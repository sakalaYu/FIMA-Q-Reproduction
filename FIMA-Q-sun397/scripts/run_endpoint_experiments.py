"""C-MSE follow-up: rank10, hard-scale repair, soft continuation, and QwT.

Standard-library launcher. Three workers share the existing physical GPU lock.
Resume skips VERIFIED complete tasks; an interrupted training task restarts.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

import run_qk_joint_experiments as qk

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'endpoint_v1'
TASKS = ('rank10', 'hard-repair', 'soft-control', 'qwt')
PRE_FIX_CODE_SHA = 'facf533c4770b3a9134b4198a0ad316fa05eedfd4e7a6485f6ca6fdc3cea1d13'
PRE_FIX_REFERENCE_TASKS = ('rank10', 'qwt')


def code_fingerprint():
    files = [ROOT / 'test_quant.py']
    for folder in ('utils', 'quantizers', 'quant_layers', 'configs'):
        files += sorted((ROOT / folder).rglob('*.py'))
    files += [ROOT / 'scripts' / name for name in
              ('run_endpoint_experiments.py', 'run_qk_joint_experiments.py', 'run_headwise_experiments.py')]
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(ROOT).as_posix().encode())
        digest.update(file.read_text(encoding='utf-8').encode())
    return digest.hexdigest()


def set_option(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command += [flag, str(value)]


def command(task, args):
    qa = qk.get_parser().parse_args(['--variant', 'C', '--reconstruction-loss', 'mse',
        '--seed', str(args.seed), '--dataset', args.dataset, '--num-workers', str(args.num_workers)])
    result = qk.command('C', qa)
    set_option(result, '--k', 10 if task == 'rank10' else 5)
    result += ['--endpoint-task', task, '--endpoint-code-sha', code_fingerprint(),
               '--endpoint-steps', str(args.repair_steps), '--endpoint-lr', str(args.repair_lr),
               '--endpoint-bound', str(args.scale_bound), '--endpoint-holdout', '128',
               '--endpoint-batch-size', '8', '--endpoint-eval-every', '200']
    mode = {'hard-repair': 'hard_scale', 'soft-control': 'soft_extra', 'diagnose': 'diagnose'}.get(task)
    if mode:
        result += ['--endpoint-mode', mode]
    if task == 'qwt':
        result += ['--qwt', '--qwt-samples', '512', '--qwt-batch-size', '8', '--qwt-ridge', '0',
                   '--qwt-branch-dtype', 'fp16', '--qwt-solver-threads', str(args.solver_threads)]
    if args.smoke:
        for flag, value in (('--calib-size', 8), ('--optim-size', 16), ('--calib-batch-size', 4),
                            ('--optim-batch-size', 4), ('--recon-iters', 2), ('--k', 2),
                            ('--qk-joint-steps', 2), ('--qk-joint-batch-size', 2),
                            ('--qk-joint-holdout', 4), ('--qk-joint-eval-every', 1),
                            ('--endpoint-steps', 2), ('--endpoint-holdout', 4),
                            ('--endpoint-batch-size', 2), ('--endpoint-eval-every', 1),
                            ('--qwt-samples', 8), ('--qwt-batch-size', 2)):
            set_option(result, flag, value)
        result.append('--skip-final-validation')
    return result


def identity(task, args):
    cmd = command(task, args)
    cmd[0] = '<python>'  # Interpreter location does not alter the recipe.
    set_option(cmd, '--dataset', str(qk.base.dataset_path(args.dataset)))
    weight = ROOT / 'checkpoints/vit_raw/vit_small_patch16_224.bin'
    digest = None
    if weight.is_file():
        h = hashlib.sha256()
        with weight.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(chunk)
        digest = h.hexdigest()
    versions = {}
    for package in ('torch', 'torchvision', 'timm', 'numpy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(protocol=PROTOCOL, task=task, command=cmd, local_model_file_sha256=digest,
                dependency_versions=versions)


def identity_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def run_name(task, args, payload=None):
    payload = payload or identity(task, args)
    return f'endpoint_{task}_seed{args.seed}_{PROTOCOL}_{identity_hash(payload)[:12]}' + ('_smoke' if args.smoke else '')


def valid_result(result, task, args, payload):
    expected_status = 'smoke_complete' if args.smoke else 'complete'
    if result.get('status') != expected_status or result.get('launch_identity') != payload:
        return False
    command_args = payload['command']
    expected_sha = command_args[command_args.index('--endpoint-code-sha') + 1]
    if result.get('task') != task or result.get('code_sha256') != expected_sha:
        return False
    cfg = result.get('cfg', {})
    expected_rank = 2 if args.smoke else 10 if task == 'rank10' else 5
    if cfg.get('k') != expected_rank or cfg.get('recon_iters') != (2 if args.smoke else 20000):
        return False
    if not result.get('fp_model_sha256'):
        return False
    counts = ['8', '16'] if args.smoke else ['128', '1024']
    if [h[0] for h in result.get('calibration_hashes', [])] != counts:
        return False
    expected_blocks = {f'blocks.{i}' for i in range(12)}
    def has_blocks(blocks):
        return len(blocks) == 12 and {b.get('block') for b in blocks} == expected_blocks
    if not has_blocks(result.get('qk_blocks', [])):
        return False
    if task in ('hard-repair', 'soft-control', 'diagnose'):
        if not has_blocks(result.get('endpoint', {}).get('blocks', [])):
            return False
    if task == 'qwt' and not has_blocks(result.get('qwt', {}).get('blocks', [])):
        return False
    if not args.smoke:
        metrics = [result.get('final', {})]
        if task == 'qwt':
            metrics.append(result.get('before_qwt', {}))
        if any(not all(isinstance(m.get(key), (int, float)) and math.isfinite(m[key])
                       for key in ('top1', 'top5', 'loss')) for m in metrics):
            return False
    return True


def find_complete(task, args, payload=None):
    payload = payload or identity(task, args)
    name = run_name(task, args, payload)
    found = []
    for path in (ROOT / 'checkpoints/quant_result').glob(f'*_{name}/endpoint_result.json'):
        try:
            result = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            continue
        if valid_result(result, task, args, payload):
            result['result_file'] = str(path)
            found.append(result)
    return sorted(found, key=lambda r: r['run_id'])[-1] if found else None


def comparable(left, right, allow_rank=False, allow_pre_fix=False):
    if left['fp_model_sha256'] != right['fp_model_sha256'] or left['calibration_hashes'] != right['calibration_hashes']:
        return False
    a, b = dict(left['cfg']), dict(right['cfg'])
    if allow_rank:
        a.pop('k', None)
        b.pop('k', None)
    versions_equal = left.get('launch_identity', {}).get('dependency_versions') == right.get('launch_identity', {}).get('dependency_versions')
    same_code = left['code_sha256'] == right['code_sha256']
    if allow_pre_fix:
        same_code = same_code or any(run['task'] in PRE_FIX_REFERENCE_TASKS and
                                     run['code_sha256'] == PRE_FIX_CODE_SHA for run in (left, right))
    return a == b and same_code and versions_equal


def summarize(args, write=True):
    runs = {task: find_complete(task, args) for task in TASKS}
    if args.include_pre_fix:
        for task in PRE_FIX_REFERENCE_TASKS:
            if runs[task] is None:
                payload = identity(task, args)
                set_option(payload['command'], '--endpoint-code-sha', PRE_FIX_CODE_SHA)
                runs[task] = find_complete(task, args, payload)
    reference = runs['qwt']
    if reference is not None and not args.smoke:
        print(f"Current C-MSE baseline (before QwT): {reference['before_qwt']['top1']:.3f}")
    rows = []
    for task, run in runs.items():
        if run is None:
            print(f'{task}: incomplete')
            continue
        row = dict(task=task, result_file=run['result_file'], final=run.get('final'),
                   code_sha256=run['code_sha256'])
        if run['code_sha256'] == PRE_FIX_CODE_SHA:
            print(f'{task}: using verified pre-fix reference')
        if task == 'qwt':
            row['before_qwt'] = run.get('before_qwt')
            row['qwt_cost'] = {key: run['qwt'].get(key) for key in
                               ('added_parameters', 'added_storage_bytes', 'added_macs_per_image', 'branch_dtype')}
        if not args.smoke:
            row['delta_historical_64p758_pp'] = run['final']['top1'] - 64.758
            matched = reference is not None and comparable(
                run, reference, allow_rank=task == 'rank10', allow_pre_fix=args.include_pre_fix)
            row['matched_current_C_MSE'] = matched
            row['delta_current_C_MSE_pp'] = run['final']['top1'] - reference['before_qwt']['top1'] if matched else None
            print(f"{task}: Top-1 {run['final']['top1']:.3f}; "
                  f"delta current C-MSE {row['delta_current_C_MSE_pp']}; "
                  f"historical {row['delta_historical_64p758_pp']:+.3f} pp")
        rows.append(row)
    budget_delta = None
    if not args.smoke and runs['hard-repair'] and runs['soft-control']:
        if comparable(runs['hard-repair'], runs['soft-control']):
            budget_delta = runs['hard-repair']['final']['top1'] - runs['soft-control']['final']['top1']
            print(f'Hard repair versus extra-soft budget control: {budget_delta:+.3f} pp')
    report = dict(protocol=PROTOCOL, code_sha256=code_fingerprint(), seed=args.seed,
                  complete_count=len(rows), expected_count=4, rows=rows,
                  current_C_MSE=reference.get('before_qwt') if reference else None,
                  hard_repair_vs_soft_control_pp=budget_delta)
    if write:
        target = ROOT / 'checkpoints/endpoint_experiments' / f'summary_seed{args.seed}_{code_fingerprint()[:12]}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temp.replace(target)
        print(f'Summary: {target}')
    return report


def launch(task, args):
    payload = identity(task, args)
    name = run_name(task, args, payload)
    cmd = command(task, args)
    if args.dry_run:
        print(name)
        print(shlex.join(cmd))
        return
    with qk.base.job_lock(name):
        if args.resume and find_complete(task, args, payload):
            print(f'Skipping verified complete task: {task}')
            return
        with qk.base.gpu_lock(args.gpu, args.gpu_slots) as uuid:
            folder = ROOT / 'checkpoints/endpoint_experiments'
            folder.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=uuid, FIMAQ_RUN_NAME=name,
                       FIMAQ_CAPTURE_CALIBRATION_HASH='1',
                       FIMAQ_ENDPOINT_IDENTITY=json.dumps(payload, sort_keys=True))
            manifest = folder / f'{name}_launch.json'
            manifest.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            with (folder / f'{name}_console.log').open('a', encoding='utf-8') as out:
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1)
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        out.write(line)
                        out.flush()
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
                if code:
                    raise subprocess.CalledProcessError(code, cmd)
        if find_complete(task, args, payload) is None:
            raise RuntimeError('Process returned without a verified complete endpoint result; inspect its log')


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=('all', 'summary', 'diagnose') + TASKS, default='all')
    p.add_argument('--worker', type=int, choices=(1, 2, 3), default=1)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--dataset', default='../imagenet_fimaq')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--gpu', default='0')
    p.add_argument('--gpu-slots', type=int, choices=(1, 2, 3), default=3)
    p.add_argument('--repair-steps', type=int, default=2000)
    p.add_argument('--repair-lr', type=float, default=0.001)
    p.add_argument('--scale-bound', type=float, default=1.2)
    p.add_argument('--solver-threads', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--include-pre-fix', action='store_true',
                   help='Summary only: include verified rank10/QwT results from before the scale-sign fix')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--smoke', action='store_true', help='16-image/two-step integration check; NOT an accuracy experiment')
    return p


def main():
    p = get_parser()
    args = p.parse_args()
    if args.num_workers < 0 or args.repair_steps < 1 or args.solver_threads < 1:
        p.error('Workers must be nonnegative, steps/solver threads positive')
    if not math.isfinite(args.repair_lr) or args.repair_lr <= 0 or not math.isfinite(args.scale_bound) or args.scale_bound <= 1:
        p.error('Repair LR must be positive and scale bound > 1')
    if args.task == 'summary':
        summarize(args, write=not args.dry_run)
        return
    if args.include_pre_fix:
        p.error('--include-pre-fix requires --task summary')
    jobs = {1: ('rank10',), 2: ('hard-repair', 'soft-control'), 3: ('qwt',)}[args.worker] if args.task == 'all' else (args.task,)
    if not args.dry_run:
        if os.name != 'posix':
            p.error('GPU runs require Linux; --dry-run works on Windows')
        data = qk.base.dataset_path(args.dataset)
        if not all((data / split).is_dir() for split in ('train', 'val')):
            p.error(f'Missing train/val under {data}')
    for task in jobs:
        launch(task, args)
    if not args.dry_run:
        summarize(args)


if __name__ == '__main__':
    main()
