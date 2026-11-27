"""Three matched W3A3 semantic-probe KD trials; Linux GPU launcher."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import run_qk_joint_experiments as qk

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'semantic_probe_v1'
JOBS = (('C', 'uniform'), ('C', 'reliable'), ('B', 'reliable'))
PARAMS = dict(probe_kd_weight=0.05, probe_kd_temperature=2.0,
              probe_kd_ridge=0.1, probe_kd_holdout=128)


def code_fingerprint():
    digest = hashlib.sha256()
    for file in ('test_quant.py', 'utils/block_recon.py', 'utils/qk_joint.py',
                 'utils/semantic_probe_kd.py', 'scripts/run_probe_kd_experiments.py'):
        digest.update(file.encode('utf-8'))
        digest.update((ROOT / file).read_text(encoding='utf-8').encode('utf-8'))
    return digest.hexdigest()


def qargs(variant, args):
    return qk.get_parser().parse_args(['--variant', variant, '--reconstruction-loss', 'mse',
        '--seed', str(args.seed), '--dataset', args.dataset, '--num-workers', str(args.num_workers)])


def command(variant, mode, args):
    cmd = qk.command(variant, qargs(variant, args))
    cmd += ['--probe-kd', mode, '--probe-kd-protocol', PROTOCOL,
            '--probe-kd-code-sha', code_fingerprint()]
    for key in PARAMS:
        cmd += ['--' + key.replace('_', '-'), str(getattr(args, key))]
    return cmd


def run_name(variant, mode, args):
    values = '_'.join(str(getattr(args, key)).replace('.', 'p') for key in PARAMS)
    return f'probekd_w3a3_seed{args.seed}_{variant}_{mode}_{PROTOCOL}_{values}_{code_fingerprint()[:10]}'


def available_runs():
    paths = list((ROOT / 'logs').glob('*probekd*.log'))
    paths += list((ROOT / 'checkpoints/quant_result').glob('*probekd*/output.log'))
    found = {}
    for path in paths:
        run = qk.base.read_run(path)
        if run is None or run['args'].get('probe_kd_protocol') != PROTOCOL:
            continue
        if run['bits'] != 3 or run['rotation_mode'] != 'all' or not run['python_seeded']:
            continue
        if [h[0] for h in run['calibration_hashes']] != ['128', '1024']:
            continue
        probes, transforms = {}, {}
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            for prefix, target in (('Probe KD result: ', probes), ('QK post-round result: ', transforms)):
                if line.startswith(prefix):
                    item = json.loads(line[len(prefix):])
                    target[item['block']] = item
        expected = {f'blocks.{i}' for i in range(12)}
        if set(probes) != expected or set(transforms) != expected:
            continue
        run['probe_blocks'] = [probes[f'blocks.{i}'] for i in range(12)]
        run['qk_blocks'] = [transforms[f'blocks.{i}'] for i in range(12)]
        found[run['run_id']] = run
    return sorted(found.values(), key=lambda r: r['run_id'])


def latest(runs, variant, mode, args):
    matches = [r for r in runs if r['seed'] == args.seed
               and r['args'].get('qk_joint_variant') == variant and r['args'].get('probe_kd') == mode
               and r['args'].get('num_workers') == args.num_workers
               and qk.base.dataset_path(r['args']['dataset']) == qk.base.dataset_path(args.dataset)
               and r['args'].get('qk_joint_protocol') == qk.PROTOCOL
               and r['args'].get('qk_joint_reconstruction_loss') == 'mse'
               and r['args'].get('probe_kd_code_sha') == code_fingerprint()
               and r['args'].get('qk_joint_rotation_rank', 0) == 0
               and r['args'].get('qk_joint_min_relative_kl', 0) == 0
               and all(r['args'].get(k) == v for k, v in qk.DEFAULTS.items())
               and all(r['args'].get(k) == getattr(args, k) for k in PARAMS)]
    return matches[-1] if matches else None


def summarize(args, write=True):
    runs = available_runs()
    references = qk.available_runs()
    rows, complete = [], {}
    for variant, mode in JOBS:
        run = latest(runs, variant, mode, args)
        if run is None:
            print(f'{variant}/{mode}: incomplete')
            continue
        complete[(variant, mode)] = run
        ref = qk.latest(references, variant, qargs(variant, args), protocol=qk.PROTOCOL)
        matched = ref is not None and all(run[k] == ref[k] for k in ('cfg', 'calibration_hashes'))
        delta = run['top1'] - ref['top1'] if matched else None
        row = dict(variant=variant, mode=mode, run=run, reference=ref,
                   matched_reference=matched, delta_no_kd_pp=delta,
                   mean_reliability=sum(b['reliability'] for b in run['probe_blocks']) / 12,
                   active_blocks=sum(b['active_steps'] > 0 for b in run['probe_blocks']))
        rows.append(row)
        print(f"{variant}/{mode}: Top-1 {run['top1']:.3f}; delta no-KD {delta}; "
              f"active blocks {row['active_blocks']}/12; mean reliability {row['mean_reliability']:.3f}")
    reliability_delta = None
    if all(('C', mode) in complete for mode in ('uniform', 'reliable')):
        left, right = complete[('C', 'reliable')], complete[('C', 'uniform')]
        if all(left[k] == right[k] for k in ('cfg', 'calibration_hashes')):
            reliability_delta = left['top1'] - right['top1']
            print(f'Reliability weighting versus uniform: {reliability_delta:+.3f} pp')
    report = dict(protocol=PROTOCOL, code_sha256=code_fingerprint(), rows=rows, complete_count=len(rows), expected_count=3,
                  reliability_delta_pp=reliability_delta)
    if write:
        folder = ROOT / 'checkpoints/probe_kd'
        folder.mkdir(parents=True, exist_ok=True)
        values = '_'.join(str(getattr(args, key)).replace('.', 'p') for key in PARAMS)
        target = folder / f'summary_seed{args.seed}_{values}_{code_fingerprint()[:10]}.json'
        temporary = target.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(target)
        print(f'Summary: {target}')
    return report


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=int, choices=(1, 2, 3), default=1)
    p.add_argument('--summary', action='store_true')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--dataset', default='../imagenet_fimaq')
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--gpu', default='0')
    p.add_argument('--gpu-slots', type=int, choices=(1, 2, 3), default=3)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    for key, default in PARAMS.items():
        p.add_argument('--' + key.replace('_', '-'), type=type(default), default=default)
    return p


def main():
    parser = get_parser()
    args = parser.parse_args()
    import math
    if args.num_workers < 0 or any(not math.isfinite(getattr(args, k)) or getattr(args, k) <= 0 for k in PARAMS):
        parser.error('Probe parameters must be finite/positive and num-workers nonnegative')
    if args.probe_kd_holdout >= qk.base.RECIPE['optim_size']:
        parser.error('Probe holdout must be smaller than 1024')
    if args.summary:
        summarize(args, write=not args.dry_run)
        return
    variant, mode = JOBS[args.worker - 1]
    cmd = command(variant, mode, args)
    name = run_name(variant, mode, args)
    if args.dry_run:
        print(name)
        print(shlex.join(cmd))
        return
    if os.name != 'posix':
        parser.error('GPU runs require Linux; use --dry-run on Windows')
    for file, marker in (('test_quant.py', '--probe-kd'), ('utils/block_recon.py', 'probe.loss('),
                         ('utils/semantic_probe_kd.py', PROTOCOL)):
        if marker not in (ROOT / file).read_text(encoding='utf-8'):
            parser.error(f'Sync {file} before running')
    data = qk.base.dataset_path(args.dataset)
    if not all((data / split).is_dir() for split in ('train', 'val')):
        parser.error(f'Missing train/val under {data}')
    with qk.base.job_lock(name):
        if args.resume and latest(available_runs(), variant, mode, args):
            print('Skipping complete matched KD run')
            summarize(args)
            return
        with qk.base.gpu_lock(args.gpu, args.gpu_slots) as uuid:
            before = {r['run_id'] for r in available_runs()}
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=uuid, FIMAQ_RUN_NAME=name, FIMAQ_CAPTURE_CALIBRATION_HASH='1')
            folder = ROOT / 'checkpoints/probe_kd'
            folder.mkdir(parents=True, exist_ok=True)
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
        new = [r for r in available_runs() if r['run_id'] not in before]
        if latest(new, variant, mode, args) is None:
            raise RuntimeError('No completed run with 12 probe and QK reports; inspect console log')
    summarize(args)


if __name__ == '__main__':
    main()
