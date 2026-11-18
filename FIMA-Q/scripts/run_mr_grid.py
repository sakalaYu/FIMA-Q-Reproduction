"""Three-arm MR experiment, isolated from existing gain/uniform runners.

Default stage is screening only. Run --stage quant after inspecting --variant
summary. Completed stages are reused by --resume; partial training restarts.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

import run_lsvit_mr as runner

ROOT = runner.controls.ROOT
VARIANTS = ('uniform', 'single', 'robust')
PROTOCOL = 'mr_grid_v1'


def code_sha():
    digest = hashlib.sha256()
    for name in ('run_mr_grid.py', 'mr_grid_impl.py'):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', required=True, choices=(*VARIANTS, 'summary'))
    p.add_argument('--stage', choices=('screen', 'quant', 'all'), default='screen')
    p.add_argument('--seed', type=int, default=3408)
    p.add_argument('--dataset', default='../imagenet_fimaq')
    p.add_argument('--fp-checkpoint', type=Path)
    p.add_argument('--gpu', default='0')
    p.add_argument('--gpu-slots', type=int, choices=(1, 2, 3), default=3)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    p.set_defaults(pair_id=None, mr_steps=3000, budget='uniform')
    return p


def recipe(args):
    return dict(protocol=PROTOCOL, variant=args.variant, source_sha256=code_sha(),
                base=runner.controls.identity('fixed', args))


def directory(args, identity):
    return ROOT / 'checkpoints/lsvit_controls/mr_grid' / (
        f'seed{args.seed}_{args.variant}_{runner.short_digest(identity)}')


def screen_result(folder, identity):
    try:
        report = json.loads((folder / 'mr_screen.json').read_text(encoding='utf-8'))
        if (report['status'] == 'complete' and report['grid_recipe'] == identity
                and len(report['blocks']) == 12 and report['total_mr_steps'] == 36000
                and (folder / 'relu_mlp.pth').is_file()):
            from_hash = hashlib.sha256((folder / 'relu_mlp.pth').read_bytes()).hexdigest()
            if from_hash == report['checkpoint_sha256']:
                return report
    except (OSError, ValueError, KeyError):
        pass
    return None


def launch_spec(args, folder, identity, screen):
    cmd, payload, _ = runner.branch_launch(args, 'relu', folder, identity['base'], screen)
    payload['grid_recipe'] = identity
    name = f'mr_grid_{args.variant}_seed{args.seed}_{runner.short_digest(payload)}'
    return cmd, payload, name


def summarize(args):
    records = {}
    for variant in VARIANTS:
        other = argparse.Namespace(**dict(vars(args), variant=variant))
        identity = recipe(other)
        folder = directory(other, identity)
        report = screen_result(folder, identity)
        if report is None:
            print(f'{variant}: no matching complete screen', flush=True)
            continue
        _, payload, name = launch_spec(other, folder, identity, report)
        found = runner.find_result(payload, name)
        blocks = report['blocks']
        records[variant] = dict(
            screen=str(folder / 'mr_screen.json'),
            mean_a3_nmse=report['mean_relu_a3_nmse'],
            mean_fp_nmse=sum(b['relu_fp_nmse'] for b in blocks) / 12,
            mean_worst_a3_nmse=sum(b['worst_a3_nmse'] for b in blocks) / 12,
            fp_teacher_kl=report['fp_teacher_kl'], seconds=report['elapsed_seconds'],
            selected_steps=[b['selected_step'] for b in blocks],
            anchors=[b['anchor_sha256'] for b in blocks],
            fp_sha=report['fp_model_sha256'], data_hashes=report['data_hashes'],
            metrics=found[1]['after'] if found else None,
            calibration_hashes=found[1]['calibration_hashes'] if found else None)
        print(f'{variant}: {json.dumps(records[variant])}', flush=True)
    baseline = records.get('uniform')
    for variant, row in records.items():
        if baseline is None or variant == 'uniform':
            continue
        matched = all(row[k] == baseline[k] for k in ('anchors', 'fp_sha', 'data_hashes'))
        if not matched:
            raise ValueError(f'{variant}: FP, data or shared 2000-step anchors differ')
        row['screen_matched'] = True
        row['nominal_error_change_pct'] = 100 * (row['mean_a3_nmse'] / max(baseline['mean_a3_nmse'], 1e-12) - 1)
        row['worst_error_change_pct'] = 100 * (row['mean_worst_a3_nmse'] / max(baseline['mean_worst_a3_nmse'], 1e-12) - 1)
        # Screening is a compute-allocation heuristic, not evidence of Top-1 gain.
        row['worth_full_reconstruction'] = (
            row['mean_a3_nmse'] <= baseline['mean_a3_nmse'] * .98
            and row['mean_worst_a3_nmse'] < baseline['mean_worst_a3_nmse']
            and row['fp_teacher_kl'] <= baseline['fp_teacher_kl'] * 1.1)
        if row['metrics'] and baseline['metrics']:
            if row['calibration_hashes'] != baseline['calibration_hashes']:
                raise ValueError('Full reconstruction calibration fingerprints differ')
            row['top1_delta_pp'] = row['metrics']['top1'] - baseline['metrics']['top1']
        print(f'{variant} vs uniform: {json.dumps({k: v for k, v in row.items() if k in ("nominal_error_change_pct", "worst_error_change_pct", "worth_full_reconstruction", "top1_delta_pp")})}', flush=True)
    if not args.dry_run:
        target = ROOT / 'checkpoints/lsvit_controls/mr_grid' / f'summary_seed{args.seed}_{code_sha()[:12]}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(f'.{os.getpid()}.tmp')
        temp.write_text(json.dumps(records, indent=2), encoding='utf-8')
        temp.replace(target)
        print(f'Summary: {target}', flush=True)


def child(args, extra):
    if not extra or extra[0] != '--':
        raise ValueError('Internal child requires -- followed by test_quant arguments')
    sys.path.insert(0, str(ROOT))
    from utils import lsvit_mr as mr
    from utils.lsvit_pair import write_json
    import mr_grid_impl
    original_screen = mr.screen_mlp_reconstruction
    mr.fit_mlp = lambda *a, **kw: mr_grid_impl.fit_mlp(*a, **kw, variant=args.variant)

    def screen(*a, **kw):
        result = original_screen(*a, **kw)
        result['grid_recipe'] = json.loads(os.environ['FIMAQ_GRID_RECIPE'])
        write_json(Path(a[2]) / 'mr_screen.json', result)
        return result

    mr.screen_mlp_reconstruction = screen
    sys.argv = ['test_quant.py', *extra[1:]]
    runpy.run_path(str(ROOT / 'test_quant.py'), run_name='__main__')


def main():
    parser = get_parser()
    args, extra = parser.parse_known_args()
    if args.child:
        child(args, extra)
        return
    if extra:
        parser.error(f'Unknown arguments: {extra}')
    if args.variant == 'summary':
        summarize(args)
        return
    identity = recipe(args)
    folder = directory(args, identity)
    screen = screen_result(folder, identity)
    if args.stage in ('screen', 'all'):
        if screen is not None and args.resume:
            print('[SCREEN COMPLETE] Reusing verified grid MR weights', flush=True)
        else:
            cmd = runner.command(args, 'screen', folder)
            cmd = [cmd[0], '-u', str(Path(__file__).resolve()), '--child', '--variant', args.variant,
                   '--', *cmd[3:]]
            name = f'mr_grid_screen_{args.variant}_{runner.short_digest(identity)}'
            if args.dry_run:
                print(shlex.join(cmd), flush=True)
            else:
                with runner.controls.qk.base.job_lock(name):
                    with runner.controls.qk.base.gpu_lock(args.gpu, args.gpu_slots) as uuid:
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                                   FIMAQ_CAPTURE_CALIBRATION_HASH='1', FIMAQ_RUN_NAME=name,
                                   FIMAQ_GRID_RECIPE=json.dumps(identity))
                        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
                screen = screen_result(folder, identity)
                if screen is None:
                    raise RuntimeError('Child returned without a matching complete grid screen')
                runner.print_screen(screen, folder)
    if args.stage in ('quant', 'all'):
        if screen is None and not args.dry_run:
            parser.error('Run the matching --stage screen first')
        cmd, payload, name = launch_spec(args, folder, identity, screen)
        if args.dry_run:
            print(shlex.join(cmd), flush=True)
            return
        with runner.controls.qk.base.job_lock(name):
            if args.resume and runner.find_result(payload, name):
                print('[WINDOW COMPLETE] Reusing verified grid quantization result', flush=True)
                return
            with runner.controls.qk.base.gpu_lock(args.gpu, args.gpu_slots) as uuid:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                           FIMAQ_CAPTURE_CALIBRATION_HASH='1', FIMAQ_RUN_NAME=name,
                           FIMAQ_METHOD_LAUNCH=json.dumps(payload))
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            if runner.find_result(payload, name) is None:
                raise RuntimeError('Missing complete grid quantization result')
            print('[WINDOW COMPLETE] Grid MR + LS reconstruction', flush=True)


if __name__ == '__main__':
    main()
