"""Screen MLP reconstruction and compare GELU/ReLU W3A3 with fixed Hadamard + LS."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

import run_lsvit_controls as controls


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=('screen', 'gelu', 'relu', 'summary', 'budget-summary'))
    parser.add_argument('--dataset', default='../imagenet_fimaq')
    parser.add_argument('--fp-checkpoint', type=Path)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--mr-steps', type=int, default=3000)
    parser.add_argument('--budget', choices=('uniform', 'residual'), default='uniform')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--gpu-slots', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.set_defaults(pair_id=None)
    return parser


def short_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]


def command(args, variant, directory):
    cmd = controls.command('fixed', args)
    cmd += ['--ls-mr', variant, '--ls-mr-steps', str(args.mr_steps)]
    if variant in ('screen', 'relu'):
        cmd += ['--ls-mr-dir', str(directory), '--ls-mr-budget', args.budget]
    if variant == 'screen':
        for option in ('--method-adaptation', '--lsvit-control'):
            index = cmd.index(option)
            del cmd[index:index + 2]
        cmd.remove('--optimize')
        cmd.append('--skip-final-validation')
    return cmd


def read_screen(directory, args):
    try:
        result = json.loads((directory / 'mr_screen.json').read_text(encoding='utf-8'))
        if (result['status'] == 'complete' and result['seed'] == args.seed and len(result['blocks']) == 12
                and result['budget'] == args.budget and result['total_mr_steps'] == 12 * args.mr_steps):
            return result
    except (OSError, ValueError, KeyError):
        pass
    return None


def print_screen(report, directory):
    print(f'FP teacher agreement: {100 * report["fp_top1_agreement"]:.2f}%', flush=True)
    print(f'Mean A3 NMSE: GELU={report["mean_gelu_a3_nmse"]:.6g}, '
          f'ReLU={report["mean_relu_a3_nmse"]:.6g}', flush=True)
    print(f'MR steps: {[block["steps"] for block in report["blocks"]]}, total={report["total_mr_steps"]}; '
          f'elapsed={report["elapsed_seconds"]:.1f}s', flush=True)
    print(f'[SCREEN COMPLETE] {directory / "mr_screen.json"}', flush=True)


def branch_launch(args, variant, directory, base, screen):
    cmd = command(args, variant, directory)
    recorded = cmd[1:]
    recorded[recorded.index('--dataset') + 1] = str(controls.qk.base.dataset_path(args.dataset))
    payload = dict(base, command=recorded, mr_variant=variant)
    payload['mr_weights_sha256'] = screen['checkpoint_sha256'] if variant == 'relu' and screen else None
    name = f'lsvit_mr_{variant}_seed{args.seed}_{short_digest(payload)}'
    return cmd, payload, name


def find_result(payload, name):
    results = (controls.ROOT / 'checkpoints/quant_result').glob(f'*_{name}/method_result.json')
    for path in sorted(results, reverse=True):
        try:
            result = json.loads(path.read_text(encoding='utf-8'))
            if (controls.valid_result(result, payload) and result.get('mr_variant') == payload['mr_variant']
                    and result.get('mr_weights_sha256') == payload['mr_weights_sha256']):
                return path, result
        except (OSError, ValueError):
            continue
    return None


def summarize(args, directory, base, screen):
    rows = []
    for variant in ('gelu', 'relu'):
        _, payload, name = branch_launch(args, variant, directory, base, screen)
        found = find_result(payload, name)
        if found is None:
            print(f'{variant}: no matching complete result', flush=True)
            continue
        path, result = found
        rows.append(dict(variant=variant, path=str(path), metrics=result['after'],
                         fp_model_sha256=result['fp_model_sha256'], calibration_hashes=result['calibration_hashes']))
        print(f'{variant}: Top-1 {result["after"]["top1"]:.3f}%', flush=True)
    matched = (len(rows) == 2 and rows[0]['fp_model_sha256'] == rows[1]['fp_model_sha256']
               and rows[0]['calibration_hashes'] == rows[1]['calibration_hashes'])
    delta = rows[1]['metrics']['top1'] - rows[0]['metrics']['top1'] if matched else None
    print(f'FP and calibration fingerprints matched: {matched}', flush=True)
    if matched:
        print(f'ReLU MR minus GELU Top-1: {delta:+.3f} pp', flush=True)
    if not args.dry_run:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / 'mr_summary.json'
        temporary = target.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(dict(rows=rows, hashes_matched=matched, top1_delta_pp=delta),
                                        indent=2), encoding='utf-8')
        temporary.replace(target)
        print(f'Summary: {target}', flush=True)


def screen_directory(args, base):
    tag = short_digest([base, args.mr_steps, args.budget])
    return controls.ROOT / 'checkpoints/lsvit_controls/mr' / f'seed{args.seed}_{args.budget}_{tag}'


def summarize_budgets(args, base):
    rows = []
    for budget in ('uniform', 'residual'):
        args.budget = budget
        directory = screen_directory(args, base)
        screen = read_screen(directory, args)
        _, payload, name = branch_launch(args, 'relu', directory, base, screen)
        found = find_result(payload, name) if screen else None
        if found is None:
            print(f'{budget}: no matching complete ReLU result', flush=True)
            continue
        path, result = found
        rows.append(dict(budget=budget, path=str(path), metrics=result['after'],
                         total_mr_steps=screen['total_mr_steps'], mr_seconds=screen['elapsed_seconds'],
                         steps_by_block={block['block']: block['steps'] for block in screen['blocks']},
                         fp_model_sha256=result['fp_model_sha256'], calibration_hashes=result['calibration_hashes']))
        print(f'{budget}: Top-1 {result["after"]["top1"]:.3f}%', flush=True)
    matched = len(rows) == 2 and all(rows[0][key] == rows[1][key] for key in (
        'fp_model_sha256', 'calibration_hashes', 'total_mr_steps'))
    delta = rows[1]['metrics']['top1'] - rows[0]['metrics']['top1'] if matched else None
    print(f'FP, calibration fingerprints and total MR steps matched: {matched}', flush=True)
    if matched:
        print(f'Residual minus uniform Top-1: {delta:+.3f} pp', flush=True)
        if not args.dry_run:
            target = directory / 'mr_budget_summary.json'
            temporary = target.with_suffix(f'.{os.getpid()}.tmp')
            temporary.write_text(json.dumps(dict(rows=rows, matched=matched, top1_delta_pp=delta),
                                            indent=2), encoding='utf-8')
            temporary.replace(target)
            print(f'Summary: {target}', flush=True)


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.mr_steps < 1 or args.num_workers < 0:
        parser.error('mr-steps must be positive and num-workers must be nonnegative')
    if (args.budget == 'residual' or args.variant == 'budget-summary') and args.mr_steps != 3000:
        parser.error('The residual budget experiment requires --mr-steps 3000')
    if args.budget == 'residual' and args.variant in ('gelu', 'summary'):
        parser.error('Use residual budget with screen or relu, and budget-summary for comparison')
    base = controls.identity('fixed', args)
    base['mr_runner_sha256'] = hashlib.sha256(Path(__file__).read_text(encoding='utf-8').encode()).hexdigest()
    if args.variant == 'budget-summary':
        summarize_budgets(args, base)
        return
    directory = screen_directory(args, base)
    screen = read_screen(directory, args)
    if args.variant == 'summary':
        summarize(args, directory, base, screen)
        return
    cmd, payload, name = branch_launch(args, args.variant, directory, base, screen)
    if args.dry_run:
        print(shlex.join(cmd), flush=True)
        return
    weight = controls.fp_checkpoint(args)
    if weight is not None and not weight.is_file():
        parser.error(f'FP checkpoint does not exist: {weight}')
    dataset = controls.qk.base.dataset_path(args.dataset)
    if not all((dataset / split).is_dir() for split in ('train', 'val')):
        parser.error(f'Missing train/val under {dataset}')
    if args.variant == 'relu' and (screen is None or not (directory / 'relu_mlp.pth').is_file()):
        parser.error('Run --variant screen with the same seed and mr-steps first')
    with controls.qk.base.job_lock(name):
        if args.variant == 'screen':
            screen = read_screen(directory, args)
            if screen and (directory / 'relu_mlp.pth').is_file():
                print_screen(screen, directory)
                return
        elif args.resume and find_result(payload, name):
            print(f'[WINDOW COMPLETE] Verified existing {args.variant} result', flush=True)
            return
        with controls.qk.base.gpu_lock(args.gpu, args.gpu_slots) as uuid:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       FIMAQ_RUN_NAME=name, FIMAQ_CAPTURE_CALIBRATION_HASH='1',
                       FIMAQ_METHOD_LAUNCH=json.dumps(payload))
            subprocess.run(cmd, cwd=controls.ROOT, env=env, check=True)
        if args.variant == 'screen':
            screen = read_screen(directory, args)
            if screen is None or not (directory / 'relu_mlp.pth').is_file():
                raise RuntimeError('MLP screening returned without a completed report and weights')
            print_screen(screen, directory)
        else:
            if find_result(payload, name) is None:
                raise RuntimeError('Quantization returned without a matching complete result')
            print(f'[WINDOW COMPLETE] {args.variant}', flush=True)


if __name__ == '__main__':
    main()
