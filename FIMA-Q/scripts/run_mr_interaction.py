"""Residual-budget operator-interaction MR experiment.

Default stage is screening only. Run --stage quant after inspecting --variant
summary. Completed stages are reused by --resume; partial training restarts.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

import run_lsvit_mr as runner

ROOT = runner.controls.ROOT
VARIANTS = ('baseline', 'joint', 'interaction')
PROTOCOL = 'mr_interaction_v1'


def code_sha():
    digest = hashlib.sha256()
    for name in ('run_mr_interaction.py', 'mr_interaction_impl.py'):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', required=True, choices=(*VARIANTS, 'summary'))
    p.add_argument('--stage', choices=('screen', 'quant', 'all'), default='screen')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--dataset', default='../imagenet_fimaq')
    p.add_argument('--fp-checkpoint', type=Path)
    p.add_argument('--gpu', default='0')
    p.add_argument('--gpu-slots', type=int, choices=(1, 2, 3), default=3)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    p.set_defaults(pair_id=None, mr_steps=3000, budget='residual')
    return p


def recipe(args):
    return dict(protocol=PROTOCOL, variant=args.variant, source_sha256=code_sha(),
                base=runner.controls.identity('fixed', args))


def directory(args, identity):
    return ROOT / 'checkpoints/lsvit_controls/mr_interaction' / (
        f'seed{args.seed}_{args.variant}_{runner.short_digest(identity)}')


def screen_result(folder, identity):
    try:
        report = json.loads((folder / 'mr_screen.json').read_text(encoding='utf-8'))
        if (report['status'] == 'complete' and report['interaction_recipe'] == identity
                and len(report['blocks']) == 12 and report['total_mr_steps'] == 36000
                and report['budget'] == 'residual' and len(report['weights_tensor_sha256']) == 64
                and [b['block'] for b in report['blocks']] == [f'blocks.{i}' for i in range(12)]
                and sorted(b['steps'] for b in report['blocks']) == [2000] * 4 + [3000] * 4 + [4000] * 4
                and all(len(b['shared_start_sha256']) == 64
                        and all(math.isfinite(v) for v in b['interaction_diagnostics'].values())
                        for b in report['blocks'])
                and (folder / 'relu_mlp.pth').is_file()):
            from_hash = hashlib.sha256((folder / 'relu_mlp.pth').read_bytes()).hexdigest()
            if from_hash == report['checkpoint_sha256']:
                return report
    except (OSError, ValueError, KeyError):
        pass
    return None


def launch_spec(args, folder, identity, screen):
    cmd, payload, _ = runner.branch_launch(args, 'relu', folder, identity['base'], screen)
    payload['interaction_recipe'] = identity
    name = f'mr_interaction_{args.variant}_seed{args.seed}_{runner.short_digest(payload)}'
    return cmd, payload, name


def normalized_command(command):
    command = list(command)
    for option in ('--fp-checkpoint', '--ls-mr-dir'):
        if option in command:
            i = command.index(option)
            del command[i:i + 2]
    return command


def historical_baseline(args, screen, payload):
    """Reuse only verified matching MR tensors, FP/data, dependencies and recipe.

    The historical source-code fingerprint is retained, not claimed identical.
    """
    if args.variant != 'baseline' or screen is None:
        return None
    for path in sorted((ROOT / 'checkpoints/quant_result').glob('*/method_result.json'), reverse=True):
        try:
            result = json.loads(path.read_text(encoding='utf-8'))
            launch = result['launch']
            if (result['args'].get('seed') != args.seed or result['args'].get('ls_mr_budget') != 'residual'
                    or result.get('mr_variant') != 'relu' or launch.get('control') != 'fixed'
                    or not runner.controls.valid_result(result, launch)
                    or result['fp_model_sha256'] != screen['fp_model_sha256']
                    or result['calibration_hashes'][1:] != screen['data_hashes']
                    or launch['dependency_versions'] != payload['dependency_versions']
                    or normalized_command(launch['command']) != normalized_command(payload['command'])):
                continue
            weights_match = result.get('mr_weights_sha256') == screen['checkpoint_sha256']
            if not weights_match:
                # Some torch versions serialize identical tensors into different file bytes.
                command = launch['command']
                old_folder = ROOT / 'checkpoints/lsvit_controls/mr' / Path(
                    command[command.index('--ls-mr-dir') + 1]).name
                old_weights = old_folder / 'relu_mlp.pth'
                if old_weights.is_file():
                    import torch
                    from utils.lsvit_pair import fingerprint
                    weights_match = fingerprint(torch.load(old_weights, map_location='cpu', weights_only=True)) == screen['weights_tensor_sha256']
            if weights_match:
                return path, result
        except (OSError, ValueError, KeyError):
            continue
    return None


def find_quant(args, screen, payload, name):
    current = runner.find_result(payload, name)
    if current:
        return current, 'current_run'
    historical = historical_baseline(args, screen, payload)
    return historical, 'historical_matching_weights_data_recipe' if historical else None


def summarize(args):
    records = {}
    for variant in VARIANTS:
        other = argparse.Namespace(**dict(vars(args), variant=variant))
        identity = recipe(other)
        folder = directory(other, identity)
        screen = screen_result(folder, identity)
        if screen is None:
            print(f'{variant}: no matching complete screen', flush=True)
            continue
        _, payload, name = launch_spec(other, folder, identity, screen)
        found, origin = find_quant(other, screen, payload, name)
        blocks = screen['blocks']
        keys = ('fp_nmse', 'fc1_w3a3_nmse', 'fc2_w3a3_nmse', 'joint_w3a3_nmse',
                'positive_interaction_nmse', 'harmful_token_fraction', 'mean_interaction_weight')
        row = dict(screen=str(folder / 'mr_screen.json'),
                   means={k: sum(b['interaction_diagnostics'][k] for b in blocks) / 12 for k in keys},
                   fp_teacher_kl=screen['fp_teacher_kl'], seconds=screen['elapsed_seconds'],
                   steps=[b['steps'] for b in blocks],
                   shared_starts=[b['shared_start_sha256'] for b in blocks],
                   fp_sha=screen['fp_model_sha256'], data_hashes=screen['data_hashes'],
                   weights_tensor_sha256=screen['weights_tensor_sha256'],
                   metrics=found[1]['after'] if found else None,
                   quant_result=str(found[0]) if found else None, quant_origin=origin,
                   calibration_hashes=found[1]['calibration_hashes'] if found else None,
                   formal_code_sha256=found[1]['launch']['code_sha256'] if found else None)
        records[variant] = row
        print(f'{variant}: {json.dumps(row)}', flush=True)
    baseline = records.get('baseline')
    for variant in ('joint', 'interaction'):
        row = records.get(variant)
        if baseline is None or row is None:
            continue
        if any(row[k] != baseline[k] for k in ('steps', 'shared_starts', 'fp_sha', 'data_hashes')):
            raise ValueError('MR budget, common starts, FP or data mismatch')
        row['matched_screen'] = True
        row['joint_error_change_pct'] = 100 * (row['means']['joint_w3a3_nmse'] / max(baseline['means']['joint_w3a3_nmse'], 1e-12) - 1)
        if row['metrics'] and baseline['metrics']:
            if row['calibration_hashes'] != baseline['calibration_hashes']:
                raise ValueError('Formal calibration data mismatch')
            row['top1_delta_vs_baseline_pp'] = row['metrics']['top1'] - baseline['metrics']['top1']
        print(f'{variant}: joint W3A3 error change {row["joint_error_change_pct"]:+.3f}%', flush=True)
    if all(records.get(v, {}).get('metrics') for v in ('joint', 'interaction')):
        a, b = records['joint'], records['interaction']
        if a['calibration_hashes'] != b['calibration_hashes'] or a['formal_code_sha256'] != b['formal_code_sha256']:
            raise ValueError('Joint/interaction formal protocol mismatch')
        records['interaction']['top1_delta_vs_joint_pp'] = b['metrics']['top1'] - a['metrics']['top1']
        print(f'Interaction minus joint: {b["metrics"]["top1"] - a["metrics"]["top1"]:+.3f} pp', flush=True)
    if not args.dry_run:
        target = ROOT / 'checkpoints/lsvit_controls/mr_interaction' / f'summary_seed{args.seed}_{code_sha()[:12]}.json'
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
    import torch
    from utils.lsvit_pair import fingerprint, write_json
    import mr_interaction_impl
    original_screen = mr.screen_mlp_reconstruction
    original_fit = mr.fit_with_residual_budget
    mr.fit_with_residual_budget = lambda *a, **kw: mr_interaction_impl.fit_residual(
        *a, **kw, variant=args.variant, original_fit=original_fit)

    def screen(*a, **kw):
        result = original_screen(*a, **kw)
        result['interaction_recipe'] = json.loads(os.environ['FIMAQ_INTERACTION_RECIPE'])
        result['weights_tensor_sha256'] = fingerprint(torch.load(Path(a[2]) / 'relu_mlp.pth', map_location='cpu', weights_only=True))
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
            print('[SCREEN COMPLETE] Reusing verified interaction MR weights', flush=True)
        else:
            cmd = runner.command(args, 'screen', folder)
            cmd = [cmd[0], '-u', str(Path(__file__).resolve()), '--child', '--variant', args.variant,
                   '--', *cmd[3:]]
            name = f'mr_interaction_screen_{args.variant}_{runner.short_digest(identity)}'
            if args.dry_run:
                print(shlex.join(cmd), flush=True)
            else:
                with runner.controls.qk.base.job_lock(name):
                    with runner.controls.qk.base.gpu_lock(args.gpu, args.gpu_slots) as uuid:
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                                   FIMAQ_CAPTURE_CALIBRATION_HASH='1', FIMAQ_RUN_NAME=name,
                                   FIMAQ_INTERACTION_RECIPE=json.dumps(identity))
                        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
                screen = screen_result(folder, identity)
                if screen is None:
                    raise RuntimeError('Child returned without a matching complete interaction screen')
                runner.print_screen(screen, folder)
    if args.stage in ('quant', 'all'):
        if screen is None and not args.dry_run:
            parser.error('Run the matching --stage screen first')
        cmd, payload, name = launch_spec(args, folder, identity, screen)
        if args.dry_run:
            print(shlex.join(cmd), flush=True)
            return
        with runner.controls.qk.base.job_lock(name):
            found, origin = find_quant(args, screen, payload, name)
            if found and (args.resume or origin != 'current_run'):
                print(f'[WINDOW COMPLETE] Reused {origin}: Top-1 {found[1]["after"]["top1"]:.3f}%; {found[0]}', flush=True)
                return
            with runner.controls.qk.base.gpu_lock(args.gpu, args.gpu_slots) as uuid:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                           FIMAQ_CAPTURE_CALIBRATION_HASH='1', FIMAQ_RUN_NAME=name,
                           FIMAQ_METHOD_LAUNCH=json.dumps(payload))
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            if runner.find_result(payload, name) is None:
                raise RuntimeError('Missing complete interaction quantization result')
            print('[WINDOW COMPLETE] Interaction MR + LS reconstruction', flush=True)


if __name__ == '__main__':
    main()
