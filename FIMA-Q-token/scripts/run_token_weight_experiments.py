"""Run matched CLS and Fisher patch weighting against the existing uniform DPLR arm."""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess

import run_formal_budget as base
from run_headwise_experiments import ROOT, dataset_path, gpu_lock, job_lock
from utils.token_weight import CLS_SCALE, MODES, PROTOCOL


def result_for(name, plan, uniform):
    smoke = plan['token_weight']['smoke']
    filename = 'token_smoke_result.json' if smoke else 'formal_budget_result.json'
    paths = sorted((ROOT / 'checkpoints/quant_result').glob(f'*_{name}/{filename}'))
    if not paths:
        return None
    path = paths[-1]
    result = base.read_json(path)
    base.check_match(result, uniform)
    weights = result['token_weight_blocks']
    expected = {'blocks.0'} if smoke else set(base.BLOCKS)
    if result['status'] != 'complete' or result['plan'] != plan or set(weights) != expected:
        raise RuntimeError(f'Incomplete or inconsistent token result: {path}')
    for row in weights.values():
        n = row['tokens']
        scale = plan['token_weight']['cls_scale']
        expected_cls = n * scale / (scale + n - 1)
        if (n < 2 or not math.isclose(row['cls'], expected_cls, rel_tol=1e-6)
                or not math.isclose(row['patch_sum'], n - expected_cls, rel_tol=1e-6)):
            raise RuntimeError(f'Token weight totals differ from the matched CLS control: {path}')
    if not smoke and (not math.isfinite(result['final']['top1']) or
                     {block: row['steps'] for block, row in result['blocks'].items()} != plan['budgets']):
        raise RuntimeError(f'Token result does not match the full budget: {path}')
    return dict(result, path=str(path))


def task(mode, smoke, common, uniform, folder, tag):
    stage = 'smoke' if smoke else 'full'
    name = f'tokenweight_{mode}_{stage}_seed{base.SEED}_{tag}'
    plan = base.build_plan('uniform', common)
    plan.update(token_weight=dict(mode=mode, cls_scale=CLS_SCALE, protocol=PROTOCOL, smoke=smoke),
                uniform_calibrated_sha256=uniform['calibrated_sha256'],
                uniform_probe_sha256=uniform['probe']['tensor_sha256'])
    return name, folder / f'{mode}_{stage}.json', plan


def summarize(folder, tag, common, uniform):
    control = uniform['final']['top1']
    arms = {'uniform': dict(top1=control, delta_vs_uniform_pp=0.0, path=uniform['path'],
                           reconstruction_seconds=uniform['reconstruction_seconds'])}
    for mode in MODES:
        name, _, plan = task(mode, False, common, uniform, folder, tag)
        result = result_for(name, plan, uniform)
        if result is not None:
            top1 = result['final']['top1']
            arms[mode] = dict(top1=top1, delta_vs_uniform_pp=top1 - control, path=result['path'],
                              reconstruction_seconds=result['reconstruction_seconds'],
                              token_weight_blocks=result['token_weight_blocks'])
    for mode, row in arms.items():
        print(f'{mode}: Top-1 {row["top1"]:.3f}%, delta {row["delta_vs_uniform_pp"]:+.3f} pp, '
              f'{row["reconstruction_seconds"] / 60:.1f} min', flush=True)
    complete = set(MODES) <= arms.keys()
    delta = arms['gradient']['top1'] - arms['cls']['top1'] if complete else None
    if complete:
        print(f'Fisher patch minus CLS: {delta:+.3f} pp', flush=True)
    target = folder / 'summary.json'
    base.write_json(target, dict(identity=common, arms=arms, complete=complete, gradient_vs_cls_pp=delta))
    print(f'Summary: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=(*MODES, 'summary'), required=True)
    parser.add_argument('--dataset', default='../imagenet_fimaq')
    parser.add_argument('--fp-checkpoint', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--gpu-slots', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.arm == 'summary' and args.smoke_only:
        parser.error('--smoke-only requires a token arm')

    reference = base.read_json(base.REFERENCE)
    uniform = base.imagenet_history()['uniform']
    original = uniform['plan']
    data = dataset_path(args.dataset)
    fp = base.controls.fp_checkpoint(args)
    mr_dir = ROOT / 'checkpoints/lsvit_controls/mr' / PurePosixPath(reference['args']['ls_mr_dir']).name
    common = dict(original, token_protocol=PROTOCOL, code_sha256=base.controls.code_fingerprint(),
                  runner_sha256=base.digest(Path(__file__)), uniform_result_sha256=base.digest(Path(uniform['path'])))
    tag = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()[:12]
    folder = ROOT / 'checkpoints/token_weight' / f'seed{base.SEED}_{tag}'
    stages = (True,) if args.smoke_only else (True, False)

    def command(plan_path):
        return base.make_command(reference, data, mr_dir, fp, plan_path) + [
            '--token-weight', args.arm, '--token-weight-cls-scale', str(CLS_SCALE)]

    if args.dry_run:
        print(f'Uniform Top-1: {uniform["final"]["top1"]:.3f}%\nOutput: {folder}')
        if args.arm != 'summary':
            for smoke in stages:
                name, plan_path, _ = task(args.arm, smoke, common, uniform, folder, tag)
                print(f'{name}: {shlex.join(command(plan_path))}')
        return

    versions = {}
    for package in original['versions']:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    if str(data) != original['dataset'] or versions != original['versions']:
        parser.error('Use the same ImageNet path and dependency versions as the original uniform arm')
    folder.mkdir(parents=True, exist_ok=True)
    if args.arm != 'summary':
        weights = mr_dir / 'relu_mlp.pth'
        if not all((data / split).is_dir() for split in ('train', 'val')):
            parser.error(f'Missing ImageNet training or validation data: {data}')
        if not weights.is_file() or not (mr_dir / 'mr_screen.json').is_file():
            parser.error(f'Restore the original uniform ReLU MR weights and report: {mr_dir}')
        if (base.digest(weights) != original['mr_weights_sha256']
                or any(reference[key] != original[key] for key in
                       ('calibration_hashes', 'fp_model_sha256', 'mr_weights_sha256'))):
            parser.error('The ReLU MR source or reference fingerprint differs from the uniform arm')
        if fp is not None and not fp.is_file():
            parser.error(f'Floating-point checkpoint missing: {fp}')
        for smoke in stages:
            name, plan_path, plan = task(args.arm, smoke, common, uniform, folder, tag)
            with job_lock(name):
                if result_for(name, plan, uniform) is not None:
                    print(f'Reusing {name}', flush=True)
                    continue
                base.write_json(plan_path, plan)
                with gpu_lock(args.gpu, args.gpu_slots) as uuid:
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                               FIMAQ_RUN_NAME=name, FIMAQ_CAPTURE_CALIBRATION_HASH='1')
                    subprocess.run(command(plan_path), cwd=ROOT, env=env, check=True)
                if result_for(name, plan, uniform) is None:
                    raise RuntimeError(f'Process exited without a complete result: {name}')
    with job_lock(f'tokenweight_summary_{tag}'):
        summarize(folder, tag, common, uniform)


if __name__ == '__main__':
    main()
