"""Run the seed-3408 equal-budget DPLR experiment with a locked ReLU MR checkpoint."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys

import run_lsvit_controls as controls
from run_headwise_experiments import ROOT, dataset_path, gpu_lock, job_lock

sys.path.insert(0, str(ROOT))
from utils.formal_budget import BLOCKS, NOMINAL_STEPS, PILOT_STEPS, VARIANTS, make_schedule


SEED = 3408
REFERENCE = ROOT / 'checkpoints/quant_result/20261111_194420_lsvit_mr_relu_seed3408_a4008f61fe80/method_result.json'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def make_command(reference, data, mr_dir, fp, plan_file):
    cmd = [sys.executable, *reference['launch']['command']]
    for name in ('--lsvit-control', '--method-adaptation', '--fp-checkpoint'):
        if name in cmd:
            idx = cmd.index(name)
            del cmd[idx:idx + 2]
    for name, value in (('--dataset', data), ('--ls-mr-dir', mr_dir), ('--recon-iters', NOMINAL_STEPS), ('--k', 5)):
        cmd[cmd.index(name) + 1] = str(value)
    cmd.extend(('--optim-metric', 'fisher_dplr', '--formal-budget-plan', str(plan_file)))
    if fp is not None:
        cmd.extend(('--fp-checkpoint', str(fp)))
    return cmd


def build_plan(variant, common, uniform=None):
    plan = dict(common, variant=variant)
    if variant != 'uniform':
        plan.update(uniform_pilot_nmse={b: uniform['blocks'][b]['hard_nmse'] for b in BLOCKS},
                    uniform_calibrated_sha256=uniform['calibrated_sha256'],
                    uniform_probe_sha256=uniform['probe']['tensor_sha256'])
    plan['budgets'] = make_schedule(variant, plan.get('uniform_pilot_nmse'), SEED)
    return plan


def completed(tag, plan):
    pattern = f'*_formalbudget_{plan["variant"]}_seed{SEED}_{tag}/formal_budget_result.json'
    paths = sorted((ROOT / 'checkpoints/quant_result').glob(pattern))
    if not paths:
        return None
    path = paths[-1]
    result = read_json(path)
    rows = result['blocks']
    cfg = result['cfg']
    if (result['plan'] != plan or result['status'] != 'complete' or set(rows) != set(BLOCKS)
            or (cfg['optim_metric'], cfg['w_bit'], cfg['a_bit'], cfg['recon_iters'], cfg['k'])
            != ('fisher_dplr', 3, 3, NOMINAL_STEPS, 5)
            or {name: row['steps'] for name, row in rows.items()} != plan['budgets']
            or any(row['probe_step'] != PILOT_STEPS or not math.isfinite(row['hard_nmse']) for row in rows.values())
            or not all(math.isfinite(result['final'][name]) for name in ('top1', 'top5', 'loss'))):
        raise RuntimeError(f'Incomplete or inconsistent result: {path}')
    return dict(result, path=str(path))


def summarize(folder, tag, common):
    uniform = completed(tag, build_plan('uniform', common))
    if uniform is None:
        print('No completed uniform arm yet', flush=True)
        return
    results = {'uniform': uniform}
    for variant in VARIANTS[1:]:
        result = completed(tag, build_plan(variant, common, uniform))
        if result is not None:
            if (result['cfg'] != uniform['cfg'] or result['probe'] != uniform['probe']
                    or result['calibrated_sha256'] != uniform['calibrated_sha256']):
                raise RuntimeError(f'{variant} does not match uniform calibration or probe fingerprints')
            results[variant] = result
    rows = {}
    for name, result in results.items():
        rows[name] = dict(path=result['path'], top1=result['final']['top1'],
                          delta_vs_uniform_pp=result['final']['top1'] - uniform['final']['top1'],
                          transformer_updates=sum(result['plan']['budgets'].values()),
                          reconstruction_seconds=result['reconstruction_seconds'], blocks=result['blocks'])
        row = rows[name]
        print(f'{name}: Top-1 {row["top1"]:.3f}%, delta {row["delta_vs_uniform_pp"]:+.3f} pp, '
              f'{row["reconstruction_seconds"] / 60:.1f} min', flush=True)
    residual_delta = (rows['residual']['top1'] - rows['shuffled']['top1']
                      if 'residual' in rows and 'shuffled' in rows else None)
    if residual_delta is not None:
        print(f'Residual minus shuffled: {residual_delta:+.3f} pp', flush=True)
    target = folder / 'summary.json'
    write_json(target, dict(arms=rows, complete=len(rows) == 3, residual_vs_shuffled_pp=residual_delta,
                            uniform_calibrated_sha256=uniform['calibrated_sha256'], identity=common))
    print(f'Summary: {target}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=(*VARIANTS, 'summary'), required=True)
    parser.add_argument('--dataset', default='../imagenet_fimaq')
    parser.add_argument('--fp-checkpoint', type=Path)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--gpu-slots', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    reference = read_json(REFERENCE)
    if (reference['status'], reference['args']['seed'], reference['mr_variant'], reference['after']['top1']) != (
            'complete', SEED, 'relu', 67.492):
        parser.error('Expected the original seed-3408 67.492% MR source record')
    mr_dir = ROOT / 'checkpoints/lsvit_controls/mr' / PurePosixPath(reference['args']['ls_mr_dir']).name
    weights = mr_dir / 'relu_mlp.pth'
    data = dataset_path(args.dataset)
    fp = controls.fp_checkpoint(args)
    versions = {}
    for package in ('torch', 'torchvision', 'timm', 'numpy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    common = dict(seed=SEED, dataset=str(data), fp_checkpoint=str(fp) if fp else None,
                  fp_model_sha256=reference['fp_model_sha256'], mr_weights_sha256=reference['mr_weights_sha256'],
                  calibration_hashes=reference['calibration_hashes'], code_sha256=controls.code_fingerprint(),
                  runner_sha256=digest(Path(__file__)), versions=versions)
    tag = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()[:12]
    folder = ROOT / 'checkpoints/formal_budget' / f'seed{SEED}_{tag}'
    plan_file = folder / f'{args.variant}.json'
    if args.dry_run:
        print(f'MR checkpoint: {weights} (present: {weights.is_file()})\nOutput: {folder}')
        if args.variant != 'summary':
            print(shlex.join(make_command(reference, data, mr_dir, fp, plan_file)))
        return
    if args.variant == 'summary':
        summarize(folder, tag, common)
        return
    if not all((data / part).is_dir() for part in ('train', 'val')):
        parser.error(f'Missing ImageNet train/val under {data}')
    if not weights.is_file() or not (mr_dir / 'mr_screen.json').is_file():
        parser.error(f'Restore the original uniform MR weights and report in {mr_dir}')
    if digest(weights) != common['mr_weights_sha256']:
        parser.error(f'Original MR checkpoint hash does not match: {weights}')
    if fp is not None and not fp.is_file():
        parser.error(f'Floating-point checkpoint missing: {fp}')
    uniform = completed(tag, build_plan('uniform', common))
    if args.variant != 'uniform' and uniform is None:
        parser.error('Run the uniform DPLR baseline first to obtain all 12 hard-path pilot scores')
    plan = build_plan(args.variant, common, uniform)
    folder.mkdir(parents=True, exist_ok=True)
    name = f'formalbudget_{args.variant}_seed{SEED}_{tag}'
    with job_lock(name):
        if completed(tag, plan) is None:
            write_json(plan_file, plan)
            print(f'{args.variant}: {plan["budgets"]}', flush=True)
            with gpu_lock(args.gpu, args.gpu_slots) as uuid:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                           FIMAQ_RUN_NAME=name, FIMAQ_CAPTURE_CALIBRATION_HASH='1')
                subprocess.run(make_command(reference, data, mr_dir, fp, plan_file), cwd=ROOT, env=env, check=True)
            if completed(tag, plan) is None:
                raise RuntimeError('Process exited without a completed budget result')
    with job_lock(f'formalbudget_summary_{tag}'):
        summarize(folder, tag, common)


if __name__ == '__main__':
    main()
