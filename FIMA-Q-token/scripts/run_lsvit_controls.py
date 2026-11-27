"""Fresh W3A3 LS controls and matched hard-path adaptive mixture runs."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

import run_method_adaptations as methods
import run_qk_joint_experiments as qk

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'ls_controls_v1'
MIX_VARIANTS = ('mix-equal', 'mix-adaptive')
# Tensor fingerprint from the completed 20261104 C-MSE + LS reference.
REFERENCE_FP_SHA = 'a78205f203d9299dd770d74b9f4273895a7d2a22260593bdbe7908e4e513463c'


def get_parser():
    p = argparse.ArgumentParser(description=__doc__)
    select = p.add_mutually_exclusive_group(required=True)
    select.add_argument('--worker', type=int, choices=(1, 2, 3))
    select.add_argument('--variant', choices=('plain', 'fixed', 'b', 'c', 'summary', *MIX_VARIANTS,
                                             'mix-prepare', 'mix-summary'))
    p.add_argument('--pair-id', help='Name of the shared calibration and first-branch check')
    p.add_argument('--dataset', default='../imagenet_fimaq')
    p.add_argument('--fp-checkpoint', type=Path, help='Explicit path to the original ViT-S FP weights')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--gpu', default='0')
    p.add_argument('--gpu-slots', type=int, choices=(1, 2, 3), default=3)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p


def fp_checkpoint(args):
    if args.fp_checkpoint is not None:
        path = args.fp_checkpoint.expanduser()
        return (path if path.is_absolute() else ROOT / path).resolve()
    directory = ROOT / 'checkpoints/vit_raw'
    for name in ('model.safetensors', 'vit_small_patch16_224.bin'):
        path = directory / name
        if path.is_file():
            return path
    return None


def command(variant, args):
    if variant in ('b', 'c'):
        qa = qk.get_parser().parse_args(['--variant', variant.upper(), '--reconstruction-loss', 'mse',
              '--dataset', args.dataset, '--seed', str(args.seed), '--num-workers', str(args.num_workers)])
        cmd = qk.command(variant.upper(), qa)
    else:
        cmd = qk.base.command(3, args.seed, variant != 'plain', args.dataset, args.num_workers)
    # --method-adaptation lsvit sets cfg.optim_metric. Omit the inherited
    # fisher_dplr CLI override to avoid misleading the console recipe.
    metric_index = cmd.index('--optim-metric')
    del cmd[metric_index:metric_index + 2]
    cmd[cmd.index('--k') + 1] = '1'
    cmd += ['--expected-fp-model-sha256', REFERENCE_FP_SHA]
    if variant == 'mix-prepare':
        cmd.remove('--optimize')
        cmd += ['--skip-final-validation', '--ls-pair-dir', str(pair_directory(args))]
    else:
        cmd += ['--method-adaptation', 'lsvit', '--lsvit-control', variant]
        if args.pair_id and variant in MIX_VARIANTS:
            folder = pair_directory(args)
            index = cmd.index('--calibrate')
            cmd[index:index + 1] = ['--load-calibrate-checkpoint', str(folder / 'calibrated.pth')]
            cmd += ['--ls-pair-dir', str(folder)]
    weight = fp_checkpoint(args)
    if weight is not None:
        cmd += ['--fp-checkpoint', str(weight)]
    return cmd


def code_fingerprint():
    digest = hashlib.sha256()
    digest.update(methods.fingerprint().encode())
    digest.update(Path(__file__).read_text(encoding='utf-8').encode())
    return digest.hexdigest()


def pair_directory(args):
    key = [args.pair_id, args.seed, str(qk.base.dataset_path(args.dataset)),
           str(fp_checkpoint(args)), args.num_workers, code_fingerprint()]
    tag = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]
    return ROOT / 'checkpoints/lsvit_controls/pairs' / f'{args.pair_id}_seed{args.seed}_{tag}'


def identity(variant, args):
    cmd = command(variant, args)
    cmd[0] = '<python>'
    cmd[cmd.index('--dataset') + 1] = str(qk.base.dataset_path(args.dataset))
    weight = fp_checkpoint(args)
    h = hashlib.sha256()
    if weight is not None and weight.is_file():
        with weight.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1048576), b''):
                h.update(chunk)
    versions = {}
    for package in ('torch', 'torchvision', 'timm', 'numpy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(protocol=PROTOCOL, control=variant, command=cmd[1:], code_sha256=code_fingerprint(),
                local_model_file_sha256=h.hexdigest() if weight is not None and weight.is_file() else None,
                expected_fp_model_sha256=REFERENCE_FP_SHA,
                dependency_versions=versions)


def run_name(payload, seed):
    ident = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f'lscontrol_{payload["control"]}_seed{seed}_{ident}'


def valid_result(result, payload):
    if (result.get('status') != 'complete' or result.get('launch') != payload
            or result.get('control') != payload['control'] or not result.get('fp_model_sha256')):
        return False
    expected_sha = payload.get('expected_fp_model_sha256')
    if expected_sha and result['fp_model_sha256'] != expected_sha:
        return False
    metrics = result.get('after', {})
    if not all(isinstance(metrics.get(k), (int, float)) and math.isfinite(metrics[k])
               for k in ('top1', 'top5', 'loss')):
        return False
    hashes = result.get('calibration_hashes', [])
    if [x[0] for x in hashes] != ['128', '1024']:
        return False
    if any(len(x) != 3 or any(len(h) != 64 for h in x[1:]) for x in hashes):
        return False
    cfg = result.get('cfg', {})
    if any(cfg.get(k) != v for k, v in dict(optim_metric='lsvit', k=1, recon_iters=20000,
            optim_size=1024, calib_size=128, optim_batch_size=32, calib_batch_size=32,
            optim_mode='qdrop', drop_prob=0.5, w_bit=3, a_bit=3).items()):
        return False
    blocks = result.get('qk_blocks', [])
    if payload['control'] in MIX_VARIANTS:
        probes = result.get('ls_mix_blocks', [])
        if '--ls-pair-dir' in payload.get('command', []):
            signature = result.get('pair_start') or {}
            hash_keys = ('snapshot_sha256', 'model_sha256', 'optimizer_sha256', 'ls_sha256',
                         'rng_sha256', 'probe_sha256')
            if (signature.get('step') != 10000 or signature.get('data_hashes') != hashes[1:]
                    or not all(re.fullmatch(r'[a-f0-9]{64}', signature.get(key, '')) for key in hash_keys)
                    or not any(p.get('block') == 'blocks.0' and p.get('pair_start') == signature for p in probes)):
                return False
        return (not blocks and len(probes) == 12
                and {p.get('block') for p in probes} == {f'blocks.{i}' for i in range(12)}
                and all(p.get('step') == 10000 and p.get('probe_samples') == 128
                        and p.get('applied_weight') == (0.5 if payload['control'] == 'mix-equal'
                                                        else p.get('suggested_weight'))
                        and p.get('applied_weight') in (0.25, 0.375, 0.5, 0.625, 0.75)
                        for p in probes))
    if payload['control'] in ('plain', 'fixed'):
        return not blocks
    variant = payload['control'].upper()
    return (len(blocks) == 12 and {b.get('block') for b in blocks} == {f'blocks.{i}' for i in range(12)}
            and all(b.get('variant') == variant and b.get('reconstruction_loss') == 'mse'
                    and b.get('protocol') == 'v2_integer_zp' for b in blocks))


def find_complete(payload, seed):
    paths = sorted((ROOT / 'checkpoints/quant_result').glob(
        f'*_{run_name(payload, seed)}/method_result.json'))
    for path in reversed(paths):
        try:
            result = json.loads(path.read_text(encoding='utf-8'))
            if valid_result(result, payload):
                return path, result
        except (OSError, ValueError):
            continue
    return None


def summarize(args, mix=False):
    rows = []
    variants = MIX_VARIANTS if mix else ('plain', 'fixed', 'b', 'c')
    for variant in variants:
        found = find_complete(identity(variant, args), args.seed)
        if found:
            path, result = found
            row = dict(control=variant, result_file=str(path), metrics=result['after'],
                       fp_model_sha256=result['fp_model_sha256'], calibration_hashes=result['calibration_hashes'])
            if mix:
                row['applied_weights'] = {p['block']: p['applied_weight'] for p in result['ls_mix_blocks']}
                if args.pair_id:
                    row['pair_start'] = result['pair_start']
            rows.append(row)
            print(f'{variant}: Top-1 {result["after"]["top1"]:.3f}; {path}', flush=True)
        else:
            print(f'{variant}: no matching complete result', flush=True)
    matched = len(rows) > 1 and all((r['fp_model_sha256'], r['calibration_hashes']) ==
                                  (rows[0]['fp_model_sha256'], rows[0]['calibration_hashes']) for r in rows)
    print(f'FP/data hashes matched across available controls: {matched}', flush=True)
    pair_verified = bool(mix and args.pair_id and matched and rows[0]['pair_start'] == rows[1]['pair_start'])
    if args.pair_id:
        print(f'First-branch signatures matched: {pair_verified}', flush=True)
    delta = (rows[1]['metrics']['top1'] - rows[0]['metrics']['top1']
             if mix and matched and (not args.pair_id or pair_verified) else None)
    if delta is not None:
        print(f'Adaptive minus equal Top-1: {delta:+.3f} pp', flush=True)
    if not mix:
        print('Historical C-MSE+LS = 65.022 (cross-run reference; NOT automatically a matched control).', flush=True)
    if not args.dry_run:
        folder = ROOT / 'checkpoints/lsvit_controls'
        folder.mkdir(parents=True, exist_ok=True)
        name = (f'mix_summary_{run_name(identity(MIX_VARIANTS[0], args), args.seed)}' if mix
                else f'summary_seed{args.seed}_{identity("plain", args)["code_sha256"][:12]}')
        target = folder / f'{name}.json'
        temp = target.with_suffix(f'.{os.getpid()}.tmp')
        summary = dict(protocol=PROTOCOL, rows=rows, hashes_matched=matched)
        if mix:
            summary['top1_delta_pp'] = delta
        if args.pair_id:
            summary['pair_verified'] = pair_verified
        temp.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        temp.replace(target)
        print(f'Summary: {target}', flush=True)


def main():
    p = get_parser()
    args = p.parse_args()
    if args.num_workers < 0:
        p.error('num-workers must be nonnegative')
    variant = args.variant or {1: 'plain', 2: 'fixed', 3: 'b'}[args.worker]
    if args.pair_id and (not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', args.pair_id)
                         or variant not in (*MIX_VARIANTS, 'mix-prepare', 'mix-summary')):
        p.error('--pair-id requires a mixture variant and a short name using letters, digits, _ or -')
    if variant == 'mix-prepare' and not args.pair_id:
        p.error('mix-prepare requires --pair-id')
    if args.pair_id and variant in MIX_VARIANTS and args.gpu_slots < 2:
        p.error('Paired branches wait for one another on GPU; use --gpu-slots 2 in both windows')
    if variant in ('summary', 'mix-summary'):
        summarize(args, mix=variant == 'mix-summary')
        return
    payload = identity(variant, args)
    cmd = command(variant, args)
    print(f'LS control {variant}: seed={args.seed}, pair={args.pair_id or "none"}', flush=True)
    if args.dry_run:
        print(shlex.join(cmd), flush=True)
        return
    weight = fp_checkpoint(args)
    if weight is not None and not weight.is_file():
        p.error(f'FP checkpoint does not exist: {weight}')
    print(f'FP source: {weight if weight is not None else "timm pretrained cache or download"}', flush=True)
    data = qk.base.dataset_path(args.dataset)
    if not all((data / split).is_dir() for split in ('train', 'val')):
        p.error(f'Missing train/val under {data}')
    folder = pair_directory(args) if args.pair_id else None
    snapshot_ready = folder and all((folder / name).is_file() for name in ('snapshot.json', 'calibrated.pth'))
    if folder and variant in MIX_VARIANTS and not snapshot_ready:
        p.error(f'Shared calibration missing in {folder}; run mix-prepare with the same --pair-id first')
    started = time.perf_counter()
    with qk.base.job_lock(run_name(payload, args.seed)):
        if variant == 'mix-prepare' and all((folder / name).is_file()
                                            for name in ('snapshot.json', 'calibrated.pth')):
            print(f'[PAIR PREPARED] {folder}', flush=True)
            return
        existing = find_complete(payload, args.seed)
        if args.resume and existing:
            print(f'[SKIPPED] Verified complete: {existing[0]}', flush=True)
        else:
            with qk.base.gpu_lock(args.gpu, args.gpu_slots) as uuid:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid,
                    FIMAQ_RUN_NAME=run_name(payload, args.seed), FIMAQ_CAPTURE_CALIBRATION_HASH='1',
                    FIMAQ_METHOD_LAUNCH=json.dumps(payload))
                if folder:
                    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                print(f'[START] {variant}', flush=True)
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            if variant == 'mix-prepare':
                if not all((folder / name).is_file() for name in ('snapshot.json', 'calibrated.pth')):
                    raise RuntimeError('Shared calibration process returned without a snapshot')
                print(f'[PAIR PREPARED] {folder}', flush=True)
                return
            if not find_complete(payload, args.seed):
                raise RuntimeError('Child returned without a verified complete LS control result')
    print(f'[WINDOW COMPLETE] {variant}; elapsed {(time.perf_counter() - started) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
