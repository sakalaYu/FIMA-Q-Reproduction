"""Three-window schedule: endpoint controls, logits KD, AdaLog/ERQ/LS-ViT."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys

import run_qk_joint_experiments as qk

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'checkpoints/quant_result/20261103_143115_endpoint_qwt_seed3407_endpoint_v1_9b042e92c4c0/endpoint_result.json'
WORKER_JOBS = {1: ('kd', 'soft-control'), 2: ('adalog', 'hard-repair'), 3: ('erq', 'lsvit')}


def fingerprint():
    paths = [ROOT / 'test_quant.py', Path(__file__).resolve()]
    for folder in ('utils', 'quantizers', 'quant_layers', 'configs'):
        paths += sorted((ROOT / folder).rglob('*.py'))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_text(encoding='utf-8').encode())
    return digest.hexdigest()


def command(method, args, checkpoint):
    if method == 'lsvit':
        qa = qk.get_parser().parse_args(['--variant', 'C', '--reconstruction-loss', 'mse',
            '--dataset', args.dataset, '--seed', '3407'])
        cmd = qk.command('C', qa)
        cmd[cmd.index('--k') + 1] = '1'
    else:
        cmd = [sys.executable, '-u', 'test_quant.py', '--model', 'vit_small', '--config', './configs/3bit/best.py',
               '--device', 'cuda', '--dataset', args.dataset, '--seed', '3407', '--val-batch-size', '64',
               '--optim-size', '1024', '--load-optimize-checkpoint', str(checkpoint), '--test-optimize-checkpoint']
    return cmd + ['--method-adaptation', method, '--adaptation-steps', str(args.kd_steps)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=int, choices=(1, 2, 3))
    p.add_argument('--method', choices=('kd', 'adalog', 'erq', 'lsvit'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--dataset', default='../imagenet_fimaq')
    p.add_argument('--gpu', default='0')
    p.add_argument('--gpu-slots', type=int, choices=(1, 2, 3), default=3)
    p.add_argument('--kd-steps', type=int, default=500)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if bool(args.worker) == bool(args.method):
        p.error('Choose --worker or --method')
    if args.kd_steps < 1:
        p.error('KD steps must be positive')
    methods = (args.method,) if args.method else WORKER_JOBS[args.worker]
    checkpoint = args.checkpoint
    if checkpoint is None and any(m != 'lsvit' for m in methods):
        if not REFERENCE.is_file():
            p.error('Pass --checkpoint with the original C-MSE model path')
        metadata = json.loads(REFERENCE.read_text(encoding='utf-8'))
        checkpoint = REFERENCE.with_name(PurePosixPath(metadata['baseline_checkpoint']).name)
    if checkpoint is not None:
        checkpoint = checkpoint.expanduser().resolve()
        if not args.dry_run and not checkpoint.is_file():
            p.error(f'Missing baseline checkpoint: {checkpoint}')
    if not args.dry_run:
        data = qk.base.dataset_path(args.dataset)
        if not all((data / split).is_dir() for split in ('train', 'val')):
            p.error(f'Missing train/val under {data}')
    for method in methods:
        if method in ('hard-repair', 'soft-control'):
            cmd = [sys.executable, '-u', 'scripts/run_endpoint_experiments.py', '--task', method,
                   '--gpu', args.gpu, '--gpu-slots', str(args.gpu_slots), '--dataset', args.dataset]
            if args.resume:
                cmd.append('--resume')
            if args.dry_run:
                print(shlex.join(cmd))
            else:
                subprocess.run(cmd, cwd=ROOT, check=True)
            continue
        cmd = command(method, args, checkpoint)
        launch = dict(method=method, command=cmd[1:], code_sha256=fingerprint())
        if checkpoint is not None:
            digest = hashlib.sha256()
            if checkpoint.is_file():
                with checkpoint.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1048576), b''):
                        digest.update(chunk)
            launch['checkpoint_sha256'] = digest.hexdigest()
        ident = hashlib.sha256(json.dumps(launch, sort_keys=True).encode()).hexdigest()[:12]
        name = f'method_{method}_{ident}'
        if args.dry_run:
            print(shlex.join(cmd))
            continue
        with qk.base.job_lock(name):
            complete = []
            for path in (ROOT / 'checkpoints/quant_result').glob(f'*_{name}/method_result.json'):
                result = json.loads(path.read_text(encoding='utf-8'))
                metrics = result.get('after', {})
                valid_metrics = all(isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key])
                                    for key in ('top1', 'top5', 'loss'))
                if result.get('launch') == launch and result.get('status') == 'complete' and valid_metrics:
                    complete.append(path)
            if args.resume and complete:
                print(f'Skipping completed {method}: {complete[-1]}', flush=True)
                continue
            with qk.base.gpu_lock(args.gpu, args.gpu_slots) as uuid:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, FIMAQ_RUN_NAME=name,
                           FIMAQ_CAPTURE_CALIBRATION_HASH='1', FIMAQ_METHOD_LAUNCH=json.dumps(launch))
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
