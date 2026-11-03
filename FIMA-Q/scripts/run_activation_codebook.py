"""Run one 3-bit activation codebook experiment on the saved C-MSE baseline."""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys

from run_headwise_experiments import dataset_path, gpu_lock, job_lock


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'checkpoints/quant_result/20261103_143115_endpoint_qwt_seed3407_endpoint_v1_9b042e92c4c0/endpoint_result.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', choices=('softmax', 'gelu'), required=True)
    parser.add_argument('--checkpoint', type=Path, help='Baked W3A3 C-MSE checkpoint before compensation')
    parser.add_argument('--dataset', default='../imagenet_fimaq')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--gpu-slots', type=int, default=3, choices=(1, 2, 3))
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.checkpoint is None:
        if not REFERENCE.is_file():
            parser.error('Reference result metadata is missing. Pass --checkpoint explicitly.')
        result = json.loads(REFERENCE.read_text(encoding='utf-8'))
        checkpoint = REFERENCE.with_name(PurePosixPath(result['baseline_checkpoint']).name)
    else:
        checkpoint = args.checkpoint.expanduser().resolve()
    data = dataset_path(args.dataset)
    command = [sys.executable, '-u', 'test_quant.py', '--model', 'vit_small', '--config', './configs/3bit/best.py',
               '--dataset', str(data), '--device', 'cuda', '--seed', '3407', '--val-batch-size', '64',
               '--optim-size', '128', '--num-workers', str(args.num_workers), '--load-optimize-checkpoint', str(checkpoint),
               '--test-optimize-checkpoint', '--activation-codebook', args.site]
    if args.dry_run:
        print(shlex.join(command))
        return
    if not checkpoint.is_file():
        parser.error(f'Baseline checkpoint is missing: {checkpoint}')
    if not all((data / split).is_dir() for split in ('train', 'val')):
        parser.error(f'Missing train/val under {data}')
    name = f'activation_codebook_{args.site}_seed3407'
    with job_lock(name), gpu_lock(args.gpu, args.gpu_slots) as uuid:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, FIMAQ_RUN_NAME=name, FIMAQ_CAPTURE_CALIBRATION_HASH='1')
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
