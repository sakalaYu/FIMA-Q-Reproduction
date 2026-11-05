"""Three foreground short controls on the same saved C-MSE+LS checkpoint."""
import argparse
import sys
from pathlib import Path

import run_method_adaptations as methods

BASELINE = methods.ROOT / ('checkpoints/quant_result/20261104_193139_method_lsvit_4373e8b51261/'
                         'vit_small_w3_a3_optimsize_1024_lsvit_dis_mode_q_qdrop.pth')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', type=int, choices=(1, 2, 3), required=True)
    p.add_argument('--checkpoint', type=Path, default=BASELINE)
    p.add_argument('--dataset', default='../imagenet_fimaq')
    p.add_argument('--gpu', default='0')
    p.add_argument('--gpu-slots', type=int, choices=(1, 2, 3), default=3)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    task = {1: 'kd', 2: 'qkcode', 3: 'qkcode_kd'}[args.worker]
    print(f'Worker {args.worker}: {task}; baseline: {args.checkpoint}', flush=True)
    sys.argv = [sys.argv[0], '--method', task, '--checkpoint', str(args.checkpoint),
                '--dataset', args.dataset, '--gpu', args.gpu, '--gpu-slots', str(args.gpu_slots),
                '--kd-steps', '500']
    if args.resume:
        sys.argv.append('--resume')
    if args.dry_run:
        sys.argv.append('--dry-run')
    methods.main()


if __name__ == '__main__':
    main()
