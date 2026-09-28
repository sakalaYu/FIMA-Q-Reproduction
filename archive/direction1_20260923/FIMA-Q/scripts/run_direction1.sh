#!/usr/bin/env bash
# Purpose: compare legacy FIMA-Q, monitored fixed scheduling and adaptive Fisher.
# Run from any directory: bash scripts/run_direction1.sh monitor
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
code_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$code_dir/.." && pwd)"
mode="${1:-monitor}"
case "$mode" in
  fixed|monitor|adaptive) ;;
  *) echo "Usage: bash scripts/run_direction1.sh [fixed|monitor|adaptive] [extra arguments]"; exit 2 ;;
esac
if [[ $# -gt 0 ]]; then shift; fi
cd "$code_dir"
export FIMAQ_RUN_NAME="direction1_${mode}_seed${SEED:-3407}"
python test_quant.py \
  --model vit_small --config ./configs/4bit/best.py \
  --dataset "${DATASET:-$repo_root/imagenet_fimaq}" \
  --load-calibrate-checkpoint "${CALIB_CHECKPOINT:-./checkpoints/quant_result/20260916_1818/vit_small_w4_a4_calibsize_128_mse.pth}" \
  --optimize --optim-metric fisher_dplr --optim-mode qdrop --drop-prob 0.5 \
  --k 15 --dis-mode q --optim-size 1024 --optim-batch-size 32 \
  --val-batch-size 64 --num-workers 8 --device cuda:0 --seed "${SEED:-3407}" \
  --fisher-schedule "$mode" --probe-size 64 --probe-interval 500 \
  --fisher-error-threshold 0.25 --fisher-patience 2 "$@"
