#!/usr/bin/env bash
# Matched W4A4 baseline/rotation run for ViT-S. Only one GPU process at a time.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-smoke}"
mkdir -p checkpoints/fisher_probe
exec 9>checkpoints/fisher_probe/gpu.lock
if ! command -v flock >/dev/null 2>&1; then
  echo 'flock is required for the single-GPU experiment.' >&2
  exit 2
fi
echo 'Waiting for the shared FIMA-Q GPU lock...'
flock 9

common=(--model vit_small --config ./configs/4bit/best.py
  --dataset "${DATASET:-../imagenet_fimaq}" --device "${DEVICE:-cuda:0}"
  --calibrate --optimize --val-batch-size 64 --num-workers 8 --seed 3407 --print-freq 10)

case "$mode" in
  smoke|baseline-smoke)
    export FIMAQ_RUN_NAME=headwise_hadamard_baseline_smoke
    python -u test_quant.py "${common[@]}" --calib-size 16 --calib-batch-size 8 \
      --optim-size 16 --optim-batch-size 8 --recon-iters 2 --skip-final-validation
    ;;
  rotation-smoke)
    export FIMAQ_RUN_NAME=headwise_hadamard_rotation_smoke
    python -u test_quant.py "${common[@]}" --headwise-hadamard --calib-size 16 \
      --calib-batch-size 8 --optim-size 16 --optim-batch-size 8 --recon-iters 2 \
      --skip-final-validation
    ;;
  baseline)
    export FIMAQ_RUN_NAME=headwise_hadamard_baseline_w4a4
    python -u test_quant.py "${common[@]}"
    ;;
  rotation)
    export FIMAQ_RUN_NAME=headwise_hadamard_rotation_w4a4
    python -u test_quant.py "${common[@]}" --headwise-hadamard
    ;;
  *)
    echo 'Usage: bash scripts/run_headwise_hadamard.sh [smoke|rotation-smoke|baseline|rotation]' >&2
    exit 2
    ;;
esac
