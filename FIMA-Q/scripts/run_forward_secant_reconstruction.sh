#!/usr/bin/env bash
# W4A4 forward-secant block reconstruction; preserves the fisher_dplr baseline.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi

mkdir -p checkpoints/fisher_probe
exec 9>checkpoints/fisher_probe/gpu.lock
if command -v flock >/dev/null 2>&1; then
  echo 'Waiting for the shared FIMA-Q GPU lock...'
  flock 9
else
  echo 'flock is required for the single-GPU experiment.' >&2
  exit 2
fi

checkpoint="${CALIB_CHECKPOINT:-./checkpoints/quant_result/20260916_1818/vit_small_w4_a4_calibsize_128_mse.pth}"
common=(--model vit_small --config ./configs/4bit/forward_secant.py
  --dataset "${DATASET:-../imagenet_fimaq}" --device "${DEVICE:-cuda:0}"
  --load-calibrate-checkpoint "$checkpoint" --optimize
  --val-batch-size 64 --num-workers 8 --seed 3407 --print-freq 10)

case "$mode" in
  smoke)
    export FIMAQ_RUN_NAME=forward_secant_smoke
    python -u test_quant.py "${common[@]}" "$@" --optim-size 64 --optim-batch-size 32 \
      --recon-iters 2 --skip-final-validation
    ;;
  full)
    export FIMAQ_RUN_NAME=forward_secant_w4a4
    python -u test_quant.py "${common[@]}" "$@" --optim-size 1024 --optim-batch-size 32 \
      --recon-iters 20000
    ;;
  *)
    echo 'Usage: bash scripts/run_forward_secant_reconstruction.sh [smoke|full] [extra arguments]' >&2
    exit 2
    ;;
esac
