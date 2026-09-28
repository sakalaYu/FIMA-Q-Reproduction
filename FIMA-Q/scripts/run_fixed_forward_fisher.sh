#!/usr/bin/env bash
# Single-GPU launcher for the fixed-epsilon, true-error directional experiment.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi

mkdir -p checkpoints/fisher_probe/fixed_results
exec 9>checkpoints/fisher_probe/gpu.lock
if command -v flock >/dev/null 2>&1; then
  echo 'Waiting for the Fisher diagnostic GPU lock...'
  flock 9
else
  echo 'flock is required to prevent overlapping runs on the single GPU.' >&2
  exit 2
fi

common=(--dataset "${DATASET:-../imagenet_fimaq}" --device "${DEVICE:-cuda:0}"
  --checkpoint "${CALIB_CHECKPOINT:-./checkpoints/quant_result/20260916_1818/vit_small_w4_a4_calibsize_128_mse.pth}")
stamp="$(date +%Y%m%d_%H%M%S)_$$"
result="checkpoints/fisher_probe/fixed_results/${stamp}_${mode}"
log="checkpoints/fisher_probe/${stamp}_fixed_${mode}.log"

case "$mode" in
  smoke)
    python -u scripts/probe_fixed_forward_fisher.py "${common[@]}" \
      --cache checkpoints/fisher_probe/fixed_smoke_images.pt --images 4 --blocks 0 \
      "$@" --epsilon 0.001 --groups 4 --output-dir "$result" 2>&1 | tee "$log"
    ;;
  pilot)
    python -u scripts/probe_fixed_forward_fisher.py "${common[@]}" \
      --cache checkpoints/fisher_probe/fixed_pilot_images.pt --images 32 --blocks 0 3 5 8 11 \
      "$@" --epsilon 0.001 --groups 4 --output-dir "$result" 2>&1 | tee "$log"
    ;;
  full)
    python -u scripts/probe_fixed_forward_fisher.py "${common[@]}" \
      --cache checkpoints/fisher_probe/fixed_full_images.pt --images 64 --blocks all \
      "$@" --epsilon 0.001 --groups 4 --output-dir "$result" 2>&1 | tee "$log"
    ;;
  secant_smoke)
    python -u scripts/probe_fixed_forward_fisher.py "${common[@]}" \
      --cache checkpoints/fisher_probe/fixed_smoke_images.pt --images 4 --blocks 0 \
      "$@" --epsilon 0.001 --groups 4 --variants single \
      --scales 0.5 0.75 1.0 1.25 1.5 --output-dir "$result" 2>&1 | tee "$log"
    ;;
  secant_pilot)
    python -u scripts/probe_fixed_forward_fisher.py "${common[@]}" \
      --cache checkpoints/fisher_probe/fixed_pilot_images.pt --images 32 --blocks 0 3 5 8 11 \
      "$@" --epsilon 0.001 --groups 4 --variants single \
      --scales 0.5 0.75 1.0 1.25 1.5 --output-dir "$result" 2>&1 | tee "$log"
    ;;
  *)
    echo 'Usage: bash scripts/run_fixed_forward_fisher.sh [smoke|pilot|full|secant_smoke|secant_pilot] [extra arguments]' >&2
    exit 2
    ;;
esac

python scripts/summarize_fixed_forward_fisher.py "$result"
echo "Results: $result"
echo "Terminal log: $log"
