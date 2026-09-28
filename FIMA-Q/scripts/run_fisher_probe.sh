#!/usr/bin/env bash
# Single-GPU sequential diagnostics. No training, no existing checkpoint edits.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
mkdir -p checkpoints/fisher_probe
# All invocations of THIS launcher on the same host share one lock.
# It cannot prevent unrelated GPU programs from being started elsewhere.
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
case "$mode" in
  smoke)
    python -u scripts/probe_fisher.py "${common[@]}" --stage both \
      --cache checkpoints/fisher_probe/smoke_images.pt --basis-size 4 --eval-size 2 \
      --ranks 2 --blocks 0 --epsilons 0.001 --test-directions 1 "$@" \
      2>&1 | tee "checkpoints/fisher_probe/${stamp}_smoke.log"
    ;;
  pilot)
    python -u scripts/probe_fisher.py "${common[@]}" --stage both \
      --cache checkpoints/fisher_probe/pilot_images.pt --basis-size 16 --eval-size 4 \
      --ranks 4 8 --blocks 0 5 11 "$@" \
      2>&1 | tee "checkpoints/fisher_probe/${stamp}_pilot.log"
    ;;
  full)
    python -u scripts/probe_fisher.py "${common[@]}" --stage both \
      --cache checkpoints/fisher_probe/full_images.pt --basis-size 48 --eval-size 16 \
      --ranks 4 8 16 --blocks all "$@" \
      2>&1 | tee "checkpoints/fisher_probe/${stamp}_full.log"
    ;;
  *) echo 'Usage: bash scripts/run_fisher_probe.sh [smoke|pilot|full] [extra arguments]' >&2; exit 2 ;;
esac
