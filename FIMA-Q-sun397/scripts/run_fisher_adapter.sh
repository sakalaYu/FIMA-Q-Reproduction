#!/usr/bin/env bash
# One-GPU comparison of plain and Fisher-weighted adapter distillation.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi

mkdir -p checkpoints/fisher_probe
exec 9>checkpoints/fisher_probe/gpu.lock
if ! command -v flock >/dev/null 2>&1; then
  echo 'flock is required for the shared single-GPU lock.' >&2
  exit 2
fi
echo 'Waiting for the shared FIMA-Q GPU lock...'
flock 9

checkpoint="${FIMAQ_BASELINE_CHECKPOINT:-./checkpoints/quant_result/20260916_1917/vit_small_w4_a4_optimsize_1024_fisher_dplr_dis_mode_q_rank_5_qdrop.pth}"
dataset="${DATASET:-}"
if [[ -z "$dataset" ]]; then
  for candidate in ../imagenet_fimaq ../../imagenet_fimaq; do
    if [[ -d "$candidate/train" && -d "$candidate/val" ]]; then
      dataset="$candidate"
      break
    fi
  done
fi
if [[ -z "$dataset" || ! -d "$dataset/train" || ! -d "$dataset/val" ]]; then
  echo 'ImageNet train/val directories not found; set DATASET=/path/to/imagenet_fimaq.' >&2
  exit 2
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "W4A4 baseline checkpoint not found: $checkpoint" >&2
  echo 'Set FIMAQ_BASELINE_CHECKPOINT=/path/to/checkpoint.pth.' >&2
  exit 2
fi
common=(--dataset "$dataset" --checkpoint "$checkpoint"
  --device "${DEVICE:-cuda:0}" --seed 3407 --num-workers 4)

case "$mode" in
  smoke)
    python -u scripts/run_fisher_adapter.py "${common[@]}" --mode fisher \
      --blocks 0,5,11 --train-count 32 --holdout-count 16 \
      --fisher-count 16 --batch-size 2 --epochs 1 --max-steps 2 "$@"
    ;;
  smoke_plain)
    python -u scripts/run_fisher_adapter.py "${common[@]}" --mode plain \
      --blocks 0,5,11 --train-count 32 --holdout-count 16 \
      --batch-size 2 --epochs 1 --max-steps 2 "$@"
    ;;
  plain)
    python -u scripts/run_fisher_adapter.py "${common[@]}" --mode plain \
      --blocks all --train-count 960 --holdout-count 64 \
      --batch-size 4 --epochs 3 --full-val "$@"
    ;;
  fisher)
    python -u scripts/run_fisher_adapter.py "${common[@]}" --mode fisher \
      --blocks all --train-count 960 --holdout-count 64 \
      --fisher-count 128 --batch-size 4 --epochs 3 --full-val "$@"
    ;;
  *)
    echo 'Usage: bash scripts/run_fisher_adapter.sh [smoke|smoke_plain|plain|fisher] [extra arguments]' >&2
    exit 2
    ;;
esac
