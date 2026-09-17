#!/usr/bin/env bash
set -e

# Usage:
#   bash scripts/run.sh verify   # verify the saved k=5 checkpoint
#   bash scripts/run.sh k15      # run FIMA-Q with k=15
#   bash scripts/run.sh mse      # run the MSE + QDrop baseline

script_dir="$(cd "$(dirname "$0")" && pwd)"
code_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$code_dir/.." && pwd)"

cd "$code_dir"

common_args=(
  --model vit_small
  --config ./configs/4bit/best.py
  --dataset "$repo_root/imagenet_fimaq"
  --optim-size 1024
  --optim-batch-size 32
  --val-batch-size 64
  --num-workers 8
  --device cuda:0
  --seed 3407
)

case "${1:-}" in
  verify)
    FIMAQ_RUN_NAME=verify_vit_small_w4a4_fisher_dplr_k5_seed3407 python test_quant.py "${common_args[@]}" \
      --load-optimize-checkpoint ./checkpoints/quant_result/20260916_1917/vit_small_w4_a4_optimsize_1024_fisher_dplr_dis_mode_q_rank_5_qdrop.pth \
      --test-optimize-checkpoint
    ;;
  k15)
    FIMAQ_RUN_NAME=vit_small_w4a4_fisher_dplr_k15_qdrop_seed3407 python test_quant.py "${common_args[@]}" \
      --load-calibrate-checkpoint ./checkpoints/quant_result/20260916_1818/vit_small_w4_a4_calibsize_128_mse.pth \
      --optimize \
      --optim-metric fisher_dplr \
      --optim-mode qdrop \
      --drop-prob 0.5 \
      --k 15
    ;;
  mse)
    FIMAQ_RUN_NAME=vit_small_w4a4_mse_qdrop_seed3407 python test_quant.py "${common_args[@]}" \
      --load-calibrate-checkpoint ./checkpoints/quant_result/20260916_1818/vit_small_w4_a4_calibsize_128_mse.pth \
      --optimize \
      --optim-metric mse \
      --optim-mode qdrop \
      --drop-prob 0.5
    ;;
  *)
    echo "请选择一个实验：verify、k15 或 mse"
    echo "例如：bash scripts/run.sh verify"
    exit 2
    ;;
esac
