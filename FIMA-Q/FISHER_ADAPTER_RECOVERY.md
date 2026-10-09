# Fisher 加权低秩误差校正：优先追求精度增益的验证实验

## 核心方案

从已有 ViT-S W4A4 FIMA-Q checkpoint 出发，冻结全部原有量化权重和量化尺度。
在每个 Transformer block 的 Attention 与 MLP 分支输出后、残差相加前，插入
零初始化的 rank-4 校正：

```text
corrected(h) = h + W_up W_down h
```

每个宽度为 384 的分支增加 `2*384*4=3072` 个参数，24 个分支共 73,728 个
参数。初始化时模型输出与基线逐值相同。使用 960 张 ImageNet train 图像做
无标签教师蒸馏，64 张互不重叠的 train 图像选择 checkpoint。训练标签不用作
优化目标；验证标签只用于报告准确率。

训练目标为完整模型输出 KL 加分支特征重建。`plain` 使用普通特征均方误差；
`fisher` 用教师预测分布抽取伪类别，计算分支输出的梯度平方平均，为通道特征
损失赋权。两种模式共用相同图像清单、rank、训练轮数、checkpoint 和随机种子。
最佳 epoch 仅按保留集的教师 KL 选择，再在 ImageNet val 上评估。结果包含
`experiment.json`、`metrics.jsonl`、`best_adapters.pt` 和 `last_adapters.pt`。

这条路线最可能快速恢复 Top-1，但额外引入浮点校正参数与推理计算，必须
同时报告参数、实际时延。CVPR 2026 Workshop 已有低秩 adapter 修复 PTQ 的
工作；“使用 adapter”不是新颖性主张。真正需要验证的是：Fisher 通道权重
是否在固定参数和数据预算下稳定超过普通 adapter。

## 服务器命令：逐项手动运行

先同步 `utils/fisher_adapter.py`、`scripts/run_fisher_adapter.py`、
`tests/test_fisher_adapter.py`。在已有 tmux 会话中开新窗口，进入窗口后运行：

```bash
tmux new-window -n fisher-adapter
```

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
python -m unittest discover -s tests -p test_fisher_adapter.py -v
```

先做 Fisher smoke，检查 checkpoint 加载、反向传播和结果保存：

```bash
flock checkpoints/fisher_probe/gpu.lock python -u scripts/run_fisher_adapter.py \
  --mode fisher --dataset ../imagenet_fimaq \
  --checkpoint ./checkpoints/quant_result/20260916_1917/vit_small_w4_a4_optimsize_1024_fisher_dplr_dis_mode_q_rank_5_qdrop.pth \
  --blocks 0,5,11 --rank 4 --train-count 32 --holdout-count 16 \
  --fisher-count 16 --batch-size 2 --epochs 1 --max-steps 2 \
  --num-workers 4 --seed 3407 --device cuda:0
```

smoke 成功后，分别启动两个完整对照。先运行 `plain`：

```bash
flock checkpoints/fisher_probe/gpu.lock python -u scripts/run_fisher_adapter.py \
  --mode plain --dataset ../imagenet_fimaq \
  --checkpoint ./checkpoints/quant_result/20260916_1917/vit_small_w4_a4_optimsize_1024_fisher_dplr_dis_mode_q_rank_5_qdrop.pth \
  --blocks all --rank 4 --train-count 960 --holdout-count 64 \
  --batch-size 4 --epochs 3 --num-workers 4 --seed 3407 \
  --device cuda:0 --full-val
```

检查 plain 完成后，再运行 `fisher`：

```bash
flock checkpoints/fisher_probe/gpu.lock python -u scripts/run_fisher_adapter.py \
  --mode fisher --dataset ../imagenet_fimaq \
  --checkpoint ./checkpoints/quant_result/20260916_1917/vit_small_w4_a4_optimsize_1024_fisher_dplr_dis_mode_q_rank_5_qdrop.pth \
  --blocks all --rank 4 --train-count 960 --holdout-count 64 \
  --fisher-count 128 --batch-size 4 --epochs 3 --num-workers 4 \
  --seed 3407 --device cuda:0 --full-val
```

若数据目录或 checkpoint 不同，只替换对应参数。`flock` 与其余 Fisher 实验
共用单 GPU 锁。每次结果在 `checkpoints/fisher_adapter/results/时间_模式/`。
请先核对 `imagenet_val_baseline` 是否约为当前本地基线的 76.442%；若明显不符，
先排查模型、预处理、checkpoint，不能用该实验作比较。

## 判断与下一步

先比较 `plain` 和 `fisher` 的 ImageNet val Top-1、教师 KL 及保留集走势。
只有 Fisher 版本在相同预算下稳定超过普通 adapter，才能把 Fisher 赋权视作
有效贡献；单次结果仅用于决定是否扩展到多 seed / W3A3 / 其他模型。

如果两种 adapter 都不能恢复精度，停止增加校正参数，转向**早层有限位移
KL 校正**：前期诊断显示第 0 块 Fisher 二次型对真实 KL 的预测误差很高，而
中后层较低。后续只针对误差大的早层，用少量真实教师 KL 更新替代纯局部
DPLR 代理，再在固定参数、固定 bit-width 下比较准确率与额外训练时间。

本地只有 checkpoint，没有 ImageNet 图像和 `timm`，因此只做了模块单测、
语法和入口检查。完整 smoke 和精度结论必须在服务器产生。
