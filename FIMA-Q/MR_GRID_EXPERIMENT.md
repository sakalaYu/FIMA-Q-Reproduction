# MR 网格稳健性实验 v1

本轮检验 MR 的训练目标，而非训练步数分配。固定 ViT-S W3A3、逐头 Hadamard、等权 LS、seed 3408、每块 3,000 次更新。新入口只增加 scripts 下的文件，不修改已有 test_quant.py、utils 或 gain 入口。

## 三项比较

| 版本 | 前 2,000 步 | 后 1,000 步 | 最终权重选择 |
|---|---|---|---|
| uniform | 原浮点＋2×截断输出 MSE | 继续原目标 | 第 3,000 步 |
| single | 同上 | 原目标＋单网格 A3 输出 MSE | 留出集名义 A3 误差最低，且浮点误差不超过第 2,000 步的 1.1 倍 |
| robust | 同上 | 原目标＋三个网格 A3 输出 MSE 的最大值 | 留出集最坏 A3 误差最低，且满足相同浮点约束 |

共同使用原 Adam、4e-5 学习率、3,000 步余弦调度、batch 4 和相同采样流。共同起点的权重、优化器、调度器和采样状态有指纹检查。候选权重可以选回第 2,000 步，但每项均执行每块 3,000 次更新，报告 attempted_steps 和 selected_step。更新次数相同不表示 FLOPs 或耗时相同。

量化辅助项权重在第 2,001～2,500 步从 0 线性升至 0.25，之后固定。名义 A3 网格从拟合集前 32 张图的当前 ReLU 激活拟合，每 250 步刷新。三个尺度为名义尺度的 0.9、1.0、1.1 倍，零点固定为整数 0。只在 fc2 输入上模拟 A3，使用 STE；不是完整 W3A3 路径。正式重建仍重新校准权重和激活尺度，并做每块 20,000 步 LS 重建。

训练使用 896 张校准图，检查点选择使用另外 128 张。正式 LS 使用原有的 1,024 张校准图，因此这 128 张不是整个量化流程的独立验证集。ImageNet 验证标签不参与本轮筛查。

## 运行

在三个终端分别运行以下一条。均在前台显示进度。

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_mr_grid.py --variant uniform --stage screen --seed 3408 --gpu 0 --gpu-slots 3 --resume
```

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_mr_grid.py --variant single --stage screen --seed 3408 --gpu 0 --gpu-slots 3 --resume
```

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_mr_grid.py --variant robust --stage screen --seed 3408 --gpu 0 --gpu-slots 3 --resume
```

三项完成后运行：

```bash
python -u scripts/run_mr_grid.py --variant summary --seed 3408
```

筛查预设标准是候选的平均名义 A3 误差至少低于本轮 uniform 2%，平均最坏网格误差下降，并且整网浮点教师 KL 不超过 uniform 的 1.1 倍。此标准仅用于节约后续算力，不能证明 Top-1 提升。

如果通过，再将 uniform 和通过的候选各跑一次 `--stage quant`。例如：

```bash
python -u scripts/run_mr_grid.py --variant uniform --stage quant --seed 3408 --gpu 0 --gpu-slots 3 --resume
python -u scripts/run_mr_grid.py --variant robust --stage quant --seed 3408 --gpu 0 --gpu-slots 3 --resume
python -u scripts/run_mr_grid.py --variant summary --seed 3408
```

结果在 `checkpoints/lsvit_controls/mr_grid/`。screen 完成标记为 `[SCREEN COMPLETE]`，不包含最终 ImageNet Top-1；quant 完成后才有 Top-1。`--resume` 复用完整阶段，未完成的阶段重新训练。运行期间保持代码、环境及 FP 权重路径不变。

## 贡献边界

GELU→ReLU MR 来自 APHQ-ViT，普通与截断输出的当前监督形式来自 GLF-Q。STE、量化感知监督、多目标约束及稳健优化均已有研究基础。single 是机制对照；robust 是本项目拟检验的“邻近网格最坏误差＋浮点拟合约束”组合方案，尚未确认学术独立性或整网收益。

uniform 对比 single 同时包含量化监督和检查点选择的影响；single 对比 robust 检验邻近网格目标与对应选择准则的整体影响。若有效，再补 robust 去掉浮点约束等消融，进一步区分机制。
