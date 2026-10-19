# Q/K 联合校准 A/B/C：三个 tmux 窗口运行

## 本次实现

三项都是 ViT-S、W3A3、seed 3407、rank=5，从浮点模型开始，应用全部固定 Hadamard 后校准。
每个 Transformer block 在原 AdaRound 重建前增加一个短校准阶段：

| 版本 | 短校准阶段优化内容 | 损失 |
|---|---|---|
| A | 每个 head 的 Q/K 联合尺度因子 | 归一化 attention KL + 归一化 block MSE |
| B | A + 有界 Q/K 对偶通道缩放 | A 的损失 + 缩放约束 |
| C | B + 每个 head 的 Cayley 正交旋转 | attention KL + 任务 DPLR + 变换约束 |

C 的 DPLR 使用全模型教师 KL 的反向梯度和量化扰动估计，包含 diagonal 与 low-rank 两项，
本次 rank=5。该短校准阶段单独估计，使用带 ridge 的 Gram 逆；后续原有 FIMA-Q 的 Fisher
刷新和 20000 步重建照常执行。A/B 使用 MSE 约束 block 输出，避免只追 attention KL。

Q/K 的变换按行向量写为 Q'=QRD、K'=KRD^{-1}，R 从已有 Hadamard 后的单位修正开始，
D 从 I 开始。D 的每个对角元素限制在 [0.5, 2]；每个 head 的尺度因子也限制在 [0.5, 2]。
QKV 权重、QKV 输入激活、Q/K 激活、其余 block 量化都在实际 fake quant 路径内计算。
权重行的尺度/零点随变换后的范围更新，并以同样的状态折叠回普通量化层。
V 和输出投影保留原先固定 Hadamard。

新增变换会折叠到 Q/K 权重和偏置，推理 checkpoint 保持原模块结构。
每个块运行时自动检查浮点等价和折叠后的量化计算等价，失败会停止任务并给出 traceback。

## 默认参数

- 初始量化：128 张图片，batch=32，与已有实验相同。
- 短校准：从现有 1024 张缓存中，前 896 张优化，最后 128 张用于候选选择。
- 每块短校准 300 步，batch=8，Adam lr=0.003；每 50 步评估一次。
- 候选选择：通过当前量化前缀、候选块和浮点后缀，在留出图片上计算完整模型教师输出 KL。
  选择优于初始状态的候选，否则回退；不使用 ImageNet 验证标签选候选。
- 留出图片只在这个短校准阶段留出，随后仍参与原有 1024 张图片的重建；不是独立测试集。
- 原有重建：batch=32、20000 步/块、rank=5、QDrop=0.5，验证 batch=64，num_workers=8。
- 短校准保存和恢复 RNG 状态，保持后续原重建的随机数序列。

## 同步到服务器的文件

必须同步这些实现文件：

1. `test_quant.py`
2. `utils/block_recon.py`
3. `utils/qk_joint.py`（新增）
4. `scripts/run_qk_joint_experiments.py`（新增）
5. `scripts/run_headwise_experiments.py`（旧汇总排除新方法，且复用 GPU 锁与配方）

运行检查还需同步：`tests/test_qk_joint.py`、`tests/test_qk_joint_runner.py`，以及已有
`tests/test_headwise_experiments.py`。保留已有 `utils/datasets.py` 的数据 hash 功能。

## 一次性检查

在服务器 fimaq 环境执行：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -m unittest discover -s tests -p 'test_qk_joint*.py' -v
python scripts/run_qk_joint_experiments.py --variant A --dry-run
```

数值与调度检查共 11 项，应全部通过。dry-run 只打印实际参数，不运行模型。
本地通过了这些 CPU 检查及 17 项旧调度回归检查；本地没有运行完整 ImageNet/CUDA 实验。

## 三个窗口同时运行

沿用一张物理 GPU 0，三个任务共享，统一设置 `--gpu-slots 3`。
每个窗口执行对应的一组命令：

窗口 1 — A：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_qk_joint_experiments.py --variant A --gpu 0 --gpu-slots 3 --resume
```

窗口 2 — B：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_qk_joint_experiments.py --variant B --gpu 0 --gpu-slots 3 --resume
```

窗口 3 — C：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_qk_joint_experiments.py --variant C --gpu 0 --gpu-slots 3 --resume
```

每个窗口跑一个完整实验。若中断，重跑原命令；完整成功的相同参数任务会跳过，失败任务重新开始。
GPU 锁与旧脚本共用，因此旧的独占任务退出后这些任务才能并发。

若需要手动指定参数，在上述命令后增加：

```bash
--qk-joint-steps 300 --qk-joint-batch-size 8 --qk-joint-holdout 128 --qk-joint-lr 0.003 --qk-joint-eval-every 50
```

这是默认值，不需要重复填写。本轮先保持三项参数相同。

## 运行时间

2026-10-17 的三窗口实测：A 从启动到完成约 2 小时 47 分，B 约 2 小时 50 分，C 约 2 小时 53 分。
其中短校准累计分别约 90、112、630 秒，重建计时（含短校准）分别为 9021、9245、9449 秒。
此前 A/B 3–4 小时、C 4–6 小时的规划估计偏保守，后续以这些实测及实际资源竞争为参考。

## 2026-10-17 修复与版本

本轮 v1 结果 A=64.388、B=64.474、C=64.240，都低于固定 all=64.650。
进一步检查发现：C 变换后重新计算的 Q/K 权重量化零点可能是小数。
UniformQuantizer 在 forward 中对零点取整，而 AdaRound 使用存储值；直接复制小数会改变
短校准选择时与 AdaRound 开始时的量化网格。现在折叠 B/C 的 Q/K 状态时明确提交整数零点。
已新增回归检查，旧实现会失败，修复后通过。

新协议标记为 `v2_integer_zp`，运行目录名称也带此后缀，`--resume` 不会跳过为旧版成功任务。
汇总仍保留并注明 v1 历史结果。若要单独确认修复效果，更新代码后仅运行 C 的原命令即可。
这项修复尚未在服务器重跑，不能把它当作已获得的提分，也不能把 A 的负增益归因于此问题。

## 汇总与回传

三项完成后：

```bash
python scripts/run_qk_joint_experiments.py --variant summary
```

汇总文件：`checkpoints/qk_joint/summary_seed3407.json`。
它会读取相同协议的原始 baseline 和固定 all，核对数据 hash/配置，再报告：

- 各版本最终 Top-1；
- 相对固定 all=64.650 的额外增益；
- 相对不旋转 baseline=63.358 的总增益；
- 12 个块中采用了几个块的修改，以及短校准的额外耗时。

若未找到相同输入的参考日志，不计算增益；不会直接用常数强行相减。
不能只因留出 KL 下降，就宣称验证集 Top-1 一定提高。

分析时同步以下小文件即可：

1. `logs/*qkjoint*.log`（3 个成功任务）
2. `checkpoints/qk_joint/summary_seed3407.json`
3. 若报错，同步 `checkpoints/qk_joint/*_console.log`，其中包含 stderr/traceback。
4. 可选：各任务目录内的 `qk_joint.json`，保存逐块结果；日志也有同样的逐块记录。

checkpoint 留在服务器，用于后续重评；日志中的最终完整验证结果才用于判断本轮是否提分。
