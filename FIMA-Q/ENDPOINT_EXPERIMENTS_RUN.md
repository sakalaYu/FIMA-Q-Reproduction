# C-MSE 后续：rank、硬量化尺度修复与 QwT

## 11 月 3 日后续三窗安排

本轮窗口 2 跑修订后的硬量化尺度修复和额外软训练对照。窗口 1 和 3 分别只替换
post-Softmax 和 post-GELU 的 3-bit 激活码本。三个窗口都在 `FIMA-Q` 目录中运行，使用同一 GPU 锁。

```bash
# 窗口 1
python -u scripts/run_activation_codebook.py --site softmax --gpu 0 --gpu-slots 3
# 窗口 2
python -u scripts/run_endpoint_experiments.py --task all --worker 2 --gpu 0 --gpu-slots 3 --resume
# 窗口 3
python -u scripts/run_activation_codebook.py --site gelu --gpu 0 --gpu-slots 3
```

窗口 1 和 3 默认从 `20261103_143115_endpoint_qwt_seed3407_endpoint_v1_9b042e92c4c0`
的 `endpoint_result.json` 读取 `baseline_checkpoint` 文件名，并从该目录加载补偿前的 `.pth`。
文件放在其他位置时，用 `--checkpoint` 指定补偿前的模型。每窗先复测同一个基线，
再用 64 张训练图挑选码本、另 64 张图按全模型教师 KL 筛选，最后全量验证。
Softmax 码本有一个零和七个对数间隔的正值，GELU 码本在八个值中给负半轴分配一到三个值。
64 张门控样本从本阶段的码本拟合中留出，最终精度以完整验证集为准。
结果存于 `checkpoints/quant_result/*_activation_codebook_*/activation_codebook_result.json`。

本轮基于 ViT-S / W3A3 / seed 3407 / C-MSE 预校准。历史参照为 64.758%，
正式 DPLR 为 rank 5、20000 步。所有任务关闭 probe KD 和额外 attention relation loss。
本实现尚无服务器完整准确率结果；本地为 CPU 数值及运行器检查。

## 实验安排

| 窗口 | 任务 | 实际内容 |
|---|---|---|
| 1 | rank10 | C-MSE 不变，仅正式 DPLR 的 rank 从 5 改为 10 |
| 2，第一项 | hard-repair | 每个 Transformer 块原 20000 步完成后，增加 2000 步硬量化尺度修复 |
| 2，第二项 | soft-control | 独立重新训练；每块原 20000 步后继续 2000 步原软舍入 DPLR＋QDrop |
| 3 | qwt | 独立重新训练 C-MSE，先验证无补偿基线，再拟合 QwT 并再次全量验证 |

因此总计 **4 个完整重建任务**，由 **3 个窗口**执行。窗口 2 两项顺序执行，
同一窗口不会同时启动两份模型。patch embedding 和分类 head 仍用原 20000 步，
修复、延长训练和 QwT 都只施加于 12 个 Transformer 块。

窗口 3 的补偿前结果是本批次 C-MSE 基线，不预先假设它一定等于历史 64.758%。
汇总同时报告历史差值与本批次匹配基线差值；比较前检查源代码、FP 权重和校准增强张量 hash。

## 代码

- `utils/hard_scale_repair.py`：路径诊断、有界尺度修复、额外训练对照。
- `utils/qwt_compensation.py`：流式线性回归补偿、FP16 分支、checkpoint 包装层重建。
- `utils/block_recon.py`：在每块原重建之后调用可选阶段。
- `test_quant.py`：参数、教师副本、补偿前后验证、模型与结果保存。
- `scripts/run_endpoint_experiments.py`：三窗口调度、GPU 锁、任务指纹、完成性检查和汇总。
- `tests/test_endpoint_numerics.py`、`tests/test_endpoint_runner.py`：CPU 检查。

同步以上文件及现有依赖到服务器后再启动任务。不要在本轮运行中途修改算法代码；
代码变化会产生新的实验指纹，旧结果不会被当作同一配置复用。
无须安装新的第三方包，使用现有 fimaq 环境。

## 硬量化修复原理

1. 在相同的 128 张校准图上测量 `soft_full`、`hard_full`、`soft_qdrop`、`hard_qdrop` 的块 MSE。
   两次 QDrop 掩码取均值，软/硬路径使用配对随机掩码；这些检查保存并恢复 RNG。
   `full` 使用实际已量化前缀输入，内部量化概率为 1；`qdrop` 同时保留原输入混合与内部随机旁路。
2. 固定权重、权重量化尺度/零点与 AdaRound 舍入变量，关闭 QDrop，使用硬舍入。
3. 对每个原有激活尺度优化 `s_new = s_old * exp(delta)`，限制倍数在 `[1/1.2, 1.2]`。
   log 倍数 Adam LR=0.001，batch=8，2000 步；目标为块输出 MSE，不添加 KD 或新的 Fisher 查询。
4. 修复训练使用前 896 张，每 200 步在末 128 张上选最小硬路径 MSE，并包含零修改候选。
   末 128 张只从本次修复更新中留出，**之前已参与原正式重建**，不是全流程独立留出集。
5. 把选中的倍数折叠回原尺度，删除训练参数化；保存后不增加推理模块。
   代码校验固定参数未改变，且折叠后误差与候选选择一致。

当前先对所有 12 块使用统一修复预算，**尚未实现按切换误差分配步数**。
需要先根据本轮诊断与准确率决定是否值得继续。

soft-control 的额外阶段与修复使用同一数量的样本、batch=8、同样的抽样序列和 2000 次更新，
保留原 DPLR＋rounding 正则＋QDrop，使用最后一次 Fisher，不新增 Fisher 查询。
保留 Adam 状态、激活 LR 从原 4e-5 重启并余弦降至零，rounding 温度固定为原终值 2。
这属于**匹配额外更新次数与样本暴露的对照**，不是严格匹配 FLOPs 或墙钟时间，也不能单独分解目标、
路径和参数集合三个因素。它检验“原训练继续更新是否已经足以产生同样收益”。

## QwT 原理与边界

参考 [QwT 官方代码](https://github.com/wujx2001/QwT) 的闭式线性补偿机制：
`y = quant_block(x) + Linear(x)`。使用现有 1024 张校准缓存的前 512 张，无额外标签或校准图。
按块依次拟合；教师与量化块看到同一个“已补偿的量化前缀输入”，目标为两者局部输出差。
使用原 FP 教师副本，避免把已经烘焙的低比特权重误当成 FP 教师。

本集成用 FP64 CPU 流式统计量及最小范数解代替显式求逆，ridge 默认为 0；
训练 R² 不大于 0 时使用零补偿。分支权重和偏置默认 FP16，最终验证真实运行此分支。
这不是新增创新，也不是对 QwT 论文结果的逐项复现。

**它是 W3A3 主干＋FP16 补偿，不能报告为纯 W3A3。**
每块完整 D×D 线性分支；384 维、12 块时增加 1,774,080 个数值参数，
FP16 约 3.38 MiB，197 tokens 时约 348,585,984 MAC/image。
结果文件记录实际增加的参数、存储字节和 MAC，不把它描述为无推理代价。
没有进行 QwT 额外一轮训练。

## 运行

每个窗口先进入同一环境：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
```

可先做本轮 CPU 数值和运行器检查：

```bash
python -m unittest discover -s tests -p 'test_endpoint_*.py' -v
```

如需先检查服务器完整接口，在正式任务前串行执行以下三条。
它们使用 16 张正式校准图、2 步重建，不验证准确率，不能代替正式实验。

```bash
python -u scripts/run_endpoint_experiments.py --task hard-repair --smoke --gpu 0 --gpu-slots 1 --resume
python -u scripts/run_endpoint_experiments.py --task soft-control --smoke --gpu 0 --gpu-slots 1 --resume
python -u scripts/run_endpoint_experiments.py --task qwt --smoke --gpu 0 --gpu-slots 1 --resume
```

正式窗口 1：

```bash
python -u scripts/run_endpoint_experiments.py --task all --worker 1 --gpu 0 --gpu-slots 3 --resume
```

正式窗口 2，自动先修复、后延长训练对照：

```bash
python -u scripts/run_endpoint_experiments.py --task all --worker 2 --gpu 0 --gpu-slots 3 --resume
```

正式窗口 3，先新 C-MSE 基线、再 QwT：

```bash
python -u scripts/run_endpoint_experiments.py --task all --worker 3 --gpu 0 --gpu-slots 3 --resume
```

三窗口的 `--gpu-slots` 必须相同。GPU0 可共享 3 份任务并不保证任意配置都不 OOM。
若出现 OOM，停止本轮尚在运行的进程，将三条命令统一改为 `--gpu-slots 1` 后重新执行，任务会排队。
已完成且配置/代码匹配的任务会跳过，失败任务重新开始。

`--resume` 是**任务级复用**，没有实现中断块/优化器步数恢复。
窗口 2 第一项已成功、第二项中断时，会保留第一项并重新开始第二项。
QwT 补偿前的基线 checkpoint 会保留，但当前运行器不会自动从它恢复中断的补偿阶段。

独立运行或重试某一项可使用 `--task rank10`、`--task hard-repair`、`--task soft-control`、`--task qwt`。
只做路径诊断可用 `--task diagnose`，仍需要先完成 C-MSE 重建。

汇总：

```bash
python scripts/run_endpoint_experiments.py --task summary
```

## 输出与同步

如果窗口 2 在硬量化修复入口报 `Repair requires finite positive activation scales`，
同步修订后的 `utils/hard_scale_repair.py` 和 `scripts/run_endpoint_experiments.py`。
建议等正在运行的旧版窗口 1、3 结束后再同步，因为旧运行器结束时会重新读取磁盘代码指纹。
用原窗口 2 命令重启。修复将有限非零的原尺度乘以有界正倍数，保留各元素符号，
并在日志中按量化器记录负尺度个数。零值或非有限值会报告具体的块和量化器。

若窗口 1、3 使用修复前的代码完成，运行以下汇总以纳入它们。此模式只读取经过核对的
修复前 rank10 和 QwT 结果，仍要求模型、校准数据、运行配置和依赖版本匹配。

```bash
python scripts/run_endpoint_experiments.py --task summary --include-pre-fix
```

- `checkpoints/endpoint_experiments/`：启动配置、console 日志、汇总 JSON。
- 对应 `checkpoints/quant_result/*_endpoint_*/`：`endpoint_result.json`、`qk_joint.json`、
  修复/对照的 `endpoint_blocks.json`，或者补偿的 `qwt.json`。
- `logs/*endpoint*.log`：完整日志。
- `.pth`：模型文件，QwT 前后的 checkpoint 分开保存。

同步结果供分析时保留上述 JSON 与日志即可；大 `.pth` 无须加入 Git，但请保留在服务器。
判定收益优先看本批次 C-MSE 差值，以及 hard-repair 相对 soft-control 的差值。
不要仅根据局部 MSE 下降或历史 64.758% 差值宣布方法有效。

## 时间

现有 2026-11-02 三任务共享 GPU 日志中，C+KD 从 16:55:37 到 19:44:18，约 2 小时 49 分；
这只是历史参照，不是新代码的实测时间。rank10 增加 Fisher 查询，修复/对照增加更新，
QwT 增加拟合和一次完整验证。按相同机器可先预留 6–9 小时覆盖本轮 4 项；
实际用时受并发、CPU 回归和验证速度影响，日志会记录各阶段耗时。
