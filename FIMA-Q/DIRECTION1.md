# 方向一：可靠性驱动的 Fisher 更新（第一版）

本次代码用于验证：DPLR 代理目标与真实输出 KL 失配时，重新估计 Fisher 是否比固定间隔更新更有效。
目前是可运行实验实现，不代表已经取得精度提升。仅校准阶段增加计算，不增加推理模块。

## 每个文件是做什么的

| 文件 | 用途 |
| --- | --- |
| `utils/fisher_reliability.py` | 可靠性判断器：拟合正比例系数，检测误差增大/相反趋势，用连续异常次数触发更新。无 PyTorch 依赖。 |
| `utils/block_recon.py` | 在原逐块重构循环接入确定性探针、Fisher 更新调度和 JSONL 诊断记录。 |
| `test_quant.py` | 提供命令行开关，拆分重构/探针数据，保存实验配置。 |
| `scripts/run_direction1.sh` | 在已有 Linux 训练环境运行三个对照模式。默认使用你的 ViT-S W4A4 校准 checkpoint。 |
| `scripts/summarize_fisher.py` | 汇总每块的真实更新次数、触发次数和诊断开销，不负责训练。 |
| `tests/test_fisher_reliability.py` | 检查趋势失配、误差失配、连续触发和重置行为。 |
| `tests/test_fisher_probe.py` | 用 CPU 小模型检查真实探针代码的公式、确定性和状态恢复；不等于完整 ViT 测试。 |
| `tests/test_fisher_schedule.py` | 用小型量化器替身运行真实重构循环，检查 monitor 不改变训练结果、adaptive 的触发与预算上限。 |

## 算法与边界

`fixed`：默认，保留原始 Fisher 更新间隔，无探针，使用全部 optim-size 数据。

`monitor`：保持原始更新间隔，增加探针诊断；不按探针触发更新。

`adaptive`：初始化后，只在连续失配时追加 Fisher 方向。第一版不替换旧方向，不自动优化秩。

每次 Fisher 更新后，先取得实际训练损失的两项归一化分母，然后在探针上得到逐图代理量 q 和 KL y。
拟合正比例系数 c = sum(q*y)/sum(q*q)，并冻结到下次 Fisher 更新。定义相对误差
E = sqrt(sum((c*q-y)^2)/sum(y^2))。若 E 比更新后的锚点误差高 0.25，或代理均值下降超过 2% 而 KL 均值上升超过 2%，则判为失配。
连续两次失配触发更新。以上是待消融的初始超参数，不是论文验证过的最优值。

探针关闭 QDrop、保留当前 soft AdaRound 权重，使用 FP 前缀与 FP 后缀，只有当前块量化。
这是与现有 Fisher 估计位置对应的 isolated-block KL，不是全量化模型 KL，也不是最终 hard-round 模型精度。
温度沿用 cfg.temp（默认 20）；KL 使用 float64 降低小数值误差。
探针不执行 backward，不更改损失计数或归一化分母，并恢复 mode、drop_prob、PyTorch RNG。

原始代码采用 floor(iters/k) 间隔；20000 步、k=15 实际调用 16 次。
fixed/monitor 保留此行为，adaptive 使用相同的实际调用次数上限（不是保证用完预算）。
因此不能只用 k 当作真实秩/计算次数，应查看日志。第一版仍使用原始矩阵求逆；奇异/病态矩阵问题没有被暗中换成另一种估计器。

## 数据公平性

monitor/adaptive 在 seed 选出的同一份 1024 张训练校准池中，前 960 张用于重构和 Fisher 估计，后 64 张作为探针；缓存后不重复随机增强。
不使用 ImageNet 验证集做触发决策。探针会影响调度，因此属于算法使用的校准数据，而非独立最终测试集。
两种模式的拆分一致，可直接比较；fixed 默认使用 1024 张重构数据，属于历史复现参考，不能单独用于归因。

初始校准通常使用同一 seed 的前 128 张样本，程序要求重构集至少保留 calib-size 张。
加载外部 checkpoint 时，必须确认它实际使用同一训练目录、样本顺序、seed 和校准规模；程序无法从旧 checkpoint 自动证明无重叠。

## 怎么运行

以下命令均在服务器的 `FIMA-Q` 目录运行，使用原本能完成复现的 Python/PyTorch/timm 环境。

```bash
# 先只监测：确认代理失配是否真实存在，固定间隔基线也留出同样的探针。
bash scripts/run_direction1.sh monitor

# 再运行自适应更新，其他默认条件一致。
bash scripts/run_direction1.sh adaptive

# 可选：原始 1024 张、无探针的历史基线。
bash scripts/run_direction1.sh fixed
```

数据或 checkpoint 在其他位置时，设置 DATASET / CALIB_CHECKPOINT 环境变量：

```bash
DATASET=/root/autodl-tmp/imagenet_fimaq bash scripts/run_direction1.sh monitor
```

下面是链路冒烟实验，只验证流程，不能用其准确率评价算法：

```bash
bash scripts/run_direction1.sh adaptive --recon-iters 100 --k 3 --probe-interval 10
```

已有入口也可直接追加 `--fisher-schedule monitor` 或 `--fisher-schedule adaptive`。
更换 W3A3 时，必须同时切换 3bit 配置和匹配的 3bit 校准 checkpoint；不能加载 W4A4 checkpoint 后仅改 bit 参数。

## 输出与判断标准

每次运行单独保存在 `checkpoints/quant_result/时间_运行名/`：

- `experiment.json`：实际命令行参数和配置。
- `probe_split.json`：校准池中的拆分位置和 seed。
- `fisher_diagnostics.jsonl`：refresh / probe / summary 事件。
- `output.log`：原有训练和精度日志；结束后仍复制到 logs。
- 原格式的优化 checkpoint：可按原入口加载；新增诊断模块不进入 state_dict。

```bash
# 将 RUN 替换为实际实验目录名；这个命令只读取和汇总日志。
python scripts/summarize_fisher.py checkpoints/quant_result/RUN/fisher_diagnostics.jsonl

# 这些测试不需要 ImageNet；探针测试需要 PyTorch。
python -m unittest discover -s tests -v
```

优先比较 monitor/adaptive 的 Top-1、实际 Fisher 次数、整次运行耗时、探针耗时、异常次数；正式结果至少三个种子。
相同的是最大 Fisher 调用预算，不是实际耗时；adaptive 可能少用预算。若精度提高但时间也明显增加，应报告准确率—成本权衡。
如果 monitor 中几乎没有持续失配，不应通过反复调小阈值制造创新动机。

后续才考虑方向筛选/替换、条件数控制和更严格的等耗时实验；不要同时修改这些变量来混淆第一版结论。
