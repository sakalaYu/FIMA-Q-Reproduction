# W3A3：在 B/C-MSE 上增加可靠性加权语义探针蒸馏

## 目标与研究边界

当前参照为固定全部 Hadamard 64.650、B-MSE 64.726、C-MSE 64.758（ViT-S、seed 3407）。
这次增加的是正式 DPLR 重建阶段的中间语义蒸馏，不修改 B/C 的 MSE 短预校准，不增加训练标签。
不是直接加载旧量化 checkpoint 继续训练：每项从同一浮点模型、同一校准配方完整运行。

FIMA-Q 的教师输出 KL 已用于估计 Fisher，B/C 的短预校准已有 attention KL。
新增约束针对的是每块输出 CLS 所承载的类别信息。无需在每一步运行教师后缀；
冻结语义探针只在校准训练时存在，推理模型不增加模块。

KD 与量化、CLS 特征蒸馏都已有相关工作，不能宣称首次结合。当前的小创新候选是：
用留出校准样本衡量各块语义探针超过常量教师先验的预测能力，再连续调整蒸馏强度。
是否具有足够新颖性及最终精度收益需要相关工作比较和实验支持，不能保证超过 64.758。

## 实现

1. 教师为相同架构的浮点模型。沿用已缓存的每块 FP CLS 特征和整网 FP softmax(logits/20)。
   将缓存概率恢复成中心化 logits，不依赖 ImageNet 类别标签。
2. 每块前 896 张图用于 ridge 线性回归，拟合标准化 CLS 特征到教师中心化 logits 的读出器。
   默认 ridge=0.1 乘特征协方差的平均对角线，求解后冻结。
3. 最后 128 张仅用于估计读出器可靠性。令 p 为教师在 T=2 的概率，p_prior 为训练部分教师平均概率：

   `r_b = clip(1 - mean KL(p || p_probe) / mean KL(p || p_prior), 0, 1)`。

   当先验 KL 接近零时令 r_b=0。该连续权重没有引入上一轮失败的 5% 候选接受门槛。
4. 蒸馏目标使用同一冻结读出器在 FP CLS 上输出的概率，学生使用量化块 CLS：

   `L_KD = T^2 KL(softmax(P_b(z_FP)/T) || softmax(P_b(z_quant)/T))`。

   不直接强迫中间学生预测去拟合真实最终 logits，避免让学生补偿探针自身的预测误差。
   当 FP/学生特征相同，该项为零；可靠性仍根据探针对真实教师最终预测的拟合能力计算。
5. 原正式重建损失变为：

   `L = L_DPLR + L_round + 0.05 * ramp(t) * r_b * L_KD / max(L_KD,first, 1e-4)`。

   等权消融使用 r_b=1。前 20% 迭代不加蒸馏，20%–40% 线性升权，此后保持权重。
   初始 KD 归一化值来自第一次激活时的重建 batch，随后固定。
   只更新原本的 AdaRound alpha 与激活量化尺度，探针无梯度参数、不进入优化器。

留出图片随后仍参与原来的 1024 张正式重建；它们不是独立最终评测集。
该方法是冻结语义代理蒸馏，不等于逐步运行完整浮点后缀的标准整网 logits KD。
线性代理可能无法表达浅层类别语义，可靠性为零的块不蒸馏。日志会记录这种情况。
本轮不使用强行可靠性下限；若全部或多数块权重为零，应先分析探针拟合能力。

## 三项对照

| worker | 底座 | 蒸馏方式 | 作用 |
|---|---|---|---|
| 1 | C-MSE | 等权语义探针 KD | 检验增加该语义约束本身是否有效 |
| 2 | C-MSE | 留出可靠性加权 KD | 与 worker 1 比较可靠性机制；与 64.758 比较新增益 |
| 3 | B-MSE | 留出可靠性加权 KD | 检验蒸馏是否仍需要 C 中的额外学习旋转；与 64.726 比较 |

统一 W3A3、ViT-S、seed 3407、rank 5、QDrop 0.5、每块 20000 步、原 Q/K 预校准 300 步。
不需要重跑以前七项。历史参照要求数据 hash 和配置匹配，代码运行差异仍需谨慎解释；
若得到的增益很小，最终需要相同环境下的配对确认。

## 同步和一次性检查

同步这次修改或新增的文件：

- `test_quant.py`
- `utils/block_recon.py`
- `utils/semantic_probe_kd.py`
- `scripts/run_probe_kd_experiments.py`
- `tests/test_semantic_probe_kd.py`
- `tests/test_probe_kd_runner.py`

保留已有最新的 `utils/qk_joint.py`、`scripts/run_qk_joint_experiments.py`、
`scripts/run_headwise_experiments.py`、`utils/datasets.py` 和原检查依赖文件。

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -m unittest discover -s tests -p 'test_semantic_probe_kd.py' -v
python -m unittest discover -s tests -p 'test_probe_kd_runner.py' -v
python scripts/run_probe_kd_experiments.py --worker 2 --dry-run
```

本地新增 8 项检查包括：拟合不消耗 RNG、探针有效性、FP 零误差、真实 AdaRound alpha 梯度、
等权/可靠性配对、常量教师回退、任务覆盖和完整结果恢复判断。旧 Q/K 检查也通过。
这不等于完整 ImageNet/CUDA 验证。本地尚无这次新增方法的 Top-1。

## 三个 tmux 窗口

窗口 1（C-MSE + 等权探针 KD）：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_probe_kd_experiments.py --worker 1 --gpu 0 --gpu-slots 3 --resume
```

窗口 2（C-MSE + 可靠性加权 KD）：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_probe_kd_experiments.py --worker 2 --gpu 0 --gpu-slots 3 --resume
```

窗口 3（B-MSE + 可靠性加权 KD）：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_probe_kd_experiments.py --worker 3 --gpu 0 --gpu-slots 3 --resume
```

每个窗口仅一项，单 GPU 0 共享三槽。若 OOM，停止这些任务并统一改成 --gpu-slots 1 排队。
--resume 只跳过完整成功且参数、源代码指纹相同的任务；中断任务从该实验开头重跑。
源代码指纹覆盖入口、重建、Q/K 实现、KD 模块和新 runner，防止改过算法后仍复用旧结果。
运行期间保持代码固定。不要与旧七项任务重复并行。

按原任务约 2 小时 40 分至 3 小时 25 分/项估计，这轮先规划三窗口共约 3–4.5 小时；
探针拟合及额外线性投影的 CUDA 耗时尚未实测，实际随 GPU/CPU 竞争变化。

## 汇总与结果解释

```bash
python scripts/run_probe_kd_experiments.py --summary
```

保存于 `checkpoints/probe_kd/summary_seed3407_*.json`，名称含参数及源代码指纹。
回传该 JSON、三条 `logs/*probekd*.log`，出错再带 `checkpoints/probe_kd/*console.log`。
可选同步各运行目录的 `probe_kd.json`；模型大文件留在服务器。

报告包含每块 r_b、教师先验/探针 KL、实际激活步数和 KD 量级，避免把未真正生效的配置误判为无收益。
先看 worker 2 是否超过 C-MSE，以及是否超过 worker 1；若只超过历史底座而不超过等权 KD，
就不能声称可靠性机制有贡献。worker 3 若与 worker 2 持平或更好，优先保留较简单的 B 路线。
即便可靠性版优于等权版，也可能部分来自总蒸馏强度降低。若初筛获益，需要再补按平均可靠性
匹配总权重的等权对照，才能更明确区分“按块分配可靠性”与“单纯降低损失系数”的作用。
若所有版本均未改善，暂停增加蒸馏损失权重，先检查探针可靠性与激活记录。不要把降低代理 KL 当作最终 Top-1 提升。
