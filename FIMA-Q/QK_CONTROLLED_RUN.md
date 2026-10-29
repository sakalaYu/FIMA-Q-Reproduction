# Q/K 实验统一说明：损失对齐、C 的因素拆分与低维旋转

## 1. FIMA-Q 实际使用的目标

当前 `configs/3bit/best.py` 的 `calib_metric='mse'`，`optim_metric='fisher_dplr'`。

| 阶段 | 原始 FIMA-Q / 固定旋转 | 新 A/B/C |
|---|---|---|
| 初始尺度搜索，128 张 | 量化算子输出的 MSE | 相同的 MSE 搜索 |
| 每块额外 300 步 Q/K 预校准 | 没有这个新增阶段 | attention KL + 块输出目标 + 参数约束 |
| 每块正式 20000 步重建，1024 张 | DPLR + AdaRound 舍入正则 | 相同的 DPLR + AdaRound 舍入正则 |

分类 head 在原代码中另用输出 KL 重建。原始 FIMA-Q 不包含新增 attention KL 预校准。
因此，阶段一致性应理解为：所有方法都保留原始 MSE 初始化和 DPLR 正式重建；主要 A/B/C
比较把新增预校准的块输出目标统一设为 DPLR。原始固定旋转作为没有这段额外优化的参照。

预校准使用 `L = L_attn/L_attn,0 + L_rec + penalty`。选择 MSE 时，
`L_rec = MSE/MSE_0`；选择 DPLR 时，`L_rec = (L_LR/L_LR,0 + L_D/L_D,0)/2`。
后面的正式 FIMA-Q 重建使用配置中的 `p1,p2` 加权两项并加 AdaRound 正则。
预校准 DPLR 在训练部分估计一次 rank 5 anchors，正式重建仍按原流程刷新 Fisher；
它们属于相同的目标形式，但不是共享同一个数值估计器。

`--reconstruction-loss auto` 保留旧默认：A/B=MSE、C=DPLR。以下统一实验全部显式指定目标，
日志、运行名与恢复检查会记录目标，避免 C-MSE 被 C-DPLR 的完成记录跳过。

## 2. 已有证据与尚未完成的归因

| 完整实验 | Top-1 | 能说明什么 |
|---|---:|---|
| 历史固定全部旋转 | 64.650 | 当时 A/B/C 的配对参照 |
| A，旧预校准 MSE | 64.388 | 这个整体设置低于参照 |
| B，旧预校准 MSE | 64.474 | 这个整体设置低于参照 |
| C v1，旧预校准 DPLR | 64.240 | 同时增加旋转并改变损失，无法单独归因 |
| 新一批固定旋转对照 | 64.524 | 同配置与缓存 hash 下的另一轮结果 |
| attention relation uniform | 64.488 | 比新对照低 0.036 pp |
| attention relation Fisher magnitude | 64.056 | 比新对照低 0.468 pp |
| attention relation alignment | 64.066 | 比新对照低 0.458 pp |

attention relation 是另一种干预，不能用它给 C 的学习旋转或 DPLR 单独定责。
截至本次本地核对，没有 C v2、C-MSE、B-DPLR 或低维旋转的完整运行结果。

此前已经完成的实现检查是：Q/K 变换保持浮点函数；临时 fake quant 与折叠后计算一致；
整数零点修复使 Uniform→AdaRound 不再改变相应量化网格。旧代码在零点检查上失败，修复后通过。
这些是数值正确性证据，不能替代完整 Top-1，也不能定量解释旧 C 的 -0.410 pp。

旧 C 接受 10/12 块，且存在仅约 0.1% 留出 KL 改善也接受的块；这提示可测试保守门槛。
短校准候选选择在当前块 AdaRound 之前、后续块仍浮点时进行，提示阶段失配的可能。
目前都属于待检验假设，不是已确认的负增益原因。

## 3. 统一七项实验

所有项都从相同开源浮点模型开始，先应用固定全部 Hadamard，然后用相同 MSE 初始化；
固定 W3A3、ViT-S、seed 3407、Fisher rank 5、QDrop 0.5、每块 20000 步正式 DPLR 重建。
新增预校准均为 300 步、batch 8、lr 0.003，每 50 步评估；896 张优化、最后 128 张选候选。

| 标记 | 预校准可训练内容 | 预校准块输出目标 | 接受门槛 |
|---|---|---|---|
| A_dplr | Q/K 量化尺度 | DPLR | KL 有改善 |
| B_mse | 尺度 + 对偶通道缩放 D | MSE | KL 有改善 |
| B_dplr | 尺度 + D | DPLR | KL 有改善 |
| C_mse | 尺度 + D + 完整正交旋转 U | MSE | KL 有改善 |
| C_dplr | 尺度 + D + 完整 U | DPLR | KL 有改善 |
| C_lowdim | 尺度 + D + 低维有界 U | DPLR | KL 有改善 |
| C_lowdim_gate | 与 C_lowdim 相同 | DPLR | KL 相对改善至少 5% |

全部采用整数零点修复，避免将已知实现问题混入 MSE/DPLR 比较。C_dplr 同时完成上一轮要求的
C v2 修复复验。A_dplr/B_dplr/C_dplr 是统一损失的主要递进比较。

四项 B/C × MSE/DPLR 形成因素拆分：

- `C_mse - B_mse`：MSE 目标下增加学习旋转的效果。
- `C_dplr - B_dplr`：DPLR 目标下增加学习旋转的效果。
- `B_dplr - B_mse`：没有额外旋转时，改变目标的效果。
- `C_dplr - C_mse`：有额外旋转时，改变目标的效果。
- 上述两个旋转效果之差：旋转与损失之间的交互。
- `C_lowdim - C_dplr`：采用低维有界旋转参数化的效果。
- `C_lowdim_gate - C_lowdim`：在同一低维版本上单独增加 5% 门槛的效果。

这些差值是最终方法的配对比较；门控会影响各块是否采用变换，也是方法的一部分。
单种子不能提供统计稳定性结论。统一汇总只对配置、数据缓存 hash 等相同的完整运行计算差值。

## 4. 实现原理与边界

Q/K 采用配对变换 `Q'=Q U D`、`K'=K U D^{-1}`。当 `U` 正交、`D` 可逆时，
浮点注意力分数 `Q'K'^T=QK^T`；量化后的误差会随坐标系与范围改变。
变换折叠到 Q/K 权重和偏置，推理不需要额外旋转模块。

完整 C 使用 Cayley 旋转。低维版本使用 `S=A B^T-B A^T`、
`U=(I-S/2)^{-1}(I+S/2)`，每 head 的因子 rank 为 4，生成元 rank 至多为 8；
这与 Fisher rank 5 是两个不同的参数。每块可训练参数约从 24972 降至 3468。
低维版本同时包含生成元 Frobenius 范数上限 2 的约束，所以其对照检验的是完整的低维有界参数化，
不能将差值全部归因于 rank 这个数字。

5% 门槛按 `(KL_before-KL_best)/max(KL_before,1e-12)` 计算，未达到时保留初始状态。
选择只使用教师概率，不使用 ImageNet 验证标签。原有无门槛低维对照使门槛作用可单独观察。

各块现在额外记录 `selected_local_objective` 和 `teacher_kl_post_round`：前者对应实际选中的候选，
区别于整个搜索中最小的局部损失；后者在真实 20000 步 AdaRound 后，用同一 128 张图片测量教师 KL。
这 128 张随后也参与正式 1024 张重建，重建后探针不是独立留出集。
探针仍使用浮点后缀，只用于辅助判断阶段变化，不能单凭它证明阶段失配的因果。

## 5. 同步与检查

同步以下实现，以及本说明和上一轮说明：

- `utils/qk_joint.py`
- `utils/block_recon.py`
- `test_quant.py`
- `scripts/run_qk_joint_experiments.py`
- `scripts/run_qk_controlled_experiments.py`
- `tests/test_qk_joint.py`
- `tests/test_qk_joint_runner.py`
- `tests/test_qk_controlled_experiments.py`

沿用已有最新 `scripts/run_headwise_experiments.py` 与 `utils/datasets.py`。

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -m unittest discover -s tests -p 'test_qk_*.py' -v
python scripts/run_qk_controlled_experiments.py --suite all --worker 1 --workers 3 --dry-run
```

## 6. 三个 tmux 窗口

单张物理 GPU 0，统一设置 `--gpu-slots 3`。每个窗口依次执行自己的任务，合计七项且不重复。
窗口 1：C_dplr → C_lowdim → C_lowdim_gate。
窗口 2：B_dplr → B_mse。窗口 3：C_mse → A_dplr。

窗口 1：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_qk_controlled_experiments.py --suite all --worker 1 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

窗口 2：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_qk_controlled_experiments.py --suite all --worker 2 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

窗口 3：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_qk_controlled_experiments.py --suite all --worker 3 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

上一轮命令若已经完成 C v2 或低维门槛版，相同参数成功记录会被复用。
`--resume` 是跳过完整成功任务，中断任务会从该实验开头重跑，并非从 AdaRound 中间迭代恢复。
若实际出现 OOM，停止这些任务并把三个命令都改成 `--gpu-slots 1` 重跑，任务将排队。

若先只做 C 归因，将三个命令中的 `--suite all` 都替换为 `--suite causal`，共四项；
以后再跑 all 会跳过已完成的相同设置。不要同时另启上一轮相同实验的命令。

旧 A/B/C 三任务共享 GPU 时，每项约 2 小时 47 分至 2 小时 53 分。
本轮 DPLR 预校准任务较多，七项三窗口运行暂估约 7–12 小时，随显存和实际 GPU 竞争变化；
单任务估算 2.5–4 小时，独占串行七项约 17.5–28 小时。它们是规划估计，不是本轮实测。

## 7. 汇总和下一步判断

```bash
python scripts/run_qk_controlled_experiments.py --summary
```

生成 `checkpoints/qk_joint/controlled_summary_seed3407.json`。同步该 JSON 和七项新的
`logs/*qkjoint*.log` 即可；报错再带对应 console.log。汇总包含各项最终 Top-1、上述差值、
匹配的历史固定旋转对照、逐块记录，以及条件满足时 C v2 相对旧 C v1 的差值。
64.650 与 64.524 两条固定对照都保留，避免基准选择变化掩盖差异。模型权重大文件留在服务器。

结果应按以下证据解释：

- 若 C-MSE 优于 B-MSE，而 C-DPLR 不优于 B-DPLR，重点检查损失与旋转的交互。
- 若两个损失下 C 都更差，现有完整旋转流程在本设置下缺乏收益证据；不能外推为所有学习旋转无效。
- 若低维版更好、加门槛又改善，支持容量约束与候选选择是值得继续的方向。
- 若各版本都低于固定旋转，先暂停扩大参数搜索，根据舍入后探针再设计同一阶段的旋转与舍入协同优化。

本轮主要是可归因的筛选实验。是否形成论文创新还需要稳定增益、机制证据及与现有可学习旋转方法的比较。
