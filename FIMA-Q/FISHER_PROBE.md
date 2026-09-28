# 新路线：结构感知 Fisher + 仅前向 Fisher 估计

本版是第一阶段**诊断与估计验证代码**，不是已经验证有效的新量化训练算法。它不修改权重，不运行 AdaRound 重构，不输出新模型精度。先验证结构差异与估计可靠性，再决定如何修改重构损失。

## 旧代码回退情况

方法 1 的新增源码、测试、两组实验结果与分析已移动到项目根目录 `archive/direction1_20260923/`。原 `test_quant.py`、`utils/block_recon.py` 恢复到提交 `4d5e8ea`。原始 checkpoint、基线日志和 ImageNet 不受影响；Git 历史没有 reset。

## 新文件分别做什么

| 文件 | 用途 |
|---|---|
| `scripts/probe_fisher.py` | 准备共享校准图像缓存、加载校准 checkpoint、执行两个研究方向的诊断。 |
| `utils/fisher_probe_model.py` | 缓存标准 ViT/DeiT 分支上下文，重放 FP 后续网络，避免每个扰动都重跑前缀。 |
| `utils/fisher_geometry.py` | 计算误差子空间、自动微分/前向差分响应及投影 Fisher。 |
| `scripts/run_fisher_probe.sh` | 单张 GPU 的 smoke/pilot/full 顺序入口，带 flock 锁和终端日志保存。 |
| `scripts/summarize_fisher_probe.py` | 将结果汇总为 CSV，可选生成 Fisher 谱 PNG 图。 |
| `tests/test_fisher_geometry.py` | 用解析例子验证子空间、Fisher 公式、前向差分和二次型。 |
| `tests/test_fisher_suffix.py` | 验证 Attention/MLP 后缀和完整前向一致、可微、hooks/模式正确恢复。 |

## 实验定义

### 结构诊断

标准 ViT-S 默认选第 0、5、11 块，分别捕获 Attention 和 MLP 输出（LayerScale/残差相加之前）。
固定 FP 输入，仅量化被测分支，获取 signed quantization errors。使用 basis 图像的误差构造未中心化 PCA 子空间 U，不取绝对值。

保留两个不同的谱：

1. 误差矩阵 DeltaZ 的奇异值谱：误差集中在哪些方向。
2. 子空间内 `U F U^T` 的特征值谱：这些方向在当前预测任务下的敏感性。

**任何一个都不能标成完整 Fisher 的谱。** 子空间维数是样本数与 max rank 的上限；rank95 是当前测量空间内的 95% 能量维数，不代表网络真实 Fisher rank。

`projected_offdiag_ratio` 是误差 PCA 坐标系内的非对角占比，不是原始 token/channel 坐标中的非对角占比。基于它不能直接决定原始 DPLR 的 alpha。
Attention/MLP 在各自的局部 FP 状态下测量，不包含跨分支 Fisher 交叉项，不能据此声称两个模块完全独立。

严格确定性模式下，谱矩阵仍在 GPU 计算；仅将已经得到的特征值复制到 CPU 计算累计能量/rank95。这样避开部分 PyTorch 版本缺失的 deterministic CUDA cumsum kernel，不改变 Fisher 矩阵或特征值。

### 仅前向估计

令 f(a)=logits(h+U^T a)/T，V 是 f 对低维系数 a 的 Jacobian。
构造 `F_sub = V^T (diag(p)-p p^T) V`，其中 p=softmax(logits/T)。

- autograd：PyTorch `autograd.functional.jvp`，作为相同方向上的自动微分参考。
- finite_difference：`[f(+s e_j)-f(-s e_j)]/(2s)`，在 no_grad 下计算，不执行 backward。
- s=epsilon*||h||_2，U 的行向量单位正交，所以是相对激活范数的扰动幅度。
- 默认 epsilon 为 1e-4、1e-3、1e-2；不预设哪个最好。
- FP32 网络，关闭 TF32；投影矩阵、KL 和差分相减使用 double 做数值汇总。极小扰动仍可能有 FP32 消减误差。

`both` 一次运行共享缓存和自动微分参考，完成两个方向的研究，避免分别运行重复计算。整个比较程序会做自动微分，但**有限差分估计分支本身仅前向**。

耗时针对构造响应 V 的步骤，GPU 同步并预热；峰值显存同时给出总 allocated 和增量 allocated。不是显卡全部 reserved 显存，也不是整个量化任务耗时。
自动微分参考为通用 JVP（可能通过双反向实现），不是已优化 FIMA-Q 实现；不能用其对比直接宣称比原 FIMA-Q 快。
前向差分每个 epsilon、每张图像需要 2*k 次后缀前向，k 增大可能更慢。
默认多个 rank 共用最大 rank 的响应，耗时记录对应最大 rank，不是每个截断 rank 的独立计时。

### 未见样本与预测测试

训练目录中随机选取固定图像，采用确定性评估预处理（没有随机 crop/flip），保存归一化图像张量及其 SHA-256。图像仍来自 train，不读取 val。
basis 和 eval 图像不重叠。对 eval 图像测试：

- 前向估计与自动微分的相对矩阵误差/响应误差。
- 未用于估计的随机组合方向引起的真实 KL，与全投影/对角/半秩二次型预测。
- eval 图像真实量化误差被子空间覆盖的能量比例。
- 真实量化误差和其子空间投影分别引起的 KL。投影二次型预测对应**投影误差 KL**，不能与完整误差混淆。

“对角/半秩”消融均在同一投影空间内，不等同于原论文的通道对角和 DPLR 实现。保持位宽一致；本版不做混合精度，也不直接推断最佳 k/alpha。

## 服务器怎么运行（单张 RTX 4090D）

先同步新文件和两个恢复后的原始入口，然后进入代码目录：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
```

先做 smoke：6 张训练图像（4 basis + 2 eval），第 0 块的 Attention/MLP，2 个方向、一个 epsilon，验证整个链路：

```bash
bash scripts/run_fisher_probe.sh smoke
```

成功后做 pilot：20 张（16+4），第 0、5、11 块，4/8 维子空间和三个 epsilon，先检查是否值得扩大：

```bash
bash scripts/run_fisher_probe.sh pilot
```

先分析 pilot，再决定是否跑 full：64 张（48+16），全部 12 个块，4/8/16 维子空间。暂不提供未经实测的运行时间保证。

```bash
bash scripts/run_fisher_probe.sh full
```

不要一次启动三个训练进程。这个入口的 flock 锁会让同机、同项目下的其他同入口任务等待，但不阻止其他脚本占用 GPU。
脚本使用 `pipefail`，异常会返回非零退出码；每次创建独立结果目录，旧结果不会覆盖。

默认读取原来的 W4A4 **校准** checkpoint。它保留 FP 权重和校准参数，适合比较 raw 与 quantized；不要使用已硬舍入的 optimize checkpoint。

```bash
# 路径不同时用环境变量修改；不会改动原文件。
DATASET=/root/autodl-tmp/imagenet_fimaq \
CALIB_CHECKPOINT=/path/to/vit_small_w4_a4_calibsize_128_mse.pth \
bash scripts/run_fisher_probe.sh pilot
```

可使用 `--stage structure` 仅做结构/自动微分诊断，或 `--stage forward` 做前向与参考对照。`both` 同时保存两类结果，推荐默认模式。
首次生成缓存后，同一路径必须使用相同样本规模、seed、模型及预处理设置；不一致会拒绝运行，改用新的 --cache 路径即可。

本版支持标准 timm ViT/DeiT，不支持 Swin、蒸馏双 head 或任意变种。每张 eval 图像会自动验证后缀重放与完整模型 logits 一致；不一致直接停止，避免悄悄测错网络。

## 结果文件与分析

控制台打印 `Results: ...`，结果在 `checkpoints/fisher_probe/results/时间/`。

| 文件 | 内容 |
|---|---|
| `experiment.json` | 参数、配置、输入样本路径/编号、数据张量/权重/代码哈希、torch/timm/GPU 信息 |
| `diagnostics.jsonl` | 各模块误差谱、参考谱、前向误差、时间/显存、KL 预测 |
| `blocks.*.attn.pt` / `blocks.*.mlp.pt` | 子空间基、逐样本参考矩阵和前向估计矩阵，不是模型 checkpoint |
| `complete.json` | 成功结束标志与诊断循环耗时；缺少它不能当成完整结果 |

```bash
# RUN 替换为控制台显示的实际目录；此脚本只汇总结果。
python scripts/summarize_fisher_probe.py checkpoints/fisher_probe/results/RUN

# 已安装 matplotlib 时可额外生成谱图。
python scripts/summarize_fisher_probe.py checkpoints/fisher_probe/results/RUN --plot
```

`matplotlib` 是可选依赖。环境中没有它时，汇总脚本仍会保留两个 CSV、提示跳过 PNG，并正常退出；无需为了分析 CSV 单独安装。

会生成 `summary.csv` 与 `prediction_errors.csv`。后者分别汇总随机组合方向、真实量化误差投影的预测误差，不能合并当成一个指标。

返回实验结果时先发 experiment.json、diagnostics.jsonl、complete.json、两个 CSV 和入口生成的终端 log。不用传共享图像缓存和模型权重。

## 如何决定下一步

1. 若误差子空间对 eval 图像真实误差覆盖很差，先改方向选择，不能仅降低 rank。
2. 若 finite difference 随 epsilon 改变剧烈，先解决步长/精度问题，再讨论仅前向重构。
3. 若估计矩阵接近参考，但二次型仍不能预测真实 KL，应检查局部近似范围，不要把误差全部归因于 Fisher 估计。
4. 若 Attention/MLP 差异只出现在单个块或少量图像，应多 seed/更多样本验证后再主张结构异质性。
5. 只有诊断成立，才将选定的近似接入原始重构过程，比较准确率与校准成本。本版没有承诺精度提升。

## 本地验证

```bash
# 解析测试与后缀逻辑测试，不需要 ImageNet。
python -m unittest discover -s tests -v
```

另有项目根目录 `tmp/smoke_fisher_probe.py` 用合成图像、真实校准权重做 CPU 链路验证，仅供本地开发，不能把其输出当作论文结果。
