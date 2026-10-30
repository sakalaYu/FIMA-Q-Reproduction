# W3A3 局部教师修复：代码与运行

## 本轮实现范围

默认从本地实际保留的固定全部 Hadamard W3A3 checkpoint 开始，历史 Top-1 为 64.650%。
本地没有 B-MSE 最终权重，不能把日志当作 checkpoint。服务器保留 B-MSE 时可以显式切换，见下文。
冻结旋转、权重及其量化尺度、零点、LayerNorm 和分类头。
逐块更新已有的激活量化尺度，包括线性层输入及两个 attention matmul 的输入尺度。
每个尺度限制在本块开始时的 `[0.9, 1.1]` 倍范围内。没有新增部署层，没有重新学习旋转，也没有重启 AdaRound。

教师使用同一个原始浮点 ViT-S checkpoint。已有 Hadamard、Q/K 变换和 LayerNorm 重参数化保持完整 block 的浮点输入输出函数，
因此原始浮点 block 可以作用在量化前缀的 h_Q 上。不能把已硬化的量化权重关闭量化后冒充浮点教师。

局部教师：`R(T_b(h_Q))`；学生：`R(Q_b(h_Q))`。E4 的 R 为浮点后续网络，E5/E6 的 R 为量化后续网络及量化头。
两条路径共享相同的量化前缀输入。训练前向保持全量化，不使用 QDrop；量化后续网络参数冻结，但用 STE 传递输入梯度。
每块开始时缓存输入、浮点块输出及教师 logits，块内不移动教师；切到下一块时重新生成缓存。
顺序固定为 block 0 到 11，不按诊断结果选块，不按 holdout 损失选择最佳步数。

## 实验定义

| 编号 | 新增修复目标 | 作用 |
|---|---|---|
| E0 / prepare | 不训练；完整验证起点 | 确认 checkpoint 加载后的精度 |
| E1 | 固定 DPLR 约束 | 同更新步数的第二阶段尺度修复对照 |
| E2 | E1 + 完整浮点教师普通 KL | 普通 KD 对照 |
| E3 | E1 + 完整浮点教师 Logit Standardization KD | 第一轮强 KD 对照 |
| E4 | E1 + 局部教师，师生共用浮点后续网络 | 局部教师的浮点路径对照 |
| E5 | E1 + 局部教师，师生共用量化后续网络 | 核心路径假设 |
| E6 | E5 + E3 的完整教师约束 | 检查完整教师是否弥补局部教师偏差 |

E4/E5 的局部 KL 公式、温度、权重、参数集合、抽样种子和更新数完全相同，只改变冻结后续网络。
不同方法更新前面的块后，后面块收到的量化输入会自然不同；它们不是共享一份全程不变的输入。
E1/E2/E3/E4/E5/E6 每块都更新 300 步，batch 8。相同更新数不等于相同 GPU 时间；E1 便宜得多，时间单独报告。

### DPLR 的准确含义

这里是新的 **固定 DPLR 约束、尺度修复第二阶段**，不是原始 FIMA-Q 20000 步 progressive AdaRound 的完全续训。
每块用 128 张训练样本估计 rank 5：当前块以 0.5 activation QDrop 生成五组梯度/扰动，
梯度来自浮点后续网络与完整浮点教师的 KL；abs-error、abs-gradient 形式与项目现有实现一致。
Gram 增加相对 `1e-6` ridge；低秩/对角项按初始全量化误差分别归一化，然后在本块修复期间固定。
所有训练组采用同一估计程序，不把“重新构造一个 Fisher 方法”混作本轮创新。
E1 只能回答新增监督是否超过相同步数的该尺度修复，不能当成额外 20000 步原始 FIMA-Q 对照。

### 数据与标签

校准图像按 `np.random.RandomState(3407)` 排序：前 1024 张训练，之后 128 张只记录诊断。
原实验按同样索引约定使用前 1024 张，因此额外的 128 张不在原 1024 图像索引中。
本轮使用确定性的教师评估预处理，不尝试恢复上一轮已丢失的随机增强张量；所有新实验共享同一缓存和 hash。
训练目标、抽样、参数更新及固定步数均不读取真实标签。probe 中的分类统计只作诊断，不影响训练。
ImageNet val 只在准备阶段和每项训练结束后评估，不用于早停、门控或选择模型。

## 同步文件

必须同步四个新文件：

- `configs/local_teacher_w3a3.json`
- `utils/local_teacher.py`
- `scripts/run_local_teacher.py`
- `scripts/run_local_teacher_experiments.py`

建议同时同步本说明及 `tests/test_local_teacher.py`。
沿用服务器已有的 fimaq 环境、quantizers、quant_layers、utils/wrap_net.py 和 run_headwise_experiments.py。

默认 checkpoint（已在本地确认存在）：

```
checkpoints/quant_result/20261012_180903_headwise_w3a3_seed3407_rotation_seeded_full/vit_small_w3_a3_optimsize_1024_fisher_dplr_dis_mode_q_rank_5_qdrop.pth
```

准备阶段完整验证，默认要求 Top-1 与 64.650 相差不超过 0.03 pp；不符合时停止训练，先排查加载和配置。
配置里的 checkpoint、dataset 等相对路径均相对于 `FIMA-Q` 项目根目录。

### v3：修复准备阶段的验证设置差异

2026-10-30 服务器以旧版设置（batch 32、确定性 cuDNN、cuDNN TF32 关闭）测得 64.594%，
与历史 64.650% 相差 0.056 pp，即净少正确 28 张。训练尚未开始，不能把它解释为蒸馏精度下降。
原日志记录的验证 batch 为 64。旧脚本未记录全部后端旗标，因此不能仅凭设置差异断言已找到全部原因。
本版把完整验证与缓存前向的 batch 分开：`eval_batch_size=32` 仍用于缓存；
`validation_batch_size=64` 用于完整 ImageNet 验证，`validation_profile=legacy` 显式设置
cuDNN benchmark=False、deterministic=False、cuDNN TF32=True、matmul TF32=False。
全部实验使用同一套预先固定的验证设置；验证完成后恢复训练的严格数值设置。

没有放宽 0.03 pp 阈值，也没有把期望基线改成已测得的较低值。
准备阶段额外保存 `baseline_check.json`（即使基线不匹配也保存）：评估配置、预处理、版本及教师/学生 hash。
更新代码后需重新 prepare；新指纹与之前失败的目录分开，已经缓存的浮点教师会复用。
如果仍不匹配，运行无需训练的对照：

```bash
python -u scripts/run_local_teacher_experiments.py --suite audit --workers 1 --gpu 0 --gpu-slots 1 --resume
```

它固定依次比较 legacy/batch64、strict/batch64、strict/batch32，保存 `evaluation_audit.json`；
后两项记录相对第一项的逐图预测分歧、修复/破坏。不会选择得分最高的配置，也不会自动开始训练。
若 prepare 成功，直接按下文运行 screen 即可，无需额外 audit。

### 不依赖 vit_raw 目录的教师加载

JSON 的 `teacher_checkpoint` 默认为 `null`。启动器先找项目中的原始浮点 `.safetensors` 或旧 `.bin`；
本地实际有 `checkpoints/vit_raw/model.safetensors`，不是之前错误写死的 `vit_small_patch16_224.bin`。
服务器没有该目录也可以运行：找不到本地教师时，使用原 `test_quant.py` 同名的
`timm.create_model('vit_small_patch16_224', pretrained=True)`，先读 timm/Hugging Face 缓存，缓存缺失才下载。
只在 prepare/check 时解析一次，保存到 `checkpoints/local_teacher/assets/vit_small_patch16_224_fp.pth`。
此后所有窗口固定使用解析后的教师文件和 hash，不在每个任务中重新选择教师。

也可以用 `--teacher-checkpoint /实际路径/model.safetensors` 或 `.pth/.bin` 指定教师；显式路径错误会停止，
不会悄悄换模型。`.safetensors` 需要 fimaq 环境安装 safetensors；缺失时会给出安装提示。
学生 checkpoint 缺失时直接停止；不会自动替换成其他精度或其他实验的学生。

## 运行

可先做 CPU 加载检查（不使用 GPU、不评估 ImageNet；首次可能解析/下载教师）：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_local_teacher_experiments.py --suite check
```

先在一个窗口运行一次准备，成功后再启动三个筛选窗口：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_local_teacher_experiments.py --suite prepare --workers 1 --gpu 0 --gpu-slots 1 --resume
```

可选但建议：先做无需训练的路径诊断。只评估缓存图像，不进行十二次完整验证。

```bash
python -u scripts/run_local_teacher_experiments.py --suite probe --workers 1 --gpu 0 --gpu-slots 1 --resume
```

三个 tmux 窗口各自进入上述目录并激活 fimaq，然后分别运行：

```bash
# 窗口 1：E3，标准化的完整教师 KD
python -u scripts/run_local_teacher_experiments.py --suite screen --worker 1 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

```bash
# 窗口 2：E4，共享浮点后续网络的局部教师
python -u scripts/run_local_teacher_experiments.py --suite screen --worker 2 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

```bash
# 窗口 3：E5，共享量化后续网络的局部教师
python -u scripts/run_local_teacher_experiments.py --suite screen --worker 3 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

这是三个进程共享物理 GPU 0，程序按 UUID 限制槽位。后续网络需要反传，旧实验可并行三个不保证本轮也装得下。
若 OOM，停止本轮其他存活进程，将上述三条的 `--gpu-slots 3` **统一**改为 `--gpu-slots 1`，重新运行。
这只改变并发，不改变实验身份；三个窗口会排队并恢复各自进度。不必先改 batch size。
GPU 锁只约束使用本项目同一锁协议的进程，不会管住另外手工运行的 Python 任务。

筛选结束，补其余三个对照：同样在三个窗口运行，把 `--suite screen` 改为 `--suite controls`。
窗口 1/2/3 分别为 E1/E2/E6。也可用 `--suite all`，同配置成功结果自动复用。

```bash
python -u scripts/run_local_teacher_experiments.py --suite controls --worker 1 --workers 3 --gpu 0 --gpu-slots 3 --resume
python -u scripts/run_local_teacher_experiments.py --suite controls --worker 2 --workers 3 --gpu 0 --gpu-slots 3 --resume
python -u scripts/run_local_teacher_experiments.py --suite controls --worker 3 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

以上三条仍是各窗口各运行一条。

汇总（不申请 GPU）：

```bash
python scripts/run_local_teacher_experiments.py --suite summary
```

### 如果服务器保留了 B-MSE 权重

可另外建立一轮 B-MSE 起点的实验。先运行：

```bash
python -u scripts/run_local_teacher_experiments.py --suite prepare --workers 1 --gpu 0 --gpu-slots 1 --resume \
  --checkpoint checkpoints/quant_result/20261029_191402_qkjoint_w3a3_seed3407_B_seeded_full_v2_integer_zp_lossmse/vit_small_w3_a3_optimsize_1024_fisher_dplr_dis_mode_q_rank_5_qdrop.pth \
  --expected-baseline-top1 64.726
```

启动器会显示 `Resolved config: .../resolved_config.json`。
后续 probe、screen、controls、summary 全部加 `--config` 指向它，例如：

```bash
python -u scripts/run_local_teacher_experiments.py --config checkpoints/local_teacher/实际指纹/resolved_config.json --suite screen --worker 1 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

窗口 2/3 分别改成 `--worker 2` / `--worker 3`。不能只在 prepare 切换 B-MSE，之后却使用默认固定旋转配置。
不同起点分目录记录，分别计算增益，不能把两种起点混在一次配对比较中。

## 中断恢复与配置

每 50 步及每块结束原子保存 `progress.pt`：完整模型、当前块位置/步数、优化器、抽样生成器、固定 DPLR 和尺度边界中心。
用原命令加 `--resume`（上面已加）继续，最多重算最近不足 50 步；不会重新做整项 12 个块。
若中断在最后验证，重跑最后验证。配置或代码变化会生成新的指纹目录，不会把新算法接到旧优化器上。
不要在一轮并行期间修改本轮源码、JSON 或输入 checkpoint。

训练参数在 JSON 中修改。第一轮不建议一起改学习率、步数、尺度边界和损失权重。
先看 E5-E4、E6-E5、E6-E3，再决定是否开放舍入；本版不支持权重舍入或额外旋转训练。

## 结果与 Git

输出目录 `checkpoints/local_teacher/<指纹>/`：

- `prepared.json`：原始完整验证、数据 hash、预处理、准备时间和显存。
- `baseline_check.json` / 可选 `evaluation_audit.json`：基线不匹配时的独立评估证据。
- `sources.json` / `sources_worker*.json`：源码、配置、原始教师及学生 checkpoint hash。
- `probe/probe.json`：两条局部教师路径的修复/破坏、分歧和输出梯度比较；小样本诊断，不代替完整验证。
- `E*/result.json`：最终完整验证、增益、修复/破坏、耗时、改变的张量、模型参数量。
- `E*/metrics.jsonl`：逐块 DPLR 条件数、更新损失、梯度、holdout KL、耗时和显存。
- `summary.json`、`*_console.log`：汇总与排错。

分析优先同步这些小文件。`.pt/.pth` 已被项目 .gitignore 排除：

- `data.pt` 是约 0.7 GB 的固定图像与教师 logits 缓存。
- `E*/progress.pt` 是续跑必需文件，留在服务器。
- `E*/model.pth` 是可再次加载的最终 state_dict。
- `baseline_predictions.pt` / `E*/predictions.pt` 保存逐图预测，留作需要时的配对分析。

state_dict 仍以浮点容器保存 fake-quant 模型，不能把其磁盘大小当成打包后的 INT3 模型大小。
本轮核对只有既有激活尺度改变，部署参数数目与 W/A 配置不增加。

## 验证范围

本地 CPU 十四项检查使用真实项目 W3A3 quantizer，覆盖 STE 前向与输入梯度、局部路径差异、标准化 KD、
固定 DPLR、尺度边界、三窗口任务划分、中断恢复后与连续训练参数完全一致，以及缺失路径、
safetensors 加载、无 vit_raw 目录时的教师一次缓存、切换学生时的基线检查。
另覆盖验证后端设置及异常恢复、按整数正确数统计精度、固定三种评估对照不按准确率选择配置。
另外在临时依赖目录安装 timm 0.9.16，实际严格加载本地 64.650% checkpoint 与原始 safetensors 教师，
以随机图像验证完整 ViT 的 STE 前向、拆分前向均为零误差，E3/E4/E5 对当前块尺度的梯度均非零且有限。
未运行服务器 GPU/ImageNet 完整实验，不能预先宣称提分。临时依赖不需要同步到服务器。
首次准备阶段会检查实际模型拆分、STE 数值和完整基线精度。

```bash
python -m unittest discover -s tests -p test_local_teacher.py -v
```
