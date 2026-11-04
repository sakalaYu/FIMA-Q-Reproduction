# C-MSE 上的 KD、AdaLog、LS-ViT 和 ERQ 适配实验

本轮固定 ViT-S、seed 3407、W3A3。窗口 1 顺序跑最终 logits 蒸馏和额外软训练对照，
窗口 2 顺序跑 AdaLog 和硬量化尺度修复，窗口 3 顺序跑 ERQ 和 LS-ViT。
每窗包含一项完整重建，四个方法各自独立，结果与自己的基线配对比较。

## 实现与来源

| 任务 | 本轮实现 | 相对原方法的适配 |
|---|---|---|
| KD | 原浮点教师最终 logits，T=2，全网络硬量化路径，更新有界激活尺度 | 从保存的 C-MSE 开始，500 步，batch=4，LR=0.0003，尺度倍数在 1/1.1 到 1.1 之间 |
| AdaLog | r=37、整数 q 的对数索引和取整尾数，post-Softmax 激活 | q 按 head 搜索，尺度保留 query-token 粒度，8 个编码中留一个给零，候选由全模型教师 KL 筛选 |
| LS-ViT | 用 sample-wise 梯度和量化扰动估计 diagonal 与 rank-one 项 | 流式 FP64 统计、相对分母下限，从浮点模型重新运行 C-MSE，正式重建使用一次 LS 估计 |
| ERQ | Aqer 的 ridge 回归，Wqer 的逐半段量化、rounding refinement 和剩余权重回归 | 限于 attention 输出投影和 MLP FC2，保留原量化网格，每段最多 8 次 rounding refinement，整层候选用教师 KL 筛选 |

AdaLog 来源为作者的 [logarithm.py](https://github.com/GoatWu/AdaLog/blob/master/quantizers/logarithm.py)
和 [matmul.py](https://github.com/GoatWu/AdaLog/blob/master/quant_layers/matmul.py)。
LS-ViT 公式参考作者的 [cal_hessian 与重建损失](https://github.com/hhh7748/LS-ViT/blob/main/utils/block_recon.py)。
ERQ 参考 [原论文的 Aqer 和 Wqer](https://arxiv.org/html/2407.06794v1)。
本轮应按适配版报告，并单独验证新增机制的收益。

LS-ViT 的 diagonal 由 mean(abs(g)*abs(d))/mean(d²) 得到，rank-one 向量由
mean(abs(g)*sqrt(sum(abs(g)*abs(d))))/mean(sum(abs(g)*abs(d))) 得到。
正式重建分别归一化这两项后相加，再加入原 rounding 正则。C 的预校准保持 MSE。
LS 统计逐批累加，分母使用相对 1e-6 下限，减小零扰动方向上的数值问题。

KD 直接优化 `T² KL(teacher_T || student_T) + 0.001 mean(delta²)`。
训练用前 896 张缓存图，末 128 张用于挑选更新状态，包含零更新状态。
这里的留出只针对新增阶段，原 C-MSE 重建可能已使用这些图。
所有权重与位宽固定，尺度保留正负号。与旧 semantic probe KD 的读出方式不同。

AdaLog 用 64 张图按 `AV` 输出误差选每个 head 的 q，门控使用另外 128 张图。
ERQ 同样用 64 张图估计协方差，另外 128 张作门控。
门控条件要求教师 KL 至少相对下降 0.1%，最终收益依据完整验证集 Top-1。
这些门控规则仍可能出现 KL 改善而 Top-1 下降，应按配对验证结果决定保留与否。

AdaLog 的 8 个编码为 7 个正值和零。当前代码模拟浮点重建值，包含按 head 的参数与按 token 的尺度，
部署还需实现查表/移位内核并测试延迟。ERQ 的新权重落回原 3-bit 行网格，保持现有推理结构。

## 同步文件

- `test_quant.py`
- `utils/block_recon.py`
- `utils/activation_codebook.py`
- `utils/method_adaptations.py`
- `utils/lsvit_adapter.py`
- `scripts/run_method_adaptations.py`
- 前次修复过的 `utils/hard_scale_repair.py` 和 `scripts/run_endpoint_experiments.py`

同步全部代码后再启动这轮实验，确保本轮源码指纹一致。如果窗口 2 已在运行，保留其日志与结果，
待它完成后再同步本轮源码，避免正在运行的调度器与磁盘代码版本不同。

## 运行

进入工作目录和环境。

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
mkdir -p logs
```

KD、AdaLog 和 ERQ 默认读取 20261103_143115 那次运行的补偿前 C-MSE checkpoint。
文件在其他位置时，在对应命令后加 `--checkpoint /完整路径/model.pth`。
两次全量验证分别报告加载后的基线和适配后的准确率，基线实际值应以本轮日志为准。

窗口 1，KD 完成后自动跑额外软训练对照。

```bash
python -u scripts/run_method_adaptations.py --worker 1 --gpu 0 --gpu-slots 3 --resume
```

窗口 2，AdaLog 完成后自动跑硬量化尺度修复。

```bash
python -u scripts/run_method_adaptations.py --worker 2 --gpu 0 --gpu-slots 3 --resume
```

窗口 3，ERQ 完成后自动跑 LS-ViT。

```bash
python -u scripts/run_method_adaptations.py --worker 3 --gpu 0 --gpu-slots 3 --resume
```

在三个已有 tmux 窗口中分别前台执行，不加后台符号，也不重定向输出。
终端直接显示训练和验证日志。调度器显示 `[START]`、`[DONE]` 和任务耗时；
整个窗口的两项任务均成功返回后显示 `[WINDOW COMPLETE]`。匹配的成功结果显示 `[SKIPPED]`。
出现 traceback 或进程非零退出时，后续任务不会继续，也不会显示窗口完成提示。
如果某项失败，修正问题后重跑同一命令，已成功且源码/命令/源 checkpoint 指纹匹配的任务会跳过。
任务内部中断会从该任务开头重新开始。需要单独执行时用 `--method kd`、`--method adalog`、
`--method erq` 或 `--method lsvit` 替代 `--worker`。

## 分析输出

每个新方法在 `checkpoints/quant_result/*_method_*/method_result.json` 中记录结果。
KD、AdaLog、ERQ 包括 before、after、top1_delta_pp、门控变化、源码和源 checkpoint 指纹。
LS-ViT 包括最终准确率、C-MSE 预校准记录和启动配置，用本轮加载成功的 C-MSE 基线作参考。
这些 JSON 和对应 output.log 可同步供分析，前台运行仍会保存模型内部的日志。
尺度修复仍写 endpoint_result.json，使用原汇总命令。

本地已经检查小模型的数值行为和原有流程兼容性。完整 ViT-S 的新准确率等待服务器运行。
已有共享 GPU 日志中的完整重建分别为 9955.761 秒和 10134.274 秒，约 2 小时 46 分和 2 小时 49 分。
这只是原重建阶段的时间，不包含新增步骤及全部加载、校准和验证。

| 窗口 | 短任务预算 | 完整重建任务预算 | 窗口总预算 |
|---|---|---|---|
| 1 | KD：15–40 分钟 | 额外软训练对照：约 3–4 小时 | 约 3.5–5 小时 |
| 2 | AdaLog：10–30 分钟 | 硬量化尺度修复：约 3–4 小时 | 约 3.5–5 小时 |
| 3 | ERQ：20–60 分钟 | LS-ViT：约 2–4 小时 | 约 2.5–5 小时 |

新适配尚未在服务器实测，表中是排程预算，CPU 求解、I/O 和三任务共享 GPU 会影响耗时。
整轮先预留 5 小时，无法保证三个窗口同时结束；不要通过增加训练步数填满空闲时间。
已完成且匹配的任务会被 --resume 跳过，相应窗口的实际剩余时间会更短。
