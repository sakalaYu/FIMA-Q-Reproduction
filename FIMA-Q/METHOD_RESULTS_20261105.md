# 2026-11-05 六项适配实验核对

全部为 ViT-S、W3A3、seed 3407。六项结果 JSON 均 status=complete，日志均有最终验证和正常结束记录。
重建任务均记录 12 个块的 C 预校准；两项 endpoint 均记录 12 个块的新增阶段。

| 方法 | Top-1 | Top-5 | 验证 loss | 相对 64.524 的差值（百分点） | 进程耗时 |
|---|---:|---:|---:|---:|---|
| 原保存 C-MSE | 64.524 | 87.332 | 1.439398 | 0 | — |
| 最终 logits KD | 64.696 | 87.436 | 1.431195 | +0.172，直接配对 | 6分24秒 |
| AdaLog 适配 | 63.732 | 86.600 | 1.483578 | -0.792，直接配对 | 5分24秒 |
| ERQ 适配 | 63.984 | 86.866 | 1.469831 | -0.540，直接配对 | 5分35秒 |
| 逐块硬尺度修复 | 63.202 | 86.394 | 1.496744 | -1.322，跨运行参考 | 2小时44分09秒 |
| 额外软训练对照 | 64.076 | 87.016 | 1.460395 | -0.448，跨运行参考 | 2小时46分11秒 |
| C-MSE + LS-ViT 曲率 | 65.022 | 87.462 | 1.410583 | +0.498，跨运行参考 | 2小时31分41秒 |

进程耗时从运行目录时间戳到日志 finished 时间，未包含调度器等待 GPU 锁的时间。
KD/AdaLog/ERQ 新增适配阶段分别为 97.67、13.94、27.28 秒，其余主要是加载和两次全量验证。
此前给短适配预留的 10–60 分钟偏保守，应以本次实测更新。

## 比较边界

三个 post 方法源 checkpoint SHA256 相同：3dcdc082d0502ecb96bcf62080e6b2d62182eedd7e2162e5247429de85a2516c。
三者加载后全量验证均为 64.524，因此各自 before/after 可直接配对。
本轮六项原浮点模型 SHA256 均为 a78205f203d9299dd770d74b9f4273895a7d2a22260593bdbe7908e4e513463c。
LS-ViT 和两个 endpoint 的 128/1024 校准图片及增强张量 hash 均与 20261103 原 C-MSE 重建匹配。
但它们重新从浮点模型重建，没有在本轮同时重跑完全不加机制的 DPLR C-MSE，故 64.524 仅为跨运行参考。
LS-ViT 超过历史 C-MSE 64.758 共 0.264 个百分点；这是当前观察到的最好结果，不等于多种子稳定增益。
LS 命令中的 k=1 配合一次 LS 统计，不表示把原 rank=5 DPLR 简单调成 rank=1；其曲率模型和初始化频率已经改变。

## 机制证据与决定

KD：500 步中选第 200 步；门控 KL 0.451995 -> 0.398865，下降约 11.75%。
Top-1、Top-5、验证 loss 同时改善。保留；优先把完全相同的 KD 应用于 65.022 的 LS checkpoint。
不能将 +0.172 直接加到 65.022，也不能说它已超过历史 64.758。

AdaLog：接受零基块 0、3、9、10，门控 KL 下降约 11.18%，但全验证集三个指标均变差。
ERQ：24 个候选只接受 blocks.1.attn.proj，门控 KL 下降约 6.81%，全验证集仍下降。
结论限于当前适配版，停止扩展这两个实现，不推断论文原方法无效。
多个候选反复使用 128 张门控图，KL 改善不能保证分类收益；这些图也并非整个原重建流程的独立留出。

硬尺度修复：接受 11/12 个块，各块局部 MSE 改善约 0.52%–3.57%，最终 Top-1 仍为 63.202。
与同代码、同数据的额外软训练 64.076 相比低 0.874 个百分点。
负尺度修复已正常工作：保留 blocks.0 和 blocks.3 的 attn.matmul2.A_quantizer 负尺度，固定参数检查均通过。
因此此次不是原来的负尺度异常导致中断，而是完成后的方法结果下降。
修复前 soft/hard full MSE 的相对差异最大约 0.0803%，该诊断下 rounding 硬化差距很小。
hard full MSE / hard QDrop MSE 约为 1.48–4.87，QDrop 与推理路径差异更显著；此比值不能单独证明其是掉点原因。
停止当前逐块 MSE 尺度修复及额外软训练，不增加这两项的步数。

LS-ViT：13 次曲率初始化（patch_embed + 12 blocks），C-MSE 候选接受 10/12 个块，最终 65.022。
保留为新的性能候选基座。它使用已有 LS-ViT 机制，不能将替换本身当作原创创新。
下一步顺序：先做 LS + 相同 KD 的短配对；再固定配方完成必要对照，区分 LS 自身收益和旋转/C 带来的增益。
不建议当前直接扩大 LS/KD 参数搜索，也不建议先在 post-GELU 继续增加量化器变体。

## 结果位置

- KD: checkpoints/quant_result/20261104_192534_method_kd_2d10b80493db/method_result.json
- AdaLog: checkpoints/quant_result/20261104_192549_method_adalog_9f4786198ff0/method_result.json
- ERQ: checkpoints/quant_result/20261104_192600_method_erq_76aae8d1ede8/method_result.json
- hard repair: checkpoints/quant_result/20261104_193118_endpoint_hard-repair_seed3407_endpoint_v1_e461e8d6ea2f/endpoint_result.json
- LS: checkpoints/quant_result/20261104_193139_method_lsvit_4373e8b51261/method_result.json
- soft control: checkpoints/quant_result/20261104_193203_endpoint_soft-control_seed3407_endpoint_v1_f093e3c3625f/endpoint_result.json

LS 保存的优化权重为：
checkpoints/quant_result/20261104_193139_method_lsvit_4373e8b51261/vit_small_w3_a3_optimsize_1024_lsvit_dis_mode_q_qdrop.pth
