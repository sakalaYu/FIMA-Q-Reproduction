# LS-ViT、旋转与蒸馏：来源核对及论文实验计划

核对日期：2026-11-05。本文为文献和本地实现分析，不是新增实验结果。

## 1. LS-ViT 与本地 LS 的关系

LS-ViT 沿用 QDrop/FIMA-Q 的逐块重建框架，主要改变代表性曲率的估计方式。原先先聚合样本再估计的方式，可能忽略样本相关曲率 H_n 与扰动 d_n 的协方差；LS 方法改为在样本级拟合 g_n 与 H d_n 的关系。

这里的 independence 不应简单解释成“不同图片不独立”。如果所有样本真的共享同一个固定 H，那么求和 g_n=H d_n 自然成立；问题是把变化的 H_n 用一个代表性 H 代替时的近似。

对角分量使用 h_j=sum_n g_nj*d_nj / sum_n d_nj²；秩一分量先令 s_n=d_n^T g_n，再计算 u=sum_n g_n sqrt(s_n)/sum_n s_n。官方与本地数值实现均对梯度、输出扰动取绝对值。重建误差的对角项和秩一项分别归一化后相加，仍包含 rounding 正则。

作者代码每块在重建初始时估计一次曲率，遍历一次校准集，包含各 minibatch 的整网反向传播。FM=1 不是只对一张图片或一个 minibatch 反向一次。相较 FIMA-Q 多次 progressive 估计，其重点是减少校准集整网梯度遍历次数。

本地 utils/lsvit_adapter.py 的上述闭式解及损失核心直接采用 LS-ViT；流式 FP64 统计、相对分母下限属于数值和内存工程适配。额外固定旋转和 C-MSE 预校准属于外部组合，不是重新发明 LS 曲率。论文应明确引用。

官方公开结果：ViT-S FP=81.39%，W3A3 FIMA-Q=64.09%，LS-ViT=64.10%；W4A4 FIMA-Q=76.65%，LS-ViT=76.67%。本地当前 C-MSE+LS 为 65.022%，名义上比官方 W3A3 LS 高 0.922 个百分点。但基座权重、预处理、量化粒度、首尾层设置、随机种子及新增预校准需对齐，不能直接称为公平超越。尤其本地当前 DPLR 主实验 k=5，与官方 FIMA-Q 对比中的 FM=15 不同。

必须补本地“原 LS”对照。目前 --method-adaptation lsvit 入口要求 C-MSE，不能直接用现有名称将其当作无旋转、无 C 的官方 LS 复现。后续应增加独立原 LS 对照入口。

## 2. 官方实验覆盖

正文：ImageNet 分类，ViT-S/B、DeiT-T/S/B、Swin-S/B；W3A3、W2A4、W2A3，W4A4 进一步见补充材料；COCO 检测和实例分割，Mask R-CNN/Cascade Mask R-CNN 与 Swin-T/S，W4A4；对角估计方法比较；同 FM 预算比较；校准耗时比较。另有样本差异及低比特下曲率近似误差的动机分析。

补充材料：对角/低秩/组合分量消融；128/256/512/1024 校准样本规模；扩展 W4A4 方法比较；与 LSQ QAT 的数据、耗时、精度比较。

实验规模服务于“估计更合理且更高效”的具体主张。增加重复实验并不能替代独立算法贡献。

## 3. 与当前旋转方案最相关的论文

| 论文 | 已有机制 | 与当前实现的关系 |
|---|---|---|
| QuaRot | 函数保持的 Hadamard 旋转，缓解 outlier | 固定旋转思想已有，不应认定为新算法 |
| SpinQuant | 学习正交旋转及量化模型优化 | 学习旋转与 Cayley 类正交参数化已有 |
| OmniQuant | 可学习等价变换，包含 Q/K 对偶通道缩放 | D 与 D^{-1} 的核心思路已有 |
| FlatQuant | 可学习仿射变换、通道缩放和量化范围优化 | 联合变换优化已有更广泛形式，具体结构不同 |
| FPTQuant | 可合并的 Q/K 缩放与旋转，解决 LLM RoPE 限制 | 与“Q/K 缩放+旋转+离线折叠”尤其接近；其受 RoPE 限制的结构与当前 ViT 每头 Cayley 不同 |
| GLF-Q (2026预印本) | ViT 的离线 Hadamard、全局/局部特征重建 | 直接覆盖 ViT 旋转和重建监督，必须纳入相关工作 |

当前实现并非完整移植上述某一篇的原算法，而是将多项已有机制适配进 FIMA-Q：固定 QK/VO 旋转、Q/K 联合尺度、有限对偶缩放、每头 Cayley 旋转、attention KL+block MSE、候选门控以及零点一致性处理。适配工作不等于已证明的算法创新。

GLF-Q 官方预印本报告 ViT-S W3A3=67.40%，但其包含 MLP 重建、归一化/激活处理和不同优化流程，不能当作当前 LS 换一项损失的公平对照。它确实说明 65.022% 不能按已有证据称为全领域最优。用户此前已放弃的 FP suffix/cross-block 研究路线不在本计划中重新提出。

## 4. 论文所需的最小证据链

先使用同一个 FP checkpoint、同一 ImageNet 校准集和固定评测流程，完成下表；FP 准确率必须实测，不能仅凭模型名称认为相同。

| 对照 | 目的 |
|---|---|
| 无旋转原 DPLR、无旋转原 LS | 区分基座算法差异 |
| 固定旋转+LS | 分离固定旋转收益 |
| B-MSE+LS、C-MSE+LS | 分离额外学习旋转的独立收益 |
| C-MSE+LS+普通 KD | 判断最终尺度 KD 是否互补 |
| C-MSE+LS+候选新机制 | 证明相对已有组合的独立贡献 |

同种子、相同样本与增强、相同候选选择规则、相同校准预算。若额外优化预算不同，增加时间或更新数匹配的普通 KD 对照。

方法确定后：优先 ImageNet-1K 的 ViT-S、DeiT-S、ViT-B，W3A3 为主、W4A4 为辅；再扩展 Swin-S。当前新增方法入口只允许 ViT-S，跨模型需先完成适配验证，不是修改模型名字即可。W2A3 用于最后检查困难条件，不要求探索期全面铺开。

强直接基线为原 FIMA-Q、原 LS-ViT、固定旋转+LS 和普通 KD。QDrop/BRECQ/PD-Quant 及 RepQ-ViT/AdaLog 等用于相应设置的主表比较；引用论文数值与本地复现值应分栏，不能用本地受限适配版代表原论文方法。

COCO 属于方法泛化证据：成熟后以 Swin-T/S+Mask R-CNN 在 W4A4 检测/实例分割，报告 APb/APm。若只宣称分类方法，不必开题阶段立即实现完整检测流水线。SUN397 校准、ImageNet 验证可检验域外校准，但仅在宣称分布鲁棒性时作为关键实验。

至少三种子配对，报告增益均值与标准差；记录教师 KL、Top-1、Top-5、验证 loss、教师/学生 top-1 一致率、类别间隔、尺度变化、饱和率、校准时间和显存。用独立门控集选择状态，最终验证集只做报告。伪量化 FLOPs 不等于真实 INT3 部署加速。

## 5. KD 改善的可执行研究顺序

### 第一步：同配方迁移，不立即扩参数

在 LS 65.022 checkpoint 上运行原普通 KD（T2、500步、batch4、LR0.0003、界限1.1）。当前涨点 0.172 来自 C-MSE+DPLR 64.524 checkpoint，不是 LS。保留 zero-update 候选，不假设增益相加。

### 第二步：分析瓶颈后改进

当前只有尺度可变，权重量化编码固定，训练状态选择在第200步；增加步数不能修复被锁定的错误编码。应先记录尺度触界比例、逐组梯度、门控与训练曲线、teacher/student top-k 差异，判断是参数受限还是数据泛化不足。

低成本候选是分组尺度预算：Q/K、V/attention输出、MLP、head 使用按敏感性确定的约束，并在相同更新数下与统一1.1界限比较。不能只给每层增加手调参数然后称为创新。

面向 Top-1 的候选是 teacher top-1 与竞争类别的 margin 蒸馏，辅以普通 KL。教师 top-1 只作无标签伪目标，对低置信教师样本限制权重；它不保证教师正确，也不是已证实的新机制。独立门控与多种子是必要条件。

### 第三步：高工作量候选——最终全量化状态的离散编码与尺度联合修正

保留相同3-bit网格，允许少量敏感投影的整数编码 q 在 {-1,0,+1} 范围调整且保持 q∈{0,...,7}，联合有界激活尺度更新。用整网 hard-path KD 评价，用局部 LS 代价或尺度空间信赖域限制损伤，限制修改预算及区域。关键是每轮候选评估和最终保存均为硬整数编码，拒绝候选可精确回退。

该机制试图突破尺度单独更新的表达能力，同时避免浮点补偿支路。不是重新在 FP suffix 上选择旋转，也不是简单增加原 AdaRound 步数。它仍与既有 rounding 优化相关，必须查重并比较普通 KD、仅编码、仅尺度、编码+尺度、去掉曲率约束等消融，不能提前认定原创或承诺涨点。

当前优先建议第一步，随后以诊断决定是否投入第三步，不建议同时扩大温度/步数/范围/旋转自由度形成大网格。

## 来源

- LS-ViT正文：https://openaccess.thecvf.com/content/CVPR2026/papers/Hwang_LS-ViT_Least-Squares_Hessian_Based_Block_Reconstruction_for_Low-Bit_Post-Training_Quantization_CVPR_2026_paper.pdf
- LS-ViT补充：https://openaccess.thecvf.com/content/CVPR2026/supplemental/Hwang_LS-ViT_Least-Squares_Hessian_CVPR_2026_supplemental.pdf
- 官方代码及表格：https://github.com/hhh7748/LS-ViT
- 官方曲率实现：https://github.com/hhh7748/LS-ViT/blob/main/utils/block_recon.py
- QuaRot：https://arxiv.org/abs/2404.00456
- SpinQuant：https://arxiv.org/abs/2405.16406
- OmniQuant：https://arxiv.org/abs/2308.13137
- FlatQuant：https://arxiv.org/abs/2410.09426
- FPTQuant：https://proceedings.mlr.press/v306/van-breugel26a.html
- GLF-Q：https://arxiv.org/abs/2609.34564
