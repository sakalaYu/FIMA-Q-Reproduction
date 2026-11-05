# C-MSE 后续短实验：最终量化状态中的 Q/K 编码与尺度

本轮是机制探索，不是已确认原创或已确认提分的方法。基座为保存的 C-MSE+LS 模型，参考 Top-1=65.022%，不重新执行20000步重建。

## 目的及边界

C 的旋转及通道缩放在 AdaRound 前选定；随后权重舍入可能限制其最终表现。当前普通 KD 又仅允许激活尺度变化。本轮检验：固定已学习的旋转，允许极少量 Q/K 权重在原整数网格移动，能否在最终硬量化状态中补足尺度优化的限制。

该实验不证明旋转本身需要改变，也不能证明旧预校准候选被后续 rounding 破坏。它只是检验当前最终模型是否还有可通过编码修正恢复的空间。

## 三个独立对照

| 窗口 | 方法 | 回答的问题 |
|---|---|---|
| 1 | 相同普通尺度 KD | 原有+0.172能否迁移到LS基座？ |
| 2 | 仅Q/K整数编码修正 | 固定尺度后，改变少量权重编码是否有价值？ |
| 3 | 编码修正后，相同尺度KD | 两种自由度能否互补？ |

三者均从同一个原始65.022 checkpoint加载，各做基线及最终全量验证。窗口3的编码阶段与窗口2算法相同，不读取窗口2输出，不依赖其先完成。

## 实现

对每块QKV权重，仅Q、K行可变。用64张校准图的T=2整网logits KL累积STE梯度，将每个权重移动一个编码的预测收益用于排序。分别生成8、32、128项的候选，保持整数编码处于0到7；用这64张图的实际硬前向KL选一个候选，再在最后128张图上用T=1教师KL门控，至少相对改善0.1%才提交，否则精确恢复原权重。

每块只扫描一次，接受后最多改变128项；12块最多1536项。V、偏置、权重量化尺度、零点、位宽、旋转均不修改。没有新浮点补偿支路，也没有新的推理矩阵。当前不引入额外LS曲率约束；LS仅是已有基座。报告采用qk_integer_code_repair_v1，不能称为已完成的曲率约束新算法。

窗口3随后执行原500步尺度KD（T=2、batch4、LR0.0003、倍率范围1/1.1到1.1），参数与窗口1相同。

候选拟合64张和门控128张互斥，但它们来自原重建使用的1024张图；门控并非整条量化流程的独立数据。门控改善不保证Top-1改善，最终判断使用完整验证集。选择预算和门控的重复使用可能过拟合，正收益后必须做独立门控及种子复验。

## 同步文件

- test_quant.py
- utils/method_adaptations.py
- utils/qk_code_repair.py
- scripts/run_method_adaptations.py
- scripts/run_c_refinement.py

其余依赖沿用上一轮。新测试在tests/test_qk_code_repair.py。

## 三窗前台命令

在每个已有tmux窗口先执行：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
```

窗口1：

```bash
python -u scripts/run_c_refinement.py --worker 1 --gpu 0 --gpu-slots 3 --resume
```

窗口2：

```bash
python -u scripts/run_c_refinement.py --worker 2 --gpu 0 --gpu-slots 3 --resume
```

窗口3：

```bash
python -u scripts/run_c_refinement.py --worker 3 --gpu 0 --gpu-slots 3 --resume
```

默认checkpoint：
checkpoints/quant_result/20261104_193139_method_lsvit_4373e8b51261/vit_small_w3_a3_optimsize_1024_lsvit_dis_mode_q_qdrop.pth

若不在此处，各命令增加 --checkpoint /实际完整路径/model.pth。不要用校准阶段calibsize文件替代optimsize文件。

窗口1参考上一轮KD进程约6分钟；共享GPU下本轮先按5–15分钟预算。窗口2、3尚无服务器实测，按10–30分钟/窗预算，整轮先预留30–45分钟，可能受I/O和GPU竞争影响。看到[WINDOW COMPLETE]才表示该窗任务成功返回；异常退出不会继续。--resume只跳过匹配的完整结果，任务内部中断需重跑该任务。

## 看什么结果

同步checkpoints/quant_result/*_method_*/method_result.json与output.log。

- before.top1应先复核是否65.022；不一致时，以本次配对为准并先排查加载。
- after/top1_delta_pp是实际增益。
- 编码阶段查看blocks、accepted、changed_entries及fit/gate KL。
- 全部候选被拒绝时，窗口2应精确保持基线；这是有效零结果，不表示未执行。
- 窗口3只有超过窗口1，才支持“编码修正带来额外价值”，仅超过原始65.022不够。

本地5项编码数值检查及5项原方法适配检查通过，包含整数边界/预算/QK位置限制、零梯度恒等、拒绝回退、接受提交、硬前向与STE一致、状态恢复与重载。完整ViT-S/CUDA结果待服务器运行，不能预报准确率。

若三项无新增正收益，停止扩大编码预算，转而补原LS及固定旋转+LS对照，重新判断C预校准是否必要，而不是不断加模块。

## 研究归属

- LS曲率公式来自LS-ViT：https://github.com/hhh7748/LS-ViT
- Hadamard和学习旋转相关机制已有QuaRot、SpinQuant等来源。
- GLF-Q已研究ViT离线Hadamard及全局局部特征重建：https://arxiv.org/abs/2609.34564

重合内容应作为引用的方法、实现依赖或baseline，不应列为本人的原创机制。当前新代码是受限整数搜索的探索实现，尚未完成全面查重或独立收益验证。论文应同时交代已有组件来源和真正新增部分。
