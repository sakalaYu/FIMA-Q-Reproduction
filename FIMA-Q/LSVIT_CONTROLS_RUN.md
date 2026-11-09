# 原LS、固定旋转+LS、B-MSE+LS：三窗受控实验

本轮目标是分离已有65.022%中的组件贡献，不预设三项一定提分。三项都从本地原浮点模型重新校准和重建，不加载最终LS checkpoint，不进行KD、编码修正、QwT或endpoint修复。

| 窗口 | --lsvit-control | 固定Hadamard | Q/K预校准 | 正式重建 |
|---|---|---|---|---|
| 1 | plain | 无 | 无 | LS |
| 2 | fixed | 全部QK/VO | 无 | LS |
| 3 | b | 全部QK/VO | B-MSE：尺度+对偶通道缩放 | LS |

plain沿用本地流式/数值稳定LS适配版，作为本地原LS对照，并非宣称与作者每一项浮点运算完全一致。三项使用相同LS实现，因此可用于分离旋转与预校准收益。

固定ViT-S、W3A3、seed3407，初始校准128张、正式重建1024张、batch32、每块20000步、QDrop0.5、每块一次LS估计（k=1）。B保持既有300步预校准、batch8、holdout128、LR0.003、每50步评估、MSE目标及v2_integer_zp。其额外预校准预算本来就比fixed多，差值表示加入整个B阶段的收益，不是时间匹配的算法比较。

## 需同步

1. test_quant.py
2. utils/lsvit_protocol.py（新增）
3. scripts/run_lsvit_controls.py（新增）

其余依赖为此前已有的scripts/run_method_adaptations.py、run_qk_joint_experiments.py、run_headwise_experiments.py、utils/lsvit_adapter.py及Q/K模块，保持完整版本。
使用原浮点权重和 ImageNet train/val。启动器优先读取 `checkpoints/vit_raw/model.safetensors`，其次读取 `checkpoints/vit_raw/vit_small_patch16_224.bin`。两份本地文件均缺失时，沿用 timm 的预训练权重缓存或下载流程。权重在其他位置时，运行和汇总命令均可加 `--fp-checkpoint /完整路径/model.safetensors`。

上次 65.022% 的日志记录了 `timm/vit_small_patch16_224.augreg_in21k_ft_in1k` 的 safetensors 加载。本地 `model.safetensors` 的 152 个张量与上次的模型指纹匹配。三窗在校准前均检查加载后模型的 SHA256，须等于 `a78205f203d9299dd770d74b9f4273895a7d2a22260593bdbe7908e4e513463c`。

## 前台运行

每个已有tmux窗口先执行：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
```

窗口1：原LS，无旋转、无预校准。

```bash
python -u scripts/run_lsvit_controls.py --worker 1 --gpu 0 --gpu-slots 3 --resume
```

窗口2：固定旋转+LS。

```bash
python -u scripts/run_lsvit_controls.py --worker 2 --gpu 0 --gpu-slots 3 --resume
```

窗口3：B-MSE+LS，无额外可学习正交旋转。

```bash
python -u scripts/run_lsvit_controls.py --worker 3 --gpu 0 --gpu-slots 3 --resume
```

全部前台显示重建与验证日志，成功完成并通过结果校验后显示[WINDOW COMPLETE]。--resume只跳过指纹匹配的完整任务，不是每块或每步恢复。先同步全部代码再同时启动；运行中不更新源码。

此前三任务共享GPU的LS完整进程为约2小时32分钟。本轮按每窗2–3小时、整轮预留3–4小时预算，实际由GPU竞争、I/O及CPU负载决定。plain/fixed少一个预校准阶段，可能稍早结束。不是几分钟的后置适配实验。

如OOM，需结束剩余本轮任务后将三条命令统一改为--gpu-slots 1重新启动，排队执行；已完成且指纹匹配的任务可跳过，未完成任务从头开始。不要在已有slots=3进程运行时混用不同的并发策略。

## 汇总与同步结果

```bash
python -u scripts/run_lsvit_controls.py --variant summary
```

每项结果在checkpoints/quant_result/*_lscontrol_*/method_result.json，保存control、实际cfg、args、模型哈希、校准图片及增强张量hash、Q/K块记录、最终准确率、权重路径、源码及依赖指纹。
plain/fixed必须没有Q/K块报告，B必须有12个B-MSE块报告，完整验证指标必须有限，才允许--resume跳过。

同步三个method_result.json、对应output.log以及checkpoints/lsvit_controls/summary*.json即可分析；不必将全部权重放入git。服务器上保留新优化权重用于后续实验。

## 如何决策

- fixed - plain：本地LS之上的固定旋转收益。
- b - fixed：LS之上加入B阶段的收益。
- 历史C的65.022 - b：额外学习旋转的初步参考。

最后一项是跨运行参考。此次改动只增加协议入口、记录及调度，未改LS估计或C数值算法，但仍须检查原浮点模型、校准图片及增强hash是否匹配。差值很小时不能直接当成稳定收益，应再跑同代码C并配对多种子。

可选同代码C重跑命令（本轮先不占三窗）：

```bash
python -u scripts/run_lsvit_controls.py --variant c --gpu 0 --gpu-slots 3 --resume
```

若fixed已接近/超过65.022，优先简化预校准；若B接近/超过C，暂停额外学习旋转；若C有明确优势，再研究其预校准目标和阶段衔接。不要将这些组件消融本身写成新的算法创新。

本地7项对照检查覆盖 CLI 配方隔离、权重路径选择、误配置拒绝、历史 C 兼容、完整结果校验及三个小模型的量化重建流程。完整 ViT-S/CUDA 需服务器执行。
