# 在 B-MSE / C-MSE W3A3 上检验局部教师蒸馏

## 当前证据与起点

固定旋转 64.650%，B-MSE 64.726%，C-MSE 64.758%，目前是 seed 3407 的结果。
新增蒸馏必须相对各自完整 checkpoint 计算，不能相对更弱起点计算后再归给蒸馏。
目前本地没有同步 local_teacher 结果，也没有 B/C 最终大权重；日志记录了服务器保存路径。
两个新 JSON 的路径和精度按对应成功日志设置。运行前必须确认服务器仍保留这两个 .pth。

## 方法与创新边界

保留已有固定旋转、B/C 预校准和正式 DPLR 重建的产物。
第二阶段每次冻结量化前缀、后续网络、权重和旋转，仅调整当前块的既有激活量化尺度。
局部教师为 R_Q(T_b(h_Q))，学生为 R_Q(Q_b(h_Q))；教师停止梯度，量化后续网络用 STE 传递输入梯度。
结合固定 DPLR 约束；E6 再加入完整浮点教师的标准化 logit 蒸馏。
固定 DPLR 是第二阶段约束，并非原始 progressive AdaRound 的完全续训。

候选贡献是：在已旋转且完整量化的 ViT 中，用相同量化环境构造局部教师，改善第二阶段尺度修复的监督。
它尚未被本项目实验验证，不保证提分或独创性。普通 KL、固定旋转、双教师损失本身都不能作为独立新点。
相关近邻至少包括 PD-Quant、MGRQ、Teacher Intervention、SQuaT；见此前检索。

## 最少先检验什么

先在 B-MSE 上完成 E3/E4/E5；若旧固定旋转上的同组实验在跑，先让它们完成，不中断。
E5-E4 检查量化后续路径；E5-E3 检查局部教师是否超过成熟全局 KD。
若出现正信号，补 controls 中 E1/E2/E6，再将胜出局部方案及最强全局 KD 放到 C-MSE 上。
后续补配对种子与 DeiT-S W3A3。所有组同起点、相同步数、可训练参数集合和验证设置。
比较时另报告总 GPU 时间；相同步数并不代表相同算力。
若只降低校准 KL、Top-1 不升，不扩大旋转自由度或简单增加步数。

## B-MSE 准备

同步两份新配置，保留现有训练源码不动。三个窗口都使用同一份 B 配置：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_local_teacher_experiments.py --config configs/local_teacher_w3a3_b_mse.json --suite prepare --workers 1 --gpu 0 --gpu-slots 1 --resume
```

准备成功后，各窗口分别执行一条：

```bash
python -u scripts/run_local_teacher_experiments.py --config configs/local_teacher_w3a3_b_mse.json --suite screen --worker 1 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

```bash
python -u scripts/run_local_teacher_experiments.py --config configs/local_teacher_w3a3_b_mse.json --suite screen --worker 2 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

```bash
python -u scripts/run_local_teacher_experiments.py --config configs/local_teacher_w3a3_b_mse.json --suite screen --worker 3 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

对应 E3/E4/E5。OOM 时停止这些任务，三条命令统一改成 --gpu-slots 1 排队恢复。
补对照时三条都将 --suite screen 改成 --suite controls，对应 E1/E2/E6。

```bash
python scripts/run_local_teacher_experiments.py --config configs/local_teacher_w3a3_b_mse.json --suite summary
```

C-MSE 使用 configs/local_teacher_w3a3_c_mse.json，其余入口相同，必须另做 prepare。
不同配置自动生成独立指纹，不混用固定旋转、B-MSE、C-MSE 的校准/训练结果。
