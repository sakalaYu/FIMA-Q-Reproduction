# C 修复复验与低维旋转试验

本说明对应上一轮两项试验。最新统一方案见 [QK_CONTROLLED_RUN.md](QK_CONTROLLED_RUN.md)：
新增 B/C × MSE/DPLR 对照，并将低维旋转与 5% 门槛拆开；统一七项运行会复用本轮相同参数的完整成功记录。

目标：以 W3A3、ViT-S、seed 3407、Fisher rank 5、固定 Hadamard 旋转的 64.650% Top-1 为参照，分别检查零点修复后的完整 C，以及低维 C。不能事先保证超过参照。

## 方法与可检验的问题

- C v2：原来的每 head 完整 Cayley 旋转，采用已修复的整数权重量化零点。此前 64.240% 是 v1，不能视为 v2 的结果。
- C v3：用 `S = A B^T - B A^T`、`R = (I-S/2)^{-1}(I+S/2)` 代替完整 `64×64` 旋转生成元，每 head 使用 rank 4 的 `A,B`。保留 Q/K 对偶缩放、Fisher DPLR 目标、同样的 300 步短校准和 20000 步 AdaRound。ViT-S 每块可训练参数从约 24972 降到约 3468。仅当留出集教师 KL 相对改善至少 5% 才折叠候选。
- 两个协议在各块完成真实 AdaRound 后增加同一 128 张校准图的教师 KL 记录 `teacher_kl_post_round`。这项无标签诊断有助于检查预校准与舍入后指标的脱节；它不是最终验证集 Top-1，也不能单凭其变化证明因果。
- v3 的低维约束与接受门槛同时改变，是一个筛选实验。若它获益，应再分别固定其中一项做消融。

## 同步与检查

同步 `utils/qk_joint.py`、`utils/block_recon.py`、`test_quant.py`、`scripts/run_qk_joint_experiments.py` 和 `tests/test_qk_joint*.py`。服务器执行：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -m unittest discover -s tests -p 'test_qk_joint*.py' -v
python scripts/run_qk_joint_experiments.py --variant C --dry-run
python scripts/run_qk_joint_experiments.py --variant C --protocol v3_lowrank_gate --rotation-rank 4 --min-relative-kl 0.05 --dry-run
```

## tmux 两个窗口

两个 C 都会做全模型 Fisher anchor 反向计算。单 GPU 上先用 `--gpu-slots 1` 排队执行，避免两个 C 同时占用显存。可以分别在两个 tmux 窗口粘贴：

窗口 1，修复后的原 C：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_qk_joint_experiments.py --variant C --gpu 0 --gpu-slots 1 --resume
```

窗口 2，低维 C：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_qk_joint_experiments.py --variant C --protocol v3_lowrank_gate --rotation-rank 4 --min-relative-kl 0.05 --gpu 0 --gpu-slots 1 --resume
```

`--resume` 根据协议、rank、门槛及完整 12 块记录区分任务；旧 v1 不会使 v2 跳过。每个 C 此前约 2 小时 53 分（三任务共享 GPU 时），单任务时每个暂估 2.5–4 小时，两项排队约 5–8 小时，实际随服务器负载变化。

## 完成后汇总

```bash
python scripts/run_qk_joint_experiments.py --variant summary
python scripts/run_qk_joint_experiments.py --variant summary --protocol v3_lowrank_gate --rotation-rank 4 --min-relative-kl 0.05
```

回传 `checkpoints/qk_joint/summary_seed3407.json`、`checkpoints/qk_joint/summary_seed3407_v3_lowrank_gate_r4_g50.json`，以及两条新 `logs/*qkjoint*.log`。这几项包含最终 Top-1、匹配的校准 hash、逐块接受情况和舍入后 KL。模型 checkpoint 不必纳入 Git。

若 v2 和 v3 均未超过 64.650%，下一步应暂停扩大旋转搜索，先用舍入后 KL 和逐块接受记录定位失配；再考虑让旋转与舍入在同一重建阶段协同优化，并将其与已有可学习旋转 PTQ 工作比较。
