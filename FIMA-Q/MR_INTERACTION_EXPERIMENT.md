# 残差预算＋MLP 有害交互加权 MR

本轮开发种子固定 3407，目标对照为已有残差 MR 的 67.052%，不是 3408 的最高纪录。固定逐头 Hadamard、等权 LS、原 GELU 教师、相同校准图、MR 总计 36,000 次更新。三个版本预算均由相同的 1,000 步试跑 A3 残差决定，使用原 2k/3k/4k 分配与学习率规则。

## 方法与归因

- baseline：完整复现原残差 MR，仅补充诊断。
- joint：每块剩余步数的前半段仍为原 MR，后半段加入完整局部 MLP W3A3 输出监督。
- interaction：与 joint 相同的全部计算路径，增加停止梯度的有害交互权重。

侧 1 表示 FC1 的输入激活和权重一起量化；侧 2 表示 FC2 的输入激活和权重一起量化。对同一输入计算 00、10、01、11 四条路径，四条路径共用同一组量化尺度。每条路径对原 GELU 教师计算逐 token 输出 MSE E00、E10、E01、E11。

delta = E11 - E10 - E01 + E00

w = 1 + stopgrad(max(delta, 0) / max(E00 + E10 + E01 + E11, epsilon))

L = L_original_MR + lambda(t) * mean(w * E11)

joint 固定 w=1。interaction 的 w 在 1～2 内。lambda 在量化感知阶段前半段从 0 升到 0.25，之后固定。每 250 步从拟合图更新网格。最终使用实际训练终点，不做留出集检查点选择，减少与上一轮的机制混淆。

采用仓库 UniformQuantizer：3-bit affine、整数零点、权重逐输出通道、激活逐张量、STE。局部网格使用拟合集前 32 张图及轻量 MSE 搜索，不含完整量化流程的 LayerNorm 重参数化、注意力误差和上游传播误差。它模拟两层线性的 W3A3 运算，不等于完整网络的量化结果；正式阶段仍独立完成原校准与 LS/AdaRound。

MR 拟合 896 张图，128 张用于诊断。正式 LS 仍使用原 1,024 张图，因此 128 张不是整个量化流程的独立测试集。更新次数相同不等于 FLOPs 相同，报告额外耗时。

## 三窗口前台运行

窗口 1，已有方法的局部复现和新诊断，不做正式重建：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_mr_interaction.py --variant baseline --stage screen --seed 3407 --gpu 0 --gpu-slots 3 --resume
```

窗口 2，普通联合量化监督：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_mr_interaction.py --variant joint --stage screen --seed 3407 --gpu 0 --gpu-slots 3 --resume
```

窗口 3，有害交互加权：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -u scripts/run_mr_interaction.py --variant interaction --stage screen --seed 3407 --gpu 0 --gpu-slots 3 --resume
```

三项出现 `[SCREEN COMPLETE]` 后汇总：

```bash
python -u scripts/run_mr_interaction.py --variant summary --seed 3407
```

先看 joint_w3a3_nmse，而非上一轮只模拟 FC2 激活的 A3 NMSE；同时看 fp_teacher_kl、有害交互量与额外耗时。交互量变小不自动意味着准确率提高。若 interaction 不优于 joint，不支持新增加权机制；不要仅凭它超过 baseline 就认定机制有效。

## 正式重建

局部诊断值得追踪时，joint 和 interaction 各做一次正式重建，两个窗口并行：

```bash
python -u scripts/run_mr_interaction.py --variant joint --stage quant --seed 3407 --gpu 0 --gpu-slots 3 --resume
```

```bash
python -u scripts/run_mr_interaction.py --variant interaction --stage quant --seed 3407 --gpu 0 --gpu-slots 3 --resume
```

汇总自动寻找历史 baseline：要求 MR 权重文件或张量完全匹配、FP 和校准数据指纹匹配、依赖版本与正式命令匹配。历史代码指纹单独保留，不声称与新版本完全一致。匹配时可复用旧准确率；若没有匹配，汇总 baseline 的 metrics 为 null。确认需要新对照后可运行：

```bash
python -u scripts/run_mr_interaction.py --variant baseline --stage quant --seed 3407 --gpu 0 --gpu-slots 3 --resume
```

此命令也会先尝试复用，失败后才运行正式 baseline。最终再次运行 summary。必须以 interaction-minus-joint 的 Top-1 验证加权贡献。开发完规则后再跨种子验证，不用验证集标签寻找 token 权重或预算。

结果在 `checkpoints/lsvit_controls/mr_interaction/`，最终准确率仍在 `checkpoints/quant_result/*/method_result.json`。`--resume` 复用完整阶段，尚未完成的训练阶段重跑。新入口仅增加 scripts 文件，不修改原实验的数值代码；本轮运行期间保持代码和权重来源不变。

## 来源边界

ReLU MR 来自 APHQ-ViT，当前原 MR 普通输出＋截断输出形式参考 GLF-Q。联合量化和 STE 属于已有思想。此处候选贡献是局部 MLP 四路径、正交互识别与停止梯度的逐 token 监督权重。它尚未获得最终 Top-1 支持，独立新颖性也仍需与 ERQ、量化重建加权等工作继续对照。
