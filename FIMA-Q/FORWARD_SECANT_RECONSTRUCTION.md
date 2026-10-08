# 前向割线块重建

该实现为原始 FIMA-Q 增加独立的 `forward_secant` 优化指标，不修改
`fisher_dplr` 基线路径。

## 方法

每次固定刷新时，程序只执行全模型前向传播，并为每个校准样本保存：

- 当前块的有符号量化误差 `d_i = q_out_i - raw_out_i`；
- 当前量化模型与全精度模型之间的输出 KL `L_i`。

割线锚点刷新和局部块重建使用相同的QDrop分布。当前W4A4配置中两者都使用
`drop_prob=0.5`，避免锚点方向来自完整量化激活、局部误差却来自随机混合激活的
分布错位。刷新结束后会再次写回训练概率，即使前向刷新异常也不会遗留错误状态。

局部块重建产生候选误差 `e_i` 后，割线项为：

```text
projection_i = <e_i, d_i> / (||d_i||² + 1e-12)
secant_loss = mean(L_i * projection_i²)
```

当 `e_i=d_i` 时，该样本的割线损失严格等于前向 KL。损失同时包含固定权重的 MSE
残差，覆盖与锚点方向正交的候选误差。`p1=p2=1`、刷新次数 `k=5`、刷新间隔和训练
迭代数均为固定配置，不包含动态选择。

两项分别用锚点的平均 KL 和锚点误差均方做固定归一化。在 `e_i=d_i` 时，割线项和
MSE残差都归一到1，不使用当前随机QDrop小批次重新决定权重。

全模型曲率信息的刷新不调用 `backward()`。局部块重建仍然需要反向传播，以优化
AdaRound 权重和激活量化尺度。

## 文件

| 文件 | 用途 |
|---|---|
| `utils/forward_secant.py` | 定义可独立测试的逐样本割线投影和 MSE 残差 |
| `utils/block_recon.py` | 缓存前向 KL/有符号方向，并接入块重建循环 |
| `configs/4bit/forward_secant.py` | 与 W4A4 基线一致的固定实验配置 |
| `scripts/run_forward_secant_reconstruction.sh` | 单 GPU smoke/full 入口 |
| `tests/test_forward_secant.py` | 验证锚点一致性、固定幅度规律和正交残差 |

## 运行

先检查完整集成流程。smoke 对全部块各运行2次重建迭代，不执行最终 ImageNet 验证：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
bash scripts/run_forward_secant_reconstruction.sh smoke
```

成功后运行完整 W4A4 实验：

```bash
bash scripts/run_forward_secant_reconstruction.sh full
```

完整实验沿用基线的1024张优化图片、batch size 32、每块20000次迭代，结束后自动验证
校准集和 ImageNet 验证集。结果保存在带有 `forward_secant_w4a4` 名称的
`checkpoints/quant_result` 子目录，并复制日志到 `logs`。

当前本地基线日志 `logs/vit_small_w4a4_fisher_dplr.log` 的 ImageNet Top-1 为
76.442%。新实验使用相同随机种子，并固定验证 batch size 为64。若新方法低于
76.342%，按预先约定停止该方向；达到或超过76.442%后再开展消融实验。
