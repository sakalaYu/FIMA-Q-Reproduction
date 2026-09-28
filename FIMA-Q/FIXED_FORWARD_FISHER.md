# 固定步长纯前向 Fisher：第二阶段实验

本阶段不训练新量化模型，也不修改已有 checkpoint。实验只验证一个固定方案：

- 所有层统一使用 `epsilon=1e-3`；
- 不选择步长、rank 或 Fisher 形式；
- 直接使用当前图像的真实量化误差 `Q(x)-x`；
- 比较单个完整误差方向和固定四个输出通道组；
- 用 JVP 作为参考，用中心差分作为纯前向估计。

## 新增文件

| 文件 | 用途 |
|---|---|
| `scripts/probe_fixed_forward_fisher.py` | 运行真实量化误差方向实验，记录精度、时间和显存 |
| `scripts/summarize_fixed_forward_fisher.py` | 生成 `summary.csv` 和 `prediction_errors.csv` |
| `scripts/run_fixed_forward_fisher.sh` | 单 GPU 入口，固定 `epsilon=1e-3` 并使用共享排队锁 |
| `utils/fisher_geometry.py` | 新增固定通道分组方向及精确重构函数 |

Attention 的探针位置位于输出投影之后，所以四组实验按最终输出通道固定分组，不能称为
Attention head 分组。单方向和四组方向都能精确重构同一个真实量化误差。

## 两种比较方式

`single` 使用完整误差的单位方向，rank 为 1。其系数等于误差范数，因此不存在 PCA
子空间覆盖不足的问题。

`grouped` 把最后一个通道维度连续切成四组。每组保留该组内的真实误差，四个分量之和
严格等于完整量化误差。分组数量固定为 4，不进行动态选择。

两种方式均在相同方向上分别计算：

1. JVP Fisher 参考值；
2. 固定步长中心差分 Fisher；
3. 对真实完整量化输出 KL 的二次近似；
4. 计算时间和 CUDA 增量显存。

## 服务器运行

先运行四张图片和第 0 块的流程检查：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
bash scripts/run_fixed_forward_fisher.sh smoke
```

成功后运行第二阶段 pilot：

```bash
bash scripts/run_fixed_forward_fisher.sh pilot
```

pilot 使用32张图片，检查第 0、3、5、8、11 块的 Attention 和 MLP。脚本结束后会自动
生成：

- `experiment.json`：固定配置、环境和源码哈希；
- `diagnostics.jsonl`：逐图像原始记录；
- `summary.csv`：JVP/前向差分误差、时间和显存；
- `prediction_errors.csv`：对真实量化 KL 的预测误差；
- `complete.json`：完整结束标志；
- 每个模块的 `.pt`：参考和前向 Fisher 小矩阵。

同步整个结果目录和终端日志即可分析。`pilot` 不会与旧探针同时占用 GPU，因为两个入口
共用同一个锁文件。

## 固定幅度割线验证

这项验证只运行单方向，并固定测试 `t=0.5、0.75、1.0、1.25、1.5`。它比较局部
Fisher 二次型和以 `t=1` 真实前向 KL 为锚点的割线方向因子，不选择幅度或模型。

先运行：

```bash
bash scripts/run_fixed_forward_fisher.sh secant_smoke
```

成功后运行：

```bash
bash scripts/run_fixed_forward_fisher.sh secant_pilot
```

结果目录会额外生成 `amplitude_errors.csv`。其中 `all_non_anchor` 汇总除 `t=1` 以外
四个幅度，可以用来判断割线因子是否真正泛化到邻近的量化误差幅度，而不是只报告锚点
上的零误差。

## 结果判断

首先检查 `complete.json`。随后比较：

- `relative_fisher_error`：中心差分相对 JVP 的矩阵误差；
- `relative_response_error`：中心差分相对 JVP 的响应误差；
- `speedup`：单次 JVP 时间除以前向差分时间；
- `relative_rmse`：Fisher 二次型对真实完整量化 KL 的预测误差。

如果单方向已经能稳定预测完整量化 KL，就优先采用单方向，不增加四组计算。如果固定四组
显著改善预测，再把四组版本带入后续 FIMA-Q 集成实验。
