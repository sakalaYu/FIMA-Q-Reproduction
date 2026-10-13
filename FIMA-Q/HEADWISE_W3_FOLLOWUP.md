# W3A3 多种子与旋转位置消融

## 本轮六个新实验

| 编号 | 位宽 | seed | 旋转 |
|---|---|---|---|
| 1 | W3A3 | 3408 | 无 |
| 2 | W3A3 | 3408 | Q/K/V + 输出投影（all） |
| 3 | W3A3 | 3409 | 无 |
| 4 | W3A3 | 3409 | all |
| 5 | W3A3 | 3407 | 仅 Q/K（qk） |
| 6 | W3A3 | 3407 | 仅 V + 输出投影（vo） |

复用已有 seed 3407 的无旋转和 all，不重新运行它们。
每个新任务都从浮点预训练模型开始，重新校准并完成重建，不加载已量化 checkpoint 再旋转。

固定配置：ViT-S、3bit/best.py、rank=5、128 张校准图片、1024 张重建图片、
校准/重建 batch=32、每块 20000 步、QDrop=0.5、验证 batch=64、num_workers=8。
与前一轮使用相同的预训练权重、数据、环境和增强设置。不要为并发改变 batch 或 rank。

## 代码位置和定义

- `utils/orthogonal_vit.py`：`apply_headwise_hadamard(..., mode='all'|'qk'|'vo')`。
- `test_quant.py`：新增 `--headwise-mode`；需与 `--headwise-hadamard` 配合使用。
- `scripts/run_headwise_experiments.py`：新增 `w3-followup`、`w3-seeds`、`w3-ablation`。
- `tests/test_orthogonal_vit.py`：浮点等价、参数不增加、未选分支不改变、RNG 不消耗等检查。
- `tests/test_headwise_experiments.py`：6 个新任务的划分、旧日志兼容、消融不混入 all、输入一致性等检查。

令每个 head 的正交矩阵为 R，按行向量记法：

- qk：Q'=QR、K'=KR，因此 Q'K'^T=QK^T；不改变 V 和输出投影。
- vo：V'=VR，同时输出投影使用对应逆变换；不改变 Q/K。
- all：两种变换同时使用，保持原先 all 实现的计算顺序和结果。

所有旋转都折叠进权重与必要的偏置，没有新增推理模块或参数。
这里的“无额外旋转算子”不等于已经测得整数推理加速。
三个模式都会在量化前用固定输入自动检查浮点模型的输出等价性。

## 服务器运行

先将本次代码同步到服务器，包括上述文件，保留已有 `utils/datasets.py` 的 hash 记录功能。
在 fimaq 环境执行一次：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
conda activate fimaq
python -m unittest discover -s tests -p 'test_headwise_experiments.py' -v
python -m unittest discover -s tests -p 'test_orthogonal_vit.py' -v
python scripts/run_headwise_experiments.py --suite w3-followup --dry-run
```

测试应为 17 项和 5 项通过；数值测试不应出现 skipped。dry-run 应列出上述六项。

当前服务器支持三个任务同时运行。本轮使用三个 tmux 窗口，共享物理 GPU 0，
每个窗口顺序执行两个任务。每个窗口先进入上述目录并激活 fimaq，然后各执行一条：

窗口 1（seed 3408 baseline，随后 seed 3408 all）：

```bash
python -u scripts/run_headwise_experiments.py --suite w3-followup --worker 1 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

窗口 2（seed 3409 baseline，随后 seed 3409 all）：

```bash
python -u scripts/run_headwise_experiments.py --suite w3-followup --worker 2 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

窗口 3（seed 3407 qk，随后 seed 3407 vo）：

```bash
python -u scripts/run_headwise_experiments.py --suite w3-followup --worker 3 --workers 3 --gpu 0 --gpu-slots 3 --resume
```

`--workers 3` 负责划分任务；`--gpu-slots 3` 允许最多三个模型进程同时运行。
三个窗口必须使用相同的 GPU 和 slots 参数。切换方案前让旧调度进程退出，
避免旧窗口继续领任务或持有独占 GPU 锁；已经成功的实验会被 `--resume` 跳过。
GPU 锁只协调使用本脚本且位于同一仓库路径的进程，其他程序仍会占显存。
若中途失败，解决原因后原命令重跑即可：`--resume` 跳过完整成功任务；失败任务从头跑。
不要删除仍在运行任务使用的 lock 文件。

也可分别运行两类实验：`--suite w3-seeds`（4 项）和 `--suite w3-ablation`（2 项）。

此前顺序执行估计总计约 13–16 小时。三个进程共享一张 GPU 的算力，
不能直接将时间除以三；并发总耗时需根据第一批三个任务的实际进度估计。

## 汇总与验收

全部完成后：

```bash
python scripts/run_headwise_experiments.py --suite summary
```

结果写入 `checkpoints/headwise_experiments/summary.json`，并在终端打印。

- `w3a3.count` 应为 3：三个 seed 的 baseline/all 配对，以及增益均值、样本标准差。
- `w3a3_seed3407_ablation.count` 应为 4：none/qk/vo/all 相对同一 baseline 的增益。
- W4A4 仍为 2/3 是预期：历史 seed 3407 未固定 Python random，未纳入统计；本轮没有要求补它。
- 每个配对或消融对照的配置、worker 数、128/1024 张样本及增强后张量 hash 必须匹配。
- 未完成、错误配置、输入不匹配的日志不会进入有效比较；部分旋转不会被当成 all。
- 三个种子的标准差是种子间波动，不是置信区间；seed 3407 单次消融不能证明跨种子机制稳定。

## 每轮分析同步哪些文件

常规结果分析的最小材料：

1. `logs/*headwise*.log` 中本轮完整日志，里面有参数、配置、输入 hash、分阶段指标、最终指标和完成标记。
2. `checkpoints/headwise_experiments/summary.json`，便于快速查看。不能只保留汇总而丢掉原始日志。
3. 对应版本的源代码、`configs/3bit/best.py`、`configs/4bit/best.py`、运行命令和 Git commit。
4. 若有报错，补终端 traceback；`output.log` 未必包含所有 stderr。

`logs/<run>.log` 与 `checkpoints/quant_result/<run>/output.log` 是同一成功任务的日志副本，
同步一份即可，优先 logs。中断任务没有最终副本时同步 output.log。
首次或环境变化时再保存 torch/timm/CUDA 版本、GPU 型号等信息。

旧 Fisher probe 分析需要其 summary.csv、prediction_errors.csv、experiment.json 等小文件；
adapter 分析需要 experiment.json 和日志。这些也适合 Git 管理。

常规比较 Top-1 不需要同步：模型 `.pth/.pt/.ckpt/.safetensors`、预训练权重、
ImageNet 图片、缓存激活、完整 Fisher 张量、优化器状态、locks、Python 缓存。
但模型 checkpoint 应保留在服务器或备份存储中：重评、错误样本分析和机制诊断时可能需要。
“不进 Git”不代表“删除”。

本次 `.gitignore` 已忽略新产生的大型模型/张量文件及锁目录，仍保留日志和 JSON/CSV。
仓库已有不少被跟踪的 `.pt/.pth`：新增 ignore 不会取消它们的跟踪，也不会缩小 Git 历史。
本次没有取消跟踪、删除模型或改写历史。已有权重的迁移需另行处理。

## 目前的证据与下一阶段

2026-10-13 核对的同协议结果：

| 设置 | baseline Top-1 | all Top-1 | 增益（百分点） |
|---|---:|---:|---:|
| W4A4 seed 3408 | 76.464 | 76.714 | +0.250 |
| W4A4 seed 3409 | 76.404 | 76.798 | +0.394 |
| W3A3 seed 3407 | 63.358 | 64.650 | +1.292 |

W4A4 配对增益均值 +0.322，样本标准差 0.102；W3A3 还只有一个 seed。
这些说明值得继续验证，尚不能独立支撑新的方法论文。

优先补的证据：

1. 本轮 W3A3 三种子和位置消融，判断信号是否稳定、来自哪里。
2. rank=15 的公平 baseline/all 配对，确认增益在更强的 FIMA-Q 配置下仍存在。
   官方 ViT-S W3A3 为 64.09，但我们当前 rank=5，不能直接把 64.650 与其差值当成公平改进。
3. 测量 QK score 误差、attention 分布误差、输出投影后的共同坐标误差、教师输出 KL、
   权重/激活的量化误差；不能只画 outlier 减少图。比较 V 等内部激活时先对齐旋转坐标。
4. 增加至少两个模型配置（例如 DeiT-S、ViT-B），比较低比特与 W4A4，报告时间和峰值显存。
5. 方法创新需超过固定旋转：例如用校准数据上可验证的任务损失预测，决定每个 head 的
   QK/VO 旋转位置或候选基。先验证评分能在独立无标签校准子集预测 KL 变化，再扩展搜索。
   对照必须包含统一固定旋转、同预算随机选择、MSE/outlier 评分；不得用验证集标签选方案。

第五项目前是待验证假设，不是已确认的新颖贡献。QuaRot 已有保持函数的注意力旋转，
SpinQuant 已有学习旋转；“加入 Hadamard”或“把旋转学起来”本身都不足以证明新意。
位置消融只能定位干预收益；重建会联动改变参数，不能仅靠最终 Top-1 宣称某种误差机制。

如果三种子增益消失，先查配对输入/配置，不继续扩张方法；若稳定但 rank=15 增益消失，
考虑是否只对有限重建预算有效，并在等精度成本下验证；若始终只有 VO 有效，就收敛到 VO
的机制诊断，避免没有证据地同时设计多个自适应组件。

参考：[FIMA-Q 官方实现](https://github.com/ShiheWang/FIMA-Q)、
[QuaRot](https://proceedings.neurips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf)、
[SpinQuant](https://arxiv.org/abs/2405.16406)。
