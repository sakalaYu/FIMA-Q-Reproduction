# 四窗口逐头旋转配对实验

服务器目录：`~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q`。
服务器实测只有 23.52 GiB 显存，四任务共享已在校准阶段触发 OOM。
现在保留四个任务窗口，但同一时间仅运行一个 GPU 任务，其余窗口等待锁。
计算逻辑在 Python 文件中，下面仅为启动命令。

## 同步文件

- `scripts/run_headwise_experiments.py`：排队、运行、完成检测、统计汇总。
- `test_quant.py`：补齐 `random.seed(seed)`，并记录种子修正标记。
- `utils/datasets.py`：由启动器启用校准输入指纹，检查配对的增强后图像是否一致。
- `utils/orthogonal_vit.py`：已有旋转实现，保留现有版本。

原入口只设置 NumPy/PyTorch 种子，未设置 Python `random`。timm 的随机裁剪使用 Python
`random`，而校准图片在主进程中预加载，因此之前的 seed 3407 配对不能保证裁剪一致。
本轮修正后，旧 3407 仅作为历史参考，不能混入新协议的均值。
新任务每次记录 128 张校准图像和 1024 张重建图像的索引及实际张量指纹；不同则拒绝配对统计。
参考 timm v0.9.2 源码：
https://github.com/huggingface/pytorch-image-models/blob/v0.9.2/timm/data/transforms.py

## 六个新任务

统一使用 ViT-S、128 张校准图像、1024 张重建图像、batch 32、每块 20000 次迭代、
Fisher rank 5、QDrop 0.5、验证 batch 64。W3A3 使用 `configs/3bit/best.py`，包含 3-bit
分类头和原配置的 8-bit patch embedding 设置；不会仅修改 W/A 而遗漏分类头位宽。

| 窗口编号 | 第一个任务 | 接着自动运行 |
| --- | --- | --- |
| 1 | W4A4 seed 3408 baseline | W3A3 seed 3407 baseline |
| 2 | W4A4 seed 3408 rotation | W3A3 seed 3407 rotation |
| 3 | W4A4 seed 3409 baseline | 无 |
| 4 | W4A4 seed 3409 rotation | 无 |

每个窗口激活原来成功运行 FIMA-Q 的环境，并进入目录：

```bash
cd ~/autodl-tmp/FIMA-Q-Reproduction/FIMA-Q
```

窗口 1：

```bash
python -u scripts/run_headwise_experiments.py --suite all --worker 1 --workers 4 --gpu 0 --gpu-slots 1 --resume
```

窗口 2：

```bash
python -u scripts/run_headwise_experiments.py --suite all --worker 2 --workers 4 --gpu 0 --gpu-slots 1 --resume
```

窗口 3：

```bash
python -u scripts/run_headwise_experiments.py --suite all --worker 3 --workers 4 --gpu 0 --gpu-slots 1 --resume
```

窗口 4：

```bash
python -u scripts/run_headwise_experiments.py --suite all --worker 4 --workers 4 --gpu 0 --gpu-slots 1 --resume
```

所有新启动窗口的 `--workers 4` 和 `--gpu-slots 1` 保持一致，只有 `--worker` 不同。
启动器会固定物理 GPU 的 UUID，子进程使用 `--device cuda`，以保留设备映射。
四个窗口使用同一套进程锁排队，不调用旧的 Bash 启动器。
槽位限制只覆盖这个 Python 启动器启动的任务，不管理用户手动启动的其他 GPU 程序。

已经失败的窗口可直接用上述对应命令重启，无须删除日志或 checkpoint。
`--gpu-slots 1` 使用独占锁，会等待先前以 `--gpu-slots 4` 启动的共享任务释放 GPU 锁，
不会再挤入这些任务。等待期间未启动模型子进程，不占用 GPU 显存。
先前仍正常运行的窗口可继续；如果也 OOM，则改用 `--gpu-slots 1` 重启。

`--resume` 跳过已完成、配置匹配、带种子标记和输入指纹的任务。中断或失败的任务从头重跑，
不支持恢复未完成的 block 重建。默认数据路径是 `../imagenet_fimaq`。
需要改路径时，对所有窗口添加相同的 `--dataset /path/to/imagenet_fimaq`。

检查任务安排但不实际运行：给任一启动命令加 `--dry-run`。

## 汇总结果

全部结束后，在任意窗口执行：

```bash
python scripts/run_headwise_experiments.py --suite summary
```

报告每个 seed 的 baseline、rotation、Top-1 增益（百分点），以及完整配对的平均增益和
样本标准差（分母 n-1）。仅一个配对时标准差为 N/A，不能当成零波动。
W4A4 默认将显示 3408/3409 两个新配对；旧 3407 在历史记录中单列。

文件：

- `checkpoints/headwise_experiments/summary.json`
- `logs/*headwise_w4a4_seed3408*seeded_full.log` 等任务日志
- `checkpoints/quant_result/*headwise*seeded_full/` 下的完整 checkpoint 与日志

脚本只采集完整实验最终的 50000 张验证结果，不把校准集准确率当作最终精度。
同一设置的重复运行只选最近一次完成结果，不作为额外种子；配置或输入指纹不一致的配对被排除。
并行结束时临时汇总可能只包含已完成任务，最后再手动运行一次 summary 获得完整统计。

## 可选：获得修正后的三种子均值

若希望本轮一次得到严格的三个 W4A4 种子（3407/3408/3409），给四个窗口命令都加
`--include-3407`，总任务数从 6 增为 8，每个窗口运行两个任务。

如果已经启动了六任务安排，不要中途更改分片参数。全部结束后，在两个窗口分别补跑：

```bash
python -u scripts/run_headwise_experiments.py --suite w4-3407 --worker 1 --workers 2 --gpu 0 --gpu-slots 1 --resume
```

```bash
python -u scripts/run_headwise_experiments.py --suite w4-3407 --worker 2 --workers 2 --gpu 0 --gpu-slots 1 --resume
```

然后重新汇总。

## 时间预算与验证范围

已完成的 W4A4 单任务独占 GPU 耗时约 2 小时 8 分钟。六任务对应约 13 小时的单任务
累计耗时；W3A3 的时间尚未实测。改为 GPU 任务顺序执行后，可暂按从头运行整批
12–16 小时预留，实际时间受硬件负载及 W3A3 实测耗时影响。
若启用额外 3407 配对，累计工作量约增三分之一。

本地已检查任务分配、汇总统计、历史实验隔离、输入指纹不一致的排除，以及命令生成。
本地没有服务器 GPU/数据，未执行这六个完整训练任务。
