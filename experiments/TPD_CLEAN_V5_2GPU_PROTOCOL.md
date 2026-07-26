# TPD-Clean-v5 两张 RTX 5090 执行协议

本文件固定 `TPD_CLEAN_V5_PROTOCOL.md` 的训练资源映射和启动行为。模型、
训练超参数、数据划分和 Gate A–E 以主协议为准。

## 1. 资源映射

只允许使用下列两张物理卡：

| 物理索引 | GPU UUID | 作业 |
| --- | --- | --- |
| 2 | `GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562` | Full/42，Capacity/3407 |
| 3 | `GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3` | Capacity/42，Full/3407 |

每张卡同时两个作业，每个作业：

- `batch_size=16`；
- `workers=0`；
- `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`；
- `AMP=false`；
- 只暴露一个 GPU UUID，训练进程内使用 `cuda:0`。

GPU 0、1 不进入 V5 启动器。每张卡在启动前至少需要 15000 MiB 空闲显存，
单个 worker 启动时至少需要 7500 MiB。

## 2. 固定路径

- 训练入口：`experiments/train_tpd_clean_v5.py`；
- sweep：`experiments/evaluate_tpd_clean_v5_pd_fa.py`；
- worker：`experiments/run_tpd_clean_v5_screen800_2x5090_worker.sh`；
- launcher：`experiments/launch_tpd_clean_v5_screen800_2x5090.sh`；
- 状态入口：`experiments/status_tpd_clean_v5_screen800_2x5090.sh`；
- source lock：
  `experiments/tpd_clean_v5_screen800_2x_source_lock.json`；
- smoke 根：
  `experiments/results/tpd_clean_v5_preflight_v1`；
- 正式结果根：
  `experiments/results/tpd_clean_v5_screen800_2x5090_v1`；
- run tag：`screen800_pd_fp32_shared2x5090_v1`；
- systemd unit 前缀：`sctransnet-tpd-clean-v5-2x-`。

## 3. 启动前置条件

launcher 的 `--preflight` 必须同时确认：

1. v5 source lock 中每个源码哈希与当前文件一致；
2. CPU all、GPU2 Full、GPU3 Capacity 三份持久 smoke 报告存在且哈希一致；
3. 三份 smoke 报告均为两步训练、step-0 exact SPD、七个 scale 梯度与
   更新非零、14 个 dense Keep tensor 梯度与更新非零、严格重载差异为零；
4. GPU UUID、物理索引和型号一致；
5. 两张卡满足四作业启动显存下限；
6. 四个目标 run 目录均不存在；
7. 四个 systemd unit 名称均未被占用。

任一条件失败时四个作业均不启动。

## 4. Worker 约束

每个 worker 在训练前后均重新验证：

- v5、v4、v3、v2、NER 共五份 source lock 及其所有源文件；
- 训练数据 fingerprint：
  `39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e`；
- 单一可见 RTX 5090；
- 当前 v5 source-lock 文件自身哈希未在运行期间变化。

每个 fresh run 训练 800 epochs，生成：

- `metrics.jsonl`：恰好 800 条连续 epoch 事件；
- `best.pth.tar`：Pd-primary；
- `best_miou.pth.tar`：mIoU-secondary；
- `last.pth.tar`：epoch 800；
- 两个 closed-interval Pd–Fa sweep；
- `summary.json`、`protocol.json`、`split.json`；
- 独立 launch manifest 和 worker log。

worker 的成功终止标记为：

```text
TPDCLEANV5_2X_COMPLETE variant=... seed=... gpu_uuid=... epochs=800
```

## 5. 运行语义

- 四个 run 都是 fresh paired training，不从 v4 或任意旧 checkpoint
  初始化；
- `Restart=no`，意外终止后不自动从不完整状态重新开始；
- 历史结果目录不覆盖、不移动；
- 官方测试集不访问；
- 四个 run 全部完成且八个 sweep 生成后，才进入独立 summarizer 和
  completion marker 阶段；
- Gate A–E 失败仍可形成“实验闭环完成”的最终 marker，但不授权 NER；
- 缺失任何正式输入时只允许报告 incomplete，不允许生成完成 marker。
