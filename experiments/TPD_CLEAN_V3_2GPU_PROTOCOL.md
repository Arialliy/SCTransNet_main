# TPD-Clean-v3 两卡交叉映射运行协议

## 1. 适用范围

本文件定义 TPD-Clean-v3 的隔离两卡运行层。模型结构、候选定义与工程门槛
继续服从 `experiments/TPD_CLEAN_V3_PROTOCOL.md`，本运行层不修改
Keep–Context–Saliency 三分支主线，不增加第四个并列分支，也不修改模型、
训练器、数据划分、损失或评估器。

现有四卡脚本及
`experiments/results/tpd_clean_v3_screen800_4x5090_v1/` 保持只读；
两卡运行不得复用、覆盖或续写其 unit、日志、checkpoint、manifest 和结果目录。

## 2. 冻结训练配置

四个 job 均从头训练，训练入口仍为
`experiments/train_tpd_clean_v3.py`，评估入口仍为
`experiments/evaluate_tpd_clean_v3_pd_fa.py`。以下配置与四卡版本逐项相同：

- 数据集：NUDT-SIRST 官方训练索引的 530/133 内部划分；
- `split_seed=20260722`，不读取官方测试索引；
- epochs `800`，batch size `16`，patch size `256`；
- Adam，base LR `0.001`，min LR `0.00001`；
- 10 epoch warmup 后 cosine decay；
- `workers=0`，FP32，六输出 BCE deep supervision；
- 每 epoch 内部验证；
- threshold `0.5`，match radius `3`，tiny area `9`；
- 同时保存 Pd-primary `best.pth.tar` 与 mIoU-primary
  `best_miou.pth.tar`，随后分别执行 Pd–Fa sweep。

两卡层不提供中断续训；目标 run directory 必须不存在。

## 3. 两卡交叉映射

允许的物理卡仅为 GPU index `2` 和 `3`。每张卡同时承载两个 job，
且通过 seed 与 variant 的交叉分配实现 counterbalance：

| job | variant | seed | GPU index | GPU UUID |
| --- | --- | ---: | ---: | --- |
| full-s42 | `tpd_clean_v3_full` | 42 | 2 | `GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562` |
| cap-s42 | `tpd_clean_v3_sal_capacity` | 42 | 3 | `GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3` |
| full-s3407 | `tpd_clean_v3_full` | 3407 | 3 | `GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3` |
| cap-s3407 | `tpd_clean_v3_sal_capacity` | 3407 | 2 | `GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562` |

因此 GPU2 同时运行 full-s42 与 cap-s3407，GPU3 同时运行 cap-s42 与
full-s3407。两张卡各包含一个 Full、一个 capacity control，以及两个不同
seed。该映射只用于模型指标筛选；由于每卡双 job，共享资源条件下不得比较
耗时、吞吐或显存效率。

## 4. 隔离命名

- 结果根：
  `experiments/results/tpd_clean_v3_screen800_2x5090_v1/`
- run tag：`screen800_pd_fp32_shared2x5090_v1`
- unit 前缀：`sctransnet-tpd-clean-v3-2x-`
- worker：
  `experiments/run_tpd_clean_v3_screen800_2x5090_worker.sh`
- launcher：
  `experiments/launch_tpd_clean_v3_screen800_2x5090.sh`
- status：
  `experiments/status_tpd_clean_v3_screen800_2x5090.sh`
- source lock：
  `experiments/tpd_clean_v3_screen800_2x_source_lock.json`

source lock schema 必须为
`sctransnet_tpd_clean_v3_screen800_2x_source_lock_v1`。锁文件缺失、是符号
链接、schema 不匹配或任一登记源码摘要漂移时，preflight 和 worker 均必须
终止。

## 5. Manifest 契约

每个 job 在独立结果根的 `launch/` 下生成 manifest。schema 固定为：

`sctransnet_tpd_clean_v3_screen800_2x5090_launch_v1`

除四卡版本已有的 paired seeds、fresh run、旧结果保留、内部验证和 FP32
策略外，`policy` 必须包含：

```json
{
  "allowed_gpu_indices": [2, 3],
  "concurrent_jobs_per_gpu": 2,
  "counterbalanced_mapping": true
}
```

worker 的 variant、seed 与 GPU UUID 必须严格匹配第 3 节表格；任意其他
组合均终止。launcher 还必须核验两个 UUID 当前分别解析为 index 2 和 3，
且均为 NVIDIA GeForce RTX 5090。

## 6. 启动与观察

创建并冻结新的 source lock 后，依次执行：

```bash
bash -n experiments/run_tpd_clean_v3_screen800_2x5090_worker.sh
bash -n experiments/launch_tpd_clean_v3_screen800_2x5090.sh
bash -n experiments/status_tpd_clean_v3_screen800_2x5090.sh
experiments/launch_tpd_clean_v3_screen800_2x5090.sh --preflight
experiments/launch_tpd_clean_v3_screen800_2x5090.sh
```

launcher 在启动前要求 GPU2、GPU3 各至少有 `15000 MiB` 空闲显存，以容纳
每卡两个并发 job。启动后使用：

```bash
experiments/status_tpd_clean_v3_screen800_2x5090.sh
```

观察四个独立 unit、epoch、summary 与 GPU2/3 显存。不得通过再次运行
launcher 来恢复已有 run。

## 7. 完成边界

每个 worker 仅在下列条件全部成立后输出
`TPDCLEANV3_2X_COMPLETE`：

1. `metrics.jsonl` 恰有 800 行；
2. `summary.json` 为内部验证完成状态；
3. `best.pth.tar` 与 `best_miou.pth.tar` 均完成 Pd–Fa sweep；
4. 新旧 source lock 与训练数据摘要在训练前后均一致。

现有四卡 finalizer 要求四个不同 GPU UUID，不适用于本两卡结果。两卡结果
必须保持独立，不得伪装为四卡 formal run；如需统一自动裁决，应另建与本
manifest schema 和“两 UUID、每卡两 job”契约一致的 finalizer/validator。
