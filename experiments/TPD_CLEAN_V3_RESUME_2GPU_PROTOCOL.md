# TPD-Clean-v3 原四卡结果的两卡续训协议

## 1. 目标与边界

本协议只定义原 TPD-Clean-v3 四卡筛选被中断后的运行恢复层。目标是在不删除
原结果的前提下，将四个未满 800 epochs 的 run 继续训练至 800 epochs，
然后完成 `best.pth.tar` 与 `best_miou.pth.tar` 的 Pd–Fa sweep。

模型仍是冻结的 Keep–Context–Saliency 三分支 TPD-Clean-v3，不增加第四个
并列分支，不修改模型结构、数据划分、训练超参、loss、checkpoint 选择规则
或评估器。原四卡脚本、原 launch manifest、原 worker 日志和已有训练产物
不得删除。

本运行属于同一 run 的透明续训，不是新的独立随机重复。续训边界、旧/新 GPU
映射与边界文件摘要必须完整披露；耗时、吞吐和显存效率不得用于模型比较。

## 2. 原 run 身份保持不变

- 结果根：
  `experiments/results/tpd_clean_v3_screen800_4x5090_v1/`
- run tag：`screen800_pd_fp32_shared4x5090_v1`
- 数据集：NUDT-SIRST 官方训练索引的 530/133 内部划分；
- `split_seed=20260722`，不读取官方测试索引；
- target epoch：`800`；
- batch size `16`，patch size `256`，`workers=0`；
- Adam，base LR `0.001`，min LR `0.00001`；
- 10 epoch warmup 后 cosine decay；
- FP32，六输出 BCE deep supervision；
- 每 epoch 内部验证；
- threshold `0.5`，match radius `3`，tiny area `9`。

续训入口固定为：

```bash
python experiments/resume_tpd_clean_v3.py \
  --run-dir RUN_DIR \
  --device cuda:0 \
  --target-epoch 800 \
  --expected-resume-epoch BOUNDARY_EPOCH \
  --resume-gpu-uuid GPU_UUID
```

worker 必须先从 `metrics.jsonl` 与 `last.pth.tar` 独立得到同一
`BOUNDARY_EPOCH`，且满足 `1 <= BOUNDARY_EPOCH < 800`，再调用续训入口。

## 3. 连续性实现与声明

续训引擎严格恢复边界 checkpoint 中的模型参数、Adam 状态与 GradScaler
状态。由于旧 checkpoint 未保存完整 RNG 状态，引擎不把重启描述为直接 RNG
恢复，而是按原 seed 重建 `workers=0` 的 DataLoader，并在不执行优化的情况
下重放全部已完成 epoch 的 shuffle、crop 与 flip 数据流，再从
`boundary_epoch + 1` 开始训练。

引擎把每次恢复写入 run directory 内的 resume provenance 与 segments
记录，并将对应摘要绑定到后续 checkpoint 和最终 summary。连续性声明限定为：

- model、Adam 与 scaler 状态已恢复；
- shuffle/crop/flip 数据流已从原 seed 重放；
- 进程发生过重启；
- 不主张 same-process continuity；
- 不主张 CUDA bitwise continuity。

四路续训会分别重放已完成 epoch 的 DataLoader 数据流。为避免四个进程在重放
阶段产生 CPU 线程过量竞争，worker 必须在调用任何 Python 进程之前固定：

```bash
v3_cpu_threads=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

该限制只约束每个续训进程使用的 CPU 数值计算线程，不改变 `workers=0`、数据
顺序、模型、优化器或训练超参。START 记录必须输出 `cpu_threads=1`；resume
manifest 的 `resource_snapshot` 必须逐项记录上述四个环境变量，policy 必须记录
`"cpu_replay_thread_cap": 1`，以便复核续训时实际采用的 CPU 上限。

外层 worker 的只读边界副本与 resume manifest 进一步绑定恢复前文件摘要、
原 GPU 和 resume GPU，使这一过程可以独立复核。

## 4. GPU2/3 交叉平衡映射

允许的物理卡仅为 GPU index `2` 与 `3`。原本就在 GPU2/3 的两个 seed3407
任务保持原卡；原 GPU0/1 的 seed42 任务迁入 GPU2/3，并使每卡恰好包含一个
Full 和一个 capacity control：

| job | variant | seed | resume GPU index | resume GPU UUID |
| --- | --- | ---: | ---: | --- |
| full-s42 | `tpd_clean_v3_full` | 42 | 3 | `GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3` |
| cap-s42 | `tpd_clean_v3_sal_capacity` | 42 | 2 | `GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562` |
| full-s3407 | `tpd_clean_v3_full` | 3407 | 2 | `GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562` |
| cap-s3407 | `tpd_clean_v3_sal_capacity` | 3407 | 3 | `GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3` |

GPU2 运行 cap-s42 与 full-s3407；GPU3 运行 full-s42 与 cap-s3407。
launcher 必须严格验证映射逐项一致、仅有两个 UUID、每个 UUID 恰出现两次，
并核验两个 UUID 当前分别解析为 index 2 与 3。

## 5. 新运行层与隔离产物

- worker：
  `experiments/run_tpd_clean_v3_resume_2x5090_worker.sh`
- launcher：
  `experiments/launch_tpd_clean_v3_resume_2x5090.sh`
- status：
  `experiments/status_tpd_clean_v3_resume_2x5090.sh`
- unit 前缀：
  `sctransnet-tpd-clean-v3-resume-2x-`
- source lock：
  `experiments/tpd_clean_v3_resume_2x_source_lock.json`
- source lock schema：
  `sctransnet_tpd_clean_v3_resume_2x_source_lock_v1`

新恢复证据统一写入原结果根下的隔离目录：

```text
resume_2x5090_v1/
├── .locks/
├── boundaries/
├── logs/
└── manifests/
```

resume manifest schema 固定为
`sctransnet_tpd_clean_v3_resume_2x5090_launch_v1`，必须记录原 GPU UUID、
resume GPU UUID/index、边界 epoch、边界 manifest 摘要、新旧 source lock
摘要、训练数据摘要和启动时资源快照。policy 必须明确：

```json
{
  "in_place_resume": true,
  "fresh_run": false,
  "original_results_preserved_by_boundary": true,
  "immutable_resume_boundary": true,
  "allowed_gpu_indices": [2, 3],
  "concurrent_jobs_per_gpu": 2,
  "counterbalanced_mapping": true,
  "cpu_replay_thread_cap": 1,
  "efficiency_comparison_allowed": false,
  "official_test_accessed": false,
  "amp": false
}
```

外层 worker 与 launcher 的运行标记统一使用
`TPDCLEANV3_RESUME_2X` 前缀；被调用的续训引擎保留其
`TPDCLEANV3_RESUME_*` provenance/epoch 标记。

## 6. 不可覆盖的 resume boundary

worker 在调用续训器前必须同时持有：

1. 原四卡 job 锁；
2. 新 resume job 锁。

随后验证：

- run directory 是第 2 节规定的唯一规范路径；
- `metrics.jsonl` epoch 从 1 连续到边界；
- `last.pth.tar` 的 epoch 与 metrics 行数完全一致；
- checkpoint 的 variant、seed、dataset、split hashes 与最后一行验证指标一致；
- protocol 的 target epoch、run tag 和内部验证策略一致；
- 原 launch manifest 的 variant、seed 与 run directory 一致；
- 所有必需文件均为普通文件且不是符号链接。

每个边界目录名包含 variant、seed 与 boundary epoch，已存在则终止，禁止覆盖。
边界原子复制并逐文件复核 SHA-256：

- `metrics.jsonl`
- `last.pth.tar`
- `best.pth.tar`
- `best_miou.pth.tar`
- `protocol.json`
- `split.json`
- 原 launch manifest
- 原 worker log

边界目录发布后设为只读。resume manifest 与 resume log 同样禁止覆盖。因此，
重复运行 launcher 不能悄悄重用或改写同一续训边界。

## 7. 启动门槛

launcher 启动前必须满足：

1. 四个原 `sctransnet-tpd-clean-v3-*.service` worker 均为 inactive；
2. 四个新 `sctransnet-tpd-clean-v3-resume-2x-*.service` 均不存在；
3. 四个原 run directory 及 metrics、last、best、best_miou、protocol、split
   均存在；
4. 新 source lock、原 v3 source lock、Clean-v2 lock 与 NER lock 均匹配；
5. GPU2/3 均为 NVIDIA GeForce RTX 5090；
6. GPU2/3 启动前各至少有 `17000 MiB` 空闲显存；
7. 新 resume manifest 与 log 均不存在。
8. worker 在首个 Python 进程之前已将 OMP、MKL、OpenBLAS 与 NumExpr 的线程
   上限固定为 `1`，且该策略已由新 source lock 绑定。

source lock 在所有恢复代码稳定后生成。语法与 preflight 命令：

```bash
bash -n experiments/run_tpd_clean_v3_resume_2x5090_worker.sh
bash -n experiments/launch_tpd_clean_v3_resume_2x5090.sh
bash -n experiments/status_tpd_clean_v3_resume_2x5090.sh
experiments/launch_tpd_clean_v3_resume_2x5090.sh --preflight
```

正式启动命令仅为：

```bash
experiments/launch_tpd_clean_v3_resume_2x5090.sh
```

状态查看：

```bash
experiments/status_tpd_clean_v3_resume_2x5090.sh
```

## 8. 完成条件

每个 worker 仅在以下条件全部成立后输出
`TPDCLEANV3_RESUME_2X_COMPLETE`：

1. 原 run 的 `metrics.jsonl` 恰有 800 行；
2. `summary.json` 为 complete，且仍标记 internal validation only；
3. `best.pth.tar` 与 `best_miou.pth.tar` 均完成 Pd–Fa sweep；
4. 新旧 source lock 与训练数据摘要在续训前后一致；
5. resume manifest、边界目录与新日志均保留。

四路完成后仍须运行适用于续训 manifest 的完整性审计和统一门槛裁决。仅完成
续训不改变“多种子证据建立前不替换 TPD-v1 主线”的既定结论。
