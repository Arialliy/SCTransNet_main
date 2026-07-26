# TPD-Clean-v4 两张 RTX 5090 运行协议

该文件只规定资源映射和运行隔离；模型公式、指标及晋级门槛见
`experiments/TPD_CLEAN_V4_PROTOCOL.md`。

## 固定任务

| 任务 | 物理 GPU | GPU UUID |
| --- | ---: | --- |
| Full, seed 42 | 2 | `GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562` |
| Capacity, seed 42 | 3 | `GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3` |
| Full, seed 3407 | 3 | `GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3` |
| Capacity, seed 3407 | 2 | `GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562` |

仅使用 GPU 2 和 3，每张卡并发两路。GPU 0 和 1 不进入
`CUDA_VISIBLE_DEVICES`、任务数组或状态检查。

## 运行约束

- 四路都是 fresh 800-epoch 训练，不能从 v3 checkpoint 续训；
- 结果根：
  `experiments/results/tpd_clean_v4_screen800_2x5090_v1`；
- run tag：`screen800_pd_fp32_shared2x5090_v1`；
- systemd unit 前缀：`sctransnet-tpd-clean-v4-2x-`；
- 每个任务固定
  `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、
  `OPENBLAS_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`；
- AMP 关闭；
- 每卡启动前至少需要 15,000 MiB 空闲显存，单 worker 自检至少
  7,500 MiB；
- 两个 seed 使用交叉 GPU 映射，墙钟时间、吞吐和能效不作为模型比较指标。

## 启动顺序

1. 所有待锁文件稳定后生成
   `experiments/tpd_clean_v4_screen800_2x_source_lock.json`；
2. 完成结构/训练/评估/运行层测试；
3. 完成 CPU、GPU2、GPU3 smoke；
4. 执行：

   ```bash
   experiments/launch_tpd_clean_v4_screen800_2x5090.sh --preflight
   ```

5. 只有 preflight 返回 `TPDCLEANV4_2X_PREFLIGHT_OK` 时才执行正式启动；
6. 启动后用：

   ```bash
   experiments/status_tpd_clean_v4_screen800_2x5090.sh
   ```

   查看四路 epoch、summary、显存和利用率。

## 结果边界

每路完成后必须具有 800 行 `metrics.jsonl`、`best.pth.tar`、
`best_miou.pth.tar`、`last.pth.tar` 和两个闭区间 Pd–Fa sweep。
V4 的阈值端点是在正式训练前预注册的，provenance 必须记录：

```text
posthoc_endpoint_completion = false
preregistered_endpoint_completion = true
endpoint_protocol_stage = before_formal_training
```

阈值 `1.0` 使用严格比较 `prediction > threshold`，对应确定的空预测工作点
`Pd=0, Fa=0`。
