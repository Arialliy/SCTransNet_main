# TPD-NER V8-MPRS-DCH V4 Tail-Aware Exact Protocol（草案）

状态：代码协议草案；尚未生成 V4 source lock，尚未授权或启动正式 GPU
训练。

## 1. 唯一候选

- variant：
  `tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on`
- parent：fresh `tpd_clean_v8_mprs_dch_full`
- relay：五节点 `q4 -> q3 -> q2`，width 8，初始化 seed 42
- NER DC scope：
  `post_centering_ner_gate_offset_not_tokenizer_mprs_dch`
- stage 4 DC support：`1`
- stage 3/2 DC support：`1-P`（`complement_tail`）
- 固定阈值：`{"4": 1.5, "3": 2.0, "2": 2.5}`
- tail support 新增参数/缓冲区：0/0

公式和阈值不是 CLI 轴。trainer 必须显式把它们传给 V4 builder，并在
architecture manifest、run identity、checkpoint identity 和 completion
summary 中重复校验。

对照角色严格分开：

- required control：V1 relay-off
  `tpd_ner_v8_mprs_dch_full_relay_off`
- paired gate predecessor：V2 relay-on
  `tpd_ner_v8_mprs_dch_v2_full_relay_on`
- structural predecessor：V3
  `tpd_ner_v8_mprs_dch_v3_full_relay_on`

正式六组件门分别执行 V4 vs V1、V4 vs V2；V3 只作为额外差值和结构前代
报告，不能替代 V1/V2 的角色。

## 2. 固定训练协议

- 数据集：NUDT-SIRST 官方训练集内部划分
- split seed：20260722
- training seed：42
- epochs：800
- batch size：16
- patch size：256
- workers：0
- validation fraction：0.20
- eval every：1 epoch
- precision：FP32，AMP=false
- base/min LR：`1e-3` / `1e-5`
- warmup：10 epochs
- loss：六个 sigmoid deep-supervision 输出的无权重 BCE 之和
- optimizer、逐 epoch schedule、数据顺序与 V3 exact 完全相同
- 固定验证阈值：0.5
- target match radius：3
- tiny area：9
- selector：沿用 V3 exact 的 Pd-primary 与 mIoU-secondary
- 独立保存 `best.pth.tar`、`best_miou.pth.tar` 和最后评估 epoch

只访问内部验证集，不访问官方测试集。

## 3. 初始化与恢复

fresh 模式只允许：

1. 从 seed 42 重新构建 V8-MPRS-DCH parent；
2. 直接构建固定 `complement_tail` V4 relay；
3. 所有 NER DC offset 从零初始化；
4. 禁止加载 V1/V2/V3 checkpoint，禁止 V3 warm-start。

exact-resume 只允许同一个 V4 run identity 的 epoch-boundary checkpoint。
在恢复 model、optimizer、scaler、RNG 或 DataLoader generator 之前，必须
校验 V4 entry schema、variant、run-id prefix、source-lock key、relay
version、DC scope、`1-P` 公式和固定阈值。V3/V4 虽然 state_dict 布局相同，
仍然禁止跨版本恢复。

## 4. 身份与产物空间

- entry schema：
  `sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_exact_entry_v1`
- source-lock key：
  `tpd_ner_v8_mprs_dch_v4_tail_aware_exact_source_lock`
- run-id prefix：
  `tpd-ner-v8-mprs-dch-v4-tail-aware-exact:`
- run tag：`formal800_exact_v4_tail_aware_seed42`
- output root：
  `experiments/results/tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1`

V4 产物不得写入、覆盖或伪装成 V3 目录和 schema。

source lock 要在 trainer、测试、本文档和公式选择证据全部稳定后单独生成。
trainer 当前已经只读绑定最终三公式 aggregate 摘要
`07f6d9b5...3ae3d7` 和完成标记摘要 `2cea9183...2d2d76`，并校验唯一入选
公式为 `complement_tail`。这两份选择产物明确不构成正式训练授权；后续
source lock 会把本代码契约一并冻结。本草案不创建 source lock，也不据此
启动训练。

## 5. GPU lane

每个进程只暴露一张卡，并始终使用逻辑 `cuda:0`。允许的物理卡仅为 GPU 2
或 GPU 3；环境必须提供对应 index、UUID、`CUDA_VISIBLE_DEVICES`、
`CUBLAS_WORKSPACE_CONFIG` 和 `PYTHONHASHSEED=42`。设备型号必须是
NVIDIA GeForce RTX 5090。

## 6. 启动门

正式训练仅在以下条件全部满足后允许：

1. V4 trainer 普通模式与 `python -O` CPU 契约测试通过；
2. fresh builder 的 manifest 严格为 `complement_tail`、`1-P` 和固定阈值；
3. V3 identity/checkpoint 在任何状态恢复前被拒绝；
4. 同 V4 epoch-boundary exact-resume 轨迹测试通过；
5. V4 source lock 已生成并复核；
6. 三公式最终选择证据摘要已写入冻结协议；
7. 输出目录不存在或符合相同 V4 exact identity。
