# SCTransNet TPD–NER V8-MPRS-DCH V3 DC knockout 诊断协议（V2）

## 1. 身份与结论边界

本协议只定义 V3 formal800 完成后的同 checkpoint、推理期 DC offset
knockout。它不是新训练，不选择新 checkpoint，不访问官方测试集，也不增加
seed。它只接受已经完成的 versioned selection-contract repair V1 聚合，
不接受因 legacy baseline 字段兼容问题而未能发布的原 frozen aggregate 路径。

固定身份：

```text
artifact kind       = dc_knockout_diagnostic
scope               = evaluation_only_same_checkpoint_counterfactual
dataset             = NUDT-SIRST
training seed       = 42
split seed          = 20260722
validation count    = 133
official test       = not accessed
formal gate impact  = false
decision authority  = false
```

`complete` 只表示诊断八行齐全、来源和哈希通过验证，不表示
`FULL_MODEL_GATE_PASSED`。本诊断不得新增、覆盖或改变正式六项 AND
裁决中的任何布尔量。

权威固定 spec：

```text
experiments/tpd_ner_v8_mprs_dch_v3_dc_knockout_spec.py
```

V2 是一次只修运行契约的版本化修订。首次 V1 启动在任何模型或数据推理前
终止，原因是 finalizer 的 evaluator 子进程环境遗漏了评估器既有的固定要求
`CUBLAS_WORKSPACE_CONFIG=:4096:8`。当时没有发布 sweep、aggregate 或完成
marker，旧 V1 输出根也不存在；V2 不复用 V1 结果并改用新的输出根。

## 2. 运行前置条件

只有以下条件全部满足才能运行：

1. canonical V3 formal800 run 具有严格 complete summary、连续 1..800
   metrics、合法 exact journal 和 checkpoint；
2. 正式 V3 training/acceptance locks 均通过只读验证；
3. 下列 repair V1 权威产物全部存在、是普通文件并且 marker 输出哈希一致：

   ```text
   comparison_selection_contract_repair_v1/
   ├── POSTPROCESS_COMPLETE_SELECTION_CONTRACT_REPAIR_V1.json
   ├── tpd_ner_v8_mprs_dch_v3_formal800_comparison_selection_contract_repair_v1.json
   └── tpd_ner_v8_mprs_dch_v3_formal800_comparison_selection_contract_repair_v1.md
   ```

4. repair report 必须满足：

   ```text
   decision = RETURN_TO_MODEL_OPTIMIZATION
   aggregate_full_model_gate_passed = false
   comparison_contract.selection_contract_repair.
     each_variant_uses_own_selected_checkpoints = true
   ```

   即 baseline、V1、V2、V3 分别比较自身内部验证选出的
   `best.pth.tar` 与 `best_miou.pth.tar`，不要求不同模型的 checkpoint
   epoch、路径或哈希相同；
5. repair attestation 必须重新验证，并绑定 repair wrapper、repair protocol、
   frozen formal postprocessor、既有协议与上游聚合；knockout source lock
   还必须直接记录 repair wrapper/protocol/attestation 的路径和 SHA-256；
6. 原始正式 V3 两份 checkpoint 与两份 learned-V3 sweep 必须存在且哈希一致；
7. 独立 knockout diagnostic source lock 已冻结并通过验证；
8. 正式训练、checkpoint、sweep、repair aggregate、marker 和 attestation
   在诊断前后只读不变。

不得为了运行本诊断修改任何 formal V3、V2、V1 或 baseline source/lock/result。

## 3. 固定 2 × 4 矩阵

源 checkpoint 固定为：

```text
best.pth.tar       -> best_validation_pd_primary
best_miou.pth.tar  -> best_validation_miou_secondary
```

每个 checkpoint 固定执行以下四个内存干预，并按此顺序保存：

| Mode | 被置零的 state keys |
|---|---|
| `zero_all_dc` | `tpd_ner.dc_offsets.4`, `.3`, `.2` |
| `zero_dc_stage4` | `tpd_ner.dc_offsets.4` |
| `zero_dc_stage3` | `tpd_ner.dc_offsets.3` |
| `zero_dc_stage2` | `tpd_ner.dc_offsets.2` |

因此 aggregate 必须恰好有八行。正式 learned-V3 两行仅作为只读 reference，
不计入这八行。

单 stage knockout 必须保留另两个 offset 的 checkpoint 学习值。干预只允许
发生在严格加载后的内存 model state；不得写派生 checkpoint。每个 mode 必须
从未经前一 mode 修改的相同 source state 重建或恢复。

`zero_all_dc` 是同一 V3 checkpoint 的推理反事实，不等价于 V2 独立训练轨迹，
不得替代 V2 structural predecessor。

## 4. 评估口径

入口固定为：

```text
experiments/evaluate_tpd_ner_v8_mprs_dch_v3_dc_knockout.py
```

输出根固定为：

```text
experiments/results/tpd_ner_v8_mprs_dch_v3_dc_knockout_v2/
```

它与 formal V3 result root 不同。evaluator 对每个 source checkpoint 生成一份
JSON，每份包含四个完整 evaluation；aggregate 只读展开成八行。

GPU 调度固定为：

```text
best.pth.tar       -> physical GPU2
                       GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
best_miou.pth.tar  -> physical GPU3
                       GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3
logical device     = cuda:0
CUDA_DEVICE_ORDER  = PCI_BUS_ID
CUBLAS_WORKSPACE_CONFIG = :4096:8
PYTHONHASHSEED      = 42
```

两个 checkpoint 可在 GPU2/3 并行；同一 checkpoint 内四个 mode 必须从
pristine source state 顺序运行。evaluator 必须同时校验
`CUDA_VISIBLE_DEVICES`、物理卡序号和 UUID 环境绑定，拒绝 GPU0/1 或未绑定
的直接运行。finalizer 必须在创建 evaluator 子进程前，将
`CUBLAS_WORKSPACE_CONFIG=:4096:8` 写入每条固定 lane 的子进程环境；只在
计划中记录但未实际传给子进程不满足本协议。evaluator 必须再次按固定 spec
严格校验该值以及正式评估入口一致的 `PYTHONHASHSEED=42`，然后才允许调用
确定性推理配置或进入模型/数据推理。

预测与阈值合同完全复用正式 V3：

```text
prediction > threshold
score dtype = float32
fixed threshold = 0.5
Fa budgets = 1e-6, 5e-6, 1e-5, 5e-5, 1e-4
no extra threshold 0 is inserted
closed upper interval includes 1 and last float32 below 1
```

每个 evaluation 必须保留完整 raw threshold points 与 provenance，而不是只
保存五个预算摘要。

固定阈值 0.5 必须覆盖：

- matched targets / 189、Pd、Fa、mIoU、false objects/image；
- matched tiny targets / 39、tiny-Pd；
- nIoU、pixel precision、pixel recall、pixel F1。

每个 Fa budget 至少保存：

- budget；
- achieved Fa；
- threshold；
- matched targets / 189；
- Pd。

所有数值必须有限；matched/Pd 必须由 count 精确重算；achieved Fa 不得超过
对应预算。

## 5. 干预完整性

每份 evaluator output 必须绑定：

- canonical run directory 与 run identity；
- source checkpoint name、role、epoch、SHA-256、state-dict SHA-256 和
  checkpoint identity；
- validation split SHA-256；
- 原始三个 DC offset 的值；
- 每个 mode 的 zeroed keys、实际 evaluated offset 值及 effective
  state-dict SHA-256；
- source checkpoint 评估前后 SHA 不变；
- non-DC state 干预前后规范 tensor SHA 相等；
- 只有请求的 DC state keys 发生变化；
- diagnostic source-lock SHA 与 knockout spec SHA；
- evaluator、formal locks、shared metric、closed interval 和
  determinism source SHA。

tensor state SHA 必须按排序后的 key、dtype、shape 和 contiguous CPU raw
bytes 计算，不能使用不稳定的临时 pickle 文件字节作为规范身份。

## 6. 聚合与 formal gate 隔离

聚合入口固定为：

```text
experiments/postprocess_tpd_ner_v8_mprs_dch_v3_dc_knockout.py
```

聚合器只能：

1. 验证独立 diagnostic source lock；
2. 验证两份 evaluator JSON 并展开八行；
3. 只从 versioned repaired aggregate 读取同 role learned-V3 正式行作为
   reference；
4. 计算 `knockout - learned` 的有符号差值；
5. 在独立 diagnostic root 原子发布 JSON、Markdown 和 marker。

aggregate schema 明确禁止 `decision`、`performance_gate_assessment` 以及正式
六项 gate 字段。允许报告 absolute metrics、signed deltas 与描述性统计，但
不得生成“通过/失败”裁决。

聚合前后必须重新验证 formal input hashes。任何 repair 或 original V3
formal artifact 变化都应使聚合失败。原 frozen aggregate 路径不得作为
fallback。

## 7. 独立 source lock

本诊断只需要一份 `diagnostic_acceptance` source lock，不需要新的 training
lock。默认路径：

```text
experiments/tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock_v2.json
```

该锁必须在 repaired formal aggregate 完成后、诊断 evaluator 启动前冻结，
并绑定：

- 只读审计父锁
  `experiments/tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock.json`，
  其 SHA-256 必须严格等于
  `89f98ecab9c1cbcd72f40b9ba9c2083076231ad240477d81a69528c0ef9c80f7`；
- `revision=2`、上述缺失 cuBLAS 环境变量的 `repair_reason`，以及首次启动
  `inference_started=false`、sweep/aggregate/marker 发布数均为 0；
- V1 输出根
  `experiments/results/tpd_ner_v8_mprs_dch_v3_dc_knockout_v1/` 在修订时
  不存在，V2 输出根为独立的
  `experiments/results/tpd_ner_v8_mprs_dch_v3_dc_knockout_v2/`，且禁止覆盖；
- 固定 evaluator 环境
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` 与 `PYTHONHASHSEED=42`；
- 当前 formal V3 training/acceptance lock hashes；
- repair V1 completion marker、aggregate JSON/Markdown；
- repair wrapper、repair protocol、repair attestation 的路径和 hashes；
- `each_variant_uses_own_selected_checkpoints=true` 及已发布 formal
  decision；
- original formal V3 learned sweeps 和两个 checkpoint hashes；
- 固定 knockout spec 与诊断 protocol；
- evaluator、aggregate、freezer/verifier 及其执行依赖 source hashes；
- seed/split、八行矩阵、无训练、无 official test、formal read-only policy。

冻结为 no-overwrite 操作；导入 freezer 模块不得产生 lock。
旧 V1 锁必须原样保留作为父级审计证据，但不允许用修改后的 V2 source
重新验证或执行；所有新运行只能使用尚未创建的 V2 默认锁。

## 8. 独立完成 marker

marker 固定命名：

```text
DC_KNOCKOUT_COMPLETE.json
```

只有八行全部通过验证、aggregate JSON/Markdown 原子发布且 formal inputs
前后不变后才能创建。marker 至少绑定：

- diagnostic marker schema、`status=complete`；
- `artifact_kind=dc_knockout_diagnostic`；
- `diagnostic_only=true`、`affects_formal_gate=false`；
- `formal_decision_authority=false`；
- row count 8 与 matrix identity SHA；
- diagnostic source-lock SHA 与 repaired formal marker SHA；
- repaired selection-contract closure identity SHA；
- 两份 sweep、aggregate JSON/Markdown SHA；
- `formal_artifacts_unchanged=true`；
- `official_test_accessed=false`。

已存在且内容及所有输出 hashes 完全一致的 marker 应幂等复用。部分输出、
错误 schema、hash 冲突或 symbolic link 必须失败，不得被误认成完成。

## 9. 结论许可

该包只能支持以下单-seed内部验证描述：

- learned V3 在同 checkpoint 推理时是否依赖全部或某一 stage DC offset；
- 某一 knockout 对 Pd、Fa、mIoU、tiny-Pd 和五预算点的方向与幅度；
- V3 失败时哪个 stage 值得后续单因素代码实验。

它不支持多 seed 稳定性、跨数据集、官方测试、统计显著性或新的 formal
success claim。
