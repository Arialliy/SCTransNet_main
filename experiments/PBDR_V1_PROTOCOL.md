# PBDR V1 零训练审计冻结协议

状态：六角色执行完成，Trigger 未通过；未授权可训练 PBDR，未启动 Formal1000。

## 冻结输入

```text
seed=42
split=img_idx/test
datasets=NUAA-SIRST, NUDT-SIRST, IRSTD-1K
roles=best_miou, best_pd
threshold=0.5, operator=>

identity_inputs.json
sha256=3c0825f7a45984ecedd85edd32d207080244a943c77ff37d4f3a7f67a6897712

dorf_v1_input_manifest.json
sha256=38bb9a2e4ae5662ae32da6b346444e6d34f5aba57ca13c5ae1dc4516f4230359

three_dataset_v2_protocol.json
sha256=00edc6413dead3678f8b4c162c74ea7d8602f55ff413cb20ad1664587380319f
```

六角色只用于零训练分析，不得作为 Formal1000 warm-start。

## 冻结实现

```text
analysis/analyze_three_dataset_pbdr_zero_training_v1.py
sha256=341b0b62841dfa065b6a01010044098c3c0a899a3c1b7f9f904400c8962532c6

analysis/compare_three_dataset_pbdr_zero_training_v1.py
sha256=264f421c51e37f00e37e111de16fa7f2cbe412291ad495d380d326afb101fda9

analysis/compare_three_dataset_dorf_v1.py
sha256=7503e738167a61103c14d251afd36ef668133caa099c3cabc7e7ce7e9cdb9cb5
```

## 统一数学设置

所有角色统一关闭 CUDA matmul 与 cuDNN TF32，float32 matmul precision 为 `highest`；
cuDNN benchmark 关闭、deterministic 开启，并启用 deterministic algorithms。禁止按角色
选择设置来追旧评估数值。旧评估 drift 只报告；PBDR 候选始终与同一次 forward 的精确
`g=0` 比较。

## 路由与保护

```text
P = nearest(dilate3(binary(q4 spatial-tail z>1.5))).detach()
z(g) = z_out + g * [P*relu(z_d0-z_out) - (1-P)*relu(z_out-z_d0)]
```

固定点：

```text
g=0                     identity
g=0.125,0.25,0.50,0.75 authorization candidates
g=1.0                   max/min oracle，不可授权
```

每张图只运行一次原模型 forward，并从 `tpd_ner.fusions["4"]`、`outc`、`outconv`
各捕获一次 raw tensor。六个点必须复用这一组张量。

## T1–T5

- T1：三个 `best_miou` 至少 2/3 安全不退化，且 matched target、unmatched predicted
  pixels、mIoU、nIoU 中至少两项严格改善；tiny matched count 不下降。
- T2：同一候选 `g` 在六角色均无冻结 DORF severe degradation。
- T3：三个 `best_miou` 各有至少一个 Current 漏检目标，其 GT 内存在 `P=1,d0>out`。
- T4：三个 `best_miou` 各有至少一个 Current unmatched FP 像素满足 `P=0,out>d0`。
- T5：六角色均满足 `2*protected_background_pixels < background_pixels`。

只有同一个 `g∈{0.125,0.25,0.50,0.75}` 同时通过 T1–T5，才授权实现可训练 PBDR；
即使通过，Formal1000 也要等模型代码、导出、梯度、resume 与 source-lock 测试完成。

## 输出

```text
results/three_dataset_pbdr_zero_training_v1/runs/<dataset>/<role>/evaluation.json
results/three_dataset_pbdr_zero_training_v1/comparison/seed42_six_role/decision.json
results/three_dataset_pbdr_zero_training_v1/comparison/seed42_six_role/decision.md
```

写入策略为 write-once。任何角色失败时停止矩阵，不留下该角色半成品。

## 最终结果

```text
decision=PBDR_GLOBAL_FIXED_G_SCREEN_FAILED
passing_authorization_gates=[]
T1 passed datasets by g=0.125/0.25/0.50/0.75: 0/3, 0/3, 1/3, 1/3
T2: pass, pass, fail, fail
T3=false
T4=false
T5=true
pbdr_implementation_authorized=false
pbdr_training_authorized=false
```

本协议至此关闭 PBDR-V1，不追加固定 g，不启动 Formal1000。完整机器结果见
`results/three_dataset_pbdr_zero_training_v1/comparison/seed42_six_role/decision.json`。
