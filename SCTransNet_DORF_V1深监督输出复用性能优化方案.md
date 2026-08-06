# SCTransNet DORF V1：深监督输出复用性能优化方案

## 1. 当前结论与优化目标

当前正式完整模型保持：

```text
TPD8 + NER4 + QFG2 + TSS-off
seed = 42
datasets = NUAA-SIRST / NUDT-SIRST / IRSTD-1K
```

GCSF 固定跳连重分配没有通过训练启动门；DS-GA V1 又表明六头监督不存在可跨三个
数据集、两个 checkpoint 角色稳定复现的全局冲突。因此下一步不再统一修改六个 BCE
权重，也不继续改 TPD、NER 或 QFG 的内部公式。

本轮直接处理当前推理图中的一个可执行性能问题：网络已经训练了融合多尺度预测的
`d0`，但正式推理只返回 `out`，没有使用 `d0`。目标是在不增加新特征分支、不修改
现有主线的前提下，判断已训练的多尺度 readout 能否改善 Pd–Fa–mIoU–nIoU 联合
工作点。Original baseline 也具有同源 `d0/outconv`，因此必须同步审计 Original，防止
把 baseline 共有的输出策略收益误记为完整模型的新增结构贡献。

## 2. 当前代码事实

正式六输出训练图计算：

```python
gt5 = upsample(gt_conv5(d5))
gt4 = upsample(gt_conv4(d4))
gt3 = upsample(gt_conv3(d3))
gt2 = upsample(gt_conv2(d2))
d0 = outconv(cat(gt2, gt3, gt4, gt5, out))
```

训练时六个输出均有独立 BCE：

```text
gt5 + gt4 + gt3 + gt2 + d0 + out
```

但正式推理路径固定返回：

```python
sigmoid(out)
```

因此 `d0` 不是未训练的新头，而是 checkpoint 中已经训练、保存并参与反向传播的
多尺度融合 readout。DORF V1 只改变最终 readout 的使用方式。

## 3. DORF V1 公式

设最终 decoder logit 为 `z_out`，多尺度融合 logit 为 `z_d0`。定义：

\[
z_{\alpha}=z_{out}+\alpha(z_{d0}-z_{out})
=(1-\alpha)z_{out}+\alpha z_{d0}
\]

\[
p_{\alpha}=\sigma(z_{\alpha})
\]

正式零训练筛选只允许以下预注册模式：

| mode | α | 含义 |
|---|---:|---|
| `current_out` | 0.00 | 当前正式输出，必须与历史结果逐指标一致 |
| `dorf_a025` | 0.25 | 小幅复用 d0 |
| `dorf_a050` | 0.50 | out 与 d0 等权 logit 融合 |
| `dorf_a075` | 0.75 | 主要使用 d0，保留部分 out |
| `d0_only` | 1.00 | 只使用已有 d0 |

禁止在看到结果后增加负 α、α>1、逐数据集 α 或更密集网格。所有融合必须发生在
sigmoid 之前的 raw logit 空间，不能混合已经 sigmoid 的概率。

## 4. 研究边界

DORF V1：

- 不新增训练参数或 persistent buffer；
- 不修改 TPD8、NER4、QFG2、decoder 或六头 BCE；
- 不重新选择输入 checkpoint；
- 不写 derived checkpoint；
- 每个 test batch 只运行一次完整模型，再从同一次 forward 取得 `z_d0/z_out`；
- `current_out` 直接采用该次模型 forward 返回的正式概率，不能通过融合公式生成；
- `sigmoid(hooked z_out)` 必须与该次模型返回概率逐元素完全一致；
- α=0 的 count 必须与绑定历史 evaluation 精确一致，浮点指标按第 7 节容差重放；
- 固定阈值仍为 0.5；阈值 1.0 只记录 `Pd=0, Fa=0` 空预测端点。

本轮是 test-selected development 性能筛选，不构成独立测试或跨 seed 稳定性结论。
两个方法在本轮都使用相同的四个非零 α 搜索预算，但这不能消除历史 Final-family
累计配方搜索预算高于 Original 的事实；后续结果必须继续披露该边界。

## 5. 十二角色实验矩阵

固定使用当前三个 TSS-off Final run，以及 three-dataset V2 中相同 seed42、相同
`img_idx` 协议的三个 Original run：

| 方法 | 数据集 | checkpoint 角色 | 模式数 |
|---|---|---|---:|
| Final TSS-off | NUAA / NUDT / IRSTD-1K | best_miou / best_pd | 5 |
| Original | NUAA / NUDT / IRSTD-1K | best_miou / best_pd | 5 |

共 `2 methods × 3 datasets × 2 roles × 5 modes = 60` 个固定 checkpoint 工作点。
Final 的主裁决为 `best_miou`，`best_pd` 只作为严重退化否决；不能把两个角色混成
平均数掩盖退化。Original 使用同一个 α 重放，作为公平基线与竞争力非退化检查，不能
为 Original 单独选择 α。

12 个 checkpoint、12 份历史 evaluation、6 份 summary/protocol、数据协议与背景像素
FP sidecar 已在任何 DORF 输出产生前写入只读输入 manifest：

```text
path=results/three_dataset_dorf_v1/manifests/dorf_v1_input_manifest.json
sha256=38bb9a2e4ae5662ae32da6b346444e6d34f5aba57ca13c5ae1dc4516f4230359
input_count=12
historical_metric_authority=bound_evaluation_json_only
checkpoint_embedded_metrics_fallback_allowed=false
```

分析器和比较器必须逐项验证 manifest 内的 path/SHA/epoch/method/dataset/role；禁止按
目录动态发现另一份 checkpoint 或 evaluation。历史 evaluation 是固定阈值指标的唯一
主要权威源；GT-background FP 的 α=0 精确整数来自 manifest 绑定的
`additive_joint_metrics_v1.json`。checkpoint 内嵌 `test_metrics` 不能作为替代源。
每项的精确文件路径只能按 manifest 中的 `run_dir` 与 `checkpoint_role` 推导为
`run_dir/checkpoints/{role}.pth.tar` 和 `run_dir/evaluations/{role}.json`，不得使用 glob、
候选搜索或其他文件名。

每个工作点至少记录：

```text
matched targets / total targets / Pd
matched tiny targets / total tiny targets / tiny-Pd
component Fa / unmatched predicted pixels
background false-positive pixels
false objects per image
mIoU / nIoU
pixel precision / recall / F1
test loss
max/mean probability difference to current_out
```

## 6. 冻结性能门

逐数据集、逐角色将候选与 `current_out` 比较，沿用上一轮完整模型级性能门。

### 6.1 Safe

必须同时满足：

```text
Δmatched_target > -2
Δmatched_tiny > -2
ΔmIoU > -0.005
ΔnIoU > -0.005
component FP reduction > -5%
background pixel FP reduction > -5%
```

### 6.2 Material

以下至少一项成立：

```text
Δmatched_target >= +2
Δmatched_tiny >= +2
ΔmIoU >= +0.005
ΔnIoU >= +0.005
component FP reduction >= +5%
background pixel FP reduction >= +5%
```

`safe-material = safe AND material`。

### 6.3 Severe

以下任一项成立即为严重退化：

```text
Δmatched_target <= -2
Δmatched_tiny <= -2
ΔmIoU <= -0.01
ΔnIoU <= -0.01
component FP increase >= 25%
background pixel FP increase >= 25%
```

两种 FP reduction 均固定为：

\[
r=(FP_{reference}-FP_{candidate})/FP_{reference}
\]

其中 component FP 使用 `unmatched_predicted_pixels`，background pixel FP 使用全部
GT-background 上的阳性预测像素。零分母语义固定为：

```text
reference=0, candidate=0: reduction=0, Safe按0判断, Material=false, Severe=false
reference=0, candidate>0: reduction=null, Safe=false, Material=false, Severe=true
```

所有门使用未舍入全精度值；Markdown 中的显示小数不参与裁决。

### 6.4 Trigger A

同一个非零 α 必须同时满足：

```text
Final best_miou safe-material datasets >= 2/3
Final severe units across best_miou + best_pd = 0/6
Final(alpha) vs Original(0): no newly true severe condition
Final(alpha) vs Original(alpha): no newly true severe condition
existing primary Final(0)-vs-Original(0) safe-material cells are preserved
alpha0 historical replay passed = true for all 12 checkpoints
all 12 checkpoint/evaluation units engineering valid = true
```

Final-vs-Original 竞争力检查对每个数据集/角色分别调用同一套
`compare_direction(Final, Original)`，并保留 `dataset × role × severe condition` 布尔
mask，不能只比较 severe 总数。设：

```text
M0  = severe_mask(Final(0),     Original(0))
Ma0 = severe_mask(Final(alpha), Original(0))
Maa = severe_mask(Final(alpha), Original(alpha))
```

正式要求 `Ma0 ⊆ M0` 且 `Maa ⊆ M0`，即两种 Original 锚点都不能出现任何新的严重
退化条件。对于 `best_miou` 中 `Final(0)` 相对 `Original(0)` 已经达到 safe-material
的具体数据集，候选在 `Final(alpha) vs Original(0)` 和
`Final(alpha) vs Original(alpha)` 两个锚点下都必须继续保持 safe-material。该规则不
要求当前混合权衡突然变成全指标全面领先。Original 自身 DORF 收益完整报告，但不参与
Final 的 2/3 主门。

通过时只授权该 α 的 DORF 生产代码；不通过时停止 DORF，不增加自适应 gate。多个 α
同时通过时，选择绝对值最小的 α，避免结果后再定义聚合分数。由于本轮 checkpoint 是
按 `out` 选择的，若要形成同政策正式 selector 比较，必须重跑 Final+Original 各三个
数据集，共 6 个 fresh formal1000 run，并都按 DORF 输出选择 best-mIoU/best-Pd；在此
之前不授权正式训练，也不能宣称 DORF 已经完成公平选模。

## 7. 工程实现

新增：

```text
analysis/analyze_three_dataset_dorf_v1.py
analysis/compare_three_dataset_dorf_v1.py
tests/test_analyze_three_dataset_dorf_v1.py
tests/test_compare_three_dataset_dorf_v1.py
```

分析器必须：

1. 严格绑定并验证上述只读输入 manifest 与其 SHA；
2. Final 使用 `build_final_inference_model_from_training_state_dict`：568 个训练 state
   key 移除 4 个训练期 TSS key 后，严格加载 564-key inference 图；
3. Original 使用 `build_paper_model('original', ..., training=False)`，对 510 个 state
   key 执行 `strict=True` 加载；
4. 两个方法始终保持 `model.eval()` 与 `mode=test`，禁止切换到 `mode=train`；
5. hook 只能精确挂到 `model.outc` 和 `model.outconv`。每 batch 先清空捕获，各自必须
   恰好调用一次，并验证 raw logits 的 shape/dtype/device/finite；hook 必须在
   `finally` 中移除；
6. `current_out` 直接使用模型正式返回概率，并逐元素核对
   `sigmoid(hooked z_out)`；其余四种模式才从同一对 raw logits 生成；
7. 在原图有效区域计算指标，排除 padding；
8. α=0 只对 manifest 绑定的历史 evaluation 重放：count 精确相等；mIoU、nIoU、
   pixel precision/recall/F1 的 absolute tolerance 为 `1e-4`，test loss 为 `1e-7`，
   其余正式浮点指标为 `1e-15`；background FP 整数与绑定 sidecar 精确相等；
9. 模型 state SHA、`mode` 与 training flag 在完整分析前后必须不变；
10. 绑定 checkpoint、summary、protocol、evaluation、数据 manifest、sidecar、evaluator
    和分析源码 SHA；
11. 写一次性 `evaluation.json`，不保存完整概率缓存或 derived checkpoint。

未来采用 DORF 后，`gt_conv2..5 + outconv` 将成为正式部署必需组件，不能再从纯
`out-only` 导出图删除；当前两种推理图本来就计算这些模块，因此本轮不增加实际 forward
计算。

比较器必须从 12 份原始 evaluation 重算 safe/material/severe、两种竞争力锚点与
Trigger A，普通 Python 和
`python -O` 结果必须逐字节一致。

## 8. 执行顺序

```text
Phase 1  完成分析器、比较器和 CPU 单元测试
Phase 2  GPU0/1/2 并行完成 Final 三数据集 best_miou/best_pd
Phase 3  GPU0/1/2 并行完成 Original 三数据集 best_miou/best_pd
Phase 4  比较器裁决并同步历史总汇
Phase 5  仅 Trigger A 通过时实现 DORF 正式推理图
Phase 6  仅另行授权时启动 Final+Original × 三数据集，共6个 fresh formal1000
```

GPU3 上已有其他训练任务，本轮不打断；DORF 零训练筛选只使用空闲的 GPU0/1/2。

## 9. 当前状态

```text
protocol_frozen=true
input_manifest_frozen=true
input_manifest_sha256=38bb9a2e4ae5662ae32da6b346444e6d34f5aba57ca13c5ae1dc4516f4230359
model_mainline_changed=false
training_loss_changed=false
dorf_v1_implementation_started=true
dorf_v1_zero_training_audit_complete=true
dorf_v1_trigger_a_passed=false
dorf_v1_production_implementation_authorized=false
dorf_v1_formal_training_authorized=false
decision=DORF_V1_ZERO_TRAINING_TRIGGER_FAILED
paper_core_established=false
stability_claim_supported=false
```

## 10. 正式执行结果

12 个固定 checkpoint/evaluation 单元已全部完成，α=0 的历史指标重放全部通过，
12/12 工程单元有效。普通 Python 与 `python -O` 的 32 项定向测试均通过；正式比较器
在普通与 `python -O` 下生成的 JSON/Markdown 逐字节一致。

| mode | α | Final best-mIoU safe-material | Final 六角色 severe | Ma0/Maa 新 severe | Trigger A |
|---|---:|---:|---:|---:|:---:|
| `dorf_a025` | 0.25 | 0/3 | 2/6 | 0/0 | false |
| `dorf_a050` | 0.50 | 0/3 | 4/6 | 2/0 | false |
| `dorf_a075` | 0.75 | 0/3 | 4/6 | 2/0 | false |
| `d0_only` | 1.00 | 0/3 | 5/6 | 3/0 | false |

最小干预 `α=0.25` 已通过双 Original 锚点的竞争力非退化检查，但仍未通过 Final
自身主门：

- NUAA-SIRST best-mIoU：少检 2 个目标，mIoU -0.001219，nIoU -0.004271；
  component FP 反而增加 3.56%，background FP 降低 3.84%；
- NUDT-SIRST best-mIoU：少检 2 个目标、少检 1 个 tiny target；mIoU/nIoU 基本持平，
  component FP 降低 12.40%、background FP 降低 4.23%；
- IRSTD-1K best-mIoU：目标与 tiny 计数不变，但所有变化均未达到 5%/0.005 实质门。

更大的 α 会进一步把 readout 推向低 FP、低响应工作区：`d0_only` 在 5/6 个 Final
单元触发 severe。虽然 NUDT-SIRST best-Pd 的 `α=0.50/0.75/1.0` 能保持目标计数并
降低 component FP，同时略升 mIoU/nIoU，但该收益没有出现在主裁决 best-mIoU，也不
能抵消 NUAA、NUDT best-mIoU 与 IRSTD best-Pd 的目标损失。

因此正式裁决为：

```text
decision=DORF_V1_ZERO_TRAINING_TRIGGER_FAILED
selected_mode=null
selected_alpha=null
dorf_v1_production_implementation_authorized=false
fresh_formal1000_launch_authorized_by_this_comparator=false
model_mainline_changed=false
training_loss_changed=false
```

DORF V1 不进入当前完整模型，不实现固定 α 生产图，也不启动 Final+Original 的 6 个
fresh formal1000。正式模型继续保持 `TPD8 + NER4 + QFG2 + TSS-off`。本轮说明
`d0` 的主要作用更接近保守虚警抑制，直接全局融合会用目标检出和区域质量换取较少 FP；
下一项结构优化必须显式保护目标响应，而不能继续做全图常系数输出平均。

正式产物：

- `results/three_dataset_dorf_v1/comparison/seed42_twelve_role/decision.md`
- `results/three_dataset_dorf_v1/comparison/seed42_twelve_role/decision.json`
- `results/three_dataset_dorf_v1/runs/` 下 12 份 `evaluation.json`
- `results/three_dataset_dorf_v1/manifests/dorf_v1_input_manifest.json`
