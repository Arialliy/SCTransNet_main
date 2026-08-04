# SCTransNet 三数据集 TSS 强度诊断与训练中执行方案

> 项目：单帧红外小目标检测  
> Original：原始 SCTransNet  
> Final 推理结构：SCTransNet + TPD8-MPRS-DCH + 五节点 NER4 Tail-Aware + QFG2-CROA  
> 训练期辅助：Target Survival Supervision（TSS）  
> 正式数据集：NUAA-SIRST、NUDT-SIRST、IRSTD-1K  
> 固定随机种子：42  
> 正式预算：1000 epochs，每 10 epochs 在各自 img_idx/test 上评估  
> 正式阈值：0.5  
> Selected checkpoint：best_miou、best_pd  
> TSS 请求权重候选：0.0025、0.005、0.01  
> TSS 标量损失比例上限：0.10  
> 当前阶段：FORMAL12_RUNNING  
> 修订日期：2026-08-03

---

# 0. 文档定位与权威顺序

本文件是当前正式实验的诊断与训练后执行补充，不重新定义已经启动的
12-run 协议，也不授权训练中途修改模型、runner、evaluator 或全局 λ
selector。

若本文件与以下冻结产物存在差异，以以下内容为准：

1. `SCTransNet_V2全数据集混合结果复盘与全局TSS配方定型方案.md`；
2. `results/three_dataset_seed42_global_tss_v2/launch/formal/launch_plan.json`；
3. launch plan 中记录 SHA256 的 runner、evaluator、selector、模型源码；
4. `results/three_dataset_v2/manifests/three_dataset_v2_protocol.json`。

当前正式 source lock 没有包含本诊断文档，因此可以修正本文件而不影响
正在运行的训练；不得修改 source lock 中的文件，否则 supervisor 会在
下一 wave 开始前拒绝继续。

当前动态状态只读取：

```text
results/three_dataset_seed42_global_tss_v2/launch/formal/launch_plan.json
results/three_dataset_seed42_global_tss_v2/launch/formal/supervisor_status.json
results/three_dataset_seed42_global_tss_v2/launch/formal/logs/
results/three_dataset_seed42_global_tss_v2/runs/
```

本次修订时第一波 NUAA Original 与 Final λ=0.0025 已在 GPU 2/3 运行。
不重复启动，不把中途 best 值当作正式结果。

---

# 1. 当前研究判断

## 1.1 已建立与未建立

当前可以建立：

```text
architecture_implementation_complete=true
architecture_frozen=true
candidate_model_retained=true
architecture_failure_supported=false
positive_tss_strength_search_authorized=true
formal12_running=true
```

当前不能建立：

```text
architecture_success=true
final_model_performance_established=true
global_tss_recipe_established=true
paper_core_established=true
stability_claim_supported=true
training_recipe_finalized=true
```

准确表述是：

> 历史固定 λ=0.005 结果证明完整模型存在正向工作点，但不同数据集和
> checkpoint 角色之间仍有 mIoU、nIoU、Pd、Fa 与 tiny-Pd 的混合权衡。
> 因此先冻结推理结构，只比较 TSS 请求强度，判断训练配方是否是当前
> 性能瓶颈之一。

`training_recipe_bottleneck_plausible=true` 只是待检验假设，不是已经
证明的结论。

## 1.2 历史结果为何支持本轮搜索

历史 V2 相对 Original 的三个独立数据集聚合为：

| checkpoint 角色 | Δmacro mIoU | Δmacro nIoU | Δmatched | Δtiny matched | Δpooled Fa |
|---|---:|---:|---:|---:|---:|
| best_miou | -0.002052 | -0.005816 | +6 | -2 | +29.9% |
| best_pd | +0.015230 | +0.012093 | +2 | 0 | -36.1% |

来源：

```text
results/four_dataset_seed42_tss_cap_v2/V2_RESULTS_SUMMARY_STOPPED_20260802.md
```

这组证据说明：

- best_pd 聚合点整体正向；
- best_miou 存在更多检出、区域质量、tiny 与 Fa 的权衡；
- 不能判断完整结构失败；
- 也不能认定 λ=0.005 已经是统一配方。

## 1.3 本轮唯一主要训练变量

三个 Final 候选之间只允许改变：

```text
requested_tss_weight ∈ {0.0025, 0.005, 0.01}
```

保持不变：

```text
TPD8-MPRS-DCH
五节点 NER4 Tail-Aware
QFG2-CROA
TSS endpoint 与 head
ratio cap = 0.10
seed = 42
epochs = 1000
eval_every = 10
threshold = 0.5
optimizer / LR / warmup / cosine
batch size / patch size / FP32
img_idx/train 与 img_idx/test
crop / augmentation 随机流
best_miou / best_pd 定义
```

正式 12-run 期间不开展：

```text
新增模型模块
TPD / NER / QFG 修改
ratio-cap 搜索
多 seed
数据集专用 λ
阈值优化
best_joint
结构消融
```

---

# 2. TSS 数学语义

## 2.1 权威公式

```text
L_seg = sum_j BCE(P_j, Y)
L_tss = sum_i BCEWithLogits(Z_i, Y_16)
Y_16 = MaxPool_16(Y)

lambda_cap =
  0.10 * stopgrad(L_seg)
  / max(stopgrad(L_tss), epsilon_fp32)

lambda_eff = min(lambda_requested, lambda_cap)
L_total = L_seg + lambda_eff * L_tss
```

代码使用：

```python
survival_loss.detach().clamp_min(epsilon)
```

不是 `survival_loss.detach() + epsilon`。

## 2.2 三个 λ 是请求上限

令：

```text
c = 0.10 * L_seg / max(L_tss, epsilon_fp32)
lambda_eff = min(lambda_requested, c)
```

| c 的区间 | λ=0.0025 | λ=0.005 | λ=0.01 |
|---|---:|---:|---:|
| c ≤ 0.0025 | c | c | c |
| 0.0025 < c ≤ 0.005 | 0.0025 | c | c |
| 0.005 < c ≤ 0.01 | 0.0025 | 0.005 | c |
| c > 0.01 | 0.0025 | 0.005 | 0.01 |

因此，三个请求值不一定在每个 minibatch 上形成三个不同的实际权重。
这也是训练后必须检查 effective-weight 分布的原因。

## 2.3 10% 上限的边界

代码约束的是：

```text
lambda_eff * L_tss <= 0.10 * L_seg
```

它是标量损失贡献上限，不是梯度范数上限。不得写成“TSS 梯度严格不超过
主分割梯度的 10%”。

## 2.4 推理结构不包含 TSS

TSS 只在训练图中读取 emb1、emb2 stride-16 endpoint。训练 checkpoint
包含 4 个 `target_survival.*` state tensor；正式推理导出删除这些状态，
保留 TPD、NER、QFG 分割路径。

因此本轮 λ 搜索改变训练目标，不改变最终推理结构。

---

# 3. 已冻结的数据协议

## 3.1 正式数据集与数量

| 数据集 | img_idx/train | img_idx/test |
|---|---:|---:|
| NUAA-SIRST | 213 | 214 |
| NUDT-SIRST | 663 | 664 |
| IRSTD-1K | 800 | 201 |

SIRST3：

```text
formal_training=false
formal_evaluation=false
global_lambda_selection=false
macro_aggregation=false
historical_only=true
```

正式入口 `experiments.three_dataset_v2_protocol` 只接受上述三个数据集。
旧 four-dataset 文件仍可保留历史常量，但不能决定本轮矩阵。

## 3.2 固定划分

只允许：

```text
datasets/<dataset>/img_idx/train_<dataset>.txt
datasets/<dataset>/img_idx/test_<dataset>.txt
```

禁止：

- 从 train 再划 model_val；
- 合并三个数据集训练；
- 改变 ID 顺序；
- 删除困难样本；
- 使用 SIRST3 的 img_idx；
- 根据结果改变划分。

## 3.3 NUAA Misc_111

正式合同：

```text
sample = NUAA-SIRST::Misc_111
split = img_idx/test
image = datasets/NUAA-SIRST/images/Misc_111.png
raw mask = datasets/NUAA-SIRST/masks/Misc_111.png
corrected mask = datasets/NUAA-SIRST/masks_corrected/Misc_111.png
image size = 325 × 220
raw mask size = 592 × 400
corrected mask size = 325 × 220
correction_id = nuaa_misc111_internal_overlay_v2
```

正式 v2 不读取 `SIRST3/masks/Misc_111.png`。原始 NUAA mask 保留且不被
覆盖；运行时不进行 resize 或左上 crop。

## 3.4 启动前数据检查

已检查 2,755 个 train/test 图像-mask 对：

```text
missing image = 0
missing effective mask = 0
size mismatch = 0
```

test 中面积不超过 9 像素的 tiny GT 数量：

```text
NUAA-SIRST = 35
NUDT-SIRST = 259
IRSTD-1K = 30
```

---

# 4. 已实现的 three-dataset v2 工程

## 4.1 实际文件

```text
experiments/three_dataset_v2_protocol.py
experiments/prepare_nuaa_misc111_overlay_v2.py
experiments/paper_three_dataset_v2.py
experiments/train_three_dataset_seed42_global_tss_v2.py
experiments/three_dataset_seed42_launch_v2.py
experiments/evaluate_three_dataset_v2.py
experiments/select_three_dataset_global_tss_recipe_v2.py
```

不得再按照旧草案创建另一套同义文件并重新启动实验。

## 4.2 已通过的启动检查

```text
normal Python tests = passed
python -O tests = passed
GPU 2 Original smoke = passed
GPU 3 Final λ=0.0025 smoke = passed
source lock = passed
pair audit = passed
```

启动前综合回归为 77 passed，并包含 21 个 subtests；普通模式与
`python -O` 均通过。

当前 resume 实现保存 model、optimizer、epoch、RNG、best_miou、best_pd
和最近 metrics 事件，并拒绝不同 λ/protocol 的状态。尚未将“完整训练与
人为中断后续训逐 tensor 完全一致”作为独立论文结果，因此本文件不声明：

```text
bitwise_exact_resume_certified=true
```

## 4.3 source lock

launch plan 已冻结：

- launcher 与 runner；
- three-dataset data protocol 与 dataset adapter；
- model builder 和全部 `model/**/*.py`；
- TSS loss；
- checkpoint/metric 实现；
- evaluator 与 metric core；
- global λ selector；
- 权威协议文档；
- three-dataset manifest、pair audit、TSS statistics。

supervisor 在每个 wave 开始前重新核对这些 SHA256。

---

# 5. 正式训练合同

## 5.1 固定超参数

```yaml
seed: 42
epochs: 1000
begin_test: 10
eval_every: 10
candidate_epochs: [10, 20, ..., 1000]
batch_size: 16
patch_size: 256
workers: 0
precision: FP32
amp: false
optimizer: Adam
base_lr: 0.001
min_lr: 0.00001
warmup_epochs: 10
schedule: linear_warmup_then_cosine
threshold: 0.5
match_radius: 3
tiny_area: 9
ratio_cap: 0.10
```

## 5.2 只保存两个 selected checkpoint

```text
best_miou.pth.tar
best_pd.pth.tar
```

不存在第三个 `last_epoch1000` selected checkpoint。

`resume/latest_training_state.pth.tar` 是每 epoch 覆盖的续训状态，不参与
论文选模；run 成功完成后删除。

### best_miou 排序

```text
1. 更高 mIoU
2. 更高 Pd
3. 更低 Fa
4. 更高 nIoU
5. 更高 tiny-Pd
6. 更低 test loss
7. 更早 epoch
```

### best_pd 排序

```text
1. 更高 Pd
2. 更低 Fa
3. 更高 tiny-Pd
4. 更高 mIoU
5. 更高 nIoU
6. 更低 test loss
7. 更早 epoch
```

每个方法、每个 λ 独立选择自己的 checkpoint；不得跨 checkpoint 拼接指标，
不得新增 best_joint。

## 5.3 paired-scratch 初始化

12 个 run 全部 fresh，不复用历史 Original checkpoint。

当前 builder 会确定性构造 Original/Final 配对初始状态，并在 protocol 中
记录：

```text
paired_initialization=true
selected_model_state_sha256
shared_state_sha256
extension_state_sha256
derived_initialization_seeds
model_construction_preserves_caller_rng_stream=true
```

训练 shuffle 使用 seed42、dataset、epoch 的确定性派生种子；crop 和增强
使用冻结的 stateless 规则。三个 λ 完成后必须核对同一数据集的
`selected_model_state_sha256` 完全一致。

不额外建立“父模型初始 checkpoint”，避免引入新的运行身份和第三类权重。

---

# 6. 正式 12-run 矩阵与排程

| 数据集 | Original | Final 0.0025 | Final 0.005 | Final 0.01 |
|---|---:|---:|---:|---:|
| NUAA-SIRST | 1 | 1 | 1 | 1 |
| NUDT-SIRST | 1 | 1 | 1 | 1 |
| IRSTD-1K | 1 | 1 | 1 | 1 |

```text
Original runs = 3
Final runs = 9
total runs = 12
waves = 6
```

固定顺序：

| wave | GPU 2 | GPU 3 |
|---:|---|---|
| 0 | NUAA Original | NUAA Final 0.0025 |
| 1 | NUAA Final 0.005 | NUAA Final 0.01 |
| 2 | NUDT Original | NUDT Final 0.0025 |
| 3 | NUDT Final 0.005 | NUDT Final 0.01 |
| 4 | IRSTD Original | IRSTD Final 0.0025 |
| 5 | IRSTD Final 0.005 | IRSTD Final 0.01 |

同一数据集的两个 wave 完成并通过产物核对后，才进入下一个数据集。训练
过程中不交换 GPU 映射、不插入其他 λ、不跳过 Original。

---

# 7. 已落地的 TSS 强度日志

Final 每个 minibatch 已记录：

```text
batch_index
sample_count
segmentation_loss
survival_loss
requested_weight
effective_weight
raw_weighted_to_seg_ratio
effective_weighted_to_seg_ratio
cap_active
counterfactual_effective_weights:
  0.0025
  0.005
  0.01
```

每个 epoch 已直接汇总：

```text
effective_weight_mean
effective_weight_p10
effective_weight_p50
effective_weight_p90
effective_weight_std
effective_weight_max
raw_weighted_to_seg_ratio_mean
effective_weighted_to_seg_ratio_mean
cap_active_batch_fraction
cap_active_sample_fraction
```

同时保留 train segmentation、survival、weighted-survival 与 total loss。

以下量可以在训练后由逐 batch 记录确定性派生，不需要修改正在运行的
runner：

```text
effective_weight_min
raw/effective ratio p50, p90, max
candidate-pair effective-weight equality rate
candidate-pair mean absolute effective-weight difference
effective-weight distribution distance
counterfactual cap-active rate
```

这些是解释性诊断，不进入 checkpoint 选择，也不改变冻结的全局 λ
selector。

若两个候选的大部分 counterfactual effective weight 相同，只能说明 ratio
cap 让这两个请求区间难以区分；不能据此声称“λ 对性能没有影响”。

---

# 8. 冻结的全局 λ 选择规则

本节必须与：

```text
experiments/select_three_dataset_global_tss_recipe_v2.py
```

完全一致。训练完成后不得换成另一套门槛或 tie-break。

## 8.1 输入

```text
datasets = [NUAA-SIRST, NUDT-SIRST, IRSTD-1K]
roles = [best_miou, best_pd]
candidates = [0.0025, 0.005, 0.01]
threshold = 0.5
selection_split = img_idx/test
seed = 42
```

每个 dataset-role 使用五个等权量：

```text
mIoU（1e-4 量化，高者优）
nIoU（1e-4 量化，高者优）
matched_target_count（高者优）
unmatched_predicted_pixels（低者优）
matched_tiny_target_count（高者优）
```

mIoU/nIoU 量化：

```text
q(x) = floor(x / 1e-4 + 0.5)
```

不直接求原始指标和，不按数据集图像数或目标数加权。

## 8.2 严重退化门

任一 dataset-role 满足以下任一条件，候选即失去资格：

1. matched target 比 Original 少至少 2 个；
2. matched tiny target 比 Original 少至少 2 个；
3. 量化 mIoU 比 Original 下降至少 50 quanta，即至少 0.005；
4. 量化 nIoU 比 Original 下降至少 50 quanta，即至少 0.005；
5. Original unmatched pixels 为 0 而 Final 大于 0，或
   `Final_pixels * 4 > Original_pixels * 5`，同时 Final 没有至少多
   检出 2 个目标。

Fa 恶化边界使用严格大于 125%；恰好 125% 不触发该条。

## 8.3 Original 双角色严格覆盖门

在一个 role 内，若 Original 在五个量上全部不差且至少一个严格更好，则该
role 被 Original 严格覆盖。

若同一数据集的 best_miou 与 best_pd 两个 role 都被严格覆盖，候选失去
资格。

## 8.4 rank population

所有 rank 必须先在三个预注册候选的完整集合上计算：

```text
rank_population=all_three_preregistered_candidates_before_eligibility_gates
```

不得先删除失败候选再重排名。

并列值使用并列位置的平均名次，不使用 dense-rank 编号。

每个候选保留：

```text
3 datasets × 2 roles × 5 metrics = 30-dimensional rank vector
```

## 8.5 Pareto 与唯一选择顺序

通过上述门槛后，在 30 维 rank 向量上做 Pareto 过滤，rank 越低越好。

非支配候选按以下顺序选择：

```text
1. 最小 worst-dataset mean rank
2. 最小 macro-dataset mean rank
3. 最大 signed metric vote vs Original
4. 更小 requested λ
```

vote 只是 Pareto 后的第三顺位 tie-break，不是额外准入门。

若无候选通过：

```text
global_tss_recipe_established=false
decision=NO_POSITIVE_GLOBAL_TSS_RECIPE_ESTABLISHED
```

不得为了继续论文流程强选 λ。

---

# 9. 阈值与 Pd–Fa 语义

## 9.1 唯一正式阈值

```text
checkpoint selection threshold = 0.5
global λ selection threshold = 0.5
main-table threshold = 0.5
```

不得按数据集或 λ 改阈值。

## 9.2 threshold=1.0

二值化为：

```python
prediction = probability > threshold
```

所以 threshold=1.0 是描述性 sweep 中的合法空预测端点：

```text
Pd = 0
Fa = 0
predicted_object_count = 0
selected_point_is_empty = true
```

它不参与 checkpoint 或 λ 选择。

预算输出的权威字段：

```json
{
  "budget": 1e-6,
  "pd_at_fa_budget": 0.0,
  "fa_at_selected_point": 0.0,
  "selected_threshold": 1.0,
  "selected_point_is_empty": true,
  "registered_grid_nonempty_feasible": false,
  "best_nonempty_point": null
}
```

若存在任意非空可行点，非空点优先于 threshold=1.0 端点。

---

# 10. 公平性与结论边界

## 10.1 单 run

每条固定 λ 与 Original 的单 run 协议匹配：

```text
same seed
same epoch budget
same candidate epochs
same img_idx
same threshold
same metric implementation
same checkpoint roles
same optimizer/schedule/augmentation contract
```

```text
per_run_protocol_matched=true
```

## 10.2 总搜索预算

```text
Original training runs = 3
Final training runs = 9
Final / Original run budget = 3.0
total_search_budget_equal=false
```

因此允许写：

> Original 与每个固定 TSS 配方使用匹配的单 run 训练和评估协议。

不得写：

> Original 与最终获胜 Final 具有相同的总超参数搜索预算。

## 10.3 test-selected

img_idx/test 同时用于：

- 每 10 epochs 评估；
- best_miou/best_pd 选模；
- 三个 λ 的统一配方选择；
- 主结果报告。

必须披露：

```text
test_selected=true
selection_is_optimistic=true
independent_test_confirmation=false
```

本轮只能建立当前 seed42、当前 img_idx/test-selected 协议下的操作性配方；
不能建立跨随机性稳定性或独立测试确认。

---

# 11. 正式训练完成后的固定流程

## Phase A：完成 12 个 run

supervisor 自动：

1. 完成当前两条 worker；
2. 检查 summary、protocol、两个 checkpoint 及其 SHA；
3. 删除成功 run 的 rolling resume；
4. 核对 source lock、pair audit、TSS statistics；
5. 进入下一 wave；
6. 直到 12/12 完成。

不要人工重复启动 launcher。

## Phase B：评估 24 个 selected checkpoint

```text
12 runs × 2 roles = 24 evaluator outputs
```

统一使用：

```text
experiments/evaluate_three_dataset_v2.py
```

每份结果必须同时包含：

```text
mIoU
nIoU
matched/total
Pd
unmatched predicted pixels
Fa
unmatched predicted objects
false objects per image
tiny matched/total
tiny-Pd
fixed threshold 0.5
descriptive Pd–Fa sweep
```

## Phase C：组装 selector 输入

每个数据集必须包含：

```text
Original best_miou
Original best_pd
Final 0.0025 best_miou
Final 0.0025 best_pd
Final 0.005 best_miou
Final 0.005 best_pd
Final 0.01 best_miou
Final 0.01 best_pd
```

同时绑定 test img_idx 文件 SHA 和有序 ID SHA。

## Phase D：执行冻结 selector

```text
experiments/select_three_dataset_global_tss_recipe_v2.py \
  --input <assembled_input.json> \
  --output <selection_output.json> \
  --launch-plan results/three_dataset_seed42_global_tss_v2/launch/formal/launch_plan.json
```

`--launch-plan` 用于核对 selector 与 data protocol 的预注册 SHA。

## Phase E：输出全部结果

必须公开：

- 三个 λ 的全部结果；
- 被退化门拒绝的原因；
- 30 维 rank/Pareto 状态；
- 最终选择或“无正 λ 成立”；
- Final 9 runs 对 Original 3 runs 的预算差异；
- TSS effective-weight 诊断；
- threshold=1.0 空预测标记；
- 当前结论边界。

---

# 12. 训练后补充诊断

以下检查可以复用 12-run 产物，但不改变主 selector。

## 12.1 effective-weight 可辨识性

计算：

```text
pairwise exact-equality rate
pairwise mean absolute difference
effective-weight distribution distance
cap-active rate
raw/effective ratio summaries
```

若两个请求值在绝大多数 minibatch 上得到相同 effective weight，应报告：

```text
candidate_pair_effectively_redundant=true
```

这影响机制解释，不回写全局 λ 选择。

## 12.2 leave-one-dataset-out

可做：

```text
NUDT + IRSTD 描述性选择 → 检查 NUAA
NUAA + IRSTD 描述性选择 → 检查 NUDT
NUAA + NUDT 描述性选择 → 检查 IRSTD
```

它只能作为域敏感性分析：

```text
participates_in_primary_lambda_selection=false
independent_test_confirmation=false
```

不得用 LODO 结果推翻或替换冻结 selector 的主裁决。

## 12.3 梯度诊断

如结果需要解释，可在训练完成后另做固定 checkpoint/固定 minibatch 的
只读梯度诊断。该诊断不进入当前 optimizer trajectory，不作为 12-run
完成条件。

---

# 13. 结果裁决

## 情况 A：selector 建立统一正 λ

```text
decision=GLOBAL_POSITIVE_TSS_RECIPE_ESTABLISHED
global_tss_recipe_established=true
global_tss_lambda=<selected>
```

允许结论：

> 在固定 seed42、固定 threshold=0.5 和既有 img_idx/test-selected
> 协议下，一个统一正 TSS 请求权重形成了通过预注册退化门和等权
> rank/Pareto 的操作性配方。

不自动推出：

```text
TSS mechanism causally established
architecture universally superior
multi-seed stability established
independent-test generalization established
```

## 情况 B：无正 λ 通过

```text
decision=NO_POSITIVE_GLOBAL_TSS_RECIPE_ESTABLISHED
global_tss_recipe_established=false
```

不得从这里直接修改 TPD、NER 或 QFG。优先增加一个后续控制：

```text
同一冻结 Final 推理结构
TSS requested weight = 0
同一 seed42 与数据协议
```

它用于区分：

```text
正 TSS 训练目标有害
vs
Final 推理结构仍有结构性退化
```

TSS-off 不属于当前 12-run，不在当前结果出现前启动。

## 情况 C：不同数据集偏好不同 λ

仍执行冻结的全局 selector，不允许按数据集分别部署不同 λ。可附加：

```text
tss_domain_sensitivity_observed=true
dataset_specific_lambda_forbidden=true
```

---

# 14. 结果表模板

## 14.1 best_miou，threshold=0.5

| Dataset | Method | λ | Epoch | mIoU ↑ | nIoU ↑ | matched/total ↑ | Pd ↑ | Fa ↓ | tiny matched/total ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA | Original | 0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUAA | Final | 0.0025 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUAA | Final | 0.005 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUAA | Final | 0.01 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUDT | Original | 0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUDT | Final | 0.0025 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUDT | Final | 0.005 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUDT | Final | 0.01 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| IRSTD | Original | 0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| IRSTD | Final | 0.0025 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| IRSTD | Final | 0.005 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| IRSTD | Final | 0.01 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 14.2 best_pd，threshold=0.5

使用与表 14.1 相同的 12 行，不跨 role 拼接。

## 14.3 全局 λ 选择

| λ | severe passed | dual-role dominance passed | Pareto eligible | R_worst ↓ | R_macro ↓ | signed vote ↑ | selected |
|---:|---|---|---|---:|---:|---:|---|
| 0.0025 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.005 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.01 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

所有 TBD 只能由正式产物填充，不使用中途 checkpoint 数值。

---

# 15. 当前项目状态

```text
decision=FORMAL12_RUNNING

architecture_implementation_complete=true
architecture_frozen=true
innovation_mainline_changed=false
new_module_design_authorized=false
candidate_model_retained=true

formal_datasets=[
  NUAA-SIRST,
  NUDT-SIRST,
  IRSTD-1K
]
sirst3_role=historical_only

seed=42
epochs=1000
eval_every=10
threshold=0.5
checkpoint_roles=[best_miou,best_pd]

requested_tss_candidates=[0.0025,0.005,0.01]
survival_ratio_cap=0.10
ratio_cap_semantics=scalar_loss_contribution_cap

planned_original_runs=3
planned_final_runs=9
planned_total_runs=12
per_run_protocol_matched=true
total_search_budget_equal=false
final_to_original_run_budget_ratio=3.0

test_selected=true
selection_is_optimistic=true
independent_test_confirmation=false

training_recipe_bottleneck_plausible=true
global_tss_recipe_established=false
training_recipe_finalized=false
final_model_performance_established=false
paper_core_established=false
stability_claim_supported=false
```

---

# 16. 最终执行结论

> 当前不再新增或修改正式训练代码，也不重复启动实验。继续由现有 supervisor
> 在 GPU 2/3 完成 12 个 fresh formal1000 run；完成后评估 24 个
> best_miou/best_pd checkpoint，并严格使用已经 source-locked 的严重退化门、
> 30 维等权 rank/Pareto 与固定 tie-break 选择统一正 TSS 配方。effective
> weight、LODO 和梯度检查只作为训练后解释性诊断。若无正 λ 通过，再设计
> Final TSS-off 控制；在此之前不修改 TPD、NER、QFG 或创新主线。

No experimental result has been generated by this document. 所有正式结果必须
来自当前 12-run、对应 evaluator 输出和 source-bound selector 产物。
