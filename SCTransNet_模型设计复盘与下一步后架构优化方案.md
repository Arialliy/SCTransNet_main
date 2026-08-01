# SCTransNet 完整模型复盘与下一步后架构优化方案

**项目对象：** 单帧红外小目标检测  
**冻结推理架构主线：** SCTransNet + TPD8-MPRS-DCH + 五节点 NER4 Tail-Aware + QFG2-CROA  
**当前候选训练期约束：** TSS，权重 0.005；推理时物理移除，最终权重尚待 P2/P3 冻结  
**主实验协议：** seed 42，1000 epochs，每 10 epochs 评估一次  
**文档日期：** 2026-08-01

---

## 1. 最终结论

### 1.1 模型设计是否成功

结论不是简单的“完全成功”或“失败”，而是：

> **模型设计已经取得有条件的实证成功，结构设计阶段可以结束；但尚未达到所有数据集、所有 checkpoint、所有固定点指标全面支配 Original 的程度。**

建议采用以下正式状态：

```text
decision=REVISE_PROTOCOL_BEFORE_POST_ARCHITECTURE_RUN

model_structure_complete=true
model_code_complete=true
post_architecture_experiment_code_complete=false
engineering_design_success=true
inference_architecture_candidate_frozen=true
architecture_redesign_required=false
innovation_mainline_changed=false

seed42_benchmark_competitiveness_supported=true
dataset_specific_test_sweep_pd_at_fa_le_1e_minus_5_advantage_observed=true
deployable_calibrated_advantage_supported=false
fixed_threshold_result=MIXED_TRADEOFF
universal_dominance=false

training_recipe_finalized=false
paper_core_established=false
stability_claim_supported=false
multiseed_current_scope=false
structural_ablation_current_scope=false
tss_zero_weight_control_current_scope=false
```

这里的“设计成功”包括三层含义：

| 层级 | 结论 | 依据 |
|---|---|---|
| 工程与结构设计 | 成功 | TPD、NER、QFG 已完整集成；TSS 仅训练期使用；无 TSS 部署图已实现 |
| 应用目标有效性 | 成功但有权衡 | seed42 的 `Fa≤1e-5` 数据集专属测试扫描中 Pd 均提高，其他预算与固定点呈 mixed trade-off |
| 全指标统一优势 | 未建立 | 固定阈值 0.5 下，NUDT、IRSTD 和 SIRST3 best-Pd 等结果仍有退化项 |
| 论文级稳定性 | 未建立 | 当前正式结果只有 seed 42，且 best checkpoint 来自测试划分上的候选 epoch 比较 |

因此，**不应继续增加第五个模块，也不应推倒 TPD、NER 或 QFG 重新设计。**

---

## 2. 为什么可以判定模型设计已经形成有效成果

### 2.1 低 Fa 区间的结果与研究目标高度一致

使用各自 best-mIoU checkpoint 时，在 `Fa ≤ 1e-5` 的预算下，Final 在以下四个**训练数据集与测试数据集同源的 dataset-specific 设置**中均获得更高 Pd：

```text
SIRST3 → SIRST3
NUAA-SIRST → NUAA-SIRST
NUDT-SIRST → NUDT-SIRST
IRSTD-1K → IRSTD-1K
```

这里的箭头表示“在哪个数据集训练、在哪个数据集测试”，不表示训练图像与测试图像相同；两者仍使用各自 `img_idx` 中互斥的 train/test ID。这里也不是“SIRST3 同一 checkpoint 在三个来源子集上的结果”。SIRST3 三来源复用实验必须单独报告，不能与上述四个数据集专属训练设置混写。

其中 IRSTD 的提升最明显：

```text
Fa ≤ 1e-6：  83/297  → 158/297
Fa ≤ 5e-6： 153/297  → 262/297
```

这些结果直接支持：

> **在 seed42、测试集阈值扫描和 `Fa≤1e-5` 这一预注册预算下，Final 的可达 Pd 高于 Original。**

“真实目标与复杂背景杂波的组件分数间隔得到改善”目前仍是待 P1 验证的解释性假设，不能由 Pd–Fa 表格直接证明。现有 sweep 还使用每个模型自身的自适应阈值集合；正式确认前必须增加 Original/Final 公共阈值集合复核。

这一观测与三个推理模块的设计逻辑一致，但尚不能把总模型差异分别归因给三个模块：

- TPD8-MPRS-DCH：在浅层 tokenization 中保留局部相位与显著性；
- 五节点 NER4：把浅层尾部目标证据持续传递给解码阶段；
- QFG2-CROA：只调制 Transformer Query，使频率先验影响注意力查询而不改变 K/V、CFN 和 decoder 主路径。

### 2.2 三个 dataset-specific train=test 设置的宏观结果仍是正向权衡

只计算 NUAA、NUDT、IRSTD 三个数据集专属训练设置，不把包含三者的 SIRST3 聚合设置重复计入。下述 macro 均为三个数据集指标的算术平均：

#### 各自 best-mIoU checkpoint

```text
macro mIoU：+0.001550
macro nIoU：+0.009503
macro Fa：降低约 15.9%
macro Pd：下降约 0.67 个百分点
```

#### 各自 best-Pd checkpoint

```text
macro mIoU：+0.023974
macro Fa：降低约 43.8%
macro Pd：下降约 0.32 个百分点
```

这说明 Final 不是单纯牺牲大量召回来换取低虚警，而是以较小的 macro Pd 损失换取区域质量和虚警改善。它构成有价值的**性能权衡和非支配候选工作区间**，但不是严格的 Pareto improvement，因为 macro Pd 同时下降。

必须同时报告 tiny-Pd 的不利变化：三个数据集上，best-mIoU 的 Δmacro tiny-Pd 为约 `-0.019048`，best-Pd 为约 `-0.017460`。因此不能把当前固定阈值结果概括为“极小目标保持全面提高”。

### 2.3 推理结构边界是干净的

最终代码将 TSS 保留为训练期辅助约束；正式部署和离线复评构建器会检查并拒绝任何 `target_survival.*` state 残留，最终部署图只保留 TPD、NER 和 QFG。训练中的在线评估只是跳过 TSS 计算，真正的物理移除发生在部署/正式离线复评图。TSS loss 与原六路分割 BCE 分开计算，并使用数据集专属正类权重：

\[
\mathcal{L}
=
\sum_{j=1}^{6} \operatorname{BCE}(P_j,Y)
+
0.005\sum_{i\in\{\mathrm{emb1},\mathrm{emb2}\}}
\operatorname{BCEWithLogits}(Z_i,Y_{16};\operatorname{pos\_weight}=w_d)
\]

其中：

\[
Y_{16}=\operatorname{MaxPool}_{16}(Y),\qquad
w_d=\frac{N_{\mathrm{negative\ cells}}}{N_{\mathrm{positive\ cells}}}.
\]

因此，当前结果只能视为冻结完整训练方案相对 Original 的整体差异，不能再拆分归因给某个模块；同时，部署图不会引入额外 TSS 分支。

---

## 3. 当前还不能宣称什么

### 3.1 不能宣称四数据集全指标全面超过 Original

固定阈值 0.5 下仍存在明确反例：

- SIRST3 best-Pd：Original 的 Pd、mIoU 和 Fa 更强；
- NUDT best-mIoU：Final 多检出 1 个目标且 nIoU 更高，但 mIoU 略低、Fa 更高；
- IRSTD best-mIoU：Final 的 nIoU 和错误目标数更好，但 Pd 和 mIoU 较低；
- epoch 1000：Final 在 IRSTD 上整体较弱。

因此，论文应使用：

```text
mixed trade-off
competitive low-false-alarm performance
higher attainable Pd at Fa≤1e-5 in the current official-test sweeps
```

而不是：

```text
uniform dominance
all metrics outperform SCTransNet
state of the art on every dataset and operating point
```

### 3.2 当前 best checkpoint 属于 benchmark-compatible test selection

Original 的公开训练脚本默认使用：

```text
seed = 42
epochs = 1000
训练过程中在 test set 上评估
依据 test mIoU 保存 best checkpoint
```

你当前对 Original 和 Final 都采用相同的 100 个候选 epoch 和相同排序规则，因此它是**公平的同协议复现**；但从严格统计意义看，仍属于 test-based checkpoint selection。

因此当前结果可作为：

> 与 SCTransNet 的 test-based selection 逻辑兼容、且 Original/Final 采用完全相同自定义候选 epoch 协议的公平配对比较。

它并非与公开 `train.py` 默认选模日程严格一致：公开脚本默认 epoch 500 后逐 epoch 测试，而当前正式实验采用 epoch 10–1000、每 10 epochs 一次、共 100 个候选点。

但不能单独作为：

> 无测试集参与的独立泛化证明。

### 3.3 当前不能支持随机种子稳定性结论

seed 42 是合理的主 seed，因为 Original SCTransNet 和 BasicIRSTD 的默认实现均使用 42。但一个 seed 只能支持：

```text
fixed_seed42_result_supported=true
```

不能支持：

```text
random_seed_stability_supported=true
```

---

## 4. 当前处于什么优化阶段

当前阶段不是“继续设计模型模块”，也不是“全面调参”。准确名称应为：

> **后架构优化阶段（Post-Architecture Optimization）**

它包含四个子问题：

1. **概率与固定阈值校准**：为什么部分预注册低 Fa 预算、尤其 `Fa≤1e-5` 的可达 Pd 更高，但 threshold 0.5 的部分数据集结果仍较弱；
2. **checkpoint 选择策略优化**：best-mIoU、best-Pd、best-low-Fa 应分别承担什么角色；
3. **训练配方最小化调优**：只有 model_val 证明退化不只是概率尺度问题时，才调 TSS 权重这一训练期标量；
4. **当前 seed42 协议闭环**：公共阈值复核、固定终点、无测试参与的新开发协议和逐图统计。多 seed 与消融按用户当前约束暂不执行，只保留为未来可选工作。

### 4.1 当前不是以下阶段

```text
不是：增加新分支
不是：重新设计 TPD
不是：重新设计 NER
不是：修改 QFG 公式
不是：同时搜索学习率、损失、数据增强和模块宽度
不是：为了某个测试集事后挑阈值
```

### 4.2 参数是否需要调整

结论是：

> **先不调模型结构参数；先完成无重训练诊断，再冻结 clean split 并成对重训。只有 model_val 证明问题不是分数整体平移，而是可达 Pd–Fa、目标—杂波间隔和 tiny-Pd 同时退化时，才允许做一维 TSS 权重搜索。**

优先级如下：

```text
固定 checkpoint 探索性诊断
→ 冻结 seed42 clean split
→ 从头成对训练并在 model_val 执行 checkpoint policy
→ 必要时仅调 TSS weight
→ calibration 冻结部署工作点
→ official test 与逐图统计
```

---

## 5. 对当前失败模式的核心判断

### 5.1 部分低 Fa 预算更强、固定 0.5 混合，首先需要检查校准

若一个模型在部分预注册低 Fa budget、尤其 `Fa≤1e-5` 上拥有更高可达 Pd，但在固定阈值 0.5 下未全面改善，可能说明：

```text
目标与杂波的相对排序改善
但概率绝对尺度或数据集间分布不一致
```

当前首先需要检验 Final 是否把真实目标排得更靠前，以及是否存在以下情况：

- 某些数据集整体输出偏低，导致 0.5 下漏检；
- 某些数据集目标区域内部概率不均匀，导致 mIoU 不稳定；
- 不同来源域的最佳 decision boundary 不一致。

这类问题首先应通过公共阈值复核、score distribution 和 component-level analysis 验证，而不是立即修改网络。若 target score 与 false score 同时下降但排序间隔不变，可能只是整体 logit 平移，不能直接归因于 TSS。

### 5.2 单调校准只能移动固定工作点，不会制造虚假的 Pd–Fa 曲线优势

设模型输出概率为 \(p\)，进行单调 logit 校准：

\[
z=\operatorname{logit}(p)
\]

\[
p_{cal}=\sigma(a z+b),\qquad a>0
\]

在不产生新的分数并列、并对阈值作一一对应映射时，该变换保持预测分数排序不变，因此：

- 完整连续阈值下的 Pd–Fa 可达集合不变；
- AUC 和排序能力不变；
- 只会改变 threshold 0.5 对应的原始概率边界。

这可以把默认 threshold 0.5 映射到一个更合适的原始工作点，但不会创造新的 Pd–Fa 曲线优势，也不等于模型表示能力提高。固定数值网格可能因采样和 `clamp` 引入离散差异，验证时必须使用逆映射阈值或公共阈值集合。

但 \(a,b\) 或阈值必须在训练集派生的 calibration split 上确定，不能在官方测试集上拟合。

---

## 6. 下一步完整执行方案

### 阶段 P0：核验并冻结现有模型与结果

当前 `results/four_dataset_seed42_v1/postprocess/` 已经包含 training gate、六阶段状态、产物清单和文件摘要；`selected_checkpoints/checkpoint_manifest.json` 已记录 16 个正式权重。P0 的任务是**复用并只读核验现有封存**，不再重复复制 checkpoint 或重复计算结果。

冻结不能只覆盖五个顶层文件，必须递归锁定真实运行依赖，至少包括：

```text
model/SCTransNet.py
model/Config.py
model/tpd_clean_v8_mprs_dch.py
model/tpd_ner_v8_mprs_dch.py
model/tpd_ner_v8_mprs_dch_v2.py
model/tpd_ner_v8_mprs_dch_v3.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py
model/tpd_sctransnet.py
model/tpd_relay.py
model/tpd_frequency_gate.py
model/tpd_frequency_gate_v2_croa.py
model/tpd_query_frequency_bridge.py
model/tpd_survival.py
model/tpd_forward_contract.py
experiments/tpd_training_loss.py
experiments/four_dataset_data_protocol_v1.py
experiments/compute_four_dataset_tss_statistics_seed42_v1.py
experiments/four_dataset_models_seed42_v1.py
experiments/train_four_dataset_original_final_seed42_exact_v1.py
experiments/train_tpd_pilot.py
experiments/evaluate_four_dataset_seed42_v1.py
experiments/four_dataset_evaluation_protocol_v1.py
```

上表只是最低显式集合；递归 manifest 还必须自动纳入正式指标实现及上述文件的所有直接本地 import，不能把这份人工清单当作完整依赖闭包。

现有可封存内容：

- Original 与 Final 的 16 个 `best_miou` / `best_pd` checkpoint；
- 100 个候选 epoch 的完整指标；
- epoch1000 固定终点指标和日志；
- 16 个数据集专属和 12 个 SIRST3 来源子集 Pd–Fa sweep；
- 数据 ID、归一化、评估器、运行依赖和文件摘要；
- seed、确定性配置、GPU 绑定和现有协议元数据。

当前**没有**单独保存 epoch1000 checkpoint，也没有永久保留成功完成时已删除的滚动续训状态、optimizer、scheduler 和末端 RNG state。若未来 clean protocol 需要逐图 epoch1000 复评或 bootstrap，必须在新训练开始前修改产物策略，不能从现有聚合指标反推。

已验证的参数事实应纳入封存清单：

| 图 | 参数量 | state keys |
|---|---:|---:|
| Original SCTransNet | 11,325,939 | 510 |
| Final 训练图（含 TSS） | 10,870,228 | 568 |
| Final 推理图（无 TSS） | 10,870,130 | 564 |
| 物理移除的 TSS | 98 | 4 |

建议状态：

```text
architecture_frozen=true
existing_seed42_results_immutable=true
epoch1000_checkpoint_available=false
terminal_rng_state_available=false
new_module_design_authorized=false
```

---

### 阶段 P1：无重训练诊断

直接使用已有 checkpoint，不启动训练。P1 只用于解释现有 seed42 结果和决定是否值得建立新的 clean protocol；不得用现有 official test 诊断结果选择正式 λ、checkpoint、阈值或 calibrator。

#### P1.0 公共阈值集合复核

当前 sweep 包含公共概率网格、logit tail 和每个模型自己的经验分位数，因此 Original 与 Final 的最终阈值集合不完全相同。首先对每个配对 checkpoint 构造：

```text
paired_thresholds
= Original thresholds
∪ Final thresholds
∪ {0.0, nextafter(1,0), 1.0}
```

两种模型都在完全相同的 `paired_thresholds` 上重新统计，核对：

- `Pd@Fa≤1e-5` 四个 dataset-specific 设置是否仍全部正向；
- 结论对公共 logit 网格密度是否稳定；
- 表中严格预算的 0 是否来自没有非空可行工作点；
- Pareto 工作区间是否依赖模型专属阈值采样。

这一复核不改变历史 Table 7；它生成新的 paired-grid 诊断表，作为是否继续校准的第一道 gate。

#### P1.1 一次推理、统一数据提取

禁止四个分析脚本分别重复模型推理。对每个选定 checkpoint 只运行一次前向，统一输出带版本和文件摘要的中间记录；概率直方图、组件记录、错误集合和阈值曲线均由该记录离线派生。

首轮只处理四个数据集的 Original/Final `best_miou` 共 8 个 checkpoint。只有当 P1 指向 SIRST3 best-Pd 特有失败时，才追加该配对 checkpoint，避免无效计算。

当前正式推理接口直接返回 sigmoid probability，并未公开最终分割 raw logits。共享提取器需要通过只读 forward hook 或等价的诊断 forward 捕获 `outc` 的 pre-sigmoid 输出；不得修改冻结模型公式、state dict 或正式推理返回契约。测试必须验证 `sigmoid(captured_logits)` 与原接口 probability 逐元素一致，随后才允许使用 logits 做尾部分布和可选 calibrator 分析。

#### P1.2 目标与虚警组件分数

对每个 GT target 和 prediction component 输出：

```text
image_id
gt_component_id
target_area
target_peak_probability
target_mean_probability
matched_component_peak
matched_component_mean
highest_unmatched_component_peak
target_false_margin
centroid_distance
component_iou
fragment_count
analysis_threshold
threshold_provenance
```

定义目标—虚警间隔：

\[
M_i=s_i^{target}-\max_j s_{ij}^{false}
\]

组件是阈值相关对象，因此必须分别在 `threshold=0.5`、配对公共网格选出的 `Fa≤1e-5` 工作点以及预注册的低 support threshold 上定义。若图像没有 false component，必须记录为 `null/no_false_component`，不能用任意常数代替。除图像内 margin 外，还应报告全数据集 target-component 与 false-component 分数分布，避免把局部 margin 误当成全局 Fa 排序。

分别比较 Original 和 Final：

- Final 是否提高多数图像的 \(M_i\)；
- IRSTD 漏检目标是否只是低于 0.5，但仍高于 false component；
- NUDT 高 Fa 是否来自少量高分背景组件；
- SIRST3 best-Pd 退化是否来自目标峰值偏低或组件碎裂。

#### P1.3 概率分布与校准

每个数据集绘制：

- GT target pixel probability histogram；
- hard-negative pixel probability histogram；
- matched component maximum-score histogram；
- unmatched component maximum-score histogram；
- reliability diagram；
- Brier score；
- Expected Calibration Error；
- threshold–Pd、threshold–Fa、threshold–mIoU 联合曲线。

普通 pixel ECE/Brier 会被大量背景像素主导，不能单独作为是否调整 TSS 的依据。必须同时报告：

- target-conditioned pixel ECE/Brier；
- hard-negative-conditioned pixel ECE/Brier；
- component-level reliability；
- 按目标面积与数据来源分层的结果；
- 原始 logits 的分布，避免 sigmoid 饱和和 `clamp` 掩盖尾部差异。

#### P1.4 目标结构错误

统计：

```text
miss
split
merge
fragmentation
attached halo
centroid shift
small-target erosion
```

诊断裁决：

| 观测结果 | 解释 | 下一步 |
|---|---|---|
| 公共阈值复核后 Final 排序仍更强，但 0.5 点较差 | 可能是工作点/概率尺度问题 | 允许设计新的 P2 clean protocol |
| Final 在公共阈值集合下都漏同一目标 | 表示/训练问题 | 先在 model_val 复现，再决定是否进入 P3 |
| Final 出现更多碎裂 | 连通性问题 | 先分析 TSS/NER evidence，不改结构 |
| Final 的 false score 与 target score 同向平移、margin 不变 | 概率尺度变化 | 不调 TSS，优先验证阈值映射 |
| model_val 上 target–false margin、tiny-Pd 与可达 Pd–Fa 同时退化 | 目标存活/训练问题 | 才允许进入 P3 |

---

### 阶段 P2：建立 seed42 clean development protocol

P2 不是把现有 full-train checkpoint 拿来后切 10% 数据做“校准”。现有权重训练时已经使用了全部 official train，不能追溯性地把其中一部分称为 held-out calibration。正式 clean protocol 必须先冻结数据角色，再从头成对训练 Original 与 Final。

#### P2.1 三类互斥数据角色

每个 official train 按图像 ID 固定划分为：

```text
official train
├── train_core   80%：只用于梯度更新和 normalization 统计
├── model_val    10%：只用于 checkpoint、lambda_s 和是否进入 P3 的选择
└── calibration  10%：只用于冻结最终部署阈值或拟合概率校准器

official test：在训练配方、checkpoint、阈值和 calibrator 全部冻结后执行
```

按现有 `img_idx` 数量，确定性配额为：

| 来源 | official train | train_core | model_val | calibration |
|---|---:|---:|---:|---:|
| NUAA-SIRST | 213 | 171 | 21 | 21 |
| NUDT-SIRST | 663 | 531 | 66 | 66 |
| IRSTD-1K | 800 | 640 | 80 | 80 |
| SIRST3 | 1676 | 1342 | 167 | 167 |

每个来源先令 `model_val=floor(0.1N)`、`calibration=floor(0.1N)`，其余全部归入 train_core。各来源内部再按目标数量、目标面积档位和无目标图像分层，并用确定性的最大余数分配使各层配额精确加总到上表；余数并列时按冻结摘要顺序裁决。SIRST3 的三类集合由三个来源的对应角色合并，因此其数量严格等于三来源之和。

固定要求如下：

- `training_seed=42`，不开展多训练 seed；
- 数据划分不再引入第二个随机 seed，而是使用冻结的 `split_policy=source_area_stratified_digest_v1`：每个分层内按固定协议标签与规范化 sample ID 生成的稳定摘要排序后确定配额；规则和输出 ID 均写入 manifest，一经生成不得因结果重排；
- Original 与 Final 复用完全相同的 ID、预处理、增强、batch、优化器和训练预算；
- train_core、model_val、calibration、official test 两两无交集；
- clean protocol 的 normalization 仅由 train_core 统计；历史 Table A 仍保留原实验的 frozen legacy normalization，两个协议不得混写；
- 对 NUAA 的 `Misc_111` 继续使用已冻结的尺寸修正规则，并在 split manifest 中记录修正后的文件摘要；
- 不得从现有 checkpoint 续训，因为它已见过 model_val 与 calibration 对应样本。

clean protocol 的 TSS 正类权重也必须重新计算：只使用 train_core 和冻结的 crop/augmentation plan 统计 `w_d=N_negative_cells/N_positive_cells`，不得复用现有 full-train 权重。所有正 `lambda_s` 候选共用同一份带摘要的 train_core TSS 统计。

当前 evaluator 的 `Fa=unmatched_predicted_component_pixels/valid_pixel_count`，所以 `1/valid_pixel_count` 只是理论最小正步长；实际分辨率还受未匹配组件面积和候选阈值集合限制。每个 model_val/calibration split 必须报告图像数、目标数、tiny 目标数、有效像素数、理论最小 Fa 步长、`floor(1e-5×valid_pixel_count)` 可容纳的虚警像素数，以及预算内非空互异可达 Fa 点数。当前 clean protocol 的主预算固定为 `Fa≤1e-5`，更严格预算只有在该 split 可分辨时才作为确认项，否则标记为 descriptive。

#### P2.2 从头训练与 checkpoint 选择

clean 训练分两批执行，避免先用未冻结的 TSS 配方训练所有数据集：

```text
第一批：SIRST3 Original + Final(lambda_s=0.005)
        → model_val 选模并执行 P3 gate
        → 如触发 P3，只在 SIRST3 搜索正 lambda_s
        → 冻结一个全局 TSS weight

第二批：按已冻结的同一 TSS weight，训练 NUAA、NUDT、IRSTD 的 dataset-specific Final
        → 同时训练或复用协议完全相同的各自 Original clean run
```

若 `lambda_s` 从 `0.005` 改为其他正值，任何已经用 `0.005` 提前训练的 Final clean run 都不再代表最终统一配方，必须在接触 calibration/official test 前按新权重从头重训并重新由 model_val 选模；协议未变化的 Original clean run 可以复用。

所有 Original 和 Final 都从随机初始化开始，以 seed42 训练 1000 epochs，并在 epoch 10、20、…、1000 生成 model_val 候选指标。两者分别选择自己的 checkpoint，但候选 epoch、配对公共阈值集合、评价器和排序规则完全相同。

对每个候选 checkpoint，先在配对公共阈值集合中按该 checkpoint 的排序规则确定**一个**低 Fa 工作点；tiny-Pd、nIoU、mIoU 和 achieved Fa 都必须取自这个同一阈值，禁止跨阈值拼接各指标最优值。

单数据集 checkpoint 的预注册排序为：

```text
1. 最大 Pd @ Fa≤1e-5
2. 最大 tiny-Pd
3. 最大 nIoU
4. 最大 mIoU
5. 更低的 achieved Fa
6. 更高的 threshold
7. 更早的 epoch
```

SIRST3 对每个 checkpoint 只能使用一个覆盖全部三来源图像的共享阈值。预算可行性按合并集合计算：

\[
Fa_{aggregate}
=
\frac{\sum_s N^{unmatched\ predicted\ pixels}_s}
{\sum_s N^{valid\ pixels}_s}
\le 10^{-5}.
\]

三来源 macro、worst-source 指标和各来源 achieved Fa 全部在这个同一共享阈值上计算，不允许三个来源各自选阈值。SIRST3 混合训练 checkpoint 的预注册排序为：

```text
1. 在 shared threshold 且 aggregate Fa≤1e-5 时最大三来源 macro Pd
2. 在同一点最大三来源中最小的 source Pd
3. 最大 macro tiny-Pd
4. 最大 macro nIoU
5. 最大 macro mIoU
6. 更低的 aggregate achieved Fa
7. 更高的 shared threshold
8. 更早的 epoch
```

这使 Pd、Fa、tiny-Pd、nIoU 和 mIoU 同时进入选模，而不是只按 mIoU 选择。`best_miou` 与 `best_pd` 仍作为 benchmark-compatible 辅助权重保存，clean protocol 的唯一主 checkpoint 由上述低 Fa 多指标规则在 model_val 上选出；除这三个可能重合的角色外，不永久保存其余候选 epoch 权重。

#### P2.3 P3 的触发门

在接触 calibration 和 official test 之前，只对第一批 SIRST3 model_val 结果计算 Final−Original 配对差值。Pd、tiny-Pd 使用 P2.2 选出的单一共享低 Fa 工作点；`target_false_margin` 使用两个模型各自该工作点，并只在两者 margin 均非 null 的同一图像上计算中位数差，同时报告有效图像数。

P3 的唯一触发式为：

```text
enter_P3 =
    (delta_macro_Pd_at_aggregate_Fa_le_1e_minus_5 < 0)
and (delta_macro_tiny_Pd < 0)
and (delta_median_target_false_margin < 0)
and (number_of_sources_with_negative_delta_Pd >= 2)
```

四项全部成立才进入 P3；任一项持平或正向均维持 `lambda_s=0.005`。单纯的 probability/logit 整体平移、threshold 0.5 退化或 ECE 变化不能触发 P3。gate 的输入 JSON、公式版本和裁决必须在 calibration 前落盘，禁止人工覆盖。

#### P2.4 冻结部署工作点

P2.4 只在 P3 被跳过或 P3 已完成后执行。先冻结一个 checkpoint，再在 calibration 上冻结阈值；calibration 不得反过来改 epoch 或 `lambda_s`。

主部署阈值使用公共候选阈值集合，按以下规则唯一确定：

```text
1. 在 Fa≤1e-5 的候选中最大化 Pd
2. 最大化 tiny-Pd
3. 最大化 mIoU
4. 最大化 nIoU
5. 最小化 achieved Fa
6. 选择更高阈值
```

上述 2–5 项均取自第 1 项选中的同一候选阈值，禁止跨阈值拼指标。Original 与 Final 各自按同一规则得到部署阈值；SIRST3 使用与 P2.2 相同的 aggregate Fa 定义和一个全局阈值，不允许为 NUAA、NUDT、IRSTD 来源分别设阈值。原始 threshold 0.5 结果仍完整报告。

直接阈值选择是主方案。若确实需要输出可解释概率，可额外拟合正斜率的 logit-affine calibrator：

```python
calibrated = torch.sigmoid(torch.exp(log_scale) * raw_logits + bias)
```

calibrator 只能在 calibration 上以预注册的 NLL 目标拟合，并报告整体、目标像素、困难负样本及连通组件条件下的 NLL/Brier/ECE。优先使用模型原始 logits，避免先 sigmoid、clamp、再 logit 带来的尾部分数失真。该单调映射只能把某个原始阈值映射到 calibrated 0.5，不能改善连续 Pd–Fa 曲线，因此不能把它写成新的检测能力。

#### P2.5 official test 的使用边界

训练配方、checkpoint、公共阈值规则、最终阈值和可选 calibrator 全部写入只读 manifest 后，才执行一次预注册的 official-test evaluation suite。该 suite 同时输出 raw-0.5、frozen-threshold、Pd@Fa budgets 和逐图记录，运行完成后不得依据 test 返回 P2/P3 重新选择。

需要如实保留历史边界：当前架构曾参考过这些 official test 上的旧实验结果，因此新流程只能称为**本轮决策冻结后的 clean evaluation**，不能声称这些数据集在整个模型研发历史中从未被访问。若 test 结果不理想，应报告为确认失败；下一轮优化必须另立版本和协议，不能在同一协议内回看重选。

P2 成功的含义是协议闭环成功，而不是预先保证 Final 全指标胜出：

```text
clean_protocol_completed=true
checkpoint_selected_on_model_val=true
threshold_selected_on_calibration=true
official_test_used_for_current_locked_protocol_reporting_only=true
```

---

### 阶段 P3：仅在 model_val 触发时做最小训练配方搜索

P3 不改 SCTransNet、TPD、五节点 NER 或 QFG 的推理结构，只允许搜索与当前失败模式直接相关的训练期标量：TSS loss weight \(\lambda_s\)。

#### P3.1 为什么只允许调整 TSS weight

- TPD、五节点 NER4 Tail-Aware 与 QFG2-CROA 是已冻结的推理创新主线；
- TSS 只在训练期监督 `emb1`、`emb2` 两个 stride-16 endpoint，部署导出时物理移除；
- 调整其权重不会改变 Final 的推理参数量、state key 或前向接口；
- 当前证据不足以授权同时修改学习率、数据增强和模块公式，否则无法判断收益来源。

#### P3.2 唯一首轮网格

```text
lambda_s ∈ {0.0025, 0.005, 0.0100}
```

- `0.0025`：较低辅助监督权重；
- `0.005`：当前训练配方；
- `0.0100`：较高辅助监督权重。

低权重或高权重对 Pd/Fa 的方向必须由实验决定，不能预先写成“减弱保守化”或“增强召回”。三个候选必须使用同一 clean split、train_core TSS 统计和训练协议；第一批 clean run 的 `0.005` 候选可以复用，当前 full-train 的 `0.005` checkpoint 不能混入比较。

`lambda_s=0` 属于 TSS zero-weight 对照，本轮按“先不做消融”的范围明确延后，不进入训练队列，也不参与最终配方选择。

#### P3.3 选择范围与规则

首轮仅在 SIRST3 混合训练协议上执行，固定 seed42、1000 epochs、相同 train_core/model_val。model_val 同时选择 `lambda_s` 和各候选内部的 checkpoint；calibration 只在最终配方冻结后选择阈值。

每个 `lambda_s`/checkpoint 仍按 P2.2 使用一个覆盖三来源的 shared threshold，以 aggregate Fa≤1e-5 判定可行；全部 tie-break 指标在同一点计算。SIRST3 的唯一排序为：

```text
1. 在 shared threshold 且 aggregate Fa≤1e-5 时最大三来源 macro Pd
2. 在同一点最大三来源中最小的 source Pd
3. 最大 macro tiny-Pd
4. 最大 macro nIoU
5. 最大 macro mIoU
6. 更低的 aggregate achieved Fa
7. 更小的正 lambda_s
8. 更早的 epoch
```

不允许按来源选择不同 `lambda_s`，也不允许用 calibration 或 official test 选择权重。若 `0.005` 胜出，冻结当前配方并停止搜索；若其他正权重胜出，仅更新训练配方版本，推理架构版本保持不变。

全局 `lambda_s` 冻结后，按 P2.2 第二批顺序完成三个 dataset-specific Final clean run；不允许按来源或数据集选择不同权重。

#### P3.4 明确冻结的项目

| 项目 | 当前决策 | 原因 |
|---|---|---|
| TPD 公式和尺度 | 冻结 | 属于目标保真主线 |
| 五节点 NER 与 tail thresholds | 冻结 | 属于多尺度节点增强主线 |
| QFG alpha/init | 冻结 | 属于频域引导主线 |
| base learning rate、optimizer、scheduler | 冻结 | 保持 Original/Final 成对协议 |
| 数据增强与 positive crop | 冻结 | 避免同时改变训练分布 |
| segmentation loss | 冻结 | 避免与 TSS weight 归因混杂 |
| threshold | 由 P2.4 冻结 | 它是部署工作点，不是模型结构参数 |

---

### 阶段 P4：固定 seed42 的确认范围与边界

#### P4.1 当前实验范围

当前只开展固定 seed42：

```text
training_seed=42
multiseed_current_scope=false
structural_ablation_current_scope=false
tss_zero_weight_control_current_scope=false
```

因此，本阶段完整执行后可以确认 seed42 下的架构竞争力和 clean protocol 结果，但仍不能确认跨随机性稳定性。`stability_claim_supported=false` 在本轮结束后仍必须保持，不能用 checkpoint 数量、bootstrap 重采样或四数据集数量替代训练 seed 重复。

#### P4.2 固定终点记录

新 clean run 在 epoch1000 必须在线生成逐图预测记录、聚合指标和末端 RNG/环境记录；这些产物只用于固定预算结果，不参与 checkpoint 选择。为控制空间，不额外永久保存 epoch1000 checkpoint，除非 epoch1000 本身被选为 `best_miou`、`best_pd` 或 clean 主 checkpoint；成功结束后删除滚动 optimizer/scheduler 状态。现有 full-train 结果只有 epoch1000 聚合指标、没有逐图记录和 epoch1000 checkpoint，因此不能补做其逐图 bootstrap。

#### P4.3 延后项目

以下项目不进入当前执行队列：

- 多 seed 成对训练；
- TPD/NER/QFG/TSS 结构消融；
- `lambda_s=0` 的 TSS zero-weight 对照；
- 多数据集重复超参数搜索。

将来若另行授权多 seed，应单独预注册 seed 列表、固定终点或 model_val 选模协议，并完整报告所有重复；不得把它写成本轮 seed42 的隐含组成部分。

---

### 阶段 P5：统计、效率与结果组织

#### P5.1 四类结果表各自承担的结论

##### 表 A：现有 benchmark-compatible best checkpoint

- seed42、1000 epochs、每 10 epochs 测试一次；
- Original 与 Final 各自的 `best_miou` / `best_pd`；
- 明确标记为 test-based candidate selection；
- 用于与当前仓库历史结果衔接，不作为 clean deployment 证据。

##### 表 B：fixed epoch1000

- 不选 checkpoint；
- raw threshold 0.5；
- 现有实验只报告已经保存的聚合指标；
- 新 clean run 同时保存 checkpoint 和逐图记录。

##### 表 C：现有 test-sweep Pd@Fa budgets

```text
5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4
```

该表必须标注为 official-test sweep / oracle-like operating envelope。当前 sweep 含模型专属经验分位数，主文结论优先引用 P1.0 的配对公共阈值复核。它能说明潜在排序能力，不能替代一个从 calibration 冻结、可直接部署的阈值。

##### 表 D：seed42 clean protocol 主表

- model_val 选择 checkpoint；
- calibration 冻结部署阈值和可选 calibrator；
- official test 只用于最后一次预注册报告；
- 同时报告 raw-0.5，以及 frozen threshold 下的 `(Pd, achieved Fa)`、tiny-Pd、mIoU、nIoU、F1 和 false-object count；
- 只有 official test 的 frozen-threshold achieved Fa 同样满足 `≤1e-5` 时，才额外记录 `deployable_budget_pass=true`；否则保留实际 Fa，不能把该 Pd 标记成 test `Pd@Fa≤1e-5`；
- SIRST3 同时报告整体、三来源分项和 source macro。

#### P5.2 配对图像 bootstrap

仅对已经冻结的 checkpoint 和阈值做 paired image bootstrap：

```text
resamples=10000
confidence_interval=95%
threshold_reselection_inside_bootstrap=false
```

SIRST3 按来源分层并以图像为 cluster 成对重采样，Original/Final 每次使用相同图像索引。报告 frozen threshold 下的 ΔmIoU、ΔnIoU、ΔF1、Δfalse-object count、ΔPd 和 ΔFa 及其区间；每次重采样同时报告实际 achieved Fa，但不重选阈值，也不把 ΔPd 写成 `ΔPd@Fa≤1e-5`。bootstrap 只能刻画当前固定模型对测试图像抽样的敏感性，不能支持训练随机性稳定性。

#### P5.3 效率证据

在同一 GPU、batch size、输入尺寸、warm-up 和重复次数下配对报告：

- training / inference 参数量与 state key 数；
- FLOPs 或 MACs；
- 单图 latency 与吞吐；
- peak inference VRAM；
- TSS 导出前后的参数和 key 差值。

效率实验必须验证当前已知边界：Original 为 11,325,939 参数、510 个 state keys；Final 训练态为 10,870,228 参数、568 keys，推理态为 10,870,130 参数、564 keys；TSS 导出移除 98 个参数和 4 个 keys。

#### P5.4 当前允许的论文结论

推荐表述：

> 在固定 seed42 下，完整模型相对仓库内成对 Original baseline 呈现数据集相关的固定工作点权衡，并在四个数据集专属训练/测试设置的 official-test sweep 中于 Fa≤1e-5 获得更高 Pd。该结果支持继续进行冻结工作点的 clean protocol 确认，但尚不支持全面支配、跨随机性稳定性或把性能差异严格归因于单个分支。

在 P1/P2 尚未完成前，不应写“更稳定的目标保持能力”；目标保持、杂波抑制和多尺度增强仍是设计假设及待验证解释，已建立的是性能工作区间，而不是完整因果机制。

---

## 7. 需要新增或复用的代码文件

原则：**优先复用现有封存和推理逻辑，只新增协议、共享记录与选择工具；当前不修改冻结模型公式。**

### 7.1 协议与现有结果核验

```text
experiments/POST_ARCHITECTURE_OPTIMIZATION_PROTOCOL_V1.md
experiments/verify_four_dataset_seed42_bundle_v1.py
experiments/build_recursive_runtime_source_manifest_v1.py
```

不再新增重复复制 checkpoint 的 `freeze_four_dataset_seed42_results_v1.py`；现有 `selected_checkpoints/checkpoint_manifest.json` 与 `postprocess/postprocess_artifact_manifest.json` 直接作为输入。

### 7.2 一次推理与离线诊断

```text
analysis/extract_original_final_inference_records_v1.py
analysis/compare_paired_common_thresholds_v1.py
analysis/analyze_component_score_margin_v1.py
analysis/analyze_target_fragmentation_v1.py
analysis/analyze_probability_calibration_v1.py
analysis/compare_original_final_error_sets_v1.py
```

后五个脚本只能读取第一个脚本生成的共享记录，不得各自加载模型重复前向。

### 7.3 clean split、训练、选模与校准

```text
experiments/build_post_architecture_clean_split_seed42_v1.py
experiments/audit_clean_split_roles_v1.py
experiments/compute_post_architecture_clean_tss_statistics_seed42_v1.py
experiments/train_post_architecture_clean_seed42_v1.py
experiments/select_checkpoint_on_model_val_v1.py
experiments/select_operating_threshold_on_calibration_v1.py
experiments/fit_monotonic_logit_calibrator_v1.py
experiments/evaluate_locked_clean_protocol_v1.py
```

### 7.4 条件触发的 TSS weight 搜索

```text
experiments/train_final_tss_weight_search_seed42_v1.py
experiments/launch_final_tss_weight_search_seed42_v1.sh
experiments/aggregate_final_tss_weight_search_seed42_v1.py
```

这些文件只在 P2.3 gate 明确触发后进入执行队列。

### 7.5 统计、效率与论文表格

```text
experiments/paired_image_bootstrap_locked_v1.py
experiments/benchmark_original_final_efficiency_v1.py
experiments/build_final_paper_tables_seed42_v1.py
```

当前不新增 multiseed launcher，也不启动消融脚本。

---

## 8. 必须增加的测试

```text
tests/test_model_sources_remain_frozen.py
tests/test_recursive_runtime_source_manifest.py
tests/test_shared_inference_record_reuse.py
tests/test_captured_logits_match_public_probability.py
tests/test_paired_common_threshold_union.py
tests/test_clean_split_roles_are_disjoint.py
tests/test_clean_tss_statistics_use_train_core_only.py
tests/test_checkpoint_selected_only_on_model_val.py
tests/test_calibration_cannot_select_lambda_or_epoch.py
tests/test_locked_test_evaluation_contract.py
tests/test_monotonic_calibrator_preserves_ranking.py
tests/test_monotonic_calibrator_preserves_pd_fa_curve.py
tests/test_raw_threshold_half_result_unchanged.py
tests/test_tss_weight_does_not_change_inference_graph.py
tests/test_inference_state_contains_no_survival_keys.py
tests/test_bootstrap_does_not_reselect_threshold.py
```

关键不变量：

1. 现有 checkpoint 的原始推理记录与历史结果在相同 evaluator 下逐元素一致；
2. Original/Final 的公共阈值集合是两者候选阈值的并集，并验证网格密度敏感性；
3. train_core、model_val、calibration、official test 两两无 ID 重叠；
4. train_core 是 normalization 与 TSS `pos_weight` 的唯一数据来源，所有正 `lambda_s` 候选复用同一统计；
5. model_val 只能选择 checkpoint、`lambda_s` 和 P3 gate，calibration 只能选择阈值或拟合 calibrator；
6. official test 运行前所有决策 manifest 已锁定，运行后程序拒绝回写选择结果；
7. 只读捕获 logits 不改变公开 probability，且 `sigmoid(captured_logits)` 逐元素一致；
8. 单调校准不改变 score 排序和连续 Pd–Fa 曲线；
9. TSS weight 改变训练 loss，但推理参数量、state key 和前向输出契约不变；
10. bootstrap 内不重新选择 checkpoint、threshold 或 calibrator；
11. Original 与 Final 使用同一 split、seed42、训练预算和 evaluator。

---

## 9. 决策树

```text
只读核验现有 seed42 结果与 16 个 best checkpoint
    ↓
P1：共享一次推理记录 + 配对公共阈值/组件/概率诊断
    ↓
先冻结 train_core / model_val / calibration，再从头成对训练
    ↓
P2.2：model_val 按低 Fa 多指标规则选择 checkpoint
    ↓
P2.3 的四项布尔触发式是否全部为真？
    ├─ 否 → 保持 lambda_s=0.005
    └─ 是 → P3：只做 TSS weight 一维搜索，再由 model_val 冻结配方
                    ↓
冻结全局 lambda_s，完成三个 dataset-specific clean run 与 model_val 选模
    ↓
P2.4：calibration 只冻结部署阈值/可选 calibrator
    ↓
锁定全部 manifest
    ↓
official test 一次性预注册评估
    ↓
P5：结果表、逐图 bootstrap、效率与边界化结论
```

official test 的结果只决定最终报告为“确认成功”还是“确认未成功”，不能在同一版本中触发回看调参。

---

## 10. 最终回答：现在到底是在调什么

### 当前冻结、不修改

```text
SCTransNet 主干与 decoder
TPD8-MPRS-DCH
五节点 NER4 Tail-Aware
QFG2-CROA
TSS 的两个 stride-16 训练端点与推理移除边界
```

### 当前首先优化

```text
1. 低 Fa 多指标 checkpoint policy
2. 从 model_val 到 calibration 的数据角色闭环
3. 可部署阈值，而不是 test sweep 上的理想阈值
4. 逐图统计与效率证据
```

### 仅在 model_val 触发时优化

```text
TSS weight：0.0025 / 0.005 / 0.01
```

`lambda_s=0` 与结构消融一起延后，不在当前队列。当前阶段最准确的名称是：

> **完整模型已完成代码设计，现处于固定 seed42 的后架构选模、部署工作点与最小训练配方优化阶段。**

---

## 11. 推荐的立即执行顺序

```text
1. 只读核验现有结果、16 个 best checkpoint 和递归运行时源码摘要
2. 对 8 个 best_miou checkpoint 各做一次推理，生成共享逐图记录
3. 完成配对公共阈值、组件分数、碎裂和条件校准诊断
4. 冻结 seed42 clean split 与三类数据角色 manifest
5. 先从头成对训练 SIRST3 Original / Final，model_val 选择主 checkpoint
6. 依据 P2.3 精确 gate 决定跳过或在 SIRST3 执行正 TSS weight 搜索
7. 冻结全局 TSS weight，再完成 NUAA / NUDT / IRSTD dataset-specific clean runs
8. 配方和 checkpoint 冻结后，calibration 选择主部署阈值并拟合可选 calibrator
9. 锁定协议与全部决策 manifest
10. 一次性执行 official-test suite，再完成结果表、逐图 bootstrap、效率与定性错误样例
```

当前不把多 seed 和消融加入队列；本轮只做 seed42，也不启动新的结构模块设计。

---

## 12. 最终项目状态建议

```text
decision=REVISE_PROTOCOL_BEFORE_POST_ARCHITECTURE_RUN

model_structure_complete=true
model_code_complete=true
post_architecture_experiment_code_complete=false
engineering_design_success=true
inference_architecture_candidate_frozen=true
architecture_redesign_required=false
innovation_mainline_changed=false
new_module_design_authorized=false

seed42_benchmark_competitiveness_supported=true
dataset_specific_test_sweep_pd_at_fa_le_1e_minus_5_advantage_observed=true
deployable_calibrated_advantage_supported=false
fixed_threshold_result=MIXED_TRADEOFF
universal_dominance=false

clean_split_pending=true
clean_retraining_pending=true
checkpoint_policy_optimization_pending=true
fixed_threshold_calibration_pending=true
training_recipe_finalized=false

broad_hyperparameter_search_authorized=false
tss_weight_minimal_search_authorized=conditional
multiseed_current_scope=false
structural_ablation_current_scope=false
tss_zero_weight_control_current_scope=false

paper_core_established=false
stability_claim_supported=false
```

---

## 参考代码与证据

1. 当前整体设计边界：  
   `SCTransNet_完整模型主线与性能驱动设计边界.md`
2. 四数据集 seed42 实验协议：  
   `SCTransNet_三数据集1000Epoch论文实验完整方案.md`
3. Final model integration and inference-head removal：  
   `model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py`
4. TPD V8-MPRS-DCH：  
   `model/tpd_clean_v8_mprs_dch.py`
5. NER V4 Tail-Aware：  
   `model/tpd_ner_v8_mprs_dch_v4_tail_aware.py`
6. QFG2-CROA：  
   `model/tpd_frequency_gate_v2_croa.py`
7. Target Survival heads、forward contract 与训练 loss：  
   `model/tpd_survival.py`、`model/tpd_forward_contract.py`、`experiments/tpd_training_loss.py`
8. 四数据集模型构建、训练、正式评价入口与评价协议：  
   `experiments/four_dataset_models_seed42_v1.py`、`experiments/train_four_dataset_original_final_seed42_exact_v1.py`、`experiments/evaluate_four_dataset_seed42_v1.py`、`experiments/four_dataset_evaluation_protocol_v1.py`
9. 当前正式结果与选中权重清单：  
   `results/four_dataset_seed42_v1/paper_results_summary.json`、`results/four_dataset_seed42_v1/paired_deltas_final_minus_original.json`、`results/four_dataset_seed42_v1/tables/table7_pd_at_fa_budgets.md`、`results/four_dataset_seed42_v1/postprocess/postprocess_status.json`、`results/four_dataset_seed42_v1/selected_checkpoints/checkpoint_manifest.json`
10. 历史 seed42 warm-start/NUDT 认证协议（仅作历史证据，不能替代本轮四数据集 scratch clean protocol）：  
    `experiments/FINAL_MODEL_CERTIFICATION_PROTOCOL_V1.md`
