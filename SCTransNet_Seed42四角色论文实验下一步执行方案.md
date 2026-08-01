# SCTransNet 完整模型：Seed 42 四角色论文实验下一步执行方案

> 审核对象：`Arialliy/SCTransNet_main` 当前 `main` 分支
> 审核日期：2026-08-01
> 适用模型：`SCTransNet + TPD8-MPRS-DCH + 五节点 NER4 Tail-Aware + QFG2-CROA + 训练期 TSS`
> 主随机种子：`42`
> 训练预算：`1000 epochs`，每 `10 epochs` 在 `model_val` 上评估一次
> 当前状态：方案文档已修正；尚未实现本协议代码，尚未启动本轮训练

---

## 0. 最终裁决

当前不再进行模块设计，也不应继续修改 TPD、NER、QFG 或 TSS 的结构定义。下一阶段应正式进入：

> **后架构论文实验优化：数据角色隔离、训练配方冻结、联合 checkpoint 选择、部署阈值校准和无重复推理评估。**

建议项目状态更新为：

```text
decision=IMPLEMENT_SEED42_FOUR_ROLE_PAPER_PROTOCOL

final_model_established=true
architecture_frozen=true
innovation_mainline_changed=false
new_module_design_authorized=false

primary_seed=42
multiseed_experiment_authorized=false
structural_ablation_authorized=false
lambda_zero_control_authorized=false

tss_positive_weight_search_authorized=conditional_only
shared_inference_cache_required=true

paper_core_established=false
stability_claim_supported=false
training_recipe_finalized=false
```

### 0.1 模型设计是否成功

模型设计本身已经成功完成，理由是：

1. 最终推理图已经明确，只保留 TPD、NER 和 QFG；
2. TSS 只在训练期提供辅助约束，推理构建器物理移除 TSS state；
3. 当前模型在 seed 42 上已经形成实际低 Fa 竞争力，而不是只有工程可运行性；
4. 当前不足来自实验协议、固定工作点和训练配方尚未最终冻结，不再是“缺少第五个模块”。

但当前只能声明：

```text
architecture_success=true
seed42_competitiveness_supported=true
universal_dominance=false
paper_core_established=false
stability_claim_supported=false
```

### 0.2 当前属于调参数吗

当前不是广义超参数搜索，也不是结构调参。只允许两类受限优化：

1. **部署校准**：`calibration` 仅选择部署阈值；
2. **条件化训练配方优化**：只有触发门成立时，才在正 TSS 权重
   `0.0025 / 0.005 / 0.01` 中选择一个，并在全部数据集统一使用。

下列内容暂不搜索：

```text
TPD 公式
NER topology / tail thresholds
QFG 结构、频带、alpha 或 gate 范围
主分割损失
optimizer / base LR / scheduler
batch size / patch size
数据增强
lambda_s = 0 对照
额外随机种子
结构消融
```

---

## 1. 代码复核结论与本轮修改边界

### 1.1 最终模型代码不需要再改公式

当前最终整模代码已经满足以下边界：

- `model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py`
  - 在一次整模前向中调用一次 `tpd_qfg.prepare(...)`；
  - prepared QFG modulation 被四个 SCTB 复用；
  - 推理类不注册 TSS heads；
  - head-free inference builder 与 survival training builder 分离。
- `model/tpd_survival.py`
  - TSS 只读取 `emb1/emb2` stride-16 endpoint；
  - 不改变 segmentation path；
  - 输出 raw survival logits。
- `experiments/tpd_training_loss.py`
  - 原六路 BCE 的加法顺序保持不变；
  - TSS 以 `lambda_s × BCEWithLogits` 叠加；
  - 正 TSS 权重要求结构化 survival logits。

因此，本轮不得通过修改最终模型文件来“提高通过率”。模型源文件进入只读 source lock。

### 1.2 当前真正需要修改的是实验代码

现有实验代码主要需要解决四个问题：

1. 既有内部划分只有 train/validation 两类，不能表达
   `train_core / model_val / calibration / official_test` 四种互斥职责；
2. 既有 checkpoint 只按 Pd-primary 和 mIoU-secondary 分开选择，未实现五指标联合选择；
3. 既有 sweep 脚本在每次调用时重新执行模型推理，固定阈值、校准、预算曲线和错误分析之间存在重复计算；
4. 既有 `best_points_under_fa_budget` 容易把“无非空可行点”编码为数值 0，需与真实 `Pd=0` 明确区分。

本轮代码变更原则为：

```text
只新增或重构 experiments/ 与 tests/ 层
不修改冻结模型数学图
不改变评估器的连通域、匹配半径和 Fa 定义
不修改 official test 内容
```

---

## 2. 四类数据角色协议

## 2.1 角色定义

每套数据必须拥有以下四类互斥角色：

| 角色 | 数据来源 | 唯一职责 | 明确禁止 |
|---|---|---|---|
| `train_core` | 官方训练集派生 | 梯度更新、训练归一化统计、TSS 类别统计 | 选择 epoch、选择阈值、报告最终测试结果 |
| `model_val` | 官方训练集派生 | 每 10 epoch 计算五指标并选择 checkpoint | 更新梯度、选择部署阈值、选择 TSS 权重以外的参数 |
| `calibration` | 官方训练集派生 | 在冻结 checkpoint 后选择部署阈值 | 选择 epoch、选择 checkpoint、选择 TSS 权重 |
| `official_test` | 官方测试集 | 冻结后的一次性论文评估和描述性曲线 | 任何训练、选模、校准或配方决策 |

访问关系必须由代码强制，而不只是文档约定：

```text
training runner        → train_core + model_val
checkpoint selector    → model_val metrics only
TSS selector           → model_val metrics only
threshold calibrator   → calibration cache only
official evaluator     → official_test cache only
```

## 2.2 推荐固定划分

从每个官方训练列表中采用：

```text
train_core  = 80%
model_val   = 10%
calibration = 10%
role_split_seed = 42
```

划分必须按以下维度进行确定性分层：

1. 数据来源；SIRST3 必须分别对 NUAA、NUDT、IRSTD 来源分层；
2. 图像 GT 目标数：`0 / 1 / 2+`；
3. 是否包含 tiny target，tiny 定义保持面积 `≤9 pixels`；
4. 在可行时平衡目标总数和 tiny-target 总数，而不仅平衡图像数。

若整数取整导致某一来源的 `model_val` 或 `calibration` 没有目标或没有 tiny target，manifest builder 应执行最小交换修复，并把交换记录写入审计文件，不能静默改变比例。

## 2.3 SIRST3 的特殊规则

SIRST3 是混合训练协议，不是第四个未知域数据集。其规则为：

```text
三个来源的 train_core 合并训练
三个来源的 model_val 合并选一个 checkpoint
三个来源的 calibration 合并选一个部署阈值
同一个 checkpoint + 同一个冻结阈值
分别评估 aggregate test 与三个来源 test breakdown
```

禁止：

```text
为 NUAA / NUDT / IRSTD 来源分别选择 SIRST3 checkpoint
为三个来源分别选择 SIRST3 部署阈值
根据来源测试结果重新校准
```

## 2.4 归一化与 TSS 统计

- 图像 `mean/std` 只能由对应实验的 `train_core` 计算；
- `model_val`、`calibration`、`official_test` 均复用该统计量；
- TSS 的 `survival_pos_weight` 只能从 `train_core` 的冻结统计流计算；
- `survival_pos_weight` 是数据不平衡统计，不属于人工搜索参数；
- 多数据集必须统一的是 `lambda_s`，不是强行统一每套数据天然不同的正负 cell 比例。

建议输出：

```text
artifacts/roles/<dataset>/role_manifest_v1.json
artifacts/roles/<dataset>/train_core.txt
artifacts/roles/<dataset>/model_val.txt
artifacts/roles/<dataset>/calibration.txt
artifacts/roles/<dataset>/official_test.txt
artifacts/normalization/<dataset>_train_core_norm_v1.json
artifacts/survival/<dataset>_train_core_survival_stats_v1.json
```

每份 manifest 必须记录：

```text
role_split_seed
source official list SHA256
role ID SHA256
image/mask content fingerprint
image count
target count
tiny-target count
source-wise counts
overlap audit
```

---

## 3. 固定训练协议

## 3.1 主实验常量

```yaml
training_seed: 42
role_split_seed: 42
epochs: 1000
eval_every: 10
candidate_epoch_count: 100
batch_size: 16
patch_size: 256
optimizer: Adam
base_lr: 1.0e-3
min_lr: 1.0e-5
warmup_epochs: 10
scheduler: warmup_cosine
precision: FP32
primary_metric_threshold: 0.5
match_radius: 3.0
tiny_area: 9
```

除 TSS 条件搜索外，Original 与 Final 使用完全相同的训练预算、数据顺序、增强随机流和评估周期。

## 3.2 从头训练与初始化公平性

论文主比较必须从头训练：

```text
parent_checkpoint = null
optimizer_state_inherited = false
scheduler_state_inherited = false
```

共有 state key 且 shape 相同的 SCTransNet 参数应通过显式 state copy 实现配对初始化，不依赖两个 builder 的调用顺序。新增模块保持冻结初始化规则：

- TPD：既定 SPD-compatible 初始化；
- NER：既定 relay 初始化和 Tail-Aware 常量；
- QFG：既定 identity/zero-terminal 初始化；
- TSS heads：严格零初始化；
- optimizer：每个模型独立创建 fresh Adam。

每个 run 必须输出 initialization manifest，并逐 tensor 验证共有 state 一致。

## 3.3 训练期间的数据访问

训练进程只能接收：

```text
train_core manifest path
model_val manifest path
```

训练 CLI 中不得出现 calibration 或 official test 路径。这样可以从能力层面避免误访问。

每 10 epochs：

1. 在 `model_val` 全图推理；
2. 固定 `threshold=0.5`；
3. 记录 Pd、Fa、tiny-Pd、nIoU、mIoU；
4. 保存候选 checkpoint；
5. 不执行 calibration sweep；
6. 不访问 official test。

建议保存：

```text
checkpoints/candidate_epoch_0010.pth.tar
...
checkpoints/candidate_epoch_1000.pth.tar
last.pth.tar
metrics_model_val.jsonl
```

---

## 4. model_val 五指标联合 checkpoint 选择

## 4.1 只建立一个权威 checkpoint 角色

本轮不再用 `best_Pd` 和 `best_mIoU` 分别承担论文主结论。权威角色改为：

```text
best_joint.pth.tar
checkpoint_role=best_model_val_joint_five_metric
```

`best_Pd`、`best_mIoU` 可保留为诊断产物，但不得用于 calibration、official test 主表或 TSS 权重选择。

## 4.2 指标输入

对每个候选 epoch，在 `model_val`、`threshold=0.5` 下构造：

\[
V_e=(Pd_e, -Fa_e, TinyPd_e, nIoU_e, mIoU_e)
\]

必须同时保存比率与原始计数：

```text
matched_target_count / target_count
unmatched_predicted_pixels / valid_pixel_count
matched_tiny_target_count / tiny_target_count
nIoU
mIoU
```

## 4.3 确定性联合选择规则

采用“Pareto 过滤 + 等权 rank aggregation”，避免人为调权重：

1. 删除非有限或审计失败的候选 epoch；
2. 在五指标上构建非支配集合；
3. 对全部有效候选分别计算五个 dense rank，最优 rank 为 1，Fa 使用升序；
4. 对非支配候选计算：

\[
R_e=r_{Pd}+r_{Fa}+r_{Tiny}+r_{nIoU}+r_{mIoU}
\]

\[
W_e=\max(r_{Pd},r_{Fa},r_{Tiny},r_{nIoU},r_{mIoU})
\]

5. 按以下 tuple 取最小值：

```text
(
  rank_sum R_e,
  worst_rank W_e,
  -matched_target_count,
  unmatched_predicted_pixels,
  -matched_tiny_target_count,
  -nIoU,
  -mIoU,
  epoch
)
```

该规则具有以下性质：

- 不会选中被另一 epoch 全面支配的 checkpoint；
- 五个指标共同参与；
- 不依赖不同量纲的人工线性权重；
- 完全确定性；
- 同分时优先更早 epoch，避免无证据的额外训练。

## 4.4 选择产物

```text
selection/model_val_joint_selection_v1.json
selection/model_val_pareto_frontier_v1.json
selection/best_joint.pth.tar
selection/best_joint_checkpoint_sha256.txt
```

选择文件必须明确：

```text
selection_data_role=model_val
selection_threshold=0.5
calibration_accessed=false
official_test_accessed=false
```

---

## 5. TSS 正权重的条件搜索

## 5.1 默认值与禁止事项

初始默认：

```text
lambda_s_default=0.005
```

本阶段暂不训练 `lambda_s=0`，因此不得形成“TSS 独立因果贡献已证明”的论文声明。

TSS 搜索只能在 SIRST3 主训练协议上触发；不得对 NUAA、NUDT、IRSTD 分别寻找不同权重。

## 5.2 触发门

先完成 SIRST3 的 Original 与 Final(`lambda_s=0.005`) seed-42 formal1000，并各自选出 `best_joint`。仅当以下三个条件同时满足时，才允许搜索：

### 条件 A：目标存活缺口

```text
Final.matched_target_count < Original.matched_target_count
OR
Final.matched_tiny_target_count < Original.matched_tiny_target_count
```

### 条件 B：模型并非整体崩塌

Final 至少在一个质量/抑制维度优于 Original：

```text
Final.Fa < Original.Fa
OR Final.nIoU > Original.nIoU
OR Final.mIoU > Original.mIoU
```

### 条件 C：工程与数据协议完整

```text
exact_resume_pass=true
role_isolation_pass=true
joint_selection_pass=true
source_lock_pass=true
nonfinite_or_cache_error=false
```

形式化为：

```text
tss_search_trigger = A and B and C
```

若触发门不成立：

```text
lambda_s_global=0.005
tss_search_skipped=true
```

## 5.3 搜索矩阵

触发后仅允许：

```text
lambda_s ∈ {0.0025, 0.005, 0.01}
```

若 `0.005` formal1000 已满足完全相同协议，则复用，不重复训练，只补 `0.0025` 与 `0.01`。

每个候选：

- 同一 `train_core/model_val`；
- seed 42；
- 1000 epochs；
- 相同初始化和数据随机流；
- 自己用 model_val 五指标联合规则选 `best_joint`；
- calibration 与 official test 保持不可见。

## 5.4 权重选择与全局冻结

三个候选的 `best_joint` 再使用同样的五指标 Pareto + rank aggregation 进行跨权重选择。完全相同时，tie-break 为：

```text
优先 0.005
其次优先离 0.005 绝对距离更小
```

最终生成：

```text
training_recipe/tss_weight_decision_v1.json
lambda_s_global=<0.0025|0.005|0.01>
selection_source=SIRST3_model_val_only
```

此后：

- SIRST3、NUAA、NUDT、IRSTD 的 Final 全部使用同一个 `lambda_s_global`；
- calibration 不得改变该权重；
- official test 不得改变该权重；
- 不允许数据集特定 lambda。

只有这一步封存后，才可更新：

```text
training_recipe_finalized=true
```

---

## 6. calibration 只选择部署阈值

## 6.1 前置条件

只有在以下内容全部冻结后才运行 calibration：

```text
model architecture
lambda_s_global
best_joint checkpoint
normalization
preprocessing
metric implementation
```

calibration 不得看到候选 epoch，也不得重新选择 checkpoint。

## 6.2 阈值候选

复用现有 audited closed-interval/adaptive threshold 生成规则，并强制包含：

```text
0.5
低概率尾部点
高概率尾部点
float32 上边界点
```

Original 与 Final、所有数据集使用完全相同的生成算法。

## 6.3 部署阈值选择规则

预注册部署预算：

```text
calibration_deploy_fa_budget=1e-5
```

先删除“预测为空”的点；空预测即使 Fa=0，也不能被当作有效部署点。

若存在非空且 `Fa≤1e-5` 的点：

1. 在可行点上构造 Pd、Fa、tiny-Pd、nIoU、mIoU 的 Pareto 集；
2. 使用与 checkpoint 相同的等权 rank aggregation；
3. 完全同分时依次选择：

```text
更高 matched_target_count
更高 matched_tiny_target_count
更低 Fa
更高 nIoU
更高 mIoU
更接近 0.5
更高 threshold
```

若不存在非空可行点：

```text
calibration_budget_reachable=false
deployment_threshold=0.5
```

不得为了强行满足预算而选取空预测点或临时扩大 Fa budget。

## 6.4 阈值的作用域

- SIRST3 mixed model：一个 aggregate calibration 阈值，原样用于 aggregate test 和三个来源 breakdown；
- 单数据集独立训练：每个训练协议可有自己的 calibration 阈值；
- Original 与 Final 可以各自校准，因为两者概率尺度不同，但必须使用同一规则和同一 calibration split；
- 不允许从 official test 反推阈值。

输出：

```text
calibration/<method>/<dataset>/deployment_threshold_v1.json
```

必须包含：

```text
checkpoint_sha256
calibration_manifest_sha256
prediction_cache_sha256
selection_rule
selected_threshold
metrics_on_calibration
budget_reachable
calibration_only=true
```

---

## 7. frozen-threshold 与 test-sweep Pd@Fa 的统计口径

这是本轮必须修复的核心报告问题。

## 7.1 固定阈值结果

论文必须同时报告两个固定点：

### Raw benchmark point

```text
threshold=0.5
```

用于与 SCTransNet/BasicIRSTD 既有口径对齐。

### Frozen calibrated deployment point

```text
threshold=tau_calibration
```

该阈值只由 calibration 选择，然后在 official test 上固定应用。

其正确名称为：

```text
Official-test metrics at the calibration-frozen threshold
```

不能将它写成 `Pd@Fa≤B`，除非它在 official test 上实际满足对应预算。应同时报告实际 test Fa。

## 7.2 test-sweep Pd@Fa

`Pd@Fa≤B` 是在 official test 概率缓存上扫描阈值得到的**描述性测试包络**：

\[
Pd@Fa\le B=\max_{\tau:Fa(\tau)\le B,\;N_{pred}(\tau)>0} Pd(\tau)
\]

它用于描述排序能力，但：

```text
不用于选择 checkpoint
不用于选择 lambda_s
不用于选择部署阈值
不回写训练配方
```

论文中应明确标注：

```text
test-sweep operating envelope / oracle envelope
```

## 7.3 无可行点的编码

如果某一 Fa budget 下没有任何非空预测点满足预算，输出必须为：

```json
{
  "budget": 1e-6,
  "reachable": false,
  "pd": null,
  "matched_target_count": null,
  "threshold": null
}
```

论文表格显示：

```text
— (unreachable)
```

不得再用数值 `0` 表示，因为 `Pd=0` 与“无非空可行点”具有不同统计含义。

## 7.4 预算点 tie-break

对每个 test Fa budget，在所有非空可行点中：

```text
1. 最大 matched_target_count / Pd
2. 最大 matched_tiny_target_count / tiny-Pd
3. 最大 nIoU
4. 最大 mIoU
5. 最低实际 Fa
6. 更高 threshold
```

所有指标必须在完整数据集上聚合后选择，禁止逐图先选点再平均。

---

## 8. 共享推理缓存

## 8.1 缓存放在实验层，不修改冻结模型

当前 QFG 已经在单次整模 forward 内 prepare 一次并被四个 SCTB 复用；TSS 也复用当前 forward 的 `emb1/emb2`。本轮新增的是**checkpoint–dataset 级共享概率缓存**，不是再修改网络图。

目的：同一个 checkpoint 在同一数据角色上的以下任务只做一次模型推理：

```text
threshold=0.5 指标
calibration 阈值选择
frozen-threshold 指标
Pd–Fa sweep
source breakdown
错误目标与组件分析
论文表格生成
```

## 8.2 缓存键

缓存唯一键至少包含：

```text
model method/variant
checkpoint SHA256
model source-lock SHA256
architecture manifest SHA256
dataset role manifest SHA256
normalization SHA256
preprocessing contract SHA256
inference precision
pad/crop rule
output probability convention
```

任一字段变化均必须创建新缓存，不能静默复用。

## 8.3 缓存内容

建议目录：

```text
inference_cache/
  <dataset>/
    <role>/
      <method>/
        <checkpoint_sha256>/
          cache_manifest.json
          index.jsonl
          probabilities/<namespaced_id>.npz
          completion_attestation.json
```

每个 image entry 保存：

```text
namespaced image ID
source dataset
original H/W
probability float32 array SHA256
target mask SHA256
preprocessing metadata
```

概率必须保存为 float32；float16 可能改变高阈值尾部排序，不适合 Pd–Fa 精确复核。

## 8.4 写入与读取规则

- write-once；默认禁止覆盖；
- 临时目录写完、校验数量与 hash 后原子 rename；
- 缓存完成前不得被 evaluator 读取；
- evaluator 默认为 cache-only，不得在缺失时隐式重新推理；
- 需要重建必须使用显式 `--rebuild-cache` 并生成新缓存版本；
- direct inference 与 cache evaluation 必须逐图逐指标一致。

## 8.5 只缓存冻结 checkpoint

训练中 100 个 candidate epoch 只记录 model_val 指标并保存 checkpoint，不为每个 epoch建立完整共享缓存。完成联合选择后，仅对 `best_joint` 建立：

```text
model_val cache（选择复核）
calibration cache（阈值选择）
official_test cache（最终固定点、sweep 和分析）
```

这样既减少重复推理，也避免不必要的磁盘占用。

---

## 9. 需要新增或修改的代码

## 9.1 协议常量与 manifest

### 新增 `experiments/paper_seed42_four_role_protocol.py`

统一定义：

```text
seed=42
epochs=1000
eval_every=10
role ratios
metric threshold=0.5
match radius/tiny area
joint selector version
calibration policy
Fa budgets
allowed lambda set
```

### 新增 `experiments/build_paper_four_role_manifests.py`

职责：

- 读取官方 train/test IDs；
- namespaced IDs；
- 统计目标与 tiny target；
- 分层拆分 train_core/model_val/calibration；
- 直接绑定 official test；
- 写 hash 和交换修复记录。

### 新增 `experiments/audit_paper_four_role_manifests.py`

硬检查：

```text
四角色两两无交集
三类 train-derived role 的并集等于 official train
外部 official test 完全一致
图像与 mask 存在且一一对应
SIRST3 来源无遗漏
hash 可重建
```

## 9.2 数据加载

### 新增 `experiments/paper_dataset.py`

不要直接破坏历史 `dataset.py`。新模块提供：

```python
build_train_core_loader(...)
build_full_image_role_loader(role="model_val" | "calibration" | "official_test")
```

要求：

- `train_core` 才允许随机 crop/augmentation；
- 其余角色全图、确定性 pad；
- role 参数使用枚举，不接受任意字符串；
- training runner 不导入 official-test builder。

## 9.3 统一训练 runner

### 新增 `experiments/train_paper_seed42_exact.py`

建议用一个 runner 支持：

```text
--method original|final
--dataset SIRST3|NUAA|NUDT|IRSTD
--tss-weight <frozen positive value for final>
```

关键断言：

```text
seed == 42
epochs == 1000
eval_every == 10
official_test path absent
calibration path absent
parent checkpoint absent
FP32
common initialization audit pass
```

需要复用既有 exact-resume 内核，保存：

```text
Python/NumPy/Torch CPU/CUDA RNG
DataLoader generator state
optimizer/scheduler
current epoch
candidate checkpoint inventory
model_val metric log
```

## 9.4 联合 checkpoint selector

### 新增 `experiments/select_model_val_joint_checkpoint.py`

实现第 4 节的 Pareto + rank aggregation，禁止接收 calibration/test 输入。输出 `best_joint.pth.tar` 及完整选择审计。

## 9.5 TSS 配方控制器

### 新增 `experiments/decide_tss_positive_weight.py`

职责：

1. 读取 SIRST3 Original/Final-0.005 的 model_val joint 结果；
2. 计算触发门；
3. 输出 `SKIP_SEARCH` 或 `RUN_POSITIVE_SEARCH`；
4. 搜索完成后在三个正权重中选择；
5. 生成全局 recipe lock。

该脚本不得接收 calibration 或 official test 文件路径。

## 9.6 共享推理缓存

### 新增 `experiments/build_shared_inference_cache.py`

只做一次模型推理并写 float32 probability cache。

### 新增 `experiments/inference_cache_contract.py`

提供：

```python
cache_key(...)
validate_cache(...)
iter_cached_predictions(...)
```

## 9.7 calibration 与 official test evaluator

### 新增 `experiments/select_calibration_deployment_threshold.py`

- 只读 calibration cache；
- 不加载模型；
- 不读取 epoch 列表；
- 实现部署预算和 unreachable 规则。

### 新增 `experiments/evaluate_official_test_from_cache.py`

输出：

```text
raw threshold 0.5
frozen calibration threshold
source breakdown
component counts
```

### 新增 `experiments/evaluate_test_pd_fa_envelope_from_cache.py`

- 只读 official test cache；
- 生成描述性 test sweep；
- 明确 `oracle_envelope=true`；
- 无可行点输出 null/reachable=false。

### 新增 `experiments/finalize_seed42_paper_comparison.py`

生成表格、曲线数据、审计清单和最终裁决，不执行模型推理。

## 9.8 Source lock

### 新增 `experiments/freeze_seed42_paper_protocol_source_lock.py`

至少锁定：

```text
冻结模型代码
训练 runner
loss routing
四角色 manifest builder/auditor
dataset loader
joint selector
cache contract
calibrator
official evaluator
sweep evaluator
metric core
所有 role manifests
normalization/TSS statistics
recipe lock
环境与 GPU 信息
```

---

## 10. 测试矩阵

## 10.1 数据角色测试

```text
四角色 pairwise disjoint
train-derived union == official train
official test exact binding
source stratification deterministic
same seed rebuild byte-identical
tiny-target quota audit
normalization cannot access non-train_core
```

## 10.2 联合选择测试

```text
dominated epoch cannot win
all five metrics affect rank
ties deterministic
earlier epoch final tie-break
nonfinite candidate rejected
best_joint checkpoint SHA matches selected candidate
calibration/test input rejected
```

## 10.3 TSS 决策测试

```text
A/B/C trigger truth table
only positive candidate set accepted
lambda=0 rejected in this protocol
calibration/test artifacts rejected
selected lambda copied unchanged to every dataset contract
```

## 10.4 缓存测试

```text
direct inference == cached probability, exact float32
fixed-threshold metrics direct == cache
sweep direct == cache
wrong checkpoint hash rejected
wrong normalization hash rejected
partial cache rejected
write-once enforcement
atomic completion
```

## 10.5 统计口径测试

```text
empty prediction is not a feasible budget point
unreachable budget outputs null, not numeric zero
frozen threshold never selected on test
Pd@Fa selected globally, not image-wise
prediction comparison remains probability > threshold
same Hungarian/component metric implementation reused
```

## 10.6 工程测试

```text
ordinary Python full related tests
python -O full related tests
CPU smoke
RTX 5090 GPU2/GPU3 smoke
exact resume tensor equality
training vs head-free inference export equality
survival state absent from inference artifact
source-lock rebuild
```

---

## 11. 正式实验矩阵

## 11.1 基础矩阵

在全局 TSS 权重冻结后：

| 训练协议 | Original | Final | Seed | Epochs |
|---|---:|---:|---:|---:|
| SIRST3 mixed | 1 run | 1 run | 42 | 1000 |
| NUAA independent | 1 run | 1 run | 42 | 1000 |
| NUDT independent | 1 run | 1 run | 42 | 1000 |
| IRSTD independent | 1 run | 1 run | 42 | 1000 |

基础总数：

```text
4 training protocols × 2 methods = 8 formal1000 runs
```

## 11.2 条件增加的 TSS 运行

若触发门成立：

```text
SIRST3 Final lambda=0.0025
SIRST3 Final lambda=0.01
```

已有完全同协议的 `lambda=0.005` 可复用。因此最多额外 2 个 formal1000 run。

## 11.3 推荐 GPU 排程

先完成 SIRST3，因为它决定 TSS 全局权重：

```text
Wave 1
GPU2: SIRST3 Original
GPU3: SIRST3 Final lambda=0.005

Conditional Wave 2
GPU2: SIRST3 Final lambda=0.0025
GPU3: SIRST3 Final lambda=0.01

After recipe freeze
Wave 3
GPU2: NUAA Original
GPU3: NUAA Final

Wave 4
GPU2: NUDT Original
GPU3: NUDT Final

Wave 5
GPU2: IRSTD Original
GPU3: IRSTD Final
```

训练期间 official test 必须保持逻辑锁定。

---

## 12. 完整执行顺序

## Phase P0：协议封存

1. 新建本协议的仓库版本；
2. 冻结所有常量、选择规则、预算和 TSS 触发门；
3. 生成 source lock 草案；
4. 禁止在看到新结果后修改选择规则。

**退出条件**：协议 hash 固定。

## Phase P1：四角色数据工程

1. 构建 SIRST3、NUAA、NUDT、IRSTD role manifests；
2. 计算 train_core-only normalization；
3. 计算 train_core-only TSS statistics；
4. 完成无重叠与来源审计。

**退出条件**：四角色审计全部通过。

## Phase P2：实验代码实现

1. 实现统一 scratch runner；
2. 实现联合 checkpoint selector；
3. 实现 TSS positive-weight controller；
4. 实现共享 inference cache；
5. 实现 calibration/test cache evaluators；
6. 实现修正后的 budget reachability schema。

**退出条件**：ordinary、`python -O`、CPU 测试全部通过。

## Phase P3：GPU preflight

1. Original/Final 两步 smoke；
2. 1000-epoch 参数协议静态审计；
3. 中断–续训短轨迹等价；
4. 训练模型导出 head-free inference 等价；
5. cache direct-vs-readback 等价。

**退出条件**：GPU2/GPU3 smoke 和 exact-resume 通过。

## Phase P4：SIRST3 主训练与 TSS 决策

1. 训练 Original formal1000；
2. 训练 Final `lambda=0.005` formal1000；
3. model_val 联合选 `best_joint`；
4. 执行 TSS 搜索触发门；
5. 如触发，补两个正权重；
6. 仅依据 SIRST3 model_val 冻结 `lambda_s_global`。

**退出条件**：`training_recipe_finalized=true`。

## Phase P5：其余单数据集训练

使用同一 `lambda_s_global` 完成 NUAA、NUDT、IRSTD 的 Original/Final formal1000，并分别用自己的 model_val 联合选择 checkpoint。

**退出条件**：8 个基础 run 与所有 `best_joint` 完整。

## Phase P6：calibration

1. 为每个 `best_joint` 建 calibration probability cache；
2. 使用 calibration-only 规则选择部署阈值；
3. 封存 threshold manifest；
4. 禁止再改变 epoch 或 lambda。

**退出条件**：所有 threshold manifests 完整且可重建。

## Phase P7：official test unlock

只有所有模型、checkpoint、TSS weight、threshold、evaluator 和 cache contract 均冻结后，生成：

```text
official_test_unlock_manifest_v1.json
```

然后统一：

1. 为 official test 建一次共享 cache；
2. 输出 threshold=0.5 固定点；
3. 输出 calibration-frozen threshold 固定点；
4. 输出 test-sweep Pd@Fa 描述性包络；
5. 输出 SIRST3 三来源 breakdown；
6. 不根据结果回到训练或 calibration。

## Phase P8：论文产物与裁决

1. 生成主表和低 Fa 表；
2. 生成 Pd–Fa 曲线；
3. 生成 checkpoint/threshold 流程图；
4. 生成复杂度和推理效率表；
5. 输出结果边界与 claim manifest。

---

## 13. 论文建议表格

### 表 1：Raw fixed threshold 0.5

| Dataset | Method | mIoU | nIoU | F1 | Pd | Fa | tiny-Pd | false objects/image |
|---|---|---:|---:|---:|---:|---:|---:|---:|

说明：checkpoint 由 model_val 五指标联合选择，阈值固定为 0.5。

### 表 2：Calibration-frozen deployment point

| Dataset | Method | calibration threshold | test mIoU | test nIoU | test Pd | actual test Fa | tiny-Pd |
|---|---|---:|---:|---:|---:|---:|---:|

说明：阈值仅由 calibration 选择；表中 Fa 为 official test 实际值。

### 表 3：Official-test Pd–Fa envelope

| Dataset | Method | ≤5e-7 | ≤1e-6 | ≤5e-6 | ≤1e-5 | ≤5e-5 | ≤1e-4 |
|---|---|---:|---:|---:|---:|---:|---:|

说明：这是 test-sweep 描述性包络，不是部署阈值结果；不可达点显示 `—`。

### 表 4：SIRST3 同一权重三来源 breakdown

只能使用同一 SIRST3 `best_joint` 和同一 SIRST3 calibration threshold。

### 表 5：训练与部署复杂度

分别报告：

```text
Original inference params/FLOPs/latency
Final inference params/FLOPs/latency
Final training-only extra TSS params
TSS inference params = 0
```

---

## 14. 通过门与状态更新

## Gate E0：工程实现

```text
四角色代码完成
全部相关测试通过
GPU smoke 通过
exact resume 通过
source lock 通过
```

## Gate D0：数据职责隔离

```text
四角色互斥
train-derived union 完整
official test 未被训练/选模/校准访问
normalization 仅 train_core
```

## Gate R0：训练配方冻结

```text
TSS trigger 已执行
lambda_s_global 已封存
所有数据集使用同一 lambda_s_global
```

通过后：

```text
training_recipe_finalized=true
```

## Gate C0：checkpoint 合规

```text
所有 checkpoint 仅由 model_val 五指标联合选择
calibration/test 不参与 epoch 决策
```

## Gate K0：校准合规

```text
calibration 只选择 threshold
selected threshold 绑定 checkpoint SHA
无空预测伪可行点
```

## Gate T0：测试统计合规

```text
raw 0.5 与 frozen threshold 分开
frozen threshold 与 test-sweep 分开
unreachable budget 不编码为 0
所有报告来自同一 official-test cache
```

## Gate P0：论文核心证据

本阶段完成后，只有当公平 official-test 结果支持预注册结论时，才重新评估：

```text
paper_core_established=true|false
```

本阶段无论结果如何均保持：

```text
stability_claim_supported=false
```

因为仍只有 seed 42。

建议 paper core 的最低结果门为：

1. SIRST3 aggregate 的 Final 不被 Original 全面支配；
2. 三个来源数据集中至少两个在 fixed-point 或低 Fa 包络上形成明确正向；
3. `Fa≤1e-5` 的总体低虚警优势可复现，且不是由空预测点产生；
4. 不出现 tiny-Pd 系统性崩塌；
5. 推理开销增量与性能收益可接受；
6. 所有结论都使用 model_val 选 checkpoint、calibration 选阈值、official test 只报告。

---

## 15. 明确禁止的操作

```text
在 official test 上选择 epoch
在 official test 上选择部署 threshold
用 test-sweep 最优阈值冒充 frozen deployment threshold
对不同测试来源使用不同 SIRST3 threshold
根据数据集分别调整 lambda_s
把不可达 Fa budget 写成 Pd=0
在看到结果后修改 joint selector 或 trigger gate
重新增加模块
开展未预注册多 seed 或结构消融并混入本轮主表
```

---

## 16. 立即执行清单

下一步应按以下顺序开始，而不是直接启动 1000-epoch 长训练：

```text
[ ] 1. 将本方案转为仓库内冻结 protocol
[ ] 2. 实现四角色 manifest builder 与 auditor
[ ] 3. 生成四套数据的 role manifests
[ ] 4. 实现 train_core-only normalization / survival statistics
[ ] 5. 实现统一 Original/Final scratch runner
[ ] 6. 实现五指标 joint checkpoint selector
[ ] 7. 实现 TSS positive-only trigger/controller
[ ] 8. 实现 shared inference cache
[ ] 9. 实现 calibration-only threshold selector
[ ] 10. 修正 test-sweep unreachable schema
[ ] 11. 完成普通模式、-O、CPU、GPU smoke、exact resume
[ ] 12. 启动 SIRST3 Original 与 Final-0.005 formal1000
[ ] 13. 执行 TSS trigger 并冻结全局权重
[ ] 14. 启动其余三数据集 formal1000
[ ] 15. 完成 calibration 后统一解锁 official test
[ ] 16. 生成论文表格、曲线与最终裁决
```

---

## 17. 最终一句话结论

> **模型结构已经设计成功并应继续冻结；当前处于论文实验协议与训练配方最终化阶段。下一步不是全面调参，而是先把四类数据职责、五指标联合选模、条件化正 TSS 权重、calibration-only 阈值以及共享推理缓存落实为代码，再以固定 seed 42 完成无职责泄漏的 1000-epoch Original/Final 公平复现。**
