# SCTransNet V2 结果复盘与下一步论文实验执行方案（修订版）

> 审核对象：`/home/ly/SCTransNet_main` 当前完整模型主线  
> 完整推理结构：`SCTransNet + TPD8-MPRS-DCH + 五节点 NER4 Tail-Aware + QFG2-CROA`  
> 训练期辅助：TSS dynamic ratio cap V2  
> 固定随机种子：`42`  
> 正式预算：`1000 epochs`，每 `10 epochs` 评估一次  
> 固定评估阈值：`0.5`  
> 当前完成：IRSTD-1K、NUDT-SIRST Final V2  
> 下一步：SIRST3、NUAA-SIRST Final V2  
> 结论边界：`paper_core_established=false`，`stability_claim_supported=false`

---

## 0. 最终裁决

### 0.1 当前模型状态

模型结构代码已经实现完成，当前冻结：

```text
SCTransNet
+ TPD8-MPRS-DCH
+ 五节点 NER4 Tail-Aware
+ QFG2-CROA
```

本轮 V2 只修改训练期 TSS 的有效权重，不修改模型表示图、测试前向、参数量、FLOPs 或部署接口。当前正式入选的 `best_miou.pth.tar` 与 `best_pd.pth.tar` 是训练图 checkpoint，仍包含四个 `target_survival.*` 参数键；只有显式调用部署导出构建器时，才会物理移除 TSS heads 并生成无 TSS 的推理模型。

当前两组结果表明：

- IRSTD-1K 与 NUDT-SIRST 的 `best_miou` 工作点均提高 Pd 和 nIoU；
- 两组 `best_miou` 同时存在少量 mIoU 回退与更高 Fa，属于混合权衡；
- 两组 `best_pd` 工作点整体明显优于 V1 Final，并形成有竞争力的 Pd–Fa–mIoU–nIoU 组合；
- 当前结果支持继续验证 V2，但不足以宣布 V2 已成为四训练制度通用配方。

因此当前状态固定为：

```text
decision=RUN_REMAINING_V2_REGIMES_FOR_RETROSPECTIVE_SCREEN

architecture_implementation_complete=true
architecture_frozen=true
innovation_mainline_changed=false
new_module_design_authorized=false

v2_key_dataset_result=POSITIVE_MIXED_TRADEOFF
v2_training_recipe_candidate=true
v2_multi_regime_validation_complete=false
v2_global_recipe_candidate_accepted=pending

paper_core_established=false
stability_claim_supported=false
training_recipe_finalized=false
```

### 0.2 当前优化阶段

当前属于：

> **后架构训练配方筛选阶段。**

当前不开展新模块设计、TPD/NER/QFG结构搜索、学习率搜索或第二项训练修改。先使用完全相同的V2代码补齐SIRST3与NUAA-SIRST。

V2源码的实际计算为：

\[
\lambda_{\mathrm{eff}}
=
\min\left(
0.005,
0.10\frac{\operatorname{sg}(L_{\mathrm{seg}})}
{\max(\operatorname{sg}(L_{\mathrm{tss}}),\epsilon_{\mathrm{fp32}})}
\right)
\]

\[
L=L_{\mathrm{seg}}+\lambda_{\mathrm{eff}}L_{\mathrm{tss}}.
\]

`lambda_eff`由停止梯度的损失标量计算；TSS损失本身仍对模型参数反向传播。

---

## 1. 当前 V2 结果的完整复盘

### 1.1 checkpoint选择口径

Original、V1 Final和V2 Final均遵循相同历史筛选口径：

```text
threshold = 0.5
candidate epochs = 10, 20, ..., 1000
每个模型从自己的100个候选epoch中独立选择最优checkpoint
```

两种选模键为：

```text
best_miou = (mIoU, Pd, -Fa, nIoU, tiny-Pd, -test_loss, -epoch)
best_pd   = (Pd, -Fa, tiny-Pd, mIoU, nIoU, -test_loss, -epoch)
```

`best_miou`实际上以连续mIoU为主，其余指标通常只在精确同分时打破平局。当前历史筛选使用对应测试列表选模，协议已标记：

```text
test_selected=true
selection_is_optimistic=true
```

因此这些结果属于benchmark-compatible工程筛选，不冒充尚未访问测试集的论文确认实验。

### 1.2 best_miou：Original、V1 Final、V2 Final三方完整比较

| 数据集 | 方法 | Epoch | mIoU ↑ | nIoU ↑ | Pd ↑ | Fa ↓ | tiny-Pd ↑ | 错误目标/图 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | Original | 270 | 0.673543 | 0.636875 | 282/297 | 2.2110e-5 | 23/30 | 0.407960 |
| IRSTD-1K | V1 Final | 470 | 0.669581 | 0.660009 | 278/297 | 2.1977e-5 | 23/30 | 0.323383 |
| IRSTD-1K | V2 Final | 180 | 0.670987 | 0.650185 | **283/297** | 2.9796e-5 | 22/30 | 0.467662 |
| NUDT-SIRST | Original | 520 | 0.945607 | 0.947437 | 935/945 | 2.5048e-6 | 258/259 | 0.034639 |
| NUDT-SIRST | V1 Final | 410 | 0.944498 | 0.948648 | 936/945 | 4.3892e-6 | 258/259 | 0.042169 |
| NUDT-SIRST | V2 Final | 500 | 0.944538 | **0.947775** | **939/945** | 5.6991e-6 | 258/259 | 0.043675 |

V2 Final相对Original：

| 数据集 | ΔmIoU | ΔnIoU | Δmatched target | ΔFa | Δtiny |
|---|---:|---:|---:|---:|---:|
| IRSTD-1K | −0.002556 | +0.013311 | +1 | +7.6863e-6（+34.8%） | −1 |
| NUDT-SIRST | −0.001069 | +0.000338 | +4 | +3.1942e-6（+127.5%） | 0 |

准确结论是：

> `best_miou@0.5`下，两个数据集的Pd与nIoU均超过Original，但代价是少量mIoU回退和更高Fa；IRSTD还少检出一个tiny目标。

V2 Final相对V1 Final的`best_miou`主要把工作点推向更高Pd，而不是让所有指标同向提高：

| 数据集 | ΔmIoU | ΔnIoU | Δmatched target | ΔFa |
|---|---:|---:|---:|---:|
| IRSTD-1K | +0.001406 | −0.009823 | +5 | +7.8192e-6 |
| NUDT-SIRST | +0.000040 | −0.000874 | +3 | +1.3099e-6 |

### 1.3 best_pd：Original、V1 Final、V2 Final三方完整比较

| 数据集 | 方法 | Epoch | mIoU ↑ | nIoU ↑ | Pd ↑ | Fa ↓ | tiny-Pd ↑ | 错误目标/图 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | Original | 230 | 0.619141 | 0.627174 | 287/297 | 4.9193e-5 | 24/30 | 0.701493 |
| IRSTD-1K | V1 Final | 320 | 0.612761 | 0.632386 | 287/297 | 5.5512e-5 | 25/30 | 0.631841 |
| IRSTD-1K | V2 Final | 370 | **0.658486** | **0.658560** | **288/297** | **2.8145e-5** | **25/30** | **0.502488** |
| NUDT-SIRST | Original | 260 | 0.915686 | 0.925523 | **941/945** | 1.3811e-5 | 258/259 | 0.085843 |
| NUDT-SIRST | V1 Final | 320 | 0.933939 | 0.936660 | 939/945 | **5.1246e-6** | 258/259 | 0.051205 |
| NUDT-SIRST | V2 Final | 540 | **0.941290** | **0.945203** | 940/945 | 5.6761e-6 | 258/259 | **0.037651** |

V2 Final相对Original：

| 数据集 | ΔmIoU | ΔnIoU | Δmatched target | ΔFa | Δtiny |
|---|---:|---:|---:|---:|---:|
| IRSTD-1K | +0.039345 | +0.031387 | +1 | −2.1047e-5（−42.8%） | +1 |
| NUDT-SIRST | +0.025604 | +0.019680 | −1 | −8.1350e-6（−58.9%） | 0 |

`best_pd`是当前V2最强的正向证据，但NUDT仍是少检出1/945个目标换取明显更低Fa与更高mIoU/nIoU的权衡，不写成全面支配。

### 1.4 当前证据能支持什么

已经支持：

```text
完整结构工程实现完成
V2在两个关键训练制度形成正向但混合的性能变化
best_miou下两个关键训练制度的Pd/nIoU超过Original
best_pd下V2整体明显改善
继续使用完全相同配方验证SIRST3/NUAA具有价值
```

尚未支持：

```text
动态cap已在四训练制度普遍有效
V2全面提高mIoU/Pd/Fa/nIoU/tiny-Pd
动态cap是终端性能变化的唯一因果原因
随机性稳定性或统计显著性
论文核心结论已经建立
```

当前只能写“结果与降低过强TSS标量贡献的假设一致”，不能写成已经证明目标存活机制。

---

## 2. V1问题与V2解释边界

V1候选checkpoint附近的加权TSS损失相对主分割损失标量占比约为：

| 训练制度 | `0.005 × L_tss / L_seg` |
|---|---:|
| SIRST3 pooled | 20.8% |
| NUAA-SIRST | 17.6% |
| NUDT-SIRST | 7.7% |
| IRSTD-1K | 31.0% |

这些数值证明固定名义权重不等于固定损失标量占比，但不等同于梯度范数，也不能单独建立因果解释。V2直接限制的是每个mini-batch的加权损失贡献比例。

当前剩余问题包括：

- `best_miou`下像素mIoU仍略低；
- `best_miou`下Fa更高；
- IRSTD tiny-Pd回退1/30；
- NUDT `best_pd`比Original少检出1/945；
- checkpoint角色之间存在明显性能端点差异。

这些问题必须通过完整指标表和后续错误分析呈现，不能只用阈值校准或换选模名称掩盖。

---

## 3. 下一步立即执行的两项正式任务

| 物理GPU | 训练制度 | 方法 | Seed | Epochs | 评估间隔 |
|---|---|---|---:|---:|---:|
| GPU 2 | SIRST3 pooled | Final V2 | 42 | 1000 | 10 |
| GPU 3 | NUAA-SIRST | Final V2 | 42 | 1000 | 10 |

严格固定：

```text
模型结构不变
lambda_max = 0.005
TSS ratio cap = 0.10
seed = 42
epochs = 1000
eval_every = 10
batch_size = 16
patch_size = 256
optimizer = Adam
原warmup + cosine
FP32
threshold = 0.5
同一img_idx清单
同一归一化
同一增强
同一评估器
```

只使用两种历史兼容选模角色：

```text
best_miou.pth.tar
best_pd.pth.tar
```

每个模型从自己的100个候选epoch中独立选模。训练期间只滚动保存一个续训状态；正式完成后删除续训状态，只保留上述两个入选checkpoint。

为确保与已完成IRSTD/NUDT完全同源，补跑前不得修改以下三个文件：

| 文件 | 当前SHA-256 |
|---|---|
| `experiments/train_four_dataset_original_final_seed42_exact_v1.py` | `2b1be09e97c1d780359fe6227a464969129c8c6cb1aa59c8125636a023ce35d7` |
| `experiments/tpd_training_loss.py` | `780fda362c9f342840671fd9e97569c2c2fad0eb25e48e9fd61122df6451e652` |
| `experiments/train_four_dataset_final_seed42_tss_cap_v2.py` | `2edfad996deb8aeac5cccdd731e4bd25a9a36791f7bc41d0e7467765b0f95d63` |

禁止在中途根据SIRST3或NUAA结果修改ratio cap、TSS权重、学习率、数据增强、阈值或选模规则。

---

## 4. V2现有日志口径

为了保持四训练制度的源码和日志口径一致，本轮不增加新的percentile累计器。共同诊断字段固定为当前已实现字段：

```text
train_tss_requested_weight
train_tss_ratio_cap
train_tss_effective_weight_mean
train_tss_weighted_loss
train_tss_weighted_to_segmentation_ratio
train_tss_cap_active_sample_fraction
train_segmentation_loss
train_survival_loss
```

其中`train_tss_cap_active_sample_fraction`表示“处于cap激活mini-batch中的样本比例”，不是严格的batch计数率。

当前IRSTD/NUDT日志无法恢复以下批次分布：

```text
lambda_eff std / p10 / p50 / p90 / min / max
raw ratio percentiles
cap_active_batch_count
```

这些字段不进入本轮四训练制度接受门。若未来干净论文协议需要细粒度诊断，应建立新日志版本，并在所有对应论文训练中统一记录；不为补日志重跑当前两组工程筛选。

---

## 5. 四训练制度聚合与裁决

### 5.1 独立性边界

SIRST3的train/test列表是NUAA、NUDT、IRSTD对应列表的严格拼接。因此：

```text
独立来源数据集 = NUAA-SIRST、NUDT-SIRST、IRSTD-1K
pooled训练制度 = SIRST3
```

三来源macro与目标总数只使用NUAA、NUDT、IRSTD。SIRST3单独报告，不能再次加入总matched target或当成第四份独立样本。

### 5.2 主判定：best_miou@0.5

主筛选始终比较每个方法各自的`best_miou@0.5`。计算：

```text
三个来源数据集的macro ΔmIoU
三个来源数据集的macro ΔnIoU
三个来源数据集的matched target总差值
三个来源数据集聚合Fa = 总错误像素 / 总有效像素
每个来源数据集的ΔmIoU、ΔnIoU、Δmatched、ΔFa、Δtiny
SIRST3 pooled的完整独立结果
```

当前筛选门是工程决策规则，不是统计显著性声明；它在SIRST3/NUAA V2结果产生前冻结。

#### `V2_RECIPE_ACCEPTED`

同时满足：

```text
A1. 三来源macro mIoU不低于Original；
A2. 三来源macro nIoU严格高于Original；
A3. 三来源matched target总数不低于Original；
A4. 三来源聚合Fa不高于Original；
A5. 任一来源均不得出现：
    mIoU下降 > 0.005，或
    nIoU下降 > 0.005，或
    matched target减少 > 2；
A6. SIRST3 pooled不得触发A5，且不得同时出现mIoU、nIoU、Pd三项下降；
A7. checkpoint、日志、协议和两个入选权重完整。
```

#### `V2_PARETO_MIXED_BUT_ADMISSIBLE`

满足：

```text
B1. 三来源macro nIoU严格高于Original；
B2. 三来源matched target总数不低于Original；
B3. A5/A6严重退化保护均未触发；
B4. 但macro mIoU或聚合Fa未满足全面接受条件。
```

含义：V2可以作为后续干净协议的候选配方，但只能声称混合权衡，不能声称全面支配。

#### `V2_RECIPE_REJECTED`

满足以下任一项：

```text
C1. 至少两个独立来源触发A5；
C2. 三来源macro nIoU不高于Original且matched target总数低于Original；
C3. SIRST3 pooled同时出现mIoU、nIoU和Pd下降，并且至少一个来源触发A5；
C4. 工程产物或协议不完整，无法形成同口径比较。
```

未满足明确接受或拒绝条件的结果一律归入`V2_PARETO_MIXED_BUT_ADMISSIBLE`并完整报告差值。

### 5.3 次判定：best_pd@0.5

`best_pd@0.5`必须完整报告mIoU、nIoU、Pd、Fa、tiny-Pd和错误目标数，但不替代主判定。它用于判断高检出端点是否形成更有价值的工作点，并作为论文辅助表。

动态cap自身的效果另做`V2 Final − V1 Final`比较；完整模型竞争力使用`V2 Final − Original`比较，两者不得混写。

---

## 6. checkpoint规则：继续使用best_miou与best_pd

当前任务不新增`best_joint`，也不改变历史选模规则。

论文兼容主checkpoint：

```text
best_miou@threshold=0.5
```

辅助高检出checkpoint：

```text
best_pd@threshold=0.5
```

原因：

- 与BasicIRSTD/SCTransNet现有评估口径保持一致；
- 不在看到当前结果后引入有利于V2的新联合排名；
- 避免连续mIoU/nIoU与离散Pd/tiny-Pd在dense rank中产生不等影响；
- 保留两个端点能够直接展示Pd–Fa–区域质量权衡。

未来若研究联合部署checkpoint，只能作为预注册的补充分析，不能替代主表的`best_miou`与`best_pd`。

---

## 7. 后续干净论文协议

本轮四训练制度V2属于历史test-selected工程筛选。若V2被接受或判为可接受的混合权衡，再进入新的数据角色协议。

只从每个数据集已有`img_idx/train`中划分：

| 来源official train | train_core | model_val | calibration |
|---|---:|---:|---:|
| NUAA-SIRST：213 | 171 | 21 | 21 |
| NUDT-SIRST：663 | 531 | 66 | 66 |
| IRSTD-1K：800 | 640 | 80 | 80 |
| SIRST3 pooled：1676 | 1342 | 167 | 167 |

SIRST3三个角色必须分别由三个来源对应角色的并集构成。现有`img_idx/test`不参加80/10/10拆分，作为最终固定测试角色。

角色职责：

| 角色 | 作用 | 禁止事项 |
|---|---|---|
| `train_core` | 梯度更新、归一化和TSS类别统计 | 选择checkpoint或阈值 |
| `model_val` | 每10 epochs选择`best_miou`与`best_pd` | 更新梯度或选择部署阈值 |
| `calibration` | checkpoint冻结后选择部署阈值 | 选择epoch、模型结构或TSS配方 |
| `fixed_test` | 最终固定评估与描述性阈值扫描 | 回写训练、选模或校准 |

必须满足：

```text
paper_role_split_seed = 42
model_training_seed = 42
两个seed字段语义分离
Original与Final复用同一角色manifest
所有候选正TSS配方复用同一角色manifest
normalization/TSS pos_weight/crop统计只来自train_core
样本ID、路径、内容摘要均执行角色重叠审计
SIRST3按来源、目标数量、目标面积/tiny状态分层
```

由于现有V1/V2已使用测试列表进行过开发与选模，后续协议只能保证“从新协议开始不再回写”，不能把`fixed_test`描述为历史上从未访问的首次盲测。论文必须披露这一开发过程。

---

## 8. 阈值与Pd@Fa报告

主表必须保留：

```text
threshold = 0.5
```

如果后续建立calibration角色，可额外报告冻结部署阈值，但不能用它替代0.5主结果。

Pd和Fa依赖阈值后的连通域、组件合并/断裂与质心匹配，因此不保证随阈值单调。论文中应称为：

> 描述性阈值–组件工作包络。

若注册阈值网格中没有任何非空预测点满足Fa budget，输出：

```json
{
  "registered_grid_nonempty_feasible": false,
  "pd": null,
  "matched_target_count": null,
  "threshold": null
}
```

“非空”必须定义为`predicted_object_count > 0`。非空但只产生虚警、Pd为0的点仍是合法点。只有枚举全部唯一FP32 score breakpoint后，才允许使用无修饰的`reachable=false`。

---

## 9. 当前实际代码与产物

V2已经实现，不新增虚构文件：

```text
experiments/tpd_training_loss.py
experiments/train_four_dataset_final_seed42_tss_cap_v2.py
```

`tpd_training_loss.py`通过可选`survival_ratio_cap`保留历史固定权重路径；不传cap时仍执行V1公式。当前单元测试覆盖：

```text
cap inactive时固定权重路径
cap active时有效贡献上限
lambda_eff无梯度路径
TSS仍保留梯度
segmentation加法顺序不变
zero/non-finite边界
FP32计算
```

现有source lock精确绑定：

```text
V1 base runner
V2 wrapper
training loss
数据manifest及其文件hash
```

它尚未覆盖完整Python依赖闭包，因此文档不声称“所有依赖源码已完整锁定”。

续训保证为epoch边界恢复：model、optimizer、RNG、best状态与已完成epoch event均保存。学习率由确定性函数计算，没有独立scheduler state；cap累计器不跨未完成epoch恢复。正式成功后续训权重自动删除。

当前与后续每组正式结果只保存：

```text
checkpoints/best_miou.pth.tar
checkpoints/best_pd.pth.tar
```

部署导出属于后续步骤。导出时必须验证：

```text
训练模型eval输出 == 导出推理模型输出
target_survival.* keys从导出state中移除
TPD/NER/QFG state完整保留
strict load通过
```

---

## 10. 执行顺序

```text
Phase 0（已完成）
封存IRSTD/NUDT V2 summary、checkpoint、日志、协议与三个源文件hash

Phase 1（立即执行）
使用完全相同V2代码在GPU2训练SIRST3
使用完全相同V2代码在GPU3训练NUAA-SIRST
固定seed42、1000 epochs、每10 epochs评估
仅保存best_miou和best_pd

Phase 2
生成三来源数据集主聚合 + SIRST3 pooled独立报告
分别生成V2−Original与V2−V1 Final比较
执行ACCEPTED / PARETO_MIXED / REJECTED裁决

Phase 3
若V2接受或可接受混合：冻结完整结构与V2候选配方
若V2拒绝：停止推广V2，但仍不立即修改结构

Phase 4
构建official-train-only四角色manifest与隔离审计
从头配对训练Original与Final
继续使用best_miou@0.5和best_pd@0.5

Phase 5
checkpoint冻结后才访问calibration
最终输出raw 0.5、calibration-frozen阈值及描述性阈值–组件工作包络
```

---

## 11. 论文结果表

### 表1：历史兼容best_miou@0.5

| Regime | Method | Epoch | mIoU | nIoU | F1 | Pd | Fa | tiny-Pd | False obj./image |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

### 表2：历史兼容best_pd@0.5

| Regime | Method | Epoch | mIoU | nIoU | F1 | Pd | Fa | tiny-Pd | False obj./image |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

### 表3：V2训练诊断

| Regime | Effective weight mean | Effective TSS/seg ratio | Cap-active sample fraction | Seg loss | TSS loss |
|---|---:|---:|---:|---:|---:|

### 表4：描述性Pd@Fa

| Regime | Method/checkpoint | ≤5e-7 | ≤1e-6 | ≤5e-6 | ≤1e-5 | ≤5e-5 | ≤1e-4 |
|---|---|---:|---:|---:|---:|---:|---:|

SIRST3 pooled与三个来源制度分开呈现，不用于重复扩充独立样本数。

---

## 12. 最终研究判断

当前已经建立：

```text
完整模型结构代码实现完成并冻结
V2在IRSTD/NUDT形成正向但混合的性能工作点
best_miou下两组Pd与nIoU超过Original
best_pd下V2具有明显综合改善
```

当前尚未建立：

```text
V2四训练制度通用配方
所有指标全面超过Original
随机性稳定性
统计显著性
论文核心结论
```

下一步唯一训练任务是：

> **保持现有V2源码和选模规则不变，在GPU2补跑SIRST3、GPU3补跑NUAA-SIRST；两组完成后再依据三来源聚合与SIRST3 pooled独立结果冻结或否决V2。**
