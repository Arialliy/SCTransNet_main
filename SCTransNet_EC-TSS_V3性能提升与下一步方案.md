# SCTransNet TSS-on 性能提升方案：EC-TSS V3.1 与下一步实验计划

> 项目：单帧红外小目标检测  
> 冻结推理结构：`SCTransNet + TPD8-MPRS-DCH + 五节点 NER4 Tail-Aware + QFG2-CROA`  
> 当前训练期辅助：Target Survival Supervision（TSS）  
> 当前统一工程默认：未建立（`null`）  
> 当前低 Fa 诊断参考：TSS-off（不是已通过裁决的统一默认）  
> 当前研究倾向：保留 TSS 创新，但不直接沿用现有固定开启方式  
> 推荐新候选：**EC-TSS V3.1 — Error-Conditioned Bidirectional Target Survival Supervision**  
> 中文：**误差条件化双向目标存活监督**  
> 推理额外开销：按设计为 0，待推理等价测试确认  
> 新增推理参数：按设计为 0，待 state-key/参数统计测试确认  
> 固定起始请求权重：`0.005`（只继承数值，不声称与旧 loss 等强度）  
> 动态标量损失比例上限：`0.10`  
> 正式数据集：NUAA-SIRST、NUDT-SIRST、IRSTD-1K  
> 正式随机种子：42  
> 固定阈值：0.5

---

# 0. 执行摘要

## 0.1 当前开启/关闭 TSS 的准确结论

现有 TSS-on 与 TSS-off 没有全面胜者：

```text
positive λ severe violations: 0.0025/0.005/0.01 = 8/5/8
TSS-off severe violations: 5
global_tss_lambda = null
tss_default_enabled = null
causal_confirmation = false
tss_effectiveness_confirmed = null
tss_harm_confirmed = null
```

| 场景 | 当前更合适的配方 |
|---|---|
| NUAA 综合性能 | 相对 `λ=0.005`，TSS-off 更有利；不构成统一默认 |
| NUDT 的 Pd / mIoU | 相对 TSS-off，TSS-on `λ=0.005` 更有利 |
| IRSTD 的 Pd / mIoU | 相对 TSS-off，TSS-on `λ=0.005` 更有利 |
| 强调低 Fa | 相对 `λ=0.005`，TSS-off 在 6 个 dataset-role 中有 5 个更低；NUDT best_pd 例外 |
| 三数据集统一配置 | 尚无明确胜者 |

与正 TSS 中严重退化项最少的描述性锚点 `λ=0.005` 比较：

```text
关闭更好：13 项
开启更好：13 项
相同：4 项
```

因此，不能把现有 `λ=0.005` 直接写成统一最终配方；也不能因为 TSS-off 在低 Fa 上更稳，就断言 TSS 创新无效。

最合理的项目策略是双轨制：

```text
当前统一工程默认：
null

低 Fa 诊断参考/被迫二选一时的临时回退候选：
TSS-off

下一研究候选：
保留 TSS heads，但重写 TSS objective，
使其处理“低于 0.5 的目标响应”和“高于 0.5 的背景响应”
```

## 0.2 为什么不能仅为“创新性”强行保留现有 TSS

创新性必须来自一个可解释、可验证的机制，而不是来自“训练时多开两个 head”。

当前正 TSS 的主要问题不是没有任何效果。下面的“作用范围过宽、域敏感和梯度冲突”是由现有结果提出、等待新训练验证的工作假设，不是已经完成的因果结论：

```text
所有正 target cells 都持续增强
所有负 cells 使用同一 BCE 机制
emb1 / emb2 使用同一 target
两个 endpoint loss 等权相加
全数据集使用一个标量 ratio cap
```

它可能同时产生两种效果：

1. 对 NUDT / IRSTD：可能增强弱目标存活并改善 Pd 或区域质量；
2. 对 NUAA / 低 Fa 区域：可能对已经正确的目标继续施压，并干扰背景抑制或区域边界。

所以，下一步不应继续搜索更多全局 λ，而应修改：

> **TSS 在哪些 cell、什么时刻、以何种方向产生监督。**

---

# 1. 当前代码中的关键瓶颈

## 1.1 当前 TSS head

当前代码在两个 stride-16 endpoint 上使用独立的 `1×1 Conv`：

```text
emb1 endpoint → 1×1 Conv → survival logit
emb2 endpoint → 1×1 Conv → survival logit
```

两路 endpoint：

```text
共享同一个空间网格
使用同一个 Y16 target
不修改 segmentation path
推理阶段可完全移除
```

因此 TSS 本身非常适合继续作为训练期创新点；不需要增加新的推理模块。

## 1.2 当前 target

当前 survival target 为：

\[
Y_{16}
=
\operatorname{MaxPool}_{16}(Y)
\]

即只要一个 16×16 cell 中存在任意目标像素，该 cell 就被标为 1。

它表达的是：

```text
该 stride-16 cell 中是否存在目标
```

而不是：

```text
该目标是否在最终 segmentation 中仍有风险
该背景 cell 是否已经产生高置信虚警
```

## 1.3 当前 loss

当前两路 survival loss 为：

\[
L_{\mathrm{tss}}
=
\sum_{i=1}^{2}
BCEWithLogits(Z_i,Y_{16};pos\_weight)
\]

然后：

\[
\lambda_{\mathrm{eff}}
=
\min
\left[
\lambda_{\mathrm{requested}},
0.10
\frac{
\operatorname{sg}(L_{\mathrm{seg}})
}{
\max(\operatorname{sg}(L_{\mathrm{tss}}),\epsilon)
}
\right]
\]

\[
L
=
L_{\mathrm{seg}}
+
\lambda_{\mathrm{eff}}L_{\mathrm{tss}}
\]

当前设计已经解决了“辅助标量 loss 过大”的问题，但没有解决：

```text
辅助梯度是否作用在真正需要救援的位置
辅助监督是否在已经正确的目标上重复增强
辅助监督是否专门压制高置信 hard negatives
emb1 / emb2 是否对主任务产生不同方向的梯度
```

## 1.4 数据集级 pos_weight 可能增加域敏感性

当前 TSS 使用一个由训练集统计得到的 `survival_pos_weight`。

不同数据集的：

```text
目标密度
正 cell 数
背景 cell 数
tiny-target 比例
```

不同，因此相同 `λ` 和相同 10% cap 仍可能形成不同的 cell-level 梯度结构。

这可能与下列现象有关，但当前正式结果的 `causal_confirmation=false`，必须作为待验证假设表述：

```text
同一 λ=0.005
在 NUDT / IRSTD 更有利
在 NUAA 和低 Fa 条件下更不稳定
```

---

# 2. 推荐修改：EC-TSS V3.1

## 2.1 名称

> **EC-TSS V3.1**  
> Error-Conditioned Bidirectional Target Survival Supervision  
> 误差条件化双向目标存活监督

## 2.2 核心思想

当前 TSS 问的是：

> 这个 cell 有没有目标？

EC-TSS 改成：

> 如果有目标，最终 segmentation 是否还没有把它可靠检出？  
> 如果没有目标，最终 segmentation 是否已经在这里产生高置信虚警？

因此，TSS 只在两类阈值错误位置工作：

```text
正向救援：
GT target cell，但最终 segmentation 置信度低

反向抑制：
GT background cell，但最终 segmentation 置信度高
```

它的设计目标是兼顾：

```text
NUDT / IRSTD 的 Pd、mIoU
+
NUAA / 低 Fa 的背景抑制
```

## 2.3 不变的部分

```text
TPD8 不变
NER4 不变
QFG2 不变
TSS heads 不变
TSS state keys 不变
推理模型不变
Y16 网格不变
lambda_requested 首轮固定 0.005（只作为描述性锚点）
ratio cap 固定 0.10
confidence threshold 固定 0.5
target-neighborhood radius 固定 3（与目标匹配半径一致，不搜索）
```

唯一核心变化：

```text
TSS loss 从“全 cell 均匀 BCE”
变成“由最终 segmentation 误差条件化的双向类平衡 loss”
```

---

# 3. EC-TSS 数学定义

## 3.1 目标邻域与背景置信度

设最终 segmentation probability 为：

\[
P_f\in[0,1]^{B\times1\times H\times W}
\]

先使用训练标签构造固定目标邻域：

\[
M
=
\operatorname{Dilate}(Y,r=3)
\]

其中 `r=3` 直接绑定正式目标匹配半径，不作为可搜索超参数。随后分别构造目标邻域和纯背景的 stride-16 置信度：

\[
Q^+_{16}
=
\operatorname{stopgrad}
\left[
\operatorname{MaxPool}_{16}(P_f\odot M)
\right]
\]

\[
Q^-_{16}
=
\operatorname{stopgrad}
\left[
\operatorname{MaxPool}_{16}(P_f\odot(1-M))
\right]
\]

必须拆分两者。若直接对整个正目标 cell 使用 `MaxPool16(P_f)`，目标之外的一个高响应可能掩盖真实目标漏检，使正向救援错误退出。stop-gradient 只阻止当前辅助 loss 沿风险图回传到 `P_f`，不代表未来迭代的风险图不随模型变化。

## 3.2 正目标救援权重

\[
R^+
=
Y_{16}
\operatorname{clamp}
\left(
\frac{0.5-Q^+_{16}}{0.5},0,1
\right)
\]

解释：

| 情况 | \(R^+\) |
|---|---:|
| GT target cell，目标邻域响应远低于 0.5 | 接近 1，强救援 |
| GT target cell，目标邻域响应达到或超过 0.5 | 0，退出正分支 |
| background cell | 0 |

## 3.3 hard-negative 抑制权重

\[
R^-
=
(1-Y_{16})
\operatorname{clamp}
\left(
\frac{Q^-_{16}-0.5}{0.5},0,1
\right)
\]

解释：

| 情况 | \(R^-\) |
|---|---:|
| background cell，纯背景响应远高于 0.5 | 接近 1，重点抑制 |
| background cell，纯背景响应达到或低于 0.5 | 0，退出负分支 |
| target cell | 0 |

## 3.4 每个 endpoint 的正负 loss

对第 \(i\) 个 TSS logit \(Z_i\)：

正类项：

\[
L_i^+
=
\frac{
\sum
R^+
\operatorname{softplus}(-Z_i)
}{
\max(\sum R^+,1)
}
\]

负类项：

\[
L_i^-
=
\frac{
\sum
R^-
\operatorname{softplus}(Z_i)
}{
\max(\sum R^-,1)
}
\]

其中：

```text
softplus(-Z) = positive BCE term
softplus(Z)  = negative BCE term
```

正负类分别按风险质量归一化，不再依赖数据集级 `pos_weight`。加入 `max(·,1)` 有两个作用：

1. 额外加入风险为零的简单 cell 不会稀释已有 hard-example；
2. 当某一分支的总风险质量低于 1 时，该分支会随风险质量继续衰减，并在风险为零时严格退出。

实现必须同时记录 `positive_risk_mass`、`negative_risk_mass` 和两个分支的激活 cell 数。风险质量归一化保证的是“不被零风险 cell 稀释”，并不保证每个数据集上的 EC-TSS loss 单调下降。

## 3.5 两个 endpoint 聚合

\[
L_{\mathrm{EC-TSS}}
=
\frac{1}{2}
\sum_{i=1}^{2}
\frac{
L_i^+ + L_i^-
}{2}
\]

初版继续保持两个 endpoint 等权，避免一次加入过多变量。

## 3.6 保留现有动态 cap

\[
\lambda_{\mathrm{eff}}
=
\min
\left[
0.005,
0.10
\frac{
\operatorname{sg}(L_{\mathrm{seg}})
}{
\max(
\operatorname{sg}(L_{\mathrm{EC-TSS}}),
\epsilon
)
}
\right]
\]

\[
L
=
L_{\mathrm{seg}}
+
\lambda_{\mathrm{eff}}L_{\mathrm{EC-TSS}}
\]

这样：

- 沿用 `0.005` 作为固定起始系数，但不声称新旧 objective 尺度等价；
- 不再进行全局 λ 搜索；
- TSS 只对阈值 0.5 下的目标低响应或背景高响应 cell 产生监督；
- 标量辅助 loss 贡献仍受 10% 上限限制，但该 cap 不是辅助梯度范数上限。

正式训练前只允许在冻结的训练 batch 上做尺度审计，记录：

```text
weighted_tss / segmentation_loss
EC-TSS / segmentation shared-gradient norm ratio
cap-active fraction
positive / negative risk mass
```

该审计只用于确认 loss 没有数值失效，不允许依据 test 指标搜索新的 λ。

---

# 4. EC-TSS 的性能改善假设

本节只定义待实验检验的性能方向，不构成因果确认或结果承诺。EC-TSS 是 stride-16 presence 辅助，对 Pd/Fa 有直接任务对应关系，但对像素边界和 mIoU 只有间接影响。

## 4.1 对 NUAA

当前 TSS-on 在 NUAA 的主要问题是：

```text
Pd / tiny-Pd 退化
或综合工作点不稳定
```

EC-TSS 的设计目标是不再对阈值下已经正确的 target cells 继续施加辅助压力。

当最终 segmentation 已经正确时：

\[
R^+\approx0
\]

正分支按定义退出；是否因此减少主分割扰动必须由训练结果验证。

对高置信背景 false cells：

\[
R^-\approx1
\]

负分支产生背景抑制辅助。

预期方向：

```text
减少当前 TSS 对 NUAA 主任务的过度干预
降低虚警
保持或恢复 Pd / tiny-Pd
```

## 4.2 对 NUDT

当前 `λ=0.005` 在 NUDT 的 Pd / mIoU 有优势。

EC-TSS 预期保留：

```text
最终 segmentation 仍弱的目标 cell
→ 强 positive rescue
```

是否保持 NUDT 的目标检出收益由正式 Gate 裁决。

同时，对于高置信背景组件：

```text
negative rescue branch
→ 提供显式抑制
```

目标是避免用 TSS-off 换低 Fa 时损失 NUDT 的区域质量。

## 4.3 对 IRSTD

IRSTD 中 TSS-on 的 Pd/mIoU 更好，而 TSS-off 的 Fa 更低。

这是 EC-TSS 需要重点检验的场景：

```text
positive branch：
保留弱目标救援

negative branch：
只处理最终 segmentation 的 hard negatives
```

目标不是简单降低 TSS 强度，而是将 TSS 分成：

```text
该救援的目标
+
该抑制的背景
```

## 4.4 对训练后期

当前 TSS 在 1000 epochs 全程使用。

EC-TSS 中，当最终 segmentation 已经稳定时：

```text
正确 target cell：R+ 下降
正确 background cell：R- 下降
```

因此风险质量具有自然衰减的条件；实际 loss 是否在训练后期下降必须记录验证，不能由公式提前保证。首轮不额外引入：

```text
手工关闭 epoch
线性 decay
cosine TSS schedule
```

这是一项待验证的数据自适应训练假设。

---

# 5. 研究创新性判断

## 5.1 创新点不是“开启 TSS”

仅开启现有 TSS：

```text
不能构成充分创新
```

因为现有机制只是：

```text
两个 1×1 presence heads
+ MaxPool16 target
+ BCEWithLogits
```

## 5.2 EC-TSS 的潜在创新点

可以将贡献表述为：

> 提出一种训练期误差条件化目标存活监督。该方法利用冻结梯度的目标邻域与纯背景 segmentation cell confidence，在浅层目标存活 endpoint 上动态区分目标救援与 hard-background 抑制，仅对固定阈值 0.5 下仍存在方向性错误的位置施加辅助约束。

核心性质：

```text
只在训练期使用
不增加推理参数
不增加推理 FLOPs
不改变 TPD/NER/QFG 主线
同时处理漏检风险和虚警风险
从“全局 loss 强度”转向“cell-level 误差选择”
```

## 5.3 创新声明边界

在正式结果通过前，只能写：

```text
novel_training_objective_proposed=true
innovation_effectiveness_established=false
```

不能提前写：

```text
EC-TSS 全面优于 TSS-off
EC-TSS 已解决所有跨数据集冲突
```

---

# 6. 两轨项目策略与冻结状态

## 6.1 当前状态

在 EC-TSS 完成前：

```text
current_unified_operational_default = null
seed42_operational_recipe_admissible = false
low_fa_diagnostic_reference = TSS-off
current_positive_anchor = 0.005
anchor_is_selected_candidate = false
positive_tss_search_closed = true
```

原因：

- TSS-off 自身仍有 5 项严重退化，未通过全局裁决；
- `λ=0.005` 同样有 5 项严重退化，只是正 λ 中的描述性锚点；
- 现有开启与关闭均没有成为三数据集统一配方。

## 6.2 论文研究候选

```text
research_candidate = EC-TSS V3.1
research_branch_status = newly_authorized_candidate
```

EC-TSS 是在旧结论 `positive_tss_search_closed=true` 之后新授权的 objective 研究分支。它不改写历史 selector，也不改变冻结的 TPD8–NER4–QFG2 推理主线。只有通过本文件的性能 Gate 后，才可以写：

```text
seed42_test_selected_operational_candidate = EC_TSS_V3_1
```

仍不能直接写成跨随机性或全局统一默认。

创新性不能成为绕过性能门槛的理由。

---

# 7. Phase 0：冻结历史输入与低成本诊断

历史目录只保存 `best_miou` 和 `best_pd` 两个 selected checkpoint，没有每轮权重。因此 Phase 0 只能对 `λ=0.005` 的三个数据集 × 两个 checkpoint 角色共 6 个 checkpoint 重算 cell 诊断；不能声称拥有逐 epoch 的 cell 或梯度证据。逐 epoch 的 EC-TSS 诊断从新训练第 1 epoch 开始记录。

Phase 0 与代码实现可以顺序衔接，但不应以大规模机制分析拖延模型实现。必须冻结：

```text
split = 各数据集既有 img_idx/test
checkpoint roles = best_miou, best_pd
checkpoint SHA256 = 历史正式产物记录
inference threshold = 0.5
match radius = 3
diagnostic augmentation = none
sample order = img_idx/test 原顺序
target/matched/missed/false-component 规则 = 正式 evaluator 规则
```

## 7.1 正负 cell 贡献分解

对当前 `λ=0.005` 重新计算：

```text
positive TSS loss
negative TSS loss
emb1 positive loss
emb1 negative loss
emb2 positive loss
emb2 negative loss
```

按数据集和 checkpoint 角色输出；历史逐 epoch 只能读取已有 scalar 日志，不能重算上述分解。

目的：

- NUAA 是否 positive rescue 过强；
- IRSTD 的 Fa 是否主要来自 negative branch 约束不足；
- emb1 / emb2 是否存在明显差异。

## 7.2 cell score 分布

记录：

```text
GT positive cell 的 Q+16 分布
GT negative cell 的 Q-16 分布
false component 所在 cell 的 Q-16
missed target 所在 cell 的 Q+16
tiny target 所在 cell 的 Q+16
```

必须比较以下分布，而不是用阈值定义本身循环证明风险方向：

```text
missed target 与 matched target 的 Q+ 分布及效应方向
false-component cell 与普通 negative cell 的 Q- 分布及效应方向
```

若方向完全相反则停止实现；若方向正确但分离度有限，仍可进入 pilot，由最终 Pd/Fa/mIoU 裁决。

## 7.3 endpoint 梯度冲突

对固定 batch 计算：

\[
g_{\mathrm{seg},i}
=
\nabla_{E_i}L_{\mathrm{seg}}
\]

\[
g_{\mathrm{tss},i}
=
\nabla_{E_i}L_{\mathrm{tss},i}
\]

\[
c_i
=
\frac{
g_{\mathrm{seg},i}\cdot g_{\mathrm{tss},i}
}{
\|g_{\mathrm{seg},i}\|
\|g_{\mathrm{tss},i}\|+\epsilon
}
\]

记录：

```text
emb1 cosine mean / p10 / negative-rate
emb2 cosine mean / p10 / negative-rate
```

用途：

- 若 emb1 冲突明显而 emb2 协同，下一阶段才考虑 endpoint gate；
- 不应在无证据时直接设置 `emb1_weight < emb2_weight`。

梯度冲突是失败分析记录，不阻塞首轮 loss、测试与 smoke，也不作为 EC-TSS V3.1 的新增模块。

---

# 8. 下一步 Phase 1：代码实现

## 8.1 不修改历史 loss

保留：

```text
experiments/tpd_training_loss.py
```

用于重建：

```text
TSS-off
λ=0.0025
λ=0.005
λ=0.01
```

新增：

```text
experiments/tpd_training_loss_ec_tss_v3_1.py
```

## 8.2 推荐接口

```python
@dataclass(frozen=True, slots=True)
class ECTSSV31TrainingLoss:
    total: torch.Tensor
    segmentation: torch.Tensor
    survival: torch.Tensor

    positive_survival: torch.Tensor
    negative_survival: torch.Tensor

    endpoint_positive_terms: tuple[torch.Tensor, ...]
    endpoint_negative_terms: tuple[torch.Tensor, ...]

    effective_survival_weight: torch.Tensor
    weighted_survival: torch.Tensor

    positive_risk_mass: torch.Tensor
    negative_risk_mass: torch.Tensor
    positive_active_cells: torch.Tensor
    negative_active_cells: torch.Tensor
```

主入口：

```python
compute_ec_tss_v3_1_training_loss(
    output,
    segmentation_target,
    segmentation_criterion,
    survival_weight=0.005,
    survival_ratio_cap=0.10,
    confidence_threshold=0.5,
    target_dilation_radius=3,
)
```

EC-TSS loss 本体不接收：

```text
survival_pos_weight
```

因为正负类在 loss 内部分别归一化。底层旧 engine 仍会传入该关键字，因此只能由新 runner adapter 接收、验证并明确忽略，不能直接替换旧 loss 调用。

## 8.3 参考核心代码

```python
def _final_segmentation_probability(output):
    return evaluator_prediction(output)


def _risk_normalized_sum(
    weighted_term: torch.Tensor,
    risk: torch.Tensor,
) -> torch.Tensor:
    denominator = risk.sum().clamp_min(1.0)
    return weighted_term.sum() / denominator


def compute_error_conditioned_terms(
    logit: torch.Tensor,
    target16: torch.Tensor,
    target_probability16: torch.Tensor,
    background_probability16: torch.Tensor,
    threshold: float = 0.5,
):
    positive_risk = target16 * torch.clamp(
        (threshold - target_probability16) / threshold, 0.0, 1.0
    )
    negative_risk = (1.0 - target16) * torch.clamp(
        (background_probability16 - threshold) / (1.0 - threshold),
        0.0,
        1.0,
    )

    positive_element = F.softplus(-logit.float())
    negative_element = F.softplus(logit.float())

    positive_loss = _risk_normalized_sum(
        positive_risk * positive_element,
        positive_risk,
    )
    negative_loss = _risk_normalized_sum(
        negative_risk * negative_element,
        negative_risk,
    )
    return positive_loss, negative_loss
```

构造风险图：

```python
with torch.no_grad():
    target_neighborhood = F.max_pool2d(
        segmentation_target.detach().float(),
        kernel_size=7,
        stride=1,
        padding=3,
    )
    target_probability16 = F.max_pool2d(
        final_probability.detach().float() * target_neighborhood,
        kernel_size=16,
        stride=16,
    )
    background_probability16 = F.max_pool2d(
        final_probability.detach().float() * (1.0 - target_neighborhood),
        kernel_size=16,
        stride=16,
    )
```

最终：

```python
survival_loss = 0.25 * (
    emb1_positive
    + emb1_negative
    + emb2_positive
    + emb2_negative
)
```

然后复用当前 10% cap 语义。

## 8.4 模型 state 不变，训练 schema 必须隔离

EC-TSS 不新增 parameter 或 buffer：

```text
TSS head state keys 不变
模型参数量不变
推理模型 state schema 不变
训练模型 state_dict keys 不变
```

但必须新增独立的训练产物 schema，例如：

```text
sctransnet_three_dataset_ec_tss_v3_1_seed42/v1
```

新 schema 记录：

```text
loss diagnostics
recipe metadata
source lock
objective_id=ec_tss_v3_1
confidence_threshold=0.5
target_dilation_radius=3
positive_normalization=risk_mass_clamp_min_1
negative_normalization=risk_mass_clamp_min_1
```

旧 TSS/TSS-off checkpoint 不得作为 EC-TSS exact-resume 起点；所有 EC-TSS run 从相同 seed 42 初始化规则独立训练。

---

# 9. 下一步 Phase 2：测试

## 9.1 数学方向测试

```text
target cell + target-neighborhood probability=0
→ positive risk=1

target cell + target-neighborhood probability>=0.5
→ positive risk=0

background cell + pure-background probability<=0.5
→ negative risk=0

background cell + pure-background probability=1
→ negative risk=1

target cell + target位置漏检但cell内远处背景高响应
→ positive risk 仍大于0，不能被远处响应掩盖
```

## 9.2 stop-gradient 测试

必须验证：

```text
risk map 不向 final segmentation probability 传播梯度
TSS logits 正常有梯度
emb1 / emb2 endpoint 正常收到 TSS 梯度
```

测试必须隔离 survival loss 并验证：

```text
autograd.grad(survival_loss, final_probability, allow_unused=True)
返回 None 或全0
```

仅检查 `requires_grad=false` 不足以证明完整反向路径正确。

## 9.3 类平衡测试

构造：

```text
相同 hard-negative cell
但额外增加大量 easy-negative cells
```

EC-TSS 的 hard-negative 项不应因 easy negatives 数量增加而被显著稀释。

## 9.4 无目标 crop

当 batch 中没有 target cell：

```text
positive loss = 0
negative loss 仍可工作
total finite
```

## 9.5 全正确状态

当：

```text
target cells Q+16>=0.5
background cells Q-16<=0.5
```

应满足：

```text
EC-TSS risk ≈ 0
EC-TSS loss ≈ 0
```

体现阈值正确状态下的严格退出。训练后期 loss 是否单调下降只记录，不作为公式必然性质。

## 9.6 cap 测试

```text
cap inactive
→ λ_eff = 0.005

cap active
→ weighted_survival <= 0.10 * segmentation

effective weight 无 grad_fn
```

同时检查 cap 只约束标量辅助 loss 比例，不把它描述为梯度范数上限。

## 9.7 历史路径隔离

```text
旧 tpd_training_loss.py 输出完全不变
旧 checkpoint 可复现
旧 selector 结论不可被 EC-TSS 改写
```

## 9.8 推理等价

```text
EC-TSS 训练模型 eval segmentation output
==
移除 TSS heads 的冻结推理模型 output
```

---

# 10. 下一步 Phase 3：工程 screen

正式 1000 epochs 前先完成：

## 10.1 CPU 单元测试

```text
普通 Python
python -O
```

## 10.2 RTX 5090 smoke

每张 GPU 至少覆盖：

```text
forward
loss
backward
optimizer step
checkpoint save
strict reload
```

## 10.3 Exact resume

连续训练与中断续训必须在以下状态一致：

```text
model
optimizer
scheduler
RNG
DataLoader generator
best_miou selector
best_pd selector
EC-TSS diagnostics
```

## 10.4 200-epoch 可续跑 pilot

建议只新增 3 个 pilot：

| 数据集 | 候选 | Epochs |
|---|---|---:|
| NUAA | EC-TSS V3.1 | formal1000 的 epoch 1–200 |
| NUDT | EC-TSS V3.1 | formal1000 的 epoch 1–200 |
| IRSTD | EC-TSS V3.1 | formal1000 的 epoch 1–200 |

三个 pilot 不是额外 run，而是三个 formal1000 run 的前缀。必须使用：

```text
planned_total_epochs=1000
pause_after_epoch=200
```

不得用 `epochs=200`，否则余弦学习率轨迹与 formal1000 不同。pilot 通过后，从 epoch 200 的 rolling state 以 `resume=required` 续到 epoch 1000，不重启、不更换目录、不改变 protocol SHA。

已有 TSS-off 与 `λ=0.005` 的前 200 epoch 轨迹只作为描述性控制；下列共同部分必须一致：

```text
初始 state
数据协议
模型与数据代码 hash
数据 manifest
初始化规则
optimizer / 1000-epoch scheduler
epoch 派生 minibatch 顺序
```

EC-TSS loss、runner 和 schema 的 hash 必然不同，不得写成“全部代码 hash 一致”。

pilot 只用于排除：

```text
loss 崩溃
TSS loss 长期为 0
Fa 明显爆炸
Pd 明显坍塌
risk map 不工作
```

pilot 使用同一个 test split 决定是否续跑，属于 test-informed/optimistic 开发决策，不用于形成最终论文结论。

冻结预算披露：

```text
当前 formal1000 run = 15
新增 EC-TSS run = 3（先投入 600 epochs，通过后补足至总计 3000 epochs）
完成后 formal1000 run = 18
Final-family : Original = 15 : 3 = 5 : 1
pilot 通过后不重新起 run
```

---

# 11. 下一步 Phase 4：formal1000

pilot 通过后，从相同 run 的 epoch 201 exact-resume：

| 数据集 | 模型 | Seed | Epochs |
|---|---|---:|---:|
| NUAA-SIRST | Final + EC-TSS V3.1 | 42 | 1000（续跑） |
| NUDT-SIRST | Final + EC-TSS V3.1 | 42 | 1000（续跑） |
| IRSTD-1K | Final + EC-TSS V3.1 | 42 | 1000（续跑） |

固定：

```text
每10 epochs评估
best_miou / best_pd
threshold=0.5
同一 img_idx
同一 evaluator
同一 Misc_111 修正
```

四卡执行布局：

```text
GPU0 = NUAA-SIRST 单卡训练
GPU1 = NUDT-SIRST 单卡训练
GPU2 = IRSTD-1K 单卡训练
GPU3 = smoke、固定训练批次尺度/风险诊断、随后 checkpoint 评估
```

不采用 DDP，不重复 run，不改变全局 batch 或 optimizer 轨迹。

比较：

```text
Original
TSS-off
TSS-on λ=0.0025
TSS-on λ=0.005（描述性锚点）
TSS-on λ=0.01
EC-TSS V3.1
```

历史五配方只读复用，不重新训练。固定阈值 0.5 的五指标与目标计数参与 Gate；Pd–Fa sweep 只做描述性报告。不得重新搜索 EC-TSS 的 λ、dilation radius、endpoint weight 或阈值。

---

# 12. EC-TSS V3.1 正式裁决 Gate

## Gate V3-A：工程

```text
3 个 formal1000 完整
checkpoint 完整
exact resume 通过
source lock 完整
普通模式与 python -O 通过
推理导出等价
```

## Gate V3-B：相对 Original 的退化底线门

对三个数据集、两个 checkpoint 角色逐项执行冻结规则：

```text
matched_target_count 比 Original 少至少 2：1 项严重退化
matched_tiny_target_count 比 Original 少至少 2：1 项严重退化
q(mIoU) 或 q(nIoU) 比 Original 低至少 50 quanta：1 项严重退化
  其中 q(x)=floor(x/0.0001+0.5)，50 quanta=0.005
unmatched_predicted_pixels：
  Original=0 且 EC>0，或 EC>125%×Original，
  且 matched_target_count 墖益不足 2：1 项严重退化
```

阶段推进要求：

```text
severe_degradation_violation_count < 5
```

即必须严格少于当前 TSS-off 和 `λ=0.005` 各自的 5 项；正式 seed42 配方通过目标仍为：

```text
severe_degradation_violation_count == 0
```

两级都要求：

```text
不存在任何数据集，
Original 在 best_miou 和 best_pd 两个角色上
均严格支配 EC-TSS
```

## Gate V3-C：保留 `λ=0.005` 描述性锚点的强项

在 `best_miou` 和 `best_pd` 两个角色上，相对当前 `λ=0.005`：

```text
NUDT：
Pd / mIoU / nIoU 的已有优势不得发生严重退化

IRSTD：
Pd / mIoU 的已有优势不得发生严重退化
```

Pd/tiny-Pd 使用原始匹配目标数；mIoU/nIoU 使用上述 `q(x)`。严重退化容差沿用 Gate V3-B，不另设测试后阈值。

## Gate V3-D：性能优先的成对比较

在每个 dataset-role 内按以下五维向量比较：

```text
matched_target_count ↑
matched_tiny_target_count ↑（可用时）
q(mIoU) ↑
q(nIoU) ↑
unmatched_predicted_pixels ↓
```

分别报告 EC-TSS 相对 Original、TSS-off 和 `λ=0.005` 的 `better/equal/worse` 总数。阶段推进要求 EC 相对 TSS-off 和 `λ=0.005` 至少各自满足 `better > worse`；若未满足，不以机制诊断替代性能结论。

原文档中的“NUAA Pd/tiny-Pd 与 IRSTD Fa 弱项”不再作为独立重复 Gate；它们已经由 Gate V3-B 和本 Gate 的失败类型表完整覆盖。

## Gate V3-E：六配方联合 Pareto 价值

每个 dataset-role 使用 Gate V3-D 的五维方向和量化，比较总体固定为：

```text
Original, TSS-off, λ=0.0025, λ=0.005, λ=0.01, EC-TSS V3.1
```

“EC-TSS 独有非支配点”定义为：没有其他配方在全部可用维度不差且至少一维更好，并且没有其他配方具有完全相同的量化向量。至少在两个 dataset-role 单元形成：

```text
EC-TSS 独有非支配点
```

且不能在任一数据集的两个角色上都被 Original、TSS-off 或 `λ=0.005` 中任一配方严格支配。Pd–Fa sweep 和 Pareto 阈值点数量只做描述性报告，不参与该 Gate。

## Gate V3-F：训练诊断完整性（不覆盖性能 Gate）

```text
每 epoch 记录两分支 risk mass、active cells、loss 和 cap-active fraction
至少在前 20 epochs 内两分支均出现过非零 risk mass
固定诊断 checkpoint 上：missed target 的 Q+ 中位数不高于 matched target
固定诊断 checkpoint 上：false-component 的 Q- 中位数不低于普通 negative
后期窗口固定为 epochs 801–1000，报告相对 epochs 1–200 的 risk mass/loss 变化
```

后期下降只作为诊断结果，不单独否决一个已经在 Pd/Fa/mIoU/nIoU 上通过的模型。Gate V3-A 至 V3-E 全部通过后：

```text
decision=EC_TSS_V3_1_SEED42_TEST_SELECTED_PASS
seed42_test_selected_operational_candidate=EC_TSS_V3_1
seed42_operational_recipe_admissible=true
global_operational_default=null
tss_training_innovation_supported=true
```

仍保持：

```text
paper_core_established=false
stability_claim_supported=false
training_recipe_finalized=false
```

因为当前仍是 seed 42、test-selected 协议。

---

# 13. 如果 EC-TSS 仍失败

不能立即增加更多模块。

## 13.1 先看 endpoint 梯度冲突

如果：

```text
emb1 negative cosine rate 明显高
emb2 主要正向
```

下一候选为：

> **Endpoint-Agreement EC-TSS**

只对与 segmentation gradient 不冲突的 endpoint 使用 TSS。

例如：

\[
a_i
=
\operatorname{clamp}
\left(
\frac{1+\cos(g_{\mathrm{seg},i},g_{\mathrm{tss},i})}{2},
0,1
\right)
\]

\[
L_{\mathrm{tss}}
=
\sum_i a_i L_i
\]

该方向与 PCGrad / GradNorm 的多任务梯度冲突思想相关，但应作为诊断支持后的第二阶段，而不是与 EC-TSS 同时加入。

## 13.2 若 Fa 仍来自目标周围多 cell 响应

检查：

```text
每个 GT component 对应多少个 Y16 positive cells
跨 cell boundary 的 tiny-target 比例
positive cell 数与 false component 数的相关性
```

若证据支持，下一候选为：

> **Object-Mass-Preserving Survival Target**

使一个目标跨多个 stride-16 cell 时，总 target mass 保持为 1，避免一个小目标产生多个完整正 cell。

这会修改 TSS target，但仍不改变推理结构。

## 13.3 若 EC-TSS 与 TSS-off 几乎相同

说明：

```text
TSS utilization 太弱
或 endpoint gradient 未进入有效主路径
```

此时进行：

```text
head weight norm
endpoint gradient norm
QFG/NER bypass
TSS knockout
```

不要继续增加 loss 复杂度。

## 13.4 若 EC-TSS 在三个数据集仍方向不一致

结论：

```text
TSS domain sensitivity remains
```

应将 TSS 定位为：

```text
可选训练辅助
而不是最终统一配方
```

仍保持 `current_unified_operational_default=null`，按应用工作区在现有配方间选择；随后回到冻结的 NER→QFG→TPD 单组件诊断顺序，而不是把 TSS-off 改写为已通过的统一默认。

---

# 14. 代码实现状态

已实现并通过普通模式与 `python -O` 测试：

```text
experiments/EC_TSS_V3_1_PROTOCOL.md

experiments/tpd_training_loss_ec_tss_v3_1.py
experiments/train_three_dataset_ec_tss_v3_1_seed42.py
experiments/launch_three_dataset_ec_tss_v3_1_seed42.py
experiments/evaluate_three_dataset_ec_tss_v3_1.py
analysis/analyze_ec_tss_v3_1_fixed_batch_scale.py
```

已实现测试：

```text
tests/test_tpd_training_loss_ec_tss_v3_1.py
tests/test_train_three_dataset_ec_tss_v3_1_seed42.py
tests/test_evaluate_compare_ec_tss_v3_1.py
tests/test_analyze_ec_tss_v3_1_fixed_batch_scale.py
```

`launcher` 的测试合并在 `test_evaluate_compare_ec_tss_v3_1.py` 中。最终正式比较器
在六个 EC-TSS checkpoint 评估完成后实现；它不阻塞三数据集训练与评估。下列深入
诊断仍保持按需触发，不作为首轮训练的前置条件：

```text
experiments/compare_finalize_ec_tss_v3_1.py
analysis/analyze_tss_cell_risk.py
analysis/analyze_tss_endpoint_gradient_conflict.py
analysis/analyze_tss_positive_cell_multiplicity.py
```

固定训练首批的 GPU3 尺度审计已经完成。NUDT-SIRST 的正式
`seed42 / epoch1 / batch0` 得到：

```text
Lseg=5.091556
Lec=0.346574
weighted_ec_tss / Lseg=3.4034e-4
ratio_cap_active=false
positive_risk_mass=0
negative_risk_mass=1913.1122
shared_segmentation_gradient_l2=15.52394
shared_weighted_ec_gradient_l2=0
```

该批次没有更新参数、没有访问 test split，也没有用于搜索 λ。首批共享 EC 梯度为 0
与 survival heads 的全零初始化一致：第一步辅助项先更新 heads，后续才可能向共享
主干传递梯度。这个尺度结果不构成性能结论。

不要修改或覆盖：

```text
experiments/tpd_training_loss.py
现有正 λ selector
现有 TSS-off comparator
历史结果目录
```

---

# 15. 推荐项目状态

```text
decision=IMPLEMENT_EC_TSS_V3_1_THEN_RUN_RESUMABLE_PILOT

architecture_implemented=true
architecture_frozen=true
inference_architecture_changed=false
innovation_mainline_changed=false

current_unified_operational_default=null
seed42_operational_recipe_admissible=false
low_fa_diagnostic_reference=tss_off
current_positive_tss_default=null
current_positive_anchor=0.005
anchor_is_selected_candidate=false
positive_tss_search_closed=true

ec_tss_v3_1_selected_as_research_candidate=true
ec_tss_v3_1_code_implemented=true
ec_tss_v3_1_formal_training_authorized=true
ec_tss_v3_1_runtime_state_source=results/three_dataset_ec_tss_v3_1_seed42/launch/formal/supervisor_status.json

requested_tss_weight=0.005
survival_ratio_cap=0.10
confidence_threshold=0.5
target_dilation_radius=3
dataset_specific_lambda_forbidden=true

new_inference_parameters=0
new_inference_flops=0

paper_core_established=false
stability_claim_supported=false
training_recipe_finalized=false
```

---

# 16. 最终执行顺序

```text
1. 封存当前 Original / TSS-off / λ=0.005 结果
2. 冻结历史 6 个 selected checkpoint 的低成本诊断输入
3. 实现 EC-TSS V3.1 新 loss 文件
4. 增加数学、梯度、类平衡、cap、resume 测试
5. 完成 CPU 测试与 GPU3 固定批次尺度审计；GPU3 两轮 smoke 与三条正式前缀并行
6. 以 planned_total_epochs=1000 运行三个 run 的 epoch 1–200
7. pilot 只判断是否存在灾难性失败
8. 通过后从同一 rolling state exact-resume 至 epoch 1000
9. 使用原 best_miou / best_pd、threshold=0.5
10. 与 Original、TSS-off、λ=0.0025/0.005/0.01 同协议比较
11. 执行 Gate V3-A 至 V3-E；V3-F 作为诊断完整性报告
12. 通过：冻结 EC-TSS 为 seed42 test-selected 候选，不外推为全局默认
13. 未通过：根据梯度冲突或 target multiplicity 进入单变量下一阶段
14. 不允许因“创新性”绕过性能 Gate
```

---

# 17. 一句话结论

> **当前开启与关闭 TSS 都没有成为统一默认。EC-TSS V3.1 值得作为新授权训练目标实现验证：它在固定阈值 0.5 下，只救援目标邻域低响应 cell，并只抑制纯背景高响应 cell；按设计不改变 TPD8–NER4–QFG2 推理主线、参数量或推理开销。能否同时改善 Pd、Fa、mIoU 和 nIoU，完全由三数据集 seed42 的实际结果裁决。**

---

# 18. 代码依据与方法参考

## 当前仓库代码依据

- `experiments/tpd_training_loss.py`
  - 六路 segmentation BCE；
  - 两路 survival BCEWithLogits；
  - `MaxPool16` target；
  - detached ratio cap。
- `model/tpd_survival.py`
  - `emb1/emb2` 两个独立 `1×1 Conv` heads；
  - 同一 stride-16 target grid；
  - segmentation path 不变；
  - inference 不需要 heads。
- `model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py`
  - 训练模型注册 TSS；
  - 推理模型严格移除 TSS；
  - TPD/NER/QFG 主线保持不变。

## 相关优化思想

- Focal-style hard example weighting：用于强调困难正负样本；
- PCGrad：用于分析和处理辅助任务与主任务之间的梯度冲突；
- GradNorm：用于分析多任务损失的梯度尺度和训练速率不平衡。

EC-TSS V3.1 的核心区别是：

> 风险权重由最终 segmentation 在目标邻域与纯背景中的阈值错误状态产生，并以 stop-gradient 方式选择 TSS 监督位置；它不是简单对 survival head 自身置信度做 focal weighting。
