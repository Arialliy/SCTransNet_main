# SCTransNet：TSS 混合结果复盘与 Query-only FG 最终模型集成方案

> **项目主线**：SCTransNet + TPD V8-MPRS-DCH + 五节点 NER V4 Tail-Aware + 可选训练期 TSS + Query-only FG
>
> **当前阶段**：TSS 裁决为 `PARETO_MIXED_TRADEOFF`；QFG-V2-CROA 的代码、测试、source-lock 与 GPU smoke 已完成；C/D 两臂正在物理 GPU2/GPU3 运行 seed42 × 800 epochs
>
> **文档日期**：2026-07-29
> **审查范围**：公开仓库 `Arialliy/SCTransNet_main` 的 SCTransNet、TPD、NER、TSS、forward contract、training loss、Query-only FG 与 bridge 代码；本地正式数值、checkpoint 和 sweep 结果以项目提供的权威结果为准。

> **设计边界修订**：本文中的“冻结”和“不修改”只约束当前 C/D
> formal800 配对实验及其 source lock，不能解释为后续模型设计永久禁止修改。
> 当前完整模型的主线、可变实现区和性能驱动修改规则以
> `SCTransNet_完整模型主线与性能驱动设计边界.md` 为准。若 C/D 的综合性能
> 不足，允许修改 QFG、TSS、损失、decoder/skip 标定以及 TPD/NER 内部公式；
> 保持 Keep–Context–Saliency 三源语义与 3+2 五节点嵌套恢复时，仍属于同一
> 研究主线。

---

## 1. 执行结论

当前 TSS 不能判定为全面成功，也不是被支配的失败方案。固定阈值 `0.5` 下，它相对同父、同配置 Control 呈现互相交叉的优势：

- 相对 `Control-best`，TSS 的 mIoU 提高，但 Fa 和错误目标数略差；
- 相对 `Control-best_mIoU`，TSS 的 Fa 和错误目标数更好，但 mIoU 更低；
- 三个工作点均为 `188/189`，TSS 没有恢复第 189 个目标；
- 完整闭区间扫描进一步确认：TSS 没有提高五预算 Pd envelope，但在相同
  `188/189` 下提供了 Control/V4 不能同时在 Fa、mIoU 和错误目标上支配的连续
  工作区间。

因此正式状态更新为：

```text
tss_fixed_threshold_status=MIXED_CONFIRMED
tss_selector_sweep_status=COMPLETE
tss_decision=PARETO_MIXED_TRADEOFF
tss_auxiliary_value_supported=true
tss_universal_improvement=false
tss_failure_established=false
tss_sweep_finalization_required=false
query_fg_engineering_authorized=true
query_fg_formal_training_authorized=true
query_fg_cd_training_status=RUNNING_GPU2_GPU3
final_tss_inclusion_decided=false
```

下一阶段不应从 TSS 结果 checkpoint 串行接入 FG，而应从**同一个冻结 V4 父 checkpoint**构造 `TSS × FG` 的 2×2 因子实验。对 seed42 的每个预注册标量指标或 Fa budget，分别报告：

1. FG-off 与 FG-on 条件下的 TSS simple effect；
2. TSS-off 与 TSS-on 条件下的 FG simple effect；
3. 两个因素的 descriptive marginal effect；
4. TSS 与 FG 的 descriptive interaction；
5. 最终推理模型是否需要保留 TSS 训练配方。

固定只使用 `seed=42`。这些差值是当前内部划分上的描述性效应，不是跨随机性统计推断；无论最终结果如何，均保持
`stability_claim_supported=false`。

推荐的最终 FG 候选为：

> **QFG-V2-CROA**
>
> **Centered RMS-normalized and Optimization-Anchored Query-only Frequency Gate**
> **中心化 RMS 稳定、优化锚定的仅 Query 频率门**

它保持现有 Query-only FG 的边界，但修复四个直接影响稳定性的代码问题：

- gate logit 未做空间中心化；
- 不同尺度的频率先验未做幅值归一化；
- 当前 `(0, 2)` 门控范围过宽；
- `alpha=0` 只保证初始前向等价，不能保证第一次优化更新不受随机 gate 投影影响。

本方案不会修改已冻结的 TPD、NER、SCTransNet 主干、K/V、CFN、decoder、六路分割损失和评估协议。

---

## 2. 当前 TSS 固定点的精确解释

### 2.1 正式数值

| 模型 / checkpoint | Epoch | selector role | Pd | Fa | mIoU | tiny-Pd | 错误目标 |
|---|---:|---|---:|---:|---:|---:|---:|
| V4-best | 422 | `best_validation_pd_primary` | 189/189 | 7.572031×10⁻⁶ | 0.926418 | 39/39 | 14 |
| V4-best_mIoU（共同父点） | 489 | `best_validation_miou_secondary` | 188/189 | 4.244926×10⁻⁶ | 0.938178 | 39/39 | 4 |
| Control-best | 37 | `best_validation_pd_primary` | 188/189 | 4.015471×10⁻⁶ | 0.934370 | 39/39 | 4 |
| Control-best_mIoU | 3 | `best_validation_miou_secondary` | 188/189 | 4.818565×10⁻⁶ | 0.940091 | 39/39 | 6 |
| TSS-best = TSS-best_mIoU | 3 | 两个 selector role、同一 state | 188/189 | 4.130199×10⁻⁶ | 0.936870 | 39/39 | 5 |

TSS 的 `best` 与 `best_mIoU` 是两个带不同 role 的 checkpoint 文件，但
state-dict SHA 均为
`3bf3c82c788f3cf2c7b6a34415b858dc1cb6c5f5fe5deb68a4c9c2718874c8d1`。
因此当前四个 A/B selector artifact 实际对应三个独立权重状态。

### 2.2 成对差值

相对 `Control-best`：


delta(mIoU) = +0.002501

delta(Fa)   = +1.147×10⁻⁷  （Fa 变差约 2.86%）

delta(error targets) = +1

相对 `Control-best_mIoU`：


delta(mIoU) = -0.003221

delta(Fa)   = -6.884×10⁻⁷  （Fa 改善约 14.29%）

delta(error targets) = -1

这说明 TSS 不是简单的“整体变好”或“整体变差”，而是在 segmentation area quality、独立虚警和 checkpoint 选择之间重新分配了误差。

### 2.3 为什么固定阈值不能完成裁决

当前三个点使用不同 checkpoint 选择准则。`best` 更偏向 Pd/Fa，`best_mIoU` 更偏向区域质量；仅比较单个固定阈值点会混合以下因素：

- 模型表示是否改善；
- 输出概率是否发生校准偏移；
- checkpoint 选择规则；
- 阈值 `0.5` 是否恰好位于该模型的有利工作区间。

因此，TSS 的正式裁决必须基于相同的阈值生成算法、共享固定网格、闭区间端点规则、Fa budgets、连通域和匹配协议下的联合 frontier，而不能根据一个固定点投票。各模型按同一冻结算法生成的 empirical-quantile/tail 补充阈值可以不同，不能误写成实际 threshold 列表逐值相同。

### 2.4 已完成 sweep 的预算与 Pareto 裁决

2026-07-29 已在物理 GPU2/GPU3 完成并分别写出四份 checkpoint-local
artifact：

| Variant | selector | Epoch | sweep artifact SHA-256 |
|---|---|---:|---|
| Control | `best` | 37 | `e7a75418267dc5cc85993ac73f3cc2a39066c5cf0e5803df3452b0b5d8635825` |
| Control | `best_mIoU` | 3 | `3e84d884d052a433f01e43ab96ac51e2985ba76c62a7ad201eaa177353d020d5` |
| TSS | `best` | 3 | `cb4d0345bd8b08b92dcaf3d69dadf3ab829c761a671b95e6691135fd57fd2adc` |
| TSS | `best_mIoU` | 3 | `36d1cbcc8b885cece7f2e75dea106f9035b7ba6534e596427a5e68e6b97a3ee6` |

五个 Fa budget 的 Pd envelope 为：

```text
V4      = [0, 188, 189, 189, 189]
Control = [0, 188, 188, 188, 189]
TSS     = [0, 188, 188, 188, 189]
budgets = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
```

因此 TSS 没有提高 Pd budget envelope，也没有恢复第 189 个目标。但在
`Fa<=5e-6` 时，TSS 可取得：

```text
Pd=188/189
Fa=3.5565598567e-6
mIoU=0.9358771374
tiny-Pd=39/39
unmatched predicted objects=6
threshold=0.65
```

该点相对 Control-best 的相同 Pd 点降低 Fa，但交换了少量 mIoU/错误目标；
相对 Control-best_mIoU 的预算点则同时具有更低 Fa、更高 mIoU 和更少错误
目标。相对 V4-best_mIoU 的预算点，TSS 具有更低 Fa 和更高 mIoU，但错误目标
更多。`threshold=0.52...0.57` 的相邻固定网格点也复现了相同 Pd 下的连续
Fa–mIoU 权衡，因此不是单个浮点阈值造成的孤立点。

冻结裁决为：

```text
tss_decision=PARETO_MIXED_TRADEOFF
tss_auxiliary_value_supported=true
tss_universal_improvement=false
```

---

## 3. TSS 的代码作用边界与当前混合结果解释

公开代码中，Target Survival Supervision 使用两个独立 `1×1 Conv` head，分别监督 `emb1` 和 `emb2` 的 stride-16 endpoint。训练 target 为：

\[
Y_{16}=\operatorname{MaxPool}_{16}(Y)
\]

总损失为：

\[
\mathcal L
=
\mathcal L_{seg}
+
\lambda_s\left[
\operatorname{BCEWithLogits}(Z_1,Y_{16})
+
\operatorname{BCEWithLogits}(Z_2,Y_{16})
\right]
\]

其中原分割损失仍是六路 BCE 的冻结加法顺序。`survival_weight=0` 时，代码不会构造 survival target，也不会读取 survival logits，因此可形成严格的 no-auxiliary control。

### 3.1 TSS 实际解决的是“cell presence”，不是精确目标定位

stride-16 cell 只回答：

> 这个 16×16 对应网格中是否存在至少一个目标像素？

它不能直接约束：

- 目标在 cell 内的亚像素位置；
- 目标是否形成单个连通响应；
- 目标边界是否紧凑；
- 第 189 个目标的最终 decoder logit 是否跨过阈值；
- 高置信目标与高频背景之间的排序间隔。

因此，TSS 可能提高某些区域的整体召回或 mIoU，但不一定恢复最后一个难目标，也不一定降低所有 Fa 工作区间。

### 3.2 辅助损失与最终分割目标存在尺度不一致

`Y16` 是 max-presence target。一个极小目标和占满 cell 的较大前景都会产生同一个正标签。该监督有利于防止目标在 tokenizer endpoint 消失，但不会惩罚 endpoint 在同一 cell 内产生过宽或分散响应。

这与当前结果吻合：TSS 能在一个比较中提高 mIoU，在另一个比较中降低 Fa，但没有形成同时改善 Pd、Fa、mIoU 的统一方向。

### 3.3 TSS 对第 189 个目标缺少直接排序约束

BCEWithLogits 对所有 stride-16 cell 汇总优化。即使使用 `pos_weight`，最后一个困难目标仍可能被大量较容易的正 cell 和负 cell 梯度淹没。若第 189 个目标的问题发生在：

- decoder 恢复阶段；
- NER mask 的局部抑制；
- Query–Key 通道相关性不足；
- 目标与结构化背景的频谱可分性不足；

则继续增大 survival loss 通常不会精确解决问题，反而可能损害 mIoU 或 Fa。

### 3.4 当前不建议继续调 TSS 权重

四份 sweep 已完成；为了保持现有 A/B 的正式可比性，仍不应修改：

- `survival_weight`；
- `survival_pos_weight`；
- endpoint 数量；
- survival target 定义；
- checkpoint 选择顺序。

否则将使已有 TSS/Control 失去正式可比性，并把下一阶段变成无边界的联合调参。

---

## 4. TSS sweep 的正式裁决规则

四个 selector artifact 已完成统一聚合，并按冻结协议得到
`PARETO_MIXED_TRADEOFF`。TSS 的两个 selector 虽共享同一 state，但已经分别
生成绑定各自 checkpoint 文件 SHA、role 和 identity 的结果文件，没有把两个
artifact 静默合并。

### 4.1 `RELATIVE_IMPROVED`

只有当 TSS 相对 V4 与同父 Control 提供严格、可复核的相对改善，而不是仅交换
Pd、Fa、mIoU 或错误目标时，才记录：

```text
tss_decision=RELATIVE_IMPROVED
tss_auxiliary_value_supported=true
tss_universal_improvement=false
```

单 seed42 内部验证仍不得把该标签扩写为跨随机性或论文级全面优越。

### 4.2 `PARETO_MIXED_TRADEOFF`

满足下列任一条件：

1. TSS 在至少一个预注册 Fa budget 上提高 Pd 包络；
2. 在相同 Pd 下，TSS 产生 Control/V4 均无法支配的更低 Fa 工作点；
3. 在相近 Pd/Fa 区间，TSS 提供独有的更高 mIoU 非支配点；
4. TSS 的独有点在两个 checkpoint 的闭区间 sweep 中可复核，不是单阈值孤点。

状态：

```text
tss_decision=PARETO_MIXED_TRADEOFF
tss_auxiliary_value_supported=true
tss_universal_improvement=false
```

### 4.3 `DOMINATED`

只有在下列条件全部成立时才能判定：

- TSS 的全部 sweep 点均被 V4 或同父 Control 弱支配；
- 所有预注册 Fa-budget 包络均无严格改善；
- 没有独有 mIoU–Pd–Fa 非支配点；
- 结果文件、阈值范围、排序和 endpoint 全部通过工程复核。

状态：

```text
tss_decision=DOMINATED
tss_auxiliary_value_supported=false
tss_keep_in_final_recipe=false
```

TSS 的该标签只决定最终训练配方和论文表述，**不阻止 Query-only FG 工程开发**。

---

## 5. 为什么最终 FG 阶段必须采用 2×2 因子设计

不建议执行：

```text
V4 → TSS checkpoint → 加 FG → 继续训练
```

这种串行方案无法区分：

- FG 是否真正有效；
- 收益是否只是继承了 TSS checkpoint；
- FG 是否只在 TSS 已改变的参数区域中有效；
- TSS 与 FG 是否发生正/负交互。

正确实验设计为：所有 arm 从**同一个冻结 V4 父 checkpoint、同一 SHA、同一训练配置和同一 seed42 随机流**开始。

| Arm | TSS loss | Query-only FG | 作用 |
|---|---:|---:|---|
| A | 0 | Off | 同父 Control |
| B | On | Off | TSS 主效应 |
| C | 0 | On | FG 主效应 |
| D | On | On | TSS+FG 联合与交互效应 |

只对已预注册、方向明确且在四个 arm 间对齐的标量估计量定义差值；不能直接对无序 Pareto 点集合做减法。

simple effects：

\[
E_{TSS\mid FG=0}=B-A
\]

\[
E_{TSS\mid FG=1}=D-C
\]

\[
E_{FG\mid TSS=0}=C-A
\]

\[
E_{FG\mid TSS=1}=D-B
\]

descriptive marginal effects：

\[
E_{TSS}^{marginal}
=
\frac{(B-A)+(D-C)}{2}
\]

\[
E_{FG}^{marginal}
=
\frac{(C-A)+(D-B)}{2}
\]

\[
E_{interaction}=D-C-B+A
\]

这些量只描述 seed42 当前内部划分，不附带显著性、置信区间或稳定性含义。

### 5.1 既有 A/B 是否可直接复用

只有满足以下条件才可复用当前 Control/TSS：

- 父 checkpoint 文件 SHA 完全一致；
- split、归一化、crop 流、seed 和 DataLoader generator 一致；
- optimizer、LR、warmup、epoch 数、AMP 与确定性设置一致；
- checkpoint 选择器和 sweep evaluator 一致；
- identity-initialized QFG 模型与旧 A/B 在初始六路输出、全部共享梯度及共享参数
  的 first-Adam 更新/逐参数状态上逐位等价；QFG 自有 optimizer state 单独记录。

任何一项不满足，就应重跑 A/B 配对 control，不能用“配置看起来一样”代替精确配对。

当前只读审计已经确认：

```text
ab_parent_checkpoint_equal=true
ab_parent_checkpoint_sha256=0ae6c0e034952e18333d8fa6ccd3bbf635cae5efa8017b06df5e00ccc4ed14ab
ab_initial_model_state_equal=true
ab_initial_model_state_sha256=935b205b5eb19e9783c4d507e468d084746ce420ad61937e28daa3799c1890ea
ab_initial_rng_equal=true
ab_data_and_split_equal=true
ab_optimizer_and_schedule_contract_equal=true
ab_source_locks_equal=true
ab_existing_run_and_checkpoint_integrity=true
ab_qfg_identity_output_gradient_first_adam_equivalence=true
ab_sweep_evaluator_core_equivalence=true
ab_reuse_status=REUSABLE
```

QFG-V2 已完成上述逐位锚定、checkpoint-local evaluator 复用和独立
write-once source-lock 审计，因此既有 A/B 正式纳入 2×2；无需为增加 C/D
重复训练无 QFG 的 A/B。

### 5.2 不允许根据 TSS sweep 标签删掉 D arm

即使 TSS 最终为 `DOMINATED`，D 仍具有价值：它能检验 TSS 是否与 FG 存在正交互。最终可能出现：

- B 被支配，但 D 优于 C：TSS 单独无效、与 FG 联合有效；
- B 有 Pareto 点，但 D 不如 C：TSS 与 FG 负交互，最终应删除 TSS；
- C、D 都有效：依据简单性和严格非支配性决定最终配方。

---

## 6. 现有 Query-only FG 代码的正确部分

公开实现的基本边界是合理的：

1. 对 encoder 的 `x1...x4` 分别执行固定 2×2 Haar 分解；
2. 支持 `high`、`low`、`high_low` 三种频率输入；
3. 将各尺度频率特征平均池化到共同 Query 网格；
4. 每个尺度用独立 projection 生成一张空间 gate；
5. gate 只插入 `q1...q4` 卷积之后、flatten/normalize 之前；
6. K、V、CFN、decoder、TPD 和 NER 的前向结构不变；
7. prepared prior 每次整模型 forward 只构造一次，在四个 SCTB 中复用。

这与 SCTransNet 的注意力语义相符。当前 SSCA 对 Query 和共享 Key 做通道相关性计算，因此空间变化的 Query gate 会改变每个通道的空间模式，再影响通道交叉注意力；它不是额外的 decoder 空间注意力支路。

---

## 7. 现有 FG V1 的四个稳定性风险

### 7.1 gate logit 未做空间中心化

当前实现近似为：

\[
Q'=Q\left(1+\tanh(\alpha)\tanh(Z)\right)
\]

随后 Query 会沿空间维做 L2 normalization。若 `Z` 含有较大的空间均值，则相当一部分门控只是对整张 Query map 做近似全局缩放，而全局缩放会被后续 normalization 抵消。

结果是：

- alpha 的有效容量被浪费在不可辨识的 DC 分量；
- 不同样本的 gate 均值会造成不必要的优化噪声；
- gate 真正有用的部分其实是空间相对变化，而不是绝对均值。

### 7.2 多尺度频率先验幅值未归一化

Haar LL、LH、HL、HH 的幅值随：

- encoder 层级；
- batch normalization 状态；
- 背景动态范围；
- 小目标强度；
- 模型训练阶段

而变化。直接送入 projection 会让 gate 学习同时承担“幅值校准”和“空间判别”，加剧跨 seed 不稳定。

### 7.3 门控范围 `(0,2)` 对严格 Fa 区域过宽

原公式中：

\[
\tanh(\alpha)\in(-1,1),\qquad \tanh(Z)\in(-1,1)
\]

所以：

\[
1+\tanh(\alpha)\tanh(Z)\in(0,2)
\]

在红外小目标中，高频不仅来自目标，也来自边缘、云层、海杂波和传感器噪声。最大两倍 Query 响应的范围可能在低 Fa budget 中放大 hard negatives。

### 7.4 `alpha=0` 只保证前向等价，不保证优化路径等价

原实现将 `alpha` 初始化为 0，因此初始：

\[
Q'=Q
\]

但对 alpha 的梯度为：

\[
\left.
\frac{\partial Q'}{\partial \alpha}
\right|_{\alpha=0}
=Q\tanh(Z)
\]

而 `Z` 来自随机初始化的 projection。于是第一次 backward 中，alpha 的更新方向已经依赖随机 gate 投影。它具有**函数零点等价**，但不具有**第一次优化更新锚定**。

项目在 V6/V7 阶段已经表明：初始输出相同并不足以保证跨 seed 稳定，必须检查零点梯度和第一 optimizer step。

### 7.5 当前 frequency source 会通过额外支路反向更新 encoder

现有 `prepare(feature)` 保留完整 autograd graph。虽然前向中只修改 Query，但 gate projection 的梯度也会通过 Haar 分支回传到 `x1...x4`。因此“Query-only”严格来说只是固定参数下的 forward boundary；训练后共享 encoder 改变，K/V 也会间接受影响。

为了使最终实验具备更清晰的因果边界，正式候选建议对 frequency conditioning source 使用 `detach()`：

```python
frequency_source = feature.detach()
```

这样：

- gate 网络仍可训练；
- Query 主路径仍正常向 encoder 反向传播；
- 只删除频率旁路对共享 encoder 的额外梯度；
- K/V 没有额外 frequency-branch 梯度污染。

---

## 8. 推荐模型：QFG-V2-CROA

### 8.1 模型名称

```text
QFG-V2-CROA
Centered RMS-normalized and Optimization-Anchored Query-only Frequency Gate
中心化 RMS 稳定、优化锚定的仅 Query 频率门
```

### 8.2 保持不变的部分

```text
Haar 2×2 fixed analysis         不变
formal mode = high_low          不变
x1/x2/x3/x4 四尺度输入          不变
对齐比例 8/4/2/1                不变
每尺度 hidden width = 8         不变
只作用 q1/q2/q3/q4              不变
插入位置：q conv 后、normalize 前 不变
prepared once / reuse 4 SCTBs   不变
K/V/CFN/decoder                 不变
TPD V8-MPRS-DCH                 不变
NER V4 Tail-Aware               不变
TSS target 和 loss              不变
```

### 8.3 新的频率先验归一化

对每个尺度：

\[
U_i=\operatorname{SelectBands}(\operatorname{Haar}(x_i))
\]

对齐到 Query 网格后：

\[
\widehat U_i=
\frac{U_i}
{\sqrt{\operatorname{mean}_{chw}(U_i^2)+\epsilon}}
\]

这里采用**每样本 full-tensor RMS**，不对 LL 做均值中心化，以保留低频符号和高低频相对结构。

### 8.4 gate logit 的空间中心化与 RMS 归一化

投影得到 raw logit：

\[
Z_i=f_i(\widehat U_i)
\]

先移除空间 DC：

\[
\widetilde Z_i=Z_i-\operatorname{mean}_{hw}(Z_i)
\]

再执行逐样本空间 RMS：

\[
\overline Z_i=
\frac{\widetilde Z_i}
{\sqrt{\operatorname{mean}_{hw}(\widetilde Z_i^2)+\epsilon}}
\]

先采用有界 arctangent map：

\[
H_i=
\frac{1}{\pi}\arctan(\pi\overline Z_i)
\]

因此：

\[
H_i\in(-0.5,0.5)
\]

非线性映射不会自动保持零均值，因此最终 gate 再中心化并缩放：

\[
G_i=
\frac{1}{2}
\left[
H_i-\operatorname{mean}_{hw}(H_i)
\right]
\]

从而在实数数学下：

\[
\operatorname{mean}_{hw}(G_i)=0,
\qquad
G_i\in(-0.5,0.5)
\]

### 8.5 Query 调制

\[
a_i=\tanh(\alpha_i)
\]

\[
F_i=1+a_iG_i
\]

\[
Q_i'=Q_i\odot F_i
\]

无论 alpha 如何变化：

\[
F_i\in(0.5,1.5)
\]

该范围与 NER V2/V4 的 skip factor 风格一致，比原 `(0,2)` 更适合控制严格 Fa 区域的风险。

### 8.6 优化锚定初始化与 epsilon 零点增益

不再使用：

```python
alpha = 0
random gate_out
```

改为：

```python
alpha_effective_init = 0.1
gate_out.weight = 0
gate_out.bias = None
```

即：

\[
\alpha_i^{param}=\operatorname{atanh}(0.1)
\]

最后一层 `1×1 Conv` 权重严格为零。初始时：

\[
Z_i=0,\quad G_i=0,\quad F_i=1,\quad Q_i'=Q_i
\]

而且：

- `dL/d alpha = 0`，因为 `G=0`；
- frequency branch 对 encoder 的额外梯度为 0；
- 共享模型的全部梯度与 FG-off control 一致；
- 第一次 Adam 更新中，仅 gate 的 terminal projection 开始学习；
- 从第二步起，gate 逐渐产生空间变化；
- 初始有效 alpha 为 0.1，使 terminal projection 能获得非零梯度。

需要明确：前向零点不代表 terminal projection 的零点梯度很小。令
\(\mathcal P(X)=X-\operatorname{mean}_{hw}(X)\)，则在 \(Z=0\)：

\[
\left.\frac{\partial \overline Z}{\partial Z}\right|_{Z=0}
=
\frac{\mathcal P}{\sqrt{\epsilon}},
\qquad
\left.\frac{\partial G}{\partial Z}\right|_{Z=0}
=
\frac{1}{2\sqrt{\epsilon}}\mathcal P
\]

正式值 `eps=1e-6`、`tanh(alpha)=0.1` 时，Query factor 对 raw logit 的
零点增益含有：

\[
\frac{0.1}{2\sqrt{10^{-6}}}=50
\]

因此 `eps` 必须进入 architecture manifest 和 source lock，不得在看到结果后
调整。GPU smoke 必须记录第一次 backward 的 terminal gradient norm、第一次
Adam step 的 `gate_out.weight` 更新 norm、factor 范围和 finite 状态；出现
非有限值或异常跃迁即工程失败。

这同时满足：

> 初始函数等价 + 初始共享梯度等价 + 第一优化步共享状态等价。

这里“共享状态等价”只指 shared parameter tensor 及其逐参数 Adam
`step/exp_avg/exp_avg_sq` 状态。FG-on optimizer 因含有额外 QFG 参数，其完整
`state_dict`、param-group 参数列表和序列化字节不可能与 FG-off optimizer
相等，不得要求整个 optimizer payload bitwise equal。

### 8.7 为什么 terminal conv 不应带 bias

空间中心化会把常数 bias 完全消除：

\[
(Z+b)-\operatorname{mean}_{hw}(Z+b)
=Z-\operatorname{mean}_{hw}(Z)
\]

因此 terminal bias 是不可辨识参数，应设为 `bias=False`，避免无效状态和无效优化器槽位。

### 8.8 参数量与 state-key 冻结常量

在 `feature_channels=(32,64,128,256)`、`mode=high_low`、
`hidden_channels=8`、四个独立 level、Haar kernel 为 persistent buffer 时：

```text
V4 parent parameters                 10,854,446
V4 parent state keys                         544
TSS parameters                                 98
TSS state keys                                  4
QFG-V2-CROA parameters                      15,684
QFG-V2-CROA state keys                          20

V4 + TSS + QFG training parameters       10,870,228
V4 + TSS + QFG training state keys              568
V4 + QFG exported inference parameters   10,870,130
V4 + QFG exported inference state keys          564
```

QFG 参数量来自每个 level 的 `4C×8` prior projection、`8×3×3`
depthwise projection、`8×1×1` terminal projection 和一个 alpha；每个 level
固定 5 个 state keys。正式 builder、checkpoint validator 和 export test 必须
逐项验证这些常量，禁止用运行时自动推导值替代冻结预期。

---

## 9. QFG-V2 核心参考代码

建议新增文件：

```text
model/tpd_frequency_gate_v2_croa.py
```

不要原地修改或覆盖已经审计的 `model/tpd_frequency_gate.py`。

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from model.tpd_frequency_gate import (
    FixedHaarAnalysis,
    PreparedQueryFrequencyLevel,
    _mode_channels,
    _select_bands,
)


RMS_EPS = 1e-6
FORMAL_ALPHA_EFFECTIVE_INIT = 0.1
FORMAL_GATE_LIMIT = 0.5


def _sample_full_tensor_rms(
    value: torch.Tensor,
    eps: float = RMS_EPS,
) -> torch.Tensor:
    working = value.float()
    rms = working.square().mean(
        dim=(1, 2, 3), keepdim=True
    ).add(eps).sqrt()
    return (working / rms).to(dtype=value.dtype)


def _spatial_center_rms(
    value: torch.Tensor,
    eps: float = RMS_EPS,
) -> torch.Tensor:
    working = value.float()
    centered = working - working.mean(
        dim=(-2, -1), keepdim=True
    )
    rms = centered.square().mean(
        dim=(-2, -1), keepdim=True
    ).add(eps).sqrt()
    return centered / rms


def _centered_bounded_arctangent_gate(
    normalized: torch.Tensor,
) -> torch.Tensor:
    bounded = torch.atan(math.pi * normalized) / math.pi
    return 0.5 * (
        bounded
        - bounded.mean(dim=(-2, -1), keepdim=True)
    )


class QueryFrequencyLevelGateV2CROA(nn.Module):
    """One Query level with centered/RMS and optimizer anchoring."""

    def __init__(
        self,
        feature_channels: int,
        *,
        mode: str = "high_low",
        hidden_channels: int = 8,
        expected_alignment: Tuple[int, int],
        detach_frequency_source: bool = True,
        alpha_effective_init: float = FORMAL_ALPHA_EFFECTIVE_INIT,
    ) -> None:
        super().__init__()
        if not 0.0 < alpha_effective_init < 1.0:
            raise ValueError("alpha_effective_init must be in (0, 1)")

        self.feature_channels = int(feature_channels)
        self.mode = str(mode)
        self.hidden_channels = int(hidden_channels)
        self.expected_alignment = tuple(expected_alignment)
        self.detach_frequency_source = bool(detach_frequency_source)
        self._prepared_owner_token = object()

        self.haar = FixedHaarAnalysis(validate_finite=True)
        self.prior_projection = nn.Conv2d(
            _mode_channels(self.feature_channels, self.mode),
            self.hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                groups=self.hidden_channels,
                bias=False,
            ),
            nn.GELU(),
        )
        self.gate_out = nn.Conv2d(
            self.hidden_channels,
            1,
            kernel_size=1,
            bias=False,
        )
        self.alpha = nn.Parameter(
            torch.tensor(math.atanh(alpha_effective_init))
        )

        # Hidden layers use one isolated deterministic seed in the model
        # factory.  The terminal map is exactly zero for identity anchoring.
        nn.init.zeros_(self.gate_out.weight)

    @staticmethod
    def _align_prior(
        prior: torch.Tensor,
        query_size: Tuple[int, int],
        expected_alignment: Tuple[int, int],
    ) -> torch.Tensor:
        ph, pw = prior.shape[-2:]
        qh, qw = query_size
        if ph % qh or pw % qw:
            raise ValueError("prior/query grids are not integer aligned")
        ratio = (ph // qh, pw // qw)
        if ratio != expected_alignment:
            raise ValueError(
                f"expected alignment {expected_alignment}, got {ratio}"
            )
        if ratio == (1, 1):
            return prior
        return torch.nn.functional.avg_pool2d(
            prior,
            kernel_size=ratio,
            stride=ratio,
        )

    def prepare(
        self,
        feature: torch.Tensor,
        query_size: Tuple[int, int],
    ) -> PreparedQueryFrequencyLevel:
        source = (
            feature.detach()
            if self.detach_frequency_source
            else feature
        )
        bands = self.haar(source)
        selected = _select_bands(bands, self.mode)
        selected = self._align_prior(
            selected,
            query_size,
            self.expected_alignment,
        )
        selected = _sample_full_tensor_rms(selected)
        hidden = self.prior_projection(selected)
        hidden = self.spatial_projection(hidden)
        raw_logits = self.gate_out(hidden)
        return PreparedQueryFrequencyLevel(
            gate_logits=raw_logits,
            query_size=tuple(query_size),
            batch_size=int(feature.shape[0]),
            _owner_token=self._prepared_owner_token,
        )

    def apply_prepared(
        self,
        query: torch.Tensor,
        prepared: PreparedQueryFrequencyLevel,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(prepared, PreparedQueryFrequencyLevel):
            raise TypeError("prepared must be a PreparedQueryFrequencyLevel")
        if prepared._owner_token is not self._prepared_owner_token:
            raise ValueError("prepared prior belongs to another QFG level")
        if tuple(query.shape[-2:]) != tuple(prepared.query_size):
            raise ValueError("prepared/query grids differ")
        if int(query.shape[0]) != int(prepared.batch_size):
            raise ValueError("prepared/query batch sizes differ")
        raw = prepared.gate_logits
        normalized = _spatial_center_rms(raw)
        gate = _centered_bounded_arctangent_gate(normalized)
        alpha = torch.tanh(self.alpha.float())
        factor = 1.0 + alpha * gate
        output = (
            query.float() * factor
        ).to(dtype=query.dtype)
        return output, gate, factor
```

### 9.1 实现注意事项

上述代码展示核心数学；正式实现还必须复用现有 V1 的：

- dtype/device 检查；
- finite 检查；
- prepared owner token 检查；
- batch/query grid metadata 检查；
- 四尺度 dataclass API；
- architecture manifest；
- parameter/buffer audit。

不要为了缩短代码删除现有防御性验证。

### 9.2 四尺度 wrapper

新 wrapper 建议保持与现有 `QueryOnlyFrequencyGate` 完全相同的调用协议：

```python
shared_query_size = tuple(emb1.shape[-2:])
prepared = qfg.prepare(
    encoder_features=(x1, x2, x3, x4),
    query_sizes=shared_query_size,
)

gated = qfg.apply_prepared(
    queries=(q1, q2, q3, q4),
    prepared=prepared,
)
q1, q2, q3, q4 = gated.queries
```

这样可直接复用 bridge 的总体结构，并降低集成风险。

---

## 10. Bridge 修改方案

建议新增：

```text
model/tpd_query_frequency_bridge_v2_croa.py
```

桥接顺序必须保持：

```text
q1/q2/q3/q4 convolution
→ QFG-V2 apply_prepared
→ rearrange
→ L2 normalize over spatial tokens
→ QKᵀ
```

K/V 代码保持逐行一致：

```python
k = attention.k(attention.mheadk(emb_all))
v = attention.v(attention.mheadv(emb_all))
```

### 10.1 不建议直接修改 `Attention_org.forward`

直接改 SCTransNet 主类会：

- 破坏冻结 baseline；
- 让非 FG 模型也依赖新参数；
- 增加 checkpoint 和 source-lock 风险；
- 难以验证 K/V 是否完全不变。

继续使用纯函数 bridge 更符合受控实验要求。

### 10.2 新 bridge 只增加可选诊断，不增加模型状态

建议加入 forward-local diagnostics collector：

```python
@dataclass
class QFGForwardDiagnostics:
    alpha: list[float]
    gate_rms: list[float]
    gate_p95: list[float]
    factor_min: list[float]
    factor_max: list[float]
    query_cosine_change: list[float]
```

默认 `diagnostics=None` 时不计算额外统计，不注册 buffer，不改变正式前向。

---

## 11. V4 + QFG + TSS 整模集成

建议新增：

```text
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa.py
```

不要修改冻结文件：

```text
model/tpd_clean_v8_mprs_dch.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware.py
model/tpd_survival.py
model/SCTransNet.py
```

### 11.1 集成原则

新模型继承 V4 Tail-Aware，并注册：

```python
self.tpd_qfg
self.target_survival  # 沿用正式 TSS 前缀；C/D 都注册，loss 决定是否使用
```

由于 V4 的 `_forward_with_relay()` 直接调用 `self.mtc.encoder(...)`，新类需要复制该 forward，并且只允许三处语义变化：

1. 从 `emb1/emb2` 保存 survival endpoint；
2. 在 encoder 前构造一次 prepared QFG；
3. 使用 `frequency_encoder_forward_v2_croa()` 代替原 encoder 调用。

其余 NER stage4→stage3→stage2、decoder、deep supervision 和返回顺序必须逐行保持一致。

### 11.2 前向流程

```text
input
→ SCTransNet encoder x1/x2/x3/x4/d5
→ V8 embeddings + five evidence nodes
→ save emb1/emb2 endpoints for optional TSS
→ QFG.prepare(detached x1/x2/x3/x4) once
→ four SCTBs:
     q conv → QFG → normalize → SSCA
     K/V unchanged
→ reconstruct + frozen double identity skip
→ NER V4 stage4 → stage3 → stage2
→ decoder and six segmentation maps
→ training only:
     structured output includes emb1/emb2 survival logits
     loss reads them only when survival_weight > 0
→ eval:
     legacy six segmentation maps
→ evaluator:
     evaluator_prediction(...) selects the final segmentation map
```

### 11.3 组合模型伪代码

```python
class TPD8NER4QFGV2CROASurvivalSCTransNet(
    TPDNERV8MPRSDCHV4SCTransNet
):
    def __init__(self, parent, *, variant, qfg_enabled=True, ...):
        super().__init__(parent, variant=variant, ...)
        self.tpd_qfg = QueryOnlyFrequencyGateV2CROA(...)
        self.target_survival = PairedTargetSurvivalHeads(32, 64)
        self.qfg_enabled = bool(qfg_enabled)

    def _forward_with_relay(self, x):
        # Copy the frozen V4 forward exactly until explicit_embeddings.
        ...
        emb1, emb2, emb3, emb4, evidence1, evidence2 = \
            self.explicit_embeddings(x1, x2, x3, x4)

        if self.qfg_enabled:
            prepared = self.tpd_qfg.prepare(
                (x1, x2, x3, x4),
                tuple(emb1.shape[-2:]),
            )
            encoded = frequency_encoder_forward_v2_croa(
                self.mtc.encoder,
                emb1, emb2, emb3, emb4,
                self.tpd_qfg,
                prepared,
            )
        else:
            encoded = self.mtc.encoder(
                emb1, emb2, emb3, emb4
            )

        # Copy frozen V4 reconstruction, NER and decoder exactly.
        segmentation = ...

        if self.training:
            return build_structured_survival_output(
                segmentation,
                emb1,
                emb2,
                self.target_survival,
            )
        # Preserve V4/TSS eval contract: return the legacy six-map tuple.
        return segmentation
```

### 11.4 沿用现有受控的瞬态 endpoint capture

不要新增无生命周期约束的缓存，例如：

```python
self._last_emb1 = emb1
self._last_emb2 = emb2
```

原因：

- 容易跨 forward 保留 graph；
- exact resume 和并行训练更难审计；
- 可能产生显存泄漏；
- 无 guard/finally 的缓存会破坏现有 forward contract。

当前正式 TSS 实现已经使用 `_survival_capture_active`、
`_captured_survival_endpoints` 和 `try/finally`：capture 只在一次同步、
非重入 forward 内有效，结束时无条件清空；评估路径不执行 survival head。
QFG 集成必须直接继承并复用该既有实现，不得增加第二套 cache，也不得跨
forward 或 optimizer step 保存 endpoint/prepared 对象。现有重入 guard 仍然
保留，因此正式运行方式固定为单进程、单线程的普通同步 forward。

---

## 12. Warm-start 和初始化规则

### 12.1 唯一父 checkpoint

C/D arm 必须使用与 A/B 相同的冻结 V4 父 checkpoint。唯一合法身份为：

```text
parent_path=experiments/results/tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1/NUDT-SIRST/tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on/seed_42_formal800_exact_v4_tail_aware_seed42/best_miou.pth.tar
parent_sha256=0ae6c0e034952e18333d8fa6ccd3bbf635cae5efa8017b06df5e00ccc4ed14ab
parent_state_dict_sha256=2b8249ffd86866597f376c80839395a3cbdbb72a68301cd8a5a6eb36595c7e75
parent_variant=tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on
parent_checkpoint_role=best_validation_miou_secondary
parent_epoch=489

existing_ab_source_lock_sha256=23edf22eee2279dc59056ef4c4855ecd0d760fc3ee6856f902d44abecd9308cf
survival_statistics_sha256=102ededc559e69442a4ec13944d0c35e70ed99301657441a3755c07629bebdc9
training_data_sha256=39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e
```

C/D 必须生成包含 QFG-V2-CROA 新源码的独立 write-once source lock；不得把既有
A/B source-lock SHA 冒充为新训练锁。比较时应核对两份 lock 中全部共享源码和
数据绑定一致，同时允许 C/D lock 合法新增 QFG 文件。

禁止：

- C 从 Control-best 继续训练；
- D 从 TSS-best 继续训练；
- 根据 FG 结果在 V4 best 和 best_mIoU 之间切换父模型。

### 12.2 Strict extension warm-start

父 state 的每个 key 必须：

```text
key equal
shape equal
dtype equal
tensor value equal
```

只允许新增：

```text
tpd_qfg.*
target_survival.*  # 沿用既有正式 TSS state prefix
```

QFG 初始化额外检查：

```text
gate_out.weight == 0 exactly
tanh(alpha) == 0.1 within FP tolerance
terminal bias absent
frequency_source_gradient == detached
factor == 1 exactly at initialization
```

### 12.3 初始化随机流隔离

QFG hidden projection 在 CPU RNG fork 内只设置 CPU default generator，避免
注册新模块改变后续 CPU 或 CUDA 全局 RNG：

```python
with torch.random.fork_rng(devices=[]):
    torch.default_generator.manual_seed(QFG_INITIALIZATION_SEED)
    qfg = QueryOnlyFrequencyGateV2CROA(...)
```

这里不能调用 `torch.manual_seed(...)`，因为它还会改写 CUDA generator，而
`devices=[]` 只负责恢复 CPU RNG。

---

## 13. 训练入口修改

建议新增：

```text
experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_exact.py
experiments/launch_tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_formal800.sh
experiments/evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pd_fa.py
experiments/finalize_tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa.py
experiments/capture_tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_smoke.py
```

### 13.1 Loss 路由

继续复用现有：

```python
loss = compute_tpd_training_loss(
    output,
    target,
    criterion,
    survival_weight=args.survival_weight,
    survival_pos_weight=args.survival_pos_weight,
)
```

Arm C：

```text
qfg_enabled=true
survival_weight=0
```

Arm D：

```text
qfg_enabled=true
survival_weight=冻结的正式 TSS 权重
```

不得为 D 单独重新搜索 survival weight。

### 13.2 训练参数

除 QFG 特有参数外，所有正式配置必须与 A/B 完全一致：

```text
parent checkpoint        same SHA
optimizer                same
shared-parameter LR      same
warmup/cosine schedule   same
epochs                   800
AMP                      same frozen setting
batch/crop/data order    paired
checkpoint selector      same
threshold sweep          same
```

QFG 参数建议使用与 shared parameters 相同 LR，避免在首个正式版本中混入 LR multiplier。模型的零 terminal 初始化已经提供渐进启动，不需要额外 warmup 或冻结阶段。

正式常量锁定为：

```text
dataset=NUDT-SIRST
split_source=img_idx/train_NUDT-SIRST.txt
internal_train_count=530
internal_validation_count=133
official_test_accessed=false
training_seed=42
split_seed=20260722
patch_size=256
batch_size=16
workers=0
epochs=800
warmup_epochs=10
base_lr=1e-4
min_lr=1e-6
amp=false
cublas_workspace_config=:4096:8
fixed_threshold=0.5
match_radius=3
tiny_area=9
fa_budgets=[1e-6,5e-6,1e-5,5e-5,1e-4]
survival_weight.C=0
survival_weight.D=0.005
survival_pos_weight=102.33587204874334
primary_selector=[maximize Pd,minimize Fa,maximize tiny-Pd,maximize mIoU,minimize val_loss]
secondary_selector=[maximize mIoU,maximize Pd,minimize Fa,maximize tiny-Pd,minimize val_loss]
formal_sweep_checkpoints=[best.pth.tar,best_miou.pth.tar]
```

既有 A/B optimizer contract 已记录：

```text
optimizer=torch.optim.Adam
param_groups=1
betas=[0.9,0.999]
eps=1e-8
weight_decay=0
amsgrad=false
foreach=null
fused=null
capturable=false
```

C/D 必须复制这些 optimizer defaults。注册额外 QFG 参数后若无法证明 shared
parameter tensor 及其逐参数 Adam 状态在 identity step 上等价，就应重跑配对
A/B；不能依据整个 optimizer `state_dict` 相等，因为两侧参数列表不同。

### 13.3 必须配对随机流

固定 seed42 中：

```text
A/B/C/D
→ same crop order
→ same augmentation decisions
→ same DataLoader generator state
→ same batch sequence
```

只有模型分支和 TSS loss 开关不同。

### 13.4 exact resume

checkpoint 必须保存并恢复：

- model；
- optimizer；
- scheduler / warmup；
- Python RNG；
- NumPy RNG；
- Torch CPU RNG；
- 每张 CUDA 卡 RNG；
- DataLoader generator；
- checkpoint selector state；
- QFG/TSS variant manifest；
- source lock hash。

连续训练与 epoch 边界中断续训必须逐 tensor 一致。

---

## 14. 正式实验矩阵

### 14.1 最低正式矩阵

固定正式 seed42：

| Run | TSS | QFG-V2 | 备注 |
|---|---:|---:|---|
| A | Off | Off | 已有 Control，可在通过复用审计后复用 |
| B | On | Off | 已有 TSS，可在通过复用审计后复用 |
| C | Off | On | 新正式 run |
| D | On | On | 新正式 run |

正式 seed：

```text
42
```

仅使用 seed42，不新增其他 seed，不执行跨 seed 稳定性裁决。

### 14.2 每个 run 保存

```text
best
best_mIoU
last
```

`last` 用于 exact-resume、终点和完整性审计，不参与正式性能 selector sweep。
仅 `best` 与 `best_mIoU` 执行相同闭区间 threshold sweep。

若既有 A/B 通过复用审计：

```text
C/D: 2 new arms × seed42 × 3 checkpoints = 6 new checkpoint artifacts
C/D: 2 new arms × seed42 × 2 selectors   = 4 new selector sweeps
A/B/C/D complete matrix                  = 12 checkpoint artifacts total
of which existing A/B                    = 6
of which newly trained C/D               = 6
```

当前 A/B 的 4 个 selector artifact 对应 3 个独立 state；四个正式 sweep 已在
GPU2/3 并行执行，只有结果文件完成身份校验与封存后才能改为 `complete`。
如果 QFG-off trajectory 审计失败并必须重跑 A/B，则本阶段才会产生
`4 arms × seed42 × 3 = 12` 个新 checkpoint；不能无条件把 12 写成新产物数。

---

## 15. 必须新增的测试

建议新增：

```text
tests/test_tpd_frequency_gate_v2_croa.py
tests/test_tpd_query_frequency_bridge_v2_croa.py
tests/test_tpd_ner_v4_qfg_v2_croa_integration.py
tests/test_tpd_ner_v4_qfg_v2_croa_survival.py
tests/test_train_tpd_ner_v4_qfg_v2_croa_exact.py
tests/test_tpd_ner_v4_qfg_v2_croa_source_lock.py
```

### 15.1 频率数学测试

- Haar impulse 的 LL/LH/HL/HH 顺序；
- `high/low/high_low` channel layout；
- 对齐比例严格为 `8/4/2/1`；
- full-tensor RMS 数值正确；
- pre-nonlinearity normalized logits 的空间均值在 FP tolerance 内为 0；
- 最终二次中心化 gate 的空间均值在 FP tolerance 内为 0；
- gate 严格在 `(-0.5,0.5)`；
- factor 严格在 `(0.5,1.5)`；
- 常数 raw logit 映射为零 gate；
- terminal bias 不存在；
- `eps=1e-6` 的零点梯度增益符合解析式，且首个 GPU step finite。

### 15.2 优化锚定测试

必须比较 FG-off 和 FG-on 初始模型：

1. 六个 segmentation output 逐元素相同；
2. loss 逐元素相同；
3. 所有 shared parameter gradient 逐元素相同；
4. 第一次 Adam step 后 shared model tensor 与逐 shared-parameter Adam state 逐元素相同；
5. 第一次 step 后只有 `gate_out.weight` 发生预期变化；
6. 第二次 step 后允许 QFG 与 control 分叉；
7. alpha 第一步保持不变；
8. frequency source detach 后，不存在 gate side-branch 对 x1...x4 的额外梯度；
9. 不比较两侧完整 optimizer payload，因为 FG-on 具有额外参数和 param-group 成员；
10. 记录 terminal gradient/update norm，验证 `eps` 零点增益未造成异常跃迁。

### 15.3 Query-only 边界测试

在同一固定参数和输入下通过 hook 验证：

```text
K before rearrange: bitwise equal
V before rearrange: bitwise equal
Q before gate: bitwise equal
Q after gate: may differ after learning
CFN input contract: unchanged
```

还必须验证 identity gate 下：

```text
frequency_encoder_forward_v2_croa
==
original encoder.forward
```

### 15.4 TPD/NER 不变性测试

- `mtc.embeddings_1/2` state 与冻结 V4 父逐 tensor 相同；
- 五个 evidence node 逐元素相同；
- NER V4 tail support mask 在 identity FG 下逐元素相同；
- q4→q3→q2 顺序不变；
- stage4/global、stage3/2 tail-aware 公式不变；
- survival_weight=0 时不构造 Y16；
- eval mode 返回 legacy 六路 segmentation tuple；
- evaluator 只通过 `evaluator_prediction(...)` 选择 final segmentation map；
- 导出无 survival heads 模型后 segmentation output 逐元素一致。

### 15.5 环境测试

```text
ordinary Python          PASS
python -O                PASS
CPU forward/backward     PASS
RTX 5090 GPU2 smoke      PASS
RTX 5090 GPU3 smoke      PASS
strict reload            PASS
exact resume             PASS
source lock              PASS
```

---

## 16. 机制诊断：防止“数值过线但机制错误”

正式训练期间只记录，不据此修改公式。

### 16.1 每尺度 QFG 统计

每个 epoch 记录：

```text
tanh(alpha_i)
gate mean / RMS / p05 / p50 / p95
factor min / max / p05 / p95
factor saturation ratio
```

异常信号：

- alpha 长期接近 0：FG 未被采用；
- alpha 快速饱和：门控过强；
- factor 大量接近 0.5 或 1.5：高风险；
- gate RMS 很高但 normalized Query 几乎不变：门控容量被 normalization 抵消。

### 16.2 Query 的真实变化

由于 Query 后续还会 L2 normalize，应测量 normalize 后而非 gate 前的变化：

\[
\Delta Q_i=1-\cos(Q_i^{norm},Q_i'^{norm})
\]

并记录：

- attention entropy 变化；
- target cell 与 hard-negative cell 的 Query change；
- 第 189 个目标对应通道的相关性变化。

### 16.3 频率依赖 counterfactual

对正式训练完成的同一 checkpoint，在不重新训练的情况下执行：

```text
dynamic frequency prior
batch-shuffled prior
spatially permuted prior
zero gate / identity gate
high-only counterfactual
low-only counterfactual
```

正式 claim 至少要求：

- dynamic prior 优于 identity；
- 打乱空间对应关系后收益下降；
- 改善不是仅由一个全局缩放或新增参数造成。

counterfactual 只用于机制支持，不替代正式固定协议结果。

### 16.4 错误组件诊断

对 V4、A、B、C、D 统一输出：

- 第 189 个 GT 的 peak、mean、rank 和 threshold margin；
- unmatched background components；
- GT 内部碎裂组件；
- attached halo；
- split/merge；
- 每个错误组件的 Haar high/low energy；
- QFG factor 与错误组件位置的相关性。

这样可以判断 FG 改善来自：

- 恢复弱目标；
- 抑制结构化背景；
- 减少目标内部碎裂；
- 仅改变概率校准。

---

## 17. 最终工程条件与相对/Pareto 性能裁决

在正式训练前不能承诺未知模型必然获得某个数值。本阶段只有工程完整性是硬
Gate；Pd、Fa、mIoU、tiny-Pd、错误目标和五预算结果必须完整报告并进入相对/
Pareto 裁决，不能把任一单项绝对值设成事后 veto。

### Gate F-A：工程完整性硬条件

全部通过：

- 唯一父点 SHA 与 strict extension warm-start；
- QFG 数学、API、owner-token、dtype/device/finite 测试；
- initial six-output、loss、shared gradient 和 shared per-parameter Adam first-step anchor；
- `eps=1e-6` 零点增益的首步 gradient/update finite 审计；
- K/V、CFN、TPD、NER、decoder 和 legacy eval contract 不变性；
- CPU forward/backward 与 RTX 5090 GPU2/GPU3 smoke；
- exact resume、strict reload、write-once source lock；
- 10,870,228/568 训练态与 10,870,130/564 推理态常量审计；
- A/B 复用审计得到明确 `REUSABLE`，否则重跑 A/B；
- 若复用 A/B，6 个新 C/D checkpoint 完整；若重跑 A/B，则 12 个新 checkpoint 完整；
- 仅所有 arm 的 `best`/`best_mIoU` selector sweep、fixed point 和 finalizer 可复核；
- ordinary Python 与 `python -O` 测试通过。

F-A 失败时不得把该实现并入最终模型，但允许修复工程问题后重新验证。

### Evidence F-B：固定阈值完整比较

对 A/B/C/D 各自 `best` 与 `best_mIoU`，在冻结阈值 `0.5` 下同时报告：

```text
Pd / matched targets
Fa
mIoU
tiny-Pd
unmatched predicted objects
false objects per image
checkpoint epoch / role / file SHA / state SHA
```

固定点用于显示实际工作点和校准偏移，不单独决定成功，也不设置
`mIoU>=0.938178`、`Pd>=188/189` 等绝对放行线。

### Evidence F-C：全局 Pareto 与相对状态

联合比较集合固定为：

```text
baseline / V1 / V2 / V3
TPD V8
NER V4
Control A
TSS B
FG-only C
TSS+FG D
```

对 C 与 D 分别输出：

```text
RELATIVE_IMPROVED
PARETO_MIXED_TRADEOFF
DOMINATED
```

`RELATIVE_IMPROVED` 表示产生可复核的严格相对改善；`PARETO_MIXED_TRADEOFF`
表示至少贡献一个其他方法无法同时在 Pd、Fa、mIoU、tiny-Pd 和错误目标上支配
的独有工作区间；`DOMINATED` 表示 selector sweep 和预算包络均无独有贡献。
“非孤立”必须由相邻阈值点或同一预算包络中的一致工作区间支持，不能由单个
浮点阈值孤点决定。

### Evidence F-D：五预算包络

预算顺序固定为：

```text
[1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
```

V4 两个 selector 的 Pd envelope 为：

```text
[0, 188, 189, 189, 189]
```

其中 V4-best 为 `[0,187,189,189,189]`，V4-best_mIoU 为
`[0,188,188,188,189]`。C/D 必须完整报告全部预算，但不要求每个预算逐项
不退化；严格改善与退化都进入 Pareto 裁决。最严格 `Fa<=1e-6` 必须单独报告。
若未超过 V1 的最严格预算，固定：

```text
strictest_fa_budget_advantage=false
universal_dominance_claim=false
```

### Evidence F-E：seed42 描述性因子效应与机制诊断

在对齐标量估计量上报告第 5 节定义的四个 simple effects、两个 marginal
effects 和 interaction。只允许写“seed42 当前内部划分上的描述性差值”，不得
写显著性或稳定性。

dynamic/shuffled/identity counterfactual、Query change 和错误组件诊断只用于
解释结果，不替代分割输出，不作为阻止模型代码完成的硬 Gate。K/V 边界、
shared-gradient anchor 和 source detach 属于 F-A 工程条件。

### Decision F-F：TSS 最终保留

满足以下任一条件时才在最终训练配方保留 TSS：

1. D 相对 C 贡献 C 无法提供的独有全局 Pareto 工作区间；
2. D 为 `RELATIVE_IMPROVED`，且该改善可由 `D-C` simple effect 复核；
3. B 为 `RELATIVE_IMPROVED` 或 `PARETO_MIXED_TRADEOFF`，同时 D 在 C/D
   联合 frontier 上不被 C 支配并提供独有点。

否则优先使用更简单的 C：

```text
final_training_uses_tss=false
final_inference_uses_tss=false
```

即使保留 TSS，导出时仍删除 `target_survival.*`，所以推理图只保留 QFG。

### Scope F-G：固定单 seed 的结论边界

本方案只运行 seed42，因此 F-G 不是跨 seed Gate，始终固定：

```text
training_seed=42
stability_claim_supported=false
paper_core_established=false
```

### 最终工程选择状态

当 F-A 全部通过，且 C 或 D 至少获得
`RELATIVE_IMPROVED` 或 `PARETO_MIXED_TRADEOFF`：

```text
decision=FINAL_QUERY_FG_STAGE_RELATIVE_PASS
query_fg_stage_success=true
final_model_engineering_selected=true
final_model_established=true
paper_core_established=false
stability_claim_supported=false
```

若 C/D 均为 `DOMINATED`，或 F-A 无法通过：

```text
decision=RETURN_TO_QUERY_FG_OPTIMIZATION
query_fg_stage_success=false
final_model_engineering_selected=false
final_model_established=false
paper_core_established=false
stability_claim_supported=false
```

不得通过修改预算、阈值生成规则、seed42、selector 或 comparator 集合制造通过。

---

## 18. 最终模型选择决策树

### 情形 1：C 通过，D 不优于 C

```text
最终训练：TPD + NER + QFG
TSS：删除
最终推理：SCTransNet + TPD + NER + QFG
```

这是最简洁的成功结果。

### 情形 2：D 通过，C 未通过或 D 严格优于 C

```text
最终训练：TPD + NER + TSS + QFG
最终推理：SCTransNet + TPD + NER + QFG
TSS heads：导出时移除
```

这证明 TSS 具有条件性价值，而不是独立全面优势。

### 情形 3：C、D 都通过且近似等价

优先 C，因为：

- 训练目标更简单；
- 因果解释更清晰；
- 不需要 survival pos-weight 和 auxiliary head；
- 复现成本更低。

只有 D 提供 C 不具备的严格相对或 Pareto 优势时才保留 TSS。

### 情形 4：C、D 都未通过

```text
FG 假设未建立
最终模型回退到已通过相对性能裁决的 NER V4
TSS 根据 sweep 单独标记 RELATIVE_IMPROVED /
PARETO_MIXED_TRADEOFF / DOMINATED
```

不应继续叠加第五个模块。

---

## 19. 失败后的唯一预注册回退

若 QFG-V2-CROA 失败，先根据机制诊断分类，不允许同时改多个变量。

### 19.1 主要问题是 high-frequency hard negatives

证据：

- gate 与背景边缘高度相关；
- Fa 上升而 Pd 不升；
- high-only counterfactual 最差；
- low-frequency 支持区域更稳定。

唯一回退：

```text
QFG-V3 mode=low
其余 CROA 公式、初始化和训练协议全部不变
```

### 19.2 gate 长期接近 identity

证据：

- alpha≈0；
- gate_out norm 很小；
- Query cosine change≈0；
- C/D 与 A/B 几乎完全重合。

先判断频率假设不被采用，不允许直接扩大 factor 到 `(0,2)`。可在独立新版本中考虑每 SCTB 独立 alpha，但不能与 mode 变化同时进行。

### 19.3 seed42 轨迹出现工程异常

优先审计：

- QFG hidden initialization；
- paired crop stream；
- alpha/gate saturation；
- hard-negative frequency correlation；
- first-step trajectory test。

该审计只判断 seed42 任务是否可复核，不得扩写为跨 seed 稳定性结论。

---

## 20. 建议文件修改清单

### 新增模型文件

```text
model/tpd_frequency_gate_v2_croa.py
model/tpd_query_frequency_bridge_v2_croa.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa.py
```

### 新增训练与评估文件

```text
experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_exact.py
experiments/evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pd_fa.py
experiments/launch_tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_formal800.sh
experiments/capture_tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_smoke.py
experiments/finalize_tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa.py
experiments/compare_tss_qfg_v2_croa_factorial.py
experiments/analyze_qfg_v2_croa_mechanism.py
experiments/TPD_NER_V4_QFG_V2_CROA_PROTOCOL.md
```

### 新增测试文件

```text
tests/test_tpd_frequency_gate_v2_croa.py
tests/test_tpd_query_frequency_bridge_v2_croa.py
tests/test_tpd_ner_v4_qfg_v2_croa_integration.py
tests/test_tpd_ner_v4_qfg_v2_croa_survival.py
tests/test_train_tpd_ner_v4_qfg_v2_croa_exact.py
tests/test_tpd_ner_v4_qfg_v2_croa_source_lock.py
```

### 必须保持冻结

```text
model/SCTransNet.py
model/tpd_clean_v8_mprs_dch.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware.py
model/tpd_survival.py
experiments/tpd_training_loss.py
现有 V4、TSS、Control 结果目录
```

---

## 21. 执行顺序

```text
1A. GPU2/3 完成并封存四份 TSS/Control selector sweep
1B. 并行实现 QFG-V2-CROA，不修改冻结文件
2. 输出 TSS=RELATIVE_IMPROVED / PARETO_MIXED_TRADEOFF / DOMINATED
3. 复验共同 V4 parent SHA 与 A/B 完整配置
4. 完成数学、identity、gradient、first-Adam-step 测试
5. 完成 V4+QFG+TSS 整模 CPU smoke
6. 完成 RTX 5090 GPU2/3 smoke
7. 完成 exact resume 与 source lock
8. 审计已有 A/B 是否可精确复用；缺失则补跑
9. 正式运行 C/D × seed42 × 800 epochs
10. 固定阈值、闭区间 sweep、五预算包络和 component audit
11. 运行 2×2 factorial 与机制 counterfactual
12. 执行 Gate F-A 和 Evidence/Decision F-B 至 F-G
13. 决定最终是否保留 TSS
14. 导出无 survival head 的最终推理模型
```

---

## 22. 推荐持久状态

在 TSS sweep 与 QFG 训练前工程闭环均已完成、C/D 正式训练运行时：

```text
decision=FORMAL_QUERY_FG_CD_RUNNING

v8_mprs_dch_frozen=true
ner_v4_tail_aware_frozen=true
ner_relative_improvement_confirmed=true

tss_fixed_threshold_status=MIXED_CONFIRMED
tss_selector_sweep_status=COMPLETE
tss_sweep_finalization_required=false
tss_decision=PARETO_MIXED_TRADEOFF
tss_auxiliary_value_supported=true
tss_universal_improvement=false
tss_failure_established=false

query_fg_v1_component_available=true
query_fg_v2_croa_selected=true
query_fg_engineering_authorized=true
query_fg_pretraining_engineering_gate=PASS
query_fg_source_lock_sha256=22be6273bde0cf6700e850b48148f017ca0170f91a7982881e9427e2d38b3cac
query_fg_formal_training_authorized=true
query_fg_cd_training_status=RUNNING_GPU2_GPU3
query_fg_c_variant=qfg_only
query_fg_c_physical_gpu=2
query_fg_d_variant=tss_qfg
query_fg_d_physical_gpu=3

mainline_changed=false
training_seed=42
final_tss_inclusion_decided=false
final_model_established=false
paper_core_established=false
stability_claim_supported=false
```

训练结束后仍需完成 C/D 的六份 checkpoint、四份 selector sweep、五预算
包络、2×2 描述性效应和最终 TSS 保留裁决；在此之前
`final_model_established=false` 保持不变。

---

## 23. 最终研究判断

TSS 当前的冻结结论是：

> **`PARETO_MIXED_TRADEOFF`：固定阈值表现混合，五预算 Pd envelope 未提升，
> 但完整 sweep 建立了其他对照不能同时支配的连续 Fa–mIoU 工作区间；TSS
> 具有条件性辅助价值，不具有全面改进结论。**

它没有恢复第 189 个目标，说明最后一个瓶颈不只是“目标是否在 stride-16 endpoint 存活”。最终模块应转向 SCTB 内部 Query 的目标—背景可分性，但必须保持严格的 Query-only 边界。

现有 FG 方向是正确的，但正式版本不应直接使用当前 `alpha=0 + random gate logits + tanh factor (0,2)` 方案。QFG-V2-CROA 通过：

- 频率输入 RMS 稳定；
- gate 空间中心化；
- 有界 `(0.5,1.5)` 调制；
- detached frequency conditioning；
- zero-terminal + nonzero-alpha 优化锚定；
- 同父 2×2 因子设计；

将性能风险和因果混杂降到最低。

该方案不能在训练前诚实地保证未知数值一定过线；它把工程正确性落实为不可
放宽的 F-A，把未知性能交给冻结的相对/Pareto 裁决，并确保最终只会产生两种
可信结果：

1. QFG 在 seed42 的工程、Pareto 和预算结果上形成可复核的相对贡献，成为当前固定协议下的最终工程模型；
2. QFG 未通过，项目保留已确认相对改进的 NER V4，而不是以选择性结果强行完成四模块故事。

无论哪种结果，均保持 `paper_core_established=false` 与
`stability_claim_supported=false`，直到未来另有更高层级证据。

---

## 24. 代码依据

- SCTransNet 主干与 SCTB：`model/SCTransNet.py`
- 五节点 NER V4 Tail-Aware：`model/tpd_ner_v8_mprs_dch_v4_tail_aware.py`
- Target Survival heads：`model/tpd_survival.py`
- Structured forward contract：`model/tpd_forward_contract.py`
- Survival/segmentation loss：`experiments/tpd_training_loss.py`
- Query-only FG：`model/tpd_frequency_gate.py`
- Query-only bridge：`model/tpd_query_frequency_bridge.py`
- 仓库：<https://github.com/Arialliy/SCTransNet_main>
