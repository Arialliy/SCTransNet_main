# SCTransNet：TPD8 诊断后完整模型级 GCSF 优化与代码修改方案

> 项目：单帧红外小目标检测
> 基线：SCTransNet
> 冻结主线：`SCTransNet + TPD8-MPRS-DCH + 五节点 NER4 Tail-Aware + QFG2-CROA`
> 训练目标：TSS OFF，仅保留原六路 segmentation BCE
> 正式数据集：NUAA-SIRST、NUDT-SIRST、IRSTD-1K
> 正式协议：seed 42、1000 epochs、每 10 epochs 评估、`best_miou / best_pd`、固定阈值 0.5
> 当前 TPD 裁决：`TPD_INCONCLUSIVE_NO_FORMULA_CHANGE`
> 本轮已诊断候选：**GCSF V1 — Global Constant-Sum Skip Fusion**
> 中文：**全局常系数和跳连重分配融合**
> 零训练裁决：`GCSF_BRANCH_AUDIT_NO_TRAINING_AUTHORIZATION`
> 下一阶段：**六路 deep-supervision 梯度审计**

---

# 0. 执行摘要

## 0.1 当前 TPD8 裁决正确

三数据集完整 TPD8 与关闭七个 TPD residual 的固定点结果如下：

| 数据集 | Full：Pd / Fa / mIoU / nIoU | All-7-off：Pd / Fa / mIoU / nIoU |
|---|---|---|
| NUAA-SIRST | `0.973384 / 1.5435e-5 / 0.796483 / 0.795348` | `0.973384 / 1.5435e-5 / 0.796205 / 0.794804` |
| NUDT-SIRST | `0.990476 / 2.7806e-6 / 0.944406 / 0.946423` | `0.989418 / 2.7117e-6 / 0.944364 / 0.946266` |
| IRSTD-1K | `0.932660 / 1.1729e-5 / 0.660312 / 0.665662` | `0.932660 / 1.1862e-5 / 0.659928 / 0.664742` |

正式证据支持：

```text
关闭七块 residual 没有提升综合性能
NUDT 关闭后少检出 1 个目标和 1 个 tiny target
NUAA 与 IRSTD 的 IoU 略降
不存在持续有害的单块 residual
24/24 counterfactual 均改变模型输出
21/21 dataset-block 组合中 target-region residual RMS > background-region RMS
```

因此：

```text
tpd_formula_change_authorized=false
tpd_residual_disable_authorized=false
tpd_new_mode_authorized=false
tpd8_retained=true
```

这不是“TPD 无效”，而是：

> TPD residual 已经真实参与目标建模，但当前整体性能瓶颈不再位于某一个 TPD block 或其局部公式。

## 0.2 下一步不再修改 TSS、QFG、TPD 局部公式

以下路径应继续冻结：

```text
TPD8-MPRS-DCH 的 K/C/S 与七个 residual
NER4 Tail-Aware 的五节点与 q4→q3→q2
QFG2-CROA 的 Query-only 频率调制
TSS objective = OFF
SCTransNet decoder 与六路输出角色
```

下一阶段应处理：

> **完整模型中原始 encoder identity 与 TPD/QFG/SCTB 变换分支如何汇合并送入 decoder。**

## 0.3 已验证的数据流与待检验的整模型假设

当前最终模型中，每个尺度先执行：

\[
T_i=\operatorname{Reconstruct}_i(\operatorname{SCTB}_i)
\]

随后代码连续两次加入原 encoder feature \(E_i\)：

\[
X_i=T_i+E_i
\]

\[
S_i=X_i+E_i=T_i+2E_i
\]

也就是说，decoder 实际接收：

\[
\boxed{S_i^{current}=T_i+2E_i}
\]

其中：

- \(E_i\)：CNN encoder 原特征；
- \(T_i\)：包含 TPD tokenization、QFG Query 调制及 SCTB 跨尺度交互后的 reconstruct 分支。

上述 `T_i+2E_i` 是已经由代码确认的真实数据流，不是推测。但它是否造成性能
瓶颈仍未被实验建立；现阶段只能把“重复 identity 可能稀释 transformed branch”
作为 GCSF 零训练诊断要检验的假设，不能提前写成失败原因。

四个尺度的 CCA 都接收已经融合的 \(T_i+2E_i\)。NER 则只存在于 L4、L3、L2：
它的 mask 由五节点 evidence、跨层 relay 与 decoder-up 特征独立生成，再乘到对应
CCA skip；L1 只有 CCA，没有 NER。五节点 evidence 还沿独立旁路进入 NER mask
生成，因此不能把全部 TPD 信息都归入 \(T_i\) 分支。

在这个准确的数据流边界内，CCA 与 L4/L3/L2 的 NER 均不能分别控制：

```text
原始局部 identity
跨尺度变换分支
```

因此提出三个**待验证假设**：

1. `2E_i` 稀释 TPD/QFG/SCTB 的目标判别信息；
2. CCA 以及 L4/L3/L2 的 NER 对两条来源不同的分支只能在融合后统一缩放；
3. 不同数据集对局部纹理和全局语义的需求不同，固定 `1:2` 融合比例容易形成 mixed trade-off。

## 0.4 本轮候选与执行结果

建议新增：

> **GCSF V1 — Global Constant-Sum Skip Fusion**
> **全局常系数和跳连重分配融合**

它不再修改任何现有局部模块，而是在四个 decoder skip 入口统一学习：

```text
跨尺度变换分支 T_i
与
重复 encoder identity 分支 E_i
```

之间的通道级重分配。

正式公式：

\[
g_i
=
\eta\tanh(a_i)
\]

\[
\boxed{
S_i^{GCSF}
=
(1+g_i)T_i
+
(2-g_i)E_i
}
\]

其中：

```text
a_i：每个尺度的逐通道可学习参数
初始化：全 0
eta：固定 0.5
```

初始化时：

\[
g_i=0
\]

所以：

\[
S_i^{GCSF}=T_i+2E_i=S_i^{current}
\]

参考实现必须保留旧模型的浮点运算顺序：

```python
baseline = (transformed + encoder) + encoder
correction = gate * transformed - gate * encoder
return baseline + correction
```

初始化时 `gate == 0`，所以有限输入下输出从旧模型的同一 `baseline` 精确起步；
训练随后才允许把部分系数从重复 identity 分支重新分配到变换分支，或反向分配。
不能把公式直接重组为 `(1 + gate) * transformed + (2 - gate) * encoder`，因为
即使 `gate=0`，浮点运算重排也不能保证逐元素零点等价。

该候选的模型代码、训练/推理双图、导出器和受门控训练入口均已实现并通过工程
测试；但 6 个 checkpoint、11 个 mode、共 66 个单元的零训练诊断没有触发训练门。
因此本轮没有启动 200-epoch pilot 或 formal1000，GCSF 不能替换当前完整模型。

---

# 1. 当前阶段的正式研究判断

完成代码与 66 单元诊断后的执行状态为：

```text
pre_revision_decision=DOCUMENT_REVISION_REQUIRED_BEFORE_GCSF_IMPLEMENTATION
decision=GCSF_ZERO_TRAINING_TRIGGER_FAILED
gcsf_zero_training_decision=GCSF_BRANCH_AUDIT_NO_TRAINING_AUTHORIZATION

tpd_decision=TPD_INCONCLUSIVE_NO_FORMULA_CHANGE
tpd8_formula_frozen=true
tpd8_residuals_retained=true

ner4_frozen=true
qfg2_frozen=true
tss_objective_enabled=false

double_identity_dataflow_verified=true
skip_fusion_performance_bottleneck_established=false
full_model_integration_hypothesis_tested=true
global_fixed_skip_reallocation_supported=false
next_candidate=DEEP_SUPERVISION_GRADIENT_AUDIT

gcsf_zero_training_diagnostic_complete=true
gcsf_code_implemented=true
gcsf_training_runner_implemented=true
gcsf_pilot_authorized=false
gcsf_formal_training_authorized=false
gcsf_training_started=false

paper_core_established=false
stability_claim_supported=false
training_recipe_finalized=false
```

当前已经成立的是：

```text
TPD residual 确实在工作
TPD 目标区域响应高于背景
关闭 residual 不能改善综合性能
```

尚未建立的是：

```text
当前固定 T_i + 2E_i 是最优整模型融合
TPD/QFG/SCTB 信息已经充分进入 decoder
四个尺度应使用相同固定分支比例
```

---

# 2. 代码级完整数据流分析

## 2.1 SCTransNet 原始 ChannelTransformer

`ChannelTransformer.forward()` 已经执行：

```python
x1 = reconstruct_1(encoded1)
...
x1 = x1 + en1
```

即：

\[
T_i+E_i
\]

## 2.2 SCTransNet 主 forward 再次加入 encoder

外层 `SCTransNet.forward()` 又执行：

```python
f1 = x1
...
x1, x2, x3, x4, _ = self.mtc(...)
x1 = x1 + f1
```

因此：

\[
(T_i+E_i)+E_i=T_i+2E_i
\]

## 2.3 完整 TPD8–NER4–QFG2 同样保留双 identity

当前完整模型显式 forward 中也是：

```python
x1 = self.mtc.reconstruct_1(encoded1) + f1
...
x1, x2, x3, x4 = x1 + f1, ...
```

所以双 identity 不是历史分析中的推测，而是正式完整模型当前真实数据流。

## 2.4 CCA、NER 与五节点 evidence 的准确作用位置

四个尺度的共同主路径是：

```text
T_i + 2E_i
→ CCA channel attention
```

其后并非四个尺度都经过 NER：

```text
L4：CCA skip → NER(q4) → decoder
L3：CCA skip → NER(q3, relay, decoder-up) → decoder
L2：CCA skip → NER(q2, relay, decoder-up) → decoder
L1：CCA skip → decoder；没有 NER
```

NER mask 不是从 `T_i+2E_i` 单一路径直接推导。它还接收由 TPD 五节点产生的
evidence，并在 L3/L2 结合跨层 relay 与 decoder-up 特征独立生成 mask。该 mask
最终乘到已经完成 CCA 的融合 skip：

\[
\operatorname{skip}_{att}(T_i+2E_i)
\]

不能做：

```text
对 E_i 抑制
同时对 T_i 保留
```

或者：

```text
对 E_i 保留
同时对 T_i 降低
```

因此，GCSF 只处理进入 CCA 前的 transformed/encoder 融合比例；它不替代
TPD 五节点 evidence→NER 的旁路，也不修改 NER 的 L4→L3→L2 递进关系。

---

# 3. GCSF V1 模型设计

## 3.1 语义定义

对尺度 \(i\in\{1,2,3,4\}\)：

\[
E_i
=
\text{CNN encoder feature}
\]

\[
T_i
=
\text{reconstructed SCTB feature}
\]

其中 \(T_i\) 已经包含：

```text
TPD8 对浅层 tokenization 的影响
QFG2 对 Query 的调制
SCTB 的跨尺度通道交互
CFN 的局部增强
reconstruct 上采样
```

当前：

\[
S_i=T_i+2E_i
\]

GCSF 定义：

\[
g_i
=
0.5\tanh(a_i)
\]

\[
S_i
=
(1+g_i)T_i
+
(2-g_i)E_i
\]

## 3.2 系数范围

解析上，有限 logit 经过 `tanh` 后满足：

\[
g_i\in(-0.5,0.5)
\]

所以：

\[
1+g_i\in(0.5,1.5)
\]

\[
2-g_i\in(1.5,2.5)
\]

两条分支系数始终为正，不允许：

```text
分支符号翻转
彻底删除 identity
彻底删除 transformed branch
```

工程验证采用包含浮点饱和端点的闭区间合同：

```text
gate ∈ [-0.5, 0.5]
transformed coefficient ∈ [0.5, 1.5]
encoder coefficient ∈ [1.5, 2.5]
```

这是数值边界，不表示有限实数 logit 在解析上必然取得 `tanh` 的端点。

## 3.3 常系数和性质

每个通道满足：

\[
(1+g_i)+(2-g_i)=3
\]

这表示分支的**系数和**保持与当前模型一致。

需要准确说明：

> GCSF 保持的是融合系数和，不保证输出激活范数或能量严格保持，因为 \(T_i\) 与 \(E_i\) 可能相关且尺度不同。

不能将其误写成严格的 activation-mass conservation。

## 3.4 参数量

默认通道：

```text
32 + 64 + 128 + 256 = 480
```

因此新增：

```text
480 个可学习标量
4 个 state keys
0 个 persistent buffer
```

GCSF 独立增量固定为 480 参数、4 个 state keys、0 个 persistent buffer。整合后
必须由构造器测试得到以下精确规模，不能只报告近似值：

```text
含 TSS heads 的训练模型：10,870,708 parameters，572 state keys
无 TSS heads 的推理模型：10,870,610 parameters，568 state keys
其中 GCSF：480 parameters，4 state keys，0 persistent buffer
```

相对不含 GCSF 的推理模型 10,870,130 参数，参数增量为：

\[
480/10{,}870{,}130
\approx0.0044\%
\]

## 3.5 零点和一阶优化锚定

初始化：

```text
a_i = 0
```

得到当前完整模型的精确输出。

对共享参数，初始 forward 系数仍为：

```text
T branch = 1
E branch = 2
```

参考 forward 固定写成：

```python
baseline = (transformed + encoder) + encoder
correction = gate * transformed - gate * encoder
return baseline + correction
```

而不是重组后的系数乘法。这样在 `gate=0` 且输入有限时，从旧模型的同一
`baseline` 起步。对共享参数，第 0 步的局部导数仍为：

\[
\frac{\partial S_i}{\partial T_i}=1+g_i=1,
\qquad
\frac{\partial S_i}{\partial E_i}=2-g_i=2
\]

所以在相同 batch、loss、训练模式、RNG 和确定性算子下，并且不存在依赖全参数
集合的全局梯度裁剪或其他耦合变换时：

```text
共享参数 forward 相同
共享参数 gradient 相同
第一次 Adam 更新后的共享参数及共享 optimizer state 相同
```

新参数的梯度为：

\[
\frac{\partial S_i}{\partial a_i}
=
0.5(T_i-E_i)
\]

因此只有 GCSF gate 在第一步开始学习。该结论只覆盖第一个更新；从第二个
forward 开始 gate 已可能非零，不能声称后续共享训练轨迹仍相同。

这一设计原则与零初始化 residual scaling / LayerScale 类思想一致，但 GCSF 的对象不是单一 residual block，而是 SCTransNet 中四尺度双分支 skip 的常系数和重分配。

---

# 4. 为什么 GCSF 符合当前证据

## 4.1 TPD residual 有目标信息，不应删除

21/21 dataset-block 组合中：

```text
target residual RMS > background residual RMS
```

说明变换分支中确实存在目标判别信息。

GCSF 不是删除这些 residual，而是解决：

```text
这些信息进入 decoder 时是否被 2× identity 稀释
```

## 4.2 All-7-off 不改善，说明局部 residual 不是主要噪声源

若 TPD residual 是主要虚警源，关闭全部 residual 应在至少部分数据集显著降低 Fa、pixel FP 或提高综合 IoU。

实际没有出现这一现象。

因此下一步不应继续寻找“有害 TPD block”，而应审查：

```text
有益 TPD/SCTB 信息在整模型中如何被融合
```

## 4.3 mixed trade-off 符合固定融合比例问题

现有模型在不同数据集表现为：

```text
某些数据集 Pd/nIoU 更好
某些工作点 Fa 更好
部分 mIoU 或 tiny-Pd 退化
```

这与固定分支比例可能产生的域差异一致：

- NUAA 可能更依赖局部形状/纹理；
- NUDT 可能更能从跨尺度语义获益；
- IRSTD 复杂背景下需要同时保持局部边界和全局目标–杂波可分性。

逐通道、逐尺度的有界重分配比统一删除第二条 identity 更安全。

---

# 5. 正式训练前的零训练整模型诊断

在编写 GCSF formal runner 前，应先使用现有三个 TSS-off `best_miou / best_pd` checkpoint 完成 branch audit。

## 5.1 提取四尺度分支

对每个尺度保存：

```text
E_i：encoder feature
T_i：reconstruct(encoded_i)
S_i：T_i + 2E_i
```

不修改 checkpoint。

所有空间统计和最终指标必须复用冻结协议：数据集各自 `img_idx` test split、原图
有效像素 support mask、同一 padding/crop 还原方式、同一目标连通域匹配语义和
固定阈值 0.5。padding 像素不得进入 RMS、Fa、pixel FP、mIoU 或 nIoU 的分子与
分母。下采样到各尺度的 target/background mask 必须由原始有效 mask 以冻结规则
生成，并保存 mask 像素数供回放校验。

## 5.2 统计指标

逐数据集、逐尺度记录：

### 分支尺度

\[
r_i
=
\frac{\operatorname{RMS}(T_i)}
{\operatorname{RMS}(E_i)+\epsilon}
\]

### 分支方向

\[
c_i
=
\operatorname{cosine}(T_i,E_i)
\]

### 目标/背景对比

\[
q_i^T
=
\frac{
\operatorname{RMS}(T_i\mid target)
}{
\operatorname{RMS}(T_i\mid background)+\epsilon
}
\]

\[
q_i^E
=
\frac{
\operatorname{RMS}(E_i\mid target)
}{
\operatorname{RMS}(E_i\mid background)+\epsilon
}
\]

除尺度汇总值外，必须按同一公式保存逐通道 `qT[i,c]` 与 `qE[i,c]`，供 Gate G-E
计算 gate 与目标/背景分支优势的对齐；不得从尺度均值反推通道结果。

### 当前融合中的有效份额

\[
p_i^T
=
\frac{\operatorname{RMS}(T_i)}
{\operatorname{RMS}(T_i)+2\operatorname{RMS}(E_i)+\epsilon}
\]

## 5.3 零训练主 counterfactual 矩阵

主矩阵只能使用 GCSF V1 在有限 logit 下可表示、且位于边界内部的固定 gate：

```text
R0：g=0，四尺度均为当前 (T+E)+E                         1 mode
N1...N4：仅对应单尺度 g=-0.25，其余尺度 g=0             4 modes
N-all：四尺度 g=-0.25                                  1 mode
P1...P4：仅对应单尺度 g=+0.25，其余尺度 g=0             4 modes
P-all：四尺度 g=+0.25                                  1 mode
合计：每 checkpoint 11 modes
```

三个数据集各有 `best_miou` 与 `best_pd` 两个 checkpoint，因此固定为：

```text
6 checkpoints × 11 modes = 66 evaluation cells
```

所有 mode 在同一次 dataloader 回放中共用输入与 mask；`R0` 必须与原模型输出及
冻结 summary 逐项重放一致。每个 mode 结束后恢复 gate/state，并校验 SHA。

`T+E` 的系数和为 2，`2T+E` 等价于 `g=1` 且超出 V1 gate 范围；二者若保留，只能
作为描述性压力测试，不能参与 Trigger、选择方向或授权正式训练。`1.5T+1.5E`
位于 `g=0.5` 的解析边界，也不进入 66-cell 主矩阵，不能替代 `g=+0.25`。

这些固定权重结果只用于零训练诊断，不作为新模型正式结果。

## 5.4 冻结的 safe / material / severe 规则

主裁决使用三个 `best_miou` checkpoint；`best_pd` 只作为高召回辅助审计和
severe veto，不能用其单独 Pd 上升替代主门。对同一 checkpoint 的候选 mode
相对 `R0` 定义：

```text
Δtarget = matched_target_count(candidate) - matched_target_count(R0)
Δtiny   = matched_tiny_target_count(candidate) - matched_tiny_target_count(R0)
ΔmIoU   = mIoU(candidate) - mIoU(R0)
ΔnIoU   = nIoU(candidate) - nIoU(R0)

Rcomp = (R0.unmatched_predicted_pixels - candidate.unmatched_predicted_pixels)
        / R0.unmatched_predicted_pixels
Rbg   = (R0.false_positive_pixels - candidate.false_positive_pixels)
        / R0.false_positive_pixels
```

这里 `Rcomp` 使用 unmatched predicted **pixels**，`Rbg` 使用所有有效
GT-background 像素上的 false-positive pixels；object count 只作描述。冻结条件：

```text
safe（全部满足）：
Δtarget > -2，Δtiny > -2
ΔmIoU > -0.005，ΔnIoU > -0.005
Rcomp > -0.05，Rbg > -0.05

material（至少一项）：
Δtarget >= 2 或 Δtiny >= 2
ΔmIoU >= 0.005 或 ΔnIoU >= 0.005
Rcomp >= 0.05 或 Rbg >= 0.05

safe_material_improvement = safe AND material

severe（任一项）：
Δtarget <= -2 或 Δtiny <= -2
ΔmIoU <= -0.01 或 ΔnIoU <= -0.01
Rcomp <= -0.25 或 Rbg <= -0.25
```

若 reference FP 为 0：candidate 也为 0 时 reduction 记 0；candidate 大于 0 时
reduction 写 JSON `null`，同时 `safe=false`、`severe=true`。全部门使用 JSON
未舍入值，不添加隐藏容差。

## 5.5 GCSF 实现与训练触发门

只有同一个非零 mode 在三个 primary `best_miou` 单元中达到：

```text
safe_material_improvement >= 2/3
severe == 0/3
```

且该 mode 在三个 `best_pd` 辅助单元中 `severe == 0/3`，才由性能反事实直接
授权 GCSF 代码实现。方向不同但同属 GCSF 可表示范围的模式若各自在不同数据集
获得 safe-material、却没有一个模式达到 2/3，只能记录
`fixed_ratio_is_domain_sensitive=true`，不能按数据集挑 mode 后拼成一次通过。

`q_i^T`、`q_i^E` 与 `p_i^T` 等分支统计只用于解释候选方向和定位尺度，均属于
描述性证据。由于 `q_i^T > q_i^E` 没有预注册效应量门槛，且分支幅值不等价于
最终 `Pd-Fa-mIoU-nIoU` 改善，它们不能单独授权 scratch pilot 或 formal1000。
即使为完成零点等价、导出和状态字典测试而预先落地 GCSF 工程代码，也不表示
训练门已经通过。V1 唯一的训练触发条件仍是上面的同一非零 mode 量化性能门。

若两类触发均不成立，则：

```text
decision=GCSF_ZERO_TRAINING_TRIGGER_FAILED
gcsf_formal_training_authorized=false
next_step=DEEP_SUPERVISION_GRADIENT_AUDIT
```

---

# 6. 代码修改方案

## 6.1 新增独立模块

新增：

```text
model/tpd_global_constant_sum_skip_fusion.py
```

下面只保留公式级简化示意；类型、形状、device/dtype、有限值、固定
`gate_limit=0.5`、state-key 与 manifest 的权威可执行合同，以已落地的
`model/tpd_global_constant_sum_skip_fusion.py` 为准，不能把此简化段直接复制为
正式实现：

```python
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class GlobalConstantSumSkipFusion(nn.Module):
    """Channel-wise reallocation between transformed and encoder branches."""

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128, 256),
        *,
        gate_limit: float = 0.5,
    ) -> None:
        super().__init__()

        normalized = tuple(int(value) for value in channels)
        if len(normalized) != 4 or any(value < 1 for value in normalized):
            raise ValueError(
                "channels must contain four positive channel counts"
            )
        if gate_limit <= 0.0 or gate_limit >= 1.0:
            raise ValueError("gate_limit must lie strictly in (0, 1)")

        self.channels = normalized
        self.gate_limit = float(gate_limit)
        self.reallocation_logits = nn.ParameterList(
            nn.Parameter(torch.zeros(1, channels_i, 1, 1))
            for channels_i in self.channels
        )

    def gate(self, level: int) -> torch.Tensor:
        if level < 0 or level >= len(self.reallocation_logits):
            raise IndexError(f"invalid fusion level {level}")
        return self.gate_limit * torch.tanh(
            self.reallocation_logits[level]
        )

    def forward_level(
        self,
        level: int,
        transformed: torch.Tensor,
        encoder: torch.Tensor,
    ) -> torch.Tensor:
        if transformed.shape != encoder.shape:
            raise ValueError(
                "transformed and encoder branches must have equal shapes"
            )
        if transformed.ndim != 4:
            raise ValueError("skip branches must be BCHW tensors")
        if transformed.shape[1] != self.channels[level]:
            raise ValueError(
                f"level {level} expected C={self.channels[level]}, "
                f"got C={transformed.shape[1]}"
            )

        gate = self.gate(level).to(
            device=transformed.device,
            dtype=transformed.dtype,
        )

        # Preserve the production reference's floating-point operation order.
        baseline = (transformed + encoder) + encoder
        correction = gate * transformed - gate * encoder
        return baseline + correction

    def forward(
        self,
        transformed: Sequence[torch.Tensor],
        encoder: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        if len(transformed) != 4 or len(encoder) != 4:
            raise ValueError("GCSF requires exactly four scales")

        return tuple(
            self.forward_level(level, transformed_i, encoder_i)
            for level, (transformed_i, encoder_i) in enumerate(
                zip(transformed, encoder)
            )
        )
```

## 6.2 新增训练/推理双类、构造器与 exporter

新增：

```text
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_gcsf.py
experiments/export_tpd_ner_v4_qfg_v2_croa_gcsf_to_inference.py
```

模型文件同时定义保留 TSS heads 的训练类和删除 TSS heads 的正式推理类，并共用
唯一 `_forward_with_relay` 实现；训练 objective 固定 TSS-off。不要覆盖当前模型文件；
统一构造器明确选择 train/inference 角色，exporter 只删除 4 个
`target_survival.*`，完整保留 4 个 `global_skip_fusion.*` 与 20 个 QFG state keys。
模型工厂、训练 runner、evaluator 没有各自复制结构定义。

## 6.3 修改 forward 的唯一位置

当前：

```python
x1 = self.mtc.reconstruct_1(encoded1) + f1
x2 = self.mtc.reconstruct_2(encoded2) + f2
x3 = self.mtc.reconstruct_3(encoded3) + f3
x4 = self.mtc.reconstruct_4(encoded4) + f4

x1, x2, x3, x4 = x1 + f1, x2 + f2, x3 + f3, x4 + f4
```

新模型：

```python
t1 = self.mtc.reconstruct_1(encoded1)
t2 = self.mtc.reconstruct_2(encoded2)
t3 = self.mtc.reconstruct_3(encoded3)
t4 = self.mtc.reconstruct_4(encoded4)

x1, x2, x3, x4 = self.global_skip_fusion(
    (t1, t2, t3, t4),
    (f1, f2, f3, f4),
)
```

此后：

```text
CCA
NER
decoder
deep supervision
```

完全不变。

## 6.4 TSS 训练合同

正式 GCSF 训练使用：

```text
survival_weight = 0
tss_objective_enabled = false
```

为最大限度减少变量，可以继续使用含 TSS heads 的训练 class，但 loss 不消费 survival logits。

部署导出仍删除：

```text
target_survival.*
```

GCSF 参数必须保留在推理模型中。

## 6.5 Architecture manifest

新增：

```python
{
    "global_skip_fusion": "gcsf_v1",
    "global_skip_fusion_levels": 4,
    "global_skip_fusion_channels": (32, 64, 128, 256),
    "global_skip_fusion_gate": "0.5*tanh(channel_logit)",
    "transformed_coefficient": "1+gate",
    "encoder_coefficient": "2-gate",
    "coefficient_sum": 3.0,
    "coefficient_sum_is_constant": True,
    "activation_norm_preserved": False,
    "initial_reference": "current_t_plus_2e_exact",
    "parameters_added": 480,
    "buffers_added": 0,
    "tpd_formula_changed": False,
    "ner_formula_changed": False,
    "qfg_formula_changed": False,
    "tss_objective_enabled": False,
}
```

## 6.6 State keys

建议固定：

```text
global_skip_fusion.reallocation_logits.0
global_skip_fusion.reallocation_logits.1
global_skip_fusion.reallocation_logits.2
global_skip_fusion.reallocation_logits.3
```

旧完整模型 checkpoint 只可用于零训练 audit 和 strict extension 工程测试：

```text
旧 key 全部严格匹配
仅允许新增四个 GCSF key
四个新 key 必须全零
```

它不得作为 GCSF pilot/formal 的 parent 或 warm start。正式候选必须从 seed42 的
scratch 初始化独立训练。

---

# 7. 单元测试与工程门

## 7.1 零点 forward 等价

当四个 logits 为 0：

```text
GCSF output
== production reference: (T + E) + E
```

必须使用相同运算顺序并逐元素相等；禁止拿重组后的 `T + 2*E` 作为 exact oracle。

整模型六个训练输出与当前 TSS-off 模型逐元素一致。

## 7.2 共享梯度等价

在 step 0：

```text
所有旧参数 gradient
==
当前 TSS-off 模型 gradient
```

新 GCSF 参数允许非零梯度。

该测试固定相同 batch、loss、train/eval 状态与 RNG，并显式确认训练配置不存在
global-norm clipping 等依赖新增 gate 梯度的全参数耦合操作。

## 7.3 第一次 Adam step

相同 batch、相同 optimizer：

```text
所有共享 model state 相同
所有共享 optimizer state 相同
仅 GCSF 参数开始分化
```

“相同”仅承诺第一次更新后的共享参数与共享 Adam state；不延伸到第二次 forward。

## 7.4 系数边界

验证：

```text
gate ∈ [-0.5, 0.5]
transformed coefficient ∈ [0.5, 1.5]
encoder coefficient ∈ [1.5, 2.5]
coefficient sum == 3
```

## 7.5 梯度有限

覆盖：

```text
T == E
T != E
全零输入
极大有限输入
FP32
```

## 7.6 Strict extension load

验证：

```text
当前 TSS-off checkpoint
→ GCSF model strict extension load
→ 所有旧 state 完全一致
→ 新 state 全零
```

## 7.7 Exact resume

连续运行与 epoch 边界续训必须在以下项目一致：

```text
model tensors
optimizer tensors
scheduler
RNG
DataLoader generator
best_miou selector
best_pd selector
GCSF diagnostics accumulator
```

## 7.8 推理导出

验证：

```text
训练模型 eval segmentation output
==
无 TSS heads 的 GCSF inference model output
```

推理 state：

```text
保留 global_skip_fusion.*
删除 target_survival.*
保留 TPD / NER / QFG
```

并严格断言：

```text
training model = 10,870,708 parameters / 572 state keys
inference model = 10,870,610 parameters / 568 state keys
GCSF delta = 480 parameters / 4 state keys / 0 persistent buffer
```

## 7.9 普通模式与 `python -O`

所有 contract 测试必须在：

```text
普通 Python
python -O
```

两种模式通过。

---

# 8. GCSF 机制日志

训练过程中每 10 epoch 保存：

## 8.1 Gate 分布

每个尺度：

```text
gate mean
gate std
gate p10
gate p50
gate p90
gate min
gate max
positive-gate channel fraction
negative-gate channel fraction
near-zero channel fraction
```

## 8.2 分支系数

```text
transformed coefficient mean
encoder coefficient mean
```

## 8.3 分支激活

```text
RMS(T_i)
RMS(E_i)
RMS(fused_i)
cos(T_i, E_i)
```

## 8.4 目标/背景对比

```text
T target/background RMS ratio
E target/background RMS ratio
fused target/background RMS ratio
```

## 8.5 与错误组件关联

记录：

```text
false component 区域 gate channel response
matched target 区域 gate channel response
pixel FP 与 transformed emphasis 的相关性
missed target 与 encoder emphasis 的相关性
```

这些机制指标不作为 checkpoint 选择条件，只用于验证设计解释。

---

# 9. 200-epoch durable pilot

## 9.1 Pilot 矩阵

| 数据集 | 候选 | TSS | Seed | Formal schedule | Pause |
|---|---|---:|---:|---:|---:|
| NUAA-SIRST | GCSF V1 | off | 42 | 1000 | epoch 200 |
| NUDT-SIRST | GCSF V1 | off | 42 | 1000 | epoch 200 |
| IRSTD-1K | GCSF V1 | off | 42 | 1000 | epoch 200 |

使用 1000-epoch 正式 scheduler，仅在 epoch 200 durable pause。

三个 run 的初始化合同固定为：

```text
seed=42
scratch=true
parent_checkpoint=None
warm_start=false
```

旧 `best_miou / best_pd` 权重只服务第 5 节零训练 audit 与 extension-load 测试，
绝不作为 pilot 初始权重。epoch 10 起按正式协议每 10 epochs 正常评估并更新
`best_miou`、`best_pd`；epoch 200 只是 durable pause，不重置 selector，也不能
丢弃 epoch 10–190 已产生的最佳权重。

## 9.2 Pilot 只检查

```text
训练有限
gate 不全部饱和
gate 不全部保持零
共享分支无 NaN/Inf
Pd/tiny 不灾难性坍塌
Fa/pixel FP 不爆炸
exact resume 可继续
```

Pilot 不用于：

```text
选择 gate_limit
修改 eta=0.5
改变训练协议
形成论文性能结论
```

这里“不形成性能结论”不等于停止选模；pilot 必须从 epoch 10 开始执行正常
checkpoint selector，续训到 1000 epochs 时沿用同一 selector state。

---

# 10. Formal1000 实验矩阵

Pilot 通过后继续同一三个 run 到 1000 epochs。

| 数据集 | 模型 | Seed | Epochs | TSS |
|---|---|---:|---:|---:|
| NUAA-SIRST | TPD8+NER4+QFG2+GCSF | 42 | 1000 | off |
| NUDT-SIRST | TPD8+NER4+QFG2+GCSF | 42 | 1000 | off |
| IRSTD-1K | TPD8+NER4+QFG2+GCSF | 42 | 1000 | off |

三者均为同一条 scratch run 从 epoch 0 开始、在 epoch 200 durable pause 后 exact
resume 至 epoch 1000；不是从历史当前模型 checkpoint 续训。固定：

```text
seed=42
scratch=true
parent_checkpoint=None
warm_start=false
evaluate_every=10
selector_start_epoch=10
```

比较对象：

```text
Original SCTransNet
当前完整 TPD8+NER4+QFG2，TSS-off
新 GCSF V1，TSS-off
```

如现有 TSS-off 与当前正式协议、初始化和数据哈希完全一致，可复用。

否则需要同步运行同样 scratch/seed42/schedule 的配对 control：

```text
当前完整模型 TSS-off
GCSF V1 TSS-off
```

此时总新 run 数从 3 增加为 6。

---

# 11. 正式性能 Gate

本轮必须联合考察：

```text
Pd
tiny-Pd
component-Fa
pixel FP
pixel precision
pixel F1
mIoU
nIoU
错误目标数
```

不能使用单一 mIoU 裁决。

## Gate G-A：工程闭环

```text
3 个 formal1000 完整
6 个 best_miou/best_pd checkpoint 可 strict load
source lock 完整
exact resume 通过
普通模式与 python -O 通过
推理导出等价
seed42 scratch init attested
parent_checkpoint is None
warm_start is false
epoch10 起 selector state 连续且 epoch200 pause 后未重置
```

## Gate G-B：primary `best_miou` 性能门

以当前完整 TSS-off 的同数据集 `best_miou` 为 reference，严格复用第 5.4 节的
valid mask、计数、safe/material/severe 公式。三个数据集必须满足：

```text
safe_material_improvement >= 2/3
severe == 0/3
```

这是 GCSF 是否提升当前完整模型的主门。所有指标使用固定阈值 0.5；Pd 必须以
`matched_target_count / total_target_count` 同时报告，不能只报小数或只看数量。

## Gate G-C：辅助 `best_pd` 严重退化否决门

以当前完整 TSS-off 的同数据集 `best_pd` 为 reference，仍使用第 5.4 节的 severe
定义，要求：

```text
severe == 0/3
```

`best_pd` 只负责高召回工作点的 severe veto；它不需要达到 2/3 safe-material，
也不能用 Pd 增长覆盖 Fa、pixel FP、mIoU、nIoU 或 tiny-Pd 的严重退化。

## Gate G-D：相对 Original 的最终模型底线

对 Original 的同数据集同角色 checkpoint，在 `3 datasets × 2 roles` 的六个单元
按第 5.4 节同向计算，要求：

```text
severe == 0/6
primary best_miou safe_material_improvement >= 2/3
```

因此 GCSF 不能只改善当前模型的一项弱点，却把最终模型推到 Original 的严重退化
区。若历史 Original 与本轮数据哈希、seed42、split、selector 或 evaluator 不完全
一致，则 G-D 不得混比，必须先补同协议 scratch control。

Pareto、rank 和阈值扫描继续输出，但全部降为描述项：候选总体、量化精度、空预测
端点和每个非支配工作点必须显式记录，任何 Pareto 点数都不能替代 G-B/G-C/G-D。

## Gate G-E：可量化的机制声称门

在三个 GCSF `best_miou` checkpoint 上定义：

```text
active_level：该尺度 mean(abs(gate)) >= 0.005
positive_channel：gate >= +0.005
negative_channel：gate <= -0.005
bidirectional_level：positive fraction >= 0.05 且 negative fraction >= 0.05
```

机制声称至少要求：

```text
至少 2/3 数据集各有 >=2 个 active_level
至少 2/3 数据集各有 >=1 个 bidirectional_level
在至少 2/3 数据集上：
median(gate | qT-qE > 0) > median(gate | qT-qE <= 0)
```

最后一项按通道使用第 5.2 节同一有效 mask 计算；若任一分组为空，则该数据集不计
通过。G-E 失败但 G-A/B/C/D 通过时，只能写
`performance_candidate_supported=true, mechanism_claim_supported=false`，不得否决
真实性能收益，也不得声称“变换分支稀释机制”已经建立。若 gate 全部近零，则记录
`skip_fusion_performance_bottleneck_established=false`。

## Gate G-F：创新边界

通过后可表述：

> 提出一种零点锚定的四尺度常系数和分支重分配融合，以协调 SCTransNet 原始局部 encoder identity 与 TPD/QFG/SCTB 变换分支。

不能表述为：

```text
严格保持激活能量
动态空间目标注意力
完全消除 encoder bias
```

因为 GCSF V1 是逐通道静态可学习重分配，不是空间动态门。

---

# 12. 裁决树

## 情况 A：GCSF 通过全部 Gate

```text
decision=GCSF_V1_SEED42_TEST_SELECTED_PASS
global_skip_fusion_supported=true
full_model_integration_improved=true
```

下一步：

```text
冻结 TPD8
冻结 NER4
冻结 QFG2
冻结 GCSF
保持 TSS-off
进入论文级复核
```

## 情况 B：Gate 学到非零，但只在单数据集改善

```text
decision=GCSF_DOMAIN_SENSITIVE_MIXED_TRADEOFF
```

解释：

```text
固定 encoder/transformed 融合确有域敏感性
但一个静态全局 channel gate 不能统一解决
```

不应按数据集选择不同 GCSF 参数作为统一模型。

后续只有在机制证据明确时，才考虑：

```text
NER evidence-conditioned spatial reallocation
```

不能直接启动。

## 情况 C：所有 gate 近零，性能等于当前模型

```text
decision=GLOBAL_SKIP_RATIO_NOT_PRIMARY_BOTTLENECK
```

下一步进入：

```text
六路深监督梯度审计
```

而不是回到 TPD/TSS/QFG 局部公式。

## 情况 D：gate 强调 transformed branch 后 Pd 提升但 Fa 上升

说明：

```text
跨尺度分支目标信息有效
但背景调制不足
```

下一步应先分析：

```text
GCSF gate × NER mask 的联合响应
```

而不是扩大 gate_limit 或增加第二个分支。

## 情况 E：GCSF 被当前 TSS-off 全面覆盖

```text
decision=GCSF_V1_REJECTED
```

回退当前完整模型，不修改历史结构。

随后进入第 13 节的整模型训练目标审计。

---

# 13. DS-GA V1：六路 Deep-Supervision 梯度审计合同

GCSF 的 66-cell 零训练分支审计未授权训练后，下一步仍保持：

```text
model = TPD8 + NER4 + QFG2
TSS objective = OFF
mainline_changed = false
loss_formula_changed = false
```

本节是 **train-only 开发诊断**。它只回答当前等权六路监督是否形成跨数据集、
跨 checkpoint 角色的持续梯度冲突；不使用 `img_idx/test` 选择 head，不产生新模型
性能，也不能直接授权修改 loss 或启动 DS V2 训练。

当前冻结目标保持：

\[
L_{seg}=L_{gt5}+L_{gt4}+L_{gt3}+L_{gt2}+L_{d0}+L_{final}
\]

六项均为 production sigmoid probability 上的 `BCELoss(reduction="mean")`，顺序
固定为 `gt5, gt4, gt3, gt2, d0, final`。审计期间禁止换成 logits BCE、Dice、IoU
loss，禁止改变 reduction、head 权重或 TSS 权重。

## 13.1 审计对象与 checkpoint 合同

固定使用当前完整 TSS-off 模型的六份 seed42 checkpoint：

```text
NUAA-SIRST：best_miou + best_pd
NUDT-SIRST：best_miou + best_pd
IRSTD-1K：best_miou + best_pd
```

`best_miou` 是 primary 梯度裁决角色，`best_pd` 是跨 checkpoint 稳健性复核角色。
六份权重必须复用 GCSF audit 已锁定的路径、文件 SHA 和 state-dict SHA；不得使用
GCSF 权重，因为 GCSF 没有获准训练，也没有训练 checkpoint。

数据仅来自各数据集自己的：

```text
img_idx/train_NUAA-SIRST.txt
img_idx/train_NUDT-SIRST.txt
img_idx/train_IRSTD-1K.txt
```

程序必须验证 train index 文件 SHA、ordered-ID SHA、图像/mask、normalization 和
correction manifest。测试集指标与标签不得进入 batch 抽取、head 选择或 Trigger。
本阶段仍属于 seed42 内部开发诊断，不支持跨随机性结论。

## 13.2 可复现 required/conditional 分层 batch 抽取

### 13.2.1 Audit epoch

schema namespace 固定为：

```text
sctransnet-ds-gradient-audit-v1
```

对整数 epoch `1...1000` 计算：

```text
sha256(namespace, 42, dataset, epoch)
```

按 digest 升序取前 32 个 epoch。每个 epoch 使用现有
`ThreeDatasetV2TrainDataset` 与 `derive_stateless_transform_plan`，从 seed42、dataset、
epoch 和 namespaced sample ID 恢复原正式 crop、flip、transpose；不得另写随机增强。

此外，required stratum 的 natural distinct-source ceiling 必须在正式
`epoch=1...1000` 上完整穷举，不能用前 32 个 audit epoch 推测。穷举结果保存 source
ID 集合、逐 source candidate count、canonical proof JSON SHA 和总数；若触发下述
自然可用性修正，batch 候选也从该份 `1...1000` 穷举 proof 中确定性选择。

### 13.2.2 Stratum 定义与可用性

只根据原图 GT mask 的 8 连通域和 crop 几何分类，不读取模型输出或 loss：

```text
required strata：
  tiny_positive：至少相交一个 component，且所有相交 component 的原图面积 <= 9
  normal_positive：至少相交一个原图面积 > 9 的 component

availability-conditional descriptive stratum：
  background_only：合法正式 crop 不与任何 GT component 相交
```

若 normal crop 同时包含 tiny component，仍归入 `normal_positive`，并记录
`mixed_tiny=true`。tiny/normal 使用 component 的**原图面积**，不能用被 crop 截断后
的面积，以免把普通目标误标成 tiny。分类优先级固定为
`normal_positive > tiny_positive > background_only`，不能为了补 tiny 配额把 mixed
crop 改判为 tiny。

`tiny_positive` 与 `normal_positive` 在三个数据集均为 required。`background_only`
只能来自正式 stateless crop 自然产生的背景子窗，禁止人为挖除目标、缩放 mask、
平移 crop 到协议外位置或生成合成背景。

正式数据事实是：NUDT-SIRST 的 663 张 train 图像均为 `256×256` 且 GT 非空；在
正式 `patch_size=256` 下每张图像只有一个完整合法 crop，因此其
`background_only` 自然候选严格为 0。NUDT 必须记录：

```text
background_only.available=false
background_only.structurally_unavailable=true
background_only.candidate_count=0
background_only.reason="all_663_train_images_are_256x256_and_gt_nonempty"
```

NUAA 与 IRSTD-1K 也必须先按相同正式协议枚举自然候选；不能预设它们必然可用。

### 13.2.3 选择与组 batch

每个候选 `(epoch, namespaced_id, transform_plan)` 的排序键为：

```text
sha256(
  namespace, 42, dataset, stratum,
  epoch, namespaced_id, augmentation_seed
)
```

按键升序选择，并冻结 required strata 的默认合同：

```text
tiny_positive：64 crops、至少 24 distinct source IDs
normal_positive：64 crops、至少 24 distinct source IDs
同一 source ID 在同一 stratum：默认最多 3 crops
batch_size=16
每个 stratum=4 batches
```

默认写为：

```text
diversity_target=24
max_repeat_cap=3
diversity_target_limited_by_natural_availability=false
```

required stratum 的 diversity/cap 由 `epoch=1...1000` 正式穷举得到的 natural
distinct-source ceiling `D` 唯一决定：

```text
若 D >= 24：
  diversity_target = 24
  max_repeat_cap = 3
  diversity_target_limited_by_natural_availability = false

若 16 <= D < 24：
  diversity_target = D
  必须覆盖全部 D 个自然 source
  max_repeat_cap = max(3, ceil(64 / D))
  diversity_target_limited_by_natural_availability = true

若 D < 16：
  required_coverage_fail = true
  不构造该 stratum batch
```

修正后的 64 条必须尽量均衡。令 `q=floor(64/D)`、`r=64 mod D`：每个 source 先取
`q` 条，再按“下一自然候选的冻结排序键”只给 `r` 个 source 各增加 1 条；因此每个
source 只能出现 `q` 或 `q+1` 次。任何 source 不得超过
`max_repeat_cap=max(3,ceil(64/D))`，也不能重复同一
`(epoch, source, transform_plan)`。因此 `D=22/23` 时虽然 cap 仍为 3，diversity
target 也必须分别放宽为 22/23；不能遗漏该边界。

NUAA 的正式穷举事实已固定为：26 个含 tiny 的 train source 中，20 个为 tiny-only、
6 个为 mixed；在 normal-first 分类下，5 个 mixed source 在 epoch 1...1000 的任何
正式 crop 中都不能成为纯 tiny，只有 `Misc_209` 可以。因此：

```text
NUAA tiny_positive natural distinct-source ceiling D=21
diversity_target=21
max_repeat_cap=max(3,ceil(64/21))=4
diversity_target_limited_by_natural_availability=true
```

均衡分配必须是 20 个 source 各 3 条、仅 1 个 source 4 条；不能让多个 source
出现 4 次。具体获得第 4 条的 source 由冻结排序键决定，不能人工选择。

`background_only` 只有在自然候选也能完整达到 `64 crops + 24 distinct IDs` 时才
整层纳入；否则不使用部分 batch，并固定记录：

```text
available=false
structurally_unavailable=true
candidate_count=0
reason=<由正式尺寸、GT 与 crop 几何给出的确定性原因>
```

可另报 `observed_natural_candidate_count` 解释覆盖情况，但正式 `candidate_count`
仍为 0。conditional stratum 不可用不是 A0 工程失败，也不允许用合成或协议外背景
补齐。因此每个数据集固定为：

```text
required only：2 strata × 4 batches = 8 batches
background naturally available：3 strata × 4 batches = 12 batches
```

该门槛已考虑真实 train mask 分布：NUAA、NUDT、IRSTD-1K 分别有
`26 / 156 / 122` 个含 tiny target 的训练 source；NUDT 的
`background_only=0` 是由 `663 × 256×256 + GT非空` 的结构事实决定。若任一
required stratum 无法达到 `64 crops + 冻结 diversity_target`，或自然 ceiling
`D<16`，才记 `coverage_fail=true`；禁止临时改 seed、人工改类别、降低公式所得
配额或重复填充。仅自然可用性修正时，才可使用已穷举的正式 epoch1...1000 候选。

同一数据集的 `best_miou` 和 `best_pd` 必须复用完全相同的 8 或 12 个 batch。
batch manifest 至少保存：

```text
sample ID / namespaced ID / source ID
audit epoch / augmentation seed / complete transform plan
stratum / mixed_tiny / batch index
required_or_conditional / available / structurally_unavailable
candidate_count / observed_natural_candidate_count / unavailable reason
natural_distinct_source_ceiling D / diversity_target / max_repeat_cap
diversity_target_limited_by_natural_availability / exhaustive proof SHA/count
input tensor SHA / mask tensor SHA
train index SHA / ordered-ID SHA
```

重新构建时 input/mask tensor SHA 必须逐条一致。

## 13.3 共享参数组合同

基于去重后的 `named_parameters()` 原顺序建立四个互斥组，并保存参数名称列表、
numel 和 SHA：

```text
encoder_shared:
  inc.*
  down_encoder1.* ... down_encoder4.*

tpd_qfg_sctb_shared:
  mtc.*
  tpd_qfg.*

ner_shared:
  tpd_ner.*

decoder_trunk_shared:
  up_decoder4.* ... up_decoder1.*
```

`gt_conv5/4/3/2`、`outconv`、`outc` 不混入 shared cosine；它们只分别报告
head-local gradient norm。验证器必须确认四个共享组无参数对象交集、无重复名称，
且实际名称集合和冻结 SHA 一致。

## 13.4 单 batch 六头梯度计算

每个 batch 使用以下固定过程：

1. 从对应 checkpoint 的原始 state 开始，`model.train()`、FP32、workers=0、
   deterministic algorithms 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；
2. forward RNG 固定为
   `sha256(namespace,42,dataset,stratum,batch_index)`，不包含 checkpoint role，保证
   两角色使用相同 dropout draw；
3. 一个 forward 同时产生六张 segmentation probability map；
4. 六项 scalar loss 必须与 production `compute_tpd_training_loss(...,
   survival_weight=0)` 的 `segmentation_terms` 逐项相等；
5. 仅使用 `torch.autograd.grad`，无 optimizer、无 `optimizer.step`、无累计 `.grad`；
   六个 head 共用该次 forward，unused 参数按同形全零向量处理；
6. train-mode forward 改写的 BN buffer 在该 batch 后恢复，完整 model state SHA 必须
   与 batch 前相等。

每个 head 记录 scalar BCE 及 16 个 per-sample BCE。对每个共享组 (G)，令
(g_{h,G}=\nabla_{\theta_G}L_h\)，记录：

```text
raw L2 norm
gradient RMS = norm / sqrt(group_numel)
norm_ratio_to_final = norm(g_h,G) / norm(g_final,G)
cosine_to_final
dot product / projection onto final gradient
unused/zero-gradient flag
```

若任一向量 norm 为 0，cosine 写 JSON `null`，不得伪造为 0。另定义：

\[
g_{aux,G}=g_{gt5,G}+g_{gt4,G}+g_{gt3,G}+g_{gt2,G}+g_{d0,G}
\]

\[
g_{total,G}=g_{aux,G}+g_{final,G}
\]

每组同时记录：

```text
cos(g_aux, g_final)
norm(g_aux) / norm(g_final)
cancellation = norm(g_total) / (sum_h norm(g_h))
```

聚合单位固定为 `dataset × role × stratum × group × head`；保存 4 个 batch 原值、
median、IQR 和正/负 cosine batch 数。不同参数组不能先拼成一个大向量再计算 cosine。

## 13.5 量化冲突定义

对 auxiliary head (h\in\{gt5,gt4,gt3,gt2,d0\})、共享组 (G)、分层 (s)，在
一个 dataset-role 的四个 batch 上定义：

### Persistent conflict：`PC(h,G,s)`

```text
有效 cosine = 4/4
median cosine(h, final) <= -0.10
cosine < 0 的 batch >= 3/4
median norm_ratio_to_final >= 0.25
```

### Aggregate conflict：`AC(G,s)`

```text
median cosine(aux, final) <= -0.10
cosine(aux, final) < 0 的 batch >= 3/4
median norm(aux) / norm(final) >= 1.50
```

### Persistent alignment：`PA(h,G,s)`

```text
有效 cosine = 4/4
median cosine(h, final) >= +0.20
cosine > 0 的 batch >= 3/4
median norm_ratio_to_final >= 0.25
```

`PA` 用于识别跨数据集方向反转。另可描述：

```text
redundant_d0 =
  median cosine(d0, final) >= 0.95
  AND median norm_ratio_to_final >= 0.50
```

但 d0 与 final 高度一致只说明冗余，不能单独授权降权。

## 13.6 Trigger A：只授权设计 DS V2

对签名 `(h,G,s)`，先定义 `D_s` 为该 stratum 实际可用的数据集集合：required
strata 的 `D_s` 固定为三个数据集；`background_only` 只包含自然候选达到完整配额
的数据集。不可用数据集不进入该 stratum 的分母，不能当作通过或失败。通过数固定：

\[
K_s=\max\left(2,\left\lceil\frac{2|D_s|}{3}\right\rceil\right)
\]

所以 `|D_s|=3` 时要求至少 2 个数据集；`background_only` 若只有两个数据集可用，
两者必须全部通过；若 `|D_s|<2`，该 stratum 无资格触发 DS V2。

只有以下条件全部成立，才允许设计一个 DS V2 候选：

```text
A0：六 checkpoint 的工程、state、epoch1...1000 ceiling proof 与
    required-strata coverage/diversity/cap 门全部通过；conditional background
    不可用已按结构事实完整登记，不视为失败

A1：存在同一个签名 (h,G,s)，且 |D_s| >= 2

A2：在同一批至少 K_s 个 D_s 数据集上，best_miou 与 best_pd 两个角色
    都同时满足 PC(h,G,s) 和对应的 AC(G,s)

A3：D_s 中不存在同一签名在一个数据集满足 PC、
    在另一个数据集满足 PA 的跨数据集方向反转

A4：h 必须是五个 auxiliary heads 之一；DS V2 只能处理通过签名涉及的 head
```

Trigger A 通过后的状态只能是：

```text
decision=DS_V2_DESIGN_AUTHORIZED_BY_PERSISTENT_GRADIENT_CONFLICT
ds_v2_design_authorized=true
ds_v2_training_authorized=false
loss_formula_changed=false
```

只有当通过签名的 `s=tiny_positive`，才允许写
`tiny_gradient_conflict_supported=true`。若冲突只出现在 normal/background，不能
声称粗尺度监督破坏 tiny target。Trigger 不允许顺带修改 TPD、NER、QFG、TSS，
也不允许直接决定具体权重；DS V2 的有界/零点锚定公式仍需单独预注册和测试。

## 13.7 Severe 与停止条件

### 工程无效

以下任一项发生，当前 checkpoint 审计无效，并停止科学裁决：

```text
任一 loss、gradient 或汇总值非有限
batch/input/mask 重放 SHA 不一致
batch 后 model state SHA 未恢复
任一 required stratum 少于 64 crops 或少于其冻结 diversity_target
required stratum 的自然 ceiling D < 16
自然 ceiling、source集合、proof SHA/count 无法由 epoch1...1000 穷举重算
受限 stratum 未覆盖全部 D 个 source，或违反 q/q+1 均衡分配
任一 source 超过 max_repeat_cap，或重复同一自然 crop tuple
声明 available=true 的 conditional background 少于 64 crops 或少于 24 distinct IDs
conditional background 不可用却未记录 structurally_unavailable/candidate_count=0/reason
为不可用 background 制造合成、协议外或部分填充 batch
任一必需组的 final gradient norm < 1e-12 超过 1/4 batch
两次 sentinel batch 重放的 loss/gradient 摘要不完全一致
参数组 name/count/numel/SHA 与冻结 manifest 不一致
```

裁决：

```text
decision=DS_AUDIT_ENGINEERING_INVALID
ds_v2_design_authorized=false
ds_v2_training_authorized=false
```

### 工程有效但不授权 DS V2

以下任一项成立即停止全局六路 reweighting 路线：

```text
不存在同一签名同时达到 A1 和 A2
冲突只存在于单数据集、单 checkpoint role，或可用数据集通过数小于 K_s
同一签名跨数据集出现 PC 对 PA 的方向反转
各数据集 culprit head 不同，且无任何签名在至少 K_s 个可用数据集通过
只有 loss/norm 大小差或 d0 redundancy，没有量化负 cosine
```

对应裁决：

```text
DS_NO_PERSISTENT_CROSS_DATASET_CONFLICT
或
DS_GLOBAL_REWEIGHTING_BLOCKED_BY_DOMAIN_REVERSAL
```

若 final gradient RMS 有效，但某 auxiliary head 的
`norm_ratio_to_final >= 1000` 在至少 2/4 batch 出现，则记：

```text
decision=DS_GRADIENT_SCALE_ANOMALY_REQUIRES_DIAGNOSIS
```

此时先排查 reduction、target shape、unused 参数和数值尺度，不能把异常直接解释为
“该 head 应降权”。

## 13.8 结果与验证合同

实现时建议形成：

```text
manifest.json
batch_records.jsonl
aggregate.json
decision.json
decision.md
```

decision 必须绑定：审计源码 SHA、六 checkpoint SHA、三个 train index SHA、batch
manifest SHA、参数名列表 SHA、运行环境，以及逐数据集 stratum availability map。
availability map 必须区分 required/conditional，并对不可用项保存
`structurally_unavailable=true`、`candidate_count=0` 和确定性原因。普通 Python 与
`python -O` validator 还必须重算每个 required stratum 的 epoch1...1000 穷举 proof
SHA/count、natural ceiling `D`、`diversity_target`、`max_repeat_cap` 与 q/q+1 分配，
并从 aggregate 原值和 availability map 重算全部 `PC/AC/PA`、`D_s/K_s` 与
Trigger；禁止只信已有 decision 字段。

即使 Trigger A 通过，也只授权下一步设计 DS V2，不授权训练。DS V2 若随后通过
独立工程门，仍必须由 seed42 scratch 训练，并联合 Pd、tiny-Pd、Fa、pixel FP、
mIoU、nIoU 和错误目标数裁决；梯度审计本身不能替代最终性能证据。

## 13.9 DS-GA V1 正式执行结果

本合同已在三数据集当前 TSS-off 完整模型的 `best_miou` 与 `best_pd` checkpoint 上
执行完毕。正式 manifest 共包含 512 个真实训练 crop、32 个唯一 batch：NUAA-SIRST
为 192/12，NUDT-SIRST 为 128/8，IRSTD-1K 为 192/12。NUDT-SIRST 的
`background_only` 因 663 张训练图像均含目标而按合同记为结构上不可用，没有制造
合成背景窗口。六份运行均通过 checkpoint、输入 batch、模型 state、sentinel replay
和参数分区核验。

构建器、分析器和比较器的 38 项定向测试在普通 Python 与 `python -O` 下均通过；
比较器从六份 raw Gram 重算得到的 aggregate 与落盘文件完全一致。

```text
decision=DS_GLOBAL_REWEIGHTING_BLOCKED_BY_DOMAIN_REVERSAL
engineering_valid=true
trigger_a_passed=false
authorized_signature_count=0
ds_v2_design_authorized=false
ds_v2_training_authorized=false
tiny_gradient_conflict_supported=false
gradient_scale_anomaly_observed=false
domain_direction_reversal_observed=true
```

两组决定性跨数据集反转均位于 `ner_shared`：

| stratum / head | NUDT-SIRST | NUAA-SIRST | IRSTD-1K |
|---|---|---|---|
| tiny / gt3 | best-mIoU：cos=-0.151577，ratio=6.9744，PC+AC | 两角色 cos≈+0.182，无 PC/PA | 两角色 cos=+0.744454/+0.614671，PA |
| normal / gt2 | best-Pd：cos=-0.305888，ratio=1.0005，PC | 两角色 cos=+0.329718/+0.605368，PA | 两角色 cos=+0.971247/+0.940586，PA |

因此，当前证据不支持把六个 BCE 头统一降权或进行全局 DS 重配：这种修改可能缓解
NUDT-SIRST 某个 checkpoint 的局部 NER 冲突，却会削弱 NUAA-SIRST 与 IRSTD-1K
已经对 final head 有利的梯度。此次裁决只否决全局 DS V2 路线，不否定现有六头监督，
也不改变 `TPD8 + NER4 + QFG2 + TSS-off` 当前正式模型。

正式产物：

- `results/three_dataset_ds_gradient_audit_v1/comparison/seed42_six_role/decision.md`
- `results/three_dataset_ds_gradient_audit_v1/comparison/seed42_six_role/decision.json`
- `results/three_dataset_ds_gradient_audit_v1/comparison/seed42_six_role/aggregate.json`
- `results/three_dataset_ds_gradient_audit_v1/runs/` 下六份 `audit.json`

---

# 14. 文件修改清单

## 新增模型

```text
model/tpd_global_constant_sum_skip_fusion.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_gcsf.py
```

## 新增分析

```text
analysis/analyze_three_dataset_gcsf_branch_audit_v1.py
analysis/compare_three_dataset_gcsf_branch_audit_v1.py
analysis/build_three_dataset_ds_gradient_audit_manifest_v1.py
analysis/analyze_three_dataset_ds_gradient_audit_v1.py
analysis/compare_three_dataset_ds_gradient_audit_v1.py
```

## 新增训练与评估

```text
experiments/export_tpd_ner_v4_qfg_v2_croa_gcsf_to_inference.py
experiments/train_three_dataset_gcsf_tss_off_seed42_v1.py
```

## 新增测试

```text
tests/test_tpd_global_constant_sum_skip_fusion.py
tests/test_tpd_ner_v4_qfg_v2_croa_gcsf_integration.py
tests/test_export_tpd_ner_v4_qfg_v2_croa_gcsf.py
tests/test_three_dataset_gcsf_branch_audit_v1.py
tests/test_compare_three_dataset_gcsf_branch_audit_v1.py
tests/test_train_three_dataset_gcsf_tss_off_seed42_v1.py
tests/test_build_three_dataset_ds_gradient_audit_manifest_v1.py
tests/test_analyze_three_dataset_ds_gradient_audit_v1.py
tests/test_compare_three_dataset_ds_gradient_audit_v1.py
```

## 不修改

```text
model/tpd_clean_v8_mprs_dch.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware.py
model/tpd_frequency_gate_v2_croa.py
experiments/tpd_training_loss.py
全部 TSS / EC-TSS 历史文件
全部 TPD residual diagnostic 结果
历史 checkpoint 和 selector
```

---

# 15. 完整执行顺序

```text
Phase 0
完成：封存 TPD_INCONCLUSIVE_NO_FORMULA_CHANGE
完成：冻结 TPD8 七个 residual

Phase 1
完成：现有 TSS-off checkpoint 上执行 E_i / T_i 分支审计
完成：6 checkpoints × 11 modes = 66 cells
结果：Trigger A=false，GCSF training authorization=false

Phase 2
完成：GCSF 独立模块、architecture manifest、strict extension loader

Phase 3
完成：零点输出、共享梯度、第一 Adam step、state、导出测试

Phase 4
完成：CPU、python -O、RTX 5090 smoke、exact resume、source lock

Phase 5（仅在 Phase 1 的量化性能 Trigger 通过后执行）
未执行：Trigger A 未通过，训练入口按合同拒绝失败 decision

Phase 6
未执行：没有 pilot，不启动 formal1000

Phase 7
未执行：没有训练后的 GCSF checkpoint

Phase 8
完成：GCSF 未触发训练，保留现有完整 TSS-off 模型
完成：六路 deep-supervision 梯度审计，6 checkpoints / 32 unique batches
结果：跨数据集方向反转，DS V2 design/training authorization=false
下一步：停止全局 DS 重加权，转入面向实际错误类型的模型结构优化
```

---

# 16. 推荐状态

```text
pre_revision_decision=DOCUMENT_REVISION_REQUIRED_BEFORE_GCSF_IMPLEMENTATION
decision=GCSF_ZERO_TRAINING_TRIGGER_FAILED
gcsf_zero_training_decision=GCSF_BRANCH_AUDIT_NO_TRAINING_AUTHORIZATION

tpd_decision=TPD_INCONCLUSIVE_NO_FORMULA_CHANGE
tpd8_frozen=true
tpd8_residuals_enabled=true

ner4_frozen=true
qfg2_frozen=true
tss_objective_enabled=false

double_identity_dataflow_verified=true
skip_fusion_performance_bottleneck_established=false
full_model_skip_fusion_hypothesis_tested=true
global_fixed_skip_reallocation_supported=false
deep_supervision_gradient_audit_complete=true
deep_supervision_global_reweighting_supported=false
deep_supervision_domain_direction_reversal_observed=true
ds_v2_design_authorized=false
ds_v2_training_authorized=false
next_candidate=NER_L4_TPR_V1
ner_l4_tpr_code_implemented=true
ner_l4_tpr_zero_training_assessment=REPRESENTABLE_CROSS_ROLE_JOINT_SIGNAL
ner_l4_tpr_formal_training_authorized=true
ner_l4_tpr_formal_training_started=true

gcsf_zero_training_audit_complete=true
gcsf_zero_training_cells=66
gcsf_trigger_a_passed=false
gcsf_code_implemented=true
gcsf_training_runner_implemented=true
gcsf_pilot_authorized=false
gcsf_formal_training_authorized=false
gcsf_training_started=false

new_inference_parameters=480
new_persistent_buffers=0
mainline_changed=false
tpd_formula_changed=false
ner_formula_changed=false
qfg_formula_changed=false

paper_core_established=false
stability_claim_supported=false
training_recipe_finalized=false
```

---

# 17. 最终结论

6 个正式 TSS-off checkpoint 已全部完成 11-mode 诊断；固定阈值仍为 0.5，阈值
1.0 只用于记录合法空预测端点。Trigger A 汇总为：

| mode | `best_miou` safe-material | 六角色 severe | 通过 |
|---|---:|---:|:---:|
| `gneg025_l1_only` | 0/3 | 0/6 | false |
| `gneg025_l2_only` | 0/3 | 0/6 | false |
| `gneg025_l3_only` | 0/3 | 2/6 | false |
| `gneg025_l4_only` | 0/3 | 3/6 | false |
| `gneg025_all_levels` | 0/3 | 6/6 | false |
| `gpos025_l1_only` | 1/3 | 2/6 | false |
| `gpos025_l2_only` | 0/3 | 1/6 | false |
| `gpos025_l3_only` | 1/3 | 2/6 | false |
| `gpos025_l4_only` | 2/3 | 1/6 | false |
| `gpos025_all_levels` | 0/3 | 6/6 | false |

最接近通过的是 `gpos025_l4_only`。它在 NUAA 与 NUDT 的 primary `best_miou`
单元都达到 safe-material，但在 IRSTD-1K `best_miou` 上少检 2 个目标，构成唯一
severe 单元：

| 数据集/角色 | Δ检出目标 | Δtiny | ΔmIoU | ΔnIoU | component FP 降幅 | background FP 降幅 | safe-material | severe |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|
| NUAA / best_miou | -1 | 0 | -0.001597 | -0.000587 | +7.56% | +2.03% | true | false |
| NUDT / best_miou | -1 | 0 | +0.000287 | -0.000098 | +28.93% | +8.29% | true | false |
| IRSTD / best_miou | -2 | 0 | +0.000283 | -0.003279 | +10.68% | +2.29% | false | true |
| NUAA / best_pd | -1 | 0 | -0.002093 | -0.000949 | +7.44% | +2.44% | true | false |
| NUDT / best_pd | -1 | 0 | +0.001677 | +0.001915 | +13.10% | +6.27% | true | false |
| IRSTD / best_pd | 0 | 0 | +0.003222 | +0.002331 | +8.00% | +3.50% | true | false |

因此正式裁决是：

```text
decision=GCSF_BRANCH_AUDIT_NO_TRAINING_AUTHORIZATION
gcsf_trigger_a_passed=false
gcsf_pilot_authorized=false
gcsf_formal_training_authorized=false
skip_fusion_performance_bottleneck_established=false
next_step=DEEP_SUPERVISION_GRADIENT_AUDIT_COMPLETED
```

这表示固定 checkpoint 上的常系数和重分配存在数据集权衡，尚不足以投入三数据集
scratch 训练；它不等价于证明任何可学习融合都无效。GCSF 工程代码保留为已测试
候选，但不进入当前完整模型。现阶段继续使用 `TPD8 + NER4 + QFG2 + TSS-off`，
主线与 TPD/NER/QFG 核心创新均未改变。

后续六路 DS-GA V1 也已完成。结果为
`DS_GLOBAL_REWEIGHTING_BLOCKED_BY_DOMAIN_REVERSAL`：工程核验通过，但没有任何签名
满足跨数据集、双 checkpoint 的设计门，且 `ner_shared` 在 NUDT-SIRST 与
NUAA-SIRST/IRSTD-1K 之间出现方向反转。因此不设计、不训练全局 DS V2，下一步改为
直接针对正式性能错误类型选择模型结构优化点。

工程验收结果为：GCSF `480 parameters / 4 state keys / 0 buffers`；训练图
`10,870,708 parameters / 572 keys`；head-free 推理图 `10,870,610 parameters /
568 keys`。普通模式与 `python -O` 均为 `33 passed + 3389 subtests passed`；RTX
5090 训练态 forward/backward smoke 通过，四组 GCSF 参数均获得有限非零梯度。

正式产物：

- `results/three_dataset_gcsf_branch_audit_v1/comparison/seed42_six_role/decision.md`
- `results/three_dataset_gcsf_branch_audit_v1/comparison/seed42_six_role/decision.json`
- `results/three_dataset_gcsf_branch_audit_v1/runs/` 下 6 份 `evaluation.json`
- `results/three_dataset_ds_gradient_audit_v1/comparison/seed42_six_role/decision.md`
- `results/three_dataset_ds_gradient_audit_v1/comparison/seed42_six_role/decision.json`
- `results/three_dataset_ds_gradient_audit_v1/runs/` 下 6 份 `audit.json`

---

# 18. 代码和研究依据

## 当前仓库代码位置

- `model/SCTransNet.py`
  - `ChannelTransformer.forward()`：`reconstruct(encoded) + encoder`
  - `SCTransNet.forward()`：再次 `x_i + f_i`
- `model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py`
  - 完整模型显式保留同样的 `reconstruct + f` 后再 `+ f`
  - QFG 只修改 Query；四尺度 CCA 接收融合 skip
- `model/tpd_sctransnet.py`
  - NER 仅在 L4/L3/L2 使用 CCA 后的融合 skip，并乘以独立生成的 mask；L1 无 NER
  - 五节点 evidence 通过旁路参与 NER mask，L3/L2 还结合 relay 与 decoder-up
- `experiments/tpd_training_loss.py`
  - TSS-off 时总损失严格等于原六项 segmentation BCE 之和

## 方法定位

GCSF 的零初始化锚定与 residual scaling、LayerScale 一类稳定优化思想相关，但其研究对象不同：

```text
不是对单个 residual block 乘一个系数
而是对 SCTransNet 四尺度 transformed/duplicated-identity skip
进行常系数和的双分支重分配
```

## 历史项目证据

历史总汇已经明确：

```text
完整模型具有竞争力，但统一配方与跨数据集稳定收益仍需优化
下一候选必须联合查看 Pd、Fa、mIoU、nIoU、tiny-Pd
不能用单一 mIoU 裁决
```

---

# 19. 后续性能结构：NER-L4-TPR

GCSF 的最接近模式 `gpos025_l4_only` 已提供明确的 L4 降 FP 方向，但问题是全图作用
导致六角色合计少检 6 个目标。后续没有重新开放 GCSF 全局融合训练，而是把同一 L4
重分配限制到 NER `q4` 判定的非目标区域：目标保护区严格保留当前融合，背景候选区
由 256 个零初始化逐通道门学习重分配。

六角色筛选结果为 `REPRESENTABLE_CROSS_ROLE_JOINT_SIGNAL`。有限-logit
`tpr_g01875` 相对无保护模式恢复 4 个目标，并相对当前模型合计减少 99 个 component
FP 和 95 个 background FP；tiny 检出不变。该结果支持实际模型训练，不改变 TPD8、
五节点 NER4 或 QFG2-CROA 主线。

当前 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 三套 seed42 / scratch / 1000-epoch
正式训练已在 GPU0/1/2 启动。详细结构、代码合同与结果见：

- `SCTransNet_NER_L4_TPR性能优化与代码实现方案.md`
- `results/three_dataset_ner_l4_tpr_zero_training_v1/comparison/seed42_six_role/decision.md`
