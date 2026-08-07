# SCTransNet PBDR‑V2 失败分析、PBDR‑V3 修正方案与非退化部署门控

> 审计对象：`Arialliy/SCTransNet_main`  
> 代码快照：`ac3fec5202c3f88aecfd83a4ea6e60c60c8be755`（2026‑08‑06，`Add PBDR V2 evidence workflow`）  
> 输入证据：仓库源码、当前 NUAA‑SIRST 双 checkpoint 正式指标、用户给出的训练与产物审计结论  
> 结论置信度：**PBDR‑V2 相对 Current 的负向结论为高置信；各失效机制的相对贡献仍需用分支归因实验确定**

---

## 1. 执行结论

### 1.1 立即决策

1. **不要启动 NUDT‑SIRST 与 IRSTD‑1K 的 PBDR‑V2 正式训练。** NUAA 已经证明当前机制相对直接父模型 Current 负向；继续跨数据集扩展只会消耗算力，不能回答失效归因。
2. **GPU0 立即用于 NUAA 的无重训分支归因、阈值/FROC 扫描和 PBDR‑V3 小规模校准训练。** GPU1/GPU2 上的 baseline 链路可继续，不必等待。
3. **废弃“从头联合训练 + 19 参数零扩展就会保持 Current”的假设。** 下一版必须从已经训练好的 Current checkpoint warm‑start，第一阶段冻结整个 Current，并冻结其 BatchNorm 运行统计，只训练校准器。
4. **移除 PBDR‑V2 的无约束 `q4` 直接残差。** `d0` 只能作为上下文特征，不能再被硬编码为“漏检救援/假警抑制”的独立证据。
5. **用非退化选择器保证部署安全。** 候选未同时满足 `Pd`、`Fa`、`mIoU`、`nIoU` 门槛时，部署清单自动回退 Current。

### 1.2 关于“确保一定提升”的严格表述

对未知标签的未来测试集，任何结构和损失都不能数学上保证 `Pd↑、Fa↓、mIoU↑、nIoU↑` 同时成立。原因并非工程保守，而是这些指标存在真实冲突：提高概率可能救回漏检，也可能新增假警；降低概率可能减少假警，也可能删掉目标。连通域匹配还使简单的像素单调性不足以保证对象级指标单调。

本方案能提供两种可执行保证：

- **工程保证：** 在一个冻结的 certification split 上，候选只有严格优于 Current 才会被选中，否则自动回退 Current。因此“最终被部署的模型”在该固定集合上不退化。
- **科学结论：** 先用训练集内部验证完成结构、checkpoint 和阈值选择，官方 test 只访问一次。此时仍应报告置信区间和多随机种子结果，而不能声称对任意未见样本绝对保证。

若继续沿用仓库当前的 test‑selected 协议，也可以保证**该已知 test 上的选中产物**不退化，但这属于乐观选择，不能包装成无偏泛化结果。

---

## 2. NUAA 结果复核

### 2.1 相对 Current 的精确变化

| checkpoint | 指标 | PBDR‑V2 | Current | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|---:|
| best_miou | 检出目标 | 254/263 | 256/263 | −2 | −0.7813%（Pd 相对值） |
| best_miou | Pd | 0.965779 | 0.973384 | −0.007605 | −0.7813% |
| best_miou | Fa | 2.4628e‑5 | 1.5435e‑5 | +9.1930e‑6 | **+59.5594%** |
| best_miou | mIoU | 0.782599 | 0.796483 | −0.013884 | −1.7432% |
| best_miou | nIoU | 0.793443 | 0.795348 | −0.001905 | −0.2395% |
| best_pd | 检出目标 | 257/263 | 257/263 | 0 | 0 |
| best_pd | Pd | 0.977186 | 0.977186 | 0 | 0 |
| best_pd | Fa | 3.6221e‑5 | 1.4749e‑5 | +2.1472e‑5 | **+145.5828%** |
| best_pd | mIoU | 0.769457 | 0.788553 | −0.019096 | −2.4217% |
| best_pd | nIoU | 0.781190 | 0.792668 | −0.011478 | −1.4480% |

### 2.2 统计解释

- `best_miou` 的 2 个目标差异本身不足以单独证明 Pd 机制性下降。以独立二项 Wilson 区间作粗略参照，`254/263` 的 95% 区间约为 `[0.9363, 0.9819]`，`256/263` 约为 `[0.9461, 0.9870]`，高度重叠。正式判断应使用逐目标配对命中和 McNemar 检验。
- 但两个 checkpoint 角色都出现 **Fa 大幅上升且 mIoU 下降**，方向一致；`best_pd` 在检出数完全相同时仍付出 145.6% 的 Fa 增量和 0.019096 的 mIoU 损失。这是比“少 2 个目标”更强的失败证据。
- 相对 Original 的混合权衡不能挽救该结论。PBDR‑V2 是 Current 的直接扩展，正确的增量消融对照是 Current；不能用更早、更弱的祖先模型替代直接父模型作为成功标准。

### 2.3 已排除的伪原因

根据产物审计，以下解释应当排除：

- 训练未完成；
- checkpoint 角色保存错误；
- state key 缺失或错配；
- 19 个 PBDR 参数没有进入优化；
- TSS 意外参与损失；
- PBDR 分支没有接入最终输出。

因此需要分析的是**模型假设、梯度路径和目标函数失败**，而不是流水线未生效。

---

## 3. 代码路径重建

### 3.1 Current 的真实部署头

基础 SCTransNet 在深监督模式下构造：

```text
gt5, gt4, gt3, gt2  -- 多尺度 raw logits，经双线性上采样到全分辨率
out                  -- 最终 decoder raw logit
d0 = Conv1x1([gt2, gt3, gt4, gt5, out])
```

训练返回六路概率：

```text
sigmoid(gt5), sigmoid(gt4), sigmoid(gt3), sigmoid(gt2), sigmoid(d0), sigmoid(out)
```

推理返回的是 `sigmoid(out)`，不是 `d0`。因此 `d0` 是受监督的辅助融合头，但不是 Current 的部署判决头。[^R2]

### 3.2 PBDR‑V2 改写了第六路监督和部署读出

PBDR‑V2 使用：

\[
\begin{aligned}
q &= \operatorname{RMSNorm}(\operatorname{stopgrad}(q_4)),\\
C &= 0.05+0.90\sigma(W_cq+b_c),\\
Q &= C\tanh(W_qq),\\
g_r &= 0.5\tanh(a_r),\qquad g_s=0.5\tanh(a_s),\\
R^+ &= C\operatorname{ReLU}(d_0-out),\\
R^- &= (1-C)\operatorname{ReLU}(out-d_0),\\
z_{route} &= out+Q+g_rR^+-g_sR^-.
\end{aligned}
\]

其直接残差上限为 `±1.0 logit`，两个 disagreement strength 的范围均为 `[-0.5, 0.5]`。[^R1]

集成代码将训练第六路从 `sigmoid(out)` 替换为 `sigmoid(routed_out)`，推理也返回 `sigmoid(routed_out)`；原始 `out` 不再拥有独立的第六路 BCE，只能通过 `d0` 和 routed path 间接接收梯度。[^R3]

### 3.3 PBDR‑V2 不是在训练好的 Current 上做后校准

模型构建器只保证第 0 步共享参数与 Current 的**初始 scratch state** 按位相同；元数据明确记录：

```text
initialization_mode = fresh_seed42_paired_scratch_extension
parent_checkpoint   = None
warm_start_used     = False
pbdr_v2_parameters_jointly_trainable = True
```

训练器随后对 `model.parameters()` 建立一个统一 Adam 优化器，所有新旧参数使用同一基础学习率。[^R5][^R6]

所以“零初始化”仅保证：

\[
z_{route}^{(0)}=z_{out}^{(0)}.
\]

它不保证：

\[
\theta_{shared}^{(t)}=\theta_{Current}^{(t)},\quad t>0.
\]

第一步以后，PBDR 的共享网络、辅助头和最终头都沿着不同目标函数进入不同优化轨迹。

### 3.4 训练目标与正式指标

训练损失是六路概率图上 BCE 的有序求和；TSS 为零时没有额外目标。[^R4]

正式评估则在固定阈值 `probability > 0.5` 后：

- `mIoU`：全数据集前景交并比；
- `nIoU`：逐图 IoU 均值；
- `Pd`：GT 与预测连通域质心在 3 像素内的一对一最大匹配；
- `Fa`：**所有未匹配预测连通域的像素数 / 有效像素数**。[^R8]

这意味着一个很小的 logit 正偏移，只要使背景跨过 0.5 并形成孤立连通域，就可能显著提高 Fa；像素平均 BCE 并没有直接表达这一拓扑代价。

---

## 4. 失败原因：按证据强度排序

| 优先级 | 失效机制 | 证据强度 | 与当前症状的对应关系 | 首要验证 |
|---:|---|---|---|---|
| 1 | `d0` 被错误当作独立证据 | 高 | 粗尺度融合光晕进入最终读出，Fa↑、边界变厚、mIoU↓ | 去掉 disagreement 分支 / 单独评估 rescue |
| 2 | 从头联合训练导致共享轨迹漂移 | 高 | 即使路由很小，raw `out` 与 Current 也已不是同一个模型 | 比较 Current 与 PBDR checkpoint 的 router‑bypass `out` |
| 3 | `q4` 直接残差过自由且空间分辨率过低 | 中高 | 宽区域正残差、阈值穿越、新增假警 | `direct_only` 与 `full-minus-direct` 归因 |
| 4 | RMS 归一化会放大弱证据；无 bias 不等于无全局偏移 | 中高 | 背景弱噪声被单位化，形成 DC/低频正偏移 | 记录归一化前 RMS、空间均值与 Q 的背景分布 |
| 5 | 零初始化存在首步梯度退化 | 高（数学确定） | 置信图首步不学习，直接残差和全局 scalar 先行 | 首 batch 梯度审计 |
| 6 | BCE 与阈值/连通域指标错位 | 高 | loss 下降不阻止 Fa 大涨；best_pd 尤其明显 | 阈值曲线、跨阈值连通域归因 |
| 7 | signed 全局强度可反转语义 | 条件性高 | rescue 可能变成 suppress，反之亦然 | 打印最终 `g_r/g_s` 的符号和值 |
| 8 | 插值网格不一致 | 中 | 小目标峰值和边界产生亚像素错位 | 统一网格的推理消融 |
| 9 | 小样本、单种子、test‑selected | 高（协议问题） | 放大模型选择方差，限制泛化结论 | 内部验证、配对 bootstrap、多种子 |

### 4.1 根因一：`d0 - out` 不具有“漏检/假警”语义可辨识性

`outconv` 是一个 `5→1` 的 1×1 卷积，因此逐像素可写成：

\[
d_0=b+w_2gt_2+w_3gt_3+w_4gt_4+w_5gt_5+w_o out.
\]

所以：

\[
d_0-out=b+\sum_{i=2}^{5}w_i gt_i+(w_o-1)out.
\]

结论：

1. `d0` 包含 `out` 本身，二者高度相关，不是两名独立判别器；
2. `d0 > out` 不等价于“这里存在被 out 漏掉的目标”；
3. `out > d0` 不等价于“这里是 out 的假警”；
4. `gt5/gt4/gt3/gt2` 来自 H/16、H/8、H/4、H/2 的粗尺度图，并被双线性上采样。它们更容易表达目标邻域和低频光晕，而不是精确边界；
5. PBDR 将原本不部署的辅助融合头通过 `ReLU(d0-out)` 重新引入最终预测，最符合当前“Fa 增加、mIoU 下降”的症状。

这不是说 `d0` 完全无用；正确用法是把 `p_d0`、`p_d0-p_out` 当作校准器输入特征，由全分辨率局部特征决定是否修正，而不是硬编码其差值方向。

### 4.2 根因二：`q4` 是 H/8 粗证据，却被允许直接修改全分辨率 logit

在 256×256 patch 下，stage‑4 `q4` 与 `up4` 对齐，空间尺寸为 32×32。PBDR 对 `q4` 做 1×1 投影后直接双线性上采样到 256×256。一个粗网格位置会影响一片全分辨率区域；对面积不超过 9 像素的 tiny target，这一分辨率不足以独立完成精确目标/背景判别。[^R3]

同时，`Q` 的幅度可达 `±1 logit`。在阈值边界 `z=0` 附近：

| logit 修正 | 概率变化 |
|---:|---:|
| +0.10 | 0.500 → 0.525 |
| +0.20 | 0.500 → 0.550 |
| +0.50 | 0.500 → 0.622 |
| +1.00 | 0.500 → 0.731 |

因此 PBDR‑V2 的上限远大于“保守校准”所需范围。只需 0.05–0.20 的正修正，就可能把大量模糊背景翻过固定 0.5 阈值。

### 4.3 根因三：全局 RMS 归一化抹掉“证据强弱”并可能放大背景噪声

当前代码对每张图、所有通道和所有空间位置计算一个全局 RMS：

\[
q=\frac{q_4}{\max(\operatorname{RMS}(q_4),10^{-6})}.
\]

问题有三层：

1. **不做中心化。** `q4` 的空间 DC 分量被保留；即使 `W_q` 没有 bias，若 `q_c(x)=\mu_c+\epsilon_c(x)`，则 `\sum_c w_c\mu_c` 仍会构成近似全图 bias。代码注释中“无 bias 可避免整图统一平移”的保证并不成立。
2. **弱图与强图都被单位化。** 一个总体能量很低、主要由噪声构成的 q4，也会被除以很小的 RMS，变成幅度约为 1 的路由输入。
3. **跨通道共用一个 RMS。** 高能通道控制所有通道的缩放，不能保证每个证据通道的统计稳定。

下一版应当“只缩小强信号，不放大弱信号”：逐通道空间中心化，并用带下限的 RMS 作分母。

### 4.4 根因四：零初始化的首步梯度顺序与设计意图相反

记：

\[
A=\operatorname{ReLU}(d_0-out),\qquad B=\operatorname{ReLU}(out-d_0).
\]

初始化时：

```text
C = 0.5
Q = 0
g_r = 0
g_s = 0
```

对路由 logit 的首步导数为：

\[
\frac{\partial z_{route}}{\partial L_c}=0,
\]

\[
\frac{\partial z_{route}}{\partial L_q}=0.5,
\]

\[
\frac{\partial z_{route}}{\partial a_r}=0.25A,
\qquad
\frac{\partial z_{route}}{\partial a_s}=-0.25B.
\]

即：

- 9 个 confidence 参数首个 optimizer step 的梯度严格为零；
- 8 个直接残差权重立即学习；
- 两个全局 signed scalar 立即学习；
- 设计中本应控制风险的空间 confidence 反而最后才开始响应。

本地 autograd 复核也得到：`confidence_projection.*.grad == 0`，而 `direct_residual_projection.weight` 与两个 strength 的梯度非零。这是公式本身决定的，不是训练偶然性。

### 4.5 根因五：零扩展没有保持 Current 的优化轨迹

PBDR‑V2 的训练输出是：

```text
gt5, gt4, gt3, gt2, d0, routed_out
```

Current 是：

```text
gt5, gt4, gt3, gt2, d0, out
```

因此 PBDR 一旦离开零状态：

- 原始 `out` 不再被第六路 BCE 直接约束；
- `out`、`d0`、粗尺度头和共享 decoder 可以共同重参数化；
- 同一个 `routed_out` 可能由“变差的 out + 补偿路由”得到较低训练 BCE，却在固定 0.5 阈值和连通域指标上更差；
- 19 个新参数很少，并不意味着整个函数只发生了 19 参数的小变化，因为全部共享参数也参与了不同的梯度场。

**关键归因实验是 PBDR checkpoint 的 router‑bypass。** 若关闭路由后 raw `out` 已显著低于 Current，则主要问题是联合训练轨迹；若 bypass 与 Current 接近而 full PBDR 变差，则主要问题是路由公式。

### 4.6 根因六：训练 BCE 不约束正式 Fa 的拓扑定义

六路平均 BCE 主要优化概率校准和逐像素误差，而 Fa 只统计未匹配连通域的面积。以下两种预测可能具有接近的 BCE，却具有完全不同的 Fa：

- 在已有目标边缘增加一些概率，但仍属于已匹配目标连通域；
- 在远端背景增加相同数量的概率并形成新孤立连通域。

PBDR‑V2 没有：

- hard‑negative component mining；
- 背景概率相对 Current 的单向约束；
- residual L1/面积预算；
- 阈值穿越惩罚；
- IoU/Jaccard surrogate；
- FROC 或 matched‑Pd/matched‑Fa 选择。

这解释了为什么 best_pd 可以保持 257 个目标，却以 145.6% 的 Fa 和明显 mIoU 损失换取。

### 4.7 根因七：signed 全局强度可以反转分支语义

`g_r=0.5*tanh(a_r)` 与 `g_s=0.5*tanh(a_s)` 允许负值：

- `g_r<0` 时，所谓 rescue 会在 `d0>out` 区域执行抑制；
- `g_s<0` 时，所谓 suppression 会在 `out>d0` 区域增加 logit。

源码注释承认其符号只是“期望非负”，并没有投影保证。最终 checkpoint 必须打印两个值后才能判断是否发生语义翻转。即使二者为正，它们仍是全图共享 scalar，不能区分目标邻域与复杂背景。

### 4.8 次要但应修正：插值网格不一致

`gt5..gt2` 使用 `bilinear, align_corners=True`，PBDR 的 q4 confidence/direct map 使用 `align_corners=False`。对极小目标和边界像素，两套坐标映射可能产生相位差。该问题很可能不是主要根因，但会放大粗尺度证据与 `d0-out` 的空间不一致。

### 4.9 协议问题：当前结果方向可信，但不是无偏 test 估计

仓库协议明确：

```text
train -> optimization
 test -> checkpoint selection + formal evaluation
```

并记录 `test_selected=True`、`selection_is_optimistic=True`、`paper_unbiased_test_supported=False`。NUAA 只有 213 个训练图，1000 epochs 且每 10 epochs 访问 test 选 checkpoint；单 seed 的模型选择方差不可忽略。[^R5][^R7]

这不会推翻 Current 与 PBDR 在同协议下的配对负向结论，但会限制“泛化失败幅度”的统计解释。

---

## 5. GPU0 立即执行的无重训归因

### 5.1 必做模型/分支矩阵

| 编号 | checkpoint | 输出 | 目的 |
|---|---|---|---|
| A0 | Current | Current `out` | 正式父模型基线 |
| A1 | PBDR‑V2 | router bypass，仅 PBDR 训练轨迹的 `out` | 测共享轨迹漂移 |
| A2 | PBDR‑V2 | full routed output | 已知正式结果复现 |
| A3 | PBDR‑V2 | `direct_only` | 定位 q4 直接残差引入的 Fa |
| A4 | PBDR‑V2 | `disagreement_only` | 定位 d0/out 分歧分支 |
| A5 | PBDR‑V2 | `rescue_only` | 检查 coarse d0 光晕 |
| A6 | PBDR‑V2 | `suppression_only` | 检查是否真的抑制背景 |
| A7 | PBDR‑V2 | strength clamp 到非负 | 检查符号翻转 |
| A8 | PBDR‑V2 | `sigmoid(d0)` | 量化辅助融合头自身质量 |

### 5.2 每个输出都做三类评估

1. **固定阈值 0.5：** 与当前正式表严格可比。
2. **阈值扫描：** 建议 `[0.20, 0.80]`，步长 `0.005` 或 `0.01`；保存完整 Pd–Fa、mIoU–Fa、nIoU–Fa 曲线。
3. **匹配工作点：**
   - 在与 Current 相同 Pd 下比较最小 Fa 和最大 mIoU；
   - 在与 Current 相同 Fa 下比较最大 Pd 和 mIoU；
   - 输出 FROC，而不是只看一个 0.5 点。

判断逻辑：

```text
A1 << A0     -> 共享网络训练轨迹漂移是主因；warm-start/freeze 必须做
A1 ≈ A0, A2 << A1 -> 路由公式是主因
A3 导致 Fa↑ -> q4 direct branch 是主因
A5 导致 Fa↑ -> d0 rescue/粗尺度光晕是主因
阈值可完全恢复 -> 校准漂移占主要部分
所有阈值都不能恢复 -> 排序/空间形状已被破坏，不只是 calibration
```

### 5.3 必须保存的诊断量

逐图、逐分支保存：

- `rescue_strength_raw`、`suppression_strength_raw` 及映射后值；
- `confidence_projection`、`direct_residual_projection` 的权重、范数和符号；
- q4 归一化前：每通道空间均值、RMS、最大值；
- `C`、`Q`、`R+`、`R-`、总 `delta_logit` 的 P1/P5/P50/P95/P99；
- GT 背景和前景上各分支的均值与正面积比例；
- 相对 Current，从 `<0.5` 变为 `>0.5` 的背景像素数和新连通域数；
- 每个新增未匹配连通域由 direct/rescue/suppression 哪一项主导；
- 按目标面积、局部对比度、图像是否空目标分层的指标。

### 5.4 可直接加入仓库的 V2 归因包装器

在 checkpoint 加载完成后替换 `model.pbdr_v2`；包装后不要再次保存 state_dict，因为 key 会多一层 `router.`。

```python
"""Evaluation-only branch attribution for a loaded PBDR-V2 checkpoint."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from tpd_persistent_evidence_residual_router_v2 import (
    PersistentEvidenceResidualRouterV2,
)

AblationMode = Literal[
    "identity",
    "full",
    "direct_only",
    "disagreement_only",
    "rescue_only",
    "suppression_only",
    "nonnegative_strengths",
]


class PBDRV2AblationWrapper(nn.Module):
    """Wrap after loading the checkpoint; do not save the wrapped state_dict."""

    def __init__(
        self,
        router: PersistentEvidenceResidualRouterV2,
        mode: AblationMode,
    ) -> None:
        super().__init__()
        self.router = router
        self.mode = mode

    @staticmethod
    def _scalar_map(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(1, 1, 1, 1)

    def forward(
        self,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        q4: torch.Tensor,
    ) -> torch.Tensor:
        diagnostics = self.router.forward_with_diagnostics(z_out, z_d0, q4)
        rescue_strength = self._scalar_map(diagnostics.rescue_strength)
        suppression_strength = self._scalar_map(
            diagnostics.suppression_strength
        )

        if self.mode == "identity":
            return z_out
        if self.mode == "full":
            return diagnostics.routed_logits
        if self.mode == "direct_only":
            return z_out + diagnostics.direct_residual
        if self.mode == "disagreement_only":
            return (
                z_out
                + rescue_strength * diagnostics.target_rescue
                - suppression_strength * diagnostics.background_suppression
            )
        if self.mode == "rescue_only":
            return z_out + rescue_strength * diagnostics.target_rescue
        if self.mode == "suppression_only":
            return z_out - suppression_strength * diagnostics.background_suppression
        if self.mode == "nonnegative_strengths":
            return (
                z_out
                + diagnostics.direct_residual
                + rescue_strength.clamp_min(0.0) * diagnostics.target_rescue
                - suppression_strength.clamp_min(0.0)
                * diagnostics.background_suppression
            )
        raise ValueError(f"unsupported PBDR-V2 ablation mode: {self.mode}")

```

---

## 6. 推荐模型：PBDR‑V3 Conservative Twin‑Gate Calibrator

### 6.1 设计原则

PBDR‑V3 不再假设 `d0>out` 就是漏检，也不允许 q4 直接生成 logit 残差。它只做以下事情：

1. 从训练好的 Current checkpoint 出发；
2. 第一阶段冻结 Current，Current 的输出成为不可移动参考；
3. 使用全分辨率最终 decoder 特征 `u1` 提供局部定位；
4. q4 只提供经过安全归一化的粗上下文；
5. `p_out`、`p_d0`、二者差值只作为上下文；
6. 两个非负空间 gate 分别表示 rescue 和 suppression；
7. 总 logit 修正严格限幅；
8. 两个 gate 完全相同初始化，因此输出**精确等于 Current**，但二者首步梯度均非零。

### 6.2 公式

令：

\[
p_0=\sigma(z_{out}),\qquad p_d=\sigma(z_{d0}),
\]

安全 q4：

\[
\tilde q_c=\frac{q_{4,c}-\operatorname{mean}_{hw}(q_{4,c})}
{\max(\operatorname{RMS}_{hw}(q_{4,c}-\operatorname{mean}_{hw}),\tau)}.
\]

这里 `τ` 是下限，保证弱证据不会被放大。

上下文：

\[
X=[\phi(u_1),\psi(\tilde q),p_0,p_d,p_d-p_0,|p_d-p_0|,U],
\]

\[
U=4p_0(1-p_0).
\]

两个非负 gate：

\[
G_r=\sigma(f_r(X)),\qquad G_s=\sigma(f_s(X)).
\]

残差预算：

\[
B=\rho+(1-\rho)U,
\]

\[
\Delta z=L\,B\,(G_r-G_s),\qquad z_{v3}=z_{out}+\Delta z,
\]

其中 `L` 建议从 `0.15` 开始，并根据漏检目标跨阈值所需 logit 分布决定是否调整，绝不直接回到 V2 的 `1.0`。

### 6.3 精确 identity 与可学习性

将两个 gate 的最终卷积权重都置零、bias 都置为同一个有限值 `b0`：

\[
G_r^{(0)}=G_s^{(0)}=\sigma(b_0),
\]

所以：

\[
\Delta z^{(0)}=0
\]

是精确成立的。同时：

\[
\frac{\partial \Delta z}{\partial f_r}=LB\sigma(b_0)(1-\sigma(b_0))\neq0,
\]

\[
\frac{\partial \Delta z}{\partial f_s}=-LB\sigma(b_0)(1-\sigma(b_0))\neq0.
\]

这解决了 PBDR‑V2 “精确 identity 但 confidence 首步无梯度”的矛盾，并保证 rescue/suppression gate 始终非负。

### 6.4 完整核心模块代码

以下代码已通过 `py_compile`、exact‑identity 和首步梯度测试：

```python
"""Conservative, exactly identity-initialized final-logit calibrator.

The module treats q4 and d0 as context, not as direct residuals.  Two
non-negative gates are initialized identically, so their difference is exactly
zero while each gate still has a non-zero first derivative.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class PBDRV3RoutingOutput:
    routed_logits: torch.Tensor
    delta_logits: torch.Tensor
    rescue_gate: torch.Tensor
    suppression_gate: torch.Tensor
    uncertainty: torch.Tensor


class ConservativeResidualCalibratorV3(nn.Module):
    """Small, bounded calibrator for a trained Current checkpoint.

    Args:
        q_channels: Number of channels in the detached q4 evidence.
        local_channels: Number of channels in the final decoder feature.
        hidden_channels: Width of the routing trunk.
        residual_limit: Maximum absolute logit correction.
        evidence_floor: Per-channel RMS floor. Values below the floor are not
            amplified, which prevents weak q4 noise from being normalized to
            unit energy.
        uncertainty_floor: Minimum fraction of the residual budget away from
            the decision boundary.
        gate_bias_init: Equal initialization for rescue/suppression logits.
            Equal gates give an exact zero delta; a finite bias gives non-zero
            gradients for both branches.
    """

    def __init__(
        self,
        *,
        q_channels: int = 8,
        local_channels: int = 32,
        hidden_channels: int = 16,
        residual_limit: float = 0.15,
        evidence_floor: float = 1.0,
        uncertainty_floor: float = 0.25,
        gate_bias_init: float = -4.0,
        detach_local_feature: bool = True,
    ) -> None:
        super().__init__()
        if q_channels < 1 or local_channels < 1 or hidden_channels < 1:
            raise ValueError("all channel counts must be positive")
        if not math.isfinite(residual_limit) or residual_limit <= 0.0:
            raise ValueError("residual_limit must be finite and positive")
        if not math.isfinite(evidence_floor) or evidence_floor <= 0.0:
            raise ValueError("evidence_floor must be finite and positive")
        if not 0.0 <= uncertainty_floor <= 1.0:
            raise ValueError("uncertainty_floor must be in [0, 1]")
        if not math.isfinite(gate_bias_init):
            raise ValueError("gate_bias_init must be finite")

        self.q_channels = int(q_channels)
        self.local_channels = int(local_channels)
        self.hidden_channels = int(hidden_channels)
        self.residual_limit = float(residual_limit)
        self.evidence_floor = float(evidence_floor)
        self.uncertainty_floor = float(uncertainty_floor)
        self.detach_local_feature = bool(detach_local_feature)

        self.local_projection = nn.Sequential(
            nn.Conv2d(local_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.q_projection = nn.Sequential(
            nn.Conv2d(q_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GELU(),
        )

        # p_out, p_d0, signed disagreement, absolute disagreement, uncertainty.
        context_channels = 2 * hidden_channels + 5
        self.routing_trunk = nn.Sequential(
            nn.Conv2d(
                context_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1, bias=True),
        )
        final_projection = self.routing_trunk[-1]
        if not isinstance(final_projection, nn.Conv2d):
            raise RuntimeError("unexpected routing trunk")
        nn.init.zeros_(final_projection.weight)
        nn.init.constant_(final_projection.bias, gate_bias_init)

    @staticmethod
    def _require_map(value: torch.Tensor, *, name: str, channels: int) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if value.ndim != 4 or value.shape[1] != channels:
            raise ValueError(
                f"{name} must be BCHW with C={channels}, got {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise TypeError(f"{name} must have a floating dtype")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} contains non-finite values")

    def _safe_q4(self, q4: torch.Tensor) -> torch.Tensor:
        # Center each q4 channel spatially.  Unlike unit-RMS normalization, the
        # denominator floor never amplifies a weak/noisy channel.
        detached = q4.detach()
        working = detached.float() if detached.dtype in (torch.float16, torch.bfloat16) else detached
        centered = working - working.mean(dim=(2, 3), keepdim=True)
        rms = torch.sqrt(centered.square().mean(dim=(2, 3), keepdim=True) + 1.0e-8)
        normalized = centered / rms.clamp_min(self.evidence_floor)
        return normalized.to(dtype=detached.dtype)

    def forward_with_diagnostics(
        self,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        q4: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> PBDRV3RoutingOutput:
        self._require_map(z_out, name="z_out", channels=1)
        self._require_map(z_d0, name="z_d0", channels=1)
        self._require_map(q4, name="q4", channels=self.q_channels)
        self._require_map(
            local_feature,
            name="local_feature",
            channels=self.local_channels,
        )
        if z_out.shape != z_d0.shape:
            raise ValueError("z_out and z_d0 must have the same shape")
        if local_feature.shape[0] != z_out.shape[0] or local_feature.shape[-2:] != z_out.shape[-2:]:
            raise ValueError("local_feature must share batch and spatial shape with z_out")
        tensors = (z_d0, q4, local_feature)
        if any(value.device != z_out.device for value in tensors):
            raise ValueError("all routing inputs must be on one device")
        if any(value.dtype != z_out.dtype for value in tensors):
            raise ValueError("all routing inputs must have one dtype")

        local = local_feature.detach() if self.detach_local_feature else local_feature
        local_context = self.local_projection(local)
        q_context = self.q_projection(self._safe_q4(q4))
        if q_context.shape[-2:] != z_out.shape[-2:]:
            q_context = F.interpolate(
                q_context,
                size=z_out.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Context is detached so Stage 1 only trains the calibrator.  If the
        # final decoder is unfrozen later, it still receives the additive
        # routed-logit gradient and optionally the local-feature gradient.
        p_out = torch.sigmoid(z_out.detach())
        p_d0 = torch.sigmoid(z_d0.detach())
        disagreement = p_d0 - p_out
        uncertainty = (4.0 * p_out * (1.0 - p_out)).clamp_(0.0, 1.0)
        residual_budget = self.uncertainty_floor + (
            1.0 - self.uncertainty_floor
        ) * uncertainty

        context = torch.cat(
            (
                local_context,
                q_context,
                p_out,
                p_d0,
                disagreement,
                disagreement.abs(),
                uncertainty,
            ),
            dim=1,
        )
        gate_logits = self.routing_trunk(context)
        gates = torch.sigmoid(gate_logits)
        rescue_gate = gates[:, 0:1]
        suppression_gate = gates[:, 1:2]

        # Both gates are non-negative.  Their equal initialization gives exact
        # identity, but their separate gradients allow immediate divergence.
        delta = (
            self.residual_limit
            * residual_budget
            * (rescue_gate - suppression_gate)
        )
        delta = delta.clamp(min=-self.residual_limit, max=self.residual_limit)
        routed = z_out + delta
        if not bool(torch.isfinite(routed).all()):
            raise FloatingPointError("routed logits contain non-finite values")
        return PBDRV3RoutingOutput(
            routed_logits=routed,
            delta_logits=delta,
            rescue_gate=rescue_gate,
            suppression_gate=suppression_gate,
            uncertainty=uncertainty,
        )

    def forward(
        self,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        q4: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_diagnostics(
            z_out,
            z_d0,
            q4,
            local_feature,
        ).routed_logits

```

---

## 7. 模型集成修改

目标文件建议新建：

```text
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v3.py
```

不要直接覆盖 V2，保留可复现实验链路。

### 7.1 安装模块

```python
from model.tpd_conservative_residual_calibrator_v3 import (
    ConservativeResidualCalibratorV3,
    PBDRV3RoutingOutput,
)


def _install_pbdr_v3(model: nn.Module) -> None:
    if hasattr(model, "pbdr_v3"):
        raise RuntimeError("PBDR-V3 integration attempted twice")
    reference = next(model.parameters())
    model.pbdr_v3 = ConservativeResidualCalibratorV3(
        q_channels=8,
        local_channels=model.outc.in_channels,  # 当前配置为 32
        hidden_channels=16,
        residual_limit=0.15,
        evidence_floor=1.0,
        uncertainty_floor=0.25,
        gate_bias_init=-4.0,
        detach_local_feature=True,
    ).to(device=reference.device, dtype=reference.dtype)
```

### 7.2 保留最终 decoder 特征并路由

把 V2 中：

```python
out = self.outc(self.up_decoder1(d2, x1))
```

改为：

```python
u1 = self.up_decoder1(d2, x1)
out = self.outc(u1)
```

`d0` 保持原公式，随后：

```python
routing = self.pbdr_v3.forward_with_diagnostics(
    z_out=out,
    z_d0=d0,
    q4=q4,
    local_feature=u1,
)
routed_out = routing.routed_logits
```

推理：

```python
if self.mode != "train":
    return torch.sigmoid(routed_out)
```

### 7.3 不要把训练辅助量伪装成第七/第八路 segmentation map

现有 `compute_tpd_training_loss` 会把 tuple 中每个元素都当作 segmentation probability 做 BCE。因此不能简单返回：

```text
(..., routed_prob, base_prob, delta)
```

否则 `base_prob` 和 `delta` 会被错误监督。

建议增加一个显式训练接口：

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PBDRV3TrainingAux:
    auxiliary_logits: tuple[torch.Tensor, ...]  # gt5, gt4, gt3, gt2, d0
    base_logits: torch.Tensor
    routed_logits: torch.Tensor
    routing: PBDRV3RoutingOutput


def forward_for_pbdr_v3_training(self, x: torch.Tensor):
    # 最好将现有 forward 主体抽到 _forward_impl，避免执行两次网络。
    probabilities, aux = self._forward_impl(x, return_pbdr_v3_aux=True)
    return probabilities, aux
```

普通 `forward` 仍保持 evaluator 所需的概率 tuple / 单张概率图契约；PBDR‑V3 独立 trainer 调用 `forward_for_pbdr_v3_training`。

---

## 8. 损失函数修改：相对 Current 的单向约束

### 8.1 目标

第一阶段 base 完全冻结，令 `p0` 为同一 forward 的 Current 输出、`p` 为 routed 输出。增加四类约束：

1. **背景不应被抬高：**
   \[
   L_{bg\uparrow}=\mathbb E_{y=0}[\operatorname{ReLU}(p-p_0-m_b)^2].
   \]
2. **目标不应被削弱：**
   \[
   L_{fg\downarrow}=\mathbb E_{y=1}[\operatorname{ReLU}(p_0-p-m_f)^2].
   \]
3. **信任域和稀疏残差：** 限制候选偏离 Current 的范围。
4. **hard‑negative top‑k：** 聚焦 Current 或 candidate 已给出较高概率的背景，而不是被海量简单背景稀释。

同时加入 soft IoU/Jaccard surrogate，使训练目标更接近 mIoU。Lovász‑Softmax 是更直接的 Jaccard surrogate，也可以作为后续消融；Focal Loss 可用于困难背景，但不应替代相对 Current 的单向约束。[^P1][^P2]

### 8.2 完整损失代码

建议新建：

```text
experiments/pbdr_v3_loss.py
```

以下代码已通过语法、前向和反向有限性测试：

```python
"""Metric-aligned objective for the conservative PBDR-V3 calibrator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class PBDRV3LossOutput:
    total: torch.Tensor
    final_bce: torch.Tensor
    soft_iou: torch.Tensor
    background_increase: torch.Tensor
    foreground_decrease: torch.Tensor
    trust_region: torch.Tensor
    residual_sparsity: torch.Tensor
    hard_negative: torch.Tensor
    deep_supervision: torch.Tensor


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = value[mask]
    if selected.numel() == 0:
        return value.new_zeros(())
    return selected.mean()


def soft_iou_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    dimensions = (1, 2, 3)
    intersection = (probability * target).sum(dim=dimensions)
    union = (probability + target - probability * target).sum(dim=dimensions)
    return (1.0 - (intersection + eps) / (union + eps)).mean()


def topk_hard_negative_loss(
    routed_logits: torch.Tensor,
    routed_probability: torch.Tensor,
    base_probability: torch.Tensor,
    target: torch.Tensor,
    *,
    candidate_floor: float = 0.05,
    topk_fraction: float = 0.02,
) -> torch.Tensor:
    if not 0.0 <= candidate_floor < 1.0:
        raise ValueError("candidate_floor must be in [0, 1)")
    if not 0.0 < topk_fraction <= 1.0:
        raise ValueError("topk_fraction must be in (0, 1]")
    negative = target < 0.5
    candidate = negative & (
        (base_probability >= candidate_floor)
        | (routed_probability >= candidate_floor)
    )
    logits = routed_logits[candidate]
    if logits.numel() == 0:
        return routed_logits.new_zeros(())
    # BCEWithLogits(z, 0) == softplus(z).
    losses = F.softplus(logits)
    k = max(1, int(math.ceil(losses.numel() * topk_fraction)))
    return losses.topk(k, sorted=False).values.mean()


def compute_pbdr_v3_loss(
    *,
    routed_logits: torch.Tensor,
    base_logits: torch.Tensor,
    delta_logits: torch.Tensor,
    target: torch.Tensor,
    auxiliary_logits: Sequence[torch.Tensor] = (),
    soft_iou_weight: float = 1.0,
    background_increase_weight: float = 8.0,
    foreground_decrease_weight: float = 4.0,
    trust_region_weight: float = 0.25,
    residual_sparsity_weight: float = 0.05,
    hard_negative_weight: float = 2.0,
    deep_supervision_weight: float = 0.0,
    background_margin: float = 0.0,
    foreground_margin: float = 0.0,
) -> PBDRV3LossOutput:
    """Train a correction head against a frozen or nearly frozen Current model.

    ``base_logits`` should be the Current logit from the same forward pass.
    Detaching it here makes the monotonic constraints one-way: the candidate
    must move relative to Current rather than moving both endpoints together.
    """
    if routed_logits.shape != target.shape or base_logits.shape != target.shape:
        raise ValueError("routed/base/target shapes must match")
    if delta_logits.shape != target.shape:
        raise ValueError("delta_logits and target shapes must match")
    if background_margin < 0.0 or foreground_margin < 0.0:
        raise ValueError("monotonic margins must be non-negative")

    target_float = target.float()
    routed_float = routed_logits.float()
    base_float = base_logits.detach().float()
    probability = torch.sigmoid(routed_float)
    base_probability = torch.sigmoid(base_float)

    final_bce = F.binary_cross_entropy_with_logits(
        routed_float,
        target_float,
        reduction="mean",
    )
    iou = soft_iou_loss(probability, target_float)

    background = target_float < 0.5
    foreground = ~background
    background_increase = _masked_mean(
        F.relu(probability - base_probability - background_margin).square(),
        background,
    )
    foreground_decrease = _masked_mean(
        F.relu(base_probability - probability - foreground_margin).square(),
        foreground,
    )
    trust_region = (probability - base_probability).square().mean()
    residual_sparsity = delta_logits.float().abs().mean()
    hard_negative = topk_hard_negative_loss(
        routed_float,
        probability,
        base_probability,
        target_float,
    )

    deep_supervision = routed_float.new_zeros(())
    if deep_supervision_weight > 0.0:
        terms = []
        for index, logits in enumerate(auxiliary_logits):
            if logits.shape != target.shape:
                raise ValueError(
                    f"auxiliary_logits[{index}] shape differs from target"
                )
            terms.append(
                F.binary_cross_entropy_with_logits(
                    logits.float(),
                    target_float,
                    reduction="mean",
                )
            )
        if terms:
            deep_supervision = sum(terms)

    total = (
        final_bce
        + soft_iou_weight * iou
        + background_increase_weight * background_increase
        + foreground_decrease_weight * foreground_decrease
        + trust_region_weight * trust_region
        + residual_sparsity_weight * residual_sparsity
        + hard_negative_weight * hard_negative
        + deep_supervision_weight * deep_supervision
    )
    return PBDRV3LossOutput(
        total=total,
        final_bce=final_bce,
        soft_iou=iou,
        background_increase=background_increase,
        foreground_decrease=foreground_decrease,
        trust_region=trust_region,
        residual_sparsity=residual_sparsity,
        hard_negative=hard_negative,
        deep_supervision=deep_supervision,
    )

```

### 8.3 权重使用方式

代码中的权重是**起始值，不是已验证最优值**。建议：

- 先固定 `residual_limit=0.15`；
- `background_increase_weight=8`、`foreground_decrease_weight=4`；
- hard‑negative 权重可从 0 线性升到 `0.5–2.0`，避免训练初期被少量极端负例支配；
- Stage 1 base 冻结时，`deep_supervision_weight=0`，因为前五路没有可训练梯度；
- Stage 2 若解冻最后 decoder，可设 `deep_supervision_weight=0.05–0.10`，不要恢复六路等权求和以压过最终校准目标。

---

## 9. Warm‑start、冻结和优化器代码

### 9.1 必须从训练完成的 Current checkpoint 加载

```python
def load_trained_current_into_pbdr_v3(
    model: nn.Module,
    checkpoint_path: Path,
) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    current_state = payload["state_dict"]
    incompatible = model.load_state_dict(current_state, strict=False)

    expected_missing = {
        key for key in model.state_dict() if key.startswith("pbdr_v3.")
    }
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            f"unexpected missing keys: {sorted(incompatible.missing_keys)}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected Current keys: {sorted(incompatible.unexpected_keys)}"
        )
```

分别从 Current `best_miou` 与 Current `best_pd` 建立两个角色候选，不要再从 paired scratch initial state 开始。

### 9.2 Stage 1：只训练 PBDR‑V3，并冻结 BatchNorm 状态

只设置 `requires_grad=False` 不够；如果整个模型仍处于 `train()`，BatchNorm running mean/variance 仍会变化，使 Current 参考悄然漂移。

```python
def configure_stage1(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.pbdr_v3.parameters():
        parameter.requires_grad_(True)

    # 冻结 Current 的 dropout/BN 行为；只让校准器进入训练模式。
    model.eval()
    model.pbdr_v3.train()


def build_stage1_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    trainable = [
        parameter
        for parameter in model.pbdr_v3.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("PBDR-V3 has no trainable parameters")
    return torch.optim.AdamW(
        [{"params": trainable, "lr": 1.0e-4, "lr_scale": 1.0}],
        lr=1.0e-4,
        weight_decay=1.0e-4,
    )
```

每步可加：

```python
torch.nn.utils.clip_grad_norm_(model.pbdr_v3.parameters(), max_norm=1.0)
```

### 9.3 现有 scheduler 必须支持 param‑group 比例

仓库当前 `set_learning_rate` 会把所有参数组写成同一个 LR。Stage 2 使用差分学习率时应改为：

```python
def set_scaled_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = base_learning_rate * float(group.get("lr_scale", 1.0))
```

### 9.4 Stage 2：只有通过 Stage 1 后才允许小范围解冻

```python
def configure_stage2(model: nn.Module) -> torch.optim.Optimizer:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.pbdr_v3.parameters():
        parameter.requires_grad_(True)
    for module in (model.up_decoder1, model.outc):
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    # 小数据集下仍建议冻结所有 BatchNorm running statistics。
    model.eval()
    model.pbdr_v3.train()

    return torch.optim.AdamW(
        [
            {
                "params": list(model.pbdr_v3.parameters()),
                "lr": 3.0e-5,
                "lr_scale": 1.0,
            },
            {
                "params": list(model.outc.parameters()),
                "lr": 1.0e-5,
                "lr_scale": 1.0 / 3.0,
            },
            {
                "params": list(model.up_decoder1.parameters()),
                "lr": 5.0e-6,
                "lr_scale": 1.0 / 6.0,
            },
        ],
        weight_decay=1.0e-4,
    )
```

如果 Stage 1 已满足门控，不要为了“继续追分”自动进入 Stage 2；解冻只增加过拟合和轨迹漂移风险。

---

## 10. 数据与 hard‑negative 策略

### 10.1 先保持原 crop 配方，隔离架构效果

第一轮 V3 应保持当前 deterministic crop 和 augmentation 不变，避免同时改变模型与数据导致无法归因。当前协议每图每 epoch 一个 256 patch，正目标偏置概率为 0.5。[^R7]

### 10.2 第二轮再加入困难背景回放

从 **development train 的 out‑of‑fold Current 预测** 中提取：

- 未匹配预测连通域；
- `p0∈[0.30,0.70]` 的背景阈值边界；
- 目标邻域外的高概率低频光斑；
- 空目标图中的高响应区域。

建议采样混合起点：

```text
40% 正目标 crop
40% hard-negative crop
20% 无偏随机 crop
```

该比例必须在内部验证上固定，不能根据 official test 调整。不要直接用 test 假警回灌训练。

### 10.3 缓存以提高 GPU0 实验吞吐

Stage 1 base 冻结，可预计算并缓存：

```text
u1, q4, out, d0, target, sample_id
```

用 FP16 保存特征、FP32 保存 logits/评估即可。缓存前后应对若干样本做 routed output 按位或严格容差核对。这样可快速扫 residual limit、损失权重和 gate width，而不重复执行整网。

---

## 11. 开发/正式验证协议

### 11.1 快速工程通道

用于尽快判断机制是否可救：

1. 现有 NUAA Current checkpoint；
2. 从 213 个 train 中冻结一个 calibration split；
3. 只在其余样本训练 V3；
4. calibration split 选 epoch、权重和阈值；
5. 当前 test 只作一次对照。

注意：现有 Current 已见过全部 213 train，因此这个通道对 base 并非完全无偏，只能作为工程筛选。

### 11.2 论文级通道

复用仓库 `train_tpd_pilot.py` 已有的 mask 分层与 `stratified_split` 逻辑：

1. 在训练前冻结 train/val IDs、hash 和 split seed；
2. Current 与 V3 的 base 都只在 dev‑train 训练；
3. V3 warm‑start dev‑train Current，并仅在 dev‑val 选 epoch、threshold、loss 权重；
4. official test 最后访问一次；
5. 至少 3 个训练 seed，报告均值、标准差和逐图 paired bootstrap；
6. Pd 使用逐目标配对命中；Fa 使用逐图 unmatched pixels 和 false objects/image 的 bootstrap。

### 11.3 固定 0.5 与阈值校准必须分开报告

- `threshold=0.5`：架构/训练协议的严格可比结果；
- `threshold=val_selected`：部署工作点；
- 不允许在 test 上扫阈值后只报告最好点；
- 可加一个只学习 temperature+bias 的 scalar calibration control。若它已恢复大部分差距，说明问题主要是 logit calibration；若不能恢复，说明空间排序/形状被破坏。

现代神经网络的置信度可出现校准误差，temperature scaling 是常见诊断控制，但它只改变标度和工作点，不能修复空间排序。[^P3]

---

## 12. 推荐实验顺序与停止规则

| 顺序 | 实验 | 训练范围 | 回答的问题 | 停止条件 |
|---:|---|---|---|---|
| E0 | A0–A8 分支归因 + threshold/FROC | 无重训 | V2 到底坏在轨迹、direct 还是 disagreement | 得到逐分支结论后立即冻结报告 |
| E1 | Current scalar temperature+bias | 2 个 scalar | 是否主要为校准漂移 | 若不能在 val 上同时守住 Pd/Fa/mIoU，停止该线 |
| E2 | V3 twin‑gate，BCE+softIoU | 仅 V3 | 结构修正是否有效 | 未优于 Current 则不解冻 base |
| E3 | 加 `L_bg↑/L_fg↓/trust/sparse` | 仅 V3 | 是否抑制 Fa 且保持 Pd | 通过角色门控即可停止 |
| E4 | 加 OOF hard negatives | 仅 V3 | 是否进一步降低未匹配连通域 | 必须在 val 配对指标上提升 |
| E5 | 小范围解冻 `outc/up_decoder1` | V3 + 最后 decoder | borderline miss 是否需要特征微调 | 任一核心指标退化立即回退 E3/E4 |

不要一次性把 E2–E5 全部合并，否则即使提升也无法知道是哪项有效。

建议 Stage 1 的训练上限远低于 1000 epochs，例如最多 100–150 epochs，并依据内部验证早停。校准头很小，长时间重复观察同一验证集只会增加选择过拟合。

---

## 13. NUAA 双角色验收门槛

### 13.1 `best_miou` 角色

以 Current 为安全基线，建议候选必须同时满足：

| 条件 | 门槛 |
|---|---:|
| matched target count | `>= 256/263` |
| Fa 非退化 | `<= 1.5435e-5` |
| 推荐严格 Fa 增益 | `<= 1.466325e-5`（至少降低 5%） |
| mIoU 严格增益 | `>= 0.798483`（至少 +0.002） |
| nIoU | `>= 0.795348` |

### 13.2 `best_pd` 角色

| 条件 | 门槛 |
|---|---:|
| matched target count | `>= 257/263` |
| Fa 非退化 | `<= 1.4749e-5` |
| 推荐严格 Fa 增益 | `<= 1.401155e-5`（至少降低 5%） |
| mIoU 严格增益 | `>= 0.790553`（至少 +0.002） |
| nIoU | `>= 0.792668` |

对于正式选择，再增加：

- mIoU 逐图配对 bootstrap 的 95% 下界 `> 0`；
- Fa ratio 的 95% 上界 `< 1`；
- Pd 逐目标配对检验不显示显著退化；
- false objects/image 的尾部风险不增加。

---

## 14. 非退化部署门控代码

以下聚合门控只能保证固定 certification split；正式版本应再接入逐图 bootstrap 结果。

```python
"""Select a candidate only when it strictly dominates Current on a frozen split."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CertificationMetrics:
    matched_target_count: int
    target_count: int
    fa: float
    miou: float
    niou: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CertificationMetrics":
        return cls(
            matched_target_count=int(value["matched_target_count"]),
            target_count=int(value["target_count"]),
            fa=float(value["fa"]),
            miou=float(value["miou"]),
            niou=float(value["niou"]),
        )


@dataclass(frozen=True, slots=True)
class CertificationDecision:
    passed: bool
    selected: str
    checks: Mapping[str, bool]
    current: CertificationMetrics
    candidate: CertificationMetrics


def certify(
    current: CertificationMetrics,
    candidate: CertificationMetrics,
    *,
    minimum_miou_gain: float = 0.002,
    maximum_fa_ratio: float = 1.0,
    require_niou_non_decrease: bool = True,
) -> CertificationDecision:
    if current.target_count != candidate.target_count:
        raise ValueError("Current and candidate target counts differ")
    checks = {
        "pd_non_regression": (
            candidate.matched_target_count >= current.matched_target_count
        ),
        "fa_non_regression": candidate.fa <= current.fa * maximum_fa_ratio,
        "miou_strict_gain": candidate.miou >= current.miou + minimum_miou_gain,
        "niou_non_regression": (
            not require_niou_non_decrease or candidate.niou >= current.niou
        ),
    }
    passed = all(checks.values())
    return CertificationDecision(
        passed=passed,
        selected="candidate" if passed else "current",
        checks=checks,
        current=current,
        candidate=candidate,
    )


def write_decision(path: Path, decision: CertificationDecision) -> None:
    payload = {
        "schema": "sctransnet_non_regression_gate/v1",
        "passed": decision.passed,
        "selected": decision.selected,
        "checks": dict(decision.checks),
        "current": asdict(decision.current),
        "candidate": asdict(decision.candidate),
        "scope": "frozen_certification_split_only",
        "unseen_test_guarantee": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

```

部署流程：

```text
train candidate
      ↓
evaluate Current and candidate on frozen certification split
      ↓
certify(...)
      ↓
passed=true  -> deployment.json 指向 candidate
passed=false -> deployment.json 指向 Current
```

这才是可以兑现的“确保最终产物不变差”：不是承诺每个新模型都会提升，而是保证失败候选不会替换 Current。

---

## 15. 对 GPU 与三数据集流水线的具体安排

### GPU0

立即执行：

1. NUAA Current/PBDR 双 checkpoint 的 A0–A8 归因；
2. 保存逐图 logits、路由量和连通域归因；
3. threshold/FROC 扫描；
4. 训练 Current scalar calibrator control；
5. 训练 NUAA PBDR‑V3 Stage 1；
6. 仅在 V3 通过内部门控后决定是否进行 hard‑negative 或 Stage 2。

### GPU1 / GPU2

- 已在运行的 baseline 不变；
- **不要在 baseline 完成后自动串行启动 PBDR‑V2；** 将队列目标改为“等待 NUAA V3 机制门控结果”；
- 若 NUAA V3 失败，继续保留 Current，停止该结构跨数据集扩展；
- 若 NUAA V3 通过，先以完全冻结的同一超参数在 NUDT/IRSTD 运行，不根据各自 test 调权重。

### 跨数据集晋级规则

```text
NUAA dev gate 通过
  -> NUDT dev gate
     -> IRSTD dev gate
```

每个数据集都保留 Current fallback。三者中任何一个失败都应如实报告数据集依赖性，而不是在其 test 上继续搜索到过线。

---

## 16. 最小测试集

新增至少以下单元测试：

```text
test_pbdr_v3_exact_identity_at_initialization
 test_pbdr_v3_rescue_and_suppression_first_step_gradients_nonzero
 test_pbdr_v3_delta_is_bounded
 test_pbdr_v3_q4_is_detached
 test_pbdr_v3_weak_q4_is_not_amplified
 test_pbdr_v3_no_persistent_forward_cache
 test_pbdr_v3_stage1_only_router_trainable
 test_pbdr_v3_stage1_batchnorm_buffers_unchanged
 test_pbdr_v3_current_checkpoint_missing_keys_exactly_router_keys
 test_pbdr_v3_training_aux_not_consumed_as_segmentation_map
 test_non_regression_gate_falls_back_to_current
 test_threshold_selection_uses_validation_not_test
```

特别要做两项回归测试：

1. 初始化后 `torch.equal(routed_logits, z_out)`；
2. 一个 optimizer step 后，两个最终 gate channel 的梯度均非零且符号相反。

---

## 17. 最终判断

PBDR‑V2 在 NUAA 上失败，不是因为模块没生效，而是因为它同时违反了一个安全校准器应有的四个条件：

1. **参考模型不固定：** 从头联合训练导致 Current 轨迹丢失；
2. **证据不独立：** `d0` 含 `out` 和粗尺度头，却被硬解释为救援/抑制证据；
3. **残差不保守：** H/8 q4 可直接施加最大 ±1 logit，且弱证据被 RMS 单位化；
4. **目标不对齐：** 六路 BCE 不约束 0.5 阈值后的未匹配连通域 Fa。

当前最合理的研究路线不是继续微调 V2 的 19 个参数，而是：

```text
先做分支归因
-> 从训练好的 Current warm-start
-> 冻结 base/BN
-> 使用全分辨率局部特征的 twin-gate bounded calibrator
-> 加相对 Current 的背景/前景单向约束和 hard-negative loss
-> 用内部验证选择阈值与 checkpoint
-> 用 Current fallback 做非退化部署
```

即使 V3 最终没有跨数据集普遍提升，这条流程仍会给出可信结论，并确保失败候选不再进入部署链路。

---

## 18. 代码证据索引

[^R1]: `model/tpd_persistent_evidence_residual_router_v2.py`，行 6–20、31–48、84–116、131–175、187–239、240–301。PBDR‑V2 公式、RMS normalization、19 参数、signed strengths、直接残差与路由实现。
[^R2]: `model/SCTransNet.py`，行 505–535、542–581。Current 的输出头、`d0` 融合、训练六路输出与推理 `out`。
[^R3]: `model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v2.py`，行 152–256。q4 所在 stage、粗尺度上采样、`d0`、PBDR 接入及训练/推理输出。
[^R4]: `experiments/tpd_training_loss.py`，行 186–230。六路 segmentation BCE 有序求和与 TSS=0 精确路径。
[^R5]: `experiments/three_dataset_pbdr_v2_models_seed42_v1.py`，行 189–305；`experiments/train_three_dataset_pbdr_v2_tss_off_seed42_v1.py`，行 239–276。paired scratch、无 warm‑start、jointly trainable、test‑selected 声明。
[^R6]: `experiments/train_four_dataset_original_final_seed42_exact_v1.py`，行 839–840、888–932。统一 Adam、统一 LR scheduler 与训练循环。
[^R7]: `experiments/three_dataset_v2_protocol.py`，行 0–56、66–132、264–341；`experiments/paper_three_dataset_v2.py`，行 90–166。split 角色、NUAA 213/214、0.5 正目标偏置 crop 和 deterministic augmentation。
[^R8]: `experiments/train_tpd_pilot.py`，行 413–524。固定阈值、IoU、对象匹配与 Fa 的精确定义。

## 19. 方法参考

[^P1]: Lin et al., **Focal Loss for Dense Object Detection**, ICCV 2017。用于困难样本聚焦和前景/背景极不平衡场景。
[^P2]: Berman et al., **The Lovász‑Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection‑over‑Union Measure in Neural Networks**, CVPR 2018。
[^P3]: Guo et al., **On Calibration of Modern Neural Networks**, ICML 2017。temperature scaling 作为校准诊断与后处理控制。
