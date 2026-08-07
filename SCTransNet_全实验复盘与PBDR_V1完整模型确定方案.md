# SCTransNet 全实验复盘与最终候选 PBDR-V1 完整模型确定方案

> 项目：单帧红外小目标检测  
> 基线：Original SCTransNet  
> 当前生产基线：`SCTransNet + TPD8-MPRS-DCH + 五节点 NER4 Tail-Aware + QFG2-CROA`  
> 当前训练目标：TSS OFF，仅使用原六路 segmentation BCE  
> 正式数据集：NUAA-SIRST、NUDT-SIRST、IRSTD-1K  
> 当前正式协议：seed 42、1000 epochs、每 10 epochs 评估、`best_miou / best_pd`、固定阈值 0.5  
> 当前总裁决：`INCONCLUSIVE_MIXED_TRADEOFF`  
> 推荐最终候选：**PBDR V1 — Persistent-Evidence Bidirectional Dual Readout**  
> 中文：**持久证据双向双读出路由**
> 本文档状态：**PBDR-V1 六角色零训练审计已完成且未通过；未实现可训练 PBDR，未启动 PBDR 训练**

---

# 0. 先给出最终判断

## 0.1 当前还没有可以诚实宣布的“最终完整模型”

目前最可靠的生产基线仍为：

```text
SCTransNet
+ TPD8-MPRS-DCH
+ NER4 Tail-Aware
+ QFG2-CROA
+ TSS OFF
```

它已经具备：

```text
完整结构
完整代码
完整训练/推理构建
三个数据集、两个选模角色，共六份正式 checkpoint
工程复现闭环
```

但尚未建立：

```text
检测族在至少 2/3 数据集形成安全且实质改善
重叠族在至少 2/3 数据集形成安全且实质改善
至少 1/3 数据集在同一 best_miou checkpoint 上同时支持两族
六角色 severe degradation 为零，且不被 Original 实质支配
```

因此当前准确状态是：

```text
architecture_implemented=true
production_baseline_frozen=true
complete_model_performance_gate_passed=false
paper_core_established=false
stability_claim_supported=false
```

## 0.2 不能在训练前保证未知模型一定提升

任何新结构在正式训练前都不能被诚实地保证一定提高四项指标。

本方案所说的“确保性能必须提升”采用论文主表式工程含义，而不是要求每个数据集、每项
指标都第一。原 SCTransNet Table I 本身就是 mIoU/nIoU/F-measure 在三数据集领先，
但 NUAA Fa 为 13.92，高于 DNA-Net 的 8.78；NUDT Pd 为 98.62，低于
DNA-Net 的 98.83；IRSTD Pd 为 93.27，低于 UIU-Net 的 93.98。原文也明确说明
SCTransNet 没有在所有 Pd 和 Fa 上取得最优，而是强调检出与虚警的平衡。

> **候选不需要在每个数据集、每项指标上全部提升。正式验收使用 M2F-SV：检测族（Pd/Fa）至少获得 2/3 数据集支持，重叠族（mIoU/nIoU）至少获得 2/3 支持，至少 1 个数据集在同一 `best_miou` 工作点同时支持两族，六角色 severe 为零，且不被 Original 实质支配。所有数据集仍必须完整报告，不对原始异量纲指标求和，不把不同 checkpoint 拼成虚构工作点。**

## 0.3 下一候选不优先修改 TPD、NER、QFG 或 TSS 局部公式

已经完成的组件级诊断表明：

```text
NER stage2：未获得 V5 开发授权
QFG：有功能影响，但没有稳定跨数据集实质收益
TPD：七个 residual 均在工作，没有持续有害 block
GCSF：全局 skip 固定比例未建立训练触发
DS：统一重加权被跨数据集梯度反转否决
DORF：保守 readout 降 FP，但丢目标
NER-L4-TPR：恢复部分目标，但重新增加 FP，正式结果为 mixed trade-off
```

所以，在当前已测试配置范围内，下一步最有依据的变量不是继续优先改某个局部模块，而是：

> **利用当前网络已经训练好的两个互补读出 `out` 与 `d0`，并使用现有 NER `q4` 持久目标证据，在目标区域执行“乐观救援”，在背景区域执行“保守抑制”。**

---

# 1. 全部正式实验给出的共同结论

## 1.0 Original 与当前 TSS-off 的正式性能锚点

下表只使用各方法自己的同名 checkpoint、固定阈值 0.5 和同一数据划分。Pd、tiny-Pd
同时给出匹配计数与数值；Fa 的分子是未匹配预测连通域像素数。所有 checkpoint 都由
`img_idx/test` 每 10 epochs 选模，因此是 `test_selected=true` 的开发结果。

| 数据集 | 方法/角色 | epoch | Pd | Fa | mIoU | nIoU | tiny-Pd |
|---|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | Current best-mIoU | 850 | 256/263；0.973384 | 225/14577078；1.5435192156e-5 | 0.796482951 | 0.795348496 | 30/35；0.857143 |
| NUAA-SIRST | Original best-mIoU | 830 | 255/263；0.969582 | 387/14577078；2.6548530508e-5 | 0.786824655 | 0.795095699 | 32/35；0.914286 |
| NUAA-SIRST | Current best-Pd | 820 | 257/263；0.977186 | 215/14577078；1.4749183616e-5 | 0.788553432 | 0.792667957 | 30/35；0.857143 |
| NUAA-SIRST | Original best-Pd | 440 | 260/263；0.988593 | 1181/14577078；8.1017608604e-5 | 0.726235741 | 0.748162904 | 34/35；0.971429 |
| NUDT-SIRST | Current best-mIoU | 420 | 936/945；0.990476 | 121/43515904；2.7805925852e-6 | 0.944406006 | 0.946423233 | 258/259；0.996139 |
| NUDT-SIRST | Original best-mIoU | 520 | 935/945；0.989418 | 109/43515904；2.5048313371e-6 | 0.945606984 | 0.947437024 | 258/259；0.996139 |
| NUDT-SIRST | Current best-Pd | 510 | 940/945；0.994709 | 290/43515904；6.6642301628e-6 | 0.937380628 | 0.939836330 | 258/259；0.996139 |
| NUDT-SIRST | Original best-Pd | 260 | 941/945；0.995767 | 601/43515904；1.3811042510e-5 | 0.915685942 | 0.925523307 | 258/259；0.996139 |
| IRSTD-1K | Current best-mIoU | 830 | 277/297；0.932660 | 618/52690944；1.1728770697e-5 | 0.660311541 | 0.665661745 | 23/30；0.766667 |
| IRSTD-1K | Original best-mIoU | 270 | 282/297；0.949495 | 1165/52690944；2.2110061266e-5 | 0.673542705 | 0.636874759 | 23/30；0.766667 |
| IRSTD-1K | Current best-Pd | 530 | 287/297；0.966330 | 1225/52690944；2.3248776868e-5 | 0.639986059 | 0.650812036 | 25/30；0.833333 |
| IRSTD-1K | Original best-Pd | 230 | 287/297；0.966330 | 2592/52690944；4.9192513993e-5 | 0.619140625 | 0.627173613 | 24/30；0.800000 |

这些数值说明当前模型已经具有竞争力，但 Current 与 Original 在三个数据集、两个角色上
仍是混合权衡，PBDR 的任务是改善这个真实锚点，而不是与论文中不同训练/测试协议的
报告值直接拼接比较。

## 1.1 TPD 主线

### 已建立

- 初代 TPD 在 NUDT 内部验证上明显优于 Original/Progressive 的若干 Pd–Fa–mIoU 工作点。
- V8-MPRS-DCH 的 residual 确实改变输出。
- 三数据集 21/21 个 dataset-block 组合中，目标区 residual RMS 高于背景区。
- 关闭全部七个 residual 没有改善综合性能。

### 未建立

- TPD 全面超过 SPD。
- 某个 TPD residual 是跨数据集持续有害块。
- 修改 TPD 公式能够解决当前 mixed trade-off。

### 裁决

```text
TPD_INCONCLUSIVE_NO_FORMULA_CHANGE
```

因此：

```text
TPD8 保留
七个 residual 保留
不设计第十种 TPD 模式
```

---

## 1.2 NER 主线

### 已建立

- NER V1–V3 未定型。
- NER V4 Tail-Aware 首次形成正式相对改善。
- 五节点证据和 `q4→q3→q2` 中继继续作为完整模型底座。
- NER `q4` tail evidence 对目标保护具有可表示信号。

### 未建立

- stage2 是稳定跨数据集瓶颈。
- NER V5-PER 应启动。
- NER-L4-TPR 可以替换全局生产模型。

### NER-L4-TPR 正式训练结果

相对当前 TSS-off Final：

| 数据集/角色 | 主要正向 | 主要负向 |
|---|---|---|
| NUAA best-mIoU | Pd 持平，Fa、mIoU、nIoU、两类 FP 同向改善 | pixel recall 略降 |
| NUAA best-Pd | Pd +1、tiny +2 | Fa、mIoU、nIoU 明显恶化 |
| NUDT best-mIoU | Pd +3 | Fa、mIoU、nIoU 退化 |
| NUDT best-Pd | Pd 持平、tiny +1、mIoU/nIoU 提升 | component FP 增加 |
| IRSTD best-mIoU | Pd +5、mIoU 提升 | Fa、background FP 大幅增加，nIoU 下降 |
| IRSTD best-Pd | 无完整优势 | Pd、tiny、Fa、IoU 均回退 |

联合结果：

```text
21 项更好
5 项相同
28 项更差
6 个 dataset-role 单元全部 incomparable
```

### 裁决

```text
NER_L4_TPR_MIXED_TRADEOFF_REPORTED_VECTOR
global_production_replacement_authorized=false
```

其最重要的研究价值是支持以下现象：

> 目标保护可以恢复 Pd，但仅在 feature fusion 层做重分配仍不足以控制最终 FP。

---

## 1.3 QFG 主线

### 已建立

- QFG 对模型输出有真实影响。
- QFG 只调制 Query，不改 K/V/CFN 和 decoder。
- QFG 的工程合同、零点初始化和推理集成完整。

### 未建立

- 任意一个 level 持续有害。
- QFG-off 比 full 更好。
- QFG V3 公式修改有证据基础。
- QFG 单独形成稳定跨数据集贡献。

### 裁决

```text
QFG_INCONCLUSIVE_NO_FORMULA_CHANGE
```

因此 QFG2 暂时冻结，不继续搜索 level 或 alpha。

---

## 1.4 TSS 主线

正式实验已经覆盖：

```text
固定正 TSS
动态 λ=0.0025 / 0.005 / 0.01
TSS-off
EC-TSS V3.1
```

结果表明：

- 正 TSS 在 NUDT/IRSTD 的部分 Pd、mIoU 工作点有用；
- TSS-off 在 NUAA、低 Fa 和部分 nIoU 工作点更稳；
- 没有任何正权重或 TSS-off 通过三数据集统一门；
- EC-TSS 有 4/6 独有非支配点，但关键性能门失败。

### 裁决

```text
EC_TSS_V3_1_PERFORMANCE_FAIL_STOP_TSS_OPTIMIZATION
```

因此最终候选继续使用：

```text
TSS OFF
```

TSS 不再作为核心创新，也不再继续调公式或 λ。

---

## 1.5 GCSF、Deep Supervision 与 DORF

### GCSF

固定比例反事实中：

- 更强调 L4 transformed branch 能降低部分 FP；
- 但会丢目标；
- 没有一个 mode 同时满足 safe-material 与零 severe。

裁决：

```text
GCSF_BRANCH_AUDIT_NO_TRAINING_AUTHORIZATION
```

### 六头 deep-supervision 梯度审计

发现：

- 局部冲突集中在 NUDT 的 NER 梯度路径；
- 相同监督头在 NUAA/IRSTD 却是正向协同；
- 存在 checkpoint-role 反转；
- 没有跨三数据集一致签名。

裁决：

```text
DS_GLOBAL_REWEIGHTING_BLOCKED_BY_DOMAIN_REVERSAL
```

### DORF

当前网络已经训练：

\[
d0=outconv(gt2,gt3,gt4,gt5,out)
\]

但历史推理只返回：

\[
out
\]

DORF 采用：

\[
z=z_{out}+\alpha(z_{d0}-z_{out})
\]

结果：

- 增大 \(\alpha\) 通常降低 FP；
- 同时增加漏检或 IoU 回退；
- 固定全图融合未通过 Trigger。

裁决：

```text
DORF_V1_ZERO_TRAINING_TRIGGER_FAILED
```

DORF 最重要的证据是：

> `d0` 是一个更保守的多尺度读出，但不能在整幅图像上统一替换 `out`，因为它会压低真实目标。

---

# 2. 当前真正剩余的性能矛盾

所有实验可以归纳成同一个冲突：

```text
目标增强方向
→ Pd / 局部 mIoU 上升
→ Fa、background FP 或 nIoU 可能变差

背景抑制方向
→ Fa / pixel FP 下降
→ Pd、tiny-Pd 或区域完整性可能变差
```

这个冲突在以下实验中反复出现：

```text
GCSF
DORF
NER-L4-TPR
TSS-on/off
best_miou / best_pd 两角色
```

因此，下一候选必须同时具备：

1. **目标区域允许增加 logit**；
2. **背景区域只允许降低高风险 logit**；
3. **不能在整图使用同一个融合方向**；
4. **必须复用已有证据，不能增加新的大分支**；
5. **初始状态必须严格等于当前完整模型**。

---

# 3. 推荐最终候选：PBDR V1

## 3.1 名称

> **PBDR V1**  
> Persistent-Evidence Bidirectional Dual Readout  
> 持久证据双向双读出路由

## 3.2 完整模型定义

候选完整模型为：

```text
SCTransNet
+ TPD8-MPRS-DCH
+ 五节点 NER4 Tail-Aware
+ QFG2-CROA
+ PBDR V1
+ TSS OFF
```

PBDR 不是新的 encoder、Transformer 或 decoder 分支。

它只复用三个已经存在的量：

```text
z_out：当前最终输出 raw logit
z_d0：现有多尺度 deep-supervision 融合 raw logit
q4：现有 NER stage4 持久目标证据
```

新增：

```text
1 个零初始化标量参数
1 个 state key
0 个 buffer
0 个卷积层
```

---

# 4. PBDR 数学设计

## 4.1 目标保护图

复用 NER-L4-TPR 已验证的 q4 目标保护定义：

\[
T_4
=
tail\_support(\operatorname{stopgrad}(q4),\kappa=1.5)
\]

\[
P_4
=
dilate_{3\times3}
\left[
\mathbf 1(T_4>0)
\right]
\]

再使用 nearest interpolation 恢复到输出分辨率：

\[
P
=
\operatorname{Up}_{nearest}(P_4)
\]

其中：

```text
P ∈ {0,1}
P 不参与反向传播
P=1 表示目标证据保护区
P=0 表示可执行保守抑制的区域
```

## 4.2 目标救援项

只在目标保护区内，当 `d0` 比 `out` 更强时增加 logit：

\[
R^+
=
P\cdot
\operatorname{ReLU}(z_{d0}-z_{out})
\]

如果：

```text
z_d0 <= z_out
```

则不修改当前目标响应。

## 4.3 背景抑制项

只在非保护区内，当 `out` 比保守 `d0` 更高时降低 logit：

\[
R^-
=
(1-P)\cdot
\operatorname{ReLU}(z_{out}-z_{d0})
\]

如果：

```text
z_out <= z_d0
```

则不执行抑制。

## 4.4 单参数双向路由

定义：

\[
g=\tanh(a)
\]

其中：

```text
a 为唯一可学习参数
a 初始化为 0
a 必须为有限数
g 的数学值域为 (-1,1)
浮点实现对 tanh 的数值饱和结果再做 nextafter 严格单位区间限幅
因此可学习 gate 在 float16/float32 中也保持 |g|<1
```

输出：

\[
\boxed{
z_{PBDR}
=
z_{out}
+
g
\left(
R^+-R^-
\right)
}
\]

当：

\[
g=0
\]

严格得到：

\[
z_{PBDR}=z_{out}
\]

作为零训练分析的闭区间边界，或在可学习模型的极限：

\[
g=1\quad\text{或}\quad g\rightarrow1
\]

则：

```text
保护区：选择 max(z_out, z_d0)
非保护区：选择 min(z_out, z_d0)
```

固定分析函数在 `g=1` 时严格得到下式；可学习 `tanh(a)` 只能逼近该边界：

\[
z_{PBDR}
=
\begin{cases}
\max(z_{out},z_{d0}), & P=1\\
\min(z_{out},z_{d0}), & P=0
\end{cases}
\]

## 4.5 为什么使用单一 g

首版不使用两个独立正/负参数，原因是：

- 减少搜索自由度；
- 强制“目标救援”和“背景抑制”共同发生；
- 避免只学会提高 Pd 而不降低 Fa；
- 新参数从 2 个降为 1 个；
- 更容易建立严格机制解释。

若最终学到：

```text
g <= 0
```

则机制 Gate 失败，PBDR 不得作为最终模型。

必须区分两类 gate：

```text
analysis_gate：零训练纯函数显式接收 g∈[0,1]，允许 g=1 oracle 边界
learned_gate：g=strict_unit(tanh(routing_logit))，routing_logit 有限且 |g|<1
```

`g=1` 可以用于验证 max/min 公式，但不能作为可学习候选通过训练授权的唯一依据。

---

# 5. PBDR 为什么可能同时改善四项核心指标

## 5.1 Pd

NER-L4-TPR 在部分数据集/角色中表明 q4 保护具有目标恢复信号。

PBDR 不在 feature 层重分配，而是在最终 logit 层：

```text
仅在 P=1 且 d0>out 时提高响应
```

这可以恢复：

- 被 final readout 低估；
- 但被多尺度 d0 保留；

的目标。

## 5.2 Fa

多数 DORF 冻结工作点表明 `d0` 相对 `out` 更保守。

PBDR 只在：

```text
P=0
且
out>d0
```

的位置降低 logit，因此专门处理缺乏持久目标证据的高响应区域。

## 5.3 mIoU

目标区域采用 max-like 救援可以补充欠分割像素；背景区域采用 min-like 抑制可以去除外溢像素。

两者均有机会提高全局交并比。

## 5.4 nIoU

nIoU 对逐图小目标完整性更敏感。

PBDR 避免全图 DORF 式统一压低，通过保护图保留目标，因此比固定全图融合更有机会同时改善逐图 IoU。

## 5.5 component-Fa 与 background FP

PBDR 的非保护区抑制会使逐像素正集收缩，因此理论上 background FP 不应因该固定 correction 增加。

这个单调性只对 `P=0` 的固定 checkpoint、固定阈值像素成立。`P=1` 区域仍可能因
`d0>out` 增加背景像素；连通域还可能因收缩发生断裂，所以不能把局部单调性写成整图
Fa 或 component-Fa 的保证。

但 component-Fa 仍可能因组件断裂而变化，所以正式 Gate 必须同时检查：

```text
component-Fa
unmatched component pixels
background FP
预测组件数
fragmentation
```

不能只看一个 Fa 小数。

---

# 6. PBDR 的创新性

PBDR 的创新点不是“再增加一个输出头”。

它是：

> 使用 NER 的持久目标证据，把已经训练好的多尺度融合读出 `d0` 与当前最终读出 `out` 分成相反方向路由：目标保护区只允许向更强读出移动，背景区只允许向更保守读出移动。

特点：

```text
目标/背景双向作用
复用现有 NER q4
复用现有 deep-supervision d0
零初始化
仅 1 个参数
不增加卷积
不增加独立预测分支
不改变 TPD/NER/QFG 公式
TSS OFF
```

需要保留的论文边界：

```text
novel_candidate_proposed=true
novelty_effectiveness_established=false
novelty_against_prior_art_verified=false
```

只有正式 Gate 通过后，才能将其写成当前实验协议下有效的最终贡献。PBDR 相对现有
条件读出、门控融合和后处理方法的外部新颖性尚未检索，不能仅凭仓库内命名宣布已建立。

---

# 7. 代码修改计划

## 7.1 新增独立模块

新增：

```text
model/tpd_persistent_evidence_bidirectional_readout.py
```

参考实现：

```python
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    relay_spatial_tail_support,
)
from model.tpd_ner_l4_target_protected_reallocation import (
    FORMAL_L4_PROTECTION_DILATION_KERNEL,
    FORMAL_L4_TAIL_Z_THRESHOLD,
    FORMAL_Q4_RELAY_CHANNELS,
)


PBDR_VERSION = "pbdr_v1_q4_protected_bidirectional_d0_out"
PBDR_LOCAL_STATE_KEYS = ("routing_logit",)


def _require_finite(value: torch.Tensor, *, name: str) -> None:
    finite = torch.isfinite(value).all()
    if value.device.type == "cuda":
        torch._assert_async(finite, f"{name} contains non-finite values")
    elif not bool(finite):
        raise FloatingPointError(f"{name} contains non-finite values")


def route_with_gate(
    z_out: torch.Tensor,
    z_d0: torch.Tensor,
    protection: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Pure routing formula used by learned forward and fixed-g audit."""
    disagreement = z_d0 - z_out
    target_rescue = protection * F.relu(disagreement)
    background_suppression = (1.0 - protection) * F.relu(-disagreement)
    return z_out + gate * (target_rescue - background_suppression)


class PersistentEvidenceBidirectionalReadout(nn.Module):
    """Route d0/out disagreement with detached NER q4 target evidence."""

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.routing_logit = nn.Parameter(
            torch.zeros((), device=device, dtype=dtype)
        )

    def gate(self) -> torch.Tensor:
        _require_finite(self.routing_logit, name="routing_logit")
        gate = torch.tanh(self.routing_logit)
        # torch.tanh(large_finite_value) can round to exactly +/-1 in
        # float16/float32.  Keep the learned gate strictly inside (-1, 1).
        one = torch.ones_like(gate)
        zero = torch.zeros_like(gate)
        strict_upper = torch.nextafter(one, zero)
        gate = torch.minimum(
            torch.maximum(gate, -strict_upper),
            strict_upper,
        )
        _require_finite(gate, name="gate")
        return gate

    @staticmethod
    def build_protection(
        q4: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if q4.ndim != 4 or q4.shape[1] != FORMAL_Q4_RELAY_CHANNELS:
            raise ValueError("PBDR requires q4 shaped Bx8xHxW")
        if not q4.is_floating_point():
            raise TypeError("PBDR q4 must be floating point")
        _require_finite(q4, name="q4")

        with torch.no_grad():
            tail = relay_spatial_tail_support(
                q4.detach(),
                z_threshold=FORMAL_L4_TAIL_Z_THRESHOLD,
            )
            binary = tail.gt(0.0).to(dtype=q4.dtype)
            protected = F.max_pool2d(
                binary,
                kernel_size=FORMAL_L4_PROTECTION_DILATION_KERNEL,
                stride=1,
                padding=FORMAL_L4_PROTECTION_DILATION_KERNEL // 2,
            )
            protected = F.interpolate(
                protected,
                size=output_size,
                mode="nearest",
            )

        return protected.detach()

    def forward(
        self,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        q4: torch.Tensor,
    ) -> torch.Tensor:
        if z_out.shape != z_d0.shape:
            raise ValueError("PBDR out/d0 logits must have equal shape")
        if z_out.ndim != 4 or z_out.shape[1] != 1:
            raise ValueError("PBDR logits must be Bx1xHxW")
        if not z_out.is_floating_point() or not z_d0.is_floating_point():
            raise TypeError("PBDR logits must be floating point")
        if z_out.device != z_d0.device or z_out.device != q4.device:
            raise ValueError("PBDR inputs must share one device")
        if z_out.dtype != z_d0.dtype or z_out.dtype != q4.dtype:
            raise ValueError("PBDR inputs must share one dtype")
        if z_out.shape[0] != q4.shape[0]:
            raise ValueError("PBDR q4/logit batch sizes must match")
        _require_finite(z_out, name="z_out")
        _require_finite(z_d0, name="z_d0")

        protection = self.build_protection(
            q4,
            tuple(z_out.shape[-2:]),
        )

        return route_with_gate(
            z_out,
            z_d0,
            protection,
            self.gate(),
        )

    def architecture_manifest(self) -> Dict[str, Any]:
        return {
            "pbdr_version": PBDR_VERSION,
            "evidence_source": "existing_ner_q4",
            "protection": "binary_tail_dilate3_nearest_upsample",
            "target_term": "P*relu(z_d0-z_out)",
            "background_term": "(1-P)*relu(z_out-z_d0)",
            "gate": "nextafter_strict_unit(tanh(routing_logit))",
            "fixed_analysis_boundary": "route_with_gate_accepts_g_equal_1",
            "initialization": "exact_zero",
            "parameters": 1,
            "state_keys": PBDR_LOCAL_STATE_KEYS,
            "persistent_buffers": 0,
        }
```

## 7.2 新增整合模型

新增：

```text
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_survival.py
```

不要覆盖当前正式模型。

整合 class 在注册 PBDR 时必须使用现有参数作为 device/dtype 参考：

```python
reference = next(self.parameters())
self.pbdr = PersistentEvidenceBidirectionalReadout(
    device=reference.device,
    dtype=reference.dtype,
)
```

必须为 PBDR 分别新增 training/inference builder、validator 和 exporter。当前 QFG
validator 使用 exact-type 检查并冻结参数量/state-key 数量，不能把 PBDR 子类直接交给
旧 validator。PBDR builder 必须显式要求：

```text
deepsuper=true
relay_enabled=true
mode=train（训练六输出）或 mode=test（单输出评估）
TSS heads 只存在于训练 class
```

## 7.3 Forward 修改位置

实际整合点必须是当前 QFG2 类中同时持有 `q4`/`out`/`d0` 局部变量的
`_forward_with_relay`，不能在外层 `forward` 返回后再做 PBDR，因为届时 `q4` 已不可访问。
训练类覆盖该方法；TSS-head-free 推理类复用同一个 PBDR 版
`_forward_with_relay`，沿用当前工程的集成方式。两类都由独立 validator 检查实际返回合同。

当前末端：

```python
out = self.outc(self.up_decoder1(d2, x1))
...
d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), dim=1))

if self.mode != "train":
    return torch.sigmoid(out)

return (
    sigmoid(gt5),
    sigmoid(gt4),
    sigmoid(gt3),
    sigmoid(gt2),
    sigmoid(d0),
    sigmoid(out),
)
```

候选改为：

```python
out = self.outc(self.up_decoder1(d2, x1))
...
d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), dim=1))

routed_out = self.pbdr(out, d0, q4)

if self.mode != "train":
    return torch.sigmoid(routed_out)

return (
    torch.sigmoid(gt5),
    torch.sigmoid(gt4),
    torch.sigmoid(gt3),
    torch.sigmoid(gt2),
    torch.sigmoid(d0),
    torch.sigmoid(routed_out),
)
```

原六项 segmentation loss 数量和加法顺序保持不变。

变化仅为：

```text
最后一项由 out 改为 routed_out
```

这里不能写成“训练完全不变”。在 `g=0` 的初始化点，全部旧参数的输出和梯度必须与
Current 逐元素相同；但 `g` 离开零点后，第六项 BCE 会通过 `routed_out` 同时改变
`out` 与 `d0` 的梯度，而 `d0` 又依赖 `gt2...gt5/out`。准确合同是：

> 不改变 TPD、NER、QFG 的结构公式和六路 loss 数量，但 PBDR 会在 learned gate 离开
> 零点后改变共享网络的梯度分配与优化轨迹。

整合测试必须分别覆盖：

```text
self.training=true,  self.mode="train" → 含 TSS 结构化训练返回，六路末项为 routed_out
self.training=false, self.mode="train" → legacy 六路返回，末项为 routed_out
self.training=false, self.mode="test"  → 单一 routed_out
deepsuper=false                       → validator 必须拒绝，不能静默绕过 PBDR
```

## 7.4 TSS 合同

继续：

```text
survival_weight=0
tss_objective_enabled=false
```

若沿用含 TSS heads 的训练 class：

- TSS logits 仍可能计算；
- loss 不读取；
- TSS 参数无梯度；
- 部署导出删除 TSS state。

## 7.5 State 与参数

PBDR 新增：

```text
pbdr.routing_logit
```

因此：

```text
新增参数：1
新增 state key：1
新增 buffer：0
训练图：10,870,229 参数，569 state keys
推理图：10,870,131 参数，565 state keys
```

当前 TSS-off checkpoint 只允许用于 strict-extension 加载等价、零训练分析和迁移 smoke：

```text
所有旧 key 严格相同
仅新增 pbdr.routing_logit
新参数必须等于 0
missing_keys == ["pbdr.routing_logit"]
unexpected_keys == []
其余 tensor 逐键完全一致
```

实现时不能把旧 checkpoint 直接 `strict=True` 加载进新模型；应先 `strict=False`，随后按
上述白名单逐项验证。这个 strict-extension 路径**不得**用作 Formal1000 性能训练的
warm start。Formal1000 固定为 seed42 scratch、fresh Adam、`parent_checkpoint=null`。

PBDR 不是零计算开销：现有 NER stage3 已经对 q4 计算一次 tail support，文档伪代码在
末端会再次计算 q4 RMS/标准化，并新增 3×3 max-pool、nearest upsample 和逐像素路由。
V1 为冻结 NER4 源码允许这次小规模重复，但必须报告参数量、单图延迟和峰值显存；不得
声称“完全复用已有 tail 计算结果”或“零推理开销”。

---

# 8. 训练前必须先做零训练方向与可用信号审计

不能直接启动 1000 epochs。

## 8.1 输入

使用当前 TSS-off 的六个正式 checkpoint：

```text
3 datasets
× best_miou / best_pd
```

每张图只执行一次 forward，捕获：

```text
z_out
z_d0
q4
target mask
```

`outc/outconv` 可使用临时 forward hook；q4 应从
`tpd_ner.fusions["4"]` 的一次模块调用中捕获，不能假设直接调用 `forward_stage` 会触发
`tpd_ner` 自身 hook。审计结束必须验证所有临时 hook 已恢复、每个 batch 的三个张量各
捕获一次。

### 8.1.1 统一推理数学设置

六角色必须统一使用下面的显式设置，禁止为了复现某个旧 JSON 而按数据集或 checkpoint
角色切换：

```text
torch.backends.cuda.matmul.allow_tf32 = false
torch.backends.cudnn.allow_tf32 = false
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.benchmark = false
torch.backends.cudnn.deterministic = true
torch.use_deterministic_algorithms(true)
```

旧三数据集评估产物没有绑定 cuDNN TF32 状态。预运行已确认，同一 NUAA checkpoint 在
当前环境切换 TF32 时可有 1–2 个背景像素跨过 0.5，且两个 checkpoint 角色不能由一个
隐式设置同时逐项复现。因此不允许反向按角色选择数学设置。本审计采用统一显式 FP32
设置，所有 T1/T2 差值均在同一个 checkpoint、同一次 forward、同一数学设置内将
`g>0` 与精确 `g=0` 比较。

旧评估 JSON 继续作为绑定来源并逐字段记录 drift，但不作为 PBDR 授权硬门。工程硬门是：

```text
checkpoint / 数据协议 / img_idx SHA 一致
g=0 raw logit 与 z_out bitwise equal
g=0 probability 与模型实际返回 bitwise equal
每个 batch 只 forward 一次，q4/out/d0 各捕获一次
```

这项修订只消除未记录的历史执行环境歧义，不放宽 T1–T5，也不改变任何 PBDR 候选指标。

这一步直接使用 `img_idx/test` 的 GT 和六个已选 checkpoint，因此属于新的
`test_selected` 配方开发，不是独立确认实验。它只能用于筛选当前 PBDR 方向，不能被
写成论文级泛化证据。

## 8.2 固定模式

从本修订版本起前瞻冻结：

```text
identity anchor：g=0
可学习范围内的授权候选：g∈{0.125, 0.25, 0.50, 0.75}
oracle 公式边界：g=1.00（只报告，不参与授权）
```

同一个 `g` 必须同时用于：

```text
三个数据集
两个 checkpoint 角色
```

不允许按数据集选 g。

这个“六角色共用一个固定 g”是有意设置的保守算力筛选，不是 PBDR 可学习性的必要
条件。正式 scratch 训练中，每个数据集、每个 checkpoint epoch 会保存各自学到的
`routing_logit`；因此固定-g 审计失败只能说明没有找到统一零训练工作点，不能证明
端到端学习后的 PBDR 不可表示。

## 8.3 必须记录的机制信号

### 目标救援可用性

在当前漏检 GT 区域统计：

```text
P=1 的目标比例
d0>out 的目标像素比例
P=1 且 d0>out 的漏检目标数
```

### 背景抑制可用性

在当前 background FP / unmatched components 中统计：

```text
P=0 的 FP 比例
out>d0 的 FP 比例
P=0 且 out>d0 的 FP 像素数
```

### 保护污染

令二值 GT 为 `Y`，必须统计：

```text
protection_occupancy = sum(P) / valid_pixels
protected_gt_coverage = sum(P*Y) / sum(Y)
protected_background_fraction = sum(P*(1-Y)) / sum(1-Y)
false_component_protected_fraction = false_component_pixels_in_P / all_false_component_pixels
protection_flip_rate = 相邻评估周期 P 发生翻转的有效像素比例（训练阶段记录）
```

## 8.4 训练授权 Trigger

先冻结比较定义：

```text
核心四项 = matched_target_count↑、unmatched_predicted_pixels↓、mIoU↑、nIoU↑
tiny-Pd = 安全项，不计入“至少两项核心改善”
FLOAT_EQ_ATOL = 1e-12，FLOAT_EQ_RTOL = 0
Fa 方向以整数 unmatched_predicted_pixels 裁决，小数 Fa 仅报告
```

`severe_degradation` 完整复用 DORF 合同：相对 Current 任一角色出现以下任一项即为
一次 severe violation：

```text
Δmatched_target_count <= -2
Δmatched_tiny_target_count <= -2
ΔmIoU <= -0.01
ΔnIoU <= -0.01
component FP pixels 增加 >= 25%
background FP pixels 增加 >= 25%
```

权威实现为 `analysis/compare_three_dataset_dorf_v1.py`，当前 SHA256：
`7503e738167a61103c14d251afd36ef668133caa099c3cabc7e7ce7e9cdb9cb5`。

至少存在一个相同固定且可学习范围内的 `g∈{0.125,0.25,0.50,0.75}` 满足：

```text
T1. best_miou 三数据集中至少 2/3：
    matched target 不下降
    unmatched predicted pixels 不增加
    mIoU / nIoU 不下降（按冻结浮点容差）
    tiny-Pd 不降
    且核心四项中至少两项严格改善

T2. 六角色 severe_degradation_violations == 0

T3. 三个 Current best_miou checkpoint 的各自全测试集均满足：
    missed_gt_objects_with_protected_rescue_pixels > 0
    其中对象必须被 Current out 在阈值 0.5 漏检，且其 GT 区内至少存在
    一个 P=1 且 d0>out 的像素

T4. 三个 Current best_miou checkpoint 的各自全测试集均满足：
    unmatched_fp_pixels_with_unprotected_suppression > 0
    其中像素必须属于 Current out 在阈值 0.5 的 unmatched 预测，且 P=0、out>d0

T5. 六角色的 protected_background_fraction 均 < 0.50
```

`0.50` 只用于排除保护图覆盖大半背景的退化情况，是本方案运行前冻结的工程上限，
不是论文机制结论；真正的背景质量仍由 unmatched/background FP 与 IoU 裁决。

`g=0` 必须逐位复现同一次 forward 的 Current 输出；与旧 Current JSON 的逐字段差异
单独报告，但不允许据此按角色切换推理数学设置。`g=1` 即使表现最好，也只作为 max/min
oracle 上界，不能单独触发训练授权。

若 Trigger 失败：

```text
pbdr_training_authorized=false
decision=PBDR_GLOBAL_FIXED_G_SCREEN_FAILED
```

本方案不启动 Formal1000，也不继续追加固定 g。该结果不写成
`PBDR_REPRESENTABILITY_NOT_ESTABLISHED`，因为固定全局 g 不是端到端学习的必要条件。

---

# 9. 单元测试计划

## 9.1 零点输出等价

当：

```text
routing_logit=0
```

必须满足：

```text
routed_out == out
```

整模型六个训练输出与当前 TSS-off 模型逐元素一致。

## 9.2 共享梯度等价

step 0：

```text
全部旧参数 gradient
==
当前 TSS-off 模型 gradient
```

新 `routing_logit` 允许非零梯度。

## 9.3 第一次 Adam step

相同 batch 与 optimizer：

```text
全部共享 model state 相同
全部共享 optimizer state 相同
仅 routing_logit 开始分化
```

## 9.4 正向公式

当：

```text
P=1
d0>out
g>0
```

输出必须提高。

当：

```text
P=1
d0<=out
```

输出不变。

## 9.5 负向公式

当：

```text
P=0
out>d0
g>0
```

输出必须降低。

当：

```text
P=0
out<=d0
```

输出不变。

## 9.6 g=1 边界

只通过纯函数 `route_with_gate(..., gate=1)` 验证：

```text
P=1 → max(out,d0)
P=0 → min(out,d0)
```

不得把 `routing_logit` 设为无穷来制造 `tanh(routing_logit)=1`。可学习模块另测所有有限
`routing_logit` 经 `nextafter` 严格限幅后均满足 `abs(g)<1`，并覆盖会使原始
`tanh` 在 float16/float32 中饱和为精确 `±1` 的大有限输入。

## 9.7 Stop-gradient

```text
protection.requires_grad == false
q4 不通过 protection 获得额外梯度
out/d0 通过 routed output 正常获得梯度
```

还必须分别记录第六项 loss 对 `routing_logit` 的目标区与背景区梯度贡献、绝对值比例和
方向。单一标量不能只凭最终 `g>0` 就证明 rescue 与 suppression 都实际参与。

## 9.8 有限性与输入合同

必须覆盖：

```text
routing_logit=±inf/NaN → 拒绝
z_out/z_d0/q4 含 inf/NaN → 拒绝
batch/device/dtype 不一致 → 拒绝
q4 通道数不是 8 → 拒绝
PBDR 新增的 z_out/z_d0/q4/routing_logit/gate 检查在 CUDA 上使用 async 路径
`relay_spatial_tail_support` 现有内部检查仍含同步 bool，V1 继承并在延迟报告中记录
```

## 9.9 State extension

```text
旧 TSS-off checkpoint 可 strict extension load
仅允许新增 pbdr.routing_logit
新增参数全零
训练图参数/state keys = 10,870,229 / 569
推理图参数/state keys = 10,870,131 / 565
```

## 9.10 Paired-scratch 初始化等价

Formal1000 不是只要同样写 `seed=42` 就算同起点。PBDR builder 必须复用 Current
的 Original/Final 共享 state 配对复制、TPD/NER/QFG 的 SHA 派生初始化子流和各模块
零点恢复语义。权威 builder 为 `experiments/four_dataset_models_seed42_v1.py`，当前
SHA256：`a7127fc334ea72b2021aa670f341b0d67a6070e82b9e780bd5c56ed555d0a4d3`。

必须通过一次同协议 Current/PBDR epoch-0 配对构建测试：

```text
PBDR 与 Current 的所有旧 state key 集合、shape、dtype 和 tensor 逐键完全相同
PBDR 只多 pbdr.routing_logit，且精确等于 0
PBDR 注册不消耗随机数，不改变 TPD/NER/QFG 初始化子流
paired-scratch metadata 写入 builder SHA、派生 seed 和两份 state SHA256
```

不能用“各自 `manual_seed(42)` 后独立构建”替代上述配对合同，否则旧参数起点
可能不同，Formal1000 比较不再是结构变量的同起点比较。

## 9.11 Exact resume

连续训练和 epoch 边界续训必须一致：

```text
model
optimizer
scheduler
RNG
DataLoader generator
best_miou
best_pd
PBDR mechanism diagnostics
```

## 9.12 推理导出

```text
训练模型 eval 输出
==
TSS-head-free PBDR inference 输出
```

导出时：

```text
保留 pbdr.routing_logit
删除 target_survival.*
保留 TPD/NER/QFG
保留 deep-supervision heads/outconv，因为 PBDR 推理需要 d0
```

同时执行第 7.3 节的三种 `self.training/self.mode` 返回合同测试，并确认
`deepsuper=false` 被 validator 明确拒绝。

---

# 10. 200-epoch durable pilot

Trigger 与全部测试通过后，运行：

| 数据集 | 模型 | Seed | 正式 scheduler | durable pause |
|---|---|---:|---:|---:|
| NUAA-SIRST | PBDR V1 | 42 | 1000 epochs | 200 |
| NUDT-SIRST | PBDR V1 | 42 | 1000 epochs | 200 |
| IRSTD-1K | PBDR V1 | 42 | 1000 epochs | 200 |

三套 run 均从 epoch 1 开始：

```text
scratch=true
fresh_adam=true
optimizer_inherited=false
scheduler_inherited=false
parent_checkpoint=null
routing_logit_init=0
old_state_init=paired_exact_current_epoch0
initialization_authority_sha256=a7127fc334ea72b2021aa670f341b0d67a6070e82b9e780bd5c56ed555d0a4d3
```

`durable pause=200` 只是同一 scratch 1000-epoch 轨迹的可恢复暂停点，不是从 Current
checkpoint 微调 200 epochs，也不是第二套 run。

Pilot 只检查：

```text
训练有限
g 离开 0
训练 batch 上 target rescue / background suppression 张量均可计算且有限
protection occupancy / flip rate 可记录
exact resume 可继续
```

epoch 200 的 `g` 符号和 test 指标全部记录，但不作为是否续跑的自适应性能条件。只要
不存在 NaN/Inf、参数从零点获得更新、产物合同完整且 exact resume 通过，就自动续跑至
1000；这样不会通过 epoch-200 test 表现额外筛选轨迹。

Pilot 不用于：

```text
选择 g
修改公式
选择 checkpoint
形成论文性能结论
```

---

# 11. Formal1000 实验

Pilot 通过后继续同一三个 run 至 1000 epochs。

Formal1000 明确禁止从六个 Current `best_miou/best_pd` checkpoint 中任一份 warm start。
零训练 audit 与 strict-extension smoke 使用旧权重，正式性能训练不使用旧权重。

比较对象：

```text
Original SCTransNet
当前完整 TSS-off Final
PBDR V1
```

辅助分析可引用：

```text
NER-L4-TPR
DORF
```

但不参与 PBDR 配方选择。

固定：

```text
seed=42
scratch=true
fresh_adam=true
parent_checkpoint=null
epochs=1000
eval_every=10
img_idx/train
img_idx/test
threshold=0.5
best_miou
best_pd
同一 evaluator
```

完整 checkpoint 选模 key 必须逐字段冻结，而不是只写角色名称：

```text
best_miou = [miou, pd, -fa, niou, tiny_pd, -test_loss, -epoch]
best_pd   = [pd, -fa, tiny_pd, miou, niou, -test_loss, -epoch]
```

当前权威实现：`experiments/train_four_dataset_original_final_seed42_exact_v1.py`，SHA256
`2b1be09e97c1d780359fe6227a464969129c8c6cb1aa59c8125636a023ce35d7`。PBDR trainer
必须直接复用或逐字段实现同一 key 语义，并由 source-lock 测试逐字段核对。

公平性边界必须同时写入最终 summary：PBDR 单次 scratch run 的数据、优化器、epoch、
评估频率和选模 key 与 Current/Original 对齐；但整个项目已经依次查看正 TSS、EC-TSS、
GCSF、DORF、NER-L4-TPR 和本轮固定-g audit，累计配方/结构搜索预算明显高于 Original。
所以 Formal1000 仍是 seed42、test-selected 的探索性开发结果，而不是独立确认。

---

# 12. “完整模型确定”硬 Gate

## 12.1 Primary role

最终完整模型只由：

```text
best_miou
```

作为主部署角色裁决。

`best_pd` 作为高召回安全角色，不允许把两个角色的有利指标拼接成一个虚构工作点。

## 12.2 Gate FM-A：M2F 跨数据集两指标族多数证据

主参考固定为当前 TSS-off Final，正向投票只来自三个真实
`best_miou` checkpoint。每个数据集等权，不对 Pd、Fa、mIoU、nIoU 的原始数值
直接求和。定义：

```text
ΔT  = candidate matched_target_count - Current matched_target_count
rFP = (Current component_false_positive_pixels
       - candidate component_false_positive_pixels)
      / Current component_false_positive_pixels
Δm  = candidate mIoU - Current mIoU
Δn  = candidate nIoU - Current nIoU
```

当 Current FP 分母为零时，candidate 也为零记 `rFP=0`；candidate 从零引入 FP
直接不通过检测安全条件。

检测族支持 `D+` 定义为：

```text
D_safe     = (ΔT > -2) AND (rFP > -0.05)
D_material = (ΔT >= 2) OR (rFP >= 0.05)
D+         = D_safe AND D_material
```

即：最多允许少检 1 个目标且 FP 增幅小于 5%；同时必须多检至少 2 个
目标，或将 FP 降低至少 5%。

重叠族支持 `O+` 定义为：

```text
O_safe     = (Δm > -0.005) AND (Δn > -0.005)
O_material = (Δm >= 0.005) OR (Δn >= 0.005)
O+         = O_safe AND O_material
```

即：mIoU/nIoU 任一项的回退都必须小于 0.005，且至少一项提高 0.005。
上述阈值复用已有 DORF 冻结 safe/material 合同，权威实现 SHA256 为
`7503e738167a61103c14d251afd36ef668133caa099c3cabc7e7ce7e9cdb9cb5`。

FM-A 同时要求：

```text
D+ 数据集数 >= 2/3
O+ 数据集数 >= 2/3
至少 1/3 数据集在同一 best_miou checkpoint 上同时满足 D+ 与 O+
```

这允许某个数据集或某项指标出现小幅真实回退，也不把 1e-4 级别的
数值波动计作模型实质成功。所有数据集结果必须完整报告。

## 12.3 Gate FM-B：Original 实质支配底线

为防止只超过 Current、但仍弱于 Original，先冻结两者 `best_miou` 的实际逐指标包络。
该包络是逐指标组合参考，不代表某一个真实 checkpoint 同时取得了全部数值。

| 数据集 | Pd 最大值 | unmatched FP pixels 最小值（Fa） | mIoU 最大值 | nIoU 最大值 | tiny-Pd 最大值 |
|---|---:|---:|---:|---:|---:|
| NUAA-SIRST | 256/263 | 225（1.5435192155794186e-5） | 0.796482950889985 | 0.795348496003674 | 32/35 |
| NUDT-SIRST | 936/945 | 109（2.504831337067018e-6） | 0.9456069844789357 | 0.947437024134738 | 258/259 |
| IRSTD-1K | 282/297 | 618（1.1728770697294776e-5） | 0.6735427048325414 | 0.6656617448460672 | 23/30 |

下表只保留为“逐项越过包络”的描述性强目标，不要求一个候选在三个数据集、
全部指标上同时越过由两个不同模型拼成的虚拟包络：

| 数据集 | Pd | unmatched FP pixels | mIoU | nIoU | tiny-Pd |
|---|---:|---:|---:|---:|---:|
| NUAA-SIRST | ≥257/263 | ≤224 | ≥0.796582950889985 | ≥0.795448496003674 | ≥32/35 |
| NUDT-SIRST | ≥937/945 | ≤108 | ≥0.9457069844789357 | ≥0.947537024134738 | ≥258/259 |
| IRSTD-1K | ≥283/297 | ≤617 | ≥0.6736427048325414 | ≥0.6657617448460672 | ≥23/30 |

FM-B 不要求候选在每个数据集超过 Original，只设置底线：任一数据集的
Original `best_miou` 都不得实质支配候选。以 Original 为 candidate、新模型为
reference，按 12.2 节相同定义重算 `D_safe/O_safe/D_material/O_material`：

```text
original_material_dominates_candidate =
    D_safe
    AND O_safe
    AND (D_material OR O_material)

original_material_dominates_candidate_dataset_count 必须等于 0
```

换言之，Original 可有小于安全带的边际回退，但如果它同时在四维上保持安全，
并在任一族达到实质改善（多检至少 2 个、FP 降低至少 5%、mIoU/nIoU 任一提高
至少 0.005），则新候选触发底线否决。Fa 以整数分子计算比例，避免科学
计数法舍入改变结果。

## 12.4 Gate FM-C：六角色 severe veto

下表是 Current `best_pd` 的检测参考值，不再作为三个数据集必须逐项不降的硬门：

| 数据集 | Current matched target 参考值 | Current tiny matched 参考值 |
|---|---:|---:|
| NUAA-SIRST | ≥257/263 | ≥30/35 |
| NUDT-SIRST | ≥940/945 | ≥258/259 |
| IRSTD-1K | ≥287/297 | ≥25/30 |

正向证据不来自 `best_pd`；它与 `best_miou` 一起构成 3 数据集 × 2 角色的
六角色安全否决。相对同数据集、同角色 Current，任一角色出现以下任一条即
`severe=true`：

```text
Δ matched target <= -2
Δ matched tiny target <= -2
Δ mIoU <= -0.01
Δ nIoU <= -0.01
component FP 增加 >= 25%
background pixel FP 增加 >= 25%
Current 某类 FP 为 0 而 candidate 从零引入该类 FP
```

六角色 `severe_role_count` 必须为 0。`best_pd` 的参考表仍完整报告，但不因其
单项得分而替代 `best_miou` 主角色裁决。

## 12.5 Gate FM-D：像素与组件口径闭合

在三个 `best_miou` 主角色中必须验证：

```text
component_false_positive_pixels == unmatched_predicted_pixels（同一 Fa 分子，只计算一次）
background FP、unmatched object、pixel F1、fragmentation 全部报告
这些附加单元另表描述，不加入 D+/O+ 投票，也不要求每个数据集逐项单调
任一单元若达到第 8.4 节 severe 条件，仍由 FM-C 否决
```

Fa 分子已在 FM-A 的 `rFP` 中参与检测族投票；FM-D 只验证字段闭合，
不把同一数值重复计算为第二个独立胜利指标。

pixel precision、pixel recall、预测组件数和逐类 unmatched component taxonomy 必须完整
报告，但不要求 precision 与 recall 同时单调，因为 FM-A 已分别约束目标检出、Fa 与 IoU。

fragmentation 使用现有 `analysis/diagnose_tpd_clean_v6_fragmentation.py` 的
`fragmented_gt_count/extra_fragments` 定义，当前 SHA256：
`98d584794a8fcf4d04352615c235fb88a7238323a59c59e1834d17cf8dd09a09`。

## 12.6 Gate FM-E：模型代码确实参与

硬条件只检查 PBDR 是否实际参与，避免把研究重心从性能重新变成单独证明原理：

```text
六个 checkpoint：所有新增路由参数有限
至少两个 best_miou checkpoint 的最终输出与无路由反事实不同
新增参数在训练中发生有限更新
```

同时报告但不设置新的后验阈值：`rescue_gt_share`、`suppression_background_share`、
保护区占比、保护背景比例、相邻评估周期 protection flip rate，以及目标/背景对
`routing_logit` 的梯度贡献。性能结论仍由 FM-A 至 FM-D 决定。

## 12.7 最终完整模型裁决

只有以下条件全部通过，才能设置：

```text
matrix_complete=true
same_recipe_across_datasets=true
detection_positive_dataset_count>=2
overlap_positive_dataset_count>=2
joint_positive_dataset_count>=1
severe_role_count=0
original_material_dominates_candidate_dataset_count=0
FM-D 口径闭合通过
FM-E 代码参与通过
```

通过后才能设置：

```text
decision=PBDR_COMPLETE_MODEL_SELECTED_SEED42_DEVELOPMENT
complete_model_architecture=
    SCTransNet+TPD8+NER4+QFG2+PBDR
tss_objective_enabled=false
complete_model_seed42_development_gate_passed=true
independent_confirmation=false
```

仍然保留：

```text
paper_core_established=false
stability_claim_supported=false
```

因为当前协议仍是：

```text
seed42
test_selected
selection_is_optimistic
```

随机性稳定性主张必须增加多 seed；论文核心还需要独立/官方测试、强对照和 PBDR
消融。两者不是“独立协议或多 seed”二选一。若继续固定 seed42，则必须持续保留
`stability_claim_supported=false`。

本 M2F-SV 规则是在查看 PBDR-V1 结果后修订的后续试验协议，必须显式标记：

```text
post_hoc_protocol_amendment=true
applies_to=PBDR_V2_and_later_scratch_runs_only
pbdr_v1_machine_decision_unchanged=true
```

它只能前瞻应用于 PBDR-V2 及之后的统一架构、统一配方 scratch run，不得追溯
改判 PBDR-V1 已冻结的 `PBDR_GLOBAL_FIXED_G_SCREEN_FAILED`。即使作描述性重算，
PBDR-V1 也不通过 M2F-SV：没有任何数据集获得 `D+`，IoU 改善远小于
0.005，且 `g=0.5/0.75` 存在 severe 角色。

---

# 13. 失败分支

## 13.1 Zero-training Trigger 失败

```text
decision=PBDR_GLOBAL_FIXED_G_SCREEN_FAILED
```

处理：

- 不按本方案启动 formal1000；
- 当前 TSS-off Final 保持生产基线；
- 不继续追加固定 g 或按数据集挑 g；
- 该结果只否定“六角色统一固定-g 零训练工作点”，不证明端到端 PBDR 不可学习。
- 若至少两个数据集存在同一候选的重叠质量正向信号，可保留 PBDR 研究族并定向修订结构；
  但失败的 V1 公式本身不得被改写成已通过。

## 13.2 g 学成负值

```text
decision=PBDR_MECHANISM_DIRECTION_REJECTED
```

说明训练偏好与设计方向相反，PBDR 不通过当前完整模型门；保留日志，不把负 gate
重新解释为双向救援机制。

## 13.3 Pd、mIoU 提升但 Fa 上升

说明目标救援有效、背景保护污染或 suppression 不足。

这不再自动判失败。若 Fa 增幅小于 5%，且 Pd 至少多检 2 个目标，则该数据集
仍可获得 `D+`；若 Fa 增幅达 25%，则触发 severe。介于 5%–25% 时该数据集
不贡献检测族正票，最终由其他数据集的多数证据决定。完整向量仍按
mixed trade-off 报告，不写成该数据集全面提升。

## 13.4 Fa 降低但 Pd 不升

说明模型更接近 target-protected DORF，尚未解决“提高检出且降低虚警”的完整目标。

这也不再自动判失败。若 FP 降低至少 5%，且最多只少检 1 个目标，该数据集可
获得 `D+`。这与原 SCTransNet 在 NUDT 上用略低 Pd 换取显著低 Fa 的主表逻辑一致。
若同时获得 `O+`，则该数据集可作为两族联合正向点。

## 13.5 只有少数数据集支持

```text
decision=PBDR_DOMAIN_SENSITIVE_MIXED_TRADEOFF
```

任一指标族只有 1/3 数据集支持时不通过；2/3 支持已经达到多数，不要求 3/3。
不能按数据集选择不同模型或配方，再拼成一个统一完整模型。

## 13.6 M2F 通过但 Original 底线未通过

说明 PBDR 改善当前模型的跨数据集主表，但至少在一个数据集上仍被 Original
以安全且实质的幅度支配。

可以作为候选保留，但不能宣布为本轮最终模型。

---

# 14. 推荐文件清单

## 模型

```text
model/tpd_persistent_evidence_bidirectional_readout.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_survival.py
```

## 分析

```text
analysis/analyze_three_dataset_pbdr_zero_training_v1.py
analysis/compare_three_dataset_pbdr_zero_training_v1.py
analysis/analyze_pbdr_rescue_suppression_alignment_v1.py
analysis/analyze_pbdr_component_fragmentation_v1.py
```

## 实验

```text
experiments/PBDR_V1_PROTOCOL.md
experiments/train_three_dataset_pbdr_tss_off_seed42.py
experiments/launch_three_dataset_pbdr_seed42.sh
experiments/evaluate_three_dataset_pbdr.py
experiments/compare_pbdr_current_original.py
experiments/finalize_pbdr_complete_model_v1.py
experiments/export_pbdr_inference_v1.py
```

## 测试

```text
tests/test_pbdr_zero_identity.py
tests/test_pbdr_shared_gradient_identity.py
tests/test_pbdr_first_adam_step_identity.py
tests/test_pbdr_bidirectional_formula.py
tests/test_pbdr_protection_stop_gradient.py
tests/test_pbdr_finite_input_contract.py
tests/test_pbdr_state_extension.py
tests/test_pbdr_paired_scratch_identity.py
tests/test_pbdr_exact_resume.py
tests/test_pbdr_return_modes.py
tests/test_pbdr_inference_export.py
tests/test_pbdr_source_lock.py
```

## 不修改

```text
TPD8 源码
NER4 源码
QFG2 源码
TSS / EC-TSS 源码
GCSF/DORF/TPR 历史代码
历史 checkpoint
历史 result 与 selector
```

---

# 15. 完整执行顺序

```text
Phase 0
封存所有历史正式结果
冻结当前 TSS-off Final 为 reference

Phase 1
从六个当前 checkpoint 捕获 q4、z_out、z_d0
执行 identity g=0、授权候选 g={0.125,0.25,0.5,0.75} 和 oracle g=1.0

Phase 2
计算 target rescue、background suppression、保护污染和 fragmentation
执行训练授权 Trigger

Phase 3
Trigger 通过后实现 PBDR 独立模块和整合模型
完成 architecture manifest 和 strict extension loader

Phase 4
完成零点输出、paired-scratch 旧 state 等价、共享梯度、第一 Adam step、双向公式、导出测试

Phase 5
完成普通 Python、python -O、RTX 5090 smoke、exact resume、source lock

Phase 6
三数据集各从 scratch 启动一个 1000-epoch scheduler，在 epoch 200 durable pause

Phase 7
Pilot 通过后继续同一三个 run 至 formal1000

Phase 8
固定 threshold=0.5，分别选择 best_miou / best_pd

Phase 9
与当前 TSS-off 和 Original 进行同角色比较
执行 FM-A 至 FM-E

Phase 10
全部通过：
确定完整模型为 TPD8+NER4+QFG2+PBDR，TSS-off

任一核心 Gate 失败：
PBDR 未通过本方案统一完整模型门，继续保留当前生产基线
如实报告 mixed/dominated 结果，不在看到结果后追加 gate、阈值或模块
```

## 15.1 PBDR-V1 零训练实测结果（2026-08-06）

六角色已使用统一 TF32-off、`highest` FP32 设置完成，六份输出的 analyzer SHA 均为
`341b0b62841dfa065b6a01010044098c3c0a899a3c1b7f9f904400c8962532c6`。
最终比较器裁决为：

```text
decision=PBDR_GLOBAL_FIXED_G_SCREEN_FAILED
passing_authorization_gates=[]
pbdr_implementation_authorized=false
pbdr_training_authorized=false
```

| g | T1（best-mIoU） | T2（六角色 severe） | T3 | T4 | T5 | 总通过 |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.125 | 否，0/3 | 是 | 否 | 否 | 是 | 否 |
| 0.250 | 否，0/3 | 是 | 否 | 否 | 是 | 否 |
| 0.500 | 否，1/3 | 否，1 个 severe 角色 | 否 | 否 | 是 | 否 |
| 0.750 | 否，1/3 | 否，1 个 severe 角色 | 否 | 否 | 是 | 否 |

三个 `best_miou` 的关键表现为：

| 数据集 | Current g=0 | 最有利的 PBDR 描述点 | Pd 变化 | Fa 变化 | mIoU 变化 | nIoU 变化 |
|---|---|---|---:|---:|---:|---:|
| NUAA-SIRST | 256/263；Fa 1.543519e-5 | 无安全改善点 | 0 | 最小为 0 | 最小门控即 -0.000165 | -0.000647 |
| NUDT-SIRST | 936/945；Fa 2.780593e-6 | g=0.75 | 0 | 0 | +0.000355 | +0.000496 |
| IRSTD-1K | 277/297；Fa 1.172877e-5 | g=0.25 | 0 | +1 component FP pixel | +0.000508 | +0.000413 |

没有任何数据集在任何授权候选点提高 `best_miou` Pd。NUAA 的 mIoU/nIoU 随门控下降；
NUDT 只改善 IoU，component Fa 不变且 background FP 增加；IRSTD 改善 IoU 时 component
Fa 上升。`g=0.5/0.75` 还在 IRSTD `best_pd` 造成 `-2` 个匹配目标的 severe 退化。

信号门进一步定位到结构限制：

```text
T3 missed target rescue：NUAA=2，NUDT=0，IRSTD-1K=4
T4 unprotected FP suppression pixels：NUAA=0，NUDT=0，IRSTD-1K=43
T5 protected background fraction：六角色均为 0.054–0.097，通过
```

NUAA/NUDT 的 unmatched FP 全部或几乎全部落在硬保护区内，当前 `(1-P)` 抑制支路无法
作用于真正需要压制的 FP；NUDT 的漏检目标又没有 `P=1,d0>out` 信号，单靠 `out/d0`
取强无法提高 Pd。因此失败不是继续细调固定 `g` 能解决的，PBDR-V1 按冻结规则关闭，
不实现、不训练、不启动 Formal1000。

下一候选若继续保持同一研究主线，应是 **PBDR-V2 Adaptive Evidence Residual Router**：

1. 保持 TPD8、五节点 NER4、QFG2 与 TSS-off 不变；
2. 用 q4 学习一个零初始化的直接 logit residual，使 `d0<=out` 的漏检目标仍有救援来源；
3. 将二值硬保护改为可学习软置信度，使高置信目标受保护、被 q4 误保护的 FP 仍可抑制；
4. rescue 与 suppression 使用独立零初始化强度，避免一个全局 `g` 同时控制相反任务；
5. 仍按同角色完整报告 Pd/Fa/mIoU/nIoU，并用 M2F-SV 裁决：检测族和
   重叠族各需 2/3 多数支持，至少一个联合正向点，六角色 severe 为零且不被
   Original 实质支配；不用加权和，也不要求每个数据集四项全部同向。

这属于对失败读出候选的定向结构修订，不改变已有完整模型主干，也不增加无关模块。

## 15.2 PBDR-V2 代码实现进度（2026-08-06）

PBDR-V2 已从方向描述收敛为 19 参数、5 state-key、0-buffer 的正式候选。
冻结公式为：

```text
q = per-sample RMS normalize(stopgrad(q4))
C = 0.05 + 0.90 * sigmoid(bilinear(conv_conf(q)))
Q = C * tanh(bilinear(conv_direct_no_bias(q)))
g+ = 0.5 * tanh(rescue_strength_raw)
g- = 0.5 * tanh(suppression_strength_raw)
z  = out + Q + g+*C*relu(d0-out) - g-*(1-C)*relu(out-d0)
```

所有新参数精确为零时，`C=0.5`、`Q=0`、`g+=g-=0`，因此 `z=out`。
实现文件为：

```text
model/tpd_persistent_evidence_residual_router_v2.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v2.py
```

训练图为 10,870,247 参数/573 state keys，推理图为 10,870,149 参数/569
state keys。三份测试文件共 14 项全部通过，已证明零点六路输出、旧参数
梯度和旧参数第一 Adam step 逐位一致，且 direct/rescue/suppression 三路可分别学习。
RTX 5090 GPU0 的 1×1×64×64 FP32 forward/backward smoke 也已通过，峰值分配显存约
107.14 MiB；FP16/BF16 autocast 的零点前向 dtype/value 也逐位一致，正式训练仍固定
FP32。

详细公式、代码合同和训练协议见：

```text
SCTransNet_PBDR_V2自适应证据残差路由设计与代码实现方案.md
```

独立 paired-scratch registry 与 trainer 已实现，训练图 573-key、推理图 569-key，
只保留 `best_miou` 和 `best_pd`。当前冻结源码已在 RTX 5090 上完成 1 图训练、1 图
测试、1 epoch 的完整引擎 smoke：阈值为 0.5，两份 selected checkpoint 均成功写入，
完成后 rolling resume 已删除。formal1000 尚未启动，因此不能提前声称性能已提升。

---

# 16. 推荐状态

```text
decision=PBDR_GLOBAL_FIXED_G_SCREEN_FAILED

current_production_model=
    SCTransNet+TPD8+NER4+QFG2
current_training_objective=TSS_OFF

tpd8_frozen=true
ner4_frozen=true
qfg2_frozen=true
tss_optimization_closed=true

next_candidate=PBDR_V2_ADAPTIVE_EVIDENCE_RESIDUAL_ROUTER
next_candidate_status=FORMAL_TRAINING_CHAIN_READY

pbdr_zero_training_audit_required=true
pbdr_zero_training_audit_implemented=true
pbdr_zero_training_audit_completed=true
pbdr_code_implemented=false
pbdr_training_authorized=false
pbdr_v1_fixed_formula_closed=true
pbdr_family_retained=true
pbdr_v2_design_authorized=true
pbdr_v2_formula_frozen=true
pbdr_v2_core_code_implemented=true
pbdr_v2_complete_model_code_implemented=true
pbdr_v2_cpu_tests_passed=true
pbdr_v2_rtx5090_smoke_passed=true
pbdr_v2_training_registry_implemented=true
pbdr_v2_trainer_implemented=true
pbdr_v2_evaluator_implemented=true
pbdr_v2_launcher_implemented=true
pbdr_v2_training_engine_smoke_passed=true
pbdr_v2_frozen_protocol_created=true
pbdr_v2_formal1000_started=true
pbdr_v2_nuaa_formal1000_status=COMPLETE_NEGATIVE_VS_CURRENT
pbdr_v2_nudt_formal1000_status=RUNNING_GPU0
pbdr_v2_irstd1k_formal1000_status=QUEUED_WAITING_FOR_BASELINE_NUDT_GPU2

complete_model_performance_gate=
    M2F_SV:
    detection_positive_datasets_ge_2
    AND overlap_positive_datasets_ge_2
    AND joint_positive_datasets_ge_1
    AND zero_severe_roles
    AND original_material_dominates_candidate_datasets_eq_0

post_hoc_protocol_amendment=true
applies_to=PBDR_V2_and_later_scratch_runs_only
pbdr_v1_machine_decision_unchanged=true

paper_core_established=false
stability_claim_supported=false
training_recipe_finalized=true
```

---

# 17. 最终结论

> **PBDR-V1 六角色零训练审计已完成，固定全局门控未通过。它在 NUDT/IRSTD 的部分工作点提高 mIoU/nIoU，但没有提高任何 best-mIoU Pd，也没有降低相应 Fa；NUAA 的 IoU 反而下降。失败来源是二值 q4 保护把 NUAA/NUDT 的 FP 同时保护，且 NUDT 漏检区不存在可由 d0 强读出救回的信号。因此不实现或训练 PBDR-V1，但保留 PBDR 研究主线。下一步只在读出候选内设计 PBDR-V2：加入 q4 直接零初始化 residual、可学习软置信度以及独立 rescue/suppression 强度，保持 TPD8、五节点 NER4、QFG2 和 TSS-off 主干不变。正式模型不要求每个数据集四项全部同向；它按 M2F-SV 裁决：检测族与重叠族均需至少 2/3 数据集支持，至少一个同时支持两族的 `best_miou` 工作点，六角色 severe 为零，且不被 Original 实质支配。该修订只前瞻用于 PBDR-V2 及之后的 scratch run，不改变 PBDR-V1 的机器裁决。**

---

# 18. 依据文件与代码

## 用户提供的权威结果汇总

```text
SCTransNet_历史模型实验结果总汇.md
```

重点包括：

- 当前总裁决；
- TPD/QFG/NER 固定权重诊断；
- GCSF；
- deep-supervision 梯度审计；
- DORF；
- NER-L4-TPR formal1000 与训练后比较。

## 仓库代码

- `model/SCTransNet.py`
  - 生成 `out`；
  - 生成 `d0=outconv(gt2,gt3,gt4,gt5,out)`；
  - 历史推理只返回 `out`。
- `model/tpd_ner_v8_mprs_dch_v4_tail_aware.py`
  - q4 tail support；
  - stop-gradient 持久证据。
- `model/tpd_ner_l4_target_protected_reallocation.py`
  - 已验证 q4 binary tail protection 与 dilation 合同。
- `model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py`
  - 完整 TPD8–NER4–QFG2 forward；
  - q4、out、d0 已在一次 forward 中可用。
- `analysis/analyze_three_dataset_dorf_v1.py`
  - raw-logit `out/d0` 融合和保守读出证据。
