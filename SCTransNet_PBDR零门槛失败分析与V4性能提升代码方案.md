# SCTransNet PBDR‑V2/V3 零正增益复盘与 PBDR‑V4 性能提升代码方案

> 审计对象：[`Arialliy/SCTransNet_main`](https://github.com/Arialliy/SCTransNet_main)  
> 公开代码快照：`106e4631b67bea6560e318e9ce593f59dabe3193`，提交说明为 `Add PBDR V3 staged validation workflow`  
> 结果依据：用户提供的 `SCTransNet_历史模型实验结果总汇(2).md`（更新时间 2026‑08‑07）  
> 本报告修订原则：**删除正增益幅度门槛；固定二值化阈值仍为 0.5；同角色第一个不同指标只要严格更好即获胜。**  
> 适用范围：NUAA‑SIRST、NUDT‑SIRST、IRSTD‑1K，`best_miou` 与 `best_pd` 两个独立角色。

---

## 1. 结论先行

### 1.1 对“不要设置门槛”的修正

这个要求是合理的。需要删除的是以下**性能接受门槛**：

- `mIoU 至少 +0.002`；
- `Fa 至少下降 5%`；
- 任意 epsilon、百分比改善或“达到某个 material gain 才算成功”的条件；
- “所有指标同时不退化”才允许使用候选的 Pareto/非退化门。

但应保留 `probability > 0.5`，因为它是 Pd、Fa、mIoU、nIoU 的**固定测量工作点**，不是性能提升门槛。新的裁决规则应为：

```text
best_miou:
  mIoU ↑ → Pd ↑ → Fa ↓ → nIoU ↑ → tiny-Pd ↑ → loss ↓

best_pd:
  Pd ↑ → Fa ↓ → tiny-Pd ↑ → mIoU ↑ → nIoU ↑ → loss ↓
```

在同一角色中，从左到右找到第一个不同项：候选严格更好即胜；完全相同则保留已有基线。代码仓库的跨数据集 PBDR‑V3 协议实际上已经采用了这种“零正增益”定义，因此下一版不应回到旧 PBDR‑V3 NUAA 协议中的 `+0.002` 等门槛。

### 1.2 当前真正的问题不是门槛，而是“比较对象”和“训练目标”

PBDR‑V2 在 NUAA 上相对 Current 明确退化，且产物审计证明路由真实参与了训练。PBDR‑V3 修复了 V2 的主要工程缺陷：改为 Current warm‑start、冻结主干、移除 q4/d0 直接残差、限制 logit 修正，并使用局部特征作为上下文；但最终仍没有超过当前可用的最强基线包络。

最关键的新发现是：PBDR‑V3 的正式跨数据集部署比较只在 **Candidate 与 Original** 之间选择，Current 仅作为诊断项。PBDR 是从 Current warm‑start 的增量模块，因此“性能提升”必须至少同时比较：

```text
Original  vs  Current  vs  Candidate
```

按上传结果做三方零门槛重算后，PBDR‑V3 在六个数据集/角色单元中**没有一个超过 Original+Current 的同角色最优包络**：

| 数据集 / 角色 | Original | Current | PBDR‑V3 | 三方零门槛胜者 |
|---|---:|---:|---:|---|
| NUAA / best_miou | mIoU 0.786825 | **0.796483** | 0.795396 | **Current** |
| NUAA / best_pd | **260/263** | 257/263 | 257/263 | **Original** |
| NUDT / best_miou | **mIoU 0.945572** | 0.944406 | 0.944794 | **Original** |
| NUDT / best_pd | **941/945** | 940/945 | 940/945 | **Original** |
| IRSTD / best_miou | **mIoU 0.673485** | 0.660312 | 0.662016 | **Original** |
| IRSTD / best_pd | Pd 都为 287/297；Fa 4.9193e‑5 | **Fa 2.3249e‑5** | Fa 2.4198e‑5 | **Current** |

因此，PBDR‑V3 的 IRSTD `best_pd` “胜 Original”是一个真实的成对结果，但不能写成“超过现有最强模型”，因为同角色 Current 的 Fa 更低。NUAA `best_miou` 也同理：Candidate 胜 Original，但低于 Current。

### 1.3 推荐路线

下一步不要继续扩大 PBDR‑V3 的训练预算，也不要再加入性能幅度门槛。建议按下面顺序执行：

1. **先做零训练残差再标定。** 对已完成的 PBDR‑V3 输出，分别调节正残差、负残差和全局 bias；固定最终阈值 0.5，用内部验证集的零门槛角色序选择。该步骤最有希望直接修复 NUDT `best_pd` “只少 1 个目标但 Fa/mIoU 大幅领先”的状态。
2. **实现 PBDR‑V4 Role‑Aligned Component Calibrator。** `best_miou` 与 `best_pd` 使用不同的残差容量和不同的组件级损失，不再只靠同一个 BCE+soft‑IoU 配方、再在 checkpoint 选择阶段区分角色。
3. **正式选择必须对比 Original+Current 包络。** Original、Current 都放入候选池；不需要任何正增益幅度，只取严格角色序最大值。
4. **第一阶段冻结主干；第二阶段作为预先定义的并行候选分支。** 第二阶段只解冻 `outc` 与 `up_decoder1`，不由性能门触发；Stage‑1、Stage‑2、V3 再标定和两个基线统一进入候选池。

---

## 2. 证据复核：PBDR‑V2 是真实失败

### 2.1 NUAA 正式结果

| 角色 | 模型 | Pd | Fa | mIoU | nIoU |
|---|---|---:|---:|---:|---:|
| best_miou | PBDR‑V2 | 254/263 = 0.965779 | 2.4628e‑5 | 0.782599 | 0.793443 |
|  | Current | 256/263 = 0.973384 | 1.5435e‑5 | 0.796483 | 0.795348 |
| best_pd | PBDR‑V2 | 257/263 = 0.977186 | 3.6221e‑5 | 0.769457 | 0.781190 |
|  | Current | 257/263 = 0.977186 | 1.4749e‑5 | 0.788553 | 0.792668 |

相对 Current：

- `best_miou` 少检 2 个目标，Fa 上升 59.56%，mIoU 下降 0.013884；
- `best_pd` 检出数相同，Fa 上升 145.58%，mIoU 下降 0.019097，nIoU 下降 0.011478；
- 训练 1000/1000 完成、100 个评估点完整、双 checkpoint 正确、573 keys 正确、19 个 PBDR 参数均由零更新为非零、TSS state 为零。

这排除了“程序没有生效”“路由未进入优化器”“checkpoint 保存错误”等伪解释。PBDR‑V2 的失败是结构和目标函数造成的真实性能退化。

### 2.2 PBDR‑V2 的代码级失败链

#### 2.2.1 `d0 - out` 不是独立的漏检/假警证据

`d0` 由 `gt2、gt3、gt4、gt5、out` 再经 `outconv` 融合，包含 `out` 自身。于是：

```text
d0 - out
= 多尺度融合误差
+ out 的自相关项
+ 粗尺度上采样光晕
+ outconv 训练轨迹变化
```

它不能被硬解释为“d0 高于 out 就是漏检救援，out 高于 d0 就是假警抑制”。粗尺度预测对小目标周围产生的光晕，也会被当成正向 rescue 证据。

#### 2.2.2 H/8 的 q4 被允许产生 ±1.0 的全分辨率直接 logit 残差

PBDR‑V2 的核心是：

```text
Q = C * tanh(conv(q4))
routed = out + Q + g_r R+ - g_s R-
```

`Q` 的幅度可接近 ±1.0 logit。q4 是 H/8 证据，双线性上采样到全分辨率后天然是块状、平滑和带光晕的；对固定 0.5 工作点而言，接近阈值的背景像素只需很小正偏移就会形成新连通域，从而直接推高 component‑Fa。

#### 2.2.3 全局 RMS 归一化会放大弱证据

V2 对整个样本的 q4 做全局 RMS 归一化，分母最小仅为 `1e-6`。当 q4 本身很弱时，它仍被归一化到近似单位能量；同时没有逐通道去均值，DC 偏置可保留并被上采样扩散。该处理破坏了“证据强度”本身的信息。

#### 2.2.4 scratch 联合训练改变了 Current 的基础轨迹

V2 不是在训练完成的 Current 后面增加一个只训练 19 参数的后校准器，而是把第六路训练输出从 `out` 改成 `routed_out` 后重新联合训练。即使路由初始输出等于 `out`，第一个更新之后梯度路径已改变，整个主干不再是 Current。零初始化只能保证第 0 步相同，不能保证 1000 个 epoch 后仍保持父模型能力。

#### 2.2.5 训练损失与正式指标拓扑不一致

像素 BCE 不知道“新产生了一个孤立连通域”，也不知道“某个 GT 目标质心附近是否至少存在一个匹配组件”。因此极少量阈值穿越像素可能几乎不影响 BCE，却让 component‑Fa 和 Pd 发生离散变化。

---

## 3. PBDR‑V3 修复了什么，为什么仍未超过基线包络

### 3.1 已经正确的部分，应当保留

PBDR‑V3 的以下设计是正确方向：

- 从同数据集、同角色 Current checkpoint warm‑start；
- Stage‑1 冻结 Current 全部参数、BatchNorm 统计和随机层行为，只训练 6,018 个 PBDR 参数；
- q4 逐通道空间去均值，并使用 RMS floor=1.0，避免弱证据被放大；
- q4 与 d0 只作为上下文，不直接注入 residual；
- 使用全分辨率 `u1` 局部解码特征；
- 初始 `routed == out` 精确成立；
- 正式跨数据集协议已经采用零正增益角色序，不需要再加 `+0.002` 或百分比门槛。

### 3.2 失败原因一：两个角色使用同一个训练目标

跨数据集 PBDR‑V3 对 `best_miou` 与 `best_pd` 都使用 `core` 配方：

```text
BCEWithLogits(routed, target) + soft-IoU(routed, target)
```

角色只在 checkpoint 选择时才出现。也就是说：

- `best_miou` worker 与 `best_pd` worker 的梯度目标完全相同；
- `best_pd` 并没有任何按目标组件等权的 rescue 损失；
- `best_miou` 也没有专门压制 unmatched predicted components 的组件损失；
- 两个 worker 仅由不同 Current parent 和不同 selection key 间接分化。

这无法稳定学习两种相反操作：`best_pd` 需要更积极地把漏检目标推过 0 logit；`best_miou` 更需要修轮廓、削假警并保持已有目标。

### 3.3 失败原因二：±0.15 logit 只覆盖非常窄的阈值邻域

V3 最大正修正为 `+0.15`。要从负 logit 跨过 0，父模型必须满足：

```text
z_out > -0.15
p_out > sigmoid(-0.15) = 0.462570
```

而实际 delta 还要乘以 uncertainty budget 和两个 gate 的差值，通常达不到理论上限。因此 V3 只能救回“已经几乎达到 0.5”的像素，无法处理概率 0.2–0.4 的真正漏检小目标。

这与正式结果高度一致：

- NUDT `best_pd` 的 Candidate 相对 Original 将 mIoU 提高 0.022223、nIoU 提高 0.014885、Fa 近乎减半，却少检 1 个目标；
- NUAA `best_pd` 的 Candidate 区域质量和 Fa 大幅优于 Original，却少检 3 个目标、少检 4 个 tiny target；
- IRSTD `best_miou` 大幅降低 Fa、提高 nIoU，却少检 4 个目标。

这不是“校准器完全不会工作”，而是它擅长抑制和轮廓校准，却没有足够正向容量救深漏检。

### 3.4 失败原因三：正负残差完全对称，但类别风险不对称

V3 使用同一个 `0.15` 上限处理 rescue 与 suppression。红外小目标任务中：

- 对 `best_pd`，漏掉一个小目标的代价是对象级 Pd 下降一个离散单位；
- 背景像素数量远多于目标像素，BCE 的总体梯度更容易支持抑制；
- 一个很小的负残差即可删掉边缘目标，而正残差却可能不足以救回深漏检。

下一版必须允许：

```text
best_pd:  positive_limit >> negative_limit
best_miou: positive_limit ≈ negative_limit，但仍由组件保护约束目标
```

### 3.5 失败原因四：四个深监督读出被压缩成一个 d0

模型已经计算了全分辨率 `gt2、gt3、gt4、gt5`，但 V3 只把它们经 `outconv` 压缩后的 d0 交给校准器。这样丢失了非常重要的可辨识信息：

- 多尺度一致高：更像持久目标；
- 仅最粗尺度高：更像上采样光晕；
- 仅 out 高而四尺度均低：更像局部噪声或过锐假警；
- 各尺度方差大：说明读出不确定，应扩大上下文判断而不是直接平均。

V4 应直接接收四路 logits，并显式构造 mean/max/min/std/consensus/spread。

### 3.6 失败原因五：双 gate 在 identity 点存在公共模态空方向

V3 用 `sigmoid(rescue) - sigmoid(suppression)`。两个 gate 在初始化时完全相等，公共方向同时抬高或降低两者不会产生一阶 residual；模型浪费一部分容量去学习两个高度相关的场。一个单一 signed residual score 可以更直接地表示正负修正，并保留精确 identity。

### 3.7 失败原因六：全分辨率上下文只有一个 3×3 主干

单个 3×3 卷积对“目标组件、邻域光晕、孤立热噪点、细长背景结构”的区分能力有限。q4 虽提供大感受野，但它上采样后缺少精确边界。V4 使用 dilation 1/2/4 的并行深度卷积，以较小参数量增加 3、5、9 像素尺度上下文。

### 3.8 失败原因七：正式比较没有使用最强基线包络

跨数据集协议记录 Candidate‑vs‑Current，但正式部署只选 Candidate 或 Original。这样可能出现：

```text
Candidate > Original
但 Current > Candidate
```

IRSTD `best_pd` 正是这种情况。下一版必须把 Original 与 Current 都放入同一个零门槛候选池，不能让增量模型通过选择一个较弱对手获得“提升”标签。

---

## 4. 评估实现的额外代码风险

仓库根目录 `metrics.py::PD_FA.update` 仍存在两个需要隔离的问题：

1. 预测组件与 GT 按遍历顺序做贪心匹配，并在命中后删除预测组件；当多个候选都在 3 像素内时，结果可能依赖 regionprops 顺序。
2. 未匹配组件由“面积值是否出现在 matched area 列表”决定，而不是按组件 ID 决定。若一个 matched component 与一个 unmatched component 面积相同，后者可能被错误排除。

这不一定解释 PBDR‑V2 的相对退化，因为同一评估器会同时作用于两模型；但它会破坏“组件级训练 atlas 与正式评估完全一致”的前提。处理原则是：

- 不回写历史指标；
- 从当前正式 evaluator 中抽出唯一的 `match_components_v2`；
- evaluator、组件 atlas、单元测试共用同一函数；
- 增加“相同面积的 matched/unmatched 组件”“多个预测组件都在 `<3` 范围内”两个测试；
- 新旧 evaluator 在所有历史保存预测上并行重放，差异单独报告。

---

## 5. 立即执行：PBDR‑V3 零训练残差再标定

这是成本最低、最可能立刻产生严格改善的步骤。V3 已保存：

```text
base_logits = out
candidate_logits = out + delta
```

新增三个标量，不改变固定 0.5 工作点：

```python
z = out + a_pos * relu(delta) - a_neg * relu(-delta) + bias
```

解释：

- `a_pos` 单独放大 rescue；
- `a_neg` 单独减弱或放大 suppression；
- `bias` 用于整体恢复接近漏检的目标；
- `(a_pos, a_neg, bias)=(0, 0, 0)` 精确包含 Current；
- `(1, 1, 0)` 精确包含原 PBDR‑V3；
- 最终二值化仍为 `sigmoid(z) > 0.5`；
- 不设置任何最小改善幅度，直接按角色序选择。

建议预注册候选网格，而不是根据 official test 调参：

```python
POS_SCALE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
NEG_SCALE = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
BIAS = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
```

网格不是性能门槛，只是候选模型参数。内部验证选择完成后固定一个组合，再访问 official test。

### 5.1 为什么 NUDT `best_pd` 最值得先做

NUDT V3 `best_pd` 相对 Original：

- Pd 仅少 1 个目标；
- Fa 从 `1.3811e‑5` 降至 `6.5263e‑6`；
- mIoU 从 `0.915652` 提高到 `0.937875`；
- nIoU 从 `0.925479` 提高到 `0.940365`。

它拥有非常大的 Fa 和区域质量余量。小幅提高 `a_pos`、减小 `a_neg` 或加入轻微正 bias，有较高概率恢复这一个目标，同时仍保持后续指标优势。

### 5.2 再标定代码片段

```python
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class ResidualCalibration:
    positive_scale: float
    negative_scale: float
    bias: float

def apply_residual_calibration(
    base_logits: torch.Tensor,
    delta_logits: torch.Tensor,
    config: ResidualCalibration,
) -> torch.Tensor:
    return (
        base_logits
        + config.positive_scale * F.relu(delta_logits)
        - config.negative_scale * F.relu(-delta_logits)
        + config.bias
    )
```

评估时必须缓存每张图的 raw logits，避免每个标量组合重复跑网络。所有组合在同一个内部验证预测缓存上计算 Pd/Fa/mIoU/nIoU；Original、Current 也进入候选池。

---

## 6. PBDR‑V4：Role‑Aligned Component Calibrator

### 6.1 设计目标

PBDR‑V4 不再追求一个统一的“既救援又抑制”校准器同时服务两个角色，而是共享代码、分开实例化：

#### `best_miou` 角色

- 首要目标：mIoU；
- 对正负 residual 使用近似平衡容量；
- 通过 suppress component loss 降低未匹配组件；
- 通过 preserve component loss 保持已有目标；
- 使用多尺度一致性修正轮廓，而不是全局提高概率。

#### `best_pd` 角色

- 首要目标：对象级 Pd；
- 正向 logit 上限显著大于负向上限；
- 每个漏检 GT component 等权，不让大目标像素数压过 tiny target；
- 允许把 `p≈0.22` 的像素推过 0.5：`sigmoid(-1.25)=0.222700`；
- suppression 只保留较小容量，避免为了少量 Fa 删除目标。

### 6.2 保留与删除

| 项目 | 决策 |
|---|---|
| Current warm‑start、主干冻结、BN 冻结 | 保留 |
| q4 逐通道 centered RMS floor | 保留 |
| d0 只作上下文 | 保留 |
| `u1` 全分辨率局部特征 | 保留 |
| twin rescue/suppression gate | 删除，改为单 signed score |
| 对称 ±0.15 residual | 删除，改为角色非对称容量 |
| uncertainty floor=0.25 | 删除，避免无条件修改高置信像素 |
| 只使用 d0 代表四个深监督读出 | 删除，直接输入 gt2–gt5 |
| BCE+soft‑IoU 同时训练两角色 | 删除，改为角色组件损失 |
| Candidate 只与 Original 比较 | 删除，改为 Original+Current 包络 |
| `+0.002`、5% 等正增益门槛 | 删除 |

### 6.3 完整核心模块代码

建议新增：

```text
model/tpd_role_aligned_residual_calibrator_v4.py
```

```python
"""Role-aligned, component-aware residual calibration for SCTransNet.

The module is an exact identity extension at initialization.  It consumes the
frozen Current model's full-resolution decoder feature, NER q4 evidence, final
readout, d0 readout, and the four individual deep-supervision logits.  Unlike
PBDR-V3, it uses one signed residual score, asymmetric positive/negative logit
budgets, and a wider multi-dilation context trunk.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


Role = Literal["best_miou", "best_pd"]


@dataclass(frozen=True, slots=True)
class PBDRV4RoutingOutput:
    routed_logits: torch.Tensor
    delta_logits: torch.Tensor
    signed_score: torch.Tensor
    rescue_budget: torch.Tensor
    suppression_budget: torch.Tensor
    uncertainty: torch.Tensor
    consensus: torch.Tensor


def _require_bchw(value: torch.Tensor, *, name: str, channels: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4 or value.shape[1] != channels:
        raise ValueError(
            f"{name} must be BCHW with C={channels}, got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


class RoleAlignedResidualCalibratorV4(nn.Module):
    """Exact-identity residual calibrator with role-specific capacity.

    ``best_pd`` receives a larger positive logit budget and a smaller negative
    budget so the head can recover targets that are farther below probability
    0.5.  ``best_miou`` uses a more balanced budget.  These are architecture
    bounds, not performance acceptance margins.
    """

    def __init__(
        self,
        *,
        role: Role,
        q_channels: int = 8,
        local_channels: int = 32,
        hidden_channels: int = 24,
        positive_limit: float | None = None,
        negative_limit: float | None = None,
        evidence_floor: float = 1.0,
        detach_context: bool = True,
    ) -> None:
        super().__init__()
        if role not in ("best_miou", "best_pd"):
            raise ValueError(f"unsupported role: {role!r}")
        if min(q_channels, local_channels, hidden_channels) < 1:
            raise ValueError("all channel counts must be positive")
        if hidden_channels % 6 != 0:
            raise ValueError("hidden_channels must be divisible by 6")
        if evidence_floor <= 0.0 or not math.isfinite(evidence_floor):
            raise ValueError("evidence_floor must be finite and positive")

        if positive_limit is None:
            positive_limit = 0.60 if role == "best_miou" else 1.25
        if negative_limit is None:
            negative_limit = 0.50 if role == "best_miou" else 0.20
        for name, value in (
            ("positive_limit", positive_limit),
            ("negative_limit", negative_limit),
        ):
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and positive")

        self.role = role
        self.q_channels = int(q_channels)
        self.local_channels = int(local_channels)
        self.hidden_channels = int(hidden_channels)
        self.positive_limit = float(positive_limit)
        self.negative_limit = float(negative_limit)
        self.evidence_floor = float(evidence_floor)
        self.detach_context = bool(detach_context)

        self.local_projection = nn.Sequential(
            nn.Conv2d(local_channels, 16, kernel_size=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.q_projection = nn.Sequential(
            nn.Conv2d(q_channels, 8, kernel_size=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.GELU(),
        )

        # p_out, p_d0, four auxiliary probabilities and eight statistics.
        scalar_context_channels = 14
        self.context_stem = nn.Sequential(
            nn.Conv2d(16 + 8 + scalar_context_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(6, hidden_channels),
            nn.GELU(),
        )
        self.context_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=hidden_channels,
                        bias=False,
                    ),
                    nn.GroupNorm(6, hidden_channels),
                    nn.GELU(),
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.context_fuse = nn.Sequential(
            nn.Conv2d(3 * hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(6, hidden_channels),
            nn.GELU(),
        )
        self.residual_head = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)

        # Exact Current identity, while d(delta)/d(weight) is non-zero.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _normalize_q4(self, q4: torch.Tensor) -> torch.Tensor:
        detached = q4.detach()
        working = detached.float()
        centered = working - working.mean(dim=(2, 3), keepdim=True)
        rms = torch.sqrt(centered.square().mean(dim=(2, 3), keepdim=True) + 1.0e-8)
        normalized = centered / rms.clamp_min(self.evidence_floor)
        return normalized.to(dtype=detached.dtype).detach()

    def forward_with_diagnostics(
        self,
        *,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        auxiliary_logits: Sequence[torch.Tensor],
        q4: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> PBDRV4RoutingOutput:
        _require_bchw(z_out, name="z_out", channels=1)
        _require_bchw(z_d0, name="z_d0", channels=1)
        _require_bchw(q4, name="q4", channels=self.q_channels)
        _require_bchw(
            local_feature,
            name="local_feature",
            channels=self.local_channels,
        )
        if len(auxiliary_logits) != 4:
            raise ValueError("auxiliary_logits must contain gt2, gt3, gt4 and gt5")

        full_resolution = (z_d0, local_feature, *auxiliary_logits)
        for index, tensor in enumerate(full_resolution):
            expected_channels = self.local_channels if index == 1 else 1
            _require_bchw(
                tensor,
                name=f"full_resolution[{index}]",
                channels=expected_channels,
            )
            if tensor.shape[0] != z_out.shape[0] or tensor.shape[-2:] != z_out.shape[-2:]:
                raise ValueError("all full-resolution inputs must match z_out")
            if tensor.device != z_out.device or tensor.dtype != z_out.dtype:
                raise ValueError("all routing inputs must share device and dtype")
        if q4.shape[0] != z_out.shape[0]:
            raise ValueError("q4 batch size must match z_out")
        if q4.device != z_out.device or q4.dtype != z_out.dtype:
            raise ValueError("q4 must share device and dtype with z_out")

        local = local_feature.detach() if self.detach_context else local_feature
        local_context = self.local_projection(local)
        q_context = self.q_projection(self._normalize_q4(q4))
        q_context = F.interpolate(
            q_context,
            size=z_out.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(dtype=z_out.dtype)

        # The parent graph remains an immutable reference during Stage 1.
        p_out = torch.sigmoid(z_out.detach())
        p_d0 = torch.sigmoid(z_d0.detach())
        aux_probability = torch.cat(
            [torch.sigmoid(value.detach()) for value in auxiliary_logits], dim=1
        )
        aux_mean = aux_probability.mean(dim=1, keepdim=True)
        aux_max = aux_probability.amax(dim=1, keepdim=True)
        aux_min = aux_probability.amin(dim=1, keepdim=True)
        aux_std = aux_probability.std(dim=1, keepdim=True, unbiased=False)
        consensus = torch.sigmoid(8.0 * (aux_probability - 0.5)).mean(
            dim=1, keepdim=True
        )
        uncertainty = (4.0 * p_out * (1.0 - p_out)).clamp(0.0, 1.0)
        support_gap = aux_mean - p_out
        spread = aux_max - aux_min

        scalar_context = torch.cat(
            (
                p_out,
                p_d0,
                aux_probability,
                aux_mean,
                aux_max,
                aux_min,
                aux_std,
                consensus,
                uncertainty,
                support_gap,
                spread,
            ),
            dim=1,
        )
        context = torch.cat((local_context, q_context, scalar_context), dim=1)
        stem = self.context_stem(context)
        multi_scale = torch.cat(
            [branch(stem) for branch in self.context_branches], dim=1
        )
        fused = self.context_fuse(multi_scale)
        signed_score = torch.tanh(self.residual_head(fused))

        if self.role == "best_pd":
            # Do not prevent a strong learned target cue from rescuing a logit
            # that lies well below zero.
            rescue_budget = torch.ones_like(uncertainty)
        else:
            rescue_budget = (
                uncertainty + F.relu(aux_max - p_out)
            ).clamp(0.0, 1.0)
        suppression_budget = (
            uncertainty
            + F.relu(p_out - aux_mean)
            + F.relu(p_out - p_d0)
        ).clamp(0.0, 1.0)

        # torch.where selects the positive branch at exact zero, so the zero
        # initialized head has a non-zero first derivative and exact identity.
        delta = torch.where(
            signed_score >= 0.0,
            self.positive_limit * rescue_budget * signed_score,
            self.negative_limit * suppression_budget * signed_score,
        )
        routed = z_out + delta
        if not bool(torch.isfinite(routed).all()):
            raise FloatingPointError("routed logits contain non-finite values")

        return PBDRV4RoutingOutput(
            routed_logits=routed,
            delta_logits=delta,
            signed_score=signed_score,
            rescue_budget=rescue_budget,
            suppression_budget=suppression_budget,
            uncertainty=uncertainty,
            consensus=consensus,
        )

    def forward(
        self,
        *,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        auxiliary_logits: Sequence[torch.Tensor],
        q4: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_diagnostics(
            z_out=z_out,
            z_d0=z_d0,
            auxiliary_logits=auxiliary_logits,
            q4=q4,
            local_feature=local_feature,
        ).routed_logits
```

该实现已经满足两个关键工程性质：

- 初始化时 `routed_logits` 与 `z_out` 逐位一致；
- 对随机输入做 BCE backward 时，零初始化 `residual_head.weight` 的全部 24 个通道均获得有限非零梯度。

默认参数量约 11,497，仅为主模型的极小增量。

---

## 7. 组件级错误 atlas

### 7.1 为什么必须用组件，而不是只用像素 mask

正式 Pd 对每个 GT component 计一次；tiny target 即使只有 1–9 个像素，也应与大目标拥有同等对象权重。正式 Fa 又只统计未匹配预测组件中的像素。像素 BCE 会按面积加权，天然不符合这两个定义。

因此，对每个角色的 frozen Current parent，在**官方训练索引**上生成静态 atlas：

- `rescue_component_ids`：没有匹配预测组件的 GT components；
- `suppress_component_ids`：没有匹配 GT 的预测 components；
- `preserve_component_ids`：已匹配的 GT components；
- 每个 component 使用独立正整数 ID，0 表示不属于该类。

匹配必须复用同一个 canonical matcher：8 连通，质心距离 `<3`，一对一。

### 7.2 数据流水线修改

每张训练图保存：

```text
<image_id>.npz
  rescue_ids:  int32 [H, W]
  suppress_ids:int32 [H, W]
  preserve_ids:int32 [H, W]
  parent_state_sha256
  matcher_source_sha256
```

`dataset.py` 的 crop、flip、resize 必须对 image、GT 和三个 ID map 使用完全相同的 stateless 变换；ID map 只能用 nearest interpolation。内部 validation 不参与 atlas 生成参数调节，official test 不生成 atlas。

---

## 8. PBDR‑V4 角色损失代码

建议新增：

```text
experiments/pbdr_v4_component_loss.py
```

```python
"""Role-aligned component loss for PBDR-V4.

The component-id maps are generated from the frozen parent prediction and the
training target with the same 8-connectivity and centroid-distance definition
used by evaluation.  IDs are positive integers; zero means outside the class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


Role = Literal["best_miou", "best_pd"]


@dataclass(frozen=True, slots=True)
class PBDRV4LossOutput:
    total: torch.Tensor
    bce: torch.Tensor
    tversky: torch.Tensor
    rescue_components: torch.Tensor
    suppress_components: torch.Tensor
    preserve_components: torch.Tensor
    foreground_drop: torch.Tensor
    background_increase: torch.Tensor
    neutral_delta: torch.Tensor


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    denominator = weight.sum()
    if int(denominator.detach().item()) == 0:
        return value.new_zeros(())
    return (value * weight).sum() / denominator


def _soft_component_peak(
    logits: torch.Tensor,
    component_ids: torch.Tensor,
    *,
    temperature: float = 0.25,
) -> list[torch.Tensor]:
    """Return one smooth maximum logit for every positive component ID."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if component_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("component_ids must use an integer dtype")
    if logits.shape != component_ids.shape:
        raise ValueError("logits and component_ids must share shape")

    peaks: list[torch.Tensor] = []
    for batch_index in range(logits.shape[0]):
        ids = torch.unique(component_ids[batch_index])
        ids = ids[ids > 0]
        for component_id in ids:
            values = logits[batch_index][component_ids[batch_index] == component_id]
            # Subtract log(n) so component size does not itself increase score.
            peak = temperature * (
                torch.logsumexp(values / temperature, dim=0)
                - torch.log(values.new_tensor(float(values.numel())))
            )
            peaks.append(peak)
    return peaks


def _positive_component_loss(
    logits: torch.Tensor,
    component_ids: torch.Tensor,
) -> torch.Tensor:
    peaks = _soft_component_peak(logits, component_ids)
    if not peaks:
        return logits.new_zeros(())
    return torch.stack([F.softplus(-peak) for peak in peaks]).mean()


def _negative_component_loss(
    logits: torch.Tensor,
    component_ids: torch.Tensor,
) -> torch.Tensor:
    peaks = _soft_component_peak(logits, component_ids)
    if not peaks:
        return logits.new_zeros(())
    return torch.stack([F.softplus(peak) for peak in peaks]).mean()


def _tversky_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    reduce_dims = tuple(range(1, probability.ndim))
    true_positive = (probability * target).sum(dim=reduce_dims)
    false_positive = (probability * (1.0 - target)).sum(dim=reduce_dims)
    false_negative = ((1.0 - probability) * target).sum(dim=reduce_dims)
    score = (true_positive + epsilon) / (
        true_positive + alpha * false_positive + beta * false_negative + epsilon
    )
    return 1.0 - score.mean()


def compute_pbdr_v4_loss(
    *,
    role: Role,
    routed_logits: torch.Tensor,
    base_logits: torch.Tensor,
    delta_logits: torch.Tensor,
    target: torch.Tensor,
    rescue_component_ids: torch.Tensor,
    suppress_component_ids: torch.Tensor,
    preserve_component_ids: torch.Tensor,
) -> PBDRV4LossOutput:
    """Compute role-specific pixel, overlap and component objectives."""

    if role not in ("best_miou", "best_pd"):
        raise ValueError(f"unsupported role: {role!r}")
    tensors = (
        base_logits,
        delta_logits,
        target,
        rescue_component_ids,
        suppress_component_ids,
        preserve_component_ids,
    )
    if any(value.shape != routed_logits.shape for value in tensors):
        raise ValueError("all loss tensors must share BCHW shape")

    routed = routed_logits.float()
    base = base_logits.detach().float()
    target_float = target.float()
    probability = torch.sigmoid(routed)
    base_probability = torch.sigmoid(base)

    bce = F.binary_cross_entropy_with_logits(routed, target_float)
    if role == "best_miou":
        tversky = _tversky_loss(
            probability, target_float, alpha=0.50, beta=0.50
        )
        weights = {
            "bce": 1.00,
            "tversky": 1.00,
            "rescue": 2.00,
            "suppress": 1.50,
            "preserve": 1.00,
            "foreground_drop": 1.00,
            "background_increase": 1.00,
            "neutral_delta": 0.01,
        }
    else:
        # Higher beta penalizes false-negative overlap more strongly.
        tversky = _tversky_loss(
            probability, target_float, alpha=0.30, beta=0.70
        )
        weights = {
            "bce": 0.50,
            "tversky": 0.75,
            "rescue": 5.00,
            "suppress": 0.50,
            "preserve": 2.00,
            "foreground_drop": 2.00,
            "background_increase": 0.25,
            "neutral_delta": 0.005,
        }

    rescue_components = _positive_component_loss(
        routed, rescue_component_ids
    )
    suppress_components = _negative_component_loss(
        routed, suppress_component_ids
    )
    preserve_components = _positive_component_loss(
        routed, preserve_component_ids
    )

    foreground = target_float >= 0.5
    background = ~foreground
    foreground_drop = _masked_mean(
        F.relu(base_probability - probability).square(), foreground
    )
    background_increase = _masked_mean(
        F.relu(probability - base_probability).square(), background
    )

    edited = (
        (rescue_component_ids > 0)
        | (suppress_component_ids > 0)
        | (preserve_component_ids > 0)
    )
    neutral_delta = _masked_mean(delta_logits.float().abs(), ~edited)

    total = (
        weights["bce"] * bce
        + weights["tversky"] * tversky
        + weights["rescue"] * rescue_components
        + weights["suppress"] * suppress_components
        + weights["preserve"] * preserve_components
        + weights["foreground_drop"] * foreground_drop
        + weights["background_increase"] * background_increase
        + weights["neutral_delta"] * neutral_delta
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("PBDR-V4 loss is non-finite")

    return PBDRV4LossOutput(
        total=total,
        bce=bce,
        tversky=tversky,
        rescue_components=rescue_components,
        suppress_components=suppress_components,
        preserve_components=preserve_components,
        foreground_drop=foreground_drop,
        background_increase=background_increase,
        neutral_delta=neutral_delta,
    )
```

### 8.1 损失解释

- `rescue_components`：对每个漏检 GT component 做 smooth max pooling，要求至少一个局部峰值越过 0 logit；每个目标等权。
- `suppress_components`：对每个未匹配预测 component 的 smooth maximum 做负类损失，只有整个组件都压下去才真正降低 Fa。
- `preserve_components`：保持父模型已经匹配的目标组件，不允许区域优化把它们删掉。
- `foreground_drop/background_increase`：以 Current 为单向参考，防止在 GT 前景降概率或在背景无约束升概率。
- `neutral_delta`：只在不属于已知错误/保护组件的区域约束 residual 稀疏。
- `best_pd` 对 rescue 和 FN 使用更高权重，`best_miou` 对 overlap 和 suppression 更平衡。

这些权重是训练配方，不是性能接受门槛；最终仍按零正增益角色序选择。

---

## 9. 模型集成修改

### 9.1 forward 修改

当前 V3 只传 `out、d0、q4、u1`：

```python
routing = self.pbdr_v3.forward_with_diagnostics(
    z_out=out,
    z_d0=d0,
    q4=q4,
    local_feature=u1,
)
```

V4 改为：

```python
routing = self.pbdr_v4.forward_with_diagnostics(
    z_out=out,
    z_d0=d0,
    auxiliary_logits=(gt2, gt3, gt4, gt5),
    q4=q4,
    local_feature=u1,
)
routed_out = routing.routed_logits
```

训练返回保持：

```python
probabilities = (
    sigmoid(gt5), sigmoid(gt4), sigmoid(gt3),
    sigmoid(gt2), sigmoid(d0), sigmoid(routed_out),
)
```

另外在 training aux 中增加：

```python
base_logits=out
auxiliary_logits=(gt2, gt3, gt4, gt5)
routing=routing
```

### 9.2 构造函数修改

```python
self.pbdr_v4 = RoleAlignedResidualCalibratorV4(role=parent_role)
```

`parent_role` 必须写入 checkpoint manifest，防止把 `best_pd` 的非对称 head 加载到 `best_miou`。

### 9.3 Stage‑1 冻结

```python
for parameter in model.parameters():
    parameter.requires_grad_(False)
for parameter in model.pbdr_v4.parameters():
    parameter.requires_grad_(True)

model.eval()
model.pbdr_v4.train()
```

严格检查：

```text
trainable names == all names prefixed by pbdr_v4.
base state SHA == parent checkpoint state SHA
all base BN buffers unchanged
initial routed logits == Current logits bitwise
```

### 9.4 Stage‑2 并行候选，不由门槛触发

为避免“通过某个门才允许解冻”，直接预先定义两个并行分支：

```text
A: Stage‑1，只有 pbdr_v4 trainable
B: Stage‑2，从 A 的 selected internal checkpoint 开始，
   额外解冻 outc 与 up_decoder1
```

推荐优化器：

```python
optimizer = torch.optim.AdamW(
    [
        {"params": model.pbdr_v4.parameters(), "lr": 1e-4},
        {"params": model.outc.parameters(), "lr": 2e-6},
        {"params": model.up_decoder1.parameters(), "lr": 1e-6},
    ],
    weight_decay=1e-4,
)
```

Stage‑2 对解冻参数增加 L2‑SP：约束其接近 parent checkpoint，而不是普通零中心 weight decay。A、B 都进入候选池；不存在“达到某个提升幅度才运行 B”的门槛。

---

## 10. 零正增益的 Original+Current 包络选择器

建议新增：

```text
experiments/pbdr_v4_zero_margin_selector.py
```

```python
"""Baseline-envelope selector with zero positive performance margin."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Literal, Sequence


Role = Literal["best_miou", "best_pd"]


@dataclass(frozen=True, slots=True)
class MetricRecord:
    name: str
    matched_target_count: int
    target_count: int
    unmatched_component_pixels: int
    valid_pixels: int
    miou: float
    niou: float
    matched_tiny_target_count: int
    tiny_target_count: int
    loss: float

    def __post_init__(self) -> None:
        integer_fields = (
            "matched_target_count",
            "target_count",
            "unmatched_component_pixels",
            "valid_pixels",
            "matched_tiny_target_count",
            "tiny_target_count",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field} must be an integer")
            if value < 0:
                raise ValueError(f"{field} must be non-negative")
        if self.matched_target_count > self.target_count:
            raise ValueError("matched_target_count exceeds target_count")
        if self.matched_tiny_target_count > self.tiny_target_count:
            raise ValueError("matched_tiny_target_count exceeds tiny_target_count")
        if self.valid_pixels <= 0:
            raise ValueError("valid_pixels must be positive")
        for field in ("miou", "niou", "loss"):
            value = float(getattr(self, field))
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")

    @property
    def pd(self) -> Fraction:
        return Fraction(self.matched_target_count, self.target_count or 1)

    @property
    def tiny_pd(self) -> Fraction:
        return Fraction(
            self.matched_tiny_target_count,
            self.tiny_target_count or 1,
        )

    @property
    def fa(self) -> Fraction:
        return Fraction(self.unmatched_component_pixels, self.valid_pixels)


def role_key(role: Role, record: MetricRecord) -> tuple[object, ...]:
    if role == "best_miou":
        return (
            record.miou,
            record.pd,
            -record.fa,
            record.niou,
            record.tiny_pd,
            -record.loss,
        )
    if role == "best_pd":
        return (
            record.pd,
            -record.fa,
            record.tiny_pd,
            record.miou,
            record.niou,
            -record.loss,
        )
    raise ValueError(f"unsupported role: {role!r}")


def select_best(role: Role, candidates: Sequence[MetricRecord]) -> MetricRecord:
    """Return the strict role-key maximum; exact ties keep the first record."""

    if not candidates:
        raise ValueError("candidate pool is empty")
    target_counts = {record.target_count for record in candidates}
    tiny_counts = {record.tiny_target_count for record in candidates}
    valid_pixels = {record.valid_pixels for record in candidates}
    if len(target_counts) != 1 or len(tiny_counts) != 1 or len(valid_pixels) != 1:
        raise ValueError("all records must bind the same evaluation split")

    winner = candidates[0]
    winner_key = role_key(role, winner)
    for record in candidates[1:]:
        key = role_key(role, record)
        if key > winner_key:  # no epsilon and no minimum effect size
            winner = record
            winner_key = key
    return winner


def select_against_baseline_envelope(
    role: Role,
    *,
    original: MetricRecord,
    current: MetricRecord,
    candidates: Sequence[MetricRecord],
) -> MetricRecord:
    # Baselines are listed first, so an exact key tie cannot replace them.
    return select_best(role, (original, current, *candidates))
```

### 10.1 候选池

每个数据集、每个角色的池应为：

```text
1. Original 同角色 checkpoint
2. Current TSS-off 同角色 checkpoint
3. PBDR-V3 residual calibration 的内部最优组合
4. PBDR-V4 Stage-1 内部最优 epoch
5. PBDR-V4 Stage-2 内部最优 epoch
```

严格 role key 最大值就是部署候选。Original 与 Current 排在池前面，因此完全相同的 key 不会被新模型替换。

### 10.2 这里没有性能门槛

选择器没有：

```text
min_delta
relative_percent
non_regression_checks
passed_gate
material_gain
```

只有严格字典序比较。Pd、tiny‑Pd 与 Fa 优先使用整数分子/分母，避免浮点末位误差把相同组件计数伪装成差异。

---

## 11. 训练协议

### 11.1 数据划分

为避免继续扩大 test-selected 偏差：

- official train index 固定拆为 80% train / 20% internal validation；
- atlas 只从 train 部分的父模型预测生成；
- internal validation 只用于 epoch、残差再标定和候选模型选择；
- official test 每个数据集只做一次统一 loader pass，同时评估全部冻结候选；
- 不根据 official test 结果重启训练或修改网格。

### 11.2 两角色分别训练

每个数据集运行：

```text
best_miou / Stage-1
best_miou / Stage-2
best_pd   / Stage-1
best_pd   / Stage-2
```

它们共享代码和数据划分，但 role、parent checkpoint、residual limits、loss weights 和 atlas 均写入 manifest。

### 11.3 建议 epoch 与学习率

- Stage‑1：150 epochs，AdamW `lr=1e-4`、`wd=1e-4`，每 5 epochs 做一次 internal validation；
- Stage‑2：50 epochs，router `1e-4`、outc `2e-6`、up_decoder1 `1e-6`；
- FP32，CUDA matmul TF32 与 cuDNN TF32 都关闭；
- checkpoint 选择只用固定 0.5 的完整 role key；
- exact key tie 取更早 epoch。

这些是训练预算和可复现实验设置，不是性能接受门槛。

### 11.4 GPU0 的优先顺序

1. 缓存三数据集六角色的 Current/V3 raw logits 与 delta；
2. 完成 residual calibration 网格并输出三方包络表；
3. 生成训练 atlas；
4. 做 V4 单 batch identity/backward、1 epoch smoke；
5. 先跑 NUDT `best_pd`，因为它离超过 Original 只差 1 个目标且有最大 Fa/mIoU 余量；
6. 再跑 NUAA `best_miou` 与 IRSTD `best_pd`；
7. 最后补其余角色，保持完整六角色报告。

---

## 12. 预期的可验证机制信号

以下不是性能门槛，而是用于判断代码是否按设计工作的诊断：

### 12.1 `best_pd`

- positive delta 应主要落在 `rescue_component_ids` 和 preserve target 邻域；
- 每个漏检 component 的 smooth peak logit 应上升；
- negative delta 在 GT component 内应显著少于背景；
- NUDT 重点检查 Original 多出的那个目标是否被恢复；
- tiny target 按 component 等权，不应被大目标像素梯度淹没。

### 12.2 `best_miou`

- suppress component 的最大 logit 应下降；
- matched target component 的存在性保持；
- `gt2–gt5` 一致区域与最终正 residual 相关；
- 仅单个粗尺度响应的区域不应普遍得到正 residual；
- 轮廓外溢和 background FP 应与 mIoU 改善同向。

### 12.3 必须保存的诊断量

```text
positive/negative delta 分位数
每类 atlas 区域的 delta 均值与 RMS
rescue/suppress/preserve component 数量
被恢复、被删除、被新建的组件 ID
阈值穿越像素数
每个 component 的 peak logit before/after
Original、Current、V3、V4 的完整角色 key
```

---

## 13. 关于“确保性能一定提升”的准确边界

### 13.1 可以严格保证的内容

在冻结的 internal validation split 上，只要把 Original 与 Current 都放入候选池，零门槛 selector 有一个简单的数学性质：

```text
selected_role_key >= max(original_role_key, current_role_key)
```

因为基线本身就是候选。这保证最终选中产物不会低于已有基线包络；如果 V3 再标定、V4 Stage‑1 或 Stage‑2 中任意一个 role key 严格更大，选择结果就是严格提升，不需要任何额外幅度门槛。

### 13.2 不能诚实预先保证的内容

任何尚未训练的新结构，都不能在看到未知 official test 或未来样本之前数学保证严格提升。若报告在实验前承诺“必然提高 Pd/Fa/mIoU/nIoU”，该承诺没有证据基础。可以做到的是：

- 让结构与损失直接对齐失败模式；
- 用 Original+Current 包络防止错误比较；
- 用 baseline-inclusive zero-margin selection 保证固定选择集不下降；
- official test 只做一次独立验证，并如实报告是否保持提升。

如果目标仅是“当前部署性能只要有任何严格提升即可”，则可按数据集、按角色选择不同 checkpoint，而不强迫一个统一模型覆盖六个角色。这与仓库现有的 `best_miou`/`best_pd` 双角色部署方式一致。

---

## 14. 推荐新增/修改文件清单

### 新增

```text
model/tpd_role_aligned_residual_calibrator_v4.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v4.py
experiments/pbdr_v4_component_loss.py
experiments/pbdr_v4_zero_margin_selector.py
analysis/build_pbdr_v4_component_atlas.py
analysis/sweep_pbdr_v3_residual_calibration.py
experiments/PBDR_V4_ZERO_MARGIN_PROTOCOL.md
tests/test_pbdr_v4_model.py
tests/test_pbdr_v4_component_loss.py
tests/test_pbdr_v4_zero_margin_selector.py
tests/test_component_matcher_duplicate_area.py
```

### 修改

```text
experiments/three_dataset_pbdr_v3_models_seed42_v1.py
  → 复制为 V4 builder，不覆盖历史 V3

experiments/train_two_dataset_pbdr_v3_stage1_v1.py
  → 复制为 V4 trainer；删除 non_regression_gate 依赖；
     official comparator 改为 Original+Current envelope

dataset.py
  → 支持三个 component-ID atlas map 与图像/GT 同步变换
```

### 不应修改

```text
PBDR-V2/V3 历史 checkpoint
既有 summary/evaluation/decision 文件
历史协议文档
历史 official-test 结果
```

---

## 15. 单元测试最小集合

1. **Identity**：V4 初始化时 routed logits 与 Current 逐位相同。
2. **First gradient**：residual head 在首个 backward 后存在有限非零梯度。
3. **Role capacity**：`best_pd.positive_limit > best_pd.negative_limit`；两个角色 manifest 不可互换。
4. **Auxiliary order**：gt2–gt5 数量、shape、dtype、device 任一不符立即报错。
5. **Atlas transform**：crop/flip 后 component IDs 与 GT 对齐，nearest interpolation 不产生新 ID。
6. **Duplicate-area matcher**：一个 matched component 与一个同面积 unmatched component 时，后者仍计入 Fa。
7. **One-to-one matcher**：两个预测组件位于一个 GT 质心 3 像素内时只能匹配一个。
8. **Zero-margin selector**：`+1e-12` 的首指标严格改善可获胜；完全相等保留前置基线。
9. **Baseline envelope**：Candidate 胜 Original 但负于 Current 时，必须选 Current。
10. **State audit**：Stage‑1 只有 `pbdr_v4.*` 可训练，所有 base BN buffer hash 不变。
11. **Export**：训练图到推理图只删除训练期 TSS heads，不丢失 V4 参数。
12. **Resume**：RNG、optimizer、model、atlas SHA、role 和 parent checkpoint 全部精确恢复。

---

## 16. 最终研究判断

PBDR‑V2 的失败原因已经足够明确：它把相关、粗糙的 `d0/q4` 当成可直接修改全分辨率 logit 的可靠证据，并在 scratch 联合训练中改变父模型轨迹；BCE 又没有约束 component‑Fa/Pd 的离散拓扑，因此出现 Fa 暴涨、mIoU 下降并不意外。

PBDR‑V3 的结构修正是有效的，但它仍停留在“通用保守概率校准”层面。其正式结果说明：

- 校准器确实能显著降低部分 Fa、改善部分 mIoU/nIoU；
- ±0.15 的窄残差无法稳定恢复对象级漏检；
- 同一个 BCE+soft‑IoU 配方不能同时承担 `best_miou` 与 `best_pd`；
- 只与 Original 比较会掩盖 Current 更强的角色点。

因此下一步不应继续调正增益门槛，也不应只增加训练 epoch。最合理的方案是：

```text
PBDR-V3 residual 再标定
+ 角色专用非对称 residual
+ 四路深监督一致性上下文
+ 与正式 Pd/Fa 定义一致的组件级损失
+ Original/Current/Candidate 三方零门槛选择
```

这条路线直接针对已有失败证据，且保留 V3 已验证正确的 warm‑start、冻结主干、精确 identity 和 q4 安全归一化。它是目前最有可能产生严格性能提升、同时避免再次把弱对手胜利误写成模型提升的方案。

---

## 17. 代码证据索引

- [最新公开提交 `106e463`](https://github.com/Arialliy/SCTransNet_main/commit/106e4631b67bea6560e318e9ce593f59dabe3193)
- [PBDR‑V2 router](https://github.com/Arialliy/SCTransNet_main/blob/106e4631b67bea6560e318e9ce593f59dabe3193/model/tpd_persistent_evidence_residual_router_v2.py)
- [PBDR‑V3 calibrator](https://github.com/Arialliy/SCTransNet_main/blob/106e4631b67bea6560e318e9ce593f59dabe3193/model/tpd_conservative_residual_calibrator_v3.py)
- [PBDR‑V3 model integration](https://github.com/Arialliy/SCTransNet_main/blob/106e4631b67bea6560e318e9ce593f59dabe3193/model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v3.py)
- [PBDR‑V3 loss](https://github.com/Arialliy/SCTransNet_main/blob/106e4631b67bea6560e318e9ce593f59dabe3193/experiments/pbdr_v3_loss.py)
- [PBDR‑V3 cross-dataset zero-margin protocol](https://github.com/Arialliy/SCTransNet_main/blob/106e4631b67bea6560e318e9ce593f59dabe3193/experiments/PBDR_V3_CROSS_DATASET_PROTOCOL.md)
- [PBDR‑V3 zero-margin role comparator](https://github.com/Arialliy/SCTransNet_main/blob/106e4631b67bea6560e318e9ce593f59dabe3193/experiments/pbdr_v3_zero_margin_role_gate.py)
- [Stage‑1 warm-start/freeze builder](https://github.com/Arialliy/SCTransNet_main/blob/106e4631b67bea6560e318e9ce593f59dabe3193/experiments/three_dataset_pbdr_v3_models_seed42_v1.py)
- [Root metric implementation](https://github.com/Arialliy/SCTransNet_main/blob/106e4631b67bea6560e318e9ce593f59dabe3193/metrics.py)
- 本地正式结果依据：`SCTransNet_历史模型实验结果总汇(2).md`
