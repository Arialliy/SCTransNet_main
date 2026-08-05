# SCTransNet：TSS 最终裁决与 NER→QFG→TPD 单组件优化方案

> 项目：单帧红外小目标检测  
> 基线：SCTransNet  
> 当前推理主线：`SCTransNet + TPD8-MPRS-DCH + 五节点 NER4 Tail-Aware + QFG2-CROA`  
> 当前训练期辅助：TSS 已完成正权重、TSS-off 与 EC-TSS V3.1 诊断  
> 正式数据集：NUAA-SIRST、NUDT-SIRST、IRSTD-1K  
> 当前开发协议：seed 42、1000 epochs、每 10 epochs 评估、`best_miou / best_pd`、阈值 0.5  
> 协议边界：现有 `img_idx/test` 同时用于选模和评估，因此本轮只是 **seed42 test-selected 开发筛选**，不是无偏论文测试  
> 本方案目的：裁决 TSS on/off，并把下一轮优化转移到推理主线中的单个模块  
> 推荐首个待验证候选：**NER V5-PER（Persistent-Evidence Positive Routing）**

---

# 0. 执行摘要

## 0.1 TSS 应该 on 还是 off

### 统一论文配方

当前不应设置：

```text
tss_default=on
```

也不能科学地宣称：

```text
tss_off 已经成为全面最优最终配方
```

正式证据表明：

- 三个旧正 TSS 请求权重 `0.0025 / 0.005 / 0.01` 均未建立跨三数据集统一配方；
- TSS-off 同样存在严重退化项，未通过全局门；
- EC-TSS V3.1 虽在 6 个 dataset-role 单元中产生 4 个独有非支配点，但未通过严重退化门、旧强项保留门和成对多数门；
- 最新正式裁决已经是：

```text
EC_TSS_V3_1_PERFORMANCE_FAIL_STOP_TSS_OPTIMIZATION
```

因此，当前最准确状态是：

```text
global_tss_default=null
tss_training_innovation_supported=false
tss_optimization_closed=true
```

### 下一轮模型优化的训练锚点

下一轮应使用：

```text
next_optimization_training_anchor=tss_off
```

这里的 TSS-off 是**诊断锚点**，不是“已经证明全面最优”的最终配方。

选择它的原因是：

1. 移除训练期辅助任务对主分割路径的梯度混杂；
2. 直接观察 TPD、NER、QFG 推理结构自身的作用；
3. 避免继续扩大已经达到 `5:1` 的 Final-family / Original 配方搜索预算；
4. 保持最终推理模型不变，因为 TSS 本来就不进入推理图。

## 0.2 TSS 能否继续开启

代码上仍可开启，但现有 `λ=0.005` 只能降级为：

```text
descriptive_domain_candidate_requiring_independent_confirmation
```

它不是已确认的域专用配方，不是默认训练模式，也不得作为下一轮结构优化的起点。原因是这一观察同样来自已被反复使用的 test-selected 开发协议。

当前观察到：

| 场景 | 当前更有利的配方 |
|---|---|
| NUAA | TSS-off 提供较强的跨角色综合折中，但不是两个角色的最低 Fa；不存在统一胜者 |
| NUDT | 旧 TSS `λ=0.005` 在部分 Pd 工作点有竞争力，区域质量并非统一领先 |
| IRSTD | 旧 TSS `λ=0.005` 在 `best_pd` 的 Pd/mIoU/nIoU 组合有竞争力，但 `best_miou` 仍存在 Fa 严重退化 |
| 三数据集统一配方 | 无正式胜者 |

所以 `λ=0.005` 目前只能用于：

- 历史结果的描述性对照；
- NUDT / IRSTD 的后续独立确认候选；
- 附录中说明 TSS 可能存在数据域依赖效应。

未经独立确认前，不得将它写成“已知部署域下可直接使用的专用配方”。若论文目标是“一套统一训练配方覆盖三数据集”，则 TSS 不应继续 on。

## 0.3 是否应停止 TSS 优化

**是。**

旧 TSS、动态正权重、TSS-off、EC-TSS V3.1 已经覆盖：

```text
全局均匀 presence supervision
正权重强度变化
无辅助目标
误差条件化正负风险
```

继续设计 TSS V4/V5 会产生三个问题：

1. 研究叙事从“模型结构创新”漂移为持续辅助 loss 搜索；
2. 同一 test-selected 协议上的后验搜索预算继续扩大；
3. 无法判断后续改进来自 TSS、NER、QFG 还是 TPD。

因此：

```text
additional_tss_objective_design_authorized=false
additional_positive_tss_lambda_search_authorized=false
```

TSS 保留为历史消融和可选训练辅助，不再作为论文核心创新点。

## 0.4 下一步优化顺序

正式顺序应固定为：

```text
NER
→ QFG
→ TPD
```

第一候选为：

> **NER V5-PER：Persistent-Evidence Positive Routing**  
> 持久证据约束的正向增强路由

它只修改 NER stage 2 的空间门控公式，不增加参数、buffer 或新分支，不修改 TPD、QFG、decoder 和训练 loss。

---

# 1. 证据复盘

## 1.1 TSS 已经完成足够充分的诊断

历史正式结果已经覆盖：

| TSS 方案 | 结果 |
|---|---|
| 旧固定正权重 | 数据集相关 mixed trade-off |
| 动态请求权重 0.0025 | 全局门失败 |
| 动态请求权重 0.005 | 违规最少，但仍失败 |
| 动态请求权重 0.01 | 全局门失败 |
| TSS-off | 与 0.005 各有 13 项更好，未形成统一优势 |
| EC-TSS V3.1 | 4/6 独有非支配点，但关键 Gate 失败 |

EC-TSS V3.1 最终成对票数为：

| 参考配方 | EC 更好 | 相同 | EC 更差 |
|---|---:|---:|---:|
| Original | 13 | 2 | 15 |
| TSS-off | 10 | 2 | 18 |
| 旧 TSS `λ=0.005` | 13 | 2 | 15 |

这说明：

```text
TSS 不是完全无效
但不能形成统一训练创新
```

## 1.2 当前推理主线仍值得保留

正式历史证据支持：

- 初代 TPD 在若干 Pd–Fa–mIoU 工作点上明显改善 Original；
- NER V4 是 NER V1–V4 中首次获得明确相对改善的版本；
- 完整模型在多个低 Fa 或错误目标工作点具有竞争力；
- 现有失败是跨数据集统一配方和严重退化门失败，不是工程故障。

因此，应写：

```text
architecture_implemented=true
architecture_frozen_for_component_diagnosis=true
architecture_failure_supported=false
architecture_global_advantage_not_established=true
```

不应写：

```text
architecture_success=true
```

因为跨数据集统一优势尚未建立。

## 1.3 为什么不应为了创新性强行开启 TSS

创新性必须同时满足：

```text
机制新颖
+
受控对照成立
+
性能证据支持
```

TSS 当前只满足前两项的一部分，性能门未通过。

将失败的辅助目标强行保留在最终训练配方中，会削弱论文逻辑：

```text
论文声称 TSS 是核心创新
但统一 Gate 明确失败
```

更合理的创新主线是：

```text
TPD8：
目标保真、相位分辨浅层 tokenization

NER：
五节点浅层证据向 decoder 的尾部感知中继

QFG：
只调制 Query 的频率条件化
```

TSS 可作为：

```text
探索性辅助监督
负结果
附录消融
```

---

# 2. 为什么下一步先优化 NER

## 2.1 NER 是最有正式正证据的中间模块

历史结果中：

- NER V1–V3 均返回模型优化；
- NER V4 Tail-Aware 首次确认相对改善；
- NER V4 成为后续完整模型底座；
- NER 直接控制 decoder stage 4/3/2 的 skip modulation。

因此，NER 是当前最值得继续投入的模块。

## 2.2 QFG 当前证据较弱

QFG-only 在因子实验中被覆盖；TSS+QFG 相对紧邻 TSS-on 的固定点增量很小。

当前 QFG 设计已经较严格：

```text
固定 Haar prior
每样本 RMS 归一化
中心化有界 gate
只修改 Query
频率源 detach
零 terminal projection
优化锚定 identity
```

在没有做层级利用率和 knockout 之前，继续调 alpha 或增加频率分支缺乏依据。

所以 QFG 排在 NER 后面。

## 2.3 TPD 应最后调整

TPD8 决定：

```text
emb1 / emb2 endpoint
五个 NER evidence nodes
QFG 的 encoder frequency source
```

修改 TPD 会同时改变 NER 和 QFG 的输入分布，归因范围最大。

因此只有 NER、QFG 均不能解决跨数据集冲突时，才回到 TPD。

---

# 3. 当前 NER V4 的待验证代码级假设

## 3.1 当前数据流

NER 使用五个 evidence nodes：

```text
h11, h12, h13
h21, h22
```

并按：

```text
q4 → q3 → q2
```

形成 relay。

Decoder 三阶段输入为：

```text
stage4: h13, h22, up4
stage3: h12, h21, q4, up3
stage2: h11, q3, up2
```

NER mask 最终以乘性形式调制 decoder skip。

## 3.2 当前 V4 Tail-Aware 公式

对 stage 3/2，代码先计算局部和父级 tail：

\[
T_s
=
\tanh(\operatorname{ReLU}(z_s-\kappa_s))
\]

再计算跨阶段持久支持：

\[
P_s
=
\sqrt{
T_s\cdot\uparrow T_{s+1}
}
\]

正式 complement-tail 支持为：

\[
B_s=1-P_s
\]

当前 shifted logits：

\[
Z_s^{V4}
=
C_s+d_sB_s
\]

其中：

- \(C_s\)：空间中心化后的 local gate logits；
- \(d_s\)：每阶段 learned DC offset；
- \(B_s\)：背景样区域支持。

最后：

\[
M_s
=
\frac{1}{\pi}
\arctan(\pi Z_s^{V4})
\]

## 3.3 待验证假设：tail support 只约束 DC，不约束 local spatial gate

当前公式中：

```text
C_s 在全部空间位置无条件生效
B_s 只乘在 d_s 上
```

也就是说，即使一个位置没有跨阶段持久目标证据，stage-2 的高分辨率 local gate 仍可以产生正向 skip enhancement。

代码事实只能说明存在这条未受持久证据约束的正向路径，不能直接证明它已经造成性能退化。历史诊断仅提供了间接线索：

```text
historical stage-wise effects are mixed
no stage-level causal conclusion is established
```

stage 2 空间分辨率最高、背景位置最多，因此我们提出它**可能**把局部纹理或杂波转化为：

```text
component-Fa
pixel FP
attached halo
```

因此，stage2 是下一轮最小变量的**候选切入点**，而不是已确认瓶颈。第 6 节零训练审计必须先验证该假设；未通过触发门时不得启动 V5 训练。

---

# 4. 新候选：NER V5-PER

## 4.1 名称

> **NER V5-PER**  
> Persistent-Evidence Positive Routing  
> 持久证据约束的正向增强路由

## 4.2 设计边界

保持不变：

```text
五个 evidence nodes
q4 → q3 → q2 relay
relay width
RMS-balanced fusion
centered gate
arctangent bounds
stagewise DC offsets
tail statistic
tail thresholds
stage4 公式
stage3 公式
TPD8
QFG2
decoder
TSS objective = off
```

只修改：

```text
stage2 local gate 的正向部分
```

## 4.3 核心公式

设 stage2 当前 centered logits 为：

\[
C_2
\]

当前背景支持：

\[
B_2=1-P_2
\]

NER V5-PER 使用：

\[
\boxed{
\widetilde C_2
=
C_2-B_2\operatorname{ReLU}(C_2)
}
\]

因此：

- 当 \(C_2>0\)：

\[
\widetilde C_2=P_2C_2
\]

正向 enhancement 按跨阶段持久上尾支持缩放。

- 当 \(C_2\le0\)：

\[
\widetilde C_2=C_2
\]

负向 suppression 在背景和目标区域均完整保留。

最终：

\[
Z_2^{V5}
=
C_2
-
B_2\operatorname{ReLU}(C_2)
+
d_2B_2
\]

\[
M_2^{V5}
=
\frac{1}{\pi}
\arctan(\pi Z_2^{V5})
\]

stage4 和 stage3 继续使用 V4：

\[
Z_s^{V5}=Z_s^{V4},
\quad s\in\{4,3\}
\]

## 4.4 直观解释

这里的 \(P_2\) 由 q2/q3 relay 的通道 RMS 上尾构造，不使用 GT，也没有概率校准。因此全文只称其为“跨阶段持久上尾支持”，不能称为目标概率；下表是待验证的设计假设，不是已经成立的数据集结论。

| 区域 | \(P_2\) | 正向 local enhancement | 负向 suppression | DC calibration |
|---|---:|---:|---:|---:|
| 跨阶段持久目标 | 高 | 保留 | 保留 | 减弱 |
| 背景样区域 | 低 | 抑制 | 保留 | 保留 |
| 中间置信区域 | 中 | 平滑缩放 | 保留 | 平滑缩放 |

这同时针对：

```text
Pd：
持久目标仍允许正向增强

Fa：
没有持久证据的背景位置不能被 stage2 正向抬高

pixel FP / halo：
负向 gate 保留，可继续抑制局部外溢
```

## 4.5 为什么不用简单的 `P2 * C2`

简单公式：

\[
P_2C_2
\]

会同时削弱：

```text
正向 enhancement
负向 suppression
```

这可能降低 Fa 抑制能力。

V5-PER 只路由正向部分，同时保留全部负向 suppression。

## 4.6 零点优化性质

实现形式：

```python
routed = centered - background_support * F.relu(centered)
```

在 `centered=0` 时，PyTorch 的 `ReLU` 零点导数为 0，因此 identity 路径 `centered` 保持一阶梯度。

当 gate 与 DC 均为零初始化时：

```text
V5 mask = 0
V4 mask = 0
relay-off skip 完全一致
```

因此 V5 不破坏已有的零点 forward anchor。

## 4.7 参数和 state

```text
新增参数：0
新增 persistent buffer：0
state keys：与 NER V4 相同
relay parameter count：不变
QFG state：不变
推理参数量：不变
```

---

# 5. 参考代码修改

## 5.1 新增模型文件

```text
model/tpd_ner_v8_mprs_dch_v5_per.py
```

参考核心类：

```python
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

from model.tpd_ner_v8_mprs_dch import RELAY_STAGE_ORDER
from model.tpd_ner_v8_mprs_dch_v2 import (
    arctangent_residual_mask,
    spatially_center_gate_logits,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    TailAwarePersistentDCOffsetEvidenceRelay,
)

SpatialSize = Tuple[int, int]
V5_PER_RELAY_VERSION = "v5_stage2_persistent_evidence_positive_routing"


class PersistentEvidencePositiveRoutingRelay(
    TailAwarePersistentDCOffsetEvidenceRelay
):
    """V4 exact at stages 4/3; route only positive stage-2 enhancement."""

    def forward_stage(
        self,
        stage: int,
        sources: Sequence[torch.Tensor],
        output_size: SpatialSize,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if stage not in RELAY_STAGE_ORDER:
            raise ValueError(f"relay stage must be 4, 3, or 2, got {stage}")

        if stage != 2:
            return super().forward_stage(stage, sources, output_size)

        relay_value = self.fusions["2"](sources, output_size)
        logits = self.gates["2"](relay_value)
        centered = spatially_center_gate_logits(logits)

        # Formal V4 complement support B2 = 1 - P2.
        background_support = self.dc_support(
            2,
            relay_value,
            sources,
            output_size,
        ).detach()

        routed_centered = (
            centered
            - background_support * F.relu(centered)
        )

        shifted = routed_centered + (
            self.dc_offsets["2"].view(1, 1, 1, 1)
            * background_support
        )

        mask = arctangent_residual_mask(shifted)
        return relay_value, mask
```

## 5.2 限制 formal mode

NER V5-PER 正式模型只允许：

```text
dc_support_mode = complement_tail
```

不应同时开放：

```text
legacy_global
direct_tail
```

避免一个新版本包含三个新的结构候选。

该约束不能只写在文档中，必须同时落地于：

```text
V5 relay constructor
training builder
inference builder
model validator
checkpoint / architecture-manifest validator
experiment variant registry
```

任何非 `complement_tail` 请求都必须立即拒绝，不得默默回退到 V4 或其他 support mode。

## 5.3 Architecture manifest

新增字段：

```python
{
    "relay_version": V5_PER_RELAY_VERSION,
    "stage4_formula": "v4_exact",
    "stage3_formula": "v4_exact",
    "stage2_positive_route": (
        "centered-(1-persistent)*relu(centered)"
    ),
    "stage2_negative_route": "unchanged_identity_path",
    "stage2_dc_support": "one_minus_persistent_tail",
    "stage2_persistent_support_gradient": "stopped",
    "parameters_added_vs_v4": 0,
    "buffers_added_vs_v4": 0,
    "state_layout_compatible_with": "ner_v4_tail_aware",
    "state_semantics_identical_to_v4": False,
}
```

这里的兼容仅指：

```text
state key 集合相同
tensor shape / dtype 合同相同
strict state_dict load 可成功
```

不表示 V4/V5 的 stage2 forward 语义相同，也不授权把 V4 best checkpoint 作为 V5 development 训练的 warm start。

## 5.4 集成模型

新增：

```text
model/tpd_ner_v8_mprs_dch_v5_per_qfg_v2_croa_survival.py
```

推荐继承当前 V4+QFG+Survival 整合类，保持 TPD8、五个 evidence nodes、`q4→q3→q2`、decoder 和 QFG2 连接不变，只替换：

```text
self.tpd_ner
```

保留：

```text
QFG prepare-once / reuse
TSS heads 注册
structured training output
head-free inference export
```

下一轮训练设置沿用已有 TSS-off 的真实合同：

```text
requested_tss_weight=0.0
tss_enabled=false
tss_heads_registered=true
tss_training_forward_computes_logits=true
tss_loss_consumes_logits=false
tss_survival_target_constructed=false
```

TSS logits 仍可计算，但 loss 不消费它们。这样与已完成的 TSS-off 训练合同保持一致。

### 5.4.1 完整训练/推理集成合同

V5 不能只新增 relay 类。在授权 GPU 训练前，必须具备下列闭环：

```text
1. V5-PER relay implementation
2. V5 + TPD8 + five-node NER + QFG2 + Survival training graph
3. head-free V5 inference graph
4. formal training builder
5. formal inference builder
6. training-model validator
7. inference-model validator
8. training-state -> head-free inference export
9. experiment variant registry / CLI selection entry
10. architecture manifest and canonical architecture id
11. evaluator builder registration
12. strict checkpoint-role and source-lock validation
```

必须验证：

```text
training eval segmentation output
== exported head-free inference output

registry-selected builder
== direct V5 formal builder

validator rejects:
non-complement_tail / wrong relay version / wrong five-node layout /
wrong QFG mode / unexpected TSS objective / missing manifest keys
```

### 5.4.2 初始化与联合训练合同

所有 V5 开发 run 必须使用 fresh seed42 初始化：

```text
fresh_seed42_initialization=true
warm_start_from_v4_best=false
resume_allowed_only_from_same_v5_run=true
```

V4 checkpoint 只允许用于零训练 knockout、state-layout 测试和固定权重反事实分析。

`保持 QFG2 不变` 的准确含义是：

```text
QFG2 architecture/config/formula frozen
QFG2 parameters jointly trainable from the frozen seed42 init policy
QFG2 parameters are not weight-frozen
```

同理，TPD8、NER 共享层和 decoder 均按既定配方联合训练；只有 TSS loss 关闭。

## 5.5 不修改历史文件

禁止覆盖：

```text
model/tpd_ner_v8_mprs_dch_v4_tail_aware.py
model/tpd_frequency_gate_v2_croa.py
model/tpd_clean_v8_mprs_dch.py
experiments/tpd_training_loss.py
EC-TSS V3.1 文件
历史 selectors
```

---

# 6. 开发训练前：零训练 NER 假设审计

在新增 development run 之前，先用现有 V4 TSS-off checkpoint 完成无需训练的 stage2-only knockout。

## 6.1 Stage2-only 三级筛选

不在启动时就对全部 stage、全部 checkpoint 重复推理。首先只做与 V5 公式直接对应的：

```text
V4 reference
stage2 mask = 0
```

保持 stage4、stage3、relay value、TPD8、QFG2、decoder 和所有权重完全不变。这是固定权重下的 mask-output sensitivity，不是“重训后删除 stage2 的性能”。

### Level 1：primary 快速筛选

```text
3 个数据集
各自 V4 TSS-off best_miou checkpoint
各 1 次 stage2-mask-off 新推理
V4 reference 结果直接复用
```

### Level 2：checkpoint-role 确认

只有 Level 1 至少出现 A 的方向性信号，才在三个 `best_pd` checkpoint 重复 stage2-only knockout。`best_pd` 仅检查结论是否过度依赖 checkpoint role，不是主裁决点。

### Level 3：机制对齐

仅对 Level 1/2 已生成的预测与 hook 缓存进行 P2 / mask / false-component 对齐，不重复完整前向。若 stage2 假设通过后还需要解释其他 stage，stage4/stage3/all-off 只作后续归因附件，不阻塞 V5 启动门。

输出：

```text
Pd
component-Fa
pixel FP
pixel precision
pixel F1
mIoU
nIoU
tiny-Pd
错误目标数
```

## 6.2 Stage2 机制统计

记录：

```text
P2 在 GT target cell 的分布
P2 在 unmatched false component cell 的分布

stage2 positive mask mass on target
stage2 positive mask mass on background
stage2 negative mask mass on target
stage2 negative mask mass on background

background positive mask 与 pixel FP 的相关性
background positive mask 与 component-Fa 的相关性
```

## 6.3 NER V5 启动触发门

触发关系固定为：

```text
A AND (B OR C)
```

其中：

```text
A. Level-1 best_miou 中至少 2/3 数据集满足：
   component-Fa 或 all-background pixel-FP 相对降低 >= 5%；
   matched-target 下降 < 2；
   matched-tiny-target 下降 < 2；
   mIoU 与 nIoU 下降均 < 0.005。

B. false component 区域的 stage2 positive mask mass
   / normal-background positive mask mass >= 1.25，
   且在至少 2/3 数据集成立；

C. P2 <= 0.25 的背景区域承载 >= 25%
   stage2 background positive mask mass，
   且在至少 2/3 数据集成立。
```

所有比例的分母为 0 时不得记为通过；必须记录原始 count、mass、区域像素数和空集语义。A/B/C 只在当前 test-selected 开发集上授权一次 V5 候选训练，不构成论文机制证据。

若触发式不满足：

```text
ner_v5_per_development_training_authorized=false
```

此时应跳到 QFG 诊断，而不是强行训练 NER V5。

---

# 7. 单元测试计划

## 7.1 State 与参数

```text
V4/V5 parameter count 完全相同
V4/V5 state key 完全相同
strict load 只验证 key/shape/layout 合同
compatibility scope = key/shape/layout only
stage2 forward semantics intentionally differ
```

## 7.2 Stage4/3 精确继承

相同输入、相同 state：

```text
V5 stage4 relay_value == V4
V5 stage4 mask == V4
V5 stage3 relay_value == V4
V5 stage3 mask == V4
```

逐元素相等。

## 7.3 Stage2 公式

### 持久支持为 1

```text
background_support = 0
V5 shifted == centered
DC 不生效
```

当前实现把 tail support 严格限制在 `[0,1)`，所以 `persistent_support=1 / background_support=0` 只能作为 mock/monkeypatch 的代数极限测试，不能表述成正常运行中可达的真实状态。另需测试真实范围：`0 <= P < 1`、`0 < B <= 1`。

### 持久支持为 0

```text
background_support = 1
positive centered 被删除
negative centered 保留
DC 完整生效
```

### centered 为负

```text
V5 local term == centered
```

### centered 为正

```text
V5 local term == persistent_support * centered
```

## 7.4 零点等价

```text
gate weight = 0
dc offset = 0
→ mask = 0
→ decoder output 与 V4 / relay-off reference 相同
```

## 7.5 梯度

```text
persistent support requires_grad=false
stage2 gate weight 有有限梯度
dc offset 有有限梯度
TPD / QFG / decoder 共享参数有有限梯度
```

## 7.6 TSS-off 合同

```text
total loss == segmentation loss
TSS logits 不被 loss 消费
TSS head 参数 grad is None
```

## 7.7 推理导出

```text
训练模型 eval segmentation output
==
head-free inference output
```

---

# 8. 200-epoch durable development milestone

通过零训练触发门和测试后，运行：

| 数据集 | 模型 | TSS | Seed | Full schedule | Pause |
|---|---|---:|---:|---:|---:|
| NUAA | NER V5-PER + QFG2 | off | 42 | 1000 | epoch 200 |
| NUDT | NER V5-PER + QFG2 | off | 42 | 1000 | epoch 200 |
| IRSTD | NER V5-PER + QFG2 | off | 42 | 1000 | epoch 200 |

`pause-after-epoch=200` 只作为 durable pause，不改变 1000-epoch scheduler。它不是独立的 200-epoch 训练，不得在通过后重新从 epoch 1 启动另一个 1000-epoch run。

```text
run identity 不变
epoch 1..200 -> durable pause
same optimizer/scheduler/scaler/RNG/sampler state
exact resume at epoch 201
epoch 201..1000 -> same run
```

epoch-200 milestone 只检查：

```text
loss / gradient / model state 全部 finite
stage2 routed mask 不是连续 3 个评估点的全零或常数场
complement_tail 是唯一 active support mode
TSS-off 时 TSS loss contribution == 0
exact resume state 完整且可继续
连续 3 个评估点不同时出现 Pd == 0 且 mIoU == 0
```

Pd/Fa/mIoU/nIoU/pixel-FP 的普通上下波动仅记录，不在 epoch 200 做性能优劣选择；历史 best epoch 很晚，不能用 200 epoch 的相对落后否决候选。该 milestone 不用于论文结果或 checkpoint 选择。

---

# 9. development1000 实验矩阵

milestone 通过后，继续同一三个 run 至 1000 epochs。

| 数据集 | 新候选 | Seed | Epochs | TSS | QFG |
|---|---|---:|---:|---:|---:|
| NUAA-SIRST | NER V5-PER | 42 | 1000 | off | QFG2 架构冻结、参数联训 |
| NUDT-SIRST | NER V5-PER | 42 | 1000 | off | QFG2 架构冻结、参数联训 |
| IRSTD-1K | NER V5-PER | 42 | 1000 | off | QFG2 架构冻结、参数联训 |

比较对象：

```text
Original
当前 V4 + QFG2 + TSS-off
新 V5-PER + QFG2 + TSS-off
```

辅助描述可继续保留：

```text
旧 TSS λ=0.005
EC-TSS V3.1
```

但它们不参与 V5 的配方搜索。

固定：

```text
img_idx/train
img_idx/test
seed42
1000 epochs
每10 epoch评估
best_miou
best_pd
threshold=0.5
同一 evaluator
```

上述口径必须在所有输出中标记：

```text
evaluation_protocol=img_idx_test_selected_development
paper_unbiased_test_supported=false
best_miou_role=primary_development_checkpoint
best_pd_role=secondary_high_recall_description
fixed_threshold=0.5
threshold_selected_on_test=false
```

该矩阵中 Original 和 V4 TSS-off 可复用同一历史开发协议的结果，因此只需新训练 3 个 V5 run。它支持内部候选筛选，不支持无偏论文外推。若后续改用独立 validation/test 协议，Original、V4、V5 必须在新协议下重新对齐，不得混用当前 checkpoint。

---

# 10. NER V5 量化 development Gate

## Gate N5-A：工程闭环

```text
3 个 development1000 完整
6 个 checkpoint 可 strict load
exact resume 通过
ordinary / python -O 测试通过
source lock 完整
head-free export 等价
fresh seed42 init attested
complement_tail-only validator passed
five-node layout unchanged
QFG2 architecture frozen but parameters jointly trained
```

任一项失败即为工程失败，不进入性能解读。

## Gate N5-B：相对 Original

使用现有冻结严重退化门。对每个 dataset-role 分别检查：

```text
matched_target_count drop >= 2                         -> 1 violation
matched_tiny_target_count drop >= 2                    -> 1 violation
mIoU drop >= 0.005                                     -> 1 violation
nIoU drop >= 0.005                                     -> 1 violation
unmatched_predicted_pixels increase > 25%
  without matched_target_count gain >= 2               -> 1 violation
reference unmatched pixels == 0 and candidate > 0
  without matched_target_count gain >= 2               -> 1 violation
```

IoU、precision 和 F1 比较先按 `1e-4` 量化；count 使用原始整数，不使用显示后小数反推。

目标：

```text
severe_degradation_violations
<
当前 V4 TSS-off 的 5 项
```

强通过条件：

```text
severe_degradation_violations == 0
```

并且不存在任何数据集：

```text
Original 在 best_miou / best_pd 两个角色
都严格支配 V5
```

`< 5` 只表示相对当前 V4 TSS-off 的阶段进步；`==0` 才标记为 `original_floor_strong_pass=true`。

## Gate N5-C：相对 V4 TSS-off

对冻结联合指标向量：

```text
mIoU↑, nIoU↑,
matched_target_count↑, matched_tiny_target_count↑,
unmatched_predicted_pixels↓,
unmatched_predicted_object_count↓,
pixel_precision↑, pixel_F1↑
```

在 3 数据集 × 2 checkpoint roles × 8 metrics 上使用等权成对票数，要求：

```text
V5 不能在任一数据集的两个角色上
均被 V4 TSS-off 严格支配

V5_vs_V4 better_metric_count
>
V5_vs_V4 worse_metric_count

relative_to_V4 severe_degradation_violations == 0
```

下列 Pareto 结果继续报告，但降级为描述项，不单独决定 Gate。若报告，必须固定候选总体为 `{Original, V4_TSS_off, V5_PER}`，使用 Gate N5-C 中的冻结指标方向、`1e-4` 量化后的精确相等规则，并同时报告每个具体非支配点：

```text
4/6 dataset-role 单元进入联合 Pareto
2/6 为 V5 独有非支配点
```

## Gate N5-D：目标问题修复

以 primary `best_miou` checkpoint 为主，要求至少 2/3 数据集相对 V4 TSS-off 满足：

```text
component-Fa 或 all-background pixel-FP 相对降低 >= 5%
matched_target_count drop < 2
matched_tiny_target_count drop < 2
mIoU drop < 0.005
nIoU drop < 0.005
```

同时三数据集任何一个 `best_miou` 单元都不得触发 Gate N5-B 中的 Fa 严重增幅规则。`best_pd` 仅作高召回辅助检查，不能用 Pd 的单独上升覆盖 Fa、pixel-FP 或区域质量退化。

## Gate N5-E：机制一致（声称门，不是性能否决门）

在 primary `best_miou` 的至少 2/3 数据集上，相对同一 V5 权重下的 V4-stage2-formula counterfactual，要求：

```text
stage2 background positive mask mass 下降 >= 10%
target persistent region positive mask mass 保留 >= 90%
background negative mask mass 保留 >= 90%
```

若 N5-A/B/C/D 通过而 N5-E 失败：

```text
performance_candidate_supported=true
per_routing_mechanism_claim_supported=false
```

即保留真实性能候选，但不声称改进来自预期的 PER 机制。反之，N5-E 通过但性能 Gate 失败时，V5 仍必须回退。机制门不得否决已经通过的真实性能，也不得拯救失败的性能。

## 10.1 决策真值表

```text
N5-A pass AND N5-B stage_progression_pass
AND N5-C pass AND N5-D pass
    -> V5 performance pass; freeze NER architecture

N5-A pass but any of N5-B/C/D fails
    -> V5 performance fail; revert V4; close NER optimization

N5-E pass/fail
    -> only controls PER mechanism claim
```

---

# 11. NER V5 之后的 QFG 计划

只有 NER stage2 零训练启动门完成裁决后，才进入 QFG。若该门未通过，
NER V5 不训练、NER 回退并冻结为 V4，然后直接进入 QFG；不要求先制造一个
V5 训练结果。

## 11.1 先做层级 knockout，不直接改公式

当前 QFG 有四个 level，各自包含：

```text
alpha
Haar kernels
prior projection
spatial projection
terminal gate
```

对冻结 checkpoint 执行：

```text
level0 alpha → 0
level1 alpha → 0
level2 alpha → 0
level3 alpha → 0
all alpha → 0
```

文档中的 `level0...level3` 使用 Python 零基编号；实现层对应现有审计原语的
`level_1_off...level_4_off`。不得因编号差异关闭错误层。

输出：

```text
每 level 的 query perturbation RMS
gate RMS
factor min/max
target vs hard-negative gate difference
Pd/Fa/mIoU/nIoU/tiny/pixel FP
```

其中 query perturbation RMS 必须由 `apply_prepared` 的输入 Query 与输出 Query
逐元素差值直接累计，不能用 `factor-1` 代替；`factor-1` RMS 作为独立的无量纲
调制强度同时报告。

## 11.2 QFG 优化触发门

主裁决固定使用三数据集、seed42、各自 TSS-off `best_miou` checkpoint、固定阈值
0.5。阈值扫描只作描述，不重选 checkpoint，也不参与下面的 level 选择。

对每个 `level-off` 或 `all-off` 模式，相对同一权重的 `full` 定义：

```text
delta_target = matched_target_off - matched_target_full
delta_tiny   = matched_tiny_target_off - matched_tiny_target_full
delta_mIoU   = mIoU_off - mIoU_full
delta_nIoU   = nIoU_off - nIoU_full
component_Fa_pixel_reduction =
    (unmatched_predicted_pixels_full - unmatched_predicted_pixels_off)
    / unmatched_predicted_pixels_full
background_pixel_FP_reduction = (background_FP_full - background_FP_off) / background_FP_full
```

这里 component-Fa 的分子固定为 `unmatched_predicted_pixels`，除以
`valid_pixel_count` 即仓库的 Fa；不得替换成 `unmatched_predicted_object_count`。
后者及 `false_objects_per_image` 只作为描述项。`background_FP` 固定为所有
GT-background 上的 `false_positive_pixels`。

某一模式在一个数据集上记为 `safe_material_improvement=true`，必须同时满足：

```text
安全条件：
delta_target > -2
delta_tiny > -2
delta_mIoU > -0.005
delta_nIoU > -0.005
component_Fa_pixel_reduction > -0.05
background_pixel_FP_reduction > -0.05

且至少一个实质收益：
delta_target >= 2
或 delta_tiny >= 2
或 delta_mIoU >= 0.005
或 delta_nIoU >= 0.005
或 component_Fa_pixel_reduction >= 0.05
或 background_pixel_FP_reduction >= 0.05
```

当分母为 0 时：`full=0, off=0` 的相对降低记 0；`full=0, off>0`
直接视为该 FP 安全条件失败，并同时记为第三数据集严重退化，不制造无穷大
百分比。

同一 level 只有在至少 2/3 数据集满足 `safe_material_improvement`，并且第三个
数据集不出现下列任一严重退化时，才定义为 `persistent_harmful_level`：

```text
matched target 下降 >= 2
或 matched tiny target 下降 >= 2
或 mIoU / nIoU 任一下降 >= 0.01
或 component-Fa unmatched pixels / background-pixel-FP 任一增加 >= 25%
或 reference 的相应 FP 为 0、candidate 的相应 FP 大于 0
```

`persistent_harmful_level` 只允许包含四个单层模式 `level0_off...level3_off`，
绝不包含 `all_off`。

只有出现至少一个 `persistent_harmful_level`，才授权设计一次 QFG V3；V3 只能
移除这些 level 的 modulation，不能增加新频率分支。

优先候选不是增加频率分支，而是：

```text
移除有害 level 的 modulation
```

即参数不增加、identity 可验证的 level-selective QFG。

`all-off` 还承担 QFG 整体去留判定。反向比较必须显式使用：

```text
delta_target = matched_target_full - matched_target_alloff
delta_tiny   = matched_tiny_target_full - matched_tiny_target_alloff
delta_mIoU   = mIoU_full - mIoU_alloff
delta_nIoU   = nIoU_full - nIoU_alloff
component_Fa_pixel_reduction =
    (unmatched_predicted_pixels_alloff - unmatched_predicted_pixels_full)
    / unmatched_predicted_pixels_alloff
background_pixel_FP_reduction =
    (background_FP_alloff - background_FP_full) / background_FP_alloff
```

此时以 `alloff` 为 denominator reference：`alloff=0, full=0` 记 0；
`alloff=0, full>0` 同时判 safety fail 和 severe veto。若 `full` 相对 `all-off`
在至少 2/3 数据集形成同样的安全实质收益，且无第三数据集严重退化，则：

```text
qfg_performance_contribution_supported=true
decision=FREEZE_QFG2_KEEP
```

若 `all-off` 自身在至少 2/3 数据集是安全实质改进且无严重退化，则：

```text
decision=DESIGN_QFG_OFF_CANDIDATE
```

功能等价固定在原始、未 padding 的全部测试像素上计算：

```text
max_abs = max(|probability_full - probability_alloff|)
mean_abs = sum(|probability_full - probability_alloff|) / 原始有效像素总数
equivalent = (max_abs <= 1e-7) AND (mean_abs <= 1e-8)
```

只有 `full` 与 `all-off` 在 3/3 数据集均 equivalent、没有任何
`persistent_harmful_level`，且 `full` 相对 `all-off` 在任何数据集都没有形成
`safe_material_improvement` 时，才允许：

```text
qfg_functional_contribution_supported=false
decision=QFG_CONTRIBUTION_UNSUPPORTED_CONSIDER_REMOVE
```

则应考虑从最终论文主模型中删除 QFG，而不是为了创新性强行保留。其余混合结果
统一记为 `QFG_INCONCLUSIVE_NO_FORMULA_CHANGE`，不授权凭单数据集最优值改公式，
直接进入 TPD block-wise 诊断。

多项条件同时出现时，决策优先级也预先冻结为：

```text
1. persistent_harmful_level 非空
   -> DESIGN_QFG_V3_REMOVE_LEVELS
2. 否则 all-off 是跨数据集安全实质改进
   -> DESIGN_QFG_OFF_CANDIDATE
3. 否则 full 相对 all-off 是跨数据集安全实质改进
   -> FREEZE_QFG2_KEEP
4. 否则 full/all-off 在 3/3 数据集功能等价，且 full 在任何数据集
   均无 safe_material_improvement
   -> QFG_CONTRIBUTION_UNSUPPORTED_CONSIDER_REMOVE
5. 否则
   -> QFG_INCONCLUSIVE_NO_FORMULA_CHANGE
```

优先级 1 高于 2 表示固定采用“能通过门时先做最小局部删层、再考虑整体关闭”的
政策，不代表局部证据强于整体证据。所有比较使用 JSON 中未舍入的原始数值，
不在上述边界之外增加隐藏容差。

---

# 12. QFG 之后的 TPD 计划

TPD 最后处理。首轮只做固定权重、固定 checkpoint 的 block residual knockout，
不得先改公式或重新训练。

## 12.1 固定对象与九种首轮模式

七个 block 按局部深度从零编号：

```text
E1.B0 / E1.B1 / E1.B2 / E1.B3 = mtc.embeddings_1.blocks[0:4]
E2.B0 / E2.B1 / E2.B2         = mtc.embeddings_2.blocks[0:3]
```

首轮只运行九种模式：

```text
full
e1b0_off / e1b1_off / e1b2_off / e1b3_off
e2b0_off / e2b1_off / e2b2_off
all7_off
```

`off` 的唯一合法实现是：在一次临时上下文中把选中 block 的完整
`saliency_scale` 向量置零，退出后逐值恢复并核对 state SHA。由于
`tanh(saliency_scale)=0`，该 block 的 residual 为零，输出变为
`activation(Keep)`。不得跳过 block、把 block 输出置零或改变
`phase_compress`；这些做法会把下采样和 Keep 路径也一起删除。

因此，`all7_off` 检验的是训练后 **TPD8 Saliency + DCH residual** 的贡献；
它仍保留 Keep/SPD 路径，不能被表述成“整个 TPD 架构关闭”。完整 TPD 相对
Original 的贡献仍由各自训练的正式主实验比较回答。

## 12.2 固定实验口径

```text
数据集：NUAA / NUDT-SIRST / IRSTD-1K
随机种子：42
checkpoint：各数据集自己的 TSS-off best_miou
split：各数据集 img_idx/test
主阈值：0.5
固定权重：NER4、QFG2、decoder 和其余参数全部不变
产物：不生成派生 checkpoint，不保存整套 probability cache
```

阈值扫描不参与本阶段裁决。为避免九模式重复执行高成本自适应扫描，只登记
`0.5` 主工作点和合法的 `1.0` 空预测端点；所有性能门只读取未舍入的
`fixed_threshold_0_5`。

同一次 full 推理额外记录下列描述性量，但它们不得越过性能门直接授权改模型：

```text
每个 block 的 tanh(saliency_scale) RMS / min / max
Keep RMS、Context-aligned RMS、Saliency-v8 RMS、residual RMS
phase-correction RMS、modulation RMS、headroom 偏离 1 的 RMS
target / background 区域 residual RMS 与二者差值
```

## 12.3 单数据集性能门

任一 candidate 相对 reference 定义：

```text
Δtarget = matched_target_count(candidate) - matched_target_count(reference)
Δtiny   = matched_tiny_target_count(candidate) - matched_tiny_target_count(reference)
ΔmIoU   = mIoU(candidate) - mIoU(reference)
ΔnIoU   = nIoU(candidate) - nIoU(reference)

Rcomp = (reference.unmatched_predicted_pixels
         - candidate.unmatched_predicted_pixels)
        / reference.unmatched_predicted_pixels
Rbg   = (reference.false_positive_pixels
         - candidate.false_positive_pixels)
        / reference.false_positive_pixels
```

这里 `Rcomp` 必须使用 unmatched predicted **pixels**，不得改用 object count；
`Rbg` 是所有 GT-background 像素上的假阳性像素数。safe 要求以下条件全部成立：

```text
Δtarget > -2
Δtiny > -2
ΔmIoU > -0.005
ΔnIoU > -0.005
Rcomp > -0.05
Rbg > -0.05
```

material 要求以下至少一项成立：

```text
Δtarget >= 2 或 Δtiny >= 2
ΔmIoU >= 0.005 或 ΔnIoU >= 0.005
Rcomp >= 0.05 或 Rbg >= 0.05
```

`safe_material_improvement = safe AND material`。严重退化为以下任一项成立：

```text
Δtarget <= -2 或 Δtiny <= -2
ΔmIoU <= -0.01 或 ΔnIoU <= -0.01
Rcomp <= -0.25 或 Rbg <= -0.25
```

若 FP reference 为零：candidate 也为零时 reduction 记 `0`；candidate 大于零时
reduction 记 JSON `null`，同时 safe 失败且 severe=true。反向比较 full 与
all7-off 时必须交换 reference 后按完全相同的公式重算，不能沿用正向百分比。
所有比较使用 JSON 未舍入值，不增加隐藏容差。

跨数据集通过固定为：至少 `2/3` 数据集达到 safe-material，且 `0/3` 数据集出现
严重退化。

## 12.4 功能差异与状态恢复门

对每种模式，在所有原始未 padding 测试像素上计算：

```text
max_abs  = max(abs(p_full - p_mode))
mean_abs = sum(abs(p_full - p_mode)) / valid_pixel_count
equivalent = (max_abs <= 1e-7 AND mean_abs <= 1e-8)
```

程序必须校验 `element_count == fixed-point valid_pixel_count`、九种模式的
valid-pixel count 相同、full 自差严格为零、每个模式结束后参数和
`saliency_scale` SHA 完全恢复。功能差异只回答 residual 是否实际进入输出，
不能单独授权修改模型。

## 12.5 首轮裁决与最多一个二阶段组合

单 block off 相对 full 跨数据集通过时，该位置记为
`persistent_harmful_block`。这只说明当前已训练权重下该位置 residual 有害，
不说明整个 block 或 TPD 机制有害。

若有害集合刚好构成共同局部深度后缀，只允许新增一个唯一 early-only 组合；
若仅一个有害 block，直接复用该单 block 结果；若有多个但不是完整后缀，只允许
新增一个 `harmful_union_off`。二阶段最多增加一种模式，并且组合模式必须用自己
的三数据集性能结果通过同一门，不能由多个单 block 结果直接推定。

冻结裁决优先级：

```text
1. 唯一 early-only / block-selective 组合自身跨数据集通过
   -> DESIGN_TPD_EARLY_ONLY_CANDIDATE
      或 DESIGN_TPD_BLOCK_SELECTIVE_CANDIDATE
2. 否则 all7-off 相对 full 跨数据集通过
   -> DESIGN_TPD_RESIDUAL_OFF_CANDIDATE
3. 否则 full 相对 all7-off 跨数据集通过
   -> FREEZE_TPD8_RESIDUAL_FULL
      tpd_residual_performance_contribution_supported=true
4. 否则仅当 full/all7-off 在 3/3 数据集功能等价、有害 block 为空，
   且 full 在任何数据集均无 safe-material 改善
   -> TPD_RESIDUAL_CONTRIBUTION_UNSUPPORTED_CONSIDER_SIMPLIFY
5. 其余
   -> TPD_INCONCLUSIVE_NO_FORMULA_CHANGE
```

首轮九模式只能确定是否需要唯一二阶段组合；只有该组合再次通过性能门，才授权
一次 fresh-training 候选。无论结果如何，本阶段均不得同时修改：

```text
phase formula
Context headroom
NER
QFG
```

## 12.6 已执行结果与裁决

本节记录已经完成的正式 TPD8 block-residual 诊断，不再是待执行计划。正式环境为：

```text
数据集：NUAA-SIRST / NUDT-SIRST / IRSTD-1K
随机种子：42
checkpoint：各数据集自己的 TSS-off best_miou
split：各数据集 img_idx/test
固定阈值：0.5
模式：full + 7 个 single-block off + all7_off，共 9 种
固定权重：NER4、QFG2、decoder 与其余参数均不变
analyzer source SHA256：3570475b13f89d629eb43fa155e85e4500f20ea8257a560e721d52bdba402abe
reference replay：3/3 PASS
model state / saliency_scale 恢复：3/3 exact PASS
派生 checkpoint / probability cache / feature cache：均未写入
```

三份正式 checkpoint SHA256 为：

```text
NUAA-SIRST：e6958eebb4a4a5493342a9faf285b2c57a5d58804f150656a87585fec3043f0a
NUDT-SIRST：0f5f6a5fe96fa86302807d132078d575495a3aff6690967785868a23400f3e84
IRSTD-1K：e8e9401500502dda0bbdc9640b830a7934fb2bc97bde706fde9adca216d965b4
```

固定阈值 `0.5` 的完整目标级指标如下；括号中保留原始计数：

| 数据集 | 模式 | Pd（matched/target） | tiny-Pd（matched/target） | Fa（unmatched pixels/valid pixels） | predicted / unmatched objects | false objects/image |
|---|---|---:|---:|---:|---:|---:|
| NUAA-SIRST | full | 0.973384030418251（256/263） | 0.8571428571428571（30/35） | 0.000015435192155794186（225/14577078） | 277 / 21 | 0.09813084112149532 |
| NUAA-SIRST | all7_off | 0.973384030418251（256/263） | 0.8571428571428571（30/35） | 0.000015435192155794186（225/14577078） | 277 / 21 | 0.09813084112149532 |
| NUDT-SIRST | full | 0.9904761904761905（936/945） | 0.9961389961389961（258/259） | 0.000002780592585184488（121/43515904） | 962 / 26 | 0.0391566265060241 |
| NUDT-SIRST | all7_off | 0.9894179894179894（935/945） | 0.9922779922779923（257/259） | 0.0000027116522731551203（118/43515904） | 961 / 26 | 0.0391566265060241 |
| IRSTD-1K | full | 0.9326599326599326（277/297） | 0.7666666666666667（23/30） | 0.000011728770697294776（618/52690944） | 321 / 44 | 0.21890547263681592 |
| IRSTD-1K | all7_off | 0.9326599326599326（277/297） | 0.7666666666666667（23/30） | 0.000011861620850824006（625/52690944） | 322 / 45 | 0.22388059701492538 |

同一工作点的区域与像素指标如下：

| 数据集 | 模式 | mIoU | nIoU | pixel precision | pixel recall | pixel F1 | background FP | test loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | full | 0.796482950889985 | 0.795348496003674 | 0.887879512311738 | 0.8855507868383404 | 0.8867136206279097 | 938 | 0.00047610574877742304 |
| NUAA-SIRST | all7_off | 0.796204974271012 | 0.7948044330649233 | 0.8876538783315405 | 0.8854315689079637 | 0.8865413309459863 | 940 | 0.0004765543438301495 |
| NUDT-SIRST | full | 0.9444060056735626 | 0.9464232331333509 | 0.9788096091789171 | 0.9641179586791453 | 0.9714082377012722 | 591 | 0.00019070613720757493 |
| NUDT-SIRST | all7_off | 0.9443636615427227 | 0.9462661933440094 | 0.9795661854485384 | 0.9633409853434576 | 0.9713858371467744 | 569 | 0.00019198017939291072 |
| IRSTD-1K | full | 0.6603115414686955 | 0.6656617448460672 | 0.8398867809057528 | 0.7554011283886061 | 0.7954067956241397 | 2093 | 0.000719415491579666 |
| IRSTD-1K | all7_off | 0.6599277978339351 | 0.6647419287628069 | 0.8402022368622645 | 0.7546442823723682 | 0.7951283166594173 | 2086 | 0.000716009629951571 |

性能门的正式聚合结果为：

```text
7 个 single-block off：
  off→full 的 safe-material 均为 0/3，severe 均为 0/3
  full→off 的 safe-material 均为 0/3，severe 均为 0/3
  persistent_harmful_block = 空集

all7_off→full：safe-material=0/3，severe=0/3
full→all7_off：safe-material=0/3，severe=0/3
```

因此，没有单 block、局部 block 组合或 `all7_off` 获得模型修改授权，也无需新增
第十种 early-only / harmful-union 模式。

功能差异不是零。七个 single-block off 在 3/3 数据集均为
`functionally_different=true`；`all7_off` 的完整差异为：

| 数据集 | max_abs | mean_abs | absolute difference sum | element count | functionally different |
|---|---:|---:|---:|---:|:---:|
| NUAA-SIRST | 0.06983965635299683 | 0.000001242181559424201 | 18.10737748188821 | 14577078 | YES |
| NUDT-SIRST | 0.4851060062646866 | 0.0000021746451257350213 | 94.63164852555312 | 43515904 | YES |
| IRSTD-1K | 0.14693868160247803 | 0.0000026068548838070876 | 137.35764469880576 | 52690944 | YES |

full 模式的生产路径统计还显示，三个数据集、七个 block 共 `21/21` 个位置均满足
`target_residual_R RMS > background_residual_R RMS`：

| 数据集 | 正 margin block 数 | target RMS − background RMS 范围 |
|---|---:|---:|
| NUAA-SIRST | 7/7 | 0.02351058485085654 ～ 0.09715297504740344 |
| NUDT-SIRST | 7/7 | 0.16223707046768612 ～ 0.8657506603283772 |
| IRSTD-1K | 7/7 | 0.4303543406596625 ～ 1.6430608371123563 |

这证明 TPD residual 实际进入输出，并呈现目标区域响应更强的描述性行为；但它不
等于跨数据集性能贡献，也不能越过固定性能门授权改公式。最终裁决为：

```text
decision=TPD_INCONCLUSIVE_NO_FORMULA_CHANGE
tpd_local_candidate_training_authorized=false
tpd_residual_off_candidate_authorized=false
tpd_residual_performance_contribution_supported=false
tpd_full_architecture_contribution_supported=false
tpd_functionally_active=true
next_step=FREEZE_CURRENT_INFERENCE_ARCHITECTURE_RETURN_TO_MODEL_LEVEL_PERFORMANCE_OPTIMIZATION
```

正式产物：

```text
results/three_dataset_tpd8_block_residual_knockout_v1/runs/NUAA-SIRST/v4_tss_off_best_miou_seed42/evaluation.json
results/three_dataset_tpd8_block_residual_knockout_v1/runs/NUDT-SIRST/v4_tss_off_best_miou_seed42/evaluation.json
results/three_dataset_tpd8_block_residual_knockout_v1/runs/IRSTD-1K/v4_tss_off_best_miou_seed42/evaluation.json
results/three_dataset_tpd8_block_residual_knockout_v1/comparison/best_miou_seed42/decision.json
results/three_dataset_tpd8_block_residual_knockout_v1/comparison/best_miou_seed42/decision.md
```

---

# 13. 论文创新组织

## 13.1 建议核心创新

### 核心 1：TPD8-MPRS-DCH

```text
目标保真浅层 tokenization
质量守恒 phase-resolved saliency
Keep/Context/Saliency 三源
```

### 核心 2：NER

```text
五节点浅层证据中继
q4→q3→q2
tail-aware / persistent-evidence routing
decoder skip modulation
```

### 核心 3：QFG（仅在后续通过时）

```text
Query-only frequency conditioning
不修改 K/V/CFN
```

## 13.2 TSS 的论文定位

TSS 不再作为核心创新。

建议写为：

> 我们进一步考察了多种训练期目标存活辅助监督，包括固定正权重、动态强度、无辅助目标以及误差条件化风险监督。尽管部分方案在特定数据集或工作点产生收益，但没有一个方案通过三数据集统一配方门，因此最终结构优化不再依赖 TSS。

这是一项有价值的负结果，而不是需要隐藏的失败。

## 13.3 不应写

```text
TSS 是最终模型核心贡献
EC-TSS 已建立统一增益
TSS-off 已全面优于所有 on 配方
QFG 已有显著独立贡献
完整架构已经全面超过 Original
```

---

# 14. 推荐项目状态

```text
decision=COMPONENT_DIAGNOSTIC_CLOSED_KEEP_TPD8_NER4_QFG2

current_inference_architecture=TPD8_MPRS_DCH_PLUS_NER4_PLUS_QFG2_CROA
current_inference_architecture_frozen=true
training_objective=TSS_OFF
tss_optimization_closed=true
tss_training_innovation_supported=false

ner_version=NER4
ner_v5_formula_modification_authorized=false
ner_v5_development_training_authorized=false
qfg_version=QFG2_CROA
qfg_formula_modification_authorized=false
qfg_optimization_authorized=false
tpd_version=TPD8_MPRS_DCH
tpd_formula_modification_authorized=false
tpd_optimization_authorized=false
tpd_residual_off_candidate_authorized=false
tpd_performance_contribution_supported=false
tpd_functionally_active=true
tpd_tenth_mode_required=false

development_protocol=seed42_img_idx_test_selected
paper_unbiased_test_supported=false
architecture_global_advantage_established=false
paper_core_established=false
stability_claim_supported=false
training_recipe_finalized=false

next_step=FREEZE_CURRENT_INFERENCE_ARCHITECTURE_RETURN_TO_MODEL_LEVEL_PERFORMANCE_OPTIMIZATION
```

---

# 15. 文件修改清单

## 新增模型

```text
model/tpd_ner_v8_mprs_dch_v5_per.py
model/tpd_ner_v8_mprs_dch_v5_per_qfg_v2_croa_survival.py
```

第二个集成文件必须同时实现并导出：

```text
V5 formal training class
V5 head-free inference class
build_formal_v5_per_training_model
build_formal_v5_per_inference_model
validate_formal_v5_per_training_model
validate_formal_v5_per_inference_model
build_inference_model_from_v5_training_state_dict
canonical architecture manifest / id
```

## 新增分析

```text
analysis/analyze_ner_stage2_mask_knockout_v1.py
analysis/compare_ner_stage2_mask_knockout_v1.py
analysis/analyze_three_dataset_qfg_level_knockout_v1.py
analysis/compare_three_dataset_qfg_level_knockout_v1.py
analysis/analyze_three_dataset_tpd8_block_residual_knockout_v1.py
analysis/compare_three_dataset_tpd8_block_residual_knockout_v1.py
```

第一个入口在同一次 stage2-mask-off 推理中同时缓存原始 V4 stage2 mask、\(P_2\)、GT 对齐和 false-component sufficient statistics，避免为 B/C 再重复完整前向；默认不保存整幅 probability arrays。

第三个入口逐数据集在内存中顺序执行 `full`、四个单 level-off 和 `all-off`，
每个模式只前向一次完整测试集；第四个入口只读取三份结果，严格执行第 11.2 节
冻结门，不重新推理。

第五个入口逐数据集执行 TPD8 的 `full`、七个 single-block off 和 `all7_off`，
同时记录生产路径 MPRS 统计、功能差异和逐模式状态恢复；第六个入口只读取三份
正式 JSON，复用相同的双向性能门并原子输出最终 Markdown/JSON 裁决。

## 新增训练与评估

```text
experiments/three_dataset_ner_v5_per_models_seed42_v1.py
experiments/train_three_dataset_ner_v5_per_tss_off_seed42.py
experiments/launch_three_dataset_ner_v5_per_seed42.py
experiments/evaluate_three_dataset_ner_v5_per.py
experiments/export_ner_v5_per_qfg_v2_croa_to_inference.py
```

比较与 finalize 入口只在 V5 获得训练授权后再新增；当前零训练门未通过时，
不得创建一个看似完整但没有合法输入结果的空闭环。

## 新增测试

```text
tests/test_ner_v5_per_model.py
tests/test_ner_v5_per_integration.py
tests/test_export_ner_v5_per_qfg_v2_croa_to_inference.py
tests/test_three_dataset_ner_v5_per_pipeline.py
tests/test_analyze_ner_stage2_mask_knockout_v1.py
tests/test_three_dataset_qfg_level_knockout_v1.py
tests/test_three_dataset_tpd8_block_residual_knockout_v1.py
tests/test_compare_three_dataset_tpd8_block_residual_knockout_v1.py
```

## 不修改

```text
现有 TSS / EC-TSS 文件
现有正权重与 TSS-off selectors
NER V4 历史模型
QFG2 核心实现
TPD8 核心实现
历史结果目录
```

---

# 16. 完整执行顺序

```text
Phase 0
封存 TSS、TSS-off、EC-TSS V3.1 最终裁决
停止所有 TSS 公式与 λ 搜索

Phase 1
实现最小 V5 relay、完整训练/推理集成骨架和工程测试
此时不启动训练

Phase 2
在现有 V4 TSS-off best_miou checkpoint 上完成三数据集
stage2-only knockout，并在同一次推理中完成 P2 / mask / component 对齐

Phase 2B
若 A AND (B OR C) 通过，授权一次 NER V5-PER development 训练
未通过则不训练 V5，转入 QFG knockout

Phase 2C
冻结 QFG 的 safe-material-improvement、persistent-harmful 与整体去留门
在三数据集 V4 TSS-off best_miou checkpoint 上执行四层和 all-off 诊断

Phase 3
完成 CPU、python -O、GPU smoke、exact resume、source lock

Phase 4
三数据集各运行一个 200-epoch durable pilot
只检查机制和灾难性失败

Phase 5
milestone 通过后从同一状态续训至 development1000

Phase 6
比较 Original、V4 TSS-off、V5-PER
执行 Gate N5-A 至 N5-E

Phase 7
NER V5 通过：
冻结 NER，进入 QFG level knockout

NER V5 失败：
回退 NER V4，直接进入 QFG level knockout

Phase 8（已完成）
QFG 完成后执行 TPD block-wise 诊断；三数据集九模式正式评估、双向性能门、
功能差异、生产路径 MPRS 统计与最终裁决均已闭环。
结果：TPD_INCONCLUSIVE_NO_FORMULA_CHANGE；无需第十模式，不授权 residual-off。
```

---

# 17. 最终结论

> **NER、QFG、TPD 三项组件诊断已经全部闭环。TSS 统一配方关闭并采用 TSS-off；NER V5 启动门未通过，保持 NER4；QFG2 与 TPD8 均已证实会实际改变输出，但固定 checkpoint 反事实没有建立跨数据集实质性能贡献，也没有建立关闭或改公式的收益。因此当前裁决是 `COMPONENT_DIAGNOSTIC_CLOSED_KEEP_TPD8_NER4_QFG2`：冻结现有 TPD8 + NER4 + QFG2 推理架构，不授权 NER V5、QFG、TPD 公式修改或 TPD residual-off，不新增第十种 TPD 模式；下一步回到完整模型级性能优化，而不是继续局部组件反事实搜索。**

## 17.1 已执行的 NER stage2 裁决

正式输入为三数据集各自 TSS-off、seed42、`best_miou` checkpoint。结果为：

| 数据集 | matched target Ref→stage2-off | component-Fa 降幅 | background pixel-FP 降幅 | mIoU / nIoU 下降 | A | C |
|---|---:|---:|---:|---:|:---:|:---:|
| NUAA-SIRST | 256→256 | -12.889% | +25.267% | 0.013999 / 0.011569 | FAIL | PASS |
| NUDT-SIRST | 936→939 | -137.190% | -5.076% | 0.008340 / 0.006390 | FAIL | PASS |
| IRSTD-1K | 277→275 | +6.796% | +9.030% | 0.004757 / 0.008172 | FAIL | PASS |

这里“降幅”为负表示误报反而增加。B 因没有与 V4 reference 对齐的概率缓存而
保持 `N/A`；但 A 已失败，所以补做 B 也不能改变当前不授权结论。

```text
decision=DO_NOT_AUTHORIZE_NER_V5_PER_DEVELOPMENT_TRAINING
ner_v5_per_development_training_authorized=false
next_step=THREE_DATASET_QFG_LEVEL_KNOCKOUT
```

裁决文件：

```text
results/ner_stage2_mask_knockout_v1/comparison/best_miou_seed42/decision.json
results/ner_stage2_mask_knockout_v1/comparison/best_miou_seed42/decision.md
```

## 17.2 已执行的 QFG level knockout 裁决

在 NER V5 未获训练授权后，使用同三份 V4 TSS-off、seed42、`best_miou`
checkpoint，依次执行 `full`、四个单层关闭和 `all_off`。`full` 与 `all_off`
固定阈值结果为：

| 数据集 | 模式 | Pd | matched target | Fa | unmatched pixels | background FP | mIoU | nIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | full | 0.973384 | 256/263 | 1.5435e-5 | 225 | 938 | 0.796483 | 0.795348 |
| NUAA-SIRST | all_off | 0.973384 | 256/263 | 1.5367e-5 | 224 | 941 | 0.796120 | 0.794875 |
| NUDT-SIRST | full | 0.990476 | 936/945 | 2.7806e-6 | 121 | 591 | 0.944406 | 0.946423 |
| NUDT-SIRST | all_off | 0.989418 | 935/945 | 2.8266e-6 | 123 | 589 | 0.944402 | 0.946550 |
| IRSTD-1K | full | 0.932660 | 277/297 | 1.1729e-5 | 618 | 2093 | 0.660312 | 0.665662 |
| IRSTD-1K | all_off | 0.932660 | 277/297 | 1.1748e-5 | 619 | 2116 | 0.659820 | 0.665169 |

四个单层关闭模式均为：

```text
safe_material_improvement=0/3
severe_degradation=0/3
persistent_harmful_level=false
```

整体关闭与反向保留比较也均为 `safe_material_improvement=0/3`。另一方面，
`full` 与 `all_off` 的 probability 在 3/3 数据集都超过冻结的功能差异阈值，
说明 QFG 确实改变网络输出，但当前固定 checkpoint 反事实没有建立跨数据集实质
性能收益，也没有建立删层或整体关闭的收益。

```text
decision=QFG_INCONCLUSIVE_NO_FORMULA_CHANGE
qfg_v3_remove_levels_authorized=false
qfg_off_candidate_authorized=false
qfg_performance_contribution_supported=false
qfg_functional_contribution_supported=true
next_step=TPD8_BLOCK_WISE_DIAGNOSTIC
```

因此当前不得设计 QFG V3，也不得直接把 QFG 改为 off。进入 TPD block-wise
诊断时保持现有 QFG2 连接和权重不变；这里的 `functional=true` 只表示关闭 QFG
会改变概率输出，不等于已证明性能贡献。

裁决文件：

```text
results/three_dataset_qfg_level_knockout_v1/comparison/best_miou_seed42/decision.json
results/three_dataset_qfg_level_knockout_v1/comparison/best_miou_seed42/decision.md
```

## 17.3 已执行的 TPD8 block-residual 裁决

QFG 裁决后保持 NER4、QFG2、decoder 和 checkpoint 权重不变，完成三数据集九模式
TPD8 诊断。完整原始指标见第 12.6 节；聚合结果为：七个 single-block off 和
`all7_off` 的正反向 safe-material 均为 `0/3`，severe 均为 `0/3`，不存在
`persistent_harmful_block`。因此没有已测单块候选，也没有需要新增模式才能验证的
多块共同信号。

所有七个 single-block off 及 `all7_off` 在 3/3 数据集均实际改变概率输出；full
生产路径的 21 个“数据集 × block”统计也全部满足目标 residual RMS 高于背景
residual RMS。它们说明 TPD residual 处于活跃状态，但没有建立其跨数据集实质
性能贡献，更不能据此授权 residual-off 或新公式。

```text
decision=TPD_INCONCLUSIVE_NO_FORMULA_CHANGE
tpd_residual_performance_contribution_supported=false
tpd_functionally_active=true
tpd_residual_off_candidate_authorized=false
tpd_tenth_mode_required=false

final_project_status=COMPONENT_DIAGNOSTIC_CLOSED_KEEP_TPD8_NER4_QFG2
next_step=FREEZE_CURRENT_INFERENCE_ARCHITECTURE_RETURN_TO_MODEL_LEVEL_PERFORMANCE_OPTIMIZATION
```

裁决文件：

```text
results/three_dataset_tpd8_block_residual_knockout_v1/comparison/best_miou_seed42/decision.json
results/three_dataset_tpd8_block_residual_knockout_v1/comparison/best_miou_seed42/decision.md
```

至此，NER、QFG、TPD 的局部组件诊断均已闭环；后续不再沿该链继续增加局部
knockout 模式，而是在冻结当前推理架构的前提下回到完整模型级性能优化。

---

# 18. 主要依据

## 用户提供的正式历史汇总

《SCTransNet 历史模型实验结果总汇》：

- 旧 TSS 无统一正收益；
- TSS-off 无全局准入；
- EC-TSS V3.1 完成三数据集 seed42 test-selected 1000-epoch 开发实验，但关键 Gate 失败；
- 正式决定停止 TSS 优化；
- 下一轮顺序为 NER→QFG→TPD。

## 当前仓库实现

- `model/tpd_ner_v8_mprs_dch_v4_tail_aware.py`
  - stage4/3/2 relay；
  - fixed tail thresholds；
  - persistent support；
  - complement-tail 只调制 DC offset。
- `model/tpd_frequency_gate_v2_croa.py`
  - QFG 只调制 Query；
  - fixed Haar、RMS normalization、bounded gate；
  - exact identity initialization。
- `model/tpd_clean_v8_mprs_dch.py`
  - Keep/Context/Saliency；
  - phase-resolved mass-preserving saliency。
- `experiments/tpd_training_loss_ec_tss_v3_1.py`
  - EC-TSS 只修改训练 objective；
  - 不新增模型参数或 buffer。
