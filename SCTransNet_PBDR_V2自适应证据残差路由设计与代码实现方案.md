# SCTransNet PBDR-V2 自适应证据残差路由设计与代码实现方案

> 日期：2026-08-06  
> 当前主干：`SCTransNet + TPD8-MPRS-DCH + 五节点 NER4 + QFG2-CROA`  
> 训练目标：`TSS OFF`，保留原六路 segmentation BCE  
> 候选模块：`PBDR-V2 Adaptive Evidence Residual Router`  
> 当前状态：**公式、核心模块、完整模型和独立训练链路均已实现；CPU/GPU/训练引擎验证通过，NUAA formal1000 已启动**

---

# 1. 为什么从 PBDR-V1 修改到 V2

PBDR-V1 的六角色零训练审计已冻结为：

```text
decision=PBDR_GLOBAL_FIXED_G_SCREEN_FAILED
pbdr_v1_implementation_authorized=false
pbdr_v1_training_authorized=false
```

它失败的核心不是 PBDR 研究方向本身，而是 V1 公式的三个限制：

1. `P∈{0,1}` 的硬保护会把 NUAA/NUDT 的部分 FP 一起锁在保护区；
2. 救援项必须满足 `d0>out`，无法救回 NUDT 中 `d0<=out` 的漏检目标；
3. rescue 与 suppression 共用一个全局 `g`，两个相反任务不能独立学习。

V2 只修改最终读出路由，不改变 TPD8、NER4、QFG2、decoder 或六路监督主线。

---

# 2. PBDR-V2 的冻结输入

一次模型 forward 内复用三个已有张量：

```text
q4    : NER stage4 的 B×8×h×w 持久证据
z_out : 当前 decoder 最终 B×1×H×W raw logit
z_d0  : 现有多尺度 deep-fusion B×1×H×W raw logit
```

`d0` 仍由原 `outconv(cat(gt2,gt3,gt4,gt5,out))` 生成，PBDR 不改变 `d0`
的输入，因此没有循环依赖。

---

# 3. 数学公式

## 3.1 q4 证据归一化

先停止 PBDR 分支对 NER4 证据生成路径的额外反向影响，然后在每张图内做
全通道、全空间 RMS 归一化：

\[
\bar q
=
\frac{\operatorname{stopgrad}(q_4)}
{\max(\operatorname{RMS}(q_4),10^{-6})}.
\]

FP16/BF16 的 reduction 必须在 FP32 计算，并使用 scale-normalized RMS
避免直接平方溢出。

## 3.2 可学习软置信度

\[
L_c
=
\operatorname{Up}_{bilinear}
\left(W_c * \bar q + b_c\right),
\]

\[
C
=
0.05+0.90\sigma(L_c).
\]

`C` 严格保持在 0.05 与 0.95 之间。因此：

- 高置信目标不会完全失去 suppression 通道；
- 低置信区域也不会完全失去 rescue 通道；
- 被 q4 误保护的 FP 不再像 V1 一样被硬锁死。

## 3.3 q4 直接残差

\[
L_q
=
\operatorname{Up}_{bilinear}
\left(W_q * \bar q\right),
\]

\[
Q=C\tanh(L_q).
\]

`W_q` 不使用 bias，避免新分支退化成整图统一 logit 平移。`Q` 不依赖
`d0>out`，因此可以处理 V1 无法救回的漏检目标。

## 3.4 独立救援与抑制

\[
R^+=C\operatorname{ReLU}(z_{d0}-z_{out}),
\]

\[
R^-=(1-C)\operatorname{ReLU}(z_{out}-z_{d0}),
\]

\[
g^+=0.5\tanh(a^+),\qquad
g^-=0.5\tanh(a^-).
\]

## 3.5 最终读出

\[
\boxed{
z_{PBDR-V2}
=
z_{out}
+Q
+g^+R^+
-g^-R^-
}
\]

所有低分辨率 logit 都固定按以下方式对齐：

```text
size=z_out.shape[-2:]
mode=bilinear
align_corners=false
```

---

# 4. 参数、初始化与等价性

| 参数 | 形状 | 数量 | 初始值 |
|---|---:|---:|---:|
| `confidence_projection.weight` | 1×8×1×1 | 8 | 0 |
| `confidence_projection.bias` | 1 | 1 | 0 |
| `direct_residual_projection.weight` | 1×8×1×1 | 8 | 0 |
| `rescue_strength_raw` | 1 | 1 | 0 |
| `suppression_strength_raw` | 1 | 1 | 0 |
| 合计 |  | **19** |  |

```text
state_key_count=5
persistent_buffer_count=0
state_prefix=pbdr_v2.
```

零点时：

\[
C=0.5,\quad Q=0,\quad g^+=0,\quad g^-=0,
\]

\[
z_{PBDR-V2}=z_{out}.
\]

因此初始六路输出、旧参数梯度和旧参数的第一次 Adam 更新都与 Current
逐位一致。首步可学参数为 `W_q/a+/a-`；置信投影在路由离开零点后
获得梯度，属于预期的两阶段启动。

`g+/g-` 为 signed coefficient。若训练后为负，必须如实报告语义反转；不会为了
证明原理而否定正向性能，但不能再把负系数解释成预定的 rescue/suppression 机制。

---

# 5. 完整模型接入

训练图：

```text
TPD8 + NER4 + QFG2-CROA + PBDR-V2 + training-only TSS heads
TSS loss weight = 0
state keys = 573
parameters = 10,870,247
```

推理图：

```text
TPD8 + NER4 + QFG2-CROA + PBDR-V2
no TSS heads
state keys = 569
parameters = 10,870,149
```

原六路训练返回中：

```text
1: sigmoid(gt5)      不变
2: sigmoid(gt4)      不变
3: sigmoid(gt3)      不变
4: sigmoid(gt2)      不变
5: sigmoid(d0)       不变
6: sigmoid(PBDR-V2)  替换原 sigmoid(out)
```

PBDR-V2 必须使用 `deepsuper=true`，因为路由显式复用 `d0`。

---

# 6. 已实现代码

## 核心路由

```text
model/tpd_persistent_evidence_residual_router_v2.py
```

已包含：

```text
stable detached RMS normalization
soft confidence
direct q4 residual
independent rescue/suppression coefficients
forward-local diagnostics
finite/shape/device/dtype validation
architecture manifest
strict zero-state validator
```

## 完整模型

```text
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v2.py
```

已包含：

```text
Survival training graph
head-free inference graph
shared forward mixin
formal builders
formal validators
exact parameter/state constants
architecture manifest
```

当前生产模型文件没有被修改，旧 checkpoint、旧 source lock 和旧结果不受影响。

---

# 7. 已完成验证

CPU 单元/集成测试：

```text
tests/test_tpd_persistent_evidence_residual_router_v2.py
tests/test_tpd_ner_v4_qfg_v2_croa_pbdr_v2_integration.py
tests/test_three_dataset_pbdr_v2_registry_trainer.py
14/14 passed
```

已建立：

```text
19 parameters / 5 state keys / 0 buffers
construction RNG neutral
zero-state six-output bitwise identity
zero-state test-mode bitwise identity
all shared state tensors bitwise equal
all shared step-0 gradients bitwise equal
all shared first-Adam parameters bitwise equal
direct residual obtains first-step gradient
rescue/suppression obtain independent first-step gradients
confidence obtains gradient after routing leaves zero anchor
training-to-inference state removal and output equivalence
nonzero routing diagnostics match the frozen reference formula
573-key PBDR resume identity rejects 568-key Current state
FP16/BF16 autocast forward dtype/value identity
```

正式训练仍固定为 FP32。AMP 测试只锁定零点前向 dtype/value；共享梯度和第一次
Adam 更新的逐位身份由 FP32 formal 合同保证。

RTX 5090 GPU0 最小 smoke：

```text
input=1x1x64x64 FP32
forward=pass
backward=pass
six output shapes=1x1x64x64
state keys=573
parameters=10,870,247
peak allocated memory=107.14 MiB
direct/rescue/suppression gradients=finite and nonzero
```

独立训练引擎 smoke：

```text
dataset=NUAA-SIRST
train_images=1
test_images=1
epochs=1
device=RTX 5090 GPU0
status=complete
protocol_document=experiments/PBDR_V2_PROTOCOL.md
threshold=0.5
training_state_keys=573
selected_checkpoints=best_miou,best_pd only
rolling_resume_removed_after_completion=true
```

---

# 8. 正式训练协议

PBDR-V2 不使用 Current checkpoint 热启动。三个数据集均从 seed42 scratch 构建：

```text
datasets=NUAA-SIRST, NUDT-SIRST, IRSTD-1K
split=each dataset img_idx
seed=42
epochs=1000
first evaluation=epoch 10
evaluation cadence=every 10 epochs
batch size=16
patch size=256
optimizer=Adam
precision=FP32
threshold=0.5
TSS weight=0
selected checkpoints=best_miou,best_pd only
```

配对 scratch 构建时，应从同一历史 builder 分别生成 Current 与 PBDR-V2，然后对
568 个共享初始 state 逐键复制和核对；五个 `pbdr_v2.*` 状态保持零。
这是初始随机性配对，不是 checkpoint warm start。

Resume 只允许同一 PBDR-V2 配方的 573-key rolling state，禁止把 Current 568-key
state 当作 resume。

---

# 9. 性能裁决

不要求每个数据集、每项指标都提升。PBDR-V2 前瞻使用 M2F-SV：

```text
detection family D+ >= 2/3 datasets
overlap family O+ >= 2/3 datasets
joint D+ and O+ >= 1 dataset
severe roles = 0/6
Original materially dominates candidate datasets = 0
```

正向投票只来自三个 `best_miou`；`best_pd` 只承担 severe veto 和完整报告。
该规则是看过 PBDR-V1 结果后冻结的后续协议：

```text
post_hoc_protocol_amendment=true
applies_to=PBDR_V2_and_later_scratch_runs_only
pbdr_v1_machine_decision_unchanged=true
```

---

# 10. 当前裁决

```text
pbdr_v1_fixed_formula_closed=true
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
pbdr_v2_nuaa_formal1000_status=RUNNING
pbdr_v2_nudt_formal1000_status=QUEUED_WAITING_FOR_BASELINE_NUAA_GPU1
pbdr_v2_irstd1k_formal1000_status=QUEUED_WAITING_FOR_BASELINE_NUDT_GPU2

current_production_model=TPD8+NER4+QFG2+TSS_OFF
candidate_model=TPD8+NER4+QFG2+PBDR_V2+TSS_OFF
production_replacement_authorized=false
paper_core_established=false
stability_claim_supported=false
```

独立 evaluator/launcher 已完成。NUAA-SIRST 已在 GPU0 以用户级持久服务
`sctransnet-pbdr-v2-nuaa-v1.service` 启动；结果目录为
`results/three_dataset_pbdr_v2_tss_off_seed42_v1/runs/NUAA-SIRST/pbdr_v2_tss_off/seed_42`。
训练源文件与 `experiments/PBDR_V2_PROTOCOL.md` 从此保持冻结。
