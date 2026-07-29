# SCTransNet + TPD V8-MPRS-DCH + NER V4 接入 Target Survival Supervision
## 下一阶段研究方案、代码修改与相对裁决协议

**文档版本：** TSS Integration Plan v2

**任务：** 单帧红外小目标检测，NUDT-SIRST

**父模型：** SCTransNet + TPD V8-MPRS-DCH + 五节点 NER V4 Tail-Aware

**下一模块：** Target Survival Supervision（TSS，训练期辅助监督）
**最后模块：** Query-only Frequency Gate（Query-only FG）

---

## 0. 执行结论

当前可以进入 Target Survival Supervision 阶段，但不应修改已经通过综合相对性能裁决的 TPD V8、NER V4 或 SCTransNet 主干。建议新增训练期扩展模型：

> **SCTransNet-TPD8-NER4-TSS1**
> SCTransNet with V8-MPRS-DCH, Tail-Aware NER V4 and dual-endpoint Target Survival Supervision

该扩展只在 `emb1`、`emb2` 的最终 stride-16 token endpoint 上增加两个独立 `1×1 Conv` presence head，并在训练时加入：

\[
\mathcal L_{\text{total}}
=
\mathcal L_{\text{seg}}
+
\lambda_s
\left[
\operatorname{BCEWithLogits}(Z_1,Y_{16})
+
\operatorname{BCEWithLogits}(Z_2,Y_{16})
\right]
\]

其中：

\[
Y_{16}=\operatorname{MaxPool}_{16}(Y)
\]

TSS 不进入分割前向路径，不参与正式推理，最终部署时完全移除。因此最终推理结构仍为：

```text
SCTransNet
+ TPD V8-MPRS-DCH
+ NER V4 Tail-Aware
```

TSS 是附着在既有 `emb1/emb2` endpoint 上的**训练期辅助模块**，不是新的 TPD 版本、NER 节点、evidence node 或推理分支。TPD 仍为 V8-MPRS-DCH，NER 仍为五节点 V4 Tail-Aware，`q4 → q3 → q2` 拓扑不变。

Query-only FG 的概念方案可以保留；TSS 通过工程完整性 Gate T-A 与推理图
不变 Gate T-B 后，才启动 FG 代码实现。TSS 的正式性能不设绝对放行门槛，
而按完整固定点、五 Fa budgets 和全局 Pareto 证据分为：

```text
RELATIVE_IMPROVED
PARETO_MIXED_TRADEOFF
DOMINATED
```

该分级决定结论措辞和 FG 父 checkpoint 的选择，不阻止 FG 设计。正式训练只使用 `seed=42`；本阶段不做多 seed 实验，也不支持跨 seed 稳定性声明。

建议项目状态：

```text
decision=TARGET_SURVIVAL_FORMAL800_RUNNING

authoritative_v4_result_accepted=true
tpd_v8_frozen=true
ner_v4_frozen=true
sctransnet_architecture_frozen=true

target_survival_core_available=true
target_survival_integrated_model_ready=true
target_survival_formal_training_authorized=true
target_survival_formal_training_running=true
target_survival_inference_graph_unchanged=true
target_survival_source_lock_sha256=23edf22eee2279dc59056ef4c4855ecd0d760fc3ee6856f902d44abecd9308cf

query_fg_conceptual_plan_available=true
query_fg_implementation_authorized=false
query_fg_formal_training_authorized=false
paper_full_model_established=false
universal_dominance=false
stability_claim_supported=false
```

> T-A/T-B 是代码与证据有效性的必要条件。Pd、Fa、mIoU、tiny-Pd、错误目标和五 Fa budgets 只进入相对/Pareto 裁决，不是阻止后续模块的绝对门槛。不得靠改阈值、换排序、合并两个 checkpoint 的有利点或只报告有利 checkpoint 制造提升结论。

---

## 1. V4 结果说明了什么

任务输入提供的权威结果为：

| Checkpoint | Epoch | Pd | Fa | mIoU | tiny-Pd | 错误目标 |
|---|---:|---:|---:|---:|---:|---:|
| V4 `best` | 422 | 189/189 | \(7.5720\times10^{-6}\) | 0.926418 | 39/39 | 14 |
| V4 `best_mIoU` | 489 | 188/189 | \(4.2449\times10^{-6}\) | 0.938178 | 39/39 | 4 |

同时：

```text
五预算包络 = [0, 188, 189, 189, 189]
两个 checkpoint 均进入五模型全局固定点 Pareto frontier
后四个预算为全局最优或并列最优
Fa≤1e-6 仍弱于 V1
decision=RELATIVE_MODEL_IMPROVEMENT_CONFIRMED_WITH_TRADEOFF
```

这里的既有模型级包络仅是历史摘要，不能替代两个 checkpoint 各自的五预算向量。TSS 阶段必须从封存的 V4 closed sweep 中分别回填 `best` 与 `best_mIoU` 的五个预算值；缺失值写 `TBD`，不得把两个 checkpoint 在不同预算上的最好值拼成一个 checkpoint 结果。

由此可得：

1. V4 已达到目标级召回上限：`best` 为 189/189。
2. 完全召回仍以较高 Fa 和较低 mIoU 为代价。
3. `best_mIoU` 更适合作为下一阶段父点：区域质量和 Fa 更好，只差 1 个目标。
4. 当前主要矛盾是**目标存活、像素完整性和低虚警之间的权衡**。
5. TSS 应负责稳定浅层 token 中的目标存在性；严格 `Fa≤1e-6` 的最终抑制更符合 Query-only FG 的职责。

因此 TSS 阶段不应把“增加响应”作为唯一目标，而应做到：

```text
恢复或稳定目标存活
+ 保持 V4 后四个 Fa budget 优势
+ 不破坏 mIoU
+ 不引入新的目标样背景组件
```

---

## 2. 现有代码已具备正确的 TSS 基础

### 2.1 Endpoint 与 V4 数据流天然对齐

`model/tpd_ner_v8_mprs_dch.py` 的 `explicit_embeddings()` 已显式返回：

```python
emb1, emb2, emb3, emb4, evidence1, evidence2
```

其中：

- `emb1`：32 通道；
- `emb2`：64 通道；
- 二者都位于输入的 stride-16 网格；
- `evidence1=(h11,h12,h13)`；
- `evidence2=(h21,h22)`。

V4 NER 与 Survival 的监督对象不同：

```text
NER：
读取 h11/h12/h13/h21/h22
→ q4 → q3 → q2
→ 调制 decoder skip

Survival：
监督最终 emb1/emb2
→ 约束浅层 tokenization 后目标仍然存在
```

所以 Survival 不是 NER 的重复分支，不需要再增加 decoder relay。

### 2.2 现有 head 是正确的训练期辅助头

`model/tpd_survival.py` 已实现：

```python
PairedTargetSurvivalHeads(
    emb1_channels=32,
    emb2_channels=64,
)
```

每个 head 为：

```text
1×1 Conv：C → 1 raw logit
```

总参数量：

\[
(32+1)+(64+1)=98
\]

其 architecture manifest 已声明：

```text
segmentation_path_modified = False
inference_heads_required = False
target_grid = stride_16_max_presence
```

因此 TSS 只增加 98 个训练参数，最终部署时可以完全移除。

### 2.3 Forward contract 已将两类输出分离

`model/tpd_forward_contract.py` 定义：

```python
TPDForwardOutput(
    segmentation=legacy_segmentation_output,
    emb1_endpoint=...,
    emb2_endpoint=...,
    emb1_survival_logits=...,
    emb2_survival_logits=...,
)
```

其中：

- `legacy_output()` 只返回原单图或六图分割输出；
- `evaluator_prediction()` 只返回最终全分辨率概率图；
- survival logits 不会混入六路 deep supervision。

不得把两个低分辨率 survival logits 追加到原六输出 tuple 中，否则会破坏 loss 和 evaluator 语义。

### 2.4 现有 loss 保持分割目标不变

`experiments/tpd_training_loss.py` 已实现：

\[
\mathcal L_{\mathrm{seg}}
=
\sum_{j=1}^{6}\operatorname{BCE}(P_j,Y)
\]

\[
\mathcal L_{\mathrm{surv}}
=
\sum_{i=1}^{2}
\operatorname{BCEWithLogits}(Z_i,Y_{16})
\]

\[
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{seg}}+\lambda_s\mathcal L_{\mathrm{surv}}
\]

当 \(\lambda_s=0\) 时，代码不会构造 `Y16`，也不会读取 survival logits，并保留原六项 BCE 的 Python 加法顺序。因此可建立严格的 `TSS-on` 与 `TSS-control` 对照。

---

## 3. 推荐模型：V4 Tail-Aware + Dual-Endpoint TSS

### 3.1 训练期完整流程

```text
红外图像
│
├─ SCTransNet encoder
│
├─ TPD V8-MPRS-DCH
│   ├─ embeddings_1 → h11 → h12 → h13 → emb1
│   └─ embeddings_2 → h21 → h22       → emb2
│
├─ 主分割路径
│   ├─ emb1/emb2/emb3/emb4 → SCTB
│   ├─ NER V4：q4 → q3 → q2
│   ├─ decoder
│   └─ 六路 segmentation probability maps
│
└─ 仅训练期辅助路径
    ├─ emb1 → 1×1 Conv → survival logit Z1
    └─ emb2 → 1×1 Conv → survival logit Z2

GT mask Y
├─ 原尺寸 → 六路 BCE
└─ MaxPool16 → Y16 → 两路 BCEWithLogits

Ltotal = Lseg + λs·Lsurv
```

### 3.2 推理期流程

```text
红外图像
→ SCTransNet + TPD V8 + NER V4
→ segmentation probability map
```

推理时：

- 不计算 survival logits；
- 不需要 `Y16`；
- 不增加分割推理 FLOPs；
- 导出模型不保留 98 个 survival 参数；
- 阈值、连通域、Pd/Fa/mIoU evaluator 全部不变。

### 3.3 Survival 梯度范围

只对 \(\mathcal L_{\mathrm{surv}}\) 反向时：

```text
有梯度：
target_survival heads
mtc.embeddings_1
mtc.embeddings_2
产生 x1/x2 的浅层 encoder

无直接 survival 梯度：
SCTB encoder
NER relay
decoder
deep-supervision heads
final output head
```

后半部分仍由原分割损失更新。这使 Survival 从目标证据源头施加约束，而不直接改写 NER mask 或 decoder。

---

## 4. 最小侵入式代码修改

### 4.1 冻结文件：禁止修改

```text
model/SCTransNet.py
model/tpd_clean_v8_mprs_dch.py
model/tpd_ner_v8_mprs_dch.py
model/tpd_ner_v8_mprs_dch_v2.py
model/tpd_ner_v8_mprs_dch_v3.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware.py
model/tpd_survival.py
model/tpd_forward_contract.py
experiments/tpd_training_loss.py
```

尤其不能为了取得 `emb1/emb2` 而修改 V4 `_forward_with_relay()`。

### 4.2 新增训练期扩展模型

新增：

```text
model/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py
```

推荐代码骨架：

```python
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple, Union

import torch
import torch.nn as nn

from model.SCTransNet import SCTransNet
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    DEFAULT_DC_SUPPORT_MODE,
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
    DEFAULT_TAIL_Z_THRESHOLDS,
    TailDCSupportMode,
    TPDNERV8MPRSDCHV4SCTransNet,
)
from model.tpd_survival import (
    PairedTargetSurvivalHeads,
    build_structured_survival_output,
    survival_parameter_count,
)

SURVIVAL_VERSION = "dual_post_tpd_endpoint_presence_v1"
SURVIVAL_STATE_PREFIX = "target_survival."
PRODUCTION_SURVIVAL_PARAMETERS = 98


def _zero_initialize_survival_heads(
    heads: PairedTargetSurvivalHeads,
) -> None:
    for module in heads.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class TPDNERV8MPRSDCHV4SurvivalSCTransNet(
    TPDNERV8MPRSDCHV4SCTransNet
):
    """Frozen V4 inference graph plus training-only endpoint supervision."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        variant: str,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode: Union[
            str,
            TailDCSupportMode,
        ] = DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds: Mapping[
            int,
            float,
        ] = DEFAULT_TAIL_Z_THRESHOLDS,
    ) -> None:
        super().__init__(
            parent,
            variant=variant,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
            dc_support_mode=dc_support_mode,
            tail_z_thresholds=tail_z_thresholds,
        )

        if self.mode != "train" or self.deepsuper is not True:
            raise RuntimeError(
                "formal Survival model requires mode='train' and deepsuper=True"
            )
        if not self.relay_enabled:
            raise RuntimeError("formal Survival model requires NER relay")

        emb1_channels = int(self.mtc.embeddings_1.blocks[0].channels)
        emb2_channels = int(self.mtc.embeddings_2.blocks[0].channels)
        if (emb1_channels, emb2_channels) != (32, 64):
            raise RuntimeError(
                "formal Survival model requires endpoint channels 32/64"
            )

        # Conv 构造不会改变调用者的全局 RNG。
        with torch.random.fork_rng(devices=[]):
            self.target_survival = PairedTargetSurvivalHeads(
                emb1_channels=emb1_channels,
                emb2_channels=emb2_channels,
            )
        _zero_initialize_survival_heads(self.target_survival)
        reference = next(self.parameters())
        self.target_survival.to(
            device=reference.device,
            dtype=reference.dtype,
        )

        if survival_parameter_count(self.target_survival) != 98:
            raise RuntimeError("unexpected Survival parameter count")

        # 非参数、非 buffer，只在一次 forward 内暂存引用。
        self._survival_capture_active = False
        self._captured_survival_endpoints: (
            Tuple[torch.Tensor, torch.Tensor] | None
        ) = None

    def explicit_embeddings(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ):
        values = super().explicit_embeddings(x1, x2, x3, x4)

        if self._survival_capture_active:
            if self._captured_survival_endpoints is not None:
                raise RuntimeError("Survival endpoint capture occurred twice")
            self._captured_survival_endpoints = (values[0], values[1])

        return values

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        # 评估和部署继续返回 V4 legacy output。
        if not self.training:
            return super().forward(x)

        if self._survival_capture_active:
            raise RuntimeError("re-entrant Survival forward is unsupported")

        self._survival_capture_active = True
        self._captured_survival_endpoints = None

        try:
            segmentation = super().forward(x)
            endpoints = self._captured_survival_endpoints
            if endpoints is None:
                raise RuntimeError(
                    "V4 forward did not expose emb1/emb2 endpoints"
                )
            emb1, emb2 = endpoints
            return build_structured_survival_output(
                segmentation,
                emb1,
                emb2,
                self.target_survival,
            )
        finally:
            self._captured_survival_endpoints = None
            self._survival_capture_active = False

    def architecture_manifest(self) -> Dict[str, Any]:
        manifest = dict(super().architecture_manifest())
        manifest.update(
            {
                "survival_version": SURVIVAL_VERSION,
                "survival_training_only": True,
                "survival_endpoints": ("emb1", "emb2"),
                "survival_endpoint_grid": "stride_16",
                "survival_target": "max_pool_16_binary_presence",
                "survival_head": "independent_conv1x1_raw_logits",
                "survival_parameters": 98,
                "survival_state_prefix": SURVIVAL_STATE_PREFIX,
                "survival_head_initialization": "exact_zero",
                "segmentation_path_modified": False,
                "inference_heads_required": False,
            }
        )
        return manifest
```

该方法具有四个关键性质：

1. 不使用 forward hook；
2. 不复制父类整段 `_forward_with_relay()`；
3. 不执行第二次 tokenizer 前向；
4. 不修改冻结 V4 源码。

上述类保持 V4 构造签名，但正式实验不得直接接受任意 `variant` 或
`dc_support_mode`。必须新增专用 formal builder：

```text
1. build_clean_v8_mprs_dch_model() 构造 raw Clean-V8 parent
2. 用 raw parent 构造 Survival 子类
3. 另建完整 V4 reference 供 parent checkpoint 比对
4. 固定 Full V8-MPRS-DCH、complement-tail、默认 thresholds
5. 固定 mode=train、deepsuper=True、relay_enabled=True
6. 核对 architecture manifest、父 state 544 keys、扩展 state 548 keys
7. 核对总参数 10,854,544、扩展参数 98
```

`load_parent_into_extension()` 只验证 state key、shape、dtype 和新增零值；
它不能验证 `context_gate`、support mode、tail thresholds 等非 state
配置。因此加载前还必须验证 V4 checkpoint identity，加载后必须验证
Survival extension manifest。

### 4.3 为什么两个 head 应全零初始化

默认随机 Conv 会在第一步给 endpoint 施加随机方向梯度。推荐将 classifier weight 和 bias 全部置零。

初始时：

\[
Z_1=Z_2=0
\]

且：

\[
\frac{\partial \mathcal L_{\mathrm{surv}}}
{\partial \text{endpoint}}
=
W^\top
\frac{\partial \mathcal L}{\partial Z}
=0
\]

因此：

- 第一个 optimizer step 中，Survival **辅助梯度贡献**只更新两个 head；
- 在相同 extension 结构、state、batch、RNG 与 optimizer 下，共享参数
  第一次更新与 \(\lambda_s=0\) control 完全一致；
- 第二步开始才向 endpoint 传递训练信号；
- 无需增加复杂 warmup 分支；
- 单输出二分类 head 不存在多通道全零对称性问题。

---

## 5. 父 checkpoint 与因果对照

### 5.1 唯一正式父点：V4 `best_mIoU`

推荐固定选择：

```text
V4 best_mIoU
epoch    = 489
Pd       = 188/189
Fa       = 4.2449e-6
mIoU     = 0.938178
tiny-Pd  = 39/39
错误目标 = 4

checkpoint_sha256 =
0ae6c0e034952e18333d8fa6ccd3bbf635cae5efa8017b06df5e00ccc4ed14ab

state_dict_sha256 =
2b8249ffd86866597f376c80839395a3cbdbb72a68301cd8a5a6eb36595c7e75
```

原因：

1. 区域质量和 Fa 优于 `best`；
2. 只缺少 1 个目标；
3. TSS 是尝试改善该目标与其他弱响应目标存活性的候选训练策略；一个漏检
   本身不能证明根因一定在 endpoint；
4. 从已经 189/189 的 `best` 出发，更可能只放大响应而损害 Fa/mIoU；
5. 父 checkpoint 必须在 TSS 训练前固定，禁止根据 TSS 结果切换。

### 5.2 复用严格 warm-start

父文件固定为：

```text
experiments/results/tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1/
NUDT-SIRST/tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on/
seed_42_formal800_exact_v4_tail_aware_seed42/best_miou.pth.tar
```

先调用 V4 `require_evaluator_checkpoint_payload()` 验证：

```text
checkpoint role = best_validation_miou_secondary
epoch = 489
variant = Full V8-MPRS-DCH
dc support = complement-tail
tail thresholds = {4:1.5, 3:2.0, 2:2.5}
seed = 42
split seed = 20260722
```

随后调用现有：

```python
from experiments.tpd_extension_warm_start import (
    load_parent_into_extension,
)

result = load_parent_into_extension(
    parent_checkpoint=parent_checkpoint,
    parent_model=fresh_v4_reference,
    extension_model=survival_extension,
    new_module_prefixes=("target_survival",),
    zero_init_prefixes=(
        "target_survival.heads.emb1.classifier",
        "target_survival.heads.emb2.classifier",
    ),
    parent_state_dict_path=("state_dict",),
    expected_parent_checkpoint_sha256=locked_parent_sha256,
)
```

必须验证：

- V4 checkpoint 每个 key 均存在；
- 共享 key 的 shape、dtype 完全一致；
- 扩展模型只新增 `target_survival.*`；
- 新增四个 tensor 全零；
- 父 tensor 加载后逐 tensor 等于 checkpoint；
- extension-only tensor 未被 warm-start 覆盖。
- extension 的非 state manifest 与固定 V4 配置一致。

### 5.3 Warm-start 是新阶段，不是跨版本 resume

```text
父模型权重：V4 best_mIoU
optimizer：重新创建
optimizer state：不继承
scheduler completed epoch：0
RNG：新阶段固定
selection state：空
```

TSS exact resume 只能恢复同一个 TSS run 的 epoch-boundary checkpoint。

父权重加载完成后，formal trainer 必须使用：

```python
request = InitializationRequest.extension_parent(
    result.provenance(),
    loaded_child_model_state_sha256=initial_model_state_sha256,
)
```

建立 completed epoch 为 0 的新 TSS 训练轨迹。不得把它登记为 `fresh`
V4，也不得把父 epoch 489 写成 TSS 已完成 epoch。

### 5.4 必须有 continued-training control

仅与冻结 V4 比较，无法区分收益来自 Survival 还是额外微调。因此正式运行至少包含：

| Variant | 父 checkpoint | 训练配置 | 唯一区别 |
|---|---|---|---|
| `tss_control` | V4 best_mIoU | 完全相同 | \(\lambda_s=0\) |
| `tss_on` | 同一 SHA | 完全相同 | \(\lambda_s>0\) |

两个 run 的初始共享 state hash 必须完全相同；两个 survival head 均为全零。

---

## 6. Loss、权重与训练常数

### 6.1 不修改公共 loss

新 trainer 调用：

```python
losses = compute_tpd_training_loss(
    output,
    target,
    criterion,
    survival_weight=SURVIVAL_WEIGHT,
    survival_pos_weight=SURVIVAL_POS_WEIGHT,
)

loss = losses.total
```

训练日志至少记录：

```text
train_total_loss
train_segmentation_loss
train_survival_loss
train_survival_emb1_loss
train_survival_emb2_loss
survival_weight
```

checkpoint 排序仍只使用原分割 validation 指标和 validation segmentation loss。不得使用 survival loss 选 checkpoint。

### 6.2 `survival_pos_weight` 必须由冻结训练流计算

测试中出现的 `10.116` 只证明 API 支持该值，不能直接当作数据集权威统计。

新增：

```text
experiments/compute_tpd_survival_target_statistics.py
experiments/tpd_survival_target_statistics_nudt_sirst_v1.json
```

在冻结训练 ID、mask 预处理和 crop stream 上计算：

\[
N_+=\sum Y_{16}
\]

\[
N_-=N_{\text{cell}}-N_+
\]

\[
w_+=\frac{N_-}{N_+}
\]

输出 JSON 并 source-lock：

```json
{
  "schema": "sctransnet_tpd_survival_target_statistics_v1",
  "dataset": "NUDT-SIRST",
  "split_sha256": "...",
  "data_sha256": "...",
  "used_train_ids_sha256": "9565f584a5429fd1e5f0451b2d9496877f6f887493dd4d9954b4e976989f245b",
  "train_image_count": 530,
  "image_sizes": [[256, 256]],
  "patch_size": 256,
  "mask_binarization": "float(mask)/255 > 0.5",
  "pool_kernel": 16,
  "pool_stride": 16,
  "full_image_equals_training_crop": true,
  "transform_preserves_positive_cell_count": true,
  "positive_cells": 1313,
  "negative_cells": 134367,
  "total_cells": 135680,
  "survival_pos_weight": 102.33587204874334
}
```

上述数值是当前只读重算得到的候选值，不得手工复制成权威常数。正式值由
统计脚本重新计算、写入 JSON 并绑定文件 SHA 后使用。当前数据全部为
`256×256`，正式 patch 也是 256，因此没有随机裁剪位置；翻转/转置也不改变
16×16 网格中的阳性 cell 数。

禁止：

- 根据 validation 结果调整 `pos_weight`；
- 每 batch 动态重算；
- 看到 Fa 变高后临时降低；
- 混用 full-image 与 random-crop 统计。

### 6.3 推荐主配置

由于是从已训练 V4 checkpoint 微调，建议：

```text
epochs                 = 800
patch_size             = 与 V4 相同
batch_size             = 与 V4 相同
split / normalization  = 与 V4 相同
optimizer               = Adam，重新初始化
base_lr                 = 1e-4
min_lr                  = 1e-6
warmup_epochs           = 10
AMP                     = False
survival_weight         = 0.005
survival_pos_weight     = 冻结训练流统计值
checkpoint selectors    = 原 Pd-primary 与 mIoU-primary
threshold / evaluator   = 不变
```

`tss_control` 必须使用同一 LR、epoch 数和调度器，以隔离 continued fine-tuning 效应。

这里的正式值已经在读取任何新阶段 validation 指标前，由 GPU2 上固定
seed42 训练流的 10-batch 梯度校准确定：

| 候选权重 | median(r) | p90(r) | 裁决 |
|---:|---:|---:|---|
| 0.01 | 0.075738 | 0.119980 | 超过预注册上限 |
| 0.005 | 0.037868 | 0.059991 | 通过并封存 |

因此 `tss_on` 的唯一正式 `survival_weight` 为 `0.005`；`0.01` 只保留为
校准中被否决的候选值，不得启动正式训练。

### 6.4 训练前梯度安全门

在固定训练 batch 上记录：

\[
r=
\frac{
\left\|
\lambda_s\nabla_{\theta_{\text{shared}}}
\mathcal L_{\text{surv}}
\right\|_2
}{
\left\|
\nabla_{\theta_{\text{shared}}}
\mathcal L_{\text{seg}}
\right\|_2+\epsilon
}
\]

在一次性 calibration clone 上，让 head 完成至少一次非零更新后要求：

```text
median(r) ∈ [0.005, 0.05]
p90(r) ≤ 0.10
```

若 `0.01` 超过上限，只允许在读取 validation 指标前启用预注册 fallback：

```text
survival_weight = 0.005
```

不得正式训练后继续搜索多个 \(\lambda_s\)。

本轮已经执行该回退并通过上述校准，后续 source lock、run identity 和
checkpoint identity 必须统一绑定 `tss_on=0.005`。

calibration clone 完成后必须丢弃；正式 control/TSS 模型从同一父 checkpoint
重新构建和 warm-start，确保两个 head 在正式 epoch 1 前仍为全零。

---

## 7. Exact trainer 修改

新增：

```text
experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact.py
```

不要修改 V4 trainer。

### 7.1 专用最小差异训练循环

现有 V4 exact kernel 的训练函数明确要求六元素 tuple，不能只替换一个
三参数 loss 函数。TSS trainer 必须拥有专用的最小差异 loop，同时继续复用
既有 data、validation、selector、exact runner 与 epoch commit 语义。

```python
from experiments.tpd_training_loss import compute_tpd_training_loss

def compute_stage_loss(
    outputs,
    target: torch.Tensor,
    criterion: nn.Module,
    *,
    survival_weight: float,
    survival_pos_weight: float,
):
    return compute_tpd_training_loss(
        outputs,
        target,
        criterion,
        survival_weight=survival_weight,
        survival_pos_weight=survival_pos_weight,
    )
```

训练 loop 将：

```python
loss = six_output_bce_loss(outputs, target, criterion)
```

替换为：

```python
losses = compute_stage_loss(
    outputs,
    target,
    criterion,
    survival_weight=args.survival_weight,
    survival_pos_weight=args.survival_pos_weight,
)
loss = losses.total
```

其余 backward、optimizer step、validation 和 selector 顺序保持不变。

每个 epoch 分别累计：

```text
train_total_loss
train_segmentation_loss
train_survival_loss
train_survival_emb1_loss
train_survival_emb2_loss
```

当 `survival_weight=0` 时，`survival_terms=()`；control 日志中的两个
per-head loss 固定写 `null`，不得为了填日志额外构造 `Y16` 或读取 logits。

### 7.2 TSS 自有身份与 checkpoint schema

不能沿用 V4 中写死的 `fresh_training=true`、
`warm_start_applied=false` 校验。新增：

```text
TSS entry schema
TSS run identity schema
TSS checkpoint schema
TSS checkpoint identity schema
TSS source-lock schema
require_tss_run_identity()
TSS EvaluatorCheckpointAdapter
```

正式初始化顺序：

```text
验证父 checkpoint schema/role/epoch/SHA/manifest
→ load_parent_into_extension()
→ 计算 loaded child state SHA
→ InitializationRequest.extension_parent(...)
→ 新建 Adam/scaler/selection state
→ 从 TSS epoch 1 开始
```

`ExactRunSpec.loss` 必须绑定 segmentation BCE、Survival BCEWithLogits、
`survival_weight`、`survival_pos_weight`、target-statistics SHA、
max-pool16 规则以及 control/on 身份。任一字段变化时不得恢复旧 TSS journal。

### 7.3 Evaluation 路径

```python
model.eval()
with torch.no_grad():
    output = model(image)
    prediction = evaluator_prediction(output)
```

扩展模型在 `eval()` 下直接返回 V4 legacy output，不计算 head。

### 7.4 Run identity 新字段

```text
parent_checkpoint_path
parent_checkpoint_sha256
parent_checkpoint_role=best_miou
parent_checkpoint_epoch=489
parent_checkpoint_state_dict_sha256
warm_start_applied=true
warm_start_schema
initialization_mode=extension_parent_warm_start
survival_version
survival_state_prefix
survival_parameters=98
survival_head_initialization=exact_zero
survival_training_only=true
survival_weight
survival_pos_weight
survival_target_statistics_sha256
segmentation_objective_unchanged=true
inference_heads_required=false
continued_training_control
```

`initial_model_state_sha256` 在父 state 加载完成且 head 全零后计算。

### 7.5 Exact-resume

必须恢复：

- 共享模型参数；
- survival head 参数；
- optimizer；
- scheduler completed epoch；
- Python / NumPy / Torch CPU / Torch CUDA RNG；
- DataLoader generator；
- selection state；
- loss 常数；
- parent checkpoint provenance；
- statistics SHA；
- source lock；
- control/on variant identity。

连续训练与中断续训必须逐 tensor 相等。

---

## 8. 部署导出

新增：

```text
experiments/export_tpd_ner_v4_survival_to_inference.py
```

```python
SURVIVAL_PREFIX = "target_survival."

def strip_survival_state_dict(state_dict):
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(SURVIVAL_PREFIX)
    }

training_state = checkpoint["state_dict"]
inference_state = strip_survival_state_dict(training_state)

inference_model = build_frozen_v4_model()
inference_model.load_state_dict(inference_state, strict=True)
inference_model.eval()
```

固定 batch 验证：

```python
training_model.eval()
inference_model.eval()

with torch.no_grad():
    a = evaluator_prediction(training_model(x))
    b = evaluator_prediction(inference_model(x))

assert torch.equal(a, b)
```

参数量应为：

```text
V4 推理模型：10,854,446
TSS 训练模型：10,854,544
部署模型：10,854,446
训练期增量：98
推理期增量：0
```

---

## 9. 必须新增的测试

```text
tests/test_tpd_ner_v8_mprs_dch_v4_survival.py
tests/test_tpd_ner_v8_mprs_dch_v4_survival_warm_start.py
tests/test_tpd_ner_v8_mprs_dch_v4_survival_loss_integration.py
tests/test_tpd_ner_v8_mprs_dch_v4_survival_export.py
tests/test_train_tpd_ner_v8_mprs_dch_v4_survival_exact.py
tests/test_tpd_ner_v8_mprs_dch_v4_survival_resume.py
tests/test_tpd_ner_v8_mprs_dch_v4_survival_source_lock.py
tests/test_tpd_ner_v8_mprs_dch_v4_survival_gpu_smoke.py
```

### 9.1 架构与 state

```text
新增参数恰好 98
新增 state key 恰好 4
新增 key 全部位于 target_survival.*
父 V4 key、shape、dtype 不变
无新增 persistent buffer
普通模式与 python -O 一致
```

### 9.2 零点等价

父 state 加载、head 全零时：

```text
V4 eval output
== TSS extension eval output
== 移除 head 后导出模型 output
```

要求逐元素 `torch.equal`。

### 9.3 `survival_weight=0` 精确对照

相同 batch、state、optimizer：

```text
segmentation loss 相等
共享参数梯度逐 tensor 相等
第一次 Adam 更新逐 tensor 相等
survival head grad 为 None
```

### 9.4 正权重梯度路由

零初始化要求把测试拆为两个阶段。

阶段 A，初始化后第一次仅对 survival loss backward：

```text
head grad                > 0
embeddings_1 grad        = None/0
embeddings_2 grad        = None/0
浅层 encoder grad         = None/0
```

只更新 head，确认 classifier weight 已非零，然后清空梯度并重新
forward。阶段 B 再仅对 survival loss backward：

```text
head grad                > 0
embeddings_1 grad        > 0
embeddings_2 grad        > 0
浅层 encoder grad         > 0

tpd_ner grad             = None/0
SCTB encoder grad         = None/0
decoder grad             = None/0
segmentation heads grad   = None/0
```

阶段 B 使用一次性测试/校准副本；正式训练模型不得继承这次 head 更新。

### 9.5 单次前向

call counter 要求：

```text
embeddings_1.forward_with_evidence：1 次
embeddings_2.forward_with_evidence：1 次
```

### 9.6 Exact resume

```text
N epoch 连续训练
==
K epoch + exact resume + N-K epoch
```

比较 model、optimizer、scheduler、RNG、selection state、loss log 和 checkpoint payload。

---

## 10. 实验矩阵

### Phase 0：父模型只读诊断

对 V4 `best_mIoU` 记录：

- 漏检目标 ID；
- 对应 `Y16` cell；
- `emb1/emb2` 正目标与背景 cell 特征；
- 最终概率峰值和阈值 margin；
- NER stage4/3/2 mask；
- 连通域结构。

### Phase 1：CPU 与 RTX 5090 smoke

| Run | 设备 | 目的 |
|---|---|---|
| CPU 2-step | CPU | structured output、loss、backward、reload |
| TSS control | GPU2 | \(\lambda_s=0\)，零点等价 |
| TSS on | GPU3 | \(\lambda_s>0\)，梯度与显存 |
| export smoke | GPU2/3 | 去 head 后逐元素等价 |

### Phase 2：seed42 paired exact800

| Run | GPU | 父 checkpoint | \(\lambda_s\) |
|---|---:|---|---:|
| `formal800_control` | GPU2 | V4 best_mIoU | 0 |
| `formal800_tss` | GPU3 | 同一 SHA | 0.005 |

截至 2026-07-29 01:23 CST，两条 seed42 exact800 轨迹已经启动：

```text
GPU2 / sctransnet-tss-control-gpu2.service / tss_control
GPU3 / sctransnet-tss-on-gpu3.service      / tss_on
```

二者都以 `mode=parent_warm_start, completed=0, next=1` 建立 child 轨迹；
首批 epoch 已成功提交，后续由同版本 exact-resume journal 保护。

两者：

- 同 split；
- 同训练 seed；
- 同初始共享 state；
- 同 optimizer/LR；
- 仅 `survival_weight` 不同；
- 保存 `best`、`best_mIoU`、`last`；
- 对前两者完成固定点和 closed sweep。

epoch 100 是同一 exact800 轨迹中的只读进度点，不另建 screen run，也不从
父 checkpoint 重新开始。仅在以下工程失败时停止：

```text
非有限 loss/gradient
输出完全塌缩
checkpoint 或 exact-resume 无效
模型/数据/设备身份不匹配
显存不足且无法完成一个正式 batch
```

Pd、Fa、mIoU、tiny-Pd 的早期波动不作为终止条件。

### Phase 3：正式比较与相对裁决

每个 run 分别使用自己的 selector 取得：

```text
best
best_mIoU
```

四个 checkpoint 均独立报告：

```text
Pd
Fa
mIoU
tiny-Pd
错误目标数
五个 Fa budget 下的 Pd/实际 Fa/mIoU/threshold
```

模型级 budget envelope 只能作为附加摘要，并必须记录每个预算来自哪个
checkpoint 与 threshold。正式实验只使用 seed42；结论固定保留
`stability_claim_supported=false`。

---

## 11. TSS 阶段工程条件与相对性能裁决

### Gate T-A：工程完整性

```text
模型、loss、warm-start、export 测试通过
普通模式和 python -O 通过
RTX 5090 smoke 通过
exact resume 通过
source lock 通过
父 checkpoint SHA 通过
statistics SHA 通过
全部 run/checkpoint/sweep 有效
```

### Gate T-B：推理图不变

```text
inference_parameter_delta=0
inference state keys 与 V4 完全相同
导出模型与训练模型 eval output 逐元素相同
阈值和 evaluator 未修改
```

T-A 与 T-B 是正式运行和后续集成必须满足的工程条件。以下性能项全部改为
裁决证据，不再作为绝对放行门槛。

### Evidence T-C：固定点完整比较

分别比较 V4、continued-training control 与 TSS 的各自 `best` 和
`best_mIoU`。每个固定点必须同时包含 Pd、Fa、mIoU、tiny-Pd 和错误目标数。
构建所有固定点的全局 Pareto frontier，不要求 TSS 在每个单项上超过 V4。

### Evidence T-D：五预算完整比较

分别报告两个 checkpoint 的五预算向量；模型 envelope 仅作附加摘要。
允许某些预算改善、另一些预算退化。不得把“所有预算逐项不退化”误写成
Pareto 要求。

### Evidence T-E：TSS 相对贡献

相对同父点、同轨迹配置的 `formal800_control`：

- 比较四个正式 checkpoint 与全部预算点；
- head survival margin、AUROC/AUPRC 和 head collapse 只作训练诊断；
- head margin 提高不能单独证明共享表示改善；
- 最终模型贡献以分割输出的 Pd、Fa、mIoU、tiny-Pd、错误目标和预算结果为准。

### 相对裁决与 FG 父点选择

```text
engineering_valid = T-A and T-B
query_fg_implementation_authorized = engineering_valid

if TSS adds a strict globally non-dominated improvement:
    relative_status=RELATIVE_IMPROVED
elif TSS contributes a non-dominated trade-off point:
    relative_status=PARETO_MIXED_TRADEOFF
else:
    relative_status=DOMINATED

stability_claim_supported=false
```

- `RELATIVE_IMPROVED`：优先使用相应 TSS checkpoint 作为 FG 父点；
- `PARETO_MIXED_TRADEOFF`：保留 TSS 与 V4/control 的非支配父点，按目标
  工作区间选择；
- `DOMINATED`：仍允许实现 FG，但优先使用 V4/control frontier 父点；
  若要检查组合交互，再比较 `V4/control+FG` 与 `TSS+FG`。

---

## 12. 可选训练诊断

以下诊断用于解释结果和定位修改方向，不替代最终 Pd、Fa、mIoU、
tiny-Pd、错误目标与五预算比较，也不阻止 Query-only FG 的代码设计。

### 12.1 Endpoint 分离度

对 `emb1`、`emb2` 记录：

\[
\Delta_i
=
\mathbb E[Z_i\mid Y_{16}=1]
-
\mathbb E[Z_i\mid Y_{16}=0]
\]

以及：

- AUROC；
- AUPRC；
- 正目标 cell recall；
- 固定 background-cell FPR 下 recall；
- hard-negative logit 分布。

### 12.2 漏检目标恢复

针对 V4 `best_mIoU` 漏掉的目标记录：

```text
emb1 logit
emb2 logit
final probability peak
预测面积
连通域数
质心距离
阈值 margin
```

若只提高 head logit 而最终目标未恢复，说明 head 吸收了辅助任务，共享表示收益不足。

### 12.3 参数漂移

按组记录：

\[
D_g=
\frac{\|\theta_g-\theta_g^{(0)}\|_2}
{\|\theta_g^{(0)}\|_2+\epsilon}
\]

分组：

```text
shallow encoder
TPD embeddings_1
TPD embeddings_2
SCTB
NER V4
decoder
output heads
```

### 12.4 NER 行为漂移

记录：

- stage4/3/2 mask mean、std、p95、p99；
- tail support coverage；
- 三个 DC offset；
- q4/q3/q2 RMS；
- skip modulation factor。

### 12.5 虚警类型

拆分：

```text
目标内部碎裂孤岛
目标附着 halo
独立背景组件
merge
split
```

---

## 13. 结果分支与修改方向

### 情形 1：TSS 与 control 基本相同

head loss 下降但 Pd/Fa/mIoU 无差异，说明当前证据只支持 head 学会了辅助
任务，没有显示最终分割改善。记录为 `DOMINATED` 或“无可见增益”，但不阻止
FG 实现；FG 父点优先选择更好的 V4/control checkpoint。

### 情形 2：Pd 提升但 Fa/mIoU 下降

这可能说明 16×16 presence 监督过强。第一预注册回退：

```text
survival_weight: λ → λ/2
```

必须从同一父 checkpoint 重新开始，不能续训失败 run。若仍无改善，保留完整
结果并进入 FG 组合比较，不把单模块绝对门槛当作最终模型的终止条件。

### 情形 3：emb1 正向、emb2 负向

不得事后直接删除一头。先完成只读梯度与 endpoint 诊断；若证据一致，再预注册 `emb1-only` 或 `emb2-only` 新模型。

### 情形 4：control 同样改善

收益主要来自 continued fine-tuning。可以保留更好 control checkpoint，但不能归因于 TSS。

### 情形 5：严格低 Fa 仍不超过 V1

如其他工作区间存在非支配改善，记录为 `PARETO_MIXED_TRADEOFF`；无论该严格
预算是否超过 V1，只要 T-A/T-B 成立，都允许继续 FG 代码设计。最终完整模型
是否保留 TSS，由 FG 组合后的综合性能决定。

---

## 14. 向 Query-only FG 的交接

T-A/T-B 成立并完成 seed42 相对裁决后：

1. 验证 TSS、control 与 V4 的非支配 checkpoint；
2. 若选择 TSS 父点，从 checkpoint 移除 `target_survival.*`；
3. 生成 V4-compatible inference parent 并固定 checkpoint SHA；
4. Query-only FG 只接 SCTB Query；
5. K/V、CFN、TPD、NER、decoder、evaluator 不变；
6. FG 与无 FG 的同一父点配对；
7. TSS 为 `DOMINATED` 时，优先使用 V4/control 父点，同时可保留
   `TSS+FG` 作为组合交互对照。

```text
训练期最终模型：
SCTransNet + TPD + NER + Survival + Query-only FG

推理期最终模型：
SCTransNet + TPD + NER + Query-only FG
```

---

## 15. 文件清单

### 新增

```text
model/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py

experiments/compute_tpd_survival_target_statistics.py
experiments/tpd_survival_target_statistics_nudt_sirst_v1.json
experiments/calibrate_tpd_survival_gradient_ratio.py
experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact.py
experiments/evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_pd_fa.py
experiments/export_tpd_ner_v4_survival_to_inference.py
experiments/launch_tpd_ner_v4_survival_formal800.sh
experiments/finalize_tpd_ner_v4_survival.py
experiments/TPD_NER_V8_MPRS_DCH_V4_SURVIVAL_PROTOCOL.md
experiments/freeze_tpd_ner_v4_survival_exact_source_lock.py
experiments/tpd_ner_v4_survival_exact_source_lock.json

tests/test_tpd_ner_v8_mprs_dch_v4_survival.py
tests/test_tpd_ner_v8_mprs_dch_v4_survival_warm_start.py
tests/test_tpd_ner_v8_mprs_dch_v4_survival_loss_integration.py
tests/test_tpd_ner_v8_mprs_dch_v4_survival_export.py
tests/test_train_tpd_ner_v8_mprs_dch_v4_survival_exact.py
tests/test_tpd_ner_v8_mprs_dch_v4_survival_resume.py
tests/test_tpd_ner_v8_mprs_dch_v4_survival_source_lock.py
tests/test_tpd_ner_v8_mprs_dch_v4_survival_gpu_smoke.py
```

### 复用但不修改

```text
model/tpd_survival.py
model/tpd_forward_contract.py
experiments/tpd_training_loss.py
experiments/tpd_extension_warm_start.py
```

### 封存不修改

```text
model/SCTransNet.py
model/tpd_clean_v8_mprs_dch.py
model/tpd_ner_v8_mprs_dch*.py
experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_exact.py
当前 V4 evaluator、checkpoint 与正式结果
```

---

## 16. 推荐执行顺序

```text
1. 固定 V4 best_mIoU epoch489、checkpoint SHA 与 state SHA
2. 实现 Survival 扩展模型和专用 formal builder
3. 固定 mode/deepsuper/variant/support/threshold/manifest
4. 实现单次前向 endpoint capture、零初始化和 device/dtype 对齐
5. 验证 V4 checkpoint identity 后执行 extension warm-start
6. 计算并固定 survival_pos_weight statistics JSON
7. 实现 TSS 自有 trainer loop、identity、checkpoint adapter 和 exact-resume
8. 完成两阶段梯度、λ=0、导出、普通模式与 python -O 测试
9. 完成 source-lock freezer 与验证
10. 完成 GPU2/3 smoke
11. T-A/T-B 成立后在 GPU2/3 并行启动 seed42 paired exact800
12. epoch100 只读检查同一轨迹，正常时继续，不重新训练
13. 分别评估两个 run 各自的 best/best_mIoU 与五预算
14. 输出 RELATIVE_IMPROVED / PARETO_MIXED_TRADEOFF / DOMINATED
15. 按全局 Pareto 选择并导出 FG 父点
16. 继续 Query-only FG 设计与组合比较
```

---

## 17. 最终研究判断

Target Survival Supervision 是当前最合理的第三模块。其价值不在于增加容量，而在于针对 V4 剩余矛盾：

> `best_mIoU` 已有较好的 Fa 和 mIoU，但仍漏检一个目标；`best` 虽检出全部
> 目标，却付出较高 Fa 和较低 mIoU。该现象支持尝试目标存活辅助监督，但不
> 预先断言漏检根因一定是 endpoint 信息丢失。

双 endpoint TSS 可以在不改变推理图、不改 NER、不改 decoder 的条件下，对浅层 token 的目标存在性施加训练期约束。现有 `tpd_survival.py`、`TPDForwardOutput`、`compute_tpd_training_loss()` 和 strict extension warm-start 已提供正确底层 contract。

本阶段应严格坚持：

```text
单次前向
双 endpoint
训练期辅助
零初始化
严格 warm-start
λ=0 paired control
原 evaluator 不变
部署时完全移除
```

seed42 paired control 能把 TSS 相对贡献与额外训练区分开。最终判断以每个
模型自己的 `best`、`best_mIoU`、完整 Pd/Fa/mIoU/tiny-Pd/错误目标和五预算
为准。TSS 单模块结果决定 FG 父点，不改变 TPD V8、五节点 NER V4 和
Query-only FG 的完整模型主线，也不以绝对门槛阻止后续模型代码设计。

---

## 18. 代码审查依据

主要依据：

1. `model/tpd_ner_v8_mprs_dch_v4_tail_aware.py`
2. `model/tpd_ner_v8_mprs_dch.py`
3. `model/tpd_survival.py`
4. `model/tpd_forward_contract.py`
5. `experiments/tpd_training_loss.py`
6. `experiments/tpd_extension_warm_start.py`
7. `tests/test_tpd_survival.py`
8. `tests/test_tpd_training_loss.py`
9. `experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_exact.py`

仓库：

- https://github.com/Arialliy/SCTransNet_main
