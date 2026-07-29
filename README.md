# TPD-SCTransNet：面向红外小目标的目标保真下采样实验

本仓库在 [SCTransNet](https://github.com/xdFai/SCTransNet) 基线上研究浅层
patch embedding 的目标保真下采样（Target-Preserving Downsampling, TPD）。
核心改动只替换 `mtc.embeddings_1` 和 `mtc.embeddings_2`；编码器、SCTB、
解码器、损失函数和输出接口保持不变，以便进行受控比较。

> 当前结论来自 **NUDT-SIRST 官方训练集的内部验证划分**，未访问官方测试集。
> TPD-Clean-v6 与 V7-DCH 正式实验使用两个随机种子；V8-MPRS-DCH、NER、
> TSS 与 QFG-V2-CROA 筛选使用 seed 42。所有结果都不足以形成跨数据集稳定性结论。

## 最新状态：TSS + QFG-V2-CROA

在 NER V4 Tail-Aware 上，本仓库完成了 TSS（训练期 Target Survival
Supervision）与 QFG-V2-CROA（Query-only Frequency Gate）的 2×2 因子实验：

| Arm | TSS | QFG | 训练变体 |
|---|---|---|---|
| A | off | off | `tss_control` |
| B | on | off | `tss_on` |
| C | off | on | `qfg_only` |
| D | on | on | `tss_qfg` |

QFG 只调制 NER query，不改 TPD tokenizer、SCTB、K/V、CFN 或 decoder 主路径；
TSS 仍仅用于训练，导出推理权重中不含 survival heads。C/D 均从相应 TSS
父 checkpoint warm start，完成 seed 42、NUDT-SIRST 530/133 内部划分、
FP32、800-epoch 正式训练和 closed Pd–Fa sweeps。

### QFG 正式训练结果（固定阈值 0.5）

| 变体/Checkpoint | Epoch | Pd ↑ | Fa ↓ | mIoU ↑ | tiny-Pd | 错误目标 |
|---|---:|---:|---:|---:|---:|---:|
| QFG-only Pd-best | 29 | 188/189 | 4.0155e-6 | 0.930844 | 39/39 | **4** |
| QFG-only mIoU-best | 3 | 188/189 | 4.8186e-6 | **0.939934** | 39/39 | 6 |
| TSS+QFG Pd-best | 136 | 188/189 | **3.6713e-6** | 0.931693 | 39/39 | 6 |
| TSS+QFG mIoU-best | 3 | 188/189 | 4.1302e-6 | 0.937018 | 39/39 | 5 |

完整联合非支配分析将 QFG-only（C）判为 **`DOMINATED`**；TSS+QFG（D）
拥有非孤立、独占的联合前沿区间，判为 **`PARETO_MIXED_TRADEOFF`**。
最终工程选择为 **`SELECT_D_TSS_QFG`**。这表示 D 在当前单 seed 内部验证集上
提供有价值的 Pareto 权衡，并不表示全面优于所有历史模型。

### 推理默认工作点

部署使用 D 的 `best_miou.pth.tar`（epoch 3）。旧版选择曾使用
Fa budget `1e-4` 下的 threshold `0.0001589997`，虽然达到 Pd `189/189`，
但带来 Fa `7.0328e-5` 和约 `2.46` 个错误目标/图。闭环 v2 因此将权威默认值
修订为固定 threshold **`0.5`**：

| Pd | tiny-Pd | Fa | mIoU | 错误目标/图 |
|---:|---:|---:|---:|---:|
| 188/189 | 39/39 | 4.1302e-6 | 0.937018 | 0.0376 |

导出推理模型含 10,870,130 个参数、564 个 state keys，保留 QFG 权重且确认
survival state 不存在。`paper_core_established=false`，
`stability_claim_supported=false`；官方测试集未访问。

主要实现与说明：

- [`model/tpd_frequency_gate_v2_croa.py`](model/tpd_frequency_gate_v2_croa.py)
- [`model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py`](model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py)
- [`experiments/train_tpd_ner_v4_qfg_v2_croa_exact.py`](experiments/train_tpd_ner_v4_qfg_v2_croa_exact.py)
- [`experiments/compare_tss_qfg_v2_croa_factorial.py`](experiments/compare_tss_qfg_v2_croa_factorial.py)
- [`SCTransNet_TSS混合结果复盘与QFG_V2最终模型集成方案.md`](SCTransNet_TSS混合结果复盘与QFG_V2最终模型集成方案.md)

## 已完成阶段：NER V4 与 Target Survival

NER V4 Tail-Aware 保留 V8-MPRS-DCH tokenizer、五个 evidence nodes、
`q4 → q3 → q2` 窄中继和原 SCTransNet decoder，只调整 stage-wise
DC offset 的空间作用域。零训练 counterfactual 从 `legacy_global`、
`direct_tail` 和 `complement_tail` 中选择了 target-protective
`complement_tail`：

\[
M_s=\frac{1}{\pi}\arctan\left[\pi\left(Z_s+d_s(1-P_s)\right)\right],
\quad s\in\{2,3\}.
\]

### NER V4 正式结果

V4 已完成 seed 42、NUDT-SIRST 530/133 内部划分、FP32、800-epoch 正式训练。

| Checkpoint | Epoch | Pd ↑ | Fa ↓ | mIoU ↑ | tiny-Pd | 错误目标 |
|---|---:|---:|---:|---:|---:|---:|
| Pd-primary `best` | 422 | **189/189** | 7.5720e-6 | 0.926418 | 39/39 | 14 |
| mIoU-secondary `best_miou` | 489 | 188/189 | **4.2449e-6** | **0.938178** | 39/39 | **4** |

V4 的五预算包络为 `[0, 188, 189, 189, 189]`；两个 checkpoint 都进入
五模型全局固定点 Pareto frontier，后四个 Fa budget 为全局最优或并列最优，
但严格 `Fa≤1e-6` 区域仍弱于 V1。正式裁决为
**`RELATIVE_MODEL_IMPROVEMENT_CONFIRMED_WITH_TRADEOFF`**：确认相对改进，
但不支持统一支配、跨种子稳定性或跨数据集结论。

### Target Survival Supervision（TSS）

TSS 在 V4 的 `emb1`、`emb2` stride-16 endpoint 上增加两个训练期
`1×1 Conv` presence head，以 max-pooled target-presence 提供辅助监督。
两个 head 共增加 98 个训练参数，不进入分割前向路径，并可从正式推理模型中完全移除。

TSS-on（survival loss 权重 `0.005`）与等结构 TSS-control（权重 `0`）均从
V4 `best_miou` checkpoint warm start，并已完成 seed-42、800-epoch 正式训练。
固定阈值 `0.5` 的内部验证结果如下：

| 变体/Checkpoint | Epoch | Pd ↑ | Fa ↓ | mIoU ↑ | tiny-Pd | 错误目标 |
|---|---:|---:|---:|---:|---:|---:|
| TSS-on Pd/mIoU-best | 3 | 188/189 | 4.1302e-6 | 0.936870 | 39/39 | 5 |
| TSS-control Pd-best | 37 | 188/189 | **4.0155e-6** | 0.934370 | 39/39 | **4** |
| TSS-control mIoU-best | 3 | 188/189 | 4.8186e-6 | **0.940091** | 39/39 | 6 |

这些固定阈值结果没有显示 TSS-on 对 control 的统一优势：TSS-on 相比 control
Pd-best 有更高 mIoU，但 Fa 和错误目标略高；control 的 mIoU-best 又取得更高
mIoU。后续完整因子分析将 TSS 与 QFG 的单独及联合贡献一并纳入最终选择。

最新实现与方案：

- [`model/tpd_ner_v8_mprs_dch_v4_tail_aware.py`](model/tpd_ner_v8_mprs_dch_v4_tail_aware.py)
- [`model/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py`](model/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py)
- [`experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact.py`](experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact.py)
- [`SCTransNet_TPD8_NER4_Target_Survival_集成方案与代码修改计划.md`](SCTransNet_TPD8_NER4_Target_Survival_集成方案与代码修改计划.md)

## 历史筛选：V8-MPRS-DCH + NER V1/V2/V3

V8-MPRS-DCH（Mass-Preserving Phase-Resolved Saliency with Deferred Context
Headroom）保持 Keep–Context–Saliency 三源主线，只将 V7 的标量 Saliency
替换为总量保持的四相位分辨表示：

```text
C0 = mean(Zp)
S0 = max(Zp) - C0
Sp = S0 + (Zp - C0) / 3
```

它不增加参数或第四语义分支，只替换 `mtc.embeddings_1/2`。在此基础上，
仓库依次完成五节点 NER 的 V1、V2 和 V3 seed-42 正式筛选；三版最终裁决均为
**`RETURN_TO_MODEL_OPTIMIZATION`**，aggregate full-model gate 均未通过。

### V1/V2/V3 固定阈值对比

以下为 Pd-primary checkpoint 在 NUDT-SIRST 530/133 内部验证集、
threshold `0.5` 下的结果：

| 模型 | Pd ↑ | Fa ↓ | mIoU ↑ | 错误目标/图 ↓ |
|---|---:|---:|---:|---:|
| SCTransNet reference | 188/189 | 1.4226e-5 | **0.919382** | 0.1278 |
| V8 + NER V1 relay-off | **189/189** | 4.8989e-5 | 0.790668 | 0.4887 |
| V8 + NER V1 relay-on | 188/189 | 1.8815e-5 | 0.909049 | 0.2782 |
| V8 + NER V2 relay-on | 188/189 | 5.6217e-6 | 0.915323 | 0.0752 |
| V8 + NER V3 relay-on | 188/189 | **4.7038e-6** | 0.903948 | **0.0451** |

V3 的 mIoU-secondary checkpoint 为 `187/189`、Fa `5.0480e-6`、
mIoU `0.935640`，并保持 tiny-Pd `39/39`。

### 正式筛选结论

- V1 的 relay-on/off 两个角色均未通过绝对门槛；
- V2 相比 V1 relay-on 明显降低 Pd-primary Fa，但相对规定的 V1 relay-off
  控制仍未通过 gate；
- V3 相对 V2 的结构前驱 gate 通过：Pd-primary 在 5/5 Fa budgets
  非劣且 1/5 更优，mIoU-secondary 在 5/5 非劣且 4/5 更优；
- V3 相对规定的 V1 relay-off 控制仍失败：Pd-primary 仅 2/5 budgets
  非劣，mIoU-secondary 虽 4/5 非劣但 0/5 严格更优；
- V3 两个 checkpoint 均保持 tiny target `39/39`，没有 tiny-Pd 回退；
- DC-offset knockout 仅为同 checkpoint 反事实诊断，不改变正式六组件裁决。

因此当前不能声称 NER 或 V8 已成为有效主线，也不能建立 paper core 或稳定性结论。
最新代码与冻结协议：

- [`model/tpd_clean_v8_mprs_dch.py`](model/tpd_clean_v8_mprs_dch.py)
- [`model/tpd_ner_v8_mprs_dch_v3.py`](model/tpd_ner_v8_mprs_dch_v3.py)
- [`experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md`](experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md)
- [`experiments/TPD_NER_V8_MPRS_DCH_V3_PROTOCOL.md`](experiments/TPD_NER_V8_MPRS_DCH_V3_PROTOCOL.md)
- [`SCTransNet_TPD_V8_MPRS_DCH_不改主线失败分析与代码修改方案.md`](SCTransNet_TPD_V8_MPRS_DCH_不改主线失败分析与代码修改方案.md)

## 已完成裁决：TPD-Clean V7-DCH

V6 的正式实验表明目标保真主线存在局部潜力，但固定阈值质量、跨种子稳定性和
相对等容量对照优势均未通过预注册门槛。后续冻结诊断发现：V6 Full 与 Capacity
在零 Saliency scale 处输出相同，但第一次优化的梯度并不相同。V7-DCH
（Deferred Context Headroom）因此将零点加强为输出与一阶优化同时等价的锚点，
只延迟 Context 对 Saliency 学习轨迹的影响，不改变 K/C/S 三源、参数量、主干网络、
损失函数、数据划分或评估协议。

V7-DCH 使用如下 Context headroom：

```text
a = tanh(saliency_scale)
H = 1 + |a|(1-|a|)V
output = Keep + Saliency_aligned × a × H
```

其中 `V` 是空间零均值、有界的 Context modulation。等容量对照固定 `H=1`。
Full 与 Capacity 的参数量均为 10,843,155。

### V7-DCH 正式训练结果

四组 fresh、FP32、800-epoch 任务均已完成，使用 seed `42/3407` 和同一
530/133 内部划分：

| Seed | 变体 | 检查点 | Epoch | Pd | tiny-Pd | Fa ↓ | mIoU ↑ |
|---:|---|---|---:|---:|---:|---:|---:|
| 42 | V7-DCH Full | Pd-primary | 414 | 187/189 | 39/39 | 9.1782e-7 | 0.929930 |
| 42 | V7-DCH Full | mIoU-primary | 694 | 187/189 | 39/39 | 2.6387e-6 | 0.939580 |
| 42 | Capacity | Pd-primary | 181 | 188/189 | 39/39 | 6.8722e-5 | 0.805532 |
| 42 | Capacity | mIoU-primary | 416 | 186/189 | 38/39 | 5.7364e-7 | 0.939605 |
| 3407 | V7-DCH Full | Pd-primary | 212 | 187/189 | 39/39 | 1.5259e-5 | 0.849535 |
| 3407 | V7-DCH Full | mIoU-primary | 557 | 183/189 | 38/39 | 3.3271e-6 | 0.923745 |
| 3407 | Capacity | Pd-primary | 522 | 186/189 | 39/39 | 1.0325e-6 | 0.928052 |
| 3407 | Capacity | mIoU-primary | 518 | 184/189 | 38/39 | 6.8837e-7 | 0.929850 |

V7 的 8 份 closed Pd–Fa sweeps、Gate A–E 汇总、Mechanism Audit M 和最终
decision 已完成并封存。正式裁决为 **`ENGINEERING_GATE_FAIL`**：
NER stage 未获授权、fragmentation mechanism 未获支持、主线不变，
`paper_core_established=false`，`stability_claim_supported=false`。
Mechanism Audit M 与工程 gate 相互独立，机制诊断不能覆盖性能裁决。

V7-DCH 实现与冻结协议：

- [`model/tpd_clean_v7_dch.py`](model/tpd_clean_v7_dch.py)
- [`experiments/TPD_CLEAN_V7_DCH_PROTOCOL.md`](experiments/TPD_CLEAN_V7_DCH_PROTOCOL.md)
- [`SCTransNet_TPD_V7_DCH_失败分析与不改主线修改计划.md`](SCTransNet_TPD_V7_DCH_失败分析与不改主线修改计划.md)

## 已完成裁决：TPD-Clean-v6

在初代 TPD 筛选实验之后，本仓库进一步实现了 TPD-Clean-v6。V6 仍只替换
SCTransNet 的两个浅层 patch embedding，不改变 backbone、SCTB、decoder、
损失函数或数据协议。它使用 Keep 投影权重派生 Context/Saliency 的共享输出坐标，
并通过空间零均值、幅值有界的 Context 增益图调制 Saliency residual。

V6 正式实验包含以下两种等容量变体，每种均训练 seed `42` 和 `3407`：

| 变体 | 说明 |
|---|---|
| `tpd_clean_v6_full` | 相位绑定 K/C/S 融合与 Context headroom 调制 |
| `tpd_clean_v6_phase_capacity` | 相同参数量与相位投影，但固定 `H=1` 的容量对照 |

四组任务均完成 800 epochs。12 个检查点可严格加载，8 份闭区间 Pd–Fa sweep、
固定阈值复算、source lock、精确续训日志以及 CPU/RTX 5090 smoke 均通过；
工程完整性 Gate E 通过。

### V6 固定阈值结果

| Seed | 变体 | 检查点 | Pd | Fa ↓ | mIoU ↑ |
|---:|---|---|---:|---:|---:|
| 42 | V6 Full | Pd-primary | **188/189** | 1.4915e-6 | 0.922945 |
| 42 | V6 Full | mIoU-primary | 187/189 | 1.7209e-6 | **0.940544** |
| 42 | Capacity | Pd-primary | **188/189** | 6.8722e-5 | 0.805532 |
| 42 | Capacity | mIoU-primary | 186/189 | **5.7364e-7** | 0.939605 |
| 3407 | V6 Full | Pd-primary | **187/189** | 4.8415e-5 | 0.860967 |
| 3407 | V6 Full | mIoU-primary | 185/189 | 1.0325e-6 | 0.924459 |
| 3407 | Capacity | Pd-primary | 186/189 | 1.0325e-6 | **0.928052** |
| 3407 | Capacity | mIoU-primary | 184/189 | **6.8837e-7** | **0.929850** |

正式裁决为 **`ENGINEERING_GATE_FAIL`**：

- Gate A（seed 42 固定阈值质量）未通过；
- Gate B（seed 42 预注册 Fa budgets）通过；
- Gate C（seed 3407 稳定性）未通过；
- Gate D（Full 相对等容量对照）未通过；
- Gate E（工程与证据完整性）通过。

因此 V6 不授权进入 NER 正式实验，也不改变既有主线结论。seed 42 显示出局部收益，
但 seed 3407 明显退化，且部分工作点被等容量对照覆盖。完整协议见
[`experiments/TPD_CLEAN_V6_PROTOCOL.md`](experiments/TPD_CLEAN_V6_PROTOCOL.md)，
整体技术与创新性复核见
[`SCTransNet_TPD_V6_整体设计正确性与创新性评估.md`](SCTransNet_TPD_V6_整体设计正确性与创新性评估.md)。

## 方法

TPD 将每个 2× 下采样单元拆成三个对齐分支：

- `keep`：`pixel_unshuffle` 后按通道压缩，保留相位信息；
- `context`：平均池化，保留局部背景上下文；
- `saliency`：最大池化减平均池化，突出局部小目标响应。

三个分支拼接后通过 `1×1` 卷积融合。大步长投影由多个 2× TPD 单元逐级完成。
实现见 [`model/tpd.py`](model/tpd.py)。

受控实验包含四个变体：

| 变体 | 说明 |
|---|---|
| `original` | 原始 SCTransNet 大步长 patch embedding |
| `progressive` | 多级 stride-2 卷积，同深度结构对照 |
| `spd` | `pixel_unshuffle + 1×1` 的 SPD 对照 |
| `tpd` | 相位保留、上下文和显著性三分支融合 |

## 初代 TPD 正式实验设置

- 数据集：NUDT-SIRST
- 数据范围：仅官方训练集，共 663 张图像
- 内部划分：530 张训练、133 张验证，划分种子 `20260722`
- 训练：800 epochs，FP32，batch size 16，patch size 256
- 模型随机种子：42
- 优化设置：初始学习率 `1e-3`，最低学习率 `1e-5`，10 epochs warmup
- 主检查点：验证集 Pd 最大；并列时依次选择更低 Fa、更高 tiny-Pd、
  更高 mIoU 和更低验证损失
- tiny target：面积不超过 9 pixels
- 匹配半径：3 pixels

四个变体共享相同的非 embedding 初始化、数据划分和训练协议。每个变体均完成
800 epochs，事件流、检查点角色、模型严格加载、指标恒等式及 Pd–Fa sweep
均通过完整性检查。

## 初代 TPD 实验结果

### Pd 主指标检查点

| 变体 | Epoch | Pd ↑ | tiny-Pd ↑ | Fa ↓ | 错误目标/图 ↓ | mIoU ↑ | nIoU ↑ | 参数量 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 456 | **0.994709** | 1.000000 | 1.4226e-5 | 0.1278 | 0.919086 | 0.923178 | 11,325,939 |
| Progressive | 300 | 0.989418 | 1.000000 | 1.0325e-6 | 0.0451 | 0.914348 | 0.914485 | 10,924,755 |
| SPD | 619 | 0.989418 | 1.000000 | **0** | **0** | **0.946542** | **0.939837** | 10,842,835 |
| TPD | 337 | **0.994709** | 1.000000 | 1.0325e-6 | 0.0226 | 0.933647 | 0.930339 | **10,827,731** |

在主检查点上，TPD 与 Original 同样检出 `188/189` 个目标，同时将 Fa 从
`1.4226e-5` 降至 `1.0325e-6`（约降低 13.8 倍），并将 mIoU 从
`0.919086` 提升至 `0.933647`。SPD 的 Pd 略低（`187/189`），但实现了零 Fa，
且 mIoU 最高。

### mIoU 次指标检查点

| 变体 | Epoch | mIoU ↑ | Pd | Fa ↓ |
|---|---:|---:|---:|---:|
| Original | 726 | 0.940738 | 0.984127 | 1.9504e-6 |
| Progressive | 611 | 0.931854 | 0.984127 | 2.1798e-6 |
| SPD | 470 | **0.949145** | **0.989418** | 4.5891e-7 |
| TPD | 457 | 0.942758 | 0.984127 | 4.5891e-7 |

### Pd–Fa 筛选结论

在五个预设 Fa budget（`1e-6`、`5e-6`、`1e-5`、`5e-5`、`1e-4`）上：

- TPD 在全部五个 budget 上优于 Original 和 Progressive；
- TPD 在后四个 budget 上优于 SPD；
- 在最严格的 `1e-6` budget 上，TPD 和 SPD 均检出 `187/189` 个目标，
  但 SPD 的实际 Fa 为 0，因此 SPD 更优；
- TPD 拥有一个独占的联合 Pd–Fa Pareto 点，但不是所有预算下的统一最优方法。

保守决策为 **`INCONCLUSIVE_MIXED_TRADEOFF`**：TPD 有潜力且不被支配，
但当前证据不足以将其确立为稳定主线。后续需要多随机种子、多数据集和官方测试集
评估。`paper_core_established=false`，`stability_claim_supported=false`。

## 数据准备

将 NUDT-SIRST 放在以下目录：

```text
datasets/NUDT-SIRST/
├── images/
├── masks/
└── img_idx/
    ├── train_NUDT-SIRST.txt
    └── test_NUDT-SIRST.txt
```

数据集、检查点和训练日志不纳入 Git 仓库。NUDT-SIRST 下载与原论文信息请参考
[官方实现仓库](https://github.com/YeRen123455/Infrared-Small-Target-Detection)。

## 运行实验

单个变体可使用统一 runner 运行：

```bash
python3 experiments/train_tpd_pilot.py \
  --variant tpd \
  --dataset NUDT-SIRST \
  --device cuda:0 \
  --epochs 800 \
  --batch-size 16 \
  --patch-size 256 \
  --workers 0 \
  --seed 42 \
  --split-seed 20260722 \
  --val-fraction 0.20 \
  --eval-every 1 \
  --base-lr 0.001 \
  --min-lr 0.00001 \
  --warmup-epochs 10 \
  --threshold 0.5 \
  --match-radius 3 \
  --tiny-area 9
```

将 `--variant` 替换为 `original`、`progressive`、`spd` 或 `tpd` 即可运行相应
对照。完整的训练、Pd–Fa 评估、汇总和审计工具位于 [`experiments/`](experiments/)；
其中 4×RTX 5090 启动脚本包含本次机器的固定 GPU UUID，迁移到其他机器前需要调整。

运行单元测试：

```bash
python3 -m unittest discover -s tests
```

## 代码结构

```text
model/tpd.py                         # TPD、SPD 和 Progressive embedding
model/tpd_clean_v6.py                # TPD-Clean-v6 与等容量对照
model/tpd_clean_v7.py                # 后续 V7 实验实现
model/tpd_clean_v7_dch.py            # V7 Deferred Context Headroom
model/tpd_clean_v8_mprs_dch.py       # V8 相位分辨 Saliency
model/tpd_ner_v8_mprs_dch.py         # NER V1
model/tpd_ner_v8_mprs_dch_v2.py      # NER V2
model/tpd_ner_v8_mprs_dch_v3.py      # NER V3
model/tpd_ner_v8_mprs_dch_v4_tail_aware.py
                                     # NER V4 Tail-Aware
model/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py
                                     # V4 + 训练期 Target Survival heads
model/tpd_frequency_gate_v2_croa.py  # QFG-V2-CROA
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py
                                     # V4 + TSS + QFG 因子模型
experiments/train_tpd_pilot.py       # 无官方测试泄漏的统一训练 runner
experiments/train_tpd_clean_v6_exact.py
                                     # V6 精确续训正式入口
experiments/train_tpd_clean_v7_dch_exact.py
                                     # V7-DCH 精确续训正式入口
experiments/train_tpd_ner_v8_mprs_dch_v3_exact.py
                                     # V8 + NER V3 正式训练入口
experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact.py
                                     # V4 + TSS 精确续训入口
experiments/train_tpd_ner_v4_qfg_v2_croa_exact.py
                                     # QFG 正式训练入口
experiments/compare_tss_qfg_v2_croa_factorial.py
                                     # A/B/C/D 因子与 Pareto 汇总
experiments/evaluate_pd_fa_sweep.py  # Pd–Fa threshold sweep
experiments/summarize_tpd_pd_fa.py   # Pd–Fa 汇总
experiments/decide_tpd_mainline_4x5090.py
                                     # 保守主线筛选决策
analysis/                             # 信息瓶颈分析
tests/                                # 模块与决策策略测试
```

设计背景和实验方案：

- [`TPD_SCTransNet_目标保真下采样实验方向.md`](TPD_SCTransNet_目标保真下采样实验方向.md)
- [`TPD_SCTransNet_主线修订版.md`](TPD_SCTransNet_主线修订版.md)
- [`SCTransNet_TPD_FG_实验设计与执行方案.md`](SCTransNet_TPD_FG_实验设计与执行方案.md)

## 与上游 SCTransNet 的关系

本仓库是 SCTransNet 的实验性派生版本，并非原论文官方结果仓库。原始模型、
论文、预训练权重和官方说明请访问
[xdFai/SCTransNet](https://github.com/xdFai/SCTransNet)。

如果使用 SCTransNet 基线，请引用原论文：

```bibtex
@article{SCTransNet,
  author  = {Yuan, Shuai and Qin, Hanlin and Yan, Xiang and Akhtar, Naveed and Mian, Ajmal},
  title   = {SCTransNet: Spatial-Channel Cross Transformer Network for Infrared Small Target Detection},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {62},
  pages   = {1--15},
  year    = {2024},
  doi     = {10.1109/TGRS.2024.3383649}
}
```

## 结果边界

README 中的初代 TPD 结果为 seed-42 内部验证实验，TPD-Clean-v6 与 V7-DCH
结果为 seed `42/3407` 内部验证实验，V8 + NER V1/V2/V3/V4、TSS 与
QFG-V2-CROA 为 seed-42 内部验证筛选。它们均不等同于 NUDT-SIRST 官方测试
成绩，也不构成跨数据集稳定性结论。
V6、V7-DCH 均为 `ENGINEERING_GATE_FAIL`；V8 + NER V1/V2/V3 均为
`RETURN_TO_MODEL_OPTIMIZATION`；V4 为
`RELATIVE_MODEL_IMPROVEMENT_CONFIRMED_WITH_TRADEOFF`；QFG-only 为
`DOMINATED`，TSS+QFG 为 `PARETO_MIXED_TRADEOFF`，最终工程选择为
`SELECT_D_TSS_QFG`。不得将局部优点描述成稳定优越性。任何后续
论文级声明都应建立在完整审计、多种子、多数据集和独立官方测试结果之上。
