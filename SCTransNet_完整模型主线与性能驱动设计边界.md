# SCTransNet 完整新模型主线与性能驱动设计边界

> 状态：当前设计约束
>
> 数据与训练范围：NUDT-SIRST 内部 530/133 划分，seed 42，formal800
>
> 当前候选：TPD V8-MPRS-DCH + 五节点 NER V4 Tail-Aware + QFG-V2-CROA；TSS 仅作为可选训练配方
> 核心原则：先固定研究问题和模型主线，再允许由 Pd、Fa、mIoU、tiny-Pd 与 Fa-budget 的综合结果驱动代码修改。

## 1. 主线不是“当前所有代码都不能改”

核心模型要解决的是红外小目标在两处连续受损的问题：

1. 浅层大步长下采样会压缩目标的相位、局部对比度和微弱响应；
2. 解码阶段的高分辨率 skip 同时携带目标和背景，弱目标可能无法被稳定恢复。

当前完整候选还用 QFG 处理跨尺度 Transformer Query 容易受高频杂波和强背景干扰的问题，但固定科学主线不依赖某一版 QFG；QFG 可以被修改、替换或在没有综合增益时移除。

因此模型主线固定为：

```text
输入
  → Keep–Context–Saliency TPD
  → SCTransNet 多尺度编码与跨尺度交互
       └─ 当前候选：在 SCTB 的 Query 路径内接入 QFG
  → 3+2 五节点 Nested Evidence Relay
  → 逐级调制 decoder skip
  → 目标分割/检测输出
```

主线固定的是“要解决什么”和“关键数据流如何解决”，不是冻结某一版归一化、某一个门控函数或某一个损失权重。

## 2. 必须保留的核心创新

### 2.1 Keep–Context–Saliency TPD

TPD 必须继续同时表达三类互补信息：

- Keep：显式重排 2×2 cell 的相位后执行可学习的 `4C→C` 压缩，保留可学习的相位响应，但不声称无损或可逆；
- Context：当前实现是 2×2 arithmetic mean/DC；把它解释为局部背景或低频参考属于模型假设；
- Saliency：表达目标相对局部背景的突出响应。

允许修改：

- Context 的计算和投影方式；
- Saliency 的定义、相位分辨形式和融合公式；
- 三源之间的门控、归一化和残差尺度；
- TPD block 的高效等价实现；
- TPD block 的参数共享方式；
- 在仍输出 `h11/h12/h13+h21/h22`、保持五节点尺度及 decoder 对位不变的前提下，调整 block 内部实现。

不允许在没有重新定义研究主线的情况下，把 TPD 退化成普通 stride convolution、纯 SPD 或与 K/C/S 无关的任意下采样器。

当前代码位置：

```text
model/tpd_clean_v8_mprs_dch.py
  TPDCleanV8MPRSDCHBlock
  TPDCleanV8MPRSDCHPatchEmbedding
```

当前实现只替换 `mtc.embeddings_1` 和 `mtc.embeddings_2`。前者含四个 2× block，后者含三个 2× block，共七个 TPD block。

### 2.2 3+2 五节点 NER

五个证据节点必须继续来自两条 TPD 浅层路径的非终端状态：

```text
x1 路径：h11, h12, h13
x2 路径：h21, h22
```

嵌套恢复顺序固定为：

```text
stage4: h13 + h22 + up4        → q4, mask4
stage3: h12 + h21 + q4 + up3   → q3, mask3
stage2: h11 + q3 + up2         →     mask2
```

它的核心价值是把下采样过程中仍存活的浅层证据显式送到解码阶段，而不是只依赖最终 tokenizer endpoint。

允许修改：

- 每个 stage 的证据对齐和融合；
- relay width；
- mask 的范围、归一化和正负证据形式；
- Tail-Aware DC 的作用域；
- `q4 → q3 → q2` 中的门控和残差更新公式；
- 同尺度、decoder-conditioned、显式 skip modulation 内部的注入公式。

必须保留：

- 3+2 五节点来源；
- 深到浅的嵌套 relay；
- NER 与 decoder 恢复之间的显式联系；
- 同尺度证据对 decoder skip 的显式调制。

若把 NER 改为直接相加到 decoder 主分支、取消同尺度 skip modulation，或改变五个节点的尺度/来源，则属于 NER 拓扑变化，而不是普通内部公式修改。

当前代码位置：

```text
model/tpd_ner_v8_mprs_dch.py
  TPDNERV8MPRSDCHSCTransNet
  explicit_embeddings
  _forward_with_relay

model/tpd_ner_v8_mprs_dch_v2.py
  RMS source balance
  centered arctangent gate

model/tpd_ner_v8_mprs_dch_v3.py
  stage-wise DC offsets

model/tpd_ner_v8_mprs_dch_v4_tail_aware.py
  TailAwarePersistentDCOffsetEvidenceRelay
  TPDNERV8MPRSDCHV4SCTransNet
```

V4 只新增 Tail-Aware 作用域；当前 NER 的 RMS/atan 和 DC 分别继承自 V2、V3。

## 3. 性能驱动的可变设计区

### 3.1 QFG 是推理期性能模块，不是不可修改的核心主线

QFG 当前从 `x1...x4` 计算固定 Haar 频率先验，只调制每个 SCTB 中归一化前的 `q1...q4`，不对 K/V、CFN 或 decoder 执行直接算子修改。训练后共享 backbone 参数会改变，所以 K/V 的数值仍可能被间接改变；`detach` 只切断频率旁路对 encoder 的额外梯度。

允许根据综合结果修改：

- 使用的频带；
- 频率先验归一化；
- gate 的空间、通道和尺度形式；
- gate 范围和初始化；
- 是否加入背景/杂波抑制条件；
- Query 路径内部的调制位置；若移动到 attention 输出之后，就不再是 Query-only；
- 四尺度参数共享方式。

当前代码位置：

```text
model/tpd_frequency_gate_v2_croa.py
model/tpd_query_frequency_bridge.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py
```

### 3.2 TSS 是可选训练监督

TSS 只在训练期监督 `emb1`、`emb2` 的 stride-16 target-presence，推理时移除两个 head。当前 C/D 两个训练模型都注册 TSS heads；C 的 `lambda=0` 表示辅助 loss 不参与反向，并不表示训练 forward 不执行 heads。evaluation 不执行 heads，导出的 inference class 则彻底移除它们。

允许修改或删除：

- target 的尺度和构造方式；
- head 结构；
- loss 类型和权重；
- 监督 endpoint；若不再监督 TPD/tokenizer endpoint，就需要改名并重新定义为一般 auxiliary supervision，而不能继续沿用当前 TSS 含义；
- 与最终分割目标的耦合方式。

若 TSS 不能使 D 相对 C 产生持续的综合优势，最终模型不保留 TSS 训练配方。TSS 不属于推理结构的必选部分。

当前代码位置：

```text
model/tpd_survival.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py
```

### 3.3 其他可修改部分

在不改变核心问题的前提下，以下内容都可以为性能服务：

- 分割、连通性、边界或目标级损失；
- deep supervision 权重；
- decoder 融合和 skip 标定；
- optimizer、学习率、warmup 和训练阶段；
- QFG/TSS/NER 的初始化；
- 非核心归一化与数值稳定实现；
- 等价的低计算量实现。

若这些修改仍不足，允许进一步修改 TPD 或 NER 的内部公式。只有当 3+2 五节点拓扑或 K/C/S 三源语义本身被替换时，才属于真正改变主线，需要建立新版本和新的对照链。

## 4. 当前完整模型的实际数据流

```text
image
  ├─ CNN encoder: x1, x2, x3, x4, d5
  │
  ├─ TPD embeddings
  │    ├─ x1 → h11 → h12 → h13 → emb1
  │    ├─ x2 → h21 → h22 → emb2
  │    ├─ x3 → emb3
  │    └─ x4 → emb4
  │
  ├─ QFG prepare
  │    └─ x1...x4 → four feature-level frequency factors
  │
  ├─ four SCTB blocks
  │    └─ each block applies the prepared factors to its own q1...q4
  │
  ├─ reconstruct multi-scale features
  │
  ├─ NER decoder
  │    ├─ stage4: h13, h22, up4 → q4/mask4
  │    ├─ stage3: h12, h21, q4, up3 → q3/mask3
  │    └─ stage2: h11, q3, up2 → mask2
  │
  └─ segmentation output and training-time deep supervision
```

TSS-on 和 C 组的架构匹配 control 都复用已经计算的 `emb1`、`emb2` endpoint，不重新执行 TPD；二者的区别是辅助 loss 权重是否为零。

## 5. 最终模型不用单一固定门槛决定

所有方案必须使用自己的：

```text
best.pth.tar       → Pd-primary role
best_miou.pth.tar  → mIoU-secondary role
```

统一比较：

- threshold 0.5 下的 Pd、Fa、mIoU、tiny-Pd 和错误目标；
- `Fa <= 1e-6, 5e-6, 1e-5, 5e-5, 1e-4` 的工作点；
- 完整阈值扫描的非支配区间；
- 相对 baseline、NER V4、A/B control 的同角色差值。

真正严格的同父、同随机流 factorial/paired 比较是 A/B/C/D；baseline 和历史 NER V4 用于统一同角色描述比较，不能扩大成严格 factorial 结论。

最终判断采用多目标原则：

1. 多检出目标是明确收益，但必须同时报告新增虚警和 mIoU 代价；
2. Pd 相同时，优先更低 Fa、更少错误目标和更高 mIoU；
3. 单个偶然阈值点不能决定模型，改善应在相邻阈值或相邻 Fa budget 上持续；
4. tiny-Pd 当前若均为 39/39，只作为不回退条件，不能把天花板结果当成模块成功；
5. 若两个模型接近等价，选择推理结构更简单、计算更少的模型；
6. 若模型只在不同应用区间各自占优，记录为真实 trade-off，而不是强行宣布全面胜出。

这是一套性能选择规则，不是要求所有模型先通过某个固定 mIoU 或 Fa 数字才能继续设计。

## 6. formal800 后的修改路由

### 6.1 Pd 或预算包络没有提高

优先检查：

- 漏检目标在 TPD、SCTB、NER 哪一阶段消失；
- QFG 是否只改变校准而没有改变目标排序；
- NER relay 是否对漏检目标产生有效 mask；
- TSS 的 cell-presence 目标是否过粗。

优先修改 QFG 的目标/背景条件、NER 的证据融合或训练目标，不先重复堆叠新的独立模块。

### 6.2 Pd 提高但 Fa 明显恶化

优先修改：

- QFG 对高频 hard negative 的抑制；
- NER 的负证据或背景条件；
- object-level false-positive loss；
- gate 范围和尺度标定。

此时不应只继续增强 Saliency，因为目标边缘和背景高频会被一起放大。

### 6.3 Pd/Fa 较好但 mIoU 下降

问题更可能位于区域恢复，而不是目标是否存活。优先修改：

- decoder 边界恢复；
- NER mask 的空间分辨率；
- 连通性或边界损失；
- shallow skip 的融合标定。

### 6.4 mIoU 提高但仍漏同一个目标

这表示主体目标分割更好，但极弱目标排序未改善。优先处理：

- Query 的目标/杂波区分；
- NER 的跨层持续证据；
- 目标级排序监督。

不把继续优化平均区域质量当成恢复漏检目标的替代方案。

## 7. 当前重复计算审计

### 7.1 已经避免的重复计算

- MPRS Saliency 使用 `Sa8 = Sa7 + ((K-bias)-Ca)/3`，复用 Keep 和 Context 投影，不构造显式 5D phase-Saliency，也不增加第二个 `4C→C` dense projection；
- QFG 的四个 feature-level frequency factor 每次整模型 forward 只 prepare 一次，在四个 SCTB 中复用；
- TSS 通过瞬态 endpoint capture 复用当前 forward 的 `emb1/emb2`，没有第二次 tokenizer forward；
- 推理模型删除 TSS head。

### 7.2 仍存在但当前不应中途修改的计算

- 每个 TPD block 同时执行 `pixel_unshuffle`、`avg_pool2d` 和 `max_pool2d`。后续可从 unshuffle 后的四 phase 直接计算 mean/max，减少两次独立输入读取；两条路径在实数代数上等价，但浮点 reduction 顺序可能不同，必须先做数值和速度测试；
- QFG 的 factor 在四个 SCTB 中重复参与乘法，但每个 SCTB 的 Query 不同，这不是可删除的重复计算；
- V4 的 persistent-tail 路径在 stage3 计算一次 `T3(q3)`，stage2 构造 parent support 时又计算一次；下一版本可使用 forward-local cache，属于只减少计算的等价优化；
- QFG 四个 level 各自保存相同的固定 Haar buffer，体积很小；更值得处理的是 `tpd_query_frequency_bridge.py` 复制 baseline attention/block/encoder、整合类复制 parent relay 的维护漂移风险，而不是误判为同一次 forward 双执行；
- bridge 的类型注解仍指向 QFG V1 类型，production 实际传入兼容 API 的 V2-CROA；当前能运行，但下一版本应修正静态类型语义；
- baseline 每个 `Attention_org` 注册了16个 forward 完全未读取的 `q*_attn*` 标量；四个 SCTB 合计64个死状态。它们不是运行卷积或重复执行，删除时需要显式 checkpoint 迁移；
- 当前 evaluation/deployment wrapper 因继承 `mode='train', deepsuper=True`，仍会计算四个 deep-supervision logits、插值和 `d0`，即使最终只返回 final map；可增加只计算 final map 的纯部署 forward；
- baseline 的双 identity addition 被完整继承，实际为 `reconstruct(encoded_i)+2*f_i`。它可能造成 bypass dilution，但不是本轮 C/D 的单独变量；若结果指向 skip 标定问题，再建立严格 identity-initialized 的可学习残差标定候选。

当前 formal800 已被 source lock 固定，训练中不修改这些执行路径。任何运行期优化都在本轮结果封存后进入新版本。

## 8. 当前阶段和下一步

当前两条正式训练：

```text
C: TPD + NER + QFG，TSS loss = 0
D: TPD + NER + QFG，TSS loss = 0.005
```

两者从同一 V4 `best_mIoU` 父 checkpoint warm-start，固定 seed 42，在物理 GPU 2 和 GPU 3 上分别训练 800 epochs。

训练完成后：

1. 分别评估 C/D 自身 `best` 和 `best_mIoU`；
2. 与 A/B、NER V4 和 baseline 做统一阈值扫描；
3. 若 C 或 D 提供有意义的综合改善，冻结最优配方；
4. 若改善不足，根据第 6 节定位问题并修改可变设计区；
5. 只有证据指向 K/C/S 语义或五节点拓扑本身无效时，才升级为主线重构。

若本轮不足，代码修改优先级由终局诊断决定：

1. 先完成 C/D 终局比较并决定 TSS：若 D 没有持续优于 C，先移除或降低 TSS；随后若保留下来的 QFG 配方已明显学习但 parent 表征后期漂移，再使用 discriminative learning rate，降低共享 V4/TPD/NER 参数学习率，只让 QFG 使用主学习率；
2. 若 QFG 长期接近 identity，再增加 SCTB×level 的 bounded strength；若高频背景主导假阳性，则优先把频带改为 `low` 或分离 LL/high-band，而不是继续放大当前 high-frequency gate；
3. 若 formal800 own-best 与 Fa-budget 包络终局仍停在 `188/189`，且对象级虚警没有下降，在 final logits 增加小权重的非对称 hard-negative/target-margin 项；若主要是区域质量问题，再重配六路 deep-supervision 权重并加入 soft-IoU/Tversky；
4. 再校准 baseline 继承的 `reconstruct+2f` 双 identity，用严格保持 step-0 输出的 bounded coefficient 让模型学习 bypass 强度；
5. 若高分辨率假阳性明确来自 relay，再做 NER V5：stage2 DC hard-off 或对 stage2/3 support 增加 bounded strength/temperature；
6. 最后才修改 TPD 内部 Context/Saliency；仍保留 K/C/S、MPRS 的 Keep-phase 投影和 3+2 五节点。

不直接继续增加第六个模块。QFG/TSS 若最终没有综合增益，应简化或移除，把计算预算留给真正改善 Pd–Fa–mIoU 的结构。

因此当前策略是：

```text
固定核心问题与核心数据流
≠ 冻结所有实现

性能不足
→ 定位瓶颈
→ 修改最相关的代码
→ 重新训练和统一评估
→ 直到形成相对 baseline 有意义提升的完整模型
```
