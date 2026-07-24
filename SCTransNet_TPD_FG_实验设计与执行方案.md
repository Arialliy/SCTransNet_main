# SCTransNet：TPD / FG 机制验证与实验执行方案

> 状态：预验证后的修订稿；TPD 仍为主线，但从“MaxPool 替换器”重定位为“进入 SCTB 前的目标保真下采样”  
> 基线仓库：SCTransNet，当前提交 `e0276283794e9a54db334c2105f6e70107312202`  
> 目标场景：NUAA-SIRST、NUDT-SIRST、IRSTD-1K  
> 默认投稿假设：TGRS / 遥感与红外小目标检测同等级别证据要求；若目标 venue 不同，应再调整篇幅和基线范围。

> 当前正式主线请以 [`TPD_SCTransNet_主线修订版.md`](./TPD_SCTransNet_主线修订版.md) 为准。本文件保留早期 MaxPool-TPD 与 FG 设计，供位置消融和条件扩展参考。

## 0. 结论先行

2026-07-22 已使用三个数据集各自的最优 checkpoint 完成冻结特征预验证。完整证据见 `analysis/results/best_checkpoint_probe_v1/验证结论.md` 和 `combined_report.md`。

当前建议修订为：**TPD-SCTransNet 仍是主线；其核心载体改为浅层 TPD Patch Embedding（优先 `emb1/emb2`）。原 MaxPool 放置方式降为位置对照，FG 只保留为第二阶段条件性候选。**

原因有三点：

1. 三数据集的 P1/P2 MaxPool 配对 rank loss 均接近 0，且 NUAA 的 Pool AP 方向会随 probe seed 翻转，不支持将其视为稳定主瓶颈。
2. `x1→emb1` 在 3/3 数据集均出现明显 AP 与 tiny-target rank 损失，图像级 bootstrap 95% CI 不跨 0；`x2→emb2` 也在三数据集同向，其中两套超过预设实际效应阈值。
3. FG 的创新空间已明显收窄。FSCNet 已使用 Haar 高频补偿、Frequency Attention Gate 和基于 SCTB 的 G-SCTB；WaveTD、MSDA-Net、ARFC-WAHNet、SWAN 也覆盖了小波、频率注意或高频方向先验。若继续 FG，必须证明与这些工作的结构和机制差异，而不能只以“频率引导 Query”作为唯一创新点。

推荐的可证伪中心假设为：

> SCTransNet 将浅层特征通过 `Conv(k=16,s=16)` 与 `Conv(k=8,s=8)` 一步映射到 SCTB 公共网格时，造成可测的微小目标信息损失；目标保真 Patch Embedding 应在不恶化虚警权衡的条件下改善 `emb1/emb2` 的可解码性，并提高相同 Fa 下的微小目标 Pd。

后文有关 MaxPool-TPD 的模块和实验表作为“同一创新在不同位置的对照路线”保留，但在 TPD Patch Embedding 的 E0–E3 最小实验完成前不作为主模型执行。只有当 Haar 高低频先验相对空间保真 embedding 提供独立增量，并且方案能与 FSCNet 等近邻方法清楚区分时，才启动 FG-Core。

---

## 1. 代码事实与必须修正的概念

### 1.1 真实尺度流

默认输入为 `B×1×256×256`，`base_channel=32`。实际张量如下：

| 节点 | 形状 | 是否进入 SCTB |
| --- | --- | --- |
| `x1 = inc(x)` | `B×32×256×256` | 是 |
| `P1 → x2` | `B×64×128×128` | 是 |
| `P2 → x3` | `B×128×64×64` | 是 |
| `P3 → x4` | `B×256×32×32` | 是 |
| `P4 → d5` | `B×256×16×16` | 否，只进入瓶颈/解码器 |
| `emb1…emb4` | 通道分别为 `32/64/128/256`，空间均为 `16×16` | 是，作为 SSCA 输入 |

对应代码：

- 四次 MaxPool：`model/SCTransNet.py:572,603-606`；
- SCTB 输入仅为 `x1…x4`：`model/SCTransNet.py:613`；
- Patch Embedding 使用 kernel/stride `[16,8,4,2]`：`model/SCTransNet.py:39-57,414-429`。

因此：

- 只有 P1–P3 会改变进入 SCTB 的 `x2…x4`；
- P4 测的是瓶颈/解码器效应，不能解释为“进入 SCTB 前的存活”；
- TPD 的机制证据必须同时测 encoder 输出与 Patch Embedding 输出。

### 1.2 SSCA 的真实注意力语义

当前实现实际是单头，`num_attention_heads=1` 被硬编码，配置中的 `num_heads=4` 未使用。默认输入下：

| 张量 | 形状 |
| --- | --- |
| `Q_i` | `B×1×C_i×256` |
| `K,V` | `B×1×480×256` |
| `A_i = Q_iK^T` | `B×1×C_i×480` |

注意力矩阵是“本级通道 × 全层级通道”，没有位置轴。FG 若使用空间门控，只能表述为：

> frequency-weighted spatial evidence for channel cross-attention

不能表述为“逐位置频率注意力”。此外，Q 在空间维执行 L2 normalize（`model/SCTransNet.py:165-169`）：

- 若门是 `B×C×1×1` 的通道标量，且放在 normalize 前，缩放会被归一化抵消；
- 可行的最小设计是 `B×1×16×16` 的空间门，在 Q 卷积后、flatten/normalize 前调制；
- 或将通道门放在 normalize 后，但需要单独验证 InstanceNorm/Softmax 是否仍保留影响。

### 1.3 三个术语需要降格

1. **PixelUnshuffle 分支不是整体无损。** PixelUnshuffle 本身可逆，但后续 `4C→C` 的 `1×1` 投影是有损压缩。建议称“先空间重排、后学习压缩”。
2. **stride-2 DWConv 不自动等于低通。** 未受约束的卷积可学成低通、高通或带通。建议称“可学习背景上下文分支”，并加入固定 BlurPool 对照。
3. **GAP 门控是场景自适应，不天然是目标自适应。** 几个像素的目标对全局平均贡献很小；只有门控统计与场景/目标条件的关联证据成立后，才可声称自适应机制有效。

### 1.4 TPD 分支接口

四级下采样器不能共享参数，接口必须分别为：

| 模块 | 输入输出契约 |
| --- | --- |
| `downsample1` | `32→32, H→H/2` |
| `downsample2` | `64→64, H/2→H/4` |
| `downsample3` | `128→128, H/4→H/8` |
| `downsample4` | `256→256, H/8→H/16` |

三个分支必须具有完全相同的空间采样原点。`3×3,s=2,p=1` 与 `2×2,s=2` pooling / PixelUnshuffle 可能存在相位差；实现后需用单像素 impulse 测试峰值坐标，不能只验证 shape。

---

## 2. P0：正式实验前必须通过的完整性门

以下任一项未完成时，不开始主实验。

### 2.1 禁止用 test 选 checkpoint

当前 `train.py:145-169` 从第 500 epoch 起每个 epoch 直接测试，`train.py:194-214` 再按 test mIoU 保存 best，属于 test-set model selection。现有 best checkpoint 只能作为工程参考，不能进入论文主表。

建议协议：

1. 从官方 train split 固定划出 20% validation，按图像内目标数和目标面积桶分层；保存一次性 split 文件，所有方法共享。
2. 训练 800 epoch；每个 epoch 在 validation 评估一次，以预先固定的 Pd-primary 规则（`Pd 最大 → 同 Pd 时 Fa 最低 → tiny-Pd 最高 → mIoU 最高 → loss 最低`）保存 `best.pth.tar`；另存 `best_miou.pth.tar` 仅作分割质量分析。这里选择的是 validation 最优 checkpoint，不是固定 epoch 800，也不是最后一轮。
3. 模型结构、超参数、checkpoint、阈值和 seed 全部冻结后，test 只运行一次。
4. 固定阈值 `0.5` 作为与原论文兼容的结果；主要目标级结论使用 validation 校准阈值。

### 2.2 修正并单元测试 Pd / Fa

当前 `metrics.py:105-123` 通过“预测连通域面积是否出现在已匹配面积列表”排除真阳性。若一个未匹配假阳性与已匹配区域面积相同，会被错误删除，Fa 可能被低估；现有匹配还依赖连通域遍历顺序。

修正要求：

- 8-connectivity 连通域；
- 以 region ID 做一对一匹配，不以面积值匹配；
- 为兼容原文，主兼容指标保留中心距离 `<3 px`；另报告 Hungarian 一对一匹配版本；
- unmatched prediction 的像素数全部计入 Fa；
- 用合成 mask 覆盖：空预测、多目标、重复面积、一个预测匹配两个 GT、边界距离恰为 3、相邻目标和尺寸不一致。

### 2.3 修复数据一致性

本地数据审计发现：

- `NUAA-SIRST/test` 的 `Misc_111` 图像为 `325×220`，mask 为 `592×400`；
- 当前 loader 会分别 padding 后再按图像尺寸裁剪，标签与图像不对齐；
- 其余五个 train/test split 未发现尺寸不一致。

处理规则：优先从数据集官方来源恢复正确配对并记录 checksum；在无法确认前，不得直接 resize mask。若必须排除，需在所有方法中统一排除并在论文中披露。

### 2.4 固化复现清单

每个 run 至少保存：

```text
run_id/
├── config.yaml              # 模型、数据、优化器、阈值、seed
├── split_manifest.json      # train/val/test 文件名与 checksum
├── environment.txt          # Python/CUDA/PyTorch/依赖/GPU
├── git_state.txt            # commit + git diff hash
├── checkpoint.pt
├── metrics.json
├── per_image_metrics.jsonl
├── per_object_metrics.jsonl
└── predictions/             # test 概率图，便于统一重算指标
```

当前 checkpoint 未保存 seed、args 或 git hash，仓库也没有 requirements；这些都应在新 runner 中补齐。

---

## 3. 数据集与尺寸分组

按本地 mask、8-connectivity 统计得到：

| Dataset | Train images | Test images | Test targets | `A≤4` | `A≤9` | `10≤A≤25` | `A>25` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NUAA-SIRST | 213 | 214 | 263 | 3 | 35 | 120 | 108 |
| NUDT-SIRST | 663 | 664 | 945 | 1 | 259 | 298 | 388 |
| IRSTD-1K | 800 | 201 | 297 | 7 | 29 | 107 | 161 |

`A≤4` 在三个 test split 中只有 `3/1/7` 个目标，单独作为主要终点统计功效不足。因此固定如下口径：

- **主要 tiny 组：`A≤9`**；
- 中等组：`10≤A≤25`；
- 较大组：`A>25`；
- `A≤4` 仅作探索性附表，必须同时报告实例数和精确置信区间。

目标面积在原始分辨率二值 GT 上按连通域像素数计算。不要在不同文档中混用 `1–4/5–16/>16` 与 `≤4/5–9/10–25/>25`。

注意：Fa 没有对应的 GT 目标尺寸，不能自然地“按目标面积分组”。尺寸分组报告 Pd、miss、实例 IoU；Fa 报全图值，或明确写成“包含该尺寸目标的图像子集 Fa”。

---

## 4. Claim–Evidence 矩阵

| Claim | 审稿人问题 | 决定性证据 | 失败时如何解释 |
| --- | --- | --- | --- |
| MaxPool 是信息瓶颈 | 损失是否其实发生在 Patch Embedding？ | pre-pool、post-pool、post-encoder、post-embedding 的统一 probe | 若 embedding 下降更大，收窄 TPD 主张或转向 embedding |
| TPD 保留目标信息 | 响应变强是否只是尺度变化？ | 局部 CNR、rank-SR、冻结 probe AUPRC、post-embedding probe | 若只在 encoder 提升而 embedding 后消失，机制链不成立 |
| TPD 改善 tiny target | 是否只是阈值更激进？ | `Pd_(A≤9)@Fa*`、Pd–Fa 曲线、miss | 若 Pd 上升伴随 Fa 同比例恶化，核心结论失败 |
| 背景分支控制虚警 | 显著性是否只增强边缘/噪声？ | edge-near、high-clutter、negative crop 分析 | 若复杂背景 Fa 上升，限制 sal 分支或缩小适用范围 |
| 动态门控必要 | 提升是否来自额外参数？ | equal/static/dynamic/shuffled-gate 对照 | 若 dynamic 不优于 static，删除“自适应”主张 |
| 存活监督与 TPD 协同 | 是否只是通用辅助监督？ | `MaxPool/TPD × no-aux/aux` 2×2 | 若两者同幅受益，定位为通用训练技术 |
| FG 提供独立频率增量 | 与 FSCNet/FAGM 有何本质不同？ | 控制容量后的 H/L 条件增量及直接近邻对比 | 若无独立增量或结构过近，不作为主线 |
| 双分辨率与 TPD 互补 | 是否只是更大模型？ | `TPD × dual-resolution` 2×2 与交互项 | 无正交增量则不放入 Full |

---

## 5. P1：先定位瓶颈，不先训练复杂模型

### 5.1 Hook 节点

对干净 SCTransNet 基线记录：

1. `x1/x2/x3/x4`：encoder 输出；
2. 每次 `pool(x_i)`：纯下采样输出；
3. `emb1…emb4`：Patch Embedding 后、进入 SSCA 前；
4. Q 卷积后、normalize 前；
5. SCTB encoded 输出；
6. reconstruct 后以及外层二次 residual 后。

最后两处必须区分，因为当前 `ChannelTransformer` 内先执行 `reconstruct(encoded_i)+en_i`，外层又执行一次 `+f_i`，解码器实际收到 `reconstruct(encoded_i)+2E_i`。

### 5.2 不直接使用全图响应比 `R_i`

原设想的全图目标/背景均值比受通道尺度、BN 和背景面积支配。改为每个目标实例的局部统计。

对节点特征先用 GT 外背景估计通道中心与尺度，再构造通道能量图 `A_i`。对第 `k` 个目标及其局部背景环 `R_ik`：

\[
\mathrm{CNR}_{ik}=
\frac{\max_{p\in M_{ik}} A_i(p)-\mu(A_i(R_{ik}))}
{\sigma(A_i(R_{ik}))+\epsilon}.
\]

定义 rank-based survival：

\[
SR_i(q)=\frac{1}{N}\sum_k
\mathbf 1\left[
\max_{p\in M_{ik}} A_i(p)>Q_q(A_i(R_{ik}))
\right].
\]

报告 `q=0.95/0.99` 及 `q∈[0.90,0.995]` 的 SR-AUC；这样无需用任意绝对响应阈值比较不同模型。

再在冻结特征上训练同结构轻量 probe，统一报告：

- pixel AUPRC；
- object Pd@固定 probe-FPR；
- `A≤9` 子集结果；
- 每图聚类 bootstrap 95% CI。

### 5.3 方向决策量

定义：

- `D_pool`：池化前到 post-encoder 的 probe 下降；
- `D_embed`：post-encoder 到 post-embedding 的 probe 下降；
- `Δ_freq`：控制原空间特征后，引入 Haar H/L 对相同 probe 的增量。

先用多 seed 基线波动确定最小实际效应 `δ`，再冻结门槛：

- `D_pool` 在至少两个数据集可重复且 CI 超过 `δ`：优先 TPD；
- `D_embed` 明显、`D_pool` 弱，且 `Δ_freq` 有独立增量：才考虑 FG；
- 主要下降在 Patch Embedding 且 `Δ_freq` 也无增量：暂停 TPD/FG，转向 target-preserving embedding 或双分辨率；
- 两者都成立：先独立跑 TPD-Core 与 FG-Core，不立即组合。

---

## 6. P2：公平的核心比较

第一轮不加 survival loss、HF loss、SoftIoU、gradient loss、decoder 高频回注、频率 K/V 或双分辨率 SCTB。所有组保留原始六项 BCE，统一只作用于 P1–P2。

| Arm | P1–P2 downsample | Haar H/L | SSCA gate | 目的 |
| --- | --- | --- | --- | --- |
| B0 | MaxPool | 无 | 无 | 干净 SCTransNet |
| B1 | AvgPool | 无 | 无 | 简单平滑 sanity baseline |
| B2 | Strided Conv | 无 | 无 | 可学习下采样 baseline |
| B3 | MaxBlurPool | 无 | 无 | 抗混叠/平移稳定 baseline |
| B4 | SPD-Conv / PixelUnshuffle+Conv | 无 | 无 | TPD keep 分支的最近算子 baseline |
| B5 | 参数匹配通用多分支 | 无 | 静态 | 控制额外容量 |
| T0 | TPD 三分支 | 无 | equal/static fusion | 多分支本身 |
| T1 | TPD 三分支 | 无 | dynamic scene gate | TPD-Core |
| F0 | MaxPool | Haar H/L | 参数匹配 static Q gate | FG 容量控制 |
| F1 | MaxPool | Haar H/L | dynamic spatial Q gate | FG-Core（条件启动） |

公平约束：

- 相同 split、seed、数据顺序、增强、epoch、optimizer、scheduler 和 checkpoint 规则；
- 报告 trainable params、完整 MAC/FLOPs 口径、显存、单图 latency；
- TPD/FG 的共同机制终点使用 post-embedding、post-Q、post-SCTB probe；
- 参数无法完全匹配时，额外加入 `MaxPool + 1×1/MLP` 容量对照；
- 所有方法从随机初始化训练；迁移旧权重只用于 smoke test，不用于主结论。

主要终点：

> `Pd_(A≤9)@Fa*`，其中 `Fa*` 为干净基线在 validation 上预注册的 Fa 预算；每个方法只在 validation 选择达到该 Fa 的阈值，再冻结到 test。

次要终点：

- 全目标 Pd、Fa、miss；
- mIoU、nIoU、F1；
- `Pd–Fa` 曲线与局部 AUC；
- `A≤9 / 10–25 / >25` 的 Pd、miss、实例 IoU；
- post-embedding probe；
- 参数、MACs/FLOPs、latency、peak memory。

固定 `0.5` 的全部指标作为兼容性附表，但不能成为唯一机制证据。

---

## 7. P3：TPD 机制消融

### 7.1 分支消融

所有分支输出均为 `B×C×H/2×W/2`。

| ID | Context | Saliency | Rearrange | 机制问题 |
| --- | ---: | ---: | ---: | --- |
| A0 |  |  |  | MaxPool baseline |
| A1 | ✓ |  |  | 背景上下文本身是否有效 |
| A2 |  | ✓ |  | 局部异常是否提高 Pd 但抬高 Fa |
| A3 |  |  | ✓ | 先重排后压缩是否有效 |
| A4 | ✓ | ✓ |  | 背景能否约束显著性 |
| A5 | ✓ |  | ✓ | 上下文与保真是否互补 |
| A6 |  | ✓ | ✓ | 两类目标敏感分支是否冗余 |
| A7 | ✓ | ✓ | ✓ | 完整分支集合 |

局部显著性第一版固定为：

\[
F_{sal}=\mathrm{MaxPool}_{2,2}(F)-\mathrm{AvgPool}_{2,2}(F),
\]

避免 `|F-AvgPool_3(F)|` 后仍未定义下采样路径。

### 7.2 融合与门控

| Fusion | 必做控制 |
| --- | --- |
| equal sum | 无可学习权重 |
| concat + `1×1` | 常规融合基线 |
| learned static logits | 控制“只是学到全局分支偏好” |
| image-dependent GAP gate | 原场景自适应设想 |
| shuffled gate at test | 跨图像打乱门权重 |
| mean gate at test | 用数据集平均门替换动态门 |

必须记录每层每分支权重的 mean/std/entropy，并按背景复杂度、目标面积、SCR 分组。若 dynamic 不优于 static，或 shuffled/mean gate 不降性能，就删除“动态自适应”主张。

### 7.3 层位置

| 设置 | 解释 |
| --- | --- |
| P1 only / P2 only / P3 only | 单层边际贡献 |
| P4 only | 仅瓶颈/解码器效应 |
| P1+P2 | 推荐核心版本 |
| P1+P2+P3 | 所有 SCTB 前池化均替换 |
| P1+P2+P3+P4 | 同时改变 SCTB 输入与瓶颈 |

累计替换实验不能代替单层实验；否则无法判断究竟是哪一级有效。

---

## 8. P4：目标存活监督

只在 TPD-Core 独立有效后加入，且必须做完整 2×2：

| Downsample | No survival | Survival |
| --- | --- | --- |
| MaxPool | TBD | TBD |
| TPD | TBD | TBD |

推荐监督节点使用尺度名而非“第几级编码器”：

- `h2: x2`，64 channels；
- `h4: x3`，128 channels；
- 条件扩展 `h8: x4`，256 channels。

`MaxPool(Y)` 的语义是“该 cell 是否含目标”，不是精细分割。因此辅助任务应明确称为 cell-level survival classification，并比较：

1. max-presence label；
2. soft occupancy：`AvgPool(Y)`；
3. bilinear soft label。

损失写为归一化均值，避免与原始六项 BCE 的求和尺度混淆：

\[
\mathcal L=\mathcal L_{SCTransNet}
+\lambda_s\frac{\sum_i\omega_i\mathcal L_{surv}^{(i)}}{\sum_i\omega_i}.
\]

`λ_s` 只在 validation 上搜索，记录每项 loss 量级和梯度范数。第一轮不要同时加入 SoftIoU、边缘损失或 HF loss。

解释规则：

- 若 survival 对 MaxPool 与 TPD 提升相近，它是通用辅助训练技术；
- 若只有 TPD 提升且 post-embedding probe 同时改善，才支持协同机制；
- 辅助头推理时移除，报告训练参数和推理参数两种口径。

---

## 9. FG 的条件性最小实验

### 9.1 为什么不作为当前默认主线

最接近的工作包括：

- [FSCNet, Infrared Physics & Technology 2025](https://doi.org/10.1016/j.infrared.2025.105825)：Haar 高频补偿、FAGM、基于 SCTB 的 G-SCTB；
- [WaveTD, Infrared Physics & Technology 2025](https://doi.org/10.1016/j.infrared.2025.105850)：DWT/IDWT 与 frequency-aware channel attention；
- [MSDA-Net, TGRS 2025](https://arxiv.org/abs/2406.02037)：高频方向先验注入；
- [ARFC-WAHNet](https://arxiv.org/abs/2505.10595)：wavelet frequency enhancement downsampling 与高低频融合；
- [SWAN](https://arxiv.org/abs/2508.01322)：Haar wavelet convolution 与 attention。

因此“并行 Haar + 高频/低频门控 + 调制 SSCA”本身不足以自动建立强创新性。

### 9.2 可运行的 Q-only 接口

在 `x1…x4` 上并行 Haar DWT；默认输入下各 band 空间尺寸为 `128/64/32/16`，再以 `8/4/2/1` 的投影对齐到 Query 的 `16×16` 网格。

最小空间门：

\[
G_i\in\mathbb R^{B\times1\times16\times16},\qquad
\widehat Q_i=Q_i\odot(1+\alpha_i\tanh G_i),
\]

其中 `α_i` 初始化为 0，以确保初始函数等于 baseline，并允许增强与抑制。全局 `net.apply(weights_init_kaiming)` 会覆盖自定义初始化，因此 identity 初始化必须在该调用后重新执行。

Q-only 门必须放在 `Attention_org` 内对已生成 Q 调制。若提前调制 `emb_i`，K/V、CFN 和 residual 路径也同时改变，不再是 Q-only 消融。

### 9.3 最小消融

| ID | Prior | Gate | 目的 |
| --- | --- | --- | --- |
| F0 | 无 | 无 | baseline |
| F1 | 无信息/参数匹配 | static | 容量控制 |
| F2 | H only | spatial Q gate | 高频增量 |
| F3 | L only | spatial Q gate | 低频上下文增量 |
| F4 | H+L | spatial Q gate | 联合条件化 |
| F5 | H+L | shuffled gate | 输入依赖性 |
| F6 | H+L | mean gate | 动态性 |

第一轮禁止加入 decoder 回注、Q/K/V 同时修改、SoftIoU、gradient loss、HF auxiliary loss。F4 通过后再逐项添加，否则无法归因。

启动 FG 主线需要同时满足：

1. `Δ_freq` 在 post-Q/post-SCTB probe 上有独立增量；
2. edge-near 与 high-clutter Fa 不恶化；
3. dynamic gate 优于 static/mean/shuffled；
4. 与 FSCNet/FAGM/G-SCTB 的结构差异能够清楚写出并被直接实验验证。

---

## 10. TPD 与 FG 的组合门

只有两者单独都通过机制证据后，才做：

| Arm | TPD | FG |
| --- | ---: | ---: |
| Base |  |  |
| TPD | ✓ |  |
| FG |  | ✓ |
| Joint | ✓ | ✓ |

对任一主要指标 `M` 计算交互项：

\[
I=M_{Joint}-M_{TPD}-M_{FG}+M_{Base}.
\]

若 Joint 不优于最佳单方法、交互项无正向证据或效率代价过大，则保留单方向。不要把组合模型直接命名为 Full 后跳过 2×2。

---

## 11. 鲁棒性、失败与效率

### 11.1 最重要的压力测试

1. **stride 相位/平移**：输入平移 `(0,0),(1,0),(0,1),(1,1)`，反向对齐预测后比较一致性和 Pd；同时按目标 centroid parity 分组。
2. **局部 SCR**：按原图目标与局部背景环的 SCR 三分位分组。
3. **edge-near**：按目标到强背景边缘的距离分组。
4. **clutter**：按 GT 外局部梯度、方差或熵分组。
5. **轻度 corruption**：归一化输入上的 Gaussian noise、blur、gamma/增益；severity 在 validation 预注册。
6. **多目标与边界目标**：相邻、合并风险、位于图像边缘的目标。
7. **hard negative crop**：从测试图像中按固定算法抽取不含 GT、但梯度/熵最高的背景 crop，报告 FP components / MPix。

TPD 的 PixelUnshuffle/stride 分支尤其需要第 1 项；TPD saliency 与 FG 高频门尤其需要第 3、4、7 项。

### 11.2 失败类型必须量化

- baseline TP → 新模型 FN；
- 新增虚警；
- 目标膨胀/光晕；
- 两个目标合并；
- 建筑边缘、云边、树枝、海杂波误增强；
- 两者共同漏检。

案例图按预定义类型和固定排序选择，不只展示成功样本。

### 11.3 效率口径

本地当前代码的工程参考：

- trainable parameters：约 `11.326 M`；
- THOP 模块统计：约 `10.119 GMAC`（会漏函数式 QK/AV、插值和自定义门控，不能直接称完整 FLOPs）；
- RTX 3090、FP32、batch=1、`1×256×256` 的单点测试约 `42.13 ms / 23.74 FPS`，peak allocated 约 `141.45 MiB`。

正式表需用同一脚本、同一 GPU、固定 warmup/iterations 重测全部方法，并补自定义算子计数。至少报告：

- trainable / inference params；
- MACs 与 FLOPs 的定义；
- latency median、P10/P90；
- FPS；
- peak allocated memory；
- 训练时额外显存与 wall-clock。

---

## 12. 统计协议

### 12.1 Seed 与阶段

- smoke：1 seed，仅查 shape、loss、梯度和保存/恢复，不进论文；
- 筛选：3 个配对 seed，优先 NUDT-SIRST + IRSTD-1K；
- 确认性主实验：5 个配对 seed，三个数据集全部运行；
- 方法间使用完全相同 seed、split 和采样顺序。

建议一次性冻结 seeds，例如 `42, 123, 3407, 2025, 2026`。

### 12.2 报告方式

- 跨 seed：mean ± SD；
- 方法差值：paired mean difference；
- test 图像级 cluster bootstrap 10,000 次，保持同一图像内多个目标成组，报告 95% CI；
- `A≤4` 使用 exact/Clopper–Pearson 区间并标记 exploratory；
- 多数据集总结果用 dataset macro-average，避免 NUDT 目标数支配；
- 一个预注册主要终点；多个主要比较使用 Holm correction；
- 同时报告效应量、绝对 miss 变化和 Fa，不只报 p 值。

成功判据不预设虚构涨点，而以以下证据链为准：

1. 主要终点的配对 CI 支持实际增益；
2. Pd–Fa 权衡未恶化；
3. post-embedding probe 与最终 tiny Pd 同向；
4. 机制控制组排除容量、静态门或阈值解释；
5. 至少两个数据集重复，并披露失败数据集。

---

## 13. 外部与算子基线矩阵

| Baseline | 类别 | 为什么必须 | 公平约束 |
| --- | --- | --- | --- |
| SCTransNet | 原始架构 | 所有改动的因果基线 | 本地干净协议重训 |
| AvgPool | 简单算子 | 检查平滑是否已足够 | 相同层与通道 |
| Strided Conv | 可学习算子 | 排除“任何可学习下采样都有效” | 参数/采样相位说明 |
| MaxBlurPool | 抗混叠 | 检查增益是否来自 shift stability | 固定 blur kernel |
| [SPD-Conv](https://arxiv.org/abs/2208.03641) | space-to-depth | PixelUnshuffle 分支的最近结构 | 相同 `4C→C` 投影 |
| [Content-Adaptive Downsampling](https://openaccess.thecvf.com/content/CVPR2023W/ECV/html/Hesse_Content-Adaptive_Downsampling_in_Convolutional_Neural_Networks_CVPRW_2023_paper.html) | 动态下采样 | 动态性近邻 | 若可运行则按同 split 重训 |
| FSCNet | 频率 + SCTB | FG 最接近工作 | 优先官方实现同协议重训 |
| MSDA-Net | 高频方向先验 | FG 高频注入近邻 | 同 split 或单列 paper protocol |
| WaveTD | 小波下采样/频率注意 | TPD/FG 共同近邻 | 不混用不同 split 的数值 |
| ARFC-WAHNet | wavelet downsampling + H/L fusion | TPD/FG 共同近邻 | 标清代码与版本 |
| SWAN | wavelet + attention | FG 当前强近邻 | 若无同协议结果则单列 |

外部论文中的数字只有在数据划分、输入尺寸、训练数据、阈值和 evaluator 全部一致时才可放入同一主表；否则只放“reported result”附表并加 dagger，不能与本地重训结果直接排名。

---

## 14. 结果表模板

### 14.1 主比较

| Dataset | Method | Stages | Seed | Params (M) | GMAC | mIoU ↑ | nIoU ↑ | F1 ↑ | Pd ↑ | Fa (1e-6) ↓ | Pd `A≤9` @ Fa* ↑ | Tiny miss ↓ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TBD | SCTransNet | — | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | MaxBlurPool | P1–P2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | SPD-Conv | P1–P2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | TPD-Core | P1–P2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | FG-Core | conditional | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 14.2 瓶颈与机制链

| Model | Hook | Local CNR ↑ | Probe AUPRC ↑ | Probe Pd@FPR ↑ | SR@95 ↑ | SR@99 ↑ | Final Pd `A≤9` ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | pre-pool | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline | post-encoder | TBD | TBD | TBD | TBD | TBD | TBD |
| Baseline | post-embedding | TBD | TBD | TBD | TBD | TBD | TBD |
| TPD | post-encoder | TBD | TBD | TBD | TBD | TBD | TBD |
| TPD | post-embedding | TBD | TBD | TBD | TBD | TBD | TBD |

### 14.3 门控必要性

| Fusion | Dynamic input | Test intervention | Pd `A≤9` @ Fa* ↑ | Fa ↓ | Gate entropy | Interpretation |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| Equal |  | — | TBD | TBD | — | TBD |
| Static learned |  | — | TBD | TBD | TBD | TBD |
| Dynamic | ✓ | — | TBD | TBD | TBD | TBD |
| Dynamic | ✓ | shuffled | TBD | TBD | TBD | TBD |
| Dynamic | ✓ | mean gate | TBD | TBD | TBD | TBD |

### 14.4 Survival 2×2

| Downsample | Survival | Post-embedding AUPRC ↑ | Pd `A≤9` @ Fa* ↑ | Fa ↓ | Training overhead | Inference overhead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MaxPool |  | TBD | TBD | TBD | TBD | TBD |
| MaxPool | ✓ | TBD | TBD | TBD | TBD | TBD |
| TPD |  | TBD | TBD | TBD | TBD | TBD |
| TPD | ✓ | TBD | TBD | TBD | TBD | TBD |

### 14.5 鲁棒性

| Condition | Severity/bin | Method | Pd `A≤9` ↑ | Fa ↓ | ΔmIoU | Failure count |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1-px shift | TBD | TBD | TBD | TBD | TBD | TBD |
| low SCR | bottom tertile | TBD | TBD | TBD | TBD | TBD |
| edge-near | TBD | TBD | TBD | TBD | TBD | TBD |
| high clutter | top tertile | TBD | TBD | TBD | TBD | TBD |
| hard negative crop | TBD | TBD | — | TBD | — | TBD |

---

## 15. 执行队列与停止条件

| Priority | Experiment | 依赖 | 进入下一阶段条件 | 停止/转向条件 |
| --- | --- | --- | --- | --- |
| P0 | 数据、split、evaluator、runner | 无 | 单元测试全部通过，baseline 可复现 | 任一 test leakage / shape mismatch 未解决 |
| P1 | 基线逐节点 probe | P0 | 定位 `D_pool/D_embed/Δ_freq` | 没有可重复瓶颈则暂停结构设计 |
| P2 | 算子 screen | P1 | TPD-Core 优于通用/容量对照 | 只靠更高 Fa 换 Pd，停止 TPD claim |
| P3 | TPD branch/gate/stage | P2 | 分支与动态机制有独立证据 | dynamic≈static/shuffled，则简化模型 |
| P4 | Survival 2×2 | P3 | 与 TPD 有协同且 post-embedding 改善 | 对两种下采样同幅增益，则降格为通用 aux |
| P5 | robustness/failure/efficiency | P3/P4 | 复杂背景与相位稳定性可接受 | Fa 或 shift sensitivity 明显恶化 |
| P6 | FG-Core | P1 + novelty audit | 独立频率增量且区别于 FSCNet | 与近邻同构或 probe 无增量则停止 |
| P7 | TPD×FG / dual-resolution | 两条单线均通过 | 正交交互与成本可接受 | 无交互则保留最佳单线 |

建议计算资源顺序：

1. 单 seed smoke，不产出论文数字；
2. NUDT-SIRST + IRSTD-1K 做算子筛选；
3. shortlist 后用 3 seeds 做机制消融；
4. 冻结模型后，三个数据集 × 5 seeds 做确认性主实验；
5. 最后才运行外部强基线和联合模型。

---

## 16. No-fabrication status

本文件没有生成或假设任何模型增益、显著性、排名或实验结果。所有 `TBD` 必须由：

1. 按冻结协议实际运行得到的结果；或
2. 协议完全匹配且来源已核验的公开结果

填写。当前仓库已有日志因 test-set checkpoint selection、数据尺寸异常、Pd/Fa 实现问题以及缺少 run manifest，不可直接作为正式论文结果。
