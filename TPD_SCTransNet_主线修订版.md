# TPD-SCTransNet 主线修订版：面向跨层 Token 化的目标保真下采样

> 状态：已完成 NUDT-SIRST 单 seed formal800 与 TPD-Clean-v2 800-epoch 四候选筛选；正式判定仍为 `INCONCLUSIVE_MIXED_TRADEOFF`
> 主线：Keep–Context–Saliency TPD 与“浅层 tokenization 的目标保真”创新主线保持不变；多种子证据前不替换主线
> 核心模块：TPD-PE（Target-Preserving Downsampling Patch Embedding）  
> 当前阶段：优化现有 Keep 投影、残差注入与融合；显式 NER 代码已隔离完成，但因下一模块门槛未通过而不启动正式训练

## 1. Pilot 后的主线判定

研究问题仍然成立，但“TPD 三分支结构就是核心创新”的结论需要撤回到待证状态。此前改变的是问题定位：

```text
原命题：编码器 MaxPool 是主要目标信息瓶颈
修订命题：进入 SCTB 前的大跨度浅层 tokenization 是主要目标信息瓶颈

原放置：TPD 替换 P1/P2/P3 MaxPool
修订放置：TPD-PE 替换 emb1/emb2；Pool-TPD 作为位置对照
```

预验证否定的是“MaxPool 是主要瓶颈”这一具体归因，并没有否定“目标保真下采样”这一研究问题；但新完成的 100-epoch pilot 又表明，当前 TPD-PE 不能仅凭动机升级为第一创新点：

- 固定阈值 Pd-best 下，SPD 与 TPD 均为 `184/189`、tiny-Pd 均为 `39/39`，但 SPD 的 Fa 为 `1.80e-5`，低于 TPD 的 `2.87e-5`，mIoU 也更高；
- Progressive 达到 `186/189`、Fa `1.56e-5`，并在四模型联合 Pd–Fa 前沿的 22 个离散点中占 13 个；
- TPD 只保留 2 个狭窄的低-Fa Pareto 点，且相应阈值接近 1，存在明显校准敏感性。

因此当前主线应表述为：**目标保真 tokenization 是待验证的问题主线，TPD、Progressive 与 SPD 是竞争性实现；只有 TPD 在全新 800-epoch、多 seed 对照中稳定胜出，TPD 才能成为方法主线。** 在此之前不得用 NER、FG 或额外损失掩盖 TPD 核心未过门的事实。

候选论文名称（仅在上述门槛通过后启用）：

> **TPD-SCTransNet: Target-Preserving Downsampling for Multi-Scale Feature Tokenization in Infrared Small Target Detection**

推荐一句话主线：

> SCTransNet 的全层级交互依赖统一网格上的多尺度表征，但浅层大跨度 tokenization 会显著削弱微小目标证据；TPD-PE 在特征投影到 SCTB 公共网格时联合保留空间相位、局部异常响应与背景上下文。

### 1.1 与成熟论文同层级的方案表述

> 为解决红外微小目标在全层级交互前被大跨度特征 tokenization 削弱的问题，本工作拟提出目标保真下采样网络 TPD-SCTransNet。该网络首先利用层次化 TPD Patch Embedding，将一次性大步长投影分解为空间相位重排、局部异常保持与背景上下文建模相协同的渐进下采样，使浅层目标证据以更可靠的形式进入 SCTB。进一步地，网络复用 TPD 过程中形成的多尺度中间证据，构建嵌套目标证据中继，并由解码语义对其进行同尺度空间调制，在绕过强压缩恢复目标定位信息的同时抑制背景边缘传播。该方法将通过统一网格表征诊断、位置与容量匹配消融、尺寸分组的 Pd–Fa 评估以及复杂背景虚警分析进行验证。

这段表述中的每一部分都必须由对应实验支持；在端到端实验完成前使用“拟提出”“将验证”，不能提前写“有效性和优越性”。

## 2. 主线修订的真实证据

预验证使用用户指定的历史最优 checkpoint，而不是固定 epoch 800：

| 数据集 | 最优 checkpoint | Test 图像 | 目标数 | $A\leq9$ |
| --- | ---: | ---: | ---: | ---: |
| NUAA-SIRST-clean213 | epoch 642 | 213 | 262 | 35 |
| NUDT-SIRST | epoch 714 | 664 | 945 | 259 |
| IRSTD-1K | epoch 586 | 201 | 297 | 29 |

所有节点使用相同 stride-16 标签和相同 probe 任务。正的 loss 表示转换后的目标可分离性下降。

| 转换 | NUAA AP / tiny rank loss | NUDT AP / tiny rank loss | IRSTD AP / tiny rank loss |
| --- | ---: | ---: | ---: |
| `x1→p1` | -0.0784 / 0.0014 | -0.0497 / 0.0028 | -0.0590 / -0.0030 |
| `x2→p2` | 0.0823 / -0.0040 | -0.0130 / -0.0001 | -0.0125 / -0.0019 |
| `x1→emb1` | **0.3633 / 0.1712** | **0.5340 / 0.2471** | **0.3143 / 0.0593** |
| `x2→emb2` | **0.4068 / 0.1042** | **0.4329 / 0.0495** | **0.2246 / 0.0559** |

结论：

- P1/P2 的 Pool rank loss 接近 0；NUAA 的 Pool AP 还会随 probe seed 翻转。
- `emb1` 在 3/3 数据集表现出显著目标信息损失。
- `emb2` 在三数据集同向，其中 NUAA、IRSTD 超过预设 0.05 rank 实际效应阈值，NUDT 为 0.0495。
- 不经 probe 训练的标准化能量 rank 同样支持 `emb1/emb2`。

证据边界：这些 checkpoint 曾依据 test mIoU 保存，因此只能用于内部路线筛选；论文中的机制结论必须用全新、validation-best、test 不可见的训练结果重新确认。

## 3. TPD 创新点的正式定义

### 3.1 SCTransNet 中真正需要修改的接口

原始四条 Patch Embedding 为：

```text
x1: C=32,  H×W       -- Conv(k=16,s=16) --> 32,  H/16×W/16
x2: C=64,  H/2×W/2   -- Conv(k=8, s=8)  --> 64,  H/16×W/16
x3: C=128, H/4×W/4   -- Conv(k=4, s=4)  --> 128, H/16×W/16
x4: C=256, H/8×W/8   -- Conv(k=2, s=2)  --> 256, H/16×W/16
```

主模型第一版只替换 `emb1/emb2`：

```text
x1 -- TPD-PE(r=16) --> emb1
x2 -- TPD-PE(r=8)  --> emb2
x3 -- original PE  --> emb3
x4 -- original PE  --> emb4
```

输出通道与空间尺寸完全不变，因此 SSCA 的 `KV_size=480`、Reconstruct、CFN 和解码器接口均不改变。

### 3.2 TPD-v1 基础单元 TPD2（历史基线定义）

把一次 stride-16/8 压缩分解成若干个相位一致的 stride-2 单元。对输入 $F\in\mathbb R^{C\times H\times W}$：

背景上下文分支：

$$
F_{ctx}=\operatorname{AvgPool}_{2,2}(F).
$$

局部异常分支：

$$
F_{sal}=\operatorname{MaxPool}_{2,2}(F)-\operatorname{AvgPool}_{2,2}(F).
$$

空间相位分支：

$$
F_{keep}=\operatorname{GConv}_{1\times1}
\left(\operatorname{PixelUnshuffle}_{2}(F)\right),
$$

其中 grouped `1×1` 将每个原通道的四个空间相位由 `4C→C` 压缩。该操作只能称为“重排后学习压缩”，不能称为整体无损。

静态融合核心：

$$
F_{out}=\operatorname{Conv}_{1\times1}
\left([F_{keep},F_{ctx},F_{sal}]\right).
$$

最小实现：

```python
class TPD2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.phase_compress = nn.Conv2d(
            4 * channels,
            channels,
            kernel_size=1,
            groups=channels,
            bias=False,
        )
        self.fuse = nn.Conv2d(3 * channels, channels, kernel_size=1)

    def forward(self, x):
        context = F.avg_pool2d(x, 2, 2)
        saliency = F.max_pool2d(x, 2, 2) - context
        keep = self.phase_compress(F.pixel_unshuffle(x, 2))
        return self.fuse(torch.cat((keep, context, saliency), dim=1))
```

第一版在每个非末级 `TPD2` 融合后使用 ReLU，末级保持线性输出；所有 progressive 对照必须采用相同的深度和激活安排。

上述 grouped Keep 与 concat–`1×1` 融合是冻结的 TPD-v1 历史定义，不是当前
TPD-Clean-v3 正式训练代码。它继续作为主线来源和对照，不应被用来描述 v3
候选的具体拓扑。

### 3.3 TPD-Clean-v3 的当前 KCS 实现

TPD-Clean-v3 不改变 Keep–Context–Saliency 三个语义源，也不增加第四个并列
分支；它只在 `embeddings_1/2` 内优化 Keep 投影、Context 校准、残差注入和
融合。

$$
K=\operatorname{Conv}^{dense}_{1\times1}(\operatorname{PixelUnshuffle}_{2}(F)),
C=\operatorname{AvgPool}_{2,2}(F),
S=\operatorname{MaxPool}_{2,2}(F)-C.
$$

Full 候选使用 Context 归一化条件码和以 dense-SPD Keep 为锚点的有界融合：

$$
\widehat C=\tanh\!\left(\frac{C-\operatorname{Mean}_{HW}(C)}{\sqrt{\operatorname{Mean}_{HW}[(C-\operatorname{Mean}_{HW}(C))^2]+\epsilon}}\right),
F_{out}=K+\tanh(\alpha_s)\odot S+\tanh(\alpha_c)\odot(S\odot\widehat C).
$$

因此 Context 只能调制 Saliency 已有的空间支持，不能在其外部独立制造
响应；两个逐通道尺度均从零初始化，所以 step 0 与 dense SPD 严格等价。
容量对照 `tpd_clean_v3_sal_capacity` 令 $\widehat C=1$，但保留相同的尺度、
参数布局、初始化和残差范围；它不是完整三分支主模型，`tpd_clean_v3_full` 才是
KCS 主候选。

当前权威实现为 `model/tpd_clean_v3.py`；训练元数据必须保持
`mainline_contract=Keep-Context-Saliency` 和
`fourth_parallel_branch_added=false`。

### 3.4 层次化 TPD-PE

$$
\operatorname{TPD\text{-}PE}_{r}(F)
=
\underbrace{\operatorname{TPD2}\circ\cdots\circ\operatorname{TPD2}}
_{\log_2 r\text{ 次}}(F).
$$

因此：

- `emb1 = TPD2 × 4`；
- `emb2 = TPD2 × 3`。

分级实现避免一次 `PixelUnshuffle(16)` 产生 `256C` 通道。以 `emb1` 为例，一次重排会产生 8192 通道；分级实现每次最多仅为 `4C`。

### 3.5 参数与容量边界

原始 `emb1+emb2` 卷积参数约为 524,384。上述 TPD-PE Lite 的卷积参数约为 50,752，不依靠参数膨胀取得优势，理论 MAC 与原 Patch Embedding 大致同阶。

这里的 50,752 是 TPD-v1 Lite 的历史统计。当前 v3 Full 与 capacity control
采用相同的 dense Keep 和同构参数布局，整网实测参数均固定为 10,843,475；
二者差异不能解释为容量差异。

正式实验仍必须报告实测参数、MAC、吞吐、单图延迟和峰值显存；不能用理论估计替代结果。

如果 TPD-PE 明显低容量导致欠拟合，应增加一个参数匹配版本，但 Lite 与 capacity-matched 两个结果必须同时保留。

## 4. 当前创新结构

### 核心创新 I：TPD-PE

通过分级的 Context、Saliency、Keep 三类信息，在浅层特征映射到 SCTB 公共网格时保留目标证据。

### 机制贡献：瓶颈定位与表征验证

使用统一网格 probe、逐目标局部 rank 和无训练能量响应，区分 Pool、encoder 与 Patch Embedding 的信息变化。正式论文版必须基于干净训练重做，不能直接使用当前 test-best 诊断充当最终结果。

### 候选创新 II：TPD-NER 嵌套目标证据中继

TPD-PE 会自然产生随后被继续压缩的中间状态：

```text
emb1 path: h11[32,H/2] → h12[32,H/4] → h13[32,H/8] → emb1[32,H/16]
emb2 path: h21[64,H/4] → h22[64,H/8] → emb2[64,H/16]
```

`emb1/emb2` 终点负责进入 SCTB 建立全层级语义，中间状态则构成 Nested Evidence Relay（NER）：

$$
q_4=\Phi_4(P(h_{13}),P(h_{22}),\operatorname{Up}(d_5)),
$$

$$
q_3=\Phi_3(P(h_{12}),P(h_{21}),\operatorname{Up}(q_4),
\operatorname{Up}(d_4)),
$$

$$
q_2=\Phi_2(P(h_{11}),\operatorname{Up}(q_3),
\operatorname{Up}(d_3)).
$$

`q4→q3→q2` 是严格自顶向下的递推顺序，不得使用同级 decoder 输出反向生成控制该输出的 `q`，否则会形成循环依赖。证据宽度保持很小，候选值为 `Ce=8`，最终宽度由 validation 和容量对照决定。

当前 decoder 已经使用 CCA 对 skip 做全局通道门控，因此 NER 不再增加通道注意力，而是在 CCA 之后、与 upsampled decoder feature 拼接之前执行局部空间调制：

$$
a_s=\operatorname{CCA}(u_s,x_s'),
$$

$$
\widetilde a_s=a_s\odot
\left(1+\tanh(G_s(q_s))\right),
$$

$$
d_s=\operatorname{Conv}([\widetilde a_s,u_s]).
$$

`G_s` 输出单通道空间图，末层零初始化，使模型初始行为严格接近原基线。NER 传递低维 TPD 证据，而不是把完整浅层特征密集拼接进 decoder。

该设计解决的是一个明确矛盾：

- SCTB 的粗网格路径负责判断目标与背景的全层级语义关系；
- NER 绕过最终强压缩，保留目标空间相位和局部峰值；
- decoder 语义在同尺度决定哪些局部证据可以进入恢复过程，从而限制背景边缘引起的 Fa。

它不是普通 U-Net++：不创建额外的高分辨率 decoder 森林，每条连接都来自 TPD-PE 原本存在的中间节点，并对应一个明确的压缩前证据尺度。

NER 必须后于 TPD-PE 核心验证，并执行 $2\times2$ 交互对照：

| Tokenizer | Relay off | 相同拓扑 Relay on |
| --- | --- | --- |
| 参数匹配 Progressive PE | P | P+N |
| TPD-PE | T | T+N |

只有当 `T+N−T` 稳定大于 `P+N−P`，并改善 $Pd_{A\leq9}@Fa^*$ 而非以更高 Fa 换取 Pd 时，NER 才能列为 TPD 的第二创新。否则它只能降为通用 skip 技巧或被删除。

### 条件训练增强：目标存活监督

仅在 TPD-PE 核心通过后，测试 `emb1/emb2` 轻量存活头：

$$
S_i=\operatorname{Conv}_{1\times1}(emb_i),\qquad
Y_{16}=\operatorname{MaxPool}_{16}(Y),
$$

$$
\mathcal L=\mathcal L_{SCTransNet}+\lambda_s
\sum_{i\in\{1,2\}}\mathcal L_{surv}(S_i,Y_{16}).
$$

它必须做 `Original/TPD × no-aux/aux` 的 $2\times2$ 交互实验。若辅助监督对原始 PE 与 TPD-PE 增益相同，则只能称为通用训练技巧，不能列为 TPD 专属创新。

### 后置扩展：FG

FG 不是第一阶段主模型。只有 TPD-PE 已有效，且 Haar 分支相对空间保真、Sobel/Laplacian 或可学习高通对照仍提供独立增量，才进入 Query 频率门控。K/V frequency token、解码回注和高频损失继续后置。

## 5. Claim–Evidence 矩阵

| 主张 | 审稿问题 | 决定性证据 | 必要对照 | 失败后的主张边界 |
| --- | --- | --- | --- | --- |
| 浅层 tokenization 是主要瓶颈 | 为什么改 PE 而不是 Pool？ | 新训练 baseline 的 `x→emb` probe、rank、CNR | `x→pool`、`pool→encoder` | 只能称探索性现象 |
| TPD-PE 保留目标证据 | 是机制有效还是容量变化？ | post-embedding probe/rank 改善 | 原始 PE、同深度 progressive conv、SPD、参数匹配通用模块 | 若只胜原始 PE，收窄为改进 embedding |
| TPD-PE 改善最终检测 | 表征改善是否转化为任务收益？ | $Pd_{A\leq9}@Fa^*$、漏检、Pd–Fa 曲线 | 干净 SCTransNet、Pool-TPD | 若没有稳定任务收益，不作为论文核心方法 |
| 正确干预位置很关键 | TPD 放哪都一样吗？ | Pool-TPD 与 TPD-PE 位置对照 | P1/P2、emb1-only、emb2-only、emb1+2 | 不声称定位贡献 |
| 三分支存在协同 | 是否只是算子堆叠？ | 单支、双支、完整三支消融 | 参数匹配 generic multi-branch | 删除无效分支，收窄机制 |
| 收益不是更深网络带来的 | progressive 深度本身是否足够？ | 同深度、同激活、近参数 progressive conv | E-cap | 若打平，不能声称 TPD 机制特殊 |
| NER 与 TPD 存在协同 | 是否只是普通 skip？ | `Progressive/TPD × Relay off/on` 交互、gate 响应和 Fa | raw relay、无 gate、参数匹配 relay | 若交互不成立，降为通用组件或删除 |
| 存活监督与 TPD 协同 | 辅助 loss 是否为通用技巧？ | $2\times2$ 交互实验 | Original/TPD × no-aux/aux | 降为普通训练技巧 |

## 6. P0：正式实验协议门

在训练 TPD 前必须完成：

1. 修复 `PD_FA` 的连通域匹配错误，并用合成 mask 单元测试验证。
2. 固定 NUAA manifest；无法恢复 `Misc_111` 时，所有模型统一使用 clean213。
3. 每个数据集只从原 train 中划固定 internal validation；test 在结构、超参数和阈值冻结前不可见。
4. 每个 run 的主模型使用 **validation Pd-best checkpoint**，不是固定 epoch，也不能再用 test 选 best；具体按 `Pd 最大 → 同 Pd 时 Fa 最低 → tiny-Pd 最高 → mIoU 最高` 选择并保存为 `best.pth.tar`。另存 `best_miou.pth.tar` 只用于分割质量分析，不替代主 checkpoint。
5. 干净 baseline 至少运行 3 个训练种子，据此确定随机波动、实际效应阈值 $\delta$ 和固定虚警预算 $Fa^*$。
6. 固定数据增强、输入尺寸、优化器、训练预算、阈值校准和保存规则。
7. 保存每图预测、每目标匹配结果、配置、seed、checkpoint SHA256、参数和环境。

停止门：split、评估器、checkpoint 规则或阈值校准任一项未固定，不启动正式 TPD 对比。

## 7. P1：最小可证伪实验

### 7.1 Operator 与位置筛选

| 编号 | `emb1/emb2` 或放置位置 | 回答的问题 | 结果 |
| --- | --- | --- | --- |
| E0 | 原始大核 stride-16/8 PE | 干净基线 | TBD |
| E-prog | 同深度 progressive conv | 是否只是渐进下采样有效 | TBD |
| E-cap | 近参数通用 progressive 模块 | 是否只是容量/深度有效 | TBD |
| E-SPD | 分级 PixelUnshuffle + projection | Keep 分支是否已足够 | TBD |
| M-pool | 原三分支 TPD 仅放 P1/P2 | MaxPool 放置位置对照 | TBD |
| T-e1 | TPD-PE 仅替换 emb1 | 第一支路贡献 | TBD |
| T-e2 | TPD-PE 仅替换 emb2 | 第二支路贡献 | TBD |
| T-e12 | TPD-PE 替换 emb1+emb2 | 推荐主模型 | TBD |

进入正式确认需同时满足：

- `T-e12` 的 post-embedding probe/rank 优于 E0；
- $Pd_{A\leq9}@Fa^*$ 优于 E0，且优势超过 P0 测得的 seed 波动；
- 至少不弱于 E-prog、E-cap 和 E-SPD；
- Fa、整体 nIoU、训练稳定性和效率没有不可接受退化。

若 E-SPD 与完整 TPD-PE 打平，删除 Context/Saliency 冗余分支。若 E-prog 打平，不能把收益归因为 TPD 三分支机制。

### 7.2 正式核心确认

结构冻结后，至少保留：

```text
E0 / E-cap / E-SPD / M-pool / T-e12
```

在三个数据集、至少三个训练种子上运行。每个 run 独立使用 validation Pd-best checkpoint，并同时保留 mIoU-best 辅助 checkpoint；阈值只在 validation 校准；最后对 test 评估一次。

报告：

- 跨 seed 均值与标准差；
- 相对同 seed baseline 的配对差；
- 按图像聚类 bootstrap 95% CI；
- 参数、MAC、延迟、吞吐和峰值显存。

## 8. P2：机制消融顺序

只有 P1 通过后，按顺序执行：

1. `emb1 / emb2 / emb1+2 / emb1+2+3` 位置消融；
2. Keep-only、Context-only、Saliency-only；
3. 三组双分支与完整三分支；
4. sum、concat+projection、静态可学习权重；
5. 局部动态门控及 mean/shuffled-gate 反事实；
6. Survival supervision 的 $2\times2$ 交互；
7. TPD-NER 的 $2\times2$ 交互、路径和空间门控消融；
8. 最后才考虑 FG 或双分辨率 SCTB。

第一版不使用 GAP 动态门控。GAP 对几个像素目标的贡献极小，只能称为 scene-conditioned，不能直接称为 target-adaptive。

## 9. 指标与困难样本

### 主指标

- IoU、nIoU、F-measure；
- Pd–Fa 曲线；
- validation 预先确定的 $Pd@Fa^*$；
- 参数、MAC、延迟、吞吐和显存。

### 尺寸分组

- 主 tiny 组：$A\leq9$；
- $10\leq A\leq25$；
- $A>25$；
- $A\leq4$ 仅作探索性结果，因为三数据集样本均较少。

每组至少报告 Pd、miss count 和可计算时的 object-level IoU。不能只报告总体 IoU。

### 机制指标

- `x_i→emb_i` 的统一网格 probe AP；
- 逐目标 local rank、robust CNR；
- TPD 前后目标响应变化；
- 三分支输出与门控图的目标/背景响应；
- 复杂边缘背景上的 false alarms。

## 10. 结果表模板

### 主结果

| Method | Dataset | Seed | IoU ↑ | nIoU ↑ | F1 ↑ | $Pd@Fa^*$ ↑ | $Pd_{A\leq9}@Fa^*$ ↑ | Fa ↓ | Params | MACs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SCTransNet-clean | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| E-cap | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| E-SPD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Pool-TPD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TPD-PE | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 机制表

| Dataset | Variant | `x1→emb1` AP loss ↓ | `x2→emb2` AP loss ↓ | Tiny rank loss ↓ | Energy rank loss ↓ | $Pd_{A\leq9}@Fa^*$ ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TBD | Original PE | TBD | TBD | TBD | TBD | TBD |
| TBD | E-cap | TBD | TBD | TBD | TBD | TBD |
| TBD | E-SPD | TBD | TBD | TBD | TBD | TBD |
| TBD | TPD-PE | TBD | TBD | TBD | TBD | TBD |

### 分支消融

| Keep | Context | Saliency | Fusion | Params | Tiny Pd ↑ | Fa ↓ | Post-embedding rank ↑ | 解释 |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| ✓ |  |  | — | TBD | TBD | TBD | TBD | Keep 是否足够 |
|  | ✓ |  | — | TBD | TBD | TBD | TBD | Context 单支 |
|  |  | ✓ | — | TBD | TBD | TBD | TBD | Saliency 单支 |
| ✓ | ✓ |  | concat | TBD | TBD | TBD | TBD | TBD |
| ✓ |  | ✓ | concat | TBD | TBD | TBD | TBD | TBD |
|  | ✓ | ✓ | concat | TBD | TBD | TBD | TBD | TBD |
| ✓ | ✓ | ✓ | concat | TBD | TBD | TBD | TBD | 完整核心 |

所有 `TBD` 必须由真实运行、匹配协议的公开结果或可审计产物填写，不得补造。

## 11. 实现位置与验证

建议新增：

```text
model/
├── tpd.py
│   ├── TPD2
│   └── TPDPatchEmbedding
└── tpd_relay.py
    ├── NestedEvidenceRelay
    └── RelayUpBlock
```

修改：

- `model/SCTransNet.py::ChannelTransformer.__init__`：仅替换 `embeddings_1/2`；
- `ChannelTransformer.forward`：在 NER 阶段额外返回 `h11/h12/h13/h21/h22`；
- `UpBlock_attention`：NER 阶段在现有 CCA 后、decoder 拼接前加入零初始化空间残差调制；
- config：增加 `embedding_type`、替换层级和消融开关；
- train：Pd-primary validation-best checkpoint、mIoU 辅助 checkpoint 与完整 provenance；
- metrics：修复 Pd/Fa 连通域匹配并增加尺寸分组。

实现单元测试：

1. `r=16/8` 输出 shape、通道和动态输入尺寸；
2. 单像素 impulse 的三个分支采样相位一致；
3. 每个分支均有非零梯度；
4. padding 不进入 probe、loss 或指标；
5. TPD-PE 与原 SCTB、Reconstruct、decoder 前向兼容；
6. NER 的零初始化输出与无 NER baseline 数值等价；
7. `q4→d4→q3→d3→q2→d2` 无循环依赖，并统一按参考张量 `size` 插值；
8. 参数、MAC、延迟统计可复现。

## 12. 执行优先级

| 优先级 | 实验 | 依赖 | 成本 | 停止条件 |
| --- | --- | --- | --- | --- |
| P0 | 评估器、split、validation-best 与 baseline seeds | 无 | 中 | 协议未固定则停止 |
| P1 | E0/E-prog/E-cap/E-SPD/M-pool/T-e1/T-e2/T-e12 | P0 | 中 | T-e12 不胜简单对照则简化/停止 |
| P2 | 三数据集三 seeds 核心确认 | P1 通过 | 高 | tiny Pd 增益不稳定或靠 Fa 换取则停止 |
| P3 | 位置、分支、融合消融 | P2 通过 | 中 | 删除无贡献组件 |
| P4 | TPD-NER 嵌套证据中继 | P2–P3 | 中 | 不胜相同拓扑普通 relay 或提高 Fa 则删除 |
| P5 | Survival supervision | P3–P4 | 中 | 无 TPD 专属交互则降为训练技巧 |
| P6 | FG 或双分辨率扩展 | P2–P5 | 高 | 无独立增量则不并入主模型 |

## 13. 当前可以与不能声称的内容

当前可以说：

> 探索性诊断显示，SCTransNet 的浅层大步长 Patch Embedding 会削弱微小目标的线性可分离性，因此将 TPD 应用于跨层 tokenization 过程。

当前不能说：

- TPD-PE 已经提高最终检测性能；
- 原三分支一定是最优结构；
- PixelUnshuffle 分支整体无损；
- MaxPool 是 SCTransNet 的主要瓶颈；
- 动态门控具有目标自适应性；
- FG 与 TPD 已被证明互补。

最终主线是：**保留 TPD 创新点，依据证据把 TPD 放到真正发生主要损失的 `emb1/emb2`，再用 Pool-TPD 和简单 progressive 对照证明“位置正确且机制必要”；只有核心成立后，才用 TPD-NER 将压缩途中存活的目标证据选择性送回 decoder。**
