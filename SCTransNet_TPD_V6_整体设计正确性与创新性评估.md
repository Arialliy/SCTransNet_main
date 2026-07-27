# SCTransNet–TPD V6 整体模型设计、技术正确性与创新性评估

- **评估日期**：2026-07-27
- **评估对象**：SCTransNet 基线、TPD V1–V6 主线、V6 TPD-PE、后续 NER / Survival / Query-only FG 规划
- **代码范围**：本地工作区 `/home/ly/SCTransNet_main` 的当前 `main` 分支及尚未提交的 V6 文件
- **结果范围**：本地冻结的 V5 正式结果、TPD-v1/SPD 锚点、V6 Full/Capacity × seeds 42/3407 的四个 800-epoch 终点、八份闭区间 sweep、正式 comparison、completion manifest 与三级复核结果
- **当前决策建议**：V6 正式工程闭环已完成；Gate B、E 通过，Gate A、C、D 未通过，正式裁决为 `ENGINEERING_GATE_FAIL`。本轮不授权 NER，下一步保持 K/C/S 三源主线，只优化 tokenizer 内部相位对齐与低虚警稳定性

> **证据边界**：本评估已直接核对本地 V6 源码、协议、单元测试、四组 checkpoint、逐轮日志、exact journal、八份 sweep、冻结 comparison JSON 与 completion 清单。四组训练均完成 `1..800` 连续记录，共 `3200/3200` epochs，12 份角色 checkpoint 均通过 CPU 严格加载；八份 sweep 均同时通过原冻结复核、strict 复核和 checkpoint-metric compatibility 复核。正式接受结果为 `authoritative_result_accepted=true`，但这表示产物与裁决可接受，不表示性能门通过。数据范围仍只是 NUDT-SIRST 官方训练集的 530/133 内部划分和两个配对 seed，未访问官方测试集，因此不能作多数据集或官方测试主张。

---

## 0. 核心结论

| 问题 | 结论 | 当前等级 |
|---|---|---:|
| 整体研究方向是否合理 | **合理。** 从 SCTransNet 的跨尺度 tokenization 入口解决浅层小目标信息压缩问题，干预位置有明确任务动机 | 良好 |
| 整体实验设计是否严谨 | **筛选协议设计较严谨且已完成正式工程闭环。** 单变量替换、配对初始化、同容量对照、SPD 精确起点和固定 Gate 均合理；V6 专用精确续训入口、轨迹等价测试、CPU/GPU 2/3 smoke、两卡串行 lane、训练与后处理独立源码锁、四任务训练、八份 sweep、comparison、completion 与三级复核均已完成 | 正式工程闭环 |
| V6 代码/公式是否正确 | **核心 forward 与当前公式一致。** 它消除了 V5 未显式约束的通道映射；状态布局和零尺度等价关系一致，bias、`eps` 和正式 AMP-off 精度边界已明确 | 条件通过 |
| V6 科学假设是否已经成立 | **尚未成立。** Gate B 证明 seed 42 存在有效 Pd–Fa 工作区间，Gate E 证明工程完整；但 Gate A 的联合固定阈值门槛、Gate C 的跨 seed 稳定性和 Gate D 的 Full/Capacity 结构优势均未建立 | 未建立 |
| 当前方案是否是正确的下一步 | **是，但对象已从“继续跑 V6”变为“实现下一版 K/C/S 内部优化”。** 保持主线和三个语义源不变，优先修复 phase collapse、低虚警阈值断层和跨 seed 不稳定，不提前叠加 NER | 正确 |
| 当前是否满足创新 | **具备中等模块级创新潜力，但尚不足以认定论文核心创新已建立** | 潜在增量创新 |
| 当前能否启动 NER | **不能。** 正式裁决已明确 `ner_stage_authorized=false` | 不授权 |

最精确的判断是：

> **V6 已完成“实现—训练—评测—接受”的工程闭环，但没有通过联合性能门；它是诊断清楚的中间候选，不是最终模型。**

创新判断则是：

> **目前满足“可形成投稿增量贡献”的潜力，但不满足“核心创新已经被证据确立”的标准。**

当前“已实现并完成正式评测的模型”和“规划整体模型”必须分开：

```text
当前已实现并完成正式评测：
SCTransNet + V6 TPD-PE（仅 embeddings_1/2；Gate A/C/D 未通过）

后续分阶段候选：
K/C/S tokenizer 内部优化
→ (five-node NER)
→ (Survival)
→ (Query-only FG)
```

五节点 NER 表示五个中间证据节点，不表示 V6 tokenizer 具有五个并列分支；
V6 tokenizer 始终只有 Keep、Context、Saliency 三种语义源。括号中的后续
模块只有在各自门槛通过后才并入下一阶段候选，不代表现在已经组成统一模型。

---

## 1. 整体模型设计处于什么位置

### 1.1 SCTransNet 的原始主线

SCTransNet 是 U 型编码器–解码器结构，其核心增量是将多尺度 encoder 特征送入 Spatial-channel Cross Transformer Block（SCTB），利用 SSCA 和 CFN 在多个 encoder 层级之间交换局部空间信息与全局通道语义，再把重建特征送回 decoder skip path。

当前 TPD 研究只替换：

```text
mtc.embeddings_1
mtc.embeddings_2
```

以下部分保持不变：

```text
CNN encoder
SCTB / SSCA / CFN
embeddings_3 / embeddings_4
decoder
loss
output heads
data split / augmentation / optimizer / metric
```

这是正确的受控实验设计，因为它把因变量尽量限定在浅层 patch tokenization。

### 1.2 必须准确界定：V6 不是整网 encoder 下采样替换

SCTransNet 主编码器仍然使用：

```python
x2 = down_encoder1(MaxPool(x1))
x3 = down_encoder2(MaxPool(x2))
x4 = down_encoder3(MaxPool(x3))
d5 = down_encoder4(MaxPool(x4))
```

V6 替换的是 encoder 特征进入 ChannelTransformer 前的 patch embedding：

```text
x1 --stride 16 tokenizer--> emb1
x2 --stride  8 tokenizer--> emb2
```

V6 将 stride 16/8 分别分解为 4/3 个连续 2× block，因此实现还要求
`x1` 的高宽可连续除以 16、`x2` 的高宽可连续除以 8；正式 256×256
输入满足该约束。两路最终都落到 16×16 的 SCTB 公共 token grid。

因此，V6 的准确定位应是：

> **面向 SCTransNet 跨尺度交互路径的目标保真浅层 tokenization。**

不宜写成：

- “替换了 SCTransNet 全部下采样”；
- “消除了 encoder 的小目标丢失”；
- “实现了整网无损下采样”。

原因有两点：

1. 主 CNN encoder 的 MaxPool 未被替换；
2. `PixelUnshuffle` 本身可逆，但后续 `4C → C` 的 `1×1` 压缩不是可逆映射，严格意义上并不“无损”。

建议论文使用以下措辞：

- `phase-explicit tokenization`；
- `rearrangement-before-compression`；
- `target-preserving tokenization`；
- `fine-detail-aware shallow tokenization`。

避免使用绝对化的 `lossless downsampling`。

### 1.3 原模型中的双重 identity bypass

本地基线代码中，ChannelTransformer 重建后先执行：

```python
x_i = reconstruct(encoded_i) + en_i
```

回到 `SCTransNet.forward` 后又执行：

```python
x_i = x_i + f_i
```

其中 `f_i` 就是进入 ChannelTransformer 前的同一 encoder feature。因此 decoder skip path 实际接收近似：

```text
Transformer reconstruction + 2 × original encoder feature
```

这是一个重要结构边界：TPD/V6 对 tokenizer 的改动旁边存在很强的 identity 通路，可能稀释 tokenizer 的实际贡献。

当前不应在 V6 中修改这条通路，因为这样会同时改变基线架构，破坏 V5/V6 与 SPD/TPD-v1 的受控比较。但在 tokenizer 主线完成后，可以把“外层第二次 identity addition”作为独立架构消融，不能与 V6 同时改。

---

## 2. V5 为什么失败，V6 修复了什么

### 2.1 V5 的核心结构问题：数值维度相同，不等于语义基相同

V5 的三个源为：

```text
K = Conv1x1(PixelUnshuffle(X))
C = AvgPool2(X)
S = MaxPool2(X) - C
```

随后将原始 `S` 直接作为 residual 加到 `K`：

```text
Y = K + selected(S)
```

问题在于：

- `K[:, o]` 是全部输入通道及四个空间 phase 的线性组合；
- `S[:, o]` 仍表示原输入特征第 `o` 个通道上的局部显著性；
- 两者虽然都有 `C` 个通道，但第 `o` 个通道不必具有相同语义。

因此，V5 隐含了一个没有保证的假设：

```text
input-channel basis ≡ dense Keep output basis
```

这不是 tensor shape 错误，而是**表示空间不对齐**。

### 2.2 V5 结果未排除该问题，但不能单独完成归因

本地冻结产物中的 V5 正式结果为：

| 工作点 | Pd | Fa | mIoU |
|---|---:|---:|---:|
| seed 42，Pd-best | 187/189 | 7.4573e-6 | 0.917607 |
| seed 42，mIoU-best | 186/189 | 2.5240e-6 | 0.935188 |
| seed 3407，最佳点 | 187/189 | 2.8682e-6 | 0.929938 |

在固定阈值 `0.5` 下，V5 seed 42 Pd-primary 相对 TPD-v1 的
`188/189、1.0325e-6、0.933647` 同时表现为更低 Pd、更高 Fa 和更低 mIoU。
在 Pd-primary 的五个预注册 Fa budget 上，V5 Full 的工作点又全部被
冻结 SPD 覆盖；seed 42 capacity control 还在三个预注册比较点严格覆盖
Full。因此 V5 没有贡献新的有用高-Pd工作区间，Gate A–D 失败、NER
不授权是正确决定。
这里不使用“V5 全曲线相对 TPD-v1 没有任何 Pareto 点”的更强表述，因为
两代 sweep 的闭区间端点采样范围并不完全相同。

这些性能结果说明 V5 没有达到晋级门槛，但不能单独证明失败原因就是表示
空间不对齐；V6 相对 V5 的结果也会同时受到 Saliency 投影与融合公式变化
影响。V5 的表示风险来自代码与公式审查，性能结果只是继续验证 V6 的动机，
不是已经完成的因果证明。

### 2.3 V6 的结构修正：从 Keep 权重派生共享输出坐标

设 `PixelUnshuffle2(X)` 对输入通道 `c` 产生四个 phase：

\[
z_{c,0},z_{c,1},z_{c,2},z_{c,3}.
\]

Keep 输出为：

\[
K_o=\sum_c\sum_{p=0}^{3}W_{o,c,p}z_{c,p}+b_o.
\]

V6 定义：

\[
W^t_{o,c}=\sum_{p=0}^{3}W_{o,c,p}.
\]

再将 Context 和 Saliency 投影到 Keep 的输出通道基：

\[
C_a=\operatorname{Conv}(C_0;W^t),\qquad
S_a=\operatorname{Conv}(S_0;W^t).
\]

这使 `C_a`、`S_a` 的输出通道由 Keep 权重约束，消除了 V5 中未显式约束的
通道映射。这里能够严格成立的是“共享权重坐标和输出通道对应”，而不是已经
由理论证明 `C_a`、`S_a` 与 `K` 具有完全相同的学习语义。

---

## 3. V6 技术正确性审查

### 3.1 Context 相位绑定投影是数学上严格成立的

令一个低分辨率 Context 特征 `C0` 在四个 phase 上重复：

\[
z_{c,p}=C_{0,c},\quad p=0,1,2,3.
\]

代入 Keep 投影：

\[
\begin{aligned}
K_o-b_o
&=\sum_c\sum_p W_{o,c,p}C_{0,c}\\
&=\sum_c\left(\sum_pW_{o,c,p}\right)C_{0,c}\\
&=\sum_cW^t_{o,c}C_{0,c}.
\end{aligned}
\]

因此 `Wt = phase-sum(Wk)` 精确复现 phase-constant/DC 输入经过原 Keep
权重后的**无偏线性响应**。若比较完整 Keep 输出，应再加 `b_o`；V6 的
`C_a/S_a` 按设计使用 `bias=None`，避免将 Keep bias 重复注入 residual。
它具有以下优点：

- 不增加参数；
- 不增加 buffer；
- 与 Keep 的输出语义绑定；
- Full 和 Capacity 可以保持完全相同的 state layout；
- 可以构造严格配对初始化。

这一部分是 V6 最扎实的理论点。

### 3.2 Saliency 投影是合理假设，但不是唯一正确投影

V6 先计算：

\[
S_0=\max_pz_p-\frac14\sum_pz_p.
\]

该操作把四个 phase 压缩为一个标量显著性幅值，然后使用相同的 `Wt` 投影。数学上，这等价于做出以下建模假设：

> `S0` 是一个 phase-invariant 特征，可视为在四个 phase 上等值复制，再通过 Keep 权重投影。

该假设是自洽的，但存在两个风险：

1. **phase location 丢失**：只知道 2×2 cell 中存在峰值，不知道峰值位于哪个 phase；
2. **phase-sum cancellation**：若某个 Keep 输出通道依赖 phase 间差异，四个权重正负抵消后 `Wt≈0`，则 Saliency 可能被压弱。

换言之：

- `Wt` 对 Context 的解释非常自然，因为 Context 属于 phase-constant/DC 模式；
- `S0=max-mean` 更接近 phase-contrast 的幅值，却也使用 DC 响应投影，这一选择需要实验验证。

因此不能说 V6 的 Saliency 对齐“理论上唯一正确”；应说它是一个**零参数、可证伪、与 Keep 权重绑定的近似对齐策略**。

### 3.3 “均值中性”表述需要收紧

V6 定义：

\[
Q=\tanh\left(
\frac{C_a-\mu(C_a)}
{\sqrt{\operatorname{mean}((C_a-\mu(C_a))^2)+\epsilon}}
\right),\qquad \epsilon=10^{-6},
\]

\[
V=0.5(Q-\mu(Q)),
\]

\[
a=\tanh(s),
\]

\[
H=1+0.5(1-|a|)V,
\]

\[
R=S_a\,aH.
\]

忽略有限精度舍入时，可以严格保证：

\[
\mu(V)=0,
\]

因此：

\[
\mu(H)=1.
\]

准确含义是：

> **Context headroom 的空间平均增益为 1；Context 在空间上重新分配 Saliency 增益。**

但一般不能推出：

\[
\mu(S_aH)=\mu(S_a).
\]

因为 `Sa` 与 `H` 可能相关：

\[
\mu(S_aH)=\mu(S_a)\mu(H)+\operatorname{Cov}(S_a,H).
\]

所以旧版表述中的“mean-neutral”最多只能用于：

- `zero-mean modulation`；
- `mean-one gain/headroom`；
- `gain-map mean-neutrality`。

不应扩展为：

- residual mean-neutral；
- feature mean-preserving；
- energy-neutral；
- fusion output mean-preserving。

当前源码 metadata 和协议标题已收紧为 `zero-mean context gain
redistribution`：它只指 `V` 的空间均值为零以及 `H` 的空间均值为一，
不指 residual 或最终输出保持均值。

模型描述已从：

```text
phase-tied, mean-neutral KCS fusion
```

收紧为：

```text
phase-tied KCS fusion with zero-mean context gain redistribution
```

或中文：

```text
相位绑定、零均值 Context 增益重分配的 KCS tokenization
```

### 3.4 有界残差设计是成立的

由于 `Q∈[-1,1]`，再次中心化并乘以 0.5 后有：

\[
V\in[-1,1].
\]

于是：

\[
H=1+0.5(1-|a|)V.
\]

可得：

\[
0.5\le H\le1.5.
\]

令 `t=|a|∈[0,1]`，最坏情况下：

\[
|aH|\le t(1.5-0.5t)\le1.
\]

因此：

\[
|R|=|S_a||aH|\le|S_a|.
\]

这一设计避免 Context 将 Saliency residual 无界放大，技术上是合理的。

### 3.5 SPD 精确起点和配对对照设计正确

当：

\[
s=0\Rightarrow a=\tanh(0)=0,
\]

则：

\[
R=0,
\qquad Y=\operatorname{activation}(K).
\]

在共享 Keep 完整 state（`weight+bias`）时，V6 step 0 与 dense SPD
逐元素相同。本地测试还覆盖：

- `pixel_unshuffle` phase 顺序；
- `Wt` 精确求和轴；
- Full/Capacity state key 一致；
- 参数量一致；
- Full/Capacity 配对初始 state 逐 tensor 相同；
- 全模型六个输出在零 gate 时与 SPD 一致；
- CPU 两个 optimizer step、梯度与参数更新有限且非零，以及 strict reload。

因此，从**实现一致性**和**对照公平性**看，V6 设计是正确的。

2026-07-26 本地重新执行 V6 core、builder 和 closed-interval evaluator
三组定向测试，共 `18 tests`，全部通过；随后执行完整 CPU 两步 smoke，
`7 tests` 全部通过，用时 `578.507s`。该 smoke 已覆盖 Full/Capacity、
六输出、step-0 dense-SPD 等价、梯度、参数更新和内存 state-dict strict
reload；该次早期输出当时尚未保存为最终源码绑定报告，也不等价于物理
GPU smoke、磁盘 checkpoint 精确续训或正式性能验证。

V6 专用 exact entry 的 `10 tests + 12 subtests` 也已通过，其中轻量连续
三轮与“一轮后 exact resume 两轮”的模型、优化器、DataLoader generator、
Python/NumPy/PyTorch 随机状态和 metrics 轨迹一致。正式入口已强制
`800 epochs / eval_every=1 / workers=0 / AMP=false / eps=1e-6`；这些是
工程闭环证据，仍不是 V6 性能证据。

最终源码绑定 smoke 报告集随后完成并通过联合验证：

- `cpu_all.json`：Full 与 Capacity 各两个 optimizer step；
- `gpu2_full.json`：物理 GPU 2、RTX 5090、对应 UUID；
- `gpu3_capacity.json`：物理 GPU 3、RTX 5090、对应 UUID；
- 三份报告共享初始模型摘要
  `a608a0121075913f16c0842f2e20b170f598073e8b671f303e864d31d7bb301b`。

CPU 最终报告在限制为 4 个计算线程后，本轮控制台观察用时约 `5.7s`
（duration 未写入报告 schema）；此前 `578.507s` 主要来自过量线程竞争，
不能作为模型固有耗时。上述报告证明预检计算和设备绑定成立，仍不代表
正式 800-epoch 性能达标。

### 3.6 FP32 结论只适用于正式 AMP-off 路径

V6 在绑定权重、投影输入、Context code 和融合系数进入相应计算前使用
`.float()`。正式协议关闭 AMP，因此正式路径确实使用 FP32。若未来在外层
开启 CUDA autocast，`conv2d` 仍属于可被 autocast 改变执行精度的算子；
仅调用 `.float()` 不能无条件保证该卷积在所有调用环境中都以 FP32 执行。

所以 metadata 和论文只能写：

```text
the phase-tied chain is evaluated in FP32 in the preregistered AMP-off
training/evaluation path
```

不能写成“该模块在任意外部调用环境下都强制 FP32”。若以后正式启用 AMP，
应在模块内部为绑定投影单独关闭 autocast，并增加 GPU dtype 测试。

此外，当前 `eps=1e-6` 是 block 的普通 Python 属性，不进入
`state_dict`。因此 checkpoint 本身不能证明该值未变；正式 exact entry、
metadata 和 source lock 必须共同绑定该常量。

### 3.7 三种“正确”必须区分

| 层级 | 当前判断 |
|---|---|
| 代码是否实现了协议公式 | **是** |
| 公式是否数学自洽 | **是，但 Saliency phase-invariant 假设需要验证** |
| 模型是否能提高 Pd/Fa/mIoU | **有局部有效工作区间，但没有达到本轮联合门槛。** Gate B 通过，说明 seed 42 的五个预注册 Fa budget 均达到 Pd floor，且四个预算点不被 SPD 覆盖；但 Gate A、C、D 失败，说明固定阈值联合性能、跨 seed 稳定性和 Full 相对 Capacity 的结构优势未建立 |

所以不能把“单元测试通过”或“Gate B 通过”直接写成“最终模型有效”。
前者证明实现正确，后者只证明一个 seed 下存在有效工作区间；最终判定必须
同时看 Pd、Fa、mIoU、跨 seed 稳定性和同容量结构对照。

### 3.8 seed 42 的 800-epoch 联合指标终点

2026-07-27 对两条 seed 42 正式运行的 800 个完整 JSONL 事件使用冻结入口
的原始字典序选择规则重算 checkpoint。两套独立只读计算结果一致；两路
epoch 均为 `1..800` 连续，数值全部有限，每轮均处理 `530` 个训练样本。
exact journal、活动 checkpoint、三个派生角色 checkpoint 均核对一致并
完成纯 CPU 严格加载。

| Variant / checkpoint role | selected epoch | Pd | Fa | mIoU | tiny-Pd | val loss |
|---|---:|---:|---:|---:|---:|---:|
| Full / Pd-primary | 419 | 188/189 | 1.49146e-6 | 0.922945 | 39/39 | 2.01911e-4 |
| Full / mIoU-primary | 535 | 187/189 | 1.72092e-6 | 0.940544 | 39/39 | 2.56192e-4 |
| Capacity / Pd-primary | 181 | 188/189 | 6.87219e-5 | 0.805532 | 39/39 | 5.13470e-4 |
| Capacity / mIoU-primary | 416 | 186/189 | 5.73639e-7 | 0.939605 | 38/39 | 1.72205e-4 |

评估不能只看 mIoU。Full 的 seed 42 固定阈值 Gate A 终点诊断为：

| Gate A 子项 | 冻结门槛 | seed 42 终点结果 | 终点状态 |
|---|---:|---:|---:|
| Pd-primary matched targets | ≥188/189 | 188/189 | 通过 |
| Pd-primary Fa | ≤5e-6 | 1.49146e-6 | 通过 |
| Pd-primary mIoU | ≥0.9336470588 | 0.922945 | 未通过 |
| mIoU-primary mIoU | ≥0.946542 | 0.940544 | 未通过 |
| mIoU-primary matched targets | ≥187/189 | 187/189 | 通过 |
| mIoU-primary Fa | ≤1e-6 | 1.72092e-6 | 未通过 |

因此 seed 42 终点为 `3/6`，结论是“训练完整但联合性能尚未达门槛”，
不是“mIoU 上升所以模型达标”。750→800 期间四个冻结选择记录均未刷新，
所有选中联合指标差值为零。两路在该窗口均未再次达到 `188/189`；Full
有 50/50 轮达到 `187/189`，Capacity 为 0/50。这说明 Full 在训练末段
稳定于较高 Pd 区间，但没有形成新的最佳联合点。前 1–800 轮达到至少
`188/189` 的频率为 Full `2/800`、Capacity `1/800`，两路均未出现
`189/189`。

Gate A 只读取 Full seed 42 的两个固定阈值终点，因此该 `3/6` 已是本轮
不可逆结果。随后 seed 3407 与八份 sweep 已完成 Gate B–E；最终 Gate B/E
通过、Gate C/D 未通过，没有改变 Gate A 失败与 NER 不授权。完整矩阵现已
作为下一轮不改变 K/C/S 主线的 tokenizer 优化依据。

回看 500→550，Full mIoU-primary 从 epoch 445 刷新到 epoch 535：
多检出一个目标，Fa 降低 `9.17822e-7`，mIoU 增加 `0.001074`，tiny-Pd
保持 `39/39`，该联合改善仍被保留。Pd-primary 没有刷新，其 mIoU 仍低于
门槛。

seed 42 终点下，Full 与 Capacity 的 Pd-primary 同为 `188/189`，Full 的 Fa
更低、mIoU 更高；在 mIoU-primary 上，Full 多检出一个目标、mIoU 更高、
tiny-Pd 也更高，Capacity 则具有更低的 Fa 和 val loss。这仍是单 seed、
完整终点的混合证据，不能代替 Gate D 的两个 seed、两个 checkpoint role、
固定阈值与五个 Fa budget 共 24 项比较。

以下 3.9–3.19 保留为 seed 3407 的运行过程记录；其中“训练中”“尚未完成”
等表述只描述当时截断状态。正式终态与最终裁决以 3.20–3.22 为准。

### 3.9 seed 3407 的共同前 250 轮运行里程碑

seed 42 两路完成后，两条串行 lane 已按预注册映射接棒 seed 3407：
物理 GPU 2 运行 Capacity，物理 GPU 3 运行 Full。两路共同完成前 250 轮
后，使用相同冻结字典序只读重算得到：

| Variant / checkpoint role | selected epoch | Pd | Fa | mIoU | tiny-Pd | val loss |
|---|---:|---:|---:|---:|---:|---:|
| Full / Pd-primary | 217 | 186/189 | 6.08057e-6 | 0.869857 | 39/39 | 3.29955e-4 |
| Full / mIoU-primary | 250 | 182/189 | 1.83564e-6 | 0.881374 | 38/39 | 3.34380e-4 |
| Capacity / Pd-primary | 246 | 186/189 | 5.96584e-6 | 0.876725 | 39/39 | 3.43198e-4 |
| Capacity / mIoU-primary | 250 | 181/189 | 1.49146e-6 | 0.880581 | 38/39 | 3.76328e-4 |

两路前 250 个事件均为 `1..250` 连续、数值有限且每轮处理 530 个训练样本；
冻结 selection policy、逐轮 `new_best` 标记、exact journal、活动 checkpoint
和前 250 行指标记录全部一致。200→250 期间，Full 的 Pd-primary 保持
`186/189`，Fa 降低 2.55 倍、mIoU 增加 `0.083842`；Full 的
mIoU-primary 的 mIoU 增加 `0.018811`、Fa 降低 3.31 倍，但选中点少
检出两个目标。Capacity 的 Pd-primary 保持 `186/189`，Fa 降低 24.12 倍、
mIoU 增加 `0.241352`、tiny-Pd 提升到 `39/39`；其 mIoU-primary 的
Fa 降低 2.77 倍、mIoU 增加 `0.062786`。

同截断下，两个 Pd-primary 均为 `186/189` 和 `39/39`；Capacity 的 Fa
比 Full 低 `1.14728e-7`、mIoU 高 `0.006868`，所以前 200 轮 Full 的
明确优势已经消失。mIoU-primary 上，Full 多检出一个目标、mIoU 高
`0.000794`、val loss 更低，但 Fa 高约 `23.08%`，仍为混合权衡。相对
seed 42 前 250 轮，seed 3407 四个角色均获得更低 Fa，却分别少检出
`2、4、2、5` 个目标；两路仍没有任何 epoch 达到 `187/189`。这些数值
只用于确认训练轨迹和配对运行正常，不能作为 Gate A–E、跨 seed 稳定性
或模型优越性的证据。

### 3.10 seed 3407 的共同前 300 轮运行里程碑

两路共同越过 epoch 300 后，严格截取 `1..300` 并按冻结字典序重算：

| Variant / checkpoint role | selected epoch | Pd | Fa | mIoU | tiny-Pd | val loss |
|---|---:|---:|---:|---:|---:|---:|
| Full / Pd-primary | 217 | 186/189 | 6.08057e-6 | 0.869857 | 39/39 | 3.29955e-4 |
| Full / mIoU-primary | 293 | 184/189 | 1.72092e-6 | 0.898366 | 38/39 | 2.94493e-4 |
| Capacity / Pd-primary | 246 | 186/189 | 5.96584e-6 | 0.876725 | 39/39 | 3.43198e-4 |
| Capacity / mIoU-primary | 291 | 183/189 | 6.88366e-7 | 0.900574 | 38/39 | 3.38471e-4 |

250→300 期间两个 Pd-primary 均未刷新。Full mIoU-primary 多检出两个
目标、Fa 降低约 `6.25%`、mIoU 增加 `0.016991`；Capacity
mIoU-primary 同样多检出两个目标、Fa 降低约 `2.167×`、mIoU 增加
`0.019993`。两路前 300 轮仍均未出现 `187/189`；Full 达到至少
`186/189` 共 3 轮，Capacity 共 2 轮。

同截断比较时，两个 Pd-primary 都是 `186/189` 和 `39/39`，Capacity
的 Fa 低 `1.14728e-7`、mIoU 高 `0.006868`，Full 只有 val loss 更低，
所以按冻结 Pd-primary 字典序 Capacity 暂时领先。mIoU-primary 上，
Capacity 的 mIoU 高 `0.002208`、Fa 低约 `2.5×`，Full 则多检出一个
目标且 val loss 更低；按三项任务指标仍是权衡，不构成严格覆盖。

相对 seed 42 的共同前 300 轮，seed 3407 Full 两个角色均少检出两个
目标，其中 mIoU-primary 还同时具有更高 Fa、更低 mIoU 和较低
tiny-Pd。seed 3407 Capacity/Pd-primary 少检出两个目标，但 Fa 低约
`11.52×`、mIoU 高 `0.071193`；Capacity/mIoU-primary 少检出三个
目标、mIoU 低 `0.023180`，仅 Fa 略低。因此 seed 3407 到 300 轮仍未
复制 seed 42 的检出水平，Full 也尚未建立相对 Capacity 的优势。

两路各 300 个事件均严格连续、数值有限且每轮处理 530 个训练样本；
目标数、tiny 目标数、Pd/tiny-Pd 计数、`new_best` 标记、exact journal、
活动 checkpoint 与主指标边界全部一致。该里程碑在当时仅是阶段诊断，
不替代后来形成的 800 epochs 终点、八份 sweep 或 Gate A–E。

### 3.11 seed 3407 的共同前 350 轮运行里程碑

严格固定两路 `1..350` 后，冻结字典序重算结果为：

| Variant / checkpoint role | selected epoch | Pd | Fa | mIoU | tiny-Pd | val loss |
|---|---:|---:|---:|---:|---:|---:|
| Full / Pd-primary | 217 | 186/189 | 6.08057e-6 | 0.869857 | 39/39 | 3.29955e-4 |
| Full / mIoU-primary | 347 | 183/189 | 1.14728e-6 | 0.907866 | 38/39 | 3.57936e-4 |
| Capacity / Pd-primary | 246 | 186/189 | 5.96584e-6 | 0.876725 | 39/39 | 3.43198e-4 |
| Capacity / mIoU-primary | 346 | 184/189 | 5.27748e-6 | 0.910301 | 38/39 | 3.25995e-4 |

300→350 期间，两个 Pd-primary 仍均未刷新。Full mIoU-primary 的
mIoU 增加 `0.009500`、Fa 降低 `1.5×`，但少检出一个目标且 val loss
增加 `6.34430e-5`；Capacity mIoU-primary 的 mIoU 增加 `0.009727`、
多检出一个目标、val loss 降低 `1.24755e-5`，但 Fa 增至原来的
`7.67×`。这些变化说明区域重叠仍在改善，但尚未转化为 Pd-primary
刷新。

同截断下，两个 Pd-primary 的 Pd 和 tiny-Pd 相同；Capacity 的 Fa 低
`1.14728e-7`、mIoU 高 `0.006868`，因此在该固定阈值角色的三项任务
指标上严格覆盖 Full。mIoU-primary 上，Capacity 的 mIoU 高
`0.002435`、多检出一个目标，Full 的 Fa 则低约 `4.6×`，仍是权衡。
两路前 350 轮均未出现 `187/189`；Full 达到至少 `186/189` 共 4 轮，
Capacity 共 2 轮。

相对 seed 42 的共同前 350 轮，seed 3407 Full/Pd-primary 少检出两个
目标，Fa 更低但 mIoU 和 val loss 更弱；Full/mIoU-primary 也少检出
两个目标、mIoU 低 `0.019879`、val loss 更高且 Fa 相同。
seed 3407 Capacity/Pd-primary 少检出两个目标，但 Fa 低约 `11.52×`、
mIoU 高 `0.071193`；Capacity/mIoU-primary 则在 Pd、Fa、mIoU、
tiny-Pd 和 val loss 上均弱于 seed 42。因此 seed 3407 到 350 轮仍未
复制 seed 42 的检出水平，Full 也没有建立相对 Capacity 的阶段优势。

两路各 350 个事件均严格连续、数值有限且每轮处理 530 个训练样本；
目标数、tiny 目标数、Pd/tiny-Pd 计数、全部 `new_best` 标记、
exact journal、活动 checkpoint 与主指标边界均一致。该截断结果仍不是
Gate A–E 终点结论。

### 3.12 seed 3407 的共同前 400 轮运行里程碑

严格截取两路 `1..400` 并按冻结字典序重算得到：

| Variant / checkpoint role | selected epoch | Pd | Fa | mIoU | tiny-Pd | val loss |
|---|---:|---:|---:|---:|---:|---:|
| Full / Pd-primary | 217 | 186/189 | 6.08057e-6 | 0.869857 | 39/39 | 3.29955e-4 |
| Full / mIoU-primary | 394 | 183/189 | 4.58911e-7 | 0.914688 | 38/39 | 3.55317e-4 |
| Capacity / Pd-primary | 246 | 186/189 | 5.96584e-6 | 0.876725 | 39/39 | 3.43198e-4 |
| Capacity / mIoU-primary | 390 | 184/189 | 2.29455e-7 | 0.920703 | 38/39 | 3.03504e-4 |

350→400 期间，两个 Pd-primary 继续没有刷新。Full mIoU-primary 的
Pd 保持 `183/189`、mIoU 增加 `0.006822`、Fa 降低 `2.5×`；
Capacity mIoU-primary 的 Pd 保持 `184/189`、mIoU 增加
`0.010402`、Fa 降低 `23×`。351–400 中 Full 只有 1 轮达到
`186/189`，Capacity 为 0 轮；两路都没有达到 `187/189`。累计前
400 轮，Full 和 Capacity 达到至少 `186/189` 的轮数分别为 5 和 2，
达到至少 `187/189` 的轮数均为 0。

在 Pd-primary 固定阈值点，两者同为 `186/189` 和 `39/39`，Capacity
的 Fa 低 `1.14728e-7`、mIoU 高 `0.006868`；在 mIoU-primary 固定
阈值点，Capacity 多检出一个目标、mIoU 高 `0.006015`、Fa 低
`2×`、val loss 也更低，tiny-Pd 相同。因此截至共同 400 轮，Capacity
在两个固定阈值 checkpoint 角色的 Pd/Fa/mIoU 三项上都严格覆盖 Full。
该结论不等于完整 Gate D，因为五个 Fa budget 与 800 轮终点仍未获得。

seed 42 同截断的 Full/Capacity 两个 Pd-primary 均为 `188/189`；
两个 mIoU-primary 均为 `187/189`。seed 3407 四个角色分别少检出
`2、4、2、3` 个目标，Fa 均更低，但 Full 两角色和
Capacity/mIoU-primary 的 mIoU 也更低。因此当前仍表现为“更低 Fa、
同时更低 Pd”的跨 seed 权衡，没有建立稳定优势。

两路各 400 个事件均严格连续、数值有限且每轮处理 530 个训练样本；
目标数、tiny 目标数、Pd/tiny-Pd 计数、全部 `new_best` 标记、
exact journal、活动 checkpoint 与主指标边界均一致。在共同 400 轮截断
时，finalizer 与权威接受入口仍处于等待；该等待随后已经结束，终局见
3.20–3.22。

### 3.13 seed 3407 的共同前 450 轮运行里程碑

严格固定两路 `1..450` 后，冻结字典序重算结果为：

| Variant / checkpoint role | selected epoch | Pd | Fa | mIoU | tiny-Pd | val loss |
|---|---:|---:|---:|---:|---:|---:|
| Full / Pd-primary | 427 | 187/189 | 4.84151e-5 | 0.860967 | 38/39 | 3.05041e-4 |
| Full / mIoU-primary | 438 | 184/189 | 2.98292e-6 | 0.916496 | 38/39 | 2.59701e-4 |
| Capacity / Pd-primary | 246 | 186/189 | 5.96584e-6 | 0.876725 | 39/39 | 3.43198e-4 |
| Capacity / mIoU-primary | 445 | 184/189 | 1.72092e-6 | 0.925743 | 38/39 | 2.93624e-4 |

400→450 期间，Full 在 epoch 427 首次达到 `187/189` 并刷新
Pd-primary：相对原选择点多检出一个目标，但 Fa 增至约 `7.96×`、
mIoU 降低 `0.008890`，tiny-Pd 由 `39/39` 降至 `38/39`。Full
mIoU-primary 多检出一个目标、mIoU 增加 `0.001808`、val loss 明显
下降，但 Fa 增至原来的 `6.5×`。Capacity Pd-primary 没有刷新；
Capacity mIoU-primary 的 mIoU 增加 `0.005041`、val loss 下降，但
Fa 增至原来的 `7.5×`。

401–450 中，Full 只有 epoch 427 达到 `187/189`，没有出现
`188/189` 或 `189/189`；Capacity 没有一轮达到 `186/189`。累计前
450 轮，Full 达到至少 `186/189` 共 6 轮，其中达到至少 `187/189`
仅 1 轮；Capacity 达到至少 `186/189` 共 2 轮，仍未达到
`187/189`。

同截断比较时，Full/Pd-primary 多检出一个目标，但 Fa 高约
`8.12×`、mIoU 低 `0.015758`、tiny-Pd 少一个，因此是明显的
Pd–Fa–mIoU 权衡，不是全面优势。mIoU-primary 的 Pd 和 tiny-Pd
相同，Capacity 的 mIoU 高 `0.009247`、Fa 低约 `1.73×`，在三项
任务指标上仍严格覆盖 Full。准确结论是：Full 开始表现出更高的 Pd
上限，Capacity 仍具有更好的低虚警与区域质量工作点。

相对 seed 42 的共同前 450 轮，seed 42 Full 的两个角色在 Pd、Fa、
mIoU、tiny-Pd 和 val loss 上均优于 seed 3407 Full。seed 3407
Capacity/Pd-primary 少检出两个目标，但 Fa 更低、mIoU 更高，仍是
权衡；seed 42 Capacity/mIoU-primary 在 Pd、Fa、mIoU 和 val loss
上均更强，tiny-Pd 相同。seed 3407 首次达到 `187/189` 是积极进展，
但只出现一次且代价较大，不能据此建立跨 seed 稳定性。

两路各 450 个事件均严格连续、数值有限且每轮处理 530 个训练样本；
目标数、tiny 目标数、Pd/tiny-Pd 计数、全部 `new_best` 标记、
exact journal、活动 checkpoint 与主指标边界均一致。该里程碑继续
服从 800 epochs 终点与完整 Gate A–E。

### 3.14 seed 3407 的共同前 500 轮运行里程碑

严格固定两路 `1..500` 后，冻结字典序重算结果为：

| Variant / checkpoint role | selected epoch | Pd | Fa | mIoU | tiny-Pd | val loss |
|---|---:|---:|---:|---:|---:|---:|
| Full / Pd-primary | 427 | 187/189 | 4.84151e-5 | 0.860967 | 38/39 | 3.05041e-4 |
| Full / mIoU-primary | 500 | 183/189 | 1.14728e-6 | 0.918487 | 38/39 | 3.92571e-4 |
| Capacity / Pd-primary | 246 | 186/189 | 5.96584e-6 | 0.876725 | 39/39 | 3.43198e-4 |
| Capacity / mIoU-primary | 480 | 185/189 | 6.88366e-7 | 0.928763 | 39/39 | 3.20231e-4 |

450→500 期间，两个 Pd-primary 均未刷新。Full mIoU-primary 的
mIoU 增加 `0.001991`、Fa 降低 `2.6×`，但少检出一个目标且
val loss 明显变差。Capacity mIoU-primary 多检出一个目标、
tiny-Pd 由 `38/39` 提升为 `39/39`、mIoU 增加 `0.003020`、Fa
降低 `2.5×`，只有 val loss 略有变差。451–500 两路均没有任何
epoch 达到 `186/189`，也没有 Pd-primary 刷新。

累计前 500 轮，Full 达到至少 `186/189` 共 6 轮，达到至少
`187/189` 仍只有 epoch 427 这一轮，未达到 `188/189`；Capacity
达到至少 `186/189` 共 2 轮，仍未达到 `187/189`。因此 Full 的
`187/189` 尚未形成可重复工作区间，两路均没有复制 seed 42 的
`188/189`。

同截断比较时，Full/Pd-primary 多检出一个目标，但 Fa 高约
`8.12×`、mIoU 低 `0.015758`、tiny-Pd 少一个。mIoU-primary 上，
Capacity 多检出两个目标、mIoU 高 `0.010276`、Fa 低约 `1.67×`、
tiny-Pd 多一个且 val loss 更低，对 Full 构成全面覆盖。当前准确结论
仍是：Full 只在 Pd-primary 提供一个高 Pd 但高代价的单点，Capacity
具有更强的低虚警与区域质量工作点。

相对 seed 42 的共同前 500 轮，seed 42 Full/Pd-primary 在全部联合
指标上优于 seed 3407 Full；seed 3407 Full/mIoU-primary 虽 Fa 更低，
但少检出三个目标、mIoU 低 `0.020984`、tiny-Pd 少一个。
seed 3407 Capacity/Pd-primary 仍是“少检出两个目标，但 Fa 和
mIoU 更好”的权衡；Capacity/mIoU-primary 只有 tiny-Pd 多一个，
Pd、Fa、mIoU 和 val loss 均弱于 seed 42。跨 seed 稳定性仍未建立。

两路各 500 个事件均严格连续、数值有限且每轮处理 530 个训练样本；
目标数、tiny 目标数、Pd/tiny-Pd 计数、全部 `new_best` 标记、
exact journal、活动 checkpoint 与主指标边界均一致。finalizer 与
在共同 500 轮截断时，权威接受入口仍处于等待；当前终局见 3.20–3.22。

### 3.15 seed 3407 的共同前 550 轮运行里程碑

严格固定两路 `1..550` 后，冻结字典序重算结果为：

| Variant / checkpoint role | selected epoch | Pd | Fa | mIoU | tiny-Pd | val loss |
|---|---:|---:|---:|---:|---:|---:|
| Full / Pd-primary | 427 | 187/189 | 4.84151e-5 | 0.860967 | 38/39 | 3.05041e-4 |
| Full / mIoU-primary | 500 | 183/189 | 1.14728e-6 | 0.918487 | 38/39 | 3.92571e-4 |
| Capacity / Pd-primary | 522 | 186/189 | 1.03255e-6 | 0.928052 | 39/39 | 2.86366e-4 |
| Capacity / mIoU-primary | 518 | 184/189 | 6.88366e-7 | 0.929850 | 38/39 | 2.93645e-4 |

500→550 期间，Full 两个角色均未刷新，501–550 甚至没有一轮达到
`185/189`。Capacity/Pd-primary 在 epoch 522 刷新：Pd 保持
`186/189`，Fa 降低 `5.78×`，mIoU 增加 `0.051327`，tiny-Pd
保持 `39/39`，val loss 也明显降低。Capacity/mIoU-primary 的
mIoU 增加 `0.001086`、val loss 下降且 Fa 不变，但 Pd 由
`185/189` 降为 `184/189`，tiny-Pd 由 `39/39` 降为 `38/39`。

高 Pd 频次为：

| 前 1–550 | ≥185/189 | ≥186/189 | ≥187/189 | ≥188/189 |
|---|---:|---:|---:|---:|
| Full | 33 | 6 | 1 | 0 |
| Capacity | 102 | 3 | 0 | 0 |

其中 501–550 窗口内，Full 四档频次全部为 0；Capacity 有 33 轮达到
至少 `185/189`、1 轮达到 `186/189`，但没有达到 `187/189`。
这说明 Capacity 已形成较稳定的 `185/189` 区间；Full 的
`187/189` 仍只有 epoch 427 一个孤立点，两路仍未出现 `188/189`。

同截断比较时，Full/Pd-primary 多检出一个目标，但 Capacity 的 Fa
低约 `46.89×`、mIoU 高 `0.067085`、tiny-Pd 多一个且 val loss
更低。mIoU-primary 上，Capacity 多检出一个目标、mIoU 高
`0.011363`、Fa 低约 `1.67×`、val loss 更低，tiny-Pd 持平，
对 Full 构成全面覆盖。Full 只保留一个高 Pd、高代价的 Pd-primary
单点。

相对 seed 42 的共同前 550 轮，seed 42 Full/Pd-primary 全面优于
seed 3407 Full；seed 3407 Full/mIoU-primary 仅 Fa 更低，但少检出
四个目标、mIoU 低 `0.022057`、tiny-Pd 少一个。seed 3407
Capacity/Pd-primary 少检出两个目标，但 Fa 低约 `66.56×`、mIoU
高 `0.122520`，属于显著工作点迁移；seed 42
Capacity/mIoU-primary 在 Pd、Fa、mIoU 和 val loss 上更强。
跨 seed 稳定性仍未建立，但 Capacity 的中高 Pd 区间已经明显比 Full
稳定。

两路各 550 个事件均严格连续、数值有限且每轮处理 530 个训练样本；
目标数、tiny 目标数、Pd/tiny-Pd 计数、全部 `new_best` 标记、
exact journal、活动 checkpoint 与主指标边界均一致。该截断结论仍不
替代 800 epochs 与完整预算扫描。

### 3.16 seed 3407 的共同前 600 轮运行里程碑

严格固定两路 `1..600` 后，冻结字典序重算结果为：

| Variant / checkpoint role | selected epoch | Pd | Fa | mIoU | tiny-Pd | val loss |
|---|---:|---:|---:|---:|---:|---:|
| Full / Pd-primary | 427 | 187/189 | 4.84151e-5 | 0.860967 | 38/39 | 3.05041e-4 |
| Full / mIoU-primary | 570 | 185/189 | 1.03255e-6 | 0.924459 | 38/39 | 2.99091e-4 |
| Capacity / Pd-primary | 522 | 186/189 | 1.03255e-6 | 0.928052 | 39/39 | 2.86366e-4 |
| Capacity / mIoU-primary | 518 | 184/189 | 6.88366e-7 | 0.929850 | 38/39 | 2.93645e-4 |

550→600 期间，Full/Pd-primary 没有刷新；Full/mIoU-primary 在
epoch 570 多检出两个目标、mIoU 增加 `0.005972`、Fa 进一步降低约
`10%`、val loss 降低 `9.34801e-5`，tiny-Pd 保持 `38/39`。
Capacity 两个冻结角色均未刷新，551–600 两路也都没有 Pd-primary
刷新。

高 Pd 频次为：

| 前 1–600 | ≥185/189 | ≥186/189 | ≥187/189 | ≥188/189 |
|---|---:|---:|---:|---:|
| Full | 41 | 10 | 1 | 0 |
| Capacity | 144 | 3 | 0 | 0 |

551–600 窗口中，Full 有 8/50 轮达到至少 `185/189`、4/50 轮达到
至少 `186/189`，但没有 `187/189`；Capacity 有 42/50 轮达到至少
`185/189`，窗口覆盖率为 `84%`，但没有达到 `186/189`。Full 相比
上一窗口的全零状态有所恢复；Capacity 的 `185/189` 区间继续保持，
但尚未向更高 Pd 档位迁移。Full 的 `187/189` 仍只有 epoch 427，
两路仍未出现 `188/189`。

同截断比较时，Full/Pd-primary 多检出一个目标；Capacity 的 Fa 低约
`46.89×`、mIoU 高 `0.067085`、tiny-Pd 多一个且 val loss 更低。
mIoU-primary 上 Full 多检出一个目标，Capacity 的 mIoU 高
`0.005390`、Fa 低 `1.5×`、val loss 略低，tiny-Pd 相同。因此 Full
开始恢复较高 Pd 的 mIoU 工作点，但仍未形成相对 Capacity 的综合优势。

相对 seed 42 的共同前 600 轮，seed 42 Full/Pd-primary 仍全面优于
seed 3407；seed 3407 Full/mIoU-primary 虽 Fa 更低，但少检出两个
目标、mIoU 低 `0.016085`、tiny-Pd 少一个。seed 3407
Capacity/Pd-primary 少检出两个目标，但 Fa 低约 `66.56×`、mIoU
高 `0.122520`；seed 42 Capacity/mIoU-primary 在 Pd、Fa、mIoU
和 val loss 上更强。seed 3407 出现恢复迹象，但跨 seed 稳定性仍未
建立。

两路各 600 个事件均严格连续、数值有限且每轮处理 530 个训练样本；
目标数、tiny 目标数、Pd/tiny-Pd 计数、全部 `new_best` 标记、
exact journal、活动 checkpoint 与主指标边界均一致。finalizer 与
在共同 600 轮截断时，权威接受入口仍等待剩余 400 个 formal epochs；
后续训练已经完成。

### 3.17 seed 3407 的共同前 650 轮运行里程碑

严格固定两路 `1..650` 后，四个冻结角色相对共同 600 轮均未刷新：

| Variant / checkpoint role | selected epoch | Pd | Fa | mIoU | tiny-Pd | val loss |
|---|---:|---:|---:|---:|---:|---:|
| Full / Pd-primary | 427 | 187/189 | 4.84151e-5 | 0.860967 | 38/39 | 3.05041e-4 |
| Full / mIoU-primary | 570 | 185/189 | 1.03255e-6 | 0.924459 | 38/39 | 2.99091e-4 |
| Capacity / Pd-primary | 522 | 186/189 | 1.03255e-6 | 0.928052 | 39/39 | 2.86366e-4 |
| Capacity / mIoU-primary | 518 | 184/189 | 6.88366e-7 | 0.929850 | 38/39 | 2.93645e-4 |

601–650 两路均没有 `new_best_pd` 或 `new_best_miou`。Full 在该窗口
没有一轮达到 `185/189`；Capacity 有 29/50 轮达到 `185/189`，但
没有一轮达到 `186/189`。Capacity 的 `185/189` 窗口覆盖率由上一
窗口的 `84%` 降为 `58%`，仍显著高于 Full。

累计高 Pd 频次为：

| 前 1–650 | ≥185/189 | ≥186/189 | ≥187/189 | ≥188/189 |
|---|---:|---:|---:|---:|
| Full | 41 | 10 | 1 | 0 |
| Capacity | 173 | 3 | 0 | 0 |

Full 的 `187/189` 仍只有 epoch 427；两路仍未出现 `188/189`。
Full 在 551–600 的短暂恢复没有延续，Capacity 继续保持更稳定的
`185/189` 常态区间。

由于冻结点未变，同截断比较也不变：Full/Pd-primary 多检出一个目标，
但 Capacity 的 Fa 低约 `46.89×`、mIoU 高 `0.067085`、tiny-Pd
多一个且 val loss 更低；mIoU-primary 上 Full 多检出一个目标，
Capacity 的 mIoU 高 `0.005390`、Fa 低 `1.5×`、val loss 略低，
tiny-Pd 相同。Full 仍只有孤立高 Pd 点，Capacity 的常态工作区间更
稳定。

seed 3407 到 650 轮仍未复制 seed 42 的高 Pd 水平，跨 seed 稳定性
仍未建立。两路各 650 个事件均严格连续、数值有限且每轮处理 530 个
训练样本；目标数、tiny 目标数、Pd/tiny-Pd 计数、全部 `new_best`
标记、exact journal、活动 checkpoint 与主指标边界均一致。

### 3.18 seed 3407 的共同前 700 轮运行里程碑

严格固定两路 `1..700` 后，四个冻结角色相对共同 650 轮仍未刷新：

| Variant / checkpoint role | selected epoch | Pd | Fa | mIoU | tiny-Pd | val loss |
|---|---:|---:|---:|---:|---:|---:|
| Full / Pd-primary | 427 | 187/189 | 4.84151e-5 | 0.860967 | 38/39 | 3.05041e-4 |
| Full / mIoU-primary | 570 | 185/189 | 1.03255e-6 | 0.924459 | 38/39 | 2.99091e-4 |
| Capacity / Pd-primary | 522 | 186/189 | 1.03255e-6 | 0.928052 | 39/39 | 2.86366e-4 |
| Capacity / mIoU-primary | 518 | 184/189 | 6.88366e-7 | 0.929850 | 38/39 | 2.93645e-4 |

651–700 两路均没有 `new_best_pd` 或 `new_best_miou`。Full 在该窗口
没有一轮达到 `185/189`；Capacity 有 37/50 轮达到 `185/189`，
覆盖率为 `74%`，但仍没有达到 `186/189`。结合上一窗口，Full 已
连续 601–700 共 100 轮没有达到 `185/189`；Capacity 最近 100 轮
有 66 轮达到 `185/189`，其常态平台继续存在。

累计高 Pd 频次为：

| 前 1–700 | ≥185/189 | ≥186/189 | ≥187/189 | ≥188/189 |
|---|---:|---:|---:|---:|
| Full | 41 | 10 | 1 | 0 |
| Capacity | 210 | 3 | 0 | 0 |

Full 的 `187/189` 仍只有 epoch 427；两路始终没有出现 `188/189`。
四个冻结点及 Full/Capacity 比较均未变化：Full 在两个角色上分别多
检出一个目标，但 Pd-primary 付出约 `46.89×` Fa、`0.067085` mIoU
和一个 tiny 目标的代价；mIoU-primary 也仍由 Capacity 保持更高
mIoU、更低 Fa 和更低 val loss。结合最近 100 轮频次，Full 的高 Pd
更接近孤立峰值，Capacity 的常态 Pd 区间更稳定。

seed 3407 到 700 轮仍未复制 seed 42 的高 Pd 水平，Full 后期频次
尤其弱，跨 seed 稳定性仍未建立。两路各 700 个事件均严格连续、
数值有限且每轮处理 530 个训练样本；目标数、tiny 目标数、
Pd/tiny-Pd 计数、全部 `new_best` 标记、exact journal、活动
checkpoint 与主指标边界均一致。在共同 700 轮截断时，后续训练仍需
完成预注册终点；实际终点已经完成，见 3.20。

### 3.19 seed 3407 的共同前 750 轮运行里程碑

严格固定两路 `1..750` 后，四个冻结角色相对共同 700 轮仍未刷新：

| Variant / checkpoint role | selected epoch | Pd | Fa | mIoU | tiny-Pd | val loss |
|---|---:|---:|---:|---:|---:|---:|
| Full / Pd-primary | 427 | 187/189 | 4.84151e-5 | 0.860967 | 38/39 | 3.05041e-4 |
| Full / mIoU-primary | 570 | 185/189 | 1.03255e-6 | 0.924459 | 38/39 | 2.99091e-4 |
| Capacity / Pd-primary | 522 | 186/189 | 1.03255e-6 | 0.928052 | 39/39 | 2.86366e-4 |
| Capacity / mIoU-primary | 518 | 184/189 | 6.88366e-7 | 0.929850 | 38/39 | 2.93645e-4 |

701–750 两路均没有 `new_best_pd` 或 `new_best_miou`。Full 在该窗口
没有一轮达到 `185/189`；Capacity 有 19/50 轮达到 `185/189`，
覆盖率为 `38%`，但仍没有达到 `186/189`。Capacity 最近四个 50 轮
窗口的 `185/189` 覆盖率依次为 `84%→58%→74%→38%`：平台仍存在，
但终点前出现减弱。Full 已连续 601–750 共 150 轮没有达到
`185/189`。

累计高 Pd 频次为：

| 前 1–750 | ≥185/189 | ≥186/189 | ≥187/189 | ≥188/189 |
|---|---:|---:|---:|---:|
| Full | 41 | 10 | 1 | 0 |
| Capacity | 229 | 3 | 0 | 0 |

Full 的 `187/189` 仍只有 epoch 427，两路始终没有出现 `188/189`。
四个冻结点及同截断比较均未变化：Full 仍只保留孤立高 Pd 点；
Capacity 的常态工作区间更稳定，但其后期 `185/189` 平台也在减弱。
最后 50 轮不能预设再次刷新。

seed 3407 到 750 轮仍未复制 seed 42 的高 Pd 水平，跨 seed 稳定性
尚未建立。两路各 750 个事件均严格连续、数值有限且每轮处理 530 个
训练样本；目标数、tiny 目标数、Pd/tiny-Pd 计数、全部 `new_best`
标记、exact journal、活动 checkpoint 与主指标边界均一致。

本段记录当时预定的自动链：四组完成判定后，物理 GPU 2 按冻结映射依次
生成 8 份 sweep，再生成 comparison、completion manifest 与 marker 并
执行 completion verify；之后再运行独立接受复核。该链及最终三级复核
随后均已完成，见 3.20–3.22。

### 3.20 四组正式训练的 800-epoch 终态

四组预注册任务均已完成，正式训练进度为 `4/4` runs、
`3200/3200` epochs。每组 `metrics.jsonl` 均为 `1..800` 连续，训练
划分一致，12 份 `best / best_miou / last` checkpoint 均通过 CPU
严格加载。物理设备映射保持为：

```text
GPU 2：Full seed 42 → Capacity seed 3407
GPU 3：Capacity seed 42 → Full seed 3407
```

固定阈值 `0.5` 的八个正式角色终点为：

| Seed | Variant / checkpoint role | epoch | Pd | Fa | mIoU | tiny-Pd |
|---:|---|---:|---:|---:|---:|---:|
| 42 | Full / Pd-primary | 419 | 188/189 | 1.49146e-6 | 0.922945 | 39/39 |
| 42 | Full / mIoU-primary | 535 | 187/189 | 1.72092e-6 | 0.940544 | 39/39 |
| 42 | Capacity / Pd-primary | 181 | 188/189 | 6.87219e-5 | 0.805532 | 39/39 |
| 42 | Capacity / mIoU-primary | 416 | 186/189 | 5.73639e-7 | 0.939605 | 38/39 |
| 3407 | Full / Pd-primary | 427 | 187/189 | 4.84151e-5 | 0.860967 | 38/39 |
| 3407 | Full / mIoU-primary | 570 | 185/189 | 1.03255e-6 | 0.924459 | 38/39 |
| 3407 | Capacity / Pd-primary | 522 | 186/189 | 1.03255e-6 | 0.928052 | 39/39 |
| 3407 | Capacity / mIoU-primary | 518 | 184/189 | 6.88366e-7 | 0.929850 | 38/39 |

该表说明 V6 不是“所有指标都变差”：seed 42 Full/Pd-primary 达到
`188/189` 且 Fa 很低，Full/mIoU-primary 也保持 `187/189`。但它同样
不是“联合性能达标”：两个 Full 角色的 mIoU 都低于冻结锚点，其中
mIoU-primary 的 Fa 还高于 `1e-6`；seed 3407 的 Full 又明显退化。

### 3.21 八份 sweep 与 Gate A–E 正式裁决

四组 × 两个 checkpoint role 的八份闭区间 sweep 均已生成。正式 Gate
结果为：

| Gate | 结果 | 关键证据 |
|---|---:|---|
| A：seed 42 固定阈值联合门槛 | **未通过（3/6）** | Pd 两项和 Pd-primary Fa 通过；两个 mIoU 门槛及 mIoU-primary Fa 未通过 |
| B：seed 42 预注册 Fa budget 与 SPD | **通过** | 五个 Pd floor 全部通过；五个预算中四个工作点不被冻结 SPD 覆盖 |
| C：seed 3407 稳定性 | **未通过** | 固定阈值六个子项全部未通过（0/6 通过）；五个预算仅 `1e-4` 一个满足 seed 差值要求 |
| D：Full 相对 Capacity | **未通过** | seed 42 通过；seed 3407 有三个 Capacity 严格覆盖点 |
| E：工程与产物完整性 | **通过（10/10）** | 四组训练、日志、checkpoint、source lock、sweep 与 smoke 均一致 |

Gate B 的 seed 42 Full/Pd-primary 预算点为：

| Fa budget | threshold | Pd | Fa | mIoU | Pd floor |
|---:|---:|---:|---:|---:|---:|
| 1e-6 | 0.89 | 187/189 | 3.44183e-7 | 0.906594 | 通过 |
| 5e-6 | 0.59 | 188/189 | 1.14728e-6 | 0.922341 | 通过 |
| 1e-5 | 0.04 | 189/189 | 7.34258e-6 | 0.860343 | 通过 |
| 5e-5 | 0.04 | 189/189 | 7.34258e-6 | 0.860343 | 通过 |
| 1e-4 | 0.04 | 189/189 | 7.34258e-6 | 0.860343 | 通过 |

因此，V6 Full 在 seed 42 上确实产生了一个可用的高 Pd 工作区间；它在
`5e-6、1e-5、5e-5、1e-4` 四个预算点不被 SPD 覆盖。最严格 `1e-6`
预算下，两者均为 `187/189`，但 SPD 的 Fa 为 0、mIoU 更高，V6 不占优。
这支持“存在竞争力工作区间”，不支持“全面超过 SPD”。

Gate C 暴露了本轮最主要的问题。seed 3407 Full/Pd-primary 相对 seed 42
在五个预算上的 matched target 为：

| Fa budget | seed 42 | seed 3407 | 要求 | 结果 |
|---:|---:|---:|---:|---:|
| 1e-6 | 187 | 5 | ≥186 | 未通过 |
| 5e-6 | 188 | 19 | ≥187 | 未通过 |
| 1e-5 | 189 | 19 | ≥188 | 未通过 |
| 5e-5 | 189 | 187 | ≥188 | 未通过 |
| 1e-4 | 189 | 188 | ≥188 | 通过 |

这不是一般幅度的随机波动，而是低 Fa 阈值区域出现明显断层。Gate D 又
显示 seed 3407 的 Capacity 在 `pd_primary@5e-6`、
`pd_primary@1e-5` 和 `miou_primary@1e-6` 三处严格覆盖 Full，因而
Context/Saliency redistribution 的稳定净收益也未建立。

正式裁决因此保持：

```text
decision=ENGINEERING_GATE_FAIL
engineering_gate_passed=false
ner_stage_authorized=false
mainline_changed=false
paper_core_established=false
stability_claim_supported=false
```

### 3.22 正式接受、复核边界与下一版输入

`authoritative_result_accepted=true` 只表示这组失败结果已经通过完整性
接受，可以作为下一版设计依据；它不把 `engineering_gate_passed=false`
改写为通过。最终接受状态为：

```text
formal_runs=4/4
formal_epochs=3200/3200
real_sweeps=8/8
strict_valid_sweeps=8/8
compatibility_valid_sweeps=8/8
frozen_acceptance_ran_first=true
```

训练、冻结后处理和 supplemental acceptance 三个既有源码锁保持不变：

```text
training:
2de1a8f75deb321b5aec4cf5dfa6bc16df8443e858e1d48a3ab6bea34de526d2

postprocess:
3cfbfda891d823c5b97d2d1a2364790c823fac9a548bbf0987444979619bd827

supplemental acceptance:
dcaf2f1b32cff5096511ba090e3149327deea1f32f2a51d4d866bb0d0cf32696
```

checkpoint-metric compatibility 层在不修改上述冻结源码和原产物的前提下，
只向内存中的审计副本补入旧 checkpoint 未保存的八个目标计数字段，并重放
每个 run 的确定性训练环境；原 checkpoint、`metrics.jsonl`、`summary.json`
及五项 checkpoint 选择指标均不改写。Pd、Fa、mIoU 与 threshold 工作点
必须原样复现，仅对 threshold-invariant `val_loss` 允许在 `1e-7` 内归一到
checkpoint 记录值。其独立源码锁为：

```text
fd3a11d4b48f0990538554d92759e8c5b6e4be178407483f64f0069a65439f93
```

该层 47 项 CPU 测试通过，八份 sweep 均实现固定阈值零差复核。completion
manifest SHA-256 为
`27b168b1b4ae59e5bd8db15702b959225b879338e32427a9b814e8640e6f0188`，
`COMPLETE.sha256` 文件 SHA-256 为
`37bcaccf3365cbe57edab94fc464fa2a41d85cc3b70eed7c5db276af16650741`。
调试过程中被替代的后处理产物已移入 `rejected_postprocess/`；原训练
checkpoint 与逐轮指标未改写，最终 comparison/completion 仅在八份接受
sweep 就绪后生成并封存。

对下一版最重要的输入不是继续证明抽象机制，而是三个可直接转化为代码的
问题：保留四相位位置身份，避免 phase-sum 抵消；消除低 Fa 区域的阈值
断层；让 Full 相对 Capacity 的收益在两个 seed 上方向一致。下一阶段因此
是 K/C/S tokenizer 内部优化，不是 NER。

---

## 4. 整体模型设计的优点

### 4.1 干预位置有任务动机

红外小目标可能只占数个像素，进入大步长 patch embedding 时容易被平滑或压缩。优先改动 `embeddings_1/2`，而非更深层 tokenization，符合“先保护浅层局部响应”的任务逻辑。

### 4.2 严格单变量设计

V6 不改 backbone、SCTB、decoder、loss 和评估协议，不接入 NER、Survival 或 Query-only FG。这样若结果变化，可以较可信地归因于 tokenizer。

### 4.3 三源语义清晰

```text
Keep     → 相位显式重排后再学习压缩的局部信息
Context  → 2×2 局部平均/DC 背景
Saliency → 局部峰值相对均值的对比幅值
```

三者与红外小目标的“微弱目标、局部对比、复杂背景”问题有清晰对应关系。

### 4.4 Full 不靠相对 Capacity 的新增容量制造优势

`Wt` 完全由 `Wk` 派生；Full 和 Capacity 参数、state key 与初始化相同。
Gate D 因而能检验 Context headroom 本身，而不是检验 Full 相对 Capacity
的额外参数量。

该结论不能扩大为“V6 相对 dense SPD 完全不增加参数”。V6 的七个
`saliency_scale` 共增加 `4×32+3×64=320` 个参数：

```text
dense SPD shallow / full model: 65,856 / 10,842,835
V6 shallow / full model:        66,176 / 10,843,155
```

准确说法是：绑定的 `Wt`、`Ca` 和 `Sa` 投影不新增可学习参数，且 Full 与
Capacity 严格参数匹配。二者不是严格计算量匹配：Capacity 仍计算
`Ca/Sa`，但跳过 Full 使用的 Context normalization。

### 4.5 从强锚点出发

零尺度使 V6 在共享 Keep `weight+bias` 时，功能上等价于一份同初始化的
dense SPD。这是结构级精确起点，不表示加载了已经训练好的 SPD checkpoint。
对于小规模 IRSTD 数据集，该起点有利于控制候选与对照的初始差异。

### 4.6 工程协议覆盖面较完整

现有协议包括：

- 固定 split；
- 固定阈值与闭区间 sweep；
- 预注册 Fa budget；
- Paired initialization；
- exact resume；
- source lock；
- GPU smoke；
- Gate A–E；
- 不因看到结果而降低门槛。

这些项目构成了当前仓库内较完整的筛选与复核协议；V6 exact entry、
轻量 exact-resume 轨迹、CPU/GPU 2/3 源码绑定 smoke 及两卡运行管理
已经落地；正式 source lock 与四任务双 lane preflight 也已通过。四组任务
已按两条串行 lane 完成，八份 sweep、comparison、completion 与三级复核
均已完成。Gate E 的 10/10 通过表明该工程协议闭环成立。

---

## 5. 当前方案的主要技术风险

### 5.1 Saliency phase collapse

`max-mean` 只保留一个 2×2 cell 的显著性幅值，丢弃峰值位于四个 phase 中的哪一个位置。V6 对 `embeddings_1` 使用 4 个连续 2× block，对 `embeddings_2` 使用 3 个，共 7 个 block；phase collapse 可能在多级压缩中累积。

可能表现为：

- 目标响应存在但定位变粗；
- mask 周围 attached halo 增大；
- Pd 提高而 mIoU 降低；
- 同一 cell 内噪声峰值被当作目标显著性。

### 5.2 phase-sum cancellation

建议按 block 计算：

\[
\rho_l=
\frac{\left\|\sum_p W_{l,p}\right\|_F}
{\sum_p\left\|W_{l,p}\right\|_F+\epsilon}.
\]

- `ρ` 高：phase-sum 投影保留较强响应；
- `ρ` 低：Keep 权重依赖 phase 差异，求和后发生显著抵消。

若 `ρ` 低且 `Sa` 能量远小于 `K`，说明 V6 的 tied projection 可能压制了局部对比信号。

### 5.3 Context 不是严格的邻域背景建模

V6 Context 来自每个 2×2 cell 的平均值，随后在整张低分辨率 feature map 上做空间中心化与 RMS normalization。它是一种局部 DC 信息与全图通道内归一化的组合，但不是传统意义上的中心–环绕背景对比或局部 ring context。

因此论文不宜声称它“显式估计目标周围背景环”。更准确的表述是：

> 利用 phase-aligned local average 构造通道内空间 headroom。

### 5.4 Max-based Saliency 可能同时增强杂波

`MaxPool-AvgPool` 对亮点、噪声尖峰、边缘和目标都会产生响应。Context headroom 是否能把目标与 hard negative 分开，正是 Full 相对 Capacity 必须证明的内容。

若 Capacity 优于 Full，应接受 Context 假设失败，而不是继续增加更复杂 gate 来“保住”该创新点。

### 5.5 强 identity bypass 可能弱化 tokenizer 梯度收益

即使 V6 token 更优，decoder 仍接收到强原始 encoder skip。可能出现：

- scale 长期接近 0；
- Full 与 Capacity 输出差异很小；
- tokenizer 指标变化明显，但最终 segmentation 指标不敏感。

这需要 block-level 机制统计，而不是只看最终 mIoU。

### 5.6 当前样本量不足以支撑稳定性结论

验证集只有 189 个目标时，一个目标对应：

\[
1/189\approx0.5291\%
\]

的 Pd 变化。`187/189` 与 `188/189` 的差异在数值上非常离散。两个 seed 可以做工程筛选，但不足以形成论文级稳定性主张；最终仍需要更多 seeds、多数据集以及官方测试集。

---

## 6. 创新性审查

### 6.1 哪些组成部分本身不是新颖点

| 组成 | 已有研究背景 | 不能单独主张的创新 |
|---|---|---|
| `PixelUnshuffle/space-to-depth + Conv` | SPD-Conv 已用于小目标和低分辨率任务 | 不能把 SPD 结构本身作为新贡献 |
| AvgPool / MaxPool / 混合 pooling | 经典 pooling 与混合 pooling 已广泛存在 | 不能把 `max-avg` 本身作为充分创新 |
| 红外小目标局部对比 | ALCNet、LCAE 等已有大量局部对比先验 | 不能只写“引入局部对比” |
| Context modulation | ACM 等已在 IRSTD 中使用上下文调制 | 不能只写“Context 调制 Saliency” |
| 保护浅层局部特征 | HintU 等已明确研究早期下采样导致的小目标信息损失 | 不能只写“防止浅层目标丢失” |
| SCTransNet 多尺度交互 | 来自基线原论文 | 不能归入新增模型贡献 |

截至 2026-07-26，还必须纳入三项比原参考列表更接近的工作：

| 最近工作 | 与 V6 的重合 | V6 仍可辨识的差异 |
|---|---|---|
| Adaptive Polyphase Sampling（CVPR 2021） | 显式处理二维下采样的四个 polyphase 分量 | APS 选择一个 phase 以改善平移一致性；V6 保留并压缩全部 phase |
| SDANet（2026） | 直接研究 IRSTD 的 strided downsampling 损失，并融合小波高低频 | SDANet 采用小波双域建模；在已核对的公开方法描述中未见从主压缩核派生 K/C/S 投影 |
| InvDet（CVPR 2026） | 用可逆早期编码与重建约束处理红外微小目标信息损失 | InvDet 的理论与整网证据更完整；在已核对的公开方法描述中未见 V6 的 K/C/S 相位权重绑定 |

这三项不会直接否定 V6，但会明显压缩“目标保真下采样”“相位感知”
和“高低频/细节保持”等宽泛表述的新颖空间。V6 必须把贡献收窄到具体的
权重派生关系和同容量因果对照。

### 6.2 V6 真正可能构成创新的部分

V6 的潜在创新不是单个算子，而是以下组合关系：

1. **面向 SCTransNet 浅层跨尺度 tokenization 的 K/C/S 表示分解**；
2. **从 dense Keep 的 phase 权重中派生零新增参数的共享输出坐标映射**；
3. **Context 与 Saliency 不增加独立 projection 参数，而与 Keep 的 phase 响应绑定**；
4. **零初始化 Saliency residual，使候选在共享 Keep `weight+bias` 时于 step 0 严格等价于 dense SPD**；
5. **Full/Capacity 同参数、同 state、同初始化，单独检验 Context gain redistribution**；
6. **有界 residual 控制 Saliency 注入幅度**。

其中 1–4 和 6 属于方法设计，5 主要是同容量因果对照设计；预注册 Gate
属于证据协议，不应包装成模型创新点。

在当前已核对文献范围内，最有辨识度的创新核心是：

> **Phase-tied shared-coordinate projection：复用 Keep 的四相位压缩权重，
> 将 Context/Saliency 映射到由 Keep 权重约束的输出通道坐标，而不引入
> 独立 projection 参数。**

这是比“max-avg 显著性”更值得作为论文方法点的内容。

### 6.3 当前创新强度评级

以下为基于本地实现、公开方法描述与现有证据的研究判断，不是期刊官方评分：

| 维度 | 评分 | 说明 |
|---|---:|---|
| 问题定义清晰度 | 8/10 | 聚焦浅层 tokenization 中的小目标压缩问题 |
| 单模块概念新颖度 | 5.5/10 | 算子多为已有成分，组合和绑定方式有一定新意 |
| 数学/控制设计 | 7/10 | phase-tied projection、SPD exact start、同容量对照较扎实，但 Saliency/DC 投影仍是假设 |
| 工程严谨性 | 9.5/10 | Gate、paired control、exact entry、恢复轨迹、CPU/GPU 2/3 smoke、两卡串行 lane、四组正式训练、四份独立 source lock、八份 sweep 与完成性接受均已落地 |
| 当前性能证据 | 5/10 | 已有两个 seed、四组 800-epoch 终点和五预算联合比较；Gate B 证明 seed 42 存在竞争力工作区间，但 Gate A/C/D 失败，综合性能和稳定性未达门槛 |
| 跨数据集稳定证据 | 1/10 | 当前没有 V6 多数据集/官方测试证据 |
| 当前论文核心成熟度 | 4/10 | 模块、代码和内部验证闭环完整，但正式裁决未建立 Full 的稳定结构优势，尚不能作为论文核心 |

### 结论

- **作为硕士/博士阶段的方法增量或一般工程型论文模块：有创新潜力。**
- **作为高水平期刊/会议的单一核心贡献：当前证据和概念强度还不够。**
- **本轮 V6 尚未稳定超过 SPD 和同容量 control，不能把候选创新写成已经成立。**
- **下一版若在不改变 K/C/S 主线的前提下通过联合门槛并完成多数据集复现，可形成中等强度模块创新。**
- **若随后 NER 也通过独立消融，整体故事可提升为更完整的架构贡献。**

### 6.4 最合适的论文叙事

不建议写成模块堆叠：

```text
TPD + NER + Survival + FG
```

更合适的统一问题链是：

```text
浅层跨尺度 tokenization 压缩小目标证据
        ↓
Phase-tied KCS tokenizer 在压缩前显式组织并对齐局部证据
        ↓
五节点 NER 中继多级 tokenization 的中间证据
        ↓
Query-only FG 在注意力查询端使用待独立对照验证的频率先验

Survival supervision：仅在训练期约束 emb1/emb2，先独立验证，再决定
是否与已经通过的 NER 候选组合；它不是推理期串行结构节点。
```

但这个故事必须按顺序建立：

1. tokenizer 单独成立；
2. NER 通过 tokenizer×relay 的交互对照；
3. Survival 通过 tokenizer×auxiliary-loss 的交互对照；
4. 只有独立模块通过后才测试组合；
5. Query FG 最后验证。

在 tokenizer 未通过前把全部模块接入，会削弱归因可信度，也更像复杂度堆叠而非创新。

---

## 7. 当前方案是否需要改公式

### 结论：不回改已冻结 V6；在新版本中优化 K/C/S 内部公式

V6 已经形成完整、可复核的失败终态。不能回改 V6 源码、checkpoint 或 Gate
来改变该结论；下一版应使用新 variant、新 builder 和新协议文件实现
phase-aware 优化。这样既保留 V6 的正式证据，又能继续推进模型代码。

因此应：

> **永久冻结 V6 作为已完成对照；下一版只在 K/C/S tokenizer 内部修复
> phase identity、phase-sum cancellation 和低虚警稳定性，不增加第四语义源。**

### 已在正式训练前完成的非公式修改

1. 将“mean-neutral fusion”改为“zero-mean gain redistribution”；
2. 将“目标保真下采样”在论文方法处限定为“目标保真跨尺度 tokenization”；
3. 明确 `PixelUnshuffle + 4C→C Conv` 不是严格无损；
4. 增加诊断导出，不注册新 parameter/buffer，不改变 forward 输出；
5. 完成 RTX 5090 GPU 2/3 smoke、源码绑定持久报告和正式 source lock。

### 下一版仍不应同时做的修改

- 不增加第四分支；
- 不增加新 loss；
- 不接 NER；
- 不接 Survival；
- 不接 Query-only FG；
- 不改 encoder MaxPool；
- 不改双 identity skip；
- 不根据中途结果更改 Gate。

---

## 8. V6 结果后可用的非阻塞诊断

以下诊断不改变 forward，也不是下一版代码实现的前置门槛。V6 正式工程与
结果接受均已闭环；不应为了增加在线 hook 改动冻结源码。能够从 checkpoint
恢复的状态可由独立脚本只读计算，但当前重心是先实现新 tokenizer 与可运行
builder，诊断只服务于结构选择，不取代模型开发。

| 诊断 | 定义或输出 | 目的 |
|---|---|---|
| scale utilization | 每 block 的 `median/p90/max(abs(tanh(scale)))` | 分支是否真正被使用 |
| residual ratio | `E[abs(R)] / (E[abs(K)]+eps)` | residual 是否过弱或过强 |
| phase cancellation | `rho_l` | 判断 `phase-sum(W)` 是否抵消 |
| aligned Saliency energy | `E[abs(Sa)]` 与 `E[abs(S0)]` | 投影后显著性是否消失 |
| gain mean | `mean_hw(H)` | 验证 gain-map 平均值为 1 |
| actual residual mean shift | `mean(Sa*H)-mean(Sa)` | 避免错误声称 residual mean-neutral |
| target lift | 目标 cell 与 hard-negative cell 的 `R/K` 差异 | 判断增强的是目标还是杂波 |
| block depth contribution | 七个 block 分别记录上述指标 | 判断后层注入是否有害 |
| attached halo | matched component 的 `pred\gt` 面积 | 解释 Pd 与 mIoU 冲突 |
| independent false alarm | 未匹配连通域数量、面积、峰值 | 解释 Fa 来源 |
| threshold margin | 最弱真目标峰值与最强假目标峰值差 | 区分结构问题与校准问题 |

优先级建议：

```text
V6 结果后只读：scale utilization / residual ratio / phase cancellation
新版本预检后选做：target lift / halo / false alarm / threshold margin
```

建议增加只读分析脚本，而非把长期 hook 固化在正式模型中：

```text
analysis/analyze_v6_block_mechanism.py
analysis/analyze_v6_component_errors.py
analysis/compare_v6_full_capacity.py
```

---

## 9. 正确的下一步执行计划

### P0：文档和 claim 修正（已完成并持续校准结果边界）

- 修正“均值中性”的边界；
- 将方法定位为 shallow cross-scale tokenization；
- 明确不做 lossless 声明；
- 记录最小非阻塞诊断指标；
- 冻结 V6 forward 公式和 Gate A–E。

### P1：工程闭环（已完成）

以下项目已全部完成：

```text
V6 专用 exact training entry
RTX 5090 smoke（物理 GPU 2/3）
exact epoch-boundary resume
source lock
paired initialization audit
full-network step-0 SPD exactness
strict checkpoint reload
CPU 两个 optimizer step
```

上述条件均已满足，四组正式实验及其后处理均已完成。

### P2：四组正式实验（已完成）

| Variant | Seed | Epochs |
|---|---:|---:|
| `tpd_clean_v6_full` | 42 | 800 |
| `tpd_clean_v6_full` | 3407 | 800 |
| `tpd_clean_v6_phase_capacity` | 42 | 800 |
| `tpd_clean_v6_phase_capacity` | 3407 | 800 |

保持：

- FP32；
- 530/133 split；
- threshold 0.5；
- `best / best_miou / last`；
- 闭区间 sweep；
- 原 Gate A–E；
- 不提前停训，不在中途选方向。

checkpoint 角色必须保持：

```text
best.pth.tar       → best_validation_pd_primary → 报告中的 pd_primary
best_miou.pth.tar  → best_validation_miou_secondary → 报告中的 miou_primary
last.pth.tar       → last_evaluated_epoch，仅用于完整性和续训
```

Gate D 同时检查两个 checkpoint role 的固定阈值和五个 budget。“严格覆盖”
定义为 Pd 不低、Fa 不高、mIoU 不低且至少一项严格更好。Gate A 中
`mIoU>=0.946542` 是冻结的六位小数门槛；SPD 锚点完整精度为
`0.9465418781725888`，两者不应描述为完整精度完全相等。

正式后处理锁保持不变。复核发现其正常 fresh-run 路径正确，但原汇总器
没有独立锁死完整阈值网格，也没有验证每个点的
`tiny_pd=matched_tiny/39`。因此新增两层只读接受代码：

```text
validate_tpd_clean_v6_strict_sweeps.py
→ 完整 base/tail/quantile/端点阈值并集
→ fixed/budget 最优点
→ Pd/tiny-Pd/目标数/Fa 像素格点等式

accept_tpd_clean_v6_formal800_results.py verify
→ 原冻结 completion verify
→ strict sweep 8/8
→ 两层同时通过后形成 frozen acceptance
```

最终 authoritative acceptance 还要求 checkpoint-metric compatibility
第三层达到 `8/8`；其边界与独立源码锁见 3.22。

新增代码与测试共 21 项 CPU 测试通过，并以独立 supplemental source lock
绑定；其 SHA-256 为
`dcaf2f1b32cff5096511ba090e3149327deea1f32f2a51d4d866bb0d0cf32696`。
正式终态为 `strict_valid=8/8`、`compatibility_valid=8/8`、
`authoritative_result_accepted=true`。这不是新的性能门槛，而是确认
所有 Gate 输入完整、计数一致并可接受。supplemental 的 5 项文件与
compatibility 的 10 项文件当前仍是工作区新文件；最终代码归档须将这
15 项完整纳入，不能只保存锁文件。

### P3：结果决策（已完成）

| 结果模式 | 解释 | 决策 |
|---|---|---|
| 全部 Gate A–E 通过，且 Gate D 在两个 seed 上通过 | Context headroom 有独立价值 | 冻结 V6，授权进入 NER 工程与交互实验阶段 |
| Capacity 优于或覆盖 Full | 当前 Context headroom 未建立独立价值 | 保持 K/C/S 研究主线，Full 不作为已建立贡献 |
| Full/Capacity 都接近 SPD，scale≈0 | 模型未使用 Saliency residual | 不启动 NER；检查梯度和 identity bypass |
| scale 活跃但 `rho` 很低、`Sa` 很弱 | phase-sum cancellation | 转向 phase-resolved V7 |
| Pd 提升、mIoU 降、halo 增加 | phase location collapse 或后层重复注入 | 优先 phase-resolved 设计，再评估 early-only mask |
| seed 42 过、3407 不过 | 稳定性不足 | 审计采样流和目标级错误，不做稳定性主张 |
| Full/Capacity 都被 SPD 覆盖 | 当前 V6 实现未建立竞争力 | 不进入 NER 工程阶段；保持目标保真/KCS 研究主线，只优化内部相位对齐 |

本轮实际落在“seed 42 有竞争力工作区间、seed 3407 不稳定，且 Capacity
在部分关键点覆盖 Full”的组合情形。因此 P3 已执行为：

```text
冻结 V6 正式产物
→ 不授权 NER
→ 保持目标保真/K/C/S 主线
→ 新建 phase-resolved K/C/S tokenizer 优化版本
```

---

## 10. V6 失败后的首选后备方向

若正式 V6 结果及诊断同时显示 `phase-sum cancellation` 或 phase location
丢失，下一步应优先设计 **Phase-Resolved Saliency Alignment**，而不是
继续增加 Context gate。它属于 K/C/S 内部 Saliency 对齐方式的优化，
不改变“浅层目标保真 tokenization”主线，也不增加第四语义源。

基本思路：

\[
D_p=\operatorname{ReLU}(z_p-C_0),
\]

保留每个 phase 的正对比响应，再使用完整的：

\[
W_{o,c,p}
\]

投影，而不是先压成单个 `S0` 后只使用：

\[
\sum_pW_{o,c,p}.
\]

该方向可同时解决：

- 峰值 phase 身份丢失；
- phase 权重求和抵消；
- Saliency 使用 DC 投影的不匹配。

同时仍可做到：

- 不增加新的语义分支；
- 不增加可学习参数；
- scale=0 时等价 SPD；
- Full/Capacity state layout 一致。

V6 的 Gate C/D 结果已经满足启动这一后备设计的工程理由。新版本必须另建
源码和协议，不能替代或回写 V6 正式产物。

---

## 11. NER、Survival 与 Query-only FG 的位置

### 11.1 NER

NER 是最可能提升“整体模型创新度”的下一层，因为它不再只处理单个
downsampling block，而是显式中继五个 tokenization 中间节点的目标证据。
“存活”是否得到监督应留给独立的 Survival 模块描述，不能仅凭 relay
结构就声称已经建立目标存活监督。

但目前 NER 代码仍与 V5 concrete class/variant 强耦合。后续 K/C/S
tokenizer 通过联合门槛后，应先抽象统一接口，例如：

2026-07-27 已对仓库现有通用 relay、显式 `q4→q3→q2` forward、V5
五节点模型和成对 builder 做只读回归，相关 30 项 CPU 测试全部通过。
这只确认 relay 核心可复用，不表示新 tokenizer composer、exact entry 或
正式 tokenizer×relay 交互矩阵已经实现，也不改变当前
`ner_stage_authorized=false` 的边界。

```python
class EvidencePatchEmbedding:
    blocks: nn.ModuleList

    def forward(self, x): ...
    def forward_with_evidence(self, x): ...
```

不要让 NER 通过 `isinstance(TPDCleanV5Block)` 识别 tokenizer。

后续 tokenizer 通过联合门槛后，也不能只训练一个 `KCS+NER` 组合。NER
必须保持原主线中的 tokenizer×relay 交互矩阵：

| Tokenizer | Relay off | 相同拓扑 Relay on |
|---|---|---|
| 参数匹配 Progressive | P | P+N |
| 达标 K/C/S tokenizer | T | T+N |

只有当两个 seed 上 `T+N−T` 的关键工作点增量方向一致，并大于
`P+N−P` 的通用 relay 增量，且最严格 Fa 区域不退化，NER 才能并入后续
整模。tokenizer 门槛通过只授权编写和运行该 NER 工程矩阵，不直接建立
NER 论文贡献，也不等于立即开始一个未经预检的 NER 正式任务。

### 11.2 Survival supervision

Survival 是训练期辅助监督，应使用交互消融，而不只是比较一个组合：

```text
dense SPD / 达标 K/C/S tokenizer × Survival off / on
```

该 `2×2` 设计用于区分“普遍有效的辅助 loss”和“对 K/C/S tokenizer 特别有效的目标
存活约束”。原始 PE 可作为补充参照。只有 Survival 自身交互成立、NER
自身交互也成立后，才测试组合；否则不把它并入最终模型。

### 11.3 Query-only FG

Query-only FG 改变注意力查询，归因链比 tokenizer 更长，应最后接入。
现有仓库已有独立 Survival、Query-only FG 与 Query bridge 组件，但还没有
面向达标 K/C/S tokenizer 的 composer、统一 forward 或正式训练入口；
“未接入”不等于“完全没有
组件代码”。

```text
达标 K/C/S tokenizer
→ K/C/S+NER（若 NER 交互门通过）
→ K/C/S+NER+Survival（若 Survival 交互门通过）
→ K/C/S+NER+Survival+Query-only FG
```

该箭头表示通过各自门槛后的整模并入顺序，不表示 Survival 跳过独立验证。
Survival 必须先完成上述 `dense SPD / 达标 K/C/S tokenizer × off / on`
的独立 `2×2`
交互门，再决定是否与已通过的 NER 候选组合。

FG 还必须比较 Haar 与空间域/其他高频先验及无 FG 对照，证明收益来自
Query 端频率引导，而不是新增参数。它不应在 tokenizer 未通过联合门槛时
并行正式训练。

---

## 12. 推荐的论文贡献表述

若后续 K/C/S 优化版最终通过，可以将第一项贡献写成：

> We introduce a phase-tied KCS tokenizer for the shallow cross-scale interaction path of SCTransNet. It performs space-to-depth rearrangement before compression, decomposes local features into Keep, Context, and Saliency sources, and derives the Context/Saliency projection from the dense Keep phase weights, constraining all three outputs to a shared learned channel coordinate without independent projection parameters.

第二项可写成：

> With shared Keep weights and bias, a zero-initialized and bounded Saliency residual makes the tokenizer functionally identical to its dense SPD reference at initialization, while a parameter- and state-matched capacity control isolates the effect of Context-based spatial gain redistribution.

必须避免以下过度主张：

```text
lossless downsampling
fully phase-preserving Saliency
residual mean-neutrality
whole-network target preservation
state-of-the-art stability（在当前阶段）
```

只有 NER 通过 tokenizer×relay 的 `2×2` 交互实验后，才能增加第三项架构
贡献：

> Explicit relay of intermediate target evidence across hierarchical tokenization nodes.

---

## 13. 最终判定

### 13.1 整体模型设计如何

**总体设计合理，筛选协议纪律较强，V6 的实现、正式训练、后处理和结果接受
均已闭环。**
它选择了一个与小目标信息压缩直接相关的入口，并保持主体网络与协议冻结。
其局限是作用范围只在 SCTransNet 的浅层跨尺度 tokenization path，且
受到主 encoder pooling 和双 identity bypass 的结构边界约束。正式结果
表明该构造有局部竞争力，但当前公式没有形成跨 seed 稳定的联合优势。

### 13.2 当前方案是否正确

**作为 V5 失败后的受控候选，V6 方案与实现是正确的。** 它消除了 V5 中未显式约束的
通道映射，参数绑定、残差边界、SPD 精确起点和配对对照均成立；但这不等于
已经证明三种来源具有完全相同的学习语义。

但正确性仅到以下程度：

```text
implementation-correct
mathematically coherent
formal-evaluation-complete
joint-performance-gate-failed
```

正式结果没有建立“把 phase-collapsed Saliency 当作四相位等值特征并使用
phase-sum/DC 权重投影能够稳定保留目标对比”这一假设。Gate C/D 为
phase-resolved 代码候选提供了工程动机，但不能单独证明 phase-sum 或
phase collapse 就是失败原因。

### 13.3 是否满足创新

**目前属于“有潜力的中等模块级创新”，还不是“已经建立的论文核心创新”。**

创新的核心不在 SPD、MaxPool、AvgPool 或 Context 本身，而在：

```text
phase-tied shared-coordinate projection
+ zero-additional-parameter Context/Saliency projection
+ SPD-exact initialization
+ parameter/state-matched causal control
```

要将后续 K/C/S 优化版本升级为可靠论文贡献，至少需要：

1. Pd、Fa、mIoU 联合门槛通过，且同容量结构优势在两个 seed 上通过；
2. Full 对 Capacity 有稳定、不可被覆盖的优势；
3. 相对 SPD、TPD-v1 和 V5 有明确、可复核的性能提升；
4. 结果后诊断支持 phase-tied projection 与目标响应变化相关；
5. 多 seed、多数据集和官方测试结果；
6. 参数量、FLOPs、延迟和稳定性报告。

### 13.4 当前项目状态

```text
decision=ENGINEERING_GATE_FAIL
project_phase=KCS_TOKENIZER_INTERNAL_OPTIMIZATION
execution_state=V6_FORMAL800_COMPLETE_ACCEPTED
v6_model_code_implemented=true
v6_thin_train_eval_wrappers=true
v6_exact_training_entry_implemented=true
v6_formal_amp_off_enforced=true
v6_two_step_smoke_entry_implemented=true
v6_cpu_two_step_execution_passed=true
v6_cpu_two_step_report_available=true
v6_physical_gpu_smoke_report_available=true
v6_smoke_report_set_verified=true
v6_exact_resume_trajectory_test_passed=true
v6_source_lock_freezer_implemented=true
v6_formal_source_lock_available=true
v6_formal_source_lock_sha256=2de1a8f75deb321b5aec4cf5dfa6bc16df8443e858e1d48a3ab6bea34de526d2
v6_postprocess_code_implemented=true
v6_postprocess_tests_passed=33
v6_postprocess_source_lock_available=true
v6_postprocess_source_lock_sha256=3cfbfda891d823c5b97d2d1a2364790c823fac9a548bbf0987444979619bd827
v6_postprocess_source_count=12
v6_postprocess_frozen_reference_count=10
v6_supplemental_strict_sweep_validator_implemented=true
v6_authoritative_result_acceptance_entry_implemented=true
v6_supplemental_acceptance_tests_passed=21
v6_supplemental_acceptance_source_lock_available=true
v6_supplemental_acceptance_source_lock_sha256=dcaf2f1b32cff5096511ba090e3149327deea1f32f2a51d4d866bb0d0cf32696
v6_supplemental_acceptance_repository_archive_pending=true
v6_checkpoint_metric_compatibility_source_lock_available=true
v6_checkpoint_metric_compatibility_source_lock_sha256=fd3a11d4b48f0990538554d92759e8c5b6e4be178407483f64f0069a65439f93
v6_checkpoint_metric_compatibility_repository_archive_pending=true
v6_strict_valid_sweeps=8/8
v6_compatibility_valid_sweeps=8/8
v6_authoritative_result_accepted=true
v6_completion_manifest_sha256=27b168b1b4ae59e5bd8db15702b959225b879338e32427a9b814e8640e6f0188
v6_completion_marker_sha256=37bcaccf3365cbe57edab94fc464fa2a41d85cc3b70eed7c5db276af16650741
v6_postfreeze_applicable_regression_tests_passed=86
v6_postfreeze_regression_tests_skipped=1
v6_prelock_absence_assertion_not_applicable_after_freeze=true
v6_formal_finalizer_completed=true
v6_two_gpu_runtime_manager_implemented=true
v6_same_gpu_concurrent_training_jobs=1
v6_formal_launch_started=true
v6_active_gpu2_run=NONE
v6_active_gpu3_run=NONE
v6_seed3407_runs=COMPLETE
v6_ner_integrated=false
generic_five_node_relay_regression_tests_passed=30
v6_ner_composer_implemented=false
v6_ner_exact_entry_implemented=false
independent_survival_and_query_fg_components_present=true
v6_preflight_complete=true
v6_partial_epoch_artifacts_available=true
v6_seed42_800_endpoint_joint_audit_passed=true
v6_full_gate_a_seed42_endpoint_subchecks_passed=3/6
v6_gate_a_seed42_passed=false
v6_current_round_ner_authorization_possible=false
v6_stage_snapshot_epoch_cutoff=800
v6_completed_formal_runs=4/4
v6_seed42_wave_epoch_progress_at_cutoff=1600/1600=100%
v6_total_formal_epoch_progress_at_cutoff=3200/3200=100%
v6_remaining_formal_epochs_at_cutoff=0
v6_seed3407_first_250_epoch_joint_audit_passed=true
v6_seed3407_first_300_epoch_joint_audit_passed=true
v6_seed3407_first_350_epoch_joint_audit_passed=true
v6_seed3407_first_400_epoch_joint_audit_passed=true
v6_seed3407_first_450_epoch_joint_audit_passed=true
v6_seed3407_first_500_epoch_joint_audit_passed=true
v6_seed3407_first_550_epoch_joint_audit_passed=true
v6_seed3407_first_600_epoch_joint_audit_passed=true
v6_seed3407_first_650_epoch_joint_audit_passed=true
v6_seed3407_first_700_epoch_joint_audit_passed=true
v6_seed3407_first_750_epoch_joint_audit_passed=true
v6_seed3407_800_endpoint_joint_audit_passed=true
v6_seed3407_common_epoch_cutoff=800
v6_seed3407_wave_epoch_progress_at_common_cutoff=1600/1600=100%
v6_total_formal_epoch_progress_at_common_cutoff=3200/3200=100%
v6_remaining_formal_epochs_at_common_cutoff=0
v6_real_sweeps_available=8/8
v6_formal_gate_evaluated=true
v6_gate_a_passed=false
v6_gate_b_passed=true
v6_gate_c_passed=false
v6_gate_d_passed=false
v6_gate_e_passed=true
v6_engineering_gate_passed=false
v6_completed_seed42_results_available=true
v6_completed_formal_matrix_available=true
v6_formula_freeze_recommended=true
v6_formal_training_completed=true
ner_stage_authorized=false
mainline_changed=false
paper_core_established=false
stability_claim_supported=false
next_stage=KCS_TOKENIZER_INTERNAL_OPTIMIZATION
```

V6 正式训练授权条件：

```text
structure_state_parameter_gradient_tests_pass
&& cpu_two_optimizer_steps_and_strict_reload_pass
&& gpu_smoke_pass
&& exact_resume_pass
&& source_lock_pass
&& paired_initialization_pass
&& step0_spd_exactness_pass
```

已完成与推荐执行链：

```text
修正文档 claim
→ 已完成 V6 专用 exact training entry
→ 已完成 CPU 两步与 strict reload 计算测试
→ 已完成 exact resume 轨迹等价测试
→ 已完成 RTX 5090 GPU 2/3 smoke
→ 已生成并验证源码绑定持久报告
→ 已完成两卡 worker/lane/launcher/status 代码
→ 已冻结并验证正式 source lock
→ 已通过四任务双 lane preflight
→ 已完成并冻结独立 postprocess/Gates 源码锁
→ 已完成 V6 Full/Capacity × seeds 42/3407 × 800 epochs
→ 已生成并严格复核八份闭区间 sweep
→ 已发布 comparison、completion manifest 与 Gate A–E 结果
→ 已接受正式裁决：Gate B/E 通过，Gate A/C/D 未通过
→ 当前保持主线，实现 K/C/S 内部相位对齐优化
→ 新 tokenizer 通过联合门槛后，才进入 NER 工程与 tokenizer×relay 交互实验
```

---

## 参考资料

1. [Arialliy/SCTransNet_main：TPD 实验仓库](https://github.com/Arialliy/SCTransNet_main)
2. [SCTransNet: Spatial-channel Cross Transformer Network for Infrared Small Target Detection](https://arxiv.org/abs/2401.15583)
3. [SCTransNet 官方实现](https://github.com/xdFai/SCTransNet)
4. [No More Strided Convolutions or Pooling: SPD-Conv](https://arxiv.org/abs/2208.03641)
5. [Asymmetric Contextual Modulation for Infrared Small Target Detection](https://openaccess.thecvf.com/content/WACV2021/html/Dai_Asymmetric_Contextual_Modulation_for_Infrared_Small_Target_Detection_WACV_2021_paper.html)
6. [Attentional Local Contrast Networks for Infrared Small Target Detection](https://arxiv.org/abs/2012.08573)
7. [Lost in UNet: Improving Infrared Small Target Detection by Underappreciated Local Features](https://arxiv.org/abs/2406.13445)
8. [LCAE-Net: Paying More Attention to Local Contrast](https://arxiv.org/abs/2411.13260)
9. [Truly Shift-Invariant Convolutional Neural Networks: Adaptive Polyphase Sampling](https://openaccess.thecvf.com/content/CVPR2021/html/Chaman_Truly_Shift-Invariant_Convolutional_Neural_Networks_CVPR_2021_paper.html)
10. [Rethinking Downsampling in UNet: Stride-Free Dual-Domain Learning for Infrared Small Target Detection](https://www.sciencedirect.com/science/article/pii/S0030399226004706)
11. [Target-Aware Invertible Encoder with Reconstruction Guidance for Infrared Small Target Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_CVPR_2026_paper.html)
12. [PyTorch Automatic Mixed Precision: Autocast Op Reference](https://docs.pytorch.org/docs/stable/amp.html)
