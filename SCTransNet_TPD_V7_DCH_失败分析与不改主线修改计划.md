 # SCTransNet–TPD V6 失败分析与不改变主线的 V7-DCH 修改计划

> 面向单帧红外小目标检测的下一阶段模型、代码、诊断与晋级方案  
> 基线：SCTransNet  
> 当前主线：浅层 Target-Preserving Downsampling / Patch Embedding  
> 已通过实现门的 P0 训练轨迹假设：**TPD-Clean V7-DCH（Deferred Context Headroom）**

---

## 0. 执行结论

### 0.1 当前裁决保持不变

V6 的正式结果、工程结果和失败裁决应全部保留，不回写、不覆盖、不通过修改指标解释为成功：

```text
decision=ENGINEERING_GATE_FAIL
authoritative_result_accepted=true
ner_stage_authorized=false
mainline_changed=false
paper_core_established=false
stability_claim_supported=false
```

Gate B 和 Gate E 说明 V6 **具备局部性能潜力且工程证据完整**；Gate A、C、D 失败则说明该潜力没有形成固定阈值质量、跨种子稳定性和相对等容量对照的可靠优势。seed 3407 的严格低 Fa 尾部存在高置信目标内部碎裂；固定阈值失败则同时包含目标内部碎片、背景/邻域响应和区域重叠质量不足，不能把两者混写为同一个原因，也不能通过后处理、放宽匹配或挑选阈值掩盖。

### 0.2 冻结诊断已完成：不启动 NER，实施 V7-DCH

冻结诊断已将下一步的 **P0 训练轨迹假设** 定为：

> **TPD-Clean V7-DCH：保持 V6 的 Keep–Context–Saliency 三源、phase-tied projection、参数布局和浅层替换范围，候选修改仅位于 Context headroom。**

V6 failure atlas、Context-off、residual-off、phase/block 诊断已经完成。
8/8 个正式 checkpoint 的 as-trained 复算与 formal sweep 数值差为 0；
56/56 个 block 的 `abs(tanh(saliency_scale))<0.5`，全局最大值为
`0.3624752164`；冻结 residual-off 没有单独解释注册失败。因此一次性
Go/No-Go 输出为 `GO_DCH_TRAJECTORY_TEST`，现在允许设置
`v7_dch_formula_frozen=true` 并实施严格配对的 fresh 训练。

该结果同时为 `CONTEXT_DIRECT_SUPPORT=false`：冻结 Context-off 没有建立
即时前向因果作用。它只授权检验 DCH 的训练轨迹假设，不授权写成
“DCH 机制已经证明”，也不授权 NER。

核心变化为：


a. V6 在零 Saliency scale 处只有**输出等价锚点**，Full 与 Capacity 的 Saliency scale 一阶梯度并不相同；

b. V7-DCH 将该锚点加强为**输出与一阶优化同时等价的锚点**；

c. Context 不再从第一次参数更新就改变 Saliency 的学习方向；其 Context residual 在零点附近是 \(O(|a|^2)\) 量级。这里的“二阶量级”只描述幅度阶数，**不声称公式在零点二阶可微**；

d. 不增加参数、不增加 buffer、不增加第四分支、不改 backbone、SCTB、decoder、loss、数据、checkpoint 选择、Pd/Fa/mIoU 定义或 Gate A–E。

### 0.3 对“顺利进入下一步”的准确理解

代码修改可以消除一个已经由公式直接确认的优化不对称，并建立完整的诊断、测试和 NER 接口准备流程；但任何模型公式都不能在训练前保证通过性能门槛。因此本方案的目标是：

> 先用冻结 V6 诊断判断 DCH 是否值得成为正式候选；若 Go，再以最少变量、最高可归因性验证它能否修复失效。NER 的硬授权条件仍然只有原 Gate A–E；碎裂机制复核只约束“机制已修复”这一研究主张，不新增 NER 性能硬门。

### 0.4 当前实际执行状态（2026-07-27）

V7-DCH 模型、普通/exact 入口、17 字段 checkpoint/summary 契约、双卡运行
链、closed sweep、Gate A–E 汇总和 Mechanism Audit M 代码均已形成。完整
DCH 定向回归结果为
`126 passed, 1 skipped, 562 subtests passed`。

CPU、物理 GPU 2 和 GPU 3 的持久 smoke 已联合通过；两个 seed 的
Full/Capacity 配对初始化、零点输出/梯度契约和第一次 Adam 更新已验证。
三类 source lock 已分别冻结并反向验证：

```text
Diagnostic SHA256=5f99bb511cb140cd502dcf41329f698b338d41e7404e6f897cf84ce3ab241a92
Training   SHA256=e67305d53b59336194541e2a9e6bec5bab3682c77232feb8be3e0fe71ea76c95
Acceptance v2 SHA256=ee7be009081b1776b6e5068c9c39b7f4429c987a44cea0a25f7c95f27fc8f130
```

旧 acceptance v1（`4fb4668d...`）和旧 diagnostic
（`edd67063...`）按原字节保留为 superseded evidence；当前 completion、
summary 和 final decision 只接受 acceptance v2。training lock 未变化，
正式训练源码身份仍为 `e67305d...`。

四任务 launcher preflight 全部判定为 `fresh`。正式训练已于
`2026-07-27 10:43:58 CST` 启动：

```text
physical GPU 2: Full/seed 42 -> Capacity/seed 3407
physical GPU 3: Capacity/seed 42 -> Full/seed 3407
```

每张卡只并发一个任务，第二个任务在同 lane 串行等待。GPU 0、1 未被本轮
任务使用。当前状态只表示工程启动成功，不表示 Gate A–E、机制主张或论文
结论已经通过。

训练后闭环已增加独立的无人值守 finalizer，并于
`2026-07-27 12:27:37 CST` 启动等待。它在任一训练 lane 活跃或四任务矩阵
不完整时以 `75` 返回并由 systemd 定时重试，不创建 sweep 或 comparison
产物；只有两条 lane 均结束且四组 800 epochs 完整后，才按下列固定顺序
执行：

```text
8 closed sweeps（按各 run 原训练卡回放 GPU 2/3）
-> comparison / Gate A–E
-> completion publish + verify
-> Mechanism Audit M（物理 GPU 2）
-> final decision
-> 全链重验与 control manifest
```

该 finalizer 是操作控制层，不扩张科学 acceptance source set；最终 control
manifest 会单独绑定四个控制文件和 27 份正式后处理产物。

---

## 1. 评审范围与证据边界

本报告基于本地仓库和正式实验资产完成交叉核对：

1. `/home/ly/SCTransNet_main` 当前源码中的 SCTransNet、V6、V7 草案、训练、评估、精确续训与协议代码；
2. V6 formal800 的比较 JSON、8 份 sweep、4 个 run 的 `best`、`best_miou`、`last` checkpoint 与运行记录；
3. seed 42/3407 的固定阈值、Fa budget 和高阈值 component 结果；
4. 对 V6/DCH 融合公式的一阶梯度推导和本地最小代码复核；
5. 对 8 个正式角色 checkpoint 的 `saliency_scale` 与 phase-sum 权重统计。

现有本地证据支持以下边界：

- seed 3407 Full/Pd-primary 的严格低 Fa 尾部存在高置信目标内部碎裂；
- 固定阈值 `0.5/0.58` 的 pixel precision 分别约为 `0.916/0.928`，因此固定阈值失败不能排除背景/邻域响应和区域覆盖不足；
- “Context 首步梯度不对称”可由代码、数学和本地最小复核直接确认；
- 正式 checkpoint 的 `saliency_scale` 未饱和，phase-sum 全局 L2 比例约为 `0.50–0.58`，当前不支持“全局严重 phase cancellation 已被确认”；
- 冻结诊断没有建立 Context headroom 的即时前向因果作用，也没有发现
  residual-off 单独解释失败；局部 block 或 decoder 放大仍属于待检验
  解释，不得把相关性写成既定因果。

---

## 2. “不改变主线”的冻结约束

以下内容必须锁定，V7-DCH 不得修改。

| 层面 | 冻结内容 |
|---|---|
| 研究问题 | 红外小目标在 SCTransNet 浅层 Transformer tokenization 中的目标保真下采样 |
| 语义主线 | Keep / Context / Saliency 三源，不增加第四并行分支 |
| 替换范围 | 仅替换 `model.mtc.embeddings_1` 与 `model.mtc.embeddings_2` |
| 主干网络 | SCTransNet encoder、四层 SCTB、Reconstruct、CCA decoder 全部不变 |
| 深监督 | 六输出结构不变 |
| 损失函数 | 原六项 BCE 求和不变 |
| 数据协议 | 530/133 split、归一化、crop、增强、batch、epoch 不变 |
| 优化协议 | Adam、学习率、warmup、cosine、FP32 不变 |
| checkpoint | `best`、`best_miou`、`last` 的选择规则不变 |
| 指标 | 8 邻域连通域、Hungarian 一对一匹配、3-pixel 半径、Fa 定义不变 |
| 正式门槛 | Gate A–E 数值与判断逻辑不变 |
| 后续模块 | 现阶段不接 NER、Survival、Query-only FG |
| V6 资产 | V6 源码、source lock、checkpoint、结果与报告只读保存 |

明确禁止以下“修复”：

- 对预测 mask 做 closing、dilation、component merge 等形态学后处理；
- 把同一 GT 周围多个预测 component 全部算作匹配；
- 增大匹配半径；
- 根据 seed 3407 的结果重新选择固定阈值或 Fa budget；
- 直接增加 Dice、IoU、connectivity、topology loss；
- 同时引入 NER、Survival 或 Query-only FG；
- 在 V7-DCH 正式训练中从 V6 checkpoint warm start。

这些操作要么改变任务协议，要么同时改变多个变量，无法判断 tokenizer 公式是否真正修复了失效。

---

## 3. V6 结果的研究含义

### 3.1 Gate 结果不是“整体无效”，而是“局部有效但不稳定”

| Gate | 结果 | 研究含义 |
|---|---|---|
| A | 失败，3/6 | seed 42 有 Pd 能力，但固定阈值下 mIoU 与部分 Fa 仍未形成完整质量优势 |
| B | 通过 | seed 42 在预注册 Fa budgets 上存在真实 Pareto 改善，说明 KCS 路径并非完全无效 |
| C | 失败 | seed 3407 的固定阈值和预算稳定性崩塌，跨种子鲁棒性是核心问题 |
| D | 失败 | seed 3407 有三个工作点被 Capacity 严格覆盖，同时 Full 在两个 budget 点覆盖 Capacity；结果是混合权衡，Context headroom 没有稳定贡献 |
| E | 通过 | 失败不是训练中断、checkpoint 错误、sweep 不完整或源码漂移造成的 |

最关键的组合证据是：

```text
Gate B 通过
+
Gate C 失败
+
Gate D 失败
```

这更符合“机制具备容量，但优化路径对种子敏感”的模式，而不是“整个 K/C/S 表示完全错误”。因此下一步应优先稳定 Context 与 Saliency 的学习时序，不应马上更换主线或堆叠 NER。

Gate A 的三个未达项必须单独进入 failure atlas，不能被 seed 3407 的严格尾部分析替代：

| checkpoint role | 未达项 | 实际差值 |
|---|---|---:|
| seed 42 Full / Pd-primary | `mIoU >= 0.9336470588` | 实际 `0.922945`，低 `0.010702` |
| seed 42 Full / mIoU-primary | `mIoU >= 0.946542` | 实际 `0.940544`，低 `0.005998` |
| seed 42 Full / mIoU-primary | `Fa <= 1e-6` | 实际 `1.720916e-6`，高 `0.720916e-6` |

因此 DCH 若被正式训练，不仅要检查严格低 Fa 尾部，还必须检查固定阈值下的区域覆盖、pixel precision/recall、mIoU 和未匹配组件构成。

### 3.2 seed 3407：strict-tail 碎裂与 fixed-threshold 混合失败

你提供的 Full/Pd-primary 结果为：

```text
threshold 0.50 : 187/189, Fa=4.84e-5, mIoU=0.86097
threshold 0.58 : 187/189, Fa=4.06e-5
Fa <= 1e-5    : threshold≈0.9999978, Pd=19/189
Fa <= 1e-6    : Pd=5/189
threshold 0.999: pixel precision=1.0,
                 predicted components=263,
                 unmatched components=104
```

当前评估代码有两个不同层面的统计：

1. `pixel precision` 根据预测像素是否落在 GT mask 内计算；
2. 正式 Fa 根据一对一目标匹配后，所有**未匹配预测连通域的像素数**计算。

因此完全可能出现：

```text
所有预测像素都位于 GT 内部
→ pixel precision = 1.0

同一个 GT 内部被切成多个预测连通域
→ Hungarian 只能匹配其中一个
→ 其余连通域全部成为 unmatched components
→ 这些像素继续计入正式 Fa
```

这说明 seed 3407 Full/Pd-primary 的**严格低 Fa 尾部**主要表现为概率场拓扑碎裂：目标内部有多个高置信峰，但峰之间的桥接像素置信度显著较低。提高阈值时，低置信桥先消失，高置信峰继续存活，组件数量不能平滑减少，最终同时造成：

- 目标被拆成多个孤岛；
- unmatched component Fa 居高不下；
- 目标区域不完整，mIoU 下降；
- 阈值继续升高后，只有极少数峰值存活，Pd 突然崩塌。

但这一结论不能外推为全部固定阈值失败原因。在 `threshold=0.5/0.58` 下 pixel precision 尚未达到 1，说明背景/邻域响应也有贡献；Gate A 的 mIoU 缺口还要求检查目标覆盖和边界质量。正式表述固定为：

> strict-tail 由 in-GT 碎裂主导；fixed-threshold 是碎裂、背景/邻域响应与区域质量缺口的混合失败。

---

## 4. V6 失效原因：已确认、强线索与待验证假设

### 4.1 证据分级

| 级别 | 原因 | 当前判断 |
|---|---|---|
| 已确认 | seed 3407 Full/Pd-primary 的严格低 Fa 尾部出现高置信目标内部碎裂 | 由高阈值 component 与 pixel precision 结果支持 |
| 已确认 | 正式 Fa 会把同一 GT 内多余碎片计为 unmatched Fa | 由评估代码的一对一匹配与 Fa 统计直接确定 |
| 已确认 | V6 Full 和 Capacity 在零 scale 输出相同，但 Saliency scale 的一阶梯度不同 | 由 V6 公式直接推导，并完成最小代码复核 |
| P0 假设 | Context headroom 是 seed 敏感性的优先检查变量 | seed 3407 有 3 个点被 Capacity 覆盖，但 Full 也在 2 个 budget 点覆盖 Capacity；现有结果不足以建立因果 |
| 次级线索 | 重复的 `MaxPool-AvgPool` residual 可能促进局部尖峰 | 符合碎裂形态，但 Full/Capacity 共享该路径，不能单独解释 Gate D |
| 待验证 | phase-sum cancellation 造成 Saliency 投影稀疏或符号抵消 | 需要读取每个 block 的 `phase_compress.weight` |
| 待验证 | phase collapse 丢失 2×2 内峰值位置 | 结构上存在，但尚不能证明是 seed 3407 崩塌主因 |
| 待验证 | 后层 block 重复注入 Saliency，抑制目标内部桥接 | 需要逐 block residual 与 target topology 关联 |
| 次级放大因素 | BCE 不约束连通性，六路深监督可能允许多个独立峰 | 基线协议共同存在，现阶段不能作为首个修改变量 |

### 4.2 为什么先处理 Context，而不是直接处理 phase

V6 Full 和 Capacity 共享：

- PixelUnshuffle Keep；
- `AvgPool` Context；
- `MaxPool-AvgPool` Saliency；
- phase-sum tied projection；
- 相同参数量、state key 和初始化；
- 相同训练、数据与评估协议。

二者唯一机制差异是 Context headroom。seed 3407 出现三个 Full 工作点被 Capacity 严格覆盖，因此 Context 调制适合作为第一个冻结诊断变量；但 Full 同时在两个预算点覆盖 Capacity，seed 42 也存在 Full 的预算优势，所以当前证据**不能确认** Context 是失效原因，也不能提前冻结 DCH 公式。

phase collapse 和 phase-sum cancellation 仍可能存在，但它们会同时影响 Full 与 Capacity。若在未完成诊断时直接切换到 phase-resolved V7，就会同时混入：

1. Saliency 表示变化；
2. Saliency 投影变化；
3. 原有 Context 首步梯度不对称继续存在。

即使性能变化，也无法判断究竟修复了什么。

---

## 5. V6 的核心代码问题：零输出锚点不是零优化锚点

### 5.1 V6 当前公式

记：

- 对齐后的 Saliency 为 \(S_a\)；
- `saliency_scale` 参数为 \(s\)；
- \(a=\tanh(s)\)；
- 归一化并中心化后的 Context modulation 为
  \[
  V=0.5\left(Q-\operatorname{mean}_{hw}(Q)\right),
  \qquad \operatorname{mean}_{hw}(V)=0;
  \]
- V6 Full 的 headroom 为
  \[
  H_6=1+0.5(1-|a|)V;
  \]
- residual 为
  \[
  R_6=S_a\,aH_6.
  \]

Capacity 固定：

\[
H_{cap}=1,\qquad R_{cap}=S_a a.
\]

初始化时 \(s=0\)，所以 \(a=0\)，两者都有：

\[
R_6=R_{cap}=0.
\]

因此 Full、Capacity 与 dense SPD 的**前向输出**严格相同。

### 5.2 但第一次反向传播已经不同

在融合 pre-activation 上，对 \(s\) 求导。由于：

\[
\left.\frac{da}{ds}\right|_{s=0}=1,
\]

V6 Full 有：

\[
\left.\frac{\partial R_6}{\partial s}\right|_{s=0}
=S_a\left(1+0.5V\right),
\]

Capacity 有：

\[
\left.\frac{\partial R_{cap}}{\partial s}\right|_{s=0}
=S_a.
\]

因此：

```text
step 0 前向输出完全一致
≠
step 0 反向梯度完全一致
```

Full 的每通道 Saliency scale 从第一次更新起就受到空间 Context 的加权。这个 scale 又是全空间共享的通道标量，所以早期 mini-batch 中目标、背景和 Context 分布的微小差异，会改变：

- scale 的初始符号；
- scale 离开零点的速度；
- 后续 Saliency residual 的全局方向；
- 不同 seed 的优化轨迹。

这种首步梯度差异提供了一条与 seed 敏感性相容的候选机制，但“相容”不等于“导致”。它是否推动 seed 3407 进入高置信碎裂路径，必须由同 checkpoint 的 Context-off、residual-off、phase/block 诊断和后续 fresh paired 训练共同区分。

### 5.3 最小实现复核

基于公开 V6 block 完成的单元级复核得到：

- 零 scale 时 Full/Capacity 输出逐元素相同；
- `phase_compress.weight` 与 bias 的初始梯度相同；
- `saliency_scale` 梯度不同；
- 因此第一次 optimizer step 后两者立即分叉。

这不是 checkpoint 统计推测，而是当前公式的直接性质。

---

## 6. 为什么当前 phase-resolved V7 草案不应原样训练

当前草案把 V6 的：

\[
S_0=\max_p Z_p-C_0
\]

改为：

\[
D_p=\operatorname{ReLU}(Z_p-C_0),
\]

并使用完整 Keep phase weight 投影 \(D_p\)。它有合理动机：保留峰值来自哪个 2×2 phase。

但当前草案存在两个问题。

### 6.1 仍保留 V6 的首步 Context 梯度不对称

草案的 headroom 仍为：

\[
H=1+0.5(1-|a|)V.
\]

所以即使 phase 表示改进，Full 与 Capacity 仍从第一次 scale 更新起分叉，当前最强稳定性问题没有被消除。

### 6.2 改变了 phase 输入的幅度与分配语义

`D_p=ReLU(Z_p-C0)` 只保证：

\[
\max_p D_p=\max_p Z_p-C_0,
\]

但一般不保证：

\[
\sum_pD_p=\max_p Z_p-C_0.
\]

令 \(S_0=\max_pZ_p-C_0\)。V6 的 phase-tied projection 等价于把同一个
\(S_0\) 复制到四个 phase 输入：

\[
[S_0,S_0,S_0,S_0],
\]

其 phase 输入的 L1/L2 量分别为 \(4S_0\) 和 \(2S_0\)。对 raw
\(D_p=[Z_p-C_0]_+\)，则有：

\[
S_0\le \|D\|_1\le3S_0,\qquad
S_0\le \|D\|_2\le\sqrt3S_0.
\]

因此不能声称 raw \(D\) 的 phase 输入“总能量高于 V6”；它在上述 L1/L2
定义下反而不高于 V6。真正的混杂是：它既改变 phase 分配，也改变相对
V6 四相复制语义的幅度。由于 \(W_p\) 带符号，投影后的 aligned output
仍可能变大或变小，不能仅由输入范数判断。这样会同时改变：

- phase 身份；
- residual 总量；
- 权重抵消结构；
- Context 与 Saliency 的交互。

因此当前草案不能作为“只修复 phase collapse”的严格单变量实验；任何
后续 MPR 方案都必须先预注册究竟守恒 L1、L2 还是 aligned-output 参照，
不能笼统写“总能量守恒”。

### 6.3 处理方式

```text
model/tpd_clean_v7.py
status=draft_frozen
formal_training_authorized=false
```

不要删除，也不要覆盖。将它保留为 phase-resolved 研究草案；只有 DCH 结果和 checkpoint 诊断证明 phase 问题仍是主要瓶颈后，再重构为后续 V8。

---

## 7. P0 候选模型：TPD-Clean V7-DCH（诊断 Go 后才冻结）

### 7.1 模型目标

V7-DCH 不改变 K/C/S 表示，候选目标只针对一个问题：

> 若冻结诊断支持 Context headroom 是优先干预变量，则让 Context 在 Saliency residual 已经开始形成后再介入，而不在零 scale 的第一次梯度中决定 Saliency 学习方向。

本节公式是预注册候选，不是已经选定的正式模型。执行第 9 节诊断并通过第
9.6 节 Go/No-Go 前：

```text
v7_dch_hypothesis_priority=P0
v7_dch_formula_frozen=false
v7_dch_implementation_authorized=false
```

DCH 全称：

> **Deferred Context Headroom**  
> 延迟式 Context 增益余量

### 7.2 保持不变的计算

输入 \(X\in\mathbb R^{B\times C\times H\times W}\)：

\[
K=\operatorname{Conv}_{1\times1}
\left(\operatorname{PixelUnshuffle}_2(X);W_k,b_k\right),
\]

\[
C_0=\operatorname{AvgPool}_2(X),
\]

\[
S_0=\operatorname{MaxPool}_2(X)-C_0,
\]

\[
W_t[o,c]=\sum_{p=0}^{3}W_k[o,4c+p],
\]

\[
C_a=\operatorname{Conv}_{1\times1}(C_0;W_t),
\qquad
S_a=\operatorname{Conv}_{1\times1}(S_0;W_t).
\]

Context code 也保持不变：

\[
\widetilde C_a=C_a-\operatorname{mean}_{hw}(C_a),
\]

\[
Q=\tanh\left(
\frac{\widetilde C_a}
{\sqrt{\operatorname{mean}_{hw}(\widetilde C_a^2)+\epsilon}}
\right),
\]

\[
V=0.5\left(Q-\operatorname{mean}_{hw}(Q)\right).
\]

### 7.3 诊断 Go 后允许冻结的唯一公式修改

令：

\[
a=\tanh(s),\qquad t=|a|.
\]

V7-DCH Full：

\[
H_7=1+t(1-t)V,
\]

\[
R_7=S_a\,aH_7,
\]

\[
Y=\operatorname{activation}(K+R_7).
\]

V7-DCH Capacity：

\[
H_{cap}=1,
\qquad
R_{cap}=S_a a.
\]

代码只需将 V6 的：

```python
headroom = 1.0 + 0.5 * (1.0 - scale.abs()) * modulation
```

替换为：

```python
magnitude = scale.abs()
headroom = 1.0 + magnitude * (1.0 - magnitude) * modulation
```

其中 `modulation` 仍为：

```python
0.5 * (context_code - mean_hw(context_code))
```

### 7.4 数学性质

#### 性质 1：零 scale 前向严格等价

当 \(a=0\)：

\[
H_7=1,
\qquad R_7=0.
\]

因此仍与 dense SPD 严格等价。

#### 性质 2：零 scale 一阶优化严格等价

因为 Context 项为：

\[
a|a|(1-|a|)V,
\]

其在零点是一阶消失项，所以：

\[
\left.\frac{\partial R_{full}}{\partial s}\right|_{0}
=
\left.\frac{\partial R_{capacity}}{\partial s}\right|_{0}
=S_a.
\]

因此在相同输入、标签、模型状态和 optimizer 状态下：

- step 0 输出一致；
- step 0 loss 一致；
- 所有参数梯度一致；
- 第一次 Adam 更新后的模型和 optimizer state 一致。

Context 只会在第一次更新后、\(|a|>0\) 时逐渐产生影响。

#### 性质 3：Context 强度自适应延迟与幅度重排

\[
t(1-t)
\]

在：

- \(t=0\) 时为 0：Saliency 尚未学习，不允许 Context 决定方向；
- 中等 \(t\) 时最大：Saliency 已建立，Context 开始重分配；
- \(t=1\) 时回到 0：Saliency 已接近饱和，避免 Context 继续扩大空间不均匀性。

这是一种由模型自身 scale 状态控制的无参数 schedule，不需要修改训练
runner 或 epoch schedule。Context residual 相对 Capacity 的新增项为：

\[
\Delta R_7=S_a\,a\,t(1-t)V,
\]

它在零点附近是 \(O(t^2)\) 量级。因为其中包含绝对值，这只是幅度阶数，
不能据此宣称公式在零点具有二阶可导性。

#### 性质 4：理论 headroom 范围更窄，但并非每个 scale 都更弱

由于：

\[
V\in[-1,1],
\qquad
0\le t(1-t)\le\frac14,
\]

所以：

\[
H_7\in[0.75,1.25].
\]

V6 的理论范围为 `[0.5, 1.5]`，所以 DCH 的全局 headroom 理论范围更窄。
但同一个 \(t\) 下，两版 headroom 的 Context 系数分别是：

\[
c_6(t)=0.5(1-t),\qquad c_7(t)=t(1-t).
\]

因此：

- \(t<0.5\)：DCH headroom 调制弱于 V6；
- \(t=0.5\)：二者相同；
- \(t>0.5\)：DCH headroom 调制强于 V6。

进一步看相对 Capacity 的**有效 Context residual**，V6 系数
\(0.5t(1-t)\) 的最大值是 \(0.125\)，DCH 系数 \(t^2(1-t)\) 的最大值是
\(4/27\approx0.14815\)。所以不能把 DCH 描述成“全程更保守”或“只改
激活时序”；准确表述是：

> DCH 延迟早期 Context，并重新安排不同 saliency scale 区域的 Context 幅度。

当前正式 checkpoint 的 \(|a|\) 最大值低于约 `0.363`，在已观察区域内
DCH 确实比 V6 弱，但这仍是“时序 + 幅度”的联合改变。正式归因必须保留
该边界。

#### 性质 5：有效 Saliency 系数仍有界

\[
|aH_7|
\le t\left(1+t(1-t)\right)
=t+t^2-t^3
\le1.
\]

因此修改不会产生超过 1 的有效 Saliency 系数。

#### 性质 6：Context map 空间均值仍为 1

因为 \(\operatorname{mean}_{hw}(V)=0\)：

\[
\operatorname{mean}_{hw}(H_7)\approx1.
\]

数学上实数运算的均值为 1；浮点实现中应使用 `torch.testing.assert_close`
而不是 `torch.equal`。准确表述仍应是“headroom map 的空间均值数值接近
1”，不能声称 residual 或融合输出均值不变。

### 7.5 为什么该修改针对当前碎裂

V6 Context 在第一次 scale 更新中就对不同空间位置赋予不同权重。若一个真实目标内部存在多个局部峰和较弱桥接区域，早期 Context 相关梯度可能推动某些通道学习“强化峰、抑制桥”，并被后续七个浅层 2× block 与解码器放大。

若诊断 Go，V7-DCH 的候选作用不是强行平滑输出，而是：

1. 先让 K/S 路径学习是否需要 Saliency；
2. 再按新的 scale schedule 允许 Context 重分配；
3. 保持 Capacity 作为相同参数、相同零点优化的严格对照；
4. 避免使用后处理掩盖概率场拓扑问题。

它是否能提高稳定性目前未知，必须先由冻结 counterfactual 决定是否值得
实现，再由两个 seed 的 800-epoch fresh paired 结果验证。不得从公式性质
直接推导 Pd、Fa 或 mIoU 必然改善。

---

## 8. 核心代码修改

本节是 **Go 后实现规范**。第 9.6 节没有输出
`GO_DCH_TRAJECTORY_TEST` 时不得创建正式
DCH 模型、训练入口或启动任务。

### 8.1 新增文件，不覆盖 V6

```text
model/tpd_clean_v7_dch.py
```

变体名称：

```text
tpd_clean_v7_dch_full
tpd_clean_v7_dch_capacity
```

保持以下 state key 结构：

```text
mtc.embeddings_1.blocks.*.phase_compress.weight
mtc.embeddings_1.blocks.*.phase_compress.bias
mtc.embeddings_1.blocks.*.saliency_scale
mtc.embeddings_2.blocks.*.phase_compress.weight
mtc.embeddings_2.blocks.*.phase_compress.bias
mtc.embeddings_2.blocks.*.saliency_scale
```

因此 V6 state dict 可以严格加载到 V7-DCH，用于只读 counterfactual 诊断；正式训练仍从 fresh paired initialization 开始。

### 8.2 核心差异补丁

```diff
-class TPDCleanV6Block(nn.Module):
+class TPDCleanV7DCHBlock(nn.Module):

-    def __init__(..., use_context_headroom: bool, ...):
+    def __init__(..., context_gate: float, ...):
         ...
-        self.use_context_headroom = bool(use_context_headroom)
+        if context_gate not in (0.0, 1.0):
+            raise ValueError(...)
+        self.context_gate = float(context_gate)

     def context_modulation(self, context_aligned):
-        if not self.use_context_headroom:
-            return torch.zeros_like(context_aligned, dtype=torch.float32)
+        if self.context_gate == 0.0:
+            return torch.zeros_like(context_aligned, dtype=torch.float32)
         code = self.context_code(context_aligned)
-        return 0.5 * (code - code.mean(...))
+        centered_code = code - code.mean(dim=(-2, -1), keepdim=True)
+        return 0.5 * centered_code

     def headroom(self, context_aligned):
         modulation = self.context_modulation(context_aligned)
         scale = torch.tanh(self.saliency_scale.float()).view(1, -1, 1, 1)
-        headroom = 1.0 + 0.5 * (1.0 - scale.abs()) * modulation
+        magnitude = scale.abs()
+        headroom = 1.0 + magnitude * (1.0 - magnitude) * modulation
         return scale, modulation, headroom
```

正式 `fusion_terms()` 还必须用 Python 常量分支使 Capacity 跳过
`context_aligned → context_code → headroom` 整条无用路径；Capacity 仍需
计算 `AvgPool`，因为 Saliency 本身定义为 `MaxPool-AvgPool`。该分支不
增加 parameter/state key，也不得改变 Full 的计算。

### 8.3 预留 NER 证据接口，但不接入 NER

在 `TPDCleanV7DCHPatchEmbedding` 中增加只读方法：

```python
def forward_with_evidence(
    self,
    x: torch.Tensor | None,
) -> tuple[torch.Tensor | None, tuple[torch.Tensor, ...]]:
    if x is None:
        return None, ()
    evidence = []
    for block in self.blocks:
        x = block(x)
        evidence.append(x)
    endpoint = x
    return endpoint, tuple(evidence[:-1])
```

普通 `forward()` 保持原执行方式，不调用 NER、不创建 relay、不改变输出：

```python
def forward(self, x):
    if x is None:
        return None
    for block in self.blocks:
        x = block(x)
    return x
```

需要测试：

```text
forward(x) == forward_with_evidence(x)[0]
```

终端 block 输出已经作为 endpoint 进入原网络，未来 NER 只取非终端
states。因此：

```text
embeddings_1: 4 blocks → 3 evidence nodes (h11,h12,h13)
embeddings_2: 3 blocks → 2 evidence nodes (h21,h22)
合计：3+2=5 nodes
```

“五节点”是 NER 的五个跨尺度证据节点，不是五个 TPD 分支；TPD 仍严格
保持 Keep/Context/Saliency 三语义源。这只是为通过 Gate 后重构 NER 的
具体类型依赖做准备，不代表 NER 获得训练授权。

---

## 9. V6 checkpoint 的只读碎裂诊断

正式 V7-DCH 编码前，先对现有 V6 checkpoint 做一次冻结诊断。诊断覆盖
4 个 run 的 `best`、`best_miou`、`last` 共 12 份 checkpoint；正式工作点
复核仍对应 Full/Capacity × 2 seeds × 2 selection roles 的 8 份 sweep。
诊断不改变 V6 结果，也不用于重新挑 checkpoint。

### 9.1 新增脚本

```text
analysis/diagnose_tpd_clean_v6_fragmentation.py
analysis/compare_v6_context_counterfactuals.py
```

输入：

```text
--checkpoint
--variant
--seed
--checkpoint-role {pd_primary,miou_primary,last}
--thresholds 0.5 0.58 0.9 0.99 0.999 0.9999 ...
--include-budget-thresholds
--output-dir
```

### 9.2 必须区分四类 unmatched component

对每个未匹配预测连通域计算与 GT 的 overlap、centroid distance 和邻域关系：

| 类型 | 定义 |
|---|---|
| `in_gt_fragment` | component 与某个 GT mask 有像素重叠，但未被一对一匹配 |
| `near_gt_duplicate` | 不重叠，但质心距某个 GT 小于匹配半径，因一对一约束未匹配 |
| `attached_or_near_gt` | 不满足前两类，但与 3-pixel GT dilation 有重叠 |
| `background_false_object` | 与任何 GT 及其邻域均无关 |

必须输出：

```text
unmatched_pixels_total
unmatched_pixels_in_gt
unmatched_pixels_near_gt
unmatched_pixels_background
fragment_fa_fraction
background_fa_fraction
```

其中：

\[
\text{fragment\_fa\_fraction}
=
\frac{\text{in-GT/near-GT unmatched pixels}}
{\text{all unmatched pixels}}.
\]

若 threshold 0.999 下该比例接近 1，则“碎裂而非背景虚警”的解释得到直接量化支持。

### 9.3 每个 GT 的拓扑指标

对每个 GT component、每个阈值记录：

```text
overlapping_prediction_components
matched_component_area
all_in_gt_prediction_area
largest_fragment_fraction
covered_gt_fraction
fragment_excess=max(0, overlapping_components-1)
```

数据集级汇总：

```text
split_target_count
mean_components_per_detected_gt
p90_components_per_detected_gt
fragment_excess_total
largest_fragment_fraction_mean
largest_fragment_fraction_p10
```

正式定义：

\[
\operatorname{split\_target\_count}(\tau)
=\sum_g\mathbf1[n_g(\tau)\ge2],
\]

即在阈值 \(\tau\) 下被两个及以上预测连通域重叠的 GT 数。另定义：

\[
\operatorname{fragment\_excess\_total}(\tau)
=\sum_g\max(0,n_g(\tau)-1).
\]

两者均为越低越好；`largest_fragment_fraction` 为越高越好。所有分母为零
的情形、未检出 GT 的处理和数据集聚合方式必须在脚本 schema 中固定。

建议定义只用于机制复核、不替代正式指标的离散碎裂面积：

\[
F_{frag}
=
\frac{1}{|T|}
\sum_{\tau\in T}
\sum_g\max(0,n_g(\tau)-1),
\]

其中 `T` 是训练前固定的阈值集合，`n_g(τ)` 是与 GT `g` 重叠的预测组件数。

### 9.4 预实现阶段的三个 V6 零训练 counterfactual

对同一个 V6 Full checkpoint 严格加载并评估：

| 条件 | 实现 | 目的 |
|---|---|---|
| `as_trained_full` | 原 V6 Full | 当前失败参考 |
| `same_weights_context_off` | 将同一 state strict-load 到 V6 Capacity | 隔离当前 Context headroom 的即时作用 |
| `same_weights_residual_off` | 所有 `saliency_scale` 临时置零 | 判断碎裂是否主要来自整个 Saliency residual |

`same_weights_dch` 不得出现在本阶段，因为公式尚未冻结、实现尚未授权。
若第 9.6 节输出 `GO_DCH_TRAJECTORY_TEST`，它才作为实现后的第四项代码
复核加入；该结果
仍不能作为 V7 正式性能结果。

这些 counterfactual 必须同时报告：

- seed 3407 strict-tail：固定阈值与 matched-Pd/matched-Fa 工作点；
- seed 42 Gate A 三个失败项对应的固定阈值 Pd、Fa、mIoU；
- pixel precision/recall、in-GT/near-GT/background unmatched pixels；
- 主指标 `fragment_excess_total`，辅指标
  `split_target_count`、`largest_fragment_fraction`。

解释规则：

| 结果 | 推断 |
|---|---|
| Context-off 在 matched operating point 降低主碎裂指标，并改善至少一个 Gate A 缺口 | 支持把 Context headroom 保留为 P0 干预变量 |
| Residual-off 才显著改善 | 问题更接近 S residual 本身，而不只是 Context |
| 三者都高度碎裂 | 需要检查训练后 Keep、decoder 或深监督概率场；不能把原因限定在 Context |

### 9.5 block 级机制日志

V6 `embeddings_1` 有 4 个 2× block，`embeddings_2` 有 3 个。每个 block 记录：

```text
median/p90/max(abs(tanh(saliency_scale)))
headroom_min/max/std/mean
coefficient_min/max/std
mean(abs(Sa))
mean(abs(residual)) / (mean(abs(Keep)) + eps)
spatial_total_variation(Sa)
spatial_total_variation(residual)
target/background residual ratio
```

phase-sum cancellation 指标建议同时使用 L1 与 L2 版本：

\[
\rho_{L1}
=
\frac{\left\|\sum_pW_p\right\|_1}
{\sum_p\|W_p\|_1+\epsilon},
\]

\[
\rho_{L2}
=
\frac{\left\|\sum_pW_p\right\|_2}
{\sqrt{4}\|W\|_2+\epsilon}.
\]

越接近 0，phase-sum projection 的抵消越强。需要按 block、输出通道和输入通道聚合，不能只给一个全模型均值。

已有全局 checkpoint 统计约为 `0.50–0.58`，因此当前不能以“全局严重
phase cancellation”作为既定原因。新脚本的作用是定位是否存在少数
block/channel 的局部异常，而不是重复证明一个预设结论。

### 9.6 DCH 公式冻结 Go/No-Go

诊断产物必须先写入只读：

```text
analysis/results/tpd_clean_v6_frozen_failure_atlas_v1/V6_FAILURE_ATLAS.json
analysis/results/tpd_clean_v6_frozen_failure_atlas_v1/V6_FAILURE_ATLAS.md
```

然后按预注册规则作一次决策。必须先区分两类问题：

1. `same_weights_context_off` 只能测量**已经训练完成的 checkpoint 中
   Context headroom 的即时前向作用**；
2. DCH 的主要假设是 Context 在零 scale 附近改变了**训练轨迹**。冻结
   checkpoint 的即时复算不能重演早期优化，也不能把“Context-off 变化很小”
   解释为该轨迹假设已经被否证。

因此不能把“Context-off 必须立即补回 Gate A 缺口”设为 DCH 训练的必要
条件。正式决策固定为：

```text
GO_DCH_TRAJECTORY_TEST =
    diagnostics_complete
    && zero_scale_gradient_asymmetry_confirmed
    && saliency_scale_not_saturated
    && !residual_off_only_explains_registered_failure
```

其中：

- `context_off_improves_seed3407_primary_fragmentation`：在 seed 3407
  Full/Pd-primary 的 matched-Pd 工作点上，主指标
  `fragment_excess_total` 下降且 Pd 不降低；若成立，只记为
  `CONTEXT_DIRECT_SUPPORT`，会提高 DCH 优先级，但不是必要条件；
- `context_off_improves_at_least_one_gate_A_deficit`：seed 42 两个正式
  role 的固定阈值复算中，三个 Gate A 未达量至少一个朝门槛方向改善，且
  其余两个不得同时恶化；若成立，同样只作为即时前向支持；
- `saliency_scale_not_saturated`：七个 block 的
  `max(abs(tanh(saliency_scale))) < 0.5`。该阈值在读取本轮完整 atlas
  结果前冻结；它保证当前 DCH 在观测 scale 区间内确实是延迟且减弱
  Context，而不是落入 `t>0.5` 的较强区间；
- `residual_off_only_explains_registered_failure`：在 seed 3407 两个
  checkpoint role 的 matched-Pd 工作点上，residual-off 均以不降低 Pd
  为前提严格改善 `fragment_excess_total`，同时至少改善 Fa 或 mIoU，
  而 Context-off 在两个 role 均不满足这一条件。只有此项成立时才优先
  返回 Saliency/phase 诊断，不冻结 DCH。

结果允许三种、但只有两种实施状态：

```text
CONTEXT_DIRECT_SUPPORT 或 GO_DCH_TRAJECTORY_TEST:
  v7_dch_formula_frozen=true
  v7_dch_implementation_authorized=true
  dch_causal_mechanism_established=false

NO_GO_DCH:
  v7_dch_formula_frozen=false
  v7_dch_implementation_authorized=false
  return_to_KCS_tokenizer_design=true
```

诊断脚本的 `INCONCLUSIVE` 只表示冻结前向复算没有建立 Context 直接作用；
若仍满足 `GO_DCH_TRAJECTORY_TEST`，可以训练严格配对的 DCH 来检验轨迹
假设，但不得把启动训练写成机制已证实。`NO_GO_DCH` 不改变 K/C/S 主线；
它只否决当前 DCH headroom。不得在诊断之后临时搜索常数或改门槛来制造
Go。

本轮实际结果已经冻结为：

```text
decision=GO_DCH_TRAJECTORY_TEST
diagnostics_complete=true
as_trained_formal_consistency=8/8_exact
input_and_state_restoration=8/8_pass
saliency_scale_not_saturated=56/56
max_abs_tanh_saliency_scale=0.3624752164
context_direct_support=false
residual_off_only_explains_registered_failure=false
v7_dch_formula_frozen=true
v7_dch_implementation_authorized=true
dch_causal_mechanism_established=false
ner_stage_authorized=false
paper_core_established=false
stability_claim_supported=false
```

因此后续实现和训练使用本节唯一预注册 DCH 公式；不得再搜索 headroom
常数。冻结诊断本身不替代 Gate A–E。

---

## 10. 工程文件修改清单

### 10.1 模型与入口

| 文件 | 操作 | 内容 |
|---|---|---|
| `model/tpd_clean_v7_dch.py` | 新增 | DCH Full/Capacity、builder、replace 函数、metadata、evidence 接口 |
| `experiments/train_tpd_clean_v7_dch.py` | 新增 | 复用 V6 runner，只替换 model builder 与 variant 校验 |
| `experiments/evaluate_tpd_clean_v7_dch_pd_fa.py` | 新增 | 复用正式 evaluator，不修改 matching 与 metric |
| `experiments/train_tpd_clean_v7_dch_exact.py` | 新增 | 复用 exact-resume 内核 |
| `experiments/TPD_CLEAN_V7_DCH_PROTOCOL.md` | 新增 | 冻结公式、矩阵、Gate 与 source lock |

### 10.2 工程闭环

```text
experiments/smoke_tpd_clean_v7_dch.py
experiments/capture_tpd_clean_v7_dch_smoke_report.py
experiments/verify_tpd_clean_v7_dch_pairing.py
experiments/launch_tpd_clean_v7_dch_formal800.sh
experiments/finalize_tpd_clean_v7_dch.py
experiments/status_tpd_clean_v7_dch.py
```

### 10.3 诊断

```text
analysis/diagnose_tpd_clean_v6_fragmentation.py
analysis/compare_v6_context_counterfactuals.py
analysis/diagnose_tpd_clean_v7_dch_mechanism.py
```

### 10.4 测试

```text
tests/test_tpd_clean_v7_dch.py
tests/test_tpd_clean_v7_dch_gradient_anchor.py
tests/test_train_tpd_clean_v7_dch.py
tests/test_train_tpd_clean_v7_dch_exact.py
tests/test_evaluate_tpd_clean_v7_dch_pd_fa.py
tests/test_tpd_clean_v7_dch_source_lock.py
tests/test_tpd_clean_v7_dch_fragmentation_audit.py
```

### 10.5 明确不修改

```text
model/SCTransNet.py
model/Config.py
experiments/train_tpd_pilot.py
dataset.py
utils.py
warmup_scheduler.py
```

若确需共享 helper，应新增薄适配文件或做不改变行为的纯重构，并用 V6 regression tests 证明旧结果路径逐元素/逐字段不变。

---

## 11. V7-DCH 必须通过的单元与集成测试

### 11.1 结构不变性

1. 只替换 `mtc.embeddings_1/2`；
2. Full 与 Capacity 参数量相同；
3. 与 V6 的 shallow state keys 完全相同；
4. V6 state dict 可 strict-load；
5. full model 参数量保持 `10,843,155`；
6. shallow embedding 参数量保持 `66,176`；
7. 没有第四 branch、额外 parameter 或 persistent buffer；
8. Capacity forward 的 Context alignment/code/headroom 调用计数严格为 0，
   但仍保留形成 Saliency 所需的 `AvgPool`。

### 11.2 数学不变量

1. `mean_hw(modulation)` 在浮点容差内为 0；
2. Full `headroom∈[0.75,1.25]`；
3. `mean_hw(headroom)` 在浮点容差内为 1；
4. `abs(scale*headroom)<=1`；
5. zero saliency 时 residual 严格为零；
6. Capacity 的 headroom 严格为 1；
7. 非零 scale、非恒定 Context 下 Full 与 Capacity 能产生差异。

均值不变量必须用：

```python
torch.testing.assert_close(actual, expected, rtol=..., atol=...)
```

不得对浮点归约结果使用 `torch.equal`。逐元素等价锚点仍可保留 exact
comparison。

### 11.3 最关键的新测试：零点梯度锚定

```python
def test_zero_scale_full_capacity_have_exact_same_gradients():
    full = build_clean_v7_dch_patch_embedding(
        "tpd_clean_v7_dch_full", channels=32, stride=2
    )
    capacity = build_clean_v7_dch_patch_embedding(
        "tpd_clean_v7_dch_capacity", channels=32, stride=2
    )
    capacity.load_state_dict(full.state_dict(), strict=True)

    x_full = torch.randn(2, 32, 16, 16, requires_grad=True)
    x_capacity = x_full.detach().clone().requires_grad_(True)

    y_full = full(x_full)
    y_capacity = capacity(x_capacity)
    assert torch.equal(y_full, y_capacity)

    loss_full = y_full.square().mean() + y_full.mean()
    loss_capacity = y_capacity.square().mean() + y_capacity.mean()
    loss_full.backward()
    loss_capacity.backward()

    assert torch.equal(x_full.grad, x_capacity.grad)
    for (name_f, param_f), (name_c, param_c) in zip(
        full.named_parameters(), capacity.named_parameters()
    ):
        assert name_f == name_c
        assert torch.equal(param_f.grad, param_c.grad)
```

还必须增加：

```text
同 optimizer 初态 + 同 batch + 同 loss
→ 第一次 Adam step 后 model state 逐 tensor 相同
→ optimizer state 逐 tensor 相同
```

这是 V7-DCH 相对 V6 最重要的工程验收项。

### 11.4 整模型测试

1. zero scale 下，与 dense SPD 的六个训练输出逐元素相同；
2. inference 输出逐元素相同；
3. `forward_with_evidence` endpoint 与普通 `forward` 相同；
4. 两个 embedding 分别返回 `states[:-1]`，证据数严格为 `3+2=5`，且五个 node 尺寸满足未来 NER contract；
5. strict save/reload；
6. CPU 两步 forward/backward；
7. RTX 5090 Full/Capacity smoke；
8. exact resume 连续与中断轨迹一致；
9. evaluator wrapper 与核心 evaluator 同字段、同数值；
10. source lock 完整。

---

## 12. 正式训练协议

### 12.1 实验矩阵

只允许物理 GPU 2、3；GPU 0、1 不得被本协议的诊断、smoke、训练、评估或
sweep 使用。每张卡一个串行 lane，交叉映射以避免 variant 与物理卡绑定：

| Lane | 物理 GPU | 顺序 1 | 顺序 2 | Epochs/run |
|---|---:|---|---|---:|
| A | 2 | Full / seed 42 | Capacity / seed 3407 | 800 |
| B | 3 | Capacity / seed 42 | Full / seed 3407 | 800 |

进程内可因 `CUDA_VISIBLE_DEVICES=2` 或 `3` 映射为逻辑 `cuda:0`，但运行
manifest 必须同时保存物理 GPU index 与 UUID。不得把逻辑编号误报为物理
GPU 0。

### 12.2 配对要求

每个 seed 内：

```text
Full 与 Capacity
→ 所有 state tensor 初始化一致
→ optimizer 初态一致
→ 数据 ID 与顺序生成器一致
→ 第一个 batch 一致
→ step 0 输出、loss、梯度一致
→ 第一次 optimizer step 后状态一致
→ 后续仅因 DCH Context 项自然分叉
```

这比 V6 的“只保证零点输出一致”更强。

### 12.3 正式训练必须 fresh start

V6 checkpoint 只用于诊断，V7-DCH 四组正式任务必须 fresh paired initialization。否则无法判断稳定性来自新公式还是 V6 已形成的优化轨迹。

### 12.4 checkpoint 与评估

保持：

```text
best.pth.tar
best_miou.pth.tar
last.pth.tar
threshold=0.5 fixed evaluation
closed-interval Pd-Fa sweep
all preregistered Fa budgets
```

工程数量固定为：

```text
4 formal runs
× 3 checkpoints/run (best, best_miou, last)
= 12 checkpoint artifacts

4 runs
× 2 formal selection roles (pd_primary, miou_primary)
= 8 closed-interval sweeps
```

`last` 只用于完整性和精确续训，不参与候选优劣选择。每份 checkpoint、
run summary 和对应 metrics artifact 必须原生保存同一套 17 个验证字段，
不得依赖后处理兼容补齐：

```text
val_loss, miou, niou,
pixel_precision, pixel_recall, pixel_f1,
pd, tiny_pd, fa, false_objects_per_image,
target_count, matched_target_count,
tiny_target_count, matched_tiny_target_count,
predicted_object_count, unmatched_predicted_object_count,
valid_pixel_count
```

不能根据中间结果改 threshold、checkpoint 角色或 Gate。

---

## 13. 晋级规则

### 13.1 Gate A–E 原样保留

V7-DCH 使用 V6 已锁定的 Gate A–E，不放宽数值，不修改 dominance 定义。

### 13.2 Mechanism Audit M：只约束机制主张，不替代正式 Gate

机制审查必须预注册，但它不新增模型性能 Gate，也不否决已满足 A–E 的
NER 工程授权。它只决定能否写出：

```text
fragmentation_mechanism_claim_supported=true
```

阈值集合必须同时包含冻结固定阈值，以及由参考模型确定后对所有模型共用的
matched-Pd/matched-Fa 工作点；不能只比较各模型自己的 `0.999`，避免把
calibration 差异误写成 topology 差异。主指标固定为
`fragment_excess_total`（越低越好），其他指标只作一致性解释：

```text
Audit M1:
seed 3407 Full 在预注册 fixed 与 matched operating points 的
fragment_excess_total 不高于 V6 Full 对应基线。

Audit M2:
seed 3407 Full 的 in-GT unmatched pixels
在预注册阈值集合上的离散均值不得高于 V6 Full。

Audit M3:
主指标 fragment_excess_total 至少在一个 matched operating point 严格改善；
split_target_count 和 fragment_fa_fraction 越低越好，
largest_fragment_fraction 越高越好，只报告方向一致性。

Audit M4:
“Capacity 全面覆盖 Full”固定为：在所有预注册 operating points 上
Capacity 的 fragment_excess_total 均不高，且至少一点更低。
M4 通过当且仅当上述全面覆盖不成立。
```

Audit M 的单独结论固定为：

```text
mechanism_audit_M_pass=true|false
fragmentation_mechanism_claim_supported=true|false
```

Audit M 不能用来替代 Pd、Fa、mIoU，也不能把“至少若干辅指标之一改善”
作为选择性成功。

### 13.3 NER 授权条件

```text
ner_stage_authorized =
    gate_A_pass
    && gate_B_pass
    && gate_C_pass
    && gate_D_pass
    && gate_E_pass
```

即使获得授权，也只表示可以开始 NER 的工程集成与受控实验，不自动表示
碎裂机制、论文核心或跨数据集稳定性成立。若 A–E 通过但 Audit M 失败，
NER 可进入受控工程实验，但论文不得声称 DCH 已修复目标内部碎裂。

---

## 14. 决策树

### 情形 1：DCH Full 两个 seed 通过 A–E，碎裂明显下降

结论：

- Context 本身可能有价值；
- 结果支持“V6 的一个主要问题是 Context 介入时序和幅度”；
- 冻结 V7-DCH tokenizer；
- 启动 NER 通用 evidence interface 重构；
- 再进行 tokenizer-only 与 tokenizer+NER 的单变量对照。

### 情形 2：DCH Capacity 稳定优于 Full

结论：

- 当前 DCH Full 被否证；
- Capacity 只保留为归因/容量对照，**不得晋升为最终主模型**；
- K/C/S 主线仍冻结，返回 Context 在 K/C/S 内的表示、调制或融合设计；
- `ner_stage_authorized=false`，不得因为 Capacity 指标更好就绕过 Full Gate。

### 情形 3：Context-off counterfactual 明显改善，但正式 DCH 仍失败

结论：

- Context 是问题来源，但 DCH 延迟程度仍不足；
- Capacity/K–S 只能继续作为诊断下界，不能成为主模型；
- 下一候选必须仍在 K/C/S 三源内修订 Context，或回到公式设计门重新预注册；
- 不应同时引入 phase-resolved 和 NER。

### 情形 4：DCH 不能改善，且 phase cancellation 指标很低

结论：

- phase-sum projection 可能压弱或改变 Saliency；
- 启动后续 **V8-MPR（Mass-Preserving Phase-Resolved Saliency）**；
- 仍保持 K/C/S 三源和 DCH headroom。

V8 仍只是未冻结后备。若要保留“把 scalar saliency \(S_0\) 分配给不同
phase”的语义，可采用无 \(\epsilon\) 偏差的显式零分支：

\[
D_p=[Z_p-C_0]_+,
\]

\[
\pi_p=
\begin{cases}
\dfrac{D_p}{\sum_jD_j},&\sum_jD_j>0,\\
0,&\sum_jD_j=0,
\end{cases}
\]

\[
S_0=\max_pZ_p-C_0,
\]

\[
S_p=S_0\pi_p.
\]

从而：

\[
\sum_pS_p=S_0.
\]

这只保证**标量 \(S_0\) 的 L1 分配守恒**，不等于保持 V6 的四相复制
\([S_0,S_0,S_0,S_0]\)：后者的 phase-input L1/L2 分别是 \(4S_0\) 和
\(2S_0\)。如果后续目标是匹配 V6 phase-input L1，则应预注册
\(S_p=4S_0\pi_p\)；如果目标是匹配 L2，则需要另一套归一化。三者不能
混称“保持 V6 总量”，也不能在看到训练结果后选择。只有 checkpoint
诊断支持 phase 问题，并另行冻结守恒范数、零分支和 Gate 后才进入该阶段。

### 情形 5：Context、Saliency residual、phase 指标都不能解释碎裂

结论：

- 需要独立审查 decoder score topology、六路 BCE 深监督和双 identity skip；
- 这将构成新的主线修改，不应伪装成 tokenizer 小修；
- 先形成新的预注册协议，再决定是否改变 loss 或 decoder。

---

## 15. 进入 NER 前的代码准备

当前 NER 实现与 V5 concrete block 类型绑定。V7-DCH 通过后，应先做接口重构，而不是立即训练：

```python
class EvidencePatchEmbeddingProtocol(Protocol):
    blocks: nn.ModuleList

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        ...

    def forward_with_evidence(
        self, x: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, tuple[torch.Tensor, ...]]:
        ...
```

NER 只依赖：

```text
embeddings_1 blocks = 4, evidence = states[:-1] = 3
embeddings_2 blocks = 3, evidence = states[:-1] = 2
selected nodes = h11, h12, h13, h21, h22（合计五节点）
channel/stride contract
```

不要再使用：

```python
isinstance(block, TPDCleanV5Block)
```

应改为能力与形状验证。重构后先验证：

- relay 关闭时整模型输出逐元素等于冻结 V7-DCH；
- state load 行为明确；
- NER 参数与 tokenizer 参数可分组统计；
- NER 正式训练仍需独立协议。

---

## 16. 推荐执行顺序

```text
1. 冻结 V6 正式结果、源码和 checkpoint
2. 对 4 runs × 3 checkpoints 构建 V6 component/fixed-threshold atlas
3. 做 V6 Full / Context-off / residual-off 零训练 counterfactual
4. 完成 phase-sum、saliency_scale 与逐 block 诊断
5. 输出并冻结 V6_FAILURE_ATLAS.json/.md
6. 按第 9.6 节执行一次 Go/No-Go
7. 若 NO_GO_DCH：停止 DCH 实现，返回 K/C/S 内部公式设计门
8. 若 GO_DCH：冻结本节 DCH 公式和 protocol，之后不得调常数
9. 新增 model/tpd_clean_v7_dch.py，补 same_weights_dch 代码复核
10. 完成结构、边界、五节点、梯度锚定与第一 Adam step 测试
11. 完成训练/评估/exact-resume 入口及三类 source locks
12. 只在物理 GPU 2/3 完成 CPU / RTX 5090 smoke
13. 验证 pairing、exact resume、source locks 和 17 字段 schema
14. 按交叉映射运行 Full/Capacity × seed 42/3407 × 800 epochs
15. 保存 4×3 checkpoints；对 4×2 roles 完成 8 份 closed sweep
16. 固定阈值复算 + Gate A–E
17. 独立执行 Mechanism Audit M
18. 仅当 A–E 全通过时授权 NER 接口集成
19. NER 仍按 tokenizer-only / tokenizer+NER 单变量实验执行
```

当前 DCH 公式作为唯一 P0 候选预注册；诊断只决定
`GO_DCH_TRAJECTORY_TEST` 或
`NO_GO_DCH`，不得搜索多个未注册 headroom 常数。只有
`GO_DCH_TRAJECTORY_TEST` 后才把该公式设为 frozen；冻结后任何公式变化
都必须使用新版本号和新协议。

---

## 17. 三类 Source lock 范围

不得用一个包含未使用草案和分析脚本的总锁替代实际执行路径。三个 lock
分别生成、分别验证，并在 comparison manifest 中关联。

### 17.1 Diagnostic lock

只绑定 V6 failure atlas 与 counterfactual 所执行的读取/评估路径：

```text
model/SCTransNet.py
model/Config.py
model/tpd_clean_v6.py
experiments/train_tpd_pilot.py
analysis/diagnose_tpd_clean_v6_fragmentation.py
analysis/compare_v6_context_counterfactuals.py
dataset.py
utils.py
V6 formal comparison/sweep/checkpoint manifest
```

### 17.2 Training lock

只绑定正式训练与精确续训路径：

```text
model/SCTransNet.py
model/Config.py
model/tpd_clean_v7_dch.py
experiments/train_tpd_pilot.py
experiments/train_tpd_clean_v7_dch.py
experiments/train_tpd_clean_v7_dch_exact.py
experiments/tpd_exact_resume.py
dataset.py
utils.py
warmup_scheduler.py
experiments/TPD_CLEAN_V7_DCH_PROTOCOL.md
```

未被正式训练 import 的 `model/tpd_clean_v7.py` 和诊断脚本不得进入 training
lock，以免无关草案变化使正式 checkpoint 锁失效。

### 17.3 Acceptance lock

绑定固定阈值复算、8 份 sweep、Gate A–E、finalize 和机制审查：

```text
model/SCTransNet.py
model/Config.py
model/tpd_clean_v7_dch.py
experiments/train_tpd_pilot.py
experiments/evaluate_tpd_clean_v7_dch_pd_fa.py
experiments/finalize_tpd_clean_v7_dch.py
analysis/diagnose_tpd_clean_v7_dch_mechanism.py
dataset.py
utils.py
experiments/TPD_CLEAN_V7_DCH_PROTOCOL.md
```

三个 manifest 均按各自用途锁定：

- train/validation ID 哈希；
- 图像与 mask 指纹；
- mean/std；
- CLI 和环境变量；
- Python、PyTorch、CUDA、cuDNN、driver；
- GPU UUID；
- deterministic、benchmark、TF32 标志；
- checkpoint 角色与 selection state；
- RNG、DataLoader generator 和 exact-resume state。

Acceptance manifest 还必须记录 4 runs、12 checkpoints、8 sweeps 的路径、
SHA256、selection role，以及 17 字段 schema 完整性。

---

## 18. 建议项目状态

冻结诊断前的历史状态为：

```text
decision=V7_DCH_DIAGNOSTIC_PREPARATION
authoritative_v6_result_accepted=true
v6_immutable=true
mainline_changed=false
current_phase_resolved_v7_status=draft_frozen
v7_dch_hypothesis_priority=P0
v7_dch_formula_frozen=false
v7_dch_implementation_authorized=false
v7_dch_code_candidate_formed=false
v7_dch_formal_training_authorized=false
ner_stage_authorized=false
paper_core_established=false
stability_claim_supported=false
```

本轮诊断输出 `GO_DCH_TRAJECTORY_TEST` 后、正式训练启动前的实施状态为：

```text
decision=V7_DCH_ENGINEERING_PREPARATION
v7_dch_formula_frozen=true
v7_dch_implementation_authorized=true
v7_dch_code_candidate_formed=true
v7_dch_formal_training_authorized=false
ner_stage_authorized=false
mainline_changed=false
paper_core_established=false
stability_claim_supported=false
```

完成代码、定向测试、GPU 2/3 smoke、exact resume 和三类 source locks 后：

```text
decision=V7_DCH_FORMAL_TRAINING_AUTHORIZED
v7_dch_code_candidate_formed=true
v7_dch_engineering_gate_pass=true
v7_dch_formal_training_authorized=true
ner_stage_authorized=false
mainline_changed=false
paper_core_established=false
stability_claim_supported=false
```

本轮当前实际状态为：

```text
decision=V7_DCH_FORMAL_TRAINING_ACTIVE
v7_dch_formula_frozen=true
v7_dch_implementation_authorized=true
v7_dch_code_candidate_formed=true
v7_dch_engineering_gate_pass=true
v7_dch_formal_training_authorized=true
v7_dch_formal_training_started=true
formal_runs_active=2
formal_runs_queued=2
formal_runs_complete=0
ner_stage_authorized=false
mainline_changed=false
dch_causal_mechanism_established=false
paper_core_established=false
stability_claim_supported=false
```

正式结果通过 A–E 后：

```text
decision=V7_DCH_TOKENIZER_GATE_PASS
ner_stage_authorized=true
mainline_changed=false
mechanism_audit_M_pass=true|false
fragmentation_mechanism_claim_supported=true|false
paper_core_established=false
stability_claim_supported=false
```

`paper_core_established` 在本协议所有阶段始终为 `false`；两 seed 的工程
Gate 不能把它改成 `conditional`。`stability_claim_supported` 仍需更多
seeds、数据集和官方测试集证据，不能仅凭两个 seed 自动设为 true。

---

## 19. 最终研究判断

当前可以确认的不是“TPD 主线失败”，也不是“立即需要更复杂的 NER”，
而是：

> V6 的 K/C/S 表示在 seed 42 显示出可用容量；Full Context headroom 在
> 零 Saliency scale 处没有保持优化锚定。seed 3407 同时出现 strict-tail
> 高置信碎裂、固定阈值混合质量缺口和部分 Capacity 覆盖，但现有证据尚未
> 证明这些结果由 Context 首步梯度差异造成。

因此，最小且可归因的下一步首先是：

> **保持 K/C/S 主线不变，先完成 V6 atlas、Context-off、residual-off 和
> phase/block 诊断；若 Go/No-Go 支持 Context 是优先干预变量，再冻结
> DCH，并把“零输出等价”提升为“零输出、零点梯度和第一 optimizer step
> 同时等价”。**

DCH 修复的是一个已确认的公式不对称，但该不对称是否造成指标失败仍是
待验证假设。phase-resolved Saliency 只能作为诊断支持后的条件后备，不能
与 Context schedule 同时修改。

---

## 20. 公开代码依据

- [SCTransNet–TPD 仓库与 V6 正式结果](https://github.com/Arialliy/SCTransNet_main)
- [当前 phase-resolved V7 草案](https://github.com/Arialliy/SCTransNet_main/blob/main/model/tpd_clean_v7.py)
- [SCTransNet 主模型](https://github.com/Arialliy/SCTransNet_main/blob/main/model/SCTransNet.py)
- [训练与连通域评估实现](https://github.com/Arialliy/SCTransNet_main/blob/main/experiments/train_tpd_pilot.py)
- [V6 正式协议](https://github.com/Arialliy/SCTransNet_main/blob/main/experiments/TPD_CLEAN_V6_PROTOCOL.md)

---

# 附录 A：`model/tpd_clean_v7_dch.py` 的 Go 后参考实现

下面的实现保持 V6 参数和 state key 布局，只修改 Context headroom，并提供
未来 NER 使用的只读 evidence 接口。它不能在
`GO_DCH_TRAJECTORY_TEST` 前合入正式运行
路径；Go 后仍须按本报告的测试矩阵复核。

```python
"""TPD-Clean-v7 DCH: deferred Context headroom for stable KCS tokenization.

This candidate preserves the V6 Keep--Context--Saliency sources, phase-tied
projection, parameter/state layout, and dense-SPD zero-scale anchor.  The only
model-formula change is the Context headroom schedule:

    H = 1 + gate * |a| * (1 - |a|) * V

where a=tanh(saliency_scale), V is the centered bounded Context code, and
gate is a Python constant (1 for Full, 0 for Capacity).  Therefore Full and
Capacity have identical output and identical first-order optimization at the
zero-scale anchor, while Context modulation activates only after the Saliency
scale leaves zero.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_CLEAN_V7_DCH_VARIANTS = (
    "tpd_clean_v7_dch_full",
    "tpd_clean_v7_dch_capacity",
)
PRIMARY_CLEAN_V7_DCH_VARIANT = "tpd_clean_v7_dch_full"

CONTEXT_HEADROOM_FLOOR = 0.75
CONTEXT_HEADROOM_CEILING = 1.25
_CONTEXT_MODULATION_SCALE = 0.5

_COMMON_SPEC: Mapping[str, object] = {
    "candidate_family": "spd_anchored_tpd_clean_v7_deferred_context_headroom",
    "mainline_contract": "Keep-Context-Saliency",
    "fourth_parallel_branch_added": False,
    "semantic_sources": ("Keep", "Context", "Saliency"),
    "phase_tied_projection": "sum_keep_weights_over_four_contiguous_phases",
    "saliency_representation": "max_pool_minus_avg_pool_unchanged_from_v6",
    "learned_scales_per_block": 1,
    "scale_parameter": "per_channel_saliency_scale",
    "zero_scale_reference": "dense_spd_exact",
    "zero_scale_first_order_reference": "capacity_exact",
    "state_compatible_with": "tpd_clean_v6",
    "shallow_embedding_parameters": 66_176,
    "full_model_parameters": 10_843_155,
}

_VARIANT_SPECS: Mapping[str, Mapping[str, object]] = {
    "tpd_clean_v7_dch_full": {
        **_COMMON_SPEC,
        "context_gate": 1.0,
        "context_reference": "phase_tied_deferred_zero_mean_gain",
        "context_modulation": "half_centered_context_code",
        "context_headroom": "one_plus_abs_scale_times_one_minus_abs_scale_times_modulation",
        "fusion_formula": (
            "K+Sa*(a*(1+abs(a)*(1-abs(a))*V));"
            "a=tanh(saliency_scale);V=0.5*(Q-mean_hw(Q))"
        ),
        "primary_candidate": True,
    },
    "tpd_clean_v7_dch_capacity": {
        **_COMMON_SPEC,
        "context_gate": 0.0,
        "context_reference": "capacity_control",
        "context_modulation": "not_computed_in_capacity_forward",
        "context_headroom": "neutral_one",
        "fusion_formula": "K+Sa*tanh(saliency_scale)",
        "primary_candidate": False,
    },
}

def _downsample_steps(stride: int) -> int:
    if stride < 2 or stride & (stride - 1):
        raise ValueError(f"stride must be a power of two >= 2, got {stride}")
    return int(math.log2(stride))

def clean_v7_dch_variant_spec(variant: str) -> Dict[str, object]:
    variant = variant.lower()
    if variant not in _VARIANT_SPECS:
        raise ValueError(
            f"Unknown Clean-v7 DCH variant {variant!r}; "
            f"choices={SUPPORTED_CLEAN_V7_DCH_VARIANTS}"
        )
    return dict(_VARIANT_SPECS[variant])

class TPDCleanV7DCHBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        activate: bool,
        *,
        context_gate: float,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be positive, got {channels}")
        if context_gate not in (0.0, 1.0):
            raise ValueError(
                f"context_gate must be exactly 0.0 or 1.0, got {context_gate}"
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.channels = int(channels)
        self.context_gate = float(context_gate)
        self.eps = float(eps)
        self.phase_compress = nn.Conv2d(4 * channels, channels, kernel_size=1)
        self.saliency_scale = nn.Parameter(torch.zeros(channels))
        self.activation = nn.ReLU(inplace=True) if activate else nn.Identity()

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                f"TPDCleanV7DCHBlock requires BxCxHxW input, got {tuple(x.shape)}"
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"TPDCleanV7DCHBlock expected {self.channels} channels, "
                f"got {x.shape[1]}"
            )
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(
                "TPDCleanV7DCHBlock requires even H/W, "
                f"got {tuple(x.shape[-2:])}"
            )

    def branches(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_input(x)
        context = F.avg_pool2d(x, kernel_size=2, stride=2)
        saliency = F.max_pool2d(x, kernel_size=2, stride=2) - context
        keep = self.phase_compress(F.pixel_unshuffle(x, 2))
        return keep, context, saliency

    def phase_tied_weight(self) -> torch.Tensor:
        weight = self.phase_compress.weight.float()
        return weight.reshape(
            self.phase_compress.out_channels,
            self.channels,
            4,
            1,
            1,
        ).sum(dim=2)

    def context_code(self, context_aligned: torch.Tensor) -> torch.Tensor:
        context_fp32 = context_aligned.float()
        centered = context_fp32 - context_fp32.mean(
            dim=(-2, -1), keepdim=True
        )
        inverse_rms = torch.rsqrt(
            centered.square().mean(dim=(-2, -1), keepdim=True) + self.eps
        )
        return torch.tanh(centered * inverse_rms)

    def context_modulation(self, context_aligned: torch.Tensor) -> torch.Tensor:
        if self.context_gate == 0.0:
            return torch.zeros_like(context_aligned, dtype=torch.float32)
        code = self.context_code(context_aligned)
        centered_code = code - code.mean(dim=(-2, -1), keepdim=True)
        return (
            self.context_gate
            * _CONTEXT_MODULATION_SCALE
            * centered_code
        )

    def headroom(
        self, context_aligned: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        modulation = self.context_modulation(context_aligned)
        scale = torch.tanh(self.saliency_scale.float()).view(1, -1, 1, 1)
        magnitude = scale.abs()
        headroom = 1.0 + magnitude * (1.0 - magnitude) * modulation
        return scale, modulation, headroom

    def fusion_terms(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        keep, context, saliency = self.branches(x)
        tied_weight = self.phase_tied_weight()
        saliency_aligned = F.conv2d(saliency.float(), tied_weight, bias=None)
        scale = torch.tanh(self.saliency_scale.float()).view(1, -1, 1, 1)
        if self.context_gate == 0.0:
            # AvgPool above is still required to form Saliency.  Capacity skips
            # only the otherwise unused Context alignment/code/headroom path.
            modulation = torch.zeros_like(saliency_aligned, dtype=torch.float32)
            headroom = torch.ones_like(saliency_aligned, dtype=torch.float32)
        else:
            context_aligned = F.conv2d(context.float(), tied_weight, bias=None)
            _, modulation, headroom = self.headroom(context_aligned)
        residual_fp32 = saliency_aligned * (scale * headroom)
        residual = residual_fp32.to(dtype=keep.dtype)
        return keep, residual, saliency_aligned, modulation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        keep, residual, _, _ = self.fusion_terms(x)
        return self.activation(keep + residual)

class TPDCleanV7DCHPatchEmbedding(nn.Module):
    def __init__(
        self,
        channels: int,
        stride: int,
        *,
        context_gate: float,
    ) -> None:
        super().__init__()
        steps = _downsample_steps(stride)
        self.blocks = nn.ModuleList(
            TPDCleanV7DCHBlock(
                channels,
                activate=index < steps - 1,
                context_gate=context_gate,
            )
            for index in range(steps)
        )

    def forward(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        for block in self.blocks:
            x = block(x)
        return x

    def forward_with_evidence(
        self,
        x: torch.Tensor | None,
    ) -> Tuple[torch.Tensor | None, Tuple[torch.Tensor, ...]]:
        """Return the unchanged endpoint plus intermediate block outputs.

        This is a read-only interface for a later NER stage.  It does not add
        parameters, buffers, relay operations, or alter the ordinary forward
        path used by tokenizer-only formal training.
        """
        if x is None:
            return None, ()
        evidence = []
        for block in self.blocks:
            x = block(x)
            evidence.append(x)
        endpoint = x
        return endpoint, tuple(evidence[:-1])

def build_clean_v7_dch_patch_embedding(
    variant: str,
    channels: int,
    stride: int,
) -> nn.Module:
    spec = clean_v7_dch_variant_spec(variant.lower())
    return TPDCleanV7DCHPatchEmbedding(
        channels,
        stride,
        context_gate=float(spec["context_gate"]),
    )

def replace_shallow_embeddings_clean_v7_dch(
    model: nn.Module,
    variant: str,
) -> Dict[str, nn.Module]:
    variant = variant.lower()
    clean_v7_dch_variant_spec(variant)
    replacements = {
        "embeddings_1": build_clean_v7_dch_patch_embedding(
            variant, channels=32, stride=16
        ),
        "embeddings_2": build_clean_v7_dch_patch_embedding(
            variant, channels=64, stride=8
        ),
    }
    model.mtc.embeddings_1 = replacements["embeddings_1"]
    model.mtc.embeddings_2 = replacements["embeddings_2"]
    return replacements

def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "CONTEXT_HEADROOM_CEILING",
    "CONTEXT_HEADROOM_FLOOR",
    "PRIMARY_CLEAN_V7_DCH_VARIANT",
    "SUPPORTED_CLEAN_V7_DCH_VARIANTS",
    "TPDCleanV7DCHBlock",
    "TPDCleanV7DCHPatchEmbedding",
    "build_clean_v7_dch_patch_embedding",
    "clean_v7_dch_variant_spec",
    "parameter_count",
    "replace_shallow_embeddings_clean_v7_dch",
]
```

> `forward_with_evidence()` 只暴露非终端 `states[:-1]`；两个 embedding
> 分别返回 3 和 2 个节点。普通 `forward()` 保持独立且不接入 relay，以
> 确保 tokenizer-only 正式路径不发生行为变化。
