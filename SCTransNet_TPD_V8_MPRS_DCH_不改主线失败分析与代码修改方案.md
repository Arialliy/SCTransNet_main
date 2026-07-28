# SCTransNet–TPD V7-DCH 失败复盘与 V8-MPRS-DCH 不改变主线修改方案

**研究任务：** 单帧红外小目标检测  
**Baseline：** SCTransNet  
**当前裁决：** `ENGINEERING_GATE_FAIL`  
**下一候选：** `TPD-Clean V8-MPRS-DCH`  
**主线状态：** 不改变  
**NER 状态：** 不授权启动  
**文档日期：** 2026-07-27

---

## 0. 审查边界与结论口径

本文依据四类当前仓库证据形成：

1. V7-DCH 已封存的 comparison JSON/Markdown、8 份 closed sweep 和 completion manifest；
2. V7-DCH 的 4 组训练目录、12 份 checkpoint、800-epoch metrics 和 source locks；
3. 当前模型、训练、评估、协议、对照实现及已生成的 Mechanism Audit M
   checkpoint report；
4. 仓库根目录的 V8 参考原型 `tpd_clean_v8_mprs_dch.py` 与最小测试
   `test_v8_mprs_dch_minimal.py`。

本文对性能裁决采用项目方提供的最新状态：

```text
Gate A = fail
Gate B = fail
Gate C = fail
Gate D = fail
Gate E = pass

decision = ENGINEERING_GATE_FAIL
v7_performance_bundle_verified = true
ner_stage_authorized = false
mainline_changed = false
paper_core_established = false
stability_claim_supported = false

comparison_sha256 =
    c5a1ab25e147e9ba8ebede7da3176f307115dad2a5d52d1b1528184fcdb6dac4
completion_manifest_sha256 =
    29ceed9bf490245e2ddc29b257c38a79f1b641272a2c8b95654641496dfb2290
completion_marker_sha256 =
    3b06d572be63e0223d3741a21ff661254f553c4c322beb8c662b4d934042206c
training_lock_sha256 =
    e67305d53b59336194541e2a9e6bec5bab3682c77232feb8be3e0fe71ea76c95
acceptance_v4_lock_sha256 =
    3b5dfb2e7ede7bf2ead48c65e58e8500bc049f129b27b3a108fef27f3d86f1ce

Mechanism Audit M = 12/12 complete
mechanism_audit_M_pass = false
fragmentation_mechanism_claim_supported = false
mechanism_report_sha256 =
    71c3c2682bcb599d3547a6583cf6c265a4d6be161bd730327b9d8789c36bde41
final_decision_sha256 =
    14626eaab03a486330bca14fa35a41ffb6e10cb4bc9b0edc23a6672267c359e3
control_manifest_sha256 =
    5e809ca8af8c18f6dd6783fa07ec015dfe4cd7e0d90e83b59c338528f324c236
```

本文已直接复核项目本地最终 comparison、completion manifest、12 份 checkpoint 的
`phase_compress`/`saliency_scale` 张量和全部 12/12 份 Mechanism Audit M checkpoint。
关于机制的判断仍严格分为：

- **已由公式和代码确认；**
- **由结果强支持；**
- **仍需 checkpoint 诊断验证。**

任何尚未完成正式训练的新结构都不能预先保证通过 Gate A–E。MPRS 当前已经从根目录
原型实现为完整模型与训练/评估候选，但根因和 fresh-training 性能仍未建立。它必须
先通过加固后的 12-checkpoint counterfactual v2；现有 smoke、exact resume 与计算
开销通过不能替代这一失败门槛。

---

# 1. 结论先行

## 1.1 当前是工程完整、性能与归因门槛未通过

V7-DCH 已完成：

```text
4 formal runs
12 checkpoints
8 closed-interval sweeps
source locks
fixed-threshold recomputation
result verification
```

Gate E 通过，说明训练、续训、checkpoint、sweep 和验收链已经完整。当前没有必要
修改已封存的 runner 或 evaluator；A--D 失败集中在候选性能与 Full/Capacity 归因：

- seed 42 未达到固定阈值质量；
- seed 42 未满足全部注册 Fa budget；
- seed 3407 明显退化；
- Full 没有持续优于同容量 Capacity。

因此不能启动 NER。现有证据证明 V7-DCH 未过门槛，但不能据此宣称某一个具体模型
假设已经被唯一否定。

## 1.2 V7-DCH 修复了“优化起点”；Saliency 表示是候选瓶颈

V7-DCH 的有效贡献是把 Context 影响延迟到二阶量级，使 Full 与 Capacity 在 `saliency_scale=0` 时具有：

- 相同前向；
- 相同输入梯度；
- 相同参数梯度；
- 相同第一次 Adam 更新。

但是它完全保留了 V6 的 Saliency：

```text
S0 = MaxPool2(X) - AvgPool2(X)
Sa = Conv1x1(S0; sum_phase(Wk))
```

这一步在投影前丢失了 2×2 cell 内峰值所属的 TL/TR/BL/BR phase，并可能因
`sum_phase(Wk)` 的方向抵消降低 Saliency 投影。

当前 12 份 checkpoint 的 phase-cancellation ratio 复算结果为：

```text
rho_mean across checkpoints = 0.512659 ... 0.546155
fraction(rho < 0.25) = 0 for every checkpoint
```

因此现有权重不支持“强 phase-sum cancellation 已被证明是主因”。更准确的候选假设是：

> **优化锚点修复成功；phase identity 丢失可能参与碎裂和跨随机轨迹退化，
> 但其贡献需要通过 V8 原型与冻结权重 counterfactual 验证。**

## 1.3 推荐下一版

下一版原型候选为：

> **TPD-Clean V8-MPRS-DCH**  
> **Mass-Preserving Phase-Resolved Saliency with Deferred Context Headroom**  
> **总量保持的相位分辨 Saliency + 延迟式 Context 余量**

只替换 V7-DCH 的 Saliency 表示：

\[
S_p=S_0+\frac{Z_p-C_0}{3}
\]

其中：

\[
Z=\operatorname{PixelUnshuffle}_2(X),\qquad
C_0=\frac14\sum_p Z_p,\qquad
S_0=\max_pZ_p-C_0
\]

其余全部保持：

- Keep 不变；
- Context 不变；
- DCH 不变；
- Full/Capacity 的唯一差异不变；
- 参数量和 state key 不变；
- zero scale 的 dense-SPD anchor 不变；
- 只替换 `mtc.embeddings_1/2`；
- backbone、SCTB、decoder、六路 BCE、数据、优化器、checkpoint、指标和 Gate A–E 均不变。

这是针对候选瓶颈的**单变量结构候选**，不是另起一条模型路线；代码与工程验证已
完成到前置复验阶段，但不自动授权四组 800-epoch 正式训练。

---

# 2. “不改变主线”的严格边界

| 层级 | 冻结内容 |
|---|---|
| Baseline | SCTransNet encoder、SCTB、decoder 和输出接口 |
| 修改位置 | 仅 `mtc.embeddings_1`、`mtc.embeddings_2` |
| 语义主线 | Keep / Context / Saliency 三源 K/C/S |
| 分支数量 | 不增加第四个并行 tokenizer 分支 |
| 参数结构 | `phase_compress` + 每通道 `saliency_scale` |
| Context 角色 | 只调制 Saliency residual |
| 深监督 | 六个输出及其 BCE 求和 |
| 数据划分 | NUDT-SIRST 既有 530/133 内部分割 |
| 训练 | FP32、Adam、warmup、cosine、800 epochs |
| checkpoint | `best`、`best_miou`、`last` |
| 评估 | threshold=0.5、闭区间 sweep、原 component matching |
| 裁决 | Gate A–E 原样继承 |
| 扩展模块 | 本轮不接 NER、Survival、Query-only FG |

V8 唯一改变：

```text
V7-DCH:
scalar Saliency
→ phase-sum projection

V8-MPRS-DCH:
mass-preserving phase-resolved Saliency
→ complete Keep phase projection
```

---

# 3. V7-DCH 结果复盘

## 3.1 固定阈值结果

| Seed | Variant | checkpoint | Pd | Fa ↓ | mIoU ↑ | 判断 |
|---:|---|---|---:|---:|---:|---|
| 42 | Full | Pd-primary | 187/189 | 9.1782e-7 | 0.929930 | Fa 好，但少 1 个目标且 mIoU 不足 |
| 42 | Full | mIoU-primary | 187/189 | 2.6387e-6 | 0.939580 | Pd 可接受，但 Fa、mIoU 均未过线 |
| 42 | Capacity | Pd-primary | 188/189 | 6.8722e-5 | 0.805532 | Pd 高，但区域质量和 Fa 崩坏 |
| 42 | Capacity | mIoU-primary | 186/189 | 5.7364e-7 | 0.939605 | Fa 低，但少目标；mIoU 与 Full 几乎相同 |
| 3407 | Full | Pd-primary | 187/189 | 1.5259e-5 | 0.849535 | 明显质量崩塌 |
| 3407 | Full | mIoU-primary | 183/189 | 3.3271e-6 | 0.923745 | Pd、Fa、mIoU 全部不足 |
| 3407 | Capacity | Pd-primary | 186/189 | 1.0325e-6 | 0.928052 | 较稳定，但 Pd 不足 |
| 3407 | Capacity | mIoU-primary | 184/189 | 6.8837e-7 | 0.929850 | 在该点严格覆盖 Full |

## 3.2 Gate A 的定量缺口

### seed 42，Full / Pd-primary

| 指标 | 实际 | 门槛 | 缺口 |
|---|---:|---:|---:|
| Pd | 187/189 | ≥188/189 | -1 个目标 |
| Fa | 9.1782e-7 | ≤5e-6 | 已满足 |
| mIoU | 0.9299302473050095 | ≥0.9336470588 | -0.0037168114949905 |

这个工作点不是普通的背景虚警问题。Fa 已达到要求，仍同时存在漏检和区域恢复不足。
该 checkpoint 的 180 点闭区间 sweep 中，所有 `threshold<0.5` 的点仍最多只有
187/189，因此不能写成“简单降低阈值即可恢复 Pd”。

### seed 42，Full / mIoU-primary

| 指标 | 实际 | 门槛 | 缺口 |
|---|---:|---:|---:|
| Pd | 187/189 | ≥187/189 | 已满足 |
| Fa | 2.6387e-6 | ≤1e-6 | 为门槛的 2.6387 倍 |
| mIoU | 0.939580 | ≥0.946542 | -0.006962 |

即使使用 mIoU 最佳 checkpoint，区域质量仍未达到 SPD 锚点附近，同时严格 Fa budget 也未通过。

## 3.3 Gate B 的含义

封存 comparison 给出的 seed42 Full/Pd-primary 五个预算点完全相同：

| Fa budget | threshold | Pd | Fa | mIoU | floor | 与 SPD 的关系 |
|---:|---:|---:|---:|---:|---:|---|
| 1e-4 | 0.55 | 187/189 | 8.030941611842105e-7 | 0.9294454155410774 | 188 | floor fail |
| 5e-5 | 0.55 | 187/189 | 8.030941611842105e-7 | 0.9294454155410774 | 188 | floor fail |
| 1e-5 | 0.55 | 187/189 | 8.030941611842105e-7 | 0.9294454155410774 | 188 | floor fail |
| 5e-6 | 0.55 | 187/189 | 8.030941611842105e-7 | 0.9294454155410774 | 188 | floor fail |
| 1e-6 | 0.55 | 187/189 | 8.030941611842105e-7 | 0.9294454155410774 | 187 | floor pass；被 SPD 严格覆盖 |

因此 Gate B 的直接失败原因是四个预算点未达到 188/189；最严格预算虽达到
187/189 floor，但没有建立相对冻结 SPD 的优势。

## 3.4 Gate C 说明不是轻微随机波动

seed 3407 的 Full / Pd-primary 相对 seed 42：

- Pd 同为 187/189；
- mIoU 从 0.929930 降到 0.849535，下降 **0.080395**；
- Fa 从 9.1782e-7 升到 1.5259e-5，约增大 **16.63 倍**。

Full / mIoU-primary 也从 187/189、0.939580 退化到 183/189、0.923745。

这不是“只差一个目标”的边缘失败，而是同一公式在不同随机轨迹下进入了不同的
响应形态。seed 同时控制初始化、数据顺序、augmentation、DataLoader 与 CUDA RNG，
不能只归因于初始化。

## 3.5 Gate D 否定了当前 Context 的稳定因果优势

V7-DCH Full 与 Capacity 的唯一公式差异是：

\[
H_{\mathrm{Full}}=1+|a|(1-|a|)V,\qquad
H_{\mathrm{Capacity}}=1
\]

seed 3407 的 mIoU-primary 中，Capacity 同时具有：

- 更高 Pd：184/189 > 183/189；
- 更低 Fa：6.8837e-7 < 3.3271e-6；
- 更高 mIoU：0.929850 > 0.923745。

因此当前不能声称 DCH Context modulation 已建立稳定收益。

---

# 4. 失败原因：证据分层

## 4.1 已由公式和代码确认

### 原因 1：DCH 只解决零点优化不对称

Full residual：

\[
R_F=S_a a\left[1+|a|(1-|a|)V\right]
\]

Capacity residual：

\[
R_C=S_a a
\]

差异：

\[
\Delta R=S_a\,a|a|(1-|a|)V
\]

在 \(a\to0\) 时，差异为 \(O(|a|^2)\)。这保证了零点公平，但不提高 Saliency 本身的表达能力。

结论：

> **零点优化锚定是正确的工程修复，但不是性能修复。**

### 原因 2：Saliency 在投影前丢失 phase identity

V7-DCH：

```python
context = F.avg_pool2d(x, 2, 2)
saliency = F.max_pool2d(x, 2, 2) - context
Wt = Wk.reshape(O, C, 4, 1, 1).sum(dim=2)
Sa = F.conv2d(saliency, Wt, bias=None)
```

对一个 2×2 cell，`S0` 只保留“峰值比平均值高多少”，不再保留峰值属于：

```text
top-left
top-right
bottom-left
bottom-right
```

中的哪一个 phase。

### 风险 3：phase-sum projection 存在方向抵消通道，但当前未见强相消

V7-DCH 的 Saliency alignment：

\[
S_a^{V7}=S_0\sum_pW_p
\]

若不同 phase 权重方向或符号不同：

\[
\sum_pW_p\approx0
\]

则 Saliency 会被抑制或改变方向。但按本文注册的

\[
\rho_o=
\frac{\left\|\sum_pW_{o,:,p}\right\|_2}
{\sum_p\left\|W_{o,:,p}\right\|_2+\epsilon}
\]

复算 12 份 checkpoint 后，整体均值为 0.512659--0.546155，且所有 checkpoint 的
`\rho<0.25` 通道比例均为 0。因此当前只能确认 phase-sum 具有信息压缩，不能确认
“强权重相消”是 seed 退化主因。

### 原因 4：phase collapse 在浅层路径中重复七次

`embeddings_1` 的 stride=16 由 4 个 2× block 构成；`embeddings_2` 的 stride=8 由 3 个 2× block 构成。两路共 7 次。

连续 phase identity 压缩可能累积为：

- 目标内部响应不连续；
- 高置信目标核心分裂成多个岛；
- 轮廓收缩或膨胀；
- mIoU 下降；
- 同一 GT 周围出现多个 prediction component。

## 4.2 由现有结果部分支持，仍需完整 checkpoint 验证

### 主假设：phase identity 压缩可能参与碎裂和随机轨迹不稳定

当前 evaluator 使用：

```text
prediction > threshold
→ 8 邻域连通域
→ GT / prediction 质心一对一 Hungarian matching
→ 未匹配 prediction component 的像素计入 Fa
```

因此，一个真实目标内部若形成多个高置信孤岛，只能有一个 prediction component 与 GT 匹配，其余孤岛会计入 unmatched objects/pixels。于是可能出现：

```text
pixel precision 很高
但 Fa 仍较高
```

全部 12/12 份 Mechanism Audit M checkpoint 在部分高阈值点观察到 fragment excess，
固定阈值下碎裂也解释了部分而非全部 Fa；这些文件均标记为 `DESCRIPTIVE_ONLY`。
例如 seed3407 Full/Pd-primary 在 threshold=0.5 时 fragment Fa fraction 约为
0.3083，说明碎裂是贡献因素之一，而不是全部来源。

结合上述 \(\rho\) 结果，当前不能写成“已经证明由 phase collapse/cancellation
导致”。V8 必须把这一点作为待否证假设，而不是既定结论。

## 4.3 次要风险：DCH Context 信号可能过弱

由于：

\[
|a|(1-|a|)\le\frac14
\]

且 Context modulation 有界，Full 的 headroom 为：

\[
0.75\le H\le1.25
\]

这保证稳定，但也意味着：

- Full 与 Capacity 在训练早期非常接近；
- Context 的差异信号较弱；
- Context 可能无法稳定形成 Gate D 优势；
- 如果 Saliency 本身已 phase-collapsed，Context 只能调制一个不完整表示。

下一版不同时修改 DCH 强度，因为那会把两个结构变量混在一起。V8 先检验新的
Saliency 表示；若 Full 仍不优于 Capacity，再单独裁决 Context 假设。

还需注意：退化明显的 Full seed3407/Pd-primary 中，
`mean(abs(tanh(saliency_scale)))≈0.0206`，低于 Full seed42/Pd-primary 的
约 0.0543。Saliency 使用不足也是并存候选原因，MPRS 仍会被相同的 scale 衰减。

## 4.4 二级风险：BCE 不直接约束目标连通性

当前训练使用六路像素 BCE 求和。BCE 不直接约束：

- 一个 GT 只形成一个 component；
- 目标内部连通；
- largest-fragment ratio；
- topology。

它可能放大碎裂，但修改 loss 会改变现有主线和历史可比性。本轮不加入 Dice、IoU、connectivity loss 或 morphology。

## 4.5 不应采用的修复

禁止：

- 为 seed 3407 单独调阈值；
- 改固定 threshold 或 Gate；
- 连接、膨胀、闭运算、面积过滤；
- 只报告有利 budget；
- 延长某个 seed 的 epoch；
- 用 NER 覆盖 tokenizer 失败；
- 同时改 loss、decoder、Context 和 Saliency；
- 正式 V8 从 V7 checkpoint warm-start。

---

# 5. V8-MPRS-DCH 模型

## 5.1 Keep 与 Context 不变

PixelUnshuffle 后：

\[
Z\in\mathbb{R}^{B\times C\times4\times H/2\times W/2}
\]

Keep：

\[
K=\operatorname{Conv}_{1\times1}(\operatorname{flatten}(Z);W_k,b_k)
\]

Context：

\[
C_0=\frac14\sum_{p=0}^{3}Z_p
\]

Context alignment：

\[
W_t[o,c]=\sum_{p=0}^{3}W_k[o,c,p]
\]

\[
C_a=\operatorname{Conv}_{1\times1}(C_0;W_t)
\]

## 5.2 新 Saliency

保留 scalar Saliency：

\[
S_0=\max_p Z_p-C_0
\]

构造 phase-resolved Saliency：

\[
\boxed{
S_p=S_0+\frac{Z_p-C_0}{3}
}
\]

使用完整 Keep phase weights：

\[
S_a^{V8}=\sum_{c,p}W_k[o,c,p]S_{c,p}
\]

## 5.3 为什么是 \(1/3\)

设：

\[
M=\max_p Z_p,\qquad m=\min_p Z_p
\]

因为其余三个 phase 均不超过 \(M\)：

\[
4C_0=\sum_pZ_p\le m+3M
\]

所以：

\[
C_0-m\le3(M-C_0)=3S_0
\]

任意 phase 满足：

\[
Z_p-C_0\ge-3S_0
\]

因此：

\[
S_p=S_0+\frac{Z_p-C_0}{3}\ge0
\]

`1/3` 是对四 phase 情况，在任意 phase 组合上仍保证非负的最大线性 contrast 系数，不是验证集调参。

## 5.4 精确不变量

### 投影前、每通道、每 cell 的总量保持

\[
\sum_pS_p
=4S_0+\frac{\sum_pZ_p-4C_0}{3}
=4S_0
\]

V7-DCH 等价于在四个 phase 上复制 \(S_0\)，总量也是 \(4S_0\)。V8 只重分配
投影前的 phase source mass。该结论不适用于带符号权重后的 \(S_a\)、residual 或
最终概率图，不得外推为端到端“信息无损”。

### 非负

\[
S_p\ge0
\]

### 平坦区域为零

若所有 phase 相等：

\[
Z_p=C_0,\quad S_0=0,\quad S_p=0
\]

### source-level 相位置换等变

交换 TL/TR/BL/BR，只会对 \(S_p\) 做同样置换。投影后的输出只有在 phase 权重也做
相同置换时才保持对应关系；固定权重下不能据此声称一像素平移等变。

### phase 权重相同时退化到 V7-DCH

若：

\[
W_p=W
\]

则：

\[
\frac13\sum_pW(Z_p-C_0)=0
\]

因此：

\[
S_a^{V8}=S_a^{V7}
\]

V8 只在 Keep 真正学习到 phase-specific weights 时产生差异。以上等式是实数代数
关系；FP32 实现测试必须使用预注册 `atol/rtol`，不能要求逐位相等。

## 5.5 与 V7-DCH 的解析关系

\[
\begin{aligned}
S_a^{V8}
&=\sum_pW_p\left(S_0+\frac{Z_p-C_0}{3}\right)\\
&=S_0\sum_pW_p+\frac13\sum_pW_p(Z_p-C_0)\\
&=S_a^{V7}+\Delta S_a
\end{aligned}
\]

其中：

\[
\boxed{
\Delta S_a=\frac13\sum_pW_p(Z_p-C_0)
}
\]

`ΔSa` 是零和 phase-contrast correction：

- 不增加第四语义源；
- 不单独形成 residual；
- 不增加参数；
- 只恢复 V7-DCH 丢失的 phase identity。

## 5.6 正式 forward 的代数复用公式

令：

\[
K-b_k=\sum_pW_pZ_p,\qquad
C_a=\sum_pW_pC_0,\qquad
S_a^{V7}=S_0\sum_pW_p
\]

则：

\[
\boxed{
\Delta S_a=\frac{(K-b_k)-C_a}{3}
}
\]

\[
\boxed{
S_a^{V8}=S_a^{V7}+\Delta S_a
}
\]

该公式与显式 \(S_p\) 投影数学等价。正式 forward 固定使用这一形式，以复用 Keep 和
Context alignment，禁止额外构造 5D `phase_saliency` 或执行完整的第二个
`4C→C` Saliency 投影。\(K\) 必须是 activation 前的 raw Keep，\(b_k\) 必须广播
减去且不得 detach。

展开后 correction 含有 Keep 的去偏线性分量，因此必须通过 bias-gradient、
target/hard-negative selectivity 和 shift stress 排除其退化为无选择性的 Keep 增益。

## 5.7 DCH 完全不变

\[
Q=\tanh\left(
\frac{C_a-\operatorname{mean}_{hw}(C_a)}
{\sqrt{\operatorname{mean}_{hw}[(C_a-\operatorname{mean}_{hw}(C_a))^2]+10^{-6}}}
\right)
\]

\[
V=0.5\left(Q-\operatorname{mean}_{hw}(Q)\right)
\]

\[
a=\tanh(\mathrm{saliency\_scale})
\]

Full：

\[
H=1+|a|(1-|a|)V
\]

Capacity：

\[
H=1
\]

融合：

\[
Y=\operatorname{activation}\left(K+S_a^{V8}aH\right)
\]

---

# 6. 对 Gate A–D 的预期作用

这些是预注册假设，不是性能结论。

| Gate | 当前问题 | V8 的针对性 |
|---|---|---|
| A | seed 42 少 1 个目标且 mIoU 低 | phase-aware channel projection 可减少目标核心收缩/断裂 |
| B | 低 Fa budget 下 Pd 不足 | 改善高阈值目标核心连续性，减少额外 unmatched islands |
| C | seed 3407 严重退化 | 检验 phase identity 恢复能否降低跨随机轨迹退化；不预设 cancellation 已成立 |
| D | Full 未持续优于 Capacity | 让 Context 调制结构更完整的 Saliency，重新检验 DCH |
| E | 工程已通过 | 复用闭环，只增加新身份和不变量测试 |

Gate D 仍存在真实风险：V8 改的是 Full/Capacity 共享的 Saliency，两者都可能改善。V8 不能数学保证 Full 必然优于 Capacity。若 Gate D 再次失败，应接受 Context 假设未成立，而不是继续修改报告或门槛。

---

# 7. 代码修改

## 7.1 核心模型文件

新增：

```text
model/tpd_clean_v8_mprs_dch.py
```

仓库根目录已有参考原型：

```text
tpd_clean_v8_mprs_dch.py
test_v8_mprs_dch_minimal.py
```

根目录参考原型最初使用直接 \(4C\) Saliency projection；现已同步改为下述代数
复用路径，并以显式四相位公式作为测试 reference。正式 `model/` 实现必须沿用
这一唯一冻结路径，不能恢复旧版重复投影。

由：

\[
K-b=\sum_pW_pZ_p,\qquad C_a=\sum_pW_pC_0
\]

可得：

\[
\boxed{
S_a^{V8}
=S_a^{V7}+\frac{(K-b)-C_a}{3}
}
\]

因此普通 forward 不构造 `phase_saliency[B,C,4,H/2,W/2]`，也不执行第二个
`4C→C` 投影：

```python
def aligned_mprs_terms(self, x):
    self._validate_input(x)

    rearranged = F.pixel_unshuffle(x, 2)
    context = F.avg_pool2d(x, kernel_size=2, stride=2)
    scalar_saliency = (
        F.max_pool2d(x, kernel_size=2, stride=2) - context
    )

    keep = self.phase_compress(rearranged)
    tied_weight = self.phase_tied_weight()
    scalar_aligned = F.conv2d(
        scalar_saliency.float(),
        tied_weight,
        bias=None,
    )
    context_aligned = F.conv2d(
        context.float(),
        tied_weight,
        bias=None,
    )

    bias = self.phase_compress.bias.float().view(1, -1, 1, 1)
    keep_linear = keep.float() - bias
    saliency_aligned = (
        scalar_aligned
        + (keep_linear - context_aligned) / 3.0
    )
    return keep, context_aligned, scalar_aligned, saliency_aligned
```

```python
def fusion_terms(self, x):
    keep, context_aligned, _, saliency_aligned = (
        self.aligned_mprs_terms(x)
    )

    if self.context_gate == 0.0:
        scale = torch.tanh(
            self.saliency_scale.float()
        ).view(1, -1, 1, 1)
        modulation = torch.zeros_like(saliency_aligned)
        headroom = torch.ones_like(saliency_aligned)
    else:
        scale, modulation, headroom = self.headroom(
            context_aligned
        )

    residual = (saliency_aligned * scale * headroom).to(keep.dtype)
    return keep, residual, saliency_aligned, modulation
```

仅诊断接口可按需构造显式 \(S_p\)，不得由普通 `forward()` 调用：

```python
def diagnostic_phase_sources(self, x):
    rearranged = F.pixel_unshuffle(x, 2)
    b, _, h, w = rearranged.shape
    phases = rearranged.reshape(b, self.channels, 4, h, w)
    context = F.avg_pool2d(x, 2, 2)
    scalar = F.max_pool2d(x, 2, 2) - context
    phase_saliency = (
        scalar.float().unsqueeze(2)
        + (phases.float() - context.float().unsqueeze(2)) / 3.0
    )
    return phases, context, scalar, phase_saliency
```

Full 与 V7 Full 一样复用 \(C_a\) 计算 DCH。Capacity 为计算 MPRS correction 也需要
\(C_a\)，但仍不计算 Context code、modulation 或 headroom。模型参数、buffer 和
state key 不增加。

## 7.2 训练入口

新增：

```text
experiments/train_tpd_clean_v8_mprs_dch.py
```

从 V7-DCH 训练入口复制，仅替换模型身份和元数据：

```python
from model.tpd_clean_v8_mprs_dch import (
    PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
    SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS,
    clean_v8_mprs_dch_variant_spec,
    parameter_count,
    replace_shallow_embeddings_clean_v8_mprs_dch,
)
```

保持断言：

```python
TOTAL_PARAMETERS = 10_843_155
SHALLOW_EMBEDDING_PARAMETERS = 66_176
```

新增 metadata：

```python
metadata.update({
    "saliency_representation": "mass_preserving_phase_resolved",
    "saliency_formula": "S_p=S0+(Z_p-C0)/3",
    "saliency_mass_invariant": "sum_p(S_p)=4*S0",
    "saliency_nonnegative": True,
    "saliency_projection": "complete_keep_weight_phase_projection",
    "saliency_forward_implementation": (
        "algebraic_reuse_scalar_aligned_keep_linear_context_aligned"
    ),
    "phase_contrast_parameters": 0,
    "phase_contrast_buffers": 0,
    "model_state_compatible_with": "tpd_clean_v7_dch",
    "cross_version_exact_resume_supported": False,
})
```

构建顺序不变：

```text
seed_everything(seed)
→ build original SCTransNet
→ initialize shared model
→ replace only embeddings_1/2
→ initialize replacements identically
→ verify parameter counts and checksums
```

## 7.3 精确续训入口

新增：

```text
experiments/train_tpd_clean_v8_mprs_dch_exact.py
```

它应是身份适配层，复用已经验证的 exact-resume 数值内核，不复制一套 RNG 恢复逻辑。
精确续训只指 V8→V8；V7→V8 只能 strict-load model state 做只读诊断，不能恢复
V7 optimizer/journal 并称为 exact resume。

必须覆盖：

- model state；
- optimizer state；
- scaler state；
- Python / NumPy / Torch CPU / CUDA RNG；
- DataLoader generator；
- checkpoint selection state；
- metrics stream；
- 连续训练与 epoch-boundary resume 逐 tensor 一致；
- V8-owned schema、variant、block type、run ID、architecture manifest、
  source-lock key、summary 和 checkpoint adapter 全部重绑。

## 7.4 评估入口

新增：

```text
experiments/evaluate_tpd_clean_v8_mprs_dch_pd_fa.py
```

继续作为闭区间 evaluator 的薄包装：

```python
base.adaptive_thresholds = adaptive_thresholds_closed_interval
base.build_model = build_clean_v8_mprs_dch_model
base.__file__ = __file__
base.main()
```

不得修改：

```text
prediction > threshold
connected-component definition
Hungarian matching
Fa counting
threshold grid
Fa budgets
mIoU/Pd/tiny-Pd definition
```

## 7.5 协议

新增：

```text
experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md
```

冻结：

```text
unique formula = S_p=S0+(Z_p-C0)/3
DCH formula = unchanged
Gate A-E = unchanged
formal matrix = Full/Capacity × seeds 42/3407
epochs = 800
initialization = paired fresh
official test accessed = false
```

## 7.6 Source locks

Training lock 至少绑定：

```text
model/SCTransNet.py
model/Config.py
model/tpd_clean_v8_mprs_dch.py
experiments/train_tpd_clean_v8_mprs_dch.py
experiments/train_tpd_clean_v8_mprs_dch_exact.py
experiments/train_tpd_pilot.py
exact-resume numerical core
dataset.py
utils.py
warmup_scheduler.py
experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md
experiments/TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md
```

Acceptance lock 至少绑定：

```text
experiments/evaluate_tpd_clean_v8_mprs_dch_pd_fa.py
experiments/evaluate_pd_fa_sweep.py
V8 fixed/sweep summarizer
V8 Gate finalizer
V8 Mechanism Audit
V8 protocol
V8 preflight amendment
V8 benchmark and CPU/GPU smoke
formal launcher and two GPU lanes
```

---

# 8. 必须新增的测试

建议新增：

```text
tests/test_tpd_clean_v8_mprs_dch.py
tests/test_train_tpd_clean_v8_mprs_dch.py
tests/test_evaluate_tpd_clean_v8_mprs_dch_pd_fa.py
tests/test_tpd_clean_v8_mprs_dch_exact_resume.py
tests/test_tpd_clean_v8_mprs_dch_source_lock.py
```

必须覆盖：

1. PixelUnshuffle phase 顺序；
2. `sum_phase(S_p) ≈ 4*S0`，冻结 FP32 `atol/rtol`；
3. `S_p >= -atol`；
4. 平坦 cell 的 `S_p ≈ 0`；
5. source-level phase permutation equivariance；投影测试同时置换 phase weights；
6. phase 权重相同时在冻结容差内退化为 V7-DCH；
7. state keys 与 V7-DCH 完全相同；
8. V7-DCH state 可 strict-load；
9. shallow 参数量 66,176；
10. full 参数量 10,843,155；
11. zero scale 时完整模型六输出等于 dense SPD；
12. zero scale 时 Full/Capacity 输出相同；
13. zero scale 时输入梯度、全部参数梯度相同；
14. 首个 Adam step model/optimizer state 相同；
15. nonzero scale 时 Full/Capacity 正常分化；
16. `forward_with_evidence()` 返回 3+2 节点；
17. 直接 \(4C\) 公式与代数复用公式在冻结容差内等价；
18. 普通 forward 不 materialize `phase_saliency`，且每 block 不执行多余
    `4C→C` Saliency conv；
19. V8 相对 V7 的 `(dx,dy)∈{0,1}²` 平移一致性不劣化超过预注册界；
20. correction/Keep correlation、target/hard-negative correction ratio 和
    Saliency scale utilization 可报告；
21. V8→V8 exact resume 逐 tensor 一致；
22. V7→V8 仅 model-state strict-load，拒绝跨版本 optimizer/journal resume；
23. CPU、物理 GPU 2、3 smoke 通过；
24. 峰值显存、吞吐和 forward latency 可复现记录。

仓库根目录已有最小参考测试：

```text
test_v8_mprs_dch_minimal.py
```

它用于验证参考原型，不替代正式 `tests/`、完整 SCTransNet、exact resume、
GPU smoke 和 evaluator 测试。

冻结的数值与计算验收口径：

```text
source mass / nonnegative:
    CPU FP32 atol=2e-6, rtol=0
optimized shortcut vs direct reference:
    CPU FP32 atol=5e-5, rtol=1e-5
    CUDA GPU2/3 atol=1e-4, rtol=1e-5
optimized vs direct x/W gradients:
    CPU atol=2e-5, rtol=1e-4
    CUDA atol=1e-4, rtol=1e-4
standard forward conv2d calls per block:
    Full=3, Capacity=3
phase_tied_weight calls per block:
    Full=1, Capacity=1
Capacity context_code/headroom calls:
    0
```

state keys、zero-scale dense-SPD 输出、zero-scale Full/Capacity 输出、首个 Adam
step 和 V8→V8 exact resume 仍要求逐 tensor 一致。

Keep-reuse 专项测试：

- \(K\) 必须等于现有 `phase_compress(PixelUnshuffle(x))`；
- `keep_linear=K-b` 与显式 no-bias conv 在冻结容差内一致；
- 修改 bias 不应改变 \(S_a^{V8}\) 超过容差；
- `max(abs(grad(mean(Sa8), bias))) <= 1e-6`；
- negative-K 样例确认 correction 使用 activation 前的 raw Keep；
- scale=0 时 correction 不得泄漏到最终 Keep 输出或 Full/Capacity 梯度。

平移/phase 专项测试：

- 四个单像素 impulse 分别落在 TL/TR/BL/BR，\(\Delta S_a\) 命中对应 \(W_p\)；
- 输入平移 2 pixels 后，去边界输出严格对应平移 1 cell；
- 输入平移 1 pixel 不预设不变，作为四 offset stress 报告；
- equal-phase weights 时退化为 V7 tied 行为。

---

# 9. 正式训练前的 checkpoint 诊断

当前状态：

```text
V7 checkpoint tensor inventory = 12/12
V7 rho/scale read-only audit = complete
V7 Mechanism Audit M = 12/12 complete, M=false
V7→V8 transplant counterfactual v1 = 12/12 complete, aggregate=false
root-level V8 reference prototype = present
formal model/experiment integration = implemented
formal800 authorization = false
```

2026-07-27 首轮 counterfactual 的四个 variant/seed 聚合组中仅两组通过：

| Variant | Seed | target lift | fragments V7→V8 | shift ratio | 组结果 |
|---|---:|---:|---:|---:|---|
| Full | 42 | 1.031930 | 303→303 | 0.998675 | pass |
| Full | 3407 | 1.084284 | 216→221 | 1.005024 | **fail** |
| Capacity | 42 | 1.634261 | 274→268 | 1.012328 | pass |
| Capacity | 3407 | 0.531865 | 322→321 | 0.995745 | **fail** |

因此该轮结果不能授权 formal800。它是冻结 V7 权重的前置诊断，不是 V8 fresh
training 的 Pd、Fa、mIoU 结果。独立复核还发现 v1 产物的 source-lock、job identity、
有序 validation IDs、有限性与空样本校验不够完整；v1 数值予以保留，但不能作为
最终授权证据。加固后的重新运行必须写入新目录，不覆盖 v1。

## 9.1 V7 checkpoint transplant counterfactual

V8 与 V7-DCH 参数和 state key 一致，因此可：

```text
V7-DCH checkpoint
→ strict-load 到 V7-DCH
→ strict-load 同一 state 到 V8
→ 同图像、同设置前向
```

覆盖全部冻结 checkpoint，避免只选择 best role：

```text
2 variants × 2 seeds × 3 checkpoint roles = 12 states
```

输出：

- V7/V8 概率图差；
- 7 个 block 的 `Sa_v7`、`Sa_v8`、`ΔSa`；
- component 数量曲线；
- fragment excess；
- largest-fragment fraction；
- in-GT unmatched pixels；
- target-core recall；
- hard-negative correction；
- 只读 Pd/Fa/mIoU counterfactual。

该诊断只回答：

> V7 已训练权重中是否含有被 phase-sum 丢弃、可由完整 phase projection 利用的信息？

它不能替代 fresh V8 training，也不能作为跨版本 exact resume。

## 9.2 phase-sum cancellation

每个 block、每个输出通道：

\[
\rho_o=
\frac{\left\|\sum_pW_{o,:,p}\right\|_2}
{\sum_p\left\|W_{o,:,p}\right\|_2+\epsilon}
\]

解释：

- \(\rho\approx1\)：phase 权重方向一致；
- \(\rho\approx0\)：phase-sum 强抵消；
- 当前 3840 个输出通道中没有 `rho<0.25`，所以不再把强 cancellation
  作为既定前提；
- 后续主要检验 input-conditioned correction 是否在目标区域具有选择性，而不是
  继续用权重 \(\rho\) 单独解释性能。

## 9.3 phase correction utilization

\[
\eta_l=
\frac{E\|S_a^{V8}-S_a^{V7}\|_1}
{E\|S_a^{V7}\|_1+\epsilon}
\]

分别统计：

- GT core；
- GT halo；
- hard-negative component；
- 普通背景。

冻结定义：

\[
\mathrm{target\_correction\_lift}=
\frac{E_{\mathrm{GT\ core}}|\Delta S_a|}
{E_{\mathrm{hard\ negative}}|\Delta S_a|+\epsilon}
\]

四 offset shift consistency 使用去边界概率图的 normalized L1 difference；报告
V8/V7 比值，不把一像素平移误写为严格等变。

预运行选择门槛：

```text
finite_rate = 100%
all 12 states strict-load = true
target_correction_lift > 1.0 in both seeds for each variant aggregate
median largest-fragment fraction does not decrease
aggregate fragment excess does not increase at registered thresholds
V8/V7 output shift-consistency ratio <= 1.10
peak-memory increase <= 10% for the optimized block path
optimized peak memory < direct-reference peak memory
```

上述门槛只决定是否值得投入四组 800-epoch 训练，不替代正式 Gate A--E。任一硬门槛
失败时，V8 保留为诊断原型，不启动 formal800。

---

# 10. 工程与实验顺序

## P0：封存 V7-DCH

```text
comparison/report ordering fixed = complete
comparison + completion hashes = sealed
Mechanism Audit M = 12/12 complete, M=false
fragmentation mechanism claim = false
final decision + control manifest = complete
```

Mechanism checkpoint、总报告、final decision 和 control manifest 已全部完成并
由摘要绑定。不得直接覆盖 `tpd_clean_v7_dch.py`；该阶段称
`sealed_and_digest_bound`，不使用无法由仓库状态证明的 `immutable=true`。

## P1：V8 模型与纯测试

```text
model code
→ syntax
→ invariants
→ state compatibility
→ zero anchor
→ gradients
→ first Adam step
→ evidence interface
```

## P2：只读失败图谱

```text
12 checkpoint counterfactual
→ rho/scale audit
→ block corrections
→ fragmentation curves
→ four-offset shift consistency
→ compute/memory microbenchmark
```

## P3：工程闭环

```text
ordinary train entry
→ exact-resume entry
→ evaluator wrapper
→ source locks
→ finalizer
→ CPU/GPU smoke
```

## P4：正式四组实验

P4 只有在 P1--P3 的所有硬门槛均通过、V8 training/acceptance source locks
冻结且 CPU/GPU2/GPU3 smoke 均通过后才授权。

| Variant | Seed | Init | Epochs |
|---|---:|---|---:|
| `tpd_clean_v8_mprs_dch_full` | 42 | paired fresh | 800 |
| `tpd_clean_v8_mprs_dch_capacity` | 42 | paired fresh | 800 |
| `tpd_clean_v8_mprs_dch_full` | 3407 | paired fresh | 800 |
| `tpd_clean_v8_mprs_dch_capacity` | 3407 | paired fresh | 800 |

固定产物：

```text
4 runs
12 checkpoints
8 closed-interval sweeps
```

中间结果不允许改变公式、epoch、checkpoint role、threshold 或 Gate。

## P5：原 Gate A–E 裁决

```text
ner_stage_authorized =
    gate_A_pass
    && gate_B_pass
    && gate_C_pass
    && gate_D_pass
    && gate_E_pass
```

只有全部通过，才进入 NER。

---

# 11. Mechanism Audit M-V8

| 指标 | 作用 |
|---|---|
| phase cancellation ratio | 验证 phase-sum 权重抵消 |
| phase correction ratio | 判断 V8 是否实际使用 phase 信息 |
| target correction lift | 判断 correction 是否偏向目标而非 hard negative |
| fragment excess | 直接量化一个 GT 周围的额外 component |
| largest-fragment fraction | 量化目标内部连贯性 |
| in-GT unmatched pixels | 区分碎裂与普通背景虚警 |
| connected-core retention | 检查阈值上升时目标核心是否连续 |
| scale utilization | 判断 Saliency residual 是否被采用 |
| Full-Capacity context delta | 判断 DCH 是否真正形成因果差异 |
| correction–Keep correlation | 排查 correction 退化为无选择性 Keep 增益 |
| four-offset shift consistency | 检查 phase-specific correction 是否放大网格偏置 |
| latency / peak memory | 验证代数复用确实消除直接 4C 路径开销 |

必须逐 block 报告 7 个下采样 block，不能只看最终 endpoint。

---

# 12. 结果决策树

## A. V8 Full 通过 A–E

```text
freeze V8 tokenizer
→ paper_core_candidate=true
→ 接入五节点 NER
```

NER 仍保持单变量：

```text
V8 tokenizer only
vs
V8 tokenizer + NER
```

## B. Full 与 Capacity 都改善，但 Gate D 仍失败

解释：

```text
phase-resolved Saliency 有效
DCH Context 未建立稳定收益
```

下一步应冻结 Saliency 结论，单独预注册 Context schedule；不能直接进入 NER，也不能临时修改 Gate D。

## C. Capacity 严格优于 Full

说明 DCH Context 有害或无益。停止继续复杂化 Context，不通过增加 Context 分支强行保留创新点。

## D. 碎裂减少、mIoU 上升，但 Pd 仍为 187/189

说明 phase 表示改善区域恢复，但漏检目标可能更早在 encoder/backbone 中消失。此时才评估 target-survival supervision，仍需独立协议。

## E. V8 与 V7-DCH 几乎无差别

检查：

- phase weights 是否近似相同；
- `ΔSa` 是否接近零；
- `saliency_scale` 是否接近零；
- shallow token path 是否被双 identity skip 稀释；
- BCE/topology mismatch 是否成为主因。

若 block correction 明显存在但最终输出无变化，继续设计 tokenizer 公式的信息价值会明显下降。

---

# 13. 风险与真实性边界

## 风险 1：V8 可能改善 Full 和 Capacity，但仍无法通过 Gate D

这是最现实的风险，因为两者共享新 Saliency。Gate D 仍取决于 DCH Context 是否在更合理的 Saliency 上产生稳定收益。

## 风险 2：phase-aware channel projection 不等于恢复空间分辨率

V8 仍把每个 block 输出压缩为：

\[
C\times H/2\times W/2
\]

它恢复的是 channel projection 中的 phase identity，不应宣传为“无损下采样”。

## 风险 3：碎裂可能主要来自 BCE/decoder

若 V8 的 block-level mechanism 改善，而最终 component 不改善，则根因可能在 loss 或 decoder。那属于下一条受控实验，不能混入 V8。

## 风险 4：两个 seed 仍不足以支持论文级稳定性

即使 Gate 通过，也只获得项目内 NER 授权。论文还需要：

- 更多 seeds；
- 多数据集；
- 官方测试集封闭评估；
- FLOPs、延迟、显存；
- 机制复现。

## 风险 5：MPRS correction 可能退化为 Keep 增益

代数复用揭示 \(\Delta S_a=((K-b)-C_a)/3\)。它是 phase-contrast 的等价表达，
但优化器可能主要利用其中与 Keep 相关的方向。必须报告 correction--Keep correlation、
bias gradient 和 target/hard-negative selectivity，不能只凭公式名称声称目标保真。

## 风险 6：phase-specific correction 可能放大网格偏置

stride-2 tokenizer 本身不具有一像素平移不变性。V8 进一步使用 phase-specific
权重，可能放大目标落在 TL/TR/BL/BR 时的差异。四 offset stress 是正式训练前硬门槛，
不是可选机制图。

## 风险 7：参数量不变不代表计算开销不变

朴素的直接 \(4C\) Saliency 投影以及旧版诊断 `old_aligned` 会增加计算与中间
张量。当前根目录原型已改为代数复用；正式实现必须保持该路径，并报告相对 V7 和
显式 direct-reference 的 latency、吞吐与峰值显存。

---

# 14. 当前实现与裁决状态

```text
decision=V8_MPRS_DCH_IMPLEMENTED_PREFLIGHT_NOT_AUTHORIZED

v7_performance_bundle_verified=true
v7_performance_bundle_sealed=true
v7_mechanism_audit_complete=true
v7_mechanism_reports_completed=12
v7_mechanism_reports_expected=12
v7_mechanism_audit_M_pass=false
fragmentation_mechanism_claim_supported=false
mainline_changed=false

v8_formula_documented=true
v8_formula_status=implemented_candidate
v8_root_reference_prototype_present=true
v8_root_minimal_test_present=true
v8_root_minimal_test_pass=true
v8_optimized_model_integrated=true
v8_formal_model_candidate_formed=true
v8_full_model_parameter_count=10843155
v8_shallow_embedding_parameter_count=66176
v8_ordinary_training_entry_implemented=true
v8_exact_training_entry_implemented=true
v8_evaluator_entry_implemented=true
v8_v8_exact_resume_test_pass=true
v8_cpu_smoke_pass=true
v8_gpu2_smoke_pass=true
v8_gpu3_smoke_pass=true
v8_compute_memory_gate_pass=true
v8_counterfactual_v1_jobs_completed=12
v8_counterfactual_v1_group_passed=2/4
v8_counterfactual_v1_gate_pass=false
v8_counterfactual_v1_evidence_audit_pass=false
v8_training_source_lock_present=false
v8_acceptance_source_lock_present=false
v8_failure_cause_established=false
v8_formal_training_authorized=false
v8_fresh_training_performed=false
v8_pd_fa_miou_available=false

ner_stage_authorized=false
paper_core_established=false
stability_claim_supported=false
```

这里的 `counterfactual_v1_evidence_audit_pass=false` 不表示 12 组数值被删除；它表示
首轮聚合没有完整绑定当前 V8 源码、ordered IDs 与所有中间身份，必须保留原产物并
以加固版本写入新目录重新运行。即使忽略该审计缺口，首轮数值本身也有两个组未过
预注册门槛，所以仍不能启动 formal800。

正式训练授权条件：

```text
model_tests_pass
&& full_model_spd_identity_pass
&& paired_initialization_pass
&& zero_anchor_gradient_pass
&& first_adam_step_pass
&& exact_resume_pass
&& source_lock_pass
&& twelve_checkpoint_counterfactual_pass
&& shift_consistency_pass
&& no_redundant_projection_pass
&& compute_memory_gate_pass
&& cpu_smoke_pass
&& gpu2_smoke_pass
&& gpu3_smoke_pass
```

---

# 15. 根目录参考原型与复验边界

仓库根目录当前存在：

```text
tpd_clean_v8_mprs_dch.py
test_v8_mprs_dch_minimal.py
```

参考原型已按本方案改成代数复用生产路径，并保留显式 \(4C\) 公式只作 reference。
当前文件哈希：

```text
tpd_clean_v8_mprs_dch.py =
    1b23a07ef30591fe918fe9ef6a7055b14ac5057c06638509f53a6a66f8bb1e28
test_v8_mprs_dch_minimal.py =
    f097323ed39f53e69f2fbd954e5b08ab133f4c98f750dde2d503a3b8b48c251d
```

使用项目 Python 的实际执行结果（均限定为 root isolated-block/minimal CPU
范围）：

```text
PASS: state compatibility
PASS: saliency mass/non-negativity/flat invariants
PASS: phase-permutation equivariance
PASS: optimized/direct forward and gradient equivalence
PASS: bias cancellation and three-convolution production path
PASS: equal-weight reduction to V7-DCH
PASS: zero-scale dense-SPD identity
PASS: zero-scale Full/Capacity gradients
PASS: first Adam-step equality
PASS: nonzero-scale Full/Capacity divergence
PASS: single-pass MPRS diagnostics
PASS: embedding shapes/evidence/parameter count
```

当前边界：

- 根目录优化原型和 isolated-block/minimal CPU 测试已通过；
- `test_v8_mprs_dch_minimal.py` 当前是直接执行脚本而非正式 pytest test item，
  且使用裸 `assert`；它只作为原型证据，不能计入正式工程 Gate；
- 正式 pytest 已使用实际 V7 类验证 state compatibility，并完成 12/12 V7
  checkpoint strict-load；
- 正式 builder 已验证 shallow=66,176、完整模型=10,843,155；
- 优化后的正式 `model/`、普通训练入口、exact 入口和 evaluator 已完成；
- 已在 NUDT-SIRST 既有 133 张内部验证图像完成首轮 12-checkpoint
  counterfactual；该轮只有 2/4 variant/seed 组通过；
- CPU、物理 GPU2、物理 GPU3 的完整六输出 SCTransNet smoke 已通过；
- V8→V8 epoch-boundary exact resume 测试已通过；V7→V8 exact resume 被拒绝；
- Full 与 Capacity 的逐字节一致首个 Adam step 已在 GPU2/3 复验通过；
- 优化路径显存 v2 同时覆盖 Full/Capacity，relative-to-V7 gate 通过；
- 加固后的 counterfactual 聚合、training lock 与 acceptance lock 尚未最终冻结；
- 未启动 V8 fresh formal800；
- 未产生新的 Pd、Fa、mIoU；
- 不能据此声称 V8 已通过性能门槛。

---

# 16. 文件清单

## 已存在的根目录参考文件

```text
tpd_clean_v8_mprs_dch.py
test_v8_mprs_dch_minimal.py
```

## 已新增并通过基础工程测试

```text
model/tpd_clean_v8_mprs_dch.py
experiments/train_tpd_clean_v8_mprs_dch.py
experiments/train_tpd_clean_v8_mprs_dch_exact.py
experiments/evaluate_tpd_clean_v8_mprs_dch_pd_fa.py
experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md
experiments/TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md
analysis/analyze_tpd_clean_v8_mprs_mechanism.py
tests/test_tpd_clean_v8_mprs_dch.py
tests/test_train_tpd_clean_v8_mprs_dch.py
tests/test_evaluate_tpd_clean_v8_mprs_dch_pd_fa.py
tests/test_tpd_clean_v8_mprs_dch_exact_resume.py
experiments/smoke_tpd_clean_v8_mprs_dch.py
analysis/benchmark_tpd_clean_v8_mprs_dch.py
tests/test_smoke_tpd_clean_v8_mprs_dch.py
tests/test_benchmark_tpd_clean_v8_mprs_dch.py
experiments/run_tpd_clean_v8_mprs_dch_formal800_2x5090_lane.sh
experiments/launch_tpd_clean_v8_mprs_dch_formal800_2x5090.sh
tests/test_tpd_clean_v8_mprs_dch_2x_runtime.py
experiments/freeze_tpd_clean_v8_mprs_dch_source_locks.py
tests/test_tpd_clean_v8_mprs_dch_source_locks.py
```

## 尚待最终冻结

```text
experiments/tpd_clean_v8_mprs_dch_exact_source_lock.json
experiments/tpd_clean_v8_mprs_dch_acceptance_source_lock.json
加固后的 counterfactual v2 12-job + aggregate report
formal800 authorization（只有全部前门槛为 true 才能生成）
```

## 不修改

```text
model/SCTransNet.py
encoder
SCTB
decoder
loss
dataset split
augmentation
optimizer
checkpoint selection
metric definitions
Gate A-E
NER model
```

---

# 17. 推荐提交顺序

```text
step 1: seal V7-DCH final report and hashes = complete
step 2: add V8 model + mathematical unit tests = complete
step 3: add frozen-checkpoint mechanism diagnostics = implemented;
        evidence hardening in progress
step 4: add ordinary/exact training entries = complete
step 5: add evaluator + protocol = complete
step 6: add CPU/GPU smoke artifacts = complete
step 7: freeze training/acceptance source locks = pending
step 8: rerun hardened 12-checkpoint preflight in a new directory = pending
step 9: issue formal authorization only if every preflight gate passes = blocked
step 10: run Full/Capacity × seeds 42/3407 × 800 = not authorized
step 11: produce 12 checkpoint / 8 sweep acceptance bundle = not started
step 12: execute unchanged Gate A-E decision = not started
```

最终顺序：

```text
V7-DCH sealed_and_digest_bound archive
→ V8-MPRS-DCH engineering closure
→ V8 four formal runs
→ unchanged Gate A-E
→ only if all pass: NER
```

---

# 18. 审查的主要仓库文件

```text
README.md
model/SCTransNet.py
model/tpd_clean_v6.py
model/tpd_clean_v7.py
model/tpd_clean_v7_dch.py
tpd_clean_v8_mprs_dch.py
test_v8_mprs_dch_minimal.py
experiments/train_tpd_pilot.py
experiments/train_tpd_clean_v7_dch.py
experiments/train_tpd_clean_v7_dch_exact.py
experiments/evaluate_pd_fa_sweep.py
experiments/evaluate_tpd_clean_v7_dch_pd_fa.py
experiments/TPD_CLEAN_V7_DCH_PROTOCOL.md
```

---

# 19. 2026-07-27 实际运行证据

## 19.1 正式代码测试

当前已通过的联合测试覆盖正式模型、完整 SCTransNet、普通 builder、评估包装、
V8→V8 exact resume、双 GPU lane 的拒绝路径以及 Full/Capacity 显存基准。
根目录最小脚本的 12 项直接执行检查也全部通过。正式测试使用项目 Python：

```text
/home/ly/BasicIRSTD/infrarenet/bin/python
```

正式 model 文件当前基线 SHA：

```text
model/tpd_clean_v8_mprs_dch.py =
    39e7b1618ea9f594f4abecc284afd0d845132218e2fab3657dbb957c78703c19
experiments/train_tpd_clean_v8_mprs_dch.py =
    f852b9fb8f6725c63d0907736418a799f979ad5ee44d5ad56cbf673cb676bd4c
experiments/evaluate_tpd_clean_v8_mprs_dch_pd_fa.py =
    3eb1cb90134183b824876efc282147470df3401a8b3e71edf73952e7c6987433
```

这些 SHA 在 source lock 冻结前仍可能因只修工程验真而更新；任何更新都必须重跑
相应测试和前置诊断，不能继续引用旧 SHA。

## 19.2 CPU/GPU smoke

三份纯 JSON 报告：

```text
analysis/results/tpd_clean_v8_mprs_smoke_v2/cpu.json
    sha256=9358b4f86495e9813e212da7e1276764c46529b24f103b9bfa41567c8ed63187
analysis/results/tpd_clean_v8_mprs_smoke_v2/gpu2.json
    sha256=716aeaf101382efbb922b6d3a30383d401a540f945add055c10a0860537615c3
analysis/results/tpd_clean_v8_mprs_smoke_v2/gpu3.json
    sha256=84dffeede42b78648116a1a38049ec4dd9e0195f673e4414e18661026f69c66c
```

三者均满足：

```text
status=complete
paired_initialization=true
paired_first_adam_step_exact=true
six_output_BCE=true
zero_scale_dense_SPD_exact=true
strict_reload=true
all_gradients_finite=true
all_updated_parameters_finite=true
MPRS blocks=7
ordinary forward conv2d per block=3
```

GPU2/3 均只暴露各自物理设备，并核验 RTX 5090 名称与 UUID。CUDA smoke 在局部
作用域内固定 deterministic algorithms、cuDNN、TF32 与
`CUBLAS_WORKSPACE_CONFIG=:4096:8`；没有改变模型公式或放宽首步逐字节一致标准。

## 19.3 显存基准 v2

报告：

```text
analysis/results/tpd_clean_v8_mprs_benchmark_v2/gpu2.json
sha256=c9d14249890786b2a958a77273b8d51e99422eaea3f897760965ac2a9a156521
compute_memory_gate_pass=true
```

它同时覆盖 Full 与 Capacity，并对三条路径使用同样预热和三轮交叉测量顺序。
以三轮 peak 的中位数判断：

| Variant | Shape | V8 optimized / V7 | V8 optimized / direct |
|---|---|---:|---:|
| Full | B16,C32,256² | 0.888887 | 0.571437 |
| Full | B16,C64,128² | 0.888895 | 0.571498 |
| Capacity | B16,C32,256² | 0.999998 | 0.571437 |
| Capacity | B16,C64,128² | 0.999996 | 0.571498 |

这证明优化实现没有重复执行显式 \(4C\) Saliency 投影；它不证明检测性能。

## 19.4 当前不得启动 formal800 的直接原因

即使正式模型、exact resume、CPU/GPU smoke 和显存门槛均已通过，首轮
counterfactual 仍有：

```text
Full/seed3407: fragment excess 216→221，失败
Capacity/seed3407: target correction lift 0.531865，失败
```

同时首轮聚合的证据身份还需加固。因此：

```text
v8_formal_training_authorized=false
formal800_started=false
new_v8_pd_fa_miou_result=false
mainline_changed=false
innovation_changed=false
```

原 15 份 v1 文件已由只增不改的归档清单绑定：

```text
analysis/results/tpd_clean_v8_mprs_counterfactual_v1/V1_ARCHIVE_EVIDENCE.json
sha256=fbc0c5be70a2548cf5f30874eed91b66ae1ae504b462f2261368c7f5b37865ec
```

下一次允许执行的运行仅是加固后的 12-checkpoint 前置复验。只有新的 aggregate
报告本身为 true、source locks 与三份 smoke/显存报告全部摘要绑定后，双 GPU
launcher 才能接受正式授权。
