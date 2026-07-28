# SCTransNet–TPD NER V3 失败复盘与 V4 Tail-Aware DC-Offset 修改方案

> 任务：在不改变 TPD–SCTransNet 主线的前提下，根据 baseline / V1 / V2 / V3 正式比较和 8 行 NER stage-wise DC-offset 消融，完成失败复盘、V4 候选公式选择、代码修改、工程验收和正式训练计划。  
> 当前裁决：`V4_COMPLEMENT_TAIL_SELECTED_FORMAL800_RUNNING`  
> V4 公式裁决：`COMPLEMENT_TAIL_SELECTED`，即 stage3/2 冻结为 `d_s(1-P_s)`；该裁决只授权实现正式 V4 训练闭环，不等于正式性能通过。  
> 文档日期：2026-07-28  
> 权威范围：正式数值以本地已封存的 V3 正式修复报告和 DC-offset knockout v2 聚合报告为准；当前工作树源码用于核对实现，不以公开分支替代封存源码和哈希。

---

## 0. 当前建议与术语边界

零训练 counterfactual 已从候选族中选定正式 V4 公式：

> **TPD-NER V4 Tail-Aware DC-Offset Calibration（V4-TA-DC）**  
> stage4 保持 V3 global，stage3/2 使用 target-protective complement
> `d_s(1-P_s)`。

本文严格区分两个概念：

```text
MPRS-DCH：
    tokenizer 内已经冻结的 Mass-Preserving Phase-Resolved Saliency
    + Deferred Context Headroom；V4 不修改它。

NER DC offset：
    tpd_ner.dc_offsets.{2,3,4}，位于 centered gate logits 之后；
    8 行 knockout 只消融这三个标量，不是消融 tokenizer DCH。
```

V4 保留 V3 的全部主结构：

```text
V8-MPRS-DCH tokenizer
→ 五个 evidence nodes：h11 / h12 / h13 / h21 / h22
→ q4 → q3 → q2 窄中继
→ CCA 后的 skip multiplicative modulation
→ 原 SCTransNet decoder
```

修改只位于 NER 的 `dc_offsets` 空间作用域：

```text
stage4：始终保留 V3 legacy global DC，作为已证据支持最强的固定项
stage3/2：已比较三种同 state、同参数的作用域
  A. legacy global：d_s
  B. direct-tail：d_s * P_s
  C. target-protective complement：d_s * (1-P_s)
```

先定义待审计的父子 relay 持续上尾响应：

\[
P_3=\sqrt{T_3(q_3)\,\uparrow T_4(q_4)},
\]

\[
P_2=\sqrt{T_2(q_2)\,\uparrow T_3(q_3)}.
\]

对 \(s\in\{3,2\}\)，三种候选作用域为：

\[
A_s^{legacy}=1,\qquad
A_s^{direct}=\operatorname{sg}(P_s),\qquad
A_s^{protect}=\operatorname{sg}(1-P_s),
\]

\[
M_s^{r}=
\frac{1}{\pi}\arctan\left[\pi\left(Z_s+d_sA_s^{r}\right)\right],
\quad r\in\{legacy,direct,protect\}.
\]

stage4 固定 \(A_4=1\)。三个候选均不增加参数、不增加 persistent
buffer、不改变 state key，并保持 V3 checkpoint 严格加载兼容。`direct-tail`
不是预先认定的最终公式：V3 两个正式 checkpoint 的 `d2/d3/d4` 全部为负，
因此直接把负 DC 限制在高能量位置可能抑制目标候选并释放背景；`protect`
候选则保留高能量位置、把负 DC 主要留给非上尾区域。

公式冻结顺序已按以下流程完成：

```text
V3 两个各自最优 checkpoint（best / best_miou）
→ 同一 state、零训练比较 legacy / direct / protect
→ 同时查看固定 0.5 的 Pd、Fa、mIoU 和五个 Pd@Fa budget
→ 按预注册规则选择 complement-tail
→ 将选中公式写入正式训练协议与 source lock
→ 以 seed 42 fresh 初始化正式训练
```

实际结果是 direct 不合格、protect/complement 唯一合格。上尾响应来自父子
relay，且子 relay 已包含父 relay 输入，因此它仍只能称为“持续响应”，不能
表述为两个独立证据的因果确认。

需要明确：**未知模型的 Pd、Fa 和 mIoU 不能在正式训练前被保证。**
零训练 counterfactual 只负责冻结候选公式，不是正式 V4 性能。只有 seed 42
fresh 800-epoch run 的六组件原门槛全部通过，才将 V4 视为成功并进入下一阶段。

---

## 1. 当前状态与审查边界

### 1.1 已确认事实

正式修复报告已确认 baseline、V1、V2、V3 均使用各自内部选择的
`best.pth.tar` 和 `best_miou.pth.tar`；要求相同的是 split、训练轴、
evaluator、checkpoint role 与选择规则，而不是四个模型共享同一 checkpoint
或相同 epoch。正式范围只有 seed 42、NUDT-SIRST 内部 `530/133` 划分。

V3 的权威结果为：

| V3 checkpoint role | Pd@0.5 | Fa@0.5 | mIoU@0.5 | Tiny-Pd@0.5 | Pd@1e-6 | Pd@5e-6 | Pd@1e-5 | Pd@5e-5 | Pd@1e-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pd-primary `best` | 188/189 | 4.703837e-6 | 0.903948 | 39/39 | 9/189 | 188/189 | 188/189 | 188/189 | 188/189 |
| mIoU-secondary `best_miou` | 187/189 | 5.048020e-6 | 0.935640 | 39/39 | 0/189 | 187/189 | 187/189 | 187/189 | 187/189 |

六组件正式裁决为：

```text
Pd-primary absolute                         false
mIoU-secondary absolute                    false
Pd-primary V3 vs V1 relay-off              false  (2/5 non-inferior, 2/5 better)
mIoU-secondary V3 vs V1 relay-off          false  (4/5 non-inferior, 0/5 better)
Pd-primary V3 vs V2 structural predecessor true   (5/5 non-inferior, 1/5 better)
mIoU-secondary V3 vs V2 predecessor        true   (5/5 non-inferior, 4/5 better)

decision = RETURN_TO_MODEL_OPTIMIZATION
```

因此 V3 不是“全面无效”：它相对 V2 的两个 paired gate 均通过；但固定
0.5 的 Fa、mIoU、严格 `Fa≤1e-6` 上尾，以及相对 V1 各自最优 role 的比较
没有同时达到门槛。Tiny-Pd 在两个 V3 role 都是 39/39，只能报告，无区分度。

GPU2/GPU3 完成的是 seed 42、同 checkpoint 的 8 行 **NER DC-offset**
knockout，不是 tokenizer MPRS-DCH 消融，也不是 fresh-training 因果实验。
它不改变上述正式裁决。

当前可以排除 checkpoint 汇总错误；不能仅凭 knockout 把 V3 的失败唯一归因
于某种空间机制。V4 的作用域设计仍是需要 counterfactual 和 fresh training
共同检验的模型假设。

### 1.2 当前代码状态

生产 `model/` 目录中已经存在：

```text
tpd_ner_v8_mprs_dch.py       # V1
tpd_ner_v8_mprs_dch_v2.py    # V2
tpd_ner_v8_mprs_dch_v3.py    # V3
tpd_ner_v8_mprs_dch_v4_tail_aware.py  # V4 三作用域正式实现
```

V4 已集成到 `model/`，根目录同名文件是指向正式实现的兼容入口；正式测试位于
`tests/test_tpd_ner_v8_mprs_dch_v4_tail_aware.py`。实现严格枚举
`legacy_global / direct_tail / complement_tail`，默认构造为
`complement_tail`，且该公式现已由零训练对照唯一选中。正常模式和
`python -O` 各 14 项 V4 测试、各 5 项 V3 回归测试均已通过；GPU2/3 的
两个 checkpoint、三作用域对照已完成并聚合。V4 exact trainer、同版本
exact-resume 守卫、正式训练 source lock、GPU2/3 单卡 lane/launcher
均已完成；source lock 绑定 39 个运行源码，SHA256 为
`90dd24dfeef2d46c820fb5c89a899cec1961a7e718053f16395e256b3c27ccf3`。

真实 NUDT-SIRST 样本的 RTX 5090 smoke 已通过：batch 2 连续两步均得到
六个输出、有限 loss/梯度、三个 `dc_offsets` 的非零梯度和严格 state
回载；正式 batch 16 单步亦通过，峰值 allocated/reserved 分别约
`7301/8388 MiB`。seed 42 fresh 800-epoch 正式轨迹已固定在物理 GPU 2
启动，启动审计已观察到前 3 个 epoch 连续写入 exact journal。当前尚未完成
的是 800 epochs 训练、各自 `best` / `best_miou` 的正式评估与六组件裁决。

2026-07-28 16:46 主机重启使服务在已提交 epoch 144 后停止；运行目录没有
`summary.json`，因此没有误判为训练完成。重启后 source lock、V4 run
identity、active checkpoint 与 metrics boundary 均重新验证通过，训练以
`mode=exact_resume completed=144 next=145` 在同一物理 GPU 2 恢复，epoch
145 已连续提交。这构成一次真实的同版本 epoch-boundary exact-resume 验证。

无论最终选择 direct 还是 protect，V8 tokenizer 仍只替换
`mtc.embeddings_1/2`，保持 Keep–Context–Saliency 主线；NER 仍使用五个
非终端 evidence node，不改 SCTB、K/V、CFN 或 decoder 拓扑。[1][2][3]

---

## 2. 当前 NER 的完整数据流

以 `256×256` patch 为例，五个 evidence node 的空间尺度为：

| 节点 | 来源 | 典型通道 | 空间尺寸 |
|---|---|---:|---:|
| `h11` | `embeddings_1` 第 1 个非终端 block | 32 | `128×128` |
| `h12` | `embeddings_1` 第 2 个非终端 block | 32 | `64×64` |
| `h13` | `embeddings_1` 第 3 个非终端 block | 32 | `32×32` |
| `h21` | `embeddings_2` 第 1 个非终端 block | 64 | `64×64` |
| `h22` | `embeddings_2` 第 2 个非终端 block | 64 | `32×32` |

NER 递推为：

\[
q_4=\Phi_4(h_{13},h_{22},\operatorname{up}(d_5)),
\]

\[
q_3=\Phi_3(h_{12},h_{21},q_4,\operatorname{up}(d_4)),
\]

\[
q_2=\Phi_2(h_{11},q_3,\operatorname{up}(d_3)).
\]

每个 `q_s` 的固定宽度为 8。门控不是把 q 直接加到 decoder，而是在 CCA 已处理的 skip 上进行乘法调制：

\[
\widehat X_s=X_s^{CCA}\odot(1+M_s).
\]

代码中的实际路径是：

```python
skip_x_att = skip_x_att * (1.0 + mask)
output = nConvs(cat(skip_x_att, up))
```

因此，NER 的收益完全依赖于 `mask` 是否在正确位置增强或抑制 encoder skip；q 本身不是一条独立的 decoder residual。[3][6]

---

## 3. V1 → V2 → V3 分别解决了什么

### 3.1 V1：建立五节点显式 NER

V1 完成了：

- 显式 `forward_with_evidence()`；
- `h11/h12/h13/h21/h22` 五节点；
- `q4→q3→q2` 递推；
- 宽度 8 的窄 relay；
- 零初始化空间 gate；
- relay-on 与 relay-off 的 step-0 输出一致。

但 V1 的融合源幅值未严格平衡，gate 的动态范围也更宽，容易出现某一路 evidence 主导或 mask 过强。

### 3.2 V2：解决数值尺度与门控范围

V2 增加两项关键约束：

1. 每个 source projection 和融合后的 relay value 都进行逐样本 full-tensor RMS normalization；
2. gate 去 bias，logits 做逐样本空间中心化，再用

\[
M=\frac{1}{\pi}\arctan(\pi Z)
\]

映射到严格接近 `(-0.5, 0.5)` 的区间，使 skip factor 保持在 `(0.5, 1.5)` 内。[4]

V2 修复的是**尺度不稳定与过强门控**，但空间中心化强制每个 stage 的 gate 均值接近 0，无法学习一个 stage 级整体偏置。

### 3.3 V3：增加 stage-wise post-centering DC

V3 在每个 stage 增加一个零初始化标量：

```python
centered_logits = spatially_center_gate_logits(logits)
shifted_logits = centered_logits + dc_offsets[str(stage)]
mask = arctangent_residual_mask(shifted_logits)
```

即：

\[
M_s=\frac{1}{\pi}\arctan\left[\pi(Z_s+d_s)\right].
\]

这允许 stage2、stage3、stage4 分别学习全局增益或抑制倾向，同时保持参数量仅增加 3。V3 仍保留零 gate、零 DC 的 relay-off 精确初始化。[5]

V3 增加的是 **NER stage 级整体校准**；它与 tokenizer 的 Deferred
Context Headroom 无关。其公式把每个 DC 广播到本 stage 全图。8 行
DC-offset knockout 能确认已训练权重对三个标量的敏感性，但不能单独证明
“全图广播”就是 V3 失败的唯一原因。

---

## 4. V3 失败原因分析

## 4.1 首要结论：NER DC offset 联合有效，但 stage 作用具有角色差异

两个各自最优 checkpoint 学到的 DC 均为负：

| checkpoint role | d2 | d3 | d4 |
|---|---:|---:|---:|
| Pd-primary | -0.168368 | -0.254669 | -0.615870 |
| mIoU-secondary | -0.274605 | -0.396151 | -0.743328 |

逐 stage knockout 相对同 role learned V3 的固定 0.5 差值为：

| knockout | Pd-role Δmatched | Pd-role ΔFa | Pd-role ΔmIoU | mIoU-role Δmatched | mIoU-role ΔFa | mIoU-role ΔmIoU |
|---|---:|---:|---:|---:|---:|---:|
| zero all DC | 0 | +9.407674e-6 | -0.007623 | +1 | +3.545087e-5 | -0.042049 |
| zero stage4 | 0 | +6.080570e-6 | -0.003578 | 0 | +1.399678e-5 | -0.017452 |
| zero stage3 | 0 | +2.294555e-7 | +0.002176 | 0 | +1.147277e-6 | -0.001235 |
| zero stage2 | 0 | +1.147277e-7 | +0.001598 | 0 | +5.736387e-7 | +0.001019 |

这支持以下受限结论：

- stage4 在两个 checkpoint role 上均是最强、方向一致的同权重正贡献；
- stage3 在 Pd-role 上呈 Fa–mIoU 权衡，在 mIoU-role 上为小幅正贡献，
  因而应称“弱且 role-dependent”，不能笼统称为已证实正向；
- stage2 在两个 role 上均表现为略降 Fa、略伤 mIoU 的混合权衡；
  Pd-role 的 `Fa≤1e-6` 下，zero-stage2 为 10/189，learned V3 为
  9/189，因此不能声称 stage2 已证明局部目标贡献；
- 八行均来自 seed 42、已训练 checkpoint 的 evaluation-only
  counterfactual，不建立跨 seed 或 fresh-training 因果结论。

所以“高分辨率全图广播可能不是最佳作用域”是合理待检验假设，但“空间支撑
错误已经被证明”不是当前证据允许的结论。

---

## 4.2 stage2 的背景暴露量远大于 stage4

对 `256×256` patch：

| Stage | Gate 尺寸 | 空间位置数 | 相对 stage4 |
|---|---:|---:|---:|
| stage4 | `32×32` | 1,024 | `1×` |
| stage3 | `64×64` | 4,096 | `4×` |
| stage2 | `128×128` | 16,384 | `16×` |

红外小目标通常只覆盖极少像素。V3 的单个 `d_2` 被广播到 16,384 个位置：

- 当 `d_2>0` 时，大量高分辨率背景 skip 被同步增强；
- 当 `d_2<0` 时，目标附近的细粒度响应也被同步压低；
- 即使 `|d_2|` 很小，乘法作用的背景位置总量仍远大于 stage4；
- 在当前 seed 42 的不同 checkpoint role / 工作点上，其影响可能表现为
  Fa 与 mIoU 的不同权衡。

这与“stage4 同权重作用最强、stage2 呈混合权衡”一致，但空间暴露只是候选
解释，必须由响应统计和三作用域 counterfactual 继续检验。

---

## 4.3 V3 的 DC 同时改变 mask 均值和空间判别斜率

V3 的门控映射为：

\[
f(u)=\frac{1}{\pi}\arctan(\pi u).
\]

其导数为：

\[
f'(u)=\frac{1}{1+\pi^2u^2}.
\]

加入全局 `d_s` 后：

\[
u=Z_s+d_s.
\]

因此，`d_s` 不仅改变 stage 的平均增益，还会把所有位置统一移向 arctan 的饱和区：

- `|d_s|` 增大时，局部 `Z_s` 的有效梯度会降低；
- stage 级校准与空间差异学习被耦合；
- 在 stage2，目标与背景均被同一偏置改变，局部对比优势可能反而减弱。

当前不能直接推出“把 DC 限制到目标上尾”就是正确答案。尤其现有
`d2/d3<0`：direct-tail 会把负抑制施加到高能量位置；protect 候选则将负抑制
主要留在非上尾位置。二者必须与 legacy global 在同 state 下比较后再冻结。

---

## 4.4 full-tensor RMS 解决幅值，但不解决“稀疏上尾与分布式杂波”的区别

V2/V3 的 RMS normalization 对每个 BCHW sample 使用一个全张量尺度。它能防止某个 source 因数值幅值更大而支配融合，但两个响应可以具有相近 RMS：

```text
A：极少数位置具有很高的小目标响应
B：大量位置具有中等强度的云边、热噪声或结构杂波
```

RMS 不能自动判断 A 比 B 更值得获得 DC 校准。空间上尾可能富集小目标，也
可能富集高亮杂波，因此它是可审计的路由信号，不是天然的目标标签。V4
候选在 RMS 平衡之后检查这一上尾结构，而不是重新扩大 relay 宽度。

---

## 4.5 三个 stage 的 evidence 质量并不对称

### stage4

```text
(h13, h22, up4)
```

- `h13` 与 `h22` 都位于 `32×32`；
- 两条浅层 tokenizer 路径在该尺度汇合；
- decoder 已具有较强语义；
- 空间范围小，背景暴露有限。

因此 q4 是当前最值得优先检查的低分辨率位置先验，但仍不能把其高能量位置
直接等同于目标。

### stage3

```text
(h12, h21, q4, up3)
```

stage3 同时获得本层 evidence 和 q4。其 response 可表达父子 relay 的持续性，
但因为 q3 本身已经包含 q4，它不是两个独立信号的确认。knockout 显示 stage3
为弱且 role-dependent 的贡献；分辨率扩大后，是否继续全图 DC 应由三作用域
counterfactual 决定。

### stage2

```text
(h11, q3, up2)
```

stage2 只有一条高分辨率 tokenizer evidence `h11`，且空间位置数最大。它可能
更偏向边界和细粒度恢复，但现有 knockout 只能确认 Fa–mIoU 混合权衡，不能
证明混合性必然来自支撑区域过宽。

---

## 4.6 NER 只调制 skip，因此 stage2 的错误会直接污染高分辨率重建

当前 NER 不直接把 q 注入 decoder，而是将 mask 乘到 CCA skip。stage2 skip 仍包含大量局部纹理、背景边缘和热噪声。若 stage2 mask 因全局 DC 整体偏正，则这些响应会一起进入后续卷积：

```text
可能结果：Pd 不一定提升，Fa 增加，mIoU 因边界扩张或离散杂点下降
```

若 mask 整体偏负：

```text
可能结果：Fa 下降，但弱小目标或目标边缘被抑制，Pd / mIoU 下降
```

这两种方向与当前 Fa–mIoU 混合权衡相容，但不是由 knockout 单独证明的
空间因果解释。

---

## 4.7 当前不应采用的修改

在 V4 中不建议同时进行以下改动：

- 不扩大 relay width；
- 不增加新的 attention branch；
- 不把 q 直接加到 decoder；
- 不修改 SCTB Query / Key / Value；
- 不加入 Survival loss；
- 不加入 Query-only FG；
- 不改 BCE 或六路 deep supervision；
- 不增加形态学后处理；
- 不根据验证结果重写 Pd/Fa/mIoU 门槛；
- 不删除 stage2 整个 NER stage。

否则无法判断性能变化究竟来自 tail-aware 支撑、容量、损失还是后处理。

---

## 5. V4 候选设计：Tail-Aware NER DC-Offset Scope

## 5.1 不变项

V4 必须原样保留：

```text
1. V8-MPRS-DCH tokenizer
2. Keep / Context / Saliency 三源
3. 仅替换 embeddings_1 / embeddings_2 的主线边界
4. h11 / h12 / h13 / h21 / h22 五节点
5. q4 → q3 → q2 顺序
6. relay width = 8
7. V2 full-tensor RMS normalization
8. bias-free 1×1 gate
9. per-sample mean_hw centering
10. atan bounded mask
11. CCA 后、concat 前的 skip multiplicative modulation
12. 六路输出、BCE、optimizer、scheduler、数据和 evaluator
13. 三个零初始化 dc_offsets state key
```

---

## 5.2 空间上尾支持

对某一级 relay value：

\[
q_s\in\mathbb{R}^{B\times C_e\times H_s\times W_s},\quad C_e=8.
\]

先计算通道 RMS energy：

\[
E_s(i,j)=
\sqrt{
\frac{1}{C_e}
\sum_{c=1}^{C_e}q_{s,c}(i,j)^2+\epsilon
}.
\]

在每个样本内做空间标准化：

\[
R_s=
\frac{E_s-\mu_{hw}(E_s)}
{\sqrt{\operatorname{mean}_{hw}[(E_s-\mu_{hw}(E_s))^2]+\epsilon}}.
\]

定义上尾支持：

\[
T_s=\tanh\left(\operatorname{ReLU}(R_s-\kappa_s)\right).
\]

性质：

- `T_s∈[0,1)`；
- 平坦 response 得到全零支持；
- 不像 top-k，负样本不被强制选出固定数量的位置；
- 不增加参数；
- 对 channel 数和绝对幅值不敏感；
- 只保留相对空间上尾。

用于零训练审计的预注册初始值：

```python
kappa4 = 1.5
kappa3 = 2.0
kappa2 = 2.5
```

越靠近高分辨率 stage，阈值越严格。这些值不是由正式结果证明的最优值，只是
在三作用域比较前固定的机制审计起点。

最终选中的作用域与阈值必须在正式 fresh training 前写入 source lock。允许
使用**训练集且仅训练集**做一次 occupancy audit；不得在看到正式 V4
验证结果后反复调参。

---

## 5.3 父子 relay 持续上尾响应

只使用本层上尾可能选中单层噪声，因此候选 B/C 利用既有
`q4→q3→q2` 递推计算父子 relay 的持续响应。需要限定：q3 已包含 q4，q2
已包含 q3，二者不是独立证据，此处的 \(P_s\) 只是可检验的持续性统计量。

### stage4

\[
P_4=1.
\]

stage4 的 NER global DC 保持 V3 完全一致。其同权重 knockout 在两个 role
上方向一致，是当前最强的 stage 证据；这仍不外推为跨 seed 结论。

### stage3

\[
P_3=\sqrt{T_3(q_3)\cdot\uparrow T_4(q_4)}.
\]

该式仅定义 \(P_3\)，是否用 \(P_3\) 或 \(1-P_3\) 调制 `d_3` 由
counterfactual 决定。

### stage2

\[
P_2=\sqrt{T_2(q_2)\cdot\uparrow T_3(q_3)}.
\]

该式仅定义 \(P_2\)，是否用 \(P_2\) 或 \(1-P_2\) 调制 `d_2` 由
counterfactual 决定。

几何平均避免两个小于 1 的 support 直接相乘后幅值过度衰减；不能把它表述为
两个独立目标证据的“确认”。

---

## 5.4 三种作用域与 stop-gradient

对 stage3/2，预注册三种作用域：

\[
A_s^{legacy}=1,
\]

\[
A_s^{direct}=\operatorname{sg}(P_s),
\]

\[
A_s^{protect}=\operatorname{sg}(1-P_s).
\]

对应：

\[
\widetilde Z_s^{r}=Z_s+d_s A_s^{r}.
\]

其中 direct 把 DC 放在持续上尾，protect 则保护持续上尾、把 DC 放在补集。
因为现有 `d2/d3` 均为负，二者的实际方向相反，必须实测而不能按名称预判。
`P_s` 在 direct/protect 的 DC 分支停止梯度。

理由：

- 防止 DC 分支通过 support 路径直接推动 q 扩大或缩小自己的作用域；
- 保持 tail support 是路由条件，而不是新的隐式目标函数；
- 使三候选保持相同参数和 state，隔离作用域差异。

普通 gate 路径仍正常反向传播，fusion 仍可通过 gate 学习。

---

## 5.5 候选门控与公式冻结

\[
M_s^{r}=
\frac{1}{\pi}
\arctan
\left[
\pi
\left(
Z_s+d_sA_s^{r}
\right)
\right].
\]

\[
\widehat X_s=X_s^{CCA}\odot(1+M_s).
\]

mask 和 skip factor 的边界仍为：

\[
M_s\in(-0.5,0.5),
\]

\[
1+M_s\in(0.5,1.5).
\]

三候选的零训练选择必须同时读取 V3 自己的 `best` 与 `best_miou`
checkpoint，报告固定 0.5 的 Pd/Fa/mIoU、false objects/image、tiny-Pd
以及五个 Pd@Fa budget。选择规则在第 10 节给出。选择结果只冻结公式，不计入
V4 六组件正式 gate。

---

## 6. 三个候选共享的严格工程性质

| 性质 | V4 状态 |
|---|---|
| SCTransNet encoder / SCTB / decoder | 不变 |
| V8 tokenizer | 不变 |
| 五 evidence nodes | 不变 |
| q4→q3→q2 | 不变 |
| relay width | 8，不变 |
| learnable parameter 数 | 与 V3 相同 |
| persistent buffer 数 | 与 V3 相同 |
| state key | 与 V3 相同 |
| V3 checkpoint strict load | 应通过 |
| stage4 forward | 与 V3 逐元素一致 |
| zero gate + zero DC 输出 | 与 relay-off 逐元素一致 |
| mask bounds | 与 V2/V3 一致 |
| stage3/2 DC 梯度 | 随最终选中的 legacy/direct/protect 作用域变化 |
| direct/protect support 对 q 的梯度 | stop-gradient |

在 step 0：

```text
gate weights = 0
dc_offsets = 0
mask = 0
```

所以：

- 六个 deep-supervision 输出与 relay-off 一致；
- shared parent 参数梯度一致；
- gate 梯度与 V3 一致；
- relay fusion 仍不因零 gate 获得非预期梯度；
- stage4 DC 梯度与 V3 一致；
- direct/protect 的 stage3/2 DC 梯度是相对 V3 有意改变的唯一首步优化差异；
- 若 legacy 获胜，则局部作用域假设未获授权，不创建“换名 V4”。

---

## 7. 三作用域比较为何是当前最小变量下一步

## 7.1 保留已经被消融证明有效的 stage4

V4 对 stage4 不做近似、不做缩放，也不做 tail 限制：

```text
P4 = 1
```

在相同权重和输入下，stage4 的 relay value、shifted logits 和 mask 应与 V3 逐元素一致。这样不会因新模型而丢失最明确的正贡献。

## 7.2 stage3 不预判 direct 或 protect

stage3 是弱且 role-dependent 的贡献，不应直接关闭，也不能预先认定
direct-tail 最优。三候选分别回答：

```text
legacy：是否仍需要全图负校准
direct：持续上尾是否应承担 DC
protect：持续上尾是否应免受负 DC，而非上尾继续受抑制
```

## 7.3 stage2 以 Pd–Fa–mIoU 联合结果裁决

三候选均保留 stage2 的普通 centered spatial gate、q2 和 decoder 拓扑。
候选差异只在 `d_2` 的作用域，不能通过单看 mIoU 或单看 Fa 决定。

需要检验：

- 固定 0.5 的 Pd、Fa、mIoU 是否形成联合改善；
- 五预算尤其 `Fa≤1e-6` 的 Pd 是否保持或提高；
- false components 是否减少；
- 39/39 tiny-Pd 是否保持；它仅报告，不参与公式单独晋级；
- direct 是否因负 DC 抑制上尾，protect 是否因补集过宽而过度全局化。

## 7.4 不通过增加容量“碰运气”

三候选参数量与 V3 相同，零训练差异可限定为：

> NER stage-wise DC 使用 global、持续上尾或持续上尾补集。

fresh training 后仍只能声称最终所选作用域与训练轨迹共同产生了结果；不能仅凭
knockout 宣称上尾因果机制已经证明。

---

## 8. 代码修改方案

## 8.1 新增核心模型文件

新增：

```text
model/tpd_ner_v8_mprs_dch_v4_tail_aware.py
```

建议不要原地修改 V3。V3 已经有正式结果，应保持 immutable。

公式冻结前的生产实现已显式支持三种 scope，且不改变任何 state。实际代码
使用按最大绝对值缩放的 RMS 与 `torch.hypot`，避免有限 FP16/BF16/FP32
极值在平方归约时溢出；下列伪代码只展示作用域逻辑：

```python
def relay_spatial_tail_support(q, z_threshold, eps=1e-6):
    energy = stable_channel_rms(q.float(), eps)
    centered = energy - stable_spatial_mean(energy)
    z = centered / stable_spatial_rms(centered, eps)
    return torch.tanh(F.relu(z - z_threshold))


def persistent_tail(stage, relay_value, sources, output_size):
    if stage == 4:
        return ones(B, 1, H, W)

    if stage == 3:
        parent = sources[2]  # q4
        local = tail(relay_value, kappa3)
        deep = upsample(tail(parent, kappa4))

    if stage == 2:
        parent = sources[1]  # q3
        local = tail(relay_value, kappa2)
        deep = upsample(tail(parent, kappa3))

    return sqrt(clamp(local * deep, min=0))


def dc_scope(policy, stage, p):
    if stage == 4 or policy == "legacy_global":
        return ones_like(p)
    if policy == "direct_tail":
        return p.detach()
    if policy == "complement_tail":
        return (1.0 - p).detach()
    raise ValueError(policy)


def forward_stage(stage, sources, output_size, policy):
    q = fusion[stage](sources, output_size)
    logits = gate[stage](q)
    z = spatial_center(logits)
    p = persistent_tail(stage, q, sources, output_size)
    a = dc_scope(policy, stage, p)
    shifted = z + dc_offset[stage] * a
    mask = atan(pi * shifted) / pi
    return q, mask
```

正式实现与根目录兼容入口为：

```text
model/tpd_ner_v8_mprs_dch_v4_tail_aware.py
SHA256: ce64dbbdfba76cdf8f3f2f331e1ac60d4e76e23a053d73a83b785e9ca1edf37f

tpd_ner_v8_mprs_dch_v4_tail_aware.py
SHA256: 403c641bc16d29d514e127e97e8a0d975b0d5722bd8b540a0818ee715233f233
```

生产文件已具备三公式同 state 诊断能力；正式训练 builder 仍必须在
counterfactual 后只允许唯一获选的作用域。不能因为生产实现已存在而跳过
公式选择。

---

## 8.2 训练入口

新增：

```text
experiments/train_tpd_ner_v8_mprs_dch_v4.py
experiments/train_tpd_ner_v8_mprs_dch_v4_exact.py
```

修改原则：

1. 复用 V3 的 dataset、optimizer、scheduler、loss、checkpoint selector；
2. 公式冻结后，builder 仅把 V3 adapter 替换为唯一获选的 V4 adapter；
3. 正式 CLI 不开放可随意修改的 tail threshold；
4. threshold 值由协议和 architecture manifest 固定；
5. exact-resume 继续保存 Python / NumPy / Torch CPU / CUDA / DataLoader generator 状态；
6. 不从 V3 checkpoint warm-start 正式训练；正式 run 使用与 V3 相同的 fresh parent 与配对初始化；
7. V3 的 `best` / `best_miou` 只用于零训练三作用域 counterfactual；
8. 正式 V4 只使用 seed 42，不增加其他 seed。

示意：

```python
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    adapt_v8_mprs_dch_parent_v4,
)

model = adapt_v8_mprs_dch_parent_v4(
    parent,
    variant=parent_variant,
    relay_enabled=relay_enabled,
    relay_width=8,
    relay_initialization_seed=42,
)
```

---

## 8.3 评估入口

新增：

```text
experiments/evaluate_tpd_ner_v8_mprs_dch_v4_pd_fa.py
```

必须复用现有：

- 固定阈值 0.5；
- closed-interval sweep；
- Hungarian matching；
- match radius；
- tiny area 定义；
- 五个 Fa budget；
- best / best_mIoU / last checkpoint 角色。

正式比较只由各模型自己内部选择的 `best` 和 `best_miou` 相同 role 参与
六组件 gate；`last` 可审计但不参与正式裁决。不得因 V4 改变连通域规则、
checkpoint selector 或后处理。

---

## 8.4 Tail mechanism audit

新增：

```text
analysis/analyze_tpd_ner_v4_tail_support.py
```

seed 42 的每个 stage、每个 checkpoint role 至少输出：

| 指标 | 定义 |
|---|---|
| `support_occupancy_all` | `P_s>0` 的全部像素比例 |
| `support_occupancy_target` | GT 邻域内 `P_s>0` 的比例 |
| `support_occupancy_background` | 非 GT 区 `P_s>0` 的比例 |
| `support_mean_target` | GT 邻域内 support 均值 |
| `support_mean_background` | 背景 support 均值 |
| `tail_separation` | target mean / background mean |
| `persistence_rate` | local tail 中得到 deeper tail 支持的比例 |
| `dc_offset` | 每 stage 学到的标量值 |
| `dc_effective_mass` | `mean(abs(d_s*A_s^r))` |
| `mask_target/background quantiles` | mask 在目标与背景的分位数 |
| `stage component delta` | 开启该 stage 后 unmatched component 的变化 |

该审计用于解释结果，不取代原性能 Gate。

---

## 8.5 Smoke、source lock 与 finalizer

新增：

```text
experiments/smoke_tpd_ner_v8_mprs_dch_v4.py
experiments/launch_tpd_ner_v4_formal800_2x5090.sh
experiments/run_tpd_ner_v4_formal800_worker.sh
experiments/finalize_tpd_ner_v4.py
experiments/TPD_NER_V4_TAIL_AWARE_PROTOCOL.md
experiments/tpd_ner_v4_source_lock.json
```

source lock 至少包括：

```text
V8 tokenizer source
V1 / V2 / V3 / V4 NER source
SCTransNet source
train / exact-resume / evaluator / finalizer
protocol
split IDs and hashes
dataset fingerprints
normalization values
PyTorch / CUDA / cuDNN / driver
GPU UUID
all CLI arguments
tail thresholds
architecture manifest
```

---

## 9. 必须增加的测试

建议新增：

```text
tests/test_tpd_ner_v8_mprs_dch_v4_tail_aware.py
```

至少包含以下测试。

### 9.1 结构与状态

1. V4 与 V3 relay state key 完全一致；
2. V4 与 V3 参数量完全一致；
3. V3 relay state 可 `strict=True` 加载到 V4；
4. V4 没有新增 persistent buffer；
5. architecture manifest 包含已冻结的唯一 scope、thresholds 和
   stop-gradient policy；
6. legacy/direct/protect 诊断模式不注册额外参数或 persistent buffer。

### 9.2 Tail support 数学性质

7. flat q 的 support 精确为 0；
8. 单点 spike 仅产生局部 support；
9. support 有限并位于 `[0,1)`；
10. FP16 / BF16 reduction 使用 FP32 且输出有限；
11. stage3 本层有 tail、q4 无 tail 时 persistent support 为 0；
12. stage2 本层有 tail、q3 无 tail 时 persistent support 为 0；
13. direct 与 protect 的最终 scope 不需要梯度。

### 9.3 与 V3 的受控关系

14. stage4 在同 state、同输入下 relay value 和 mask 与 V3 逐元素一致；
15. legacy global 三个 stage 与 V3 逐元素一致；
16. direct 使用 \(P_s\)，protect 使用 \(1-P_s\)，二者互补；
17. zero gate / zero DC 时三个 scope 的 mask 均精确为 0；
18. step 0 的六个整模输出与 relay-off 逐元素一致；
19. shared parent、gate 和 fusion 的第一步梯度满足配对预期；
20. stage4 DC 第一梯度与 V3 一致；
21. stage3/2 DC 梯度按选中 scope 累计；
22. 非零 DC 时三个 scope 在 stage3/2 按设计分化；
23. skip factor 始终严格位于 `(0.5,1.5)`。

### 9.4 训练系统

24. 两步 CPU forward/backward；
25. RTX 5090 两步 smoke；
26. strict checkpoint reload；
27. uninterrupted 与 exact-resume 逐 tensor 一致；
28. best / best_mIoU / last 均可被 evaluator 严格加载；
29. source-lock mismatch 必须 fail closed。

正式测试与根目录兼容测试入口为：

```text
tests/test_tpd_ner_v8_mprs_dch_v4_tail_aware.py
SHA256: eff88c18ae98137fe3fe3e29b6d6a9c968b8e9aab4e619a4033f4c95c0169446

test_tpd_ner_v8_mprs_dch_v4_tail_aware.py
SHA256: f0e5e778d28cde3974edd365ff0740555fea224c1ce247177c338b8a81d7a102
```

上述测试在隐藏 GPU 的 CPU 环境中，正常模式与 `python -O` 均为 14/14
通过；V3 回归测试正常模式与 `python -O` 均为 5/5 通过。

---

## 10. 正式训练前的零训练诊断

在写 launcher 前，对 seed 42 已封存 V3 的两个各自最优 checkpoint 做一次
**零训练三作用域 counterfactual**：

```text
分别加载 V3 best / best_miou state
→ 构建同 state、同参数、同 evaluator 的诊断模型
→ 不更新参数
→ legacy global / direct-tail / target-protective complement
→ 每个 role 分别输出固定 0.5 与完整 closed-interval sweep
```

该诊断必须回答：

- stage4 是否确实逐元素不变；
- legacy 三 stage 是否与原 V3 逐元素一致；
- direct 对负 `d_3/d_2` 是否抑制了高能量目标候选；
- protect 是否在保留目标上尾的同时继续抑制非上尾背景；
- 两个 checkpoint role 的固定 Pd、Fa、mIoU 如何变化；
- 五个 Pd@Fa budget，尤其 `Fa≤1e-6`，是否非劣或提高；
- stage2 support 是否仍过于广泛。

该结果只能作为**公式冻结诊断**，不能作为 V4 正式性能，因为 V3 权重是在
legacy global 公式下训练得到的。

在任何结果出现前固定以下判断：

```text
实现门：
  stage4 非逐元素一致                         => 实现错误，停止
  legacy global 非三 stage 逐元素一致         => 实现错误，停止
  state key / 参数量 / evaluator 任一不同      => 比较无效，停止

局部作用域资格门（direct、protect 分别对 legacy）：
  每个 role 五预算至少 4/5 Pd 非劣             => 必须
  两个 role 合计至少一个预算 Pd 严格提高        => 必须
  固定 0.5 不得同时在 matched、Fa、mIoU 上
  被 legacy Pareto 支配                        => 必须

选择：
  只有一个局部候选合格                         => 冻结该候选
  两个均合格且一方在两个 role 的固定指标和
  五预算上联合支配另一方                       => 冻结支配者
  两个均合格但形成混合权衡                     => FORMULA_INCONCLUSIVE，不训练
  两个均不合格或 legacy 最强                    => LOCAL_SCOPE_REJECTED，不训练
```

选择时同时保存每行 checkpoint、state、scope、阈值、固定指标、五预算和源码
哈希。Tiny-Pd 相对 39/39 的变化必须披露，但仅报告、不参与候选资格或正式
gate。不得用多个 threshold 反复搜索验证指标，也不得把 legacy 重新命名为
V4 成功。

### 10.1 已完成结果与公式裁决

GPU2 评估 V3 自己的 Pd-primary `best`（epoch 253），GPU3 评估 V3 自己的
mIoU-secondary `best_miou`（epoch 489）。两个进程均先证明
`legacy_global` 与冻结 V3 sweep 的全部预测派生字段 canonical exact，再评估
direct 和 complement；六行均 strict-load 同一 source state，未写派生
checkpoint。

| Role | Scope | matched@0.5 | Fa@0.5 | mIoU@0.5 | false objects/image | 五预算 matched：`1e-6 / 5e-6 / 1e-5 / 5e-5 / 1e-4` |
|---|---|---:|---:|---:|---:|---|
| Pd-primary | legacy | 188/189 | 4.703837e-6 | 0.903948 | 0.045113 | 9 / 188 / 188 / 188 / 188 |
| Pd-primary | direct | 188/189 | 5.048020e-6 | 0.905775 | 0.052632 | 9 / 188 / 188 / 188 / 188 |
| Pd-primary | complement | 188/189 | 4.703837e-6 | 0.904118 | 0.045113 | **10** / 188 / 188 / 188 / 188 |
| mIoU-secondary | legacy | 187/189 | 5.048020e-6 | 0.935640 | 0.052632 | 0 / 187 / 187 / 187 / 187 |
| mIoU-secondary | direct | 187/189 | 6.768937e-6 | 0.934342 | 0.082707 | 0 / 187 / 187 / 187 / 187 |
| mIoU-secondary | complement | 187/189 | 5.277476e-6 | 0.935994 | 0.045113 | 0 / 187 / 187 / 187 / 187 |

六行固定 tiny-Pd 均为 39/39。按预注册资格门：

```text
direct_tail:
  两个 role 的预算计数虽均为 5/5 非劣，
  但跨 role 没有任何严格 Pd 提升；
  且在 mIoU-secondary 固定点被 legacy 在 matched/Fa/mIoU 上 Pareto 支配；
  qualifies=false。

complement_tail:
  两个 role 均为 5/5 预算非劣；
  Pd-primary 的 Fa≤1e-6 从 9/189 提到 10/189；
  两个固定点均未被 legacy Pareto 支配；
  qualifies=true。

decision=COMPLEMENT_TAIL_SELECTED
selected_formula_mode=complement_tail
formal_v4_formula_selected=true
```

这只冻结 stage3/2 的正式公式为 \(d_s(1-P_s)\)。它没有使 V3 旧权重达到 V4
正式 absolute gate，不能替代 seed 42 fresh 800-epoch 训练。

冻结产物：

```text
best counterfactual JSON SHA256:
2813043c31009d38c579823efd2d01f68a7d90f7aaef81a826ca4202c40eee5d

best_miou counterfactual JSON SHA256:
f1720e98638349e8d0e454c6ad2f5d10caa418c76e77059db1f034d936a24480

aggregate JSON SHA256:
07f6d9b5bdabcc5df1a323485bbf590fb8e297a8361998816cfb256b803ae3d7

aggregate Markdown SHA256:
a80bfc1d3cc4f1463298c86d1136138431cb4499fc5c2ac672d7e29058fdd9a8
```

---

## 11. Tail threshold 的冻结策略

三作用域 counterfactual 已使用并冻结：

```text
kappa4 = 1.5
kappa3 = 2.0
kappa2 = 2.5
```

以下训练集 occupancy 项只保留为后续诊断记录，不再具有调整 threshold 的
权限：

```text
q4 tail：应覆盖多数目标粗位置，但不能覆盖大面积背景
P3：应比 T3 更稀疏，并在目标区域保持明显高于背景的 support
P2：应最稀疏；若几乎全零，direct 接近关闭 DC，而 protect 接近 legacy global
```

建议记录但不替代性能 Gate 的参考范围：

| 指标 | 参考目标 |
|---|---:|
| q4 background occupancy | `<10%` |
| P3 background occupancy | `<5%` |
| P2 background occupancy | `<2%` |
| target/background support ratio | `>1`，越高越好 |

V3 checkpoint counterfactual 已经完成，因此不得再从任何网格重选
threshold。正式 V4 必须继续使用 `kappa4/3/2 = 1.5/2.0/2.5`。

---

## 12. 正式实验矩阵

原则：复制 V3 的固定训练轴，但严格保持当前单 seed 范围。公式冻结后只新增
一个 seed 42 的 V4 fresh run；baseline、V1、V2、V3 已封存结果继续只读，
不因 V4 重训或改变它们的 checkpoint。

| Candidate | Formula | Seed | Checkpoint selection |
|---|---|---:|---|
| V4-TA-DC | counterfactual 唯一获选的 direct 或 protect | 42 | 模型自己的 `best` / `best_miou` |

该唯一正式 run：

```text
800 epochs
FP32
相同 530/133 split
相同 augmentation
相同 Adam / LR / warmup / cosine
相同 eval cadence
相同 best / best_mIoU / last
相同 closed-interval sweep
相同五个 Fa budget
不 early-stop
不从 V3 warm-start
```

正式 V4 seed-42 run 可按当前资源约束使用物理 GPU2/GPU3；GPU 编号只属于
launcher 与运行协议，不能写入模型定义。

---

## 13. V4 的正式通过条件

## 13.1 原门槛不得修改

V4 是否通过，继续由已封存的正式门槛判定。不得因 V4 接近门槛而：

- 改 threshold；
- 改 Fa budget；
- 改 checkpoint 排序；
- 用其他 seed 替换固定 seed 42；
- 用 counterfactual 或 smoke 代替正式 run；
- 用后处理掩盖模型输出。

## 13.2 六组件正式门槛

V4 使用自己的内部验证 `best.pth.tar` 作为 Pd-primary，使用自己的
`best_miou.pth.tar` 作为 mIoU-secondary。baseline、V1、V2 也只使用各自
相同 role 的最优 checkpoint；不要求 checkpoint epoch 或旧 protocol 文本相同。

固定阈值 0.5 的 absolute 门：

| Role | 最低 matched / Pd | 最高 Fa | 最低 mIoU |
|---|---:|---:|---:|
| Pd-primary | 188/189 | 1e-6 | 0.933647 |
| mIoU-secondary | 187/189 | 1e-6 | 0.946542 |

两个 role 均必须通过全部五个 absolute budget：

| Fa budget | 最低 matched / Pd |
|---:|---:|
| 1e-6 | 187/189 |
| 5e-6 | 188/189 |
| 1e-5 | 188/189 |
| 5e-5 | 188/189 |
| 1e-4 | 188/189 |

paired gate 对两个 checkpoint role 分别计算：

```text
V4 vs V1 relay-off：
  五预算至少 4/5 non-inferior，且至少 1/5 strictly better

V4 vs V2 structural predecessor：
  五预算至少 4/5 non-inferior，且至少 1/5 strictly better
```

六个必须同时为 true 的组件是：

```text
1. pd_primary_absolute
2. miou_secondary_absolute
3. pd_primary_v4_vs_v1
4. miou_secondary_v4_vs_v1
5. pd_primary_v4_vs_v2
6. miou_secondary_v4_vs_v2
```

baseline 只报告差值，不影响裁决。Tiny-Pd 必须继续报告相对 39/39 是否回退，
但它不是第七个独立 pass gate。`last` checkpoint 也不参与六组件裁决。

## 13.3 机制审计不能替代性能

即使 tail support 看起来集中、stage2 背景 occupancy 很低，只要上述
Pd/Fa/mIoU、五预算或 paired gate 任一未过，结论仍然是：

```text
RETURN_TO_MODEL_OPTIMIZATION
```

只有全部原门槛通过，才能设置：

```text
decision=NER_V4_GATE_PASS
v4_tail_aware_accepted=true
next_model_stage_authorized=true
```

---

## 14. 失败保护与后备边界

direct 的主要风险是 support 过窄或负 DC 抑制目标上尾；protect 的主要风险是
补集过宽、退化为近似 global。三作用域 counterfactual 不允许在看到结果后
新增第四种即时公式。

### V4b：stage2 DC hard-off

```text
P4 = 1
P3 = 由后续独立计划预注册的作用域
P2 = 0
```

注意：

- stage2 的普通 V2 centered spatial gate 仍保留；
- 仅禁止 `d_2`；
- 不删除 q2，不改变 decoder；
- 仍不增加参数。

V4b 只在以下条件下进入下一轮，而不能在看到正式 V4 验证性能后即时替换：

```text
V4 完成且失败；
机制审计证明 stage2 effective DC 与背景 false component 明确正相关；
seed 42 的 best / best_miou 两个 role 均支持关闭 stage2 DC；
```

当前不先用 V4b 取代三作用域比较，因为 stage2 knockout 是 Fa–mIoU 混合权衡，
尚未证明稳定负向。V4b 不是本轮候选，也不能用于替换失败的正式 V4 结果。

---

## 15. V4 通过后的下一步

V4 通过原门槛后：

1. 冻结 `V8-MPRS-DCH + V4-TA-DC NER`；
2. 生成完整 source lock 和 architecture manifest；
3. 在当前固定 seed-42 范围内不立即叠加多个模块；
4. 下一模块只能在 Target Survival Supervision 与 Query-only FG 中选择一个做单变量实验；
5. 不同时修改 tokenizer、NER、loss 和 SCTB；
6. 当前结论仍只限 NUDT-SIRST 内部 `530/133` 验证集。

V4 通过只说明：

> 在 seed 42、当前内部验证划分下，五节点 NER 的获选 DC 作用域通过了
> Pd、Fa、mIoU、五预算和 paired-control 六组件门槛。

Tiny-Pd 若仍为 39/39 只表示没有回退，不能声称 tiny-target 得到提升。V4
通过也不会自动证明论文核心、跨数据集稳定性或通用性。

---

## 16. 建议项目状态

```text
decision=V4_COMPLEMENT_TAIL_SELECTED_FORMAL800_RUNNING

baseline_v1_v2_v3_comparison_accepted=true
v3_authoritative_result_accepted=true
v3_verdict=RETURN_TO_MODEL_OPTIMIZATION
v3_immutable=true

ner_dc_offset_knockout_complete=true
stage4_role=strong_same_checkpoint_positive_seed42
stage3_role=weak_role_dependent
stage2_role=mixed

mainline_changed=false
v8_tokenizer_changed=false
ner_topology_changed=false
loss_changed=false
evaluator_changed=false

v4_scope_candidates=legacy_global,direct_tail,complement_tail
v4_three_scope_counterfactual_complete=true
v4_formula_decision=COMPLEMENT_TAIL_SELECTED
v4_selected_formula_mode=complement_tail
v4_selected_formula_stage3_2=d*(1-P)
v4_tail_aware_formula_selected=true
v4_three_scope_production_implementation_ready=true
v4_production_implementation_ready=true
v4_targeted_engineering_tests_passed=true
v4_full_repo_tests_passed=false
v4_gpu_smoke_passed=true
v4_formal_batch16_gpu_smoke_passed=true
v4_exact_resume_contract_passed=true
v4_live_restart_replay_exercised=true
v4_live_restart_resume_boundary=144_to_145
v4_source_lock_passed=true
v4_source_lock_sha256=90dd24dfeef2d46c820fb5c89a899cec1961a7e718053f16395e256b3c27ccf3
v4_formal_training_authorized=true
v4_formal_training_running=true
v4_formal_training_physical_gpu=2
v4_first_exact_epochs_committed=true
v4_formal800_complete=false

next_model_stage_authorized=false
paper_core_established=false
stability_claim_supported=false
```

正式启动条件：

```text
syntax_and_unit_tests_pass
&& strict_v3_state_compatibility_pass
&& stage4_exact_equivalence_pass
&& legacy_global_exact_equivalence_pass
&& zero_output_identity_pass
&& three_scope_counterfactual_complete
&& one_local_scope_candidate_selected
&& cpu_smoke_pass
&& rtx5090_smoke_pass
&& exact_resume_pass
&& source_lock_pass
```

---

## 17. 执行顺序

```text
已完成：封存 V3 和 8 行 NER DC-offset knockout
→ 已完成：生产实现支持 legacy / direct / complement，不修改 V3
→ 已完成：数学/状态/stage4/legacy 等价单元测试
→ 已完成：固定 thresholds = 1.5 / 2.0 / 2.5
→ 已完成：seed42 V3 best / best_miou 三作用域零训练 counterfactual
→ 已完成：按预注册资格门唯一选择 complement_tail
→ 已完成：获选公式已是生产 V4 默认且 manifest 明确
→ 已完成：实现并验证 V4 exact trainer 与同版本 exact-resume 守卫
→ 已完成：固定 39 个运行源码与正式 source lock
→ 已完成：CPU / RTX 5090 两步 smoke 与正式 batch-16 压力检查
→ 进行中：seed42 V4 单个 fresh 800-epoch run（物理 GPU 2）
→ 固定阈值 + closed sweep
→ tail / component / gradient mechanism audit
→ 使用各模型自己的 best / best_miou 按六组件原门槛裁决
→ 全部门槛通过后才进入下一模型阶段
```

---

## 18. 结论

V3 的正式证据没有否定五节点拓扑，也不能归结为 tokenizer MPRS-DCH
失效。seed-42 的 NER DC-offset knockout 表明：stage4 的同权重全局负校准
作用最强且跨两个 checkpoint role 方向一致；stage3 较弱且 role-dependent；
stage2 呈 Fa–mIoU 混合权衡。它提示 stage3/2 的全图作用域值得检查，但没有
单独证明全图广播就是失败原因。

三作用域最小变量比较已经完成：

> stage4 保留 NER global DC；stage3/2 在 legacy global、direct-tail
> `d·P` 与 complement-tail `d·(1-P)` 中唯一选择 complement-tail。

三个公式保持 V3 参数和 state 兼容。由于现有 `d2/d3` 均为负，direct 和
protect 具有相反作用方向，不能预先宣布 direct 最优。父子 relay 也不是独立
证据，因此 \(P_s\) 只称为持续上尾响应。

complement 已通过公式资格门并进入 seed 42 fresh 800-epoch 正式训练；
当前训练已启动但尚未完成，因此仍不是正式性能通过。正式性能仍由
Pd、Fa、mIoU、五预算和 V1/V2 paired
comparison 的六组件门槛裁决。

---

## 本地权威代码、结果与协议

1. `model/tpd_clean_v8_mprs_dch.py`：V8-MPRS-DCH tokenizer。
2. `model/tpd_ner_v8_mprs_dch.py`：五节点与 `q4→q3→q2`。
3. `model/tpd_ner_v8_mprs_dch_v2.py`：RMS、空间中心化、atan gate。
4. `model/tpd_ner_v8_mprs_dch_v3.py`：NER post-centering DC offset。
5. `model/tpd_sctransnet.py`：CCA skip 的乘法调制位置。
6. `model/tpd_ner_v8_mprs_dch_v4_tail_aware.py`：三作用域 V4 正式实现。
7. `experiments/evaluate_tpd_ner_v8_mprs_dch_v4_tail_formula_counterfactual.py`：GPU2/3 零训练对照。
8. `experiments/results/tpd_ner_v8_mprs_dch_v4_tail_formula_counterfactual_v1/NUDT-SIRST/tpd_ner_v8_mprs_dch_v4_tail_formula_counterfactual_aggregate.json`：公式裁决。
9. `experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_exact.py`：唯一公式、seed 42 的正式 exact trainer。
10. `experiments/tpd_ner_v8_mprs_dch_v4_tail_aware_exact_source_lock.json`：39 个运行源码与训练数据绑定。
11. `experiments/run_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_1x5090_lane.sh`：固定 GPU UUID、fresh / exact-resume lane。
12. `experiments/launch_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_1x5090.sh`：正式后台启动、预检与状态入口。
13. `experiments/results/tpd_ner_v8_mprs_dch_v3_exact_v1/NUDT-SIRST/comparison_selection_contract_repair_v1/tpd_ner_v8_mprs_dch_v3_formal800_comparison_selection_contract_repair_v1.json`。
14. `experiments/results/tpd_ner_v8_mprs_dch_v3_dc_knockout_v2/NUDT-SIRST/comparison_aggregate_field_repair_v1/tpd_ner_v8_mprs_dch_v3_dc_knockout_comparison_aggregate_field_repair_v1.json`。
