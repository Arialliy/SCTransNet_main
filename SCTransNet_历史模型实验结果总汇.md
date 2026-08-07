# SCTransNet 历史模型实验结果总汇

更新时间：2026-08-05

## 1. 汇总结论

本文件汇总本仓库截至当前已经产生正式性能结果的模型，并将结果按实验协议分开报告。
核心结论如下。

1. 初代 TPD 在 NUDT-SIRST 内部验证集上明确优于 Original 和 Progressive 的综合
   Pd–Fa–mIoU 工作点，但没有全面超过最近结构对照 SPD。
2. TPD-Clean V6、V7-DCH 的局部结果为正，但跨 seed 表现和等容量对照结果不稳定，
   两版均未成为最终模型。
3. 五节点 NER V1–V3 未达到综合性能要求；NER V4 Tail-Aware 首次获得明确的相对
   性能改善，并成为后续完整模型的 NER 版本。
4. 原始 TSS 辅助项没有形成跨数据集统一正收益；QFG-only 被联合比较覆盖；NUDT
   内部 formal800 阶段曾保留 TSS+QFG 完整模型，是因为它提供了有效的 Pd–Fa Pareto
   工作区间，不代表它已成为跨数据集统一配方。
5. 四数据集 1000-epoch 结果显示完整模型具有数据集相关的混合收益：SIRST3、NUAA、
   NUDT 和 IRSTD-1K 上都有局部改善，但不存在“所有数据集、两个 checkpoint 角色、
   所有指标均超过 Original”的结论。
6. 三数据集 TSS 强度诊断表明固定正权重与 TSS-off 都存在明显退化项，尚未找到可作为
   三数据集统一默认值的旧 TSS 配方。
7. EC-TSS V3.1 的 seed 42、三数据集、1000-epoch 训练与六个 checkpoint 复评已经
   完成。它在 6 个 dataset-role 单元中贡献 4 个独有非支配点，但严重退化门、旧强项
   保留门和成对多数门均未通过，因此不能成为统一 TSS 配方。
8. NER→QFG→TPD 的三数据集固定权重组件诊断已闭环：NER stage 2 启动门
   未通过，QFG 和 TPD 均为“存在功能影响但没有跨数据集实质改善”。因此当前
   保持 `TPD8 + NER4 + QFG2`、`TSS OFF`，不启动 NER V5、QFG/TPD 公式修改或新的重训。

当前总裁决仍为：

```text
decision = INCONCLUSIVE_MIXED_TRADEOFF
ec_tss_v3_1_decision = EC_TSS_V3_1_PERFORMANCE_FAIL_STOP_TSS_OPTIMIZATION
ner_stage2_decision = DO_NOT_AUTHORIZE_NER_V5_PER_DEVELOPMENT_TRAINING
qfg_decision = QFG_INCONCLUSIVE_NO_FORMULA_CHANGE
tpd_decision = TPD_INCONCLUSIVE_NO_FORMULA_CHANGE
paper_core_established = false
stability_claim_supported = false
```

这不表示模型设计失败。它表示完整模型已经证明具有竞争力，但统一配方和跨数据集稳定
收益仍需继续优化。

## 2. 读表口径

### 2.1 两类正式实验不能直接横向排名

| 实验族 | 数据与划分 | Epoch | Seed | checkpoint 选择 | 用途 |
|---|---|---:|---|---|---|
| NUDT 内部筛选 | NUDT-SIRST 官方训练集内部 530/133 划分 | 800 | 主要为 42；V6/V7 及 TSS-on 补充轨迹另含 3407 | 内部验证集 | 模块与架构筛选 |
| 多数据集实验 | 各数据集自己的 `img_idx/train`、`img_idx/test` | 1000 | 42 | test split 每 10 epoch 评估并选模 | 数据集内性能记录 |

多数据集 checkpoint 是 `test_selected=true`、`selection_is_optimistic=true`。因此它们是
当前固定协议下的数据集内比较结果，不是独立测试或跨随机性结论。

### 2.2 指标定义

- `Pd`：固定阈值二值图中，按 8 连通组件和质心距离 `<3` 做一对一匹配后的
  `匹配 GT 目标数/GT 目标总数`。存在目标计数时统一写为
  `检出目标数/目标总数（Pd 小数）`，例如 `188/189（0.994709）`。
- `Fa`：未与 GT 目标匹配的预测组件所含像素数除以有效像素总数，越低越好。它是
  历史 SCTransNet 的 component-Fa，不等于所有背景误预测像素率；已匹配组件向背景
  外溢的像素不会计入该 Fa。
- `pixel precision / recall / F1`：逐像素 TP/FP/FN 口径，用于约束 Pd 和 component-Fa
  无法反映的轮廓外溢与大连通域问题；最终判断必须和 Pd、Fa、mIoU、nIoU 联合查看。
- `mIoU`：全局累计交并比，越高越好。
- `nIoU`：逐图 IoU 的平均值，越高越好。
- `tiny-Pd`：tiny target 的目标级检出率或匹配计数。
- `best_pd`：按该次实验冻结的 Pd 优先规则选择的高召回 checkpoint，不代表综合性能最优。
- `best_miou`：按 mIoU 优先规则选择的 checkpoint。
- 所有本文件固定点结果均使用阈值 `0.5`；Pd–Fa budget 扫描是另一套工作点分析，
  不能替代固定阈值表。

## 3. 模型演进关系

| 阶段 | 模型 | 相对前一阶段的主要变化 | 最终状态 |
|---|---|---|---|
| Baseline | Original SCTransNet | 原始 patch embedding | 对照基线 |
| TPD 对照 | Progressive | 多次 stride-2 替代一次大步长 | 被 TPD 超过 |
| TPD 对照 | SPD | `pixel_unshuffle + dense 1×1` | 严格低 Fa/mIoU 强对照 |
| TPD V1 | Keep–Context–Saliency TPD | 三源目标保真下采样 | 有效但与 SPD 权衡 |
| TPD-Clean V6 | 相位绑定 K/C/S + Context headroom | 清理归因并做等容量对照 | 未通过综合要求 |
| V7-DCH | Deferred Context Headroom | 加强零点及一阶优化锚点 | 未通过综合要求 |
| V8-MPRS-DCH | 四相位分辨 Saliency | 保持 K/C/S 主线，不增加第四语义分支 | 作为 NER 底座 |
| NER V1–V3 | 五 evidence nodes + 窄中继 | 逐步压低 Fa、调整 relay/DC offset | V1–V3 未定型 |
| NER V4 | Tail-Aware complement-tail | 调整 stage-wise DC offset 空间作用域 | 相对改善确认 |
| TSS | 两个训练期 survival heads | stride-16 目标存在性辅助监督 | 单独效果混合 |
| QFG V2-CROA | Query-only frequency gate | 只调制 NER query | QFG-only 被覆盖 |
| Final D | TPD V8 + NER V4 + TSS + QFG | 完整训练模型；部署移除 TSS heads | 当前完整模型 |
| EC-TSS V3.1 | 错误条件化 TSS | 将正/负区域风险分别归一化 | formal1000 完成；混合权衡，未成为统一配方 |

TPD-Clean V4/V5 仅有 preflight，V8-MPRS-DCH 也没有独立于 NER 的正式 800-epoch 性能行；
它们不被伪装成已有性能结果。

## 4. 初代 TPD：NUDT 内部 formal800

实验条件：seed 42、530/133 内部划分、固定阈值 0.5。

### 4.1 best-Pd 角色

| 模型 | Epoch | Pd | tiny-Pd | Fa | 错误目标/图 | mIoU | nIoU | 参数量 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 456 | 188/189（0.994709） | 39/39（1.000000） | 1.4226e-5 | 0.1278 | 0.919086 | 0.923178 | 11,325,939 |
| Progressive | 300 | 187/189（0.989418） | 39/39（1.000000） | 1.0325e-6 | 0.0451 | 0.914348 | 0.914485 | 10,924,755 |
| SPD | 619 | 187/189（0.989418） | 39/39（1.000000） | 0 | 0 | **0.946542** | **0.939837** | 10,842,835 |
| TPD | 337 | 188/189（0.994709） | 39/39（1.000000） | 1.0325e-6 | 0.0226 | 0.933647 | 0.930339 | 10,827,731 |

TPD 相比 Original 保持 188/189（0.994709）的 Pd，将 Fa 降低约 13.8 倍，并将 mIoU 提升
0.014561。相对 SPD，TPD 多检出 1 个目标，但 SPD 的 Fa 为 0 且 mIoU 更高。

### 4.2 best-mIoU 角色

| 模型 | Epoch | mIoU | Pd | Fa |
|---|---:|---:|---:|---:|
| Original | 726 | 0.940738 | 186/189（0.984127） | 1.9504e-6 |
| Progressive | 611 | 0.931854 | 186/189（0.984127） | 2.1798e-6 |
| SPD | 470 | **0.949145** | **187/189（0.989418）** | 4.5891e-7 |
| TPD | 457 | 0.942758 | 186/189（0.984127） | 4.5891e-7 |

五个 Fa budget 上，TPD 全部优于 Original 和 Progressive；相对 SPD，TPD 在后四个
预算点更有利，但在最严格 `Fa≤1e-6` 工作区间由 SPD 更优。因此该阶段裁决为
`INCONCLUSIVE_MIXED_TRADEOFF`。

## 5. TPD-Clean V6 与 V7-DCH：NUDT 内部 formal800

### 5.1 TPD-Clean V6

| Seed | 变体 | checkpoint | Epoch | Pd | Fa | mIoU |
|---:|---|---|---:|---:|---:|---:|
| 42 | V6 Full | best-Pd | 419 | 188/189（0.994709） | 1.4915e-6 | 0.922945 |
| 42 | V6 Full | best-mIoU | 535 | 187/189（0.989418） | 1.7209e-6 | 0.940544 |
| 42 | Capacity | best-Pd | 181 | 188/189（0.994709） | 6.8722e-5 | 0.805532 |
| 42 | Capacity | best-mIoU | 416 | 186/189（0.984127） | 5.7364e-7 | 0.939605 |
| 3407 | V6 Full | best-Pd | 427 | 187/189（0.989418） | 4.8415e-5 | 0.860967 |
| 3407 | V6 Full | best-mIoU | 570 | 185/189（0.978836） | 1.0325e-6 | 0.924459 |
| 3407 | Capacity | best-Pd | 522 | 186/189（0.984127） | 1.0325e-6 | 0.928052 |
| 3407 | Capacity | best-mIoU | 518 | 184/189（0.973545） | 6.8837e-7 | 0.929850 |

V6 的 seed 42 有局部正收益，但 seed 3407 明显不稳定，且不能持续超过等容量对照。
裁决：`ENGINEERING_GATE_FAIL`。

### 5.2 V7-DCH

| Seed | 变体 | checkpoint | Epoch | Pd | tiny-Pd | Fa | mIoU |
|---:|---|---|---:|---:|---:|---:|---:|
| 42 | V7-DCH Full | best-Pd | 414 | 187/189（0.989418） | 39/39（1.000000） | 9.1782e-7 | 0.929930 |
| 42 | V7-DCH Full | best-mIoU | 694 | 187/189（0.989418） | 39/39（1.000000） | 2.6387e-6 | 0.939580 |
| 42 | Capacity | best-Pd | 181 | 188/189（0.994709） | 39/39（1.000000） | 6.8722e-5 | 0.805532 |
| 42 | Capacity | best-mIoU | 416 | 186/189（0.984127） | 38/39（0.974359） | 5.7364e-7 | 0.939605 |
| 3407 | V7-DCH Full | best-Pd | 212 | 187/189（0.989418） | 39/39（1.000000） | 1.5259e-5 | 0.849535 |
| 3407 | V7-DCH Full | best-mIoU | 557 | 183/189（0.968254） | 38/39（0.974359） | 3.3271e-6 | 0.923745 |
| 3407 | Capacity | best-Pd | 522 | 186/189（0.984127） | 39/39（1.000000） | 1.0325e-6 | 0.928052 |
| 3407 | Capacity | best-mIoU | 518 | 184/189（0.973545） | 38/39（0.974359） | 6.8837e-7 | 0.929850 |

V7 改善了设计锚点，但实际性能仍存在 seed 3407 退化和等容量对照不占优问题。
裁决：`ENGINEERING_GATE_FAIL`。

## 6. V8-MPRS-DCH + 五节点 NER：NUDT 内部 formal800

### 6.1 NER V1–V4 的 best-Pd 角色

| 模型 | Epoch | Pd | Fa | mIoU | nIoU | tiny-Pd | 错误目标/图 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original closed-interval 重评 | 456 | 188/189（0.994709） | 1.4226e-5 | 0.919382 | 0.923449 | 39/39（1.000000） | 0.1278 |
| NER V1 relay-off | 289 | **189/189（1.000000）** | 4.8989e-5 | 0.790668 | 0.788853 | 39/39（1.000000） | 0.4887 |
| NER V1 relay-on | 461 | 188/189（0.994709） | 1.8815e-5 | 0.909049 | 0.913220 | 39/39（1.000000） | 0.2782 |
| NER V2 relay-on | 295 | 188/189（0.994709） | 5.6217e-6 | 0.915323 | 0.915798 | 39/39（1.000000） | 0.0752 |
| NER V3 relay-on | 253 | 188/189（0.994709） | **4.7038e-6** | 0.903948 | 0.904935 | 39/39（1.000000） | **0.0451** |
| NER V4 Tail-Aware | 422 | **189/189（1.000000）** | 7.5720e-6 | **0.926418** | **0.928662** | 39/39（1.000000） | 0.1053 |

这里的 Original 行是 NER 联合比较阶段对历史 checkpoint 的当前 closed-interval 重评；
第 4 节的 `0.919086/0.923178` 是原训练 `summary.json` 存储值。两者来自同一
checkpoint，但评估实现版本不同，不能把 `0.000296` 的差异写成模型变化。后文完整
模型相对 Original 的 `mIoU +0.01764` 采用本节 closed-interval 重评口径。

### 6.2 NER 各版 best-mIoU 角色

| 模型 | Epoch | Pd | Fa | mIoU | nIoU | tiny-Pd | 错误目标/图 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NER V1 relay-off | 577 | 187/189（0.989418） | **6.8837e-7** | **0.946618** | **0.942128** | 39/39（1.000000） | **0.0301** |
| NER V1 relay-on | 513 | 187/189（0.989418） | 2.7535e-6 | 0.938779 | 0.936131 | 39/39（1.000000） | 0.0677 |
| NER V2 relay-on | 454 | 185/189（0.978836） | 4.9333e-6 | 0.939389 | 0.934285 | 38/39（0.974359） | **0.0301** |
| NER V3 relay-on | 489 | 187/189（0.989418） | 5.0480e-6 | 0.935640 | 0.936383 | 39/39（1.000000） | 0.0526 |
| NER V4 Tail-Aware | 489 | **188/189（0.994709）** | 4.2449e-6 | 0.938178 | 0.937557 | 39/39（1.000000） | **0.0301** |

V1–V3 的正式裁决均为 `RETURN_TO_MODEL_OPTIMIZATION`。V4 同时恢复了
best-Pd 的 189/189（1.000000），并在 best-mIoU 保持 188/189（0.994709）、
4 个错误目标；因此裁决升级为
`RELATIVE_MODEL_IMPROVEMENT_CONFIRMED_WITH_TRADEOFF`，并成为后续完整模型底座。

## 7. TSS、QFG 与最终完整模型：NUDT 内部 formal800

### 7.1 TSS 因子结果

| 变体/checkpoint | Epoch | Pd | Fa | mIoU | nIoU | tiny-Pd | 错误目标 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TSS-on best-Pd/best-mIoU | 3 | 188/189（0.994709） | 4.1302e-6 | 0.936870 | 0.936312 | 39/39（1.000000） | 5 |
| TSS-control best-Pd | 37 | 188/189（0.994709） | **4.0155e-6** | 0.934370 | 0.934263 | 39/39（1.000000） | **4** |
| TSS-control best-mIoU | 3 | 188/189（0.994709） | 4.8186e-6 | **0.940091** | 0.940000 | 39/39（1.000000） | 6 |

旧 TSS-on 没有统一优于 control：它改善了 control best-Pd 的 mIoU，但 Fa 略高；
control best-mIoU 的 mIoU 又更高。

### 7.2 QFG 与 TSS+QFG

| 变体/checkpoint | Epoch | Pd | Fa | mIoU | nIoU | tiny-Pd | 错误目标 |
|---|---:|---:|---:|---:|---:|---:|---:|
| QFG-only best-Pd | 29 | 188/189（0.994709） | 4.0155e-6 | 0.930844 | 0.931702 | 39/39（1.000000） | 4 |
| QFG-only best-mIoU | 3 | 188/189（0.994709） | 4.8186e-6 | **0.939934** | 0.939891 | 39/39（1.000000） | 6 |
| TSS+QFG best-Pd | 136 | 188/189（0.994709） | **3.6713e-6** | 0.931693 | 0.931583 | 39/39（1.000000） | 6 |
| TSS+QFG best-mIoU | 3 | 188/189（0.994709） | 4.1302e-6 | 0.937018 | 0.936369 | 39/39（1.000000） | 5 |

QFG-only 的联合比较结果为 `DOMINATED`；TSS+QFG 为
`PARETO_MIXED_TRADEOFF`，工程选择为 `SELECT_D_TSS_QFG`。

### 7.3 固定 seed42 工程认证

部署点是完整模型 D 的 `best_miou`、epoch 3、阈值 0.5：

| Pd | Fa | mIoU | nIoU | tiny-Pd | 错误目标/图 |
|---:|---:|---:|---:|---:|---:|
| 188/189（0.994709） | 4.1302e-6 | 0.937018 | 0.936369 | 39/39（1.000000） | 0.0376 |

相对 Original 的 best-Pd 固定点，完整模型保持 Pd 和 tiny-Pd，将 Fa 约降低 3.44 倍、
mIoU 提升约 0.01764、错误目标从 17 降至 5。相对紧邻模型 TSS-on，QFG 仅将 mIoU
提高约 0.000148，不能表述为显著或统一优势。

补充 seed 3407 的 TSS-on 复现实验结果如下；它不参与固定 seed42 主判定：

| checkpoint | Epoch | Pd | Fa | mIoU | nIoU |
|---|---:|---:|---:|---:|---:|
| best-Pd | 101 | 188/189（0.994709） | 2.9829e-6 | 0.931908 | 0.930085 |
| best-mIoU | 1 | 188/189（0.994709） | 4.7038e-6 | 0.937697 | 0.936522 |

## 8. 四数据集 Original 与 Final：formal1000

这里的 `Final` 指 `TPD V8-MPRS-DCH + NER V4 Tail-Aware + TSS + QFG-V2-CROA`。
各数据集独立训练，seed 42，每 10 epoch 评估，使用各自 `img_idx`。

### 8.1 best-mIoU checkpoint

| 数据集 | 模型 | Epoch | mIoU | nIoU | Pd | Fa | tiny-Pd | 错误目标/图 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SIRST3 | Original | 580 | 0.827791 | 0.861934 | 0.967442 | 9.4869e-6 | 0.950617 | 0.078777 |
| SIRST3 | Final | 600 | **0.832626** | **0.864422** | 0.967442 | **8.8641e-6** | **0.956790** | 0.080630 |
| NUAA-SIRST | Original | 830 | 0.786825 | 0.795096 | **0.969582** | 2.6549e-5 | **0.914286** | 0.126168 |
| NUAA-SIRST | Final | 720 | **0.796547** | **0.799259** | 0.961977 | **1.6670e-5** | 0.857143 | **0.102804** |
| NUDT-SIRST | Original | 520 | **0.945607** | 0.947437 | 0.989418 | **2.5048e-6** | 0.996139 | **0.034639** |
| NUDT-SIRST | Final | 410 | 0.944498 | **0.948648** | **0.990476** | 4.3892e-6 | 0.996139 | 0.042169 |
| IRSTD-1K | Original | 270 | **0.673543** | 0.636875 | **0.949495** | 2.2110e-5 | 0.766667 | 0.407960 |
| IRSTD-1K | Final | 470 | 0.669581 | **0.660009** | 0.936027 | **2.1977e-5** | 0.766667 | **0.323383** |

### 8.2 best-Pd checkpoint

| 数据集 | 模型 | Epoch | mIoU | nIoU | Pd | Fa | tiny-Pd | 错误目标/图 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SIRST3 | Original | 290 | **0.811000** | **0.852927** | **0.978738** | **1.9741e-5** | **0.969136** | **0.141798** |
| SIRST3 | Final | 270 | 0.800218 | 0.848096 | 0.978073 | 2.7053e-5 | 0.966049 | 0.145505 |
| NUAA-SIRST | Original | 440 | 0.726236 | 0.748163 | **0.988593** | 8.1018e-5 | **0.971429** | 0.500000 |
| NUAA-SIRST | Final | 590 | **0.786317** | **0.790728** | 0.980989 | **2.0306e-5** | 0.885714 | **0.149533** |
| NUDT-SIRST | Original | 260 | 0.915686 | 0.925523 | **0.995767** | 1.3811e-5 | 0.996139 | 0.085843 |
| NUDT-SIRST | Final | 320 | **0.933907** | **0.936627** | 0.993651 | **5.1246e-6** | 0.996139 | **0.051205** |
| IRSTD-1K | Original | 230 | **0.619141** | 0.627174 | 0.966330 | **4.9193e-5** | 0.800000 | 0.701493 |
| IRSTD-1K | Final | 320 | 0.612761 | **0.632386** | 0.966330 | 5.5512e-5 | **0.833333** | **0.631841** |

### 8.3 四数据集结果解读

- SIRST3 best-mIoU：Final 的 mIoU、nIoU、Fa 和 tiny-Pd 改善，Pd 持平；但
  best-Pd 角色由 Original 更优。
- NUAA best-mIoU：Final 提高 mIoU/nIoU并降低 Fa，但 Pd/tiny-Pd 回退；best-Pd
  角色则显著降低 Fa并提高区域质量，Pd 仍低于 Original。
- NUDT best-mIoU：Final 的 Pd/nIoU略高，但 Original 的 mIoU/Fa更好；best-Pd
  角色 Final 以轻微 Pd 回退换取更低 Fa和更高区域质量。
- IRSTD-1K：Final 改善 nIoU、Fa或错误目标数中的部分指标，但 mIoU/Pd存在回退。

所以完整模型是“数据集相关的正向竞争力”，而不是统一全面超过 Original。

## 9. 三数据集旧 TSS 强度诊断：formal1000

范围已收敛为 NUAA-SIRST、NUDT-SIRST、IRSTD-1K；SIRST3 不参与统一配方选择。
以下全部为 seed 42、固定阈值 0.5、test-selected 结果。

单个 run 的训练与评估协议一致，但配方搜索预算不同：Original 只有 3 个 run，
Final-family 包含三个正 λ 加 TSS-off，共 12 个 run，搜索预算比为 `4.0`。TSS-off
也是在读取正 λ 的 test-selected 结果后新增的诊断。因此本节适合比较已观察工作点，
不能写成 Original 与整个 Final 配方族具有相同搜索预算。

### 9.1 NUAA-SIRST

| checkpoint | 配方 | Epoch | mIoU | nIoU | Pd | Fa | tiny-Pd |
|---|---|---:|---:|---:|---:|---:|---:|
| best-mIoU | Original | 830 | 0.786825 | 0.795096 | 255/263（0.969582） | 2.6549e-5 | 32/35（0.914286） |
| best-mIoU | Final, TSS λ=0.0025 | 770 | 0.793272 | 0.785947 | 252/263（0.958175） | 1.7013e-5 | 30/35（0.857143） |
| best-mIoU | Final, TSS λ=0.005 | 680 | 0.788944 | 0.792508 | 251/263（0.954373） | 1.7150e-5 | 29/35（0.828571） |
| best-mIoU | Final, TSS λ=0.01 | 680 | **0.797386** | 0.791021 | 253/263（0.961977） | **8.3693e-6** | 30/35（0.857143） |
| best-mIoU | Final, TSS-off | 850 | 0.796483 | **0.795349** | **256/263（0.973384）** | 1.5435e-5 | 30/35（0.857143） |
| best-Pd | Original | 440 | 0.726236 | 0.748163 | **260/263（0.988593）** | 8.1018e-5 | **34/35（0.971429）** |
| best-Pd | Final, TSS λ=0.0025 | 230 | 0.746279 | 0.764466 | 257/263（0.977186） | 5.7007e-5 | 33/35（0.942857） |
| best-Pd | Final, TSS λ=0.005 | 550 | 0.778930 | 0.792287 | 258/263（0.980989） | 3.2311e-5 | 31/35（0.885714） |
| best-Pd | Final, TSS λ=0.01 | 330 | 0.784420 | 0.787630 | 259/263（0.984791） | 2.4353e-5 | 33/35（0.942857） |
| best-Pd | Final, TSS-off | 820 | **0.788553** | **0.792668** | 257/263（0.977186） | **1.4749e-5** | 30/35（0.857143） |

### 9.2 NUDT-SIRST

| checkpoint | 配方 | Epoch | mIoU | nIoU | Pd | Fa | tiny-Pd |
|---|---|---:|---:|---:|---:|---:|---:|
| best-mIoU | Original | 520 | 0.945607 | 0.947437 | 935/945（0.989418） | **2.5048e-6** | 258/259（0.996139） |
| best-mIoU | Final, TSS λ=0.0025 | 610 | **0.946686** | **0.949784** | 938/945（0.992593） | 4.2743e-6 | 258/259（0.996139） |
| best-mIoU | Final, TSS λ=0.005 | 500 | 0.944538 | 0.947775 | **939/945（0.993651）** | 5.6991e-6 | 258/259（0.996139） |
| best-mIoU | Final, TSS λ=0.01 | 590 | 0.942368 | 0.947814 | 932/945（0.986243） | 4.5041e-6 | 257/259（0.992278） |
| best-mIoU | Final, TSS-off | 420 | 0.944406 | 0.946423 | 936/945（0.990476） | 2.7806e-6 | 258/259（0.996139） |
| best-Pd | Original | 260 | 0.915686 | 0.925523 | **941/945（0.995767）** | 1.3811e-5 | 258/259（0.996139） |
| best-Pd | Final, TSS λ=0.0025 | 630 | **0.942560** | **0.946929** | 939/945（0.993651） | **5.0786e-6** | 258/259（0.996139） |
| best-Pd | Final, TSS λ=0.005 | 540 | 0.941290 | 0.945203 | 940/945（0.994709） | 5.6761e-6 | 258/259（0.996139） |
| best-Pd | Final, TSS λ=0.01 | 300 | 0.926336 | 0.933891 | 938/945（0.992593） | 8.4567e-6 | 258/259（0.996139） |
| best-Pd | Final, TSS-off | 510 | 0.937381 | 0.939836 | 940/945（0.994709） | 6.6642e-6 | 258/259（0.996139） |

### 9.3 IRSTD-1K

| checkpoint | 配方 | Epoch | mIoU | nIoU | Pd | Fa | tiny-Pd |
|---|---|---:|---:|---:|---:|---:|---:|
| best-mIoU | Original | 270 | 0.673543 | 0.636875 | 282/297（0.949495） | 2.2110e-5 | 23/30（0.766667） |
| best-mIoU | Final, TSS λ=0.0025 | 280 | **0.686740** | 0.653590 | 270/297（0.909091） | 3.5243e-5 | 22/30（0.733333） |
| best-mIoU | Final, TSS λ=0.005 | 180 | 0.670987 | 0.650185 | **283/297（0.952862）** | 2.9796e-5 | 22/30（0.733333） |
| best-mIoU | Final, TSS λ=0.01 | 180 | 0.676715 | 0.630574 | 277/297（0.932660） | 3.1485e-5 | 22/30（0.733333） |
| best-mIoU | Final, TSS-off | 830 | 0.660312 | **0.665662** | 277/297（0.932660） | **1.1729e-5** | **23/30（0.766667）** |
| best-Pd | Original | 230 | 0.619141 | 0.627174 | 287/297（0.966330） | 4.9193e-5 | 24/30（0.800000） |
| best-Pd | Final, TSS λ=0.0025 | 320 | 0.613793 | 0.632749 | 287/297（0.966330） | 5.9346e-5 | 25/30（0.833333） |
| best-Pd | Final, TSS λ=0.005 | 370 | **0.658486** | **0.658560** | **288/297（0.969697）** | 2.8145e-5 | 25/30（0.833333） |
| best-Pd | Final, TSS λ=0.01 | 530 | 0.627583 | 0.641612 | 287/297（0.966330） | 2.8696e-5 | **26/30（0.866667）** |
| best-Pd | Final, TSS-off | 530 | 0.639986 | 0.650812 | 287/297（0.966330） | **2.3249e-5** | 25/30（0.833333） |

### 9.4 统一配方裁决

按冻结的严重退化计数：

| 配方 | 严重退化项数 |
|---|---:|
| TSS λ=0.0025 | 8 |
| TSS λ=0.005 | 5 |
| TSS λ=0.01 | 8 |
| TSS-off | 5 |

TSS-off 相对 λ=0.005 的逐项比较为 13 项更好、13 项更差、4 项相同，不能形成统一
优势；最终诊断为：

```text
decision = TSS_OFF_NOT_GLOBALLY_ADMISSIBLE_SEED42_TEST_SELECTED
off_gate_eligible = false
tss_default_enabled = null
global_tss_lambda = null
```

## 10. EC-TSS V3.1：三数据集 formal1000 最终结果

EC-TSS V3.1 针对旧 TSS 的两个实际问题进行修改：背景容易产生大面积辅助损失，且
目标/背景风险被同一归一化掩盖。新目标分别构造目标遗漏风险与背景虚警风险，再分别
归一化，并限制加权辅助损失相对分割损失的比例。它只改变训练目标；正式评估时已移除
TSS heads，TPD8–NER4–QFG2 推理结构、参数量和推理路径均未改变。

实验条件：各数据集自己的 `img_idx/train`、`img_idx/test`，seed 42，1000 epochs，
每 10 epochs 测试并按冻结规则保存 `best_miou` 和 `best_pd`，表中均为阈值 0.5。
单个 run 的训练与评估协议一致，但累计搜索预算并不相等：Original 为 3 个 run，
Final-family 的正 λ、TSS-off 与 EC-TSS 合计 15 个 run，即 `5:1`。因此以下结论是
已观察固定点的比较，不能写成整个配方族与 Original 具有相同搜索预算。

### 10.1 六个正式 checkpoint

| 数据集 | checkpoint | Epoch | mIoU | nIoU | Pd | component-Fa | pixel precision | pixel F1 | 错误目标/图 | tiny-Pd |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | best-mIoU | 980 | 0.789707 | 0.794494 | 253/263（0.961977） | 1.7150e-5 | 0.892670 | 0.882498 | 0.107477 | 31/35（0.885714） |
| NUAA-SIRST | best-Pd | 510 | 0.780742 | 0.787516 | 256/263（0.973384） | 2.0855e-5 | 0.877973 | 0.876873 | 0.112150 | 30/35（0.857143） |
| NUDT-SIRST | best-mIoU | 410 | 0.944564 | 0.947240 | 937/945（0.991534） | 2.8266e-6 | 0.979708 | 0.971492 | 0.037651 | 257/259（0.992278） |
| NUDT-SIRST | best-Pd | 460 | 0.942968 | 0.949620 | 939/945（0.993651） | 7.6064e-6 | 0.967903 | 0.970647 | 0.045181 | 258/259（0.996139） |
| IRSTD-1K | best-mIoU | 830 | 0.668879 | 0.675449 | 281/297（0.946128） | 1.2981e-5 | 0.844414 | 0.801590 | 0.268657 | 24/30（0.800000） |
| IRSTD-1K | best-Pd | 320 | 0.618942 | 0.632799 | 285/297（0.959596） | 4.9553e-5 | 0.729682 | 0.764625 | 0.592040 | 24/30（0.800000） |

### 10.2 component-Fa 与真实像素 FP 的区别

使用冻结 `img_idx/test` mask 在 CPU 上恢复聚合 TP/FP/FN/TN；36/36 个历史与 EC 固定点
均通过 pixel precision、recall、F1、mIoU 恒等式校验，未重新推理、未重选 checkpoint。
EC 六点如下：

| 数据集 | checkpoint | component-Fa | pixel FP/全部有效像素 | 倍数 | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | best-mIoU | 1.7150e-5 | 6.0369e-5 | 3.52× | 7319 | 880 | 1069 | 14567810 |
| NUAA-SIRST | best-Pd | 2.0855e-5 | 7.0041e-5 | 3.36× | 7346 | 1021 | 1042 | 14567669 |
| NUDT-SIRST | best-mIoU | 2.8266e-6 | 1.2984e-5 | 4.59× | 27279 | 565 | 1036 | 43487024 |
| NUDT-SIRST | best-Pd | 7.6064e-6 | 2.1004e-5 | 2.76× | 27562 | 914 | 753 | 43486675 |
| IRSTD-1K | best-mIoU | 1.2981e-5 | 3.8773e-5 | 2.99× | 11088 | 2043 | 3446 | 52674367 |
| IRSTD-1K | best-Pd | 4.9553e-5 | 8.2063e-5 | 1.66× | 11672 | 4324 | 2862 | 52672086 |

这说明历史 `Fa` 数值不能解释为“全部背景误预测率”。它仍可用于和 SCTransNet 历史
结果同口径比较，但综合性能判断还必须查看 pixel precision/F1、pixel FP、mIoU/nIoU
和错误目标数。机器可读结果见
[`additive_joint_metrics_v1.json`](results/three_dataset_ec_tss_v3_1_seed42/comparison/additive_joint_metrics_v1.json)。

### 10.3 冻结 Gate 与历史配方比较

| Gate | 正式结果 | 解释 |
|---|---:|---|
| V3-A 工程闭环 | 通过 | 三个 formal1000、六个 strict-load 复评、checkpoint 指标回放与源绑定完整 |
| V3-B 相对 Original 严重退化为零 | **未通过：5 项** | NUAA 两角色目标计数回退；NUAA best-Pd tiny 回退；NUDT/IRSTD best-Pd 各回退目标 |
| V3-C 保留 λ=0.005 旧强项 | **未通过：4 项** | NUDT best-mIoU 少 2 个目标；IRSTD 两角色目标回退，且 IRSTD best-Pd mIoU 回退 396 quanta |
| V3-D 相对 TSS-off 与 λ=0.005 正向票更多 | **未通过** | 相对 TSS-off 为 10/2/18；相对 λ=0.005 为 13/2/15（更好/相同/更差） |
| V3-E 六配方联合 Pareto | 通过：4/6 | NUAA best-mIoU、NUDT 两角色、IRSTD best-mIoU 为 EC 独有非支配点 |
| V3-F 深入机制诊断 | 未完整；非否决门 | 1000-epoch 风险轨迹完整，固定 checkpoint cell 分离诊断未补；性能 Gate 已足以裁决 |

冻结五指标的完整成对票数为：

| 参考配方 | EC 更好 | 相同 | EC 更差 |
|---|---:|---:|---:|
| Original | 13 | 2 | 15 |
| TSS-off | 10 | 2 | 18 |
| TSS λ=0.005 | 13 | 2 | 15 |

加入 pixel precision、pixel F1 和错误目标数后的补充审计仍为混合权衡；`best_pd` 只能
视为高召回工作点，不能单靠 Pd 数值判定 EC-TSS 更优。EC 在 4/6 单元仍具有联合
Pareto 价值，但 NUAA best-Pd 被 TSS-off 严格支配，IRSTD best-Pd 被 TSS-off、
`λ=0.005` 和 `λ=0.01` 严格支配。

### 10.4 本阶段裁决

```text
decision = EC_TSS_V3_1_PERFORMANCE_FAIL_STOP_TSS_OPTIMIZATION
seed42_test_selected_operational_candidate = null
seed42_operational_recipe_admissible = false
global_operational_default = null
tss_training_innovation_supported = false
paper_core_established = false
stability_claim_supported = false
training_recipe_finalized = false
```

该裁决表示 EC-TSS **没有成为三数据集统一训练配方**，不是代码或训练中断，也不表示
其所有工作点都无效。TSS 此后只保留为可选训练辅助，不作为最终统一创新点继续扩展；
推理主线仍是 TPD8–NER4–QFG2。下一轮模型优化回到 NER→QFG→TPD 的单组件诊断顺序，
先处理跨数据集性能冲突，再决定是否修改现有模块；不立即叠加新的推理模块。

## 11. 当前最可靠的模型判断

### 已经成立

- TPD 是有效的结构方向，明显优于 Original/Progressive 的若干关键工作点。
- NER V4 相比 NER V1–V3 更适合作为完整模型的 NER 版本。
- 完整模型 D 在 NUDT 内部 seed42 以及多数据集的部分工作区间具有竞争力，特别是
  若干低 Fa 或错误目标数工作点。
- 模型优化必须同时看 Pd、Fa、mIoU、nIoU 和 tiny-Pd，不能只看 mIoU。
- EC-TSS V3.1 的训练、选模和正式复评已经闭环；它有 4/6 个非支配单元，但没有
  通过三数据集统一配方要求。

### 尚未成立

- TPD 全面超过 SPD。
- 完整模型在三个或四个数据集、两个 checkpoint 角色和所有指标上统一超过 Original。
- 旧 TSS 的某个固定正权重或 TSS-off 可以作为全局默认配方。
- EC-TSS 形成跨三数据集、两 checkpoint 角色的统一正式性能提升。
- 跨 seed 稳定性或独立测试泛化。

### 下一次模型决策标准

下一轮 NER→QFG→TPD 单组件优化仍逐数据集、逐 checkpoint 角色与 Original 和当前
无 TSS/旧 Final 对照比较：

1. Pd 是否保持或提高，以及具体多检出/少检出多少目标；
2. Fa 是否降低，不能用很小的 mIoU 改善换取大幅 Fa 上升；
3. mIoU 与 nIoU 是否至少有一个稳定改善，另一个不能严重退化；
4. tiny-Pd 是否保持；
5. 三个数据集是否减少严重退化项，而不是只在单一数据集获得最好数值。

新候选必须完成正式 1000-epoch 的 `best_miou` 与 `best_pd` 两角色评估；不能因为单一
数据集或单一 Pd 数值改善就接入下一模块。

## 12. NER/QFG/TPD 固定权重组件诊断

实验口径：seed 42、三个数据集各自的 TSS-off `best_miou` checkpoint、
`img_idx/test`、固定阈值 0.5。这是 test-selected 开发协议下的固定权重反事实诊断，
不是独立测试或多 seed 稳定性实验。

### 12.1 NER stage 2 启动门

NER stage2-only mask knockout 的冻结触发式为 `A AND (B OR C)`。正式聚合结果为：

| Gate | 通过数据集 | 要求 | 结果 |
|---|---:|---:|:---:|
| A | 0/3 | 至少 2/3 | FAIL |
| B | 0/3 | 至少 2/3 | FAIL |
| C | 3/3 | 至少 2/3 | PASS |

由于 A 失败，整体启动门失败；决策为
`DO_NOT_AUTHORIZE_NER_V5_PER_DEVELOPMENT_TRAINING`。因此不授权 NER V5-PER
development1000 训练，NER 保持 V4 Tail-Aware。

### 12.2 QFG 固定权重裁决

QFG 四个单 level knockout 均没有形成稳定的跨数据集影响；`all_off`
相对 full 与 full 相对 `all_off` 的 safe-material improvement 都是 0/3，
严重退化也都是 0/3。同时 full/`all_off` 在 3/3 数据集上存在功能差异，
所以 QFG 不能直接删除，也没有证据支持设计 QFG V3。正式决策为：

```text
qfg_decision = QFG_INCONCLUSIVE_NO_FORMULA_CHANGE
qfg_v3_remove_levels_authorized = false
qfg_off_candidate_authorized = false
```

### 12.3 TPD full 与 all7-off 完整固定点

`all7_off` 只把七个 TPD block 的完整 `saliency_scale` 向量置零，仍保留
Keep/SPD 下采样路径；因此它评估的是已训练 TPD8 residual 的固定权重贡献，
不是“关闭整个 TPD”。`unmatched` 为 component-Fa 的未匹配预测像素分子，
`background FP` 为所有 GT-background 上的阳性预测像素数。

| 数据集 | 模式 | Pd | tiny-Pd | unmatched | background FP | Fa | mIoU | nIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | full | 256/263（0.973384030） | 30/35（0.857142857） | 225 | 938 | 1.5435192156e-5 | 0.796482951 | 0.795348496 |
| NUAA-SIRST | all7_off | 256/263（0.973384030） | 30/35（0.857142857） | 225 | 940 | 1.5435192156e-5 | 0.796204974 | 0.794804433 |
| NUDT-SIRST | full | 936/945（0.990476190） | 258/259（0.996138996） | 121 | 591 | 2.7805925852e-6 | 0.944406006 | 0.946423233 |
| NUDT-SIRST | all7_off | 935/945（0.989417989） | 257/259（0.992277992） | 118 | 569 | 2.7116522732e-6 | 0.944363662 | 0.946266193 |
| IRSTD-1K | full | 277/297（0.932659933） | 23/30（0.766666667） | 618 | 2093 | 1.1728770697e-5 | 0.660311541 | 0.665661745 |
| IRSTD-1K | all7_off | 277/297（0.932659933） | 23/30（0.766666667） | 625 | 2086 | 1.1861620851e-5 | 0.659927798 | 0.664741929 |

三个数据集的变化都很小且方向混合：NUAA 的 Pd、tiny-Pd、unmatched 和 Fa
不变，但区域 IoU 略降；NUDT 用少 1 个目标和少 1 个 tiny target 换取更低的
unmatched 与 background FP；IRSTD-1K 保持 Pd/tiny-Pd，但 component-Fa 像素与
IoU 退化。没有一个方向达到冻结的 material 改善门。

### 12.4 七个单 block 与 all7 跨数据集门

| 模式 | off 相对 full safe-material | off 相对 full severe | full 相对 off safe-material | full 相对 off severe | persistent harmful |
|---|---:|---:|---:|---:|:---:|
| e1b0_off | 0/3 | 0/3 | 0/3 | 0/3 | false |
| e1b1_off | 0/3 | 0/3 | 0/3 | 0/3 | false |
| e1b2_off | 0/3 | 0/3 | 0/3 | 0/3 | false |
| e1b3_off | 0/3 | 0/3 | 0/3 | 0/3 | false |
| e2b0_off | 0/3 | 0/3 | 0/3 | 0/3 | false |
| e2b1_off | 0/3 | 0/3 | 0/3 | 0/3 | false |
| e2b2_off | 0/3 | 0/3 | 0/3 | 0/3 | false |
| all7_off | 0/3 | 0/3 | 0/3 | 0/3 | 不适用 |

七个单 block 的跨数据集 material improvement 和 severe/harm 均为 0/3，
因此 `persistent_harmful_block_ids=[]`。`all7_off` 相对 full 和 full 相对
`all7_off` 也都是 safe-material 0/3、severe 0/3。

### 12.5 功能差异、区域统计与最终决策

full 与 `all7_off` 的概率输出在 3/3 数据集上都超过冻结的功能等价容差：

| 数据集 | max-abs probability difference | mean-abs probability difference | 功能差异 |
|---|---:|---:|:---:|
| NUAA-SIRST | 0.069839656 | 1.2421815594e-6 | YES |
| NUDT-SIRST | 0.485106006 | 2.1746451257e-6 | YES |
| IRSTD-1K | 0.146938682 | 2.6068548838e-6 | YES |

full 推理的七个 block 区域统计中，目标区 residual RMS 在三个数据集均高于
背景区 residual RMS，即 21/21 个 dataset-block 组合都为正差值。三数据集的七块
`target RMS - background RMS` 范围分别为：NUAA `0.023510585–0.097152975`、
NUDT `0.162237070–0.865750660`、IRSTD-1K `0.430354341–1.643060837`。

这些结果证明 TPD residual 确实进入最终输出，并且其响应在聚合意义上更偏向
目标区；但它们没有建立 full 或任一 off 方向的跨数据集性能优势。因此正式决策为：

```text
tpd_decision = TPD_INCONCLUSIVE_NO_FORMULA_CHANGE
tpd_local_candidate_training_authorized = false
tpd_residual_off_candidate_authorized = false
requires_new_tenth_mode = false
```

当前模型状态冻结为 `TPD8 + NER4 + QFG2`、`TSS OFF`。不授权 NER V5、
QFG V3、TPD 公式修改、第十种固定权重组合模式或由本轮诊断触发的 fresh training。

### 12.6 正式裁决与机器可读路径

- NER decision Markdown：[`results/ner_stage2_mask_knockout_v1/comparison/best_miou_seed42/decision.md`](results/ner_stage2_mask_knockout_v1/comparison/best_miou_seed42/decision.md)
- NER decision JSON：[`results/ner_stage2_mask_knockout_v1/comparison/best_miou_seed42/decision.json`](results/ner_stage2_mask_knockout_v1/comparison/best_miou_seed42/decision.json)
- QFG decision Markdown：[`results/three_dataset_qfg_level_knockout_v1/comparison/best_miou_seed42/decision.md`](results/three_dataset_qfg_level_knockout_v1/comparison/best_miou_seed42/decision.md)
- QFG decision JSON：[`results/three_dataset_qfg_level_knockout_v1/comparison/best_miou_seed42/decision.json`](results/three_dataset_qfg_level_knockout_v1/comparison/best_miou_seed42/decision.json)
- TPD decision Markdown：[`results/three_dataset_tpd8_block_residual_knockout_v1/comparison/best_miou_seed42/decision.md`](results/three_dataset_tpd8_block_residual_knockout_v1/comparison/best_miou_seed42/decision.md)
- TPD decision JSON：[`results/three_dataset_tpd8_block_residual_knockout_v1/comparison/best_miou_seed42/decision.json`](results/three_dataset_tpd8_block_residual_knockout_v1/comparison/best_miou_seed42/decision.json)
- TPD 三数据集输入 evaluation：
  [`NUAA`](results/three_dataset_tpd8_block_residual_knockout_v1/runs/NUAA-SIRST/v4_tss_off_best_miou_seed42/evaluation.json)、
  [`NUDT`](results/three_dataset_tpd8_block_residual_knockout_v1/runs/NUDT-SIRST/v4_tss_off_best_miou_seed42/evaluation.json)、
  [`IRSTD-1K`](results/three_dataset_tpd8_block_residual_knockout_v1/runs/IRSTD-1K/v4_tss_off_best_miou_seed42/evaluation.json)。

## 13. GCSF 完整模型级跳连重分配诊断

实验口径：seed 42，NUAA-SIRST、NUDT-SIRST、IRSTD-1K 各自 `img_idx/test`；
使用当前完整 `TPD8 + NER4 + QFG2 + TSS-off` 的 `best_miou` 与 `best_pd`
checkpoint。每份 checkpoint 计算当前 `(T+E)+E` 和 10 个 GCSF 可表示固定比例，
共 `6 checkpoints × 11 modes = 66` 个单元。正式指标阈值为 0.5；阈值 1.0 仅记录
`Pd=0, Fa=0` 的空预测端点，不参与裁决。

### 13.1 当前模型六角色重放基准

| 数据集 | checkpoint | Pd（数值） | tiny-Pd（数值） | Fa | mIoU | nIoU | unmatched | background FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | best_miou | 256/263（0.973384030） | 30/35（0.857142857） | 1.5435192156e-5 | 0.796482951 | 0.795348496 | 225 | 938 |
| NUAA-SIRST | best_pd | 257/263（0.977186312） | 30/35（0.857142857） | 1.4749183616e-5 | 0.788553432 | 0.792667957 | 215 | 820 |
| NUDT-SIRST | best_miou | 936/945（0.990476190） | 258/259（0.996138996） | 2.7805925852e-6 | 0.944406006 | 0.946423233 | 121 | 591 |
| NUDT-SIRST | best_pd | 940/945（0.994708995） | 258/259（0.996138996） | 6.6642301628e-6 | 0.937380628 | 0.939836330 | 290 | 1005 |
| IRSTD-1K | best_miou | 277/297（0.932659933） | 23/30（0.766666667） | 1.1728770697e-5 | 0.660311541 | 0.665661745 | 618 | 2093 |
| IRSTD-1K | best_pd | 287/297（0.966329966） | 25/30（0.833333333） | 2.3248776868e-5 | 0.639986059 | 0.650812036 | 1225 | 2682 |

六份 `current_g0` 全部通过既有正式 evaluation 的逐指标重放核验，说明此次反事实
比较没有改变数据、checkpoint、推理图或指标定义。

### 13.2 Trigger A 汇总

| 固定 GCSF mode | `best_miou` safe-material | 六角色 severe | Trigger A |
|---|---:|---:|:---:|
| gneg025_l1_only | 0/3 | 0/6 | false |
| gneg025_l2_only | 0/3 | 0/6 | false |
| gneg025_l3_only | 0/3 | 2/6 | false |
| gneg025_l4_only | 0/3 | 3/6 | false |
| gneg025_all_levels | 0/3 | 6/6 | false |
| gpos025_l1_only | 1/3 | 2/6 | false |
| gpos025_l2_only | 0/3 | 1/6 | false |
| gpos025_l3_only | 1/3 | 2/6 | false |
| gpos025_l4_only | 2/3 | 1/6 | false |
| gpos025_all_levels | 0/3 | 6/6 | false |

Trigger A 要求同一非零 mode 在 `best_miou` 上至少 2/3 safe-material，并且六个
数据集/角色单元 severe 为 0。没有 mode 同时满足两项。

### 13.3 最接近通过的 `gpos025_l4_only` 绝对性能

该模式只在 L4 使用 `g=+0.25`，即将 L4 从 `1.0T+2.0E` 改为
`1.25T+1.75E`。其余尺度不变。

| 数据集 | checkpoint | Pd（数值） | tiny-Pd（数值） | Fa | mIoU | nIoU | unmatched | background FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | best_miou | 255/263（0.969581749） | 30/35（0.857142857） | 1.4268977637e-5 | 0.794885570 | 0.794761351 | 208 | 919 |
| NUAA-SIRST | best_pd | 256/263（0.973384030） | 30/35（0.857142857） | 1.3651569951e-5 | 0.786460601 | 0.791718658 | 199 | 800 |
| NUDT-SIRST | best_miou | 935/945（0.989417989） | 258/259（0.996138996） | 1.9762889448e-6 | 0.944692796 | 0.946325635 | 86 | 542 |
| NUDT-SIRST | best_pd | 939/945（0.993650794） | 258/259（0.996138996） | 5.7909862105e-6 | 0.939057320 | 0.941751046 | 252 | 942 |
| IRSTD-1K | best_miou | 275/297（0.925925926） | 23/30（0.766666667） | 1.0476183535e-5 | 0.660594728 | 0.662382679 | 552 | 2045 |
| IRSTD-1K | best_pd | 287/297（0.966329966） | 25/30（0.833333333） | 2.1388874718e-5 | 0.643207569 | 0.653143190 | 1127 | 2588 |

相对当前模型，它在 NUAA/NUDT `best_miou` 上以各少检 1 个目标换取更低 FP，均
达到冻结的 safe-material 门；但 IRSTD-1K `best_miou` 少检 2 个目标，触发唯一
severe 单元。`best_pd` 三项均未 severe，仍不能抵消 primary veto。

### 13.4 正式裁决

```text
decision = GCSF_BRANCH_AUDIT_NO_TRAINING_AUTHORIZATION
gcsf_trigger_a_passed = false
gcsf_pilot_authorized = false
gcsf_formal_training_authorized = false
skip_fusion_performance_bottleneck_established = false
next_step = DEEP_SUPERVISION_GRADIENT_AUDIT
```

GCSF 的 480 参数训练/推理代码、导出器和 seed42 scratch runner 已实现并完成工程
测试，但训练入口会重放六份输入和 decision；当前失败 decision 已被实际拒绝。因此
没有产生 GCSF 训练 checkpoint，也不能报告训练后 GCSF 性能。当前正式完整模型仍为
`TPD8 + NER4 + QFG2 + TSS-off`，主线和既有创新点不变。

工程合同为 `480 parameters / 4 state keys / 0 buffers`；训练图为
`10,870,708 parameters / 572 keys`，head-free 推理图为
`10,870,610 parameters / 568 keys`。普通模式与 `python -O` 均通过
`33 tests + 3389 subtests`，RTX 5090 训练态 forward/backward smoke 也已通过。

- 正式裁决：[`decision.md`](results/three_dataset_gcsf_branch_audit_v1/comparison/seed42_six_role/decision.md)
- 机器可读裁决：[`decision.json`](results/three_dataset_gcsf_branch_audit_v1/comparison/seed42_six_role/decision.json)
- 六份输入：[`results/three_dataset_gcsf_branch_audit_v1/runs/`](results/three_dataset_gcsf_branch_audit_v1/runs/)

## 14. 六头 Deep-Supervision 梯度审计

在 GCSF 未获得训练授权后，继续检查当前完整模型的六个等权 BCE 监督头是否形成可
跨数据集复现的梯度冲突。审计固定 seed 42，使用 NUAA-SIRST、NUDT-SIRST、
IRSTD-1K 各自 `img_idx/train`，并同时覆盖当前 TSS-off 完整模型的 `best_miou` 与
`best_pd` checkpoint。正式 manifest 共 512 个真实 crop、32 个唯一 batch；六份
`audit.json` 全部通过输入重放、checkpoint、模型 state、RNG、sentinel 和参数分区
核验。构建器、分析器和比较器的 38 项定向测试在普通 Python 与 `python -O` 下均
通过，六份 raw Gram 的独立重算结果与正式 aggregate 完全一致。

### 14.1 正式裁决

```text
decision=DS_GLOBAL_REWEIGHTING_BLOCKED_BY_DOMAIN_REVERSAL
engineering_valid=true
trigger_a_passed=false
signature_count=60
domain_reversal_signature_count=2
authorized_signature_count=0
ds_v2_design_authorized=false
ds_v2_training_authorized=false
tiny_gradient_conflict_supported=false
gradient_scale_anomaly_observed=false
```

审计分层可用性为：tiny 与 normal 在三数据集均可用；background 只在 NUAA-SIRST
和 IRSTD-1K 可用。NUDT-SIRST 的 663 张正式训练图像都含目标，因此其背景分层按
合同记为结构上不可用，没有生成合成背景样本。

### 14.2 决定性跨数据集反转

| 分层 / 参数组 / 监督头 | NUDT-SIRST | NUAA-SIRST | IRSTD-1K |
|---|---|---|---|
| tiny / NER / gt3 | best-mIoU：cos=-0.151577，ratio=6.9744，PC+AC；best-Pd 转为 +0.089162 | 两角色约 +0.181/+0.182，无稳定冲突 | 两角色 +0.744454/+0.614671，均 PA |
| normal / NER / gt2 | best-Pd：cos=-0.305888，ratio=1.0005，PC；best-mIoU 为 +0.086451 | 两角色 +0.329718/+0.605368，均 PA | 两角色 +0.971247/+0.940586，均 PA |

这两组结果说明，局部冲突集中在 NUDT-SIRST 的 NER 梯度路径，而且随 checkpoint
角色变化；同一梯度在 NUAA-SIRST 或 IRSTD-1K 上却是正向协同。六十个候选签名中
没有一个同时满足跨数据集和双 checkpoint 授权门。因此不应统一降低六头权重，也不
应启动全局 DS V2 训练。该结果不代表现有六头监督失败，只表示“全局统一重加权”不是
当前可复用的性能优化方向。

当前正式完整模型仍为 `TPD8 + NER4 + QFG2 + TSS-off`；TPD、NER、QFG 主线和创新点
均未改变。本轮没有训练新模型，也没有产生新的性能 checkpoint。

- 正式裁决：[`decision.md`](results/three_dataset_ds_gradient_audit_v1/comparison/seed42_six_role/decision.md)
- 机器可读裁决：[`decision.json`](results/three_dataset_ds_gradient_audit_v1/comparison/seed42_six_role/decision.json)
- 聚合结果：[`aggregate.json`](results/three_dataset_ds_gradient_audit_v1/comparison/seed42_six_role/aggregate.json)
- 六份审计：[`results/three_dataset_ds_gradient_audit_v1/runs/`](results/three_dataset_ds_gradient_audit_v1/runs/)

## 15. DORF V1 深监督输出复用筛选

当前 Original 与 Final 都训练了多尺度融合 readout
`d0=outconv(gt2,gt3,gt4,gt5,out)`，但历史正式推理只返回 `out`。DORF V1 在 raw
logit 空间预注册 `α=0.25/0.50/0.75/1.0`，使用同一个 α 同时重放 Final TSS-off 与
Original 的三数据集 best-mIoU/best-Pd，共 12 checkpoint、60 个工作点。固定阈值为
0.5；没有重新选 checkpoint，也没有训练新权重。

12 个 checkpoint、evaluation、summary/protocol、数据协议和背景 FP sidecar 均在
首个输出前冻结 SHA。12/12 α=0 历史重放与工程核验通过；普通 Python 和
`python -O` 的 32 项定向测试均通过，正式比较输出逐字节一致。

### 15.1 Trigger A

| mode | α | Final best-mIoU safe-material | Final 六角色 severe | Ma0/Maa 新 severe | 结果 |
|---|---:|---:|---:|---:|:---:|
| `dorf_a025` | 0.25 | 0/3 | 2/6 | 0/0 | FAIL |
| `dorf_a050` | 0.50 | 0/3 | 4/6 | 2/0 | FAIL |
| `dorf_a075` | 0.75 | 0/3 | 4/6 | 2/0 | FAIL |
| `d0_only` | 1.00 | 0/3 | 5/6 | 3/0 | FAIL |

`Ma0` 表示 `Final(α)` 对 `Original(0)`，`Maa` 表示 `Final(α)` 对
`Original(α)`；数字为相对当前 Final(0)/Original(0) 新增的 severe 条件数。

### 15.2 最小干预 `α=0.25` 的主角色

| 数据集 | ΔPd 目标计数 | Δtiny | ΔmIoU | ΔnIoU | component FP reduction | background FP reduction | severe |
|---|---:|---:|---:|---:|---:|---:|:---:|
| NUAA-SIRST | -2 | 0 | -0.001219 | -0.004271 | -3.56% | +3.84% | YES |
| NUDT-SIRST | -2 | -1 | +0.000021 | +0.000159 | +12.40% | +4.23% | YES |
| IRSTD-1K | 0 | 0 | -0.000006 | -0.001034 | +0.49% | +0.72% | NO，但无 material gain |

更大的 α 总体进一步降低 FP，但也增加目标漏检或 IoU 回退。例外是 NUDT-SIRST
best-Pd：`α=0.50/0.75/1.0` 均保持 940/945 和 tiny 258/259，同时降低 component
FP、略升 mIoU/nIoU；然而这一优势没有在主裁决 best-mIoU 重现，不能作为跨数据集
统一输出。

### 15.3 正式裁决

```text
decision=DORF_V1_ZERO_TRAINING_TRIGGER_FAILED
selected_mode=null
selected_alpha=null
dorf_v1_production_implementation_authorized=false
fresh_formal1000_launch_authorized_by_this_comparator=false
model_mainline_changed=false
training_loss_changed=false
```

因此不实现固定 α DORF 生产图，不启动 6 个 selector-aligned fresh formal1000。当前
正式完整模型仍为 `TPD8 + NER4 + QFG2 + TSS-off`。DORF 表明 d0 更像保守的低 FP
readout，直接全图平均无法同时保住 Pd 与区域质量；后续结构候选必须显式保护目标响应。

- 正式裁决：[`decision.md`](results/three_dataset_dorf_v1/comparison/seed42_twelve_role/decision.md)
- 机器可读裁决：[`decision.json`](results/three_dataset_dorf_v1/comparison/seed42_twelve_role/decision.json)
- 12 份 evaluation：[`results/three_dataset_dorf_v1/runs/`](results/three_dataset_dorf_v1/runs/)
- 冻结输入 manifest：[`dorf_v1_input_manifest.json`](results/three_dataset_dorf_v1/manifests/dorf_v1_input_manifest.json)

## 16. NER-L4-TPR 目标保护重分配候选

GCSF 的 `gpos025_l4_only` 在六角色中持续降低两类 FP，但合计比当前 Final 少检
6 个目标；DORF 也重复出现低响应伴随 Pd/IoU 回退。NER-L4-TPR 因此不再全图统一
调整，而是用现有五节点 NER 的 `q4` tail evidence 生成停止梯度的目标保护区，仅在
非目标区域学习 L4 Transformer/Encoder 常系数和重分配。

新增模块只有一个 `(1,256,1,1)` 零初始化参数，共 256 parameters / 1 state key /
0 buffers。训练图为 `10,870,484 parameters / 569 keys`，head-free 推理图为
`10,870,386 parameters / 565 keys`。当前 Final 权重零扩展后，六个 segmentation
输出在 CPU 和 RTX 5090 上均逐位一致；RTX 5090 backward 中 256 个门参数全部获得
有限非零梯度。严格导出 569→565 只删除四个 TSS 训练键。

### 16.1 六角色固定 checkpoint 筛选

三数据集 `best_miou` 与 `best_pd` 共六角色已完成固定阈值 0.5 筛选。联合比较同时
使用当前 `g=0` 和无保护 `gpos025_l4_only` 两个参照：前者判断 FP/Fa 是否保留，
后者判断被全局重分配压掉的目标是否恢复。没有用单一指标门或跨量纲加权和。

```text
assessment=REPRESENTABLE_CROSS_ROLE_JOINT_SIGNAL
representable_cross_role_joint_mode=tpr_g01875
finite_logit_pareto_modes=tpr_g00625,tpr_g0125,tpr_g01875
```

| 模式 | 目标恢复单元 | 两类 FP 下降单元 | 联合单元 | best-mIoU 联合 | best-Pd 联合 | ΣΔ目标 vs 无保护 | ΣΔcomponent FP vs 当前 | ΣΔbackground FP vs 当前 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `tpr_g00625` | 5/6 | 3/6 | 2/6 | 2/3 | 0/3 | +5 | -30 | -27 |
| `tpr_g0125` | 5/6 | 3/6 | 2/6 | 2/3 | 0/3 | +5 | -64 | -64 |
| `tpr_g01875` | 4/6 | 5/6 | 3/6 | 1/3 | 2/3 | +4 | -99 | -95 |
| `tpr_g025`（边界） | 4/6 | 5/6 | 3/6 | 1/3 | 2/3 | +4 | -136 | -129 |

`tpr_g01875` 相对无保护重分配恢复 4 个目标，同时相对当前模型合计降低 99 个
component FP 像素和 95 个 background FP 像素；六角色 tiny 检出总数保持不变。
但它相对当前模型仍合计少检 2 个目标，说明这是支持训练的架构信号，不是训练后最终
性能结论。

### 16.2 正式训练完成状态

正式执行决定为 `authorize_formal_training`。GPU 1-epoch smoke 已通过，新门 256/256
参数发生有限更新。NUAA-SIRST、NUDT-SIRST、IRSTD-1K 三套 seed42、scratch、
1000-epoch、TSS-off 实验均已完成；每套实验从 epoch 10 起每 10 epochs 评估，并且只
保留各自的 `best_miou` 与 `best_pd` checkpoint。三份 `summary.json` 和六份正式权重
均已落盘，训练进程正常结束，GPU 已释放。

```text
NUAA-SIRST=1000/1000 complete
NUDT-SIRST=1000/1000 complete
IRSTD-1K=1000/1000 complete
seed=42
threshold=0.5
selection_split=img_idx/test
test_selected=true
selection_is_optimistic=true
independent_test_confirmation=false
```

### 16.3 NER-L4-TPR 正式绝对性能

下表中的 Pd 和 tiny-Pd 同时给出匹配计数与数值；Fa 为未匹配预测连通域像素除以
有效像素数。`best_miou` 与 `best_pd` 是两个独立的选模角色，不能跨角色拼接指标。

| 数据集 | 角色 | epoch | Pd（匹配/总数；值） | tiny-Pd（匹配/总数；值） | Fa ↓ | mIoU ↑ | nIoU ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | best-mIoU | 710 | 256/263；0.973384 | 30/35；0.857143 | 1.447478e-5 | 0.797080 | 0.803596 |
| NUAA-SIRST | best-Pd | 310 | 258/263；0.980989 | 32/35；0.914286 | 3.107619e-5 | 0.766000 | 0.782239 |
| NUDT-SIRST | best-mIoU | 430 | 939/945；0.993651 | 258/259；0.996139 | 5.676086e-6 | 0.940423 | 0.944872 |
| NUDT-SIRST | best-Pd | 460 | 940/945；0.994709 | 259/259；1.000000 | 8.387738e-6 | 0.938033 | 0.942877 |
| IRSTD-1K | best-mIoU | 240 | 282/297；0.949495 | 23/30；0.766667 | 3.769149e-5 | 0.670300 | 0.658517 |
| IRSTD-1K | best-Pd | 230 | 285/297；0.959596 | 22/30；0.733333 | 2.941682e-5 | 0.639739 | 0.647207 |

像素分类质量与两类 FP 如下。component FP 是未匹配预测连通域中的像素数；
background FP 是全部落在 GT 背景上的预测前景像素数。二者含义不同，不能互相替代。

| 数据集 | 角色 | Precision ↑ | Recall ↑ | F1 ↑ | component FP px ↓ | background FP px ↓ | 未匹配预测目标 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | best-mIoU | 0.895818 | 0.878517 | 0.887083 | 211 | 857 | 22 |
| NUAA-SIRST | best-Pd | 0.867445 | 0.867549 | 0.867497 | 453 | 1112 | 40 |
| NUDT-SIRST | best-mIoU | 0.976488 | 0.962211 | 0.969297 | 247 | 656 | 29 |
| NUDT-SIRST | best-Pd | 0.968402 | 0.967650 | 0.968026 | 365 | 894 | 34 |
| IRSTD-1K | best-mIoU | 0.781376 | 0.825031 | 0.802610 | 1986 | 3355 | 97 |
| IRSTD-1K | best-Pd | 0.770226 | 0.790629 | 0.780294 | 1550 | 3428 | 98 |

### 16.4 相对优化前 TSS-off Final 的同角色差值

比较严格使用各方法自己的同名 checkpoint：NER-L4-TPR `best_miou` 只对比原 Final
`best_miou`，`best_pd` 只对比原 Final `best_pd`。正的 `ΔPd/Δtiny/ΔmIoU/ΔnIoU`
表示 NER-L4-TPR 更高；负的 `ΔFa/Δcomponent FP/Δbackground FP` 表示虚警更少。

| 数据集 | 角色 | ΔPd 目标数 | Δtiny | ΔFa | ΔmIoU | ΔnIoU | Δcomponent FP px | Δbackground FP px | 核心解释 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| NUAA-SIRST | best-mIoU | 0 | 0 | -9.604120e-7 | +0.000597 | +0.008248 | -14 | -81 | Pd 不变且 Fa、IoU、两类 FP 同向改善 |
| NUAA-SIRST | best-Pd | +1 | +2 | +1.632700e-5 | -0.022553 | -0.010429 | +238 | +292 | 更高总 Pd/tiny-Pd 换取更差 Fa 与 IoU |
| NUDT-SIRST | best-mIoU | +3 | 0 | +2.895493e-6 | -0.003983 | -0.001552 | +126 | +65 | 更高 Pd 换取轻微 IoU 与 Fa 回退 |
| NUDT-SIRST | best-Pd | 0 | +1 | +1.723508e-6 | +0.000652 | +0.003041 | +75 | -111 | Pd 持平、tiny/IoU 改善，但两种 FP 方向不一致 |
| IRSTD-1K | best-mIoU | +5 | 0 | +2.596272e-5 | +0.009989 | -0.007145 | +1368 | +1262 | 更高 Pd/mIoU，但虚警代价显著且 nIoU 回退 |
| IRSTD-1K | best-Pd | -2 | -3 | +6.168043e-6 | -0.000247 | -0.003605 | +325 | +746 | Pd、tiny-Pd、Fa、IoU 核心指标均回退 |

需要补充两点：NUAA best-mIoU 虽然核心检测/分割指标同向改善，但 pixel recall 比原
Final 低 0.007034，因此在完整九维报告向量上仍属于不可比关系；IRSTD best-mIoU 的
Pd 与 mIoU 提升也不能抵消显著增加的两类 FP。这里不使用单指标或加权和把权衡改写
成“全面胜出”。

### 16.5 三数据集联合正式裁决

训练后比较器在 3 数据集 × 2 checkpoint 角色 × 9 个报告单元上进行无加权、逐单元
比较。相对优化前 TSS-off Final，NER-L4-TPR 为 21 个单元更好、5 个相同、28 个更差；
相对 Original 为 29 个更好、3 个相同、22 个更差。六个“数据集/角色”单元均为
`incomparable`，因此最终分类为：

```text
status=complete
classification=NER_L4_TPR_MIXED_TRADEOFF_REPORTED_VECTOR
candidate_vs_current_final_tss_off=incomparable
candidate_vs_original=incomparable
global_production_replacement_authorized=false
candidate_retained=true
mainline_changed=false
model_success_claim_made=false
```

这不是“模块完全失败”：NUAA best-mIoU 建立了 Pd 不变、Fa 更低且 mIoU/nIoU 更高的
正向工作点；NUDT 和 IRSTD best-mIoU 也提高了目标检出数。但它没有形成三数据集、
双 checkpoint 角色的一致提升，尤其 IRSTD best-Pd 明确回退。因此 NER-L4-TPR 可作为
受数据域和工作区影响的有效候选保留，不能直接替换当前全局正式模型。当前生产基线
仍保持 `TPD8 + NER4 + QFG2 + TSS-off`。

### 16.6 正式产物

- 联合筛选：[`decision.md`](results/three_dataset_ner_l4_tpr_zero_training_v1/comparison/seed42_six_role/decision.md)
- 机器结果：[`decision.json`](results/three_dataset_ner_l4_tpr_zero_training_v1/comparison/seed42_six_role/decision.json)
- 正式训练决定：[`execution_decision.json`](results/three_dataset_ner_l4_tpr_zero_training_v1/comparison/seed42_six_role/execution_decision.json)
- 正式训练目录：[`results/three_dataset_l4_tpr_tss_off_seed42_v1/`](results/three_dataset_l4_tpr_tss_off_seed42_v1/)
- NUAA-SIRST 正式摘要：[`summary.json`](results/three_dataset_l4_tpr_tss_off_seed42_v1/runs/NUAA-SIRST/final_tss_off_ner_l4_tpr_v1/seed_42/summary.json)
- NUDT-SIRST 正式摘要：[`summary.json`](results/three_dataset_l4_tpr_tss_off_seed42_v1/runs/NUDT-SIRST/final_tss_off_ner_l4_tpr_v1/seed_42/summary.json)
- IRSTD-1K 正式摘要：[`summary.json`](results/three_dataset_l4_tpr_tss_off_seed42_v1/runs/IRSTD-1K/final_tss_off_ner_l4_tpr_v1/seed_42/summary.json)
- 训练后比较器：[`compare_three_dataset_ner_l4_tpr_posttraining_v1.py`](analysis/compare_three_dataset_ner_l4_tpr_posttraining_v1.py)

## 17. PBDR-V1 固定双读出零训练审计

PBDR-V1 使用当前 TSS-off Final 的六个 checkpoint，在每张测试图的一次 forward 中捕获
q4、out 与 d0，并比较 `g=0/0.125/0.25/0.5/0.75/1.0`。正式授权候选只有中间四个；
`g=1` 是不可授权的 max/min oracle。六角色均使用 seed42、`img_idx/test`、阈值 0.5、
统一 TF32-off FP32 设置。

```text
decision=PBDR_GLOBAL_FIXED_G_SCREEN_FAILED
passing_authorization_gates=[]
pbdr_implementation_authorized=false
pbdr_training_authorized=false
formal1000_started=false
```

| g | T1 通过数据集 | T2 severe | T3 | T4 | T5 | 总通过 |
|---:|---:|:---:|:---:|:---:|:---:|:---:|
| 0.125 | 0/3 | 0 | 否 | 否 | 是 | 否 |
| 0.250 | 0/3 | 0 | 否 | 否 | 是 | 否 |
| 0.500 | 1/3 | 1 | 否 | 否 | 是 | 否 |
| 0.750 | 1/3 | 1 | 否 | 否 | 是 | 否 |

三个 best-mIoU Current 锚点及最有利描述点如下。Pd 给出数值和匹配计数；Fa 使用
unmatched component pixels / valid pixels。

| 数据集 | 点 | Pd（匹配/总数；值） | Fa ↓ | mIoU ↑ | nIoU ↑ | component FP px ↓ | background FP px ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | Current g=0 | 256/263；0.973384 | 1.543519e-5 | 0.796761 | 0.795636 | 225 | 936 |
| NUAA-SIRST | g=0.125 | 256/263；0.973384 | 1.543519e-5 | 0.796596 | 0.794988 | 225 | 953 |
| NUDT-SIRST | Current g=0 | 936/945；0.990476 | 2.780593e-6 | 0.944373 | 0.946329 | 121 | 592 |
| NUDT-SIRST | g=0.75 | 936/945；0.990476 | 2.780593e-6 | 0.944728 | 0.946825 | 121 | 633 |
| IRSTD-1K | Current g=0 | 277/297；0.932660 | 1.172877e-5 | 0.660251 | 0.665585 | 618 | 2093 |
| IRSTD-1K | g=0.25 | 277/297；0.932660 | 1.174775e-5 | 0.660759 | 0.665998 | 619 | 2112 |

固定 PBDR 没有在任何 best-mIoU 数据集增加 Pd。NUDT 的 IoU 改善没有降低 Fa；IRSTD
的 IoU 改善伴随 Fa 上升；NUAA 从最小门控开始即降低 mIoU/nIoU。目标救援信号为
NUAA/NUDT/IRSTD=`2/0/4` 个漏检目标，非保护区 FP 抑制信号为 `0/0/43` 个像素。
因此硬保护图同时保护了 NUAA/NUDT 的误检，而双读出在 NUDT 漏检目标上没有更强 d0
可用。PBDR-V1 固定公式关闭，不实现、不训练；但 `g=0.75` 在 NUDT 与 IRSTD 同时提高
mIoU/nIoU，说明 PBDR 研究族保留正向重叠质量信号。后续应改为 q4 直接 residual 与
可学习软保护，而不是继续调固定 g。

这里不把“所有数据集、所有指标同时提升”作为论文模型必要条件。原
SCTransNet Table I 也不是所有 Pd/Fa 都最优：NUAA Fa 13.92 高于 DNA-Net 8.78，
NUDT Pd 98.62 低于 DNA-Net 98.83，IRSTD Pd 93.27 低于 UIU-Net 93.98。

后续完整模型前瞻采用 M2F-SV 门：在三个 `best_miou` 上，检测族（Pd/Fa）
至少获得 2/3 数据集的安全实质改善，重叠族（mIoU/nIoU）也至少获得 2/3，
且至少一个数据集在同一 checkpoint 上同时支持两族；六个 `best_miou/best_pd`
角色无 severe，且没有数据集被 Original 实质支配。安全/实质阈值复用已有冻结
DORF 合同：检测实质改善为多检至少 2 个目标或 FP 下降至少 5%，重叠实质改善为
mIoU/nIoU 任一提高至少 0.005。

这是看过 PBDR-V1 结果后的协议修订，标记为 `post_hoc_protocol_amendment=true`，只适用于
PBDR-V2 及之后的统一 scratch run，不追溯改判 PBDR-V1。PBDR-V1 在该门下仍不通过：
检测族没有任何 `D+`，其 IoU 改善远小于 0.005，且高 g 存在 severe 角色；但这些
定向信号足以支持继续设计 PBDR-V2。

- 最终裁决：[`decision.md`](results/three_dataset_pbdr_zero_training_v1/comparison/seed42_six_role/decision.md)
- 机器结果：[`decision.json`](results/three_dataset_pbdr_zero_training_v1/comparison/seed42_six_role/decision.json)
- 冻结协议：[`PBDR_V1_PROTOCOL.md`](experiments/PBDR_V1_PROTOCOL.md)
- 六角色结果：[`results/three_dataset_pbdr_zero_training_v1/runs/`](results/three_dataset_pbdr_zero_training_v1/runs/)

## 18. PBDR-V2 自适应证据残差路由 formal1000（进行中）

PBDR-V2 在冻结的 `TPD8 + NER4 + QFG2-CROA + TSS-off` 主干上增加 19 个
readout 参数。训练图为 573 state keys，推理图为 569 state keys。三个数据集使用
seed42、各自 `img_idx`、1000 epochs、epoch 10 起每 10 epochs 评估、固定阈值 0.5，
并分别保存自己的 `best_miou` 与 `best_pd`。

当前执行状态：

```text
NUAA-SIRST=1000/1000 complete
NUDT-SIRST=running on GPU0
IRSTD-1K=queued for GPU2 after the existing baseline NUDT run
```

### 18.1 NUAA-SIRST 正式结果

| 角色 | 模型 | epoch | Pd（匹配/总数；值） | tiny-Pd | Fa ↓ | mIoU ↑ | nIoU ↑ | 预测组件/误检组件 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| best-mIoU | PBDR-V2 | 550 | 254/263；0.9657794677 | 30/35；0.8571428571 | 2.4627706595e-5 | 0.7825994015 | 0.7934425659 | 289/35 |
| best-mIoU | Current TSS-off | 850 | 256/263；0.9733840304 | 30/35；0.8571428571 | 1.5435192156e-5 | 0.7964829509 | 0.7953484960 | 277/21 |
| best-mIoU | Original | 830 | 255/263；0.9695817490 | 32/35；0.9142857143 | 2.6548530508e-5 | 0.7868246549 | 0.7950956988 | 282/27 |
| best-Pd | PBDR-V2 | 330 | 257/263；0.9771863118 | 32/35；0.9142857143 | 3.6221250926e-5 | 0.7694566814 | 0.7811902030 | 292/35 |
| best-Pd | Current TSS-off | 820 | 257/263；0.9771863118 | 30/35；0.8571428571 | 1.4749183616e-5 | 0.7885534318 | 0.7926679569 | 275/18 |
| best-Pd | Original | 440 | 260/263；0.9885931559 | 34/35；0.9714285714 | 8.1017608604e-5 | 0.7262357414 | 0.7481629044 | 367/107 |

相对 Current，PBDR-V2 的 NUAA `best_miou` 少检 2 个目标，Fa 增加 59.56%，
mIoU 下降 0.013884；`best_pd` 的总 Pd 持平、tiny-Pd 多 2 个，但 Fa 增加
145.58%，mIoU/nIoU 分别下降 0.019097/0.011478。因此 NUAA 不是可接受的正向工作点，
也不是由单个指标造成的轻微权衡。相对 Original 则仍是混合结果：PBDR-V2 的 Fa
略低，但 Pd 与重叠质量较弱。

正式产物审计已确认：1000 个 epoch 连续完整、100 个评估点完整；目录仅保留两份
selected checkpoint；两份权重均为 573 keys，4 个 TSS state 精确为零，19 个 PBDR
标量均已从零学习为非零。由此可排除“路由没有参与训练”这一解释，当前问题属于
PBDR-V2 配方在 NUAA 上的真实性能退化。

- PBDR-V2 NUAA 摘要：[`summary.json`](results/three_dataset_pbdr_v2_tss_off_seed42_v1/runs/NUAA-SIRST/pbdr_v2_tss_off/seed_42/summary.json)
- Current NUAA 摘要：[`summary.json`](results/three_dataset_tss_off_seed42_v1/runs/NUAA-SIRST/final_tss_off/seed_42/summary.json)
- 冻结协议：[`PBDR_V2_PROTOCOL.md`](experiments/PBDR_V2_PROTOCOL.md)

## 19. 权威结果来源

- 初代 TPD、V6、V7、NER、TSS、QFG 与工程认证摘要：[`README.md`](README.md)
- 初代 TPD 研究裁决：[`TPD_SCTransNet_主线修订版.md`](TPD_SCTransNet_主线修订版.md)
- V6 评估：[`SCTransNet_TPD_V6_整体设计正确性与创新性评估.md`](SCTransNet_TPD_V6_整体设计正确性与创新性评估.md)
- V7 复盘：[`SCTransNet_TPD_V7_DCH_失败分析与不改主线修改计划.md`](SCTransNet_TPD_V7_DCH_失败分析与不改主线修改计划.md)
- V8/NER V1–V3：[`SCTransNet_TPD_V8_MPRS_DCH_不改主线失败分析与代码修改方案.md`](SCTransNet_TPD_V8_MPRS_DCH_不改主线失败分析与代码修改方案.md)
- NER V4：[`SCTransNet_NER_V3失败复盘与V4_Tail_Aware修改方案.md`](SCTransNet_NER_V3失败复盘与V4_Tail_Aware修改方案.md)
- TSS/QFG：[`SCTransNet_TSS混合结果复盘与QFG_V2最终模型集成方案.md`](SCTransNet_TSS混合结果复盘与QFG_V2最终模型集成方案.md)
- 固定 seed42 工程认证：[`SCTransNet_最终模型稳定性认证与论文级闭环方案.md`](SCTransNet_最终模型稳定性认证与论文级闭环方案.md)
- 四数据集 best-mIoU 表：[`results/four_dataset_seed42_v1/tables/table2_best_miou.md`](results/four_dataset_seed42_v1/tables/table2_best_miou.md)
- 四数据集 best-Pd 表：[`results/four_dataset_seed42_v1/tables/table4a_best_pd.md`](results/four_dataset_seed42_v1/tables/table4a_best_pd.md)
- 四数据集机器可读汇总：`results/four_dataset_seed42_v1/paper_results_summary.json`
- 三数据集正 TSS 结果：`results/three_dataset_seed42_global_tss_v2/`
- 三数据集 TSS-off 结果与裁决：`results/three_dataset_tss_off_seed42_v1/`
- EC-TSS V3.1 方案：[`SCTransNet_EC-TSS_V3性能提升与下一步方案.md`](SCTransNet_EC-TSS_V3性能提升与下一步方案.md)
- EC-TSS V3.1 最终比较：[`results/three_dataset_ec_tss_v3_1_seed42/comparison/ec_tss_v3_1_final_comparison.md`](results/three_dataset_ec_tss_v3_1_seed42/comparison/ec_tss_v3_1_final_comparison.md)
- 36 点联合像素 sidecar：[`results/three_dataset_ec_tss_v3_1_seed42/comparison/additive_joint_metrics_v1.md`](results/three_dataset_ec_tss_v3_1_seed42/comparison/additive_joint_metrics_v1.md)
- GCSF 固定分支重分配裁决：[`results/three_dataset_gcsf_branch_audit_v1/comparison/seed42_six_role/decision.md`](results/three_dataset_gcsf_branch_audit_v1/comparison/seed42_six_role/decision.md)
- DS-GA 六头梯度审计裁决：[`results/three_dataset_ds_gradient_audit_v1/comparison/seed42_six_role/decision.md`](results/three_dataset_ds_gradient_audit_v1/comparison/seed42_six_role/decision.md)
- DORF V1 十二角色裁决：[`results/three_dataset_dorf_v1/comparison/seed42_twelve_role/decision.md`](results/three_dataset_dorf_v1/comparison/seed42_twelve_role/decision.md)
- NER-L4-TPR 六角色筛选：[`results/three_dataset_ner_l4_tpr_zero_training_v1/comparison/seed42_six_role/decision.md`](results/three_dataset_ner_l4_tpr_zero_training_v1/comparison/seed42_six_role/decision.md)
- NER-L4-TPR 三数据集正式训练：[`results/three_dataset_l4_tpr_tss_off_seed42_v1/`](results/three_dataset_l4_tpr_tss_off_seed42_v1/)
- NER-L4-TPR 训练后比较器：[`analysis/compare_three_dataset_ner_l4_tpr_posttraining_v1.py`](analysis/compare_three_dataset_ner_l4_tpr_posttraining_v1.py)

本文件只汇总已经落盘并完成口径核对的正式结果；后续模型完成后，应追加对应结果，
不覆盖或改写历史表。
