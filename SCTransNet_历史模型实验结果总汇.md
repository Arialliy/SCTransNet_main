# SCTransNet 历史模型实验结果总汇

更新时间：2026-08-04

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

当前总裁决仍为：

```text
decision = INCONCLUSIVE_MIXED_TRADEOFF
ec_tss_v3_1_decision = EC_TSS_V3_1_PERFORMANCE_FAIL_STOP_TSS_OPTIMIZATION
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

## 12. 权威结果来源

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

本文件只汇总已经落盘的正式结果；后续模型完成后，应追加对应正式结果，不覆盖或
改写历史表。
