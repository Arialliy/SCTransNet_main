# SCTransNet 历史模型实验结果总汇

更新时间：2026-08-08

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
9. PBDR-V2 在 NUAA-SIRST 失败后停止向 NUDT-SIRST、IRSTD-1K 扩展。PBDR-V3
   Stage-1 已完成三数据集同角色 Original 对比：NUDT 两角色均保留 Original；
   IRSTD-1K 的 `best_miou` 保留 Original、`best_pd` 采用 PBDR-V3；NUAA 的指标级
   结果为 `best_miou` 候选胜、`best_pd` Original 胜，但因历史 Original 缺少完全
   匹配的双 TF32-off 精度证明，只能作为 advisory，不能绑定替换。
10. PBDR-V4 已完成 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 的五族联合正式评估。
    六个角色的胜者计数为 Original 4、Current 2，V3-calibrated、V4-Stage1、
    V4-Stage2 均为 0。V4 两阶段只在 NUAA `best_miou` 相对 Original 严格提高
    第一排序项 mIoU，但仍低于 Current，因此没有突破 Original/Current 既有包络。
    本轮没有性能接受门槛；正式胜者属于 official-test 运营选择，必须按 optimistic
    selection 解读。
11. IRSTD-BGCR 已完成严格 train-only 的 3-fold OOF：800 张 official-train 图像按
    267/267/266 划分，epoch 0 与每 5 epoch 至 120 均在完整 held fold 上评估并精确
    汇总。最终选择为 **epoch 0（Current）**；训练后 mIoU 最佳的 epoch 5 仍低
    Current 0.607261 pp，epoch 120 低 1.617915 pp。因此 BGCR 不替换 Current，
    full-selected 只生成 epoch-0 identity 审计候选。该结论没有访问 official test，
    也不能与用户给定的独立训练 Baseline official 历史向量混为同一口径。

当前总裁决仍为：

```text
decision = INCONCLUSIVE_MIXED_TRADEOFF
ec_tss_v3_1_decision = EC_TSS_V3_1_PERFORMANCE_FAIL_STOP_TSS_OPTIMIZATION
ner_stage2_decision = DO_NOT_AUTHORIZE_NER_V5_PER_DEVELOPMENT_TRAINING
qfg_decision = QFG_INCONCLUSIVE_NO_FORMULA_CHANGE
tpd_decision = TPD_INCONCLUSIVE_NO_FORMULA_CHANGE
pbdr_v3_cross_dataset_status = complete
pbdr_v3_nudt_selected = best_miou:original,best_pd:original
pbdr_v3_irstd_selected = best_miou:original,best_pd:candidate
pbdr_v3_nuaa_status = advisory_complete_binding_blocked
pbdr_v4_cross_dataset_status = complete
pbdr_v4_nuaa_selected = best_miou:current,best_pd:original
pbdr_v4_nudt_selected = best_miou:original,best_pd:original
pbdr_v4_irstd_selected = best_miou:original,best_pd:current
pbdr_v4_family_wins = original:4,current:2,v3_calibrated:0,v4_stage1:0,v4_stage2:0
pbdr_v4_envelope_breakthrough = false
pbdr_v4_operational_test_selected = true
pbdr_v4_selection_is_optimistic = true
irstd_bgcr_oof_status = complete_train_only
irstd_bgcr_selected_epoch = 0
irstd_bgcr_strictly_improves_current = false
irstd_bgcr_deployment_decision = KEEP_CURRENT
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
| IRSTD-BGCR train-only OOF | IRSTD-1K official-train 800 张，固定 3-fold 267/267/266 | 120；每 5 epoch | 42 | 三折 held-fold 充分统计精确 pooled；含 epoch 0 | 仅用于开发期选择，不是 official-test 结果 |

多数据集 checkpoint 是 `test_selected=true`、`selection_is_optimistic=true`。因此它们是
当前固定协议下的数据集内比较结果，不是独立测试或跨随机性结论。
BGCR OOF 也必须单独读表：它与用户给定的独立训练 Baseline official 历史结果使用
不同样本范围和选择协议，禁止直接拼成一个“单 checkpoint 全指标向量”。

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

## 18. PBDR-V2 自适应证据残差路由 formal1000（NUAA 后停止扩展）

PBDR-V2 在冻结的 `TPD8 + NER4 + QFG2-CROA + TSS-off` 主干上增加 19 个
readout 参数。训练图为 573 state keys，推理图为 569 state keys。三个数据集使用
seed42、各自 `img_idx`、1000 epochs、epoch 10 起每 10 epochs 评估、固定阈值 0.5，
并分别保存自己的 `best_miou` 与 `best_pd`。

当前执行状态：

```text
NUAA-SIRST=1000/1000 complete
NUDT-SIRST=not_launched
IRSTD-1K=not_launched
cross_dataset_extension=stopped_after_NUAA_failure
```

这里覆盖 2026-08-05 曾记录的“NUDT 运行中、IRSTD 排队中”临时状态。后续失败分析
明确要求不启动这两套 PBDR-V2 正式训练；实际跨数据集继续实验使用的是修正后的
PBDR-V3，见第 19 节。

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

## 19. PBDR-V3 保守双门校准器：三数据集 Stage-1 正式结果

PBDR-V3 在 Current 主干之后增加有界的 rescue/suppression 双门校准器；Stage-1 只训练
PBDR-V3 参数并冻结主干。正式比较严格使用同数据集、同角色 Original checkpoint，
二值化固定为 `probability > 0.5`，未做阈值搜索。这里不设置正增益门槛：按冻结的
角色优先序，在第一个不同项上只要严格更好即获胜。NUDT/IRSTD 的 epoch 不参与
跨模型比较，六项指标完全相同则保留 Original。

- `best_miou`：mIoU ↑ → Pd ↑ → Fa ↓ → nIoU ↑ → tiny-Pd ↑ → test loss ↓。
- `best_pd`：Pd ↑ → Fa ↓ → tiny-Pd ↑ → mIoU ↑ → nIoU ↑ → test loss ↓。

NUAA 的 post-hoc advisory 在上述六项之后还有 `earlier_epoch` 第七项；本次两个角色
都在第一项即完成裁决，因此 epoch 没有影响实际胜者。

NUDT-SIRST 和 IRSTD-1K 的 Original 均在同一 FP32、CUDA matmul TF32-off、cuDNN
TF32-off 实现下重新评估；历史 Original checkpoint 本身仍是
`test_selected=true`、`selection_is_optimistic=true`。NUAA 使用权威历史固定 0.5
结果，但该历史产物没有明确证明两个 TF32 开关均关闭，所以只允许指标级 advisory。
这项 NUAA 比较是 `post_hoc_not_preregistered`，发生在 official result 之后；adjudicator
没有重新加载数据或模型、没有重访 official test，也没有覆盖此前的部署产物。

### 19.1 角色裁决与 checkpoint

| 数据集 | 角色 | Candidate epoch | Original epoch | 第一个决定项 | 指标胜者 | 绑定部署 |
|---|---|---:|---:|---|---|---|
| NUAA-SIRST | best_miou | 30 | 830 | mIoU `0.795395869 > 0.786824655` | **Candidate（advisory）** | 不绑定；沿用此前 Current-based 部署 |
| NUAA-SIRST | best_pd | 25 | 440 | Pd `257/263 < 260/263` | **Original（advisory）** | 不绑定；沿用此前 Current-based 部署 |
| NUDT-SIRST | best_miou | 95 | 520 | mIoU `0.944794189 < 0.945572339` | **Original** | **Original** |
| NUDT-SIRST | best_pd | 90 | 260 | Pd `940/945 < 941/945` | **Original** | **Original** |
| IRSTD-1K | best_miou | 15 | 270 | mIoU `0.662016336 < 0.673484761` | **Original** | **Original** |
| IRSTD-1K | best_pd | 20 | 230 | Pd 同为 `287/297`；Fa `2.419770654e-5 < 4.919251399e-5` | **Candidate** | **PBDR-V3 Candidate** |

因此四个具有绑定资格的 NUDT/IRSTD 角色中，PBDR-V3 只在 IRSTD-1K
`best_pd` 获得部署胜利。NUDT `best_pd` 虽然 Candidate 的 mIoU、nIoU 和 Fa 更好，
但 Pd 少检 1 个目标；Pd 是该角色的第一排序项，后续指标不能覆盖这一回退。

### 19.2 全部核心检测、重叠与损失指标

以下每格均为 `Candidate / Original（Candidate - Original）`。mIoU、nIoU、Pd、
tiny-Pd 越高越好；Fa、test loss 越低越好。

| 数据集 / 角色 | mIoU ↑ | nIoU ↑ | Pd ↑ | Fa ↓ | tiny-Pd ↑ | test loss ↓ |
|---|---:|---:|---:|---:|---:|---:|
| NUAA / best_miou | `0.795395869 / 0.786824655 (+0.008571214)` | `0.792876492 / 0.795095699 (-0.002219207)` | `0.973384030 / 0.969581749 (+0.003802281)` | `1.570959557e-5 / 2.654853051e-5 (-1.083893494e-5)` | `0.857142857 / 0.914285714 (-0.057142857)` | `4.751602795e-4 / 4.942663793e-4 (-1.910609977e-5)` |
| NUAA / best_pd | `0.789941470 / 0.726235741 (+0.063705728)` | `0.793992233 / 0.748162904 (+0.045829329)` | `0.977186312 / 0.988593156 (-0.011406844)` | `1.474918362e-5 / 8.101760860e-5 (-6.626842499e-5)` | `0.857142857 / 0.971428571 (-0.114285714)` | `4.950236295e-4 / 5.762233480e-4 (-8.119971853e-5)` |
| NUDT / best_miou | `0.944794189 / 0.945572339 (-0.000778150)` | `0.946810964 / 0.947406904 (-0.000595939)` | `0.990476190 / 0.989417989 (+0.001058201)` | `2.780592585e-6 / 2.504831337e-6 (+2.757612481e-7)` | `0.996138996 / 0.996138996 (0)` | `1.905135181e-4 / 2.382386971e-4 (-4.772517899e-5)` |
| NUDT / best_pd | `0.937875478 / 0.915652026 (+0.022223451)` | `0.940364516 / 0.925479141 (+0.014885375)` | `0.994708995 / 0.995767196 (-0.001058201)` | `6.526349539e-6 / 1.381104251e-5 (-7.284692971e-6)` | `0.996138996 / 0.996138996 (0)` | `2.896802722e-4 / 1.944135359e-4 (+9.526673635e-5)` |
| IRSTD / best_miou | `0.662016336 / 0.673484761 (-0.011468425)` | `0.668341492 / 0.636851511 (+0.031489981)` | `0.936026936 / 0.949494949 (-0.013468013)` | `1.210834256e-5 / 2.211006127e-5 (-1.000171870e-5)` | `0.766666667 / 0.766666667 (0)` | `7.207705747e-4 / 2.842100198e-4 (+4.365605550e-4)` |
| IRSTD / best_pd | `0.641038931 / 0.619140625 (+0.021898306)` | `0.652740697 / 0.627173613 (+0.025567084)` | `0.966329966 / 0.966329966 (0)` | `2.419770654e-5 / 4.919251399e-5 (-2.499480746e-5)` | `0.833333333 / 0.800000000 (+0.033333333)` | `5.596538098e-4 / 4.431423889e-4 (+1.165114210e-4)` |

### 19.3 全部像素分类指标

以下每格为 `Candidate / Original`。

| 数据集 / 角色 | Pixel Precision ↑ | Pixel Recall ↑ | Pixel F1 ↑ |
|---|---:|---:|---:|
| NUAA / best_miou | `0.890628764 / 0.899242142` | `0.881497377 / 0.862899380` | `0.886039545 / 0.880695991` |
| NUAA / best_pd | `0.896874231 / 0.799420476` | `0.868860277 / 0.888054363` | `0.882645028 / 0.841409692` |
| NUDT / best_miou | `0.978680712 / 0.980281589` | `0.964647713 / 0.963906057` | `0.971613546 / 0.972024859` |
| NUDT / best_pd | `0.965526935 / 0.958463505` | `0.970369062 / 0.953487551` | `0.967941943 / 0.955969053` |
| IRSTD / best_miou | `0.832196657 / 0.810134523` | `0.764001651 / 0.799711022` | `0.796642393 / 0.804889027` |
| IRSTD / best_pd | `0.797292458 / 0.709924564` | `0.765859364 / 0.828815192` | `0.781259870 / 0.764776840` |

### 19.4 全部目标计数、预测组件与样本规模

以下成对值均为 `Candidate / Original`。NUAA 的历史 adjudication 额外保存了
unmatched predicted pixels；NUDT/IRSTD 的本次 evaluation schema 未单列该字段，
因此不由 Fa 反推并伪装成原始记录。

| 数据集 / 角色 | Pd 命中/总数 | tiny 命中/总数 | 预测对象数 | 未匹配预测对象数 ↓ | 错误对象/图 ↓ | 未匹配预测像素 ↓ | 有效像素数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA / best_miou | `256/263 / 255/263` | `30/35 / 32/35` | `278 / 282` | `22 / 27` | `0.102803738 / 0.126168224` | `229 / 387` | `14,577,078` |
| NUAA / best_pd | `257/263 / 260/263` | `30/35 / 34/35` | `275 / 367` | `18 / 107` | `0.084112150 / 0.500000000` | `215 / 1,181` | `14,577,078` |
| NUDT / best_miou | `936/945 / 935/945` | `258/259 / 258/259` | `961 / 958` | `25 / 23` | `0.037650602 / 0.034638554` | 未单列 | `43,515,904` |
| NUDT / best_pd | `940/945 / 941/945` | `258/259 / 258/259` | `981 / 998` | `41 / 57` | `0.061746988 / 0.085843373` | 未单列 | `43,515,904` |
| IRSTD / best_miou | `278/297 / 282/297` | `23/30 / 23/30` | `322 / 364` | `44 / 82` | `0.218905473 / 0.407960199` | 未单列 | `52,690,944` |
| IRSTD / best_pd | `287/297 / 287/297` | `25/30 / 24/30` | `382 / 428` | `95 / 141` | `0.472636816 / 0.701492537` | 未单列 | `52,690,944` |

测试样本规模分别为 NUAA-SIRST 214 图、NUDT-SIRST 664 图、IRSTD-1K 201 图；
所有表项阈值均为 0.5。NUDT/IRSTD 的正式测试各只构造一次 loader、完整遍历一次，
没有结果驱动重试、正增益门槛、阈值扫描或第二次 official-test pass。

### 19.5 结果解释与口径限制

- NUAA `best_miou` 的第一项 mIoU 严格提高，所以指标级判 Candidate；其 nIoU 和
  tiny-Pd 仍回退。`best_pd` 的 Pd 少 3 个目标，因此由 Original 获胜，不能用后续
  mIoU/Fa 改善覆盖第一排序项。
- NUDT `best_miou` 的 Candidate 多检 1 个目标且 loss 更低，但 mIoU 首项略低；
  `best_pd` 的 Candidate 大幅提高 mIoU/nIoU并降低 Fa，却少检 1 个目标。两角色均
  按冻结角色顺序保留 Original。
- IRSTD `best_miou` 的 Candidate 降低 Fa并提高 nIoU，但 mIoU 和 Pd 都回退；
  `best_pd` 的 Pd 完全持平，Candidate 在第二项 Fa 上严格更低，同时 mIoU、nIoU、
  tiny-Pd、Precision、F1 和错误对象数也更好，因此是唯一绑定采用的 PBDR-V3 点。
- NUDT/IRSTD 的 Original 数值来自 PBDR-V3 裁决使用的匹配精度重评；NUAA Original
  来自未重访 official test 的权威历史 fixed-0.5 产物。NUDT/IRSTD 同一 checkpoint
  与第 8、9 节历史 summary 存在末位差异时，以本节裁决产物为准，不把差异解释成
  模型本身发生变化。

### 19.6 正式产物

- 跨数据集最终报告：[`FINAL_RUN_REPORT.md`](results/two_dataset_pbdr_v3_stage1_v1/FINAL_RUN_REPORT.md)
- NUDT-SIRST 正式 evaluation：[`evaluation.json`](results/two_dataset_pbdr_v3_stage1_v1/runs/NUDT-SIRST/formal/evaluation.json)
- IRSTD-1K 正式 evaluation：[`evaluation.json`](results/two_dataset_pbdr_v3_stage1_v1/runs/IRSTD-1K/formal/evaluation.json)
- NUAA-SIRST Original 零门槛 adjudication：[`original_zero_margin_role_adjudication_v1.json`](results/nuaa_pbdr_v3_stage1_v1/original_zero_margin_role_adjudication_v1.json)
- 跨数据集冻结协议：[`PBDR_V3_CROSS_DATASET_PROTOCOL.md`](experiments/PBDR_V3_CROSS_DATASET_PROTOCOL.md)

## 20. PBDR-V4 角色对齐组件校准器：三数据集五族零门槛正式结果

PBDR-V4 将 Original、Current TSS-off、内部选择的 V3 residual recalibration、
V4-Stage1 和 V4-Stage2 同时放入每个数据集/角色的冻结候选池。二值化工作点固定为
<code>probability > 0.5</code>。这里没有性能接受门槛：
<code>performance_acceptance_margin=null</code>，没有最小增益、百分比、epsilon、
non-regression gate 或 materiality threshold。同角色从左到右找到第一个不同指标，
只要严格更好即获胜；完整 role key 相同才按固定族序保留较早者：
<code>Original > Current > V3-calibrated > V4-Stage1 > V4-Stage2</code>。

- <code>best_miou</code>：mIoU ↑ → Pd ↑ → Fa ↓ → nIoU ↑ → tiny-Pd ↑ → loss ↓。
- <code>best_pd</code>：Pd ↑ → Fa ↓ → tiny-Pd ↑ → mIoU ↑ → nIoU ↑ → loss ↓。

需要明确披露：三份正式 bundle 均为 <code>operational_test_selected=true</code>、
<code>selection_is_optimistic=true</code>。因此这些结果是冻结协议下从 official
test 五族候选中产生的运营选择，不是独立未见测试集结果，也不能单独支持跨 seed
泛化结论。

### 20.1 内部选择与 official 前冻结

Stage-1 训练 150 epochs、每 5 epochs 内部验证，只更新 <code>pbdr_v4.*</code>；
Stage-2 从 Stage-1 冻结选中点出发训练 50 epochs，并以低学习率预定义解冻
<code>outc.*</code> 与 <code>up_decoder1.*</code>。Stage-2 是固定并行候选，不由
性能幅度门触发。12 份训练摘要均为 <code>complete</code>、
<code>official_test_accessed=false</code>、
<code>performance_acceptance_margin=null</code>。

| 数据集 | 角色 | 阶段 | epoch | mIoU | nIoU | Pd | Fa | tiny-Pd | loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | best_miou | Stage1 | 105 | 0.913967611 | 0.906658408 | 60/60；1.000000000 | 0.000000000e+00 | 8/8；1.000000000 | 1.764814913e-04 |
| NUAA-SIRST | best_miou | Stage2 | 35 | 0.916666667 | 0.909111530 | 60/60；1.000000000 | 0.000000000e+00 | 8/8；1.000000000 | 1.722234226e-04 |
| NUAA-SIRST | best_pd | Stage1 | 10 | 0.902426944 | 0.895346397 | 60/60；1.000000000 | 0.000000000e+00 | 8/8；1.000000000 | 2.009988869e-04 |
| NUAA-SIRST | best_pd | Stage2 | 30 | 0.899358658 | 0.891333933 | 60/60；1.000000000 | 0.000000000e+00 | 8/8；1.000000000 | 2.039926121e-04 |
| NUDT-SIRST | best_miou | Stage1 | 5 | 0.986327811 | 0.987690929 | 189/189；1.000000000 | 0.000000000e+00 | 39/39；1.000000000 | 2.854784728e-05 |
| NUDT-SIRST | best_miou | Stage2 | 5 | 0.986001609 | 0.987297722 | 189/189；1.000000000 | 0.000000000e+00 | 39/39；1.000000000 | 2.917018764e-05 |
| NUDT-SIRST | best_pd | Stage1 | 150 | 0.992134831 | 0.990743043 | 189/189；1.000000000 | 0.000000000e+00 | 39/39；1.000000000 | 1.738168952e-05 |
| NUDT-SIRST | best_pd | Stage2 | 35 | 0.993409420 | 0.992465045 | 189/189；1.000000000 | 0.000000000e+00 | 39/39；1.000000000 | 1.536360662e-05 |
| IRSTD-1K | best_miou | Stage1 | 135 | 0.787212787 | 0.725200894 | 228/230；0.991304348 | 5.435943604e-06 | 25/26；0.961538462 | 1.700513025e-04 |
| IRSTD-1K | best_miou | Stage2 | 50 | 0.786559293 | 0.725644235 | 228/230；0.991304348 | 5.602836609e-06 | 25/26；0.961538462 | 1.690291960e-04 |
| IRSTD-1K | best_pd | Stage1 | 120 | 0.703865337 | 0.655232536 | 230/230；1.000000000 | 2.794265747e-05 | 26/26；1.000000000 | 2.738195856e-04 |
| IRSTD-1K | best_pd | Stage2 | 10 | 0.707445475 | 0.657232916 | 230/230；1.000000000 | 2.670288086e-05 | 26/26；1.000000000 | 2.699564444e-04 |

按冻结角色序比较 Stage1/Stage2，内部验证中 Stage2 在 NUAA <code>best_miou</code>、
NUDT <code>best_pd</code>、IRSTD <code>best_pd</code> 更优，Stage1 在另外 3 个
角色更优。正式集的 V4 内部比较只有 IRSTD <code>best_miou</code> 发生反转，由
Stage2 超过 Stage1；但两个阶段最终均未赢得任何五族角色，内部选择信号没有转化为
对既有基线包络的正式胜利。

内部冻结的 V3 再标定配置与充分统计如下；六次 sweep 都只访问内部验证集，每次
固定评估 378 个预定义配置，未访问 official test：

| 数据集 | 角色 | positive scale | negative scale | bias | internal intersection/union | Pd | Fa pixels | tiny-Pd |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUAA-SIRST | best_miou | 0 | 0.25 | -0.1 | 1797/1968 | 60/60 | 0/2768375 | 8/8 |
| NUAA-SIRST | best_pd | 0 | 1.5 | +0.15 | 1793/1967 | 60/60 | 0/2768375 | 8/8 |
| NUDT-SIRST | best_miou | 3 | 1.5 | +0 | 6146/6221 | 189/189 | 0/8716288 | 39/39 |
| NUDT-SIRST | best_pd | 4 | 0.25 | -0.15 | 6174/6218 | 189/189 | 1/8716288 | 39/39 |
| IRSTD-1K | best_miou | 2 | 1.5 | +0 | 10838/13774 | 228/230 | 233/41943040 | 25/26 |
| IRSTD-1K | best_pd | 0 | 0 | -0.15 | 10160/13964 | 229/230 | 730/41943040 | 25/26 |

六个五族候选池均在 official pass 前冻结，候选数均为 5，
<code>official_test_accessed=false</code>。冻结时发现 V3 选中项名称带
<code>grid-NNN-</code> 前缀，而原 freezer 按无前缀名称比较。正式 amendment 只
修正该字符串合同；calibration 数值、模型状态、split、metric、训练产物和
evaluator 均未改变，并在 official test 访问前完成冻结。

### 20.2 五族正式胜者

| 数据集 | 角色 | 五族胜者 | 第一个决定项 |
|---|---|---|---|
| NUAA-SIRST | best_miou | **Current** | mIoU 0.796761047 为五族最高 |
| NUAA-SIRST | best_pd | **Original** | Pd 260/263（0.988593156）为五族最高 |
| NUDT-SIRST | best_miou | **Original** | mIoU 0.945572339 为五族最高 |
| NUDT-SIRST | best_pd | **Original** | Pd 941/945（0.995767196）为五族最高 |
| IRSTD-1K | best_miou | **Original** | mIoU 0.673484761 为五族最高 |
| IRSTD-1K | best_pd | **Current** | Pd 与 Original 同为 287/297；Fa 2.326775546e-5 < 4.919251399e-5 |

胜者计数为 <code>Original=4</code>、<code>Current=2</code>、
<code>V3-calibrated=0</code>、<code>V4-Stage1=0</code>、
<code>V4-Stage2=0</code>。本轮完整复现 Original/Current 两族包络；
“未突破包络”不表示 V4 每个单项都更差，也不表示 V4 被两族在全部指标上
Pareto 支配。

### 20.3 全部 30 个正式点：核心检测、重叠与损失指标

以下是三数据集 × 两角色 × 五族的全部绝对值；粗体族为该角色五族胜者。Fa 同时
列出未匹配组件像素数、有效像素数与比率，不能与全部 pixel FP 混为同一口径。

| 数据集 | 角色 | 族 | mIoU ↑ | nIoU ↑ | Pd ↑ | Fa ↓（unmatched/valid；值） | tiny-Pd ↑ | loss ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NUAA | best_miou | Original | 0.786824655 | 0.795095699 | 255/263；0.969581749 | 387/14577078；2.654853051e-05 | 32/35；0.914285714 | 4.941946333e-04 |
| NUAA | best_miou | **Current** | 0.796761047 | 0.795635585 | 256/263；0.973384030 | 225/14577078；1.543519216e-05 | 30/35；0.857142857 | 4.760764221e-04 |
| NUAA | best_miou | V3-calibrated | 0.795266272 | 0.792624930 | 256/263；0.973384030 | 229/14577078；1.570959557e-05 | 30/35；0.857142857 | 4.754221070e-04 |
| NUAA | best_miou | V4-Stage1 | 0.795042387 | 0.792385795 | 255/263；0.969581749 | 248/14577078；1.701301180e-05 | 30/35；0.857142857 | 4.827409685e-04 |
| NUAA | best_miou | V4-Stage2 | 0.795185902 | 0.793067370 | 256/263；0.973384030 | 229/14577078；1.570959557e-05 | 30/35；0.857142857 | 4.855870506e-04 |
| NUAA | best_pd | **Original** | 0.726164944 | 0.748136395 | 260/263；0.988593156 | 1182/14577078；8.108620946e-05 | 34/35；0.971428571 | 5.761868922e-04 |
| NUAA | best_pd | Current | 0.788553432 | 0.792667957 | 257/263；0.977186312 | 215/14577078；1.474918362e-05 | 30/35；0.857142857 | 4.945913217e-04 |
| NUAA | best_pd | V3-calibrated | 0.791017316 | 0.794048174 | 257/263；0.977186312 | 217/14577078；1.488638532e-05 | 30/35；0.857142857 | 4.932320414e-04 |
| NUAA | best_pd | V4-Stage1 | 0.790514250 | 0.792715242 | 256/263；0.973384030 | 244/14577078；1.673860838e-05 | 30/35；0.857142857 | 5.081712216e-04 |
| NUAA | best_pd | V4-Stage2 | 0.788894709 | 0.791059407 | 256/263；0.973384030 | 250/14577078；1.715021351e-05 | 30/35；0.857142857 | 5.137823374e-04 |
| NUDT | best_miou | **Original** | 0.945572339 | 0.947406904 | 935/945；0.989417989 | 109/43515904；2.504831337e-06 | 258/259；0.996138996 | 2.307008644e-04 |
| NUDT | best_miou | Current | 0.944373335 | 0.946329107 | 936/945；0.990476190 | 121/43515904；2.780592585e-06 | 258/259；0.996138996 | 1.726302978e-04 |
| NUDT | best_miou | V3-calibrated | 0.945120265 | 0.947188327 | 936/945；0.990476190 | 125/43515904；2.872513001e-06 | 258/259；0.996138996 | 1.721731227e-04 |
| NUDT | best_miou | V4-Stage1 | 0.944494491 | 0.946452136 | 936/945；0.990476190 | 120/43515904；2.757612481e-06 | 258/259；0.996138996 | 1.724867205e-04 |
| NUDT | best_miou | V4-Stage2 | 0.943983619 | 0.945995019 | 935/945；0.989417989 | 119/43515904；2.734632377e-06 | 257/259；0.992277992 | 1.731157868e-04 |
| NUDT | best_pd | **Original** | 0.915652026 | 0.925479141 | 941/945；0.995767196 | 601/43515904；1.381104251e-05 | 258/259；0.996138996 | 1.944075372e-04 |
| NUDT | best_pd | Current | 0.937382763 | 0.939827569 | 940/945；0.994708995 | 291/43515904；6.687210267e-06 | 258/259；0.996138996 | 2.235516537e-04 |
| NUDT | best_pd | V3-calibrated | 0.937745450 | 0.940372814 | 940/945；0.994708995 | 283/43515904；6.503369435e-06 | 258/259；0.996138996 | 2.228216258e-04 |
| NUDT | best_pd | V4-Stage1 | 0.937261840 | 0.940064407 | 940/945；0.994708995 | 295/43515904；6.779130683e-06 | 258/259；0.996138996 | 2.391608610e-04 |
| NUDT | best_pd | V4-Stage2 | 0.938474663 | 0.940902383 | 940/945；0.994708995 | 286/43515904；6.572309747e-06 | 258/259；0.996138996 | 2.352089295e-04 |
| IRSTD | best_miou | **Original** | 0.673484761 | 0.636851511 | 282/297；0.949494949 | 1165/52690944；2.211006127e-05 | 23/30；0.766666667 | 2.842009839e-04 |
| IRSTD | best_miou | Current | 0.660251398 | 0.665585204 | 277/297；0.932659933 | 618/52690944；1.172877070e-05 | 23/30；0.766666667 | 7.194182393e-04 |
| IRSTD | best_miou | V3-calibrated | 0.662042503 | 0.667671699 | 278/297；0.936026936 | 674/52690944；1.279157193e-05 | 23/30；0.766666667 | 7.223466873e-04 |
| IRSTD | best_miou | V4-Stage1 | 0.661479503 | 0.667361092 | 278/297；0.936026936 | 698/52690944；1.324705817e-05 | 23/30；0.766666667 | 7.222208308e-04 |
| IRSTD | best_miou | V4-Stage2 | 0.663588314 | 0.670794150 | 277/297；0.932659933 | 695/52690944；1.319012239e-05 | 23/30；0.766666667 | 7.146023454e-04 |
| IRSTD | best_pd | Original | 0.619140625 | 0.627173613 | 287/297；0.966329966 | 2592/52690944；4.919251399e-05 | 24/30；0.800000000 | 4.431502857e-04 |
| IRSTD | best_pd | **Current** | 0.639832723 | 0.650690649 | 287/297；0.966329966 | 1226/52690944；2.326775546e-05 | 25/30；0.833333333 | 5.571619307e-04 |
| IRSTD | best_pd | V3-calibrated | 0.637095826 | 0.647134645 | 286/297；0.962962963 | 1147/52690944；2.176844659e-05 | 25/30；0.833333333 | 5.584963549e-04 |
| IRSTD | best_pd | V4-Stage1 | 0.623871201 | 0.631785562 | 286/297；0.962962963 | 2064/52690944；3.917181670e-05 | 25/30；0.833333333 | 5.970769833e-04 |
| IRSTD | best_pd | V4-Stage2 | 0.625609377 | 0.631768912 | 286/297；0.962962963 | 1983/52690944；3.763455064e-05 | 25/30；0.833333333 | 5.914675829e-04 |

本节 Original 数值来自本轮 publication bundle 的新 canonical matcher 与联合 pass；
若与第 19 节历史裁决存在少量末位差异，以本节 bundle 为本轮五族比较的权威值。

### 20.4 全部 24 个非 Original 点相对 Original 的核心差值

所有差值均为“候选减 Original”。mIoU、nIoU、Pd、tiny-Pd 为正改善；Fa、loss
为负改善。<code>角色序结果</code> 严格按冻结 role key 在第一个不同项裁决，
不设置 epsilon 或任何正增益门槛。

| 数据集 | 角色 | 族 | ΔmIoU | ΔnIoU | ΔPd（命中数；值） | ΔFa（像素；值） | Δtiny-Pd（命中数；值） | Δloss | 角色序结果 vs O |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| NUAA | best_miou | Current | +0.009936392 | +0.000539886 | +1；+0.003802281 | -162；-1.111333835e-05 | -2；-0.057142857 | -1.811821117e-05 | **胜（mIoU）** |
| NUAA | best_miou | V3-calibrated | +0.008441617 | -0.002470769 | +1；+0.003802281 | -158；-1.083893494e-05 | -2；-0.057142857 | -1.877252627e-05 | **胜（mIoU）** |
| NUAA | best_miou | V4-Stage1 | +0.008217732 | -0.002709904 | +0；+0.000000000 | -139；-9.535518710e-06 | -2；-0.057142857 | -1.145366476e-05 | **胜（mIoU）** |
| NUAA | best_miou | V4-Stage2 | +0.008361247 | -0.002028329 | +1；+0.003802281 | -158；-1.083893494e-05 | -2；-0.057142857 | -8.607582675e-06 | **胜（mIoU）** |
| NUAA | best_pd | Current | +0.062388487 | +0.044531562 | -3；-0.011406844 | -967；-6.633702584e-05 | -4；-0.114285714 | -8.159557054e-05 | **负（Pd）** |
| NUAA | best_pd | V3-calibrated | +0.064852372 | +0.045911779 | -3；-0.011406844 | -965；-6.619982413e-05 | -4；-0.114285714 | -8.295485084e-05 | **负（Pd）** |
| NUAA | best_pd | V4-Stage1 | +0.064349305 | +0.044578847 | -4；-0.015209125 | -938；-6.434760108e-05 | -4；-0.114285714 | -6.801567061e-05 | **负（Pd）** |
| NUAA | best_pd | V4-Stage2 | +0.062729765 | +0.042923012 | -4；-0.015209125 | -932；-6.393599595e-05 | -4；-0.114285714 | -6.240455486e-05 | **负（Pd）** |
| NUDT | best_miou | Current | -0.001199004 | -0.001077797 | +1；+0.001058201 | +12；+2.757612481e-07 | +0；+0.000000000 | -5.807056660e-05 | **负（mIoU）** |
| NUDT | best_miou | V3-calibrated | -0.000452074 | -0.000218577 | +1；+0.001058201 | +16；+3.676816642e-07 | +0；+0.000000000 | -5.852774176e-05 | **负（mIoU）** |
| NUDT | best_miou | V4-Stage1 | -0.001077848 | -0.000954767 | +1；+0.001058201 | +11；+2.527811441e-07 | +0；+0.000000000 | -5.821414393e-05 | **负（mIoU）** |
| NUDT | best_miou | V4-Stage2 | -0.001588721 | -0.001411884 | +0；+0.000000000 | +10；+2.298010401e-07 | -1；-0.003861004 | -5.758507763e-05 | **负（mIoU）** |
| NUDT | best_pd | Current | +0.021730737 | +0.014348428 | -1；-0.001058201 | -310；-7.123832243e-06 | +0；+0.000000000 | +2.914411655e-05 | **负（Pd）** |
| NUDT | best_pd | V3-calibrated | +0.022093423 | +0.014893673 | -1；-0.001058201 | -318；-7.307673075e-06 | +0；+0.000000000 | +2.841408868e-05 | **负（Pd）** |
| NUDT | best_pd | V4-Stage1 | +0.021609814 | +0.014585267 | -1；-0.001058201 | -306；-7.031911827e-06 | +0；+0.000000000 | +4.475332381e-05 | **负（Pd）** |
| NUDT | best_pd | V4-Stage2 | +0.022822637 | +0.015423242 | -1；-0.001058201 | -315；-7.238732763e-06 | +0；+0.000000000 | +4.080139232e-05 | **负（Pd）** |
| IRSTD | best_miou | Current | -0.013233362 | +0.028733694 | -5；-0.016835017 | -547；-1.038129057e-05 | +0；+0.000000000 | +4.352172554e-04 | **负（mIoU）** |
| IRSTD | best_miou | V3-calibrated | -0.011442258 | +0.030820188 | -4；-0.013468013 | -491；-9.318489340e-06 | +0；+0.000000000 | +4.381457034e-04 | **负（mIoU）** |
| IRSTD | best_miou | V4-Stage1 | -0.012005258 | +0.030509581 | -4；-0.013468013 | -467；-8.863003100e-06 | +0；+0.000000000 | +4.380198469e-04 | **负（mIoU）** |
| IRSTD | best_miou | V4-Stage2 | -0.009896447 | +0.033942639 | -5；-0.016835017 | -470；-8.919938880e-06 | +0；+0.000000000 | +4.304013615e-04 | **负（mIoU）** |
| IRSTD | best_pd | Current | +0.020692098 | +0.023517036 | +0；+0.000000000 | -1366；-2.592475853e-05 | +1；+0.033333333 | +1.140116449e-04 | **胜（Fa）** |
| IRSTD | best_pd | V3-calibrated | +0.017955201 | +0.019961032 | -1；-0.003367003 | -1445；-2.742406741e-05 | +1；+0.033333333 | +1.153460692e-04 | **负（Pd）** |
| IRSTD | best_pd | V4-Stage1 | +0.004730576 | +0.004611949 | -1；-0.003367003 | -528；-1.002069729e-05 | +1；+0.033333333 | +1.539266975e-04 | **负（Pd）** |
| IRSTD | best_pd | V4-Stage2 | +0.006468752 | +0.004595299 | -1；-0.003367003 | -609；-1.155796336e-05 | +1；+0.033333333 | +1.483172972e-04 | **负（Pd）** |

V4 相对 Original 的角色序结果合计为 2 胜、10 负：仅 NUAA
<code>best_miou</code> 的 Stage1、Stage2 首项 mIoU 严格更高；两者仍被 Current
的更高 mIoU 覆盖。

### 20.5 全部 30 个正式点：像素分类与目标/组件明细

TP/FP/FN、Precision、Recall、F1、预测对象和未匹配对象均直接来自同一 bundle，
不由舍入比率反推。<code>组件像素</code> 是未匹配预测组件中的像素数；Fa 正是
该数除以 valid pixels，而不是全部 pixel FP 除以 valid pixels。

| 数据集 | 角色 | 族 | Precision ↑ | Recall ↑ | F1 ↑ | TP/FP/FN | intersection/union | 预测/未匹配对象 ↓ | 错误对象/图 ↓ | 组件像素 ↓ | valid pixels | 样本 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA | best_miou | Original | 0.899242142 | 0.862899380 | 0.880695991 | 7238/811/1150 | 7238/9199 | 282/27 | 0.126168224 | 387 | 14577078 | 214 |
| NUAA | best_miou | **Current** | 0.888105200 | 0.885670005 | 0.886885931 | 7429/936/959 | 7429/9324 | 277/21 | 0.098130841 | 225 | 14577078 | 214 |
| NUAA | best_miou | V3-calibrated | 0.890709724 | 0.881258941 | 0.885959130 | 7392/907/996 | 7392/9295 | 278/22 | 0.102803738 | 229 | 14577078 | 214 |
| NUAA | best_miou | V4-Stage1 | 0.888369305 | 0.883285646 | 0.885820182 | 7409/931/979 | 7409/9319 | 280/25 | 0.116822430 | 248 | 14577078 | 214 |
| NUAA | best_miou | V4-Stage2 | 0.889636932 | 0.882212685 | 0.885909254 | 7400/918/988 | 7400/9306 | 278/22 | 0.102803738 | 229 | 14577078 | 214 |
| NUAA | best_pd | **Original** | 0.799334693 | 0.888054363 | 0.841362173 | 7449/1870/939 | 7449/10258 | 367/107 | 0.500000000 | 1182 | 14577078 | 214 |
| NUAA | best_pd | Current | 0.898527410 | 0.865641392 | 0.881777886 | 7261/820/1127 | 7261/9208 | 275/18 | 0.084112150 | 215 | 14577078 | 214 |
| NUAA | best_pd | V3-calibrated | 0.895601029 | 0.871363853 | 0.883316212 | 7309/852/1079 | 7309/9240 | 276/19 | 0.088785047 | 217 | 14577078 | 214 |
| NUAA | best_pd | V4-Stage1 | 0.870224589 | 0.896161183 | 0.883002467 | 7517/1121/871 | 7517/9509 | 278/22 | 0.102803738 | 244 | 14577078 | 214 |
| NUAA | best_pd | V4-Stage2 | 0.866812478 | 0.897711016 | 0.881991215 | 7530/1157/858 | 7530/9545 | 278/22 | 0.102803738 | 250 | 14577078 | 214 |
| NUDT | best_miou | **Original** | 0.980281589 | 0.963906057 | 0.972024859 | 27293/549/1022 | 27293/28864 | 958/23 | 0.034638554 | 109 | 43515904 | 664 |
| NUDT | best_miou | Current | 0.978774515 | 0.964117959 | 0.971390955 | 27299/592/1016 | 27299/28907 | 962/26 | 0.039156627 | 121 | 43515904 | 664 |
| NUDT | best_miou | V3-calibrated | 0.977796847 | 0.965848490 | 0.971785943 | 27348/621/967 | 27348/28936 | 961/25 | 0.037650602 | 125 | 43515904 | 664 |
| NUDT | best_miou | V4-Stage1 | 0.980328694 | 0.962740597 | 0.971455044 | 27260/547/1055 | 27260/28862 | 962/26 | 0.039156627 | 120 | 43515904 | 664 |
| NUDT | best_miou | V4-Stage2 | 0.982019713 | 0.960586262 | 0.971184746 | 27199/498/1116 | 27199/28813 | 962/27 | 0.040662651 | 119 | 43515904 | 664 |
| NUDT | best_pd | **Original** | 0.958463505 | 0.953487551 | 0.955969053 | 26998/1170/1317 | 26998/29485 | 998/57 | 0.085843373 | 601 | 43515904 | 664 |
| NUDT | best_pd | Current | 0.964690604 | 0.970686915 | 0.967679470 | 27485/1006/830 | 27485/29321 | 980/40 | 0.060240964 | 291 | 43515904 | 664 |
| NUDT | best_pd | V3-calibrated | 0.965949064 | 0.969803991 | 0.967872689 | 27460/968/855 | 27460/29283 | 981/41 | 0.061746988 | 283 | 43515904 | 664 |
| NUDT | best_pd | V4-Stage1 | 0.962375546 | 0.972911884 | 0.967615033 | 27548/1077/767 | 27548/29392 | 983/43 | 0.064759036 | 295 | 43515904 | 664 |
| NUDT | best_pd | V4-Stage2 | 0.965253404 | 0.971287304 | 0.968260954 | 27502/990/813 | 27502/29305 | 983/43 | 0.064759036 | 286 | 43515904 | 664 |
| IRSTD | best_miou | **Original** | 0.810134523 | 0.799711022 | 0.804889027 | 11623/2724/2911 | 11623/17258 | 364/82 | 0.407960199 | 1165 | 52690944 | 201 |
| IRSTD | best_miou | Current | 0.839874531 | 0.755332324 | 0.795363159 | 10978/2093/3556 | 10978/16627 | 321/44 | 0.218905473 | 618 | 52690944 | 201 |
| IRSTD | best_miou | V3-calibrated | 0.823360987 | 0.771638916 | 0.796661339 | 11215/2406/3319 | 11215/16940 | 324/46 | 0.228855721 | 674 | 52690944 | 201 |
| IRSTD | best_miou | V4-Stage1 | 0.809090909 | 0.783817256 | 0.796253582 | 11392/2688/3142 | 11392/17222 | 323/45 | 0.223880597 | 698 | 52690944 | 201 |
| IRSTD | best_miou | V4-Stage2 | 0.809824213 | 0.786087794 | 0.797779485 | 11425/2683/3109 | 11425/17217 | 321/44 | 0.218905473 | 695 | 52690944 | 201 |
| IRSTD | best_pd | Original | 0.709924564 | 0.828815192 | 0.764776840 | 12046/4922/2488 | 12046/19456 | 428/141 | 0.701492537 | 2592 | 52690944 | 201 |
| IRSTD | best_pd | **Current** | 0.804146288 | 0.757946883 | 0.780363405 | 11016/2683/3518 | 11016/17217 | 380/93 | 0.462686567 | 1226 | 52690944 | 201 |
| IRSTD | best_pd | V3-calibrated | 0.814016375 | 0.745630934 | 0.778324416 | 10837/2476/3697 | 10837/17010 | 377/91 | 0.452736318 | 1147 | 52690944 | 201 |
| IRSTD | best_pd | V4-Stage1 | 0.713906112 | 0.831842576 | 0.768375226 | 12090/4845/2444 | 12090/19379 | 408/122 | 0.606965174 | 2064 | 52690944 | 201 |
| IRSTD | best_pd | V4-Stage2 | 0.717565879 | 0.829984863 | 0.769692136 | 12063/4748/2471 | 12063/19282 | 405/119 | 0.592039801 | 1983 | 52690944 | 201 |

### 20.6 全部 24 个非 Original 点相对 Original 的像素/目标差值

以下仍是“候选减 Original”；Precision、Recall、F1、TP 为正通常更好，FP、FN、
union、预测/未匹配对象和组件像素需结合任务角色与前表共同解释。

| 数据集 | 角色 | 族 | ΔPrecision | ΔRecall | ΔF1 | ΔTP | ΔFP | ΔFN | Δunion | Δ预测对象 | Δ未匹配对象 | Δ错误对象/图 | Δ组件像素 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA | best_miou | Current | -0.011136942 | +0.022770625 | +0.006189940 | +191 | +125 | -191 | +125 | -5 | -6 | -0.028037383 | -162 |
| NUAA | best_miou | V3-calibrated | -0.008532418 | +0.018359561 | +0.005263139 | +154 | +96 | -154 | +96 | -4 | -5 | -0.023364486 | -158 |
| NUAA | best_miou | V4-Stage1 | -0.010872837 | +0.020386266 | +0.005124191 | +171 | +120 | -171 | +120 | -2 | -2 | -0.009345794 | -139 |
| NUAA | best_miou | V4-Stage2 | -0.009605210 | +0.019313305 | +0.005213263 | +162 | +107 | -162 | +107 | -4 | -5 | -0.023364486 | -158 |
| NUAA | best_pd | Current | +0.099192717 | -0.022412971 | +0.040415713 | -188 | -1050 | +188 | -1050 | -92 | -89 | -0.415887850 | -967 |
| NUAA | best_pd | V3-calibrated | +0.096266337 | -0.016690510 | +0.041954039 | -140 | -1018 | +140 | -1018 | -91 | -88 | -0.411214953 | -965 |
| NUAA | best_pd | V4-Stage1 | +0.070889896 | +0.008106819 | +0.041640294 | +68 | -749 | -68 | -749 | -89 | -85 | -0.397196262 | -938 |
| NUAA | best_pd | V4-Stage2 | +0.067477786 | +0.009656652 | +0.040629042 | +81 | -713 | -81 | -713 | -89 | -85 | -0.397196262 | -932 |
| NUDT | best_miou | Current | -0.001507074 | +0.000211902 | -0.000633904 | +6 | +43 | -6 | +43 | +4 | +3 | +0.004518072 | +12 |
| NUDT | best_miou | V3-calibrated | -0.002484742 | +0.001942433 | -0.000238916 | +55 | +72 | -55 | +72 | +3 | +2 | +0.003012048 | +16 |
| NUDT | best_miou | V4-Stage1 | +0.000047105 | -0.001165460 | -0.000569815 | -33 | -2 | +33 | -2 | +4 | +3 | +0.004518072 | +11 |
| NUDT | best_miou | V4-Stage2 | +0.001738124 | -0.003319795 | -0.000840113 | -94 | -51 | +94 | -51 | +4 | +4 | +0.006024096 | +10 |
| NUDT | best_pd | Current | +0.006227099 | +0.017199364 | +0.011710418 | +487 | -164 | -487 | -164 | -18 | -17 | -0.025602410 | -310 |
| NUDT | best_pd | V3-calibrated | +0.007485560 | +0.016316440 | +0.011903637 | +462 | -202 | -462 | -202 | -17 | -16 | -0.024096386 | -318 |
| NUDT | best_pd | V4-Stage1 | +0.003912041 | +0.019424333 | +0.011645981 | +550 | -93 | -550 | -93 | -15 | -14 | -0.021084337 | -306 |
| NUDT | best_pd | V4-Stage2 | +0.006789900 | +0.017799753 | +0.012291901 | +504 | -180 | -504 | -180 | -15 | -14 | -0.021084337 | -315 |
| IRSTD | best_miou | Current | +0.029740009 | -0.044378698 | -0.009525869 | -645 | -631 | +645 | -631 | -43 | -38 | -0.189054726 | -547 |
| IRSTD | best_miou | V3-calibrated | +0.013226464 | -0.028072107 | -0.008227688 | -408 | -318 | +408 | -318 | -40 | -36 | -0.179104478 | -491 |
| IRSTD | best_miou | V4-Stage1 | -0.001043614 | -0.015893766 | -0.008635445 | -231 | -36 | +231 | -36 | -41 | -37 | -0.184079602 | -467 |
| IRSTD | best_miou | V4-Stage2 | -0.000310310 | -0.013623228 | -0.007109543 | -198 | -41 | +198 | -41 | -43 | -38 | -0.189054726 | -470 |
| IRSTD | best_pd | Current | +0.094221724 | -0.070868309 | +0.015586565 | -1030 | -2239 | +1030 | -2239 | -48 | -48 | -0.238805970 | -1366 |
| IRSTD | best_pd | V3-calibrated | +0.104091811 | -0.083184258 | +0.013547576 | -1209 | -2446 | +1209 | -2446 | -51 | -50 | -0.248756219 | -1445 |
| IRSTD | best_pd | V4-Stage1 | +0.003981548 | +0.003027384 | +0.003598387 | +44 | -77 | -44 | -77 | -20 | -19 | -0.094527363 | -528 |
| IRSTD | best_pd | V4-Stage2 | +0.007641316 | +0.001169671 | +0.004915296 | +17 | -174 | -17 | -174 | -23 | -22 | -0.109452736 | -609 |

### 20.7 结果解释与模型裁决

- NUAA <code>best_miou</code> 中 V4 两阶段都相对 Original 提高首项 mIoU；但
  Current 的 mIoU 为 0.796761047，仍高于 Stage1 的 0.795042387 和 Stage2 的
  0.795185902。
- NUAA <code>best_pd</code> 中 V4 的 mIoU、nIoU、Fa 和 loss 相对 Original
  改善，但少检 4 个目标且 tiny target 少检 4 个；Pd 是第一排序项。
- NUDT <code>best_miou</code> 中 Stage1 多检 1 个目标，但 mIoU 首项降低
  0.001077848；Stage2 的 mIoU 更低且 tiny target 少检 1 个。
- NUDT <code>best_pd</code> 中 Stage2 相对 Original 的 mIoU/nIoU 分别提高
  0.022822637/0.015423242，Fa 降低 7.238732763e-6，但少检 1 个目标。
- IRSTD <code>best_miou</code> 中 Stage2 的 nIoU 提高 0.033942639、Fa 更低，
  但 mIoU 下降 0.009896447 且少检 5 个目标；<code>best_pd</code> 两阶段也都
  少检 1 个总目标。
- IRSTD <code>best_pd</code> 的 Current 与 Original 同为 287/297，第二项 Fa
  从 4.919251399e-5 降至 2.326775546e-5，因此由 Current 获胜。

结论不是“V4 没有局部作用”，而是局部收益伴随角色首要指标回退，或已被更强的
Original/Current 工作点覆盖。按本轮冻结协议，不采用 V3-calibrated、V4-Stage1
或 V4-Stage2 作为新的统一部署族。

### 20.8 一次 official pass、完整性审计与边界

NUAA、NUDT、IRSTD 的正式样本数分别为 214、664、201。每个数据集只构造并完整
遍历一次 loader；同一遍历内联合评估两角色 × 五族共 10 个候选，forward 总数
分别为 2,140、6,640、2,010，且每个候选的 forward 数严格等于样本数。三份
bundle 均满足：

~~~text
loader_iteration_count=1
official_probability_or_logit_cache_written=false
official_sweep_performed=false
performance_acceptance_margin=null
operational_test_selected=true
selection_is_optimistic=true
~~~

因此没有 official 概率/logit 缓存、阈值 sweep、结果驱动重试或第二次
official-test pass。固定 <code>>0.5</code> 是指标测量工作点，不是性能提升门槛。

独立只读审计共执行 2,599 项断言，结果为 PASS、0 anomaly：30/30 指标恒等式、
135 个 source-lock 唯一文件和 30 个候选产物字节 SHA 均一致；审计未读取数据集
test 索引，也未构造 loader。claim preflight digest 的格式与自哈希有效，但
preflight body 未单独持久化，因此最终产物不能独立重算这一项 digest；这不影响
bundle、候选池、source lock、split、metric 恒等式与一次 pass 证据。

关键哈希如下：

| 产物 | SHA-256 |
|---|---|
| source lock（semantic） | <code>96af51690eb9270f76a2a37cbb778ede45f57dcf6fb36e2eba357dfacdef8ba6</code> |
| split projection（semantic） | <code>edf6fffb47f52693dbdd6c82209ff2b2259095b7aed5b91b28aab0e83112d1fb</code> |
| freezer amendment（semantic） | <code>30582e4c6623b5d514c08b3f0afefad6e50a807f28521f7cfe620f9c147f54ec</code> |
| NUAA-SIRST publication bundle | <code>a8bf36f7aa6d7df3de5132842a3e875d50cfb538acc46e32acbf803b462d8ea4</code> |
| NUAA-SIRST claim | <code>4f4f1e15b5af693296eb24002e9eca0df6de234bf46d790b6f6e16b922561b23</code> |
| NUAA-SIRST joint candidate pool | <code>686089b17789a6279401580bb521aaba7bcb686587a3cc13a1212ae21fe7ffdb</code> |
| NUDT-SIRST publication bundle | <code>30b081d7c2e96a7c1dfe314daad1ec7516543dcbb7c4ea94c2699070f6bbedcc</code> |
| NUDT-SIRST claim | <code>b9604cac598f6e4abfca2a71535310a84a27a1e49a35beba1a565c340172b1fd</code> |
| NUDT-SIRST joint candidate pool | <code>cceeaf671040257333e598e4eeddf4d3711b2c24f4da00e0925afef75786b7d5</code> |
| IRSTD-1K publication bundle | <code>96b705259db09276986236154238f0311a40878f56aeafb21a06f8b1d7c6a46f</code> |
| IRSTD-1K claim | <code>301de205bbbdee75815f16ccecd0fbbbc6f1760e2ec64e7c66614e67e784678a</code> |
| IRSTD-1K joint candidate pool | <code>597f4636483fe015771a1d60a7839f7b840f54b3c2cd7eabcddb9707a51323b7</code> |

### 20.9 正式产物

- V4 方案：[SCTransNet_PBDR零门槛失败分析与V4性能提升代码方案.md](SCTransNet_PBDR零门槛失败分析与V4性能提升代码方案.md)
- 冻结协议：[PBDR_V4_PROTOCOL.md](experiments/PBDR_V4_PROTOCOL.md)
- source lock：[source_lock.json](results/pbdr_v4_v1/protocol/source_lock.json)
- split projection：[split_projection.json](results/pbdr_v4_v1/protocol/split_projection.json)
- freezer amendment：[freeze_selected_grid_name_amendment_v1.json](results/pbdr_v4_v1/protocol/freeze_selected_grid_name_amendment_v1.json)
- 六个候选池：[candidate_pools/](results/pbdr_v4_v1/candidate_pools/)
- 六份 V3 内部校准：[v3_calibration/](results/pbdr_v4_v1/v3_calibration/)
- 十二份 V4 训练摘要：[training/](results/pbdr_v4_v1/training/)
- NUAA 正式 bundle：[publication_bundle.json](results/pbdr_v4_v1/official/NUAA-SIRST/publication_bundle.json)
- NUDT 正式 bundle：[publication_bundle.json](results/pbdr_v4_v1/official/NUDT-SIRST/publication_bundle.json)
- IRSTD 正式 bundle：[publication_bundle.json](results/pbdr_v4_v1/official/IRSTD-1K/publication_bundle.json)


## 21. PBDR-V5 目标保持型组件约束：失败定位、三数据集内部训练与最终裁决

本节记录 PBDR-V4 失败后的内部定位、较小 V5 目标保持型组件约束实现、三数据集
30-epoch 正式内部训练、空闲 GPU 迁移、全部内部评估点和最终模型裁决。V5 全程
只使用 development-train 与冻结 internal-validation；没有重新访问 official
test、test index 或 official loader。固定概率工作点仍为严格 <code>&gt;0.5</code>，
<code>performance_acceptance_margin=null</code>；这里的 0.5 是测量阈值，不是性能
接受门槛，任何完整 role key 上的严格提升都可以胜出。

### 21.1 用户给定的独立训练 Baseline 与可直接比较的完整模型

用户给定的独立训练 Baseline 同时包含“全程最佳 mIoU”与“epoch 1000 完整指标”。
这两个 checkpoint 不能拼接成一个单点向量。此前本节的直接比较表误把最佳 mIoU
与 epoch-1000 的 nIoU/F1/Pd/Fa 放在同一行；现按原始语义拆开。除 Fa 外均为
百分数，Fa 单位为 <code>×10⁻⁶</code>。

独立 Baseline 的最佳 checkpoint 只确认一个指标：

| 数据集 | 独立 Baseline 全程最佳 mIoU (%) |
|---|---:|
| NUAA-SIRST | **76.70** |
| NUDT-SIRST | **93.99** |
| IRSTD-1K | **67.74** |

独立 Baseline 的 epoch-1000 完整向量为：

| 数据集 | Epoch 1000 mIoU | nIoU | F1 | Pd | Fa |
|---|---:|---:|---:|---:|---:|
| NUAA-SIRST | 74.64 | 77.92 | 85.48 | 95.06 | 16.81 |
| NUDT-SIRST | 93.13 | 93.85 | 96.44 | 98.84 | 6.83 |
| IRSTD-1K | 66.65 | 66.82 | 79.97 | 93.27 | 11.60 |

上两表保留用户给定的公开显示精度。对 IRSTD-1K，本地 checkpoint/训练日志
还能绑定两个更精确的工作点：

| IRSTD 独立 Baseline checkpoint | mIoU | nIoU | F1 | Pd | Fa ×10⁻⁶ |
|---|---:|---:|---:|---:|---:|
| epoch 713 operational best | **67.7357929%** | 67.1640%* | 80.7586%* | 93.2659933% | 20.8005386 |
| epoch 1000 terminal | 66.6485671% | 66.8172%* | 79.9749%* | 93.2659933% | **11.5959201** |

`*` 表示训练日志只保留百分数四位小数。epoch 713 是从 epoch 500 起反复在
official test 上评估并按 test mIoU 选出的 operational best，不是未见测试集选择；
因此它只作历史对比目标，不做 BGCR teacher。

因此与 Current 完整模型的数值并列也分两张表。第一张只比较各自已报告的最佳
mIoU，不把其它 epoch-1000 指标带入：

| 数据集 | 独立 Baseline 最佳 mIoU → Current mIoU（差值） |
|---|---:|
| NUAA-SIRST | 76.7000 → 79.6761 (**+2.9761 pp**) |
| NUDT-SIRST | 93.9900 → 94.4373 (**+0.4473 pp**) |
| IRSTD-1K | 67.7357929 → 66.0251398 (**−1.7106531 pp**) |

第二张使用独立 Baseline 的 epoch-1000 完整向量；Current 仍是其正式 best-mIoU
checkpoint，因此这是明确标注 checkpoint 语义的数值并列，不声称两边选择时点相同：

| 数据集 | mIoU (%) | nIoU (%) | F1 (%) | Pd (%) | Fa (×10⁻⁶) |
|---|---:|---:|---:|---:|---:|
| NUAA-SIRST | 74.6400 → 79.6761 (**+5.0361 pp**) | 77.9200 → 79.5636 (**+1.6436 pp**) | 85.4800 → 88.6886 (**+3.2086 pp**) | 95.0600 → 97.3384 (**+2.2784 pp**) | 16.8100 → 15.4352 (**−1.3748**) |
| NUDT-SIRST | 93.1300 → 94.4373 (**+1.3073 pp**) | 93.8500 → 94.6329 (**+0.7829 pp**) | 96.4400 → 97.1391 (**+0.6991 pp**) | 98.8400 → 99.0476 (**+0.2076 pp**) | 6.8300 → 2.7806 (**−4.0494**) |
| IRSTD-1K | 66.6485671 → 66.0251398 (**−0.6234273 pp**) | 66.8172 → 66.5585 (**−0.2587 pp**) | 79.9749 → 79.5363 (**−0.4386 pp**) | 93.2659933 → 93.2659933 (**0.0000 pp**) | 11.5959201 → 11.7287707 (**+0.1328506**) |

直接结论：

- Current 完整模型在 NUAA-SIRST 与 NUDT-SIRST 相对独立 Baseline 成功；
- Current 在 IRSTD-1K 仍低于独立 Baseline epoch-713 operational best mIoU
  1.7106531 pp；若与 Baseline epoch-1000 mIoU 并列，差距为 0.6234273 pp；
- V5 只有内部验证结果，禁止把 V5 内部数值与上表独立 Baseline 直接相减，也
  不能把 V5 描述成新的 official 结果。

因此，如果必须选择一个统一、完整、当前可部署的设计模型，仍选择
<code>Current = TPD8 + NER4 + QFG2，TSS-off</code>；但它尚未解决 IRSTD-1K。

### 21.2 内部失败定位

development-train atlas 的组件监督分布如下：

| 数据集/角色 | preserve 组件/像素 | rescue 组件/像素 | suppress 组件/像素 | 决定性问题 |
|---|---:|---:|---:|---|
| NUDT / best_pd | 729 / 23,144 | **0 / 0** | 11 / 32 | 训练与内部验证都没有漏检救援信号 |
| NUAA / best_miou | 207 / 7,231 | 3 / 34 | 8 / 14 | 内部目标已全检出，主要是轮廓边界校准 |
| IRSTD / best_miou | 949 / 48,170 | 16 / 148 | 83 / 882 | rescue 稀少且与 peak-only 目标错配，halo/附着 FP 主导 |

具体定位如下：

- NUDT internal Current 已为 189/189、tiny 39/39，仅有 1 个未匹配像素。
  V4-Stage1 的 15 次上穿、0 次下穿把该像素通过新增像素并入匹配组件，暴露了
  通过“桥接”绕过 Fa 的拓扑捷径。V4-Stage2 虽达 189/189、Fa=0，但没有自然
  rescue 样本，不能从内部证据学会恢复 official 上相对 Original 少掉的目标。
- NUAA internal Current 已为 60/60、tiny 8/8、Fa=0。V4-Stage2 相对 Current
  的收益来自 TP +4、FP −10、FN −4 的边界像素变化；最弱目标峰值仍远高于
  0-logit 决策边界，因此不是目标峰值保活问题。
- IRSTD internal Current 为 228/230、tiny 25/26。两个漏检分别为
  XDU641 component 2（area=3，Current peak=−1.19466）和 XDU202 component 1
  （area=11，Current peak=−1.52051）。V4-Stage1 后仅到 −0.76039/−1.13811，
  V4-Stage2 后为 −0.77564/−1.12914；best-mIoU router 的正残差上限 +0.6
  在结构上不足以让它们越过 0。另有 atlas rescue 目标已经有正峰却因连通性/
  质心匹配失败，说明 peak-only rescue 与 Pd matcher 存在错配。Stage1 相对
  Current 虽 TP +442、FN −442，却 FP +484，新增 FP 主要是与真目标连通的
  边缘/halo，而不是独立假目标。

### 21.3 V5 设计、冻结训练协议与空闲 GPU 迁移

V5 是一个较小的目标保持型微调臂：

- 从不可变 V4-Stage1 selected checkpoint 初始化；
- 仅训练 <code>pbdr_v4.* + outc.* + up_decoder1.*</code>，其余 Current
  参数和 buffer 必须保持冻结；
- 保留 V4 的 BCE、role-Tversky、rescue、suppress、neutral 与 L2-SP；
- 用逐 preserve-component 的 frozen-Current smooth-peak 零 margin no-drop
  约束替换绝对峰值抬升；
- 对 Current 在各 preserve component 内的正支撑采用等组件、单侧 logit
  no-drop；背景项采用“每样本 active background 概率增加”再等样本平均；
- 30 epochs、每 5 epochs 内部评估、batch size 16、seed 42、FP32；
- epoch 0 必须进入候选池作为 fail-safe；固定家族顺序为
  Original → Current → V3-calibrated → V4-Stage1 → V4-Stage2 → V5，
  完全同 role key 时保留更早家族。

NUAA 在原绑定 GPU3 完整运行。NUDT/IRSTD 开始时分别与用户已有进程共享 GPU0/1；
用户要求改用空闲 GPU 后，在完整 rolling checkpoint 处停下，并迁移至完全空闲的
GPU2/GPU3。迁移只在独立 launcher 中改变硬件 allowlist，未改模型、loss、
optimizer、数据、随机状态或选择规则：

| 数据集 | 迁移源 epoch | 原 GPU UUID | 空闲 GPU UUID | 连续性 |
|---|---:|---|---|---|
| NUDT-SIRST | 15 | GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70 | GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562 | 源 4 条评估历史是目标 7 条历史的精确前缀 |
| IRSTD-1K | 11 | GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640 | GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3 | 源 [0,5,10] 与目标前缀逐项一致 |

两次迁移的 run identity、optimizer group signature 和 RNG rolling state 均连续；
expected/observed GPU UUID 一致。定向测试最终为 **38/38 passed**，只出现第三方
<code>thop</code> 的弃用警告。

### 21.4 三角色六族全部内部选择结果

下表仅为冻结 internal-validation。mIoU、nIoU、F1 均为百分数；Fa 单位为
<code>×10⁻⁶</code>。V3 使用 sweep 的 selected candidate，而不是 anchor。

| 数据集/角色 | 家族 | Epoch | mIoU | nIoU | F1 | Pd | tiny-Pd | Fa |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NUDT / best_pd | Original | — | 94.426902 | 94.815836 | 97.133577 | 189/189 | 39/39 | 4.474382 |
| NUDT / best_pd | Current | — | 99.260569 | 99.153471 | 99.628913 | 189/189 | 39/39 | 0.114728 |
| NUDT / best_pd | V3-calibrated | — | 99.292377 | 99.190865 | 99.644932 | 189/189 | 39/39 | 0.114728 |
| NUDT / best_pd | V4-Stage1 | 150 | 99.213483 | 99.074304 | 99.605189 | 189/189 | 39/39 | **0** |
| NUDT / best_pd | **V4-Stage2** | **35** | **99.340942** | **99.246504** | **99.669382** | **189/189** | **39/39** | **0** |
| NUDT / best_pd | V5 | 30 | 99.261163 | 99.146427 | 99.629212 | 189/189 | 39/39 | **0** |
| NUAA / best_miou | Original | — | 90.060852 | 89.557872 | 94.770544 | 60/60 | 8/8 | 0 |
| NUAA / best_miou | Current | — | 91.001011 | 89.984355 | 95.288512 | 60/60 | 8/8 | 0 |
| NUAA / best_miou | V3-calibrated | — | 91.310976 | 90.521519 | 95.458167 | 60/60 | 8/8 | 0 |
| NUAA / best_miou | V4-Stage1 | 105 | 91.396761 | 90.665841 | 95.505024 | 60/60 | 8/8 | 0 |
| NUAA / best_miou | **V4-Stage2** | **35** | **91.666667** | **90.911153** | **95.652174** | **60/60** | **8/8** | **0** |
| NUAA / best_miou | V5 | 10 | 91.430020 | 90.661203 | 95.523179 | 60/60 | 8/8 | 0 |
| IRSTD / best_miou | Original | — | 67.816491 | 63.260352 | 80.822201 | 221/230 | 25/26 | 27.322769 |
| IRSTD / best_miou | Current | — | 78.270510 | 71.933132 | 87.810945 | 228/230 | 25/26 | 5.531311 |
| IRSTD / best_miou | V3-calibrated | — | 78.684478 | 72.464001 | 88.070860 | 228/230 | 25/26 | 5.555153 |
| IRSTD / best_miou | **V4-Stage1** | **135** | **78.721279** | 72.520089 | **88.093907** | **228/230** | **25/26** | **5.435944** |
| IRSTD / best_miou | V4-Stage2 | 50 | 78.655929 | **72.564424** | 88.052974 | 228/230 | 25/26 | 5.602837 |
| IRSTD / best_miou | V5 | 0 | **78.721279** | 72.520089 | **88.093907** | **228/230** | **25/26** | **5.435944** |

V3 selected 配置分别为：NUDT <code>pos=4, neg=0.25, bias=−0.15</code>，
NUAA <code>pos=0, neg=0.25, bias=−0.10</code>，IRSTD
<code>pos=2, neg=1.5, bias=0</code>。

### 21.5 V5 全部 21 个内部评估点

以下包括 epoch 0 与每 5 epoch 的全部结果。ΔmIoU 为相对同一 V5 run 的 epoch 0，
单位为百分点；Fa 单位为 <code>×10⁻⁶</code>。

| 数据集 | Epoch | mIoU (%) | ΔmIoU pp | nIoU (%) | F1 (%) | Pd | tiny-Pd | Fa | TP/FP/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NUDT | 0 | 99.213483 | 0 | 99.074304 | 99.605189 | 189/189 | 39/39 | 0 | 6181/37/12 |
| NUDT | 5 | 99.229163 | +0.015680 | 99.095751 | 99.613090 | 189/189 | 39/39 | 0 | 6179/34/14 |
| NUDT | 10 | 99.229163 | +0.015680 | 99.095751 | 99.613090 | 189/189 | 39/39 | 0 | 6179/34/14 |
| NUDT | 15 | 99.197432 | −0.016051 | 99.069691 | 99.597099 | 189/189 | 39/39 | 0 | 6180/37/13 |
| NUDT | 20 | 99.229163 | +0.015680 | 99.088490 | 99.613090 | 189/189 | 39/39 | 0 | 6179/34/14 |
| NUDT | 25 | 99.261163 | +0.047680 | 99.139817 | 99.629212 | 189/189 | 39/39 | 0 | 6180/33/13 |
| NUDT | **30** | **99.261163** | **+0.047680** | **99.146427** | **99.629212** | **189/189** | **39/39** | **0** | **6180/33/13** |
| NUAA | 0 | 91.396761 | 0 | 90.665841 | 95.505024 | 60/60 | 8/8 | 0 | 1806/108/62 |
| NUAA | 5 | 91.253792 | −0.142969 | 90.454492 | 95.426910 | 60/60 | 8/8 | 0 | 1805/110/63 |
| NUAA | **10** | **91.430020** | **+0.033259** | **90.661203** | **95.523179** | **60/60** | **8/8** | **0** | **1803/104/65** |
| NUAA | 15 | 91.299949 | −0.096812 | 90.458097 | 95.452142 | 60/60 | 8/8 | 0 | 1805/109/63 |
| NUAA | 20 | 91.286727 | −0.110034 | 90.495839 | 95.444915 | 60/60 | 8/8 | 0 | 1802/106/66 |
| NUAA | 25 | 91.374937 | −0.021825 | 90.733565 | 95.493107 | 60/60 | 8/8 | 0 | 1801/103/67 |
| NUAA | 30 | 91.332995 | −0.063766 | 90.598357 | 95.470199 | 60/60 | 8/8 | 0 | 1802/105/66 |
| IRSTD | **0** | **78.721279** | **0** | **72.520089** | **88.093907** | **228/230** | **25/26** | **5.435944** | **11032/1883/1099** |
| IRSTD | 5 | 78.312743 | −0.408536 | 72.204649 | 87.837517 | 228/230 | 25/26 | 6.413460 | 11093/2034/1038 |
| IRSTD | 10 | 78.378761 | −0.342518 | 72.248238 | 87.879028 | 228/230 | 25/26 | 6.127357 | 11071/1994/1060 |
| IRSTD | 15 | 78.314956 | −0.406323 | 72.276085 | 87.838909 | 228/230 | 25/26 | 6.198883 | 11080/2017/1051 |
| IRSTD | 20 | 78.439031 | −0.282247 | 72.339609 | 87.916899 | 228/230 | 25/26 | 5.745888 | 11045/1950/1086 |
| IRSTD | 25 | 78.535927 | −0.185351 | 72.431265 | 87.977729 | 228/230 | 25/26 | 6.055832 | 11061/1953/1070 |
| IRSTD | 30 | 78.364023 | −0.357256 | 72.263696 | 87.869764 | 228/230 | 25/26 | 6.222725 | 11065/1989/1066 |

IRSTD 所有训练评估点的 Pd 均为 228/230、tiny-Pd 均为 25/26，没有救回任何
新目标。最接近的 epoch 25 相对 epoch 0 虽 TP +29、FN −29，却 FP +70，mIoU
仍下降 0.185351 pp；这再次确认瓶颈是边界/附着 halo FP，而不是目标保持失败。

### 21.6 零门槛内部裁决与最优模型

| 数据集/角色 | 既有五族包络 | V5 selected epoch | V5 相对包络 | 六族胜者 | V5 严格提升 |
|---|---|---:|---|---|---|
| NUDT / best_pd | V4-Stage2 | 30 | Pd/Fa/tiny-Pd 同；mIoU −0.079779 pp，nIoU −0.100078 pp，F1 −0.040170 pp | **V4-Stage2** | false |
| NUAA / best_miou | V4-Stage2 | 10 | mIoU −0.236646 pp，nIoU −0.249950 pp，F1 −0.128995 pp；Pd/Fa/tiny-Pd 同 | **V4-Stage2** | false |
| IRSTD / best_miou | V4-Stage1 | 0 | 全部指标与充分统计完全相同；冻结顺序保留更早家族 | **V4-Stage1** | false |

最终结论：

1. V5 在 **0/3** 个焦点角色上严格超过既有内部包络，停止采用 V5；
2. NUDT 的内部完整 role-key 顺序为
   V4-Stage2 > V5 > V4-Stage1 > V3-calibrated > Current > Original；
3. NUAA 的六族胜者仍是 V4-Stage2；V5 只比自己的 V4-Stage1 初始化提高
   0.033259 pp mIoU；
4. IRSTD 的 V5 selected epoch=0，等价于安全回退到 V4-Stage1；
5. V4-Stage2/V4-Stage1 是特定内部角色的最佳校准候选，不是已经证明可统一
   替代 Current 的完整模型；
6. 统一完整模型仍以 Current 为主；其 NUAA/NUDT 已胜独立 Baseline，IRSTD
   仍需要专项优化；
7. 已完成的一次 official 六角色联合评估中，Original 胜 4 个角色、Current
   胜 2 个角色、V4 胜 0 个角色。因此 V4 是本节三个焦点内部角色的局部最优，
   不是 official/统一部署最优。

### 21.7 是否继续优化与下一步

V5 本轮开发终止，不继续对相同结果做 epoch、权重或阈值 sweep。若继续，必须另立
预注册的新协议（例如 V5.1），而不是根据 official 结果回调：

- IRSTD：只在 rescue map 放宽正残差预算，或让 <code>outc/up_decoder1</code>
  真正抬高低峰；增加组件核心定位/质心约束与邻接外圈抑制；加入 attached-boundary
  FP/halo penalty，使 TP 增益不再被更多连通 FP 抵消；
- NUDT：atlas 没有自然 rescue 样本。只有在新协议预注册受控 logit erosion/
  peak-drop counterfactual hard-positive 臂时才继续，不能声称当前 V5 已学会救援；
- NUAA：目标已全检出且 Fa=0，不再增加统一 peak boost；若继续只应研究可跨 split
  外推的轻量 boundary-band 校准与对 Current 的蒸馏。

### 21.8 完整性审计、哈希与产物

三份 V5 summary、两份迁移 ledger 和统一 internal summary 的自哈希均已重放
通过；NUAA、NUDT、IRSTD 的 selected candidate 都可由严格 builder 从 595-key
训练状态导出并严格加载 591-key inference 状态。builder 元数据均为：

~~~text
dataset_loader_imported=false
dataset_index_accessed=false
official_test_data_accessed=false
official_test_accessed=false
performance_acceptance_margin=null
~~~

关键哈希：

| 产物 | SHA-256 |
|---|---|
| V5 source manifest | <code>c80489a2bf1b707d4f9ac99bc2b729fbeccbe8a0578b6b972954e2fa9eaeca08</code> |
| NUAA summary | <code>c4b06254fd1e5d243c6a80f6d570b1cebb219c525b4ad1a778aee045ebda5a14</code> |
| NUDT summary | <code>d0ede7330e2d2ca533b6952df52597bd6a9902dcf7e93577de1814fa1f7bb5d2</code> |
| IRSTD summary | <code>84f0814c923f683535d5f6dca50fb09394d39e68bb472cc13a222217d3eac581</code> |
| NUDT GPU migration ledger | <code>5a3b68eb596aa02631dc026835f99162013ac47e59092428b718e4290494ea3a</code> |
| IRSTD GPU migration ledger | <code>7b62084c385a33480f4acec50d59eb417534e6026ffb42d6c696a96dd8cd9b0b</code> |
| unified internal summary | <code>de755ae9d73bc801d52d8fc69cf186c627bd4d5351a95650bfacf05d73009151</code> |

正式内部产物：

- 失败定位：[failure_localization_bundle.json](results/pbdr_v5_v1/diagnostics/failure_localization_bundle.json)
- 冻结内部协议：[PBDR_V5_INTERNAL_PROTOCOL.md](experiments/PBDR_V5_INTERNAL_PROTOCOL.md)
- V5 loss：[pbdr_v5_target_preservation_loss.py](experiments/pbdr_v5_target_preservation_loss.py)
- V5 trainer：[train_three_dataset_pbdr_v5_v1.py](experiments/train_three_dataset_pbdr_v5_v1.py)
- 空闲 GPU 迁移入口：[resume_pbdr_v5_on_idle_gpu.py](experiments/resume_pbdr_v5_on_idle_gpu.py)
- NUAA summary：[summary.json](results/pbdr_v5_v1/training/NUAA-SIRST/best_miou/summary.json)
- NUDT summary：[summary.json](results/pbdr_v5_v1/training_idle_gpu/NUDT-SIRST/best_pd/summary.json)
- IRSTD summary：[summary.json](results/pbdr_v5_v1/training_idle_gpu/IRSTD-1K/best_miou/summary.json)
- 统一机器可读汇总：[internal_summary.json](results/pbdr_v5_v1/comparison/internal_summary.json)
- 统一人类可读汇总：[INTERNAL_SUMMARY.md](results/pbdr_v5_v1/comparison/INTERNAL_SUMMARY.md)


## 22. IRSTD-BGCR 正式 train-only 3-fold OOF：完整结果与不替换裁决

本节只汇总 canonical 目录
[`results/irstd_bgcr_v1/`](results/irstd_bgcr_v1/) 中的正式 train-only 产物。
`fold_0_pre_cpu_thread_fix`、`fold_1_pre_cpu_thread_fix` 和
`frozen_context_cache.incomplete_pre_binarization_contract` 均是被正式合同排除的前置产物，
不进入任何数值、选择或哈希结论。

### 22.1 数据、训练与选择口径

- 数据范围为 IRSTD-1K 的 800 张 <code>official_train_only</code> 图像；固定三折为
  267/267/266。每个 fold 用另外两折训练，并在完整 held fold 上评估；三个 held fold
  是 800 张图像的不重叠全集。
- seed=42、FP32、TF32-off；训练 epoch 为 1–120，epoch 0 和每 5 epoch 至 120
  评估。每个训练样本每 epoch 恰好出现一次；error-aware 三类与 counterfactual
  0/1/2 在这一前提下确定性平衡，不做 rescue 样本重复过采样。
- OOF 不是三个四舍五入后 fold 指标的算术平均。mIoU、Pd、Fa、像素
  TP/FP/FN 先汇总可加充分统计再计算；nIoU 与 loss 使用三折保存的精确有理数和。
- 固定 <code>probability_threshold=0.5</code>、比较为
  <code>strict_greater_than</code>。这是二值化测量工作点，不是性能接受门槛；
  <code>performance_acceptance_margin=null</code>，完整 role key 只需严格改善即可胜出。
- epoch 0 的两个 residual terminal 为精确零，因此 BGCR 输出与冻结 Current 位级一致。
  `Baseline-epoch1000` 只作为同一 train-only OOF 投影上的冻结参考 logits，不参与最终
  BGCR 推理，也不是第 21.1 节用户给定的独立训练 Baseline official 历史表。
- 正式 [`oof_selection.json`](results/irstd_bgcr_v1/oof_selection.json) 给出
  <code>selected_epoch=0</code>、<code>strictly_improves_epoch0_miou=false</code>、
  <code>strictly_improves_epoch0_full_role_key=false</code>。因此正式裁决是
  **保留 Current，不用训练后的 BGCR 替换 Current**。

### 22.2 Current、同投影 Baseline1000 与 BGCR 关键工作点

mIoU、nIoU、F1、Pd、tiny-Pd 为百分数；Fa 单位为 <code>×10⁻⁶</code>；
BCE loss 是逐样本均值的原始小数。TP/FP/FN 是逐像素统计，与 component-Fa
不是同一分子。`ΔmIoU` 均相对 epoch-0 Current。
本表所有差值都只在同一个 800-sample train-only OOF 投影内计算；**不计算这里的
Current/BGCR 与第 21.1 节 historical independent-Baseline official 数值之间的跨
split 差值**。

| OOF 模型/工作点 | Epoch | mIoU | nIoU | F1 | Pd | Fa | tiny-Pd | TP / FP / FN | BCE loss | ΔmIoU (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Current（正式 selected）** | **0** | **78.224854** | **72.858877** | **87.782206** | 98.493724 | **5.311966** | 90.769231 | 52977 / 7275 / 7472 | **0.000167799** | 0 |
| Baseline-epoch1000（冻结 OOF 参考） | 0 | 78.109365 | 73.124301 | 87.709442 | 98.242678 | 8.845329 | 88.461538 | 53237 / 7708 / 7212 | 0.000170304 | −0.115489 |
| **BGCR 训练后 mIoU 最佳** | **5** | **77.617593** | **72.579284** | **87.398542** | **98.744770** | 7.424355 | 90.769231 | 55131 / 10580 / 5318 | 0.000180026 | **−0.607261** |
| BGCR 训练后 mIoU 最低 | 25 | 75.655825 | 70.490142 | 86.140981 | 98.744770 | 10.194778 | 92.307692 | 53152 / 9806 / 7297 | 0.000202298 | −2.569029 |
| BGCR 训练终点 | 120 | 76.606939 | 71.255282 | 86.754166 | 98.577406 | 8.788109 | 90.769231 | 50962 / 6075 / 9487 | 0.000203092 | −1.617915 |

解释：Baseline1000 在 nIoU 上比 Current 高 0.265423 pp，但 mIoU 低 0.115489 pp，
且 Fa、tiny-Pd 更差；它只是同 train-only OOF 投影上的参考，不覆盖独立 official
Baseline 历史。BGCR epoch 5 把 Pd 提高 0.251046 pp、减少 FN 2154，但同时增加
FP 3305，导致 mIoU、nIoU 和 F1 全部回退。epoch 120 同样没有形成可采用的整体改善。

各关键点相对 Current 的完整差值如下；指标差值单位为 pp，Fa 仍为
<code>×10⁻⁶</code>，loss 为绝对差值：

| 工作点 | ΔmIoU | ΔnIoU | ΔF1 | ΔPd | ΔFa | Δtiny-Pd | ΔTP | ΔFP | ΔFN | Δloss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline1000 | −0.115489 | +0.265423 | −0.072764 | −0.251046 | +3.533363 | −2.307692 | +260 | +433 | −260 | +0.000002505 |
| BGCR epoch 5 | −0.607261 | −0.279593 | −0.383664 | +0.251046 | +2.112389 | 0 | +2154 | +3305 | −2154 | +0.000012227 |
| BGCR epoch 25 | −2.569029 | −2.368735 | −1.641225 | +0.251046 | +4.882812 | +1.538462 | +175 | +2531 | −175 | +0.000034499 |
| BGCR epoch 120 | −1.617915 | −1.603595 | −1.028039 | +0.083682 | +3.476143 | 0 | −2015 | −1200 | +2015 | +0.000035293 |

逐折局部轨迹如下。fold 2 确实在 epoch 5 局部提高 0.196097 pp，但 fold 0/1 的
训练后最佳仍低于各自 epoch 0；协议要求同一全局 epoch 精确 pooled，不能把三个 fold
各自最优 epoch 拼成一个候选。

| Fold | Epoch-0 Current mIoU | 该 fold 训练后最佳 epoch / mIoU | 差值 (pp) |
|---:|---:|---:|---:|
| 0 | 77.339854 | 40 / 76.935374 | −0.404479 |
| 1 | 79.079643 | 10 / 79.021401 | −0.058242 |
| 2 | 78.164465 | 5 / 78.360562 | +0.196097 |

### 22.3 epoch 0/5/…/120 的完整 pooled OOF 轨迹

下表直接来自 `selection.epoch_summaries[].metrics`；epoch 0 是 identity Current，
其余 24 行均为训练后 BGCR。所有训练后 mIoU 都低于 epoch 0；其中最高为 epoch 5，
最低为 epoch 25（75.655825%）。

| Epoch | mIoU | nIoU | F1 | Pd | Fa (×10⁻⁶) | tiny-Pd | TP | FP | FN | BCE loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0** | **78.224854** | **72.858877** | **87.782206** | 98.493724 | **5.311966** | 90.769231 | 52977 | 7275 | 7472 | **0.000167799** |
| **5** | **77.617593** | **72.579284** | **87.398542** | 98.744770 | 7.424355 | 90.769231 | 55131 | 10580 | 5318 | **0.000180026** |
| 10 | 77.279821 | 72.312206 | 87.184002 | 98.828452 | 7.419586 | 92.307692 | 53973 | 9392 | 6476 | 0.000182272 |
| 15 | 76.308490 | 71.199291 | 86.562468 | 99.079498 | 9.708405 | 93.076923 | 54192 | 10568 | 6257 | 0.000196924 |
| 20 | 76.420634 | 71.318108 | 86.634576 | 98.912134 | 8.878708 | 92.307692 | 53363 | 9379 | 7086 | 0.000195755 |
| 25 | 75.655825 | 70.490142 | 86.140981 | 98.744770 | 10.194778 | 92.307692 | 53152 | 9806 | 7297 | 0.000202298 |
| 30 | 76.177413 | 71.036601 | 86.478069 | 98.661088 | 9.660721 | 91.538462 | 53037 | 9174 | 7412 | 0.000197499 |
| 35 | 77.082990 | 71.647828 | 87.058604 | 98.828452 | 7.681847 | 92.307692 | 51392 | 6222 | 9057 | 0.000189291 |
| 40 | 77.036285 | 71.675357 | 87.028809 | 98.828452 | 7.858276 | 92.307692 | 51612 | 6548 | 8837 | 0.000192121 |
| 45 | 77.004932 | 71.536016 | 87.008798 | 98.661088 | 8.668900 | 91.538462 | 51524 | 6461 | 8925 | 0.000194164 |
| 50 | 77.353142 | 71.975409 | 87.230642 | 98.828452 | 7.977486 | 91.538462 | 52119 | 6929 | 8330 | 0.000195073 |
| 55 | 76.938587 | 71.653476 | 86.966431 | 98.912134 | 8.063316 | 92.307692 | 51515 | 6507 | 8934 | 0.000201250 |
| 60 | 76.730668 | 71.476891 | 86.833450 | 98.744770 | 7.381439 | 91.538462 | 50953 | 5956 | 9496 | 0.000195572 |
| 65 | 77.033809 | 71.721447 | 87.027229 | 99.079498 | 7.214546 | 93.846154 | 51266 | 6101 | 9183 | 0.000194730 |
| 70 | 76.713788 | 71.442544 | 86.822640 | 98.744770 | 7.925034 | 90.769231 | 51119 | 6187 | 9330 | 0.000199734 |
| 75 | 77.206267 | 71.866784 | 87.137175 | 98.661088 | 8.726120 | 91.538462 | 51590 | 6372 | 8859 | 0.000195266 |
| 80 | 76.732077 | 71.516543 | 86.834352 | 98.744770 | 8.516312 | 92.307692 | 50990 | 6003 | 9459 | 0.000199246 |
| 85 | 76.576059 | 71.185996 | 86.734362 | 98.577406 | 7.495880 | 90.769231 | 50652 | 5697 | 9797 | 0.000195815 |
| 90 | 76.748299 | 71.329187 | 86.844738 | 98.661088 | 7.944107 | 90.769231 | 51208 | 6273 | 9241 | 0.000202410 |
| 95 | 76.711279 | 71.352578 | 86.821033 | 98.577406 | 8.945465 | 90.769231 | 51069 | 6124 | 9380 | 0.000201598 |
| 100 | 76.521961 | 71.178313 | 86.699650 | 98.493724 | 8.916855 | 90.000000 | 50907 | 6077 | 9542 | 0.000204066 |
| 105 | 76.631822 | 71.251883 | 86.770120 | 98.493724 | 8.826256 | 90.000000 | 51023 | 6133 | 9426 | 0.000203553 |
| 110 | 76.625952 | 71.247967 | 86.766357 | 98.577406 | 8.854866 | 90.769231 | 51003 | 6112 | 9446 | 0.000203387 |
| 115 | 76.615547 | 71.264728 | 86.759686 | 98.577406 | 8.783340 | 90.769231 | 50957 | 6061 | 9492 | 0.000202959 |
| **120** | **76.606939** | **71.255282** | **86.754166** | **98.577406** | **8.788109** | **90.769231** | **50962** | **6075** | **9487** | **0.000203092** |

### 22.4 Cache、GPU、测试资产、候选与哈希

正式 cache 是 800 项、只读、host-RAM-resident 的冻结上下文缓存。每项包含
image/target、Current 的 `u1` 与六个 logits、Baseline1000 logits、两个 component-ID
图和八个 atlas bool mask；训练进程启动时逐项验证并常驻内存，后续 epoch 不重复解压。

| 产物/合同 | 正式记录 |
|---|---|
| Cache schema / status | `sctransnet_irstd_bgcr_frozen_context_cache_v1/v1` / `complete` |
| Cache sample count | 800 |
| Cache manifest file SHA-256 | `90c5a1fce85920ded133183a9f1b7f01083d7c7e774e1d6a4d52609033e68ec3` |
| Cache manifest semantic SHA-256 | `cda0859ffb55d4cf7237e5b0750387b1a624ff4e6c27d008c27a11a4811ff0c3` |
| Cache identity SHA-256 | `3d5a4a0a1013c78003d6e22bffa11846be1b903193b9ab1df1a8320c3c7a4734` |
| Fold manifest / assignment SHA-256 | `8ec2db388d083c8fe5e3750b5e66ba280b9796ca35f514e88323dfc029953fd0` / `a7ce375391e27e53bdad5f67599d470b336f70c304e22a96b8aa3fef6283c583` |
| OOF selection file / semantic SHA-256 | `4f238f74ed2bf7fa1467ec33679f35dd9f2a2d5963353fc2601d88219b8afe41` / `315e26bd252b5d260d9bd644f9ffa4f76bfaa59afa4ceaabd06f04e209b13365` |
| Full-selected summary file / self SHA-256 | `89f611310bac6527e087dbffdbfbf247d7dae750091cb4d5359b8a79eef71699` / `51895ef812ad7632c66cea26fbea210940533a2401dab8aa79639d36cfcf62dd` |
| Current 564-key base semantic SHA-256 | `f3745109e889cc6f25e42a43e698c5a43516ddc96a1364ffc78ab4b6b09d7f4f`；final audit 为 bitwise equal、全部 frozen、全部 base grad 为 `None` |
| Integrated candidate | 595 keys；file SHA `ad9049d85772673e60d390d2284f5995e52f36edcab53995fb3309163a573903`；state-semantic SHA `e6a81e9b3ed5a8d76528ed7df45c9e2a7c1b6b9ee3cb48c460f68402d1facfbf` |

GPU 只影响执行位置，没有改变 seed、数据、fold、loss、优化器或选择合同：

| 任务 | GPU | UUID |
|---|---|---|
| fold 0 | NVIDIA GeForce RTX 5090 | `4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562` |
| fold 1 | NVIDIA GeForce RTX 5090 | `8d68eb9e-49d3-67f6-f715-6ef2ac4975c3` |
| fold 2 | NVIDIA GeForce RTX 5090 | `4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562` |
| full-selected epoch-0 identity build | NVIDIA GeForce RTX 5090 | `4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562` |

代码侧现有 6 个 BGCR 专项测试文件，覆盖 cache、head/loss、
model contract、pipeline、run contract 和 OOF selector：

- [`test_irstd_bgcr_cache.py`](tests/test_irstd_bgcr_cache.py)
- [`test_irstd_bgcr_core.py`](tests/test_irstd_bgcr_core.py)
- [`test_irstd_bgcr_model_contract.py`](tests/test_irstd_bgcr_model_contract.py)
- [`test_irstd_bgcr_pipeline.py`](tests/test_irstd_bgcr_pipeline.py)
- [`test_irstd_bgcr_run_contract.py`](tests/test_irstd_bgcr_run_contract.py)
- [`test_select_irstd_bgcr_oof_v1.py`](tests/test_select_irstd_bgcr_oof_v1.py)

2026-08-08 最终复验用上述 6 个文件完整收集了 **82 个 pytest item**，
结果为 **82 passed、5 warnings、0 failures/errors**，用时 129.31 s；警告只是
4 条 `thop`/`distutils` 弃用警告和 1 条测试断言中的 tensor-to-scalar 警告。
可复核记录见
[`PYTEST_VERIFICATION_20260808.md`](results/irstd_bgcr_v1/PYTEST_VERIFICATION_20260808.md)。
该测试命令只运行合成/合同测试，没有构造或访问 official-test index、loader
或 evaluation。Cache、三折 summary、selector、full summary 和 candidate 的正式产物
均声明以下五项为 false：
`official_test_accessed`、`official_test_index_opened`、
`official_test_index_parsed`、`official_test_loader_built`、
`official_evaluation_performed`；同时 `performance_acceptance_margin=null`，cache 兼容字段
`margin=null`。因此本节只能支持 train-only OOF 的“不替换 Current”裁决，不能生成
新的 official-test 性能声明。

正式产物：

- 冻结方案：[SCTransNet_IRSTD专项优化_冻结主模型_BGCR代码方案.md](SCTransNet_IRSTD专项优化_冻结主模型_BGCR代码方案.md)
- 三折 manifest：[fold_manifest.json](results/irstd_bgcr_v1/fold_manifest.json)
- 三折 summary：[fold 0](results/irstd_bgcr_v1/fold_0/summary.json)、[fold 1](results/irstd_bgcr_v1/fold_1/summary.json)、[fold 2](results/irstd_bgcr_v1/fold_2/summary.json)
- OOF selection：[oof_selection.json](results/irstd_bgcr_v1/oof_selection.json)
- Cache commit：[COMMITTED.json](results/irstd_bgcr_v1/frozen_context_cache/COMMITTED.json)
- Full-selected summary：[summary.json](results/irstd_bgcr_v1/full_selected/summary.json)
- Integrated candidate：[integrated_candidate.pth.tar](results/irstd_bgcr_v1/full_selected/integrated_candidate.pth.tar)


## 23. 权威结果来源

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
- PBDR-V2 失败分析与 V3 方案：[`SCTransNet_PBDR_V2_failure_analysis_and_V3_plan.md`](SCTransNet_PBDR_V2_failure_analysis_and_V3_plan.md)
- PBDR-V3 跨数据集最终报告：[`results/two_dataset_pbdr_v3_stage1_v1/FINAL_RUN_REPORT.md`](results/two_dataset_pbdr_v3_stage1_v1/FINAL_RUN_REPORT.md)
- PBDR-V3 NUAA advisory 裁决：[`results/nuaa_pbdr_v3_stage1_v1/original_zero_margin_role_adjudication_v1.json`](results/nuaa_pbdr_v3_stage1_v1/original_zero_margin_role_adjudication_v1.json)
- PBDR-V4 零门槛方案：[`SCTransNet_PBDR零门槛失败分析与V4性能提升代码方案.md`](SCTransNet_PBDR零门槛失败分析与V4性能提升代码方案.md)
- PBDR-V4 冻结协议：[`experiments/PBDR_V4_PROTOCOL.md`](experiments/PBDR_V4_PROTOCOL.md)
- PBDR-V4 official publication bundles：[`results/pbdr_v4_v1/official/`](results/pbdr_v4_v1/official/)
- PBDR-V4 候选池与训练证据：[`results/pbdr_v4_v1/`](results/pbdr_v4_v1/)
- PBDR-V5 内部冻结协议：[PBDR_V5_INTERNAL_PROTOCOL.md](experiments/PBDR_V5_INTERNAL_PROTOCOL.md)
- PBDR-V5 失败定位：[failure_localization_bundle.json](results/pbdr_v5_v1/diagnostics/failure_localization_bundle.json)
- PBDR-V5 六族内部汇总：[INTERNAL_SUMMARY.md](results/pbdr_v5_v1/comparison/INTERNAL_SUMMARY.md)
- PBDR-V5 机器可读证据：[results/pbdr_v5_v1/](results/pbdr_v5_v1/)
- IRSTD-BGCR 冻结方案：[SCTransNet_IRSTD专项优化_冻结主模型_BGCR代码方案.md](SCTransNet_IRSTD专项优化_冻结主模型_BGCR代码方案.md)
- IRSTD-BGCR train-only OOF 选择：[oof_selection.json](results/irstd_bgcr_v1/oof_selection.json)
- IRSTD-BGCR cache、三折与 full-selected 证据：[results/irstd_bgcr_v1/](results/irstd_bgcr_v1/)

本文件只汇总已经落盘并完成口径核对的正式结果；后续模型完成后，应追加对应结果，
不覆盖或改写历史表。
