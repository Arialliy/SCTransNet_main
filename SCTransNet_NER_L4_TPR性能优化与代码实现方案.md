# SCTransNet NER-L4-TPR 性能优化与代码实现方案

## 1. 当前结论

当前正式完整模型仍为：

```text
TPD8 + NER4 + QFG2-CROA + TSS-off
```

已经完成的 GCSF 与 DORF 固定权重诊断共同表明：全图统一降低响应可以减少虚警，
但会同时压低真实目标，形成“Fa 下降、Pd 或 IoU 回退”的重复结果。下一步不再搜索
另一个全局常数，而是把已经观察到的 L4 正向重分配限制在 NER 判定的非目标区域。

本候选命名为：

```text
NER-L4-TPR
NER-conditioned L4 Target-Protected Reallocation
```

TPD8 的三分支 patch embedding、NER4 的五节点证据链、QFG2-CROA 的 Query-only
频率调制及六头 segmentation 训练损失均保持不变。

## 2. 要解决的性能问题

当前 L4 融合为：

```text
B4 = (T4 + E4) + E4 = T4 + 2E4
```

其中 `T4` 是 Transformer 重建特征，`E4` 是 CNN encoder 第四层特征。历史
`gpos025_l4_only` 把它改成 `1.25T4 + 1.75E4` 后，六个 checkpoint 角色的两类
FP 均下降，但总计少检 6 个目标。由此可见，L4 正向重分配具有降 Fa 信号，主要问题
是它无差别作用于目标区和背景区。

NER-L4-TPR 的目标是：

1. NER 高置信目标保护区严格保留当前 `T4 + 2E4`；
2. 只在非目标区域学习 L4 常系数和重分配；
3. 零初始化时严格回到当前完整模型；
4. 最终同时检查 Pd、Fa、mIoU、nIoU、tiny-Pd，而不是只看 mIoU。

## 3. 模型公式

现有 NER 第四层证据为：

```text
q4 = Phi4(h13, h22, up4)
```

`q4` 不依赖 L4 skip 融合结果，因此可以在 L4 CCA 之前先计算，不形成循环依赖。

使用 NER4 已冻结的 stage-4 tail 阈值 `kappa4=1.5` 生成保护区：

```text
S4 = TailSupport(q4; kappa4=1.5)
P4 = MaxPool3x3(1[S4 > 0])
A4 = 1 - P4
```

`P4` 与 `A4` 均停止梯度；NER 只提供空间路由，不被新的融合门反向改变。

新增逐通道门：

```text
G4 = 0.25 * tanh(a4)
a4 shape = (1, 256, 1, 1)
```

`a4` 全零初始化。最终 L4 融合采用保留原运算顺序的 baseline-plus-delta 形式：

```text
B4 = (T4 + E4) + E4
X4 = B4 + A4 * (G4*T4 - G4*E4)
```

等价系数为：

```text
X4 = (1 + A4*G4)T4 + (2 - A4*G4)E4
```

两支系数之和恒为 3。`P4=1` 时 `A4=0`，目标保护区严格使用当前融合；`P4=0`
时只在背景候选区执行重分配；`a4=0` 时全图输出与当前完整模型一致，同时 `a4`
仍可获得训练梯度。

## 4. 代码范围

新增：

```text
model/tpd_ner_l4_target_protected_reallocation.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_l4_tpr.py
tests/test_tpd_ner_l4_target_protected_reallocation.py
```

预计工程合同：

| 项目 | 当前 Final | NER-L4-TPR | 增量 |
|---|---:|---:|---:|
| 训练图参数 | 10,870,228 | 10,870,484 | +256 |
| 训练图 state keys | 568 | 569 | +1 |
| 推理图参数 | 10,870,130 | 10,870,386 | +256 |
| 推理图 state keys | 564 | 565 | +1 |
| persistent buffers | 0 | 0 | 0 |

现有正式文件不原地改写，候选使用独立模型类和独立训练入口。TSS 训练头仍可注册以
保持六输出训练图兼容，但 TSS loss 权重固定为 0，推理导出时仍只删除四个 TSS state。

## 5. 工程验收

代码完成后至少验证：

1. 核心模块零门输出与当前 `(T4+E4)+E4` 一致；
2. 完整模型迁移当前 Final 权重并补零门后，最终输出一致；
3. 目标保护区输出严格等于当前融合；
4. 背景区输出严格符合常系数和重分配公式；
5. `q4`、`mask4`、`up4` 均只计算一次；
6. 路由图停止梯度，门参数获得有限非零梯度；
7. 训练图到 head-free 推理图严格导出和加载；
8. 普通 Python 与 `python -O` 测试结果一致。

## 6. 性能筛选与训练顺序

正式训练不使用历史 checkpoint warm-start。历史 seed42 Final checkpoint 只用于快速
筛选，且由 manifest 明确绑定数据集、`best_miou`/`best_pd` 角色、路径和 SHA。

快速筛选固定：

```text
datasets = NUAA-SIRST, NUDT-SIRST, IRSTD-1K
roles = best_miou, best_pd
seed = 42
threshold = 0.5
modes = current_g0, tpr_g00625, tpr_g0125, tpr_g01875, tpr_g025
```

筛选同时记录目标检出计数与 Pd 数值、tiny-Pd、两种 FP、Fa、mIoU、nIoU、pixel
precision/recall/F1。阈值 1.0 只作为空预测端点记录，不参与 0.5 工作点比较。

这一筛选只决定是否值得投入训练资源，不把一个人为门槛写成最终论文结论。重点检查：

1. 相比无保护的 `gpos025_l4_only`，目标损失是否明显恢复；
2. 六角色中多数是否仍保留 FP/Fa 下降；
3. `best_miou` 与 `best_pd` 是否出现同向性能信号；
4. 是否存在 Pd、Fa、mIoU、nIoU 的可用联合工作点。

若目标保护有效，则按现有三数据集协议分别从头训练：

```text
seed = 42
epochs = 1000
split = each dataset img_idx/train and img_idx/test
evaluation = every 10 epochs
selection = best_mIoU and best_Pd independently
threshold = 0.5
datasets = NUAA-SIRST, NUDT-SIRST, IRSTD-1K
```

训练后以每个数据集自己的当前 Final 为直接对照，联合报告 Pd、Fa、mIoU、nIoU、
tiny-Pd 及错误目标数；不与不同训练/测试协议下的论文表格数值直接替代比较。

## 7. 当前状态

```text
current_production_model=TPD8_NER4_QFG2_CROA_TSS_OFF
candidate=NER_L4_TPR_V1
mainline_changed=false
tpd_formula_changed=false
ner_five_node_evidence_changed=false
qfg_formula_changed=false
tss_objective_enabled=false
candidate_code_implementation=in_progress
candidate_training_started=false
```
