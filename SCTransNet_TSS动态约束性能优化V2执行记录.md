# SCTransNet TSS 动态约束性能优化 V2 执行记录

## 1. 本轮目标

本轮只处理完整模型在 IRSTD-1K 上 Pd、mIoU 回退，以及 NUDT-SIRST 上收益不足的问题。固定 seed 42、数据划分、归一化、1000 epochs、每 10 epochs 测试、阈值 0.5、优化器与学习率策略全部保持不变。

TPD、NER、TSS、QFG 的模型代码和推理图不修改。本轮不是新模块，也不改变完整模型的主线和创新点。

## 2. 失败定位

V1 正式日志显示，在完整模型 best_mIoU checkpoint 附近，固定 TSS 辅助项占分割损失的比例约为：

| 数据集 | `0.005 × L_tss / L_seg` |
|---|---:|
| SIRST3 | 20.8% |
| NUAA-SIRST | 17.6% |
| NUDT-SIRST | 7.7% |
| IRSTD-1K | 31.0% |

虽然四个数据集使用相同的名义权重 0.005，但 TSS 正样本权重和损失尺度不同，导致辅助任务对主分割任务的实际作用不一致。IRSTD-1K 的干预最强，与其 Pd 和 mIoU 回退方向一致。

## 3. 唯一代码修改

原训练目标为：

```text
L = L_seg + 0.005 × L_tss
```

V2 改为逐训练批次计算：

```text
lambda_eff = min(
    0.005,
    0.10 × stopgrad(L_seg) / (stopgrad(L_tss) + float32_eps)
)
L = L_seg + lambda_eff × L_tss
```

因此：

- TSS 的实际损失贡献不超过当前分割损失的 10%；
- 原比例低于 10% 时，仍使用完整的 0.005 权重；
- `lambda_eff` 由已停止梯度的损失计算，不引入权重本身的反向传播路径；
- 不修改测试阶段前向、阈值或指标定义。

## 4. 正式运行范围

先运行性能问题最直接的两个数据集：

- 物理 GPU 2：IRSTD-1K Final V2，seed 42，1000 epochs；
- 物理 GPU 3：NUDT-SIRST Final V2，seed 42，1000 epochs。

结果独立保存到：

```text
/home/ly/SCTransNet_main/results/four_dataset_seed42_tss_cap_v2
```

复用 V1 已冻结的数据清单，但不覆盖 V1 日志、权重或结果。训练期间只滚动保留一个续训状态；正式完成后只保留 `best_miou` 和 `best_pd` 两个入选 checkpoint。

## 5. 判断方式

在固定阈值 0.5 下，分别比较 V2 Final 与同数据集 V1 Original 的各自最优 checkpoint。Pd、Fa、mIoU、nIoU、tiny-Pd 均需报告，不用单一指标替代综合判断。

本轮首先判断 IRSTD-1K 的 Pd/mIoU 回退能否消除，同时检查 NUDT-SIRST 的 Pd–Fa–mIoU 工作点是否保持或改善。若结果正向，再用同一 V2 训练策略补跑 SIRST3 与 NUAA-SIRST；在看到这两个关键数据集结果前，不叠加第二项训练修改。
