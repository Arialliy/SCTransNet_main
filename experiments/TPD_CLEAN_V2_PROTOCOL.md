# TPD-Clean-v2 单种子结构筛选协议

## 状态与研究边界

本实验新增候选，不覆盖 `TPD-v1`，也不改写已经完成的 formal800
结论。只有新候选完成本轮 800 epochs、随后通过配对多种子确认，才讨论
是否升级主线。

当前运行使用共享 GPU 和 CPU。模型指标可用于结构筛选；墙钟时间、吞吐
与能效不得用于方法间比较。

## 固定问题

本轮只回答两个问题：

1. 将 TPD 的 grouped Keep 换成与 SPD 完全相同的 dense Keep 后，
   Context/Saliency 是否仍产生增量？
2. Context 与 Saliency 哪个分支负责检出变化，哪个分支可能增加虚警？

## 候选

| Variant | Keep | Context | Saliency | 作用 |
| --- | --- | --- | --- | --- |
| `grouped_keep` | grouped `4C→C` | — | — | 分离旧 TPD Keep 投影 |
| `tpd_clean_ctx` | dense `4C→C` | 零初始化有界残差 | — | 检验 Context |
| `tpd_clean_sal` | dense `4C→C` | — | 零初始化有界残差 | 检验 Saliency |
| `tpd_clean_full` | dense `4C→C` | 零初始化有界残差 | 零初始化有界残差 | 完整 TPD-Clean |

三个 Clean 候选在残差尺度为零时与 SPD 使用相同 Keep 起点。固定
`tanh(gamma)` 将每通道残差系数限制在 `[-1, 1]`。

## 固定训练协议

- 数据：NUDT-SIRST 官方训练部分；
- 划分：530/133 内部训练/验证，`split_seed=20260722`；
- 模型种子：42；
- 训练：800 epochs，batch size 16，patch size 256，FP32；
- 优化器、学习率、warmup、增强和 checkpoint 规则与已完成 formal800 相同；
- 主 checkpoint：validation Pd 最大，其后依次比较 Fa、tiny-Pd、mIoU、loss；
- 辅助 checkpoint：validation mIoU 最大；
- 官方 test：不访问；
- 每个 checkpoint 完成相同的 Pd–Fa 阈值扫描。

## 筛选规则

不使用 100-epoch 排名。epoch 350–400 才进行第一次结构判断；未被明确
覆盖的候选继续到 800。

候选进入配对多种子确认至少需要：

- 在预先固定的 Fa budgets 上不被 SPD 与当前 TPD 全部覆盖；
- 严格低 Fa 区域没有明显破坏 SPD 的工作点；
- 较宽松预算没有丢掉当前 TPD 的检出能力，或提供清楚的结构简化收益；
- mIoU-best 没有出现不可接受退化；
- 固定阈值与阈值扫描的结论方向一致。

tiny-Pd 当前为 `39/39` 天花板，不作为本轮候选排序依据。Pareto 点数量
也不作为证据强弱指标。
