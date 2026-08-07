# PBDR-V5 目标保持型组件约束微调：内部冻结协议

## 1. 先定位、后实现

V5 只能在
[`failure_localization_bundle.json`](../results/pbdr_v5_v1/diagnostics/failure_localization_bundle.json)
已经独占落盘后实现。定位过程只读取 V4 的 development-train atlas、内部验证摘要、
内部缓存和 checkpoint，不读取或构造 official test index/loader。

用户给出的独立训练 Baseline 只是一组标量参考，没有绑定可加载 checkpoint。因此它进入
最终数值比较，但不能伪装成内部候选或参与 checkpoint 选择。

## 2. V5 只做一个小型微调臂

V5 不增加主干、解码器或路由器结构。从每个角色已经选定且不可变的 V4-Stage1
checkpoint 初始化，只训练现有的：

- `pbdr_v4.*`
- `outc.*`
- `up_decoder1.*`

其余参数、全部 BatchNorm 状态和全部非许可 buffer 保持冻结。Stage2 的 `outc` 与
`up_decoder1` 继续用 L2-SP 锚定同角色 Current。

固定训练对象：

1. NUDT-SIRST / `best_pd`
2. NUAA-SIRST / `best_miou`
3. IRSTD-1K / `best_miou`

## 3. 损失只改变保护归一化

保留 V4 的 BCE、role-Tversky、component-equal rescue、component-equal suppress、
neutral-delta 和 Stage2 L2-SP。删除 V4 的绝对 preserve、全前景平均 drop、全背景平均
increase 三项，并用其原角色权重一对一替换为：

1. frozen-Current 相对 smooth-peak no-drop；
2. 每个 preserve component 等权的 Current-positive-support logit no-drop；
3. 只在实际发生正向概率变化的背景像素上平均的 active-background no-increase。

三个新项都使用精确零 margin，不加入 epsilon、最小增益或性能门槛。固定工作点仍为
`probability > 0.5`。

## 4. 固定预算

- seed：42
- 精度：FP32，TF32 关闭
- epoch：30
- 每 5 epoch 内部验证一次
- batch size：16
- optimizer/LR：沿用 V4-Stage2 的三个 AdamW 参数组
- 不扫权重、不重启试验、不按中途结果加 epoch

训练开始前先评估 epoch 0。epoch 0 与 5/10/15/20/25/30 使用同一个完整 role key；
完全相同保留更早 epoch。`performance_acceptance_margin = null`。

## 5. 选择与停止

内部候选固定顺序为 Original、Current、V3-calibrated、V4-Stage1、V4-Stage2、V5。
角色 key 不变：

- `best_miou`：mIoU ↑ → Pd ↑ → Fa ↓ → nIoU ↑ → tiny-Pd ↑ → loss ↓
- `best_pd`：Pd ↑ → Fa ↓ → tiny-Pd ↑ → mIoU ↑ → nIoU ↑ → loss ↓

V5 只要严格超过既有内部包络即可保留，没有正增益幅度门槛。若不超过，则如实判定该
小型 V5 无效，不调权、不补 epoch。整个 V5 内部阶段不得再次访问已经使用过的 official
test；最终是否需要新的独立评估，必须在候选和代码完全冻结后另行决定。

## 6. 预先声明的能力边界

- NUDT `best_pd` 的训练 atlas 没有 rescue component；V5 不能预先声称会恢复内部从未
  出现的漏检模式。
- IRSTD 的两个内部漏检峰值低于路由器单独可恢复范围；V5 允许末端解码参数微调，但
  target-preservation 仍只是一项保险约束，不构成必然超过独立 Baseline 的承诺。
- NUAA 的主要问题是边界校准外推，而不是目标峰值不足；V5 不统一抬高已检目标。
