# TPD-Clean-v2 下一模块固定门槛

## 状态

本门槛在四路候选均未到 epoch 100 时登记，早于协议规定的 epoch 350–400
第一次结构判断。它只决定是否允许设计或启动一个附加模块，不改变 TPD-v1
主线，不改变 Keep–Context–Saliency 三分支创新点，也不建立论文级结论。

该文件独立于训练 source lock，避免修改正在运行的四路实验依赖文件。
机器可读版本为 `experiments/tpd_clean_next_module_gate_v1.json`。

## 第一门槛：epoch 350–400 允许开始模块设计

`tpd_clean_full` 必须同时满足：

| 检查项 | 硬门槛 |
| --- | ---: |
| Pd-primary 固定阈值检出 | ≥ 188/189 |
| Pd-primary 固定阈值 Fa | ≤ 5e-6 |
| Pd-primary 固定阈值 mIoU | ≥ 0.9336470588 |
| mIoU-secondary 固定阈值 mIoU | ≥ 0.9427577483 |
| mIoU-secondary 固定阈值检出 | ≥ 186/189 |
| mIoU-secondary 固定阈值 Fa | ≤ 1e-6 |

通过只允许完成下一模块的结构说明、接口和消融计划；不允许据此提前停止当前
800 epochs，也不允许启动正式新模块训练。未通过则继续现有四路，不增加模块。

## 第二门槛：800 epochs 后允许正式启动下一模块

以下条件必须全部满足：

1. 四候选均完成 800 epochs，`best` 与 `best_miou` 两类 sweep 全部通过审计。
2. `tpd_clean_full` 的 Pd-primary 固定阈值达到 `188/189`、`Fa≤5e-6`、
   `mIoU≥0.9336470588`。
3. 在 `Fa≤1e-6`，至少达到 `187/189`；在 `5e-6、1e-5、5e-5、1e-4`
   四个较宽预算上，每个都至少达到 `188/189`。
4. 至少一个预设 Fa budget 严格优于冻结 SPD 与 TPD-v1，不能被二者联合覆盖。
5. mIoU-secondary 固定阈值达到 `mIoU≥0.9427577483`、`Pd≥186/189`、
   `Fa≤1e-6`。
6. `tpd_clean_full` 至少在一个预算上优于 `tpd_clean_ctx`，并至少在一个预算
   上优于 `tpd_clean_sal`；在最严格 `1e-6` 预算上不得弱于任一单分支。
7. 固定阈值和预算扫描不能给出相反方向的判断。

通过后，只允许把一个新模块作为 TPD 的附加扩展进入实验；不会自动替换主线。
未通过时，下一步只能优化当前 Keep 投影、残差注入或融合方式，不能直接增加
第四个并列分支。

## 理想目标（不作为硬门槛）

- `Fa≤1e-6` 时达到 `188/189`；
- mIoU-secondary 达到 SPD 的 `0.9491444867`。

## 证据边界

这些门槛来自已经冻结的 TPD-v1/SPD 单种子结果，只是工程阶段门。即使全部
通过，`paper_core_established`、`stability_claim_supported` 和
`three_branch_necessity_established` 仍保持 `false`，后续仍需配对多种子确认。
