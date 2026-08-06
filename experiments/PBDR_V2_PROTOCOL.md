# PBDR-V2 seed42 三数据集 formal1000 冻结协议

> protocol version: `pbdr_v2_tss_off_seed42_v1`  
> frozen date: `2026-08-06`  
> candidate: `TPD8 + NER4 + QFG2-CROA + PBDR-V2 + TSS-off`

本文件只定义 PBDR-V2 正式训练身份。训练开始后不得修改；进度和结果写入结果目录
以及仓库研究复盘文档，不回写本文件。

## 1. 模型

PBDR-V2 使用已有 `q4`、`out` 和 `d0`：

```text
q  = RMS-normalize(stopgrad(q4))
C  = 0.05 + 0.90 * sigmoid(bilinear(conv_conf(q)))
Q  = C * tanh(bilinear(conv_direct_no_bias(q)))
g+ = 0.5 * tanh(rescue_strength_raw)
g- = 0.5 * tanh(suppression_strength_raw)
z  = out + Q + g+*C*relu(d0-out) - g-*(1-C)*relu(out-d0)
```

固定实现身份：

```text
integration_version=v4_qfg_v2_croa_pbdr_v2_v1
new_parameters=19
new_state_keys=5
new_buffers=0
training_parameters=10870247
training_state_keys=573
inference_parameters=10870149
inference_state_keys=569
training_only_tss_state_keys=4
tss_loss_weight=0
```

所有 PBDR-V2 新参数精确零初始化；初始六路输出、旧参数梯度和第一次 Adam 的旧参数
更新在 formal FP32 合同下与 Current 逐位一致。FP16/BF16 只验证前向输出 dtype/value
身份，不属于 formal1000 训练合同。

## 2. 数据与训练

三个数据集各自独立 scratch 训练、各自在自己的 `img_idx/test` 上选模：

```text
datasets=NUAA-SIRST,NUDT-SIRST,IRSTD-1K
split=each_dataset_img_idx
seed=42
epochs=1000
first_evaluation_epoch=10
evaluation_cadence=10
batch_size=16
patch_size=256
optimizer=Adam
base_lr=0.001
min_lr=0.00001
warmup_epochs=10
schedule=manual_linear_warmup_then_cosine
segmentation_loss=ordered_sum_BCE_over_six_outputs
precision=FP32
threshold=0.5
match_radius=3
tiny_area=9
selected_checkpoints=best_miou,best_pd
```

三个数据集使用同一模型、公式、初始化、损失和超参数；不使用 Current checkpoint
热启动。Current 的 568 个共享 scratch state 只用于核对并逐位安装同一 seed42 初始状态，
五个新增 PBDR-V2 state 保持零。

## 3. checkpoint 与恢复

只长期保存：

```text
checkpoints/best_miou.pth.tar
checkpoints/best_pd.pth.tar
```

训练过程中允许一个覆盖式 rolling resume state，完成后删除。Resume 必须同时匹配：

```text
schema
dataset
seed=42
recipe_id=pbdr_v2_tss_off
architecture_id
integration_version
training_state_key_count=573
protocol_sha256
planned_total_epochs=1000
GPU UUID binding
```

Current 568-key、PBDR-V1 以及其他配方 state 均不得作为本协议 resume。

## 4. 选模与报告

双角色沿用统一 selector：

```text
best_miou = [mIoU, Pd, -Fa, nIoU, tiny-Pd, -loss, -epoch]
best_pd   = [Pd, -Fa, tiny-Pd, mIoU, nIoU, -loss, -epoch]
```

每个角色完整报告 `Pd / Fa / mIoU / nIoU / tiny-Pd`、目标计数和误检计数。Pd 与 Fa
必须联合解释；不得用单个 Pd 数值替代完整工作点。

## 5. 完整模型裁决

不要求每个数据集、每项指标全部提升。PBDR-V2 及后续候选前瞻采用 M2F-SV：

```text
detection_positive_best_miou_datasets >= 2/3
overlap_positive_best_miou_datasets >= 2/3
joint_detection_and_overlap_positive_datasets >= 1
severe_role_count_across_best_miou_and_best_pd = 0
original_materially_dominates_candidate_dataset_count = 0
```

`best_pd` 只承担严重退化检查和完整报告，不提供正向票。该规则不追溯改变
PBDR-V1 的既有机器裁决。
