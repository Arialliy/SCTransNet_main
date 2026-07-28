# TPD-NER V8-MPRS-DCH V2 单 seed 正式训练与评估协议

状态：代码与版本清单尚未最终冻结；正式训练、正式评估和结果聚合均未启动。
所有结果字段在实际产物完成并通过审计前均为 `TBD`。本协议只定义固定
`seed=42` 的工程验证，不安排多 seed。

## 1. 唯一新训练模型

V2 只允许一个可训练身份：

```text
tpd_ner_v8_mprs_dch_v2_full_relay_on
```

它保留 V8-MPRS-DCH Full parent、Keep/Context/Saliency 三语义源、五个证据
节点及 `q4 -> q3 -> q2` 递归中继。V2 只修改 relay 内部数值控制：

- 每个对齐后的 source projection 执行 per-sample full-tensor RMS
  normalization；
- fusion 的 ReLU 输出再次执行相同 RMS normalization；
- bias-free gate logits 按样本减去空间均值；
- residual mask 固定为 `atan(pi*z)/pi`，范围严格位于 `(-0.5, 0.5)`；
- 三个 gate 权重保持零初始化。

固定结构字段：

```text
parent variant        = tpd_clean_v8_mprs_dch_full
relay enabled         = true
relay version         = v2_rms_centered_arctangent
relay width           = 8
relay init seed       = 42
relay RMS eps         = 1e-6
gate bias             = false
gate centering        = per_sample_mean_hw
mask mapping          = atan(pi*z)/pi
relay parameters      = 11,288
total parameters      = 10,854,443
deep-supervision heads= 6
```

不得训练 `v2_relay_off`，也不得为它创建 run、checkpoint、sweep 或结果列。

## 2. Relay-off 恒等对照复用

正式 paired control 固定为当前 V1 运行：

```text
tpd_ner_v8_mprs_dch_full_relay_off
```

V2 的 relay-off adapter 仅作为不可训练的身份 probe：

```text
adapt_v8_mprs_dch_parent_v2(..., relay_enabled=False)
    -> adapt_v8_mprs_dch_parent(..., relay_enabled=False)
```

恒等条件必须同时成立：

1. V1 off 与 V2 off probe 使用同一个
   `tpd_clean_v8_mprs_dch_full` parent；
2. model seed 均为 `42`；
3. 返回类、state keys、forward path 和全部 state tensor 完全相同；
4. V2-on 的公共 state 与 V1-off 初始 state 逐 tensor 相同；
5. V2-on 仅新增 16 个以 `tpd_ner.` 开头的 state keys；
6. V2 正式 variant 集合不包含任何 off 身份。

聚合必须重新验证上述代码合同，并验证 V1 off 与 V2-on 的数据拆分、训练轴和
checkpoint role。仅写“结构相近”或参数量接近，不构成恒等对照。

## 3. 固定数据与训练轴

```text
dataset                = NUDT-SIRST
official train count   = 663
internal train/val     = 530 / 133
training seed          = 42
split seed             = 20260722
epochs                 = 800
batch size             = 16
patch size             = 256
workers                = 0
optimizer              = Adam
base/min LR            = 1e-3 / 1e-5
warmup                 = 10 epochs, then cosine decay
AMP                    = false
precision              = FP32
eval every             = 1 epoch
fixed threshold        = 0.5
centroid match radius  = 3
tiny target area       = A <= 9
loss                   = unweighted sum of BCE over six post-sigmoid outputs
official test accessed = false
```

V1 off 与 V2-on 必须具有相同的 ordered 530/133 IDs、split hashes 和以上完整
训练轴。checkpoint 选择都只能来自内部 validation：

```text
best.pth.tar       -> best_validation_pd_primary
best_miou.pth.tar  -> best_validation_miou_secondary
```

V1 off 是否通过 V2 候选绝对门槛不属于成功条件；它只承担严格 paired
control 的角色。

## 4. V2 训练入口与单卡调度

ordinary 兼容入口：

```text
experiments/train_tpd_ner_v8_mprs_dch_v2.py
output root = experiments/results/tpd_ner_v8_mprs_dch_v2_formal800_seed42_v1
run tag     = formal800_fp32_seed42_v2
```

正式 exact-resume 入口：

```text
experiments/train_tpd_ner_v8_mprs_dch_v2_exact.py
output root = experiments/results/tpd_ner_v8_mprs_dch_v2_exact_v1
run tag     = formal800_exact_v2_seed42
```

正式调度只创建一个 V2-on 任务，并只暴露一张 RTX 5090。lane/launcher
接受物理 GPU `2` 或 `3`，但一次只能选择其中一张；不得同时创建 off
任务，不得使用 GPU `0/1`。

fresh 与 exact-resume 只能作用于同一 V2 run identity。任何 V1 checkpoint、
V1 active journal、虚构 V2-off checkpoint 或不同 seed/split 的产物都必须在
恢复模型、optimizer、scaler、RNG 和 DataLoader 状态之前被拒绝。

## 5. 独立 V2 身份与版本清单

V2 ordinary/exact 必须拥有独立的：

- entry schema；
- checkpoint schema；
- checkpoint-identity schema；
- architecture-manifest schema；
- completion-summary schema；
- run ID prefix；
- source-lock key/schema/path；
- output root 和 run tag。

V2 exact 训练版本清单默认路径为：

```text
experiments/tpd_ner_v8_mprs_dch_v2_exact_source_lock.json
```

acceptance 版本清单默认路径为：

```text
experiments/tpd_ner_v8_mprs_dch_v2_acceptance_source_lock.json
```

两个 JSON 只能在所有 V2 训练、协议、evaluator、lane、launcher、
postprocess 和直接运行依赖最终冻结后，由版本清单工具以 no-overwrite 方式
生成。本阶段不得预先创建空清单，也不得修改 V1 manifests。

训练清单必须覆盖 V2 ordinary/exact、V2 model、本协议，以及实际复用的 V1
训练内核、V1 model 与共享 parent 依赖。acceptance 清单必须覆盖 V2
evaluator、单卡调度、postprocess、V1 off evaluator、SCTransNet reference
evaluator、共享 Pd/Fa metric core 和闭区间阈值实现。

## 6. 正式评估矩阵

正式聚合固定为六行：

| 来源 | Variant | Pd-primary | mIoU-secondary |
|---|---|---:|---:|
| 当前同口径参考 | baseline SCTransNet | 复用并重验 | 复用并重验 |
| 严格 paired control | V1 relay-off | 复用并重验 | 复用并重验 |
| 新候选 | V2 relay-on | 新评估 | 新评估 |

V2 evaluator 只接受 V2-on 的完整 800-epoch run，只评估
`best.pth.tar` 和 `best_miou.pth.tar`。V1 off 的两份 sweep 必须继续由现有
V1 evaluator 验证；baseline 的两份 sweep必须继续由同口径 SCTransNet
reference evaluator 验证。V2 postprocess 对 V1 off 和 baseline 仅执行只读
复用，不得重写、复制替换或重新生成 V1 result。

三者使用同一 shared metric core、`prediction > threshold` 判定、闭概率区间、
固定 threshold grid、tail logit step、match radius、tiny area 和五个 Fa
budget：

```text
1e-6, 5e-6, 1e-5, 5e-5, 1e-4
```

每份 sweep 必须绑定当前绝对 run/checkpoint 路径、checkpoint role/name/SHA、
seed/split、validation split SHA、evaluator 路径/SHA、训练产物 SHA 和 run
identity。V2 exact run identity 还必须证明训练时只暴露一张已登记的物理
GPU 2 或 3，逻辑设备为 `cuda:0`，设备型号为 RTX 5090，记录的
`device_uuid`、`physical_gpu_uuid` 与 `CUDA_VISIBLE_DEVICES` 三者一致；
CPU smoke 轨迹不得进入正式评估。旧 checkpoint、旧 evaluator 或不完整
JSON 均不可复用。

## 7. V2 候选绝对门槛

Pd-primary checkpoint 在阈值 `0.5` 必须同时满足：

```text
matched targets >= 188 / 189
Pd              >= 188 / 189
Fa              <= 1e-6
mIoU            >= 0.933647
```

mIoU-secondary checkpoint 在阈值 `0.5` 必须同时满足：

```text
mIoU            >= 0.946542
matched targets >= 187 / 189
Pd              >= 187 / 189
Fa              <= 1e-6
```

两个 candidate checkpoint 都必须满足各自对应的绝对门槛。

每个 candidate checkpoint 的 Pd@Fa 门槛：

| Fa budget | 最低 matched targets | 最低 Pd |
|---:|---:|---:|
| `1e-6` | 187 / 189 | `187/189` |
| `5e-6` | 188 / 189 | `188/189` |
| `1e-5` | 188 / 189 | `188/189` |
| `5e-5` | 188 / 189 | `188/189` |
| `1e-4` | 188 / 189 | `188/189` |

## 8. Paired 门槛与最终成功条件

对每个 checkpoint role，V2-on 相对对应 V1-off 的五个 Pd@Fa 点必须：

```text
non-inferior budget count >= 4 / 5
strictly-better budget count >= 1 / 5
```

最终成功仅在以下四项全部成立时允许：

1. V2-on Pd-primary 绝对门槛通过；
2. V2-on mIoU-secondary 绝对门槛通过；
3. Pd-primary 的 V2-on vs V1-off paired 门槛通过；
4. mIoU-secondary 的 V2-on vs V1-off paired 门槛通过。

baseline 负责报告改进幅度，不额外改变上述成功逻辑。V1-off 自身不要求通过
候选绝对门槛。任一候选或 paired 门槛失败时，decision 必须返回模型优化，
不能用 baseline 优势、单一 checkpoint 或单一指标替代。

## 9. 聚合发布

postprocess 只有在以下条件同时成立后才能运行：

- V2-on summary 为 `complete` 且 metrics 为连续 `1..800`；
- V1-off summary 为 `complete` 且 metrics 为连续 `1..800`；
- V1 off 和 baseline 的四份 reference sweep 均存在并通过当前身份重验；
- V2 training/acceptance manifests 均存在并通过只读验证。

两份 V2 sweep 按 checkpoint role 顺序执行。中断产生的不完整或身份错误的
V2 sweep 只能移入该 V2 run 的 `rejected_postprocess/<timestamp>/`，不得删除
或覆盖。最终 JSON、Markdown 与 completion marker 使用隔离临时文件和原子
发布；一份已完成、一份缺失时允许补齐，内容冲突时先保留到
`rejected_postprocess` 再重新发布。

## 10. 结论边界

本闭环只支持 NUDT-SIRST 官方训练集的 530/133 内部拆分和固定 seed 42。
不得据此声称跨 seed 稳定性、跨数据集泛化或官方测试集性能。正式报告必须
同时展示 Pd、Fa、mIoU、false objects/image 和五档 Pd@Fa，不得只看 mIoU。

## 11. 实现文件

| 部分 | V2 专属文件 |
|---|---|
| ordinary trainer | `experiments/train_tpd_ner_v8_mprs_dch_v2.py` |
| exact-resume trainer | `experiments/train_tpd_ner_v8_mprs_dch_v2_exact.py` |
| smoke | `experiments/smoke_tpd_ner_v8_mprs_dch_v2.py` |
| 正式 evaluator | `experiments/evaluate_tpd_ner_v8_mprs_dch_v2_pd_fa.py` |
| 单卡 lane | `experiments/run_tpd_ner_v8_mprs_dch_v2_formal800_1x5090_lane.sh` |
| 单卡 launcher | `experiments/launch_tpd_ner_v8_mprs_dch_v2_formal800_1x5090.sh` |
| 版本清单工具 | `experiments/freeze_tpd_ner_v8_mprs_dch_v2_source_locks.py` |
| postprocess/aggregate | `experiments/postprocess_tpd_ner_v8_mprs_dch_v2_formal800.py` |
| relay model | `model/tpd_ner_v8_mprs_dch_v2.py` |
