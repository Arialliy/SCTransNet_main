# TPD-NER V8-MPRS-DCH 单 seed 正式训练与评估协议

状态：exact-resume 主训练入口、ordinary 兼容入口、闭区间 Pd/Fa 评估入口及
其定向测试已完成；本协议不声称正式 800-epoch 结果已经产生。所有结果值在
实际运行和审计完成前均为 `TBD`。

## 1. 模型身份与结构边界

本阶段以 `tpd_clean_v8_mprs_dch_full` 完整模型为唯一 parent，在其五个既有
分级下采样证据节点上接入 Nested Evidence Relay（NER）：

```text
embeddings_1: h11 -> h12 -> h13 -> emb1
embeddings_2: h21 -> h22 -> emb2

relay: q4(h13,h22,up4) -> q3(h12,h21,q4,up3) -> q2(h11,q3,up2)
```

五节点的布局固定为 `3 + 2`，中继顺序固定为 `q4 -> q3 -> q2`，宽度固定为
`8`。五节点是中间证据，不是五个新的输入分支；TPD 仍只有 Keep、Context、
Saliency 三个语义源。

exact 与 ordinary 两个入口都只接受下列两个显式 Full 身份：

| Variant | 正式角色 | Relay | 参数量 |
|---|---|---:|---:|
| `tpd_ner_v8_mprs_dch_full_relay_off` | TPD-only 严格同初始化对照 | off | 10,843,155 |
| `tpd_ner_v8_mprs_dch_full_relay_on` | TPD+NER 主候选 | on | 10,854,446 |

Relay-on 只增加 `tpd_ner.*` 下的 11,291 个参数。Relay-off 不注册
`tpd_ner` 子模块。两者从相同 seed 的 fresh Full parent 构建，公共参数初始
state 必须一致；Relay 的局部初始化 seed 固定为 `42`。Capacity 不属于本协议
的正式训练或评估矩阵。

## 2. 唯一 seed 与数据拆分

本协议冻结为单 seed：

```text
model/training seed = 42
relay initialization seed = 42
split seed = 20260722
dataset = NUDT-SIRST
official train count = 663
internal train/validation = 530 / 133
```

不得使用 seed `3407`，不得追加 multi-seed 矩阵，也不得以其他 seed 的
checkpoint 续跑。训练 CLI 会主动拒绝非 `42` 的 `--seed`、非 `20260722`
的 `--split-seed`；ordinary 入口还直接限制其余训练轴，exact 主路径将其余
trajectory 轴写入原生 identity，正式评估器拒绝任何偏离本协议的产物。

拆分只来自 `img_idx/train_NUDT-SIRST.txt`。训练和 checkpoint 选择均只使用
内部 530/133 拆分；不得访问官方测试集。`split.json` 保存排序无关的既有
hash；exact 主路径还在原生 run identity 中绑定有序 train/validation
fingerprints，ordinary 入口则在 `split.json` 增加有序 ID hash。

## 3. 正式训练矩阵

本阶段只新训练两个 fresh run：

| Variant | Seed | Split seed | Epochs |
|---|---:|---:|---:|
| `tpd_ner_v8_mprs_dch_full_relay_off` | 42 | 20260722 | 800 |
| `tpd_ner_v8_mprs_dch_full_relay_on` | 42 | 20260722 | 800 |

固定训练设置：

- patch size：256
- batch size：16
- workers：0
- optimizer：Adam
- base/min learning rate：`1e-3 / 1e-5`
- warmup：10 epochs，随后 cosine decay
- AMP：关闭，FP32
- 每个 epoch 执行内部验证
- 固定验证阈值：0.5
- centroid match radius：3
- tiny target area：`A <= 9`
- exact 主路径 run tag：`formal800_exact_v1`
- ordinary 兼容路径 run tag：`formal800_fp32_seed42_v1`

模型在 `mode=train, deepsuper=True` 下必须返回六个 post-sigmoid Tensor。
每个 Tensor 的形状必须与 target 完全一致，训练目标固定为：

```text
loss = BCE(output_1,target) + ... + BCE(output_6,target)
```

不是 logits loss，不允许只训练最后一路，也不允许给六路添加新权重。入口在
训练时主动检查“恰好六路”语义，并在 loss、验证指标或 checkpoint 写入阶段
阻止非有限值；checkpoint 使用同目录原子替换。

正式主路径使用 `train_tpd_ner_v8_mprs_dch_exact.py`。它把模型、optimizer、
scaler、epoch、手工 LR 位置、Python/NumPy/Torch/CUDA RNG、DataLoader
generator、完整 metrics history 和选择历史绑定进 exact journal；中断后只允许
同一 combination variant 做 epoch-boundary exact resume。Fresh 示例：

```bash
PYTHONHASHSEED=42 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=<registered-GPU-UUID> \
TPD_NER_V8_MPRS_DCH_PHYSICAL_GPU_INDEX=<2-or-3> \
TPD_NER_V8_MPRS_DCH_PHYSICAL_GPU_UUID=<registered-GPU-UUID> \
python experiments/train_tpd_ner_v8_mprs_dch_exact.py \
  --variant tpd_ner_v8_mprs_dch_full_relay_off \
  --fresh \
  --device cuda:0

PYTHONHASHSEED=42 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=<registered-GPU-UUID> \
TPD_NER_V8_MPRS_DCH_PHYSICAL_GPU_INDEX=<2-or-3> \
TPD_NER_V8_MPRS_DCH_PHYSICAL_GPU_UUID=<registered-GPU-UUID> \
python experiments/train_tpd_ner_v8_mprs_dch_exact.py \
  --variant tpd_ner_v8_mprs_dch_full_relay_on \
  --fresh \
  --device cuda:0
```

中断后使用相同命令、相同环境、相同 variant，将 `--fresh` 改为
`--exact-resume`。启动前必须生成并核验
`experiments/tpd_ner_v8_mprs_dch_exact_source_lock.json`；source lock 必须覆盖
exact 入口声明的全部 `RUNTIME_SOURCE_PATHS`（其中包含本协议）及冻结训练数据
digest。评估器源码 hash 由每份 sweep 的 evaluator provenance 另行绑定。

exact 主路径默认输出根目录为：

```text
experiments/results/tpd_ner_v8_mprs_dch_exact_v1/
```

每个 exact 运行目录必须是：

```text
NUDT-SIRST/<variant>/seed_42_formal800_exact_v1/
```

`train_tpd_ner_v8_mprs_dch.py` 只保留为无中断 ordinary 兼容入口；其输出根目录
和 run tag 分别为 `tpd_ner_v8_mprs_dch_formal800_seed42_v1/` 与
`formal800_fp32_seed42_v1`。最终评估器严格接受 exact-primary 与
ordinary-compatibility 两种 schema，但正式调度优先 exact。

同一目录已存在时不得覆盖；off/on 的 run ID、目录、architecture ID 和
checkpoint identity 必须独立，禁止交叉加载或交叉续跑。

## 4. Checkpoint 与训练产物

每个完整 run 必须产生：

- `protocol.json`
- `split.json`
- 连续 800 行的 `metrics.jsonl`
- Pd-primary `best.pth.tar`
- mIoU-secondary `best_miou.pth.tar`
- 最后评估 epoch `last.pth.tar`
- `summary.json`，且 `status=complete`

三个 checkpoint role 固定为：

```text
best.pth.tar       -> best_validation_pd_primary
best_miou.pth.tar  -> best_validation_miou_secondary
last.pth.tar       -> last_evaluated_epoch
```

exact 主路径的 compatibility checkpoint 顶层必须保存 exact NER checkpoint
schema、derived checkpoint schema、原生 exact `run_identity`、
`checkpoint_identity`、variant、parent、relay 开关/宽度、role、split hash、
模型/optimizer/scaler state、三个 state digest、source exact checkpoint digest
和完整内部验证指标。原生 exact identity 绑定 architecture ID、builder manifest、
source locks、有序 split/data fingerprints、训练合同和 RNG/选择策略，不能伪装
成 ordinary identity。

ordinary 兼容 checkpoint 使用独立 ordinary schema，并在 checkpoint identity
中绑定 role 与实际文件名。评估器对两种来源分别严格校验，再在 sweep 输出的
`evaluated_checkpoint_identity` 中统一记录 artifact mode、role、文件名和 hash。
评估器只接受完成 run 中的 `best.pth.tar` 与 `best_miou.pth.tar`，不以
`last.pth.tar` 生成正式 sweep。

## 5. 主比较与严格配对对照

最终主比较为：

```text
baseline SCTransNet
    vs new Full relay-off = TPD-only V8-MPRS-DCH Full
    vs TPD+NER V8-MPRS-DCH Full relay-on
```

`full_relay_off` 本身就是本轮新训练的 TPD-only 严格同初始化对照，不再额外
放一个外部 TPD-only V8 Full 正式列。旧 TPD-only 结果可以作为历史参考附表，
但不占正式主比较列，也不能替代本轮 relay-off。

baseline SCTransNet 是唯一正式外部参考。纳入最终表格前，它必须能够证明：

- dataset 均为 NUDT-SIRST；
- model/training seed 均为 42；
- split seed 为 20260722，且 530/133 ID 与 hash 完全一致；
- 训练为 800 epochs、内部验证选择、官方测试未访问；
- fixed-threshold 和 Pd/Fa sweep 使用同一指标实现、匹配口径和预算；
- 对比的是清楚标识的 Pd-primary 或 mIoU-secondary checkpoint。

若 baseline 或另列的历史 TPD-only 产物不能满足上述同口径条件，应标记为
不可比较，而不是用近似运行、seed 3407、多 seed 平均或不同拆分替代。

## 6. 正式闭区间评估

两个新 run 的 `best.pth.tar` 和 `best_miou.pth.tar` 均执行一次正式 sweep，
因此本阶段应产生 4 份新的 NER sweep。评估复用既有
`evaluate_pd_fa_sweep.py` 的预测、连通域、one-to-one centroid matching、
Fa、mIoU 和 checkpoint 审计实现；阈值生成固定使用 V8 继承的闭概率区间函数。
正式来源优先为 exact run 发布的 compatibility checkpoint；ordinary run 仅作为
兼容来源。两种来源均须先通过各自 schema、run identity、checkpoint identity
和完整性审计。

预测判定保持：

```text
prediction > threshold
```

闭区间必须包含最后一个小于 1 的 float32 阈值及上边界阈值，使无预测点
`Pd=0, Fa=0` 明确进入候选集合。禁止在看到结果后更改 threshold grid、tail
logit step、额外阈值、匹配半径、tiny area 或 Fa budget。

每份最终评估 JSON 必须覆盖：

1. 固定阈值 `0.5`：
   - Pd
   - Fa
   - mIoU
   - false objects per image
2. 五个固定 Fa budget 下的最佳 Pd：

```text
1e-6, 5e-6, 1e-5, 5e-5, 1e-4
```

每个 budget 点必须实际存在，所记录 Fa 不得超过对应 budget。输出中的
`final_metric_coverage` 必须显式列出上述四个固定阈值指标和五个
`Pd@Fa-budget`，缺失任意字段时评估失败而不是写出不完整结果。

GPU 评估示例：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python experiments/evaluate_tpd_ner_v8_mprs_dch_pd_fa.py \
  --run-dir <formal-run-directory> \
  --checkpoint best.pth.tar \
  --device cuda:0

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python experiments/evaluate_tpd_ner_v8_mprs_dch_pd_fa.py \
  --run-dir <formal-run-directory> \
  --checkpoint best_miou.pth.tar \
  --device cuda:0
```

CPU 审计可使用 `--device cpu`。正式评估器会主动拒绝非 800 epoch、
非固定预算、非固定阈值扫描设置、非正式 checkpoint 名称、off/on 身份串换、
不一致的 schema/run ID/architecture ID/role，以及不完整的最终指标。

## 7. 结果前固定的性能门槛

以下门槛在任何正式结果产生前冻结。比例与目标数必须同时记录；本拆分的锚定
target 总数为 189。

Pd-primary checkpoint 在阈值 0.5 必须同时满足：

```text
matched targets >= 188 / 189
Pd >= 188 / 189 = 0.9947089947...
Fa <= 1e-6
mIoU >= 0.933647
```

mIoU-selected checkpoint（`best_miou.pth.tar`）在阈值 0.5 必须同时满足：

```text
mIoU >= 0.946542
matched targets >= 187 / 189
Pd >= 187 / 189 = 0.9894179894...
Fa <= 1e-6
```

Pd@Fa budget 门槛为：

| Fa budget | 最低 matched targets | 最低 Pd |
|---:|---:|---:|
| `1e-6` | 187 / 189 | `187/189` |
| `5e-6` | 188 / 189 | `188/189` |
| `1e-5` | 188 / 189 | `188/189` |
| `5e-5` | 188 / 189 | `188/189` |
| `1e-4` | 188 / 189 | `188/189` |

完整 relay-on 还必须相对同一次正式矩阵的 relay-off，在五个预算点中至少
4 点 Pd 不差，并至少 1 点 Pd 严格更好。该配对门槛必须由 off/on 两份同 role
sweep 聚合后裁决，不能由单 checkpoint 文件自行宣告。评估 JSON 会分别记录
绝对门槛的要求、observed target/matched-target 数、比例和布尔检查，并将
relay-on 配对门标记为待聚合。

任一绝对门槛或 relay-on 配对门槛未通过，结论必须是返回代码/训练优化；不得
宣称最终成功，不得通过更换 seed、预算、阈值、checkpoint role 或拆分绕过。

## 8. 汇总表最低字段

每个 checkpoint 一行，至少报告：

| 来源 | Variant | Seed | Split seed | Checkpoint role | matched/189 + Pd@0.5 | Fa@0.5 | mIoU@0.5 | False objects/image@0.5 | matched/189 + Pd@1e-6 | matched/189 + Pd@5e-6 | matched/189 + Pd@1e-5 | matched/189 + Pd@5e-5 | matched/189 + Pd@1e-4 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 外部参考 | baseline SCTransNet | 42 | 20260722 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 本阶段严格对照 | Full relay-off = TPD-only | 42 | 20260722 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 本阶段候选 | Full relay-on | 42 | 20260722 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Pd-primary 与 mIoU-secondary checkpoint 不得混成一行或挑选后只报告较优者；
两种 checkpoint 的 sweep 均保留，主表采用哪个 role 必须预先一致并在表头注明。

## 9. 结论边界

本协议只支持 seed 42、NUDT-SIRST 内部 530/133 验证上的配对工程结论。不得从
单 seed 推导稳定性、多随机性、跨数据集或官方测试集结论。不得用 smoke、
未完成 epoch、中途最优值或外部不同拆分结果填充正式表。

Relay-on 的有效性必须参照本轮 relay-off TPD-only 严格配对对照，并联合
检查 Pd、Fa、mIoU、false objects/image 与五个预算点；不能只凭单一 mIoU、
单个阈值或单个 checkpoint 宣称成立。

## 10. 实现入口

| 部分 | 文件 |
|---|---|
| 单 seed exact-resume 主 trainer | `experiments/train_tpd_ner_v8_mprs_dch_exact.py` |
| ordinary 兼容 trainer | `experiments/train_tpd_ner_v8_mprs_dch.py` |
| 正式 closed-interval evaluator | `experiments/evaluate_tpd_ner_v8_mprs_dch_pd_fa.py` |
| CPU/GPU 两步训练检查 | `experiments/smoke_tpd_ner_v8_mprs_dch.py` |
| GPU 2/3 双任务启动器 | `experiments/launch_tpd_ner_v8_mprs_dch_formal800_2x5090.sh` |
| 单卡任务执行器 | `experiments/run_tpd_ner_v8_mprs_dch_formal800_2x5090_lane.sh` |
| 训练与评估版本清单生成器 | `experiments/freeze_tpd_ner_v8_mprs_dch_source_locks.py` |
| 五节点模型/adapter | `model/tpd_ner_v8_mprs_dch.py` |
| exact trainer 定向测试 | `tests/test_train_tpd_ner_v8_mprs_dch_exact.py` |
| ordinary trainer 定向测试 | `tests/test_train_tpd_ner_v8_mprs_dch.py` |
| evaluator 定向测试 | `tests/test_evaluate_tpd_ner_v8_mprs_dch_pd_fa.py` |
| smoke 定向测试 | `tests/test_smoke_tpd_ner_v8_mprs_dch.py` |
| 版本清单定向测试 | `tests/test_tpd_ner_v8_mprs_dch_source_locks.py` |
