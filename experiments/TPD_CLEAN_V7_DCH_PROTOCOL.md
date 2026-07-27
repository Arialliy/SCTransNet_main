# TPD-Clean V7-DCH：延迟式 Context 余量正式协议

状态：V6 冻结 failure atlas 已完成，预注册决策为
`GO_DCH_TRAJECTORY_TEST`。因此本协议冻结唯一的 V7-DCH 公式并授权实现、
smoke、精确续训验证和四组 fresh formal800。该状态不是
`CONTEXT_DIRECT_SUPPORT`，也不表示机制、论文核心或跨随机性稳定性已经
成立。

## 1. 决策来源与结论边界

只读决策产物：

```text
analysis/results/tpd_clean_v6_frozen_failure_atlas_v1/V6_FAILURE_ATLAS.json
SHA256 6ed938a7e9e7652df24ec1ebe9cc0c680458e20d0547c7cffc2e7c8d897a1317
```

决策条件：

```text
diagnostics_complete=true
zero_scale_gradient_asymmetry_confirmed=true  # 公式与代码证据
saliency_scale_not_saturated=true
residual_off_only_explains_registered_failure=false
context_direct_support=false
```

56 个 checkpoint/block 的 `max(abs(tanh(saliency_scale)))` 全部小于
预注册上界 `0.5`，观测全局最大值为 `0.3624752163887024`。冻结
counterfactual 没有建立 Context 的直接作用；本轮只检验“延迟 Context
介入是否能改善 fresh 训练轨迹”。

始终保持：

```text
dch_causal_mechanism_established=false
paper_core_established=false
stability_claim_supported=false
```

上述字段只能由后续预注册证据更新，不能由启动训练、单个 checkpoint 或
单个 mIoU 数值更新。

## 2. 主线与唯一公式

主线仍是三个既定语义源，不能增加第四个并列 tokenizer 分支：

```text
K  = Conv1x1(PixelUnshuffle2(X); Wk, bk)
C0 = AvgPool2(X)
S0 = MaxPool2(X) - C0
```

继续使用 V6 的无参数相位绑定投影：

```text
Wt[o,c] = sum_{p=0..3} Wk[o,4c+p]
Ca = Conv1x1(C0; Wt, bias=None)
Sa = Conv1x1(S0; Wt, bias=None)
```

Context code 与 Saliency scale：

```text
Q = tanh((Ca - mean_hw(Ca))
         / sqrt(mean_hw((Ca - mean_hw(Ca))^2) + 1e-6))
V = 0.5 * (Q - mean_hw(Q))
a = tanh(saliency_scale)
```

本轮冻结的唯一 DCH Full 公式：

```text
H = 1 + abs(a) * (1 - abs(a)) * V
R = Sa * (a * H)
Y = activation(K + R)
```

Capacity 对照：

```text
H = 1
R = Sa * a
Y = activation(K + R)
```

设计不变量：

- Keep、Context、Saliency 的定义与 V6 相同；
- Context 只调制 Saliency residual；
- `Wt` 只从 Keep 权重派生，不增加参数或 persistent buffer；
- 每个 block 只有一个逐通道 `saliency_scale`；
- Full/Capacity 参数、state keys 和配对初始化完全相同；
- zero scale 时两者与 dense SPD 前向逐元素相同；
- zero scale 时 Full/Capacity 的输入梯度、全部参数梯度、第一步 Adam
  model state 与 optimizer state 相同；
- `mean_hw(V)=0`、`0.75<=H<=1.25`、`abs(a*H)<=1`；
- `abs(R)<=abs(Sa)`，zero Saliency 时 residual 为零；
- DCH Context 项在零点附近为 `O(abs(a)^2)`；不声明其二阶可微；
- 只替换 `mtc.embeddings_1/2`，不改 backbone、SCTB、decoder、loss、
  数据、增强、优化器、checkpoint 选择或 Pd/Fa/mIoU 定义。

Capacity 是同容量归因对照，不能替代 K/C/S Full 成为论文主模型。

## 3. 正式矩阵与设备映射

变体：

```text
tpd_clean_v7_dch_full
tpd_clean_v7_dch_capacity
```

种子：`42`、`3407`。四组均从配对 fresh initialization 开始，不从 V6
checkpoint warm start。

只使用物理 GPU 2 和 3。每卡一个串行 lane，映射固定为：

| Lane | 物理 GPU | UUID | 顺序 1 | 顺序 2 |
|---|---:|---|---|---|
| A | 2 | `GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562` | Full / 42 | Capacity / 3407 |
| B | 3 | `GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3` | Capacity / 42 | Full / 3407 |

进程内可通过 UUID 映射成逻辑 `cuda:0`，但 run manifest 必须记录物理
index、UUID、逻辑 device 和映射方式。GPU 0、1 不属于本协议。

## 4. 训练、续训与 checkpoint

- 数据：NUDT-SIRST 官方训练索引的既有 530/133 内部分割；
- 不访问官方测试集；
- patch size：256；
- batch size：16；
- epochs：800；
- eval every：1；
- workers：0；
- AMP：关闭；
- `eps=1e-6`；
- optimizer、学习率、warmup、增强继承冻结 runner；
- 初始化模式只允许 `fresh` 或 epoch 边界 `exact_resume`；
- 每个 seed 的 Full/Capacity 使用相同 state、数据 ID、顺序生成器和第一
  batch。

checkpoint 角色：

| 文件 | checkpoint role | comparison role |
|---|---|---|
| `best.pth.tar` | `best_validation_pd_primary` | `pd_primary` |
| `best_miou.pth.tar` | `best_validation_miou_secondary` | `miou_primary` |
| `last.pth.tar` | `last_evaluated_epoch` | 完整性与续训 |

总产物固定为：

```text
4 formal runs
12 checkpoints（每 run: best / best_miou / last）
8 closed-interval sweeps（每 run: pd_primary / miou_primary）
```

中间结果不允许改变 epoch、阈值、checkpoint role、公式或 Gate。

## 5. 原生 17 字段验证 schema

每次 validation、`metrics.jsonl`、run summary 和三种 checkpoint 都必须
原生保存同一套字段：

```text
val_loss, miou, niou,
pixel_precision, pixel_recall, pixel_f1,
pd, tiny_pd, fa, false_objects_per_image,
target_count, matched_target_count,
tiny_target_count, matched_tiny_target_count,
predicted_object_count, unmatched_predicted_object_count,
valid_pixel_count
```

不得在训练结束后补齐 checkpoint 指标。选择策略仍同时使用
Pd、Fa、tiny-Pd、mIoU、loss，不能只看 mIoU。

## 6. Pd–Fa evaluator

- 固定阈值：`0.5`；
- 比较符号：`prediction > threshold`；
- 继承正式 adaptive threshold grid；
- 训练前已冻结两个闭区间端点：
  `nextafter(float32(1), 0)` 与 `1.0`；
- 五个 Fa budget：
  `1e-6 / 5e-6 / 1e-5 / 5e-5 / 1e-4`；
- matching、Pd、Fa、mIoU、tiny-Pd 和 component 定义不修改；
- DCH evaluator 只是 V6 闭区间核心的身份隔离薄包装，并绑定 DCH builder
  与 DCH variant。

## 7. Gate A–E：完全继承 V6，不放宽

### Gate A：seed 42 固定阈值

- Pd-primary：至少 `188/189`、`Fa<=5e-6`、
  `mIoU>=0.9336470588`；
- mIoU-primary：`mIoU>=0.946542`、至少 `187/189`、
  `Fa<=1e-6`。

### Gate B：seed 42 预算下限

- `Fa<=1e-6`：至少 `187/189`；
- 其余四个 budget：均至少 `188/189`；
- 至少一个 budget 点不被冻结 SPD 同时在 Pd、Fa、mIoU 上覆盖。

### Gate C：seed 3407 稳定性底线

- Pd-primary：至少 `188/189`、`Fa<=5e-6`、
  `mIoU>=0.920000`；
- mIoU-primary：`mIoU>=0.940000`、至少 `186/189`、
  `Fa<=1e-6`；
- 五个 budget 至少四个达到 seed 42 对应 Pd 的 `-1` 目标以内。

### Gate D：Full 对 Capacity

- 对两个 seed、两个 checkpoint role 的固定阈值和五个 budget 检查；
- 严格覆盖固定为 Pd 不低、Fa 不高、mIoU 不低，且至少一项严格更好；
- 任一 seed 不允许 Capacity 在任一注册工作点严格覆盖 Full；
- 每个 seed 至少一个非空预测 budget 上 Full 严格覆盖 Capacity；
- 阈值 `1.0` 空预测端点不能单独构成 Full 优势。

### Gate E：工程完整性

- 四个 run 均完成可审计的 800 epochs 或精确续接；
- 12 个 checkpoint 全部存在、角色正确并可 strict-load；
- 8 个 sweep 完整，固定阈值复算一致且五个 budget 均有工作点；
- split、训练协议、数据、source locks 与 evaluator 摘要一致；
- CPU、物理 GPU 2 和 3 smoke 通过；
- 原生 17 字段 schema 在 metrics、summary、checkpoint 中完整。

NER 工程授权仍严格为：

```text
ner_stage_authorized =
    gate_A_pass && gate_B_pass && gate_C_pass
    && gate_D_pass && gate_E_pass
```

任何一个 Gate 失败时，主线仍保持 K/C/S，但不能绕过门槛进入 NER 正式
训练。

## 8. 五节点接口与机制审查边界

tokenizer-only formal800 不接 NER。模型只准备通用 evidence interface：

```text
embeddings_1: states[:-1] -> 3 nodes
embeddings_2: states[:-1] -> 2 nodes
total -> 5 nodes
```

Mechanism Audit M 独立报告 `fragment_excess_total`、in-GT unmatched
pixels、split target 与 largest-fragment 方向；它只决定
`fragmentation_mechanism_claim_supported`，不替代 Gate A–E，也不以一个
辅指标或单一阈值判定模型性能。

## 9. 三类 source lock

三个 lock 分别生成并验证，不使用一个总锁：

1. **Diagnostic**：V6 frozen atlas、counterfactual 与汇总执行路径；
2. **Training**：以 DCH exact entry 的 `RUNTIME_SOURCE_PATHS` 为唯一
   路径权威，绑定 DCH model、普通/exact entry、exact-resume 数值内核、
   协议及实际 eager local imports；若 DCH exact wrapper 复用 V6 exact，
   V6 exact 入口及其实际 eager imports 必须纳入。正式执行链的
   worker/lane/launcher 以及 worker preflight 直接 import 的 smoke report
   verifier 同样纳入；status、smoke 生成/capture 和 tests 不在正式训练
   执行路径，由持久 smoke/工程验证产物另行绑定；
3. **Acceptance**：DCH evaluator、fixed/sweep、Gate A–E、finalizer、
   Mechanism Audit M、协议及其 eager local imports。

训练 lock 与 acceptance lock 必须独立；诊断脚本和未执行的
`model/tpd_clean_v7.py` 不得进入 training lock。source lock 只绑定代码
与冻结输入；每个 run manifest 另行记录：

```text
train/validation ID hash, image/mask fingerprint, mean/std,
CLI, relevant environment, Python/PyTorch/CUDA/cuDNN/driver,
physical GPU index/UUID, deterministic/benchmark/TF32,
checkpoint role/selection state, RNG/DataLoader/exact-resume state
```

Acceptance comparison manifest 还必须绑定 4 runs、12 checkpoints、
8 sweeps 的路径、SHA256、role 与原生 17 字段完整性。

## 10. 启动门槛

必须依次完成：

1. 模型结构、公式、zero-scale、梯度、第一 Adam step 与五节点测试；
2. 普通 evaluator、exact-resume 和原生 17 字段测试；
3. CPU smoke；
4. 物理 GPU 2/3 RTX 5090 smoke；
5. 三类 source lock 输入完整并生成后反向验证；
6. pairing manifest 验证；
7. 才能启动四组 formal800。

本协议不预填任何 DCH 性能结果。所有 Pd、Fa、mIoU、Gate 和 Audit M
数值只能来自上述正式产物。
