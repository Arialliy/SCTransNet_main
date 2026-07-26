# TPD-Clean-v6：相位绑定、均值中性的 KCS 融合筛选协议

状态：V5 四组 800 epoch 与八个闭区间 sweep 已完成，最终报告为
`ENGINEERING_GATE_FAIL`；V6 独立模型代码与工程入口正在实现，尚未启动
正式训练。

## 1. 决策依据与边界

V5 的工程产物完整，Gate E 全部通过，但 Gate A、B、C、D 未通过：

- seed 42 Full 的 Pd-primary 为 `187/189`、`Fa=7.4573e-6`、
  `mIoU=0.917607`；
- seed 42 Full 的 mIoU-primary 为 `186/189`、
  `Fa=2.5240e-6`、`mIoU=0.935188`；
- seed 3407 Full 的两个 checkpoint 均为 `187/189`、
  `Fa=2.8682e-6`、`mIoU=0.929938`；
- seed 42 的 capacity control 在三个预注册工作点覆盖 Full。

因此，V5 不授权进入 NER 正式训练。V6 继续只优化 TPD patch embedding
内部的 Keep–Context–Saliency 融合，不改变目标保真 tokenization 主线，
不增加第四个并列 tokenizer 分支。本轮仍只替换
`mtc.embeddings_1/2`；backbone、SCTB、decoder、损失、数据划分、增强、
优化器、checkpoint 选择和 Pd–Fa 定义保持不变。

Baseline、TPD-v1、Clean-v2/v3/v4/v5、NER、目标存活监督、FG 代码和全部
既有实验产物保持原状。V6 使用独立源码、结果根、日志和 source lock，
从配对的共享初始化重新训练，不从 V5 checkpoint 初始化。

## 2. 模型公式

每个 2× 下采样单元仍只有三个既定语义源：

```text
K  = Conv1x1(PixelUnshuffle2(X); Wk, bk)
C0 = AvgPool2(X)
S0 = MaxPool2(X) - C0
```

由 Keep 的 dense 1×1 权重直接派生相位绑定投影，不新增参数：

```text
Wt[o,c] = sum_{p=0..3} Wk[o,4c+p]
Ca = Conv1x1(C0; Wt, bias=None)
Sa = Conv1x1(S0; Wt, bias=None)
```

这里 `4c+p` 与 PyTorch `pixel_unshuffle(..., 2)` 的通道顺序严格一致。
Full 的 Context code、均值中性调制和有界残差为：

```text
Q = tanh((Ca - mean_hw(Ca)) / rms_hw(Ca - mean_hw(Ca)))
V = 0.5 * (Q - mean_hw(Q))
a = tanh(saliency_scale)
H = 1 + 0.5 * (1 - abs(a)) * V
R = Sa * (a * H)
Y = activation(K + R)
```

容量对照计算相同的 K/C/S 与相位绑定投影，但令 `V=0`：

```text
H = 1
R = Sa * a
Y = activation(K + R)
```

设计契约：

- 语义源仍为 K/C/S，Context 只调制 Saliency；
- `saliency_scale` 是每通道向量，每个 block 只有这一组可学习尺度；
- `Wt` 完全由 `Wk` 派生，不注册新参数或 buffer；
- Full 与 capacity control 的参数、状态键和初始 state 完全相同；
- `mean_hw(V)=0`，因此 Context 只重新分配空间增益，不改变其平均基线；
- `0.5 <= H <= 1.5`，并且 `|a*H| <= 1`，故 `|R| <= |Sa|`；
- `S0=0` 时 `R=0`；
- scale 从零初始化，step 0 与 dense SPD 逐元素相同；
- Context code、绑定投影和有界系数使用 FP32，residual 末端再转回特征
  dtype；
- 仅 `mtc.embeddings_1/2` 被替换，不接 relay、NER、额外输出或 hook。

## 3. 候选矩阵

| 变体 | Context headroom | 身份 |
| --- | --- | --- |
| `tpd_clean_v6_full` | `H=1+0.5(1-|a|)V` | KCS 主候选 |
| `tpd_clean_v6_phase_capacity` | `H=1` | 同容量、同相位投影对照 |

每个变体训练 seed `42` 和 `3407`，共四个 fresh run。正式任务只使用物理
GPU 2 和 3，并交叉映射 variant/seed，避免单张卡固定对应同一候选。

## 4. 训练与评估协议

- 数据：NUDT-SIRST 官方训练索引的既有 530/133 内部分割；
- 不访问官方测试集；
- patch size：256；
- batch size：16；
- epochs：800；
- optimizer、学习率、warmup、增强：继承冻结的
  `experiments/train_tpd_pilot.py`；
- AMP：关闭，正式训练与 checkpoint 选择使用 FP32；
- 主 checkpoint：validation Pd 最大；同 Pd 时依次选择更低 Fa、
  更高 tiny-Pd、更高 mIoU；
- 辅助 checkpoint：validation mIoU 最大；
- 固定阈值：0.5；
- sweep 在训练前即包含 `nextafter(float32(1), 0)` 与阈值 `1.0`；
- 五个预注册 Fa budget：
  `1e-6 / 5e-6 / 1e-5 / 5e-5 / 1e-4`；
- 正式入口必须使用精确 epoch 边界续训内核，保存模型、优化器、
  DataLoader generator 和全部随机状态。

中途 epoch 仅用于运行状态检查，不改变 800 epoch 终点，不提前宣布候选
胜出。

## 5. 代码与运行前置门槛

正式训练前必须完成：

1. 结构公式、状态键、参数量和梯度测试；
2. CPU 两步 forward/backward/strict-reload；
3. 物理 GPU 2、3 各至少一份 RTX 5090 两步 smoke；
4. step 0 全模型六个输出与 dense SPD 逐元素一致；
5. Full/Control 配对初始化逐 tensor 一致；
6. 精确续训的连续运行与中断后续训轨迹逐 tensor 一致；
7. 训练、evaluator、smoke、协议与测试进入独立 source lock。

任一项缺失时不启动四组正式任务。

## 6. 工程晋级门槛

V6 沿用 V4/V5 的固定数值锚点，不因已看到 V5 结果而降低标准。

### Gate A：seed 42 固定阈值工作点

- Pd-primary：至少 `188/189`；
- Pd-primary：`Fa <= 5e-6`；
- Pd-primary：`mIoU >= 0.9336470588`；
- mIoU-primary：`mIoU >= 0.946542`；
- mIoU-primary：至少 `187/189`；
- mIoU-primary：`Fa <= 1e-6`。

### Gate B：seed 42 预算下限

- `Fa <= 1e-6`：至少 `187/189`；
- 其余四个预算：均至少 `188/189`；
- 至少一个预注册预算点不被冻结 SPD 同时在 Pd、Fa 和 mIoU 上覆盖。

### Gate C：seed 3407 稳定性底线

- Pd-primary：至少 `188/189`、`Fa <= 5e-6`、
  `mIoU >= 0.920000`；
- mIoU-primary：`mIoU >= 0.940000`、至少 `186/189`、
  `Fa <= 1e-6`；
- 五个预算中至少四个达到 seed 42 对应 Pd 的 `-1` 目标以内。

### Gate D：Full 对容量对照

- 任一 seed 的固定阈值和五个预算上，不允许 capacity control 同时以
  不低 Pd、不高 Fa、不低 mIoU 严格覆盖 Full；
- 每个 seed 至少一个预注册预算上，Full 必须严格优于 capacity control；
- Full 的优势不能只来自阈值 `1.0` 空预测端点。

### Gate E：工程完整性

- 四个 run 均为连续或精确续接的 800 epoch；
- 每个 run 同时具有 `best`、`best_miou`、`last` checkpoint；
- checkpoint 可严格重建加载，模型、split、训练协议和 evaluator 哈希
  一致；
- CPU 与物理 GPU 2/3 的 RTX 5090 smoke 均通过；
- 固定阈值结果可在 sweep 中复算一致；
- 所有五个 Fa budget 均有有效工作点。

## 7. Claim–evidence 对应与决策

| 工程问题 | 对照 | 证据 | 通过条件 |
| --- | --- | --- | --- |
| V6 是否达到下一模块门槛 | SPD、TPD-v1、V6 Full | Gate A–C | A–C 全通过 |
| Context 调制是否优于同容量版本 | V6 Full/Capacity | Gate D | 两个 seed 均通过 |
| 结果是否可训练、续训、评估和复核 | 四个 V6 run | Gate E | 全部子项通过 |

- 任一门槛失败：`engineering_gate_passed=false`，不启动 NER 正式训练，
  主线和创新点保持不变；
- 全部门槛通过：只将 V6 标记为可进入五节点 NER 的工程候选；
- 即使全部通过，也不自动改变历史
  `paper_core_established=false` 或
  `stability_claim_supported=false`；
- 本协议不预填任何 V6 结果，全部数值必须来自正式运行产物。
