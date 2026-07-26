# TPD-Clean-v4：单门控 KCS 融合筛选协议

状态：模型代码实现中；正式训练尚未启动。

## 1. 目的与边界

TPD-Clean-v4 只修复 v3 暴露出的融合自由度过松问题，不改变
TPD-SCTransNet 的目标保真 tokenization 主线，不改变
Keep–Context–Saliency 三个语义源，也不增加第四个并列 tokenizer 分支。

本轮只替换 `mtc.embeddings_1/2`。SCTransNet backbone、SCTB、decoder、
损失、数据划分、增强、优化器、checkpoint 选择和 Pd–Fa 定义保持不变。
TPD-NER、目标存活监督和 FG 均不进入本轮训练。

旧 TPD-v1、Clean-v2、Clean-v3、formal800、NER 源码和全部结果保持原状。
V4 使用独立源码、结果根、运行单元、日志和 source lock。

## 2. 模型公式

每个 2× 下采样单元保留：

```text
K = Conv1x1(PixelUnshuffle2(X))
C = AvgPool2(X)
S = MaxPool2(X) - C
```

Full 的 Context code 为：

```text
Q = tanh((C - mean_hw(C)) / rms_hw(C - mean_hw(C)))
```

融合改为：

```text
L = saliency_scale + 0.5 * tanh(context_scale) * Q
R = S * tanh(L)
Y = activation(K + R)
```

容量对照只令 `Q=1`，其余参数、状态键、初始化和残差范围完全相同。

设计契约：

- `|R| <= |S|`；
- `S=0` 时 `R=0`；
- Context 不能在 Saliency 支持之外制造响应；
- 两个 scale 从零初始化，step 0 与 dense SPD 逐元素相同；
- Full 与 capacity control 参数量和初始 state 完全相同；
- Context 归一化和 logit 在 FP32 中计算，再转换回特征 dtype。

## 3. 候选矩阵

| 变体 | Context code | 身份 |
| --- | --- | --- |
| `tpd_clean_v4_full` | centered spatial RMS + tanh | KCS 主候选 |
| `tpd_clean_v4_sal_capacity` | constant one | 同容量 Saliency 对照 |

每个变体训练 seed `42` 和 `3407`，共四个 run。两张 RTX 5090 仅使用物理
GPU 2 和 3，每张卡顺序或受控并发执行两个 run；GPU 0 和 1 不使用。

## 4. 固定训练协议

- 数据：NUDT-SIRST 官方训练索引的既有 530/133 内部分割；
- 不访问官方测试集；
- patch size：256；
- batch size：16；
- epoch：800；
- optimizer、学习率、warmup、AMP、增强：继承冻结的
  `experiments/train_tpd_pilot.py` 协议；
- 主 checkpoint：validation Pd 最大，同 Pd 时依次选择更低 Fa、
  更高 tiny-Pd、更高 mIoU；
- 辅助 checkpoint：validation mIoU 最大；
- 固定阈值：0.5；
- Pd–Fa sweep 从第一版开始包含
  `nextafter(float32(1), 0)` 和阈值 `1.0`；
- 五个预注册 Fa budget：
  `1e-6 / 5e-6 / 1e-5 / 5e-5 / 1e-4`。

epoch 350–400 只做运行健康和趋势检查，不提前宣布结构胜出，也不因普通
指标波动停止 800 轮训练。

## 5. 工程晋级门槛

V4 Full 必须全部满足，才允许进入 TPD-NER 正式训练。

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

- 任一 seed 的固定阈值和五个预算上，不允许 capacity control
  同时以不低 Pd、不高 Fa、不低 mIoU 严格支配 Full；
- 每个 seed 至少一个预注册预算上，Full 必须严格优于 capacity control；
- Full 的优势不能只来自阈值 `1.0` 空预测端点。

### Gate E：工程完整性

- 四个 run 均为连续可审计的 800 epoch；
- 每个 run 同时具有 `best`、`best_miou`、`last` checkpoint；
- checkpoint 可严格重建加载，模型、split、训练协议和 evaluator 哈希一致；
- CPU 与 RTX 5090 smoke 均通过；
- 固定阈值结果可在 sweep 中复算一致；
- 所有五个 Fa budget 均有有效工作点。

## 6. 决策

- 任一门槛失败：`engineering_gate_passed=false`，TPD-v1 主线保持不变，
  不启动 NER 正式训练，继续只优化 TPD-PE。
- 全部门槛通过：只将 V4 标为可进入 NER 的工程候选，并启动
  `TPD/Progressive × NER off/on` 代码与训练阶段。
- 即使全部通过，也不自动设置
  `paper_core_established=true` 或
  `stability_claim_supported=true`，且不自动覆盖既有主线。
