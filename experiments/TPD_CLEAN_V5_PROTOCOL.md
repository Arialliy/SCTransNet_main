# TPD-Clean-v5：正向 Context 选择器 KCS 融合筛选协议

状态：模型代码、CPU/RTX 5090 两步预检和独立复核已通过；正式
screen800 训练待 source lock 冻结后启动。

## 1. 目的与边界

TPD-Clean-v4 的四组 800 epoch 实验已完成，但 Gate A、B、C、D 未通过。
V4 的主要问题不是三路主线，而是同一个有符号 logit 中两个可学习尺度的
冗余，以及 Context 在空间位置上反转 Saliency 注入方向。

V5 不改变 TPD-SCTransNet 的目标保真 tokenization 主线，仍只保留
Keep–Context–Saliency 三个语义源，不增加第四个并列 tokenizer 分支。
本轮仍只替换 `mtc.embeddings_1/2`；backbone、SCTB、decoder、损失、
数据划分、增强、优化器、checkpoint 选择和 Pd–Fa 定义保持不变。
TPD-NER、目标存活监督和 FG 不进入本轮训练。

Baseline、TPD-v1、Clean-v2/v3/v4、既有 NER 代码和全部历史结果保持原状。
V5 使用独立源码、结果根、运行单元、日志和 source lock，且必须从共享
初始化重新训练，禁止用 v4 checkpoint 热启动。

## 2. 模型公式

每个 2× 下采样单元仅使用三个既定语义源：

```text
K = Conv1x1(PixelUnshuffle2(X))
C = AvgPool2(X)
S = MaxPool2(X) - C
```

Full 的 Context code 和正向选择器为：

```text
Q = tanh((C - mean_hw(C)) / rms_hw(C - mean_hw(C)))
P = 1 + 0.5 * Q
```

融合为：

```text
R = S * tanh(saliency_scale * P)
Y = activation(K + R)
```

容量对照计算同一个 `Q`，但最终固定 `P=1`。两者参数布局、状态键和
初始化完全相同，每个 2× block 只有一个可学习 `saliency_scale`。

设计契约：

- 仍然只有 K/C/S 三源；
- `0.5 <= P <= 1.5`；
- `|R| <= |S|`；
- `S=0` 时 `R=0`；
- Context 只能正向选择现有 Saliency 响应的幅度，不能创建新支持或翻转
  全局尺度方向；
- 唯一 scale 从零初始化，step 0 与 dense SPD 逐元素相同；
- Full 与 capacity control 参数量、完整初始 state 相同；
- Q、P 和 bounded coefficient 在 FP32 中计算，只在 residual 末端转换
  回特征 dtype；
- 仅 `mtc.embeddings_1/2` 被替换，不接入 relay、NER 或额外输出。

## 3. 候选矩阵

| 变体 | Context selector | 身份 |
| --- | --- | --- |
| `tpd_clean_v5_full` | `1 + 0.5Q` | KCS 主候选 |
| `tpd_clean_v5_sal_capacity` | constant one | 同容量 Saliency 对照 |

每个变体训练 seed `42` 和 `3407`，共四个 fresh run。只使用物理
GPU 2 和 3，每张卡受控并发两个 run，并用交叉映射平衡 seed/variant：

- GPU 2：Full/42，Capacity/3407；
- GPU 3：Capacity/42，Full/3407。

GPU 0 和 1 不进入本轮任务。

## 4. 固定训练协议

- 数据：NUDT-SIRST 官方训练索引的既有 530/133 内部分割；
- 不访问官方测试集；
- patch size：256；
- batch size：16；
- epoch：800；
- optimizer、学习率、warmup、AMP、增强：继承冻结的
  `experiments/train_tpd_pilot.py` 协议；
- AMP：关闭，训练和 checkpoint 选择使用既有 FP32 路径；
- 主 checkpoint：validation Pd 最大，同 Pd 时依次选择更低 Fa、
  更高 tiny-Pd、更高 mIoU；
- 辅助 checkpoint：validation mIoU 最大；
- 固定阈值：0.5；
- Pd–Fa sweep 从正式训练前即包含
  `nextafter(float32(1), 0)` 和阈值 `1.0`；
- 五个预注册 Fa budget：
  `1e-6 / 5e-6 / 1e-5 / 5e-5 / 1e-4`。

epoch 350–400 只做运行健康和趋势检查，不提前宣布结构胜出，也不因普通
指标波动停止 800 轮训练。

## 5. 工程晋级门槛

V5 Full 必须全部满足，才允许进入 TPD-NER 正式训练。门槛沿用 v4 的
同一组数值锚点，避免看过 v4 结果后降低标准。

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
- CPU 与物理 GPU 2/3 的 RTX 5090 smoke 均通过；
- 固定阈值结果可在 sweep 中复算一致；
- 所有五个 Fa budget 均有有效工作点。

## 6. 决策

- 任一门槛失败：`engineering_gate_passed=false`，既有 TPD 主线和创新点
  保持不变，不启动 NER 正式训练，继续只优化 TPD-PE 的实现；
- 全部门槛通过：只将 V5 标记为可进入 NER 的工程候选，并启动
  五节点 NER 的独立接入与 `TPD/Progressive × NER off/on` 训练阶段；
- 即使全部通过，也不自动设置 `paper_core_established=true` 或
  `stability_claim_supported=true`，且不覆盖历史实验结论。
