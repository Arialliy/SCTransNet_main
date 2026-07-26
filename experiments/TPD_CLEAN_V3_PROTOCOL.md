# TPD-Clean-v3 KCS 融合优化协议

## 1. 目标与边界

本阶段只优化浅层 `embeddings_1/2` 内的 Keep–Context–Saliency 融合。
它不增加第四个并列分支，不接入 NER，不改 encoder、SCTB、decoder、loss、
数据划分或评估器，也不改写 formal800、TPD-Clean-v2、冻结 SPD/TPD-v1
及 NER 的任何产物。

TPD-Clean-v2 的直接现象是：Full 的 Saliency 门控量级相对 sal-only
下降，而未经校准的稠密 Context 残差参与相加；Full 在高 Pd 区域的 Fa
高于 sal-only。因此 v3 不再让 Context 独立产生空间响应，而是让 Context
只条件化 Saliency。

## 2. 冻结模型

每个 2× 下采样单元保持三路：

\[
\begin{aligned}
K &= W_k(\operatorname{PixelUnshuffle}_2(X)),\\
C &= \operatorname{AvgPool}_2(X),\\
S &= \operatorname{MaxPool}_2(X)-C,\\
D &= C-\operatorname{Mean}_{HW}(C),\\
Q_c &= \tanh\left(D\cdot
(\operatorname{Mean}_{HW}(D^2)+10^{-6})^{-1/2}\right).
\end{aligned}
\]

主候选 `tpd_clean_v3_full`：

\[
Y=A\left[K+S\odot\left(\tanh(g_s)+\tanh(g_c)\odot Q_c\right)\right].
\]

容量对照 `tpd_clean_v3_sal_capacity` 令 \(Q_c=1\)：

\[
Y_{\mathrm{cap}}=A\left[K+S\odot
\left(\tanh(g_s)+\tanh(g_c)\right)\right].
\]

容量对照保留相同的两组 gate、参数数、状态键、初始化和最大残差范围，
但不使用 Context 信息。它只用于区分 Context 条件化与第二组 Saliency
缩放容量，不是候选主线。

硬结构约束：

- Keep 使用与 SPD 相同的 dense `4C→C` 1×1 projection；
- 仍然只有 Keep、Context、Saliency 三路；
- \(S=0\) 时 Context 交互残差严格为零；
- \(|Q_c|\leq1\)，Context 交互幅度不超过 Saliency；
- 两个 gate 零初始化，step-0 与 SPD 六输出逐位相同；
- 不使用 v2 checkpoint warm start；
- 浅层参数固定为 `66,496`，整网参数固定为 `10,843,475`。

## 3. 四卡预登记运行

数据集仅使用 NUDT-SIRST 官方训练索引的 530/133 内部划分；
`split_seed=20260722`，不读取官方测试索引。训练配置与既有 formal800
保持一致：

- epochs `800`；
- batch size `16`，patch size `256`；
- Adam，base LR `1e-3`，min LR `1e-5`；
- 10 epoch warmup 后 cosine decay；
- FP32，六输出 BCE deep supervision；
- 每 epoch 内部验证；
- 同时保存 Pd-primary `best.pth.tar` 与 mIoU-primary
  `best_miou.pth.tar`。

预登记模型种子为 `42` 与 `3407`。四个 job 固定映射：

| job | variant | seed | GPU UUID |
| --- | --- | ---: | --- |
| full-s42 | `tpd_clean_v3_full` | 42 | `GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70` |
| cap-s42 | `tpd_clean_v3_sal_capacity` | 42 | `GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640` |
| full-s3407 | `tpd_clean_v3_full` | 3407 | `GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562` |
| cap-s3407 | `tpd_clean_v3_sal_capacity` | 3407 | `GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3` |

两模型在同一 seed 下必须具有相同初始 state checksum、相同训练样本顺序
和相同增广随机流。四卡允许与现有任务共享；结果只比较模型指标，不比较耗时、
吞吐或显存效率。

独立结果根：

`experiments/results/tpd_clean_v3_screen800_4x5090_v1/`

## 4. 运行前门槛

两个候选必须全部通过：

1. CPU 前向、反向、两步更新；
2. RTX 5090 前向、反向、两步更新；
3. 六个输出形状与有限值检查；
4. 14 个 KCS gate tensor 和 14 个 dense Keep tensor 均有有限非零梯度并更新；
5. step-0 与 SPD 六输出逐位相等；
6. state dict `strict=True` 重建；
7. 重建后六输出最大绝对差为 0；
8. 两候选完整初始化 checksum 相同；
9. Clean-v2 与 NER 既有 source lock 仍全部匹配。

任一项失败则不启动 formal800。

## 5. 800 epochs 后的工程门槛

四个 run、八份 checkpoint 与八份 Pd–Fa sweep 必须完整。`seed=42`
用于与冻结旧结果直接比较；`seed=3407` 用于检查新模型与容量对照的配对方向。

主候选进入下一轮设计必须同时满足：

1. `seed=42` Pd-primary 固定阈值：`Pd≥188/189`、`Fa≤5e-6`、
   `mIoU≥0.9336470588`；
2. `seed=42` mIoU-primary 固定阈值：`mIoU≥0.946542`、
   `Pd≥187/189`、`Fa≤1e-6`；
3. `seed=42` 在 `Fa≤1e-6` 达到至少 `187/189`，在
   `5e-6、1e-5、5e-5、1e-4` 四个预算均达到至少 `188/189`；
4. `seed=42` 至少一个预设预算严格优于冻结 SPD，且在
   `Fa≤5e-6` 不弱于 v2 sal-only；
5. 两个 seed 上，Full 均不能被同 seed capacity control 在
   `Pd、Fa、mIoU` 联合支配；
6. 两个 seed 上，Full 至少在一个低 Fa 预算或 mIoU 上严格优于
   capacity control，并且较宽预算 Pd 不退化超过一个目标；
7. 固定阈值与预算扫描不能给出相反的总体方向。

上述门槛只决定是否继续优化本模型或设计后续模块。即使通过，两个 seed、
一个内部验证划分仍不足以替换现有 TPD-v1 主线；正式替换仍需至少三个配对
seed、更多数据集与既定统计确认。
