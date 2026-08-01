# SCTransNet 最终模型四数据集、固定 Seed 42、1000 Epoch 论文实验完整方案

> 文件名为兼容既有引用保留“`三数据集1000Epoch`”；本文实际定义为 SIRST3 加三个单数据集，共四个训练范围。

## 0. 结论先行

当前应进入**完整模型的四数据集同协议 test-selected 对比实验准备阶段**，不再设计或增加新模块。

本轮正式实验固定为四个训练范围：

```text
SIRST3
NUAA-SIRST
NUDT-SIRST
IRSTD-1K
```

每个训练范围都分别训练：

```text
Original SCTransNet：scratch、seed 42、1000 epochs
Final SCTransNet：   scratch、seed 42、1000 epochs
```

正式训练总数为：

```text
4 training regimes × 2 methods × 1 seed = 8 formal1000 runs
```

四个训练范围使用各自现有的 `train_<dataset>.txt` 和 `test_<dataset>.txt`，不重新划分。Original 与 Final 在每个训练范围内分别选择自己的 `best_miou` 和 `best_pd`，不要求同一 epoch。

同时保留旧版方案的两层逻辑：

1. **SIRST3 混合训练层**：对齐原 SCTransNet“一份混合权重分别测试 NUAA-SIRST、NUDT-SIRST、IRSTD-1K”的协议；
2. **三个单数据集训练层**：分别在 NUAA-SIRST、NUDT-SIRST、IRSTD-1K 上训练和测试，验证专用训练条件下的性能。

SIRST3 是三个来源数据集的并集，不是第四个独立数据来源；但它是本方案中的第四个独立训练设置。

由于用户要求直接沿用现有 train/test `img_idx` 且比较各自最优 checkpoint，本方案采用与原 SCTransNet 训练代码一致的 test-selected 选模方式。所有最优结果必须标注：

```text
test_selected = true
selection_is_optimistic = true
```

正式权重产物只保存每个 run 的 `best_miou` 与 `best_pd`；epoch `10, 20, …, 1000` 的其余候选只写入指标日志，不保存权重。epoch 1000 的数值可由该日志作为固定终点诊断汇总，但没有对应 checkpoint。训练期间允许保留一个滚动覆盖的续训状态，成功完成后删除。

---

# 1. 当前研究问题

冻结的最终模型为：

```text
SCTransNet
+ TPD8-MPRS-DCH
+ 五节点 NER4 Tail-Aware
+ QFG2-CROA
+ TSS（仅训练期，weight=0.005）
```

推理图为：

```text
SCTransNet + TPD + NER + QFG
```

TSS head 与对应 state 在部署权重中移除。

当前 NUDT-SIRST 内部验证结果只能支持：

```text
final_model_established=true
```

尚不能支持：

```text
paper_core_established=true
stability_claim_supported=true
```

论文级实验需要回答四个问题：

1. **从头训练时**，完整模型是否仍优于 Original，而不是依赖历史阶段 warm-start？
2. 完整模型的收益是否在 SIRST3、NUAA-SIRST、NUDT-SIRST、IRSTD-1K 四个训练设置下成立？
3. 在唯一固定 seed 42 下，Final 相对同协议 Original 的多指标差值是否为正？
4. TPD、NER、QFG 与训练期 TSS 是否在后续消融中提供可归因贡献？

本轮不开展多 seed，因此无论结果如何：

```text
stability_claim_supported = false
multiseed_replication_supported = false
```

---

# 2. 为什么第一步应做 SIRST3 混合训练

SCTransNet 官方实现将 SIRST3 定义为 NUAA-SIRST、NUDT-SIRST 与 IRSTD-1K 的组合，并明确采用“一套权重分别测试三套数据”的结果口径。原论文训练设置为：

```text
optimizer     = Adam
initial LR    = 1e-3
minimum LR    = 1e-5
schedule      = cosine annealing
batch size    = 16
patch size    = 256 × 256
epochs        = 1000
pretraining   = none
augmentation  = random crop + flip + rotation
```

因此，第一组正式实验必须同时满足：

```text
SIRST3 mixed training
+ 1000 epochs
+ Original/Final 同协议从头训练
+ 同一权重测试三套数据
```

这组实验直接回答：

> 在与 SCTransNet 论文一致的多数据域训练条件下，完整模型是否提供更好的统一红外小目标表示能力？

仅做三个单数据集训练无法直接对齐 SCTransNet 官方“一份权重覆盖三套数据”的主结果；仅做 SIRST3 又无法回答每个数据集专用训练下的性能。因此，本方案同时执行 SIRST3 和三个单数据集训练，共四个训练范围。

---

# 3. 两层实验体系

## 3.1 第一层：SIRST3 混合训练主实验

这是论文主表和核心结论来源。

```text
SIRST3 train union
→ Original 从头训练 1000 epochs
→ Final 从头训练 1000 epochs
→ 各自在 test_SIRST3 上选自己的 best_miou / best_pd
→ 同一 checkpoint 分别测试三个官方 test split
```

每个模型只有 seed 42。SIRST3 checkpoint 先在完整 `test_SIRST3.txt` 上选定，再原样用于三个来源测试集，不得针对三个来源分别重新选择。

正确口径：

```text
Final seed 42 best_miou on test_SIRST3
→ test NUAA-SIRST
→ test NUDT-SIRST
→ test IRSTD-1K
```

错误口径：

```text
NUAA-SIRST 选 epoch A
NUDT-SIRST 选 epoch B
IRSTD-1K 选 epoch C
```

后者破坏“一套 SIRST3 权重测试三个来源”的定义。

## 3.2 第二层：三数据集独立训练实验

在主实验完成后，再开展：

```text
NUAA-SIRST train  → NUAA-SIRST test
NUDT-SIRST train → NUDT-SIRST test
IRSTD-1K train   → IRSTD-1K test
```

Original 和 Final 在每个数据集上都独立从头训练 1000 epochs，训练 seed 均固定为 42。每个 dataset × method run 使用本数据集自己的 test 指标选择 `best_miou` 和 `best_pd`。

该实验回答：

> 当模型只接触单一数据域时，TPD、NER、QFG 是否仍然有效？

它与 SIRST3 混合训练共同构成本轮“四数据集”完整实验，不是可选项，也不由 SIRST3 checkpoint 的来源拆分结果替代。

---

# 4. 当前代码中不能直接用于论文主实验的部分

## 4.1 旧 `train.py` 的 checkpoint 属于 test-selected

当前旧训练入口在训练过程中构建 `TestSetLoader`，默认从约第 500 epoch 开始在官方 test split 上评估。为降低遗漏 epoch 500 之前最佳点的风险，本轮按用户最终确认改为从 epoch 10 起每 10 epochs 评价一次；Original 与 Final 使用完全相同的候选集合和排序规则。

本方案沿用现有 `img_idx` 且不创建额外 validation，因此继续使用该选择方式，但必须准确标注：

```text
candidate_epochs = 10,20,...,1000
eval_every = 10
selection_threshold = 0.5
selection_source = current dataset test split
test_selected = true
selection_is_optimistic = true
```

该结果适合与原 SCTransNet 代码的 best-checkpoint 口径做同协议比较，但不能描述成“测试集未参与选模”的无偏泛化估计。本轮按用户确认的存储规则，只冻结 `best_miou` 与 `best_pd` 两类 checkpoint；epoch 1000 只保留在线评价数值。

## 4.2 四个训练范围使用各自冻结的 legacy normalization

为保证 Original 与 Final 同协议，并与现有 SCTransNet 实现一致，主实验冻结当前代码映射：

| Training regime | Mean | Std |
|---|---:|---:|
| SIRST3 | 101.06385040283203 | 34.619606018066406 |
| NUAA-SIRST | 101.06385040283203 | 34.619606018066406 |
| NUDT-SIRST | 107.80905151367188 | 33.02274703979492 |
| IRSTD-1K | 87.4661865234375 | 39.71953201293945 |

必须注明：SIRST3 数值是原代码硬编码并复用 NUAA-SIRST 数值，不得声称它是由 1676 张 SIRST3 train 图像重新计算所得。

在“一套 SIRST3 权重测试三个来源”的评估中，NUAA-SIRST、NUDT-SIRST、IRSTD-1K 三个测试集必须继续使用 `train_dataset_name=SIRST3` 对应的 SIRST3 normalization。三个单数据集独立训练则使用各自训练范围的 normalization。

## 4.3 当前最终训练入口是 warm-start 专用入口

现有 C/D 正式训练入口从 V4 `best_miou` checkpoint 初始化，并且固定 seed 42。它适合重现当前内部认证，但不适合回答“最终架构能否从头训练成功”。

论文主实验应新增独立入口：

```text
initialization_mode = true_scratch
parent_checkpoint   = none
optimizer_state     = new
scheduler_state     = new
completed_epoch     = 0
```

不要修改或覆盖已经封存的旧入口和结果目录。

## 4.4 当前 TSS 正样本权重是 NUDT 专用统计

现有 TSS `survival_pos_weight` 来自 NUDT-SIRST 530 张、256×256 固定图像的统计。SIRST3 包含不同分辨率、不同目标密度的数据，不能直接复用该数值。

需要分别为四个训练范围重新计算：

```text
positive stride-16 cells
negative stride-16 cells
survival_pos_weight = negative / positive
```

统计必须分别基于各自 `train_<dataset>.txt` 和被冻结的训练 crop 规则，不能读取任何 test mask。

---

# 5. 数据准备协议

## 5.1 四个训练范围与现有索引

四个训练范围全部使用当前目录中已有的 `img_idx`，不重新划分：

```text
dataset_root = /home/ly/SCTransNet_main/datasets
```

| Training regime | Train index | Train | Test index | Test |
|---|---|---:|---|---:|
| SIRST3 | `train_SIRST3.txt` | 1676 | `test_SIRST3.txt` | 1079 |
| NUAA-SIRST | `train_NUAA-SIRST.txt` | 213 | `test_NUAA-SIRST.txt` | 214 |
| NUDT-SIRST | `train_NUDT-SIRST.txt` | 663 | `test_NUDT-SIRST.txt` | 664 |
| IRSTD-1K | `train_IRSTD-1K.txt` | 800 | `test_IRSTD-1K.txt` | 201 |

索引 SHA-256：

| Dataset | Train index SHA-256 | Test index SHA-256 |
|---|---|---|
| SIRST3 | `75c32b896b95e29b89edc1f5231f619f275c2b54da0264934e6e0df13d7e7d9a` | `67a0f48b536ea6e2f8c895868c4bcd16c66c7c0a6280fd05ef7cd366d78b8922` |
| NUAA-SIRST | `324e5dadcb6cc9fc2a99a5f5dedd06ad4de77b2ed826e4ceffda8b6a784da0b4` | `e49023203a323c247306b314f23c8b3b917093a26984067792355adff7a8386e` |
| NUDT-SIRST | `e0a79f7c3d42548ba7d7dad9d2d336012b63a6bc5081e89e286f0f45036f8ec3` | `a463c52ee64b1c803c4a322fe090aaf6bc360844898e3943bb7c64a8e551b86e` |
| IRSTD-1K | `689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744` | `8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579` |

SIRST3 的 train/test 索引分别是三个来源数据集对应 train/test 索引的严格拼接。因此：

```text
SIRST3 train = 213 + 663 + 800 = 1676
SIRST3 test  = 214 + 664 + 201 = 1079
```

SIRST3 与三个单数据集结果共享底层样本，论文中应称为“四个训练设置”，不能声称它们是四份统计独立的数据来源。

## 5.2 数据目录与 `Misc_111` 修正门禁

当前 SIRST3 目录已具备 2755 对图像/mask，训练和测试均可由官方 loader 读取。SIRST3 中的 `Misc_111` 已修正为：

```text
image = 325 × 220
mask  = 325 × 220
corrected mask sha256 =
7e20ff7267737f367d2ea0545289152710225fe871d7c34c34b2d97c66b06fff
```

但是独立 NUAA-SIRST 目录当前仍为：

```text
image = 325 × 220
mask  = 592 × 400
original mask sha256 =
1bec16e5b0413d08f5b01c70faac97c72454586b03d10129fde778db4194a4aa
```

原 SCTransNet loader 对此只会分别 pad 并打印 `111`，测试循环再按图像尺寸对 mask 做左上角裁剪；它没有完成正确的 resize 或几何对齐。因此在 NUAA 独立实验启动前，必须：

1. 建立带版本号的 correction manifest，将 `NUAA-SIRST::Misc_111` 映射到已经确认的 `325×220` 修正版；
2. 复核 `Misc_111` 图像与 mask 尺寸一致；
3. 记录修正前后 SHA-256；
4. Original 与 Final 使用完全相同的数据版本；
5. 不得删除 `Misc_111` 后仍将测试集写成 214 张。

不要直接覆盖原始 NUAA mask。原文件保留为 raw 数据，正式 loader 通过 correction manifest 解析修正版路径，从而使旧实验指纹和本轮修正记录都可追踪。

在该门禁完成前：

```text
four_dataset_suite_ready = false
nuaa_dataset_ready = false
```

## 5.3 固定 split 协议

本轮不创建内部 validation，也不从现有训练索引中再次抽样：

```text
dataset_split_source = existing_img_idx
dataset_split_seed = not_applicable
training_seed = 42
```

每个训练范围必须冻结：

- 原始 train/test ID 顺序；
- train/test ID SHA-256；
- 图像和 mask 文件指纹；
- train/test 无交集检查；
- 实际样本数、目标数和有效像素数；
- 数据修正记录。

由于 best checkpoint 直接根据各自 test split 选择，结果必须始终标注 `test_selected=true`。

## 5.4 混合训练采样策略

主实验使用**自然频率拼接采样**：

```text
每张 official-train 图像每个 epoch 出现一次
整个 union 统一 shuffle
不使用 dataset-balanced sampler
```

原因是这最接近 SCTransNet 官方 SIRST3 训练方式。三个单数据集训练也各自在自己的 train 索引内自然频率采样。

数据集平衡采样可作为补充消融，但不能在看到主结果后临时启用。

## 5.5 训练 crop 与增强

统一使用：

```text
patch size        = 256 × 256
small image       = 右侧/底部零填充后裁剪
positive bias     = pos_prob 0.5
horizontal flip   = 0.5
vertical flip     = 0.5
transpose/rotate  = 0.5（按仓库既有实现）
```

同一数据集内的 Original 与 Final 必须看到完全相同的样本顺序、crop 坐标和增强变换。

正式 runner 必须将数据随机性改为无状态派生：

```text
augmentation_seed =
stable_sha256_uint64(protocol_seed, dataset_name, epoch, namespaced_sample_id)
```

禁止使用 Python 内置 `hash()`，因为它不适合作为跨进程持久协议。每个 run 还必须建立独立的 DataLoader generator，并从固定 seed 42 和 dataset name 稳定派生其 seed，不能依赖模型构建后剩余的全局 RNG。这样两个模型的训练数据流可逐样本配对复核。

---

# 6. 模型与初始化公平性

## 6.1 正式比较对象

| Method ID | 训练图 | 推理图 |
|---|---|---|
| `original_scratch` | Original SCTransNet | Original SCTransNet |
| `final_scratch` | SCTransNet + TPD + NER + QFG + TSS | SCTransNet + TPD + NER + QFG |

不允许：

- Final 从历史 V4 checkpoint warm-start；
- Original 从预训练权重开始；
- 一个模型继承 optimizer，另一个使用 fresh optimizer；
- 两个模型使用不同的数据增强、LR 或训练 epoch。

## 6.2 配对初始化

仅使用同一个 `torch.manual_seed(seed)` 还不够，因为 Final 新增模块会改变 RNG 消耗顺序，从而使共享主干参数初始值不同。

推荐实施严格的 paired initialization：

1. 构建并初始化 Original；
2. 构建 Final；
3. 对两个模型中名称、shape、dtype 均一致的公共 state，逐 tensor 将 Original 初始值复制到 Final；
4. TPD、NER、QFG 等 Final-only 参数使用由主 seed 派生的独立子 seed；
5. TSS classifier 保持严格零初始化；
6. QFG terminal projection 保持严格零初始化；
7. 输出 shared-state hash 和 extension-state hash。

派生 seed 示例：

```text
base seed = 42
TPD seed  = stable_sha256_uint64(42, "tpd")
NER seed  = stable_sha256_uint64(42, "ner")
QFG seed  = stable_sha256_uint64(42, "qfg")
TSS seed  = not used; heads are zero
```

这些是由训练 seed 42 确定性生成的模块随机子流，不是额外实验 seed；所有运行的对外 `training_seed` 仍唯一为 42。

## 6.3 随机种子

本轮所有正式训练唯一使用：

```text
seed = 42
```

seed 42 同时决定：

- Original 公共初始化；
- Final 公共初始化配对；
- Final-only 参数初始化；
- DataLoader shuffle；
- crop 与 augmentation；
- Python、NumPy、Torch CPU/CUDA RNG。

四个训练范围都使用相同的 seed 值，但每个数据集是独立训练进程和独立 checkpoint。不得在看到结果后切换 seed。

单 seed 可以用于固定协议下的 Original–Final 配对比较，但不能支持训练随机性稳定性结论。

---

# 7. 1000 Epoch 训练协议

## 7.1 统一训练配置

| 项目 | 配置 |
|---|---|
| Epochs | `1000`，所有任务完整训练，不提前停止 |
| Optimizer | Adam |
| Base LR | `1e-3` |
| Minimum LR | `1e-5` |
| Scheduler | 10-epoch warmup + cosine decay，Original/Final 相同 |
| Batch size | `16` |
| Patch size | `256` |
| Precision | FP32，AMP 关闭 |
| Workers | `0` 作为最严格确定性主设置 |
| Deep supervision | 开启，六个分割输出 |
| Segmentation loss | 六项 BCE 相加 |
| TSS weight | Final=`0.005`；Original 无 TSS |
| Eval interval | epoch 10、20、…、1000，在本数据集 test split 每 10 epochs 评价一次 |
| Default threshold | `0.5` |
| Match radius | `<3 pixels` |
| Connectivity | 8 邻域 |
| Tiny target | GT area `≤9 pixels` |

说明：公开论文只写了 cosine annealing；仓库实际工具对该 scheduler 包含 10-epoch warmup。为了与当前工程和历史模型保持一致，建议两个方法统一采用仓库实际调度实现，并在论文 Implementation Details 中明确写出 warmup，而不是只写“cosine”。

## 7.2 Original 损失


after six deep-supervision predictions:

\[
\mathcal{L}_{Original}
=\sum_{j=1}^{6}\operatorname{BCE}(P_j,Y)
\]

## 7.3 Final 损失

\[
\mathcal{L}_{Final}
=\sum_{j=1}^{6}\operatorname{BCE}(P_j,Y)
+0.005\sum_{k\in\{emb1,emb2\}}
\operatorname{BCEWithLogits}(Z_k,Y_{16})
\]

其中：

\[
Y_{16}=\operatorname{MaxPool}_{16}(Y)
\]

TSS 只影响训练，不参与 checkpoint 选择，也不进入推理图。

## 7.4 TSS pos-weight

为四个训练范围分别生成冻结统计：

```text
survival_pos_weight = negative stride-16 cells / positive stride-16 cells
```

最佳做法是基于各自被冻结的 1000-epoch crop 计划精确统计。若实现成本过高，可使用预注册数量的 deterministic crop repetitions，但必须在训练前固定重复次数，不得根据结果调整。四份统计不得互相复用，除非原始计数完全一致且有审计证据。

---

# 8. Checkpoint 选择协议

## 8.1 每个方法选择自己的最优 checkpoint

“公平比较”不等于两个方法强制使用相同 epoch。不同模型的收敛速度可能不同，因此 Original 与 Final 应各自在同一数据集、同一候选 epoch 和同一排序规则下选择自己的 checkpoint。

每个 run 保存：

```text
best_miou.pth.tar
best_pd.pth.tar
```

不保存 100 个候选 epoch 的全部权重，也不保存 `last_epoch1000` 权重。为支持中断续训，运行中仅维护一个覆盖式 resume state；run 成功结束后删除该临时状态。

固定选择协议：

```text
candidate_epochs = 10,20,...,1000
eval_every = 10
selection_threshold = 0.5
selection_loss = test_loss
test_selected = true
```

四个训练范围分别选模：

```text
SIRST3 Original / Final → test_SIRST3
NUAA Original / Final   → test_NUAA-SIRST
NUDT Original / Final   → test_NUDT-SIRST
IRSTD Original / Final  → test_IRSTD-1K
```

每个 dataset × method run 都允许选择自己的 epoch。SIRST3 选定的 checkpoint 在随后三个来源测试中必须原样复用。

## 8.2 主论文 checkpoint

建议将 `best_miou` 作为论文主工作点，与当前最终部署角色一致；`best_pd` 作为目标检测优先的第二工作点。

### `best_miou` 选择顺序

```text
1. 最大当前 test split mIoU
2. 更高 Pd
3. 更低 Fa
4. 更高 nIoU
5. 更高 tiny-Pd
6. 更低 test segmentation loss
7. 更早 epoch
```

### `best_pd` 选择顺序

```text
1. 最大当前 test split Pd
2. 更低 Fa
3. 更高 tiny-Pd
4. 更高 mIoU
5. 更高 nIoU
6. 更低 test loss
7. 更早 epoch
```

## 8.3 测试阶段禁止的行为

- 不得为 SIRST3 权重的三个来源测试集重新选三个 checkpoint；
- 不得用 SIRST3 checkpoint 替代三个单数据集独立训练结果；
- 不得在 `best_miou` 与 `best_pd` 两个预定义角色之外事后增加更有利的 checkpoint 角色；
- 不得在预注册的 epoch `10,20,…,1000` 之外扩展候选集合；
- 不得为 Original 与 Final 使用不同的评价频率或排序键；
- 不得为同一结果表中的方法分别重新校准阈值；
- 历史阈值 `0.000159` 不进入主论文默认点。

所有 best 结果均需标明它们由对应 test split 选择。

---

# 9. 评估协议

## 9.1 主固定点

所有主表均使用：

```text
threshold = 0.5
```

输出指标：

- mIoU；
- nIoU；
- F1/F-measure；
- Pd；
- Fa；
- tiny-Pd；
- false-object count；
- matched / missed target count。

## 9.2 Pd–Fa 曲线

固定点之外，对冻结 checkpoint 做完整阈值扫描：

```text
Fa budgets = 0.5e-6, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4
```

至少输出：

- 每个数据集的 Pd–Fa 曲线；
- 各 budget 下最大 Pd；
- Pareto frontier；
- 每个 budget 的实际 Fa、阈值和检出目标原始计数；
- Final 相对 Original 的配对差值。

阈值扫描用于说明可调工作区间，不替代固定 `0.5` 的默认部署结果。

不能把“某方法占有多少个 Pareto 点”作为证据强弱，因为点数受阈值采样密度和去重规则影响。

## 9.3 严格 evaluator 与论文兼容 evaluator

建议主论文全部重训模型统一使用当前冻结的严格 evaluator：

```text
8-connectivity
one-to-one matching
centroid radius < 3
fixed threshold comparison
```

为了与 SCTransNet 论文公开数值核对，可以额外运行 legacy-compatible evaluator，但必须单独成表并标注：

```text
strict evaluator results
legacy-compatible reproduction results
```

不能把两个 evaluator 的数值混在同一排名表中。

## 9.4 统计报告

固定 seed 42 的每个指标报告：

```text
Original value
Final value
paired delta = Final - Original
raw counts
```

可对同一数据集内 Original 与 Final 做 image-level paired bootstrap：

```text
bootstrap repetitions = 10,000
resampling unit        = image
confidence interval    = 95%
```

每次重采样都重新汇总 mIoU、nIoU、Pd 和 Fa，而不是只对已经汇总的单个数值做统计。

bootstrap 只描述“给定已经选定的 seed 42 checkpoint”时，当前测试图像集合上的条件性样本不确定性。它不覆盖 test-selected checkpoint 的选择过程，不能替代不同训练 seed，也不能将 `stability_claim_supported` 改为 true。

---

# 10. 完整实验矩阵

## Stage 0：工程准备与冻结

必须先完成：

1. 四数据集 `img_idx` manifest；
2. 四数据集 train/test disjoint audit；
3. 建立 `Misc_111` 版本化 correction manifest 并冻结原始/修正版哈希；
4. 四个训练范围 legacy normalization manifest；
5. 四份训练集专用 TSS statistics；
6. scratch Original builder；
7. scratch Final builder；
8. paired initialization；
9. exact resume；
10. inference export 去除 TSS；
11. 四数据集 runner 与 evaluator；
12. source lock。

当前新增 runner 尚未实现，现有 exact trainer 仍面向 NUDT、800 epochs 和 parent warm-start。因此：

```text
runner_ready = false
formal1000_running = false
```

## Stage 1：四数据集 2-epoch smoke

对四个训练范围依次执行：

| GPU | Model | Seed | Epochs |
|---|---|---:|---:|
| GPU 2 | 当前数据集 Original scratch | 42 | 2 |
| GPU 3 | 当前数据集 Final scratch | 42 | 2 |

检查：

- forward/backward 有限；
- TSS loss 非零且有限；
- QFG/NER/TPD 梯度存在；
- checkpoint strict reload；
- exact resume 与连续运行一致；
- Final training model 导出 inference model 后输出逐元素一致；
- epoch 10 之前不会进入正式 test-selected 评价，之后仅每 10 epochs 评价一次；
- 两个模型使用同一数据集、相同 batch ID、crop 和增强。

## Stage 2：四数据集正式训练

### 固定 seed 42 正式矩阵

| Wave | GPU 2 | GPU 3 |
|---|---|---|
| 1 | SIRST3 Original | SIRST3 Final |
| 2 | NUAA-SIRST Original | NUAA-SIRST Final |
| 3 | NUDT-SIRST Original | NUDT-SIRST Final |
| 4 | IRSTD-1K Original | IRSTD-1K Final |

总计：

```text
4 datasets × 2 methods × 1 seed = 8 formal1000 runs
```

每个 run 输出：

```text
1000 条 epoch records
best_miou
best_pd
epoch1000 metric record（无 checkpoint）
img_idx manifest
normalization manifest
initialization manifest
source lock
```

运行中的 exact-resume state 每个 epoch 覆盖写入，成功结束后删除，不属于正式 checkpoint 产物。

## Stage 3：checkpoint 选择与固定阈值评估

每个数据集内分别选择 Original 和 Final 的：

```text
best_miou
best_pd
```

结果数量拆分为：

```text
4 datasets × 2 methods × 2 selected roles = 16 个主/次工作点
4 datasets × 2 methods × 1 epoch1000 metric-only role = 8 个固定终点诊断
合计 = 16 个 checkpoint records + 8 个 metric-only records
```

其中：

- `best_miou` 为主表；
- `best_pd` 为检测优先工作点表；
- epoch 1000 只作为选模敏感性诊断表，不保存或复评权重。

## Stage 4：SIRST3 一份权重测试三个来源

对 SIRST3 Original 与 SIRST3 Final 已选定的两个 checkpoint 角色，额外分别评估：

```text
NUAA-SIRST test
NUDT-SIRST test
IRSTD-1K test
```

必须：

- 三个来源共用对应 SIRST3 checkpoint；
- 三个来源共用 SIRST3 normalization；
- 不为任何来源重新选择 epoch；
- 与三个单数据集独立训练结果分表。

该表用于对齐原 SCTransNet 的“一份 SIRST3 权重测试三个数据集”口径。

这些来源测试是 `test_SIRST3` 的组成子集，不是额外的独立验证。每条结果必须记录：

```text
source_subset_of_selection = true
selection_parent = test_SIRST3
```

## Stage 5：Pd–Fa 扫描与汇总

对 8 个正式 run 的 `best_miou` 和 `best_pd` 进行统一阈值扫描。SIRST3 checkpoint 另生成三个来源测试的曲线。

输出：

- 固定阈值 0.5 四数据集对照表；
- epoch 1000 在线固定终点数值表（无对应 checkpoint）；
- 预注册 Fa budget 表；
- SIRST3 一权重三来源表；
- Original–Final 配对差值；
- 可选 image-level paired bootstrap；
- checkpoint、数据、评价器与配置哈希。

## Stage 6：模块消融

至少在 SIRST3 mixed protocol 下从头训练以下模型：

| Ablation ID | TPD | NER | QFG | TSS train loss |
|---|---:|---:|---:|---:|
| A0 Original | ✗ | ✗ | ✗ | ✗ |
| A1 TPD | ✓ | ✗ | ✗ | ✗ |
| A2 TPD+NER | ✓ | ✓ | ✗ | ✗ |
| A3 TPD+NER+QFG | ✓ | ✓ | ✓ | ✗ |
| A4 Full Final | ✓ | ✓ | ✓ | ✓，0.005 |

建议：

- A0–A4 全部固定 seed 42；
- 优先在 SIRST3 mixed protocol 下完成；
- 不以增加其他 seed 作为本轮前置条件；
- 所有消融仍训练 1000 epochs，不使用历史 staged checkpoint。

额外零训练反事实：

- Final checkpoint 的 QFG alpha knockout；
- NER mask neutralization；
- TPD saliency scale neutralization；
- 移除 TSS head 后推理等价性。

零训练 knockout 只能作为机制证据，不能替代正式消融训练。

## Stage 7：外部方法公平比较

最少包含：

- Original SCTransNet；
- DNANet；
- UIU-Net；
- ACM/ALCNet 中至少一个；
- 至少两个具有公开代码的近期 IRSTD 方法。

优先级：

```text
同代码环境重训 > 官方公开权重重评估 > 论文 reported numbers
```

若只引用 reported numbers，需要用 `†` 标注，并说明其训练、阈值或 evaluator 可能不同。不能对 reported numbers 做与本方法相同的统计显著性比较。

---

# 11. 推荐代码修改

所有新增工作使用新文件，不修改已封存的最终模型、旧训练入口和历史结果。

## 11.1 数据与 manifest

```text
experiments/build_four_dataset_imgidx_manifest_v1.py
experiments/audit_four_dataset_pairs_v1.py
experiments/create_and_verify_nuaa_misc111_correction_v1.py
experiments/freeze_four_dataset_legacy_norm_v1.py
experiments/compute_four_dataset_tss_statistics_seed42_v1.py
experiments/paper_four_dataset_v1.py
```

### `build_four_dataset_imgidx_manifest_v1.py`

职责：

- 读取四个训练范围现有 train/test txt；
- 校验图像与 mask 存在；
- 校验 train/test 不相交；
- 校验 SIRST3 是三个来源对应 split 的拼接；
- 写出 canonical JSON/JSONL；
- 写出文件和 ID SHA-256；
- 记录实际图像数量、尺寸和目标数量。

### `paper_four_dataset_v1.py`

职责：

- 通过 manifest 访问四个训练范围；
- 返回 `image, mask, source_dataset, sample_id`；
- 实现 stateless crop/augmentation；
- train/test 两种模式；
- test 使用完整图像并 pad 到 32 的倍数；
- 对图像和 mask 尺寸不一致直接报错，禁止静默左上角裁剪；
- 记录每次正式 test 评价的 checkpoint 角色。

## 11.2 模型构建

```text
experiments/build_paper_original_and_final_v1.py
```

建议 API：

```python
build_paired_paper_models(
    dataset_name: str,
    seed: int,
    final_with_tss: bool = True,
) -> tuple[OriginalSCTransNet, FinalSCTransNet, dict]
```

该 builder 需要输出：

- 两个模型参数量；
- 公共 state key；
- 公共 state tensor hash；
- Final-only state key；
- TSS/QFG 零初始化检查；
- 初始化 seed 派生记录；
- `warm_start_used=false`。

现有 formal builder 对 seed 42 和历史父 checkpoint 有严格约束，因此论文 builder 应独立存在，不能把 `--fresh` 继续解释成 parent warm-start。

## 11.3 论文训练入口

```text
experiments/train_four_dataset_original_final_seed42_exact_v1.py
experiments/launch_four_dataset_original_final_seed42_2x5090_v1.sh
```

建议 CLI：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3 \
python3 experiments/train_four_dataset_original_final_seed42_exact_v1.py \
  --method final_scratch \
  --dataset SIRST3 \
  --dataset-root /home/ly/SCTransNet_main/datasets \
  --manifest experiments/manifests/four_dataset_imgidx_v1.json \
  --normalization-manifest experiments/manifests/four_dataset_legacy_norm_v1.json \
  --tss-statistics experiments/manifests/four_dataset_tss_seed42_v1.json \
  --seed 42 \
  --physical-gpu-index 3 \
  --expected-gpu-uuid GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3 \
  --device cuda:0 \
  --epochs 1000 \
  --begin-test 10 \
  --eval-every 10 \
  --batch-size 16 \
  --patch-size 256 \
  --base-lr 0.001 \
  --min-lr 0.00001 \
  --warmup-epochs 10 \
  --threshold 0.5 \
  --match-radius 3 \
  --tiny-area 9 \
  --fresh-scratch
```

Original 使用相同命令，仅修改：

```text
--method original_scratch
--physical-gpu-index 2
--expected-gpu-uuid GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
CUDA_VISIBLE_DEVICES=GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562
```

每个 wave 再将 `--dataset` 依次替换为 `NUAA-SIRST`、`NUDT-SIRST`、`IRSTD-1K`。每个 run 使用独立输出目录，禁止覆盖日志或 checkpoint。

GPU 2/3 只作为物理卡编号记录。每个 worker 必须先用对应 GPU UUID 将进程限制为单卡，进程内统一使用 `cuda:0`；禁止同时混用物理 `cuda:2/3` 和单卡可见映射。

## 11.4 测试与汇总

```text
experiments/select_four_dataset_test_checkpoints_v1.py
experiments/evaluate_four_dataset_seed42_v1.py
experiments/evaluate_sirst3_three_official_tests_v1.py
experiments/evaluate_four_dataset_pd_fa_sweep_v1.py
experiments/bootstrap_four_dataset_paired_comparison_v1.py
experiments/finalize_four_dataset_seed42_paper_results_v1.py
experiments/export_four_dataset_final_inference_v1.py
```

`evaluate_sirst3_three_official_tests_v1.py` 必须：

- 验证 checkpoint 已提前冻结；
- 验证 default threshold 为 0.5；
- 验证三个测试集共用训练 normalization；
- 只读取无 TSS 的 Final inference model；
- 每次评估写入模型、数据、evaluator 和 checkpoint SHA。

## 11.5 测试文件

```text
tests/test_four_dataset_imgidx_and_counts.py
tests/test_nuaa_misc111_alignment.py
tests/test_four_dataset_legacy_normalization.py
tests/test_four_dataset_stateless_augmentation.py
tests/test_paper_paired_initialization.py
tests/test_paper_true_scratch_initialization.py
tests/test_four_dataset_tss_statistics.py
tests/test_four_dataset_exact_resume.py
tests/test_test_selected_checkpoint_policy.py
tests/test_final_tss_strip_equivalence.py
tests/test_three_dataset_checkpoint_reuse.py
tests/test_paper_source_lock.py
```

---

# 12. 训练前硬性检查

必须全部通过：

```text
[ ] SIRST3 1676/1079 已复核
[ ] NUAA-SIRST 213/214 已复核
[ ] NUDT-SIRST 663/664 已复核
[ ] IRSTD-1K 800/201 已复核
[ ] 四个训练范围 train/test 无交集
[ ] 八份 img_idx SHA-256 已冻结
[ ] NUAA Misc_111 correction manifest 已启用
[ ] NUAA loader 解析后的 Misc_111 为 325×220，mask SHA-256 与 SIRST3 修正版一致
[ ] 四个训练范围 legacy normalization 已冻结
[ ] 四份 TSS pos-weight 已冻结
[ ] Original 与 Final 都是 true scratch
[ ] shared-state 初始化逐 tensor 一致
[ ] Final-only 参数初始化符合协议
[ ] TSS heads 严格零初始化
[ ] QFG terminal projections 严格零初始化
[ ] 四个训练范围 2-epoch smoke 均通过
[ ] 2-epoch exact resume 等价
[ ] epoch 10、20、…、1000 test-selected 行为与协议一致
[ ] Final training graph → inference graph 输出等价
[ ] 普通 Python 与 python -O 测试通过
[ ] GPU 2/3 smoke 通过
[ ] source lock 与 environment manifest 完整
[ ] runner_ready=true
```

---

# 13. 论文级成功门槛

实验结果不能通过事后修改阈值、checkpoint、seed 或数据划分改变结论。本阶段不设置“只看 mIoU”或单指标硬门槛，而按 Pd、Fa、mIoU、nIoU、F1 和 tiny-Pd 的整体变化分类。

## Gate P-A：工程完整性

- 8 个正式 run 均完成 1000 epochs；
- exact resume 通过；
- checkpoint、日志、manifest、sweep 完整；
- test-selected 行为被完整记录；
- TSS 不存在于推理 state；
- 所有结果可由 bundle 独立重建。

## Gate P-B：公平性

- Original/Final 使用完全相同的训练样本与增强流；
- 训练预算、optimizer、scheduler、batch、patch 一致；
- 两个方法均为从头训练；
- 公共参数初始 state 配对一致；
- 数据增强采用稳定摘要派生种子，Original/Final 的样本、crop 与变换逐项一致；
- DataLoader generator 独立于模型构建 RNG；
- 同一数据集内每个方法依据同一 test-selected 规则选择自己的 checkpoint；
- 四个训练范围都只使用 seed 42。

## Gate P-C：跨数据集相对性能

按以下四类裁决：

```text
POSITIVE_MULTI_METRIC
Final 在主要数据集上改善一个或多个关键指标，
同时 Pd、Fa、mIoU 不出现无法解释的整体退化。

POSITIVE_PD_FA_TRADEOFF
Final 在主要 Fa budget 提高 Pd，或明显降低 Fa，
但在 mIoU 或极端工作区间付出有限代价。

INCONCLUSIVE_MIXED_TRADEOFF
Original 与 Final 在不同数据集或指标上各有优势，
尚无清晰总体方向。

NEGATIVE_DOMINATED
Final 在 Pd、Fa、mIoU 及预算扫描上整体被 Original 支配。
```

主判断必须同时检查：

- 四个 dataset-specific `best_miou` 结果；
- 四个 dataset-specific `best_pd` 结果；
- 八个 epoch 1000 在线固定终点数值；
- SIRST3 一份权重测试三个来源；
- 六个预注册 Fa budget；
- 原始目标与虚警计数。

## Gate P-D：固定 seed 结论边界

若 Final 获得正向结果，可以更新：

```text
fixed_seed42_four_dataset_performance_supported = true
```

但本轮始终保持：

```text
stability_claim_supported = false
multiseed_replication_supported = false
```

跨四个训练设置都提升，说明 seed 42 下具有跨数据设置的一致性；它不等同于跨随机初始化稳定性。

## Gate P-E：模块贡献

- A1 相对 A0 支持 TPD；
- A2 相对 A1 支持 NER；
- A3 相对 A2 支持 QFG，至少在 Pareto 或一个主要指标上有独有贡献；
- A4 相对 A3 支持 TSS 训练约束，或至少证明 TSS 是必要的优化稳定器；
- 若某模块只产生权衡，应按 tradeoff 写入论文，不能声称全面提升。

只有 P-A、P-B、P-C 以及必要消融得到合理支持，才建议评估：

```text
paper_core_established=true
```

`stability_claim_supported` 不随本轮结果改变。

---

# 14. 结果不同时的解释规则

## 情况 1：四个训练设置下 Final 多指标整体优于 Original

结论：

```text
固定 seed 42 下，完整架构具有跨训练设置的整体收益
```

随后执行完整消融和外部方法比较；不将其写成多 seed 稳定性结论。

## 情况 2：只在 NUDT 提升，NUAA/IRSTD 退化

结论：

```text
模型对 NUDT 场景有明显适配，但跨数据设置收益尚未建立
```

论文主张应收缩为 NUDT-oriented improvement，不能写成四数据集通用改进。

## 情况 3：Final from-scratch 失败，但历史 staged model 成功

说明收益可能来自：

```text
architecture + staged curriculum / warm-start
```

此时可新增一个预注册的 `final_curriculum` 实验作为训练策略对照，但不能继续把历史结果解释成纯架构收益。

## 情况 4：固定阈值混合，但 sweep 有独有 Pareto 点

可判为：

```text
PARETO_MIXED_TRADEOFF
```

论文应写“改善特定低 Fa 或高召回工作区间”，不能写“全面超过”。

## 情况 5：Final 被 Original 在大多数训练设置和指标上支配

结论：

```text
完整模型代码成立，但论文核心性能假设未成立
```

不能通过切换测试阈值、隐藏不利指标或只报告 NUDT 结果改变结论。

---

# 15. 论文表格规划

## Table 1：数据集与划分

| Dataset/training regime | Train | Test | Split source | Seed | Epochs |
|---|---:|---:|---|---:|---:|
| SIRST3 | 1676 | 1079 | existing img_idx | 42 | 1000 |
| NUAA-SIRST | 213 | 214 | existing img_idx | 42 | 1000 |
| NUDT-SIRST | 663 | 664 | existing img_idx | 42 | 1000 |
| IRSTD-1K | 800 | 201 | existing img_idx | 42 | 1000 |

## Table 2：四数据集各自训练的 best-mIoU 主结果

| Train/Test dataset | Method | Selected epoch | mIoU ↑ | nIoU ↑ | F1 ↑ | Pd ↑ | Fa ×10⁻⁶ ↓ | tiny-Pd ↑ | False objects ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SIRST3 | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SIRST3 | Final | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUAA-SIRST | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUAA-SIRST | Final | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUDT-SIRST | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUDT-SIRST | Final | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| IRSTD-1K | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| IRSTD-1K | Final | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

该表所有行均标注 `seed=42`、`test_selected=true`。

## Table 3：SIRST3 一套权重、三个来源测试

| Test source | SIRST3-trained method | Checkpoint role | mIoU ↑ | nIoU ↑ | F1 ↑ | Pd ↑ | Fa ↓ |
|---|---|---|---:|---:|---:|---:|---:|
| NUAA-SIRST | Original | best_miou | TBD | TBD | TBD | TBD | TBD |
| NUAA-SIRST | Final | best_miou | TBD | TBD | TBD | TBD | TBD |
| NUDT-SIRST | Original | best_miou | TBD | TBD | TBD | TBD | TBD |
| NUDT-SIRST | Final | best_miou | TBD | TBD | TBD | TBD | TBD |
| IRSTD-1K | Original | best_miou | TBD | TBD | TBD | TBD | TBD |
| IRSTD-1K | Final | best_miou | TBD | TBD | TBD | TBD | TBD |

三个来源必须复用各自方法在 `test_SIRST3` 上选定的同一 checkpoint。

主表展示 `best_miou`；`best_pd` 的另外 6 条来源结果采用相同表结构放入 Table 3b 或补充材料，不得只计算而不归档。

## Table 4：best-Pd 与 epoch-1000 数值诊断

分别采用 Table 2 的结构形成两张子表：

- Table 4a：每个 run 自己的 `best_pd`；
- Table 4b：每个 run 在 epoch 1000 在线评价得到的指标；该表没有 checkpoint 路径或 SHA，也不用于三来源复评。

## Table 5：模块消融

| TPD | NER | QFG | TSS | Params | mIoU | nIoU | F1 | Pd | Fa |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

## Table 6：效率

| Method | Inference Params | Training-only Params | FLOPs | FPS | Latency median/p95 | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Original |  | 0 |  |  |  |  |
| Final |  | 98 |  |  |  |  |

最终模型公开代码当前记录的推理参数量约为 10.87M，TSS 训练期仅增加 98 个参数；论文最终数值仍应由统一 profiler 在同一输入与硬件下重新测量。

## Table 7：低 Fa 预算

| Dataset | Method | Pd@0.5e-6 | Pd@1e-6 | Pd@5e-6 | Pd@1e-5 | Pd@5e-5 | Pd@1e-4 |
|---|---|---:|---:|---:|---:|---:|---:|

---

# 16. 论文图形规划

1. **整体架构图**：SCTransNet + TPD + NER + QFG；TSS 用虚线标注“training only”。
2. **四数据集 Pd–Fa 曲线**：固定 seed 42 的 Original vs Final。
3. **定量柱状图**：四个训练设置的 mIoU、Pd、Fa 配对差值。
4. **可视化对比**：输入、GT、Original、Final、误差图。
5. **正向与负向案例**：不得只选最终模型优势图像。
6. **模块响应图**：TPD saliency、NER masks、QFG factors。
7. **收敛曲线**：train loss、test-selected mIoU、Pd、Fa，并明确其选择用途。
8. **固定 seed 配对差值图**：四个训练设置的 Original–Final delta，不命名为随机稳定性图。

可视化样本应按预注册规则选择，例如：

- 最大 mIoU 改善；
- 最大 mIoU 退化；
- tiny target；
- 多目标；
- 云层/海面/地面强杂波；
- Original false alarm 被 Final 消除；
- Final 新增 false alarm 的失败案例。

---

# 17. 论文 Experimental Setup 可直接采用的描述框架

> We evaluate Original SCTransNet and the frozen final architecture under four training regimes: SIRST3, NUAA-SIRST, NUDT-SIRST, and IRSTD-1K. For each regime, both methods are trained independently from scratch for 1,000 epochs using the existing train/test index files and a fixed training seed of 42. The two methods share the same optimizer, learning-rate schedule, batch size, crop size, data order, augmentation policy, and checkpoint-selection rule. To reduce the risk of missing an earlier performance peak while limiting evaluation cost, epochs 10, 20, ..., 1,000 are evaluated on the corresponding test split, and each method independently selects and preserves only its best-mIoU and best-Pd checkpoints from these 100 candidates. We explicitly label these results as test-selected. Epoch-1,000 metrics are retained only as an online numeric diagnostic, without preserving an additional checkpoint. For the SIRST3 regime, each checkpoint selected on the combined SIRST3 test split is reused without reselection on the NUAA-SIRST, NUDT-SIRST, and IRSTD-1K test subsets, matching the one-weight-for-three-datasets setting. The default operating threshold is 0.5. We report mIoU, nIoU, F-measure, Pd, Fa, tiny-target Pd, false-object counts, and Pd under predefined Fa budgets.

论文中还应明确：

- Final 从头训练，不继承当前 NUDT staged checkpoint；
- TSS weight 为 0.005，只用于训练；
- TSS state 从推理模型中移除；
- SIRST3 一权重三来源评估使用同一 SIRST3 normalization；
- 三个单数据集训练使用各自 legacy normalization；
- 全部结果固定 seed 42，不声称多 seed 稳定性；
- best 结果由 test split 选模，属于 optimistic/test-selected；
- 每个 run 只保存 `best_miou` 与 `best_pd` 两个正式 checkpoint；
- epoch 1000 仅报告训练时在线评价数值，不保存第三份 checkpoint。

---

# 18. 推荐的实际执行顺序

```text
1. 冻结 PAPER_FOUR_DATASET_SEED42_PROTOCOL_V1
2. 冻结四套现有 img_idx 与 SHA-256
3. 建立 Misc_111 版本化 correction manifest 并完成数据复核
4. 冻结四个 legacy normalization 配置
5. 计算四份 TSS class statistics
6. 实现 true-scratch Original/Final paired builder
7. 实现四数据集 runner、checkpoint selector 与 evaluator
8. 完成四数据集 2-epoch smoke 和 exact-resume 测试
9. Wave 1：GPU2/3 启动 SIRST3 Original/Final，seed 42，1000 epochs
10. Wave 2：GPU2/3 启动 NUAA Original/Final，seed 42，1000 epochs
11. Wave 3：GPU2/3 启动 NUDT Original/Final，seed 42，1000 epochs
12. Wave 4：GPU2/3 启动 IRSTD-1K Original/Final，seed 42，1000 epochs
13. 冻结 8 个 run 各自的 best_miou / best_pd
14. 用 SIRST3 checkpoint 一次性评估三个来源 test
15. 完成 fixed-0.5 表、Pd–Fa sweep 和可选 paired bootstrap
16. 按多指标规则做四数据集主实验裁决
17. 开展 A0–A4 固定 seed 42 从头训练消融
18. 重训或重评外部开源 baselines
19. 测量参数量、FLOPs、延迟、FPS、显存
20. 生成论文表格、曲线和正负可视化案例
21. 最终决定 fixed_seed42_four_dataset_performance_supported / paper_core_established
```

---

# 19. 最终建议状态

在开始实验前：

```text
decision=ENTER_FOUR_DATASET_SEED42_RUNNER_IMPLEMENTATION
architecture_frozen=true
new_module_design_authorized=false
paper_experiment_authorized=true
training_seed=42
epochs=1000
formal_runs=8
runner_ready=false
nuaa_dataset_ready=false
formal1000_running=false
stability_claim_supported=false
```

工程准备全部通过后：

```text
decision=START_FOUR_DATASET_SEED42_FORMAL1000
training_regimes=SIRST3,NUAA-SIRST,NUDT-SIRST,IRSTD-1K
epochs=1000
methods=original_scratch,final_scratch
seed=42
formal_runs=8
gpu_assignment=2,3
default_threshold=0.5
test_selected=true
```

当前不应继续基于现有 NUDT staged checkpoint 扩展模块。最优先的工程任务是通过版本化 correction manifest 修正 NUAA `Misc_111`、实现四数据集统一 runner，并通过四套 smoke test。随后依次完成四个训练范围的 Original/Final 配对训练。

最终实验定义为：

> **SIRST3、NUAA-SIRST、NUDT-SIRST、IRSTD-1K 四个训练范围，各自使用现有 img_idx；Original 与完整最终模型均从头训练 1000 epochs、唯一固定 seed 42，并只冻结、比较各自的 best-mIoU 与 best-Pd checkpoint。**

其中 SIRST3 结果还要按原 SCTransNet 协议，将同一 SIRST3 checkpoint 原样评估三个来源测试集。该完整矩阵用于判断最终模型在固定 seed 42 下是否真正取得跨数据设置的性能收益。
