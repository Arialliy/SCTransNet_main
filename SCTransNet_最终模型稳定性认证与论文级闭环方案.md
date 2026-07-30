# SCTransNet 最终模型稳定性认证、官方测试与论文级闭环方案

> 适用模型：**SCTransNet + TPD V8-MPRS-DCH + 五节点 NER V4 Tail-Aware + QFG-V2-CROA**
> 训练期辅助：Target Survival Supervision（TSS）
> 推理期结构：TSS head 已严格移除
> 当前决策：`SELECT_D_TSS_QFG`
> 当前证据边界：NUDT-SIRST 官方训练集 530/133 内部划分、split seed 20260722、训练 seed 42
> 当前评估边界：默认阈值 0.5，同时已有完整 threshold sweep 与五个 Fa budget；最终模型正式闭环尚未访问官方 test

---

## 0. 2026-07-30 本轮固定 seed 42 执行口径

本节是用户确认后的本轮执行约束，优先于本文后续仍保留的多 seed
论文级扩展蓝图。

必须区分两类产物：

1. 既有模块阶段的 seed-42 正式结果：
   - `tpd_ner_v4_survival_exact_v1/.../seed_42_formal800_tss`
   - `tpd_ner_v4_qfg_v2_croa_exact_v2_optimized/.../seed_42_formal800_tss_qfg`
2. 本轮最终模型认证新实验：
   - 必须使用新的 run identity、合同、日志、exact journal 和输出目录；
   - 不得把上述旧目录复制、改名或直接登记为本轮新实验。

本轮新实验固定为：

```text
certification_training_seed = 42
builder_compatibility_seed = 42
extension_initialization_seed = 42
parent_training_seed = 42
split_seed = 20260722
arms = [B / V4-stack+TSS, D / Full-stack]
epochs = 800
threshold = 0.5
trainable_scope = all_model_parameters
optimizer = fresh Adam for each arm
parent_checkpoint_usage = initialization_only
GPU2 = B
GPU3 = D
output_root = experiments/results/final_model_seed42_certification_replay_v1
```

旧 seed-42 B/D 只作为历史模块结果和数值交叉核验；本轮新 seed-42 B/D
才作为当前认证实验。两者即使因确定性训练得到完全相同的 tensor、曲线或指标，
也必须保持独立身份并分别报告。

seed 3407 若已经启动，只保留为补充压力轨迹，不替代、不参与本轮固定 seed-42
主判定。seed 426780603 本轮取消，不得由旧 launcher 自动启动。

本轮固定 seed 42 的结果只能支持单随机性的工程复验。无论结果是否提升，均保持：

```text
paper_core_established = false
stability_claim_supported = false
multiseed_replication_supported = false
```

本轮性能判断必须同时报告 Pd、Fa、mIoU、tiny-Pd 和错误目标，并比较 B、D
各自按预注册规则选出的最优 checkpoint；不得要求两个方法来自相同 epoch，
也不得跨 checkpoint 拼接指标。冻结部署权重、默认阈值 0.5、模型主线和创新点
均不因本轮 replay 改变。

后文关于 3407、426780603、3/5 个全新 seed、跨数据集和官方 test 的内容保留为
未来扩大证据等级时的论文级蓝图，不属于本轮 GPU 执行队列。

---

## 1. 执行结论

当前最终模型的**工程选择成立**，但下一阶段不应继续增加模块，也不应重新调整默认阈值。正确的下一步是进入：

> **Final Model Certification（最终模型认证）**

认证阶段保持以下内容完全冻结：

```text
SCTransNet backbone
TPD V8-MPRS-DCH
五节点 NER V4 Tail-Aware
QFG-V2-CROA
TSS weight = 0.005 的训练配方
默认 checkpoint = D / best_miou.pth.tar / epoch 3
默认 threshold = 0.5
匹配半径、tiny 定义、Pd/Fa/mIoU 实现
现有部署权重和 v2 manifest
```

只新增四类认证代码，不再增加模型模块：

1. QFG 功能利用率、同 checkpoint 反事实诊断与相邻模型增量比较；
2. 本轮固定 seed-42 独立认证 replay 入口，以及未来可选的多随机种子扩展入口；
3. 独立官方测试与跨数据集评估入口；
4. 统计置信区间、效率和论文汇总工具。

当前不能将 `paper_core_established` 或 `stability_claim_supported` 设为 true。原因不是最终模型没有相对改进，而是证据仍只有一个已参与模型选择的训练 seed 和一个内部划分，并且最终 D 在固定点对 `B / V4-stack+TSS` 的增益很小。

必须特别说明：**任何代码方案都不能在训练和独立测试之前诚实保证未知性能一定过线。** 本方案将“必须通过门槛”落实为：门槛预注册、不得事后降低、不得挑 seed、不得在官方测试集调阈值，只有证据全部满足才发布通过结论。

### 1.1 正式训练必须是累计整模全参数训练

本文中的 A/B/C/D 都是完整 SCTransNet child 模型，不是将 TPD、NER、TSS 或 QFG 拆出来单独训练。统一采用如下累计结构命名：

| Arm | Exact variant | 完整训练 child | 有效 eval / 目标部署图 | 证据角色 |
|---|---|---|---|---|
| Original | `original` | 原始 SCTransNet | 原始 SCTransNet | 端到端 baseline |
| A / V4-stack control child | `tss_control` | SCTransNet + TPD + 五节点 NER；TSS head 注册但 loss weight=0 | SCTransNet + TPD + 五节点 NER | `A→B` 的 TSS control |
| B / V4-stack+TSS | `tss_on` | SCTransNet + TPD + 五节点 NER + TSS loss/head | SCTransNet + TPD + 五节点 NER | TSS 训练增量 |
| C / V4-stack+QFG | `qfg_only` | SCTransNet + TPD + 五节点 NER + QFG；TSS head 注册但 loss weight=0 | SCTransNet + TPD + 五节点 NER + QFG | `C−A`：TSS-off 条件下的 QFG effect |
| D / Full-stack | `tss_qfg` | SCTransNet + TPD + 五节点 NER + TSS loss/head + QFG | SCTransNet + TPD + 五节点 NER + QFG | `D−B`：TSS-on 条件下的 QFG effect；最终完整模型 |

正式训练规则固定为：

```text
parent_checkpoint_role = best_miou
parent_checkpoint_usage = initialization_only
trainable_scope = all_model_parameters
optimizer = one single-param-group Adam per child over all model.parameters()
```

A/B/C/D 原始训练 checkpoint 都保留 `target_survival.*` state；在 validation/eval forward 中不使用 TSS。当前只有最终 D 的 deployment inference artifact 经过严格验证，物理上移除了 TSS 属性和 state keys。因此“推理图移除 TSS”不能泛化成“所有训练 checkpoint 都不含 TSS state”。

因此：

- “B / TSS-only”只表示“相对 D 没有 QFG”，不表示只训练 TSS；下文统一改称 `B / V4-stack+TSS`；
- “D / TSS+QFG”是上下游模块全部接入后的完整模型；下文统一改称 `D / Full-stack`；
- 同父 checkpoint 的作用是提供相同起点，不表示父模型权重被冻结；
- 每个 arm 都是独立 child 任务，父模型实例不会与 child 同时参与训练；“累计”描述的是结构与权重谱系；
- `QFG-off` 和逐层 knockout 只属于零训练诊断，不是新的训练 arm；
- 增量证据比较 `A→B`、`B→D`，最终性能比较 `Original→D`。

---

## 2. 当前最终结果的准确解释

### 2.1 默认部署点

依据当前已封存结果：

| 指标 | 最终 D / Full-stack |
|---|---:|
| checkpoint | `best_miou.pth.tar` |
| epoch | 3 |
| threshold | 0.5 |
| Pd | 188/189 = 0.994709 |
| Fa | 4.1302e-6 |
| mIoU | 0.937018 |
| tiny-Pd | 39/39 |
| 错误目标 | 5 |

相对 baseline 的 Pd-primary 固定点：

- Pd 与 tiny-Pd 保持不变；
- Fa 约降低 3.44 倍；
- mIoU 提升约 0.01764；
- 错误目标从 17 降至 5，减少约 70.6%。

因此，当前可以成立的结论是：

> 在 seed 42 的 NUDT-SIRST 内部验证划分上，最终模型在保持目标检出率的同时，数值上改善了背景抑制和区域质量；D 还向联合 Pd–Fa frontier 贡献了一个非孤立阈值区间。该结论是单 seed 内部验证结果，不等同于统计显著或跨随机性稳定。

### 2.2 不能夸大的部分

最终 D 相对 V4-stack 或 V4-stack+TSS 并非统一优势：

- V4 `best_mIoU`：188/189，Fa 4.2449e-6，mIoU 0.938178，错误目标 4；
- B / V4-stack+TSS：188/189，Fa 4.1302e-6，mIoU 0.936870，错误目标 5；
- D / Full-stack：188/189，Fa 4.1302e-6，mIoU 0.937018，错误目标 5。

因此在默认固定点上：

- D 相对 B 的 mIoU 只增加约 `0.000148`；
- D 与 B 的 Pd、Fa、tiny-Pd 和错误目标相同；
- D 相对 V4 仅小幅降低 Fa，但 mIoU 略低、错误目标多 1 个；
- C / V4-stack+QFG 被判为 `DOMINATED`；
- 现有选择规则支持 D 这一完整组合配方，并确认 D 存在非孤立 frontier 区间，但当前 factorial 明确不能单独建立 TSS 或 QFG 的因果归因。

这意味着下一阶段最重要的科学问题不是“再加一个模块”，而是：

> **完整 D 能否稳定复现对 Original 的端到端收益，并且 QFG 在 B→D 的相邻累计比较中提供可重复增量？**

---

## 3. 本地代码与封存产物复核结果

> 审查范围说明：本方案已直接读取本地 `main` 分支代码、v2 deployment/default-point/reproducibility manifest、final-selection、factorial 和各 checkpoint-local sweep。数值以本地封存 JSON/manifest 为权威，工作树代码用于结构和哈希复核。

### 3.1 最终推理图是正确闭合的

最终整模类位于：

- `model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py`

代码中定义了两个不同角色：

1. `TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet`：训练模型，包含 TSS heads；
2. `TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet`：部署模型，不注册 TSS heads，仅保留 TPD、NER 和 QFG。

本地封存代码的固定规模为：

| 项目 | 数值 |
|---|---:|
| QFG 参数 | 15,684 |
| TSS 训练参数 | 98 |
| 训练整模参数 | 10,870,228 |
| 推理整模参数 | 10,870,130 |
| 推理 state keys | 564 |

部署验证函数明确检查：

```text
不存在 target_survival 属性
不存在 target_survival.* state keys
QFG state keys 完整
TPD/NER/QFG 架构 manifest 完整
推理参数量与 state-key 数固定
```

因此“TSS 只在训练使用、推理完全移除”在代码边界上是成立的。

### 3.2 QFG 的作用边界保持清晰

`model/tpd_frequency_gate_v2_croa.py` 中的 QFG-V2-CROA 具有以下固定设计：

1. 对四级 encoder feature 做固定 2×2 Haar 分析；
2. 高频使用绝对幅值，频率模式为 `high_low`；
3. 每个样本进行 full-tensor RMS normalization；
4. 空间 gate logits 进行中心化和 RMS normalization；
5. 使用中心化 arctangent gate，范围严格位于 `(-0.5, 0.5)`；
6. Query factor 严格位于 `(0.5, 1.5)`；
7. frequency source 使用 `detach()`，避免旁路梯度直接进入 encoder feature；
8. terminal `gate_out.weight` 精确零初始化；
9. `tanh(alpha)` 的有效初值为 0.1；
10. 初始整模输出和共享参数首步更新保持锚定；
11. 每次整模前向只 `prepare()` 一次，四层 SCTB 复用同一个 forward-local modulation；
12. 只修改 Q，不修改 K、V、CFN 或 decoder。

整模 `_forward_with_relay()` 的顺序为：

```text
x1...x4 encoder features
→ TPD embeddings + 五节点 evidence
→ QFG.prepare(x1...x4)
→ SCTB 中 q-conv 后应用 Query-only factor
→ reconstruct(encoded) + encoder identity
→ 再加一次外层 encoder identity（实际等价于 reconstruct(encoded) + 2×identity，与原始 SCTransNet 两级残差路径一致）
→ NER q4 → q3 → q2
→ decoder
→ segmentation output
```

主线与创新点没有发生漂移。

### 3.3 父 checkpoint 与 child 完整模型的关系

当前 A/B/C/D 不是从随机初始化各自从头训练，也不是冻结父网络只训练新增模块。准确流程是：

```text
构造独立 child 模型
→ 从同一个不可变 V4 best-mIoU checkpoint 复制 544 个共享 state
→ 保留 TSS/QFG extension-only state 的 builder 初值
→ 为该 child 新建独立 Adam
→ 对 child 的全部 model.parameters() 训练 800 个 child epochs
```

这里不可变的是父 checkpoint 的身份和字节，不是 child 中继承参数的 `requires_grad`。父 reference 实例不进入 child 的前向、loss 或优化；A、B、C、D 各自训练、互不续接。

初始化方式的准确名称是 `extension_parent_warm_start`。只有同一个 child arm 中断后继续，才叫 `exact_resume`。

### 3.4 当前 seed-42 入口存在一个认证阶段必须处理的工程限制

当前正式训练入口：

- `experiments/train_tpd_ner_v4_qfg_v2_croa_exact.py`

具有以下硬编码：

```python
TRAINING_SEED = 42
...
if seed != TRAINING_SEED:
    raise ValueError("formal QFG exact builder requires seed=42")
```

模型 formal builder 也通过 `_require_formal_seed()` 强制 seed 42。更准确地说，当前 formal 合同把以下三类值都固定为 42，现有入口不能分别改变；QFG 初始化本身已经通过独立常量和 `fork_rng` 隔离，并非底层 RNG 实现完全混在一起：

1. formal builder 的兼容种子；
2. QFG hidden projection 的初始化种子；
3. 数据顺序、裁剪、增强和训练轨迹种子。

这是 seed-42 正式闭环的正确保护措施，但它使跨种子认证无法直接运行。**不能修改已封存的 formal800 文件和 source lock**，应新增 replication-only runner，将三类种子显式拆开。

### 3.5 当前优化器对完整 child 的全部参数使用同一学习率

正式 A/B/C/D runner 都会为自己的 child 新建单参数组 Adam；以 C/D runner 为例：

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

并使用：

```text
800 epochs
10 epochs warmup
cosine decay 到 1e-6
每 epoch 评估
```

D 的部署 checkpoint 出现在 child epoch 3。这只说明预注册选择规则在该 child 轨迹的早期 checkpoint 取得当前最佳 mIoU，不足以单独证明 parent 已位于最优区、QFG 只在短窗口有效或后期发生 parent drift。这些解释必须由 epoch trajectory audit 检验。

这不是要求现在更改学习率；当前模型和训练配方已冻结，首先原样复现。只有认证失败并结束本轮协议后，才能启动一个显式标记的新训练优化版本，不能把 differential LR 或新 schedule 混入当前认证。

---

## 4. 下一阶段的总策略

下一阶段分为五层，执行顺序不可颠倒：

```text
F0 现有成果、完整 arm 命名和所有 Gate 只读封存
→ F1 零训练功能审计与逐图统计缓存
→ F2 本轮固定 seed-42 累计整网独立 replay
→ （未来可选）多 seed 精确复现
→ F4a 外部数据集适配、train/validation 与效率验证（不访问 test）
→ 恢复或重建缺失 comparator checkpoint
→ 同时冻结 NUDT 与两个外部数据集的 test contracts
→ F3 NUDT official test 与 F4b 外部 test 一次性执行
→ 失败案例与统一汇总
→ 论文核心裁决
```

### 为什么不立即访问官方测试集

NUDT-SIRST 官方 test 文件物理上已经位于本机，通用数据加载器也能读取；因此不能声称机器层面的严格未知或从未接触。当前能够核验的是：**最终模型正式闭环 manifest 均标记 `official_test_accessed=false`，本轮审查没有运行官方 test 评估。**

应先完成：

- 模型、checkpoint、阈值和比较方法封存；
- seed 列表封存；
- 官方测试统计方案封存；
- 禁止 test-time sweep 的代码审计。

然后由锁定入口一次性运行官方测试。论文中只能表述为“最终模型协议未用 test 调参，并按预注册合同一次性评估”，不能扩大为机器或仓库历史上的绝对盲测。

---

## 5. F0：不可变封存

### 5.1 不允许修改的文件

至少包括：

```text
model/tpd_clean_v8_mprs_dch.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware.py
model/tpd_frequency_gate_v2_croa.py
model/tpd_query_frequency_bridge.py
model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py
model/tpd_survival.py
experiments/tpd_training_loss.py
experiments/tpd_exact_runner.py
experiments/tpd_extension_warm_start.py
experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact.py
experiments/train_tpd_ner_v4_qfg_v2_croa_exact.py
experiments/evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_pd_fa.py
experiments/evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa.py
experiments/evaluate_pd_fa_sweep.py
experiments/publish_tpd_ner_v4_qfg_v2_croa_default_operating_point_v2.py
现有 final selection、deployment、reproducibility 和 source-lock 文件
```

### 5.2 新增 tag/manifest

当前推荐 tag 尚不存在。`parent_lock` 绑定已经存在的冻结模型源码 commit：

```text
frozen_model_source_commit = a295f751470c3414bb453d702451cecde41a1524
```

本文当前仍是未跟踪工作树文件，必须先纳入明确的 certification commit；不能对当前 HEAD 打 tag 后声称已经包含本文。

由于文件不能包含承载自身的未来 commit SHA，`parent_lock` 不绑定 certification commit。新增：

```text
experiments/FINAL_MODEL_CERTIFICATION_PROTOCOL_V1.md
experiments/final_model_certification_parent_lock_v1.json
experiments/final_model_certification_source_lock_v1.json
experiments/final_model_replication_seed_contract.json
experiments/final_model_certification_release_attestation_v1.json
```

流程为：先提交 protocol/locks/本文，再在 commit 后生成只绑定该 certification commit 的 release attestation，最后将 attestation 提交并对其 commit 发布 tag `tpd8-ner4-qfg2-final-seed42-v1`。这样不存在 commit 自引用。

`parent_lock` 绑定既有模型和数据事实；`source_lock` 绑定新增认证 runner、collector、official-test evaluator 及其复用的冻结源码。parent lock 应绑定：

- `frozen_model_source_commit=a295f751470c3414bb453d702451cecde41a1524`；
- 最终 inference checkpoint SHA-256；
- D training checkpoint SHA-256；
- deployment-v2 manifest SHA-256；
- final-selection SHA-256；
- threshold 0.5；
- dataset split hashes；
- normalization；
- evaluator source hashes；
- 最终模型正式闭环的 `official_test_accessed=false`。

现有 deployment-v2、final-selection、reproducibility-manifest-v2 和 source locks 已经绑定大部分上游事实。新 parent lock 应通过 SHA 引用这些现有权威文件，不重复复制整份内容。

在认证 runner、逐图统计缓存、官方测试锁和对应测试实现之前，工程状态固定为：

```text
certification_design_reviewed=true
certification_design_complete=false
certification_implementation_complete=false
certification_execution_authorized=false
```

### 5.3 所有 Gate 共用的多指标判定合同

本项目不能只看 mIoU，也不能在结果出来后从 Pd、Fa、mIoU 中挑一项有利指标。所有确认性比较统一报告固定阈值 0.5 下的：

```text
Pd
Fa
mIoU
tiny-Pd
false objects
```

并以 image-level paired simultaneous 95% CI 计算 treatment `T` 相对 control `C` 的：

```text
ΔPd
ΔFa
ΔmIoU
Δtiny-Pd
Δfalse objects
```

共同判定式为：

```text
Pd_NI       := L95(ΔPd)       >= -δPd
tiny_NI     := L95(Δtiny-Pd)  >= -δtiny
mIoU_SUP    := L95(ΔmIoU)     > 0
mIoU_NI     := L95(ΔmIoU)     >= -δmIoU
Fa_SUP      := U95(ΔFa)       < 0
Fa_NI       := U95(ΔFa)       <= δFa
false_NI    := U95(Δfalse_objects_per_image) <= δfalse

MIOU_ROUTE :=
    Pd_NI
    and tiny_NI
    and false_NI
    and mIoU_SUP
    and Fa_NI

FA_ROUTE :=
    Pd_NI
    and tiny_NI
    and false_NI
    and Fa_SUP
    and mIoU_NI

TRADEOFF_PASS := MIOU_ROUTE or FA_ROUTE
```

规则：

- 本协议默认 `δPd=δtiny=δmIoU=δFa=δfalse=0`，不人为制造宽松非劣界限；若未来使用非零 `δ`，必须升协议版本并在看该版本任何结果前给出独立应用依据；
- `D→Original` 的确认性主路线固定为 `FA_ROUTE`；`D→B` 的确认性主路线固定为 `MIOU_ROUTE`；不能在不同 seed 或数据集上切换路线；
- 单 seed/单数据集“方向一致”定义为对应主路线的全部 point-estimate 不等式成立；4/5 seed 或 2/3 数据集的一致性都沿同一预选路线计算；
- `DOMINATES(C,T)` 定义为：C 相对 T 在 Pd、tiny-Pd、mIoU、Fa、false objects 五项上全部满足方向正确的非劣条件，并且至少一项满足相应 superiority 条件；所有“实质支配”均使用这一唯一算法；
- 每个 treatment-control family 固定包含上述五个主指标，使用 Bonferroni percentile simultaneous 95% CI：每项采用双侧 99% percentile CI，以保证该五指标 family 的覆盖率不低于 95%；
- paired bootstrap 固定 `10,000` 次，RNG seed 固定 `20260730`，同一 family 的所有方法共享完全相同的重采样索引；
- Fa 使用 `ΔFa`；Fa 为 0 时不得使用未定义的原始比值、倍数或 `log(Fa ratio)`；
- 五个 Fa budget 全量报告，其中 `Pd@Fa≤5e-6` 是 key secondary result，不参与 Gate 通过判定，也不得事后改选为主结果；
- 单 checkpoint 比较使用 image-cluster paired bootstrap；多 seed 聚合必须以 seed 为最高层配对单位或使用预注册 hierarchical bootstrap，不能把不同 seed 的图像直接池化成彼此独立样本。

hierarchical bootstrap 固定实现为：先对 paired seeds 有放回重采样，再在每个抽中的 seed 内对图像有放回重采样；同一次 draw 中 treatment/control 共用 seed 和图像索引。跨数据集 aggregate 对三个固定数据集等权，数据集本身不重采样，只在各数据集内部执行相同的 seed→image 两级重采样，再对数据集效应取等权平均。

F1 数值诊断容差固定为：

```text
full/off probability output equivalent:
    max_abs <= 1e-7 and mean_abs <= 1e-8
repeat inference equivalent:
    max_abs <= 1e-7
nontrivial factor use:
    max_level(mean(abs(factor - 1))) > 1e-4
```

---

## 6. F1：零训练功能审计

F1 不启动新训练，只回答“冻结 D checkpoint 是否实际使用 QFG”。它不能单独证明 QFG 的训练期因果收益。

### 6.1 QFG exact knockout

在同一个 D inference checkpoint 上计算：

```text
full QFG
all-level QFG off
level-1 off
level-2 off
level-3 off
level-4 off
```

最干净的 knockout 是将对应 level 的 `alpha` 临时置零：


y = Q \cdot (1 + \tanh(\alpha)G)

当 `alpha=0` 时 factor 精确为 1，不需要修改其他参数。

参考实现：

```python
from contextlib import contextmanager
import torch

@contextmanager
def temporary_qfg_knockout(model, levels=None):
    qfg_levels = model.tpd_qfg.levels
    selected = range(len(qfg_levels)) if levels is None else tuple(levels)
    snapshots = {
        index: qfg_levels[index].alpha.detach().clone()
        for index in selected
    }
    try:
        with torch.no_grad():
            for index in selected:
                qfg_levels[index].alpha.zero_()
        yield
    finally:
        with torch.no_grad():
            for index, value in snapshots.items():
                qfg_levels[index].alpha.copy_(value)
```

输出：

- full/off 概率图 `max_abs`、`mean_abs`；
- threshold 0.5 的 Pd、Fa、mIoU、tiny-Pd、错误目标；
- 五个 Fa budget 的包络；
- full/off component 差异；
- 每个 QFG level 的独立作用。

### 6.2 QFG 利用率

每层记录：


t_\ell = \tanh(\alpha_\ell)


u_\ell = \mathbb{E}|F_\ell-1|

以及：

```text
alpha effective value
gate mean/rms/p5/p50/p95/min/max
factor mean/rms/p5/p50/p95/min/max
target-cell factor
hard-negative factor
target/background factor contrast
```

由于 gate 在空间上中心化，factor 均值接近 1 并不足以证明无作用；应关注空间方差和目标/背景差异。

### 6.3 Gate M：功能激活、训练增量与独立测试分开判定

#### Gate M-functional：同 checkpoint 功能激活

在 F0 中先锁定数值等价容差和 factor 非平凡变化容差，再要求：

1. full 与 QFG-off 的输出差异超过数值等价容差；
2. 至少一层 factor 的空间变化超过测量容差；
3. 重复推理得到一致结论；
4. 目标、hard-negative、普通背景的网格映射和区域定义在审计前固定；
5. full/off/逐层关闭的全部结果完整报告，不按结果挑 level。

这只能设置：

```text
qfg_functionally_active=true
```

#### Gate M-train：相邻累计整网训练增量

QFG 的性能贡献必须由新确认性 seeds 上的 `D / Full-stack` 对 `B / V4-stack+TSS` 验证：

1. 两个 arm 从同一个 V4 parent checkpoint 初始化；
2. 两个 arm 使用相同数据顺序、增强、预算和 checkpoint 选择规则；
3. D 相对 B 满足预注册的 `MIOU_ROUTE`。

#### Gate M-test：独立测试功能依赖

在锁定的独立 test 上，full 不被同 checkpoint 的 QFG-off 按 F0 定义实质支配。该比较不选择部署模型，也不替代 D→Original 的主性能比较。

只有 `M-functional`、`M-train` 与 `M-test` 同时通过，才可设置：

```text
qfg_functional_contribution_supported=true
```

Gate S-P 只依赖 `M-train`，不等待官方 test，因而不会与 Gate T 形成循环。不能仅因为 D 在历史候选池中被选中或单 checkpoint knockout 有差异，就写成 QFG 训练因果贡献已经成立。

### 6.4 配对 bootstrap

新增 image-level paired bootstrap，比较：

```text
final D vs Original SCTransNet
final D vs SPD
final D vs V4
final D vs B / V4-stack+TSS
full QFG vs QFG-off
```

应以图像为重采样单位，而不是将像素或目标视为独立样本。每次重采样重新累计：

- matched GT 数；
- GT 总数；
- unmatched component pixels；
- valid image pixels；
- intersection/union；
- tiny target 计数；
- false component 数。

每个 checkpoint/mode 只运行一次推理，缓存逐图充分统计和必要的概率图。bootstrap 仅从缓存做 CPU 重采样，避免每次 bootstrap 重跑模型。严格使用 F0 固定的 `10,000` 次 paired bootstrap、RNG seed `20260730` 和五指标 Bonferroni percentile simultaneous CI：

```text
ΔPd
ΔFa
ΔmIoU
Δtiny-Pd
Δfalse objects
```

若补充报告连续性修正后的 Fa ratio，修正规则必须在 F0 固定，并同时保留原始错误像素、有效像素和错误目标数。

### 6.5 新增文件

```text
analysis/audit_final_qfg_functional_use.py
analysis/collect_final_model_validation_statistics.py
analysis/bootstrap_final_model_paired_metrics.py
analysis/audit_final_model_component_errors.py
analysis/audit_final_model_epoch_trajectory.py
```

---

## 7. F2：累计整模固定 seed-42 replay 与未来多随机种子扩展

### 7.0 本轮与未来扩展的边界

本轮只执行第 0 节锁定的一对新 seed-42 B/D replay。下面关于多随机种子、
confirmatory seed、3/5 seed Gate 的规定只在用户以后明确启动论文级扩展时生效，
不能据此把本轮 seed-42 替换为其他 trajectory seed。

### 7.1 父 checkpoint 与独立全参数子轨迹

A、B、C、D 的准确训练语义是：

> **同一 V4 checkpoint 初始化的独立全参数子轨迹训练。**

V4 parent 是独立完成的 fresh formal800 模型。A/B/C/D 分别构造自己的 child 模型实例，再从不可变的 V4 `best_miou.pth.tar` 执行严格 `extension_parent_warm_start`：

```text
parent checkpoint epoch = 489
parent checkpoint SHA-256 = 0ae6c0e034952e18333d8fa6ccd3bbf635cae5efa8017b06df5e00ccc4ed14ab
copied shared state keys = 544
```

随后每个 arm：

- 新建自己的 Adam，不继承 parent optimizer、schedule、epoch 或训练轨迹；
- 对 child 整模全部参数更新，继承的 SCTransNet/TPD/NER 参数并未冻结；
- 使用独立 run 目录、journal 和 checkpoint 选择；
- child epoch 从 1 重新计数，不能与 parent epoch 相加；
- parent reference 只做结构与 state-layout 校验，不进入前向、loss 或优化。

A、B、C、D 互不作为对方的 parent。尤其 D 不从 B checkpoint 继续训练；D−B 比较的是同父初始化下加入 QFG 的条件增量。

### 7.2 种子证据分层

以下三类 seed 不能混为同一等级：

| Seed | 角色 | 是否进入论文稳定性分母 |
|---:|---|---|
| 42 | 已参与模型开发和选择的既有结果 | 否 |
| 3407 | 已知历史压力轨迹 | 否，只作压力报告 |
| 426780603 | 由部署 inference artifact SHA `997027bb...` 首段确定的 hash-seed | 可作预注册复现 seed，但必须在运行前写入 lock |

hash-seed 的唯一来源固定为：

```text
hash_seed_source = deployment_inference_artifact_sha256
source_sha256 = 997027bb2cc59e0e16ef85beba2c78ab8b3e195de962acbe7c97adc8c007c63a
hash_seed = int("997027bb", 16) & 0x7fffffff = 426780603
```

不能改用 D training source checkpoint SHA `890c8cf0...`，否则会得到另一个数值 `151817456`。

证据等级：

1. 只保留 seed 42：最终模型工程选择仍成立，但 `stability_claim_supported=false`；
2. seed 42 + 3407 + 426780603：可完成工程压力与复现筛查，但不能把 2/3 规则写成确认性稳定结论；
3. 至少 3 个全新、结果未知、运行前锁定的 paired seeds：可写“多 seed 复现支持”；
4. 至少 5 个全新 paired seeds：才进入本文的论文稳定性 Gate。

新增确认性 seed 列表以尚未运行过的 `final_model_certification_source_lock_v1.json` 文件 SHA-256 为唯一输入，避免 seed contract 与自身哈希循环依赖。按 8 个十六进制字符依次取块并执行 `int(block, 16) & 0x7fffffff`，跳过 42、3407、426780603、0 和重复值；不足 5 个时对 `source_lock_sha256 + counter` 再做 SHA-256 继续取值。

最终 seed 列表在任何确认性训练前一次性写入 `final_model_replication_seed_contract.json` 并封存。seed 42 和 3407 不得计入通过率分母。

### 7.3 训练矩阵：完整模型比较与模块增量比较并行

#### 工程复现最小矩阵

不重训 seed 42 的 A/B/C/D，也不重扫已有 seed-42 B/D checkpoint。新增最小矩阵为：

| Trajectory seed | B / V4-stack+TSS | D / Full-stack | 目的 |
|---:|---|---|---|
| 3407 | 800 child epochs | 800 child epochs | 已知压力轨迹 |
| 426780603 | 800 child epochs | 800 child epochs | 确定性 hash-seed 复现 |

这四项都是独立全参数 child runs，不是单模块训练。每个 seed 下：

- B 和 D 从相同 V4 checkpoint 加载相同的 544 个共享 state；
- 两者使用相同 trajectory seed、DataLoader order、增强、optimizer、schedule 和预算；
- B 新增 TSS，D 新增 TSS+QFG，TSS weight 均为 0.005；
- D 不读取 B 的训练 checkpoint；
- 两者按同一规则各自选择自己的 `best_miou` 作为主 checkpoint；`best` 作为预注册次要 checkpoint，不能在看到 test 后二选一。

四个新 run 各保存 `last/best/best_miou`，即 12 份 checkpoint 产物；只对 `best` 和 `best_miou` 扫描，共 8 份 sweep。12 份 checkpoint 不是 12 次独立实验。

#### 论文稳定性最小矩阵

若 B/D 始终共享 seed-42 V4 parent，则无论 child trajectory 使用多少 seeds，都只能建立：

```text
fixed_parent_child_trajectory_stability_supported
```

不能写成完整训练流水线稳定。完整 Gate S-P 要求每个全新确认性 seed 至少运行：

```text
Original SCTransNet, fresh seed s
V4 parent, fresh seed s
B / V4-stack+TSS, warm-start from V4 parent seed s
D / Full-stack, warm-start from V4 parent seed s
```

其中：

- `Original→D` 回答完整模型最终性能是否稳定提高；
- `B→D` 回答 QFG 在已经接入 TPD、NER、TSS 后是否提供稳定增量；
- seed-matched V4 parent 把 parent 训练随机性纳入完整流水线，而不只改变 child trajectory；
- 如论文要单独声称 TSS 的稳定贡献，必须再加入 A / V4-stack control child，比较 `A→B`；
- 只有要建立完整 TSS×QFG interaction 时才补 C / V4-stack+QFG，不预先把机制扩展实验混入性能主线。

执行量必须区分：

| 层级 | 新训练 runs | checkpoint 产物 | sweeps |
|---|---:|---:|---:|
| 工程筛查：3407/426780603 × B/D | 4 | 12 | 8 |
| 5 seeds 固定 parent × Original/B/D | 15 | 45 | 30 |
| 5 seeds 完整流水线 × Original/V4-parent/B/D | 20 | 60 | 40 |
| 工程筛查 + 完整流水线 | 24 | 72 | 48 |
| 至少 3 seeds 的 Original-budget-matched | +3 | +9 | +6 |
| 工程筛查 + 完整流水线 + 最小预算控制 | 至少 27 | 至少 81 | 至少 54 |

以上仍不含 A/C 或外部数据集；不得用“4 runs / 8 sweeps”代表整个论文闭环。

### 7.4 训练预算比较合同

D 的两阶段训练与模型选择预算包含 V4 parent formal800 run 和 D child formal800 run；Original 标准 baseline 是 fresh formal800。必须区分：

```text
two-stage experiment/model-selection budget = V4 run 800 + D-child run 800
selected checkpoint update lineage = V4 best_miou epoch 489 → D-child best_miou epoch 3
```

两阶段都完成 800 epochs 和逐 epoch validation，但部署权重并没有经历“parent 最后 epoch + child 最后 epoch”的简单 1600 次连续更新。`Original-budget-matched` 匹配的是训练与模型选择计算预算，不是把部署权重 epoch 编号相加。因此主比较明确解释为：

```text
complete prescribed recipe performance
```

它验证最终可部署配方的性能，不自动证明收益与额外优化预算无关。所有表格必须同时报告累计 optimizer steps、累计 validation selection opportunities 和 GPU-hours。

要设置 `paper_core_established=true`，还必须增加 `Original-budget-matched` 敏感性控制：至少 3 个全新 paired seeds，使用与 V4-parent+D-child 谱系相同的累计 optimizer steps 和 validation opportunities，checkpoint 仍只由内部 validation 选择。如果 D 只胜过标准 Original800、却不支持相对预算匹配控制的非支配结论，论文只能主张“完整训练配方更优”，不能把全部收益归因于模型结构。

### 7.5 Seed contract 与真实 runner 映射

避免使用容易误解的 `parent_construction_seed`。新合同至少记录：

```python
@dataclass(frozen=True)
class ReplicationSeedScheduleContract:
    certification_source_lock_sha256: str
    trajectory_seeds: tuple[int, ...]
    builder_compatibility_seed: int = 42
    split_seed: int = 20260722
```

训练前的 seed schedule 不包含尚未生成的 seed-matched V4 parent SHA。工程固定-parent 筛查可直接引用既有 parent SHA；完整流水线则按顺序：

```text
锁定 seed schedule
→ 训练并选择该 seed 的 fresh V4 parent
→ 生成 per-seed child_initialization_manifest（记录 parent checkpoint SHA）
→ 才允许启动该 seed 的 B/D children
```

`builder_compatibility_seed=42` 只用于复用冻结 builder 并构造 child/reference state layout；共享权重随后由 per-seed manifest 指定的 parent checkpoint 覆盖，它不是重新训练或重新选择父模型。

第一轮工程复现固定 extension 初始化为 42，只改变 trajectory seed，回答“相同完整起点下训练轨迹是否可复现”。若要认证初始化稳定性，主论文轮必须同时改变 extension initialization 和 trajectory seed。当前冻结 QFG builder 内部固定初始化 42，尚不能直接完成后者；必须实现 replication-only 参数化 initializer、单独测试并生成新 source lock，不能只改 dataclass 就宣称初始化已变化。

现有 B 与 D 的冻结入口不同：

```text
B source: experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact.py
D source: experiments/train_tpd_ner_v4_qfg_v2_croa_exact.py
```

因此新增一个共享 replication core 和两个薄入口：

```text
experiments/final_model_replication_seed_contract.py
experiments/final_model_child_initialization_manifest.py
experiments/final_model_replication_exact_core.py
experiments/train_final_model_replication_b_exact.py
experiments/train_final_model_replication_d_exact.py
```

上述两个入口只覆盖工程 B/D 筛查。Gate S-P 还必须增加 Original、fresh V4 parent 和 Original-budget-matched 的独立入口；否则只能执行固定 parent child-trajectory 复现，不能执行完整流水线认证。

真实实现流程应为：

```text
arm-specific build_selected_model(..., seed=42)
→ 新 replication initialization adapter
   （内部恰好调用一次 load_parent_into_extension(...)）
→ 重置 Python / NumPy / Torch CPU / Torch CUDA RNG 到 trajectory_seed
→ DataLoader generator.manual_seed(trajectory_seed)
→ 新建 Adam(model.parameters(), lr=1e-4)
→ 新建 ExactRunSpec(seed=trajectory_seed)
```

仓库不存在 `build_frozen_model()`，不得把旧文档伪代码直接复制成实现。`PYTHONHASHSEED` 必须在启动 Python 进程前设置为 trajectory seed；GPU 2/3 各自通过 `CUDA_VISIBLE_DEVICES` 暴露单卡后，进程内设备使用 `cuda:0`。

旧 B/D `initialization_plan()` 会校验 seed-42 formal identity，并且内部已经调用 warm-start。新 replication adapter 必须复用底层 `load_parent_into_extension()` 但只调用一次；不得先运行旧 `initialization_plan()` 再次加载 parent。

### 7.6 精确续训要求

`extension_parent_warm_start` 只负责由共同 parent 初始化一个新 child；它不是 exact resume。只有同一个 child arm、同一 trajectory seed、同一 source lock 和同一 parent SHA 的中断任务才能 exact resume。

每个新 run 必须保存并恢复：

```text
Python RNG
NumPy RNG
Torch CPU RNG
Torch CUDA RNG states
DataLoader generator state
model state
optimizer state
scaler state
completed child epoch
selection state
manual LR schedule state
```

连续运行与 child epoch 边界中断续训应逐 tensor 等价。

### 7.7 Gate S：工程复现与论文稳定性分开

#### Gate S-E：工程复现完整

- 所有预注册 runs、checkpoint、sweep、日志和数据指纹完整；
- ordinary 与 `python -O` 验证一致；
- strict load、exact resume、source lock 和 run identity 验证通过；
- run identity 明确区分 builder、extension initialization、trajectory 和 split seed；
- 每个 checkpoint 的固定点、预算点和 sweep 来自同一次逐图缓存，不跨 checkpoint 拼指标。

Gate S-E 只允许设置：

```text
engineering_replication_complete=true
```

#### Gate S-R：多 seed 复现支持

至少 3 个全新 paired seeds，聚合 `D→Original` 满足 `FA_ROUTE`，同时完整报告 `D→B` 的 `MIOU_ROUTE`。通过后最多设置：

```text
multiseed_replication_supported=true
stability_claim_supported=false
```

#### Gate S-P：论文稳定性

至少 5 个全新 paired seeds，并同时满足：

1. 每个 seed 都 fresh 训练自己的 V4 parent，B/D 只从 seed-matched V4 parent warm-start；
2. 聚合 D 相对 Original 满足 `FA_ROUTE`；
3. 聚合 D 相对 B 满足 Gate M-train；
4. 至少 4/5 seeds 的 D→Original point estimate 满足同一 `FA_ROUTE` 方向；
5. 任一 seed 上 D 均不被 Original 或 B 按 `DOMINATES` 定义支配；
6. 不使用 seed 42 或已知压力 seed 3407 填充通过率；
7. 不跨 checkpoint、metric 或 Fa budget 事后拼接结论。

若五个 seeds 仍共享 seed-42 V4 parent，只能设置 `fixed_parent_child_trajectory_stability_supported=true`，Gate S-P 不通过。完整 Gate S-P 也只表示内部划分上的训练随机性稳定；最终 `stability_claim_supported=true` 还必须结合官方 test 与跨数据集 Gate。

---

## 8. F3：官方测试一次性评估

### 8.1 访问前必须冻结

新增：

```text
experiments/FINAL_MODEL_OFFICIAL_TEST_PROTOCOL_V1.md
experiments/freeze_final_model_official_test_contract.py
experiments/final_model_official_test_contract_v1.json
```

当前本地权重库存检查只确认了 Original 与 A/B/D 等后续模型；Progressive、SPD 和早期 TPD 的聚合 JSON 存在，但对应 `best`/`best_miou` checkpoint 文件尚未找到。聚合 JSON 不能替代官方 test 推理权重，因此：

```text
official_test_comparator_checkpoint_inventory_complete=false
Gate F3 execution=blocked
```

F0 必须先二选一并写入合同：

1. 恢复原始 Progressive/SPD/TPD checkpoint，逐个校验 SHA、结构和既有 sweep；
2. 若无法恢复，按冻结代码和训练协议重新训练这些 comparator，并明确标记为 reconstructed runs，不能冒充旧 checkpoint。

三者 checkpoint 库存闭合前不得生成正式 official-test contract。

合同必须固定：

```text
测试模式 = 当前 530/133 训练得到的冻结 checkpoint 直接测试
每个方法自己的 checkpoint SHA
checkpoint 选择模式 = each_method_own_frozen_validation_selection
每个方法各自的选择指标、候选集合和选择产物 SHA
threshold = 0.5
禁止 threshold sweep
match radius = 3
tiny area = 9
所有 metric 实现 SHA
不允许 test-time augmentation
不允许根据 test 结果更换 checkpoint
不允许根据 test 结果更换默认点
dataset receipts:
    experiments/results/final_model_locked_tests_v1/NUDT-SIRST/test_access_receipt_v1.json
    experiments/results/final_model_locked_tests_v1/NUAA-SIRST/test_access_receipt_v1.json
    experiments/results/final_model_locked_tests_v1/IRSTD-1k/test_access_receipt_v1.json
atomic master receipt:
    experiments/results/final_model_locked_tests_v1/atomic_test_access_receipt_v1.json
```

master receipt 绑定三个 dataset receipt、全部结果 SHA 和执行顺序；任一子结果缺失时 master 状态必须为 incomplete，不能部分宣布一次性测试闭环。

本轮 F3 明确选择“当前 530/133 训练 checkpoint 直接测试”，不在访问 test 前用全部 663 张官方 train 重训。若以后执行全官方 train 重训，必须建立新的训练和 checkpoint 选择合同，并作为另一组证据单独报告，不能与本轮 checkpoint 混在同一比较表中。

主比较遵循“各方法使用各自已经封存的 validation 最优 checkpoint”，不要求 baseline 与 V1/V2/V3/D 来自相同 checkpoint 文件角色。允许 baseline 使用自己的 Pd-primary 选择、D 使用自己的 deployment-v2 选择，但每个方法的候选集合、选择准则、checkpoint SHA 和 selection artifact 必须在 test 前固定，且只能由 train/internal-validation 信息产生。

为检查 checkpoint 角色差异是否影响结论，另预注册一张 common-role `best_miou` 敏感性表；它是完整报告，不得在看到 test 后与主表二选一。同一个方法的一行结果必须来自同一 checkpoint，不能拼接多个 checkpoint 的 Pd、Fa 和 mIoU。

### 8.2 一次性运行方式

official-test contract 分成两个固定 panel，并逐项列出 checkpoint SHA；同一次 locked execution 完整运行，不能从多个 seeds 中选择一个“代表 checkpoint”。

```text
Legacy seed-42 panel（9 passes）:
    Original SCTransNet
    Progressive
    SPD
    TPD
    V4 parent checkpoint
    A / V4-stack control child
    B / V4-stack+TSS
    D / Full-stack
    D / Full-stack with QFG-off counterfactual

Confirmatory full-pipeline panel:
    对 5 个确认性 seeds，逐 seed 运行
        Original
        seed-matched V4 parent
        B
        D full
        D QFG-off
    共 25 passes

Budget-matched sensitivity panel:
    逐项运行至少 3 个 Original-budget-matched seed checkpoints
```

其中 SPD 是 TPD 最近且当前仍具竞争力的结构对照，不能从论文级闭环中删除。Progressive 用于排除“只把大步长卷积改成多次 stride-2”这一简单解释。QFG-off 是冻结 checkpoint 的功能对照，不是新的可选部署模型，不得根据 test 结果在 full/off 中选择。所有确认性 seed 结果和 hierarchical aggregate 都必须发布。

### 8.3 官方测试 Gate T

主判定只有一个：

```text
confirmatory 5-seed hierarchical aggregate:
D / Full-stack vs Original SCTransNet
at threshold 0.5
must satisfy FA_ROUTE
```

Legacy seed-42 panel 用于与既有内部结论对照，不单独触发 Gate T。

并同时要求：

1. 原始错误像素、有效像素、错误目标、matched/total GT 完整发布；
2. `DOMINATES(SPD,D)=false`；
3. B→D 和 full→QFG-off 只承担预注册的模块增量/功能次判定，不能替代完整模型主判定；
4. 不得在 Fa 与 mIoU 中根据 test 结果任选一个更有利门槛；
5. D 的 `false_NI` 成立，且 `DOMINATES(Original,D)=false`；
6. 无论 Gate 是否通过，都保存和报告所有预注册方法的结果。

若官方测试仅表现为 Pareto tradeoff，可支持“相对改进伴随权衡”；不能写 universal dominance。

---

## 9. F4：跨数据集与效率验证

### 9.1 数据集

固定为 NUDT-SIRST 加两个外部公开红外小目标数据集，共三个数据集：

```text
NUAA-SIRST
IRSTD-1k
```

每个数据集必须：

- 使用其官方 train/test 协议，或在无官方划分时预注册 split；
- 训练期只使用 train/validation；
- 测试集只访问一次；
- 固定 threshold 0.5 作为主部署点；
- 可报告 validation-derived sweep，但不能在 test 上调阈值；
- 每个数据集分别训练同一冻结架构；这不是把 NUDT 权重直接零样本迁移到其他数据集；
- Original、SPD、B、D 使用相同数据、增强、训练预算和 seed 列表；各方法自己的 checkpoint 选择规则必须在训练前预注册，并且只使用 train/internal-validation 信息；
- 三个数据集的 split、seed、训练预算、baseline 和 test contract 在第一次执行任一正式 test 前同时封存。

### 9.2 数据适配代码

通用 `dataset.py` 已能读取 NUDT-SIRST、NUAA-SIRST 和 IRSTD-1k，但当前 exact runner/evaluator 高度绑定 NUDT-SIRST。数据注册表是复现治理增强，不是读取数据的硬前置；新 exact runner 仍必须完成数据集参数化：

```python
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    image_dir: str
    mask_dir: str
    train_index: str
    test_index: str
    image_suffixes: tuple[str, ...]
    mask_suffixes: tuple[str, ...]
    normalization_scope: str = "train_only"
```

新增：

```text
experiments/final_model_dataset_registry.py
experiments/final_model_infrared_dataset.py
experiments/train_final_model_multidataset_exact.py
experiments/evaluate_final_model_multidataset.py
```

### 9.3 论文级跨数据集门槛

Gate X 固定为：

1. NUDT-SIRST、NUAA-SIRST、IRSTD-1k 三个数据集全部完成，且每个新增数据集至少使用 3 个全新 paired seeds；
2. D 相对 Original 在至少 2/3 数据集满足预注册的 `FA_ROUTE`；
3. 任一数据集均满足 `DOMINATES(Original,D)=false`；
4. 跨数据集 hierarchical aggregate 满足同一 `FA_ROUTE`；
5. D 相对 B 在至少 2/3 数据集沿预注册 mIoU-route 方向一致；
6. 完整报告 D 与 SPD 的比较；仅“进入 Pareto frontier”不足以单独判定通过。

如果新增数据集只使用固定 seed，只能报告“跨数据集数值复现”，`Gate X=false`，不能用于 `paper_core_established` 或 `stability_claim_supported`。

### 9.4 失败案例与困难场景

论文级闭环还应固定样本选择规则，并报告：

```text
tiny target
低对比度
强 clutter / hard negative
边缘附近目标
密集或相邻目标
典型 false-positive
Original/Progressive/SPD/TPD/Full-stack 各方案失败案例
```

不得只挑 D 成功、baseline 失败的图。代表性、困难和共同失败案例都要按预注册规则采样。

### 9.5 效率与 Gate E

新增：

```text
experiments/profile_final_model_inference.py
```

固定报告：

```text
parameter count
state size
FLOPs/MACs
batch=1 latency
throughput
peak CUDA memory
256×256 与原图尺寸
FP32 和可选 AMP
100 warmup + 1000 timed iterations
至少 5 次独立计时重复的 median、p5、p95
CUDA synchronize 与 CUDA events
GPU 型号/UUID、驱动、CUDA、PyTorch
```

同时比较：

```text
SCTransNet baseline
V4
final D
```

QFG 固定 Haar、projection 和 factor 计算应被计入推理开销；TSS 不应计入。

Gate E 定义为：

```text
source/data/checkpoint/evaluator hashes 完整
checkpoint strict-load 与部署图验证通过
TSS 不进入推理图
exact-resume 与确定性评估验证通过
参数、state size、FLOPs、延迟、吞吐、显存完整报告
Original/V4/D 使用相同硬件、输入和计时协议
累计 optimizer steps、validation opportunities 与 GPU-hours 完整报告
至少 3 个全新 seeds 的 Original-budget-matched 敏感性控制完成
```

若论文不主张“轻量化”，Gate E 只检查报告完整性，不设置开销通过线；若要主张低开销，必须在测量前另行锁定最大参数、延迟和显存增幅。

---

## 10. 实现状态与建议新增文件

### 10.1 当前真实实现状态

| 项目 | 状态 |
|---|---|
| seed-42 训练、评估、选择、deployment-v2、reproducibility-v2 | `complete` |
| 本文认证设计 | `implemented_for_fixed_parent_engineering_scope` |
| F0 protocol / parent lock | `complete` |
| certification source lock | `complete_write_once_verified` |
| certification tag / release attestation | `pending` |
| F1 逐图 collector、六模式 knockout、五预算、区域统计、paired bootstrap | `implementation_complete_execution_pending` |
| F2 replication shared core、B/D runners、launcher、resume resolver | `complete_execution_active` |
| 本轮新 seed-42 B/D 独立 replay | `running_on_physical_gpu_2_3_after_real_exact_resume_probe` |
| seed 3407 B/D 累计整模训练 | `B_complete_800_D_exact_resume_available_741_supplement_only` |
| seed 426780603 B/D 累计整模训练 | `cancelled_for_current_fixed_seed42_scope` |
| 本轮 seed-42 四 checkpoint 独立 sweep 适配器 | `implementation_pending_execution_pending` |
| F3 locked official-test runner 与 access receipt | `pending` |
| Progressive/SPD/early-TPD comparator checkpoints | `missing_or_unrecovered` |
| F4 multidataset exact runner、失败案例、profile | `pending` |
| 论文级确认性全流水多 seed 训练 | `not_authorized_by_engineering_v1_runner` |

因此当前已不是纯设计蓝图。旧多 seed v1 工程入口已经存在，但本轮必须先完成
新的固定 seed-42 独立 replay successor，随后才进入 GPU 执行；论文级
fresh-parent、多数据集和正式测试闭环仍未完成：

```text
FIXED_SEED42_CERTIFICATION_REPLAY_EXECUTION_ACTIVE
```

### 10.2 协议与锁

```text
experiments/FINAL_MODEL_CERTIFICATION_PROTOCOL_V1.md
experiments/FINAL_MODEL_OFFICIAL_TEST_PROTOCOL_V1.md
experiments/final_model_certification_parent_lock_v1.json
experiments/final_model_certification_source_lock_v1.json
experiments/final_model_seed42_certification_replay_contract_v2.json
experiments/final_model_seed42_certification_replay_manifests_v2/
experiments/final_model_seed42_certification_replay_source_lock_v4.json
experiments/final_model_replication_seed_contract.json
experiments/final_model_certification_release_attestation_v1.json
experiments/final_model_official_test_contract_v1.json
experiments/freeze_final_model_certification_source_lock.py
experiments/freeze_final_model_official_test_contract.py
```

### 10.3 零训练审计与共享逐图缓存

```text
analysis/collect_final_model_validation_statistics.py
analysis/audit_final_qfg_functional_use.py
analysis/run_final_qfg_six_mode_audit.py
analysis/audit_final_model_epoch_trajectory.py
```

现有六模式 runner 已把 paired bootstrap、component difference、QFG factor/gate 与 target、hard-negative、ordinary-background 区域统计集成到同一次逐图缓存闭环。collector 的缓存键强制包含 checkpoint SHA、dataset hash、133 张 validation ID hash、normalization hash、evaluator SHA、certification source-lock SHA、match radius、tiny area 和 inference/knockout mode。

### 10.4 多 seed 累计整模训练

```text
experiments/final_model_replication_seed_contract.py
experiments/final_model_child_initialization_manifest.py
experiments/final_model_replication_exact_core.py
experiments/final_model_seed42_certification_replay_contract.py
experiments/final_model_seed42_certification_replay_exact_core.py
experiments/train_final_model_seed42_certification_replay_b_exact.py
experiments/train_final_model_seed42_certification_replay_d_exact.py
experiments/run_final_model_seed42_certification_replay_pair_2x5090.sh
experiments/train_final_model_replication_original_exact.py
experiments/train_final_model_replication_v4_parent_exact.py
experiments/train_final_model_replication_b_exact.py
experiments/train_final_model_replication_d_exact.py
experiments/train_final_model_replication_original_budget_matched_exact.py
experiments/prepare_final_model_engineering_replication.py
experiments/run_final_model_replication_seed_pair_2x5090.sh
experiments/launch_final_model_replication_2x5090.sh
experiments/launch_final_model_full_pipeline_replication_2x5090.sh
experiments/watch_final_model_engineering_replication.py
experiments/summarize_final_model_engineering_replication.py
experiments/evaluate_final_model_engineering_replication_pd_fa.py
experiments/decide_final_model_stability.py
```

其中 v1 已实现并封存的是固定 parent 的 B/D 工程矩阵；Original、fresh-V4、Original-budget-matched 和 seed-matched parent 属于后续论文级 runner，当前工程入口会主动拒绝 confirmatory seeds。

### 10.5 官方测试、跨数据集与效率

```text
experiments/evaluate_final_model_official_test_locked.py
experiments/summarize_final_model_official_test.py
experiments/results/final_model_locked_tests_v1/{dataset}/test_access_receipt_v1.json
experiments/results/final_model_locked_tests_v1/atomic_test_access_receipt_v1.json
experiments/final_model_dataset_registry.py
experiments/final_model_infrared_dataset.py
experiments/train_final_model_multidataset_exact.py
experiments/evaluate_final_model_multidataset.py
experiments/profile_final_model_inference.py
```

### 10.6 测试

```text
tests/test_final_model_seed_contract.py
tests/test_final_model_replication_exact.py
tests/test_final_model_seed_matched_parent.py
tests/test_final_model_budget_matched_control.py
tests/test_final_qfg_knockout.py
tests/test_final_model_statistics_cache.py
tests/test_final_model_official_test_lock.py
tests/test_final_model_bootstrap.py
tests/test_final_model_inference_profile.py
```

---

## 11. 关键代码修改原则

### 11.1 只新增，不修改封存文件

新 runner 通过独立 adapter 复用现有：

```text
experiments/tpd_exact_runner.py
experiments/tpd_extension_warm_start.py
experiments/tpd_training_loss.py
experiments/evaluate_pd_fa_sweep.py
既有 evaluator 与 source-lock utilities
```

但不能改现有 formal800 入口的常量、判断或已有 source lock。统一使用 `final_model_certification_source_lock_v1.json` 绑定所有会消费确认性 seeds 的实现：Original、fresh-V4、B、D、Original-budget-matched、multidataset runners、共享 core、collector、locked test evaluator、profile、summary 和全部复用源码。不再另造同义的 replication source lock。

### 11.2 run identity 必须显式区分种子

示例：

```json
{
  "architecture_family": "tpd8_ner4_qfg2_croa",
  "frozen_parent_checkpoint_sha256": "0ae6c0e0...",
  "builder_compatibility_seed": 42,
  "extension_initialization_seed": 42,
  "trajectory_seed": 426780603,
  "split_seed": 20260722,
  "default_threshold": 0.5,
  "official_test_accessed": false
}
```

### 11.3 checkpoint 不允许跨 seed resume

必须拒绝：

```text
seed 42 journal → seed 3407 run
B checkpoint → D exact resume
旧 source lock → 新 replication runner
```

只允许同 arm、同 seed、同 source lock、同 parent SHA 的 exact resume。

跨 seed 任务始终从同一父 checkpoint 新建 child trajectory；它不是从另一个 seed 的 child checkpoint 继续。

### 11.4 官方测试脚本必须无 sweep API

正式 test 脚本不应接收：

```text
--threshold-min
--threshold-max
--num-thresholds
--fa-budget
--select-best
```

只允许：

```text
--locked-contract
--output
```

模型 checkpoint、test index、threshold 0.5、normalization 和 evaluator hashes 全部从 contract 读取。正式入口每个合同只能生成一个 write-once result/access receipt，不能通过 CLI 替换模型或数据。

### 11.5 逐图缓存避免重复计算

F1 在 seed-42 验证集上需要的最小推理 pass 为：

```text
Original
SPD
V4
B
D full
D all-level QFG-off
D level-1 off
D level-2 off
D level-3 off
D level-4 off
```

当前可立即执行的是除 SPD 外的 9 个 core passes；恢复或重训 SPD checkpoint 后补 1 个 SPD pass，论文级缓存合计 10 个。QFG 利用率 hook 并入 D full pass；固定阈值、五预算、component atlas 和 bootstrap 全部从缓存重算，不再次加载模型。现有聚合 sweep JSON 只做交叉核验，不能替代逐图 paired bootstrap 所需的充分统计。

---

## 12. 推荐 GPU 执行顺序

GPU 调度固定只使用物理 GPU 2/3。显存占用属于启动时状态，必须在每次 launcher 启动前记录 GPU UUID、可用显存和已有本项目进程；文档不能把某次空闲快照写成永久保证。

精确依赖顺序为：

### 第 0 步：核验与封存

```text
核验现有 seed-42 v2 closure
创建 parent lock
明确 hash_seed=426780603
冻结统一 Gate、checkpoint 角色和所有 δ
完成 comparator checkpoint inventory
冻结 recipe-level 与 budget-matched 比较合同
```

### 第 1 步：实现与 dry-run

```text
实现所有确认性 seed 消费者：
Original、fresh-V4、B、D、Original-budget-matched、
multidataset runners、replication shared core、collector、locked evaluators
ordinary 与 python -O 测试
CPU smoke
短程训练 smoke
中断续训逐 tensor 等价测试
```

全部通过后生成 write-once `final_model_certification_source_lock_v1.json`，再由其 SHA 派生并封存 seed schedule。该 lock 生成后不再修改任何 seed-consuming runner；需要修改时必须升版本、重新 dry-run，并产生新的 seed schedule，旧 schedule 不得沿用。

### 第 2 步：F1 零训练审计

```text
先完成 9 个当前可用的 validation core passes
恢复/重训 SPD checkpoint 后补第 10 个 pass
从缓存完成 fixed-0.5、五预算、bootstrap、component atlas
只读现有 metrics/journal 完成 epoch trajectory audit
```

### 第 3 步：GPU 2/3 本轮固定 seed-42 配对 replay

```text
GPU 2: new certification seed 42, B / V4-stack+TSS
GPU 3: new certification seed 42, D / Full-stack
```

两臂使用本轮 successor 合同、独立 run identity 和新输出目录，从共同 V4 parent
分别 warm-start，并各自从 child epoch 1 训练到 800。旧 seed-42 目录不得作为
本轮输出目录。seed 3407 只作补充；seed 426780603 不启动。

### 第 4 步：评估与工程裁决

```text
本轮新 B、D 各自评估 best 与 best_miou
合计 4 份 checkpoint-local sweeps
旧 seed-42 B/D 只作历史交叉核验
汇总固定 seed-42 Gate M-functional 与工程复验结果
```

这一步完成的是固定 seed-42 工程复验，不自动产生论文稳定性结论。如果以后决定
声明稳定性，再单独按 Gate S-P 的全新 paired seed 协议执行，不用 seed 42/3407
填充数量。

### 第 4.5 步：论文稳定性与预算控制

```text
按 5 个全新 seeds 训练 seed-matched Original/V4-parent/B/D
按至少 3 个全新 seeds 训练 Original-budget-matched
完成 hierarchical bootstrap、Gate M-train 与 Gate S-P
```

若跳过此步，可以继续保留模型部署结论，但不能设置论文稳定性或 paper-core 状态。

### 第 5 步：一次性测试

NUDT 官方 test、NUAA-SIRST、IRSTD-1k 的合同必须先同时冻结。完成各数据集 train/validation 后，再用 locked evaluator 一次性执行全部预注册 test；不得先看 NUDT test 再修改外部数据集协议。

---

## 13. 最终论文裁决规则

### 13.1 可设置 `paper_core_established=true` 的条件

必须同时满足：

```text
Gate M：M-functional + M-train + M-test 全部成立
Gate S-P：至少 5 个全新 paired seeds 的内部稳定性成立
Gate T：NUDT 官方 test 的 D→Original 主比较成立，且 D 不被 SPD 实质支配
Gate X：NUDT + NUAA-SIRST + IRSTD-1k 的跨数据集支持成立
Gate E：工程、复现和效率证据完整
```

### 13.2 可设置 `stability_claim_supported=true` 的条件

至少需要同时满足：

```text
Gate S-P：5 个全新、结果未知、预注册 paired seeds
每个 seed 使用自己 fresh 训练的 seed-matched V4 parent
seed 42 与 3407 不进入通过率分母
相同模型、数据、预算、checkpoint 选择和评估协议
至少 4/5 seed 的 D→Original point estimate 满足同一 FA_ROUTE 方向
任一 seed 上 DOMINATES(Original,D)=false 且 DOMINATES(B,D)=false
Gate T 与 Gate X 同时支持
Gate E 的训练预算匹配控制完成
```

只有 3 个全新 seeds 时，只能设置 `multiseed_replication_supported=true`，不能设置广义稳定性为 true。

### 13.3 允许的论文表述

通过全部门槛后，可以写核心性能表述：

> 本文提出一种由目标保真相位下采样、五节点尾部感知证据中继和仅 Query 频率调制组成的红外小目标检测框架，并在训练阶段加入目标存活监督且不将该辅助 head 带入部署图。完整模型在预注册的多随机种子、官方测试和多个数据集上获得更优的 Pd–Fa–mIoU 综合权衡。

只有 A→B 的性能与目标存活诊断共同支持时，才进一步写“TSS 改善浅层目标存活”；只有 M-functional、M-train 与 M-test 同时支持时，才写“QFG 提供独立增量贡献”。

仍不应写：

```text
在所有 Fa budget 上统一最优
全面支配所有基线
绝对解决小目标丢失
无损下采样
跨场景必然稳定
```

除非后续数据确实支持这些更强主张。

---

## 14. 当前事实状态与流程状态

```text
# 已有 manifest/产物支持的事实
final_model_established=true
final_model_engineering_selected=true
selected_recipe=D_TSS_QFG
inference_architecture=TPD8_NER4_QFG2_CROA
training_only_tss=true
inference_tss_removed=true
default_checkpoint=D_best_miou_epoch3
default_threshold=0.5
historical_seed42_operational_closure=complete
new_seed42_certification_replay=running
new_seed42_replay_old_stage_results_reused=false
new_seed42_posttraining_tooling=complete_execution_waiting_for_epoch800
new_seed42_completion_watcher=active_waiting_for_formal800
new_seed42_completion_source_lock=complete
new_seed42_completion_source_lock_sha256=8ce245e3f609bd929ae9405daaae11a4d6f5aa470965c2896139238cfcf43ee7
official_test_accessed_by_final_closure=false
paper_core_established=false
stability_claim_supported=false

# 本文修订后的建议流程状态
decision=RUN_NEW_FIXED_SEED42_CERTIFICATION_REPLAY
architecture_frozen=true
mainline_changed=false
new_module_design_authorized=false
fallback_EF_authorized=false
current_execution_seed=42
seed3407_role=supplement_only_not_in_current_gate
seed426780603_role=cancelled_not_scheduled
multiseed_replication_authorized_for_current_execution=false
multiseed_replication_execution_ready=not_in_current_queue
official_test_authorized=false
certification_design_reviewed=true
certification_design_complete_for_engineering_scope=true
certification_design_complete_for_paper_scope=false
certification_protocol=complete
certification_parent_lock=complete
certification_source_lock=complete
certification_source_lock_sha256=d6334b4f863e06cd0fa744723025b6bdf1fe76a7d0664cfe84d472a19e09d13f
certification_tag=pending
f1_tooling=complete_execution_pending
f2_replication_contract_and_runner=complete
fixed_seed42_b_d_formal800_execution=active
fixed_seed42_four_checkpoint_sweeps=pending_epoch800
fixed_seed42_paired_gate=pending_sweeps
final_completion_attestation=pending_f1_and_seed42_gate
f3_locked_official_test_runner=pending
f4_multidataset_profile=pending
comparator_checkpoint_inventory_complete=false
engineering_certification_execution_ready=true
paper_certification_execution_ready=false
qfg_functional_contribution_supported=pending
```

---

## 15. 最终建议

当前最优行动不是继续增加模型模块，而是先把本文从设计转成经过测试的认证代码。完整模型已经完成 seed-42 工程选择；后续工作不会把 TPD、NER、TSS 或 QFG 拆成孤立单模块训练。

最先执行的三个任务应固定为：

```text
1. 保持 GPU2/GPU3 上的新 seed-42 B/D 累计整模训练到各自 800 epochs
2. 分别评估 B/D 各自的 best_mIoU 与 best，共完成 4 份 checkpoint-local sweeps 和固定 Gate
3. 对冻结部署 D 执行 F1 六模式 QFG 审计、CPU deep verification，并生成一次性最终认证声明
```

若当前目标只是完成可部署的新模型，seed-42 权重可以继续保留，额外多 seed 不是模型代码成功的必要条件；此时必须维持 `stability_claim_supported=false`。若目标是论文稳定性主张，则按 Gate S-P 完成全新 seeds，不能用 seed 42/3407 填数量。

旧 F2 的 3407/426780603 执行队列不属于本轮固定 seed-42 Gate。其合同、
runner 和 tests 作为未来多随机性扩展工具保留，但不得把未完成的旧执行写成已完成，
也不得由旧 launcher 在本轮自动接续 426780603。

只有 Gate M、S、T、X、E 对应的实现和证据全部完成后，才能更新论文状态。若结果不支持，应保留当前 seed-42 部署产物并缩小结论范围；不得通过改变 threshold、删除不利 seed 或重选 checkpoint 来改变预注册裁决。

---

## 本地权威代码与产物位置

- `model/tpd_frequency_gate_v2_croa.py`
- `model/tpd_query_frequency_bridge.py`
- `model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py`
- `experiments/train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact.py`
- `experiments/train_tpd_ner_v4_qfg_v2_croa_exact.py`
- `experiments/tpd_extension_warm_start.py`
- `experiments/tpd_exact_runner.py`
- `experiments/evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa.py`
- `experiments/publish_tpd_ner_v4_qfg_v2_croa_default_operating_point_v2.py`
- `experiments/tpd_ner_v4_qfg_v2_croa_operational_closure_source_lock_v2.json`
- `experiments/results/tpd_ner_v4_qfg_v2_croa_exact_v2_optimized/NUDT-SIRST/reproducibility_manifest_v2/manifest.json`
- `experiments/final_model_seed42_certification_replay_contract_v2.json`
- `experiments/final_model_seed42_certification_replay_source_lock_v4.json`
- `experiments/run_final_model_seed42_certification_replay_pair_2x5090.sh`
- `experiments/final_model_seed42_certification_replay_posttraining.py`
- `experiments/run_final_model_seed42_certification_replay_posttraining_2x5090.sh`
- `experiments/final_model_seed42_certification_completion.py`
- `experiments/run_final_model_seed42_certification_completion.sh`
- `experiments/freeze_final_model_seed42_certification_completion_source_lock.py`
- `experiments/final_model_seed42_certification_completion_source_lock_v1.json`
