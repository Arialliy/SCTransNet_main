# SCTransNet 正 TSS 全局配方未建立后的 TSS-off Seed42 配对诊断与执行方案

> 项目：单帧红外小目标检测  
> 基线：SCTransNet  
> 冻结推理结构：`SCTransNet + TPD8-MPRS-DCH + 五节点 NER4 Tail-Aware + QFG2-CROA`  
> 当前训练期辅助：Target Survival Supervision（TSS）  
> 当前正式裁决：`NO_POSITIVE_GLOBAL_TSS_RECIPE_ESTABLISHED`  
> 主随机种子：`42`  
> 训练预算：`1000 epochs`  
> 评估周期：每 `10 epochs`  
> 固定阈值：`0.5`  
> 正式数据集：`NUAA-SIRST / NUDT-SIRST / IRSTD-1K`
> 选择集：各数据集 `img_idx/test`  
> 结论边界：`fixed_seed42_img_idx_test_selected_paired_diagnostic`  
> 稳定性结论：不由本阶段建立

---

## 0. 执行摘要

三个正 TSS 请求权重上限均未通过跨数据集统一严重退化门：

| TSS 请求权重上限 | 严重退化违规数 | 裁决 |
|---:|---:|---|
| `0.0025` | 8 | 拒绝 |
| `0.005` | 5 | 拒绝 |
| `0.01` | 8 | 拒绝 |

正式状态为：

```text
decision=NO_POSITIVE_GLOBAL_TSS_RECIPE_ESTABLISHED
global_tss_recipe_established=false
global_tss_lambda=null
```

这一结果不表示训练任务失败，也不表示 TPD、NER、QFG 组成的推理结构已经失败。它只证明：

> 在固定 seed 42、固定阈值 0.5、相同 `img_idx` 协议、相同 checkpoint 选择规则和相同 10% 动态标量损失上限下，`0.0025 / 0.005 / 0.01` 三个正 TSS 请求权重均不能成为统一覆盖 NUAA-SIRST、NUDT-SIRST 和 IRSTD-1K 的默认训练配方。

下一步最有判别力、变量最少的实验不是继续细搜正权重，也不是修改模型结构，而是：

> **保持完整 TPD+NER+QFG 推理结构不变，运行三个数据集的 TSS-off（`survival_weight=0`）Seed42 配对诊断。**

这 3 个 run 可以回答：

> 在同一架构、同一 seed 和当前 test-selected 选模协议下，关闭 TSS 后预先规定的性能向量如何变化。

它不单独建立跨随机性因果或稳定性主张。

---

# 1. 当前裁决是否正确

当前裁决是正确且必要的。

## 1.1 已经建立的结论

```text
positive_global_tss_recipe_rejected=true
global_tss_lambda=null
fixed_lambda_005_not_eligible_under_frozen_seed42_protocol=true
dataset_dependent_tss_tradeoff_supported=true
```

其中 `0.005` 虽然违规数最低，但它只是描述性锚点，正式 selector 没有选中任何正候选：

```text
违规最少的描述性锚点
≠
通过门槛
≠
可以降低门槛后强行选用
```

`0.005` 仍因以下问题未通过：

- NUAA-SIRST 的 Pd 退化；
- NUAA-SIRST 的 tiny-Pd 退化；
- IRSTD-1K `best_miou` 的 Fa 增加。

因此，不能将其设置为全局默认 TSS 配方。

正式字段应保持：

```text
selected_positive_candidate=null
descriptive_fewest_violation_anchor=0.005
```

## 1.2 尚未建立的结论

当前结果还不能证明：

```text
TSS 本身一定有害
TPD8 + NER4 + QFG2 推理结构失败
关闭 TSS 一定优于正 TSS
三个正 λ 对训练产生了充分可分的实际干预
```

尤其需要注意，三个数值是 TSS 的**请求权重上限**，实际有效权重仍受动态 ratio cap 控制。

---

# 2. 正 TSS 搜索为什么应当结束

当前不应继续搜索：

```text
0.003
0.004
0.006
0.0075
0.008
```

原因如下。

## 2.1 预注册候选已经全部未通过

本轮已完成：

```text
0.0025
0.005
0.01
```

三档半倍、原值、两倍搜索，足以判断当前正权重区间没有形成统一跨数据集配方。

## 2.2 继续细搜会扩大 test-selected 偏差

当前协议中：

```text
img_idx/test
→ 每 10 epochs 选择 checkpoint
→ 参与全局 TSS 权重裁决
```

继续在同一批 test 上增加候选，会进一步增加超参数选择自由度，使结果更乐观，削弱论文可信度。

## 2.3 更细的正权重不能回答“TSS 是否应启用”

三个正候选全部开启 TSS。即使继续细搜，也无法区分：

```text
TSS 权重没选对
```

与：

```text
TSS 辅助监督本身不适合作为全局配方
```

因此应正式关闭正权重搜索：

```text
positive_tss_search_closed=true
additional_positive_lambda_search_authorized=false
```

---

# 3. 代码级解释：为什么必须做 TSS-off

动态 TSS loss 可写为：

\[
L_{\mathrm{seg}}
=
\sum_j BCE(P_j,Y)
\]

\[
L_{\mathrm{tss}}
=
\sum_i BCEWithLogits(Z_i,Y_{16})
\]

\[
\lambda_{\mathrm{cap}}
=
\rho
\frac{
\operatorname{stopgrad}(L_{\mathrm{seg}})
}{
\max(
\operatorname{stopgrad}(L_{\mathrm{tss}}),
\epsilon
)
}
\]

\[
\lambda_{\mathrm{eff}}
=
\min(
\lambda_{\mathrm{requested}},
\lambda_{\mathrm{cap}}
)
\]

\[
L_{\mathrm{total}}
=
L_{\mathrm{seg}}
+
\lambda_{\mathrm{eff}}L_{\mathrm{tss}}
\]

其中：

```text
rho = 0.10
```

当：

```text
survival_weight = 0
```

现有 loss 路径应当直接退化为：

\[
L_{\mathrm{total}}=L_{\mathrm{seg}}
\]

即：

- 不构造 survival target；
- loss 不读取、不校验 survival logits；
- TSS 不进入总损失；
- TSS 不向共享主干传播梯度；
- segmentation 六项 BCE 保持原始加法顺序。

必须区分 loss 路径与训练 forward：

```text
训练 forward：仍计算并返回两路 survival logits
λ=0 loss：不消费这些 logits
反向传播：TSS 计算图不连接 Ltotal，因此 TSS 参数 grad=None
```

为了保持配对实验只改变辅助 loss，本阶段不绕过、不删除 TSS forward。

因此，TSS-off 是区分以下两种解释的最干净实验：

```text
A. mixed trade-off 主要由训练期 TSS 引起
B. mixed trade-off 即使没有 TSS 仍存在，问题位于推理结构或主训练路径
```

---

# 4. 下一步正式实验：三个 TSS-off run

## 4.1 实验矩阵

只新增 3 个正式任务：

| 数据集 | 模型 | TSS | Seed | Epochs | 阈值 |
|---|---|---:|---:|---:|---:|
| NUAA-SIRST | Final | Off，`λ=0` | 42 | 1000 | 0.5 |
| NUDT-SIRST | Final | Off，`λ=0` | 42 | 1000 | 0.5 |
| IRSTD-1K | Final | Off，`λ=0` | 42 | 1000 | 0.5 |

保持完全不变：

```text
TPD8-MPRS-DCH
五节点 NER4 Tail-Aware
QFG2-CROA
TSS head 结构
训练模型 class
img_idx/train
img_idx/test
normalization
augmentation
optimizer
scheduler
batch size
patch size
eval_every=10
best_miou / best_pd 排序
threshold=0.5
evaluator
Misc_111 修正合同
```

唯一有效变化：

```text
survival_weight = 0.0
```

## 4.2 Original 是否需要重训

当前 3 个 Original 已经完成下列合同核对，可直接复用，不重训：

```text
img_idx 文件 SHA
数据文件 SHA
Misc_111 修正 manifest SHA
normalization
seed
初始化规则
augmentation 随机流
optimizer/scheduler
1000 epoch 预算
每 10 epoch 评估
checkpoint tie-break
evaluator SHA
```

已有 selector 记录 `per_run_protocol_matched=true`，3 个 Original 均为 seed42/1000 epochs/每 10 epochs 评估，且数据和 evaluator 合同匹配。TSS-off protocol 建立后应再机械复核一次。如新 evaluator 的 source SHA 不同，优先重新评估已保存的 Original `best_miou / best_pd` checkpoint，不因此重训 Original。

## 4.3 搜索预算披露

TSS-off 完成后，Final 家族已查看的配方是 3 个正 λ 加 1 个 off：

```text
per_run_protocol_matched=true
total_recipe_search_budget_equal=false
final_training_runs=12
original_training_runs=3
final_to_original_recipe_search_ratio=4.0
tss_off_added_after_positive_test_results=true
test_selected=true
```

这一预算差异必须进入最终 summary 和论文限制。

---

# 5. TSS-off 的推荐实现方式

## 5.1 保留同一训练模型类

建议继续构建：

```text
SCTransNet
+ TPD8
+ NER4
+ QFG2
+ 已注册的 TSS heads
```

但将 loss 配置设为：

```yaml
survival_weight: 0.0
survival_ratio_cap: 0.10
```

ratio cap 在零权重下不产生实际作用，但保留字段可以减少配置差异。

不建议：

- 切换到另一个无 TSS 训练模型类；
- 删除 TSS heads 后重新初始化模型；
- 使用不同 state schema；
- 修改 forward；
- 修改部署导出代码。

这样能够确保：

```text
同一模型 class
同一共享参数初始化
同一 checkpoint schema
同一冻结训练核心
唯一差异是辅助 loss 是否参与优化
```

但运行身份必须独立：

```text
method=final_tss_off
recipe_id=final_tss_off
requested_tss_weight=0.0
tss_enabled=false
run_directory=.../final_tss_off/seed_42
```

不得把 Final TSS-off 写成 Original，也不得复用任一正 λ 的 resume 目录。

## 5.2 TSS head 在 TSS-off 下的预期行为

训练 checkpoint 仍可包含：

```text
target_survival.heads.emb1.classifier.weight
target_survival.heads.emb1.classifier.bias
target_survival.heads.emb2.classifier.weight
target_survival.heads.emb2.classifier.bias
```

但在 `survival_weight=0` 时：

```text
TSS 参数不应获得梯度
TSS 参数不应发生更新
TSS 参数应保持初始化值
TSS 参数仍在 optimizer param group 中，但不应创建 Adam state
```

如果 TSS heads 初始为全零，则训练结束时也应保持全零。

部署导出时继续删除所有 `target_survival.*` state。当前 evaluator 是在内存中构建无 TSS 推理图；如需独立 deployment checkpoint，必须在本阶段显式落盘并记录 SHA。

---

# 6. TSS-off 启动前必须完成的复核

## 6.1 正 λ 的有效权重可辨识性

现有 9 个正 λ run 的 1000-epoch/minibatch 日志已完整，启动前必须将以下结果写入独立、带 SHA 的封存 JSON，不得在 TSS-off 结果出现后重算规则。

```text
effective_lambda_mean
effective_lambda_p10
effective_lambda_p50
effective_lambda_p90
effective_lambda_max
cap_active_rate
raw_tss_ratio
effective_tss_ratio
```

全程 sample-weighted 聚合为：

| 数据集 | λ | effective mean | p10 | p50 | p90 | cap batch fraction | raw ratio | effective ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IRSTD-1K | .0025 | .001749 | .000739 | .001827 | .002500 | 70.88% | .1788 | .0923 |
| IRSTD-1K | .005 | .002074 | .000711 | .001848 | .003847 | 95.90% | .3625 | .0989 |
| IRSTD-1K | .01 | .002182 | .000720 | .001867 | .003882 | 99.19% | .7149 | .0997 |
| NUAA-SIRST | .0025 | .002227 | .001397 | .002500 | .002500 | 33.86% | .0999 | .0750 |
| NUAA-SIRST | .005 | .003346 | .001380 | .003383 | .005000 | 73.64% | .1946 | .0927 |
| NUAA-SIRST | .01 | .004031 | .001380 | .003454 | .007760 | 94.54% | .3857 | .0987 |
| NUDT-SIRST | .0025 | .002391 | .002050 | .002500 | .002500 | 15.14% | .0524 | .0441 |
| NUDT-SIRST | .005 | .004306 | .002163 | .005000 | .005000 | 32.46% | .0948 | .0579 |
| NUDT-SIRST | .01 | .007040 | .002070 | .008991 | .010000 | 52.91% | .1984 | .0761 |

`0.005` 与 `0.01` 在各数据集反事实 effective weight 相同的 batch 比例为：

```text
IRSTD-1K=96.08%
NUAA-SIRST=74.81%
NUDT-SIRST=34.18%
```

因此固定记录：

```text
lambda_005_vs_010_not_fully_identifiable=true
```

这表示两者在 IRSTD/NUAA 上被 ratio cap 明显压缩，不表示三个正 λ 完全相同。它不改变“无正全局配方”的裁决，但论文中不能写成：

> 增强 TSS 已被充分否证。

更准确的写法是：

> 在当前 ratio cap 下，提高请求权重上限没有形成可接受的全局配方。

## 6.2 严重退化类型分解

不能只保留总违规数：

| λ | 总违规数 |
|---:|---:|
| 0.0025 | 8 |
| 0.005 | 5 |
| 0.01 | 8 |

还应输出：

| λ | Pd 违规 | tiny 违规 | Fa 违规 | mIoU 违规 | nIoU 违规 | strict domination |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0025 | 4 | 1 | 1 | 1 | 1 | 0 |
| 0.005 | 2 | 2 | 1 | 0 | 0 | 0 |
| 0.01 | 4 | 1 | 2 | 0 | 1 | 0 |

用于判断：

- Pd/tiny 退化是否来自目标存活冲突；
- Fa 退化是否来自背景激活或组件碎裂；
- 区域指标退化是否来自 segmentation 表达；
- 是否存在被 Original 全面覆盖的候选。

## 6.3 分数据集、分 checkpoint 违规矩阵

至少输出：

```text
NUAA best_miou
NUAA best_pd
NUDT best_miou
NUDT best_pd
IRSTD best_miou
IRSTD best_pd
```

并检查：

```text
违规是否随 λ 单调变化
```

如果没有单调关系，则问题不是简单的“TSS 太强”或“TSS 太弱”，而可能是：

- batch 级有效权重差异；
- 目标密度差异；
- endpoint 监督与主任务梯度冲突；
- checkpoint 轨迹波动。

---

# 7. TSS-off 的测试计划

不重复已有基础测试。当前已有：

```text
tests/test_tpd_training_loss.py
tests/test_tpd_ner_v8_mprs_dch_v4_survival_loss_integration.py
tests/test_four_dataset_models_seed42_v1.py
```

它们已覆盖 λ=0 loss 等价、TSS 无梯度、第一次 Adam 更新等价以及推理去除 TSS 后输出等价。本阶段只新增针对当前 Final QFG2 图和 TSS-off 运行合同的测试：

```text
tests/test_train_three_dataset_tss_off_seed42_v1.py
tests/test_tss_off_preflight_and_launcher_v1.py
tests/test_compare_tss_off_positive_original_v1.py
```

## 7.1 Loss 等价

必须验证：

\[
L_{\mathrm{total}}
=
L_{\mathrm{seg}}
\]

逐元素或严格浮点等价。

测试内容：

```text
不构造 survival target
训练 forward 会计算 logits，但 loss 不读取 survival logits
不调用 survival criterion
六项 segmentation loss 顺序不变
```

## 7.2 梯度隔离

验证：

```text
共享 segmentation 参数有梯度
target_survival.* 参数 grad is None
```

## 7.3 TSS 参数不更新

训练若干 step 后：

```text
target_survival.* state == initial target_survival.* state
optimizer param group 包含 TSS 参数
optimizer state 不包含 TSS 参数状态
```

## 7.4 Exact resume

连续训练与中断续训应在以下状态一致：

```text
model tensors
optimizer tensors
scheduler
RNG
DataLoader generator
best_miou state
best_pd state
```

## 7.5 部署导出

验证：

```text
训练模型 eval segmentation output
==
移除 TSS 后的推理模型 output
```

且推理权重中：

```text
不存在 target_survival.*
```

selected 训练 checkpoint 只保存 model state；滚动 resume 才保存 optimizer/RNG。两者不得混写。

---

# 8. 建议新增的实验文件

不要覆盖正 TSS 12-run 的历史代码，新增独立阶段：

```text
experiments/train_three_dataset_tss_off_seed42_v1.py
experiments/launch_three_dataset_tss_off_seed42_v1.py
experiments/launch_three_dataset_tss_off_seed42_v1.sh
experiments/evaluate_three_dataset_tss_off_seed42_v1.py
experiments/compare_tss_off_positive_original_v1.py
experiments/finalize_tss_off_diagnostic_v1.py
experiments/preflight_three_dataset_tss_off_seed42_v1.py
experiments/tss_off_diagnostic_common_v1.py
```

辅助分析：

```text
experiments/analyze_positive_tss_effective_weights_v1.py
experiments/summarize_tss_violation_types_v1.py
```

正式输出根目录固定为：

```text
/home/ly/SCTransNet_main/results/three_dataset_tss_off_seed42_v1
```

运行目录固定为：

```text
runs/{dataset}/final_tss_off/seed_42
```

新文件应调用现有冻结训练/评估核心，不复制指标实现或数据集逻辑。旧 12-run 源码与历史产物保持不变。

## 8.1 配置合同

```yaml
datasets:
  - NUAA-SIRST
  - NUDT-SIRST
  - IRSTD-1K

seed: 42
epochs: 1000
eval_every: 10
threshold: 0.5

checkpoint_roles:
  - best_miou
  - best_pd

architecture:
  tpd: V8-MPRS-DCH
  ner: V4-Tail-Aware
  qfg: V2-CROA

tss:
  mode: off
  survival_weight: 0.0
  survival_ratio_cap: 0.10
  heads_registered_during_training: true
  heads_removed_for_inference: true
```

---

# 9. TSS-off 的正式比较对象

聚合报告应包含：

```text
Original
TSS-off
TSS λ=0.0025
TSS λ=0.005
TSS λ=0.01
```

但裁决需要回答两个独立问题。

## 9.1 TSS-off 是否优于 Original

为主裁决，用于判断：

```text
TPD+NER+QFG 推理结构在不使用 TSS 时
是否在当前 seed42/test-selected 协议下通过统一跨数据集严重退化门
```

它必须完全复用旧 selector 的：

```text
每数据集、每 checkpoint role 严重退化规则
Original 双 role 严格支配规则
mIoU/nIoU q=floor(x/1e-4+0.5) 量化
Pd/tiny/Fa 的整数计数语义
```

主门定义为：

```text
off_gate_eligible =
    severe_degradation_violations == 0
    and original_dual_role_dominated_datasets == []
```

“违规数降到可接受范围”不是合法条件；必须为 0。

## 9.2 TSS-off 是否优于正 TSS

为次级诊断，用于记录：

```text
TSS-off 与每一个正 λ 在预定 30 维指标向量上的关系
```

对 `0.0025 / 0.005 / 0.01` 分别输出：

```text
dominates
dominated
equal
incomparable
```

关系在三数据集×两 role×五指标的同一量化向量上计算。不得只与事后锚点 `0.005` 比较。

旧 selector 的 rank population 固定为三个正 λ，不得把 λ=0 塞入旧 selector 或重写旧 rank。如生成四 Final 配方 rank，只能作为已提前冻结 population 的描述性补充，不参与 `off_gate_eligible` 主裁决。

不得为 TSS-off 单独制定更宽松门槛。

---

# 10. TSS-off 的唯一裁决合同

本阶段不使用可重叠的 A–E 主观分支，而使用两个正交输出轴。

## 10.1 Axis 1：TSS-off 相对 Original 的主门

```text
off_gate_eligible=true
当且仅当：
    severe_degradation_violations == []
    and original_dual_role_dominated_datasets == []
```

若为 true：

```text
decision=TSS_OFF_OPERATIONALLY_ADMISSIBLE_SEED42_TEST_SELECTED
tss_default_enabled=false
seed42_operational_recipe_admissible=true
final_training_recipe_established=false
causal_confirmation=false
stability_claim_supported=false
```

解释只能是：

> 在当前 seed42、img_idx/test 选模和严重退化门下，无 TSS 的完整架构配方未被 Original 门拒绝。

这不等于“全面优于 Original”，也不等于论文级最终配方已建立。

若为 false：

```text
decision=TSS_OFF_NOT_GLOBALLY_ADMISSIBLE_SEED42_TEST_SELECTED
architecture_global_advantage_not_established=true
component_level_architecture_diagnosis=true
```

下一步按预先顺序进入单组件诊断：

```text
1. NER relay 的目标/背景调制
2. QFG 各层实际利用率与 knockout
3. TPD 对 tiny-target 与组件连通性的影响
```

仍不得同时修改多个模块。

## 10.2 Axis 2：TSS-off 与三个正 λ 的次级关系

分别输出：

```text
off_vs_0p0025=dominates|dominated|equal|incomparable
off_vs_0p005=dominates|dominated|equal|incomparable
off_vs_0p01=dominates|dominated|equal|incomparable
```

其中：

```text
dominates:
    off 在所有有效量化 cell 上不差，且至少一个 cell 更好
dominated:
    正 λ 在所有有效量化 cell 上不差，且至少一个 cell 更好
equal:
    所有有效量化 cell 相同
incomparable:
    双方都存在至少一个更好 cell
```

次级关系不改变 Axis 1 的主门结果。

## 10.3 诊断表述规则

- 如 off 相对一个或多个正 λ 改善，只记录 `TSS_OFF_IMPROVED_PREDECLARED_DIAGNOSTIC_CONTRAST`，不写 `TSS_HARM_CONFIRMED`。
- 如某正 λ 支配 off，只写“该正 λ 在预定向量上优于 off”，不写“TSS 确实有效”。
- 如全部 equal，才记录 `TSS_EFFECT_NOT_IDENTIFIABLE_UNDER_QUANTIZED_ENDPOINTS=true`。
- 如为 incomparable，必须保留 mixed trade-off，不得用单一平均分强制排序。

---

# 11. 当前禁止事项

现在不要：

```text
继续搜索更多正 λ
按数据集选择不同 λ
降低严重退化门
修改固定阈值 0.5
新增 best_joint
重新引入 SIRST3 选配方
修改 TPD / NER / QFG
增加第五个推理模块
```

特别禁止：

```text
NUAA 用 0.0025
NUDT 用 0.005
IRSTD 用 0.01
```

因为这无法建立统一模型配方。

---

# 12. 论文表述建议

## 12.1 当前正 TSS 负结果

建议写：

> 在冻结完整推理结构的条件下，我们比较了三个正 TSS 请求权重上限。虽然部分候选在单个数据集或指标上取得改善，但没有一个候选能够同时通过 NUAA-SIRST、NUDT-SIRST 和 IRSTD-1K 的预注册严重退化门。因此，我们未将正 TSS 设置为统一训练配方，并进一步通过 TSS-off 实验区分辅助监督效应与推理结构效应。

## 12.2 不应写

```text
TSS 完全无效
完整模型训练失败
三个 λ 已否证所有可能的 TSS 机制
模型结构必须推倒重做
```

## 12.3 TSS-off 完成后的可能主张

若 TSS-off 成功：

> 在固定 seed42 和当前 img_idx/test 选模协议下，冻结的 TPD–NER–QFG 无 TSS 配方通过了预定严重退化门；正 TSS 辅助监督未建立统一跨数据集配方。

若 TSS-off 仍混合：

> 在固定 seed42 和当前 img_idx/test 选模协议下，关闭 TSS 后完整架构仍未通过预定门，需要进一步对冻结推理结构进行单组件诊断。

---

# 13. 启动 Gate

## Gate O1：正 TSS 阶段封存

```text
三个正 λ 结果完整
违规类型矩阵完整
effective λ 日志完整
source lock 完整
正λ封存产物的 SHA 已记录
```

## Gate O2：TSS-off 等价性

```text
total loss == segmentation loss
TSS 参数无梯度
TSS 参数不更新
TSS 参数 grad=None
Adam param group 含 TSS，但 Adam state 无 TSS 条目
部署导出不含 TSS
```

## Gate O3：协议一致

```text
三个 img_idx 协议不变
seed42 不变
1000 epochs 不变
每10 epoch评估不变
best_miou / best_pd 不变
threshold=0.5 不变
共享初始 state SHA 与正λ Final 一致
TSS head 初始 state SHA 一致
sampler 顺序和 augmentation 输入序列一致
首 batch segmentation 输出一致
```

## Gate O4：工程完整

```text
普通模式测试通过
python -O 测试通过
GPU2/3 smoke 通过
exact resume 通过
独立 final_tss_off recipe/run/resume 目录
新 comparator 和两轴裁决谓词已 source-lock
```

只有 O1–O4 全部通过，才启动三个 TSS-off formal1000 run。

## Gate O5：训练后闭环

```text
3/3 run 均为 1000 epochs
3/3 summary 与 metrics 完整
6/6 best_miou / best_pd checkpoint 存在且可严格加载
6/6 threshold=0.5 评估完整
Axis 1 / Axis 2 裁决产物完整
最终 artifact SHA 已封存
```

O5 通过后才能更新项目裁决。

---

# 14. 推荐项目状态

```text
decision=PREPARE_TSS_OFF_PAIRED_SEED42_DIAGNOSTIC

architecture_implemented=true
architecture_frozen_for_diagnostic=true
architecture_global_advantage_not_established=true
architecture_failure_supported=false

positive_global_tss_recipe_established=false
global_tss_lambda=null
positive_tss_search_closed=true

selected_positive_candidate=null
descriptive_fewest_violation_anchor=0.005

tss_off_stage_pending_gates=true
tss_off_formal_runs_required=3
tss_off_formal_runs_completed=0

causal_confirmation=false
test_selected=true
selection_is_optimistic=true

final_training_recipe_established=false
final_inference_architecture_candidate=
    SCTransNet+TPD8+NER4+QFG2

paper_core_established=false
stability_claim_supported=false
training_recipe_finalized=false
```

---

# 15. 最终执行顺序

```text
1. 封存三个正 λ 的全部结果和 source lock
2. 聚合 effective λ 与 cap-active rate
3. 输出违规类型与 dataset-role 矩阵
4. 新增 TSS-off protocol、runner、tests
5. 验证 λ=0 的 loss/gradient/state 等价性
6. 完成 GPU smoke 与 exact resume
7. GPU2 运行 NUAA TSS-off，GPU3 同时运行 NUDT TSS-off
8. GPU2 完成 NUAA 后立即自动续跑 IRSTD TSS-off，不等待 NUDT 完成
9. 三个 run 均为 seed42，1000 epochs，resume=auto
10. 选择各自 best_miou / best_pd，threshold=0.5
11. Axis 1 使用旧严重退化门比较 TSS-off 与 Original
12. Axis 2 对三个正 λ 分别输出 dominates/dominated/equal/incomparable
13. 按两轴合同决定是否保留无 TSS 操作性配方或进入单组件诊断
```

---

# 16. 一句话结论

> **三个正 TSS 权重均未能建立统一全局配方，因此正权重搜索停止；下一步保持 TPD8+NER4+QFG2 结构不变，在三个正式数据集上运行 TSS-off（λ=0）Seed42 配对诊断，并严格按预先冻结的 Original 主门与逐正λ次级关系裁决。**

---

# 17. 相关代码位置

- 仓库：`https://github.com/Arialliy/SCTransNet_main`
- 动态训练 loss：`experiments/tpd_training_loss.py`
- TSS heads：`model/tpd_survival.py`
- 最终训练/推理模型：  
  `model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py`
- 三数据集协议与评估文件：应以新增 v2/v1-off 文件实现，避免覆盖历史 four-dataset 协议。
