# TPD-Clean V8-MPRS-DCH 前置诊断勘误 V1

**状态：** 在任何 V8 fresh formal800 之前生效  
**日期：** 2026-07-27  
**适用范围：** 只修订前置诊断的测量与身份绑定  
**不改变：** 模型公式、Keep–Context–Saliency 主线、Full/Capacity 定义、训练矩阵、
800 epochs、Pd/Fa/mIoU、Fa budgets、Gate A–E 和 NER 授权规则

---

## 1. 为什么需要勘误

首轮 12-checkpoint counterfactual 已完成，但只允许保留为 v1 描述性产物。复核发现：

1. GT-core 与 hard-negative mask 独立 adaptive-max-pool 后可能落入同一个粗 cell；
2. V7/V8 分别选择“自己覆盖到的 GT”会让漏掉的难例从 V8 样本池消失；
3. `torch.roll` 加输出裁边不能证明卷回输入没有经全局网络影响内部；
4. 首轮 job/aggregate 没有完整绑定当前 V8 源码、有序 validation IDs 和全部输入；
5. aggregate 没有从 operating points 独立重算所有拓扑量。

因此：

```text
counterfactual_v1_artifacts_deleted = false
counterfactual_v1_gate_authoritative = false
counterfactual_v2_new_output_required = true
formal800_authorized_by_v1 = false
```

---

## 2. Correction 选择性：下采样后强制互斥

图像尺度定义保持不变：

- V7 probability `>0.5`；
- 与 radius-3 GT dilation 完全不相交的整个 V7 prediction component 为
  hard negative；
- 原始二值 GT 为 target core。

对每个 MPRS block 的输出尺寸：

\[
T_{\rm raw}=\operatorname{AdaptiveMaxPool}(\mathrm{GT\ core})>0
\]

\[
N_{\rm raw}=\operatorname{AdaptiveMaxPool}(\mathrm{hard\ negative})>0
\]

冻结 target-priority 去重：

\[
\boxed{T=T_{\rm raw},\qquad N=N_{\rm raw}\land\neg T_{\rm raw}}
\]

因此同一个 block cell 不能同时进入分子与分母。每个 checkpoint job 的每个 block
聚合后都必须满足：

```text
target_count > 0
hard_negative_count > 0
target_mask & hard_negative_mask = empty
```

任一计数为空直接判该 job 无效；禁止使用 `max(1, count)` 形成数值。

仍按 3 checkpoint roles × 133 ordered validation images × 7 blocks 聚合：

\[
\mathrm{lift}=
\frac{\operatorname{mean}_{T}|\Delta S_a|}
{\operatorname{mean}_{N}|\Delta S_a|+10^{-6}}
\]

门槛不变：

```text
每个 variant 的 seed42 和 seed3407 均要求 lift > 1.0
```

---

## 3. 拓扑：固定 V7 reference GT，做配对统计

每个 checkpoint role 和其 V7 mechanism registry 的每个数值阈值上：

1. 先固定 `V7-covered GT set`：至少被一个 V7 prediction component 覆盖的 GT；
2. V7 与 V8 都只在这一组完全相同、按 GT identity 配对的集合上统计；
3. 对 reference GT，若 V8 没有任何 overlap component：
   - `largest_fragment_fraction_v8 = 0`；
   - `v8_reference_coverage = 0`；
   - 该 GT 不得从 V8 样本池删除；
4. 每个 GT 的 fragment excess 定义为
   `max(0, overlap_component_count - 1)`；
5. operating-point payload 必须保存逐 GT 配对值，aggregate 必须从这些逐 GT 值
   独立重算，不接受未复算的缓存总量。

同一 variant/seed 聚合 3 roles 和全部 registry points 后，要求：

```text
V8 aggregate fragment excess <= V7
V8 pooled paired largest-fragment median >= V7
V8 covered reference GT count >= V7 covered reference GT count
```

最后一项等价于：V8 不能靠丢掉 V7 已覆盖 GT 来通过碎裂门槛。

---

## 4. 位移项重新命名为环面网格偏移压力测试

输入、offset 和计算仍保持首轮定义：

```text
used_val_ids 原顺序前 16 张
offsets = (0,0), (0,1), (1,0), (1,1)
torch.roll
输出对齐后每边 crop 16 pixels
normalized L1
V8/V7 <= 1.10
```

但其含义冻结为：

> **toroidal grid-offset stress（环面网格偏移压力）**

不得再声称 crop 能阻止卷回区域经 SCTB 或大感受野传播到内部，也不得将其写成纯
平移等变测量。它只比较相同环面压力下 V8 是否比 V7 放大网格相位敏感性。

所有 base、shift 和 topology probability 必须来自正式 `model.forward()` 的同一
数值路径。diagnostic interface 只读取已经计算的 MPRS 项；若它同时返回网络输出，
必须在冻结容差内与正式输出核对，不能混用两条不同乘法结合顺序的输出做 Gate。

---

## 5. Job、数据和源码身份

每个 v2 job 必须绑定并由 aggregate 复核：

```text
variant
seed
checkpoint role
checkpoint path + SHA256
V7 protocol/split/summary/metrics SHA256
V7 source-lock SHA256 + training-data SHA256
registry path + canonical SHA256
registry exact numeric operating points
133 used_val_ids 原顺序
ordered validation IDs SHA256
validation image/mask content fingerprint
model/tpd_clean_v8_mprs_dch.py SHA256
experiments/train_tpd_clean_v8_mprs_dch.py SHA256
analysis/analyze_tpd_clean_v8_mprs_mechanism.py SHA256
主协议 SHA256
本勘误 SHA256
device identity
```

四个 shard/job 组的 V8 源码、协议、勘误、数据和 ordered-ID 摘要必须完全相同。
aggregate 必须验证 expected 12-job 矩阵，拒绝 stale、错 role、错 seed、错 registry、
空 evidence、非有限数值或混合源码结果。

---

## 6. 产物与授权

首轮目录保持不变：

```text
analysis/results/tpd_clean_v8_mprs_counterfactual_v1/
```

勘误后的所有 job 与 aggregate 只能写入：

```text
analysis/results/tpd_clean_v8_mprs_counterfactual_v2/
```

v2 aggregate 必须同时报告：

- 4 个 variant/seed group；
- 12 个 checkpoint job；
- 每个 group 的 7-block correction 统计；
- paired topology、coverage 和 shift stress；
- 所有 source/data/job bindings；
- `counterfactual_gate_pass`。

只有 v2 aggregate 为 true，且 Full/Capacity 显存 v2、CPU/GPU2/GPU3 smoke、
V8 exact resume、training source lock 与 acceptance source lock 全部通过，才可生成
`formal_training_authorized=true`。否则 launcher 必须拒绝启动，且不得产生新的 V8
Pd、Fa、mIoU 声明。
