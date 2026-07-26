# TPD-NER-v5：五节点嵌套证据中继冻结协议

状态：V5-NER 模型、训练入口、闭区间评估器和 CPU 两步预检代码已完成；
正式训练尚未启动。只有 TPD-Clean-v5 的 4×800 实验完成且 Gate A–E
全部通过，才允许创建正式 NER 运行目录和启动训练。

## 1. 主线与本轮边界

本轮不改变 TPD-SCTransNet 的目标保真 tokenization 主线。TPD 仍只有
Keep、Context、Saliency 三个语义源，不增加第四个并列 tokenizer 分支。

“五节点”指两条既有分级下采样路径产生的五个中间证据张量：

```text
embeddings_1: h11 -> h12 -> h13 -> emb1
embeddings_2: h21 -> h22 -> emb2
```

它们不是五个输入分支。`emb1/emb2` 仍进入原 SCTB；五个非终点状态只供
窄宽度 Nested Evidence Relay（NER）使用。

本轮只比较 tokenizer 与 NER 的交互。目标存活监督、Query-only FG、
额外频率损失、K/V 修改和 decoder 频率回注均不进入本轮。

## 2. 固定结构

### 2.1 TPD tokenizer

TPD 侧直接复用冻结的 `tpd_clean_v5_full`：

```text
K = Conv1x1(PixelUnshuffle2(X))
C = AvgPool2(X)
S = MaxPool2(X) - C
Q = tanh((C - mean_hw(C)) / rms_hw(C - mean_hw(C)))
P = 1 + 0.5 * Q
Y = activation(K + S * tanh(saliency_scale * P))
```

每个 2× block 只有一个零初始化 `saliency_scale`，Context 没有独立
scale。TPD 只替换 `mtc.embeddings_1/2`。

### 2.2 参数匹配 Progressive 对照

Progressive 使用与 V5 相同的 4/3 级下采样深度。每级为：

```text
Y = activation(Conv2d(C,C,kernel=2,stride=2,bias=True)(X)
               * (1 + tanh(channel_gain)))
```

`channel_gain` 为 C 维零初始化参数且参与每次前向。每级参数量严格为
`4*C^2 + 2*C`，与一个 V5 block 相同；两条浅层 embedding 的总参数量
均为 66,176。

### 2.3 五节点 NER

固定证据递推为：

```text
q4 = Phi4(h13, h22, up(d5))
q3 = Phi3(h12, h21, q4, up(d4))
q2 = Phi2(h11, q3, up(d3))
```

顺序严格为 `q4 -> d4 -> q3 -> d3 -> q2 -> d2`，不存在同级回路。
中继宽度固定为 `Ce=8`。每一级只产生一个单通道空间门控，在原 CCA
处理 skip 后、与上采样 decoder 特征拼接前执行：

```text
skip_modulated = skip_after_CCA * (1 + tanh(Gs(qs)))
```

三个 `Gs` 的末层权重和偏置均从零初始化。因此对同一 tokenizer 和同一
seed，Relay-on 与 Relay-off 的公共 state 完全一致，step 0 的六个深监督
输出逐元素一致；Relay-on 只增加 `tpd_ner.*` 参数。

生产维度参数契约：

| 项目 | 固定值 |
| --- | ---: |
| Relay-off 总参数 | 10,843,155 |
| Relay-on 总参数 | 10,854,446 |
| NER 新增参数 | 11,291 |
| 其中空间 gate 参数 | 27 |
| 浅层 embedding 参数 | 66,176 |

## 3. 四变体交互矩阵

| Tokenizer | Relay off | Relay on |
| --- | --- | --- |
| TPD-Clean-v5 Full | `tpd_clean_v5_full_relay_off` | `tpd_clean_v5_full_relay_on` |
| 参数匹配 Progressive | `progressive_relay_off` | `progressive_relay_on` |

每个配对必须使用相同模型 seed 和完全相同的公共初始化。两个 Relay-on
变体使用 tokenizer 无关的局部随机流初始化 NER，使 NER 初始 state 可直接
比较，同时不扰动公共参数。

## 4. 正式训练前置门

只有下列条件全部满足，才允许编写/启用正式 worker 与 launcher：

1. TPD-Clean-v5 四个 run 均完成 800 epochs；
2. V5 completion marker 与最终汇总均存在；
3. V5 Gate A、B、C、D、E 全部通过，且
   `engineering_gate_passed=true`；
4. V5 训练源码锁、后处理源码锁和最终输入哈希一致；
5. 本协议对应的 V5-NER source lock 校验通过；
6. CPU 及指定 GPU 的四变体两步预检均通过；
7. 不覆盖 baseline、TPD-v1、Clean-v2/v3/v4/v5 或旧 NER 产物。

Gate 未通过时，V5-NER 保持“代码完成、未接入正式训练”的隔离状态。

## 5. 训练与评估设计

正式阶段使用与 V5 相同的 NUDT-SIRST 530/133 内部分割、数据增强、
FP32 训练、checkpoint 选择规则和闭区间 Pd–Fa 评估器；不访问官方
测试集。

第一轮计划使用配对 seed `42` 和 `3407`，四变体各训练 800 epochs，
共 8 个 fresh run。资源映射、并发数和启动批次在 V5 Gate 通过后另建
执行协议冻结；不得在本文件中把当前运行卡的瞬时状态写成永久前提。

每个 run 必须保存：

- 连续 800 行 `metrics.jsonl`；
- Pd-primary、mIoU-secondary 和 epoch-800 checkpoint；
- 固定阈值 0.5 指标；
- `best` 与 `best_miou` 的闭区间 Pd–Fa sweep；
- 五个预注册 Fa budget：
  `1e-6 / 5e-6 / 1e-5 / 5e-5 / 1e-4`；
- split、protocol、源码、checkpoint 和评估器哈希；
- 逐目标尺寸分组结果，至少单列 `A<=9`。

所有结果字段在正式运行完成前均为 `TBD`，不得用 smoke 或训练中途值填充。

## 6. NER 晋级门

NER 是否并入整模由交互增量决定，不由单个 Relay-on 的绝对指标决定。
对每个 seed 分别计算：

```text
Delta_T = TPD_on - TPD_off
Delta_P = Progressive_on - Progressive_off
Interaction = Delta_T - Delta_P
```

晋级至少要求：

1. 两个 seed 的 Relay-on 都完成 800 epochs，且审计输入齐全；
2. `TPD_on` 在至少一个预注册 Fa budget 提高 Pd，且该点不以更高 Fa
   或更低 mIoU 被 `TPD_off` 同时覆盖；
3. `A<=9` 的 Pd 改善不能只来自放宽 Fa；
4. 两个 seed 的关键预算上 `Interaction` 方向一致，且 TPD 的中继增量
   大于 Progressive 的通用中继增量；
5. 在最严格 `Fa<=1e-6` 区域，NER 不得使 TPD 的可用工作点退化；
6. Relay gate、fusion 和 tokenizer 控制参数均实际获得有限梯度并更新；
7. 参数量、峰值显存和推理耗时单独报告。

通过仅表示 NER 可以作为 TPD 的第二模块接入后续整模；不自动设置
`paper_core_established=true` 或 `stability_claim_supported=true`。
未通过时只保留 TPD 核心主线，NER 保持独立候选，不用存活监督或 FG
掩盖本轮交互结果。

## 7. 当前代码状态

| 部分 | 文件 | 状态 |
| --- | --- | --- |
| 四变体模型与五节点前向 | `model/tpd_ner_v5.py` | 已完成 |
| 训练 builder/CLI | `experiments/train_tpd_ner_v5.py` | 已完成 |
| 闭区间评估器 | `experiments/evaluate_tpd_ner_v5_pd_fa.py` | 已完成 |
| 四变体两步预检 | `experiments/smoke_tpd_ner_v5.py` | 已完成 |
| 模型/训练/评估/预检测试 | `tests/test_*tpd_ner_v5*.py` | 已完成 |
| source lock | `experiments/tpd_ner_v5_source_lock.json` | 随本协议冻结 |
| 正式 worker/launcher | 独立后续文件 | Gate 通过前禁止启用 |
| 目标存活监督接入 | 独立后续阶段 | 未接入 |
| Query-only FG 接入 | 独立后续阶段 | 未接入 |
