# SCTransNet TPD–NER V8-MPRS-DCH V3 固定实验协议

## 1. 协议状态与目标

- 候选模型：
  `tpd_ner_v8_mprs_dch_v3_full_relay_on`
- 数据集：NUDT-SIRST 官方训练集的固定内部划分；
- 训练/验证图像数：530/133；
- train/model/relay seed：42；
- split seed：20260722；
- 正式训练：800 epochs；
- 当前不访问官方测试集；
- 当前只运行单个固定 seed，不安排多 seed；
- V2 formal800 完成并生成正式裁决前，不启动 V3 formal800。

本协议的目标是验证：在完整 V8-MPRS-DCH + 五节点 NER 模型中，
增加三个中心化后的分层 DC offset，能否同时改善 Pd、Fa、mIoU
以及五档 Fa-budget 工作点。

V3 的代码合同已经建立，但尚无训练结果。因此：

- “V3 工程实现通过”已经可以成立；
- “V3 性能正向”在 formal800 与五预算评估完成前不能成立。

## 2. 不允许变化的模型主线

以下内容必须保持：

1. SCTransNet 可比主干；
2. Keep–Context–Saliency 三路 patch embedding；
3. MPRS-DCH 主模块；
4. 五个证据节点；
5. `q4 → q3 → q2` 递归 relay 顺序；
6. 六个 deep-supervision 输出；
7. relay width 8；
8. V2 的逐源 RMS、融合后 RMS、空间中心化和有界
   arctangent residual mask；
9. NUDT-SIRST 530/133 划分、训练轴、优化器、调度器、损失和
   checkpoint 选择规则。

V3 不增加第四条 tokenizer 分支，也不删除 Keep、Context 或
Saliency。

## 3. V3 唯一结构修改

权威实现：

- `model/tpd_ner_v8_mprs_dch_v3.py`
- 类：
  `TPDNERV8MPRSDCHV3SCTransNet`
- relay：
  `RMSBalancedCenteredDCOffsetEvidenceRelay`
- relay version：
  `v3_rms_centered_arctangent_post_center_dc`

每个 decoder relay stage 仅增加一个可学习标量：

```text
centered_logits = logits - mean_hw(logits)
shifted_logits  = centered_logits + dc_offset[stage]
mask            = atan(pi * shifted_logits) / pi
```

stage 为 4、3、2，因此总共增加三个参数：

```text
tpd_ner.dc_offsets.4
tpd_ner.dc_offsets.3
tpd_ner.dc_offsets.2
```

固定合同：

- 三个 offset 全部零初始化；
- gate 仍为 `bias=False`；
- mask 仍严格位于 `(-0.5, 0.5)`；
- skip factor 仍严格位于 `(0.5, 1.5)`；
- relay 参数：11,291；
- 完整模型参数：10,854,446；
- relay state keys：19；
- step 0 时，V3、V2 和 relay-off 的六个输出逐元素相等；
- V3/V2 checkpoint 不能 strict 互载；
- V3 不改变父模型和 tokenizer 参数的初始值。

该修改针对 V2 的直接代码限制：V2 的 gate logits 在映射前被空间
中心化，因而只能表达相对空间差异，不能学习每个 stage 的整体背景
偏移。V3 保留空间中心化，同时恢复三个受控的全局校准自由度。

## 4. 正式训练合同

权威入口：

```text
experiments/train_tpd_ner_v8_mprs_dch_v3_exact.py
```

固定参数：

| 项目 | 固定值 |
|---|---:|
| dataset | NUDT-SIRST |
| epochs | 800 |
| batch size | 16 |
| patch size | 256 |
| workers | 0 |
| validation fraction | 0.20 |
| eval every | 1 |
| base learning rate | 1e-3 |
| minimum learning rate | 1e-5 |
| warmup epochs | 10 |
| threshold | 0.5 |
| match radius | 3.0 |
| tiny area | 9 |
| precision | FP32 |
| AMP | false |
| train/model/relay seed | 42 |
| split seed | 20260722 |

损失保持为六个 sigmoid 输出的等权 BCE 总和：

```text
L = BCE(o1,y) + BCE(o2,y) + BCE(o3,y)
  + BCE(o4,y) + BCE(o5,y) + BCE(o6,y)
```

首个 V3 formal800 不同时改变 loss 权重、优化器、scheduler、
tokenizer、MPRS-DCH 或 NER 拓扑。

## 5. GPU 与精确续训

- 只允许物理 GPU2 或 GPU3；
- 不使用 GPU0 或 GPU1；
- worker 内只暴露一个已登记 GPU UUID；
- 进程内部设备固定为 `cuda:0`；
- 不要求等待整张卡完全空闲，但启动前必须满足显存需求；
- 同时最多运行一个 V3 formal 训练任务。

版本私有环境变量：

```text
TPD_NER_V8_MPRS_DCH_V3_PHYSICAL_GPU_INDEX
TPD_NER_V8_MPRS_DCH_V3_PHYSICAL_GPU_UUID
```

允许的初始化方式：

1. `--fresh`；
2. 同一 V3 variant、同一 run identity、epoch 边界处的
   `--exact-resume`。

明确禁止：

- 从 V1 checkpoint 精确续训；
- 从 V2 checkpoint 精确续训；
- 从 relay-off checkpoint 精确续训；
- 从不同 variant 或不同 source-lock identity 精确续训；
- 在恢复 optimizer、scaler、scheduler 或 RNG 之后才发现版本不符。

跨版本 checkpoint 必须在任何训练状态恢复前被拒绝。

## 6. 对照矩阵

最终报告必须包含 8 行：

| 模型 | `best` | `best_miou` |
|---|---:|---:|
| baseline SCTransNet | 1 | 1 |
| V1 relay-off | 1 | 1 |
| V2 relay-on | 1 | 1 |
| V3 relay-on | 1 | 1 |

角色定义：

- required control：V1 relay-off；
- structural predecessor：V2 relay-on；
- baseline：只作完整基线报告，不直接决定 V3 放行；
- 仅新运行 V3 的两个 sweep；
- baseline、V1 和 V2 sweep 只读重验，不重新生成或修改。

两个 checkpoint role：

1. `best.pth.tar`：
   `best_validation_pd_primary`；
2. `best_miou.pth.tar`：
   `best_validation_miou_secondary`。

## 7. 指标与阈值

固定阈值 0.5 必须报告：

- Pd；
- matched targets / 189；
- tiny-Pd；
- matched tiny targets / 39；
- Fa；
- false objects/image；
- mIoU；
- nIoU；
- pixel precision、recall、F1。

五个预注册 Fa budgets：

```text
1e-6, 5e-6, 1e-5, 5e-5, 1e-4
```

每个预算点必须由完整闭区间 sweep 的原始点重新推导，不能只读取
已有摘要。阈值集合必须包含：

- 0；
- 1；
- float32 中严格小于 1 的最大值；
- 固定阈值 0.5；
- 预设额外阈值；
- 合法的经验分位阈值；
- 模型预测分数产生的自适应阈值。

## 8. V3 绝对门槛

### 8.1 `best` / Pd-primary

固定阈值 0.5 同时满足：

```text
matched targets >= 188/189
Fa              <= 1e-6
mIoU            >= 0.933647
```

五预算满足：

```text
Pd@Fa<=1e-6 >= 187/189
Pd@其余四个预算 >= 188/189
```

### 8.2 `best_miou`

固定阈值 0.5 同时满足：

```text
matched targets >= 187/189
Fa              <= 1e-6
mIoU            >= 0.946542
```

五预算使用与 Pd-primary 相同的最低 matched-target 合同。

## 9. 双配对门槛

对每个 checkpoint role 分别执行配对比较。

### 9.1 V3 对 V1 relay-off

五个 Fa budgets 中：

```text
V3 matched targets >= V1 matched targets：至少 4/5
V3 matched targets >  V1 matched targets：至少 1/5
```

### 9.2 V3 对 V2 relay-on

五个 Fa budgets 中：

```text
V3 matched targets >= V2 matched targets：至少 4/5
V3 matched targets >  V2 matched targets：至少 1/5
```

V2 自身是否通过 V3 的绝对门槛不参与 V3 决策。V2 是直接结构前驱，
而不是 V3 的绝对放行前置条件。

## 10. 最终六项 AND 裁决

以下六项必须全部通过：

1. Pd-primary absolute；
2. mIoU-secondary absolute；
3. Pd-primary V3-vs-V1 paired；
4. mIoU-secondary V3-vs-V1 paired；
5. Pd-primary V3-vs-V2 paired；
6. mIoU-secondary V3-vs-V2 paired。

全部通过：

```text
FULL_MODEL_GATE_PASSED
```

任一失败：

```text
RETURN_TO_MODEL_OPTIMIZATION
```

V1 的 absolute gate、V2 的 absolute gate和 baseline 不加入上述
六项 AND。

tiny-Pd 必须报告，但当前 39/39 已出现天花板，因此不作为独立放行
项；如果 V3 tiny-Pd 低于现有 39/39，则必须在最终报告中单独标记为
退化。

## 11. epoch 401–450 诊断门

该门用于尽早判断 DC offset 是否朝降低 Fa 的方向工作，不替代
epoch800 的正式裁决。

V3 在 epoch 401–450 应同时观察：

```text
median false pixels/epoch                   <= 40
mean unmatched predicted objects/epoch      <= 7.5
Pd>=183 且 Fa<=5e-6 的 epoch 数             >= 25/50
mean Pd 相比 V2 同窗下降                     <= 1 个目标
mean mIoU 相比 V2 同窗下降                   <= 0.003
```

未通过该诊断门时：

- formal 训练产物仍应完整保存；
- 先检查三个 learned DC offset、各 stage mask、错误像素和错误目标；
- 不得把诊断门伪装为最终性能结论；
- 后续修改仍需保持 SCTransNet、Keep–Context–Saliency、
  MPRS-DCH 和五节点 NER 主线。

## 12. 源锁与产物冻结顺序

V3 使用两份独立锁：

```text
experiments/tpd_ner_v8_mprs_dch_v3_exact_source_lock.json
experiments/tpd_ner_v8_mprs_dch_v3_acceptance_source_lock.json
```

生成/验证入口：

```text
experiments/freeze_tpd_ner_v8_mprs_dch_v3_source_locks.py
```

冻结顺序：

1. 完成 V3 model、exact trainer、协议和测试；
2. 把完整 V3 training runtime source closure 写入 training lock；
3. 完成 evaluator、postprocess、smoke、handoff、lane 和 launcher；
4. 把 acceptance closure 写入 acceptance lock；
5. acceptance lock 必须绑定 V3 training-lock SHA；
6. acceptance lock 必须绑定当前 V2 training/acceptance 两份锁；
7. V2 和 V3 的 training data SHA 必须一致；
8. launcher 与 lane 启动前验证 V3 两份锁以及 V2 上游锁；
9. 锁文件只允许首次发布，不覆盖；
10. 一旦 checkpoint 嵌入锁 SHA，不再修改该 run 的锁。

任何后续版本修改必须使用新的 schema、锁文件和 run root，不能把
旧 checkpoint 解释为新版本。

## 13. 正式执行顺序

```text
V2 formal800 完成
  -> V2 best / best_miou 五预算 sweep
  -> V2 canonical JSON / Markdown / marker
  -> 验证 V2 未通过或需要继续优化
  -> V3 CPU 工程测试
  -> V3 GPU2/3 确定性 smoke
  -> V3 fresh formal800
  -> 如中断，仅同版本 exact-resume
  -> V3 两 checkpoint sweep
  -> 8 行矩阵与六项 AND 裁决
```

handoff 不允许：

- 停止、重启或修复 V2 训练；
- 修改 V1、V2 或 baseline 参考 sweep；
- 在 V2 formal marker 不完整时启动 V3；
- 同时启动两条 V3 formal trajectory。

## 14. V3 未通过后的修改顺序

若 V3 未通过，按以下顺序处理，每次只修改一类因素：

1. 检查三个 offset 是否学习为有效负背景校准，以及哪个 stage
   贡献主要错误目标；
2. 若 offset 方向正确但幅度不足，优先调整 V3 内部 offset 的学习
   速率或有界缩放，不改主线；
3. 若 DC 校准已降低 Fa 但 Pd/mIoU 下降，再调整 relay residual
   强度上限；
4. 只有结构校准稳定后，才单独测试六头 loss 权重；
5. 每个后续版本重新建立 model、trainer、测试、锁和结果目录；
6. 不同时混入 tokenizer、MPRS-DCH、NER 拓扑和 loss 多项修改。

最终目标始终是完整模型的 Pd–Fa–mIoU 综合提升，而不是只证明某个
中间机理。

## 15. 结论边界

本协议只允许建立：

- NUDT-SIRST 固定 530/133 内部划分；
- 单 seed 42；
- 两个 checkpoint role；
- 五个 Fa budgets；
- baseline/V1/V2/V3 的内部验证比较。

本协议不支持：

- 多 seed 稳定性结论；
- 跨数据集结论；
- 官方测试集结论；
- 仅凭 tiny-Pd 39/39 建立目标保真因果结论。

当前截至 V2 epoch600 的证据显示低 Fa 仍是主要瓶颈。V3 是针对该
瓶颈的下一段模型代码，不是已经得到性能确认的最终模型。
