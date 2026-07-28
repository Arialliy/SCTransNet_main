# TPD-Clean V8-MPRS-DCH 正式实验协议

## 0. 前置诊断勘误

`TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md` 是本协议的规范性组成部分，
并在前置诊断口径冲突时优先。它不改变模型、训练、Pd/Fa/mIoU 或 Gate A–E，只修正：

- 下采样后 target/hard-negative mask 互斥；
- 固定 V7-covered GT 的配对拓扑统计与 coverage 非下降；
- `torch.roll` 项的环面网格压力解释；
- job、ordered IDs、数据、源码、registry 与 aggregate 的摘要绑定。

首轮 `counterfactual_v1` 目录保留但不授权训练；勘误后的运行必须写入
`counterfactual_v2` 新目录。

## 1. 模型身份

- 候选：`tpd_clean_v8_mprs_dch_full`
- 容量对照：`tpd_clean_v8_mprs_dch_capacity`
- 主线：Keep–Context–Saliency，不增加第四语义分支
- 五节点：`embeddings_1` 的 3 个非终端节点与 `embeddings_2` 的 2 个非终端节点
- 只替换：`mtc.embeddings_1`、`mtc.embeddings_2`
- 总参数：10,843,155
- 浅层 embedding 参数：66,176

五节点是跨层证据节点，不是五个并行分支。模型的语义来源仍只有 Keep、Context、
Saliency 三类。

## 2. 唯一结构变量

V7-DCH 的 Keep、Context、DCH headroom、SCTransNet encoder、SCTB、decoder 和
六路 BCE 均保持不变。V8 唯一改变 Saliency 表示：

\[
C_0=\frac14\sum_{p=0}^{3}Z_p,\qquad
S_0=\max_p Z_p-C_0,
\]

\[
S_p=S_0+\frac{Z_p-C_0}{3}.
\]

其逐通道逐 cell 约束为：

\[
\sum_p S_p=4S_0,\qquad S_p\ge0.
\]

正式 forward 使用与显式四相位投影等价的复用式：

\[
S_a^{V8}=S_a^{V7}+\frac{(K-b)-C_a}{3}.
\]

显式五维相位张量只允许出现在测试与诊断接口，不进入普通 forward。Full 和
Capacity 都固定为每 block 三次卷积；没有新增参数或持久 buffer。

## 3. DCH 与对照

令 \(a=\tanh(s)\)，\(V\) 为 V7-DCH 的零均值 Context modulation。

- Full：\(H=1+|a|(1-|a|)V\)，输出 \(K+S_a^{V8}(aH)\)
- Capacity：\(H=1\)，输出 \(K+S_a^{V8}a\)

两者参数、初始化和 Saliency 完全配对，只由固定的 Context gate 区分。

## 4. 正式训练矩阵

| Variant | Seed | 初始化 | Epochs |
|---|---:|---|---:|
| Full | 42 | paired fresh | 800 |
| Capacity | 42 | paired fresh | 800 |
| Full | 3407 | paired fresh | 800 |
| Capacity | 3407 | paired fresh | 800 |

固定训练设置：

- 数据：NUDT-SIRST 官方训练集的既有 530/133 内部分割
- 官方测试集：不访问
- 输入 patch：256
- batch size：16
- optimizer：Adam
- loss：六路 post-sigmoid BCE 求和
- AMP：关闭
- workers：0
- 每个 epoch 评估
- warmup、cosine、阈值、匹配半径、tiny 面积定义均继承 V7-DCH
- 物理 GPU：只允许 2、3；每个训练进程只看到其分配 GPU 的 UUID

每组固定保存：

- Pd-primary `best.pth.tar`
- mIoU-secondary `best_miou.pth.tar`
- 最后评估 epoch `last.pth.tar`
- exact journal、`metrics.jsonl`、`protocol.json`、`summary.json`

V8→V8 支持 epoch-boundary exact resume。V7→V8 只允许 model state 的
`strict=True` 只读诊断，不恢复 V7 optimizer、journal 或 RNG。

## 5. 正式训练前门槛

以下项目必须全部通过才启动四组 800 epochs：

1. 正式模型、完整 SCTransNet、数学不变量及梯度测试通过；
2. V7-DCH 与 V8 state keys 完全相同，12/12 checkpoint 可 strict-load；
3. zero scale 时完整六输出与 dense SPD 一致；
4. Full/Capacity 的 zero-scale 输出、共享梯度和首个 Adam step 一致；
5. 普通 forward 每 block 三次卷积且不物化显式 phase-Saliency；
6. V8→V8 exact resume 逐 tensor 一致；
7. 12-checkpoint counterfactual 全部 finite；
8. 每个 variant 聚合后，两个 seed 均满足
   `target_correction_lift > 1.0`；
9. 注册阈值上的 median largest-fragment fraction 不下降，aggregate
   fragment excess 不增加；
10. 四 offset 输出位移一致性 `V8/V7 <= 1.10`；
11. 优化路径峰值显存相对 V7 增幅不超过 10%，且低于显式参考路径；
12. CPU、物理 GPU 2 和 3 的完整模型 smoke 通过；
13. training source lock 与 acceptance source lock 校验通过。

前置门槛只决定是否投入正式训练，不替代最终 Pd、Fa、mIoU 裁决。

Correction、拓扑与位移检查的精确定义由规范性勘误 V1 冻结：

- target/hard-negative adaptive pool 后使用 target-priority 去重，且每 block
  两类计数均必须非零；
- topology 使用固定 V7-covered GT identity 做 V7/V8 配对，V8 未覆盖记 0，
  并要求 reference coverage 不下降；
- 位移项仍使用原顺序前 16 张、四 offset、`torch.roll`、crop16 和 normalized L1，
  但只称为 toroidal grid-offset stress，不声称卷回影响被排除；
- 所有 base、shift、topology probability 来自同一正式 forward 数值路径；
- v2 job 和 aggregate 必须绑定 ordered IDs、数据、registry、输入与当前 V8 源码。

正式 Pd、Fa、mIoU 不使用上述前置诊断聚合口径，仍覆盖全部 133 张内部验证图像。

## 6. 评估与 Gate

预测、连通域、Hungarian matching、Fa 计数、tiny-Pd、mIoU 与 closed-interval
阈值生成均继承 V7-DCH，不做修改。正式产物为 12 份 checkpoint 和 8 份
closed-interval sweep。

固定 Fa budgets：

```text
1e-6, 5e-6, 1e-5, 5e-5, 1e-4
```

Gate A–E 完整继承 V7-DCH，并联合使用 Pd、Fa、mIoU；不能只凭 mIoU 或单个
checkpoint 作出结论。只有 A、B、C、D、E 全部通过，才授权接入 NER。

## 7. 结果边界

两 seed 内部验证只用于项目阶段裁决。即使 Gate A–E 全部通过，也不自动建立
跨随机性、跨数据集或官方测试集结论。训练过程中不得依据中间指标改变公式、
epoch、checkpoint role、阈值、Fa budget 或 Gate。
