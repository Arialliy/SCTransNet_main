# SCTransNet：冻结完整主模型后的 IRSTD-1K 专项优化方案

**方案名称：IRSTD-BGCR（Baseline-Guided Core–Ring Repair）**  
**目标模型：`TPD8 + NER4 + QFG2，TSS-off`**  
**更新时间：2026-08-08**

**执行修订：2026-08-08（代码、缓存、三折 OOF 与全量候选均已封存）**

> 用户补充权重根目录 `/home/ly/SCTransNet` 后，已找到并严格绑定
> IRSTD 独立 Baseline 的 epoch 713 best-mIoU 和固定 epoch 1000 两个
> checkpoint。为避免把历史 official-test 选点信息带入新头训练，
> **正式 teacher 固定为 epoch 1000**；epoch 713 只是待超越的 operational
> Baseline 参考，不参与 loss。本轮只访问冻结 official-train 投影，
> `official_test_accessed=false`；不重新构造 official-test loader/index。

> **最终执行裁决：BGCR 未成功。** 三折 OOF 的零门槛 selector 选择
> `epoch 0`，即与 Current 精确相同的恒等候选；所有非零训练 epoch 的
> pooled mIoU 都更低。因此 BGCR **不替换 Current**，封存的 full candidate
> 仅用于身份与复现审计，不作为性能模型发布。

---

## 0. 最终结论

本轮已经完成实现、验证、缓存、三折训练、OOF 选择和全量候选封存。
最终仍不继续修改完整主模型，也不再设计 PBDR-V6：

- `TPD8 + NER4 + QFG2，TSS-off` 作为统一主模型冻结；
- PBDR-V4、PBDR-V5 已完成实现、训练和审计，V5 未超过内部包络，停止采用；
- NUAA-SIRST、NUDT-SIRST 已超过独立 Baseline，不再投入结构搜索；
- 本轮开始前唯一未闭环的是 IRSTD-1K：Current 的 official mIoU 为 **66.0251%**，独立
  Baseline epoch 713 为 **67.7357929%**，按已舍入 Current 计算约差
  **−1.7107 个百分点**；
- 已按 **IRSTD 专项、冻结主干、附加式修复** 完成 BGCR；三折 OOF
  选回 epoch 0，证明本次新增头没有产生可采用的内部提升。

本轮实现的是一个仅用于 IRSTD 的 **Baseline-Guided Core–Ring Repair（BGCR）头**：

1. 从冻结 Current 提取全分辨率 `u1`、`out`、`d0`、`gt2–gt5`；
2. 加入原图局部对比度，而不再依赖粗尺度 `q4`；
3. 将误差显式拆为“目标核心需要上调”和“匹配组件外的附着 halo/桥接区域需要下调”；
4. 用严格绑定的固定 epoch-1000 Baseline 生成 Current-vs-Baseline
   优势图；epoch-713 best 不进入训练图；
5. 通过 logit core-drop 与 ring-injection 反事实，主动制造自然训练集中稀缺的漏检和附着 halo；
6. 训练只更新约 **27.2K** 个新参数，Current 的 564 个推理 state keys 和全部 buffer 保持不变。

本方案**不设最小增益幅度**：不要求 `+0.005`、不要求百分比改善、不要求所有指标同时不退化。IRSTD 的选择顺序以 mIoU 为第一项；任意严格正的 mIoU 改善都算改善。固定 `probability > 0.5` 仅是既有测量工作点，不是性能接受门槛。

需要把“确保提升”分成两个可验证层次：

- **工程上可保证**：主模型不会被训练破坏；epoch 0 是与 Current 精确相同的恒等候选；内部选择结果不会低于 Current。
- **未知 official test 上的严格正提升**：必须由新的、独立授权的
  冻结评估协议确认。V4 已消耗的 official 访问不得被 BGCR
  重用，也不得根据已知 official 结果回调本轮 loss。

实际 selector 触发了第一层安全回退：`selected_epoch=0`、
`strictly_improves_epoch0_miou=false`。因此不存在需要申请新 official
评估的 BGCR 性能候选，本轮直接保留 Current。

---

## 1. 冻结边界

### 1.1 永久冻结的部分

以下内容不再修改：

```text
TPD8
NER4 Tail-Aware
QFG2-CROA
TSS OFF
encoder / decoder
up_decoder1
outc
deep-supervision heads
Current checkpoint 中的全部参数与 buffer
```

BGCR 不属于主模型结构重设计，而是 IRSTD 域专用的最终读出修复臂。新代码放在独立文件中，Current 源文件保持不动。

### 1.2 明确停止的方向

不再执行：

- PBDR-V5 的 epoch、loss 权重或路由上限 sweep；
- PBDR-V6 通用校准器；
- 全局 `out/d0` 固定混合；
- 重新修改 TPD、NER、QFG；
- 重新启用 TSS；
- 解冻 `outc.*` 或 `up_decoder1.*`；
- 用 component-Fa 单独代表像素分割质量；
- 通过 official-test 阈值搜索制造结果。

---

## 2. 当前性能缺口

用户表格的列语义是“最佳 mIoU checkpoint”与“epoch 1000 指标”并列，
不是一个 checkpoint 的五项指标。绑定后的正确两个工作点为：

| Baseline checkpoint | mIoU | nIoU | F1 | Pd | Fa ×10⁻⁶ | 用途 |
|---|---:|---:|---:|---:|---:|---|
| epoch 713 best | **67.7357929%** | 67.1640%* | 80.7586%* | 93.2659933% | 20.8005386 | operational 对比目标，不做 teacher |
| epoch 1000 | 66.6485671% | 66.8172%* | 79.9749%* | 93.2659933% | **11.5959201** | 本轮唯一固定 teacher |

`*` 日志只保留了百分数四位小数，不得伪造更多精度。Current 的
official 工作点仍为 mIoU `66.0251%`、nIoU `66.5585%`、F1 `79.5363%`、
Pd `93.2660%`、Fa `11.7288×10⁻⁶`，但本轮不重新读取 official test。

两个重要判断：

1. epoch 1000 与 Current 的 Pd/Fa 接近，但 mIoU 仍高约 `0.6235 pp`；
   epoch 713 的 mIoU 更高、Fa 却更差。因而主要矛盾确实是像素交并
   质量和轮廓工作点，不能把 best mIoU 与 epoch-1000 的低 Fa 拼成
   一个不存在的 Baseline 单点。
2. Current 在 NUAA/NUDT 已胜 Baseline，说明不能为了 IRSTD 再动统一主干，否则会重新打开已经关闭的跨域冲突。

重要污染声明：原 Baseline `train.py` 从 epoch 500 起每 epoch 评估
official test，epoch 713 是按 official-test mIoU 选出的 operational best，不是
未见测试集的模型选择。因此它可作为用户已指定的对比目标，但禁止
进入 BGCR 训练 teacher。

---

## 3. 代码路径复核

### 3.1 Current 的正式部署输出是 `out`，不是 `d0`

Current 的 `_forward_with_relay`：

- `u1 = up_decoder1(d2, x1)`；
- `out = outc(u1)`；
- `gt2–gt5` 只用于 deep supervision；
- `d0 = outconv(cat(gt2,gt3,gt4,gt5,out))`；
- `mode != train` 时返回 `sigmoid(out)`。

因此 IRSTD 专项头应以 `out` 为不可变锚点；`d0/gt2–gt5` 只可作为上下文，不应再次作为全图直接混合输出。

代码依据：

- [Current 主模型实现](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py)

### 3.2 V4 的结构上限无法救回已定位的 IRSTD 漏检

V4 `best_miou` 的正向 logit 上限为 `+0.60`。内部定位得到两个漏检峰值：

```text
XDU641 component 2: Current peak = -1.19466
XDU202 component 1: Current peak = -1.52051
```

即使 V4 正残差完全饱和：

```text
-1.19466 + 0.60 = -0.59466 < 0
-1.52051 + 0.60 = -0.92051 < 0
```

两者在结构上都不可能越过固定 0-logit 决策边界。V4 Stage-1/Stage-2 的实际峰值也仍为负。因此，继续调 V4/V5 的保持损失不能修复这一容量错配。

代码依据：

- [V4 角色校准器](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/model/tpd_role_aligned_residual_calibrator_v4.py)

### 3.3 V4 atlas 没有单独表示“匹配组件的附着 halo”

V4 atlas 只有三类：

```text
rescue  = unmatched target components
suppress = unmatched predicted components
preserve = matched target components
```

但 IRSTD 的关键 FP 经常属于**已经匹配到目标的预测组件**：预测轮廓沿目标边缘向外扩张，或通过窄桥连接到背景块。此类像素：

- 会进入 mIoU 的 union；
- 会降低 pixel precision/F1；
- 但不属于 unmatched prediction component；
- 因而不会进入 V4 `suppress_component_ids`；
- component-Fa 还可能因“桥接后成为 matched component”而看起来更好。

这就是 V4/V5 能减少未匹配组件，却仍不能提高 IRSTD mIoU 的核心拓扑漏洞。

代码依据：

- [V4 component atlas](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/experiments/pbdr_v4_component_atlas.py)
- [V4 component loss](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/experiments/pbdr_v4_component_loss.py)

### 3.4 V5 改的是保护归一化，不是错误表示

V5 替换了三项：

- preserve smooth-peak no-drop；
- preserve component 内 Current-positive support no-drop；
- active background probability no-increase。

这些修改能防止简单退化，但没有解决：

- 正残差容量不足；
- 附着 halo 未单独标注；
- 自然 rescue 样本稀缺；
- 缺乏原图局部对比度；
- 目标核心和外圈使用同一个通用残差场。

代码依据：

- [V5 target-preservation loss](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/experiments/pbdr_v5_target_preservation_loss.py)
- [V5 internal protocol](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/experiments/PBDR_V5_INTERNAL_PROTOCOL.md)

---

## 4. V4/V5 连续失败的完整原因

### 4.1 误差类别极度不平衡

IRSTD development-train atlas：

| 类别 | 组件数 | 像素数 |
|---|---:|---:|
| preserve | 949 | 48,170 |
| rescue | 16 | 148 |
| suppress | 83 | 882 |

rescue 像素只有 preserve 像素约 **0.31%**。普通像素 BCE/Tversky 和随机正样本 crop 无法为 16 个自然漏检产生稳定的专门梯度。

### 4.2 默认数据采样只保证“有目标”，不保证“有错误”

仓库默认训练集使用 `pos_prob=0.5` 的随机 crop，再做翻转/转置。它只能增加含目标 patch 的概率，不能定向采到：

- 低峰漏检目标；
- 匹配组件的附着 halo；
- 桥接像素；
- Current 与 Baseline 发生互补的区域。

代码依据：

- [dataset.py](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/dataset.py)
- [utils.py random_crop](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/utils.py)

### 4.3 粗尺度模型证据无法区分“目标核心”和“目标光晕”

V4 使用 `q4`、deep readout 共识、`d0` 和 `u1`。其中 `q4` 为粗尺度语义，在小目标区域上插值后天然宽化；`d0/gt2–gt5` 又与 `out` 高度相关。它们能表达“这里可能有目标”，却不一定能表达“哪几个像素是核心、哪几个像素是外溢”。

BGCR 实现因此加入原图局部高通、局部均值差和局部标准差，让修复头看到真实红外局部对比度。

### 4.4 V5 实际梯度再次证明瓶颈是 halo，不是目标保持

IRSTD V5：

- selected epoch = 0，与 V4-Stage1 完全相同；
- 训练后的所有评估点 mIoU 都低于 epoch 0；
- 最接近的 epoch 25 相对 epoch 0：TP `+29`、FN `−29`，但 FP `+70`，mIoU 仍下降 `0.185351 pp`；
- 所有点 Pd 都是 `228/230`，没有救回新目标。

这说明模型不是“不会增加前景”，而是增加的前景不够紧凑：每获得一些 TP，同时制造更多附着 FP。继续强化 peak preservation 会进一步保护错误轮廓。

### 4.5 单一内部验证 split 与 official 差异较大

V4/V5 内部 IRSTD `best_miou` 约 78.7%，而 official Current 约 66.0%。这表明单一 internal validation 对 official 域的代表性有限。下一轮不应继续用单一 split 选 epoch，而应在训练集内做固定的分层 K-fold OOF 选择，最后只训练一次全量 head。

---

## 5. 已实现版本：IRSTD-BGCR

### 5.1 结构图

```text
                                             ┌─ z_gt2
                                             ├─ z_gt3
input x ──> frozen Current ──> u1, z_out, d0 ├─ z_gt4
   │                                         └─ z_gt5
   │
   └─ fixed local-contrast pyramid: x, x-mean3, x-mean7, local_std5

[u1 projection + local contrast + detached readout statistics]
                         │
                         ▼
                  3-scale context trunk
                     │          │
                 core gate   halo gate
                     │          │
              positive arm   negative arm
                     └──── delta logits ────> z_out + delta ────> sigmoid
```

### 5.2 为什么删除 `q4`

BGCR 不把 `q4` 作为输入。原因不是否定 NER4，而是 IRSTD 专项修复需要像素级 core/ring 区分：

- `q4` 已服务于稳定主模型；
- 其空间尺度过粗，插值容易扩大激活；
- V4 已证明加入 `q4` 后仍不能解决附着 halo；
- 全分辨率 `u1` 和原图局部对比度更适合做边界修复。

### 5.3 双臂公式

令：

- `g_core(x) ∈ [0,1]`：目标核心门；
- `g_halo(x) ∈ [0,1]`：附着 halo 门；
- `s_pos(x), s_neg(x) ∈ [-1,1]`：两个零初始化残差信号；
- 正向上限 `A+ = 2.25`；
- 负向上限 `A− = 1.25`。

则：

```text
Δ+(x) = A+ · g_core(x) · s_pos(x)
Δ−(x) = A− · g_halo(x) · s_neg(x)
Δ(x)  = Δ+(x) − Δ−(x)
z'(x) = z_out(x) + Δ(x)
```

`A+=2.25` 不是性能门槛，而是表达容量。它覆盖已定位的 `−1.52051` 漏检峰值，同时正向修改只能通过 core gate 进入。两个 residual terminal 均精确零初始化，因此初始 `z' == z_out`。

### 5.4 参数和 state 合同

本报告所给实现经过独立 smoke test：

```text
new parameters        = 27,220
parameter tensors     = 29
persistent buffers    = 2
new state keys         = 31
Current inference keys = 564
integrated keys        = 595
integrated parameters  = 10,897,350
identity max_abs_diff  = 0.0
positive terminal grad = non-zero
negative terminal grad = non-zero
```

这只是独立语法/前向/反向检查；合入仓库后仍需执行完整 Current state、CUDA、数据和导出审计。

---

## 6. IRSTD 专项 error atlas

V4 的三类 atlas 扩展为：

| map | 定义 | 用途 |
|---|---|---|
| `target_component_ids` | 全部 GT 组件 ID | component peak、centroid |
| `rescue_component_ids` | Current 未匹配 GT | 自然漏检 |
| `core_target` | 每个 GT 的形态学核心 | 正向 core gate |
| `attached_halo` | 与 GT 已匹配的 prediction component 中，落在 GT 外的像素 | 解决 component-Fa 盲区 |
| `detached_false_positive` | 未匹配 prediction components | 独立 FP |
| `outer_ring` | GT 膨胀外圈 | halo 反事实注入 |
| `far_background` | 目标膨胀区之外 | 保持远背景 |
| `baseline_rescue` | Baseline 正确、Current 漏掉的 GT 像素 | 向 Baseline 学习互补 |
| `baseline_halo_advantage` | Current FP、Baseline 正确抑制的背景像素 | 向 Baseline 学习轮廓 |

其中最重要的新增项是 `attached_halo`。它直接优化 mIoU 中的真实 FP，而不是只优化 unmatched component-Fa。

---

## 7. 必须增加反事实错误生成

自然 atlas 只有 16 个 rescue component，无法支撑一个稳定 rescue 头。BGCR 对冻结 Current logit 做三种等概率输入：

```text
mode 0: clean
mode 1: core-drop       在 target core 内减 0.8–2.2 logit
mode 2: ring-injection  在 target outer ring 内加 0.5–1.5 logit
```

这不会改原图、GT 或 Current 权重。它把训练变成一个定向去噪任务：

- core-drop 强制学习“只在真实核心上抬高低峰”；
- ring-injection 强制学习“只在目标外圈上压低附着响应”；
- clean 保证真实分布不被合成错误完全取代。

与普通图像增强相比，logit 反事实直接作用于已定位的失败变量，且不会让冻结主干发生域漂移。

---

## 8. 损失函数

BGCR 只服务于 IRSTD `best_miou`。总损失：

```text
L = 1.00 L_BCE
  + 2.00 L_soft-IoU
  + 0.75 L_core-gate
  + 0.75 L_halo-gate
  + 1.50 L_component-peak
  + 0.50 L_centroid
  + 2.00 L_halo-probability
  + 0.50 L_far-background-no-increase
  + 0.25 L_direction
  + 0.01 L_neutral-delta
```

### 8.1 关键项

- `L_soft-IoU`：直接对齐 mIoU，不再让 BCE 的大量易背景像素主导。
- `L_component-peak`：每个 GT 等权，确保 tiny/弱目标不会因面积小而被稀释。
- `L_centroid`：在目标局部 ROI 内对齐软质心，与 `<3` 像素的目标匹配口径方向一致。
- `L_halo-probability`：专门压低 attached halo、detached FP 和 Baseline 优势背景。
- `L_far-background-no-increase`：只约束相对 Current 的正向背景变化，不阻碍背景下降。
- `L_direction`：要求 positive arm 在 core 上向上、negative arm 在 halo 上向下。

这里的 `1e-6` 仅是除法数值稳定项，与性能接受 margin 无关。

---

## 9. Baseline-guided 训练

### 9.1 严格绑定 Baseline

2026-08-07 已在用户指定的 `/home/ly/SCTransNet` 完成两个权重绑定：

```text
operational_best_path
  /home/ly/SCTransNet/checkpoints/IRSTD-1K/SCTransNet_best_mIoU.pth.tar
operational_best_epoch       713
operational_best_file_sha256 5f702bba036f43b62fc82d349b75344f9f6c04b2b68a143311a0b48050b3371b
operational_best_raw_state_semantic_sha256
  8d314d45f68de9b6747c5ada4ea7efc4f62a423a4bdf50db2f2d60bd8509d022
operational_best_normalized_state_semantic_sha256
  5ecf6f812f00e323ab5f8cec55d0ca86ea9f7db2225080bbc0ea947f44e181a4
operational_best_trainable   false
operational_best_teacher     false

formal_teacher_path
  /home/ly/SCTransNet/checkpoints/IRSTD-1K/SCTransNet_1000.pth.tar
formal_teacher_epoch         1000
formal_teacher_file_sha256   b4cb66be6e4a410dfd902ba050da82d0b666dd071bfb2c5477a7c3173ff07bc5
formal_teacher_raw_state_semantic_sha256
  972e7c15f8da8142da85112f535fb555a86293e12d7341d7c5be653fb4076d9b
formal_teacher_normalized_state_semantic_sha256
  1961ed8ee278fde09508145fe537324172599bfa704c181dc53f756578070b5c
formal_teacher_state_keys    510
formal_teacher_trainable     false
formal_teacher_enabled       true
```

`SCTransNet_best_mIoU.pth.tar` 与 `SCTransNet_713_best.pth.tar` 是同 inode 硬链接。
上表的 `raw_state_semantic_sha256` 对 checkpoint 内带精确 `model.` 前缀的
510-key mapping 计算；`normalized_state_semantic_sha256` 只在严格验证后去掉
这一前缀再计算。二者不是可互换的同一标签。
Baseline 的 `model/SCTransNet.py` 与当前仓库同名文件 SHA 均为
`5fb7ce711f190ead2bfcc910d2971266b2561e643c9f8a524d2032ffd48c0aeb`；510-key
state 已在同架构上 `strict=True` 加载通过。

### 9.2 利用 Baseline，而不在推理时依赖 Baseline

训练时构造：

```text
baseline_rescue = GT & (Baseline > 0) & (Current <= 0)
baseline_halo_advantage = ~GT & (Current > 0) & (Baseline <= 0)
```

这些区域只从固定 epoch-1000 teacher 计算，并且只在 checkpoint、
source lock 和 train-only logits manifest 全部验证通过时加入
core/halo supervision。epoch-713 best 不构建 teacher logits。
最终 BGCR 推理只需要 Current 和新 head，不需要同时运行 Baseline。

这比盲目追逐 `67.74%` 更有效：它把“Baseline 为什么更好”转为可学习的像素与组件差异。

---

## 10. 数据与选择协议

### 10.1 缓存冻结上下文

对冻结的 IRSTD official-train ID 投影执行一次冻结 Current，缓存：

```text
image
mask
u1
out
d0
gt2
gt3
gt4
gt5
Current component match
Baseline epoch-1000 logits
BGCR atlas maps
```

训练阶段从缓存裁 patch，避免每个 epoch 重跑 10.87M 参数主干，也从工程上保证主干绝不更新。
缓存中的 `u1/out/d0/gt2–gt5` 保持 FP32；缓存和集成推理在同一
全图输入上必须逐位一致。不得将 `Current(full_image)` 的特征裁块
声称为 `Current(image_patch)` 的结果：两者因 padding 和全局 QFG/Transformer
不等价。本协议的训练对象是**全图冻结 context 上的 repair head**：
裁取 `272×272` 外层 context patch，仅对中心 `256×256` 计算损失。
这里的 halo 合同固定为每边 **8 px**：`272 = 8 + 256 + 8`，loss
只能读取 `[8:264, 8:264]` 中心区域。BGCR head 的最大空间感受半径
必须等于 8 px；不得通过新增卷积、空间归一化或全局池化暗中扩大。
单元测试必须证明同一冻结全图 context 上，`272×272` crop 的中心
`256×256` 输出与 full-context head 前向逐点一致。

普通 `nn.GroupNorm` 会同时在组内通道和 `H/W` 上统计均值、方差，因而
即使卷积已有 8 px halo，`272×272` crop 与 `512×512` 全图的中心归一化
结果仍不相同。正式 head 必须使用逐像素 `LocalGroupNorm2d`：每个空间
位置只在本位置的组内通道上归一化，绝不跨 `H/W`；这样 8 px halo
才足以保证中心等价，同时保持每通道 affine 参数布局不变。

### 10.2 error-aware crop

每个 batch 的中心来源固定为三类等概率：

```text
1/3 target core or rescue component
1/3 attached halo / baseline-halo region
1/3 random image location
```

这不是结果驱动采样，也不扫描比例。它只是防止 148 个 rescue 像素被 48,170 个 preserve 像素淹没。

### 10.3 固定 K-fold OOF

不再依赖单一 internal validation：

1. 按图像级目标数量、目标面积和空背景标记，对已冻结的
   official-train ID 投影固定分层 3-fold；不解析任何 test index；
2. 每 fold 训练 BGCR，记录每 5 epoch 的 validation mIoU；
3. 对三个互斥 validation fold 的充分统计做一次精确聚合，选择
   **整个 OOF 投影上 role key 最大**的 epoch；不对三个已舍入
   mIoU 做算术平均；
4. 排序与已冻结 V4/V5 `best_miou` 一致：精确 mIoU、精确 Pd、
   精确 `-Fa`、nIoU、tiny-Pd、`-loss`、较早 epoch；F1 完整报告但
   不插入已冻结 role key；
5. 用确定的 epoch 数在完整 train 上训练一个最终 head；
6. 冻结代码、state 和 manifest。本轮最终 head 只是 internal candidate，
   不再访问已消耗的 official test。

每个 fold summary 还必须携带同一 validation fold 上的
`baseline1000_metric_row`，身份固定为 `Baseline-epoch1000`。selector 用与
BGCR 完全相同的充分统计协议精确汇总为
`baseline1000_internal_oof_metrics`；该字段只表示 train-only OOF 内部参考，
不是 official 结果，也不得与独立训练 Baseline 的 official 数值直接相减。

Current 主模型在其原训练期已见过完整 official-train；OOF 只检验新增
27.2K repair head 的外推，不得将其描述为主模型从未见过的独立验证。

### 10.4 零门槛排序

```python
def irstd_role_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        Fraction(int(row["intersection_pixels"]), int(row["union_pixels"])),
        Fraction(int(row["matched_target_count"]), int(row["target_count"])),
        -Fraction(int(row["unmatched_component_pixels"]), int(row["valid_pixel_count"])),
        float(row["nIoU"]),
        Fraction(int(row["matched_tiny_target_count"]), int(row["tiny_target_count"])),
        -float(row["loss"]),
        -float(row["epoch"]),
    )
```

候选池始终包含 epoch 0 identity。没有 epsilon、最小百分点、百分比或全指标否决项。任意严格更高的 mIoU 都是提升。

---

## 11. 训练配置

推荐固定一次，不做权重 sweep：

```text
Dataset            IRSTD-1K only
Parent             frozen Current best_miou
Trainable          irstd_repair.* only
Precision          FP32, TF32 off
Seed               42
Epoch              120
Validation         every 5 epochs
Batch size         16 cached context patches
Optimizer          AdamW
LR                 3e-4
Weight decay       1e-4
Scheduler          cosine to 1e-6
Gradient clipping  1.0
Counterfactual     deterministic balanced clean/core-drop/ring-injection
Teacher            fixed independent Baseline epoch 1000 only
Internal threshold probability > 0.5
Performance margin none
Official test      prohibited in this run
```

主模型保持 `eval()`，BGCR head 单独 `train()`。optimizer 必须只接收 `model.irstd_repair.parameters()`。

---

## 12. 代码修改清单

```text
NEW  model/irstd_core_ring_repair.py
NEW  model/tpd8_ner4_qfg2_irstd_crr.py
NEW  experiments/irstd_error_atlas.py
NEW  experiments/irstd_logit_counterfactual.py
NEW  experiments/irstd_core_ring_loss.py
NEW  experiments/irstd_baseline_teacher.py
NEW  experiments/cache_irstd_frozen_context_v1.py
NEW  experiments/train_irstd_bgcr_v1.py
NEW  experiments/irstd_bgcr_run_contract.py
NEW  experiments/select_irstd_bgcr_oof_v1.py
NEW  tests/test_irstd_bgcr_core.py
NEW  tests/test_irstd_bgcr_model_contract.py
NEW  tests/test_irstd_bgcr_run_contract.py
NEW  tests/test_irstd_bgcr_cache.py
NEW  tests/test_select_irstd_bgcr_oof_v1.py
NEW  tests/test_irstd_bgcr_pipeline.py

UNCHANGED
     model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py
     all TPD/NER/QFG implementation files
     Current checkpoint
```

其中 `tests/test_irstd_bgcr_run_contract.py` 同时覆盖独立 Baseline teacher
的 epoch-713 禁用与 epoch-1000 唯一 teacher 合同；cache、OOF selector
分别使用独立测试文件，避免把 train-only、append-only 与 official 禁止
边界仅留在综合 smoke 中。

---

## 13. 完整核心代码

### 13.1 `model/irstd_core_ring_repair.py`

```python
"""IRSTD-only core/ring repair head for a frozen SCTransNet Current model.

The parent TPD8+NER4+QFG2/TSS-off network is never updated.  This module reads
frozen high-resolution decoder features, frozen readout logits and fixed local
contrast features from the input image.  Its two terminal residual heads are
exactly zero-initialized, so construction is an exact identity mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F


IRSTD_CRR_VERSION: Final[str] = "irstd_core_ring_repair_v1"
LOCAL_CHANNELS: Final[int] = 32
HIDDEN_CHANNELS: Final[int] = 32
POSITIVE_LOGIT_LIMIT: Final[float] = 2.25
NEGATIVE_LOGIT_LIMIT: Final[float] = 1.25


@dataclass(frozen=True, slots=True)
class IRSTDCoreRingRepairOutput:
    routed_logits: torch.Tensor
    delta_logits: torch.Tensor
    positive_delta: torch.Tensor
    negative_delta: torch.Tensor
    core_gate_logits: torch.Tensor
    halo_gate_logits: torch.Tensor
    core_gate: torch.Tensor
    halo_gate: torch.Tensor


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


class LocalGroupNorm2d(nn.Module):
    """逐像素组归一化：只沿组内通道聚合，绝不跨 H/W。"""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        *,
        eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if (
            num_groups < 1
            or num_channels < 1
            or num_channels % num_groups != 0
            or num_channels // num_groups < 2
        ):
            raise ValueError("invalid local group-normalization dimensions")
        self.num_groups = int(num_groups)
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = value.shape
        grouped = value.reshape(
            batch,
            self.num_groups,
            channels // self.num_groups,
            height,
            width,
        )
        working = grouped.float()
        mean = working.mean(dim=2, keepdim=True)
        variance = working.var(dim=2, keepdim=True, unbiased=False)
        normalized = (working - mean) * torch.rsqrt(variance + self.eps)
        normalized = normalized.reshape(batch, channels, height, width).to(
            dtype=value.dtype
        )
        return (
            normalized * self.weight.to(value.dtype).view(1, -1, 1, 1)
            + self.bias.to(value.dtype).view(1, -1, 1, 1)
        )


def _conv_norm_act(
    in_channels: int,
    out_channels: int,
    *,
    kernel_size: int = 3,
    dilation: int = 1,
) -> nn.Sequential:
    padding = dilation * (kernel_size // 2)
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        LocalGroupNorm2d(_group_count(out_channels), out_channels),
        nn.GELU(),
    )


class IRSTDCoreRingRepairHead(nn.Module):
    """Compact IRSTD specialization head with exact Current initialization.

    Positive capacity is deliberately larger than PBDR-V4 best-mIoU's +0.60:
    the localized IRSTD misses include peaks near -1.20 and -1.52 logits, so a
    core-only branch needs enough representational range to cross zero.  The
    large range is safe only because it is multiplied by a learned core gate
    trained from target-core and counterfactual rescue supervision.
    """

    def __init__(
        self,
        *,
        local_channels: int = LOCAL_CHANNELS,
        hidden_channels: int = HIDDEN_CHANNELS,
        positive_limit: float = POSITIVE_LOGIT_LIMIT,
        negative_limit: float = NEGATIVE_LOGIT_LIMIT,
        detach_context: bool = True,
    ) -> None:
        super().__init__()
        if local_channels < 1 or hidden_channels < 8:
            raise ValueError("invalid IRSTD repair channel configuration")
        if not math.isfinite(positive_limit) or positive_limit <= 0.0:
            raise ValueError("positive_limit must be finite and positive")
        if not math.isfinite(negative_limit) or negative_limit <= 0.0:
            raise ValueError("negative_limit must be finite and positive")

        self.local_channels = int(local_channels)
        self.hidden_channels = int(hidden_channels)
        self.detach_context = bool(detach_context)
        self.register_buffer(
            "positive_limit",
            torch.tensor(float(positive_limit), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "negative_limit",
            torch.tensor(float(negative_limit), dtype=torch.float32),
            persistent=True,
        )

        self.local_projection = _conv_norm_act(self.local_channels, 16, kernel_size=1)
        self.contrast_projection = _conv_norm_act(4, 8, kernel_size=3)

        # 13 scalar maps: p_out, p_d0, four auxiliary probabilities, their
        # mean/max/min/std, uncertainty, support gap and spread.
        scalar_channels = 13
        context_channels = 16 + 8 + scalar_channels
        self.context_stem = _conv_norm_act(
            context_channels,
            self.hidden_channels,
            kernel_size=3,
        )
        branch_channels = self.hidden_channels // 2
        self.context_branches = nn.ModuleList(
            [
                _conv_norm_act(
                    self.hidden_channels,
                    branch_channels,
                    kernel_size=3,
                    dilation=dilation,
                )
                for dilation in (1, 2, 3)
            ]
        )
        self.context_fuse = _conv_norm_act(
            branch_channels * 3,
            self.hidden_channels,
            kernel_size=1,
        )

        self.core_gate_head = nn.Conv2d(self.hidden_channels, 1, kernel_size=1)
        self.halo_gate_head = nn.Conv2d(self.hidden_channels, 1, kernel_size=1)
        self.positive_residual_head = nn.Conv2d(
            self.hidden_channels, 1, kernel_size=1
        )
        self.negative_residual_head = nn.Conv2d(
            self.hidden_channels, 1, kernel_size=1
        )

        # Sparse prior for both semantic gates.  Exact identity comes from the
        # two zero terminal residual heads, not from saturating a gate.
        nn.init.normal_(self.core_gate_head.weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(self.core_gate_head.bias, -1.5)
        nn.init.normal_(self.halo_gate_head.weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(self.halo_gate_head.bias, -1.5)
        nn.init.zeros_(self.positive_residual_head.weight)
        nn.init.zeros_(self.positive_residual_head.bias)
        nn.init.zeros_(self.negative_residual_head.weight)
        nn.init.zeros_(self.negative_residual_head.bias)

    @staticmethod
    def _validate_one_channel(
        value: torch.Tensor,
        *,
        name: str,
        reference: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.ndim != 4 or value.shape[1] != 1 or min(value.shape) < 1:
            raise ValueError(f"{name} must be non-empty BCHW with C=1")
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if reference is not None:
            if value.shape != reference.shape:
                raise ValueError(f"{name} must match z_out shape")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"{name} must match z_out device/dtype")

    def _validate_inputs(
        self,
        *,
        image: torch.Tensor,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        z_gt2: torch.Tensor,
        z_gt3: torch.Tensor,
        z_gt4: torch.Tensor,
        z_gt5: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> None:
        self._validate_one_channel(z_out, name="z_out")
        for name, value in (
            ("image", image),
            ("z_d0", z_d0),
            ("z_gt2", z_gt2),
            ("z_gt3", z_gt3),
            ("z_gt4", z_gt4),
            ("z_gt5", z_gt5),
        ):
            self._validate_one_channel(value, name=name, reference=z_out)
        if local_feature.ndim != 4 or local_feature.shape[1] != self.local_channels:
            raise ValueError(
                f"local_feature must be BCHW with C={self.local_channels}"
            )
        if local_feature.shape[0] != z_out.shape[0] or local_feature.shape[-2:] != z_out.shape[-2:]:
            raise ValueError("local_feature must match z_out batch/spatial shape")
        if local_feature.device != z_out.device or local_feature.dtype != z_out.dtype:
            raise ValueError("local_feature must match z_out device/dtype")

    @staticmethod
    def _local_contrast(image: torch.Tensor) -> torch.Tensor:
        mean3 = F.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
        mean7 = F.avg_pool2d(image, kernel_size=7, stride=1, padding=3)
        mean_sq5 = F.avg_pool2d(image.square(), kernel_size=5, stride=1, padding=2)
        mean5 = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
        std5 = (mean_sq5 - mean5.square()).clamp_min(0.0).sqrt()
        return torch.cat((image, image - mean3, image - mean7, std5), dim=1)

    def forward_with_diagnostics(
        self,
        *,
        image: torch.Tensor,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        z_gt2: torch.Tensor,
        z_gt3: torch.Tensor,
        z_gt4: torch.Tensor,
        z_gt5: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> IRSTDCoreRingRepairOutput:
        self._validate_inputs(
            image=image,
            z_out=z_out,
            z_d0=z_d0,
            z_gt2=z_gt2,
            z_gt3=z_gt3,
            z_gt4=z_gt4,
            z_gt5=z_gt5,
            local_feature=local_feature,
        )

        if self.detach_context:
            image_context = image.detach()
            local_context_input = local_feature.detach()
            readouts = tuple(
                value.detach()
                for value in (z_out, z_d0, z_gt2, z_gt3, z_gt4, z_gt5)
            )
        else:
            image_context = image
            local_context_input = local_feature
            readouts = (z_out, z_d0, z_gt2, z_gt3, z_gt4, z_gt5)

        out_ctx, d0_ctx, gt2_ctx, gt3_ctx, gt4_ctx, gt5_ctx = readouts
        p_out = torch.sigmoid(out_ctx)
        p_d0 = torch.sigmoid(d0_ctx)
        aux = torch.cat(
            tuple(torch.sigmoid(value) for value in (gt2_ctx, gt3_ctx, gt4_ctx, gt5_ctx)),
            dim=1,
        )
        aux_mean = aux.mean(dim=1, keepdim=True)
        aux_max = aux.amax(dim=1, keepdim=True)
        aux_min = aux.amin(dim=1, keepdim=True)
        aux_std = aux.std(dim=1, keepdim=True, unbiased=False)
        uncertainty = 4.0 * p_out * (1.0 - p_out)
        support_gap = aux_mean - p_out
        spread = aux_max - aux_min
        scalar_context = torch.cat(
            (
                p_out,
                p_d0,
                aux,
                aux_mean,
                aux_max,
                aux_min,
                aux_std,
                uncertainty,
                support_gap,
                spread,
            ),
            dim=1,
        )

        local = self.local_projection(local_context_input)
        contrast = self.contrast_projection(self._local_contrast(image_context))
        stem = self.context_stem(torch.cat((local, contrast, scalar_context), dim=1))
        multi_scale = torch.cat(
            tuple(branch(stem) for branch in self.context_branches), dim=1
        )
        fused = self.context_fuse(multi_scale)

        core_gate_logits = self.core_gate_head(fused)
        halo_gate_logits = self.halo_gate_head(fused)
        core_gate = torch.sigmoid(core_gate_logits)
        halo_gate = torch.sigmoid(halo_gate_logits)

        # Two separately supervised residual arms.  Their terminal projections
        # are exact zero at initialization.  Directional loss prevents the
        # positive arm from becoming negative and the negative arm from becoming
        # positive after optimization.
        positive_signal = torch.tanh(self.positive_residual_head(fused))
        negative_signal = torch.tanh(self.negative_residual_head(fused))
        positive_delta = self.positive_limit.to(z_out.dtype) * core_gate * positive_signal
        negative_delta = self.negative_limit.to(z_out.dtype) * halo_gate * negative_signal
        delta = positive_delta - negative_delta
        routed = z_out + delta

        return IRSTDCoreRingRepairOutput(
            routed_logits=routed,
            delta_logits=delta,
            positive_delta=positive_delta,
            negative_delta=negative_delta,
            core_gate_logits=core_gate_logits,
            halo_gate_logits=halo_gate_logits,
            core_gate=core_gate,
            halo_gate=halo_gate,
        )

    def forward(
        self,
        *,
        image: torch.Tensor,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        z_gt2: torch.Tensor,
        z_gt3: torch.Tensor,
        z_gt4: torch.Tensor,
        z_gt5: torch.Tensor,
        local_feature: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_with_diagnostics(
            image=image,
            z_out=z_out,
            z_d0=z_d0,
            z_gt2=z_gt2,
            z_gt3=z_gt3,
            z_gt4=z_gt4,
            z_gt5=z_gt5,
            local_feature=local_feature,
        ).routed_logits


__all__ = [
    "IRSTD_CRR_VERSION",
    "IRSTDCoreRingRepairHead",
    "IRSTDCoreRingRepairOutput",
    "LOCAL_CHANNELS",
    "NEGATIVE_LOGIT_LIMIT",
    "POSITIVE_LOGIT_LIMIT",
]

```

### 13.2 `model/tpd8_ner4_qfg2_irstd_crr.py`

```python
"""IRSTD-only CRR extension of the frozen TPD8+NER4+QFG2/TSS-off graph.

This file is intentionally separate from the production Current model.  It
copies Current's forward ordering so the original implementation and checkpoint
remain immutable.  Only ``irstd_repair.*`` parameters are trainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SCTransNet import SCTransNet
from model.tpd_ner_v8_mprs_dch import (
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    DEFAULT_DC_SUPPORT_MODE,
    DEFAULT_TAIL_Z_THRESHOLDS,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    FORMAL_SURVIVAL_VARIANT,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
)
from model.tpd_query_frequency_bridge import frequency_encoder_forward
from model.irstd_core_ring_repair import (
    IRSTDCoreRingRepairHead,
    IRSTDCoreRingRepairOutput,
)


IRSTD_CRR_STATE_PREFIX = "irstd_repair."


@dataclass(frozen=True, slots=True)
class FrozenIRSTDContext:
    local_feature: torch.Tensor
    out_logits: torch.Tensor
    d0_logits: torch.Tensor
    gt2_logits: torch.Tensor
    gt3_logits: torch.Tensor
    gt4_logits: torch.Tensor
    gt5_logits: torch.Tensor


class TPD8NER4QFG2IRSTDCRRInferenceSCTransNet(
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet
):
    """Frozen Current plus one dataset-specific identity-initialized repair head."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        variant: str = FORMAL_SURVIVAL_VARIANT,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode: str = DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds: Mapping[int, float] = DEFAULT_TAIL_Z_THRESHOLDS,
        repair_initialization_seed: int = 42,
    ) -> None:
        super().__init__(
            parent,
            variant=variant,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
            dc_support_mode=dc_support_mode,
            tail_z_thresholds=tail_z_thresholds,
        )
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(repair_initialization_seed)
            repair = IRSTDCoreRingRepairHead(
                local_channels=int(self.outc.in_channels),
            )
        reference = next(self.parameters())
        repair.to(device=reference.device, dtype=reference.dtype)
        self.irstd_repair = repair
        self.freeze_current()

    def freeze_current(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(IRSTD_CRR_STATE_PREFIX))
        # The parent remains in evaluation mode even while the repair head trains.
        super().train(False)
        self.irstd_repair.train(True)

    def train(self, mode: bool = True):
        super().train(False)
        self.irstd_repair.train(mode)
        return self

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        parameters = tuple(self.irstd_repair.parameters())
        if not parameters or not all(parameter.requires_grad for parameter in parameters):
            raise RuntimeError("IRSTD repair parameter contract is broken")
        if any(
            parameter.requires_grad
            for name, parameter in self.named_parameters()
            if not name.startswith(IRSTD_CRR_STATE_PREFIX)
        ):
            raise RuntimeError("a frozen Current parameter became trainable")
        return parameters

    def _frozen_current_context(self, x: torch.Tensor) -> FrozenIRSTDContext:
        # Exact ordering copied from Current's _forward_with_relay.  No formula,
        # tensor, coefficient or normalization in Current is changed.
        x1 = self.inc(x)
        x2 = self.down_encoder1(self.pool(x1))
        x3 = self.down_encoder2(self.pool(x2))
        x4 = self.down_encoder3(self.pool(x3))
        d5 = self.down_encoder4(self.pool(x4))
        f1, f2, f3, f4 = x1, x2, x3, x4
        emb1, emb2, emb3, emb4, evidence1, evidence2 = self.explicit_embeddings(
            x1, x2, x3, x4
        )
        h11, h12, h13 = evidence1
        h21, h22 = evidence2
        prepared_qfg = self.tpd_qfg.prepare(
            (x1, x2, x3, x4),
            tuple(tuple(embedding.shape[-2:]) for embedding in (emb1, emb2, emb3, emb4)),
        )
        encoded1, encoded2, encoded3, encoded4, _ = frequency_encoder_forward(
            self.mtc.encoder,
            emb1,
            emb2,
            emb3,
            emb4,
            self.tpd_qfg,
            prepared_qfg,
        )
        x1 = self.mtc.reconstruct_1(encoded1) + f1
        x2 = self.mtc.reconstruct_2(encoded2) + f2
        x3 = self.mtc.reconstruct_3(encoded3) + f3
        x4 = self.mtc.reconstruct_4(encoded4) + f4
        x1, x2, x3, x4 = x1 + f1, x2 + f2, x3 + f3, x4 + f4

        up4, skip4 = self.up_decoder4.prepare(d5, x4)
        q4, mask4 = self.tpd_ner.forward_stage(
            4, (h13, h22, up4), tuple(up4.shape[-2:])
        )
        d4 = self.up_decoder4.finish(up4, skip4, mask4)
        up3, skip3 = self.up_decoder3.prepare(d4, x3)
        q3, mask3 = self.tpd_ner.forward_stage(
            3, (h12, h21, q4, up3), tuple(up3.shape[-2:])
        )
        d3 = self.up_decoder3.finish(up3, skip3, mask3)
        up2, skip2 = self.up_decoder2.prepare(d3, x2)
        _, mask2 = self.tpd_ner.forward_stage(
            2, (h11, q3, up2), tuple(up2.shape[-2:])
        )
        d2 = self.up_decoder2.finish(up2, skip2, mask2)
        u1 = self.up_decoder1(d2, x1)
        out = self.outc(u1)

        gt5 = F.interpolate(
            self.gt_conv5(d5), scale_factor=16, mode="bilinear", align_corners=True
        )
        gt4 = F.interpolate(
            self.gt_conv4(d4), scale_factor=8, mode="bilinear", align_corners=True
        )
        gt3 = F.interpolate(
            self.gt_conv3(d3), scale_factor=4, mode="bilinear", align_corners=True
        )
        gt2 = F.interpolate(
            self.gt_conv2(d2), scale_factor=2, mode="bilinear", align_corners=True
        )
        d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), dim=1))
        return FrozenIRSTDContext(
            local_feature=u1.detach(),
            out_logits=out.detach(),
            d0_logits=d0.detach(),
            gt2_logits=gt2.detach(),
            gt3_logits=gt3.detach(),
            gt4_logits=gt4.detach(),
            gt5_logits=gt5.detach(),
        )

    def forward_for_irstd_training(
        self,
        x: torch.Tensor,
        *,
        base_logits_override: torch.Tensor | None = None,
    ) -> tuple[IRSTDCoreRingRepairOutput, FrozenIRSTDContext]:
        with torch.no_grad():
            context = self._frozen_current_context(x)
        base = context.out_logits if base_logits_override is None else base_logits_override
        routing = self.irstd_repair.forward_with_diagnostics(
            image=x.detach(),
            z_out=base,
            z_d0=context.d0_logits,
            z_gt2=context.gt2_logits,
            z_gt3=context.gt3_logits,
            z_gt4=context.gt4_logits,
            z_gt5=context.gt5_logits,
            local_feature=context.local_feature,
        )
        return routing, context

    def _forward_with_relay(self, x: torch.Tensor):
        routing, _ = self.forward_for_irstd_training(x)
        return torch.sigmoid(routing.routed_logits)


def load_current_into_frozen_base_strictly(
    model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
    current_state: Mapping[str, torch.Tensor],
) -> dict[str, int]:
    """Load exactly the Current keys while retaining identity BGCR state."""
    if not isinstance(current_state, Mapping):
        raise TypeError("current_state must be a state mapping")
    integrated_state = model.state_dict()
    base_keys = tuple(
        key for key in integrated_state if not key.startswith(IRSTD_CRR_STATE_PREFIX)
    )
    repair_keys = tuple(
        key for key in integrated_state if key.startswith(IRSTD_CRR_STATE_PREFIX)
    )
    if set(current_state) != set(base_keys):
        missing = sorted(set(base_keys) - set(current_state))
        unexpected = sorted(set(current_state) - set(base_keys))
        raise RuntimeError(
            f"Current state contract differs: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    merged = dict(integrated_state)
    for key in base_keys:
        value = current_state[key]
        if value.shape != integrated_state[key].shape:
            raise RuntimeError(f"Current tensor shape differs for {key}")
        merged[key] = value.detach().clone()
    model.load_state_dict(merged, strict=True)
    loaded = model.state_dict()
    for key in base_keys:
        if not torch.equal(loaded[key].detach().cpu(), current_state[key].detach().cpu()):
            raise RuntimeError(f"Current tensor changed while loading {key}")
    model.freeze_current()
    return {
        "current_keys": len(base_keys),
        "repair_keys": len(repair_keys),
        "integrated_keys": len(loaded),
    }


__all__ = [
    "FrozenIRSTDContext",
    "IRSTD_CRR_STATE_PREFIX",
    "TPD8NER4QFG2IRSTDCRRInferenceSCTransNet",
    "load_current_into_frozen_base_strictly",
]

```

### 13.3 `experiments/irstd_error_atlas.py`

```python
"""Build IRSTD-specific target-core and attached-halo supervision maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from experiments.component_matching_v2 import match_components_v2


@dataclass(frozen=True, slots=True)
class IRSTDErrorAtlas:
    target_component_ids: np.ndarray
    rescue_component_ids: np.ndarray
    core_target: np.ndarray
    attached_halo: np.ndarray
    detached_false_positive: np.ndarray
    outer_ring: np.ndarray
    halo_target: np.ndarray
    far_background: np.ndarray
    baseline_rescue: np.ndarray
    baseline_halo_advantage: np.ndarray


def _component_core(component: np.ndarray) -> np.ndarray:
    area = int(component.sum())
    if area <= 4:
        return component.copy()
    distance = ndimage.distance_transform_edt(component)
    maximum = float(distance.max(initial=0.0))
    core = component & (distance >= max(1.0, 0.5 * maximum))
    if not bool(core.any()):
        flat_index = int(np.argmax(distance))
        core.flat[flat_index] = True
    return core


def build_irstd_error_atlas(
    *,
    current_logits: np.ndarray,
    target_mask: np.ndarray,
    ring_radius: int = 3,
    far_background_radius: int = 7,
    baseline_logits: np.ndarray | None = None,
) -> IRSTDErrorAtlas:
    """Construct maps that V4/V5 did not represent explicitly.

    ``attached_halo`` includes every prediction pixel outside the GT mask that
    belongs to a prediction component already matched to a target.  This is the
    topology class hidden from component-Fa and from V4's unmatched-component
    suppress atlas.
    """
    logits = np.asarray(current_logits, dtype=np.float32)
    target = np.asarray(target_mask, dtype=np.bool_)
    if logits.ndim != 2 or target.ndim != 2 or logits.shape != target.shape:
        raise ValueError("current_logits and target_mask must be aligned 2D arrays")
    if ring_radius < 1 or far_background_radius <= ring_radius:
        raise ValueError("invalid morphology radii")

    prediction = logits > 0.0
    if baseline_logits is None:
        baseline = None
    else:
        baseline = np.asarray(baseline_logits, dtype=np.float32)
        if baseline.shape != logits.shape:
            raise ValueError("baseline_logits must match current_logits")
    match = match_components_v2(
        prediction_mask=prediction,
        target_mask=target,
    )
    target_ids = np.asarray(match.target_id_map, dtype=np.int32)
    prediction_ids = np.asarray(match.prediction_id_map, dtype=np.int32)

    core = np.zeros_like(target, dtype=np.bool_)
    for component_id in np.unique(target_ids):
        if component_id <= 0:
            continue
        component = target_ids == component_id
        core |= _component_core(component)

    rescue_ids = np.where(
        np.isin(target_ids, np.asarray(match.unmatched_target_ids, dtype=np.int32)),
        target_ids,
        0,
    ).astype(np.int32, copy=False)

    attached_halo = np.zeros_like(target, dtype=np.bool_)
    for pair in match.matches:
        prediction_component = prediction_ids == int(pair.prediction_id)
        attached_halo |= prediction_component & ~target

    detached_false_positive = np.isin(
        prediction_ids,
        np.asarray(match.unmatched_prediction_ids, dtype=np.int32),
    )
    ring_structure = ndimage.generate_binary_structure(2, 2)
    outer_ring = ndimage.binary_dilation(
        target,
        structure=ring_structure,
        iterations=ring_radius,
    ) & ~target
    far_background = ~ndimage.binary_dilation(
        target,
        structure=ring_structure,
        iterations=far_background_radius,
    )

    if baseline is None:
        baseline_rescue = np.zeros_like(target, dtype=np.bool_)
        baseline_halo_advantage = np.zeros_like(target, dtype=np.bool_)
    else:
        baseline_prediction = baseline > 0.0
        baseline_rescue = target & baseline_prediction & ~prediction
        baseline_halo_advantage = ~target & prediction & ~baseline_prediction

    # The supervised negative class contains observed topology errors and, when
    # available, Current false-positive pixels already corrected by the bound
    # independent Baseline teacher.  Pure outer-ring pixels are supplied by the
    # counterfactual generator rather than marked negative in every clean image.
    halo_target = attached_halo | detached_false_positive | baseline_halo_advantage

    return IRSTDErrorAtlas(
        target_component_ids=np.ascontiguousarray(target_ids),
        rescue_component_ids=np.ascontiguousarray(rescue_ids),
        core_target=np.ascontiguousarray(core),
        attached_halo=np.ascontiguousarray(attached_halo),
        detached_false_positive=np.ascontiguousarray(detached_false_positive),
        outer_ring=np.ascontiguousarray(outer_ring),
        halo_target=np.ascontiguousarray(halo_target),
        far_background=np.ascontiguousarray(far_background),
        baseline_rescue=np.ascontiguousarray(baseline_rescue),
        baseline_halo_advantage=np.ascontiguousarray(baseline_halo_advantage),
    )


__all__ = ["IRSTDErrorAtlas", "build_irstd_error_atlas"]

```

### 13.4 `experiments/irstd_logit_counterfactual.py`

```python
"""Counterfactual logit corruption for scarce IRSTD rescue/halo examples."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CounterfactualBatch:
    logits: torch.Tensor
    halo_target: torch.Tensor
    mode: torch.Tensor


def corrupt_irstd_logits(
    *,
    current_logits: torch.Tensor,
    core_target: torch.Tensor,
    outer_ring: torch.Tensor,
    observed_halo_target: torch.Tensor,
    generator: torch.Generator,
) -> CounterfactualBatch:
    """Mix clean, core-drop and ring-injection samples within one batch.

    Modes are sampled per image: 0=clean, 1=core attenuation, 2=attached-ring
    injection.  The parent network stays frozen; only its detached output logit
    is corrupted, turning rare error types into dense supervised examples.
    """
    if current_logits.ndim != 4 or current_logits.shape[1] != 1:
        raise ValueError("current_logits must be BCHW with C=1")
    for name, value in (
        ("core_target", core_target),
        ("outer_ring", outer_ring),
        ("observed_halo_target", observed_halo_target),
    ):
        if value.shape != current_logits.shape:
            raise ValueError(f"{name} must match current_logits")

    batch = current_logits.shape[0]
    device = current_logits.device
    dtype = current_logits.dtype
    mode = torch.randint(0, 3, (batch,), generator=generator, device=device)
    drop_scale = torch.empty(batch, 1, 1, 1, device=device, dtype=dtype).uniform_(
        0.8, 2.2, generator=generator
    )
    ring_scale = torch.empty(batch, 1, 1, 1, device=device, dtype=dtype).uniform_(
        0.5, 1.5, generator=generator
    )
    core_mode = (mode == 1).view(batch, 1, 1, 1).to(dtype)
    ring_mode = (mode == 2).view(batch, 1, 1, 1).to(dtype)

    corrupted = current_logits.detach().clone()
    corrupted = corrupted - core_mode * drop_scale * core_target.to(dtype)
    corrupted = corrupted + ring_mode * ring_scale * outer_ring.to(dtype)
    halo_target = torch.maximum(
        observed_halo_target.to(dtype),
        ring_mode * outer_ring.to(dtype),
    )
    return CounterfactualBatch(logits=corrupted, halo_target=halo_target, mode=mode)


__all__ = ["CounterfactualBatch", "corrupt_irstd_logits"]

```

### 13.5 `experiments/irstd_core_ring_loss.py`

```python
"""IRSTD-only objective for the frozen-main core/ring repair head."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn.functional as F


IRSTD_CRR_LOSS_VERSION = "irstd_core_ring_loss_v1"
LOSS_WEIGHTS: Mapping[str, float] = {
    "bce": 1.0,
    "soft_iou": 2.0,
    "core_gate": 0.75,
    "halo_gate": 0.75,
    "component_peak": 1.5,
    "centroid": 0.5,
    "halo_probability": 2.0,
    "far_background_no_increase": 0.5,
    "direction": 0.25,
    "neutral_delta": 0.01,
}


@dataclass(frozen=True, slots=True)
class IRSTDCoreRingLossOutput:
    total: torch.Tensor
    bce: torch.Tensor
    soft_iou: torch.Tensor
    core_gate: torch.Tensor
    halo_gate: torch.Tensor
    component_peak: torch.Tensor
    centroid: torch.Tensor
    halo_probability: torch.Tensor
    far_background_no_increase: torch.Tensor
    direction: torch.Tensor
    neutral_delta: torch.Tensor

    def detached_scalars(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu().item())
            for name in self.__dataclass_fields__
        }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=value.dtype)
    return (value * weights).sum() / weights.sum().clamp_min(1.0)


def _balanced_binary_logit_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    positive = positive_mask.bool()
    negative = ~positive
    terms: list[torch.Tensor] = []
    if bool(positive.any()):
        terms.append(F.softplus(-logits[positive]).mean())
    if bool(negative.any()):
        terms.append(F.softplus(logits[negative]).mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def _soft_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    target_float = target.float()
    reduce_dims = tuple(range(1, probability.ndim))
    intersection = (probability * target_float).sum(dim=reduce_dims)
    union = (
        probability + target_float - probability * target_float
    ).sum(dim=reduce_dims)
    # Numerical stabilizer only; this is not a performance acceptance margin.
    score = (intersection + 1.0e-6) / (union + 1.0e-6)
    return 1.0 - score.mean()


def _component_peak_loss(
    logits: torch.Tensor,
    target_component_ids: torch.Tensor,
    *,
    temperature: float = 0.25,
) -> torch.Tensor:
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    losses: list[torch.Tensor] = []
    for batch_index in range(logits.shape[0]):
        component_ids = torch.unique(target_component_ids[batch_index])
        component_ids = component_ids[component_ids > 0]
        for component_id in component_ids:
            mask = target_component_ids[batch_index] == component_id
            values = logits[batch_index][mask]
            smooth_peak = temperature * (
                torch.logsumexp(values / temperature, dim=0)
                - torch.log(values.new_tensor(float(values.numel())))
            )
            losses.append(F.softplus(-smooth_peak))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _component_centroid_loss(
    logits: torch.Tensor,
    target_component_ids: torch.Tensor,
    *,
    roi_radius: int = 4,
) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    height, width = probability.shape[-2:]
    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=probability.device, dtype=probability.dtype),
        torch.arange(width, device=probability.device, dtype=probability.dtype),
        indexing="ij",
    )
    losses: list[torch.Tensor] = []
    kernel_size = 2 * roi_radius + 1
    for batch_index in range(probability.shape[0]):
        component_ids = torch.unique(target_component_ids[batch_index])
        component_ids = component_ids[component_ids > 0]
        for component_id in component_ids:
            component = (target_component_ids[batch_index] == component_id).float()
            roi = F.max_pool2d(
                component.unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=roi_radius,
            ).squeeze(0)
            mass = probability[batch_index] * roi
            denominator = mass.sum().clamp_min(1.0e-6)
            pred_y = (mass.squeeze(0) * y_grid).sum() / denominator
            pred_x = (mass.squeeze(0) * x_grid).sum() / denominator
            target_denominator = component.sum().clamp_min(1.0)
            target_y = (component.squeeze(0) * y_grid).sum() / target_denominator
            target_x = (component.squeeze(0) * x_grid).sum() / target_denominator
            scale = component.sum().sqrt().clamp_min(1.0) + float(roi_radius)
            losses.append(
                ((pred_y - target_y).square() + (pred_x - target_x).square())
                / scale.square()
            )
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def compute_irstd_core_ring_loss(
    *,
    routed_logits: torch.Tensor,
    current_logits: torch.Tensor,
    target: torch.Tensor,
    target_component_ids: torch.Tensor,
    core_target: torch.Tensor,
    halo_target: torch.Tensor,
    far_background: torch.Tensor,
    core_gate_logits: torch.Tensor,
    halo_gate_logits: torch.Tensor,
    positive_delta: torch.Tensor,
    negative_delta: torch.Tensor,
    delta_logits: torch.Tensor,
) -> IRSTDCoreRingLossOutput:
    """Optimize IRSTD mIoU while explicitly separating target core and halo."""
    reference_shape = routed_logits.shape
    float_inputs = {
        "current_logits": current_logits,
        "target": target,
        "core_target": core_target,
        "halo_target": halo_target,
        "far_background": far_background,
        "core_gate_logits": core_gate_logits,
        "halo_gate_logits": halo_gate_logits,
        "positive_delta": positive_delta,
        "negative_delta": negative_delta,
        "delta_logits": delta_logits,
    }
    if routed_logits.ndim != 4 or routed_logits.shape[1] != 1:
        raise ValueError("routed_logits must be BCHW with C=1")
    for name, value in float_inputs.items():
        if value.shape != reference_shape:
            raise ValueError(f"{name} must match routed_logits shape")
        if value.device != routed_logits.device:
            raise ValueError(f"{name} must match routed_logits device")
    if target_component_ids.shape != reference_shape:
        raise ValueError("target_component_ids must match routed_logits shape")
    if target_component_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("target_component_ids must be an integer tensor")

    routed = routed_logits.float()
    current = current_logits.detach().float()
    target_float = target.float()
    core_mask = core_target >= 0.5
    halo_mask = halo_target >= 0.5
    far_background_mask = far_background >= 0.5

    bce = F.binary_cross_entropy_with_logits(routed, target_float)
    soft_iou = _soft_iou_loss(routed, target_float)
    core_gate = _balanced_binary_logit_loss(core_gate_logits.float(), core_mask)
    halo_gate = _balanced_binary_logit_loss(halo_gate_logits.float(), halo_mask)
    component_peak = _component_peak_loss(routed, target_component_ids)
    centroid = _component_centroid_loss(routed, target_component_ids)

    probability = torch.sigmoid(routed)
    current_probability = torch.sigmoid(current)
    halo_probability = _masked_mean(probability.square(), halo_mask)
    far_background_no_increase = _masked_mean(
        F.relu(probability - current_probability).square(),
        far_background_mask,
    )

    # The positive arm should raise core logits; the negative arm is subtracted
    # from the base logit and therefore should be non-negative on halo pixels.
    direction = (
        _masked_mean(F.relu(-positive_delta.float()).square(), core_mask)
        + _masked_mean(F.relu(-negative_delta.float()).square(), halo_mask)
    )
    edited = core_mask | halo_mask
    neutral_delta = _masked_mean(delta_logits.float().abs(), ~edited)

    components = {
        "bce": bce,
        "soft_iou": soft_iou,
        "core_gate": core_gate,
        "halo_gate": halo_gate,
        "component_peak": component_peak,
        "centroid": centroid,
        "halo_probability": halo_probability,
        "far_background_no_increase": far_background_no_increase,
        "direction": direction,
        "neutral_delta": neutral_delta,
    }
    total = sum(LOSS_WEIGHTS[name] * value for name, value in components.items())
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("IRSTD core/ring loss is non-finite")
    return IRSTDCoreRingLossOutput(total=total, **components)


__all__ = [
    "IRSTD_CRR_LOSS_VERSION",
    "IRSTDCoreRingLossOutput",
    "LOSS_WEIGHTS",
    "compute_irstd_core_ring_loss",
]

```

---

## 14. 训练入口关键段

```python
# The wrapper parent must be a raw V8 SCTransNet.  Passing an already-built
# Current graph would register NER/QFG twice and is forbidden.
raw_parent, _ = build_clean_v8_mprs_dch_model(FORMAL_SURVIVAL_VARIANT, 42)
model = TPD8NER4QFG2IRSTDCRRInferenceSCTransNet(raw_parent)

# The authority checkpoint is the 568-key training graph.  Validate it first,
# then remove exactly the four named Survival-only keys.  Arbitrary filtering
# or loading 568 keys directly into the 595-key integrated graph is forbidden.
current_568, current_binding = load_audited_irstd_current_best_miou()
current_564 = strip_exact_survival_state(current_568, SURVIVAL_STATE_KEYS)
load_current_into_frozen_base_strictly(
    model,
    current_564,
    current_binding=current_binding,
)
validate_formal_irstd_bgcr_model(model, current_inference_state=current_564)
model.train(True)  # override keeps Current eval and only repair head train

optimizer = torch.optim.AdamW(
    model.trainable_parameters(),
    lr=3.0e-4,
    weight_decay=1.0e-4,
)

for batch in train_loader:
    image = batch["image"].cuda(non_blocking=True)
    target = batch["target"].cuda(non_blocking=True)
    context = batch["frozen_current_context"].cuda(non_blocking=True)

    corrupted = corrupt_irstd_logits(
        current_logits=context.out_logits,
        core_target=batch["core_target"].cuda(),
        outer_ring=batch["outer_ring"].cuda(),
        observed_halo_target=batch["halo_target"].cuda(),
        generator=train_generator,
    )
    routing = model.forward_repair_from_context(
        image=image,
        context=context,
        base_logits_override=corrupted.logits,
    )

    loss = compute_irstd_core_ring_loss(
        routed_logits=routing.routed_logits,
        current_logits=context.out_logits,
        target=target,
        target_component_ids=batch["target_component_ids"].cuda(),
        core_target=torch.maximum(
            batch["core_target"].cuda(),
            batch["baseline_rescue"].cuda(),
        ),
        halo_target=corrupted.halo_target,
        far_background=batch["far_background"].cuda(),
        core_gate_logits=routing.core_gate_logits,
        halo_gate_logits=routing.halo_gate_logits,
        positive_delta=routing.positive_delta,
        negative_delta=routing.negative_delta,
        delta_logits=routing.delta_logits,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
    optimizer.step()
```

非缓存 smoke 也只能调用一次 `forward_for_irstd_training`。正式缓存
runner 直接调用 `forward_repair_from_context`，禁止为同一 batch 重跑两次
parent。现有 Current validator 使用 exact-type 检查，不能直接验证 BGCR
子类；必须先验证独立 exact Current reference，再由 BGCR 专用 validator
证明集成图的 564 个 base tensors 逐位一致。

---

## 15. 已加入的工程测试

### 15.1 精确恒等初始化

```python
with torch.no_grad():
    current_probability = current(image)
    candidate_probability = bgcr(image)
assert torch.equal(current_probability, candidate_probability)
```

### 15.2 冻结合同

```python
for name, parameter in model.named_parameters():
    if name.startswith("irstd_repair."):
        assert parameter.requires_grad
    else:
        assert not parameter.requires_grad
```

训练前后：

```python
assert_state_equal(before_current_state, after_current_state)
assert_buffer_equal(before_current_buffers, after_current_buffers)
```

### 15.3 梯度合同

```python
loss.total.backward()
assert all_finite_nonzero(model.irstd_repair.positive_residual_head.weight.grad)
assert all_finite_nonzero(model.irstd_repair.negative_residual_head.weight.grad)
assert all(parameter.grad is None for name, parameter in model.named_parameters()
           if not name.startswith("irstd_repair."))
```

### 15.4 atlas 拓扑单元测试

构造一个 GT 组件和一个带长尾的 matched prediction：

```text
GT:       ###
pred:     ####------##
```

必须满足：

```text
prediction component 被 matcher 匹配到 GT
attached_halo 包含 GT 外的全部尾部
该尾部不依赖 unmatched_prediction_ids
```

这是 BGCR 区别于 V4/V5 的决定性测试。

### 15.5 counterfactual 测试

- clean：logit 不变；
- core-drop：只降低 core；
- ring-injection：只增加 outer ring；
- 原 GT、Current context、main state 不变；
- 同 seed 输出逐位一致。

---

## 16. 结果解释规则

不设置人工最小幅度。只做严格排序：

### 16.1 相对 Current

```text
candidate mIoU > Current mIoU  -> 性能提升
candidate mIoU = Current mIoU  -> 看 nIoU/F1/Pd/Fa；完全相同保留 Current
candidate mIoU < Current mIoU  -> 不替换 Current
```

本轮上述裁决限定在同一冻结 internal OOF 投影。Original、Current 和
BGCR OOF 都必须用完全相同的 ID 顺序、target hash、matcher 和
`probability > 0.5` 计算。内部胜出不自动替换 official 部署模型。

### 16.2 相对独立 Baseline

“三数据集全面提升”完成的条件只是：

```text
BGCR IRSTD exact mIoU > exact independent-Baseline mIoU
```

不再附加 `+0.005`、`+0.1 pp` 或百分比要求。epoch-713 best 的 mIoU
已恢复为 `67.73579291877417%`；但本轮没有新的 official-test 访问授权，
所以 internal OOF 结果不得与该 official 数值跨 split 做胜负宣称。

其他指标必须完整报告，但不作为 mIoU 专项目标的否决门。若出现极端 Pd/Fa 异常，应如实解释，而不是用加权分数隐藏。

---

## 17. 设计动机：为什么执行前认为该方案比继续 V5 更可能成功

| 已定位问题 | V5 | BGCR |
|---|---|---|
| 主模型是否被动 | 解冻 `outc/up_decoder1` | **全部冻结** |
| 低峰容量 | V4 best-mIoU 最大 +0.60 | **core-only +2.25** |
| 附着 halo | 未单独建图 | **显式 `attached_halo`** |
| 原图局部对比度 | 无 | **3/7 高通 + local std** |
| 自然 rescue 稀缺 | 仅自然样本 | **core-drop 反事实** |
| halo 样本 | 仅已有背景变化 | **ring-injection 反事实** |
| Baseline 信息 | 只有最终标量 | **固定 epoch-1000 teacher 优势图** |
| 主要目标 | 通用 role loss | **IRSTD mIoU 直接优化** |
| internal 选择 | 单 split | **固定分层 OOF** |
| 初始安全性 | epoch 0 fallback | **精确 identity + epoch 0** |

---

## 18. 风险和对应诊断

### 风险 A：正向分支再次制造 halo

检查：

```text
positive_delta inside core
positive_delta in outer ring
TP gain / FP gain
```

若 positive delta 主要出现在 ring，说明 core gate 未学到形态；先检查 atlas/crop，不调性能门槛。

### 风险 B：负向分支压掉 tiny target

检查：

```text
halo gate 与 core gate 重叠率
tiny target core 上 negative_delta
component peak before/after
```

### 风险 C：内部 OOF 正、最终负

这通常是域代表性或 Baseline 复现不一致。应检查：

- train/test normalization；
- full-image padding；
- TF32；
- canonical matcher；
- Current source SHA；
- Baseline source/checkpoint SHA；
- cache 是否来自同一 Current graph。

不应再次访问 official test 后回调 loss 权重。

### 风险 D：BGCR 只复制 Baseline，未超过 Baseline

Baseline-guided atlas 的目的不是强制模仿全部 Baseline，而是学习其相对 Current 的优势区域。`soft-IoU + GT core/ring` 仍是主监督，因此 BGCR 可保留 Current 已有优势并学习 Baseline 的互补。

---

## 19. 执行顺序（已完成）

1. **已冻结并哈希 Current**：564 keys、全部参数和 buffer；
2. **已绑定 independent Baseline**：epoch 1000 作唯一 teacher，epoch 713 best
   只作 operational 参考，禁止用 Original 冒充；
3. **已构建 Current+GT+epoch1000-teacher 的 core/ring/attached-halo atlas**；
4. **已生成冻结 Current context 与 epoch-1000 Baseline logits cache**；
5. **已运行 BGCR 单元测试与 GPU identity/backward smoke**；
6. **已固定 3-fold OOF，并分别训练 120 epochs**；
7. **已按零门槛、精确充分统计 role key 选择 epoch 0**；
8. **已在完整 official-train 投影封存 epoch-0 full internal candidate**；
9. **已冻结代码、checkpoint、source lock、OOF manifest**；
10. **只汇总 internal OOF 的 Current/BGCR/Baseline-1000**，未访问 official test；
11. BGCR OOF 没有严格胜过 Current，故最终执行分支为：**保留 Current**。

---

## 20. 最终研究判断

本轮开始前，证据把剩余问题定位为 **IRSTD 的域专用输出几何误差**：

- 两个自然漏检峰值超出 V4 的正残差能力；
- 更主要的 mIoU 损失来自 matched component 外溢、附着 halo 和桥接；
- V5 的 TP 增长被更大的 FP 增长抵消；
- component-Fa 的 unmatched-component 口径不能覆盖这一问题；
- 单一内部 split 不足以支持外推。

BGCR 据此完成实现并接受了完整三折 OOF 检验，但实测所有非零 epoch 的
mIoU 都低于 epoch-0 Current。因而原先“轻量 core/ring 修复可以提高
Current”的性能假设已被本轮否定。最终裁决是：

> **保留冻结的 `TPD8 + NER4 + QFG2，TSS-off` Current；BGCR 仅保留为失败实验与可复现工程资产，不进入部署模型。**

该结果不说明工程实现无效：恒等回退、冻结主模型、无 official 污染、精确
OOF 聚合都按合同工作；它只说明这套监督与轻量头没有把已定位误差转成
可泛化的 mIoU 增益。不能再通过调门槛或挑单 fold 把失败结果改写为成功。

---

## 21. 实际执行结果与最终裁决

### 21.1 工程验证、mask、缓存与执行优化

2026-08-08 的最终定向测试集已全部通过：

| 测试文件 | 通过数 |
|---|---:|
| `tests/test_irstd_bgcr_cache.py` | 10 |
| `tests/test_irstd_bgcr_core.py` | 13 |
| `tests/test_irstd_bgcr_model_contract.py` | 14 |
| `tests/test_irstd_bgcr_pipeline.py` | 12 |
| `tests/test_irstd_bgcr_run_contract.py` | 18 |
| `tests/test_select_irstd_bgcr_oof_v1.py` | 15 |
| **合计** | **82/82** |

最终复验命令、退出码和警告摘要已落盘到
[`PYTEST_VERIFICATION_20260808.md`](results/irstd_bgcr_v1/PYTEST_VERIFICATION_20260808.md)；
结果为 `82 passed, 5 warnings, 0 failures/errors`。

mask 合同也在完整 800 张 official-train 投影上回放：原 PNG 的
`strict > 127.5` 与缓存 normalized target 的 `strict > 0.5` 在
**800/800** 张图上逐像素一致。该口径同时写入
`frozen_context_cache/identity.json` 的 `target_binarization`；这解决了
少数带灰度抗锯齿像素的 mask 被 `>0` 错误扩张的问题。缓存最终状态为：

```text
sample_count                         800
container_compression                store
sample payload bytes                 37,752,819,200 (35.1601 GiB)
cache manifest file SHA256           90c5a1fce85920ded133183a9f1b7f01083d7c7e774e1d6a4d52609033e68ec3
cache COMMITTED file SHA256          ff3d3f71eeb3b28f6ac1dffedb7a750ac439d2fd85a0c866d8a662e06c7f98c0
cache manifest semantic SHA256       cda0859ffb55d4cf7237e5b0750387b1a624ff4e6c27d008c27a11a4811ff0c3
cache access                         validate_once_then_host_RAM_resident
```

`store`-only NPZ 避免每个 batch 的解压开销；trainer 启动时只做一次
manifest、sidecar、container 和 array SHA 校验，随后把 cache 保持在 host
RAM，epoch 内不重新打开 NPZ。GPU 与线程执行证据为：

- fold 0、fold 2 使用空闲物理 GPU 2，RTX 5090，UUID
  `GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562`；
- fold 1 使用空闲物理 GPU 3，RTX 5090，UUID
  `GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3`；
- 首轮诊断发现本机 PyTorch 默认 96 个 CPU 线程使 16-sample cached
  collate 因调度/内存竞争慢逾 60 倍；两份
  `fold_{0,1}_pre_cpu_thread_fix` 诊断运行都只到 epoch 0，未进入 OOF；
- 正式协议固定 `cpu_intraop_threads=4`、`cpu_interop_threads=1` 后，三个
  fold 均完成 epoch 0–120，并各写出 25 个互斥验证点。

上述优化只改变数据搬运与 CPU 执行线程，不改变 cache tensor、采样计划、
优化器、loss、FP32/TF32-off 或 role key。三个正式 summary 的文件 SHA256
依次为：

```text
fold 0  d5c4dfa3b6642c8b4c386290a6de4d09bac78b788a88068b5fa39fd8959036dd
fold 1  f103a25084a19f586462b3818bb543b711a1a36d76a409b3ebc729ba3967b0a8
fold 2  7bd574ed83f0561dc54714fdb61ac6b5d60b2efe9db024d18f7741f0acbf7eb9
```

### 21.2 三折 OOF 实际结果

三个 validation fold 的样本数为 `267/267/266`。selector 先合并 TP、
union、组件命中数、Fa numerator 等充分统计，再计算一次指标；没有对
fold 百分数取平均。最终 pooled 结果为：

| 同一 train-only OOF 投影 | Epoch | mIoU | nIoU | F1 | Pd | Fa ×10⁻⁶ | TP / FP / FN pixels |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current（BGCR identity） | **0** | **78.2248538184%** | 72.8588773649% | **87.7822056155%** | **98.4937238494%** | **5.3119659424** | 52,977 / 7,275 / 7,472 |
| 独立 Baseline epoch 1000 | 0 | 78.1093651422% | **73.1243005041%** | 87.7094419823% | 98.2426778243% | 8.8453292847 | 53,237 / 7,708 / 7,212 |

这里的 Fa 仍使用仓库冻结的 unmatched-component pixel numerator；表末的
FP 是全部 pixel FP，二者不能互相替代。Current 在这个相同 OOF 投影上的
mIoU 比 epoch-1000 Baseline 高 `0.1154886762 pp`，但这不是 official-test
结论，也不能与 epoch-713 的 historical official 数值跨 split 相减。

25 个候选 epoch 中，所有非零 epoch 的 pooled mIoU 都低于 epoch 0；
最高的非零候选是 epoch 5，mIoU 为 `77.6175928142%`，仍比 Current 低
`0.6072610042 pp`。零门槛 selector 因而给出：

```text
selected_epoch                         0
strictly_improves_epoch0_miou          false
strictly_improves_epoch0_full_role_key false
OOF selection file SHA256
  4f238f74ed2bf7fa1467ec33679f35dd9f2a2d5963353fc2601d88219b8afe41
```

这不是“门槛太高”导致的拒绝：本协议没有正 margin；只要 mIoU 有任意严格
正增益就会选中。实际结果说明训练后的 BGCR 修改均未超过恒等 Current。

### 21.3 full candidate、独立回放与模型裁决

selector 选中 epoch 0 后，全量阶段按合同只封存恒等初始化候选：

```text
integrated candidate file SHA256
  ad9049d85772673e60d390d2284f5995e52f36edcab53995fb3309163a573903
integrated state keys                 595
integrated state semantic SHA256
  e6a81e9b3ed5a8d76528ed7df45c9e2a7c1b6b9ee3cb48c460f68402d1facfbf
full summary file SHA256
  89f611310bac6527e087dbffdbfbf247d7dae750091cb4d5359b8a79eef71699
candidate state == formal initial     true（独立逐 tensor 位级回放）
```

cache、三个 fold、OOF selector、full summary 和 candidate 均记录：
`official_test_accessed=false`、`official_test_index_opened=false`、
`official_test_index_parsed=false`、`official_test_loader_built=false`、
`official_evaluation_performed=false`。本轮没有构造或读取 official-test
loader/index。

最终结论是：**BGCR 的代码与工程合同设计成功，但性能设计失败**。它没有
在同一 train-only OOF 投影上严格提高 Current，最终 full candidate 又与
formal epoch-0 初始化位级相同，所以 **BGCR 不替换 Current**；当前最优完整
模型仍是冻结的 `TPD8 + NER4 + QFG2，TSS-off` Current。

---

## 22. 依据与来源

- 仓库：[Arialliy/SCTransNet_main](https://github.com/Arialliy/SCTransNet_main)
- Current 主模型：[tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py)
- PBDR-V4 校准器：[tpd_role_aligned_residual_calibrator_v4.py](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/model/tpd_role_aligned_residual_calibrator_v4.py)
- PBDR-V4 atlas：[pbdr_v4_component_atlas.py](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/experiments/pbdr_v4_component_atlas.py)
- PBDR-V4 loss：[pbdr_v4_component_loss.py](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/experiments/pbdr_v4_component_loss.py)
- PBDR-V5 loss：[pbdr_v5_target_preservation_loss.py](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/experiments/pbdr_v5_target_preservation_loss.py)
- PBDR-V5 protocol：[PBDR_V5_INTERNAL_PROTOCOL.md](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/experiments/PBDR_V5_INTERNAL_PROTOCOL.md)
- 数据加载与增强：[dataset.py](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/dataset.py)、[utils.py](https://raw.githubusercontent.com/Arialliy/SCTransNet_main/main/utils.py)
- 冻结 cache commit：`results/irstd_bgcr_v1/frozen_context_cache/COMMITTED.json`
- 三折 summary：`results/irstd_bgcr_v1/fold_{0,1,2}/summary.json`
- OOF 选择：`results/irstd_bgcr_v1/oof_selection.json`
- full 候选与 summary：`results/irstd_bgcr_v1/full_selected/`
- 本轮实验事实：`SCTransNet_历史模型实验结果总汇.md`
