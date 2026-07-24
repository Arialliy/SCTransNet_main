# 目标保真下采样 SCTransNet

> **英文暂定名称：TPD-SCTransNet: Target-Preserving Downsampling SCTransNet for Infrared Small Target Detection**

本方向不修改 SSCA、CFN，也不涉及跨域泛化和单纯的损失函数堆叠，而是研究一个更基础的问题：

> **红外小目标在进入 SCTB 之前，是否已经被连续下采样削弱甚至丢失？**

SCTransNet 论文指出，多次下采样会造成小目标空间信息丢失，进而影响不同层级之间的特征交互。原始 SCTransNet 的编码器采用四组残差块和最大池化，并将四级编码特征统一映射至 $H/16\times W/16$ 后送入 SCTB。

因此可以提出如下核心假设：

> **SCTransNet 擅长建模全层级语义，但 Transformer 只能重组仍然存在的特征；已经在下采样阶段消失的小目标信息，无法仅靠注意力机制重新恢复。**

---

## 一、核心改进：目标保真下采样模块 TPD

将原始下采样算子：

```python
MaxPool2d(kernel_size=2, stride=2)
```

替换为多分支的 **Target-Preserving Downsampling（TPD）** 模块。

### 1. 背景语义分支

采用步长卷积或低通下采样获取平滑的背景语义：

$$
F_{\mathrm{ctx}}
=
\mathrm{DWConv}_{3\times3,s=2}(F).
$$

该分支主要负责保留建筑、云层、山脉、海面等连续背景结构，与 SCTransNet 原本强调的背景连续性建模保持一致。

### 2. 局部显著性分支

通过局部最大响应与局部均值之间的差异突出小目标：

$$
F_{\mathrm{sal}}
=
\mathrm{MaxPool}(F)-\mathrm{AvgPool}(F).
$$

也可以使用：

$$
F_{\mathrm{sal}}
=
\left|F-\mathrm{AvgPool}_{3\times3}(F)\right|.
$$

随后再进行步长为 2 的映射。

该分支不直接假定目标一定是一个结构清晰的物体，而是尽可能保留局部异常响应。

### 3. 无损重排分支

使用 PixelUnshuffle 将每个 $2\times2$ 邻域重新排列到通道维：

$$
F_{\mathrm{keep}}
=
\mathrm{Conv}_{1\times1}
\left(
\mathrm{PixelUnshuffle}_{2}(F)
\right).
$$

PixelUnshuffle 不直接丢弃 $2\times2$ 邻域中的像素，而是将空间信息转移到通道维，因此更适合保护仅由少量像素构成的红外小目标响应。

### 4. 自适应门控融合

三个分支对不同场景的贡献并不相同，因此设计动态门控权重：

$$
[g_{\mathrm{ctx}},g_{\mathrm{sal}},g_{\mathrm{keep}}]
=
\mathrm{Softmax}
\left(
\mathrm{MLP}(\mathrm{GAP}(F))
\right).
$$

最终输出为：

$$
F_{\mathrm{out}}
=
\mathrm{Conv}_{1\times1}
\left(
 g_{\mathrm{ctx}}F_{\mathrm{ctx}}
+g_{\mathrm{sal}}F_{\mathrm{sal}}
+g_{\mathrm{keep}}F_{\mathrm{keep}}
\right).
$$

直观上：

- 简单天空背景可能更依赖局部显著性分支；
- 建筑、山地等复杂背景可能更依赖背景语义分支；
- 极小、模糊目标可能更依赖无损重排分支。

---

## 二、加入“目标存活监督”

仅替换下采样模块可能不够，因为网络不一定会主动保留小目标。因此，可以在每一级编码器下采样后增加一个轻量辅助预测头：

$$
S_i=\mathrm{Conv}_{1\times1}(E_i).
$$

对应标签不建议直接使用普通双线性插值缩小，因为一个 $1\sim2$ 像素目标可能在插值后变成接近零的响应。建议使用最大池化生成多尺度标签：

$$
Y_i=\mathrm{MaxPool}_{2^i}(Y).
$$

只要原始目标区域中存在一个正像素，缩小后的标签仍然能够保留目标。

目标存活损失定义为：

$$
\mathcal L_{\mathrm{surv}}
=
\sum_{i=1}^{4}
\omega_i
\mathcal L_{\mathrm{seg}}(S_i,Y_i).
$$

总损失为：

$$
\mathcal L
=
\mathcal L_{\mathrm{SCTransNet}}
+
\lambda_s\mathcal L_{\mathrm{surv}}.
$$

原始 SCTransNet 已经对五级显著图及融合输出进行深监督。这里的“目标存活监督”与原始深监督不同：它监督的是**编码器中的目标信息在下采样后是否仍然存在**，而不是只监督解码器输出。

---

## 三、进一步增强：双分辨率 SCTB

前述 TPD 是基础版本。验证有效后，可以进一步设计双分辨率跨层建模。

原始 SCTransNet 将不同编码层全部映射到 $H/16\times W/16$ 后进行全层级交互。可以改成：

- 粗分辨率分支：$H/16\times W/16$，负责背景全局语义；
- 细分辨率分支：$H/8\times W/8$，只接收浅层目标敏感特征；
- 最后融合细粒度目标特征与粗粒度背景特征。

可表示为：

$$
O_i^{c}
=
\mathrm{SCTB}_{c}(I_1^{c},I_2^{c},I_3^{c},I_4^{c}),
$$

$$
O_i^{f}
=
\mathrm{SCTB}_{f}(I_1^{f},I_2^{f}),
$$

$$
O_i
=
O_i^{c}
+
\alpha_i\cdot\mathrm{Down}(O_i^{f}).
$$

由此形成明确的功能解耦：

- 粗尺度负责判断“这一片背景是什么”；
- 细尺度负责判断“这里是否存在极小异常目标”。

第一版不建议直接加入双分辨率 SCTB。应先证明下采样阶段确实存在目标信息损失，再逐步添加这一模块。

---

## 四、实验设计

### 1. 主对比实验

保持 SCTB、CFN、解码器和训练设置全部不变，仅更改下采样模块：

| 方法 | 下采样方式 | 目标存活监督 |
|---|---|---|
| Baseline | MaxPool | 无 |
| Variant A | Strided Conv | 无 |
| Variant B | PixelUnshuffle + Conv | 无 |
| Variant C | 双分支 TPD | 无 |
| Variant D | 三分支 TPD | 无 |
| Variant E | 三分支 TPD | 有 |
| Full | TPD + 双分辨率 SCTB | 有 |

对比时应严格使用相同的随机种子、训练轮数、输入尺寸、数据增强和优化器配置，否则无法判断性能增益是否来自下采样模块。

### 2. 分阶段替换实验

不同下采样层的重要性并不相同，因此需要比较：

| 设置 | 第 1 层 | 第 2 层 | 第 3 层 | 第 4 层 |
|---|---:|---:|---:|---:|
| A | TPD | MaxPool | MaxPool | MaxPool |
| B | TPD | TPD | MaxPool | MaxPool |
| C | TPD | TPD | TPD | MaxPool |
| D | TPD | TPD | TPD | TPD |

预期上，只替换前两层或前三层可能更合适。浅层保留更多目标空间信息，而深层更侧重背景语义，全部替换不一定取得最优结果。

### 3. 按目标尺寸分组评估

不能只报告整体 IoU。该方向是否成立，关键在于是否真正改善极小目标检测。

建议按标注区域面积分组：

- 极小目标：$A\leq4$ 像素；
- 小目标：$5\leq A\leq9$；
- 中等目标：$10\leq A\leq25$；
- 较大目标：$A>25$。

每个尺寸区间分别报告：

- Pd；
- Fa；
- IoU；
- nIoU；
- F-measure；
- 漏检数量。

对于这一方向，**极小目标上的 Pd 和漏检数量**比整体 IoU 更重要。

---

## 五、增加“目标存活率”分析指标

为了证明改进并非黑盒式涨点，可以直接测量目标特征经过每一级编码器后的保留情况。

设第 $i$ 层特征图为 $E_i$，缩放后的目标掩膜为 $Y_i$，定义目标—背景响应比：

$$
R_i
=
\frac{
\mathrm{Mean}(|E_i|\odot Y_i)
}{
\mathrm{Mean}(|E_i|\odot(1-Y_i))+\epsilon
}.
$$

$R_i$ 表示目标区域响应相对于背景区域响应的比值。

可以对比原始 MaxPool 与 TPD：

| 模型 | $R_1$ | $R_2$ | $R_3$ | $R_4$ |
|---|---:|---:|---:|---:|
| SCTransNet |  |  |  |  |
| TPD-SCTransNet |  |  |  |  |

还可以增加目标存活率：

$$
SR_i
=
\frac{
\#\{\text{第 }i\text{ 层仍有显著响应的目标}\}
}{
\#\{\text{全部目标}\}
}.
$$

如果 TPD 在浅层和中层显著提高 $R_i$ 与 $SR_i$，同时极小目标 Pd 上升，就能形成完整证据链：

> 下采样更加保真  
> $\rightarrow$ 编码器中的目标响应更强  
> $\rightarrow$ SCTB 获得更完整的目标信息  
> $\rightarrow$ 极小目标漏检下降。

---

## 六、需要重点完成的消融实验

至少包含以下实验：

1. 仅使用背景语义分支；
2. 仅使用局部显著性分支；
3. 仅使用 PixelUnshuffle 分支；
4. 背景语义分支 + 局部显著性分支；
5. 背景语义分支 + PixelUnshuffle 分支；
6. 三分支直接相加；
7. 三分支动态门控；
8. 加入目标存活监督；
9. 多尺度标签使用双线性插值与最大池化的对比；
10. 替换前两层、前三层和全部四层的对比。

原始 SCTransNet 的消融实验说明，SSCA 对整体性能提升较大，因此建议保持 SSCA 不变，专门验证：

> **向 SSCA 输入保留得更好的目标特征，是否能够进一步提升检测性能。**

这样可以建立更清晰的因果关系，避免同时修改多个 Transformer 模块后难以解释增益来源。

---

## 七、创新性边界

单独完成以下操作，创新性通常偏弱：

- 将 MaxPool 替换为 Strided Conv；
- 将 MaxPool 替换为 PixelUnshuffle；
- 单纯增加小波下采样；
- 只增加编码器辅助损失。

一个相对完整的研究方案应包含三个相互支撑的部分：

1. **目标—背景双特性下采样**：同时保留局部异常和背景连续性；
2. **图像自适应门控**：根据不同场景动态选择下采样路径；
3. **目标存活监督与分析指标**：证明目标信息确实得到保留。

三部分结合后，研究内容才能从“替换一个算子”提升为一个完整、可验证的问题。

---

## 八、可能出现的问题与应对策略

最大的风险是：局部显著性分支不仅会保留真实目标，也可能保留高亮噪声、建筑边缘和云层局部异常，从而提高 Fa。

可以通过以下策略控制：

- 不让局部显著性分支单独决定输出；
- 使用背景语义分支提供抑制信息；
- 门控权重由全局特征生成；
- 对孤立的高响应背景区域增加负样本约束；
- 只在前两级编码器使用局部显著性分支；
- 在复杂背景下限制显著性分支的最大门控权重。

SCTransNet 的核心优势之一是利用深层背景连续性降低虚警，因此新模块不能只追求目标增强，还必须避免破坏原有的背景建模能力。

---

## 九、最适合先实现的第一版

第一版建议严格控制变量，只完成以下修改：

1. 在前两次下采样中，使用 `PixelUnshuffle + 局部显著性分支 + 1×1 融合` 替换 MaxPool；
2. 在第二、第三级编码器后增加目标存活辅助头；
3. 其余 SCTB、CFN、CCA 和解码器完全保持不变；
4. 按目标面积分组统计 Pd、Fa 和漏检数；
5. 可视化原模型与新模型在 $E_1\sim E_4$ 中的目标响应变化。

该版本具有以下优点：

- 代码改动集中；
- 变量控制清晰；
- 计算开销相对可控；
- 容易建立性能增益与下采样保真之间的因果关系；
- 与 SCTransNet 的全层级语义交互机制衔接紧密。

最终可以将研究主线概括为：

> **原始 SCTransNet 负责全层级语义交互，TPD 模块负责确保极小目标能够“活着”进入全层级交互。**

---

## 十、建议的实现文件结构

```text
model/
├── SCTransNet.py
├── target_preserving_downsample.py
│   ├── ContextDownsampleBranch
│   ├── SaliencyDownsampleBranch
│   ├── PixelUnshuffleBranch
│   └── TargetPreservingDownsample
├── survival_head.py
│   └── TargetSurvivalHead
└── losses.py
    ├── SegmentationLoss
    └── TargetSurvivalLoss
```

建议先实现 `TargetPreservingDownsample`，并保留与原始 `MaxPool2d` 相同的输入、输出空间缩放关系，以减少对后续网络代码的影响。

---

## 十一、推荐实验执行顺序

1. 复现原始 SCTransNet 基线；
2. 仅用 PixelUnshuffle 替换第一个 MaxPool；
3. 加入局部显著性分支；
4. 加入背景语义分支；
5. 比较直接融合与动态门控融合；
6. 加入目标存活辅助监督；
7. 分析不同目标尺寸上的 Pd、Fa 和漏检数；
8. 计算 $R_i$ 与 $SR_i$；
9. 验证前两层、前三层和全层替换；
10. 在基础版本稳定后，再尝试双分辨率 SCTB。

---

## 十二、来源说明

本文档中的 SCTransNet 基线结构、四级编码与最大池化、统一到 $H/16\times W/16$ 的 Patch Embedding、SCTB/SSCA/CFN、多尺度深监督和评价指标等背景信息，来自用户提供的 SCTransNet 论文。TPD、多分支下采样、目标存活监督、目标存活率指标以及双分辨率 SCTB 属于在该基线上的实验设计建议。
