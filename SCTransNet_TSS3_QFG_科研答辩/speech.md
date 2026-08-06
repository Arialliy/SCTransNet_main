## Slide 1: SCTransNet 目标保真与 TSS 设计

今天汇报的是我们在 SCTransNet baseline 上的当前最新方案。核心不是简单增加注意力模块，而是沿着目标证据的生命周期，分别处理下采样保真、浅层证据中继、Query 频率抑噪和训练期目标存活监督。后面我会先说明为什么产生这个想法，再逐层介绍完整模型、四个模块以及三种 TSS 强度的实验结果。

## Slide 2: 研究背景：红外小目标的三类困难

红外小目标的困难首先来自像素少，连续下采样很容易把弱响应抹掉；其次，边缘、纹理和热噪声同样表现为局部高频，容易形成虚警；第三，编码器和解码器的语义层次不同，定位与边界恢复会受到影响。因此我们不能只看 Mean IoU，还要同时观察 Pd、Fa 和 tiny-Pd，后续方案也围绕这些矛盾展开。

## Slide 3: Baseline：SCTransNet 整体架构与 SCTB

先看 baseline。四级编码器产生 E1 到 E4，经过 Patch Embedding 对齐为 16×16 token，再由 SCTB 完成全层级通道语义交互。输出恢复到原分辨率，与编码特征残差相加，经 CCA 选择性送入解码器，最后进行多尺度监督。这里要强调：SCTB 很擅长重组已有信息，但它开始工作时，信息已经经历了 token 化。

## Slide 4: 研究缺口：跨层交互之前，目标证据是否仍然存活

我们的出发点就在这里：如果弱目标在 stride embedding 阶段已经被平均或抑制，后续 attention 再强也只能重组剩余信息，无法凭空恢复消失的证据。同时 Query 中的高频杂波还可能干扰跨尺度相关性。因此问题不是继续堆叠注意力，而是保证进入注意力之前仍有可靠目标证据，并在训练中直接检查这种存活状态。

## Slide 5: 方案映射：在 Baseline 的什么位置改、为什么改

这页把四个问题和修改位置一一对齐。前两级 embedding 用 TPD 做目标保真；SCTB 之前暴露的浅层证据通过 NER 中继到解码尾部；SSCA 的 Query 用 QFG 做有界频率调制；emb1 和 emb2 endpoint 增加训练期 TSS 监督。K/V、CFN 和 Decoder 主体不变，所以每项修改的作用位置和影响边界都可以追踪。

## Slide 6: 完整模型：SCTransNet + TPD + NER + QFG + TSS

完整模型中蓝色是 baseline 主路，橙色是 TPD 和 NER，紫色是 QFG，绿色虚线是只在训练期存在的 TSS。阅读顺序从左到右：TPD 替换前两级 embedding，NER 收集五个证据节点并按 q4、q3、q2 中继，QFG 只进入 SSCA Query，TSS 从两个 endpoint 产生目标存在监督。推理时裁掉 TSS，主干部署路径保持完整。

## Slide 7: TPD V8-MPRS-DCH：目标保真下采样

TPD 只替换 embeddings_1 和 embeddings_2。内部把信息分为 Keep、Context 和 Saliency 三路；MPRS 用四相位表示减少显著性在降采样中的损失，DCH 则延迟 Context 对 Saliency 学习轨迹的影响。最终 endpoint 继续进入 SCTB 主路，evidence 输出给 NER 和 TSS，形成主任务与辅助证据的明确分工。

## Slide 8: NER V4 Tail-Aware：五节点证据中继

NER 从 emb1 暴露三个节点 h11、h12、h13，从 emb2 暴露两个节点 h21、h22，共五个浅层证据源。中继从 q4 开始，经 q3 到 q2，每级融合本地证据、上一级 relay 和 decoder upsample，再用 gate 调制对应 decoder skip。它不连接 q1，避免末端过度扰动；Tail-Aware 的 complement_tail 进一步限制补偿作用范围。

## Slide 9: QFG2-CROA：Query-only 频率调制

QFG 的输入来自 E1 到 E4，固定 Haar 变换分离 high 和 low 频率先验。频率源采用 stop-gradient，再通过 CROA 有界门控生成 g1 到 g4，只调制 Q1 到 Q4。共享 K/V、CFN 和 Decoder 不变。这样做的目的，是抑制 Query 中的杂波干扰，同时保留 baseline 的语义检索边界。

## Slide 10: TSS-on：双端点目标存活监督

TSS 读取 emb1 和 emb2 的 stride-16 endpoint，分别通过独立的 1×1 卷积输出 cell-presence logits。标签由 GT 经过 16×16 max-pooling 得到，两个 BCE 相加形成 L_tss，并与分割损失联合训练。它直接回答目标证据在深层入口前是否仍然存在；部署时两套 head 完全移除，因此没有额外推理分支。

## Slide 11: 训练协议：动态 TSS 权重与 10% Ratio Cap

主损失是六个分割输出的 BCE 有序求和，TSS 损失来自两个 endpoint。有效权重 λ_eff 取请求值和 10% ratio cap 的较小者，并对损失比值使用 stop-gradient，避免辅助权重分支反向影响主任务。三组实验的唯一主变量是 λ_req 等于 0.0025、0.005 或 0.01，其余训练设置保持一致。

## Slide 12: 实验设置：Original + Final 三种 TSS 参数

实验部分只保留 Original 和当前 Final 完整结构。Final 的 TPD、NER、QFG 与 TSS-on 结构完全固定，只比较 λ=0.0025、0.005、0.01 三种 TSS 参数。三个数据集均采用 seed 42、formal1000、固定阈值和一致评估口径，并分别报告 best-Mean-IoU 与 best-Pd，避免把不同目标的 checkpoint 混在一起。

## Slide 13: 最终结果：NUAA-SIRST（Original + TSS三参数）

NUAA 每个 role 都展示 Original 加三种 Final 参数。best-Mean-IoU 下，λ=0.01 达到 0.797386，并把 Fa 降到 8.3693×10⁻⁶；best-Pd 下它的区域质量和 Fa 同样较好。但 Original 仍保持 Pd 260/263 和 tiny-Pd 34/35，因此正确结论是 Final 改善区域质量与虚警控制，而不是全面超过 Original。

## Slide 14: 最终结果：NUDT-SIRST（Original + TSS三参数）

NUDT 的参数偏好不同。best-Mean-IoU 下，λ=0.0025 获得最高 Mean IoU 0.946686 和 nIoU 0.949784；best-Pd 下 Original 的 Pd 为 941/945，仍是最高，λ=0.005 的 940/945 最接近。Final 的低参数配置提高区域质量或降低 best-Pd 固定点的 Fa，但 Original 保留部分指标优势。

## Slide 15: 最终结果：IRSTD-1K（Original + TSS三参数）

IRSTD-1K 的 role 分化最明显。best-Mean-IoU 下 λ=0.0025 得到 0.686740；best-Pd 下 λ=0.005 更均衡，Mean IoU 0.658486、nIoU 0.658560、Pd 288/297、Fa 2.8145×10⁻⁵。λ=0.01 的 tiny-Pd 为 26/30。这里没有一个参数在两个 role 的所有指标上同时领先。

## Slide 16: 综合对比：Original 与 Final 三参数的收益边界

把 Original 与三种 Final 参数并排后，可以看到 NUAA 更偏向 λ=0.01 的区域质量与低 Fa，NUDT 更偏向 λ=0.0025 或 0.005，IRSTD 则随 checkpoint role 改变。Final 在多个 Mean IoU、nIoU 和 Fa 指标上具有竞争力，但 Original 在部分 Pd、tiny-Pd 或 Fa 上仍占优，所以不能宣称跨数据集统一、全面最优。

## Slide 17: 总结：最终模型、三参数结果与结论边界

最终结构由 SCTransNet、TPD V8、NER V4、QFG2-CROA 与 TSS-on 构成。实验部分只比较 Original 与这一固定 Final 结构下的三种正 TSS 参数，没有展示失败过程或中间模型。当前证据说明 Final 在多个区域质量与虚警指标上有竞争力，但最优参数依赖数据集与 checkpoint role；下一步应重点讨论跨数据集稳定性和 Pd–Fa–tiny-Pd 权衡。
