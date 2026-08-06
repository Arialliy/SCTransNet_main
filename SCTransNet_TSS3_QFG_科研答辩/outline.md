# SCTransNet 目标保真与训练期 TSS 设计汇报

> 用途：组会 / 科研答辩，详细介绍当前模型、核心模块设计与截止到三种正 TSS 配方的实验结果  
> 范围：SCTransNet + TPD V8-MPRS-DCH + NER V4 Tail-Aware + QFG2-CROA + TSS-on  
> 实验截止：TSS `λ=0.0025 / 0.005 / 0.01`；不纳入 TSS-off、EC-TSS、GCSF、DORF 等后续实验  
> 建议：17 页，16:9，中文，科研答辩风

## Slide 1：封面｜SCTransNet 目标保真与 TSS 设计

- 副标题：从浅层 token 保真、证据中继到 Query 频率调制与目标存活监督
- 研究任务：红外小目标分割 / 检测
- 汇报内容：研究动机、完整模型、模块细节、训练协议、三种 TSS 实验
- 版式角色：封面，使用 baseline 网络图的局部作为淡化技术背景
- 必需源图：
  - SCTransNet 原始整体网络图；严格输入资产；不改结构、文字和箭头

    ![SCTransNet architecture](/home/ly/SCTransNet_main/Fig/picture2.png)

## Slide 2：研究背景｜红外小目标的三类困难

- 目标尺度小、像素少，多次下采样后易丢失
- 局部显著性与背景高频杂波相似，容易形成虚警
- encoder–decoder 语义鸿沟影响定位与边界恢复
- Pd、Fa、Mean IoU、tiny-Pd 存在多目标权衡
- 版式角色：问题定义，左侧论文证据图，右侧三类挑战
- 必需源图：
  - SCTransNet 论文 Fig.1；严格输入资产；保持实验面板与标注

    ![SCTransNet motivation](/home/ly/SCTransNet_main/Fig/picture01.png)

## Slide 3：Baseline｜SCTransNet 整体架构与 SCTB

- 四级 encoder 提取多尺度局部特征
- patch embedding 将四级特征统一为 16×16 token
- SCTB 包含 SSCA 跨层通道交互和 CFN 互补增强
- Feature Mapping 回到对应分辨率，与 decoder skip 融合
- 五尺度 saliency map 参与深监督并融合输出
- 版式角色：baseline 架构详解
- 必需源图：
  - 参考 PPT 中的完整 baseline 前向图；严格输入资产；保留 encoder、Patch Embedding、SCTB×4、Feature Mapping、残差相加、CCA、decoder 与多尺度监督关系

    ![SCTransNet baseline forward](/home/ly/SCTransNet_main/SCTransNet_TSS3_QFG_科研答辩/assets/baseline_reference/baseline_forward.png)

## Slide 4：研究缺口｜跨层交互之前，目标证据是否仍然存活

- SCTB 在 token 化之后才开始全层交互
- 弱小目标可能在 stride patch embedding 中先被平均或抑制
- 后续 attention 只能重组剩余信息，无法恢复已消失证据
- Query 中的高频杂波可能干扰跨尺度相关性
- 研究思路：下采样保真 + 证据中继 + Query 抑噪 + 训练期存活监督
- 版式角色：因果链与科学问题

## Slide 5：方案映射｜在 Baseline 的什么位置改、为什么改

- Baseline `embeddings_1/2`：stride token 化可能先损失弱小目标 → 替换为 TPD V8，保留 endpoint 与显著性证据
- Baseline SCTB 之前：跨层交互只能重组尚存信息 → 增加 NER V4，将浅层证据沿 `q4 → q3 → q2` 中继到 decoder skip
- Baseline SSCA Query：局部高频杂波会进入相关性计算 → 增加 QFG2-CROA，仅对 Q1–Q4 作有界频率调制
- Baseline 深监督：主要约束最终/多尺度 mask → 增加 TSS-on，在 emb1/emb2 endpoint 直接监督 target presence
- 设计约束：不改 K/V、CFN、decoder 主体；TSS 推理时裁剪，避免把训练辅助头带入部署
- 版式角色：以 baseline 完整模型图为底图，四个编号修改点与“问题→改动→目标”侧栏一一对应
- 必需源图：
  - 参考 PPT baseline 完整模型页；严格输入资产；作为修改定位依据

    ![Baseline full model](/home/ly/SCTransNet_main/SCTransNet_TSS3_QFG_科研答辩/assets/baseline_reference/baseline_full_model.png)

## Slide 6：完整模型｜SCTransNet + TPD + NER + QFG + TSS

- 蓝色：SCTransNet baseline 主路径
- 橙色：TPD V8-MPRS-DCH 替换 `embeddings_1/2`
- 橙色窄带：NER V4 Tail-Aware 的 `q4 → q3 → q2` 证据中继
- 紫色：QFG2-CROA 只调制 SSCA Query
- 绿色虚线：TSS-on 只在训练期连接 emb1/emb2 endpoint
- 版式角色：新绘完整模型总图，主视觉占页面 80%
- 必需新绘资产：
  - `assets/figures/full_model_tss_qfg.png`；严格依据代码数据流，明确训练图与推理图边界

## Slide 7：模块一｜TPD V8-MPRS-DCH 目标保真下采样

- 只替换 `mtc.embeddings_1/2`，其余 token 化与主干保持不变
- Keep / Context / Saliency 三源分解的语义与数据流
- MPRS 用四相位分辨表示保留显著性总量
- DCH 延迟 Context 对 Saliency 学习轨迹的影响
- endpoint 输出继续进入 SCTB，evidence 输出供 NER/TSS 使用
- 版式角色：TPD 内部模块放大图
- 必需新绘资产：`assets/figures/tpd_v8_detail.png`

## Slide 8：模块二｜NER V4 Tail-Aware 五节点证据中继

- emb1 暴露 h11/h12/h13，emb2 暴露 h21/h22
- 五节点按 `q4 → q3 → q2` 建立窄带 relay
- 每级融合 local evidence、parent relay 与 decoder upsample
- Tail-Aware `complement_tail` 限制 DC offset 的空间作用域
- 三级 gate 只调制对应 decoder skip，不连接 q1
- 版式角色：五节点到三级 relay 的结构图
- 必需新绘资产：`assets/figures/ner_v4_detail.png`

## Slide 9：模块三｜QFG2-CROA Query-only 频率调制

- 输入为 encoder 四级特征 E1–E4
- 固定 Haar 2×2 变换提取 high/low 频率先验
- 频率源 stop-gradient，通过有界 gate 形成四级 Query 调制量
- 仅修改 SSCA 的 Q1–Q4，K/V、CFN、decoder 不变
- CROA 约束频率调制幅度，保留 baseline 语义主路
- 版式角色：QFG 频率分解—门控—Query 调制数据流
- 参考定位：沿用 baseline SSCA 的 Q/K/V 生成关系，明确紫色 QFG 支路只进入 Q，不进入共享 K/V
- 必需源图：
  - 参考 PPT 的 SSCA Q/K/V 构造页；严格作为结构参照

    ![Baseline SSCA QKV](/home/ly/SCTransNet_main/SCTransNet_TSS3_QFG_科研答辩/assets/baseline_reference/baseline_ssca_qkv.png)
- 必需新绘资产：`assets/figures/qfg2_croa_detail.png`

## Slide 10：模块四｜TSS-on 双端点目标存活监督

- TSS 读取 emb1 / emb2 的 stride-16 endpoint
- 两个独立 `1×1 Conv` 输出 cell-presence logits
- 监督标签为 GT 经 16×16 max-pooling 后的二值 target-presence
- `L_tss = L_emb1 + L_emb2`，与六输出分割损失联合训练
- TSS heads 只存在训练 checkpoint，部署导出时完全移除
- 版式角色：双 endpoint、标签生成、loss 和部署裁剪图
- 必需新绘资产：`assets/figures/tss_on_detail.png`

## Slide 11：训练协议｜动态 TSS 权重与 10% Ratio Cap

- `L_seg = Σ BCE_k`，六输出分割损失有序求和
- `L_tss = L_emb1 + L_emb2`
- `λ_eff = min(λ_req, 0.10·stopgrad(L_seg)/max(stopgrad(L_tss), ε))`
- `L_total = L_seg + λ_eff·L_tss`
- 三种 TSS-on 设计的唯一主变量：`λ_req ∈ {0.0025, 0.005, 0.01}`
- 版式角色：公式、动态 cap 曲线和梯度边界解释

## Slide 12：实验设置｜Original + Final 三种 TSS 参数

- Original：原始 SCTransNet；Final 结构固定为 TPD V8 + NER V4 + QFG2-CROA + TSS-on
- Final 只改变 `λ=0.0025 / 0.005 / 0.01`
- 数据集：NUAA-SIRST、NUDT-SIRST、IRSTD-1K；seed 42、formal1000、阈值 0.5
- 分别报告 best-Mean-IoU 与 best-Pd；不出现 TSS-off、失败方案或模型迭代过程

## Slide 13：最终结果｜NUAA-SIRST：Original + 三参数

- 两张四行精确表：Original、Final λ=0.0025、0.005、0.01
- best-Mean-IoU 下 λ=0.01 获得最高 Mean IoU 0.797386 与最低 Fa 8.3693×10⁻⁶
- best-Pd 下 Original 保持 Pd 260/263 与 tiny-Pd 34/35；Final 改善区域质量和 Fa
- 结论：Final 提升区域质量与虚警控制，但未全面超过 Original 的检测率

## Slide 14：最终结果｜NUDT-SIRST：Original + 三参数

- 两张四行精确表：Original、Final λ=0.0025、0.005、0.01
- best-Mean-IoU 下 λ=0.0025 获得最高 Mean IoU 0.946686 与 nIoU 0.949784
- best-Pd 下 Original 保持最高 Pd 941/945；λ=0.005 最接近，λ=0.0025 区域质量更强
- 结论：Final 提高部分区域质量指标，但 Original 保留部分优势

## Slide 15：最终结果｜IRSTD-1K：Original + 三参数

- 两张四行精确表：Original、Final λ=0.0025、0.005、0.01
- best-Mean-IoU 下 λ=0.0025 获得 Mean IoU 0.686740
- best-Pd 下 λ=0.005 获得 Mean IoU 0.658486、nIoU 0.658560、Pd 288/297、Fa 2.8145×10⁻⁵
- 结论：不同 checkpoint role 呈现不同参数偏好

## Slide 16：综合对比｜Original 与 Final 三参数的收益边界

- NUAA、NUDT、IRSTD 的最优参数不同
- Final 在多个 Mean IoU、nIoU 与 Fa 指标上具有竞争力
- Original 在部分 Pd、tiny-Pd 或 Fa 指标仍占优
- 三种 λ 不存在跨数据集统一、全面最优；仅总结最终结构与 Original 对照

## Slide 17：总结｜最终模型、三参数结果与结论边界

- 最终模型：SCTransNet + TPD V8 + NER V4 + QFG2-CROA + TSS-on
- 三种 TSS 参数均在同一最终结构上评估
- Final 的参数偏好依赖数据集与 checkpoint role
- 不呈现失败过程、中间模型、TSS-off 或后续设计，不宣称全面领先

## 源图与新绘资产映射（待确认）

| 资产 | 使用页 | 用途 | 要求 |
|---|---:|---|---|
| `/home/ly/SCTransNet_main/Fig/picture2.png` | 1 | baseline 整体架构背景 | 严格输入，不改结构/标签/箭头 |
| `/home/ly/SCTransNet_main/Fig/picture01.png` | 2 | baseline 研究动机 | 严格输入，不改实验面板与标注 |
| `assets/baseline_reference/baseline_forward.png` | 3 | baseline 完整前向流程 | 来自参考 PPT，严格保留数据流 |
| `assets/baseline_reference/baseline_full_model.png` | 5 | baseline 修改位置映射 | 来自参考 PPT，四处改动编号定位 |
| `assets/baseline_reference/baseline_ssca_qkv.png` | 9 | QFG 与 baseline SSCA 接口 | 来自参考 PPT，作为 Q-only 结构参照 |
| `assets/figures/full_model_tss_qfg.png` | 6 | 完整模型总图 | 待审批后由选定图像后端绘制 |
| `assets/figures/tpd_v8_detail.png` | 7 | TPD 模块细节 | 严格按代码数据流绘制 |
| `assets/figures/ner_v4_detail.png` | 8 | NER 模块细节 | 严格显示 3+2 节点和 q4/q3/q2 |
| `assets/figures/qfg2_croa_detail.png` | 9 | QFG 模块细节 | 严格显示 E1–E4 来源和 Query-only |
| `assets/figures/tss_on_detail.png` | 10 | TSS-on 模块细节 | 严格区分训练图与部署图 |

> 实验页的表格仅使用本地文档中的精确数字，不生成虚构曲线或改写数据。
