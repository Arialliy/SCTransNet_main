# SCTransNet V2 三独立数据集结果复盘与全局 TSS 配方定型方案

> 项目：红外小目标检测  
> 基线：Original SCTransNet  
> 冻结模型：SCTransNet + TPD8-MPRS-DCH + 五节点 NER4 Tail-Aware + QFG2-CROA  
> 训练期辅助：Target Survival Supervision（TSS）  
> 后续数据集：NUAA-SIRST、NUDT-SIRST、IRSTD-1K  
> 数据划分：严格使用各数据集现有 img_idx/train 与 img_idx/test  
> 固定随机种子：42  
> 训练轮数：1000 epochs  
> 测试频率：每 10 epochs  
> benchmark 阈值：0.5  
> checkpoint 角色：best_miou、best_pd  
> 当前结果裁决：INCONCLUSIVE_MIXED_TRADEOFF  
> 当前执行裁决：FORMAL12_LAUNCH_AUTHORIZED

---

# 0. 本次修订结论

本文件按以下约束定型：

1. 后续只使用 NUAA-SIRST、NUDT-SIRST、IRSTD-1K。
2. SIRST3 仅保留为历史部分结果，不参与后续训练、测试、配方选择或聚合。
3. 数据划分严格使用各数据集已有 img_idx/train 与 img_idx/test。
4. 不再从 train 内部划分 train_core、model_val 或 calibration。
5. 每个方法独立选择自己的 best_miou 和 best_pd，不新增 best_joint。
6. checkpoint 每 10 epochs 在对应 img_idx/test 上评估和选择。
7. 三个正 TSS 候选由三个数据集的 img_idx/test 结果等权选择统一 λ。
8. 固定 threshold=0.5，不进行阈值定型。
9. 固定 seed 42，不扩展多 seed。
10. 模型结构和创新主线保持不变。
11. 本次执行已同步实现 three-dataset v2 协议、runner、launcher、evaluator 和 selector；代码实现完成与正式训练启动是两个独立状态。

当前状态：

    architecture_implementation_complete=true
    architecture_frozen=true
    innovation_mainline_changed=false
    new_module_design_authorized=false

    data_protocol=existing_img_idx_train_test
    checkpoint_selection_split=img_idx_test
    checkpoint_roles=[best_miou,best_pd]
    best_joint_selector_required=false
    benchmark_threshold=0.5

    current_v2_decision=INCONCLUSIVE_MIXED_TRADEOFF
    v2_candidate_retained=true
    v2_global_recipe_established=false
    v2_universal_dominance=false

    future_datasets=[NUAA-SIRST,NUDT-SIRST,IRSTD-1K]
    sirst3_future_role=historical_only

    final_model_performance_established=false
    paper_core_established=false
    stability_claim_supported=false
    training_recipe_finalized=false

    three_dataset_v2_protocol_implemented=true
    evaluator_and_selector_implemented=true
    implementation_tests_passed=true
    indexed_pair_preflight_passed=true
    real_gpu_smoke_passed=true
    formal_results_available=false
    posttraining_batch_orchestrator_complete=false

---

# 1. 审核边界与 img_idx 口径

## 1.1 本地证据来源

本方案以本地工程和真实产物为准：

- 模型工程：/home/ly/SCTransNet_main
- Original baseline：/home/ly/SCTransNet
- TSS loss：experiments/tpd_training_loss.py
- 三数据集 V2 runner：experiments/train_three_dataset_seed42_global_tss_v2.py
- 三数据集 V2 launcher：experiments/three_dataset_seed42_launch_v2.py
- 三数据集 V2 evaluator：experiments/evaluate_three_dataset_v2.py
- 全局 λ selector：experiments/select_three_dataset_global_tss_recipe_v2.py
- 旧训练循环（仅复用 optimization/checkpoint loop）：experiments/train_four_dataset_original_final_seed42_exact_v1.py
- 结果汇总：results/four_dataset_seed42_tss_cap_v2/V2_RESULTS_SUMMARY_STOPPED_20260802.md

不使用外部仓库链接替代本地源码与本地产物。

## 1.2 三个数据集的固定 img_idx

| 数据集 | train img_idx | train 数量 | test img_idx | test 数量 |
|---|---|---:|---|---:|
| NUAA-SIRST | train_NUAA-SIRST.txt | 213 | test_NUAA-SIRST.txt | 214 |
| NUDT-SIRST | train_NUDT-SIRST.txt | 663 | test_NUDT-SIRST.txt | 664 |
| IRSTD-1K | train_IRSTD-1K.txt | 800 | test_IRSTD-1K.txt | 201 |

固定文件摘要：

| 文件 | SHA256 |
|---|---|
| NUAA train | 324e5dadcb6cc9fc2a99a5f5dedd06ad4de77b2ed826e4ceffda8b6a784da0b4 |
| NUAA test | e49023203a323c247306b314f23c8b3b917093a26984067792355adff7a8386e |
| NUDT train | e0a79f7c3d42548ba7d7dad9d2d336012b63a6bc5081e89e286f0f45036f8ec3 |
| NUDT test | a463c52ee64b1c803c4a322fe090aaf6bc360844898e3943bb7c64a8e551b86e |
| IRSTD train | 689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744 |
| IRSTD test | 8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579 |

正式 runner 必须保持列表原始顺序，不允许：

- 重新随机划分；
- 从 train 抽取 validation；
- 合并三个数据集；
- 使用 SIRST3 的 img_idx；
- 静默删除样本；
- 按运行结果修改列表。

seed 42 只控制：

- 参数初始化；
- 训练样本 shuffle；
- crop 与增强随机性；
- DataLoader 的确定性顺序。

seed 42 不改变 img_idx 数据划分。

## 1.3 train 与 test 的用途

img_idx/train：

- 梯度更新；
- normalization 统计；
- TSS pos_weight；
- 数据增强与随机 crop。

img_idx/test：

- 每 10 epochs 固定阈值 0.5 评估；
- 选择 best_miou；
- 选择 best_pd；
- 三个正 λ 候选的全局配方选择；
- 最终结果表与描述性阈值 sweep。

因此本方案属于：

    img_idx_test_selected_protocol=true

必须如实限定：

- test 同时承担周期性评估和 checkpoint 选择；
- λ 也由三个 test 结果选择；
- 结果适合与当前工程和相同协议的 Original 比较；
- 不能把该 test 描述为从未参与模型选择的独立确认集；
- 不能据此建立多随机性稳定性或未见数据泛化主张。

这是用户明确指定的实验协议，不再引入额外内部划分。

## 1.4 SIRST3 历史状态

SIRST3 停止于完成 epoch 742：

- formal1000 未完成；
- 截至 epoch 740 有 74 个十轮评估点；
- best_miou 与 best_pd 已保存；
- rolling resume 仍存在；
- 后续不恢复；
- 不进入三数据集全局 λ 选择。

记录：

    sirst3_historical_status=STOPPED_PARTIAL
    sirst3_used_for_future_selection=false

# 2. 冻结模型与创新主线

## 2.1 最终推理结构

结构保持：

    Original SCTransNet
    + TPD8-MPRS-DCH
    + 五节点 NER4 Tail-Aware
    + QFG2-CROA

当前不修改：

- TPD 的五路目标保真设计；
- MPRS 与 DCH；
- NER 的五节点 relay topology；
- Tail-Aware 支持建模；
- QFG2-CROA 频率调制；
- 主分割路径；
- 六输出深监督；
- optimizer、基础学习率和学习率轨迹；
- patch size、batch size 与数据增强；
- TSS head 的 endpoint 与结构；
- ratio cap 0.10。

## 2.2 TSS 的定位

TSS 是训练期辅助监督：

- 读取 emb1 与 emb2 的 stride-16 endpoint；
- 两个独立 1×1 Conv 产生 cell-presence logits；
- 训练 checkpoint 包含 target_survival 参数；
- 正式部署导出时显式去除 TSS heads；
- 去除后推理路径仍为 TPD + NER + QFG。

因此应准确表述为：

    training_checkpoint_contains_tss_heads=true
    deployment_export_contains_tss_heads=false
    inference_requires_tss_heads=false

不能把训练 checkpoint 和部署导出权重混为一谈。

## 2.3 与代码一致的动态 TSS 公式

主分割损失为六个输出 BCE 的有序求和：

    L_seg = sum(k=1..6) BCE_k

两个 Survival endpoint 的损失为：

    L_tss = L_emb1 + L_emb2

V2 有效权重与当前代码一致：

    lambda_eff = min(
        lambda_req,
        rho * stopgrad(L_seg) / max(stopgrad(L_tss), epsilon)
    )

其中：

    lambda_req = 候选正权重
    rho        = 0.10
    epsilon    = 当前 FP32 dtype 的 machine epsilon

总损失：

    L_total = L_seg + lambda_eff * L_tss

准确的梯度说明：

- 只有计算动态系数时使用 stop-gradient；
- 原始 Lseg 和 Ltss 仍保留正常计算图；
- lambda_eff 本身不接收梯度；
- TSS loss 仍通过共享特征向网络反向传播；
- 当 cap 生效时，TSS 加权损失不超过主分割损失的 10%。

原文使用 Ltss + epsilon 的写法与实现不完全一致，现已改为 max(Ltss, epsilon)。

## 2.4 历史日志与新 runner 诊断范围

已停止的 four-dataset V2 历史日志每个 epoch 只记录：

- requested TSS weight；
- ratio cap；
- effective weight 的样本加权均值；
- weighted TSS loss；
- weighted TSS loss 与 segmentation loss 的聚合比值；
- cap-active 的样本加权比例。

这些历史日志没有保存：

- 每个 batch 的原始比值；
- effective weight 的 p10、p50、p90、std、max；
- 每个 batch 的 Lseg 与 Ltss 配对值；
- 三个候选 λ 的逐 batch 反事实有效权重。

因此不能从旧日志精确补算 0.0025、0.005、0.01 的 batch 级反事实覆盖率，不能把估算值写成真实结果。新 `train_three_dataset_seed42_global_tss_v2.py` 已实现逐 batch 的 Lseg/Ltss、原始/有效比值、effective weight p10/p50/p90/std/max、cap-active 和三候选反事实有效权重；这些字段只从新 12-run 产物中读取。

---

# 3. V2 三独立数据集结果复盘

所有差值均为 V2 减 Original。

## 3.1 best_miou checkpoint

| 数据集 | ΔmIoU | ΔnIoU | Δmatched targets | ΔFa | Δtiny targets |
|---|---:|---:|---:|---:|---:|
| NUAA-SIRST | +0.002120 | −0.002587 | −4 | −35.4% | −3 |
| NUDT-SIRST | −0.001069 | +0.000338 | +4 | +127.5% | 0 |
| IRSTD-1K | −0.002556 | +0.013311 | +1 | +34.8% | −1 |

三个独立数据集聚合：

    macro ΔmIoU         = -0.000502
    macro ΔnIoU         = +0.003687
    matched target 总差 = +1
    tiny target 总差    = -4
    aggregate ΔFa       = +24.5%

结论：

- mIoU 基本持平；
- nIoU 与总 matched targets 略有提高；
- aggregate Fa 明显升高；
- tiny target 总检出减少；
- 该 checkpoint 角色没有形成跨指标整体提升。

## 3.2 best_pd checkpoint

| 数据集 | ΔmIoU | ΔnIoU | Δmatched targets | ΔFa | Δtiny targets |
|---|---:|---:|---:|---:|---:|
| NUAA-SIRST | +0.052694 | +0.044125 | −2 | −60.1% | −3 |
| NUDT-SIRST | +0.025604 | +0.019680 | −1 | −58.9% | 0 |
| IRSTD-1K | +0.039345 | +0.031387 | +1 | −42.8% | +1 |

三个独立数据集聚合：

    macro ΔmIoU         = +0.039214
    macro ΔnIoU         = +0.031730
    matched target 总差 = -2
    tiny target 总差    = -2
    aggregate ΔFa       = -49.7%

结论：

- mIoU、nIoU 和 Fa 明显改善；
- 最大 matched target 数量轻微下降；
- tiny target 总检出轻微下降；
- IRSTD-1K 是最完整的正向点；
- NUAA 和 NUDT 属于非支配混合权衡，不能称为单向全面提高。

## 3.3 V2 相对 V1 Final

三个独立数据集聚合差值：

| checkpoint 角色 | macro ΔmIoU | macro ΔnIoU | Δmatched | Δtiny | aggregate ΔFa |
|---|---:|---:|---:|---:|---:|
| best_miou | −0.002052 | −0.005816 | +6 | −2 | +29.9% |
| best_pd | +0.015230 | +0.012093 | +2 | 0 | −36.1% |

这说明动态 cap 确实改变了训练轨迹：

- 在 best_pd 角色上总体更有利；
- 在 best_miou 角色上出现更多检出与更高 Fa 的交换；
- 不能仅依据其中一个 checkpoint 宣布 V2 成功或失败。

## 3.4 SIRST3 历史结果

SIRST3 只作为历史附注：

| checkpoint 角色 | ΔmIoU | ΔnIoU | Δmatched | ΔFa | Δtiny |
|---|---:|---:|---:|---:|---:|
| best_miou，部分 | +0.000906 | +0.001922 | +8 | +25.6% | +3 |
| best_pd，部分 | −0.006502 | −0.007851 | −4 | +2.2% | −2 |

这些数值来自停止于 epoch 742 的部分训练，不参与本方案任何后续选择。

## 3.5 当前研究裁决

正确裁决为：

    decision=INCONCLUSIVE_MIXED_TRADEOFF
    v2_global_recipe_established=false
    v2_universal_dominance=false
    final_model_performance_established=false
    paper_core_established=false
    stability_claim_supported=false

该裁决不表示模型结构失败，而表示：

- V2 有正向工作区间；
- 当前 λ=0.005、cap=0.10 尚未形成三数据集统一的整体优势；
- 下一步应优化和选择训练配方；
- 不应新增网络模块来掩盖现有训练配方问题。

---

# 4. checkpoint 选择规则

## 4.1 只保留两个既定角色

每个方法、数据集和 λ 独立保存：

    best_miou.pth.tar
    best_pd.pth.tar

明确不新增：

    best_joint.pth.tar

best_miou 与 best_pd 分别表示区域质量优先端点和目标检出优先端点。新增第三种 selector 只会改变评价目标，不能解决训练配方本身的混合权衡。

## 4.2 best_miou 固定字典序

只读取对应数据集 img_idx/test 的 threshold=0.5 指标：

    1. 最大 mIoU
    2. 最大 Pd
    3. 最小 Fa
    4. 最大 nIoU
    5. 最大 tiny-Pd
    6. 最小 test loss
    7. 更早 epoch

等价 key：

    (mIoU, Pd, -Fa, nIoU, tiny-Pd, -test_loss, -epoch)

## 4.3 best_pd 固定字典序

只读取对应数据集 img_idx/test 的 threshold=0.5 指标：

    1. 最大 Pd
    2. 最小 Fa
    3. 最大 tiny-Pd
    4. 最大 mIoU
    5. 最大 nIoU
    6. 最小 test loss
    7. 更早 epoch

等价 key：

    (Pd, -Fa, tiny-Pd, mIoU, nIoU, -test_loss, -epoch)

## 4.4 单 run 协议一致性与搜索预算差异

- Original 选择自己的 best_miou 与 best_pd。
- 每个 Final λ 选择自己的 best_miou 与 best_pd。
- 不要求 Original 与 Final 使用相同 epoch。
- 不允许把 baseline checkpoint 来源强行设成与 Final 相同。
- 不允许跨 checkpoint 拼接指标。
- 一行结果必须来自同一 checkpoint、同一阈值、同一 test 列表。
- best_miou 与 best_pd 可以偶然选择同一 epoch；仍保留两个角色记录。
- 每个 checkpoint 保存 epoch、选择 key、test 指标、权重摘要和源码版本。

准确的比较表述是：

- 单个 Original 与单个 Final run 使用相同数据、seed、epochs、optimizer、scheduler、增强、评估频率和阈值，属于单 run 协议一致；
- Original 只有 3 个训练 run；
- Final 因搜索三个正 λ，共有 9 个训练 run；
- Final 的训练搜索预算是 Original 的 3 倍，且 TSS 还会增加少量训练期计算；
- 因此不能写“Original 与 Final 的总搜索预算完全公平”；
- 必须同时报告单 run GPU-hours、Original 总 GPU-hours、Final 搜索总 GPU-hours和峰值显存。

固定披露：

    per_run_protocol_matched=true
    original_training_runs=3
    final_training_runs=9
    final_to_original_run_budget_ratio=3.0
    total_search_budget_equal=false

## 4.5 评估日程和保存范围

评估 epoch：

    10, 20, 30, ..., 1000

每个完整 run 共 100 个候选点。

长期保存：

- best_miou.pth.tar；
- best_pd.pth.tar；
- metrics.jsonl；
- protocol.json；
- train/test img_idx 摘要；
- summary.json；
- 运行版本记录。

滚动 resume 每个 epoch 覆盖保存，训练成功结束后可按协议移除。它不是第三个 selected checkpoint。

---

# 5. img_idx 数据协议与 NUAA 修正

## 5.1 唯一允许的数据路径

每个数据集只能读取：

    datasets/<dataset>/img_idx/train_<dataset>.txt
    datasets/<dataset>/img_idx/test_<dataset>.txt
    datasets/<dataset>/images
    datasets/<dataset>/masks

唯一例外是 NUAA-SIRST::Misc_111：该精确样本 ID 必须读取第 5.3 节规定路径与摘要的 masks_corrected overlay。除此之外，任何样本都不得读取 masks_corrected。

新 runner 的 CLI 只接受：

    dataset in {NUAA-SIRST, NUDT-SIRST, IRSTD-1K}

出现 SIRST3 即拒绝运行。

## 5.2 img_idx 完整性检查

启动前完整性预检必须验证：

- train 与 test 样本 ID 无交集；
- 每个列表无重复 ID；
- 所有 ID 都能解析到图像与 mask；
- 样本顺序与文件一致；
- 图像和 loader 解析后的 effective mask 尺寸一致；
- Misc_111 的 592 x 400 raw mask 只允许作为保留原件，不得进入指标计算；
- Original 与三个 Final λ 使用完全相同的 train/test 列表；
- 训练统计只读取 train；
- test 不进入 normalization、TSS pos_weight 或梯度更新。

上述预检已经由 three-dataset v2 launcher 完成：共检查 2,755 个
train/test 图像-mask 对，缺失与尺寸错误均为 0；正式 launcher 会在每个
wave 开始前继续复核协议文件与预检产物摘要。

每个 run 记录：

    dataset
    train_img_idx_path
    train_img_idx_sha256
    train_count
    test_img_idx_path
    test_img_idx_sha256
    test_count
    ordered_train_ids_digest
    ordered_test_ids_digest

## 5.3 NUAA Misc_111 固定修正

NUAA test 列表第 91 个样本为 Misc_111。

已核对：

    image:             325 x 220
    raw NUAA mask:     592 x 400
    corrected mask:    325 x 220

固定摘要：

    image_sha256
      = 72561a22b2d1e09a167563f1f3dab7ee04153aabd87579df749ca15ecf3e60b1

    raw_mask_sha256
      = 1bec16e5b0413d08f5b01c70faac97c72454586b03d10129fde778db4194a4aa

    corrected_mask_sha256
      = 7e20ff7267737f367d2ea0545289152710225fe871d7c34c34b2d97c66b06fff

三数据集 runner 不再依赖 SIRST3 路径。已验证修正 mask 的相同字节已放入 NUAA 内部 overlay：

    datasets/NUAA-SIRST/masks_corrected/Misc_111.png

规则：

- 保留原始 592 x 400 mask，不覆盖、不删除；
- loader 对 Misc_111 只读取摘要匹配的 325 x 220 修正 mask；
- 不允许运行时左上裁剪；
- 不允许运行时 resize；
- 图像与修正 mask 必须逐项尺寸一致；
- overlay 路径和三项摘要写入 protocol.json；
- 其他 NUAA 样本仍读取原 masks 目录。

历史 four_dataset correction manifest 仍保留旧记录，但不再是正式输入。以下四项已落地：

1. 已创建上述 NUAA 内部同字节 overlay；
2. 已生成只包含三数据集的 v2 protocol manifest；
3. 新 loader 已只解析 NUAA overlay；
4. 正式数据绑定已结构化限定为 NUAA、NUDT 和 IRSTD-1K。

该修正只处理 NUAA 的一个既有 test 样本，不代表使用 SIRST3 数据集进行实验。

---

# 6. 三数据集全局 TSS 配方选择

## 6.1 固定搜索范围

    seed=42
    epochs=1000
    test_interval=10
    threshold=0.5
    precision=FP32
    ratio_cap=0.10

只比较：

    lambda_req in {0.0025, 0.005, 0.01}

本轮不同时修改：

- ratio cap；
- optimizer；
- 基础学习率或 scheduler；
- augmentation；
- batch size 或 patch size；
- 数据集专用 λ；
- 模型结构；
- λ=0 机制对照。

Original 是性能基准，不属于 λ 候选。

## 6.2 训练矩阵

| 数据集 | Original | Final-0.0025 | Final-0.005 | Final-0.01 |
|---|---:|---:|---:|---:|
| NUAA-SIRST | 1 | 1 | 1 | 1 |
| NUDT-SIRST | 1 | 1 | 1 | 1 |
| IRSTD-1K | 1 | 1 | 1 | 1 |

总计：

    Original runs = 3
    Final runs    = 9
    total runs    = 12

12 个 run 全部 fresh training，固定 seed42。旧结果只作历史复盘，不替代新配方矩阵中的任何 run。

九个 Final run 完成前，不允许依据先完成的数据集提前冻结 λ。

## 6.3 全局选择输入

对每个 λ，读取三个数据集各自：

    best_miou: mIoU, nIoU, matched, Fa, tiny matched
    best_pd:   mIoU, nIoU, matched, Fa, tiny matched

数据集、checkpoint 角色和指标均等权：

    dataset_equal_weight=true
    checkpoint_role_equal_weight=true
    metric_equal_weight=true
    image_count_weighting=false
    target_count_weighting=false

尽管三个数据集 test 数量不同，λ 排名不按图像数或目标数加权，避免 NUDT 单独主导统一配方。

## 6.4 三个 λ 之间的排名

统一相等判定：

- 对 mIoU、nIoU 定义 q(x)=floor(x/0.0001+0.5)，按 q 值比较，q 相同才并列；
- Pd 使用 matched target 整数计数，越高越好；
- tiny-Pd 使用 tiny matched target 整数计数，越高越好；
- Fa 使用 unmatched predicted pixels 整数计数，越低越好；
- 并列使用平均名次；
- 第一名为 1，第二名为 2，第三名为 3。

对数据集 d：

    R_dataset(d, λ)
      = 两个 checkpoint 角色、五项指标的平均候选名次

    R_worst(λ)
      = 三个 R_dataset 中的最大值

    R_macro(λ)
      = 三个 R_dataset 的算术平均

名次永远在全部三个预注册候选 `{0.0025,0.005,0.01}` 上先一次性计算；严重退化门和 Original 覆盖门只改变后续的 eligibility/Pareto 集合，不剔除失败候选后重算剩余候选的名次。固定字段：

    rank_population=all_three_preregistered_candidates_before_eligibility_gates

每个候选还保留三数据集、双角色、五指标的完整方向统一 rank 向量，共 30 个单元。任何全局选择都只使用这些等权 rank 和后续 Pareto 关系，不对 mIoU、nIoU、Pd、Fa、tiny-Pd 的原始数值直接求和。

## 6.5 相对 Original 的严重退化保护

每个 λ、数据集和 checkpoint 角色先分别检查：

1. matched target 不得比 Original 少 2 个或更多；
2. tiny matched target 不得比 Original 少 2 个或更多；
3. mIoU 量化后必须满足 `q_original-q_final<50`；`>=50` 即触发退化门；
4. nIoU 量化后必须满足 `q_original-q_final<50`；`>=50` 即触发退化门；
5. Fa 使用同一数据集的 unmatched predicted pixels 整数比较；仅当 `final_pixels*4 > original_pixels*5` 时触发，正好 125% 不触发；触发时 matched target 必须至少多 2 个；
6. Original Fa 为 0 而 Final Fa 大于 0 时，同样按第 5 条处理。

任一项不满足，该 λ 不进入全局可选集合，不能由其他指标的微小改善抵消。

## 6.6 相对 Original 的等权 vote

方向：

    mIoU       higher
    nIoU       higher
    matched    higher
    Fa pixels  lower
    tiny match higher

计分：

- mIoU、nIoU 使用与候选排名相同的 q(x)，q 相同为平；
- matched 与 tiny matched 使用整数计数；
- Fa 只按 unmatched predicted pixels 投票；
- Fa rate 是派生展示值，不重复计票；
- 改善 +1，回退 -1，平 0。

    S_role(d, role, λ)
      = 五项 vote 之和

    S_dataset(d, λ)
      = S_role(d, best_miou, λ) + S_role(d, best_pd, λ)

上述 vote 只在 Pareto 过滤后作为第三顺位的并列裁决，不额外设置“单数据集 vote 必须非负”或“多数数据集 vote 必须为正”的准入门。这样全局准入仍由预注册的严重退化门、Original 双角色严格覆盖门与 30 维等权 rank Pareto 共同决定，而 vote 不能越过 rank/Pareto 改变准入集合。

唯一的 Original 准入条件是：任一数据集不能在两个 checkpoint 角色上都被 Original 严格覆盖。

严格覆盖定义：

> 按上述量化值和整数计数，Original 在五项指标上全部不差，并且至少一项严格更好。

若没有候选通过：

    global_tss_recipe_established=false
    decision=NO_POSITIVE_GLOBAL_TSS_RECIPE_ESTABLISHED

不能为了继续流程而强行指定 λ。

## 6.7 等权 rank Pareto 过滤

严重退化保护与 Original 准入条件通过后，对候选的 30 维方向统一 rank 向量执行 Pareto 过滤。

候选 A 覆盖候选 B，当且仅当：

- A 在 30 个 rank 单元上均不差于 B；
- 至少一个 rank 单元严格优于 B。

被覆盖候选不能成为全局 λ。输出必须记录：

    pareto_eligible
    pareto_dominated
    pareto_dominated_by
    rank_vector

Pareto 只使用等权 rank，不使用原始指标和，也不按图像数、目标数或数据集规模加权。

## 6.8 唯一 λ 选择顺序

在通过全部条件且位于 Pareto 集合的候选中：

    1. 最小 R_worst
    2. 最小 R_macro
    3. 最大的三数据集 S_dataset 总和
    4. 更小 lambda_req

输出：

    global_tss_lambda
    candidate_ranking
    per_dataset_rank
    rank_vector
    pareto_dominated_by
    per_dataset_role_vote
    per_dataset_total_vote
    selection_split=img_idx/test
    test_selected=true

选择完成后不能根据某个单独指标再次更改 λ。

---

# 7. 阈值和指标

## 7.1 唯一主阈值

所有 checkpoint 选择、λ 选择和主结果固定：

    threshold=0.5

不单独保留 calibration 数据，不选择部署阈值，不通过改阈值补偿模型性能。

threshold=1.0 仅允许出现在描述性 Pd-Fa sweep 中，禁止参与 best_miou、best_pd、全局 λ 或主结果选择。

## 7.2 主结果必须同时报告

    mIoU
    nIoU
    matched target count / total target count
    Pd
    unmatched predicted pixel count
    Fa
    unmatched predicted object count
    false objects per image
    matched tiny target count / total tiny target count
    tiny-Pd

不能只看 mIoU，也不能只看 Pd。

若某个 test 没有 tiny GT，则该数据集的 tiny 指标为 NA，并从三个 λ 的相应排名与 vote 中共同剔除，分母同步减少；不能把 NA 编码为 0 或 -1。启动前审计已确认三个 test 均定义 tiny 指标：NUAA-SIRST、NUDT-SIRST、IRSTD-1K 的面积不超过 9 像素 tiny GT 数量分别为 35、259、30。

## 7.3 描述性 Pd-Fa sweep

sweep 只用于展示同一 checkpoint 的工作区间，不回写 checkpoint 或 λ。

闭区间包含 threshold=1.0。判定为 probability > threshold 时，threshold=1.0 是空预测：

    Fa=0
    Pd=0

若预算下没有非空预测点满足条件，输出：

~~~json
{
  "budget": 1e-6,
  "pd_at_fa_budget": 0.0,
  "selected_threshold": 1.0,
  "selected_point_is_empty": true,
  "registered_grid_nonempty_feasible": false,
  "best_nonempty_point": null
}
~~~

连通域和目标匹配随阈值可能非单调，因此 sweep 是描述性 threshold-component envelope，不替代 threshold=0.5 主表。

---

# 8. 推理缓存与三数据集聚合

## 8.1 每个 checkpoint 只推理一次

缓存键：

    dataset
    ordered test img_idx digest
    model state digest
    model source digest
    normalization digest
    preprocessing digest
    effective test mask manifest digest
    Misc_111 overlay path and corrected mask digest
    mask resolution rule source digest
    checkpoint role
    epoch
    dtype

同一 float32 probability cache 用于：

- threshold=0.5；
- Pd-Fa sweep；
- component analysis；
- 论文表格。

## 8.2 聚合口径

每个数据集先单独报告完整数值。

同时提供：

- macro mIoU；
- macro nIoU；
- macro Pd；
- macro Fa；
- macro tiny-Pd；
- pooled matched / pooled targets；
- pooled tiny matched / pooled tiny targets；
- pooled unmatched pixels / pooled valid pixels；
- total unmatched predicted objects；
- total false objects / total test images。

宏平均体现数据集等权；pooled 指标反映全部样本总量。两者同时报告，不能只报告受 NUDT 样本量影响的 pooled 数值。

---

# 9. 代码修改范围

## 9.1 模型结构保持不变

保持只读：

    model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py
    model/tpd_frequency_gate_v2_croa.py
    model/tpd_survival.py
    model/tpd_relay.py

TSS 实际计算已经使用 clamp_min(epsilon)，不改核心 loss。

以下三处 Ltss + eps 的说明文字已经修正为与 clamp_min 实现一致：

- experiments/tpd_training_loss.py 的模块公式说明；
- experiments/train_four_dataset_final_seed42_tss_cap_v2.py 的模块说明；
- V2 runner 写入 protocol.json 的公式元数据。

## 9.2 新 experiments 入口

实际实现：

    experiments/three_dataset_v2_protocol.py
    experiments/prepare_nuaa_misc111_overlay_v2.py
    experiments/paper_three_dataset_v2.py
    experiments/train_three_dataset_seed42_global_tss_v2.py
    experiments/three_dataset_seed42_launch_v2.py
    experiments/evaluate_three_dataset_v2.py
    experiments/select_three_dataset_global_tss_recipe_v2.py

旧 four-dataset runner 只复用通用的 optimization/checkpoint loop，不再提供本轮数据集矩阵、数据 manifest、运行目录或全局 λ 身份。

明确不新增：

    build_three_dataset_roles
    select_model_val_best_joint
    launch_sirst3_positive_tss_search

历史 V2 runner 将 requested weight 固定为 0.005，不能原样承担三候选训练。新 Final runner 已实现：

- 只接受 0.0025、0.005、0.01 三个候选值；
- 把候选值写入运行身份、protocol、checkpoint 和 resume；
- loss、cap-active 判定和全部 TSS 日志读取同一个冻结候选值；
- resume 时拒绝候选值不一致的状态；
- 禁止运行中通过 CLI 或配置文件改变候选值。

## 9.3 新训练 TSS 诊断

每个 epoch 记录：

    requested_weight
    effective_weight_mean
    effective_weight_p10
    effective_weight_p50
    effective_weight_p90
    effective_weight_std
    effective_weight_max
    raw_weighted_to_seg_ratio_mean
    effective_weighted_to_seg_ratio_mean
    cap_active_batch_fraction
    cap_active_sample_fraction

这些字段只解释训练强度，不参与 checkpoint 选择。

## 9.4 checkpoint 与 resume

resume 必须保存或确定性重建：

- model state；
- optimizer state；
- completed epoch；
- RNG state；
- epoch 对应训练顺序；
- 当前 best_miou；
- 当前 best_pd；
- metrics 边界；
- img_idx 摘要；
- 运行身份。

学习率按 completed epoch 确定性计算，不虚构独立 scheduler state。

---

# 10. 测试要求

## 10.1 img_idx

必须验证：

- 六个固定 img_idx 文件摘要一致；
- train/test 数量为 213/214、663/664、800/201；
- 每个列表无重复；
- train/test 无交集；
- 顺序没有变化；
- Original 和三个 Final λ 输入完全一致；
- SIRST3 输入被拒绝。

## 10.2 Misc_111

必须验证：

- NUAA test 第 91 个 ID 为 Misc_111；
- image 为 325 x 220；
- raw mask 为 592 x 400；
- corrected overlay 为 325 x 220；
- 三项摘要匹配；
- loader 读取 corrected overlay；
- raw mask 未被覆盖；
- 不执行 crop 或 resize 修正。

## 10.3 checkpoint selector

必须验证：

- best_miou 每一级 tie；
- best_pd 每一级 tie；
- 每个方法独立选模；
- 只读取对应 img_idx/test；
- 不生成 best_joint；
- 不跨 checkpoint 拼指标；
- 最终 tie 选择更早 epoch；
- 普通 Python 与 python -O 一致。

## 10.4 TSS loss

必须验证：

- ratio_cap=None 保持历史行为；
- cap inactive 时 effective weight 等于 requested λ；
- cap active 时有效比例不超过 0.10；
- effective weight 不接收梯度；
- TSS 对共享特征保留梯度；
- 六项 segmentation loss 顺序不变；
- denominator 与 clamp_min 实现一致；
- FP32 finite。

## 10.5 全局 λ selector

必须验证：

- 只接受三个固定正候选；
- 三数据集等权；
- 两 checkpoint 角色等权；
- 五指标方向正确；
- Fa 只按 unmatched predicted pixels 投票；
- 排名与 vote 使用相同的 0.0001 量化函数；
- 严重退化保护先于 vote；
- Pareto 只读取 30 维等权 rank 向量；
- selector 不直接求原始指标和；
- 输出每个被覆盖候选及其覆盖者；
- 没有候选通过时不强选；
- tie 最终选择更小 λ；
- 输入来源只能是三个 img_idx/test。

## 10.6 推理导出

必须验证：

- 训练 checkpoint 含 4 个 target_survival.* 状态张量键，共 98 个标量参数；
- 部署导出删除这 4 个状态张量键；
- 导出前后 segmentation 输出逐元素一致；
- TPD、NER、QFG 参数完整加载；
- 缓存与直接评估指标一致。

---

# 11. 执行阶段

## Phase 0：历史结果封存

状态：

    completed

保留：

- 三数据集 Original、V1、V2 summary；
- best_miou、best_pd；
- metrics.jsonl；
- threshold sweep；
- SIRST3 停止记录；
- 当前结果总表。

## Phase 1：img_idx 与 Misc_111 协议

状态：

    completed

任务：

1. 固定六个 img_idx 文件摘要；
2. 检查列表数量、顺序、重复与交集；
3. 建立 NUAA 内部版本化 Misc_111 overlay；
4. 输出三数据集协议文件；
5. 确认未来入口不接受 SIRST3。

完成条件：

    img_idx_protocol_ready=true
    nuaa_misc111_overlay_ready=true

## Phase 2：runner 与 selector

启动所需核心实现状态：

    completed_and_tested

任务：

1. 继承现有每 10 epoch 测试；
2. 只保留 best_miou、best_pd；
3. 实现三个固定正 λ；
4. 增加 TSS 诊断；
5. 实现三数据集等权 λ selector；
6. 实现独立 three-dataset v2 evaluator；
7. 实现 threshold=0.5 主评估和带空预测标志的 Pd-Fa sweep；
8. 同一次 evaluator 调用只推理一次，并复用同一组 float32 概率数组生成固定 0.5 指标和描述性 sweep；
9. 完成全部测试。

12-run 完成后的 24 份 checkpoint 批量评估、selector 输入组装和总表导出
仍需由后训练编排入口完成。该入口不改变已经冻结的 evaluator、selector、
阈值语义或 λ 算法，因此不阻止训练启动，但必须在 Phase 4 读取结果前完成。

真实 GPU 冒烟已在物理 GPU 2/3 上分别完成一条 Original 和一条
Final(lambda=0.0025) 的 1-epoch 极小样本训练；两条链路均成功执行前向、
反向、测试并仅保存 best_miou 与 best_pd。冒烟结果不进入性能表。

完成条件：

    implementation_tests_passed=true
    best_joint_absent=true
    sirst3_future_path_absent=true

## Phase 3：12 个 seed42 formal1000

全部 fresh：

    3 Original
    9 Final

每个 run：

    exact img_idx train/test
    seed 42
    1000 epochs
    test every 10 epochs
    threshold 0.5
    own best_miou
    own best_pd

九个 Final 都会访问各自 img_idx/test，因为 test 同时用于 checkpoint 与 λ 选择。该事实必须写入结果说明。

## Phase 4：全局 λ 与最终汇总

读取 12 个 run 的 img_idx/test 结果。

### 有候选通过

    global_tss_recipe_established=true
    global_tss_lambda=<selected>

冻结：

- 一个统一 λ；
- ratio cap；
- 三个 Final best_miou 与三个 Final best_pd，共 6 份 Final 权重；
- 三个 Original best_miou 与三个 Original best_pd，共 6 份对照权重；
- threshold=0.5。

输出选定 λ 的三数据集正式对比，同时保留另外两个 λ 的配方搜索表。

### 无候选通过

    global_tss_recipe_established=false
    decision=NO_POSITIVE_GLOBAL_TSS_RECIPE_ESTABLISHED

如实报告三候选结果，不修改模型结构，不强行定型全局 TSS。

---

# 12. 结果表模板

## 12.1 best_miou，threshold=0.5

| Dataset | Method | Epoch | mIoU ↑ | nIoU ↑ | matched/total ↑ | Pd ↑ | Fa ↓ | false objects ↓ | tiny matched/total ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUAA | Final | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUDT | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUDT | Final | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| IRSTD | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| IRSTD | Final | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 12.2 best_pd，threshold=0.5

| Dataset | Method | Epoch | mIoU ↑ | nIoU ↑ | matched/total ↑ | Pd ↑ | Fa ↓ | false objects ↓ | tiny matched/total ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUAA | Final | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUDT | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NUDT | Final | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| IRSTD | Original | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| IRSTD | Final | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 12.3 λ 排名

| λ | R_NUAA | R_NUDT | R_IRSTD | R_worst | R_macro | S_total | Eligible |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.0025 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.005 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.01 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## 12.4 相对 Original vote

| λ | Dataset | S_best_miou | S_best_pd | S_dataset | Severe degradation | Strictly dominated in both roles |
|---:|---|---:|---:|---:|---|---|
| 0.0025 | NUAA | TBD | TBD | TBD | TBD | TBD |
| 0.0025 | NUDT | TBD | TBD | TBD | TBD | TBD |
| 0.0025 | IRSTD | TBD | TBD | TBD | TBD | TBD |
| 0.005 | NUAA | TBD | TBD | TBD | TBD | TBD |
| 0.005 | NUDT | TBD | TBD | TBD | TBD | TBD |
| 0.005 | IRSTD | TBD | TBD | TBD | TBD | TBD |
| 0.01 | NUAA | TBD | TBD | TBD | TBD | TBD |
| 0.01 | NUDT | TBD | TBD | TBD | TBD | TBD |
| 0.01 | IRSTD | TBD | TBD | TBD | TBD | TBD |

所有 TBD 必须来自真实运行，不补造数值。

---

# 13. 结论边界

## 13.1 当前可以写

> 在固定 seed42 和数据集既有 img_idx train/test 协议下，动态 TSS V2 改变了完整模型的性能工作区间：best_pd checkpoint 上区域质量和虚警整体改善，而 best_miou checkpoint 上存在检出、tiny-target 与虚警之间的混合权衡。因此后续保持架构冻结，只比较三个统一正 TSS 候选，并继续使用 best_miou 与 best_pd 两个既定 checkpoint 角色。

## 13.2 当前不能写

    V2 在三个数据集全面优于 Original
    lambda=0.005 已是全局最优
    TSS 已在所有指标上建立优势
    SIRST3 已完成 formal1000
    img_idx/test 是未参与选择的独立确认集
    多随机种子稳定性已经建立
    最终模型论文核心已经建立

## 13.3 新实验完成后的允许表述

若统一 λ 通过三数据集准入条件，可写：

> 在固定 seed42、固定 threshold=0.5 和既有 img_idx train/test 协议下，同一正 TSS 配方在 NUAA-SIRST、NUDT-SIRST 与 IRSTD-1K 上表现出整体竞争力。

仍需附加：

> checkpoint 和 λ 均依据 img_idx/test 结果选择，结论限于该协议下的模型比较，不构成独立测试确认或跨随机性稳定性证据。

---

# 14. 最终状态与执行结论

    decision=FORMAL12_LAUNCH_AUTHORIZED

    architecture_implementation_complete=true
    architecture_frozen=true
    innovation_mainline_changed=false
    new_module_design_authorized=false

    data_protocol=existing_img_idx_train_test
    internal_resplit=false
    checkpoint_selection_split=img_idx_test
    global_lambda_selection_split=three_img_idx_tests
    img_idx_test_selected_protocol=true

    future_datasets=[NUAA-SIRST,NUDT-SIRST,IRSTD-1K]
    sirst3_future_role=historical_only

    primary_seed=42
    epochs=1000
    test_interval=10
    benchmark_threshold=0.5
    threshold_calibration=false

    checkpoint_roles=[best_miou,best_pd]
    best_joint_selector_required=false
    cross_checkpoint_metric_stitching=false

    positive_tss_candidates=[0.0025,0.005,0.01]
    survival_ratio_cap=0.10
    dataset_specific_tss=false

    planned_original_runs=3
    planned_final_runs=9
    planned_total_runs=12
    per_run_protocol_matched=true
    final_to_original_run_budget_ratio=3.0
    total_search_budget_equal=false

    three_dataset_v2_protocol_implemented=true
    evaluator_and_selector_implemented=true
    implementation_tests_passed=true
    indexed_pair_preflight_passed=true
    indexed_pair_count=2755
    indexed_pair_error_count=0
    tiny_gt_counts=[35,259,30]
    real_gpu_smoke_passed=true
    training_launch_authorized=true
    formal_runtime_state_source=results/three_dataset_seed42_global_tss_v2/launch/formal/launch_plan.json

    current_v2_decision=INCONCLUSIVE_MIXED_TRADEOFF
    v2_global_recipe_established=false
    final_model_performance_established=false
    paper_core_established=false
    stability_claim_supported=false
    training_recipe_finalized=false
    formal_training_started_at_protocol_freeze=false
    posttraining_batch_orchestrator_complete=false

一句话执行结论：

> 后续严格使用 NUAA、NUDT 和 IRSTD-1K 各自现有 img_idx/train 与 img_idx/test，不再内部重划分；固定 seed42、1000 epochs、每10 epochs在 test 上选择各方法自己的 best_miou 和 best_pd，由三个数据集 test 结果等权选择统一正 TSS 配方，并明确所有结果属于 img_idx/test-selected 协议。
