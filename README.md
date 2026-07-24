# TPD-SCTransNet：面向红外小目标的目标保真下采样实验

本仓库在 [SCTransNet](https://github.com/xdFai/SCTransNet) 基线上研究浅层
patch embedding 的目标保真下采样（Target-Preserving Downsampling, TPD）。
核心改动只替换 `mtc.embeddings_1` 和 `mtc.embeddings_2`；编码器、SCTB、
解码器、损失函数和输出接口保持不变，以便进行受控比较。

> 当前结论来自 **NUDT-SIRST 官方训练集的内部验证划分**，未访问官方测试集。
> 正式实验仅使用一个随机种子，因此不能据此声称 TPD 已稳定优于所有对照。

## 方法

TPD 将每个 2× 下采样单元拆成三个对齐分支：

- `keep`：`pixel_unshuffle` 后按通道压缩，保留相位信息；
- `context`：平均池化，保留局部背景上下文；
- `saliency`：最大池化减平均池化，突出局部小目标响应。

三个分支拼接后通过 `1×1` 卷积融合。大步长投影由多个 2× TPD 单元逐级完成。
实现见 [`model/tpd.py`](model/tpd.py)。

受控实验包含四个变体：

| 变体 | 说明 |
|---|---|
| `original` | 原始 SCTransNet 大步长 patch embedding |
| `progressive` | 多级 stride-2 卷积，同深度结构对照 |
| `spd` | `pixel_unshuffle + 1×1` 的 SPD 对照 |
| `tpd` | 相位保留、上下文和显著性三分支融合 |

## 正式实验设置

- 数据集：NUDT-SIRST
- 数据范围：仅官方训练集，共 663 张图像
- 内部划分：530 张训练、133 张验证，划分种子 `20260722`
- 训练：800 epochs，FP32，batch size 16，patch size 256
- 模型随机种子：42
- 优化设置：初始学习率 `1e-3`，最低学习率 `1e-5`，10 epochs warmup
- 主检查点：验证集 Pd 最大；并列时依次选择更低 Fa、更高 tiny-Pd、
  更高 mIoU 和更低验证损失
- tiny target：面积不超过 9 pixels
- 匹配半径：3 pixels

四个变体共享相同的非 embedding 初始化、数据划分和训练协议。每个变体均完成
800 epochs，事件流、检查点角色、模型严格加载、指标恒等式及 Pd–Fa sweep
均通过完整性检查。

## 实验结果

### Pd 主指标检查点

| 变体 | Epoch | Pd ↑ | tiny-Pd ↑ | Fa ↓ | 错误目标/图 ↓ | mIoU ↑ | nIoU ↑ | 参数量 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original | 456 | **0.994709** | 1.000000 | 1.4226e-5 | 0.1278 | 0.919086 | 0.923178 | 11,325,939 |
| Progressive | 300 | 0.989418 | 1.000000 | 1.0325e-6 | 0.0451 | 0.914348 | 0.914485 | 10,924,755 |
| SPD | 619 | 0.989418 | 1.000000 | **0** | **0** | **0.946542** | **0.939837** | 10,842,835 |
| TPD | 337 | **0.994709** | 1.000000 | 1.0325e-6 | 0.0226 | 0.933647 | 0.930339 | **10,827,731** |

在主检查点上，TPD 与 Original 同样检出 `188/189` 个目标，同时将 Fa 从
`1.4226e-5` 降至 `1.0325e-6`（约降低 13.8 倍），并将 mIoU 从
`0.919086` 提升至 `0.933647`。SPD 的 Pd 略低（`187/189`），但实现了零 Fa，
且 mIoU 最高。

### mIoU 次指标检查点

| 变体 | Epoch | mIoU ↑ | Pd | Fa ↓ |
|---|---:|---:|---:|---:|
| Original | 726 | 0.940738 | 0.984127 | 1.9504e-6 |
| Progressive | 611 | 0.931854 | 0.984127 | 2.1798e-6 |
| SPD | 470 | **0.949145** | **0.989418** | 4.5891e-7 |
| TPD | 457 | 0.942758 | 0.984127 | 4.5891e-7 |

### Pd–Fa 筛选结论

在五个预设 Fa budget（`1e-6`、`5e-6`、`1e-5`、`5e-5`、`1e-4`）上：

- TPD 在全部五个 budget 上优于 Original 和 Progressive；
- TPD 在后四个 budget 上优于 SPD；
- 在最严格的 `1e-6` budget 上，TPD 和 SPD 均检出 `187/189` 个目标，
  但 SPD 的实际 Fa 为 0，因此 SPD 更优；
- TPD 拥有一个独占的联合 Pd–Fa Pareto 点，但不是所有预算下的统一最优方法。

保守决策为 **`INCONCLUSIVE_MIXED_TRADEOFF`**：TPD 有潜力且不被支配，
但当前证据不足以将其确立为稳定主线。后续需要多随机种子、多数据集和官方测试集
评估。`paper_core_established=false`，`stability_claim_supported=false`。

## 数据准备

将 NUDT-SIRST 放在以下目录：

```text
datasets/NUDT-SIRST/
├── images/
├── masks/
└── img_idx/
    ├── train_NUDT-SIRST.txt
    └── test_NUDT-SIRST.txt
```

数据集、检查点和训练日志不纳入 Git 仓库。NUDT-SIRST 下载与原论文信息请参考
[官方实现仓库](https://github.com/YeRen123455/Infrared-Small-Target-Detection)。

## 运行实验

单个变体可使用统一 runner 运行：

```bash
python3 experiments/train_tpd_pilot.py \
  --variant tpd \
  --dataset NUDT-SIRST \
  --device cuda:0 \
  --epochs 800 \
  --batch-size 16 \
  --patch-size 256 \
  --workers 0 \
  --seed 42 \
  --split-seed 20260722 \
  --val-fraction 0.20 \
  --eval-every 1 \
  --base-lr 0.001 \
  --min-lr 0.00001 \
  --warmup-epochs 10 \
  --threshold 0.5 \
  --match-radius 3 \
  --tiny-area 9
```

将 `--variant` 替换为 `original`、`progressive`、`spd` 或 `tpd` 即可运行相应
对照。完整的训练、Pd–Fa 评估、汇总和审计工具位于 [`experiments/`](experiments/)；
其中 4×RTX 5090 启动脚本包含本次机器的固定 GPU UUID，迁移到其他机器前需要调整。

运行单元测试：

```bash
python3 -m unittest discover -s tests
```

## 代码结构

```text
model/tpd.py                         # TPD、SPD 和 Progressive embedding
experiments/train_tpd_pilot.py       # 无官方测试泄漏的统一训练 runner
experiments/evaluate_pd_fa_sweep.py  # Pd–Fa threshold sweep
experiments/summarize_tpd_pd_fa.py   # Pd–Fa 汇总
experiments/decide_tpd_mainline_4x5090.py
                                     # 保守主线筛选决策
analysis/                             # 信息瓶颈分析
tests/                                # 模块与决策策略测试
```

设计背景和实验方案：

- [`TPD_SCTransNet_目标保真下采样实验方向.md`](TPD_SCTransNet_目标保真下采样实验方向.md)
- [`TPD_SCTransNet_主线修订版.md`](TPD_SCTransNet_主线修订版.md)
- [`SCTransNet_TPD_FG_实验设计与执行方案.md`](SCTransNet_TPD_FG_实验设计与执行方案.md)

## 与上游 SCTransNet 的关系

本仓库是 SCTransNet 的实验性派生版本，并非原论文官方结果仓库。原始模型、
论文、预训练权重和官方说明请访问
[xdFai/SCTransNet](https://github.com/xdFai/SCTransNet)。

如果使用 SCTransNet 基线，请引用原论文：

```bibtex
@article{SCTransNet,
  author  = {Yuan, Shuai and Qin, Hanlin and Yan, Xiang and Akhtar, Naveed and Mian, Ajmal},
  title   = {SCTransNet: Spatial-Channel Cross Transformer Network for Infrared Small Target Detection},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  volume  = {62},
  pages   = {1--15},
  year    = {2024},
  doi     = {10.1109/TGRS.2024.3383649}
}
```

## 结果边界

README 中的新增结果均为本仓库的 seed-42 内部验证实验。它们不等同于 NUDT-SIRST
官方测试成绩，也不构成跨数据集稳定性结论。为了避免误读，任何后续论文级声明都应
建立在多种子、多数据集和独立官方测试结果之上。
