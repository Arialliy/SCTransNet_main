# EC-TSS V3.1 frozen execution protocol

This file is the compact machine-facing companion to
`SCTransNet_EC-TSS_V3性能提升与下一步方案.md`.

## Frozen identity

```text
objective_id=ec_tss_v3_1
training_seed=42
datasets=NUAA-SIRST,NUDT-SIRST,IRSTD-1K
selection_split=img_idx/test
planned_total_epochs=1000
pilot_pause_epoch=200
begin_test=10
eval_every=10
threshold=0.5
checkpoint_roles=best_miou,best_pd
requested_tss_weight=0.005
survival_ratio_cap=0.10
confidence_threshold=0.5
target_dilation_radius=3
positive_normalization=risk_mass_clamp_min_1
negative_normalization=risk_mass_clamp_min_1
final_probability_source=evaluator_prediction
```

## Architecture boundary

- The TPD8, five-node NER4 Tail-Aware, and QFG2-CROA inference path is frozen.
- Existing survival heads and model state keys are reused.
- EC-TSS changes only the training objective.
- The historical TSS loss, selectors, launchers, and result directories are read-only.
- A training checkpoint from another TSS recipe is not a valid EC-TSS resume source.

## Objective

Let `P` be the final segmentation probability, `Y` the binary mask,
`Y16=MaxPool16(Y)`, and `M=Dilate(Y,r=3)`. Risk maps are detached from `P`:

```text
Q_pos = MaxPool16(stopgrad(P) * M)
Q_neg = MaxPool16(stopgrad(P) * (1-M))
R_pos = Y16 * clamp((0.5-Q_pos)/0.5, 0, 1)
R_neg = (1-Y16) * clamp((Q_neg-0.5)/0.5, 0, 1)
```

For endpoint logit `Z_i`:

```text
L_i_pos = sum(R_pos * softplus(-Z_i)) / max(sum(R_pos), 1)
L_i_neg = sum(R_neg * softplus( Z_i)) / max(sum(R_neg), 1)
L_ec = 0.25 * (L_1_pos + L_1_neg + L_2_pos + L_2_neg)
lambda_eff = min(0.005, 0.10 * stopgrad(L_seg) / max(stopgrad(L_ec), eps))
L_total = L_seg + lambda_eff * L_ec
```

The cap constrains the scalar weighted auxiliary loss, not its gradient norm.

## Resumable pilot

The pilot is the first 200 epochs of each formal run, not a separate run.
The learning-rate horizon remains 1000 epochs. At epoch 200 the runner writes
the complete rolling state and returns a paused status. Continuation must use
the same directory, protocol digest, model, optimizer, RNG state, and derived
DataLoader order with `resume=required`.

## GPU lanes

```text
GPU0: NUAA-SIRST
GPU1: NUDT-SIRST
GPU2: IRSTD-1K
GPU3: smoke, fixed-batch scale/risk diagnostics, then checkpoint evaluation
```

The three formal training runs remain independent single-GPU runs. No DDP,
duplicate run, global-batch change, or altered optimizer trajectory is added
only to occupy the fourth device.

The launcher first executes the epoch-200 prefix. It may continue to epoch
1000 only after engineering checks find finite loss, non-empty selected
checkpoints, complete rolling state, and nonzero use of both EC-TSS branches.
The 200-epoch metrics are development evidence and remain test-informed.

## Result scope

All EC-TSS artifacts use an independent root and schema. Only selected
`best_miou` and `best_pd` checkpoints are retained after successful epoch
1000 completion; the rolling state exists only while a run is incomplete.
Historical Original, TSS-off, and positive-lambda artifacts are never
overwritten or re-selected.
