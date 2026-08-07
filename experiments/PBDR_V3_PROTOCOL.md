# PBDR-V3 Conservative Twin-Gate Calibration Protocol

Status: frozen before the first PBDR-V3 training run.

## Scope

PBDR-V3 is a new, independent experiment line.  It does not overwrite or
resume PBDR-V2.  The stopped NUDT-SIRST PBDR-V2 epoch-64 state remains an
archived result and IRSTD-1K PBDR-V2 remains unstarted.

The first gate is NUAA-SIRST only.  GPU 0 is the only authorized device;
existing baseline services on GPUs 1--3 are outside this protocol.

## Ordered execution

1. Re-evaluate the frozen Current and PBDR-V2 NUAA checkpoints with the A0--A8
   branch-attribution matrix and a descriptive threshold/FROC sweep.
2. Build one PBDR-V3 candidate from each frozen Current role:
   `best_miou` and `best_pd`.
3. Warm-start all pre-PBDR state from the selected Current checkpoint.
4. Stage 1 freezes every Current parameter and all Current BatchNorm/dropout
   behavior.  Only `pbdr_v3.*` parameters are trainable.
5. Select epoch and any deployable threshold on a deterministic internal split
   of the official NUAA training set.  The official test index is not opened
   by the trainer.
6. Access the official NUAA test set once only when the internal certification
   gate passes.  Otherwise the deployment decision points to Current without
   evaluating the failed candidate on test.
7. Do not extend to NUDT-SIRST or IRSTD-1K until the NUAA mechanism gate passes.

## Frozen identity

- dataset: `NUAA-SIRST`
- training seed: `42`
- internal split seed: `20260722`
- internal validation fraction: `0.20`
- crop/augmentation: existing frozen three-dataset V2 stateless 256-pixel crop
- train normalization: existing frozen NUAA train normalization
- precision: FP32; cuDNN TF32 and CUDA matmul TF32 disabled
- Stage-1 optimizer: AdamW, base LR `1e-4`, weight decay `1e-4`
- maximum Stage-1 epochs: `150`
- validation cadence: every `5` epochs, including the final epoch
- fixed comparable threshold: probability `> 0.5`
- descriptive threshold grid: `[0.20, 0.80]`, step `0.01`
- object matching: one-to-one centroid distance `< 3` pixels
- tiny target: connected-component area `<= 9` pixels
- bounded residual limit: `0.15` logit
- q4 evidence RMS floor: `1.0`
- uncertainty floor: `0.25`
- equal rescue/suppression gate bias initialization: `-4.0`
- checkpoint policy: prefer epochs passing the fixed-0.5 internal gate, then
  apply the parent-role metric ordering; retain one candidate plus rolling
  exact-resume state

Every run records SHA-256 hashes for the parent checkpoint, split manifest,
protocol document, model/loss/trainer/evaluator sources, and their explicit
runtime dependencies.  Resume requires exact identity and restores Python,
NumPy, CPU Torch, CUDA Torch, model, optimizer, epoch, and selection state.

## Stage-1 loss recipes

The executable supports two predeclared recipes so their effects are not
silently combined:

- `core`: routed final BCE + soft-IoU; all relative-Current constraint weights
  are zero.
- `constrained`: routed final BCE + soft-IoU + background-increase (`8.0`) +
  foreground-decrease (`4.0`) + trust region (`0.25`) + residual sparsity
  (`0.05`) + hard-negative loss, whose weight ramps linearly from `0` to `2.0`
  over the first 20 epochs.

Deep-supervision loss is zero in Stage 1 because the five auxiliary heads and
the shared graph are frozen.

## Internal certification

For either Current parent role, a candidate passes only if all checks hold on
the frozen internal validation split at threshold 0.5:

- matched target count does not decrease;
- Fa does not increase;
- mIoU improves by at least `0.002`;
- nIoU does not decrease.

Failure writes a machine-readable decision selecting Current.  It does not
authorize Stage 2 or official-test candidate evaluation.

## Official-test deployment gate

If internal certification passes, compare candidate and its exact Current
parent at fixed threshold 0.5.  The role-specific minimum checks are:

### Parent `best_miou`

- matched targets `>= 256 / 263`
- Fa `<= 1.5435192155794186e-5`
- mIoU `>= 0.798482950889985`
- nIoU `>= 0.795348496003674`

### Parent `best_pd`

- matched targets `>= 257 / 263`
- Fa `<= 1.4749183615536667e-5`
- mIoU `>= 0.7905534317984362`
- nIoU `>= 0.7926679569324805`

The deployment manifest always names exactly one artifact.  A failed candidate
selects the immutable Current parent checkpoint.  This is a non-regression
guarantee on the frozen certification set, not a guarantee for arbitrary
future data.

## Stop conditions

- Any source/checkpoint/split hash mismatch: fail closed.
- Any non-PBDR trainable parameter or changing base BatchNorm buffer: fail
  closed.
- Any non-finite tensor, loss, gradient, or metric: fail closed.
- `core` failing its internal gate forbids base unfreezing; the predeclared
  `constrained` recipe may still run as the explicit E3 ablation.
- `core` passing its internal gate skips the same-role `constrained` worker;
  this prevents recipe selection from being repeated on the official test.
- `constrained` failing the internal gate ends the PBDR-V3 NUAA line and keeps
  Current deployed.
- Stage 2, hard-negative replay from cached components, and cross-dataset
  training require a new explicit protocol amendment.
