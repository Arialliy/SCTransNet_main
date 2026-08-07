# SCTransNet PBDR-V4 formal protocol

Status: implementation protocol.  This file governs only new PBDR-V4
artifacts; it does not reinterpret or overwrite historical V2/V3 results.

## Objective and comparison rule

PBDR-V4 is selected at the fixed probability decision rule
`probability > 0.5`.  There is no minimum gain, relative percentage,
non-regression gate, epsilon, or materiality threshold.  For each dataset and
checkpoint role, selection is the strict lexicographic maximum over this
frozen family order:

1. Original, same role;
2. Current TSS-off, same role;
3. selected PBDR-V3 residual recalibration;
4. selected PBDR-V4 Stage-1 checkpoint;
5. selected PBDR-V4 Stage-2 checkpoint.

An exact role-key tie retains the earlier family.  The role keys are:

- `best_miou`: higher exact mIoU, higher exact Pd, lower exact Fa, higher
  nIoU, higher exact tiny-Pd, lower loss;
- `best_pd`: higher exact Pd, lower exact Fa, higher exact tiny-Pd, higher
  exact mIoU, higher nIoU, lower loss.

For epoch selection, `-epoch` is appended after the complete role key.  A
Candidate-vs-Current pass flag must never precede the role key.

## Frozen split authority

The existing V3 formal split manifests for NUAA-SIRST, NUDT-SIRST, and
IRSTD-1K are projected read-only.  The split is not reconstructed.  Its
official-train, development-train, and internal-validation ID order, source
file hash, and canonical split hash are bound.  Atlas construction uses only
development-train IDs.  Calibration and checkpoint selection use only
internal-validation IDs.

This is a model-selection split, not an independent holdout: Original and
Current parents were trained on the official training set.  The benchmark
test sets have also been used by historical experiments.  The final pass is
therefore described as one new PBDR-V4 protocol pass, not an unseen
confirmatory test.

## Fixed metric semantics

- probability and target comparisons are strict `> 0.5`;
- components are 8-connected (`connectivity=2` in the 2-D skimage API);
- centroid match distance is strict `< 3` pixels;
- one-to-one assignment maximizes cardinality first and minimizes total
  centroid distance second using the current formal Hungarian cost;
- tiny targets have component area `<= 9` pixels;
- Fa counts pixels belonging to unmatched predicted component IDs.

The V4 matcher is versioned separately.  Legacy greedy matching code is not
modified, and historical metrics are not recomputed or relabeled.

## PBDR-V3 residual recalibration

Internal validation evaluates all 378 pre-registered configurations:

- positive scale: `0, 0.5, 1, 1.5, 2, 3, 4`;
- negative scale: `0, 0.25, 0.5, 0.75, 1, 1.5`;
- bias: `-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.30`.

The routed logit is `base + s_pos*relu(delta) - s_neg*relu(-delta) + bias`.
`(0,0,0)` must exactly replay Current and `(1,1,0)` must exactly replay V3.
Only one selected configuration per dataset and role is frozen before the
official pass; no official calibration sweep is permitted.

## Role-bound V4 graph

The calibrator consumes explicitly named raw logits `out`, `d0`, `gt2`,
`gt3`, `gt4`, `gt5`, raw `q4`, and final decoder feature `u1`.  The auxiliary
logit order cannot be supplied as an untyped sequence.  Its terminal residual
head is zero initialized, so initial routed logits are exactly Current.

Role code and positive/negative residual limits are persistent checkpoint
buffers.  Loading a checkpoint from the other role, or one with different
limits, is an error even under `strict=False`.  Dataset, role, stage, parent
checkpoint, split authority, atlas, source lock, and architecture identity are
also checkpoint-payload fields.

## Component atlas and loss

For each dataset and role, frozen Current predictions on development-train
IDs generate three int32 ID maps with the canonical matcher:

- rescue: unmatched target components;
- suppress: unmatched predicted components;
- preserve: matched target components.

Image, target, and all three maps share one stateless transformation in this
order: bottom/right pad, crop, axis-0 flip, axis-1 flip, transpose, contiguous
conversion.  The current training chain performs no resize.

The role loss contains BCE, role-specific Tversky, equal-per-component rescue,
suppress, and preserve terms, one-sided foreground/background protections,
and a small neutral-delta term.  Stage-2 protections use a separate frozen
Current logit reference; the moving candidate base is never labeled Current.

## Stage-1 and Stage-2

Stage-1 runs 150 epochs with validation every 5 epochs.  Only `pbdr_v4.*`
parameters are trainable.  All Current parameters and buffers remain bitwise
unchanged.

Stage-2 runs 50 epochs from the immutable selected Stage-1 checkpoint.
Trainable parameters are exactly `pbdr_v4.*`, `outc.*`, and
`up_decoder1.*`.  Router, outc, and up_decoder1 learning rates are respectively
`1e-4`, `2e-6`, and `1e-6`.  L2-SP anchors the Stage-2 base parameters to the
Current parent.  Every BatchNorm module remains in eval mode; every base
BatchNorm running buffer, including those under `up_decoder1`, remains
bitwise Current.

Both stages use FP32, AdamW weight decay `1e-4`, deterministic algorithms,
and disabled CUDA matmul and cuDNN TF32.  Stage-2 always runs as a predeclared
parallel candidate and is not conditional on a performance gate.

## Checkpoint and resume contract

Epoch checkpoints are append-only.  A rolling state is atomically replaced;
the final selection and summary are exclusive commits.  Resume validates
schema, dataset, role, stage, epoch bounds, model and optimizer parameter-group
structure, RNG states, split/atlas/source-lock hashes, parent or Stage-1
initialization hash, and current best-checkpoint bytes.  Resume immediately
replays the stage state audit.

`resume=never` rejects any prior state.  `resume=required` requires a valid
complete or rolling artifact.  A valid complete resume returns the existing
artifact without changing its bytes.  Formal launch must name the resume mode
explicitly.

## One-pass official boundary

Before a claim is created, every frozen candidate, checkpoint, role, source
lock, data binding, and maximum-size forward is validated.  The official
dataset/index/loader may be constructed only after an exclusive claim commit.
One loader and one iteration update online metrics for all five frozen
families.  Official probabilities and logits are not cached, and no sweep or
checkpoint change occurs after the claim.

A crash after claim and before publication-bundle commit is a consumed,
terminal failure; a second pass is forbidden.  Once the publication bundle is
committed, later invocations may only validate it and materialize missing
views without constructing the loader.  If official results choose the
deployed family, the result records `operational_test_selected=true` and
`selection_is_optimistic=true`.

