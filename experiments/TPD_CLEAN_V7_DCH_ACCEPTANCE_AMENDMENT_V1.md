# TPD-Clean V7-DCH acceptance amendment V1

## Scope

This amendment records one post-training verifier correction discovered after
all four formal 800-epoch runs and all eight closed-interval Pd--Fa sweeps had
completed. It changes no model, dataset, split, seed, optimizer, training
schedule, checkpoint, evaluator, threshold, metric, sweep, or Gate A--E rule.

## Correct canonical field

The ExactRunSpec producer stores the verified run environment at:

`run_identity.training_contract.environment`

The completion validator incorrectly attempted to read:

`run_identity.environment`

The validator is corrected to read only the canonical nested field. It does
not accept the incorrect top-level shape as a fallback.

## Evidence boundary

- The formal training protocol is unchanged.
- The training source lock remains
  `experiments/tpd_clean_v7_dch_exact_source_lock.json`.
- The four run protocols and twelve checkpoints are unchanged.
- The eight existing Pd--Fa sweep files are unchanged and must be validated
  before reuse.
- Checkpoint selection and all fixed-threshold Pd, Fa, and mIoU values are
  unchanged.
- Gate A--E thresholds and equations are unchanged.
- No NER formal run is authorized by this amendment.

The immutable superseded locks remain historical evidence:

- archived diagnostic v1 SHA-256:
  `edd670631f8e058e82d7bdddd68a21b1de46a1d3b02a35ae6fa7e2de22734695`
- diagnostic v1 supersession record SHA-256:
  `86512d73fc6aa0bb8ebbf38b272392a55f56c0667af0d534f4bb0f4927a4219b`
- diagnostic v1 SHA-256:
  `5f99bb511cb140cd502dcf41329f698b338d41e7404e6f897cf84ce3ab241a92`
- acceptance v1 SHA-256:
  `4fb4668d1eb97e3c6a28a60efbfad4ea9ac3423d98f16f7d411a09cebb5b68d7`
- acceptance v2 SHA-256:
  `ee7be009081b1776b6e5068c9c39b7f4429c987a44cea0a25f7c95f27fc8f130`

Current governance advances to diagnostic v2 and acceptance v3. Their hashes
are emitted by the exclusive source-lock writer after the corrected sources
and regression tests are final.
