# Final Model Certification Protocol V1

## 1. Frozen subject

This protocol certifies the already selected complete model without adding or
removing model modules:

```text
SCTransNet
+ TPD V8-MPRS-DCH
+ five-node NER V4 Tail-Aware
+ QFG-V2-CROA
+ TSS weight 0.005 during training only
```

The selected method is `D / Full-stack` (`method_id=d_tss_qfg`,
`variant=tss_qfg`).  Its deployment graph retains TPD, NER, and QFG and
physically omits the TSS heads and `target_survival.*` state.

The frozen model-source commit is:

```text
a295f751470c3414bb453d702451cecde41a1524
```

F0 does not change that source snapshot, the selected weights, or the
authoritative deployment operating point.

## 2. What “parent” means

`final_model_certification_parent_lock_v1.json` is the upstream-evidence
parent of later certification work.  Its name does not mean that a second
model participates in inference or shares an optimizer with a child.

The V4 checkpoint

```text
experiments/results/
tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1/NUDT-SIRST/
tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on/
seed_42_formal800_exact_v4_tail_aware_seed42/best_miou.pth.tar
```

is used once to initialize the shared 544 state entries of each cumulative
child arm.  Its role is `initialization_only`:

```text
construct one independent child model
→ copy the common V4 parent state
→ keep extension-only TSS/QFG builder initialization
→ create a new Adam optimizer for that child
→ train all child model parameters
```

The parent optimizer, scheduler, epoch counter, and training trajectory are
not inherited.  The parent model instance is not part of the child forward
pass and is not trained alongside the child.  A/B/C/D are independent after
their common initialization and never continue from one another.  Only an
interrupted run of the same arm may use exact resume.

The arm semantics are fixed as follows:

| Arm | Exact variant | Complete child training graph | Evaluation graph |
|---|---|---|---|
| Original | `original` | Original SCTransNet, independently trained | Original SCTransNet |
| A | `tss_control` | SCTransNet + TPD + five-node NER; TSS registered with weight 0 | SCTransNet + TPD + five-node NER |
| B | `tss_on` | A graph + TSS loss/head, weight 0.005 | SCTransNet + TPD + five-node NER |
| C | `qfg_only` | A graph + QFG; TSS registered with weight 0 | SCTransNet + TPD + five-node NER + QFG |
| D | `tss_qfg` | B graph + QFG | SCTransNet + TPD + five-node NER + QFG |

Thus “each arm trains independently” and “the arms use the same parent
checkpoint” are simultaneous facts: independence starts immediately after
the common initialization.

## 3. Frozen seed-42 identity

The authoritative engineering selection remains:

```text
decision                    = SELECT_D_TSS_QFG
training checkpoint         = D / best_miou.pth.tar
checkpoint epoch            = 3
checkpoint role             = best_validation_miou_secondary
deployment threshold        = 0.5
selected method             = d_tss_qfg
selected variant            = tss_qfg
```

At the authoritative threshold 0.5 on the frozen internal validation split:

| Metric | Value |
|---|---:|
| Pd | 188 / 189 |
| Fa | 4.1301985432330825e-6 |
| mIoU | 0.9370177924736262 |
| tiny-Pd | 39 / 39 |
| false objects | 5 |

The v1 final-selection file establishes the method/checkpoint selection.  The
deployment-v2 manifest and default-operating-point profile establish threshold
0.5 as the authoritative inference default; the older low-threshold point in
the legacy selection record is retained only as historical evidence.

## 4. Data and evaluator contract

The frozen seed-42 evidence uses only the NUDT-SIRST official training set
with the pre-existing internal split:

```text
split seed                   = 20260722
used train images            = 530
used validation images       = 133
train ID SHA-256             = 9565f584a5429fd1e5f0451b2d9496877f6f887493dd4d9954b4e976989f245b
validation ID SHA-256        = 86247e5970f93224c64005e1ac7f3a933bafb37baf279ab71fce5670ae925e06
normalization mean           = 107.28969064388635
normalization standard dev.  = 32.74261895755552
official test accessed       = false
```

The fixed evaluation contract is:

```text
prediction comparison        = prediction > threshold
default threshold            = 0.5
matching radius              = 3 pixels
tiny-target area             = at most 9 pixels
Fa budgets                   = 1e-6, 5e-6, 1e-5, 5e-5, 1e-4
primary metrics              = Pd, Fa, mIoU, tiny-Pd, false objects
```

Later certification tools must import or exactly reuse the frozen metric
implementation.  They may not silently substitute another connected-component
rule, matching rule, threshold comparison, normalization, or split.

## 5. Parent-lock scope

The parent lock binds by repository-relative path and SHA-256:

- the frozen Git commit, tree, and selected model/evaluator sources;
- the immutable V4 initialization checkpoint;
- the selected D training checkpoint and head-free inference artifact;
- deployment-v2, default-profile, final-selection, split, protocol, selected
  sweep, factorial, and reproducibility-manifest evidence;
- the training, post-training, and operational source locks;
- the seed, split, normalization, checkpoint, threshold, metric, and claim
  boundaries extracted from those files.

The lock deliberately excludes:

- its own bytes and SHA-256;
- the lock-generator source;
- this protocol file;
- any future certification commit;
- any future release attestation.

This avoids self-reference.  The parent lock binds the already existing model
facts.  A later source lock will bind new certification programs, and a later
release attestation will bind the certification commit and tag.

The lock uses canonical UTF-8 pretty JSON with one trailing newline.
Publication is write-once: an existing destination is never replaced, even
when its bytes are identical.  Verification rebuilds the complete live payload
and requires byte-for-byte canonical equality.

Commands:

```bash
python experiments/freeze_final_model_certification_parent_lock.py \
  --plan

python experiments/freeze_final_model_certification_parent_lock.py \
  --write-once

python experiments/freeze_final_model_certification_parent_lock.py \
  --verify
```

## 6. Certification state and claim boundary

At F0:

```text
final_model_engineering_selected       = true
final_model_established                = true
paper_core_established                 = false
stability_claim_supported              = false
official_test_claim                    = false
cross_seed_stability_claim             = false
internal_validation_only               = true

certification_design_reviewed          = true
certification_design_complete          = false
certification_implementation_complete  = false
certification_execution_authorized     = false
```

`final_model_established=true` is the existing engineering-selection field,
not a paper-level stability statement.  F0 creates no new performance result
and does not authorize official-test execution.  The above false fields may be
changed only by the later, separately bound certification gates.

