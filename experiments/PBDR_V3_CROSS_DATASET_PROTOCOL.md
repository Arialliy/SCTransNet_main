# PBDR-V3 cross-dataset Original-comparison protocol

Status: frozen before the first NUDT-SIRST or IRSTD-1K PBDR-V3 run.

This is a versioned amendment to, not an edit of,
`experiments/PBDR_V3_PROTOCOL.md`. It was authorized after the NUAA Stage-1
run. The scope is exactly `NUDT-SIRST` and `IRSTD-1K`; SIRST3 is outside the
three-dataset PBDR line. Existing PBDR-V2 and NUAA PBDR-V3 artifacts remain
immutable.

## Meaning of “no threshold”

The probability-to-mask comparison remains fixed at `probability > 0.5` so
metrics are defined consistently. There is no positive performance margin,
epsilon, percentage-gain requirement, or Pareto/non-regression requirement.
At the first different metric in the frozen role order, any strict improvement
is sufficient. Exact performance equality retains Original.
Target count and tiny-target count are same-split bindings, not performance
terms: both denominators must be identical, and both Pd ratios must exactly
match their integer matched/total counts (with zero total mapped to `0.0`).

The same-role comparison order is:

```text
best_miou:
  mIoU higher, then Pd higher, then Fa lower, then nIoU higher,
  then tiny-Pd higher, then test loss lower

best_pd:
  Pd higher, then Fa lower, then tiny-Pd higher, then mIoU higher,
  then nIoU higher, then test loss lower
```

Epoch is an earlier-epoch tie-break only when selecting among checkpoints from
one Stage-1 training run. It is not treated as cross-model performance and is
not part of Candidate-versus-Original deployment comparison.

## Frozen inputs

- training seed: `42`
- internal split seed: `20260722`
- internal validation fraction: `0.20` of the official training index
- dataset root: exact canonical path
  `/home/ly/SCTransNet_main/datasets`
- data protocol manifest: exact canonical path
  `/home/ly/SCTransNet_main/results/three_dataset_v2/manifests/three_dataset_v2_protocol.json`,
  file SHA-256
  `00edc6413dead3678f8b4c162c74ea7d8602f55ff413cb20ad1664587380319f`
- Current parents: same-dataset/same-role frozen checkpoints from
  `results/three_dataset_tss_off_seed42_v1`
- Original comparators: same-dataset/same-role frozen checkpoints from
  `results/four_dataset_seed42_v1/selected_checkpoints/checkpoint_manifest.json`
- threshold: fixed `0.5`; no threshold optimization or threshold sweep
- Stage 1: Current graph and BatchNorm state frozen; only the 6,018 PBDR-V3
  parameters train
- optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`
- epochs: `150`; internal evaluation every `5` epochs
- loss recipe: `core` only, unchanged from NUAA Stage 1
- precision: FP32; CUDA matmul TF32 and cuDNN TF32 both disabled
- crop/augmentation, normalization, matching radius (`<3` pixels),
  connectivity (`8`), and tiny area (`<=9` pixels): unchanged from the frozen
  three-dataset protocol

The two formal jobs may run concurrently only with this mapping:

```text
NUDT-SIRST -> physical GPU 0, UUID GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70
IRSTD-1K   -> physical GPU 1, UUID GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640
```

Existing baseline processes on physical GPUs 2 and 3 are outside scope and
must not be stopped, signalled, or modified.

## Train-only selection

Each dataset has one `best_miou/core` and one `best_pd/core` worker. The
trainer opens only that dataset's official training index, creates the frozen
internal split, and selects the maximum fixed-0.5 role key. Exact key ties use
the earlier Stage-1 epoch. A zero-margin Candidate-versus-Current comparison is
recorded as diagnostic evidence but does not change the official comparator
from Original and does not trigger an additional recipe.

Neither a failed nor a passed internal comparison authorizes test-dependent
retuning. Every completed, integrity-valid role contributes exactly its one
internally selected candidate to the predeclared dataset-level evaluation.

## One dataset-level official evaluation

Official-test access is dataset-global, not per role. The evaluator must:

1. validate both completed role runs, both candidate checkpoints, both Current
   parents, both Original checkpoints, all source locks, and both frozen
   training splits before any official-test claim;
2. create one dataset-owned access claim with `O_CREAT|O_EXCL` before test
   index construction; the claim authority is the non-overridable canonical
   path under this protocol's fixed result root (alternate result, run,
   claim, evaluation, or deployment paths are forbidden);
3. construct that dataset's official test loader exactly once;
4. in one ordered loader pass, collect fixed-0.5 predictions for both V3
   candidates, both exact Current bypasses (diagnostic only), and both
   same-role Original models under the same precision controls;
5. compute all metrics with the same implementation and choose Candidate or
   Original independently for each role using the frozen zero-margin order;
6. atomically commit one immutable publication bundle containing the complete
   evaluation and both deployment templates, then materialize one evaluation
   plus two hash-bound role deployment records. If materialization is
   interrupted after the bundle commit, a later invocation may only validate
   that bundle and restore missing views without constructing or iterating a
   test loader again.

The one-pass output must record test ID order, input-index hashes, target and
valid-pixel counts, checkpoint/state hashes, runtime-source hashes, both TF32
flags, all metric deltas and regressions, and the decisive metric. Original's
historical `test_selected=true` and `selection_is_optimistic=true` status must
remain disclosed even though Original is re-evaluated under matched precision.

If the access claim already exists, or any source/checkpoint/data binding is
different, the evaluator fails closed and does not create another loader.
There is no second official evaluation after observing results.

A completed trainer invocation is also immutable: an exact `resume=auto` or
`resume=required` request validates and returns its existing summary without
rewriting the selected checkpoint, certification, or summary bytes.

## Ordered execution

1. static/unit tests;
2. bounded GPU smoke for both datasets and roles under a separate smoke root;
   the validation prefix must extend through the first frozen tiny-target
   sample (at least 5 NUDT or 6 IRSTD validation images), so the complete
   role key has finite tiny-Pd; both dataset services start concurrently, but
   the launcher must wait for and aggregate their final worker exit statuses
   before this gate can pass;
3. formal Stage-1 training for both roles on both datasets;
4. source/checkpoint/split preflight for the dataset-global evaluators;
5. exactly one official pass for NUDT-SIRST and one for IRSTD-1K;
6. immutable report and deployment manifests; no result-driven retry.

Result root:

```text
results/two_dataset_pbdr_v3_stage1_v1/
```
