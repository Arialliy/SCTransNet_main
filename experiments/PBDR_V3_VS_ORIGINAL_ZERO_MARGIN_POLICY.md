# PBDR-V3 versus Original zero-margin role policy

## Status and timing

This policy was requested after the NUAA PBDR-V3 official-test artifacts had
already been produced. It is therefore a post-hoc metric-comparison revision,
not a pre-registered confirmatory rule. The original `evaluation.json` and
`deployment.json` files remain immutable evidence of the earlier Current-based
gate.

The revision must use only existing, hash-bound JSON/checkpoint artifacts. It
must not load the dataset, construct a test loader, run a model, or consume
another official-test access claim.

## Comparator

The baseline is the same-role `Original SCTransNet` checkpoint in:

```text
results/four_dataset_seed42_v1/selected_checkpoints/checkpoint_manifest.json
```

Comparison uses the official fixed threshold `0.5`, the same NUAA test target
count, and the role-specific ordering already frozen in that Original
manifest. There is no minimum effect-size or margin:

```text
best_miou:
  higher mIoU
  then higher Pd
  then lower Fa
  then higher nIoU
  then higher tiny-Pd
  then lower test loss
  then earlier epoch

best_pd:
  higher Pd
  then lower Fa
  then higher tiny-Pd
  then higher mIoU
  then higher nIoU
  then lower test loss
  then earlier epoch
```

The advisory metric winner is the candidate if and only if its complete
lexicographic key is strictly greater than the same-role Original key. Exact
equality keeps Original. This implements “the first differing role-ordered
metric must strictly improve” without inventing a weighted score or a positive
gain threshold.

## Reporting requirements

Every decision must also report, without hiding trade-offs:

- matched target count and Pd;
- Fa and its unmatched-prediction-pixel numerator when derivable;
- mIoU and nIoU;
- tiny-target count and tiny-Pd;
- which metrics improved, tied, or regressed;
- the Original checkpoint's `test_selected=true` and
  `selection_is_optimistic=true` disclosure.

The historical Original artifacts do not explicitly attest that both CUDA
matmul TF32 and cuDNN TF32 were disabled. PBDR-V3 does. Therefore the overlay
must separate the advisory metric winner from a binding deployment decision:
`binding_eligible=false`, `binding_selected=null`, and
`status=blocked_precision_provenance`. The earlier deployment remains
effective. The comparison is not a claim of fully matched precision
provenance.

## Output contract

The adjudicator writes a new overlay artifact and never overwrites the earlier
deployment:

```text
results/nuaa_pbdr_v3_stage1_v1/original_zero_margin_role_adjudication_v1.json
```

The overlay binds the Original manifest, Original protocol, both Original
formal fixed-0.5 evaluations and checkpoints, both V3 one-use access claims,
evaluation/deployment/protocol/summary/split artifacts, the candidate
checkpoints, this policy, and the adjudicator source by SHA-256. It records
`official_test_reaccessed=false`.
