# V3 DC-knockout V2 aggregate-field repair V1

## Scope

The V2 DC-knockout diagnostic already has a frozen six-source lock and two
completed four-row checkpoint sweeps.  Both sweeps pass the frozen
`validate_checkpoint_sweep` implementation.  No new evaluation or training
is part of this repair.

The frozen aggregate fails while loading the learned V3 formal reference.
The canonical formal aggregate stores each `pd_at_fa_budget` point with the
field `fa`.  The diagnostic postprocessor instead requests
`point["achieved_fa"]`.  `achieved_fa` is the normalized diagnostic report
field, not the canonical formal input field.

## One allowed correction

The versioned wrapper performs exactly this mapping:

`formal_row.pd_at_fa_budget[*]["fa"] -> normalized["achieved_fa"]`

All remaining formal-row identity checks are retained.  The wrapper directly
reuses the frozen V2 diagnostic functions:

- `validate_checkpoint_sweep`
- `build_report`
- `render_markdown`

It does not alter the model, checkpoints, sweep values, thresholds, signed
delta definition, row order, formal comparison, or formal decision.

## Bound inputs

The repair attestation binds the exact:

- frozen V2 diagnostic postprocessor;
- V2 diagnostic source lock;
- `best.pth.tar` four-row sweep;
- `best_miou.pth.tar` four-row sweep;
- repaired formal V3 aggregate;
- versioned wrapper; and
- this protocol.

The generated report additionally records the attestation file hash, so the
report explicitly binds the V2 lock, both sweeps, formal aggregate, wrapper,
protocol, and attestation.

## Publication isolation

`--verify-only` and `--plan` are read-only.  `--aggregate-only`, when
explicitly invoked, may publish only below:

`experiments/results/tpd_ner_v8_mprs_dch_v3_dc_knockout_v2/NUDT-SIRST/comparison_aggregate_field_repair_v1/`

It never targets the frozen `comparison/` directory and refuses conflicting
existing files.

## Claim boundary

The repaired package remains an eight-row, evaluation-only, seed-42 internal
validation diagnostic.  It has no formal-decision authority, performs no GPU
work, and does not change the registered full-model gate.
