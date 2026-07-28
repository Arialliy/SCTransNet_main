# V3 formal800 selection-contract repair V1

## Scope

This repair is limited to the post-training comparison contract in
`postprocess_tpd_ner_v8_mprs_dch_v3_formal800.py`.  It does not change the
V3 model, training, checkpoints, evaluators, sweeps, registered performance
gate, or any baseline/V1/V2 artifact.

The frozen postprocessor first calls the completed V2 comparison contract,
which correctly checks `selection_source` and `checkpoint_policy` only
between V2 relay-on and V1 relay-off.  Its V3 extension then incorrectly
requires the legacy baseline protocol to:

1. contain the newer top-level
   `selection_source == "internal_validation_only"` field; and
2. use byte-identical `checkpoint_policy` wording to V1.

The baseline protocol predates the new top-level field.  Its policy instead
states exactly:

> best.pth.tar is Pd-primary; best_miou.pth.tar is a secondary analysis
> checkpoint; all selection uses internal validation only

Both immutable baseline sweeps also record
`audit.selection_source == "internal_validation_only"` and
`official_test_accessed == false`.  The completed V2 aggregate binds both
baseline sweep hashes, labels their rows as
`same_protocol_external_reference`, declares
`scope == "single_seed_internal_validation"`, and declares
`official_test_accessed == false`.

## Allowed correction

The versioned wrapper may replace exactly one frozen logic function in
memory:

`same_split_and_training_contract`

The replacement retains all frozen checks for training axes, ordered split
identity, normalization, optimizer, loss, selection rules, model roles, and
comparison design.  It additionally requires exact equality across V3, V2,
and V1 for:

- `selection_source`, with the exact value
  `internal_validation_only`; and
- `checkpoint_policy`.

For the legacy baseline only, internal-validation selection is established
by the conjunction of:

- the exact baseline `checkpoint_policy` text above;
- both immutable baseline sweeps, including their audit selection source,
  official-test isolation, validation split identity, checkpoint roles, and
  successful reference-artifact validation; and
- the completed, hash-bound V2 aggregate and completion marker.

No other comparison or gate logic may be replaced.  The wrapper calls the
frozen `aggregate_and_write` implementation and only redirects its three
publication paths.

## Publication isolation

If explicitly invoked with `--aggregate-only`, the repair writes only below:

`experiments/results/tpd_ner_v8_mprs_dch_v3_exact_v1/NUDT-SIRST/comparison_selection_contract_repair_v1/`

The output names are:

- `tpd_ner_v8_mprs_dch_v3_formal800_comparison_selection_contract_repair_v1.json`
- `tpd_ner_v8_mprs_dch_v3_formal800_comparison_selection_contract_repair_v1.md`
- `POSTPROCESS_COMPLETE_SELECTION_CONTRACT_REPAIR_V1.json`

Existing frozen V3 comparison paths are never publication targets of this
wrapper.

## Attestation and execution policy

The immutable repair attestation binds:

- the frozen V3 postprocessor source;
- this repair wrapper source;
- this protocol;
- both V3 formal sweeps;
- all four compared protocols;
- both baseline sweeps; and
- the completed V2 aggregate JSON, Markdown, and completion marker.

Every binding is verified before contract evaluation or publication.
Symlinks, missing bindings, extra bindings, and hash mismatches are hard
failures.

`--verify-only` and `--plan` are read-only.  `--aggregate-only` performs no
training and no model evaluation; it revalidates and aggregates already
existing sweeps through the frozen implementation.  The repair introduces no
GPU work.

## Claim boundary

The repaired output remains a fixed-seed-42, NUDT-SIRST 530/133 internal
validation comparison.  It does not add a cross-seed, cross-dataset, or
official-test claim.  It does not alter any performance value or acceptance
threshold.
