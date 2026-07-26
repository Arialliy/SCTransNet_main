# TPD-Clean-v3 closed-interval sweep recovery

## Scope

This recovery completes the threshold domain for the eight finished
TPD-Clean-v3 candidate sweeps. It does not retrain a model, change a
checkpoint, change any of the 3,200 metric rows, or modify formal800,
TPD-Clean-v2, frozen SPD/TPD-v1, or TPD-NER artifacts.

The original evaluator sampled thresholds strictly below one. The
`tpd_clean_v3_sal_capacity`, seed 3407, Pd-primary checkpoint emitted FP32
scores saturated at exactly `1.0`; therefore its original grid contained no
point under three strict Fa budgets. That was a threshold-coverage gap, not a
missing training result.

## Closed FP32 threshold domain

The recovery wrapper preserves every original point and uniformly appends to
all eight candidate sweeps:

- `0.9999999403953552`, the largest FP32 value below one, which isolates the
  exact-one score plateau;
- `1.0`, the empty-prediction endpoint because evaluation uses
  `prediction > threshold`.

The endpoint has `Pd=0` and `Fa=0`; it cannot manufacture a detection gain.
All five registered Fa budgets are recomputed with the unchanged selection
key. Frozen reference sweeps are not regenerated.

## Recovery artifacts

- wrapper: `experiments/evaluate_tpd_clean_v3_pd_fa_closed_interval.py`
  (`4c1f2e5423d71326e20921a933be62b8830c1547dbdf6a7a1819e36c85971165`)
- independent audit:
  `experiments/audit_tpd_clean_v3_closed_interval_sweeps.py`
  (`f938a5537c491f99482ae0d376a96e8baa0b978e6c29bfa25c24a2f2517d3783`)
- focused test:
  `tests/test_evaluate_tpd_clean_v3_pd_fa_closed_interval.py`
  (`5243ac18eb6f036bafe2ab2e2cf30f8286080fac1f8fef44084de324a963e4c2`)
- immutable recovery root:
  `experiments/results/tpd_clean_v3_screen800_4x5090_v1/resume_2x5090_v1/recovery_threshold1_20260726_113926/`
- machine-readable audit:
  `closed_interval_sweep_audit.json`
  (`00ed5b68ceec7de852d6216406e1450ba0c005dafaf90626b9dd53a55294cfec`)
- original eight sweeps and the failed incomplete report are retained under
  `original/`.

The audit requires, for all eight sweeps:

1. checkpoint identity and threshold-0.5 metrics are unchanged;
2. every original threshold point is byte-for-byte equal as a JSON value;
3. only the two registered boundary thresholds are added;
4. wrapper invocation and SHA-256 provenance match;
5. score dtype and exact-one score count are recorded;
6. all five budget optima exactly recompute and none remains null.

## Source-lock boundary

The original training lock
`9d735db3ad8b6fa0dc10a32afb666c0cd1e9757c61121293c85ed5869020058b`
and resume lock
`6f28b5999c585e7b0c028a02c7c19d254259938108bd24ecaf3feb3e5fbd702b`
remain unchanged. Only the resume postprocessing lock is extended to bind
this recovery wrapper, audit, test, and protocol before the canonical
summarizer and completion validator are rerun.

The recovery does not authorize automatic mainline replacement. The final
seven-gate report remains the sole engineering decision artifact and must keep
`automatic_mainline_replacement=false` and `mainline_changed=false`.
