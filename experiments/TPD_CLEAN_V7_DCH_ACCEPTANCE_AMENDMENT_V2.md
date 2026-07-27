# TPD-Clean V7-DCH acceptance amendment V2

## Scope

This amendment records one final-report rendering correction discovered after
acceptance v3 produced a comparison JSON/Markdown pair but before any
`completion_inputs.json` or `COMPLETE.sha256` was created. It changes no model,
dataset, split, seed, training configuration, checkpoint, evaluator, sweep,
metric, Gate A--E equation, threshold, or NER authorization rule.

## Root cause and correction

The comparison JSON is serialized with canonical key sorting. The Gate-D
Markdown table was previously rendered by iterating an in-memory mapping, so a
canonical JSON write/read round trip changed only the order of its 24 rows.
Every row and value remained identical.

The renderer now uses the protocol order explicitly:

1. seeds `42`, then `3407`;
2. checkpoint roles in `ROLE_SPECS` order;
3. fixed threshold first, followed by the five `BUDGET_KEYS` in their
   registered order.

Gate A--E is also rendered in explicit A-to-E order. The completion validator
continues to require that the published Markdown be reproduced exactly from
the published JSON; that check is not bypassed or weakened.

## Superseded unsealed comparison pair

The acceptance-v3 pair is preserved under:

`experiments/results/tpd_clean_v7_dch_formal800_2x5090_v1/NUDT-SIRST/comparison/superseded_acceptance_v3_markdown_order_v1`

- JSON SHA-256:
  `980bccdb248e24f08f5b95f7d9c3b9ec8da337a4d05e48c48e52e36486455caf`
- Markdown SHA-256:
  `3db8bbb5be4727077ab4b9cae23d828dbd41f62b0084988d78179fa0a72de8f8`
- Archive evidence SHA-256:
  `9c94a460c4a7601025152ea087901f63230c0425131e78150ddba55417780f03`

The archived JSON is semantically identical to a report rederived from the
exact inputs when its generated timestamp is excluded. The archived Markdown
equals the renderer output from that rederived report. The only failure was
that the old order-dependent renderer could not reproduce it from the
canonical JSON reload. This pair was never sealed as a completed result and is
not accepted as current.

## Lock and experiment boundary

- The training source lock remains
  `e67305d53b59336194541e2a9e6bec5bab3682c77232feb8be3e0fe71ea76c95`.
- The superseded diagnostic v2 lock is
  `902987310b86404b5cf72bb8e23359020508483ecc810cf6a881c67947e4b1d9`.
- The superseded acceptance v3 lock is
  `f319f4b4b1cd05ad97504b8fc317e8c24abb3736d5292ec64e85647731df5a45`.
- All four 800-epoch runs, twelve checkpoints, and eight sweep files remain
  unchanged.
- The comparison must be regenerated under the new current acceptance lock.
- No NER formal run is authorized by this amendment.

Current governance advances to diagnostic v3 and acceptance v4. Their hashes
are emitted only after all corrected sources and regression tests are final.
