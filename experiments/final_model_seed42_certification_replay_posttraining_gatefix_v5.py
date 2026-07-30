#!/usr/bin/env python3
"""Additive seed42 Gate manifest-validation context repair.

The frozen seed42 post-training adapter already defines a strict four-result
manifest overlay.  Paired analysis runs its manifest validator inside that
overlay, but Gate invokes the same validator after leaving the overlay.  The
reused replication validator therefore sees its restored historical
eight-result/two-seed constants and rejects the valid seed42 four-result
manifest before Gate adjudication.

This successor changes only that validator's context: every seed42 manifest
validation is executed inside the already frozen seed42 evaluator overlay.
The original validator, all manifest fields, the paired result, Gate policy,
metrics, checkpoints, caches, and write-once behavior remain unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from experiments import (
    final_model_seed42_certification_replay_posttraining as frozen_posttraining,
)
from experiments import (
    final_model_seed42_certification_replay_posttraining_metricsfix_v4
    as metricsfix_v4,
)


SCHEMA = (
    "sctransnet_final_model_seed42_certification_replay_"
    "posttraining_gatefix_v5"
)
_FROZEN_REPLAY_MANIFEST_VALIDATOR = (
    frozen_posttraining._validate_replay_manifest_for_paired
)


def seed42_overlay_bound_manifest_validator(
    manifest: Mapping[str, Any],
) -> tuple[
    dict[tuple[int, str, str], dict[str, Any]],
    dict[tuple[int, str], dict[str, Any]],
]:
    """Run the unchanged manifest validator under its seed42 constants."""

    with frozen_posttraining._evaluator_overlay():
        return _FROZEN_REPLAY_MANIFEST_VALIDATOR(manifest)


def main(argv: Sequence[str] | None = None) -> None:
    """Run v4 with only the seed42 manifest-validator context repaired."""

    with frozen_posttraining._temporary_attributes(
        frozen_posttraining,
        {
            "_validate_replay_manifest_for_paired": (
                seed42_overlay_bound_manifest_validator
            ),
        },
    ):
        metricsfix_v4.main(argv)


if __name__ == "__main__":
    main()
