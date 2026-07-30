#!/usr/bin/env python3
"""Run the v4 metrics attestation verifier under the seed42 Gate repair."""

from __future__ import annotations

from typing import Sequence

from experiments import (
    final_model_seed42_certification_completion_metricsfix_attestation_v4
    as metricsfix_attestation,
)
from experiments import (
    final_model_seed42_certification_replay_posttraining as frozen_posttraining,
)
from experiments import (
    final_model_seed42_certification_replay_posttraining_gatefix_v5
    as gatefix_v5,
)


SCHEMA = (
    "sctransnet_final_model_seed42_certification_metricsfix_"
    "attestation_gatefix_entry_v5"
)


def main(argv: Sequence[str] | None = None) -> None:
    """Build or verify the v4 attestation with local Gate validation."""

    with frozen_posttraining._temporary_attributes(
        frozen_posttraining,
        {
            "_validate_replay_manifest_for_paired": (
                gatefix_v5.seed42_overlay_bound_manifest_validator
            ),
        },
    ):
        metricsfix_attestation.main(argv)


if __name__ == "__main__":
    main()
