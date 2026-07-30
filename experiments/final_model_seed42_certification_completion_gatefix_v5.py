#!/usr/bin/env python3
"""Run the base completion verifier under the seed42 Gate repair context."""

from __future__ import annotations

from typing import Sequence

from experiments import final_model_seed42_certification_completion as completion
from experiments import (
    final_model_seed42_certification_replay_posttraining as frozen_posttraining,
)
from experiments import (
    final_model_seed42_certification_replay_posttraining_gatefix_v5
    as gatefix_v5,
)


SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_"
    "gatefix_entry_v5"
)


def main(argv: Sequence[str] | None = None) -> None:
    """Run every base completion action with the same local Gate fix."""

    with frozen_posttraining._temporary_attributes(
        frozen_posttraining,
        {
            "_validate_replay_manifest_for_paired": (
                gatefix_v5.seed42_overlay_bound_manifest_validator
            ),
        },
    ):
        completion.main(argv)


if __name__ == "__main__":
    main()
