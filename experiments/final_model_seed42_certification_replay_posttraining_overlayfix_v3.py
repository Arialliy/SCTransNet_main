#!/usr/bin/env python3
"""Additive seed42 post-training entry with a build-local replay overlay.

The frozen v1 post-training adapter keeps ``replay_trainer_overlay`` active
for the complete shared-evaluator call.  The shared evaluator performs a
second live input validation while writing the checkpoint-local result; that
validation must see the original frozen trainer source-lock function.

This successor changes only the lifetime of the existing overlay.  It enters
the overlay for the actual model-construction call and restores the trainer
before checkpoint loading, prediction collection, cache sealing, and result
validation.  All model, checkpoint, evaluator, metric, and write-once logic
continues to come from the frozen v1 modules.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from experiments import (
    final_model_seed42_certification_replay_posttraining as frozen_posttraining,
)


SCHEMA = (
    "sctransnet_final_model_seed42_certification_replay_"
    "posttraining_overlayfix_v3"
)
_FROZEN_BOUND_SHARED_EVALUATOR_LOADER = (
    frozen_posttraining.evaluator._load_bound_shared_evaluator
)


@contextlib.contextmanager
def build_local_trajectory_model_builder(
    request: frozen_posttraining.evaluator.CheckpointEvaluationRequest,
    inputs: frozen_posttraining.replay_core.ReplayInputs,
) -> Iterator[Callable[[str, int], Any]]:
    """Yield the frozen builder while limiting its overlay to one call."""

    frozen_posttraining._validate_request_shape(request)

    def build_model(variant: str, seed: int) -> Any:
        frozen_posttraining._equal(
            "shared evaluator variant",
            variant,
            request.variant,
        )
        frozen_posttraining._equal(
            "shared evaluator seed",
            seed,
            frozen_posttraining.TRAJECTORY_SEED,
        )
        with frozen_posttraining.replay_core.replay_trainer_overlay(
            inputs
        ) as trainer:
            return trainer.build_selected_model(
                variant,
                seed,
                eps=trainer.FORMAL_EPS,
            )

    yield build_model


def seed42_source_bound_shared_evaluator(*args, **kwargs):
    """Keep the dynamic sweep's evaluator hash on the seed42 adapter."""

    evaluator, state = _FROZEN_BOUND_SHARED_EVALUATOR_LOADER(
        *args,
        **kwargs,
    )
    source_binding = frozen_posttraining._evaluation_source_binding()
    adapter = source_binding.get("checkpoint_local_adapter")
    if not isinstance(adapter, dict):
        frozen_posttraining._fail(
            "seed42 source binding omits checkpoint-local adapter"
        )
    expected_path = Path(str(adapter.get("path"))).resolve()
    frozen_posttraining._equal(
        "seed42 checkpoint-local adapter path",
        expected_path,
        Path(frozen_posttraining.__file__).resolve(),
    )
    evaluator.__file__ = str(expected_path)
    return evaluator, state


def main(argv: Sequence[str] | None = None) -> None:
    """Run the frozen CLI with only its builder context manager replaced."""

    with frozen_posttraining._temporary_attributes(
        frozen_posttraining,
        {
            "_trajectory_model_builder": (
                build_local_trajectory_model_builder
            ),
        },
    ), frozen_posttraining._temporary_attributes(
        frozen_posttraining.evaluator,
        {
            "_load_bound_shared_evaluator": (
                seed42_source_bound_shared_evaluator
            ),
        },
    ):
        frozen_posttraining.main(argv)


if __name__ == "__main__":
    main()
