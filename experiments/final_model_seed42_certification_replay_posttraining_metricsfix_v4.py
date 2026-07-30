#!/usr/bin/env python3
"""Additive seed42 post-training metric-view compatibility entry.

The frozen shared sweep emits all 17 checkpoint validation fields after
checking them against the checkpoint, summary, and complete metrics log.  The
engineering request and all base closure verifiers intentionally use the
11-field paper-metric projection.  The frozen adapter compared those mappings
before projecting the shared result, so six valid auxiliary fields caused a
strict key-set mismatch even though every common value was identical.

This successor first rechecks the complete 17-field mapping against the
checkpoint, selected summary metrics, metrics.jsonl event, and the original
fixed-threshold audit.  It then applies the already frozen 11-field projection
and rebuilds only the corresponding fixed-threshold audit through the same
frozen audit function.  Every remaining validator and write stays unchanged.
The v3 build-local overlay and evaluator identity repairs are reused directly.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments import (
    final_model_seed42_certification_replay_posttraining as frozen_posttraining,
)
from experiments import (
    final_model_seed42_certification_replay_posttraining_overlayfix_v3
    as overlayfix_v3,
)


SCHEMA = (
    "sctransnet_final_model_seed42_certification_replay_"
    "posttraining_metricsfix_v4"
)
AUXILIARY_METRICS = (
    "niou",
    "pixel_f1",
    "pixel_precision",
    "pixel_recall",
    "predicted_object_count",
    "val_loss",
)
EXPECTED_FULL_METRICS = frozenset(
    (
        *frozen_posttraining.summary_core.METRICS,
        *AUXILIARY_METRICS,
    )
)
_FROZEN_CHECKPOINT_LOCAL_VALIDATOR = (
    frozen_posttraining.evaluator.validate_checkpoint_local_result
)
_FROZEN_FIXED_THRESHOLD_AUDIT = (
    frozen_posttraining.evaluator.sweep_core.audit_fixed_threshold_checkpoint
)


def _validate_full_metric_sources(
    metrics: Mapping[str, Any],
    request: frozen_posttraining.evaluator.CheckpointEvaluationRequest,
) -> dict[str, Any]:
    frozen_posttraining.evaluator._verify_request_files_unchanged(request)
    if set(metrics) != EXPECTED_FULL_METRICS:
        frozen_posttraining.evaluator._fail(
            "shared checkpoint validation metric fields differ from the "
            "registered 17-field source view"
        )
    ready: dict[str, Any] = {}
    for name, value in metrics.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            frozen_posttraining.evaluator._fail(
                f"shared checkpoint validation metric {name} "
                "must be finite numeric"
            )
        ready[name] = copy.deepcopy(value)

    checkpoint = frozen_posttraining.evaluator.sweep_core.torch.load(
        request.checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        frozen_posttraining.evaluator._fail(
            "selected checkpoint payload is not an object"
        )
    checkpoint_metrics = checkpoint.get("validation_metrics")
    frozen_posttraining.evaluator._canonical_equal(
        "shared/checkpoint full validation metrics",
        ready,
        checkpoint_metrics,
    )

    summary = frozen_posttraining.evaluator.sweep_core.load_json_object(
        request.run_directory / "summary.json"
    )
    summary_keys = (
        ("best_miou_validation_metrics",)
        if request.checkpoint_filename == "best_miou.pth.tar"
        else (
            "best_validation_metrics",
            "best_pd_validation_metrics",
        )
    )
    for summary_key in summary_keys:
        frozen_posttraining.evaluator._canonical_equal(
            f"shared/summary {summary_key} full validation metrics",
            ready,
            summary.get(summary_key),
        )

    events = frozen_posttraining.evaluator.sweep_core.load_complete_metrics(
        request.run_directory / "metrics.jsonl",
        frozen_posttraining.replay_contract.FORMAL_EPOCHS,
    )
    event = events[request.checkpoint_epoch - 1]
    event_metrics = {name: event.get(name) for name in ready}
    frozen_posttraining.evaluator._canonical_equal(
        "shared/metrics-log full validation metrics",
        ready,
        event_metrics,
    )
    frozen_posttraining.evaluator._verify_request_files_unchanged(request)
    return ready


def projected_checkpoint_metrics_validator(
    payload: Mapping[str, Any],
    request: frozen_posttraining.evaluator.CheckpointEvaluationRequest,
    *,
    execution_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the full source view, then persist the frozen paper view."""

    if not isinstance(payload, Mapping):
        frozen_posttraining.evaluator._fail(
            "shared evaluator result must be an object"
        )
    observed = payload.get("checkpoint_validation_metrics")
    if not isinstance(observed, Mapping):
        frozen_posttraining.evaluator._fail(
            "shared evaluator checkpoint validation metrics are missing"
        )
    full_metrics = _validate_full_metric_sources(observed, request)
    fixed_point = payload.get("fixed_threshold_0_5")
    if not isinstance(fixed_point, Mapping):
        frozen_posttraining.evaluator._fail(
            "shared evaluator fixed-threshold point is missing"
        )
    expected_raw_audit = _FROZEN_FIXED_THRESHOLD_AUDIT(
        dict(fixed_point),
        full_metrics,
    )
    frozen_posttraining.evaluator._canonical_equal(
        "shared full fixed-threshold checkpoint audit",
        payload.get("fixed_threshold_0_5_checkpoint_audit"),
        expected_raw_audit,
    )

    projected = frozen_posttraining.evaluator._validate_checkpoint_metrics(
        full_metrics,
        label="shared evaluator checkpoint validation metrics",
    )
    frozen_posttraining.evaluator._canonical_equal(
        "projected result/request checkpoint validation metrics",
        projected,
        request.checkpoint_validation_metrics,
    )
    projected_audit = _FROZEN_FIXED_THRESHOLD_AUDIT(
        dict(fixed_point),
        projected,
    )
    normalized = copy.deepcopy(dict(payload))
    normalized["checkpoint_validation_metrics"] = projected
    normalized["fixed_threshold_0_5_checkpoint_audit"] = projected_audit
    return _FROZEN_CHECKPOINT_LOCAL_VALIDATOR(
        normalized,
        request,
        execution_context=execution_context,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run v3 with only the checkpoint-local result validator adapted."""

    with frozen_posttraining._temporary_attributes(
        frozen_posttraining.evaluator,
        {
            "validate_checkpoint_local_result": (
                projected_checkpoint_metrics_validator
            ),
        },
    ):
        overlayfix_v3.main(argv)


if __name__ == "__main__":
    main()
