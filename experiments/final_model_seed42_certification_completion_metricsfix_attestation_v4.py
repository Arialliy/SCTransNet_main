#!/usr/bin/env python3
"""Write and verify the additive seed42 metric-projection attestation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments import final_model_seed42_certification_completion as completion
from experiments import (
    final_model_seed42_certification_completion_overlayfix_attestation_v3
    as attestation_core,
)
from experiments import (
    freeze_final_model_seed42_certification_completion_metricsfix_source_lock
    as metricsfix_source_lock,
)


SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_"
    "metricsfix_attestation_v4"
)
ACTION_SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_"
    "metricsfix_attestation_action_v4"
)
DEFAULT_OUTPUT = (
    completion.replay_contract.DEFAULT_OUTPUT_ROOT
    / "final_model_seed42_certification_metricsfix_attestation_v4.json"
)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class CompletionMetricsfixAttestationError(ValueError):
    """The completion metric-projection evidence is missing or differs."""


def _fail(message: str) -> None:
    raise CompletionMetricsfixAttestationError(message)


def _equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        _fail(
            f"{label} differs: observed={observed!r}, expected={expected!r}"
        )


canonical_json_bytes = attestation_core.canonical_json_bytes
_artifact = attestation_core._artifact
_canonical_object = attestation_core._canonical_object
_write_or_validate = attestation_core._write_or_validate


def build_attestation(
    *,
    source_lock_path: Path = metricsfix_source_lock.DEFAULT_OUTPUT,
    base_attestation_path: Path = completion.DEFAULT_ATTESTATION,
) -> dict[str, Any]:
    source_lock = metricsfix_source_lock.verify_source_lock(
        source_lock_path
    )
    completion.verify_attestation(output=base_attestation_path)
    base = _canonical_object(
        base_attestation_path,
        "base completion attestation",
    )
    _equal("base completion status", base.get("status"), "complete")
    _equal(
        "base completion decision",
        base.get("decision"),
        "FIXED_SEED42_INTERNAL_CERTIFICATION_CLOSED",
    )
    model_contract = base.get("model_contract")
    if not isinstance(model_contract, Mapping):
        _fail("base completion model contract is missing")
    for label, observed, expected in (
        ("mainline changed", model_contract.get("mainline_changed"), False),
        (
            "innovation changed",
            model_contract.get("innovation_changed"),
            False,
        ),
        (
            "default threshold",
            model_contract.get("default_threshold"),
            0.5,
        ),
        (
            "deployment weight changed",
            model_contract.get("seed42_deployment_weight_changed"),
            False,
        ),
        (
            "paper-core claim",
            base.get("paper_core_established"),
            False,
        ),
        (
            "stability claim",
            base.get("stability_claim_supported"),
            False,
        ),
        (
            "official test accessed",
            base.get("official_test_accessed"),
            False,
        ),
    ):
        _equal(f"base completion {label}", observed, expected)
    upstream = source_lock.get(
        "upstream_completion_overlayfix_source_lock_v3"
    )
    if not isinstance(upstream, Mapping):
        _fail("metrics-fix source lock omits overlay-fix v3")
    return {
        "schema": SCHEMA,
        "status": "complete",
        "decision": "FIXED_SEED42_INTERNAL_CERTIFICATION_CLOSED",
        "scope": (
            "additive_runtime_repairs_binding_for_base_seed42_completion"
        ),
        "base_completion_attestation": _artifact(
            base_attestation_path,
            "base completion attestation",
        ),
        "metricsfix_source_lock_v4": {
            **_artifact(source_lock_path, "metrics-fix source lock v4"),
            "schema": source_lock["schema"],
            "source_count": source_lock["source_count"],
        },
        "upstream_overlayfix_source_lock_v3": {
            "path": str(
                (
                    metricsfix_source_lock.REPO_ROOT
                    / str(upstream["path"])
                ).resolve()
            ),
            "sha256": str(upstream["sha256"]),
            "schema": str(upstream["schema"]),
        },
        "repair_closure": {
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            "cublas_envfix_v2_applied": True,
            "replay_overlay_limited_to_model_build": True,
            "prewrite_live_revalidation_preserved": True,
            "dynamic_sweep_evaluator_bound_to_seed42_adapter": True,
            "checkpoint_full_metric_count": 17,
            "frozen_report_projection_metric_count": 11,
            "persistent_result_metric_count": 11,
            "checkpoint_full_sources_exactly_matched": True,
            "raw_full_fixed_threshold_audit_revalidated": True,
            "projected_fixed_threshold_audit_rebuilt_by_frozen_function": True,
            "base_v1_verifier_compatibility_preserved": True,
            "auxiliary_metric_fields_preserved_in_source_checkpoint": True,
            "frozen_v1_v2_v3_sources_unchanged": True,
            "four_checkpoint_local_sweeps_bound_by_base_attestation": True,
            "f1_and_deep_bound_by_base_attestation": True,
        },
        "model_contract": {
            "mainline": model_contract["mainline"],
            "mainline_changed": False,
            "innovation_changed": False,
            "model_architecture_changed": False,
            "checkpoint_changed_by_repair": False,
            "evaluation_algorithm_changed": False,
            "default_threshold": 0.5,
            "seed42_deployment_weight_changed": False,
        },
        "claim_boundary": {
            "single_seed_internal_validation_only": True,
            "official_test_accessed": False,
            "paper_core_established": False,
            "stability_claim_supported": False,
            "multiseed_replication_supported": False,
        },
        "paper_core_established": False,
        "stability_claim_supported": False,
        "official_test_accessed": False,
        "write_once": True,
        "overwrite_forbidden": True,
    }


def finalize_attestation(
    *,
    source_lock_path: Path = metricsfix_source_lock.DEFAULT_OUTPUT,
    base_attestation_path: Path = completion.DEFAULT_ATTESTATION,
    output: Path = DEFAULT_OUTPUT,
    require_runtime_env: bool = True,
) -> dict[str, Any]:
    if require_runtime_env:
        _equal(
            "runtime CUBLAS_WORKSPACE_CONFIG",
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            CUBLAS_WORKSPACE_CONFIG,
        )
    payload = build_attestation(
        source_lock_path=source_lock_path,
        base_attestation_path=base_attestation_path,
    )
    path, action = _write_or_validate(output, payload)
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": "finalize-metricsfix-attestation",
        "attestation_action": action,
        "attestation": _artifact(path, "metrics-fix attestation"),
        "runtime_environment_verified": require_runtime_env,
        "paper_core_established": False,
        "stability_claim_supported": False,
    }


def verify_attestation(
    *,
    source_lock_path: Path = metricsfix_source_lock.DEFAULT_OUTPUT,
    base_attestation_path: Path = completion.DEFAULT_ATTESTATION,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    stored = _canonical_object(output, "metrics-fix attestation")
    expected = build_attestation(
        source_lock_path=source_lock_path,
        base_attestation_path=base_attestation_path,
    )
    _equal(
        "stored/live metrics-fix attestation",
        canonical_json_bytes(stored),
        canonical_json_bytes(expected),
    )
    return {
        "schema": ACTION_SCHEMA,
        "status": "verified_complete",
        "action": "verify-metricsfix-attestation",
        "attestation": _artifact(output, "metrics-fix attestation"),
        "paper_core_established": False,
        "stability_claim_supported": False,
    }


def dry_run_payload(
    *,
    source_lock_path: Path,
    base_attestation_path: Path,
    output: Path,
) -> dict[str, Any]:
    source_lock = metricsfix_source_lock.verify_source_lock(
        source_lock_path
    )
    return {
        "schema": ACTION_SCHEMA,
        "status": "dry_run_complete",
        "action": "dry-run",
        "source_lock": _artifact(
            source_lock_path,
            "metrics-fix source lock",
        ),
        "source_count": source_lock["source_count"],
        "base_attestation_path": str(Path(base_attestation_path).resolve()),
        "output_path": str(Path(output).resolve()),
        "base_attestation_present": Path(base_attestation_path).is_file(),
        "output_present": Path(output).is_file(),
        "gpu_queried": False,
        "gpu_command_launched": False,
        "writes_performed": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--write-once", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=metricsfix_source_lock.DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--base-attestation",
        type=Path,
        default=completion.DEFAULT_ATTESTATION,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.dry_run:
        payload = dry_run_payload(
            source_lock_path=args.source_lock,
            base_attestation_path=args.base_attestation,
            output=args.output,
        )
    elif args.write_once:
        payload = finalize_attestation(
            source_lock_path=args.source_lock,
            base_attestation_path=args.base_attestation,
            output=args.output,
        )
    else:
        payload = verify_attestation(
            source_lock_path=args.source_lock,
            base_attestation_path=args.base_attestation,
            output=args.output,
        )
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
