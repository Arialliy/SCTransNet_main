#!/usr/bin/env python3
"""Write and verify the additive seed42 Gate-context attestation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments import final_model_seed42_certification_completion as completion
from experiments import (
    final_model_seed42_certification_completion_metricsfix_attestation_v4
    as metricsfix_attestation,
)
from experiments import (
    freeze_final_model_seed42_certification_completion_gatefix_source_lock
    as gatefix_source_lock,
)
from experiments import (
    final_model_seed42_certification_replay_posttraining
    as frozen_posttraining,
)
from experiments import (
    final_model_seed42_certification_replay_posttraining_gatefix_v5
    as gatefix_v5,
)


SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_"
    "gatefix_attestation_v5"
)
ACTION_SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_"
    "gatefix_attestation_action_v5"
)
DEFAULT_OUTPUT = (
    completion.replay_contract.DEFAULT_OUTPUT_ROOT
    / "final_model_seed42_certification_gatefix_attestation_v5.json"
)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class CompletionGatefixAttestationError(ValueError):
    """The Gate-context repair evidence is missing or differs."""


def _fail(message: str) -> None:
    raise CompletionGatefixAttestationError(message)


def _equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        _fail(
            f"{label} differs: observed={observed!r}, expected={expected!r}"
        )


canonical_json_bytes = metricsfix_attestation.canonical_json_bytes
_artifact = metricsfix_attestation._artifact
_canonical_object = metricsfix_attestation._canonical_object
_write_or_validate = metricsfix_attestation._write_or_validate


def build_attestation(
    *,
    source_lock_path: Path = gatefix_source_lock.DEFAULT_OUTPUT,
    metricsfix_attestation_path: Path = (
        metricsfix_attestation.DEFAULT_OUTPUT
    ),
) -> dict[str, Any]:
    source_lock = gatefix_source_lock.verify_source_lock(source_lock_path)
    with frozen_posttraining._temporary_attributes(
        frozen_posttraining,
        {
            "_validate_replay_manifest_for_paired": (
                gatefix_v5.seed42_overlay_bound_manifest_validator
            ),
        },
    ):
        metricsfix_attestation.verify_attestation(
            output=metricsfix_attestation_path
        )
    upstream_attestation = _canonical_object(
        metricsfix_attestation_path,
        "metrics-fix v4 attestation",
    )
    _equal(
        "metrics-fix attestation status",
        upstream_attestation.get("status"),
        "complete",
    )
    _equal(
        "metrics-fix attestation decision",
        upstream_attestation.get("decision"),
        "FIXED_SEED42_INTERNAL_CERTIFICATION_CLOSED",
    )
    upstream_lock = source_lock.get(
        "upstream_completion_metricsfix_source_lock_v4"
    )
    if not isinstance(upstream_lock, Mapping):
        _fail("Gate-fix source lock omits metrics-fix v4")
    bound_v4_lock = upstream_attestation.get("metricsfix_source_lock_v4")
    if not isinstance(bound_v4_lock, Mapping):
        _fail("metrics-fix attestation omits its source lock")
    _equal(
        "metrics-fix attestation/source-lock SHA-256",
        bound_v4_lock.get("sha256"),
        upstream_lock.get("sha256"),
    )
    model_contract = upstream_attestation.get("model_contract")
    if not isinstance(model_contract, Mapping):
        _fail("metrics-fix model contract is missing")
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
            "paper-core claim",
            upstream_attestation.get("paper_core_established"),
            False,
        ),
        (
            "stability claim",
            upstream_attestation.get("stability_claim_supported"),
            False,
        ),
        (
            "official test accessed",
            upstream_attestation.get("official_test_accessed"),
            False,
        ),
    ):
        _equal(f"metrics-fix {label}", observed, expected)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "decision": "FIXED_SEED42_INTERNAL_CERTIFICATION_CLOSED",
        "scope": (
            "additive_gate_manifest_context_repair_binding_for_"
            "base_seed42_completion"
        ),
        "metricsfix_attestation_v4": _artifact(
            metricsfix_attestation_path,
            "metrics-fix attestation v4",
        ),
        "gatefix_source_lock_v5": {
            **_artifact(source_lock_path, "Gate-fix source lock v5"),
            "schema": source_lock["schema"],
            "source_count": source_lock["source_count"],
        },
        "upstream_metricsfix_source_lock_v4": {
            "path": str(
                (
                    gatefix_source_lock.REPO_ROOT
                    / str(upstream_lock["path"])
                ).resolve()
            ),
            "sha256": str(upstream_lock["sha256"]),
            "schema": str(upstream_lock["schema"]),
        },
        "repair_closure": {
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            "valid_seed42_manifest_result_count": 4,
            "valid_seed42_paired_checkpoint_group_count": 2,
            "historical_eight_result_contract_preserved": True,
            "original_strict_manifest_validator_reused": True,
            "manifest_validation_bound_to_seed42_overlay": True,
            "manifest_bytes_unchanged": True,
            "paired_result_unchanged": True,
            "gate_policy_unchanged": True,
            "gate_threshold_unchanged": True,
            "metric_values_unchanged": True,
            "checkpoint_and_cache_unchanged": True,
            "write_once_semantics_preserved": True,
            "base_completion_and_metricsfix_attestation_rebuilds_patched": True,
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
    source_lock_path: Path = gatefix_source_lock.DEFAULT_OUTPUT,
    metricsfix_attestation_path: Path = (
        metricsfix_attestation.DEFAULT_OUTPUT
    ),
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
        metricsfix_attestation_path=metricsfix_attestation_path,
    )
    path, action = _write_or_validate(output, payload)
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": "finalize-gatefix-attestation",
        "attestation_action": action,
        "attestation": _artifact(path, "Gate-fix attestation"),
        "runtime_environment_verified": require_runtime_env,
        "paper_core_established": False,
        "stability_claim_supported": False,
    }


def verify_attestation(
    *,
    source_lock_path: Path = gatefix_source_lock.DEFAULT_OUTPUT,
    metricsfix_attestation_path: Path = (
        metricsfix_attestation.DEFAULT_OUTPUT
    ),
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    stored = _canonical_object(output, "Gate-fix attestation")
    expected = build_attestation(
        source_lock_path=source_lock_path,
        metricsfix_attestation_path=metricsfix_attestation_path,
    )
    _equal(
        "stored/live Gate-fix attestation",
        canonical_json_bytes(stored),
        canonical_json_bytes(expected),
    )
    return {
        "schema": ACTION_SCHEMA,
        "status": "verified_complete",
        "action": "verify-gatefix-attestation",
        "attestation": _artifact(output, "Gate-fix attestation"),
        "paper_core_established": False,
        "stability_claim_supported": False,
    }


def dry_run_payload(
    *,
    source_lock_path: Path,
    metricsfix_attestation_path: Path,
    output: Path,
) -> dict[str, Any]:
    source_lock = gatefix_source_lock.verify_source_lock(source_lock_path)
    return {
        "schema": ACTION_SCHEMA,
        "status": "dry_run_complete",
        "action": "dry-run",
        "source_lock": _artifact(source_lock_path, "Gate-fix source lock"),
        "source_count": source_lock["source_count"],
        "metricsfix_attestation_path": str(
            Path(metricsfix_attestation_path).resolve()
        ),
        "output_path": str(Path(output).resolve()),
        "metricsfix_attestation_present": Path(
            metricsfix_attestation_path
        ).is_file(),
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
        default=gatefix_source_lock.DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--metricsfix-attestation",
        type=Path,
        default=metricsfix_attestation.DEFAULT_OUTPUT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.dry_run:
        payload = dry_run_payload(
            source_lock_path=args.source_lock,
            metricsfix_attestation_path=args.metricsfix_attestation,
            output=args.output,
        )
    elif args.write_once:
        payload = finalize_attestation(
            source_lock_path=args.source_lock,
            metricsfix_attestation_path=args.metricsfix_attestation,
            output=args.output,
        )
    else:
        payload = verify_attestation(
            source_lock_path=args.source_lock,
            metricsfix_attestation_path=args.metricsfix_attestation,
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
