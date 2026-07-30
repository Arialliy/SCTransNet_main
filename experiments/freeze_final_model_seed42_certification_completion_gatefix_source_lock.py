#!/usr/bin/env python3
"""Freeze and verify the additive seed42 Gate-context repair layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

from experiments import (
    freeze_final_model_seed42_certification_completion_metricsfix_source_lock
    as metricsfix_source_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_"
    "gatefix_source_lock_v5"
)
ACTION_SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_"
    "gatefix_source_lock_action_v5"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/"
    "final_model_seed42_certification_completion_gatefix_source_lock_v5.json"
)
EXPECTED_METRICSFIX_SOURCE_LOCK_SHA256 = (
    "93d205ab4e3dc952c607d6d57334b3f083dd7bd76e9e7bb2ed57e704dcb24784"
)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
SOURCE_RELATIVE_PATHS = (
    "experiments/"
    "final_model_seed42_certification_replay_posttraining_gatefix_v5.py",
    "experiments/"
    "final_model_seed42_certification_completion_gatefix_v5.py",
    "experiments/"
    "final_model_seed42_certification_metricsfix_attestation_gatefix_v5.py",
    "experiments/"
    "final_model_seed42_certification_python_gatefix_v5.sh",
    "experiments/"
    "run_final_model_seed42_certification_completion_gatefix_v5.sh",
    "experiments/"
    "freeze_final_model_seed42_certification_completion_gatefix_source_lock.py",
    "experiments/"
    "final_model_seed42_certification_completion_gatefix_attestation_v5.py",
    "tests/"
    "test_final_model_seed42_certification_completion_gatefix.py",
)


class CompletionGatefixSourceLockError(ValueError):
    """The Gate-context repair lock or one of its sources differs."""


def _fail(message: str) -> None:
    raise CompletionGatefixSourceLockError(message)


def _equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        _fail(
            f"{label} differs: observed={observed!r}, expected={expected!r}"
        )


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _regular(path: Path, label: str) -> Path:
    source = Path(path)
    if source.is_symlink():
        _fail(f"{label} must not be a symlink: {source}")
    try:
        metadata = source.stat()
    except FileNotFoundError:
        _fail(f"{label} is missing: {source}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a regular file: {source}")
    return source.resolve()


def file_sha256(path: Path) -> str:
    source = _regular(path, "file")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    source = Path(path).resolve()
    try:
        return source.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        _fail(f"path lies outside repository: {source}")


def _verified_upstream(
    path: Path = metricsfix_source_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    payload = metricsfix_source_lock.verify_source_lock(path)
    _equal(
        "metrics-fix source-lock schema",
        payload.get("schema"),
        metricsfix_source_lock.SCHEMA,
    )
    upstream_sha256 = file_sha256(path)
    _equal(
        "metrics-fix source-lock SHA-256",
        upstream_sha256,
        EXPECTED_METRICSFIX_SOURCE_LOCK_SHA256,
    )
    return {
        "path": _repo_relative(Path(path)),
        "schema": metricsfix_source_lock.SCHEMA,
        "sha256": upstream_sha256,
        "source_count": payload["source_count"],
    }


def build_source_lock(
    *,
    metricsfix_source_lock_path: Path = metricsfix_source_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    upstream = _verified_upstream(metricsfix_source_lock_path)
    source_sha256 = {
        relative: file_sha256(REPO_ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }
    return {
        "schema": SCHEMA,
        "status": "locked",
        "lock_kind": "additive_seed42_gate_manifest_context_fix",
        "scope": (
            "seed42_four_result_manifest_validation_under_frozen_"
            "seed42_evaluator_overlay"
        ),
        "source_count": len(source_sha256),
        "source_sha256": source_sha256,
        "upstream_completion_metricsfix_source_lock_v4": upstream,
        "repair_contract": {
            "observed_failure_stage": "gate_prewrite_manifest_validation",
            "valid_seed42_manifest_result_count": 4,
            "valid_seed42_paired_checkpoint_group_count": 2,
            "historical_replication_manifest_result_count": 8,
            "historical_replication_trajectory_seeds": [3407, 426780603],
            "seed42_replay_trajectory_seeds": [42],
            "paired_analyze_already_used_seed42_overlay": True,
            "gate_validator_previously_called_after_overlay_exit": True,
            "original_strict_manifest_validator_reused": True,
            "seed42_overlay_lifetime_extended_only_for_manifest_validation": True,
            "posttraining_gate_runs_under_seed42_manifest_overlay": True,
            "base_completion_verify_posttraining_runs_under_same_overlay": True,
            "base_completion_finalize_attestation_runs_under_same_overlay": True,
            "base_completion_verify_attestation_runs_under_same_overlay": True,
            "metricsfix_attestation_rebuild_runs_under_same_overlay": True,
            "historical_eight_result_contract_unchanged": True,
            "manifest_bytes_changed": False,
            "paired_result_changed": False,
            "gate_policy_changed": False,
            "gate_threshold_changed": False,
            "metric_values_changed": False,
            "checkpoint_or_cache_changed": False,
            "write_once_semantics_preserved": True,
            "successor_attestation_written_after_base_completion": True,
        },
        "execution_contract": {
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            "gpu_assignment": {"b": 2, "d": 3},
            "existing_valid_four_sweeps_reused": True,
            "existing_valid_manifest_reused": True,
            "existing_valid_paired_result_reused": True,
            "resume_from_idempotent_full_runner": True,
            "mainline_changed": False,
            "innovation_changed": False,
            "model_architecture_changed": False,
            "checkpoint_changed": False,
            "evaluation_algorithm_changed": False,
            "seed42_deployment_weights_changed": False,
            "default_threshold": 0.5,
        },
        "claim_boundary": {
            "paper_core_established": False,
            "stability_claim_supported": False,
            "multiseed_replication_supported": False,
            "official_test_accessed": False,
        },
        "overwrite_forbidden": True,
    }


def verify_source_lock(
    path: Path = DEFAULT_OUTPUT,
    *,
    metricsfix_source_lock_path: Path = metricsfix_source_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    source = _regular(path, "completion Gate-fix source lock")
    raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompletionGatefixSourceLockError(
            "completion Gate-fix source lock is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        _fail("completion Gate-fix source lock must contain one object")
    _equal(
        "completion Gate-fix source-lock canonical bytes",
        raw,
        canonical_json_bytes(payload),
    )
    expected = build_source_lock(
        metricsfix_source_lock_path=metricsfix_source_lock_path
    )
    _equal(
        "stored/live completion Gate-fix source lock",
        canonical_json_bytes(payload),
        canonical_json_bytes(expected),
    )
    return payload


def freeze_source_lock(
    path: Path = DEFAULT_OUTPUT,
    *,
    metricsfix_source_lock_path: Path = metricsfix_source_lock.DEFAULT_OUTPUT,
) -> tuple[Path, str]:
    destination = Path(path).expanduser()
    if destination.is_symlink():
        _fail(
            "Gate-fix source-lock output must not be a symlink: "
            f"{destination}"
        )
    content = canonical_json_bytes(
        build_source_lock(
            metricsfix_source_lock_path=metricsfix_source_lock_path
        )
    )
    if destination.exists():
        existing = _regular(destination, "existing Gate-fix source lock")
        _equal(
            "stored/live Gate-fix source-lock bytes",
            existing.read_bytes(),
            content,
        )
        return existing, "skipped_identical_locked"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve(), "created"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-once", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metricsfix-source-lock",
        type=Path,
        default=metricsfix_source_lock.DEFAULT_OUTPUT,
    )
    parser.add_argument("--require-runtime-env", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.write_once and args.require_runtime_env:
        _fail("--require-runtime-env is only valid with --verify")
    if args.verify:
        payload = verify_source_lock(
            args.output,
            metricsfix_source_lock_path=args.metricsfix_source_lock,
        )
        action = "verified"
        output = Path(args.output).resolve()
    else:
        output, action = freeze_source_lock(
            args.output,
            metricsfix_source_lock_path=args.metricsfix_source_lock,
        )
        payload = verify_source_lock(
            output,
            metricsfix_source_lock_path=args.metricsfix_source_lock,
        )
    runtime_value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if args.require_runtime_env:
        _equal(
            "runtime CUBLAS_WORKSPACE_CONFIG",
            runtime_value,
            CUBLAS_WORKSPACE_CONFIG,
        )
    print(
        json.dumps(
            {
                "schema": ACTION_SCHEMA,
                "status": "complete",
                "action": action,
                "output": str(output),
                "output_sha256": file_sha256(output),
                "source_count": payload["source_count"],
                "metricsfix_source_lock_sha256": (
                    payload[
                        "upstream_completion_metricsfix_source_lock_v4"
                    ]["sha256"]
                ),
                "cublas_workspace_config": (
                    CUBLAS_WORKSPACE_CONFIG
                    if args.require_runtime_env
                    else runtime_value
                ),
                "runtime_environment_verified": args.require_runtime_env,
                "gpu_queried": False,
                "gpu_command_launched": False,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
