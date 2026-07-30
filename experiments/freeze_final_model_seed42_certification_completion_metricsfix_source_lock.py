#!/usr/bin/env python3
"""Freeze and verify the additive seed42 checkpoint-metrics repair layer."""

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
    freeze_final_model_seed42_certification_completion_overlayfix_source_lock
    as overlayfix_source_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_"
    "metricsfix_source_lock_v4"
)
ACTION_SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_"
    "metricsfix_source_lock_action_v4"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/"
    "final_model_seed42_certification_completion_metricsfix_source_lock_v4.json"
)
EXPECTED_OVERLAYFIX_SOURCE_LOCK_SHA256 = (
    "6aeca56da4f84cb9621a0a7fa3953e6a32afc5b33e860901fc476d6f3aedd7a2"
)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
SOURCE_RELATIVE_PATHS = (
    "experiments/"
    "final_model_seed42_certification_replay_posttraining_metricsfix_v4.py",
    "experiments/"
    "final_model_seed42_certification_python_metricsfix_v4.sh",
    "experiments/"
    "run_final_model_seed42_certification_completion_metricsfix_v4.sh",
    "experiments/"
    "freeze_final_model_seed42_certification_completion_metricsfix_source_lock.py",
    "experiments/"
    "final_model_seed42_certification_completion_metricsfix_attestation_v4.py",
    "tests/"
    "test_final_model_seed42_certification_completion_metricsfix.py",
)


class CompletionMetricsfixSourceLockError(ValueError):
    """The additive metric-projection lock or one of its sources differs."""


def _fail(message: str) -> None:
    raise CompletionMetricsfixSourceLockError(message)


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
    path: Path = overlayfix_source_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    payload = overlayfix_source_lock.verify_source_lock(path)
    _equal(
        "overlay-fix source-lock schema",
        payload.get("schema"),
        overlayfix_source_lock.SCHEMA,
    )
    upstream_sha256 = file_sha256(path)
    _equal(
        "overlay-fix source-lock SHA-256",
        upstream_sha256,
        EXPECTED_OVERLAYFIX_SOURCE_LOCK_SHA256,
    )
    envfix = payload.get("upstream_completion_envfix_source_lock_v2")
    if not isinstance(envfix, Mapping):
        _fail("overlay-fix source lock omits env-fix v2")
    return {
        "path": _repo_relative(Path(path)),
        "schema": overlayfix_source_lock.SCHEMA,
        "sha256": upstream_sha256,
        "source_count": payload["source_count"],
        "envfix_source_lock_v2": {
            "path": str(envfix["path"]),
            "schema": str(envfix["schema"]),
            "sha256": str(envfix["sha256"]),
        },
    }


def build_source_lock(
    *,
    overlayfix_source_lock_path: Path = overlayfix_source_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    upstream = _verified_upstream(overlayfix_source_lock_path)
    source_sha256 = {
        relative: file_sha256(REPO_ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }
    return {
        "schema": SCHEMA,
        "status": "locked",
        "lock_kind": "additive_checkpoint_metrics_projection_fix",
        "scope": (
            "seed42_checkpoint_full_source_validation_then_11_metric_view"
        ),
        "source_count": len(source_sha256),
        "source_sha256": source_sha256,
        "upstream_completion_overlayfix_source_lock_v3": upstream,
        "repair_contract": {
            "shared_sweep_checkpoint_metric_count": 17,
            "frozen_report_projection_metric_count": 11,
            "persistent_result_metric_count": 11,
            "auxiliary_fields_causing_key_mismatch": [
                "niou",
                "pixel_f1",
                "pixel_precision",
                "pixel_recall",
                "predicted_object_count",
                "val_loss",
            ],
            "shared_full_metrics_already_checked_against_checkpoint": True,
            "shared_full_metrics_already_checked_against_summary": True,
            "shared_full_metrics_already_checked_against_metrics_log": True,
            "full_17_fields_checked_against_checkpoint": True,
            "full_17_fields_checked_against_summary": True,
            "full_17_fields_checked_against_metrics_log": True,
            "raw_full_fixed_threshold_audit_exactly_revalidated": True,
            "persistent_projection_uses_frozen_metric_projection": True,
            "projected_fixed_threshold_audit_uses_frozen_function": True,
            "all_11_persistent_values_require_exact_request_match": True,
            "base_v1_result_and_closure_verifier_compatibility_preserved": True,
            "prewrite_live_revalidation_preserved": True,
            "frozen_v1_v2_v3_sources_unchanged": True,
            "successor_attestation_written_after_base_completion": True,
        },
        "execution_contract": {
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            "gpu_assignment": {"b": 2, "d": 3},
            "existing_write_once_summary_reused": True,
            "existing_complete_prediction_caches_recomputed_and_validated": True,
            "cache_only_fast_path_used": False,
            "existing_valid_results_reused": True,
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
    overlayfix_source_lock_path: Path = overlayfix_source_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    source = _regular(path, "completion metrics-fix source lock")
    raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompletionMetricsfixSourceLockError(
            "completion metrics-fix source lock is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        _fail("completion metrics-fix source lock must contain one object")
    _equal(
        "completion metrics-fix source-lock canonical bytes",
        raw,
        canonical_json_bytes(payload),
    )
    expected = build_source_lock(
        overlayfix_source_lock_path=overlayfix_source_lock_path
    )
    _equal(
        "stored/live completion metrics-fix source lock",
        canonical_json_bytes(payload),
        canonical_json_bytes(expected),
    )
    return payload


def freeze_source_lock(
    path: Path = DEFAULT_OUTPUT,
    *,
    overlayfix_source_lock_path: Path = overlayfix_source_lock.DEFAULT_OUTPUT,
) -> tuple[Path, str]:
    destination = Path(path).expanduser()
    if destination.is_symlink():
        _fail(
            "metrics-fix source-lock output must not be a symlink: "
            f"{destination}"
        )
    content = canonical_json_bytes(
        build_source_lock(
            overlayfix_source_lock_path=overlayfix_source_lock_path
        )
    )
    if destination.exists():
        existing = _regular(
            destination,
            "existing metrics-fix source lock",
        )
        _equal(
            "stored/live metrics-fix source-lock bytes",
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
        "--overlayfix-source-lock",
        type=Path,
        default=overlayfix_source_lock.DEFAULT_OUTPUT,
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
            overlayfix_source_lock_path=args.overlayfix_source_lock,
        )
        action = "verified"
        output = Path(args.output).resolve()
    else:
        output, action = freeze_source_lock(
            args.output,
            overlayfix_source_lock_path=args.overlayfix_source_lock,
        )
        payload = verify_source_lock(
            output,
            overlayfix_source_lock_path=args.overlayfix_source_lock,
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
                "overlayfix_source_lock_sha256": (
                    payload[
                        "upstream_completion_overlayfix_source_lock_v3"
                    ]["sha256"]
                ),
                "runtime_environment_verified": bool(
                    args.require_runtime_env
                ),
                "cublas_workspace_config": (
                    runtime_value if args.require_runtime_env else None
                ),
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
