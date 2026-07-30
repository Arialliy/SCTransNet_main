#!/usr/bin/env python3
"""Freeze and verify the additive seed42 overlay-scope repair layer."""

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
    freeze_final_model_seed42_certification_completion_envfix_source_lock
    as envfix_source_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_"
    "overlayfix_source_lock_v3"
)
ACTION_SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_"
    "overlayfix_source_lock_action_v3"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/"
    "final_model_seed42_certification_completion_overlayfix_source_lock_v3.json"
)
EXPECTED_ENVFIX_SOURCE_LOCK_SHA256 = (
    "d1cb351e90d055ff4ed0e5b294e3faa841de3b94a158c325d8b98e6c808650f3"
)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
FROZEN_POSTTRAINING_RELATIVE_PATH = (
    "experiments/final_model_seed42_certification_replay_posttraining.py"
)
SOURCE_RELATIVE_PATHS = (
    "experiments/"
    "final_model_seed42_certification_replay_posttraining_overlayfix_v3.py",
    "experiments/"
    "final_model_seed42_certification_python_overlayfix_v3.sh",
    "experiments/"
    "run_final_model_seed42_certification_completion_overlayfix_v3.sh",
    "experiments/"
    "final_model_seed42_certification_completion_overlayfix_attestation_v3.py",
    "experiments/"
    "freeze_final_model_seed42_certification_completion_overlayfix_source_lock.py",
    "tests/"
    "test_final_model_seed42_certification_completion_overlayfix.py",
)


class CompletionOverlayfixSourceLockError(ValueError):
    """The additive overlay-scope lock or one of its sources differs."""


def _fail(message: str) -> None:
    raise CompletionOverlayfixSourceLockError(message)


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
    path: Path = envfix_source_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    payload = envfix_source_lock.verify_source_lock(path)
    _equal(
        "env-fix source-lock schema",
        payload.get("schema"),
        envfix_source_lock.SCHEMA,
    )
    upstream_sha256 = file_sha256(path)
    _equal(
        "env-fix source-lock SHA-256",
        upstream_sha256,
        EXPECTED_ENVFIX_SOURCE_LOCK_SHA256,
    )
    completion = payload.get("upstream_completion_source_lock_v1")
    if not isinstance(completion, Mapping):
        _fail("env-fix source lock omits completion source-lock v1")
    source_sha256 = completion.get("source_sha256")
    if source_sha256 is not None:
        _fail("unexpected nested source map in env-fix upstream binding")
    completion_path = REPO_ROOT / str(completion.get("path"))
    completion_payload = (
        envfix_source_lock.completion_source_lock.verify_source_lock(
            completion_path
        )
    )
    completion_sources = completion_payload.get("source_sha256")
    if not isinstance(completion_sources, Mapping):
        _fail("completion source-lock v1 source map is missing")
    frozen_posttraining_sha256 = completion_sources.get(
        FROZEN_POSTTRAINING_RELATIVE_PATH
    )
    if not isinstance(frozen_posttraining_sha256, str):
        _fail("completion source-lock v1 omits frozen post-training adapter")
    _equal(
        "frozen post-training adapter live SHA-256",
        file_sha256(REPO_ROOT / FROZEN_POSTTRAINING_RELATIVE_PATH),
        frozen_posttraining_sha256,
    )
    return {
        "path": _repo_relative(Path(path)),
        "schema": envfix_source_lock.SCHEMA,
        "sha256": upstream_sha256,
        "source_count": payload["source_count"],
        "completion_source_lock_v1": {
            "path": str(completion["path"]),
            "schema": str(completion["schema"]),
            "sha256": str(completion["sha256"]),
        },
        "frozen_posttraining_adapter": {
            "path": FROZEN_POSTTRAINING_RELATIVE_PATH,
            "sha256": frozen_posttraining_sha256,
        },
    }


def build_source_lock(
    *,
    envfix_source_lock_path: Path = envfix_source_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    upstream = _verified_upstream(envfix_source_lock_path)
    source_sha256 = {
        relative: file_sha256(REPO_ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }
    return {
        "schema": SCHEMA,
        "status": "locked",
        "lock_kind": "additive_completion_overlay_scope_fix",
        "scope": "seed42_posttraining_model_build_overlay_lifetime_only",
        "source_count": len(source_sha256),
        "source_sha256": source_sha256,
        "upstream_completion_envfix_source_lock_v2": upstream,
        "repair_contract": {
            "failure_stage": (
                "checkpoint_local_result_prewrite_live_revalidation"
            ),
            "frozen_behavior": (
                "replay_trainer_overlay_spanned_shared_evaluator_main"
            ),
            "successor_behavior": (
                "replay_trainer_overlay_spans_model_build_call_only"
            ),
            "dynamic_sweep_evaluator_file_binding": (
                FROZEN_POSTTRAINING_RELATIVE_PATH
            ),
            "dynamic_sweep_evaluator_hash_matches_seed42_adapter": True,
            "prewrite_live_revalidation_preserved": True,
            "frozen_posttraining_adapter_unchanged": True,
            "frozen_completion_v1_unchanged": True,
            "envfix_v2_unchanged": True,
            "python_module_redirection_exact_match_only": True,
            "all_other_python_invocations_forwarded_verbatim": True,
            "successor_attestation_written_after_base_completion": True,
        },
        "execution_contract": {
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            "gpu_assignment": {"b": 2, "d": 3},
            "existing_write_once_summary_reused": True,
            "existing_complete_prediction_caches_validated": True,
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
    envfix_source_lock_path: Path = envfix_source_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    source = _regular(path, "completion overlay-fix source lock")
    raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompletionOverlayfixSourceLockError(
            "completion overlay-fix source lock is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        _fail("completion overlay-fix source lock must contain one object")
    _equal(
        "completion overlay-fix source-lock canonical bytes",
        raw,
        canonical_json_bytes(payload),
    )
    expected = build_source_lock(
        envfix_source_lock_path=envfix_source_lock_path
    )
    _equal(
        "stored/live completion overlay-fix source lock",
        canonical_json_bytes(payload),
        canonical_json_bytes(expected),
    )
    return payload


def freeze_source_lock(
    path: Path = DEFAULT_OUTPUT,
    *,
    envfix_source_lock_path: Path = envfix_source_lock.DEFAULT_OUTPUT,
) -> tuple[Path, str]:
    destination = Path(path).expanduser()
    if destination.is_symlink():
        _fail(
            "overlay-fix source-lock output must not be a symlink: "
            f"{destination}"
        )
    content = canonical_json_bytes(
        build_source_lock(
            envfix_source_lock_path=envfix_source_lock_path
        )
    )
    if destination.exists():
        existing = _regular(
            destination,
            "existing overlay-fix source lock",
        )
        _equal(
            "stored/live overlay-fix source-lock bytes",
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
        "--envfix-source-lock",
        type=Path,
        default=envfix_source_lock.DEFAULT_OUTPUT,
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
            envfix_source_lock_path=args.envfix_source_lock,
        )
        action = "verified"
        output = Path(args.output).resolve()
    else:
        output, action = freeze_source_lock(
            args.output,
            envfix_source_lock_path=args.envfix_source_lock,
        )
        payload = verify_source_lock(
            output,
            envfix_source_lock_path=args.envfix_source_lock,
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
                "envfix_source_lock_sha256": (
                    payload[
                        "upstream_completion_envfix_source_lock_v2"
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
