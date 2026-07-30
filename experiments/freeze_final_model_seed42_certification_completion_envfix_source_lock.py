#!/usr/bin/env python3
"""Freeze and verify the additive seed42 completion env-fix layer.

This successor never rewrites the already frozen completion source-lock v1.
It binds only the small recovery wrapper, this builder/verifier, and their
test, while treating completion source-lock v1 as an immutable upstream.
"""

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
    freeze_final_model_seed42_certification_completion_source_lock
    as completion_source_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_envfix_"
    "source_lock_v2"
)
ACTION_SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_envfix_"
    "source_lock_action_v2"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/"
    "final_model_seed42_certification_completion_envfix_source_lock_v2.json"
)
EXPECTED_COMPLETION_SOURCE_LOCK_SHA256 = (
    "8ce245e3f609bd929ae9405daaae11a4d6f5aa470965c2896139238cfcf43ee7"
)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
ORIGINAL_COMPLETION_RELATIVE_PATH = (
    "experiments/run_final_model_seed42_certification_completion.sh"
)
SOURCE_RELATIVE_PATHS = (
    "experiments/"
    "freeze_final_model_seed42_certification_completion_envfix_source_lock.py",
    "experiments/"
    "run_final_model_seed42_certification_completion_envfix_v2.sh",
    "tests/test_final_model_seed42_certification_completion_envfix.py",
)


class CompletionEnvfixSourceLockError(ValueError):
    """The additive env-fix lock or its live sources differ."""


def _fail(message: str) -> None:
    raise CompletionEnvfixSourceLockError(message)


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
    path: Path = completion_source_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    payload = completion_source_lock.verify_source_lock(path)
    _equal(
        "completion source-lock schema",
        payload.get("schema"),
        completion_source_lock.SCHEMA,
    )
    upstream_sha256 = file_sha256(path)
    _equal(
        "completion source-lock SHA-256",
        upstream_sha256,
        EXPECTED_COMPLETION_SOURCE_LOCK_SHA256,
    )
    source_sha256 = payload.get("source_sha256")
    if not isinstance(source_sha256, Mapping):
        _fail("completion source-lock source map is missing")
    locked_completion_sha256 = source_sha256.get(
        ORIGINAL_COMPLETION_RELATIVE_PATH
    )
    if not isinstance(locked_completion_sha256, str):
        _fail("completion source-lock omits the original completion shell")
    _equal(
        "original completion shell live SHA-256",
        file_sha256(REPO_ROOT / ORIGINAL_COMPLETION_RELATIVE_PATH),
        locked_completion_sha256,
    )
    return {
        "path": _repo_relative(Path(path)),
        "schema": completion_source_lock.SCHEMA,
        "sha256": upstream_sha256,
        "source_count": payload["source_count"],
        "original_completion_shell": {
            "path": ORIGINAL_COMPLETION_RELATIVE_PATH,
            "sha256": locked_completion_sha256,
        },
    }


def build_source_lock(
    *,
    completion_source_lock_path: Path = completion_source_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    upstream = _verified_upstream(completion_source_lock_path)
    source_sha256 = {
        relative: file_sha256(REPO_ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }
    return {
        "schema": SCHEMA,
        "status": "locked",
        "lock_kind": "additive_completion_runtime_environment_fix",
        "scope": "seed42_completion_cublas_workspace_env_only",
        "source_count": len(source_sha256),
        "source_sha256": source_sha256,
        "upstream_completion_source_lock_v1": upstream,
        "runtime_environment": {
            "name": "CUBLAS_WORKSPACE_CONFIG",
            "required_value": CUBLAS_WORKSPACE_CONFIG,
            "wrapper_forces_value": True,
            "caller_value_may_override_required_value": False,
        },
        "execution_contract": {
            "original_completion_script_unchanged": True,
            "original_completion_arguments_forwarded_verbatim": True,
            "original_completion_executed_via_exec": True,
            "existing_write_once_summary_reused": True,
            "existing_valid_results_reused": True,
            "mainline_changed": False,
            "innovation_changed": False,
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
    completion_source_lock_path: Path = completion_source_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    source = _regular(path, "completion env-fix source lock")
    raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompletionEnvfixSourceLockError(
            "completion env-fix source lock is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        _fail("completion env-fix source lock must contain one object")
    _equal(
        "completion env-fix source-lock canonical bytes",
        raw,
        canonical_json_bytes(payload),
    )
    expected = build_source_lock(
        completion_source_lock_path=completion_source_lock_path
    )
    _equal(
        "stored/live completion env-fix source lock",
        canonical_json_bytes(payload),
        canonical_json_bytes(expected),
    )
    return payload


def freeze_source_lock(
    path: Path = DEFAULT_OUTPUT,
    *,
    completion_source_lock_path: Path = completion_source_lock.DEFAULT_OUTPUT,
) -> tuple[Path, str]:
    destination = Path(path).expanduser()
    if destination.is_symlink():
        _fail(f"env-fix source-lock output must not be a symlink: {destination}")
    content = canonical_json_bytes(
        build_source_lock(
            completion_source_lock_path=completion_source_lock_path
        )
    )
    if destination.exists():
        existing = _regular(destination, "existing env-fix source lock")
        _equal("stored/live env-fix source-lock bytes", existing.read_bytes(), content)
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
        "--completion-source-lock",
        type=Path,
        default=completion_source_lock.DEFAULT_OUTPUT,
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
            completion_source_lock_path=args.completion_source_lock,
        )
        action = "verified"
        output = Path(args.output).resolve()
    else:
        output, action = freeze_source_lock(
            args.output,
            completion_source_lock_path=args.completion_source_lock,
        )
        payload = verify_source_lock(
            output,
            completion_source_lock_path=args.completion_source_lock,
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
                "completion_source_lock_sha256": (
                    payload["upstream_completion_source_lock_v1"]["sha256"]
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

