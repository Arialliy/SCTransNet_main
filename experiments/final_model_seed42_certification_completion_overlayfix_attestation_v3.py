#!/usr/bin/env python3
"""Write and verify the additive seed42 completion-repair attestation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

from experiments import final_model_seed42_certification_completion as completion
from experiments import (
    freeze_final_model_seed42_certification_completion_overlayfix_source_lock
    as overlayfix_source_lock,
)


SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_"
    "overlayfix_attestation_v3"
)
ACTION_SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_"
    "overlayfix_attestation_action_v3"
)
DEFAULT_OUTPUT = (
    completion.replay_contract.DEFAULT_OUTPUT_ROOT
    / "final_model_seed42_certification_overlayfix_attestation_v3.json"
)
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class CompletionOverlayfixAttestationError(ValueError):
    """The completion repair evidence is missing or differs."""


def _fail(message: str) -> None:
    raise CompletionOverlayfixAttestationError(message)


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


def _sha256(path: Path, label: str) -> str:
    source = _regular(path, label)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, label: str) -> dict[str, str]:
    source = _regular(path, label)
    return {
        "path": str(source),
        "sha256": _sha256(source, label),
    }


def _canonical_object(path: Path, label: str) -> dict[str, Any]:
    source = _regular(path, label)
    raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompletionOverlayfixAttestationError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        _fail(f"{label} must contain one object")
    _equal(f"{label} canonical bytes", raw, canonical_json_bytes(payload))
    return payload


def _write_or_validate(
    path: Path,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    destination = Path(path).expanduser()
    if destination.is_symlink():
        _fail(f"overlay-fix attestation must not be a symlink: {destination}")
    content = canonical_json_bytes(payload)
    if destination.exists():
        stored = _regular(destination, "existing overlay-fix attestation")
        _equal(
            "stored/live overlay-fix attestation",
            stored.read_bytes(),
            content,
        )
        return stored, "skipped_identical_complete"
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
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            concurrent = _regular(
                destination,
                "concurrently created overlay-fix attestation",
            )
            _equal(
                "concurrent/live overlay-fix attestation",
                concurrent.read_bytes(),
                content,
            )
    finally:
        temporary.unlink(missing_ok=True)
    stored = _regular(destination, "written overlay-fix attestation")
    _equal(
        "written overlay-fix attestation bytes",
        stored.read_bytes(),
        content,
    )
    return stored, "created"


def build_attestation(
    *,
    source_lock_path: Path = overlayfix_source_lock.DEFAULT_OUTPUT,
    base_attestation_path: Path = completion.DEFAULT_ATTESTATION,
) -> dict[str, Any]:
    source_lock = overlayfix_source_lock.verify_source_lock(source_lock_path)
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
        "upstream_completion_envfix_source_lock_v2"
    )
    if not isinstance(upstream, Mapping):
        _fail("overlay-fix source lock omits env-fix v2")
    return {
        "schema": SCHEMA,
        "status": "complete",
        "decision": "FIXED_SEED42_INTERNAL_CERTIFICATION_CLOSED",
        "scope": (
            "additive_runtime_repair_binding_for_base_seed42_completion"
        ),
        "base_completion_attestation": _artifact(
            base_attestation_path,
            "base completion attestation",
        ),
        "overlayfix_source_lock_v3": {
            **_artifact(source_lock_path, "overlay-fix source lock v3"),
            "schema": source_lock["schema"],
            "source_count": source_lock["source_count"],
        },
        "upstream_envfix_source_lock_v2": {
            "path": str(
                (
                    overlayfix_source_lock.REPO_ROOT
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
            "frozen_completion_v1_unchanged": True,
            "frozen_posttraining_v1_unchanged": True,
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
    source_lock_path: Path = overlayfix_source_lock.DEFAULT_OUTPUT,
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
        "action": "finalize-overlayfix-attestation",
        "attestation_action": action,
        "attestation": _artifact(path, "overlay-fix attestation"),
        "runtime_environment_verified": require_runtime_env,
        "paper_core_established": False,
        "stability_claim_supported": False,
    }


def verify_attestation(
    *,
    source_lock_path: Path = overlayfix_source_lock.DEFAULT_OUTPUT,
    base_attestation_path: Path = completion.DEFAULT_ATTESTATION,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    stored = _canonical_object(output, "overlay-fix attestation")
    expected = build_attestation(
        source_lock_path=source_lock_path,
        base_attestation_path=base_attestation_path,
    )
    _equal(
        "stored/live overlay-fix attestation",
        canonical_json_bytes(stored),
        canonical_json_bytes(expected),
    )
    return {
        "schema": ACTION_SCHEMA,
        "status": "verified_complete",
        "action": "verify-overlayfix-attestation",
        "attestation": _artifact(output, "overlay-fix attestation"),
        "paper_core_established": False,
        "stability_claim_supported": False,
    }


def dry_run_payload(
    *,
    source_lock_path: Path,
    base_attestation_path: Path,
    output: Path,
) -> dict[str, Any]:
    source_lock = overlayfix_source_lock.verify_source_lock(
        source_lock_path
    )
    return {
        "schema": ACTION_SCHEMA,
        "status": "dry_run_complete",
        "action": "dry-run",
        "source_lock": _artifact(
            source_lock_path,
            "overlay-fix source lock",
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
        default=overlayfix_source_lock.DEFAULT_OUTPUT,
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
