#!/usr/bin/env python3
"""Freeze/verify the successor lock for seed42 posttraining and completion.

The training identity remains bound by replay source-lock v4.  This successor
adds only the posttraining and final orchestration layer, and directly binds
their launchers and tests.  It never rewrites replay v4, the F0 certification
source lock, or the parent lock.
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
    freeze_final_model_certification_parent_lock as parent_lock,
)
from experiments import (
    freeze_final_model_certification_source_lock as certification_source_lock,
)
from experiments import (
    freeze_final_model_seed42_certification_replay_source_lock
    as replay_source_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_source_lock_v1"
)
ACTION_SCHEMA = (
    "sctransnet_final_model_seed42_certification_completion_source_lock_action_v1"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/"
    "final_model_seed42_certification_completion_source_lock_v1.json"
)

SOURCE_RELATIVE_PATHS = (
    "experiments/freeze_final_model_seed42_certification_completion_source_lock.py",
    "experiments/final_model_seed42_certification_replay_posttraining.py",
    "experiments/run_final_model_seed42_certification_replay_posttraining_2x5090.sh",
    "tests/test_final_model_seed42_certification_replay_posttraining.py",
    "experiments/final_model_seed42_certification_completion.py",
    "experiments/run_final_model_seed42_certification_completion.sh",
    "tests/test_final_model_seed42_certification_completion.py",
    "analysis/verify_final_qfg_six_mode_audit_deep.py",
    "experiments/evaluate_final_model_engineering_replication_pd_fa.py",
    "experiments/analyze_final_model_engineering_paired_screen.py",
    "experiments/adjudicate_final_model_engineering_gate.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "experiments/evaluate_tpd_clean_v6_pd_fa.py",
    "experiments/evaluate_tpd_clean_v8_mprs_dch_pd_fa.py",
    "experiments/evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa.py",
    "experiments/evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa.py",
    "experiments/evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_pd_fa.py",
    "tests/test_evaluate_final_model_engineering_replication_pd_fa.py",
    "tests/test_analyze_final_model_engineering_paired_screen.py",
    "tests/test_adjudicate_final_model_engineering_gate.py",
    "tests/test_evaluate_tpd_clean_v6_pd_fa.py",
    "tests/test_evaluate_tpd_clean_v6_pd_fa_checkpoint_compat.py",
    "tests/test_evaluate_tpd_clean_v8_mprs_dch_pd_fa.py",
    "tests/test_evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa.py",
    "tests/test_evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa.py",
    "tests/test_evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_pd_fa.py",
)


class CompletionSourceLockError(ValueError):
    """The completion source lock or its live source tree differs."""


def _fail(message: str) -> None:
    raise CompletionSourceLockError(message)


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
    source = _regular(path, "source")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    source = Path(path).resolve()
    try:
        return source.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        _fail(f"path lies outside repository: {source}")


def _upstream_binding(path: Path, schema: str) -> dict[str, str]:
    return {
        "path": _relative(path),
        "schema": schema,
        "sha256": file_sha256(path),
    }


def build_source_lock(
    *,
    replay_source_lock_path: Path = replay_source_lock.DEFAULT_OUTPUT,
    certification_source_lock_path: Path = certification_source_lock.DEFAULT_OUTPUT,
    parent_lock_path: Path = parent_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    replay_source_lock.verify_source_lock(
        replay_source_lock_path,
        upstream_source_lock_path=certification_source_lock_path,
        parent_lock_path=parent_lock_path,
    )
    certification_source_lock.verify_source_lock(
        certification_source_lock_path,
        repo_root=REPO_ROOT,
    )
    parent_lock.verify_parent_lock(parent_lock_path, repo_root=REPO_ROOT)
    source_sha256 = {
        relative: file_sha256(REPO_ROOT / relative)
        for relative in SOURCE_RELATIVE_PATHS
    }
    return {
        "schema": SCHEMA,
        "status": "locked",
        "lock_kind": "seed42_posttraining_and_final_completion_sources",
        "scope": "new_seed42_replay_plus_frozen_deployment_qfg_audit",
        "source_count": len(source_sha256),
        "source_sha256": source_sha256,
        "upstream_locks": {
            "seed42_replay_source_lock_v4": _upstream_binding(
                Path(replay_source_lock_path),
                replay_source_lock.SCHEMA,
            ),
                "f0_certification_source_lock_v1": _upstream_binding(
                    Path(certification_source_lock_path),
                    certification_source_lock.SCHEMA,
            ),
            "certification_parent_lock_v1": _upstream_binding(
                Path(parent_lock_path),
                parent_lock.LOCK_SCHEMA,
            ),
        },
        "frozen_model": {
            "mainline": "SCTransNet+TPD8+five-node-NER4+QFG2-CROA",
            "mainline_changed": False,
            "innovation_changed": False,
            "seed42_deployment_weights_changed": False,
            "default_threshold": 0.5,
        },
        "execution_scope": {
            "trajectory_seed": 42,
            "new_replay_run_count": 2,
            "new_replay_sweep_count": 4,
            "supplementary_seed_3407_used_in_current_gate": False,
            "seed_426780603_scheduled": False,
            "old_seed42_stage_results_used_as_new_replay": False,
            "f1_subject": "frozen_seed42_deployment_d",
        },
        "readiness": {
            "f0_protocol_and_locks_complete": True,
            "f1_runner_and_deep_verifier_complete": True,
            "f2_full_parameter_contract_runner_tests_complete": True,
            "new_seed42_posttraining_runner_tests_complete": True,
            "top_level_completion_runner_tests_complete": True,
            "new_seed42_formal_execution_complete": False,
            "final_attestation_execution_complete": False,
        },
        "claim_boundary": {
            "single_seed_internal_validation_only": True,
            "official_test_accessed": False,
            "paper_core_established": False,
            "stability_claim_supported": False,
            "multiseed_replication_supported": False,
        },
        "overwrite_forbidden": True,
    }


def verify_source_lock(
    path: Path = DEFAULT_OUTPUT,
    *,
    replay_source_lock_path: Path = replay_source_lock.DEFAULT_OUTPUT,
    certification_source_lock_path: Path = certification_source_lock.DEFAULT_OUTPUT,
    parent_lock_path: Path = parent_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    source = _regular(path, "completion source lock")
    raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CompletionSourceLockError(
            "completion source lock is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        _fail("completion source lock must contain one object")
    _equal(
        "completion source-lock canonical bytes",
        raw,
        canonical_json_bytes(payload),
    )
    expected = build_source_lock(
        replay_source_lock_path=replay_source_lock_path,
        certification_source_lock_path=certification_source_lock_path,
        parent_lock_path=parent_lock_path,
    )
    _equal(
        "stored/live completion source lock",
        canonical_json_bytes(payload),
        canonical_json_bytes(expected),
    )
    return payload


def freeze_source_lock(
    path: Path = DEFAULT_OUTPUT,
    *,
    replay_source_lock_path: Path = replay_source_lock.DEFAULT_OUTPUT,
    certification_source_lock_path: Path = certification_source_lock.DEFAULT_OUTPUT,
    parent_lock_path: Path = parent_lock.DEFAULT_OUTPUT,
) -> tuple[Path, str]:
    destination = Path(path).expanduser()
    if destination.is_symlink():
        _fail(f"completion source-lock output must not be a symlink: {destination}")
    payload = build_source_lock(
        replay_source_lock_path=replay_source_lock_path,
        certification_source_lock_path=certification_source_lock_path,
        parent_lock_path=parent_lock_path,
    )
    content = canonical_json_bytes(payload)
    if destination.exists():
        stored = _regular(destination, "existing completion source lock")
        _equal("stored/live completion source lock", stored.read_bytes(), content)
        return stored, "skipped_identical_locked"
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
                "concurrently created completion source lock",
            )
            _equal("concurrent/live completion source lock", concurrent.read_bytes(), content)
            return concurrent, "skipped_identical_locked"
    finally:
        temporary.unlink(missing_ok=True)
    stored = _regular(destination, "written completion source lock")
    _equal("written completion source-lock bytes", stored.read_bytes(), content)
    return stored, "created"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--replay-source-lock",
        type=Path,
        default=replay_source_lock.DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--certification-source-lock",
        type=Path,
        default=certification_source_lock.DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--parent-lock",
        type=Path,
        default=parent_lock.DEFAULT_OUTPUT,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    common = {
        "replay_source_lock_path": args.replay_source_lock,
        "certification_source_lock_path": args.certification_source_lock,
        "parent_lock_path": args.parent_lock,
    }
    if args.freeze:
        path, action = freeze_source_lock(args.output, **common)
        payload = {
            "schema": ACTION_SCHEMA,
            "status": "locked",
            "action": action,
            "output": str(path),
            "sha256": file_sha256(path),
        }
    else:
        stored = verify_source_lock(args.output, **common)
        payload = {
            "schema": ACTION_SCHEMA,
            "status": "verified_locked",
            "output": str(Path(args.output).resolve()),
            "sha256": file_sha256(args.output),
            "source_count": stored["source_count"],
        }
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


__all__ = [
    "ACTION_SCHEMA",
    "CompletionSourceLockError",
    "DEFAULT_OUTPUT",
    "SCHEMA",
    "SOURCE_RELATIVE_PATHS",
    "build_source_lock",
    "canonical_json_bytes",
    "file_sha256",
    "freeze_source_lock",
    "main",
    "verify_source_lock",
]


if __name__ == "__main__":
    main()
