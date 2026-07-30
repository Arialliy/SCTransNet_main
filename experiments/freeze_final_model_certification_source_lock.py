#!/usr/bin/env python3
"""Freeze or verify the source lock used by final-model engineering B/D runs.

This v1 lock authorizes only the fixed-parent engineering replication matrix.
It deliberately marks confirmatory full-pipeline execution unavailable until
the Original/fresh-V4/budget-matched runners and parameterized extension
initializer are implemented and tested under a successor lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    freeze_final_model_certification_parent_lock as parent_lock,
)
from experiments import (  # noqa: E402
    freeze_tpd_ner_v4_qfg_v2_croa_exact_source_lock as d_source_lock,
)
from experiments import (  # noqa: E402
    freeze_tpd_ner_v4_survival_exact_source_lock as b_source_lock,
)


SCHEMA = "sctransnet_final_model_certification_source_lock_v1"
ACTION_SCHEMA = "sctransnet_final_model_certification_source_lock_action_v1"
DEFAULT_OUTPUT_RELATIVE = (
    "experiments/final_model_certification_source_lock_v1.json"
)
DEFAULT_OUTPUT = REPO_ROOT / DEFAULT_OUTPUT_RELATIVE
B_UPSTREAM_LOCK = (
    REPO_ROOT / "experiments/tpd_ner_v4_survival_exact_source_lock.json"
)
D_UPSTREAM_LOCK = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json"
)

# Ordered explicitly for review.  The output lock is excluded to avoid an
# output/self-hash cycle.  The freezer and every verifier it executes are
# included because their live behavior is part of launch authorization.
SOURCE_PATHS = (
    "experiments/freeze_final_model_certification_source_lock.py",
    "experiments/freeze_final_model_certification_parent_lock.py",
    (
        "experiments/"
        "freeze_tpd_ner_v4_qfg_v2_croa_operational_closure_v2.py"
    ),
    "experiments/freeze_tpd_ner_v4_survival_exact_source_lock.py",
    "experiments/freeze_tpd_ner_v4_qfg_v2_croa_exact_source_lock.py",
    "experiments/freeze_tpd_clean_v8_mprs_dch_source_locks.py",
    "experiments/FINAL_MODEL_CERTIFICATION_PROTOCOL_V1.md",
    "experiments/final_model_certification_parent_lock_v1.json",
    "experiments/tpd_ner_v4_survival_exact_source_lock.json",
    (
        "experiments/"
        "tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json"
    ),
    "analysis/collect_final_model_validation_statistics.py",
    "analysis/audit_final_qfg_functional_use.py",
    "analysis/run_final_qfg_six_mode_audit.py",
    "experiments/final_model_replication_seed_contract.py",
    "experiments/final_model_child_initialization_manifest.py",
    "experiments/final_model_replication_exact_core.py",
    "experiments/train_final_model_replication_b_exact.py",
    "experiments/train_final_model_replication_d_exact.py",
    "experiments/prepare_final_model_engineering_replication.py",
    "experiments/watch_final_model_engineering_replication.py",
    "experiments/summarize_final_model_engineering_replication.py",
    "experiments/run_final_model_replication_seed_pair_2x5090.sh",
    "experiments/launch_final_model_replication_2x5090.sh",
    "tests/test_freeze_final_model_certification_parent_lock.py",
    "tests/test_final_model_statistics_cache.py",
    "tests/test_final_qfg_knockout.py",
    "tests/test_run_final_qfg_six_mode_audit.py",
    "tests/test_final_model_replication_seed_contract.py",
    "tests/test_final_model_replication_exact.py",
    "tests/test_freeze_final_model_certification_source_lock.py",
    "tests/test_summarize_final_model_engineering_replication.py",
)


class CertificationSourceLockError(ValueError):
    """The final-model certification source lock is invalid or changed."""


def _fail(message: str) -> None:
    raise CertificationSourceLockError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail(f"value is not canonical JSON: {exc}")


def _relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("source path must be a non-empty string")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or str(pure) != value
    ):
        _fail(f"source path is not canonical repository-relative: {value}")
    return value


def _repo_file(repo_root: Path, relative: str, label: str) -> Path:
    canonical = _relative_path(relative)
    root = Path(repo_root).resolve()
    path = root / canonical
    if path.is_symlink():
        _fail(f"{label} must not be a symlink: {path}")
    try:
        metadata = path.stat()
    except FileNotFoundError:
        _fail(f"{label} is missing: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a regular file: {path}")
    if not path.resolve().is_relative_to(root):
        _fail(f"{label} escapes repository root: {path}")
    return path.resolve()


def sha256_file(path: str | os.PathLike[str]) -> str:
    value = Path(path)
    if value.is_symlink():
        _fail(f"cannot hash symlink: {value}")
    try:
        metadata = value.stat()
    except FileNotFoundError:
        _fail(f"cannot hash missing file: {value}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"cannot hash non-regular file: {value}")
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_source_lock_payload(
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if len(set(SOURCE_PATHS)) != len(SOURCE_PATHS):
        _fail("source path list contains duplicates")
    source_sha256 = {
        relative: sha256_file(
            _repo_file(root, relative, f"source {relative}")
        )
        for relative in sorted(SOURCE_PATHS)
    }
    verified_parent = parent_lock.verify_parent_lock(
        root / parent_lock.DEFAULT_OUTPUT_RELATIVE_PATH,
        repo_root=root,
    )
    b_payload = b_source_lock.verify_source_lock(
        root
        / "experiments/tpd_ner_v4_survival_exact_source_lock.json",
        dataset_dir=root / "datasets",
    )
    d_payload = d_source_lock.verify_source_lock(
        root
        / "experiments/"
        "tpd_ner_v4_qfg_v2_croa_exact_source_lock_v2_optimized.json",
        dataset_dir=root / "datasets",
    )
    return {
        "schema": SCHEMA,
        "status": "locked",
        "lock_kind": "final_model_engineering_replication_sources",
        "execution_scope": "fixed_parent_engineering_b_d_only",
        "source_count": len(source_sha256),
        "source_sha256": source_sha256,
        "upstream_locks": {
            "certification_parent": {
                "path": parent_lock.DEFAULT_OUTPUT_RELATIVE_PATH,
                "sha256": source_sha256[
                    parent_lock.DEFAULT_OUTPUT_RELATIVE_PATH
                ],
                "schema": verified_parent["schema"],
            },
            "b_formal800": {
                "path": (
                    "experiments/"
                    "tpd_ner_v4_survival_exact_source_lock.json"
                ),
                "sha256": source_sha256[
                    "experiments/"
                    "tpd_ner_v4_survival_exact_source_lock.json"
                ],
                "schema": b_payload["schema"],
            },
            "d_formal800": {
                "path": (
                    "experiments/"
                    "tpd_ner_v4_qfg_v2_croa_exact_"
                    "source_lock_v2_optimized.json"
                ),
                "sha256": source_sha256[
                    "experiments/"
                    "tpd_ner_v4_qfg_v2_croa_exact_"
                    "source_lock_v2_optimized.json"
                ],
                "schema": d_payload["schema"],
            },
        },
        "frozen_model": {
            "mainline": "SCTransNet+TPD8+five-node-NER4+QFG2-CROA",
            "mainline_changed": False,
            "innovation_changed": False,
            "builder_compatibility_seed": 42,
            "split_seed": 20260722,
            "default_threshold": 0.5,
        },
        "engineering_matrix": {
            "trajectory_seeds": [3407, 426780603],
            "arms": ["b", "d"],
            "run_count": 4,
            "fixed_parent_seed": 42,
            "full_parameter_child_training": True,
        },
        "readiness": {
            "f0_parent_protocol_complete": True,
            "f1_cache_and_knockout_primitives_complete": True,
            "f1_executable_six_mode_audit_complete": True,
            "f1_six_mode_audit_execution_complete": False,
            "f2_engineering_b_d_runners_complete": True,
            "engineering_b_d_execution_ready": True,
            "confirmatory_full_pipeline_execution_ready": False,
            "paper_stability_gate_execution_ready": False,
        },
        "pending_confirmatory_implementations": [
            "parameterized_extension_initializer",
            "fresh_original_runner",
            "fresh_v4_parent_runner",
            "original_budget_matched_runner",
            "seed_matched_child_parent_manifests",
            "locked_multidataset_and_official_test_evaluators",
        ],
        "official_test_accessed": False,
        "overwrite_forbidden": True,
    }


def verify_source_lock(
    path: str | os.PathLike[str] = DEFAULT_OUTPUT,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        _fail(f"source lock must be a regular file: {source}")
    raw = source.read_bytes()
    try:
        observed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot parse source lock: {exc}")
    if not isinstance(observed, Mapping):
        _fail("source lock must contain one object")
    expected = build_source_lock_payload(repo_root=repo_root)
    if dict(observed) != expected:
        _fail("source lock differs from the live source closure")
    if raw != canonical_json_bytes(expected):
        _fail("source lock is not canonical JSON")
    return expected


def write_source_lock_once(
    path: str | os.PathLike[str] = DEFAULT_OUTPUT,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    destination = Path(path)
    if destination.is_symlink():
        _fail(f"refusing to write through symlink: {destination}")
    if destination.exists():
        raise FileExistsError(
            f"write-once source lock already exists: {destination}"
        )
    payload = build_source_lock_payload(repo_root=repo_root)
    content = canonical_json_bytes(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
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
    verified = verify_source_lock(destination, repo_root=repo_root)
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": "write-once",
        "output": str(destination.resolve()),
        "output_sha256": sha256_file(destination),
        "source_count": verified["source_count"],
        "verified": True,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--write-once", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.plan:
        payload = build_source_lock_payload(repo_root=args.repo_root)
        result = {
            "schema": ACTION_SCHEMA,
            "status": "ready",
            "action": "plan",
            "output": str(args.output.resolve()),
            "output_exists": args.output.exists(),
            "would_write": False,
            "payload_sha256": hashlib.sha256(
                canonical_json_bytes(payload)
            ).hexdigest(),
            "source_count": payload["source_count"],
        }
    elif args.write_once:
        result = write_source_lock_once(
            args.output,
            repo_root=args.repo_root,
        )
    else:
        payload = verify_source_lock(
            args.output,
            repo_root=args.repo_root,
        )
        result = {
            "schema": ACTION_SCHEMA,
            "status": "complete",
            "action": "verify",
            "output": str(args.output.resolve()),
            "output_sha256": sha256_file(args.output),
            "source_count": payload["source_count"],
            "verified": True,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
