#!/usr/bin/env python3
"""Freeze/verify the successor sources for the new seed-42 B/D replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from experiments import (
    final_model_seed42_certification_replay_contract as replay_contract,
)
from experiments import (
    freeze_final_model_certification_parent_lock as parent_lock,
)
from experiments import (
    freeze_final_model_certification_source_lock as upstream_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (
    "sctransnet_final_model_seed42_certification_replay_source_lock_v4"
)
ACTION_SCHEMA = (
    "sctransnet_final_model_seed42_certification_replay_source_lock_action_v3"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "experiments/final_model_seed42_certification_replay_source_lock_v4.json"
)
SOURCE_PATHS = (
    "experiments/final_model_seed42_certification_replay_contract.py",
    "experiments/final_model_seed42_certification_replay_exact_core.py",
    "experiments/freeze_final_model_seed42_certification_replay_source_lock.py",
    "experiments/train_final_model_seed42_certification_replay_b_exact.py",
    "experiments/train_final_model_seed42_certification_replay_d_exact.py",
    "experiments/run_final_model_seed42_certification_replay_pair_2x5090.sh",
    "tests/test_final_model_seed42_certification_replay.py",
)


class Seed42ReplaySourceLockError(ValueError):
    """The successor source-lock payload or live source tree differs."""


def _fail(message: str) -> None:
    raise Seed42ReplaySourceLockError(message)


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


def _regular(path: Path, label: str) -> Path:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        _fail(f"{label} must be a regular non-symlink file: {value}")
    return value.resolve()


def build_payload(
    *,
    contract_path: Path = replay_contract.DEFAULT_CONTRACT,
    manifest_directory: Path = replay_contract.DEFAULT_MANIFEST_DIRECTORY,
    source_lock_path: Path = upstream_lock.DEFAULT_OUTPUT,
    parent_lock_path: Path = parent_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    contract = replay_contract.load_contract(
        contract_path,
        source_lock_path=source_lock_path,
        parent_lock_path=parent_lock_path,
    )
    upstream_lock.verify_source_lock(source_lock_path, repo_root=REPO_ROOT)
    parent_lock.verify_parent_lock(parent_lock_path, repo_root=REPO_ROOT)
    manifests = {}
    for arm in ("b", "d"):
        path = replay_contract.manifest_path(manifest_directory, arm)
        replay_contract.load_child_manifest(
            path,
            arm=arm,
            contract_path=contract_path,
            source_lock_path=source_lock_path,
            parent_lock_path=parent_lock_path,
        )
        manifests[arm] = {
            "path": path.resolve().relative_to(REPO_ROOT).as_posix(),
            "sha256": replay_contract.file_sha256(path),
        }
    sources = {}
    for relative in SOURCE_PATHS:
        path = _regular(REPO_ROOT / relative, f"successor source {relative}")
        sources[relative] = replay_contract.file_sha256(path)
    return {
        "schema": SCHEMA,
        "status": "locked",
        "lock_kind": "new_seed42_certification_replay_sources",
        "scope": contract["scope"],
        "frozen_model": contract["frozen_model"],
        "trajectory_seed": replay_contract.TRAJECTORY_SEED,
        "split_seed": replay_contract.SPLIT_SEED,
        "default_threshold": replay_contract.DEFAULT_THRESHOLD,
        "run_count": 2,
        "arms": ["b", "d"],
        "output_root": contract["replay_identity"]["output_root"],
        "contract": {
            "path": Path(contract_path)
            .resolve()
            .relative_to(REPO_ROOT)
            .as_posix(),
            "sha256": replay_contract.file_sha256(contract_path),
        },
        "child_manifests": manifests,
        "upstream_locks": {
            "certification_source_lock_v1": {
                "path": Path(source_lock_path)
                .resolve()
                .relative_to(REPO_ROOT)
                .as_posix(),
                "sha256": replay_contract.file_sha256(source_lock_path),
            },
            "certification_parent_lock_v1": {
                "path": Path(parent_lock_path)
                .resolve()
                .relative_to(REPO_ROOT)
                .as_posix(),
                "sha256": replay_contract.file_sha256(parent_lock_path),
            },
        },
        "source_count": len(sources),
        "source_sha256": sources,
        "legacy_seed42_result_imported": False,
        "legacy_exact_journal_imported": False,
        "supplementary_seed_3407_primary_gate": False,
        "seed_426780603_scheduled": False,
        "official_test_accessed": False,
        "overwrite_forbidden": True,
    }


def verify_source_lock(
    path: Path = DEFAULT_OUTPUT,
    *,
    contract_path: Path = replay_contract.DEFAULT_CONTRACT,
    manifest_directory: Path = replay_contract.DEFAULT_MANIFEST_DIRECTORY,
    upstream_source_lock_path: Path = upstream_lock.DEFAULT_OUTPUT,
    parent_lock_path: Path = parent_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    source = _regular(path, "seed-42 replay source lock")
    raw = source.read_bytes()
    try:
        observed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot parse seed-42 replay source lock: {exc}")
    if not isinstance(observed, dict):
        _fail("seed-42 replay source lock must contain one object")
    if raw != canonical_json_bytes(observed):
        _fail("seed-42 replay source lock is not canonical JSON")
    expected = build_payload(
        contract_path=contract_path,
        manifest_directory=manifest_directory,
        source_lock_path=upstream_source_lock_path,
        parent_lock_path=parent_lock_path,
    )
    if observed != expected:
        _fail("seed-42 replay source lock differs from live sources")
    return observed


def write_once(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    content = canonical_json_bytes(dict(payload))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        _fail(f"refusing to write through symlink: {destination}")
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != content:
            raise FileExistsError(
                f"write-once source lock already differs: {destination}"
            )
        return destination
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
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-once", action="store_true")
    action.add_argument("--verify", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.write_once:
        path = write_once(args.output, build_payload())
        verify_source_lock(path)
        action = "write-once"
    else:
        path = args.output
        verify_source_lock(path)
        action = "verify"
    print(
        json.dumps(
            {
                "schema": ACTION_SCHEMA,
                "status": "complete",
                "action": action,
                "output": str(Path(path).resolve()),
                "output_sha256": replay_contract.file_sha256(path),
                "source_count": len(SOURCE_PATHS),
                "verified": True,
                "gpu_used": False,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
