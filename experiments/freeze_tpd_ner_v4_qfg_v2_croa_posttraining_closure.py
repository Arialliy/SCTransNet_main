#!/usr/bin/env python3
"""Write or verify the independent QFG formal800 post-training source lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    tpd_ner_v4_qfg_v2_croa_posttraining_policy as policy,
)


def build_lock_payload() -> dict[str, Any]:
    source_sha256 = {
        relative: policy.sha256_file(REPO_ROOT / relative)
        for relative in policy.POSTTRAINING_SOURCE_PATHS
    }
    if list(source_sha256) != list(policy.POSTTRAINING_SOURCE_PATHS):
        raise RuntimeError("post-training source order changed")
    for path, expected in (
        (policy.TRAINING_LOCK_PATH, policy.TRAINING_LOCK_SHA256),
        (policy.FROZEN_AUTHORITY_PATH, policy.FROZEN_AUTHORITY_SHA256),
        (
            policy.FROZEN_AUTHORITY_MARKER_PATH,
            policy.FROZEN_AUTHORITY_MARKER_SHA256,
        ),
    ):
        if policy.sha256_file(path) != expected:
            raise ValueError(f"frozen prerequisite changed: {path}")
    payload = {
        "schema": policy.LOCK_SCHEMA,
        "status": "complete",
        "lock_kind": "post_training_closure",
        "candidate_family": "tpd_ner_v4_qfg_v2_croa_formal800",
        "dataset": "NUDT-SIRST",
        "training_seed": 42,
        "split_seed": 20260722,
        "official_test_accessed": False,
        "source_count": len(source_sha256),
        "source_sha256": source_sha256,
        "policy_summary": policy.policy_summary(),
        "policy_summary_sha256": policy.policy_summary_sha256(),
        "training_source_lock": {
            "path": str(policy.TRAINING_LOCK_PATH.relative_to(REPO_ROOT)),
            "sha256": policy.TRAINING_LOCK_SHA256,
        },
        "frozen_authority": {
            "path": str(policy.FROZEN_AUTHORITY_PATH.relative_to(REPO_ROOT)),
            "sha256": policy.FROZEN_AUTHORITY_SHA256,
            "marker_path": str(
                policy.FROZEN_AUTHORITY_MARKER_PATH.relative_to(REPO_ROOT)
            ),
            "marker_sha256": policy.FROZEN_AUTHORITY_MARKER_SHA256,
        },
        "policy": {
            "write_once": True,
            "overwrite_forbidden": True,
            "runtime_sources_regular_non_symlink_files": True,
            "runtime_sources_repo_local": True,
            "source_lock_self_excluded": True,
            "training_lock_unchanged": True,
            "terminal_reports_must_record_lock_sha256": True,
            "deployment_manifest_must_record_lock_sha256": True,
        },
    }
    return policy.canonical(payload)


def _atomic_create(path: Path, content: bytes) -> None:
    path = Path(path).resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace closure source lock: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise NotADirectoryError(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def write_once(path: Path) -> dict[str, Any]:
    output = Path(path).expanduser().resolve()
    content = policy.canonical_json_bytes(build_lock_payload())
    _atomic_create(output, content)
    _, binding = policy.load_closure_lock(output, verify_sources=True)
    return {
        "schema": policy.LOCK_ACTION_SCHEMA,
        "status": "complete",
        "action": "write_once",
        "output": str(output),
        "output_sha256": binding["sha256"],
        "source_count": binding["source_count"],
        "policy_summary_sha256": binding["policy_summary_sha256"],
        "verified": True,
    }


def verify(path: Path) -> dict[str, Any]:
    output = Path(path).expanduser().resolve()
    raw = policy.regular_file(output, "post-training closure source lock").read_bytes()
    expected = policy.canonical_json_bytes(build_lock_payload())
    if raw != expected:
        raise ValueError("post-training closure source lock differs from live closure")
    _, binding = policy.load_closure_lock(output, verify_sources=True)
    return {
        "schema": policy.LOCK_ACTION_SCHEMA,
        "status": "complete",
        "action": "verify",
        "output": str(output),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "source_count": binding["source_count"],
        "policy_summary_sha256": binding["policy_summary_sha256"],
        "verified": True,
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-once", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, default=policy.DEFAULT_LOCK_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _argument_parser().parse_args(argv)
    result = write_once(args.output) if args.write_once else verify(args.output)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


__all__ = ["build_lock_payload", "main", "verify", "write_once"]


if __name__ == "__main__":
    main()
