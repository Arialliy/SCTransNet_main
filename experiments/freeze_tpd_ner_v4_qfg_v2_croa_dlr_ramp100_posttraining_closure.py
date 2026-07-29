#!/usr/bin/env python3
"""Create or verify the independent paired-DLR post-training source lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (
    tpd_ner_v4_qfg_v2_croa_dlr_ramp100_posttraining_policy as policy,
)


def build_lock_payload() -> dict[str, Any]:
    source_sha256 = {
        relative: policy.sha256_file(policy.REPO_ROOT / relative)
        for relative in policy.POSTTRAINING_SOURCE_PATHS
    }
    if len(source_sha256) != len(policy.POSTTRAINING_SOURCE_PATHS):
        raise ValueError("DLR closure source list contains duplicates")
    if (
        policy.sha256_file(policy.TRAINING_LOCK_PATH)
        != policy.TRAINING_LOCK_SHA256
    ):
        raise ValueError("DLR training source lock differs")
    if (
        policy.sha256_file(policy.REFERENCE_CLOSURE_LOCK_PATH)
        != policy.REFERENCE_CLOSURE_LOCK_SHA256
    ):
        raise ValueError("reference closure source lock differs")
    if (
        policy.sha256_file(policy.LEGACY_V1_LOCK_PATH)
        != policy.LEGACY_V1_LOCK_SHA256
    ):
        raise ValueError("superseded v1 DLR closure lock differs")
    policy.verify_final_evaluation_sources()
    return {
        "schema": policy.LOCK_SCHEMA,
        "status": "complete",
        "lock_kind": "post_training_closure",
        "candidate_family": (
            "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800"
        ),
        "source_count": len(source_sha256),
        "source_sha256": source_sha256,
        "training_source_lock": {
            "path": str(policy.TRAINING_LOCK_PATH.resolve()),
            "sha256": policy.TRAINING_LOCK_SHA256,
        },
        "reference_closure_source_lock": {
            "path": str(policy.REFERENCE_CLOSURE_LOCK_PATH.resolve()),
            "sha256": policy.REFERENCE_CLOSURE_LOCK_SHA256,
        },
        "supersession": policy.supersession_summary(),
        "final_evaluation_source_sha256": dict(
            policy.FINAL_EVALUATION_SOURCE_SHA256
        ),
        "policy_summary": policy.policy_summary(),
        "policy_summary_sha256": policy.policy_summary_sha256(),
        "sweep_public_interface": policy.interface_summary(),
        "write_once": True,
        "overwrite_forbidden": True,
        "idempotent_existing_identical_is_success": True,
        "official_test_accessed": False,
    }


def _atomic_create(path: Path, content: bytes) -> bool:
    output = Path(path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        if (
            output.is_file()
            and not output.is_symlink()
            and output.read_bytes() == content
        ):
            return False
        raise FileExistsError(f"refusing to replace DLR closure lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise NotADirectoryError(output.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError:
            if (
                output.is_file()
                and not output.is_symlink()
                and output.read_bytes() == content
            ):
                return False
            raise
        directory_descriptor = os.open(str(output.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def verify(path: Path) -> dict[str, Any]:
    payload, binding = policy.load_closure_lock(path, verify_sources=True)
    expected = build_lock_payload()
    if policy.canonical(payload) != policy.canonical(expected):
        raise ValueError("DLR closure lock payload differs from live sources")
    return {
        "schema": policy.LOCK_ACTION_SCHEMA,
        "status": "complete",
        "action": "verify",
        "output": binding["path"],
        "output_sha256": binding["sha256"],
        "source_count": binding["source_count"],
        "verified": True,
        "writes_performed": False,
    }


def write_once(path: Path) -> dict[str, Any]:
    output = policy.require_v2_lock_target(path)
    payload = build_lock_payload()
    content = policy.canonical_json_bytes(payload)
    written = _atomic_create(output, content)
    result = verify(output)
    result["action"] = "write_once" if written else "verify"
    result["writes_performed"] = written
    result["idempotent_resume"] = True
    return result


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
