#!/usr/bin/env python3
"""Freeze the four-source QFG V2 operational closure.

The source lock is deliberately independent from the frozen training and
post-training closures.  It binds the runtime controller, default operating
point publisher, and both reproducibility-manifest generators.  The fallback
receipt is bound only by its schema and repository-relative contract path; a
particular receipt instance is not source material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

LOCK_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_"
    "operational_closure_source_lock_v2"
)
ACTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_"
    "operational_closure_action_v2"
)

SOURCE_PATHS = (
    (
        "experiments/"
        "control_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_fallback.py"
    ),
    (
        "experiments/"
        "publish_tpd_ner_v4_qfg_v2_croa_default_operating_point_v2.py"
    ),
    (
        "experiments/"
        "generate_tpd_ner_v4_qfg_v2_croa_reproducibility_manifest.py"
    ),
    (
        "experiments/"
        "generate_tpd_ner_v4_qfg_v2_croa_reproducibility_manifest_v2.py"
    ),
)

UPSTREAM_LOCK_RELATIVE_PATH = (
    "experiments/"
    "tpd_ner_v4_qfg_v2_croa_posttraining_closure_source_lock.json"
)
UPSTREAM_LOCK_SHA256 = (
    "315f091b75078e65b871946cecae92893e8915bb3951b6fc4dcf3a52c984cbbd"
)

RECEIPT_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "fallback_receipt_v1"
)
RECEIPT_RELATIVE_PATH = (
    "experiments/results/"
    "tpd_ner_v4_qfg_v2_croa_exact_v2_optimized/"
    "NUDT-SIRST/fallback_control/"
    "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_fallback_receipt.json"
)

DEFAULT_OUTPUT_RELATIVE_PATH = (
    "experiments/"
    "tpd_ner_v4_qfg_v2_croa_operational_closure_source_lock_v2.json"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / DEFAULT_OUTPUT_RELATIVE_PATH


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _regular_file(path: Path, label: str) -> Path:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {value}")
    return value


def sha256_file(path: Path) -> str:
    value = _regular_file(path, "SHA-256 input")
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=str,
        )
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            canonical(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _resolved_repo(repo_root: Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def _resolved_output(repo_root: Path, output: Path | None) -> Path:
    if output is None:
        return repo_root / DEFAULT_OUTPUT_RELATIVE_PATH
    return Path(output).expanduser().resolve()


def _source_hashes(repo_root: Path) -> dict[str, str]:
    _require(
        len(set(SOURCE_PATHS)) == len(SOURCE_PATHS),
        "operational source list contains duplicates",
    )
    return {
        relative: sha256_file(
            repo_root / relative,
        )
        for relative in SOURCE_PATHS
    }


def _verify_upstream_lock(repo_root: Path) -> Path:
    path = repo_root / UPSTREAM_LOCK_RELATIVE_PATH
    actual = sha256_file(path)
    _require(
        actual == UPSTREAM_LOCK_SHA256,
        "upstream 15-source post-training closure lock differs",
    )
    return path


def build_lock_payload(
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    root = _resolved_repo(repo_root)
    source_sha256 = _source_hashes(root)
    _verify_upstream_lock(root)
    return canonical(
        {
            "schema": LOCK_SCHEMA,
            "status": "complete",
            "lock_kind": "operational_closure",
            "candidate_family": "tpd_ner_v4_qfg_v2_croa",
            "source_count": len(source_sha256),
            "source_sha256": source_sha256,
            "upstream_posttraining_closure_source_lock": {
                "path": UPSTREAM_LOCK_RELATIVE_PATH,
                "sha256": UPSTREAM_LOCK_SHA256,
                "source_count": 15,
            },
            "receipt_contract": {
                "schema": RECEIPT_SCHEMA,
                "path": RECEIPT_RELATIVE_PATH,
                "binding_scope": "schema_and_contract_path_only",
                "receipt_content_in_source_lock": False,
                "receipt_may_be_absent_at_freeze_time": True,
            },
            "policy": {
                "canonical_pretty_json": True,
                "write_once": True,
                "overwrite_forbidden": True,
                "idempotent_existing_identical_is_success": True,
                "runtime_sources_regular_non_symlink_files": True,
                "runtime_sources_repo_local": True,
                "source_lock_self_excluded": True,
                "upstream_15_source_lock_unchanged": True,
                "official_test_accessed": False,
            },
            "official_test_accessed": False,
        }
    )


def _path_state(path: Path) -> dict[str, Any]:
    value = Path(path)
    if value.is_symlink():
        return {"state": "invalid_symlink", "path": str(value)}
    if not value.exists():
        return {"state": "missing", "path": str(value)}
    if not value.is_file():
        return {"state": "invalid_not_file", "path": str(value)}
    return {
        "state": "regular_file",
        "path": str(value),
        "sha256": sha256_file(value),
    }


def preflight(
    repo_root: Path = REPO_ROOT,
    output: Path | None = None,
) -> dict[str, Any]:
    root = _resolved_repo(repo_root)
    target = _resolved_output(root, output)
    sources = {
        relative: _path_state(root / relative)
        for relative in SOURCE_PATHS
    }
    upstream = _path_state(root / UPSTREAM_LOCK_RELATIVE_PATH)
    missing_sources = [
        relative
        for relative, state in sources.items()
        if state["state"] == "missing"
    ]
    invalid_sources = [
        relative
        for relative, state in sources.items()
        if state["state"] not in {"missing", "regular_file"}
    ]
    upstream_ready = (
        upstream.get("state") == "regular_file"
        and upstream.get("sha256") == UPSTREAM_LOCK_SHA256
    )
    inputs_ready = (
        not missing_sources
        and not invalid_sources
        and upstream_ready
    )

    output_state = _path_state(target)
    if inputs_ready:
        expected = canonical_json_bytes(build_lock_payload(root))
        if output_state["state"] == "regular_file":
            output_state["content"] = (
                "identical"
                if target.read_bytes() == expected
                else "conflict"
            )
        elif output_state["state"] == "missing":
            output_state["content"] = "absent"
    else:
        output_state["content"] = "not_evaluated"

    if missing_sources or upstream.get("state") == "missing":
        status = "pending"
    elif (
        invalid_sources
        or not upstream_ready
        or output_state["state"] not in {"missing", "regular_file"}
        or output_state.get("content") == "conflict"
    ):
        status = "blocked"
    else:
        status = "ready"

    return canonical(
        {
            "schema": ACTION_SCHEMA,
            "status": status,
            "action": "preflight",
            "repo_root": str(root),
            "output": str(target),
            "source_count": len(SOURCE_PATHS),
            "sources": sources,
            "missing_sources": missing_sources,
            "invalid_sources": invalid_sources,
            "upstream_posttraining_closure_source_lock": {
                **upstream,
                "expected_sha256": UPSTREAM_LOCK_SHA256,
                "verified": upstream_ready,
            },
            "receipt_contract": {
                "schema": RECEIPT_SCHEMA,
                "path": RECEIPT_RELATIVE_PATH,
                "content_required": False,
            },
            "output_state": output_state,
            "publish_ready": (
                status == "ready"
                and output_state.get("content") == "absent"
            ),
            "verify_ready": (
                status == "ready"
                and output_state.get("content") == "identical"
            ),
            "writes_performed": False,
        }
    )


def _atomic_create(path: Path, content: bytes) -> bool:
    output = Path(path).expanduser().resolve()
    if output.exists() or output.is_symlink():
        if (
            output.is_file()
            and not output.is_symlink()
            and output.read_bytes() == content
        ):
            return False
        raise FileExistsError(
            f"refusing to replace operational closure source lock: {output}"
        )
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


def verify(
    repo_root: Path = REPO_ROOT,
    output: Path | None = None,
) -> dict[str, Any]:
    root = _resolved_repo(repo_root)
    target = _resolved_output(root, output)
    raw = _regular_file(
        target,
        "operational closure source lock",
    ).read_bytes()
    expected = canonical_json_bytes(build_lock_payload(root))
    _require(
        raw == expected,
        "operational closure source lock differs from live sources",
    )
    return canonical(
        {
            "schema": ACTION_SCHEMA,
            "status": "complete",
            "action": "verify",
            "output": str(target),
            "output_sha256": hashlib.sha256(raw).hexdigest(),
            "source_count": len(SOURCE_PATHS),
            "upstream_posttraining_closure_source_lock_sha256": (
                UPSTREAM_LOCK_SHA256
            ),
            "receipt_schema": RECEIPT_SCHEMA,
            "receipt_path": RECEIPT_RELATIVE_PATH,
            "verified": True,
            "writes_performed": False,
        }
    )


def publish(
    repo_root: Path = REPO_ROOT,
    output: Path | None = None,
) -> dict[str, Any]:
    root = _resolved_repo(repo_root)
    target = _resolved_output(root, output)
    content = canonical_json_bytes(build_lock_payload(root))
    written = _atomic_create(target, content)
    result = verify(root, target)
    result["action"] = "publish" if written else "verify"
    result["writes_performed"] = written
    result["idempotent_resume"] = True
    return canonical(result)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--publish", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.preflight:
        result = preflight(args.repo_root, args.output)
    elif args.publish:
        result = publish(args.repo_root, args.output)
    else:
        result = verify(args.repo_root, args.output)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


__all__ = [
    "ACTION_SCHEMA",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_OUTPUT_RELATIVE_PATH",
    "LOCK_SCHEMA",
    "RECEIPT_RELATIVE_PATH",
    "RECEIPT_SCHEMA",
    "SOURCE_PATHS",
    "UPSTREAM_LOCK_RELATIVE_PATH",
    "UPSTREAM_LOCK_SHA256",
    "build_lock_payload",
    "canonical",
    "canonical_json_bytes",
    "main",
    "preflight",
    "publish",
    "sha256_file",
    "verify",
]


if __name__ == "__main__":
    raise SystemExit(main())
