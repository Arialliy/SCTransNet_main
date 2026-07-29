#!/usr/bin/env python3
"""Plan, write once, or verify the QFG-V2-CROA exact source lock.

Importing this module is read-only.  ``--plan`` computes the complete lock
without publishing it, ``--write-once`` atomically creates a previously absent
file, and ``--verify`` compares an existing lock with the live training
closure.  The freezer itself is not part of the numerical runtime source set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    train_tpd_ner_v4_qfg_v2_croa_exact as trainer,
)
from experiments.freeze_tpd_clean_v8_mprs_dch_source_locks import (  # noqa: E402
    training_data_contract,
)


FREEZER_PATH = Path(__file__).resolve()
DATASET = "NUDT-SIRST"
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets"
DEFAULT_OUTPUT = trainer.DEFAULT_EXACT_SOURCE_LOCK_PATH
LOCK_SCHEMA = trainer.EXACT_SOURCE_LOCK_SCHEMA
LOCK_KIND = "training"
PLAN_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_exact_source_lock_plan_v1"
)
ACTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_exact_source_lock_action_v1"
)
EXPECTED_CONSUMER_KEYS = frozenset(
    {
        trainer.SOURCE_LOCK_KEY,
        "training_data",
        "survival_target_statistics",
        "parent_checkpoint",
    }
)


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, "
            f"observed={observed!r}"
        )


def _regular_file(path: Path, label: str) -> Path:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise ValueError(
            f"{label} must be a regular non-symlink file: {value}"
        )
    return value


def _repo_relative_regular_file(
    path: Path,
    *,
    repo_root: Path,
    label: str,
) -> tuple[Path, str]:
    root = Path(repo_root).resolve()
    raw = Path(path)
    _regular_file(raw, label)
    resolved = raw.resolve()
    try:
        relative_path = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} lies outside the repository: {raw}") from exc
    canonical = root / relative_path
    if canonical != resolved:
        raise ValueError(f"{label} has a non-canonical repository path: {raw}")
    return resolved, relative_path.as_posix()


def _sha256_regular_file(path: Path, label: str) -> str:
    value = _regular_file(path, label)
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_source_sha256(
    *,
    repo_root: Path = REPO_ROOT,
    source_paths: Sequence[Path] | None = None,
) -> dict[str, str]:
    """Hash the complete, order-independent numerical runtime closure."""

    root = Path(repo_root).resolve()
    declared = (
        trainer.RUNTIME_SOURCE_PATHS
        if source_paths is None
        else tuple(Path(path) for path in source_paths)
    )
    if not declared:
        raise ValueError("QFG runtime source set is empty")
    records: dict[str, str] = {}
    freezer_relative = FREEZER_PATH.relative_to(REPO_ROOT).as_posix()
    for index, source in enumerate(declared):
        resolved, relative = _repo_relative_regular_file(
            Path(source),
            repo_root=root,
            label=f"runtime source[{index}]",
        )
        if relative == freezer_relative and root == REPO_ROOT.resolve():
            raise ValueError(
                "source-lock freezer is not a numerical training runtime source"
            )
        if relative in records:
            raise ValueError(f"duplicate QFG runtime source: {relative}")
        records[relative] = _sha256_regular_file(
            resolved,
            f"runtime source {relative}",
        )
    return dict(sorted(records.items()))


def survival_target_statistics_binding() -> dict[str, Any]:
    """Bind the validated target-survival statistics artifact."""

    resolved, relative = _repo_relative_regular_file(
        trainer.DEFAULT_TARGET_STATISTICS_PATH,
        repo_root=REPO_ROOT,
        label="QFG target-survival statistics",
    )
    digest = _sha256_regular_file(
        resolved,
        "QFG target-survival statistics",
    )
    statistics = trainer.load_survival_target_statistics(resolved)
    _require_equal(
        "QFG target-statistics SHA",
        statistics.get("sha256"),
        digest,
    )
    _require_equal(
        "QFG target-statistics schema",
        statistics.get("schema"),
        trainer.TARGET_STATISTICS_SCHEMA,
    )
    return {
        "path": relative,
        "sha256": digest,
        "schema": trainer.TARGET_STATISTICS_SCHEMA,
        "used_train_ids_sha256": statistics["used_train_ids_sha256"],
        "positive_cells": statistics["positive_cells"],
        "negative_cells": statistics["negative_cells"],
        "total_cells": statistics["total_cells"],
        "survival_pos_weight": statistics["survival_pos_weight"],
    }


def parent_checkpoint_binding() -> dict[str, Any]:
    """Bind the immutable V4 best-mIoU extension parent."""

    resolved, relative = _repo_relative_regular_file(
        trainer.PARENT_CHECKPOINT_PATH,
        repo_root=REPO_ROOT,
        label="QFG parent checkpoint",
    )
    digest = _sha256_regular_file(resolved, "QFG parent checkpoint")
    _require_equal(
        "QFG parent checkpoint SHA",
        digest,
        trainer.PARENT_CHECKPOINT_SHA256,
    )
    return {
        "path": relative,
        "sha256": digest,
        "state_dict_sha256": trainer.PARENT_STATE_DICT_SHA256,
        "epoch": trainer.PARENT_CHECKPOINT_EPOCH,
        "role": trainer.PARENT_CHECKPOINT_ROLE,
        "variant": trainer.PARENT_VARIANT,
    }


def build_source_lock_payload(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    repo_root: Path = REPO_ROOT,
    source_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Build the deterministic QFG lock payload without writing it."""

    root = Path(repo_root).resolve()
    if root != trainer.REPO_ROOT.resolve():
        raise ValueError(
            "QFG source lock must be built against trainer.REPO_ROOT"
        )
    variants = trainer.supported_candidate_variants()
    _require_equal(
        "QFG source-lock variant matrix",
        variants,
        trainer.SUPPORTED_CANDIDATE_VARIANTS,
    )
    data = training_data_contract(Path(dataset_dir), DATASET)
    sources = runtime_source_sha256(
        repo_root=root,
        source_paths=source_paths,
    )
    statistics = survival_target_statistics_binding()
    parent = parent_checkpoint_binding()
    formal = trainer.formal_contract()
    _require_equal(
        "formal target-statistics SHA",
        formal.get("survival_target_statistics_sha256"),
        statistics["sha256"],
    )
    _require_equal(
        "formal parent checkpoint SHA",
        formal.get("parent_checkpoint_sha256"),
        parent["sha256"],
    )
    return {
        "schema": LOCK_SCHEMA,
        "lock_kind": LOCK_KIND,
        "candidate_family": "v4_tail_aware_qfg_v2_croa_tss_pair",
        **data,
        "variants": list(variants),
        "qfg_variant": trainer.QFG_VARIANT,
        "tss_variants": dict(trainer.FORMAL_TSS_VARIANTS),
        "formal_contract": formal,
        "survival_target_statistics": statistics,
        "survival_target_statistics_sha256": statistics["sha256"],
        "parent_checkpoint": parent,
        "parent_checkpoint_sha256": parent["sha256"],
        "source_count": len(sources),
        "source_sha256": sources,
        "policy": {
            "official_test_accessed": False,
            "training_seed": trainer.TRAINING_SEED,
            "split_seed": trainer.SPLIT_SEED,
            "extension_parent_initialization": True,
            "fresh_weight_initialization": False,
            "same_variant_exact_resume_only": True,
            "paired_variants": list(variants),
            "physical_gpu_choices": [2, 3],
            "logical_device": "cuda:0",
            "write_once": True,
            "overwrite_forbidden": True,
            "runtime_sources_regular_non_symlink_files": True,
            "runtime_sources_repo_local": True,
            "freezer_executed_during_training": False,
            "freezer_in_runtime_source_set": False,
        },
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = _regular_file(path, "QFG source lock")
    try:
        payload = json.loads(value.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid QFG source-lock JSON: {value}") from exc
    if not isinstance(payload, dict):
        raise ValueError("QFG source lock must contain one JSON object")
    return payload


def _output_path(path: Path) -> Path:
    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    return Path(os.path.abspath(raw))


def _require_new_output(path: Path) -> Path:
    output = _output_path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite existing QFG source lock: {output}"
        )
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir():
        raise NotADirectoryError(
            "source-lock parent must be an existing regular directory: "
            f"{parent}"
        )
    return output


def _write_new_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    """Atomically publish one new file with a no-replace hard link."""

    output = _require_new_output(path)
    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.write-once.tmp"
    )
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary source-lock path exists: {temporary}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output, follow_symlinks=False)
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_descriptor = os.open(output.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return output


def verify_source_lock(
    path: Path = DEFAULT_OUTPUT,
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> dict[str, Any]:
    """Verify live equality and the trainer's exact four-key contract."""

    lock_path = _output_path(path)
    observed = _load_json_object(lock_path)
    expected = build_source_lock_payload(dataset_dir=dataset_dir)
    _require_equal("QFG source-lock payload", observed, expected)
    consumer = trainer.source_lock_contract(
        observed["training_data_sha256"],
        lock_path,
        trainer.DEFAULT_TARGET_STATISTICS_PATH,
    )
    _require_equal(
        "trainer source-lock key set",
        frozenset(consumer),
        EXPECTED_CONSUMER_KEYS,
    )
    expected_consumer = {
        trainer.SOURCE_LOCK_KEY: _sha256_regular_file(
            lock_path,
            "QFG source lock",
        ),
        "training_data": observed["training_data_sha256"],
        "survival_target_statistics": (
            observed["survival_target_statistics_sha256"]
        ),
        "parent_checkpoint": observed["parent_checkpoint_sha256"],
    }
    _require_equal(
        "trainer source-lock consumer contract",
        consumer,
        expected_consumer,
    )
    return observed


def plan_source_lock(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    destination = _require_new_output(output)
    payload = build_source_lock_payload(dataset_dir=dataset_dir)
    return {
        "schema": PLAN_SCHEMA,
        "status": "ready",
        "action": "plan",
        "output": str(destination),
        "output_exists": False,
        "would_write": False,
        "overwrite_forbidden": True,
        "payload_sha256": payload_sha256(payload),
        "training_data_sha256": payload["training_data_sha256"],
        "survival_target_statistics_sha256": (
            payload["survival_target_statistics_sha256"]
        ),
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "source_count": payload["source_count"],
    }


def write_source_lock_once(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    destination = _require_new_output(output)
    payload = build_source_lock_payload(dataset_dir=dataset_dir)
    _write_new_atomic(destination, payload)
    verified = verify_source_lock(destination, dataset_dir=dataset_dir)
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": "write-once",
        "output": str(destination),
        "output_sha256": _sha256_regular_file(
            destination,
            "written QFG source lock",
        ),
        "payload_sha256": payload_sha256(verified),
        "post_write_verified": True,
        "overwrite_forbidden": True,
        "training_data_sha256": verified["training_data_sha256"],
        "survival_target_statistics_sha256": (
            verified["survival_target_statistics_sha256"]
        ),
        "parent_checkpoint_sha256": verified["parent_checkpoint_sha256"],
        "source_count": verified["source_count"],
    }


def verify_source_lock_action(
    *,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    destination = _output_path(output)
    payload = verify_source_lock(destination, dataset_dir=dataset_dir)
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "action": "verify",
        "output": str(destination),
        "output_sha256": _sha256_regular_file(
            destination,
            "verified QFG source lock",
        ),
        "payload_sha256": payload_sha256(payload),
        "verified": True,
        "training_data_sha256": payload["training_data_sha256"],
        "survival_target_statistics_sha256": (
            payload["survival_target_statistics_sha256"]
        ),
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "source_count": payload["source_count"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, exclusively write, or verify the QFG-V2-CROA "
            "exact800 training source lock"
        )
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--write-once", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(list(sys.argv[1:] if argv is None else argv))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.plan:
        result = plan_source_lock(
            dataset_dir=args.dataset_dir,
            output=args.output,
        )
    elif args.write_once:
        result = write_source_lock_once(
            dataset_dir=args.dataset_dir,
            output=args.output,
        )
    else:
        result = verify_source_lock_action(
            dataset_dir=args.dataset_dir,
            output=args.output,
        )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


__all__ = [
    "ACTION_SCHEMA",
    "DATASET",
    "DEFAULT_DATASET_DIR",
    "DEFAULT_OUTPUT",
    "EXPECTED_CONSUMER_KEYS",
    "FREEZER_PATH",
    "LOCK_KIND",
    "LOCK_SCHEMA",
    "PLAN_SCHEMA",
    "build_source_lock_payload",
    "main",
    "parent_checkpoint_binding",
    "parse_args",
    "payload_sha256",
    "plan_source_lock",
    "runtime_source_sha256",
    "survival_target_statistics_binding",
    "verify_source_lock",
    "verify_source_lock_action",
    "write_source_lock_once",
]


if __name__ == "__main__":
    main()
