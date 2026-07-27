#!/usr/bin/env python3
"""Create the independent V6 checkpoint-metric compatibility lock once."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_tpd_clean_v6_pd_fa_checkpoint_compat as compat  # noqa: E402
from experiments import summarize_tpd_clean_v6_formal800 as summary  # noqa: E402


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError as exc:
        raise ValueError(f"path lies outside repository: {path}") from exc


def _hash_sources(relatives: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in relatives:
        if relative in hashes:
            raise ValueError(f"duplicate compatibility source: {relative}")
        path = REPO_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"compatibility source is not a regular file: {path}"
            )
        hashes[relative] = compat.sha256_file(path)
    if not hashes:
        raise ValueError("compatibility source set is empty")
    return hashes


def build_source_lock_payload() -> dict[str, Any]:
    frozen_paths: dict[str, str] = {}
    frozen_hashes: dict[str, str] = {}
    for name, (path, expected_sha) in compat.EXPECTED_LOCK_BINDINGS.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"frozen lock is not a regular file: {path}")
        actual_sha = compat.sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"frozen lock digest differs: {name}")
        frozen_paths[name] = _relative(path)
        frozen_hashes[name] = actual_sha
    evaluator_hashes = {
        _relative(compat.FROZEN_EVALUATOR): compat.sha256_file(
            compat.FROZEN_EVALUATOR
        ),
        _relative(compat.GENERIC_BASE_EVALUATOR): compat.sha256_file(
            compat.GENERIC_BASE_EVALUATOR
        ),
    }
    training_lock = json.loads(
        compat.EXPECTED_LOCK_BINDINGS["training_source_lock"][0].read_text(
            encoding="utf-8"
        )
    )
    for relative, digest in evaluator_hashes.items():
        if training_lock.get("source_sha256", {}).get(relative) != digest:
            raise ValueError(
                f"training lock does not bind base evaluator: {relative}"
            )
    sources = _hash_sources(sorted(compat.COMPATIBILITY_SOURCE_RELATIVES))
    return {
        "schema": compat.SOURCE_LOCK_SCHEMA,
        "scope": (
            "Post-freeze in-memory checkpoint-metric audit compatibility, "
            "eight-sweep runner, read-only validation, and final acceptance"
        ),
        "candidate_root": _relative(summary.DEFAULT_CANDIDATE_ROOT),
        "frozen_lock_paths": frozen_paths,
        "frozen_lock_sha256": frozen_hashes,
        "base_evaluator_sha256": evaluator_hashes,
        "source_count": len(sources),
        "source_sha256": sources,
        "policy": {
            "audit_supplement_is_in_memory_only": True,
            "checkpoint_rewrite_forbidden": True,
            "metrics_rewrite_forbidden": True,
            "sweep_overwrite_forbidden": True,
            "original_checkpoint_validation_metrics_preserved": True,
            "sweep_task_metric_points_preserved": True,
            "sweep_val_loss_normalized_to_checkpoint": True,
            "raw_fixed_audit_preserved_before_normalization": True,
            "formal_inference_replays_training_environment": True,
            "base_evaluator_artifact_digest_preserved": True,
            "old_acceptance_runs_before_compatibility_acceptance": True,
            "direct_wrapper_requires_frozen_arguments": True,
            "formal_runner_requires_cuda": True,
            "shared_postprocess_lock_required": True,
            "non_strict_numeric_delta_limits": (
                compat.NON_STRICT_NUMERIC_DELTA_LIMITS
            ),
        },
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
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


def write_new_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise NotADirectoryError(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    return path


def freeze_source_lock(
    output: Path = compat.DEFAULT_COMPATIBILITY_SOURCE_LOCK,
) -> tuple[Path, dict[str, Any]]:
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    payload = build_source_lock_payload()
    return write_new_json(output, payload), payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the V6 checkpoint-metric compatibility sources"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=compat.DEFAULT_COMPATIBILITY_SOURCE_LOCK,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output, payload = freeze_source_lock(args.output)
    print(
        f"WROTE {output} sources={payload['source_count']} "
        f"sha256={compat.sha256_file(output)}",
        flush=True,
    )


__all__ = [
    "build_source_lock_payload",
    "freeze_source_lock",
    "main",
    "parse_args",
    "write_new_json",
]


if __name__ == "__main__":
    main()
