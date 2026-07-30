#!/usr/bin/env python3
"""Validate and summarize each arm's own selected engineering checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import final_model_replication_exact_core as core  # noqa: E402
from experiments import final_model_replication_seed_contract as seeds  # noqa: E402
from experiments import prepare_final_model_engineering_replication as prepare  # noqa: E402
from experiments import watch_final_model_engineering_replication as watcher  # noqa: E402


SCHEMA = "sctransnet_final_model_engineering_replication_summary_v1"
DEFAULT_OUTPUT = (
    watcher.DEFAULT_OUTPUT_ROOT / "engineering_replication_summary_v1.json"
)
CHECKPOINTS = (
    ("best_miou.pth.tar", "primary_best_miou"),
    ("best.pth.tar", "secondary_best_pd"),
)
METRICS = (
    "pd",
    "fa",
    "miou",
    "tiny_pd",
    "false_objects_per_image",
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)


class EngineeringSummaryError(ValueError):
    """A completed engineering run or checkpoint is inconsistent."""


def _fail(message: str) -> None:
    raise EngineeringSummaryError(message)


def canonical_json_bytes(value: Any) -> bytes:
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


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        _fail(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_projection(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        _fail("validation metrics must be a mapping")
    missing = [name for name in METRICS if name not in metrics]
    if missing:
        _fail(f"validation metrics omit fields: {missing}")
    return {name: metrics[name] for name in METRICS}


def metric_delta(
    d_metrics: Mapping[str, Any],
    b_metrics: Mapping[str, Any],
) -> dict[str, float]:
    projected_d = metric_projection(d_metrics)
    projected_b = metric_projection(b_metrics)
    return {
        name: float(projected_d[name]) - float(projected_b[name])
        for name in (
            "pd",
            "fa",
            "miou",
            "tiny_pd",
            "false_objects_per_image",
        )
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"cannot parse {label}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} is not an object")
    return value


def _checkpoint_record(
    inputs: core.ReplicationInputs,
    run_dir: Path,
    checkpoint_name: str,
    selection_role: str,
) -> dict[str, Any]:
    checkpoint_path = run_dir / checkpoint_name
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    trainer = inputs.definition.trainer
    if inputs.definition.arm == core.ARM_B:
        validated = trainer.require_evaluator_checkpoint_payload(
            payload,
            expected_variant=inputs.definition.variant,
        )
    else:
        validated = trainer.require_evaluator_checkpoint_payload(
            payload,
            expected_variant=inputs.definition.variant,
        )
    if _sha256_file(checkpoint_path) != checkpoint_sha256:
        _fail("checkpoint changed during summary validation")
    expected_role = {
        "best_miou.pth.tar": "best_validation_miou_secondary",
        "best.pth.tar": "best_validation_pd_primary",
    }[checkpoint_name]
    if validated.get("checkpoint_role") != expected_role:
        _fail(f"{checkpoint_name} role mismatch")
    return {
        "selection_role": selection_role,
        "filename": checkpoint_name,
        "path": str(checkpoint_path.resolve()),
        "sha256": checkpoint_sha256,
        "epoch": validated["epoch"],
        "checkpoint_role": validated["checkpoint_role"],
        "metrics": metric_projection(validated["validation_metrics"]),
    }


def summarize_run(
    *,
    arm: str,
    trajectory_seed: int,
    output_root: Path,
    source_lock_path: Path,
    seed_contract_path: Path,
    manifest_directory: Path,
) -> dict[str, Any]:
    initialization_path = prepare.manifest_path(
        manifest_directory,
        seed=trajectory_seed,
        arm=arm,
    )
    inputs = core.validate_replication_inputs(
        arm=arm,
        trajectory_seed=trajectory_seed,
        schedule_path=seed_contract_path,
        initialization_manifest_path=initialization_path,
        certification_source_lock_path=source_lock_path,
    )
    run_dir = watcher.run_directory(output_root, trajectory_seed, arm)
    summary = _load_json(run_dir / "summary.json", "run summary")
    if summary.get("status") != "complete":
        _fail(f"run is not complete: {run_dir}")
    with core.replication_trainer_overlay(inputs):
        checkpoints = [
            _checkpoint_record(
                inputs,
                run_dir,
                checkpoint_name,
                selection_role,
            )
            for checkpoint_name, selection_role in CHECKPOINTS
        ]
    return {
        "arm": arm,
        "variant": inputs.definition.variant,
        "trajectory_seed": trajectory_seed,
        "run_directory": str(run_dir.resolve()),
        "summary_sha256": _sha256_file(run_dir / "summary.json"),
        "seed_contract_sha256": inputs.schedule_sha256,
        "source_lock_sha256": inputs.source_lock_sha256,
        "child_initialization_manifest_sha256": (
            inputs.initialization_sha256
        ),
        "checkpoints": checkpoints,
    }


def build_summary(
    *,
    output_root: Path = watcher.DEFAULT_OUTPUT_ROOT,
    source_lock_path: Path,
    seed_contract_path: Path = prepare.DEFAULT_SEED_CONTRACT,
    manifest_directory: Path = prepare.DEFAULT_MANIFEST_DIRECTORY,
) -> dict[str, Any]:
    runs = [
        summarize_run(
            arm=arm,
            trajectory_seed=trajectory_seed,
            output_root=output_root,
            source_lock_path=source_lock_path,
            seed_contract_path=seed_contract_path,
            manifest_directory=manifest_directory,
        )
        for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS
        for arm in core.SUPPORTED_ARMS
    ]
    by_identity = {
        (run["trajectory_seed"], run["arm"]): run for run in runs
    }
    comparisons = []
    for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS:
        b_run = by_identity[(trajectory_seed, core.ARM_B)]
        d_run = by_identity[(trajectory_seed, core.ARM_D)]
        b_primary = b_run["checkpoints"][0]["metrics"]
        d_primary = d_run["checkpoints"][0]["metrics"]
        comparisons.append(
            {
                "trajectory_seed": trajectory_seed,
                "checkpoint_policy": (
                    "each_arm_own_best_miou_primary"
                ),
                "d_minus_b": metric_delta(d_primary, b_primary),
                "b_metrics": b_primary,
                "d_metrics": d_primary,
            }
        )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "scope": "fixed_parent_engineering_b_d_only",
        "run_count": len(runs),
        "checkpoint_count": len(runs) * len(CHECKPOINTS),
        "checkpoint_selection": {
            "primary": "each_arm_own_best_miou",
            "secondary": "each_arm_own_best_pd",
            "cross_arm_shared_checkpoint_epoch_required": False,
        },
        "fixed_threshold": 0.5,
        "pd_fa_sweep_complete": False,
        "official_test_accessed": False,
        "runs": runs,
        "paired_best_miou_comparisons": comparisons,
        "claim_boundary": {
            "engineering_replication_complete": True,
            "paper_stability_supported": False,
            "full_pipeline_stability_supported": False,
        },
    }


def write_once(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    content = canonical_json_bytes(dict(payload))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        _fail(f"refusing to write through symlink: {destination}")
    if destination.exists():
        if destination.read_bytes() != content:
            raise FileExistsError(f"summary already differs: {destination}")
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
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=watcher.DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--seed-contract",
        type=Path,
        default=prepare.DEFAULT_SEED_CONTRACT,
    )
    parser.add_argument(
        "--manifest-directory",
        type=Path,
        default=prepare.DEFAULT_MANIFEST_DIRECTORY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = build_summary(
        output_root=args.output_root,
        source_lock_path=args.source_lock,
        seed_contract_path=args.seed_contract,
        manifest_directory=args.manifest_directory,
    )
    destination = write_once(args.output, payload)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(destination.resolve()),
                "sha256": _sha256_file(destination),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
