#!/usr/bin/env python3
"""Independent post-training closure for the single-seed V3 NER candidate.

Only the two V3 relay-on checkpoint roles may be evaluated here.  The
SCTransNet baseline, V1 relay-off required control, and V2 relay-on structural
predecessor are immutable upstream sweeps: they are rebound and revalidated,
never regenerated or repaired by this module.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_v3_pd_fa as v3_eval,
)
from experiments import (  # noqa: E402
    freeze_tpd_ner_v8_mprs_dch_v3_source_locks as v3_freeze,
)
from experiments import (  # noqa: E402
    postprocess_tpd_ner_v8_mprs_dch_formal800 as v1_post,
)
from experiments import (  # noqa: E402
    postprocess_tpd_ner_v8_mprs_dch_v2_formal800 as v2_post,
)
from experiments import train_tpd_ner_v8_mprs_dch_v3_exact as v3_exact  # noqa: E402


SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_v3_posttraining_aggregate_v1"
READINESS_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_posttraining_readiness_v1"
)
COMPLETE_MARKER_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_postprocess_complete_v1"
)
REJECTED_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_rejected_postprocess_v1"
)
DATASET = v3_eval.DATASET
TRAINING_SEED = 42
SPLIT_SEED = 20260722
EXPECTED_EPOCHS = 800
TARGET_COUNT = 189
TINY_TARGET_COUNT = 39
CHECKPOINTS = ("best.pth.tar", "best_miou.pth.tar")
CHECKPOINT_ROLES = dict(v3_eval.CHECKPOINT_ROLES)
ROLE_NAMES = {
    "best_validation_pd_primary": "pd_primary",
    "best_validation_miou_secondary": "miou_secondary",
}
FA_BUDGETS = tuple(v3_eval.FA_BUDGETS)
BUDGET_KEYS = tuple(v3_eval.BUDGET_KEYS)
VARIANT_V3_ON = v3_eval.VARIANT
VARIANT_V2_ON = v3_eval.V2_PREDECESSOR
VARIANT_V1_OFF = v3_eval.V1_CONTROL
BASELINE_VARIANT = "baseline_sctransnet"
V3_RESULT_ROOT = v3_exact.DEFAULT_OUTPUT_ROOT
V3_RUN_DIR = (
    V3_RESULT_ROOT
    / DATASET
    / VARIANT_V3_ON
    / f"seed_{TRAINING_SEED}_{v3_exact.FORMAL_RUN_TAG}"
)
V2_RUN_DIR = v2_post.V2_RUN_DIR
V1_OFF_RUN_DIR = v2_post.V1_OFF_RUN_DIR
BASELINE_RUN_DIR = v2_post.BASELINE_RUN_DIR
V3_EVALUATOR = (
    REPO_ROOT / "experiments/evaluate_tpd_ner_v8_mprs_dch_v3_pd_fa.py"
)
TRAINING_LOCK = v3_eval.DEFAULT_TRAINING_LOCK
ACCEPTANCE_LOCK = v3_eval.DEFAULT_ACCEPTANCE_LOCK
COMPARISON_DIR = V3_RESULT_ROOT / DATASET / "comparison"
JSON_OUTPUT = (
    COMPARISON_DIR / "tpd_ner_v8_mprs_dch_v3_formal800_comparison.json"
)
MARKDOWN_OUTPUT = (
    COMPARISON_DIR / "tpd_ner_v8_mprs_dch_v3_formal800_comparison.md"
)
COMPLETE_MARKER = COMPARISON_DIR / "POSTPROCESS_COMPLETE.json"
GPU_UUIDS = dict(v3_exact.PHYSICAL_GPU_UUIDS)
COMPLETE_FIXED_FIELDS = (
    "matched_target_count",
    "target_count",
    "pd",
    "fa",
    "miou",
    "false_objects_per_image",
    "threshold",
    "tiny_pd",
    "matched_tiny_target_count",
    "tiny_target_count",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
)


class IncompleteTraining(RuntimeError):
    """Raised when V3 or an immutable upstream trajectory is incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _finite_number(location: str, value: Any) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{location} must be finite")
    return float(value)


def _normalize_complete_fixed(
    point: Any,
    *,
    location: str,
) -> dict[str, Any]:
    """Normalize the complete fixed-0.5 metric view required by V3 reports."""

    _require(isinstance(point, Mapping), f"{location} is missing")
    missing = [name for name in COMPLETE_FIXED_FIELDS if name not in point]
    _require(not missing, f"{location} lacks fixed metrics: {missing}")
    counts: dict[str, int] = {}
    for name, expected in (
        ("target_count", TARGET_COUNT),
        ("tiny_target_count", TINY_TARGET_COUNT),
    ):
        value = point.get(name)
        _require(type(value) is int, f"{location}.{name} is invalid")
        _require(value == expected, f"{location}.{name} differs")
        counts[name] = value
    for name, total_name in (
        ("matched_target_count", "target_count"),
        ("matched_tiny_target_count", "tiny_target_count"),
    ):
        value = point.get(name)
        _require(type(value) is int, f"{location}.{name} is invalid")
        _require(
            0 <= value <= counts[total_name],
            f"{location}.{name} is invalid",
        )
        counts[name] = value
    finite = {
        name: _finite_number(f"{location}.{name}", point.get(name))
        for name in (
            "pd",
            "fa",
            "miou",
            "false_objects_per_image",
            "threshold",
            "tiny_pd",
            "niou",
            "pixel_precision",
            "pixel_recall",
            "pixel_f1",
        )
    }
    _require(
        math.isclose(
            finite["pd"],
            counts["matched_target_count"] / counts["target_count"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        f"{location}.pd differs from counts",
    )
    _require(
        math.isclose(
            finite["tiny_pd"],
            (
                counts["matched_tiny_target_count"]
                / counts["tiny_target_count"]
            ),
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        f"{location}.tiny_pd differs from counts",
    )
    _require(
        math.isclose(
            finite["threshold"],
            0.5,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        f"{location}.threshold differs",
    )
    for name in (
        "pd",
        "fa",
        "miou",
        "threshold",
        "tiny_pd",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
    ):
        _require(
            0.0 <= finite[name] <= 1.0,
            f"{location}.{name} lies outside [0, 1]",
        )
    _require(
        finite["false_objects_per_image"] >= 0.0,
        f"{location}.false_objects_per_image is negative",
    )
    return {
        "matched_target_count": counts["matched_target_count"],
        "target_count": counts["target_count"],
        **finite,
        "matched_tiny_target_count": counts[
            "matched_tiny_target_count"
        ],
        "tiny_target_count": counts["tiny_target_count"],
    }


def _tiny_pd_audit(fixed: Mapping[str, Any]) -> dict[str, Any]:
    matched = fixed.get("matched_tiny_target_count")
    total = fixed.get("tiny_target_count")
    _require(type(matched) is int, "V3 matched tiny-target count is invalid")
    _require(total == TINY_TARGET_COUNT, "V3 tiny-target count differs")
    regressed = matched < TINY_TARGET_COUNT
    return {
        "reference": "perfect_tiny_target_recall_39_of_39",
        "required_matched_tiny_target_count": TINY_TARGET_COUNT,
        "required_tiny_target_count": TINY_TARGET_COUNT,
        "observed_matched_tiny_target_count": matched,
        "observed_tiny_target_count": total,
        "observed_tiny_pd": fixed["tiny_pd"],
        "tiny_pd_regressed": regressed,
        "independent_pass_gate": False,
        "report_only": True,
    }


def _metrics_progress(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        return {
            "exists": False,
            "event_count": 0,
            "last_epoch": 0,
            "contiguous_from_one": False,
        }
    epochs: list[Any] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        epochs.append(value.get("epoch"))
    return {
        "exists": True,
        "event_count": len(epochs),
        "last_epoch": epochs[-1] if epochs else 0,
        "contiguous_from_one": epochs == list(range(1, len(epochs) + 1)),
    }


def inspect_v3_progress(run_dir: Path = V3_RUN_DIR) -> dict[str, Any]:
    directory = Path(run_dir)
    metrics = _metrics_progress(directory / "metrics.jsonl")
    summary_path = directory / "summary.json"
    summary: dict[str, Any] | None = None
    if summary_path.is_file() and not summary_path.is_symlink():
        summary = load_json(summary_path)
    required = (
        "protocol.json",
        "split.json",
        "metrics.jsonl",
        "summary.json",
        "best.pth.tar",
        "best_miou.pth.tar",
        "last.pth.tar",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if summary is not None:
        _require(
            summary.get("schema") == v3_exact.COMPLETION_SUMMARY_SCHEMA,
            "V3 completion summary schema differs",
        )
        for name, expected in {
            "variant": VARIANT_V3_ON,
            "seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "structural_predecessor": VARIANT_V2_ON,
            "required_control": VARIANT_V1_OFF,
            "relay_off_retrained": False,
        }.items():
            _require(
                summary.get(name) == expected,
                f"V3 completion summary differs: {name}",
            )
    complete = bool(
        not missing
        and summary is not None
        and summary.get("status") == "complete"
        and metrics["event_count"] == EXPECTED_EPOCHS
        and metrics["last_epoch"] == EXPECTED_EPOCHS
        and metrics["contiguous_from_one"]
    )
    return {
        "variant": VARIANT_V3_ON,
        "run_dir": str(directory.resolve()),
        "metrics": metrics,
        "summary_exists": summary is not None,
        "summary_status": None if summary is None else summary.get("status"),
        "missing_required_artifacts": missing,
        "complete": complete,
    }


def inspect_training_readiness() -> dict[str, Any]:
    v3_on = inspect_v3_progress()
    v2_on = v2_post.inspect_v2_progress()
    v1_off = v1_post.inspect_run_progress(VARIANT_V1_OFF)
    ready = bool(
        v3_on["complete"] and v2_on["complete"] and v1_off["complete"]
    )
    return {
        "schema": READINESS_SCHEMA,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "expected_epochs": EXPECTED_EPOCHS,
        "multi_seed_scheduled": False,
        "v3_on": v3_on,
        "v2_on_read_only_structural_predecessor": v2_on,
        "v1_off_read_only_required_control": v1_off,
        "required_runs_complete": ready,
        "posttraining_action": "evaluate_v3_only" if ready else "wait",
    }


def formal_gate_contract() -> dict[str, Any]:
    """Validate the preregistered gate without consulting result values."""

    contract = v3_eval.performance_gate_contract()
    _require(isinstance(contract, Mapping), "V3 gate contract is missing")
    v2_contract = v2_post.v2_eval.performance_gate_contract()
    for key in (
        "anchor_target_count",
        "pd_primary_fixed_threshold_0_5",
        "miou_secondary_fixed_threshold_0_5",
        "pd_at_fa_budget",
    ):
        _require(
            contract.get(key) == v2_contract.get(key),
            f"V3 absolute gate differs from the existing V2 gate: {key}",
        )
    paired_expectations = {
        "paired_v3_on_vs_v1_off_each_checkpoint_role": VARIANT_V1_OFF,
        "paired_v3_on_vs_v2_on_each_checkpoint_role": VARIANT_V2_ON,
    }
    for name, reference in paired_expectations.items():
        value = contract.get(name)
        _require(isinstance(value, Mapping), f"V3 gate lacks {name}")
        expected = {
            "reference": reference,
            "candidate": VARIANT_V3_ON,
            "minimum_non_inferior_budget_count": 4,
            "minimum_strictly_better_budget_count": 1,
            "budget_count": 5,
        }
        _require(dict(value) == expected, f"V3 gate differs: {name}")
    for name, expected in {
        "v1_off_absolute_gate_required": False,
        "v2_predecessor_absolute_gate_required": False,
        "baseline_affects_decision": False,
        "tiny_pd_reported_not_independent_gate": True,
    }.items():
        _require(contract.get(name) is expected, f"V3 gate differs: {name}")
    _require(
        tuple(contract.get("all_required_components", ()))
        == (
            "pd_primary_absolute",
            "miou_secondary_absolute",
            "pd_primary_v3_vs_v1",
            "miou_secondary_v3_vs_v1",
            "pd_primary_v3_vs_v2",
            "miou_secondary_v3_vs_v2",
        ),
        "V3 required decision-component registry differs",
    )
    return json.loads(
        json.dumps(dict(contract), sort_keys=True, allow_nan=False)
    )


def verify_frozen_manifests() -> dict[str, Any]:
    """Read-only verification; absent V3 locks are a production hard stop."""

    for label, path in (
        ("V3 training", TRAINING_LOCK),
        ("V3 acceptance", ACCEPTANCE_LOCK),
    ):
        if not Path(path).is_file() or Path(path).is_symlink():
            raise FileNotFoundError(f"{label} source lock is unavailable: {path}")
    formal_gate_contract()
    training = v3_freeze.verify_training_lock(TRAINING_LOCK)
    acceptance = v3_freeze.verify_acceptance_lock(
        ACCEPTANCE_LOCK,
        TRAINING_LOCK,
    )
    v3 = v3_eval.verify_frozen_manifests(
        training_lock_path=TRAINING_LOCK,
        acceptance_lock_path=ACCEPTANCE_LOCK,
    )
    upstream = v2_post.verify_frozen_manifests()
    return {
        "v3_training_source_lock": str(Path(TRAINING_LOCK).resolve()),
        "v3_training_source_lock_sha256": sha256_file(TRAINING_LOCK),
        "v3_acceptance_source_lock": str(Path(ACCEPTANCE_LOCK).resolve()),
        "v3_acceptance_source_lock_sha256": sha256_file(ACCEPTANCE_LOCK),
        "v3_training_data_sha256": training["training_data_sha256"],
        "upstream_v2_training_data_sha256": acceptance[
            "upstream_v2_training_data_sha256"
        ],
        "v3_training_source_count": training["source_count"],
        "v3_acceptance_source_count": acceptance["source_count"],
        "v3_evaluation_source_binding": dict(v3),
        "upstream_v2_and_v1_source_bindings": dict(upstream),
    }


def sweep_path(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unexpected checkpoint: {checkpoint}")
    return Path(run_dir) / f"pd_fa_sweep_{Path(checkpoint).stem}.json"


def _checkpoint_role(checkpoint: str) -> str:
    try:
        return CHECKPOINT_ROLES[checkpoint]
    except KeyError as exc:
        raise ValueError(f"unexpected checkpoint: {checkpoint}") from exc


def _run_artifact_sha256(
    run_dir: Path,
    checkpoint: str,
    evaluator: Path,
) -> dict[str, str]:
    return {
        "protocol.json": sha256_file(run_dir / "protocol.json"),
        "split.json": sha256_file(run_dir / "split.json"),
        "summary.json": sha256_file(run_dir / "summary.json"),
        "metrics.jsonl": sha256_file(run_dir / "metrics.jsonl"),
        "checkpoint": sha256_file(run_dir / checkpoint),
        "evaluator": sha256_file(evaluator),
    }


def current_v3_binding(checkpoint: str) -> dict[str, Any]:
    role = _checkpoint_role(checkpoint)
    run_dir = V3_RUN_DIR.resolve()
    artifact_identity = v3_eval.validate_run_artifacts(run_dir, checkpoint)
    hashes = _run_artifact_sha256(run_dir, checkpoint, V3_EVALUATOR)
    _require(
        artifact_identity.get("variant") == VARIANT_V3_ON,
        "V3 current variant differs",
    )
    _require(
        artifact_identity.get("checkpoint_role") == role,
        "V3 current checkpoint role differs",
    )
    _require(
        artifact_identity.get("structural_predecessor") == VARIANT_V2_ON,
        "V3 current structural predecessor differs",
    )
    _require(
        artifact_identity.get("required_control") == VARIANT_V1_OFF,
        "V3 current required control differs",
    )
    split = load_json(run_dir / "split.json")
    split_hashes = split.get("hashes")
    _require(isinstance(split_hashes, Mapping), "V3 split hashes are missing")
    validation_sha = split_hashes.get("used_val_sha256")
    _require(isinstance(validation_sha, str), "V3 validation SHA is missing")
    _require(
        artifact_identity.get("validation_split_sha256") == validation_sha,
        "V3 artifact validation split SHA differs",
    )
    return {
        "variant": VARIANT_V3_ON,
        "run_dir": run_dir,
        "checkpoint_path": (run_dir / checkpoint).resolve(),
        "checkpoint_name": checkpoint,
        "checkpoint_role": role,
        "checkpoint_sha256": hashes["checkpoint"],
        "validation_split_sha256": validation_sha,
        "evaluator_path": V3_EVALUATOR.resolve(),
        "evaluator_sha256": hashes["evaluator"],
        "artifact_sha256": hashes,
        "artifact_identity": artifact_identity,
    }


def _upstream_paths() -> tuple[Path, ...]:
    paths = set(v2_post._reference_paths())
    paths.update(
        (
            Path(v1_post.TRAINING_LOCK),
            Path(v1_post.ACCEPTANCE_LOCK),
            Path(v2_post.TRAINING_LOCK),
            Path(v2_post.ACCEPTANCE_LOCK),
        )
    )
    paths.update(
        V2_RUN_DIR / name
        for name in (
            "protocol.json",
            "split.json",
            "summary.json",
            "metrics.jsonl",
            *CHECKPOINTS,
        )
    )
    paths.update(sweep_path(V2_RUN_DIR, name) for name in CHECKPOINTS)
    return tuple(sorted(path.resolve() for path in paths))


def upstream_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for path in _upstream_paths():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        stat = path.stat()
        snapshot[str(path)] = {
            "sha256": sha256_file(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return snapshot


def _normalize_v3_sweep(
    payload: Mapping[str, Any],
    *,
    checkpoint: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = binding.get("artifact_identity")
    _require(isinstance(artifact, Mapping), "V3 artifact identity is missing")
    _require(
        payload.get("validation_split_sha256")
        == binding.get("validation_split_sha256"),
        "V3 sweep validation split SHA differs",
    )
    v3_eval.validate_output_identity(payload, artifact_audit=artifact)
    fixed = _normalize_complete_fixed(
        v3_eval._normalize_fixed(payload.get("fixed_threshold_0_5")),
        location="V3 fixed_threshold_0_5",
    )
    raw_budgets = v3_eval._normalize_budgets(payload)
    budgets = {
        key: {
            "matched_target_count": point["matched_target_count"],
            "target_count": point["target_count"],
            "pd": point["pd"],
            "fa": point["achieved_fa"],
            "threshold": point["threshold"],
        }
        for key, point in raw_budgets.items()
    }
    gate = payload.get("performance_gate_assessment")
    _require(isinstance(gate, Mapping), "V3 absolute gate is missing")
    expected_gate = v3_eval._absolute_gate(
        _checkpoint_role(checkpoint),
        fixed,
        raw_budgets,
    )
    _require(dict(gate) == expected_gate, "V3 absolute gate differs")
    return {
        "source": "new_v3_candidate",
        "variant": VARIANT_V3_ON,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "checkpoint": checkpoint,
        "checkpoint_role": _checkpoint_role(checkpoint),
        "checkpoint_epoch": payload.get("checkpoint_epoch"),
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "run_directory": str(binding["run_dir"]),
        "run_identity": dict(artifact["run_identity"]),
        "source_checkpoint_identity": dict(
            artifact["checkpoint_identity"]
        ),
        "fixed_threshold_0_5": fixed,
        "pd_at_fa_budget": budgets,
        "absolute_gate": dict(gate),
        "tiny_pd_regressed": _tiny_pd_audit(fixed)[
            "tiny_pd_regressed"
        ],
    }


def validate_v3_sweep(
    path: Path,
    *,
    checkpoint: str,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = (
        current_v3_binding(checkpoint)
        if binding is None
        else dict(binding)
    )
    return _normalize_v3_sweep(
        load_json(path),
        checkpoint=checkpoint,
        binding=current,
    )


def load_upstream_rows() -> dict[tuple[str, str], dict[str, Any]]:
    """Revalidate all six upstream rows without repairing any artifact."""

    rows = v2_post.load_all_rows()
    expected = {
        (variant, checkpoint)
        for variant in (
            BASELINE_VARIANT,
            VARIANT_V1_OFF,
            VARIANT_V2_ON,
        )
        for checkpoint in CHECKPOINTS
    }
    _require(set(rows) == expected, "upstream six-row matrix differs")
    run_dirs = {
        BASELINE_VARIANT: BASELINE_RUN_DIR,
        VARIANT_V1_OFF: V1_OFF_RUN_DIR,
        VARIANT_V2_ON: V2_RUN_DIR,
    }
    complete_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for variant, checkpoint in sorted(expected):
        key = (variant, checkpoint)
        payload = load_json(sweep_path(run_dirs[variant], checkpoint))
        complete_fixed = _normalize_complete_fixed(
            payload.get("fixed_threshold_0_5"),
            location=f"{variant}:{checkpoint} fixed_threshold_0_5",
        )
        existing_fixed = rows[key].get("fixed_threshold_0_5")
        _require(
            isinstance(existing_fixed, Mapping),
            f"{variant}:{checkpoint} normalized fixed metrics are missing",
        )
        for name, value in existing_fixed.items():
            _require(
                complete_fixed.get(name) == value,
                f"{variant}:{checkpoint} fixed metric differs: {name}",
            )
        complete = dict(rows[key])
        complete["fixed_threshold_0_5"] = complete_fixed
        complete_rows[key] = complete
    return complete_rows


def load_all_rows() -> dict[tuple[str, str], dict[str, Any]]:
    rows = load_upstream_rows()
    for checkpoint in CHECKPOINTS:
        binding = current_v3_binding(checkpoint)
        rows[(VARIANT_V3_ON, checkpoint)] = validate_v3_sweep(
            sweep_path(V3_RUN_DIR, checkpoint),
            checkpoint=checkpoint,
            binding=binding,
        )
    return rows


def evaluation_command(
    *,
    checkpoint: str,
    python: Path | str = sys.executable,
    device_mode: str = "gpu23",
    physical_gpu: int = 2,
) -> tuple[list[str], dict[str, str], Path]:
    _checkpoint_role(checkpoint)
    if device_mode not in ("gpu23", "cpu"):
        raise ValueError(f"unexpected device mode: {device_mode}")
    gpu_key = str(physical_gpu)
    if gpu_key not in GPU_UUIDS:
        raise ValueError("V3 evaluation physical GPU must be 2 or 3")
    command = [
        str(python),
        str(V3_EVALUATOR.resolve()),
        "--run-dir",
        str(V3_RUN_DIR.resolve()),
        "--checkpoint",
        checkpoint,
        "--device",
        "cuda:0" if device_mode == "gpu23" else "cpu",
        "--expected-epochs",
        str(EXPECTED_EPOCHS),
    ]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(TRAINING_SEED)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if device_mode == "gpu23":
        environment["CUDA_VISIBLE_DEVICES"] = GPU_UUIDS[gpu_key]
    else:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    return command, environment, sweep_path(V3_RUN_DIR, checkpoint)


def _require_v3_owned_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    root = Path(os.path.abspath(V3_RESULT_ROOT))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"refusing to quarantine a non-V3 artifact: {absolute}"
        ) from exc
    cursor = root
    if cursor.is_symlink():
        raise ValueError(f"V3 result root is a symlink: {cursor}")
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink() and (
            cursor != absolute or cursor.is_dir()
        ):
            raise ValueError(
                f"V3 artifact ancestry contains a symlink: {cursor}"
            )
    return absolute


def _new_rejected_directory(parent: Path) -> Path:
    checked_parent = _require_v3_owned_path(parent)
    root = checked_parent / "rejected_postprocess"
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError(f"rejected-postprocess directory is a symlink: {root}")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for index in range(1000):
        candidate = root / (stamp if index == 0 else f"{stamp}.{index:03d}")
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("could not allocate a V3 rejected-postprocess directory")


def quarantine_v3_postprocess_artifacts(
    paths: Iterable[Path],
    *,
    parent: Path,
    reason: str,
) -> dict[str, Any]:
    checked_parent = _require_v3_owned_path(parent)
    existing: list[Path] = []
    for raw_path in paths:
        path = _require_v3_owned_path(Path(raw_path))
        if path.exists() or path.is_symlink():
            existing.append(path)
    if not existing:
        return {"moved": {}, "reason": reason}
    destination = _new_rejected_directory(checked_parent)
    moved: dict[str, str] = {}
    for source in existing:
        target = destination / source.name
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        os.replace(source, target)
        moved[str(source)] = str(target)
    descriptor = os.open(
        destination / "reason.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    content = _canonical_bytes(
        {
            "schema": REJECTED_SCHEMA,
            "reason": reason,
            "moved": moved,
        }
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "directory": str(destination),
        "moved": moved,
        "reason": reason,
    }


def _run_v3_evaluation(
    *,
    checkpoint: str,
    python: Path | str,
    device_mode: str,
    physical_gpu: int,
) -> dict[str, Any]:
    command, environment, output = evaluation_command(
        checkpoint=checkpoint,
        python=python,
        device_mode=device_mode,
        physical_gpu=physical_gpu,
    )
    _require_v3_owned_path(output)
    binding = current_v3_binding(checkpoint)
    rejected: list[dict[str, Any]] = []
    if output.exists() or output.is_symlink():
        try:
            row = validate_v3_sweep(
                output,
                checkpoint=checkpoint,
                binding=binding,
            )
        except Exception as exc:
            rejected.append(
                quarantine_v3_postprocess_artifacts(
                    [output],
                    parent=output.parent,
                    reason=(
                        "existing V3 sweep failed current binding: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
        else:
            return {
                "variant": VARIANT_V3_ON,
                "checkpoint": checkpoint,
                "status": "reused_valid_existing",
                "output": str(output),
                "row": row,
                "rejected_previous_outputs": rejected,
            }
    try:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
    except BaseException as exc:
        if output.exists() or output.is_symlink():
            rejected.append(
                quarantine_v3_postprocess_artifacts(
                    [output],
                    parent=output.parent,
                    reason=(
                        "V3 evaluator exited before a valid result was "
                        f"accepted: {type(exc).__name__}: {exc}"
                    ),
                )
            )
        raise
    post_binding = current_v3_binding(checkpoint)
    try:
        row = validate_v3_sweep(
            output,
            checkpoint=checkpoint,
            binding=post_binding,
        )
    except BaseException as exc:
        if output.exists() or output.is_symlink():
            rejected.append(
                quarantine_v3_postprocess_artifacts(
                    [output],
                    parent=output.parent,
                    reason=(
                        "new V3 sweep failed current binding: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
        raise
    return {
        "variant": VARIANT_V3_ON,
        "checkpoint": checkpoint,
        "status": "completed",
        "output": str(output),
        "row": row,
        "rejected_previous_outputs": rejected,
    }


def run_v3_evaluations(
    *,
    python: Path | str = sys.executable,
    device_mode: str = "gpu23",
    physical_gpu: int = 2,
) -> list[dict[str, Any]]:
    readiness = inspect_training_readiness()
    if readiness["required_runs_complete"] is not True:
        raise IncompleteTraining(
            "V3-on and immutable V2-on/V1-off trajectories must each have "
            "a complete summary and contiguous 1..800 metrics"
        )
    formal_gate_contract()
    before = upstream_snapshot()
    load_upstream_rows()
    _require(
        upstream_snapshot() == before,
        "upstream files changed during read-only preflight",
    )
    try:
        results = [
            _run_v3_evaluation(
                checkpoint=checkpoint,
                python=python,
                device_mode=device_mode,
                physical_gpu=physical_gpu,
            )
            for checkpoint in CHECKPOINTS
        ]
    finally:
        _require(
            upstream_snapshot() == before,
            "upstream files changed during V3 evaluation",
        )
    return results


def same_split_and_training_contract() -> dict[str, Any]:
    """Prove the V3/V2/V1/baseline comparison axes without changing them."""

    upstream = v2_post.same_split_and_training_contract()
    run_dirs = {
        VARIANT_V3_ON: V3_RUN_DIR,
        VARIANT_V2_ON: V2_RUN_DIR,
        VARIANT_V1_OFF: V1_OFF_RUN_DIR,
        BASELINE_VARIANT: BASELINE_RUN_DIR,
    }
    protocols = {
        name: load_json(path / "protocol.json")
        for name, path in run_dirs.items()
    }
    splits = {
        name: load_json(path / "split.json")
        for name, path in run_dirs.items()
    }
    arguments: dict[str, Mapping[str, Any]] = {}
    for name, protocol in protocols.items():
        value = protocol.get("arguments")
        _require(isinstance(value, Mapping), f"{name} arguments are missing")
        arguments[name] = value
    fixed_axes = dict(upstream["fixed_training_axes"])
    for source, source_arguments in arguments.items():
        for name, expected in fixed_axes.items():
            _require(
                source_arguments.get(name) == expected,
                f"{source} training axis differs: {name}",
            )
    reference_split = splits[VARIANT_V1_OFF]
    for source, split in splits.items():
        for name, expected in {
            "dataset": DATASET,
            "split_seed": SPLIT_SEED,
            "used_train_count": 530,
            "used_val_count": 133,
            "full_official_train_count": 663,
            "official_test_accessed": False,
        }.items():
            _require(
                split.get(name) == expected,
                f"{source} split differs: {name}",
            )
        for name in ("used_train_ids", "used_val_ids", "hashes"):
            _require(
                split.get(name) == reference_split.get(name),
                f"{source} ordered split differs: {name}",
            )
    for field in (
        "normalization",
        "optimizer",
        "loss",
        "primary_selection_rule",
        "secondary_selection_rule",
    ):
        expected = protocols[VARIANT_V1_OFF].get(field)
        _require(expected is not None, f"V1 off protocol lacks {field}")
        for source, protocol in protocols.items():
            _require(
                protocol.get(field) == expected,
                f"{source} protocol differs: {field}",
            )
    for source, protocol in protocols.items():
        _require(
            protocol.get("selection_source")
            == "internal_validation_only",
            f"{source} selection source differs",
        )
        _require(
            protocol.get("checkpoint_policy")
            == protocols[VARIANT_V1_OFF].get("checkpoint_policy"),
            f"{source} checkpoint policy differs",
        )
    _require(
        arguments[VARIANT_V3_ON].get("parent_variant")
        == arguments[VARIANT_V2_ON].get("parent_variant")
        == arguments[VARIANT_V1_OFF].get("parent_variant")
        == "tpd_clean_v8_mprs_dch_full",
        "V3/V2/V1 parent variants differ",
    )
    _require(
        arguments[VARIANT_V3_ON].get("relay_enabled") is True
        and arguments[VARIANT_V2_ON].get("relay_enabled") is True
        and arguments[VARIANT_V1_OFF].get("relay_enabled") is False,
        "V3/V2/V1 relay identities differ",
    )
    design = protocols[VARIANT_V3_ON].get("comparison_design")
    _require(isinstance(design, Mapping), "V3 comparison design is missing")
    _require(
        design.get("required_control") == VARIANT_V1_OFF,
        "V3 required control differs",
    )
    _require(
        design.get("structural_predecessor") == VARIANT_V2_ON,
        "V3 structural predecessor differs",
    )
    _require(
        design.get("relay_off_retrained") is False,
        "V3 relay-off retraining policy differs",
    )
    return {
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "multi_seed_scheduled": False,
        "same_fixed_training_axes": True,
        "same_normalization": True,
        "same_optimizer": True,
        "same_loss": True,
        "same_selection_rules": True,
        "same_checkpoint_roles": True,
        "same_ordered_train_ids": True,
        "same_ordered_validation_ids": True,
        "same_split_hashes": True,
        "official_test_accessed": False,
        "required_control": VARIANT_V1_OFF,
        "structural_predecessor": VARIANT_V2_ON,
        "relay_off_retrained": False,
        "fixed_training_axes": fixed_axes,
        "upstream_v2_contract": upstream,
    }


def _paired_gate(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    contract_key: str = (
        "paired_v3_on_vs_v1_off_each_checkpoint_role"
    ),
) -> dict[str, Any]:
    _require(
        reference.get("checkpoint_role") == candidate.get("checkpoint_role"),
        "paired rows have different checkpoint roles",
    )
    contracts = formal_gate_contract()
    contract = contracts.get(contract_key)
    _require(isinstance(contract, Mapping), f"unknown paired gate: {contract_key}")
    _require(
        reference.get("variant") == contract["reference"],
        "paired reference variant differs",
    )
    _require(
        candidate.get("variant") == contract["candidate"],
        "paired candidate variant differs",
    )
    reference_prefix = (
        "v1_off"
        if contract["reference"] == VARIANT_V1_OFF
        else "v2_on"
    )
    comparisons: dict[str, Any] = {}
    non_inferior = 0
    strictly_better = 0
    for key in BUDGET_KEYS:
        reference_point = reference["pd_at_fa_budget"][key]
        candidate_point = candidate["pd_at_fa_budget"][key]
        reference_count = int(reference_point["matched_target_count"])
        candidate_count = int(candidate_point["matched_target_count"])
        no_worse = candidate_count >= reference_count
        better = candidate_count > reference_count
        non_inferior += int(no_worse)
        strictly_better += int(better)
        comparisons[key] = {
            f"{reference_prefix}_matched_target_count": reference_count,
            "v3_on_matched_target_count": candidate_count,
            f"{reference_prefix}_pd": reference_point["pd"],
            "v3_on_pd": candidate_point["pd"],
            "v3_on_non_inferior": no_worse,
            "v3_on_strictly_better": better,
        }
    passed = bool(
        non_inferior >= contract["minimum_non_inferior_budget_count"]
        and strictly_better
        >= contract["minimum_strictly_better_budget_count"]
    )
    return {
        "checkpoint_role": reference["checkpoint_role"],
        "reference_variant": contract["reference"],
        "candidate_variant": contract["candidate"],
        "comparisons": comparisons,
        "non_inferior_budget_count": non_inferior,
        "strictly_better_budget_count": strictly_better,
        "required_non_inferior_budget_count": contract[
            "minimum_non_inferior_budget_count"
        ],
        "required_strictly_better_budget_count": contract[
            "minimum_strictly_better_budget_count"
        ],
        "budget_count": contract["budget_count"],
        "passed": passed,
    }


def _predecessor_paired_gate(
    predecessor: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    return _paired_gate(
        predecessor,
        candidate,
        contract_key="paired_v3_on_vs_v2_on_each_checkpoint_role",
    )


def _absolute_components(gate: Mapping[str, Any]) -> dict[str, bool]:
    fixed_checks = gate.get("fixed_threshold_checks")
    budget_checks = gate.get("budget_checks")
    _require(
        isinstance(fixed_checks, Mapping),
        "V3 absolute fixed-threshold checks are missing",
    )
    _require(
        isinstance(budget_checks, Mapping),
        "V3 absolute budget checks are missing",
    )
    _require(
        set(budget_checks) == set(BUDGET_KEYS),
        "V3 absolute budget-check matrix differs",
    )
    fixed_passed = bool(fixed_checks) and all(
        value is True for value in fixed_checks.values()
    )
    budgets_passed = all(
        isinstance(budget_checks[key], Mapping)
        and budget_checks[key].get("passed") is True
        for key in BUDGET_KEYS
    )
    absolute_passed = bool(
        gate.get("absolute_checkpoint_gate_passed") is True
    )
    _require(
        absolute_passed == (fixed_passed and budgets_passed),
        "V3 absolute aggregate differs from fixed/budget components",
    )
    return {
        "fixed_threshold_passed": fixed_passed,
        "all_fa_budgets_passed": budgets_passed,
        "absolute_checkpoint_gate_passed": absolute_passed,
    }


def _compare_to_baseline(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        candidate["checkpoint_role"] == baseline["checkpoint_role"],
        "candidate and baseline roles differ",
    )
    fixed_candidate = candidate["fixed_threshold_0_5"]
    fixed_baseline = baseline["fixed_threshold_0_5"]
    budgets = {}
    for key in BUDGET_KEYS:
        candidate_point = candidate["pd_at_fa_budget"][key]
        baseline_point = baseline["pd_at_fa_budget"][key]
        budgets[key] = {
            "delta_matched_targets": (
                int(candidate_point["matched_target_count"])
                - int(baseline_point["matched_target_count"])
            ),
            "delta_pd": (
                float(candidate_point["pd"]) - float(baseline_point["pd"])
            ),
            "candidate_achieved_fa": candidate_point["fa"],
            "baseline_achieved_fa": baseline_point["fa"],
        }
    return {
        "variant": candidate["variant"],
        "checkpoint_role": candidate["checkpoint_role"],
        "fixed_threshold_0_5_delta_candidate_minus_baseline": {
            "matched_targets": (
                int(fixed_candidate["matched_target_count"])
                - int(fixed_baseline["matched_target_count"])
            ),
            "pd": float(fixed_candidate["pd"]) - float(fixed_baseline["pd"]),
            "fa": float(fixed_candidate["fa"]) - float(fixed_baseline["fa"]),
            "miou": (
                float(fixed_candidate["miou"])
                - float(fixed_baseline["miou"])
            ),
            "false_objects_per_image": (
                float(fixed_candidate["false_objects_per_image"])
                - float(fixed_baseline["false_objects_per_image"])
            ),
        },
        "pd_at_fa_budget": budgets,
    }


def build_report(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    lock_bindings: Mapping[str, Any],
    comparison_contract: Mapping[str, Any],
    upstream_before: Mapping[str, Any],
    upstream_after: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        (variant, checkpoint)
        for variant in (
            BASELINE_VARIANT,
            VARIANT_V1_OFF,
            VARIANT_V2_ON,
            VARIANT_V3_ON,
        )
        for checkpoint in CHECKPOINTS
    }
    _require(set(rows) == expected, "V3 aggregate eight-row matrix differs")
    _require(
        dict(upstream_before) == dict(upstream_after),
        "upstream baseline/V1/V2 artifacts changed during aggregation",
    )
    gate_contract = formal_gate_contract()
    candidate_absolute: dict[str, Any] = {}
    absolute_components: dict[str, Any] = {}
    paired_v1: dict[str, Any] = {}
    paired_v2: dict[str, Any] = {}
    tiny_pd_by_role: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []
    ordered_rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        role = _checkpoint_role(checkpoint)
        role_name = ROLE_NAMES[role]
        complete_role_rows: dict[str, dict[str, Any]] = {}
        for variant in (
            BASELINE_VARIANT,
            VARIANT_V1_OFF,
            VARIANT_V2_ON,
            VARIANT_V3_ON,
        ):
            complete = dict(rows[(variant, checkpoint)])
            complete["fixed_threshold_0_5"] = _normalize_complete_fixed(
                complete.get("fixed_threshold_0_5"),
                location=f"{variant}:{checkpoint} fixed_threshold_0_5",
            )
            complete_role_rows[variant] = complete
        baseline = complete_role_rows[BASELINE_VARIANT]
        v1_off = complete_role_rows[VARIANT_V1_OFF]
        v2_on = complete_role_rows[VARIANT_V2_ON]
        v3_on = complete_role_rows[VARIANT_V3_ON]
        tiny_audit = _tiny_pd_audit(v3_on["fixed_threshold_0_5"])
        recorded_tiny_regression = v3_on.get("tiny_pd_regressed")
        if recorded_tiny_regression is not None:
            _require(
                type(recorded_tiny_regression) is bool
                and recorded_tiny_regression
                is tiny_audit["tiny_pd_regressed"],
                "V3 normalized tiny-Pd regression marker differs",
            )
        v3_on["tiny_pd_regressed"] = tiny_audit["tiny_pd_regressed"]
        tiny_pd_by_role[role_name] = tiny_audit
        gate = v3_on.get("absolute_gate")
        _require(isinstance(gate, Mapping), "V3 candidate gate is missing")
        candidate_absolute[role_name] = dict(gate)
        absolute_components[role_name] = _absolute_components(gate)
        paired_v1[role_name] = _paired_gate(v1_off, v3_on)
        paired_v2[role_name] = _predecessor_paired_gate(v2_on, v3_on)
        comparisons.append(_compare_to_baseline(v3_on, baseline))
        ordered_rows.extend(
            [dict(baseline), dict(v1_off), dict(v2_on), dict(v3_on)]
        )
    pd_absolute = absolute_components["pd_primary"]
    miou_absolute = absolute_components["miou_secondary"]
    pd_v1_paired = bool(paired_v1["pd_primary"]["passed"])
    miou_v1_paired = bool(paired_v1["miou_secondary"]["passed"])
    pd_v2_paired = bool(paired_v2["pd_primary"]["passed"])
    miou_v2_paired = bool(paired_v2["miou_secondary"]["passed"])
    aggregate_tiny_pd_regressed = any(
        audit["tiny_pd_regressed"] for audit in tiny_pd_by_role.values()
    )
    success_components = {
        "v3_best_fixed_threshold_absolute": pd_absolute[
            "fixed_threshold_passed"
        ],
        "v3_best_all_fa_budgets_absolute": pd_absolute[
            "all_fa_budgets_passed"
        ],
        "v3_best_absolute": pd_absolute[
            "absolute_checkpoint_gate_passed"
        ],
        "v3_best_miou_fixed_threshold_absolute": miou_absolute[
            "fixed_threshold_passed"
        ],
        "v3_best_miou_all_fa_budgets_absolute": miou_absolute[
            "all_fa_budgets_passed"
        ],
        "v3_best_miou_absolute": miou_absolute[
            "absolute_checkpoint_gate_passed"
        ],
        "v3_on_pd_primary_absolute": pd_absolute[
            "absolute_checkpoint_gate_passed"
        ],
        "v3_on_miou_secondary_absolute": miou_absolute[
            "absolute_checkpoint_gate_passed"
        ],
        "best_paired_v3_on_vs_v1_off": pd_v1_paired,
        "best_miou_paired_v3_on_vs_v1_off": miou_v1_paired,
        "best_paired_v3_on_vs_v2_on_predecessor": pd_v2_paired,
        "best_miou_paired_v3_on_vs_v2_on_predecessor": miou_v2_paired,
        "pd_primary_paired_v3_on_vs_v1_off": pd_v1_paired,
        "miou_secondary_paired_v3_on_vs_v1_off": miou_v1_paired,
        "pd_primary_paired_v3_on_vs_v2_on": pd_v2_paired,
        "miou_secondary_paired_v3_on_vs_v2_on": miou_v2_paired,
        "v1_off_absolute_gate_required": False,
        "v2_predecessor_absolute_gate_required": False,
        "baseline_affects_decision": False,
        "tiny_pd_affects_decision": False,
    }
    required_decision_components = {
        name: success_components[name]
        for name in (
            "v3_best_fixed_threshold_absolute",
            "v3_best_all_fa_budgets_absolute",
            "v3_best_miou_fixed_threshold_absolute",
            "v3_best_miou_all_fa_budgets_absolute",
            "best_paired_v3_on_vs_v1_off",
            "best_miou_paired_v3_on_vs_v1_off",
            "best_paired_v3_on_vs_v2_on_predecessor",
            "best_miou_paired_v3_on_vs_v2_on_predecessor",
        )
    }
    preregistered_required_components = {
        "pd_primary_absolute": success_components[
            "v3_on_pd_primary_absolute"
        ],
        "miou_secondary_absolute": success_components[
            "v3_on_miou_secondary_absolute"
        ],
        "pd_primary_v3_vs_v1": pd_v1_paired,
        "miou_secondary_v3_vs_v1": miou_v1_paired,
        "pd_primary_v3_vs_v2": pd_v2_paired,
        "miou_secondary_v3_vs_v2": miou_v2_paired,
    }
    _require(
        tuple(preregistered_required_components)
        == tuple(gate_contract["all_required_components"]),
        "reported V3 gate components differ from preregistration",
    )
    aggregate_passed = all(required_decision_components.values())
    _require(
        aggregate_passed
        == all(preregistered_required_components.values()),
        "expanded and preregistered V3 decisions differ",
    )
    sweep_bindings = {}
    run_dirs = {
        BASELINE_VARIANT: BASELINE_RUN_DIR,
        VARIANT_V1_OFF: V1_OFF_RUN_DIR,
        VARIANT_V2_ON: V2_RUN_DIR,
        VARIANT_V3_ON: V3_RUN_DIR,
    }
    for variant, checkpoint in sorted(expected):
        path = sweep_path(run_dirs[variant], checkpoint)
        sweep_bindings[f"{variant}:{checkpoint}"] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "multi_seed_scheduled": False,
        "official_test_accessed": False,
        "scope": "single_seed_internal_validation",
        "row_count": len(ordered_rows),
        "rows": ordered_rows,
        "preregistered_performance_gate_contract": gate_contract,
        "v3_candidate_absolute_gate_by_role": candidate_absolute,
        "v3_candidate_absolute_components_by_role": absolute_components,
        "paired_v3_on_vs_v1_off_gate_by_role": paired_v1,
        "paired_v3_on_vs_v2_on_gate_by_role": paired_v2,
        "v3_tiny_pd_regression_by_role": tiny_pd_by_role,
        "aggregate_tiny_pd_regressed": aggregate_tiny_pd_regressed,
        "tiny_pd_regression_affects_decision": False,
        "success_components": success_components,
        "preregistered_required_components": (
            preregistered_required_components
        ),
        "required_decision_components": required_decision_components,
        "aggregate_full_model_gate_passed": aggregate_passed,
        "decision": (
            "FULL_MODEL_GATE_PASSED"
            if aggregate_passed
            else "RETURN_TO_MODEL_OPTIMIZATION"
        ),
        "comparisons_vs_baseline": comparisons,
        "comparison_contract": dict(comparison_contract),
        "upstream_read_only": {
            "variants": [
                BASELINE_VARIANT,
                VARIANT_V1_OFF,
                VARIANT_V2_ON,
            ],
            "before": dict(upstream_before),
            "after": dict(upstream_after),
            "unchanged": True,
            "repair_attempted": False,
        },
        "bindings": {
            **dict(lock_bindings),
            "v3_evaluator": str(V3_EVALUATOR.resolve()),
            "v3_evaluator_sha256": sha256_file(V3_EVALUATOR),
            "v2_evaluator": str(v2_post.V2_EVALUATOR.resolve()),
            "v2_evaluator_sha256": sha256_file(v2_post.V2_EVALUATOR),
            "v1_off_evaluator": str(v1_post.NER_EVALUATOR.resolve()),
            "v1_off_evaluator_sha256": sha256_file(v1_post.NER_EVALUATOR),
            "baseline_evaluator": str(
                v1_post.BASELINE_EVALUATOR.resolve()
            ),
            "baseline_evaluator_sha256": sha256_file(
                v1_post.BASELINE_EVALUATOR
            ),
            "postprocess": str(Path(__file__).resolve()),
            "postprocess_sha256": sha256_file(Path(__file__).resolve()),
            "sweeps": sweep_bindings,
        },
        "claim_boundary": {
            "single_seed_only": True,
            "cross_seed_stability_claim": False,
            "cross_dataset_claim": False,
            "official_test_claim": False,
            "tiny_pd_reported": True,
            "tiny_pd_reference": "perfect_tiny_target_recall_39_of_39",
            "tiny_pd_regressed": aggregate_tiny_pd_regressed,
            "tiny_pd_is_independent_pass_gate": False,
            "tiny_pd_regression_does_not_change_six_component_gate": True,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V8-MPRS-DCH + five-node NER V3 formal800 comparison",
        "",
        f"- Decision: `{report['decision']}`",
        (
            "- Aggregate full-model gate passed: "
            f"`{str(report['aggregate_full_model_gate_passed']).lower()}`"
        ),
        "- Scope: fixed seed 42, NUDT-SIRST internal 530/133 validation",
        "- Baseline, V1 relay-off, and V2 relay-on files changed: `false`",
        "- Official test accessed: `false`",
        "",
        "## Fixed threshold 0.5 and Pd@Fa",
        "",
        "| Source | Variant | Role | Pd@0.5 | Fa@0.5 | mIoU@0.5 | False objects/image@0.5 | Tiny-Pd@0.5 | Tiny matched/total@0.5 | nIoU@0.5 | Pixel precision@0.5 | Pixel recall@0.5 | Pixel F1@0.5 | Pd@1e-6 | Pd@5e-6 | Pd@1e-5 | Pd@5e-5 | Pd@1e-4 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        fixed = row["fixed_threshold_0_5"]
        budget_cells = []
        for key in BUDGET_KEYS:
            point = row["pd_at_fa_budget"][key]
            budget_cells.append(
                f"{point['matched_target_count']}/{point['target_count']} "
                f"({point['pd']:.9f}; Fa={point['fa']:.9g})"
            )
        lines.append(
            f"| {row['source']} | {row['variant']} | "
            f"{ROLE_NAMES[row['checkpoint_role']]} | "
            f"{fixed['matched_target_count']}/{fixed['target_count']} "
            f"({fixed['pd']:.9f}) | {fixed['fa']:.9g} | "
            f"{fixed['miou']:.9f} | "
            f"{fixed['false_objects_per_image']:.9f} | "
            f"{fixed['tiny_pd']:.9f} | "
            f"{fixed['matched_tiny_target_count']}/"
            f"{fixed['tiny_target_count']} | "
            f"{fixed['niou']:.9f} | "
            f"{fixed['pixel_precision']:.9f} | "
            f"{fixed['pixel_recall']:.9f} | "
            f"{fixed['pixel_f1']:.9f} | "
            + " | ".join(budget_cells)
            + " |"
        )
    lines.extend(["", "## Preregistered gate outcome", ""])
    for role, components in report[
        "v3_candidate_absolute_components_by_role"
    ].items():
        lines.append(
            f"- V3-on `{role}` absolute gate: "
            f"`{str(components['absolute_checkpoint_gate_passed']).lower()}` "
            f"(fixed={str(components['fixed_threshold_passed']).lower()}, "
            f"all-budgets={str(components['all_fa_budgets_passed']).lower()})"
        )
    for role, gate in report[
        "paired_v3_on_vs_v1_off_gate_by_role"
    ].items():
        lines.append(
            f"- `{role}` V3-vs-V1 required-control gate: "
            f"`{str(gate['passed']).lower()}` "
            f"({gate['non_inferior_budget_count']}/5 non-inferior, "
            f"{gate['strictly_better_budget_count']}/5 strictly better)"
        )
    for role, gate in report[
        "paired_v3_on_vs_v2_on_gate_by_role"
    ].items():
        lines.append(
            f"- `{role}` V3-vs-V2 structural-predecessor gate: "
            f"`{str(gate['passed']).lower()}` "
            f"({gate['non_inferior_budget_count']}/5 non-inferior, "
            f"{gate['strictly_better_budget_count']}/5 strictly better)"
        )
    lines.extend(["", "## Tiny-target regression audit", ""])
    for role, audit in report["v3_tiny_pd_regression_by_role"].items():
        lines.append(
            f"- V3-on `{role}` tiny-Pd regression: "
            f"`{str(audit['tiny_pd_regressed']).lower()}` "
            f"({audit['observed_matched_tiny_target_count']}/"
            f"{audit['observed_tiny_target_count']}; "
            f"tiny-Pd={audit['observed_tiny_pd']:.9f})"
        )
    lines.append(
        "- Any V3 role below 39/39: "
        f"`{str(report['aggregate_tiny_pd_regressed']).lower()}`; "
        "this is explicitly report-only and does not alter the six-component "
        "formal gate."
    )
    lines.extend(
        [
            "",
            "V1 relay-off is the required paired control and V2 relay-on is "
            "the paired structural predecessor. Their sweeps are read-only. "
            "The baseline reports deltas only.",
            "",
            "## Conclusion boundary",
            "",
            "- This report covers only the fixed seed-42 internal-validation "
            "model decision.",
            "- It does not claim cross-seed stability, cross-dataset transfer, "
            "or official-test performance.",
            "- Tiny-Pd regression is disclosed against 39/39 separately and "
            "is not an independent pass gate.",
            "",
        ]
    )
    return "\n".join(lines)


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
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


def _atomic_publish_new(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
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
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(path) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _completion_marker_bytes(
    report: Mapping[str, Any],
    json_bytes: bytes,
    markdown_bytes: bytes,
) -> bytes:
    return _canonical_bytes(
        {
            "schema": COMPLETE_MARKER_SCHEMA,
            "status": "complete",
            "decision": report["decision"],
            "aggregate_full_model_gate_passed": report[
                "aggregate_full_model_gate_passed"
            ],
            "aggregate_tiny_pd_regressed": report[
                "aggregate_tiny_pd_regressed"
            ],
            "tiny_pd_regression_affects_decision": False,
            "outputs": {
                JSON_OUTPUT.name: hashlib.sha256(json_bytes).hexdigest(),
                MARKDOWN_OUTPUT.name: hashlib.sha256(
                    markdown_bytes
                ).hexdigest(),
            },
        }
    )


def write_report(report: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    _require_v3_owned_path(COMPARISON_DIR)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    if COMPARISON_DIR.is_symlink():
        raise ValueError("V3 comparison directory may not be a symlink")
    json_bytes = _canonical_bytes(report)
    markdown_bytes = render_markdown(report).encode("utf-8")
    marker_bytes = _completion_marker_bytes(
        report,
        json_bytes,
        markdown_bytes,
    )
    expected = {
        JSON_OUTPUT: json_bytes,
        MARKDOWN_OUTPUT: markdown_bytes,
    }
    conflicts = [
        path
        for path, content in expected.items()
        if (path.exists() or path.is_symlink())
        and (
            not path.is_file()
            or path.is_symlink()
            or path.read_bytes() != content
        )
    ]
    if conflicts:
        quarantine_v3_postprocess_artifacts(
            [JSON_OUTPUT, MARKDOWN_OUTPUT, COMPLETE_MARKER],
            parent=COMPARISON_DIR,
            reason="existing V3 report differs from the rebound aggregate",
        )
    for path, content in expected.items():
        if not path.exists() and not path.is_symlink():
            _atomic_publish_new(path, content)
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.read_bytes() == content,
            f"V3 report publication differs: {path}",
        )
    if COMPLETE_MARKER.exists() or COMPLETE_MARKER.is_symlink():
        marker_valid = bool(
            COMPLETE_MARKER.is_file()
            and not COMPLETE_MARKER.is_symlink()
            and COMPLETE_MARKER.read_bytes() == marker_bytes
        )
        if not marker_valid:
            quarantine_v3_postprocess_artifacts(
                [COMPLETE_MARKER],
                parent=COMPARISON_DIR,
                reason="existing V3 completion marker differs",
            )
    if not COMPLETE_MARKER.exists() and not COMPLETE_MARKER.is_symlink():
        _atomic_publish_new(COMPLETE_MARKER, marker_bytes)
    _require(
        COMPLETE_MARKER.is_file()
        and not COMPLETE_MARKER.is_symlink()
        and COMPLETE_MARKER.read_bytes() == marker_bytes,
        "V3 completion marker differs",
    )
    return JSON_OUTPUT, MARKDOWN_OUTPUT, COMPLETE_MARKER


def aggregate_and_write() -> tuple[dict[str, Any], tuple[Path, Path, Path]]:
    readiness = inspect_training_readiness()
    if readiness["required_runs_complete"] is not True:
        raise IncompleteTraining(
            "aggregate requires complete V3-on, V2-on, and V1-off seed-42 "
            "runs with contiguous 1..800 metrics"
        )
    locks = verify_frozen_manifests()
    before = upstream_snapshot()
    comparison_contract = same_split_and_training_contract()
    rows = load_all_rows()
    after = upstream_snapshot()
    report = build_report(
        rows,
        lock_bindings=locks,
        comparison_contract=comparison_contract,
        upstream_before=before,
        upstream_after=after,
    )
    report["readiness_binding"] = readiness
    return report, write_report(report)


def execution_plan(
    *,
    python: Path | str = sys.executable,
    device_mode: str = "gpu23",
    physical_gpu: int = 2,
) -> dict[str, Any]:
    evaluations = []
    for checkpoint in CHECKPOINTS:
        command, _, output = evaluation_command(
            checkpoint=checkpoint,
            python=python,
            device_mode=device_mode,
            physical_gpu=physical_gpu,
        )
        evaluations.append(
            {
                "variant": VARIANT_V3_ON,
                "checkpoint": checkpoint,
                "physical_gpu_index": (
                    physical_gpu if device_mode == "gpu23" else None
                ),
                "command": command,
                "output": str(output),
            }
        )
    return {
        "readiness": inspect_training_readiness(),
        "new_evaluation_count": len(evaluations),
        "new_evaluations": evaluations,
        "v2_on_evaluations": 0,
        "v1_off_evaluations": 0,
        "baseline_evaluations": 0,
        "upstream_mode": "read_only_reuse_revalidation_no_repair",
        "preregistered_performance_gate_contract": formal_gate_contract(),
        "aggregate_row_count": 8,
        "aggregate_outputs": [
            str(JSON_OUTPUT),
            str(MARKDOWN_OUTPUT),
            str(COMPLETE_MARKER),
        ],
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "multi_seed_scheduled": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess the single-seed V3 five-node NER candidate"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run-now", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--device-mode",
        choices=("gpu23", "cpu"),
        default="gpu23",
    )
    parser.add_argument(
        "--physical-gpu",
        type=int,
        choices=(2, 3),
        default=2,
    )
    return parser.parse_args(argv)


def _print_json(value: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.status:
        _print_json(inspect_training_readiness())
        return
    if args.plan:
        verify_frozen_manifests()
        _print_json(
            execution_plan(
                python=args.python,
                device_mode=args.device_mode,
                physical_gpu=args.physical_gpu,
            )
        )
        return
    if args.aggregate_only:
        report, paths = aggregate_and_write()
        print(
            f"AGGREGATE decision={report['decision']} "
            f"json={paths[0]} markdown={paths[1]} marker={paths[2]}",
            flush=True,
        )
        return
    verify_frozen_manifests()
    run_v3_evaluations(
        python=args.python,
        device_mode=args.device_mode,
        physical_gpu=args.physical_gpu,
    )
    report, paths = aggregate_and_write()
    print(
        f"COMPLETE decision={report['decision']} "
        f"json={paths[0]} markdown={paths[1]} marker={paths[2]}",
        flush=True,
    )


__all__ = [
    "ACCEPTANCE_LOCK",
    "BASELINE_RUN_DIR",
    "BUDGET_KEYS",
    "CHECKPOINTS",
    "COMPARISON_DIR",
    "COMPLETE_MARKER",
    "JSON_OUTPUT",
    "MARKDOWN_OUTPUT",
    "TRAINING_LOCK",
    "TRAINING_SEED",
    "V1_OFF_RUN_DIR",
    "V2_RUN_DIR",
    "V3_RUN_DIR",
    "aggregate_and_write",
    "build_report",
    "current_v3_binding",
    "evaluation_command",
    "execution_plan",
    "formal_gate_contract",
    "inspect_training_readiness",
    "load_all_rows",
    "run_v3_evaluations",
    "same_split_and_training_contract",
    "upstream_snapshot",
    "validate_v3_sweep",
    "write_report",
]


if __name__ == "__main__":
    main()
