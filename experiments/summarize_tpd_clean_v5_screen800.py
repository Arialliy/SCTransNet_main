#!/usr/bin/env python3
"""Audit TPD-Clean-v5 screen800 artifacts and evaluate protocol Gates A--E.

The summarizer is intentionally post-training only.  It does not train, repair,
or infer a missing result.  A gate is evaluated only after all four candidate
runs, all checkpoint roles, both sweeps per run, the frozen SPD reference, and
the pre-registered CPU/RTX-5090 smoke reports pass their integrity checks.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CANDIDATE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_clean_v5_screen800_2x5090_v1"
)
DEFAULT_FORMAL_REFERENCE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_pe_formal800_4x5090_v1"
)
DEFAULT_SMOKE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_clean_v5_preflight_v1"
)
DEFAULT_TRAINING_SOURCE_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v5_screen800_2x_source_lock.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CANDIDATE_ROOT / "NUDT-SIRST/comparison"

JSON_OUTPUT_NAME = "tpd_clean_v5_screen800_comparison.json"
MARKDOWN_OUTPUT_NAME = "tpd_clean_v5_screen800_comparison.md"
SCHEMA = "sctransnet_tpd_clean_v5_screen800_comparison_v1"

DATASET = "NUDT-SIRST"
VARIANTS = ("tpd_clean_v5_full", "tpd_clean_v5_sal_capacity")
PRIMARY_VARIANT = "tpd_clean_v5_full"
CONTROL_VARIANT = "tpd_clean_v5_sal_capacity"
SEEDS = (42, 3407)
RUN_TAG = "screen800_pd_fp32_shared2x5090_v1"
EXPECTED_EPOCHS = 800
EXPECTED_TRAIN_COUNT = 530
EXPECTED_VAL_COUNT = 133
EXPECTED_TARGET_COUNT = 189
EXPECTED_SPLIT_SEED = 20260722
EXPECTED_TOTAL_PARAMETERS = 10_843_155
EXPECTED_SHALLOW_PARAMETERS = 66_176
EXPECTED_FAMILY = "spd_anchored_tpd_clean_v5_positive_context_selector"
EXPECTED_EVALUATOR = REPO_ROOT / "experiments/evaluate_tpd_clean_v5_pd_fa.py"
PROTOCOL_PATH = REPO_ROOT / "experiments/TPD_CLEAN_V5_PROTOCOL.md"
BUDGET_KEYS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")
LAST_FLOAT32_BELOW_ONE = 0.9999999403953552
EXPECTED_TRAINING_DATA_SHA256 = (
    "39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
)
EXPECTED_TRAINING_SOURCE_FILES = frozenset(
    {
        "dataset.py",
        "experiments/TPD_CLEAN_V5_2GPU_PROTOCOL.md",
        "experiments/TPD_CLEAN_V5_PROTOCOL.md",
        "experiments/capture_tpd_clean_v5_smoke_report.py",
        "experiments/evaluate_pd_fa_sweep.py",
        "experiments/evaluate_tpd_clean_v5_pd_fa.py",
        "experiments/fingerprint_tpd_training_data.py",
        "experiments/launch_tpd_clean_v5_screen800_2x5090.sh",
        "experiments/run_tpd_clean_v5_screen800_2x5090_worker.sh",
        "experiments/smoke_tpd_clean_v3.py",
        "experiments/smoke_tpd_clean_v5.py",
        "experiments/status_tpd_clean_v5_screen800_2x5090.sh",
        "experiments/tpd_clean_screen800_source_lock.json",
        "experiments/tpd_clean_v3_screen800_source_lock.json",
        "experiments/tpd_clean_v4_screen800_2x_source_lock.json",
        "experiments/tpd_ner_v1_source_lock.json",
        "experiments/train_tpd_clean_v3.py",
        "experiments/train_tpd_clean_v5.py",
        "experiments/train_tpd_pilot.py",
        "model/Config.py",
        "model/SCTransNet.py",
        "model/tpd.py",
        "model/tpd_clean_v3.py",
        "model/tpd_clean_v5.py",
        "tests/test_evaluate_tpd_clean_v5_pd_fa.py",
        "tests/test_smoke_tpd_clean_v5.py",
        "tests/test_tpd_clean_v5.py",
        "tests/test_tpd_clean_v5_2x_runtime.py",
        "tests/test_tpd_clean_v5_runner.py",
        "tests/test_train_tpd_clean_v5.py",
        "utils.py",
        "warmup_scheduler.py",
    }
)
SMOKE_SOURCE_BINDINGS = {
    "capture_source_sha256": (
        "experiments/capture_tpd_clean_v5_smoke_report.py"
    ),
    "model_source_sha256": "model/tpd_clean_v5.py",
    "smoke_source_sha256": "experiments/smoke_tpd_clean_v5.py",
    "train_source_sha256": "experiments/train_tpd_clean_v5.py",
}
EXPECTED_SMOKE_REPORTS = {
    "cpu_all.json": {
        "cuda_visible_devices": None,
        "device": "cpu",
        "device_name": "cpu",
        "batch_size": 2,
        "patch_size": 32,
        "variants": list(VARIANTS),
    },
    "gpu2_full.json": {
        "cuda_visible_devices": "2",
        "device": "cuda:0",
        "device_name": "NVIDIA GeForce RTX 5090",
        "batch_size": 2,
        "patch_size": 64,
        "variants": [PRIMARY_VARIANT],
    },
    "gpu3_capacity.json": {
        "cuda_visible_devices": "3",
        "device": "cuda:0",
        "device_name": "NVIDIA GeForce RTX 5090",
        "batch_size": 2,
        "patch_size": 64,
        "variants": [CONTROL_VARIANT],
    },
}
EXPECTED_SCALE_PARAMETER_NAMES = frozenset(
    {
        *{
            f"embeddings_1.blocks.{index}.saliency_scale"
            for index in range(4)
        },
        *{
            f"embeddings_2.blocks.{index}.saliency_scale"
            for index in range(3)
        },
    }
)
EXPECTED_PHASE_PARAMETER_NAMES = frozenset(
    {
        *{
            f"embeddings_1.blocks.{index}.phase_compress.{parameter}"
            for index in range(4)
            for parameter in ("bias", "weight")
        },
        *{
            f"embeddings_2.blocks.{index}.phase_compress.{parameter}"
            for index in range(3)
            for parameter in ("bias", "weight")
        },
    }
)
GPU_ASSIGNMENTS = {
    (PRIMARY_VARIANT, 42): "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    (CONTROL_VARIANT, 42): "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
    (PRIMARY_VARIANT, 3407): "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
    (CONTROL_VARIANT, 3407): "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
}
ROLE_SPECS = {
    "pd_primary": {
        "checkpoint": "best.pth.tar",
        "sweep": "pd_fa_sweep_best.pth.json",
        "checkpoint_role": "best_validation_pd_primary",
        "summary_epoch": "best_pd_epoch",
        "summary_metrics": "best_pd_validation_metrics",
    },
    "miou_primary": {
        "checkpoint": "best_miou.pth.tar",
        "sweep": "pd_fa_sweep_best_miou.pth.json",
        "checkpoint_role": "best_validation_miou_secondary",
        "summary_epoch": "best_miou_epoch",
        "summary_metrics": "best_miou_validation_metrics",
    },
}
REQUIRED_INTEGRITY_CHECKS = (
    "summary_complete",
    "metrics_complete_contiguous_finite",
    "metadata_consistent",
    "official_test_isolated",
    "split_hashes_recomputed_consistent",
    "checkpoint_role_epoch_metrics_consistent",
    "global_selection_keys_recomputed",
    "state_dict_strict_load",
    "fixed_threshold_object_metrics_exact",
)
ENGINEERING_INTEGRITY_KEYS = (
    "four_runs_contiguous_800_epochs",
    "twelve_checkpoints_present_and_strict_load",
    "eight_closed_interval_sweeps",
    "model_split_protocol_evaluator_hashes_consistent",
    "cpu_and_rtx5090_smoke_passed",
    "fixed_threshold_reproduction_exact",
    "all_five_budgets_available",
    "preregistered_endpoint_provenance",
)


class IncompleteArtifact(ValueError):
    """A required artifact is missing, unfinished, or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IncompleteArtifact(message)


def _regular(path: Path, label: str) -> None:
    _require(
        path.is_file() and not path.is_symlink(),
        f"{label}: missing, linked, or non-regular file: {path}",
    )


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _regular(path, label)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise IncompleteArtifact(f"{label}: invalid JSON: {exc}") from exc
    _require(isinstance(payload, dict), f"{label}: expected a JSON object")
    _require_finite_tree(payload, label)
    return payload


def _require_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), f"{label}: non-finite number")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_tree(item, f"{label}[{index}]")


def _sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label}: invalid SHA-256",
    )
    return str(value)


def _same_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def _validate_point(
    value: Any, label: str, *, require_threshold: bool = True
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label}: point must be an object")
    point = dict(value)
    for key in ("matched_target_count", "target_count"):
        observed = point.get(key)
        _require(
            isinstance(observed, int) and not isinstance(observed, bool),
            f"{label}: {key} must be an integer",
        )
    _require(
        int(point["target_count"]) == EXPECTED_TARGET_COUNT,
        f"{label}: target_count must be {EXPECTED_TARGET_COUNT}",
    )
    _require(
        0 <= int(point["matched_target_count"]) <= EXPECTED_TARGET_COUNT,
        f"{label}: invalid matched_target_count",
    )
    for key in ("pd", "fa", "miou"):
        observed = point.get(key)
        _require(
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and math.isfinite(float(observed)),
            f"{label}: {key} must be finite numeric",
        )
    _require(0.0 <= float(point["pd"]) <= 1.0, f"{label}: Pd outside [0,1]")
    _require(0.0 <= float(point["fa"]), f"{label}: Fa must be non-negative")
    _require(
        0.0 <= float(point["miou"]) <= 1.0,
        f"{label}: mIoU outside [0,1]",
    )
    expected_pd = int(point["matched_target_count"]) / EXPECTED_TARGET_COUNT
    _require(
        math.isclose(float(point["pd"]), expected_pd, rel_tol=0.0, abs_tol=1e-15),
        f"{label}: Pd/count mismatch",
    )
    if require_threshold:
        threshold = point.get("threshold")
        _require(
            isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and math.isfinite(float(threshold))
            and 0.0 <= float(threshold) <= 1.0,
            f"{label}: threshold must be finite inside [0,1]",
        )
    return point


def _metric_subset(
    observed: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    for key, value in expected.items():
        _require(key in observed, f"{label}: missing metric {key}")
        _require(
            _same_number(observed[key], value),
            f"{label}: metric {key} differs",
        )


def _pd_selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["pd"]),
        -float(row["fa"]),
        float(row["tiny_pd"]),
        float(row["miou"]),
        -float(row["val_loss"]),
    )


def _miou_selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["miou"]),
        float(row["pd"]),
        -float(row["fa"]),
        float(row["tiny_pd"]),
        -float(row["val_loss"]),
    )


def _sweep_key(point: Mapping[str, Any]) -> tuple[float, ...]:
    tiny_pd = point.get("tiny_pd")
    return (
        float(point["pd"]),
        -float(point["fa"]),
        float(tiny_pd) if tiny_pd is not None else -1.0,
        float(point["miou"]),
        -abs(float(point["threshold"]) - 0.5),
    )


def _best_under_budget(
    points: Sequence[Mapping[str, Any]], budget: float
) -> Mapping[str, Any] | None:
    feasible = [point for point in points if float(point["fa"]) <= budget]
    return max(feasible, key=_sweep_key) if feasible else None


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    no_worse = (
        int(left["matched_target_count"]) >= int(right["matched_target_count"])
        and float(left["fa"]) <= float(right["fa"])
        and float(left["miou"]) >= float(right["miou"])
    )
    strict = (
        int(left["matched_target_count"]) > int(right["matched_target_count"])
        or float(left["fa"]) < float(right["fa"])
        or float(left["miou"]) > float(right["miou"])
    )
    return no_worse and strict


def _covers(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether ``left`` weakly covers ``right`` on Pd/Fa/mIoU."""
    return (
        int(left["matched_target_count"]) >= int(right["matched_target_count"])
        and float(left["fa"]) <= float(right["fa"])
        and float(left["miou"]) >= float(right["miou"])
    )


def _point_digest(point: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "matched_target_count": int(point["matched_target_count"]),
        "target_count": int(point["target_count"]),
        "pd": float(point["pd"]),
        "fa": float(point["fa"]),
        "miou": float(point["miou"]),
        "threshold": (
            float(point["threshold"]) if "threshold" in point else None
        ),
    }


def _get_run(
    runs: Mapping[Any, Any], variant: str, seed: int
) -> Mapping[str, Any]:
    for key in ((variant, seed), f"{variant}/seed_{seed}"):
        if key in runs:
            run = runs[key]
            _require(isinstance(run, Mapping), f"run {key}: invalid record")
            return run
    raise IncompleteArtifact(f"missing normalized run {variant}/seed={seed}")


def evaluate_engineering_gate(
    runs: Mapping[Any, Any],
    spd_reference: Mapping[str, Any],
    engineering_integrity: Mapping[str, bool],
) -> dict[str, Any]:
    """Evaluate TPD_CLEAN_V5_PROTOCOL.md Gates A--E without defaults."""
    normalized = {
        (variant, seed): _get_run(runs, variant, seed)
        for variant in VARIANTS
        for seed in SEEDS
    }
    full42 = normalized[(PRIMARY_VARIANT, 42)]
    full3407 = normalized[(PRIMARY_VARIANT, 3407)]

    a_pd = full42["roles"]["pd_primary"]["fixed_threshold_0_5"]
    a_miou = full42["roles"]["miou_primary"]["fixed_threshold_0_5"]
    gate_a_subchecks = {
        "pd_primary_matched_at_least_188": int(a_pd["matched_target_count"])
        >= 188,
        "pd_primary_fa_at_most_5e_6": float(a_pd["fa"]) <= 5e-6,
        "pd_primary_miou_at_least_0_9336470588": float(a_pd["miou"])
        >= 0.9336470588,
        "miou_primary_miou_at_least_0_946542": float(a_miou["miou"])
        >= 0.946542,
        "miou_primary_matched_at_least_187": int(
            a_miou["matched_target_count"]
        )
        >= 187,
        "miou_primary_fa_at_most_1e_6": float(a_miou["fa"]) <= 1e-6,
    }
    gate_a = {
        "passed": all(gate_a_subchecks.values()),
        "subchecks": gate_a_subchecks,
        "observed": {
            "pd_primary": _point_digest(a_pd),
            "miou_primary": _point_digest(a_miou),
        },
    }

    full42_budgets = full42["roles"]["pd_primary"]["budgets"]
    spd_budgets = spd_reference["roles"]["pd_primary"]["budgets"]
    budget_floor_checks: dict[str, Any] = {}
    spd_non_dominance_checks: dict[str, Any] = {}
    for budget in BUDGET_KEYS:
        point = full42_budgets[budget]
        minimum = 187 if budget == "1e-06" else 188
        budget_floor_checks[budget] = {
            "passed": (
                int(point["matched_target_count"]) >= minimum
                and float(point["fa"]) <= float(budget) + 1e-15
            ),
            "minimum_matched_target_count": minimum,
            "observed": _point_digest(point),
        }
        spd_point = spd_budgets[budget]
        covered = _covers(spd_point, point)
        spd_non_dominance_checks[budget] = {
            "full_not_covered_by_spd": not covered,
            "full": _point_digest(point),
            "spd": _point_digest(spd_point),
        }
    gate_b_subchecks = {
        "all_five_budget_pd_floors": all(
            check["passed"] for check in budget_floor_checks.values()
        ),
        "at_least_one_budget_not_covered_by_frozen_spd": any(
            check["full_not_covered_by_spd"]
            for check in spd_non_dominance_checks.values()
        ),
    }
    gate_b = {
        "passed": all(gate_b_subchecks.values()),
        "subchecks": gate_b_subchecks,
        "budget_floors": budget_floor_checks,
        "frozen_spd_comparisons": spd_non_dominance_checks,
    }

    c_pd = full3407["roles"]["pd_primary"]["fixed_threshold_0_5"]
    c_miou = full3407["roles"]["miou_primary"]["fixed_threshold_0_5"]
    seed_budget_checks: dict[str, Any] = {}
    within_one_count = 0
    for budget in BUDGET_KEYS:
        count42 = int(full42_budgets[budget]["matched_target_count"])
        count3407 = int(
            full3407["roles"]["pd_primary"]["budgets"][budget][
                "matched_target_count"
            ]
        )
        passed = count3407 >= count42 - 1
        within_one_count += int(passed)
        seed_budget_checks[budget] = {
            "passed": passed,
            "seed42_matched_target_count": count42,
            "seed3407_matched_target_count": count3407,
            "minimum_seed3407": count42 - 1,
        }
    gate_c_subchecks = {
        "pd_primary_matched_at_least_188": int(c_pd["matched_target_count"])
        >= 188,
        "pd_primary_fa_at_most_5e_6": float(c_pd["fa"]) <= 5e-6,
        "pd_primary_miou_at_least_0_920000": float(c_pd["miou"]) >= 0.92,
        "miou_primary_miou_at_least_0_940000": float(c_miou["miou"]) >= 0.94,
        "miou_primary_matched_at_least_186": int(
            c_miou["matched_target_count"]
        )
        >= 186,
        "miou_primary_fa_at_most_1e_6": float(c_miou["fa"]) <= 1e-6,
        "at_least_four_budgets_within_seed42_minus_one": within_one_count >= 4,
    }
    gate_c = {
        "passed": all(gate_c_subchecks.values()),
        "subchecks": gate_c_subchecks,
        "budget_stability": seed_budget_checks,
        "budget_stability_pass_count": within_one_count,
        "observed": {
            "pd_primary": _point_digest(c_pd),
            "miou_primary": _point_digest(c_miou),
        },
    }

    gate_d_seeds: dict[str, Any] = {}
    for seed in SEEDS:
        full = normalized[(PRIMARY_VARIANT, seed)]
        control = normalized[(CONTROL_VARIANT, seed)]
        comparisons: dict[str, Any] = {}
        capacity_dominance: list[str] = []
        strict_budget_advantages: list[str] = []
        nonempty_strict_advantages: list[str] = []
        for role_name in ROLE_SPECS:
            full_role = full["roles"][role_name]
            control_role = control["roles"][role_name]
            fixed_label = f"{role_name}.fixed_threshold_0_5"
            fixed_dominated = _dominates(
                control_role["fixed_threshold_0_5"],
                full_role["fixed_threshold_0_5"],
            )
            if fixed_dominated:
                capacity_dominance.append(fixed_label)
            comparisons[fixed_label] = {
                "capacity_dominates_full": fixed_dominated,
                "full": _point_digest(full_role["fixed_threshold_0_5"]),
                "capacity": _point_digest(
                    control_role["fixed_threshold_0_5"]
                ),
            }
            for budget in BUDGET_KEYS:
                label = f"{role_name}.budget.{budget}"
                full_point = full_role["budgets"][budget]
                control_point = control_role["budgets"][budget]
                dominated = _dominates(control_point, full_point)
                if dominated:
                    capacity_dominance.append(label)
                strict = _dominates(full_point, control_point)
                if strict:
                    strict_budget_advantages.append(label)
                    if float(full_point["threshold"]) < 1.0:
                        nonempty_strict_advantages.append(label)
                comparisons[label] = {
                    "capacity_dominates_full": dominated,
                    "full_strictly_dominates_capacity": strict,
                    "full_advantage_not_empty_endpoint": (
                        strict and float(full_point["threshold"]) < 1.0
                    ),
                    "full": _point_digest(full_point),
                    "capacity": _point_digest(control_point),
                }
        subchecks = {
            "capacity_never_dominates_full": not capacity_dominance,
            "full_strict_at_one_or_more_registered_budgets": bool(
                strict_budget_advantages
            ),
            "full_advantage_not_only_threshold_1_empty_endpoint": bool(
                nonempty_strict_advantages
            ),
        }
        gate_d_seeds[str(seed)] = {
            "passed": all(subchecks.values()),
            "subchecks": subchecks,
            "capacity_dominated_points": capacity_dominance,
            "full_strict_budget_advantages": strict_budget_advantages,
            "nonempty_full_strict_budget_advantages": (
                nonempty_strict_advantages
            ),
            "comparisons": comparisons,
        }
    gate_d = {
        "passed": all(item["passed"] for item in gate_d_seeds.values()),
        "per_seed": gate_d_seeds,
        "comparison_policy": {
            "capacity_dominance": (
                "Pd no lower, Fa no higher, mIoU no lower, at least one strict"
            ),
            "full_strict_advantage": (
                "strict Pareto dominance: Pd no lower, Fa no higher, "
                "mIoU no lower, and at least one strict"
            ),
            "empty_endpoint_excluded": True,
        },
    }

    _require(
        set(engineering_integrity) == set(ENGINEERING_INTEGRITY_KEYS),
        "Gate E integrity key set differs from the protocol contract",
    )
    gate_e = {
        "passed": all(engineering_integrity.values()),
        "subchecks": dict(engineering_integrity),
    }

    checks = {
        "gate_a_seed42_fixed_threshold": gate_a,
        "gate_b_seed42_budget_and_spd": gate_b,
        "gate_c_seed3407_stability": gate_c,
        "gate_d_full_vs_capacity": gate_d,
        "gate_e_engineering_integrity": gate_e,
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "protocol_section": "TPD_CLEAN_V5_PROTOCOL.md section 5",
    }


def _read_metrics(path: Path, variant: str, label: str) -> list[dict[str, Any]]:
    _regular(path, label)
    lines = path.read_text(encoding="utf-8").splitlines()
    _require(
        len(lines) == EXPECTED_EPOCHS,
        f"{label}: metrics rows={len(lines)}, expected {EXPECTED_EPOCHS}",
    )
    rows: list[dict[str, Any]] = []
    for epoch, raw in enumerate(lines, start=1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise IncompleteArtifact(
                f"{label}: invalid metrics JSON at epoch {epoch}: {exc}"
            ) from exc
        _require(isinstance(row, dict), f"{label}: metrics row not an object")
        _require(row.get("epoch") == epoch, f"{label}: non-contiguous epoch")
        _require(row.get("variant") == variant, f"{label}: metrics variant")
        _require(
            row.get("processed_train_samples") == EXPECTED_TRAIN_COUNT,
            f"{label}: processed sample count differs at epoch {epoch}",
        )
        _require_finite_tree(row, f"{label}.metrics[{epoch}]")
        _validate_point(row, f"{label}.metrics[{epoch}]", require_threshold=False)
        for key in ("tiny_pd", "val_loss", "train_loss", "learning_rate"):
            _require(
                isinstance(row.get(key), (int, float))
                and not isinstance(row.get(key), bool)
                and math.isfinite(float(row[key])),
                f"{label}: metrics {key} invalid at epoch {epoch}",
            )
        rows.append(row)
    return rows


def _validate_model_metadata(
    model: Mapping[str, Any], variant: str, label: str
) -> None:
    expected = {
        "variant": variant,
        "candidate_family": EXPECTED_FAMILY,
        "primary_candidate": variant == PRIMARY_VARIANT,
        "mainline_contract": "Keep-Context-Saliency",
        "fourth_parallel_branch_added": False,
        "fusion_support": "positive_context_selected_saliency",
        "fusion_formula": (
            "K+S*tanh(saliency_scale*(1+0.5*context_code))"
        ),
        "context_selector_floor": 0.5,
        "context_selector_ceiling": 1.5,
        "learned_scales_per_block": 1,
        "residual_bound": "absolute_residual_at_most_absolute_saliency",
        "zero_scale_reference": "dense_spd_exact",
        "total_parameters": EXPECTED_TOTAL_PARAMETERS,
        "trainable_parameters": EXPECTED_TOTAL_PARAMETERS,
        "shallow_embedding_parameters": EXPECTED_SHALLOW_PARAMETERS,
    }
    expected["context_code"] = (
        "centered_spatial_rms_tanh_fp32"
        if variant == PRIMARY_VARIANT
        else "centered_spatial_rms_tanh_fp32_ignored"
    )
    expected["context_reference"] = (
        "positive_selector"
        if variant == PRIMARY_VARIANT
        else "capacity_control"
    )
    expected["context_selector"] = (
        "positive_centered_0p5_to_1p5"
        if variant == PRIMARY_VARIANT
        else "neutral_one"
    )
    for key, value in expected.items():
        _require(model.get(key) == value, f"{label}: model metadata {key}")
    _sha(model.get("shared_initialization_sha256"), f"{label}.shared_init")
    _sha(model.get("full_initialization_sha256"), f"{label}.full_init")


def _validate_protocol(
    protocol: Mapping[str, Any], variant: str, seed: int, label: str
) -> None:
    arguments = protocol.get("arguments")
    _require(isinstance(arguments, dict), f"{label}: protocol arguments missing")
    expected = {
        "variant": variant,
        "dataset": DATASET,
        "dataset_dir": str((REPO_ROOT / "datasets").resolve()),
        "output_root": str(DEFAULT_CANDIDATE_ROOT.resolve()),
        "run_tag": RUN_TAG,
        "device": "cuda:0",
        "epochs": EXPECTED_EPOCHS,
        "batch_size": 16,
        "patch_size": 256,
        "workers": 0,
        "seed": seed,
        "split_seed": EXPECTED_SPLIT_SEED,
        "val_fraction": 0.2,
        "eval_every": 1,
        "base_lr": 0.001,
        "min_lr": 0.00001,
        "warmup_epochs": 10,
        "threshold": 0.5,
        "match_radius": 3.0,
        "tiny_area": 9,
        "amp": False,
        "max_train_images": None,
        "max_val_images": None,
    }
    for key, value in expected.items():
        _require(
            key in arguments and _same_number(arguments[key], value),
            f"{label}: protocol argument {key}",
        )
    _require(
        protocol.get("official_test_accessed") is False,
        f"{label}: official-test isolation missing",
    )
    _require(
        protocol.get("optimizer") == "Adam"
        and protocol.get("loss")
        == "sum of BCE over six deep-supervision outputs",
        f"{label}: optimizer or loss contract",
    )
    environment = protocol.get("environment")
    _require(
        isinstance(environment, dict)
        and environment.get("device") == "cuda:0"
        and environment.get("device_name") == "NVIDIA GeForce RTX 5090",
        f"{label}: training device environment",
    )
    model = protocol.get("model")
    _require(isinstance(model, dict), f"{label}: protocol model missing")
    _validate_model_metadata(model, variant, label)


def _validate_split(split: Mapping[str, Any], label: str) -> dict[str, str]:
    expected = {
        "dataset": DATASET,
        "source": "img_idx/train_NUDT-SIRST.txt",
        "official_test_accessed": False,
        "split_seed": EXPECTED_SPLIT_SEED,
        "full_internal_train_count": EXPECTED_TRAIN_COUNT,
        "full_internal_val_count": EXPECTED_VAL_COUNT,
        "used_train_count": EXPECTED_TRAIN_COUNT,
        "used_val_count": EXPECTED_VAL_COUNT,
    }
    for key, value in expected.items():
        _require(split.get(key) == value, f"{label}: split {key}")
    hashes = split.get("hashes")
    _require(isinstance(hashes, dict), f"{label}: split hashes missing")
    result = {
        key: _sha(hashes.get(key), f"{label}.{key}")
        for key in (
            "full_internal_train_sha256",
            "full_internal_val_sha256",
            "used_train_sha256",
            "used_val_sha256",
        )
    }
    _require(
        result["full_internal_train_sha256"] == result["used_train_sha256"]
        and result["full_internal_val_sha256"] == result["used_val_sha256"],
        f"{label}: full/used split hashes differ",
    )
    return result


def _validate_summary(
    summary: Mapping[str, Any],
    protocol: Mapping[str, Any],
    split_hashes: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
    variant: str,
    seed: int,
    label: str,
) -> None:
    expected = {
        "status": "complete",
        "variant": variant,
        "dataset": DATASET,
        "seed": seed,
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
    }
    for key, value in expected.items():
        _require(summary.get(key) == value, f"{label}: summary {key}")
    model = summary.get("model")
    _require(isinstance(model, dict), f"{label}: summary model missing")
    _validate_model_metadata(model, variant, label)
    _require(model == protocol.get("model"), f"{label}: model metadata drift")
    _require(
        summary.get("split_hashes") == dict(split_hashes),
        f"{label}: summary split hashes drift",
    )
    best_pd = max(rows, key=_pd_selection_key)
    best_miou = max(rows, key=_miou_selection_key)
    _require(
        summary.get("best_epoch") == best_pd["epoch"]
        and summary.get("best_pd_epoch") == best_pd["epoch"],
        f"{label}: global Pd selection differs",
    )
    _require(
        summary.get("best_miou_epoch") == best_miou["epoch"],
        f"{label}: global mIoU selection differs",
    )
    for key, selected in (
        ("best_validation_metrics", best_pd),
        ("best_pd_validation_metrics", best_pd),
        ("best_miou_validation_metrics", best_miou),
    ):
        metrics = summary.get(key)
        _require(isinstance(metrics, dict), f"{label}: summary {key} missing")
        _metric_subset(selected, metrics, f"{label}.{key}")


def _load_and_strict_check_checkpoints(
    run_dir: Path,
    variant: str,
    seed: int,
    summary: Mapping[str, Any],
    protocol: Mapping[str, Any],
    split_hashes: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    try:
        import torch
        from experiments.train_tpd_clean_v5 import build_clean_v5_model
    except Exception as exc:
        raise IncompleteArtifact(f"{label}: cannot import v5 builder: {exc}") from exc

    specs = {
        "pd_primary": (
            "best.pth.tar",
            "best_validation_pd_primary",
            int(summary["best_pd_epoch"]),
            summary["best_pd_validation_metrics"],
        ),
        "miou_primary": (
            "best_miou.pth.tar",
            "best_validation_miou_secondary",
            int(summary["best_miou_epoch"]),
            summary["best_miou_validation_metrics"],
        ),
        "last": (
            "last.pth.tar",
            "last_evaluated_epoch",
            EXPECTED_EPOCHS,
            {
                key: rows[-1][key]
                for key in summary["best_pd_validation_metrics"]
            },
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for role, (filename, checkpoint_role, epoch, expected_metrics) in specs.items():
        path = run_dir / filename
        _regular(path, f"{label}/{filename}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise IncompleteArtifact(
                f"{label}/{filename}: checkpoint load failed: {exc}"
            ) from exc
        _require(isinstance(checkpoint, dict), f"{label}/{filename}: not a dict")
        for key, value in {
            "variant": variant,
            "dataset": DATASET,
            "seed": seed,
            "split_seed": EXPECTED_SPLIT_SEED,
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
            "checkpoint_role": checkpoint_role,
            "epoch": epoch,
        }.items():
            _require(
                checkpoint.get(key) == value,
                f"{label}/{filename}: checkpoint {key}",
            )
        _require(
            checkpoint.get("model_metadata") == protocol.get("model"),
            f"{label}/{filename}: checkpoint model metadata",
        )
        _require(
            checkpoint.get("split_hashes") == dict(split_hashes),
            f"{label}/{filename}: checkpoint split hashes",
        )
        metrics = checkpoint.get("validation_metrics")
        _require(isinstance(metrics, dict), f"{label}/{filename}: metrics missing")
        _metric_subset(metrics, expected_metrics, f"{label}/{filename}.metrics")
        state = checkpoint.get("state_dict")
        _require(isinstance(state, dict) and state, f"{label}/{filename}: state")
        _require(
            all(
                isinstance(tensor, torch.Tensor)
                and bool(torch.isfinite(tensor).all())
                for tensor in state.values()
            ),
            f"{label}/{filename}: non-finite state tensor",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            model, metadata = build_clean_v5_model(variant, seed)
        _require(metadata == protocol.get("model"), f"{label}: rebuilt metadata")
        try:
            incompatible = model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise IncompleteArtifact(
                f"{label}/{filename}: strict state load failed: {exc}"
            ) from exc
        _require(
            not incompatible.missing_keys and not incompatible.unexpected_keys,
            f"{label}/{filename}: strict state incompatibility",
        )
        result[role] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "epoch": epoch,
            "checkpoint_role": checkpoint_role,
            "strict_load": True,
        }
        del model
        del checkpoint
    return result


def _validate_sweep(
    run_dir: Path,
    path: Path,
    checkpoint: Mapping[str, Any],
    role_name: str,
    variant: str,
    seed: int,
    expected_metrics: Mapping[str, Any],
    split_hashes: Mapping[str, str],
    evaluator_sha: str,
    label: str,
) -> dict[str, Any]:
    payload = _load_json(path, label)
    spec = ROLE_SPECS[role_name]
    for key, value in {
        "variant": variant,
        "dataset": DATASET,
        "seed": seed,
        "split_seed": EXPECTED_SPLIT_SEED,
        "validation_count": EXPECTED_VAL_COUNT,
        "official_test_accessed": False,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_role": spec["checkpoint_role"],
        "validation_split_sha256": split_hashes["used_val_sha256"],
    }.items():
        _require(payload.get(key) == value, f"{label}: sweep {key}")
    _require(
        payload.get("checkpoint_sha256") == checkpoint["sha256"],
        f"{label}: checkpoint hash",
    )
    _require(
        Path(str(payload.get("checkpoint", ""))).resolve()
        == Path(str(checkpoint["path"])).resolve(),
        f"{label}: checkpoint path",
    )

    points_raw = payload.get("points")
    _require(isinstance(points_raw, list) and points_raw, f"{label}: no points")
    points: list[dict[str, Any]] = []
    previous = -math.inf
    for index, raw in enumerate(points_raw):
        point = _validate_point(raw, f"{label}.points[{index}]")
        threshold = float(point["threshold"])
        _require(threshold > previous, f"{label}: thresholds not increasing")
        previous = threshold
        points.append(point)
    _require(
        len(points) >= 2
        and float(points[-2]["threshold"]) == LAST_FLOAT32_BELOW_ONE
        and float(points[-1]["threshold"]) == 1.0,
        f"{label}: closed interval endpoints missing",
    )
    _require(
        int(points[-1]["matched_target_count"]) == 0
        and float(points[-1]["pd"]) == 0.0
        and float(points[-1]["fa"]) == 0.0,
        f"{label}: threshold-1 point is not empty",
    )
    _require(
        float(points[-1]["miou"]) == 0.0
        and int(points[-1].get("predicted_object_count", -1)) == 0
        and int(points[-1].get("unmatched_predicted_object_count", -1)) == 0,
        f"{label}: threshold-1 object counts or mIoU are not empty",
    )
    provenance = payload.get("threshold_provenance")
    _require(isinstance(provenance, dict), f"{label}: provenance missing")
    expected_provenance = {
        "posthoc_endpoint_completion": False,
        "preregistered_endpoint_completion": True,
        "endpoint_protocol_stage": "before_formal_training",
        "closed_probability_interval": True,
        "score_dtype": "float32",
        "last_float32_below_one": LAST_FLOAT32_BELOW_ONE,
        "last_float32_semantics": "exact_one_score_plateau",
        "upper_boundary_threshold": 1.0,
        "upper_boundary_comparison": "prediction > threshold",
        "upper_boundary_semantics": "empty_prediction_pd0_fa0",
    }
    for key, value in expected_provenance.items():
        _require(
            provenance.get(key) == value,
            f"{label}: threshold provenance {key}",
        )
    _require(
        provenance.get("added_thresholds")
        == [LAST_FLOAT32_BELOW_ONE, 1.0],
        f"{label}: added thresholds",
    )
    _require(
        provenance.get("total_unique_threshold_count") == len(points),
        f"{label}: threshold count",
    )

    fixed = _validate_point(
        payload.get("fixed_threshold_0_5"), f"{label}.fixed"
    )
    _require(float(fixed["threshold"]) == 0.5, f"{label}: fixed threshold")
    fixed_from_points = next(
        (point for point in points if float(point["threshold"]) == 0.5), None
    )
    _require(fixed == fixed_from_points, f"{label}: fixed point differs")
    checkpoint_validation_metrics = payload.get(
        "checkpoint_validation_metrics"
    )
    _require(
        isinstance(checkpoint_validation_metrics, dict)
        and checkpoint_validation_metrics == dict(expected_metrics),
        f"{label}: checkpoint validation metrics differ",
    )
    _metric_subset(
        fixed,
        checkpoint_validation_metrics,
        f"{label}.fixed_checkpoint_metrics",
    )
    _metric_subset(fixed, expected_metrics, f"{label}.fixed")
    fixed_audit = payload.get("fixed_threshold_0_5_checkpoint_audit")
    _require(isinstance(fixed_audit, dict), f"{label}: fixed audit missing")
    _require(
        float(fixed_audit.get("max_abs_non_strict_numeric_delta", math.inf))
        == 0.0,
        f"{label}: fixed checkpoint reproduction differs",
    )

    budgets = payload.get("best_points_under_fa_budget")
    _require(
        isinstance(budgets, dict) and set(budgets) == set(BUDGET_KEYS),
        f"{label}: budget keys",
    )
    normalized_budgets: dict[str, dict[str, Any]] = {}
    for budget in BUDGET_KEYS:
        _require(budgets[budget] is not None, f"{label}: null budget {budget}")
        point = _validate_point(budgets[budget], f"{label}.budget.{budget}")
        recomputed = _best_under_budget(points, float(budget))
        _require(
            recomputed is not None and point == recomputed,
            f"{label}: budget {budget} is not recomputed optimum",
        )
        normalized_budgets[budget] = point

    audit = payload.get("audit")
    _require(isinstance(audit, dict), f"{label}: audit missing")
    _require(
        audit.get("expected_epochs") == EXPECTED_EPOCHS
        and audit.get("metrics_event_count") == EXPECTED_EPOCHS
        and audit.get("metrics_epoch_range") == [1, EXPECTED_EPOCHS]
        and audit.get("summary_status") == "complete"
        and audit.get("selection_source") == "internal_validation_only",
        f"{label}: audit completeness",
    )
    integrity = audit.get("integrity_checks_passed")
    _require(
        isinstance(integrity, dict)
        and set(integrity) == set(REQUIRED_INTEGRITY_CHECKS)
        and all(value is True for value in integrity.values()),
        f"{label}: integrity key set or value differs",
    )
    artifacts = audit.get("artifact_sha256")
    _require(
        isinstance(artifacts, dict)
        and set(artifacts)
        == {
            "protocol.json",
            "split.json",
            "summary.json",
            "metrics.jsonl",
            "checkpoint",
            "evaluator",
        },
        f"{label}: artifact hash key set differs",
    )
    _require(
        artifacts.get("checkpoint") == checkpoint["sha256"]
        and artifacts.get("evaluator") == evaluator_sha,
        f"{label}: evaluator/checkpoint hash",
    )
    for filename in ("protocol.json", "split.json", "summary.json", "metrics.jsonl"):
        _require(
            artifacts.get(filename) == sha256_file(run_dir / filename),
            f"{label}: artifact hash {filename}",
        )
    return {
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_role": spec["checkpoint_role"],
        "fixed_threshold_0_5": copy.deepcopy(fixed),
        "budgets": copy.deepcopy(normalized_budgets),
        "checkpoint": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "sweep": str(path.resolve()),
        "sweep_sha256": sha256_file(path),
        "validation_split_sha256": payload["validation_split_sha256"],
        "evaluator_sha256": evaluator_sha,
        "fixed_threshold_reproduction_exact": True,
        "all_budgets_available": True,
        "preregistered_endpoint_provenance": True,
    }


def _validate_launch(
    candidate_root: Path,
    run_dir: Path,
    variant: str,
    seed: int,
    training_lock: Path,
    training_lock_sha: str,
    label: str,
) -> dict[str, Any]:
    path = candidate_root / "launch" / f"{variant}_seed{seed}.json"
    launch = _load_json(path, f"{label}/launch")
    expected = {
        "schema": "sctransnet_tpd_clean_v5_screen800_2x5090_launch_v1",
        "variant": variant,
        "seed": seed,
        "candidate_family": EXPECTED_FAMILY,
        "gpu_uuid": GPU_ASSIGNMENTS[(variant, seed)],
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "training_data_sha256": (
            "39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
        ),
        "source_lock_sha256": training_lock_sha,
    }
    for key, value in expected.items():
        _require(launch.get(key) == value, f"{label}: launch {key}")
    _require(
        Path(str(launch.get("run_directory", ""))).resolve()
        == run_dir.resolve(),
        f"{label}: launch run directory",
    )
    _require(
        Path(str(launch.get("source_lock", ""))).resolve()
        == training_lock.resolve(),
        f"{label}: launch source lock",
    )
    policy = launch.get("policy")
    _require(isinstance(policy, dict), f"{label}: launch policy missing")
    for key, value in {
        "paired_variants": True,
        "pre_registered_seeds": [42, 3407],
        "fresh_run": True,
        "warm_start": False,
        "old_results_preserved": True,
        "shared_resource_screening": True,
        "efficiency_comparison_allowed": False,
        "official_test_accessed": False,
        "amp": False,
        "allowed_gpu_indices": [2, 3],
        "concurrent_jobs_per_gpu": 2,
        "counterbalanced_mapping": True,
        "cpu_threads_per_job": 1,
    }.items():
        _require(policy.get(key) == value, f"{label}: launch policy {key}")
    _require(
        policy.get("thread_environment")
        == {
            "OMP_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "NUMEXPR_NUM_THREADS": 1,
        },
        f"{label}: thread environment",
    )
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def load_candidate_run(
    candidate_root: Path,
    variant: str,
    seed: int,
    training_lock: Path,
    training_lock_sha: str,
    evaluator_sha: str,
) -> dict[str, Any]:
    label = f"{variant}/seed={seed}"
    run_dir = (
        candidate_root
        / DATASET
        / variant
        / f"seed_{seed}_{RUN_TAG}"
    )
    _require(
        run_dir.is_dir() and not run_dir.is_symlink(),
        f"{label}: missing run directory",
    )
    protocol_path = run_dir / "protocol.json"
    split_path = run_dir / "split.json"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    protocol = _load_json(protocol_path, f"{label}/protocol")
    split = _load_json(split_path, f"{label}/split")
    summary = _load_json(summary_path, f"{label}/summary")
    rows = _read_metrics(metrics_path, variant, label)
    _validate_protocol(protocol, variant, seed, label)
    split_hashes = _validate_split(split, label)
    _validate_summary(
        summary, protocol, split_hashes, rows, variant, seed, label
    )
    checkpoints = _load_and_strict_check_checkpoints(
        run_dir,
        variant,
        seed,
        summary,
        protocol,
        split_hashes,
        rows,
        label,
    )
    roles: dict[str, Any] = {}
    for role_name, spec in ROLE_SPECS.items():
        roles[role_name] = _validate_sweep(
            run_dir,
            run_dir / spec["sweep"],
            checkpoints[role_name],
            role_name,
            variant,
            seed,
            summary[spec["summary_metrics"]],
            split_hashes,
            evaluator_sha,
            f"{label}/{role_name}",
        )
    launch = _validate_launch(
        candidate_root,
        run_dir,
        variant,
        seed,
        training_lock,
        training_lock_sha,
        label,
    )
    log_path = candidate_root / "logs" / f"{variant}_seed{seed}.log"
    _regular(log_path, f"{label}/worker log")
    log_lines = log_path.read_text(encoding="utf-8").splitlines()
    completion = (
        f"TPDCLEANV5_2X_COMPLETE variant={variant} seed={seed} "
        f"gpu_uuid={GPU_ASSIGNMENTS[(variant, seed)]} epochs=800"
    )
    completion_lines = [line for line in log_lines if line == completion]
    _require(
        len(completion_lines) == 1,
        f"{label}: worker completion line count={len(completion_lines)}, expected 1",
    )
    forbidden_log_evidence = [
        line
        for line in log_lines
        if (
            "TPDCLEANV5_2X_ABORT" in line
            or "traceback" in line.lower()
            or "out of memory" in line.lower()
            or "resume" in line.lower()
            or re.search(r"\boom\b", line, flags=re.IGNORECASE) is not None
        )
    ]
    _require(
        not forbidden_log_evidence,
        f"{label}: worker log contains failure or resume evidence",
    )
    return {
        "variant": variant,
        "seed": seed,
        "run_directory": str(run_dir.resolve()),
        "roles": roles,
        "model": copy.deepcopy(summary["model"]),
        "split_hashes": split_hashes,
        "metrics_event_count": len(rows),
        "checkpoints": checkpoints,
        "launch": launch,
        "artifacts": {
            "protocol.json": sha256_file(protocol_path),
            "split.json": sha256_file(split_path),
            "summary.json": sha256_file(summary_path),
            "metrics.jsonl": sha256_file(metrics_path),
            "worker.log": sha256_file(log_path),
        },
    }


def load_spd_reference(
    formal_root: Path, validation_split_sha: str
) -> dict[str, Any]:
    run_dir = (
        formal_root
        / DATASET
        / "spd"
        / "seed_42_formal800_pd_fp32_4x5090_v1"
    )
    sweep_path = run_dir / "pd_fa_sweep_best.pth.json"
    checkpoint_path = run_dir / "best.pth.tar"
    protocol_path = run_dir / "protocol.json"
    split_path = run_dir / "split.json"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    sweep = _load_json(sweep_path, "frozen SPD sweep")
    protocol = _load_json(protocol_path, "frozen SPD protocol")
    split = _load_json(split_path, "frozen SPD split")
    summary = _load_json(summary_path, "frozen SPD summary")
    rows = _read_metrics(metrics_path, "spd", "frozen SPD")
    _regular(checkpoint_path, "frozen SPD checkpoint")
    _require(sweep.get("variant") == "spd", "frozen SPD variant")
    _require(sweep.get("seed") == 42, "frozen SPD seed")
    _require(sweep.get("dataset") == DATASET, "frozen SPD dataset")
    _require(
        sweep.get("checkpoint_role") == "best_validation_pd_primary",
        "frozen SPD checkpoint role",
    )
    _require(
        sweep.get("validation_split_sha256") == validation_split_sha,
        "frozen SPD validation split differs",
    )
    protocol_arguments = protocol.get("arguments")
    _require(
        isinstance(protocol_arguments, dict),
        "frozen SPD protocol arguments missing",
    )
    for key, value in {
        "variant": "spd",
        "dataset": DATASET,
        "epochs": EXPECTED_EPOCHS,
        "seed": 42,
        "split_seed": EXPECTED_SPLIT_SEED,
        "eval_every": 1,
        "threshold": 0.5,
        "amp": False,
    }.items():
        _require(
            key in protocol_arguments
            and _same_number(protocol_arguments[key], value),
            f"frozen SPD protocol argument {key}",
        )
    _require(
        protocol.get("official_test_accessed") is False,
        "frozen SPD protocol official-test isolation differs",
    )
    split_hashes = _validate_split(split, "frozen SPD")
    _require(
        split_hashes["used_val_sha256"] == validation_split_sha,
        "frozen SPD split manifest differs",
    )
    for key, value in {
        "status": "complete",
        "variant": "spd",
        "dataset": DATASET,
        "seed": 42,
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
    }.items():
        _require(summary.get(key) == value, f"frozen SPD summary {key}")
    _require(
        summary.get("split_hashes") == split_hashes,
        "frozen SPD summary split hashes differ",
    )
    recomputed_best = max(rows, key=_pd_selection_key)
    _require(
        summary.get("best_epoch") == recomputed_best["epoch"]
        and summary.get("best_pd_epoch") == recomputed_best["epoch"],
        "frozen SPD summary best epoch differs",
    )
    summary_metrics = summary.get("best_pd_validation_metrics")
    _require(
        isinstance(summary_metrics, dict)
        and summary.get("best_validation_metrics") == summary_metrics,
        "frozen SPD summary Pd metrics differ",
    )
    _metric_subset(recomputed_best, summary_metrics, "frozen SPD summary metrics")
    checkpoint_sha = sha256_file(checkpoint_path)
    _require(
        sweep.get("checkpoint_sha256") == checkpoint_sha,
        "frozen SPD checkpoint hash differs",
    )
    _require(
        Path(str(sweep.get("checkpoint", ""))).resolve()
        == checkpoint_path.resolve(),
        "frozen SPD checkpoint path differs",
    )
    checkpoint_metrics = sweep.get("checkpoint_validation_metrics")
    _require(
        isinstance(checkpoint_metrics, dict)
        and checkpoint_metrics == summary_metrics,
        "frozen SPD checkpoint validation metrics differ",
    )
    points_raw = sweep.get("points")
    _require(isinstance(points_raw, list) and points_raw, "frozen SPD points")
    points = [
        _validate_point(point, f"frozen SPD point {index}")
        for index, point in enumerate(points_raw)
        ]
    budgets = sweep.get("best_points_under_fa_budget")
    _require(
        isinstance(budgets, dict) and set(budgets) == set(BUDGET_KEYS),
        "frozen SPD budgets differ",
    )
    normalized: dict[str, dict[str, Any]] = {}
    for budget in BUDGET_KEYS:
        _require(budgets[budget] is not None, f"frozen SPD budget {budget} null")
        point = _validate_point(budgets[budget], f"frozen SPD budget {budget}")
        recomputed = _best_under_budget(points, float(budget))
        _require(
            recomputed is not None and point == recomputed,
            f"frozen SPD budget {budget} differs from sweep",
        )
        normalized[budget] = point
    fixed = _validate_point(sweep.get("fixed_threshold_0_5"), "frozen SPD fixed")
    _require(float(fixed["threshold"]) == 0.5, "frozen SPD fixed threshold")
    _require(
        any(point == fixed for point in points),
        "frozen SPD fixed point is absent from sweep",
    )
    _metric_subset(fixed, checkpoint_metrics, "frozen SPD fixed metrics")
    fixed_audit = sweep.get("fixed_threshold_0_5_checkpoint_audit")
    _require(
        isinstance(fixed_audit, dict)
        and float(
            fixed_audit.get(
                "max_abs_non_strict_numeric_delta",
                math.inf,
            )
        )
        == 0.0,
        "frozen SPD fixed checkpoint reproduction differs",
    )
    audit = sweep.get("audit")
    _require(isinstance(audit, dict), "frozen SPD audit missing")
    _require(
        audit.get("expected_epochs") == EXPECTED_EPOCHS
        and audit.get("metrics_event_count") == EXPECTED_EPOCHS
        and audit.get("metrics_epoch_range") == [1, EXPECTED_EPOCHS]
        and audit.get("summary_status") == "complete"
        and audit.get("selection_source") == "internal_validation_only",
        "frozen SPD audit completeness differs",
    )
    audit_integrity = audit.get("integrity_checks_passed")
    _require(
        isinstance(audit_integrity, dict)
        and set(audit_integrity) == set(REQUIRED_INTEGRITY_CHECKS)
        and all(value is True for value in audit_integrity.values()),
        "frozen SPD audit integrity differs",
    )
    artifact_hashes = audit.get("artifact_sha256")
    expected_artifact_paths = {
        "protocol.json": protocol_path,
        "split.json": split_path,
        "summary.json": summary_path,
        "metrics.jsonl": metrics_path,
        "checkpoint": checkpoint_path,
        "evaluator": REPO_ROOT / "experiments/evaluate_pd_fa_sweep.py",
    }
    _require(
        isinstance(artifact_hashes, dict)
        and set(artifact_hashes) == set(expected_artifact_paths),
        "frozen SPD artifact hash key set differs",
    )
    for name, artifact_path in expected_artifact_paths.items():
        _regular(artifact_path, f"frozen SPD artifact {name}")
        _require(
            artifact_hashes.get(name) == sha256_file(artifact_path),
            f"frozen SPD current artifact hash differs: {name}",
        )
    return {
        "seed": 42,
        "audit": {
            "expected_epochs": EXPECTED_EPOCHS,
            "metrics_event_count": len(rows),
            "summary_status": summary["status"],
            "integrity_checks_passed": copy.deepcopy(audit_integrity),
            "artifact_sha256": copy.deepcopy(artifact_hashes),
            "fixed_threshold_reproduction_exact": True,
        },
        "roles": {
            "pd_primary": {
                "fixed_threshold_0_5": fixed,
                "budgets": normalized,
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "sweep": str(sweep_path.resolve()),
                "sweep_sha256": sha256_file(sweep_path),
            }
        },
    }


def validate_training_source_lock(path: Path) -> tuple[dict[str, Any], str]:
    payload = _load_json(path, "v5 training source lock")
    expected_metadata = {
        "schema": "sctransnet_tpd_clean_v5_screen800_2x_source_lock_v1",
        "candidate_family": EXPECTED_FAMILY,
        "mainline_contract": "Keep-Context-Saliency",
        "fourth_parallel_branch_added": False,
        "resource_protocol": "two_rtx5090_counterbalanced_shared",
        "allowed_gpu_indices": [2, 3],
        "concurrent_jobs_per_gpu": 2,
        "variants": list(VARIANTS),
        "model_seeds": list(SEEDS),
        "training_data_sha256": EXPECTED_TRAINING_DATA_SHA256,
    }
    for key, expected in expected_metadata.items():
        _require(
            payload.get(key) == expected,
            f"v5 training source-lock metadata differs: {key}",
        )
    entries = payload.get("source_sha256")
    _require(
        isinstance(entries, dict)
        and set(entries) == set(EXPECTED_TRAINING_SOURCE_FILES),
        "source-lock source path set differs from the frozen 32-file contract",
    )
    for relative, expected in entries.items():
        _sha(expected, f"training source digest {relative}")
        source = REPO_ROOT / relative
        _regular(source, f"training source {relative}")
        _require(
            sha256_file(source) == expected,
            f"training source hash drift: {relative}",
        )
    frozen_locks = {
        "frozen_v4_source_lock_sha256": (
            "experiments/tpd_clean_v4_screen800_2x_source_lock.json"
        ),
        "frozen_v3_source_lock_sha256": (
            "experiments/tpd_clean_v3_screen800_source_lock.json"
        ),
        "frozen_v2_source_lock_sha256": (
            "experiments/tpd_clean_screen800_source_lock.json"
        ),
        "frozen_ner_source_lock_sha256": (
            "experiments/tpd_ner_v1_source_lock.json"
        ),
    }
    for metadata_key, relative in frozen_locks.items():
        _require(
            payload.get(metadata_key) == entries[relative],
            f"source-lock recursive binding differs: {metadata_key}",
        )
    evaluator_relative = "experiments/evaluate_tpd_clean_v5_pd_fa.py"
    _require(
        entries.get(evaluator_relative) == sha256_file(EXPECTED_EVALUATOR),
        "source lock does not bind the v5 evaluator",
    )
    smoke_entries = payload.get("smoke_sha256")
    _require(
        isinstance(smoke_entries, dict)
        and set(smoke_entries) == set(EXPECTED_SMOKE_REPORTS),
        "source lock does not bind exactly three V5 smoke reports",
    )
    for name, digest in smoke_entries.items():
        _sha(digest, f"source-lock smoke digest {name}")
    return payload, sha256_file(path)


def validate_smoke_reports(
    smoke_root: Path,
    training_lock_payload: Mapping[str, Any],
) -> dict[str, Any]:
    bound = training_lock_payload.get("smoke_sha256")
    _require(
        isinstance(bound, dict)
        and set(bound) == set(EXPECTED_SMOKE_REPORTS),
        "training source lock smoke binding differs",
    )
    reports: dict[str, dict[str, Any]] = {}
    initial_checksums: set[str] = set()
    for name, expected in EXPECTED_SMOKE_REPORTS.items():
        path = smoke_root / name
        envelope = _load_json(path, f"smoke/{name}")
        _require(
            sha256_file(path) == bound[name],
            f"smoke/{name}: digest differs from training source lock",
        )
        _require(
            envelope.get("schema")
            == "sctransnet_tpd_clean_v5_persisted_smoke_v1"
            and envelope.get("status") == "complete",
            f"smoke/{name}: envelope contract",
        )
        _require(
            envelope.get("cuda_visible_devices")
            == expected["cuda_visible_devices"],
            f"smoke/{name}: CUDA visibility provenance",
        )
        source_entries = training_lock_payload["source_sha256"]
        for envelope_key, relative in SMOKE_SOURCE_BINDINGS.items():
            _require(
                envelope.get(envelope_key) == source_entries[relative],
                f"smoke/{name}: source binding differs for {envelope_key}",
            )
        report = envelope.get("report")
        _require(isinstance(report, dict), f"smoke/{name}: report missing")
        _require(
            report.get("schema") == "sctransnet_tpd_clean_v5_smoke_v1"
            and report.get("status") == "complete"
            and report.get("paired_initialization") is True
            and report.get("device") == expected["device"]
            and report.get("device_name") == expected["device_name"]
            and report.get("batch_size") == expected["batch_size"]
            and report.get("patch_size") == expected["patch_size"]
            and report.get("steps") == 2
            and report.get("seed") == 42
            and report.get("learned_scales_per_block") == 1
            and report.get("context_selector_range") == [0.5, 1.5]
            and report.get("fusion_formula")
            == "K+S*tanh(saliency_scale*(1+0.5*context_code))"
            and report.get("residual_bound")
            == "absolute_residual_at_most_absolute_saliency",
            f"smoke/{name}: report contract",
        )
        variants = report.get("variants")
        _require(isinstance(variants, list), f"smoke/{name}: variants")
        _require(
            [entry.get("variant") for entry in variants]
            == expected["variants"],
            f"smoke/{name}: ordered variant matrix",
        )
        for entry in variants:
            variant = str(entry.get("variant"))
            expected_context_code = (
                "centered_spatial_rms_tanh_fp32"
                if variant == PRIMARY_VARIANT
                else "centered_spatial_rms_tanh_fp32_ignored"
            )
            _require(
                entry.get("status") == "complete"
                and entry.get("output_count") == 6
                and entry.get("step_zero_exact_spd") is True
                and entry.get("strict_rebuild_load") is True
                and float(entry.get("strict_reload_max_abs_difference", math.inf))
                == 0.0
                and entry.get("total_parameters") == EXPECTED_TOTAL_PARAMETERS
                and entry.get("shallow_embedding_parameters")
                == EXPECTED_SHALLOW_PARAMETERS
                and entry.get("context_code") == expected_context_code,
                f"smoke/{name}/{variant}: invariant failed",
            )
            _sha(
                entry.get("initial_model_checksum"),
                f"smoke/{name}/{variant}: initial checksum",
            )
            _sha(
                entry.get("trained_model_checksum"),
                f"smoke/{name}/{variant}: trained checksum",
            )
            initial_checksums.add(str(entry["initial_model_checksum"]))
            expected_lengths = {
                "scale_gradient_l1": EXPECTED_SCALE_PARAMETER_NAMES,
                "scale_update_l1": EXPECTED_SCALE_PARAMETER_NAMES,
                "phase_gradient_l1": EXPECTED_PHASE_PARAMETER_NAMES,
                "phase_update_l1": EXPECTED_PHASE_PARAMETER_NAMES,
            }
            for key, expected_names in expected_lengths.items():
                values = entry.get(key)
                _require(
                    isinstance(values, dict)
                    and set(values) == set(expected_names)
                    and all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        and float(value) > 0.0
                        for value in values.values()
                    ),
                    f"smoke/{name}/{variant}: {key}",
                )
            losses = entry.get("losses")
            _require(
                isinstance(losses, list)
                and len(losses) == 2
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in losses
                ),
                f"smoke/{name}/{variant}: losses",
            )
        cuda_memory = report.get("cuda_memory")
        if expected["device"] == "cpu":
            _require(cuda_memory is None, f"smoke/{name}: CPU CUDA memory")
        else:
            _require(
                isinstance(cuda_memory, dict)
                and set(cuda_memory)
                == {"peak_allocated_mib", "peak_reserved_mib"}
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    and float(value) > 0.0
                    for value in cuda_memory.values()
                ),
                f"smoke/{name}: CUDA memory report",
            )
        reports[name] = envelope
    _require(
        len(initial_checksums) == 1,
        "smoke reports: candidate initial states are not exactly paired",
    )
    return {
        "root": str(smoke_root.resolve()),
        "binding": "training_source_lock.smoke_sha256",
        "paired_initialization_sha256": next(iter(initial_checksums)),
        "reports": {
            name: {
                "path": str((smoke_root / name).resolve()),
                "sha256": bound[name],
                "cuda_visible_devices": EXPECTED_SMOKE_REPORTS[name][
                    "cuda_visible_devices"
                ],
            }
            for name in sorted(bound)
        },
        "passed": True,
    }


def _json_runs(
    runs: Mapping[tuple[str, int], Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        f"{variant}/seed_{seed}": copy.deepcopy(runs[(variant, seed)])
        for seed in SEEDS
        for variant in VARIANTS
        if (variant, seed) in runs
    }


def _build_report_strict(
    candidate_root: Path,
    formal_reference_root: Path,
    smoke_root: Path,
    training_source_lock: Path,
) -> dict[str, Any]:
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    reasons: list[str] = []
    runs: dict[tuple[str, int], dict[str, Any]] = {}
    training_lock_payload: dict[str, Any] = {}
    training_lock_sha = ""
    evaluator_sha = ""
    smoke: dict[str, Any] = {}

    try:
        training_lock_payload, training_lock_sha = validate_training_source_lock(
            training_source_lock
        )
        evaluator_sha = sha256_file(EXPECTED_EVALUATOR)
    except IncompleteArtifact as exc:
        reasons.append(f"training source lock: {exc}")
    if training_lock_sha:
        for variant in VARIANTS:
            for seed in SEEDS:
                try:
                    runs[(variant, seed)] = load_candidate_run(
                        candidate_root,
                        variant,
                        seed,
                        training_source_lock,
                        training_lock_sha,
                        evaluator_sha,
                    )
                except IncompleteArtifact as exc:
                    reasons.append(f"{variant}/seed={seed}: {exc}")
    if training_lock_payload:
        try:
            smoke = validate_smoke_reports(
                smoke_root,
                training_lock_payload,
            )
        except IncompleteArtifact as exc:
            reasons.append(f"smoke reports: {exc}")

    common = {
        "schema": SCHEMA,
        "generated_at_utc": generated,
        "status": "incomplete" if reasons else "complete",
        "scope": {
            "dataset": DATASET,
            "candidate_variants": list(VARIANTS),
            "model_seeds": list(SEEDS),
            "epochs": EXPECTED_EPOCHS,
            "split_seed": EXPECTED_SPLIT_SEED,
            "fa_budgets": list(BUDGET_KEYS),
            "official_test_accessed": False,
            "candidate_root": str(candidate_root.resolve()),
            "formal_reference_root": str(formal_reference_root.resolve()),
            "smoke_root": str(smoke_root.resolve()),
            "training_source_lock": str(training_source_lock.resolve()),
            "protocol": str(PROTOCOL_PATH.resolve()),
        },
        "candidate_runs": _json_runs(runs),
        "gate_evaluated": False,
        "engineering_gate_passed": None,
        "mainline_changed": False,
        "paper_core_established": False,
        "stability_claim_supported": False,
        "ner_stage_authorized": False,
    }
    if reasons:
        return {
            **common,
            "incomplete_reasons": reasons,
            "required_before_gate_evaluation": {
                "candidate_runs": 4,
                "metrics_rows_per_run": EXPECTED_EPOCHS,
                "candidate_checkpoints": 12,
                "candidate_sweeps": 8,
                "candidate_budget_points_per_sweep": 5,
                "smoke_reports": 3,
                "frozen_spd_reference": 1,
            },
            "decision": "INCOMPLETE",
        }

    split_fingerprints = {
        (
            run["split_hashes"]["used_train_sha256"],
            run["split_hashes"]["used_val_sha256"],
        )
        for run in runs.values()
    }
    _require(len(split_fingerprints) == 1, "candidate splits differ")
    validation_sha = next(iter(split_fingerprints))[1]
    for seed in SEEDS:
        full = runs[(PRIMARY_VARIANT, seed)]["model"]
        control = runs[(CONTROL_VARIANT, seed)]["model"]
        _require(
            full["full_initialization_sha256"]
            == control["full_initialization_sha256"],
            f"seed={seed}: paired full initialization differs",
        )
        _require(
            full["shared_initialization_sha256"]
            == control["shared_initialization_sha256"],
            f"seed={seed}: paired shared initialization differs",
        )
    spd = load_spd_reference(formal_reference_root, validation_sha)

    integrity = {
        "four_runs_contiguous_800_epochs": (
            len(runs) == 4
            and all(
                run["metrics_event_count"] == EXPECTED_EPOCHS
                for run in runs.values()
            )
        ),
        "twelve_checkpoints_present_and_strict_load": (
            sum(len(run["checkpoints"]) for run in runs.values()) == 12
            and all(
                checkpoint["strict_load"] is True
                for run in runs.values()
                for checkpoint in run["checkpoints"].values()
            )
        ),
        "eight_closed_interval_sweeps": (
            sum(len(run["roles"]) for run in runs.values()) == 8
        ),
        "model_split_protocol_evaluator_hashes_consistent": (
            all(
                role["evaluator_sha256"] == evaluator_sha
                for run in runs.values()
                for role in run["roles"].values()
            )
            and training_lock_payload["source_sha256"][
                "experiments/evaluate_tpd_clean_v5_pd_fa.py"
            ]
            == evaluator_sha
        ),
        "cpu_and_rtx5090_smoke_passed": smoke.get("passed") is True,
        "fixed_threshold_reproduction_exact": all(
            role["fixed_threshold_reproduction_exact"] is True
            for run in runs.values()
            for role in run["roles"].values()
        ),
        "all_five_budgets_available": all(
            role["all_budgets_available"] is True
            and set(role["budgets"]) == set(BUDGET_KEYS)
            for run in runs.values()
            for role in run["roles"].values()
        ),
        "preregistered_endpoint_provenance": all(
            role["preregistered_endpoint_provenance"] is True
            for run in runs.values()
            for role in run["roles"].values()
        ),
    }
    gate = evaluate_engineering_gate(runs, spd, integrity)
    gate_passed = bool(gate["passed"])
    return {
        **common,
        "status": "complete",
        "frozen_spd_reference": spd,
        "smoke_validation": smoke,
        "training_source_lock_sha256": training_lock_sha,
        "evaluator_sha256": evaluator_sha,
        "engineering_integrity": integrity,
        "engineering_gate": gate,
        "gate_evaluated": True,
        "engineering_gate_passed": gate_passed,
        "ner_stage_authorized": gate_passed,
        "validation": {
            "candidate_run_count": 4,
            "candidate_metrics_event_count": 3200,
            "candidate_checkpoint_count": 12,
            "candidate_sweep_count": 8,
            "paired_initialization": True,
            "paired_split_fingerprint": True,
        },
        "decision": (
            "ENGINEERING_GATE_PASS" if gate_passed else "ENGINEERING_GATE_FAIL"
        ),
        "decision_boundary": {
            "gate_only_controls_next_engineering_stage": True,
            "automatic_mainline_replacement": False,
            "mainline_changed": False,
            "paper_core_established": False,
            "stability_claim_supported": False,
        },
    }


def build_report(
    candidate_root: Path,
    formal_reference_root: Path,
    smoke_root: Path,
    training_source_lock: Path,
) -> dict[str, Any]:
    """Build a report while preserving null-gate semantics on every gap."""

    try:
        return _build_report_strict(
            candidate_root,
            formal_reference_root,
            smoke_root,
            training_source_lock,
        )
    except IncompleteArtifact as exc:
        return {
            "schema": SCHEMA,
            "generated_at_utc": dt.datetime.now(
                dt.timezone.utc
            ).isoformat(),
            "status": "incomplete",
            "scope": {
                "dataset": DATASET,
                "candidate_variants": list(VARIANTS),
                "model_seeds": list(SEEDS),
                "epochs": EXPECTED_EPOCHS,
                "split_seed": EXPECTED_SPLIT_SEED,
                "fa_budgets": list(BUDGET_KEYS),
                "official_test_accessed": False,
                "candidate_root": str(candidate_root.resolve()),
                "formal_reference_root": str(
                    formal_reference_root.resolve()
                ),
                "smoke_root": str(smoke_root.resolve()),
                "training_source_lock": str(
                    training_source_lock.resolve()
                ),
                "protocol": str(PROTOCOL_PATH.resolve()),
            },
            "candidate_runs": {},
            "incomplete_reasons": [f"post-training audit: {exc}"],
            "required_before_gate_evaluation": {
                "candidate_runs": 4,
                "metrics_rows_per_run": EXPECTED_EPOCHS,
                "candidate_checkpoints": 12,
                "candidate_sweeps": 8,
                "candidate_budget_points_per_sweep": 5,
                "smoke_reports": 3,
                "frozen_spd_reference": 1,
            },
            "gate_evaluated": False,
            "engineering_gate_passed": None,
            "mainline_changed": False,
            "paper_core_established": False,
            "stability_claim_supported": False,
            "ner_stage_authorized": False,
            "decision": "INCOMPLETE",
        }


def _metric_cell(point: Mapping[str, Any]) -> str:
    return (
        f"{int(point['matched_target_count'])}/{int(point['target_count'])}; "
        f"Fa={float(point['fa']):.8g}; "
        f"mIoU={float(point['miou']):.6f}; "
        f"thr={float(point['threshold']):.9g}"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    status = (
        "INCOMPLETE"
        if report["status"] == "incomplete"
        else str(report["decision"])
    )
    lines = [
        "# TPD-Clean-v5 screen800 comparison",
        "",
        f"Status: **{status}**",
        "",
        "NUDT-SIRST official-training-index internal 530/133 split only; "
        "official test data were not accessed.",
        "",
    ]
    if report["status"] == "incomplete":
        lines.extend(["## Incomplete artifacts", ""])
        lines.extend(
            f"- {reason}" for reason in report.get("incomplete_reasons", [])
        )
        lines.extend(
            [
                "",
                "Gate A–E were not evaluated and missing values were not "
                "converted into gate failures.",
                "",
                "- `engineering_gate_passed=null`",
                "- `mainline_changed=false`",
                "- `paper_core_established=false`",
                "- `stability_claim_supported=false`",
                "",
            ]
        )
        return "\n".join(lines)

    lines.extend(
        [
            "## Fixed threshold 0.5",
            "",
            "| Seed | Variant | Pd-primary | mIoU-primary |",
            "| ---: | --- | --- | --- |",
        ]
    )
    runs = report["candidate_runs"]
    for seed in SEEDS:
        for variant in VARIANTS:
            run = runs[f"{variant}/seed_{seed}"]
            lines.append(
                f"| {seed} | `{variant}` | "
                f"{_metric_cell(run['roles']['pd_primary']['fixed_threshold_0_5'])} | "
                f"{_metric_cell(run['roles']['miou_primary']['fixed_threshold_0_5'])} |"
            )
    lines.extend(
        [
            "",
            "## Five registered Fa budgets (Pd-primary)",
            "",
            "| Seed | Variant | Budget | Result |",
            "| ---: | --- | ---: | --- |",
        ]
    )
    for seed in SEEDS:
        for variant in VARIANTS:
            role = runs[f"{variant}/seed_{seed}"]["roles"]["pd_primary"]
            for budget in BUDGET_KEYS:
                lines.append(
                    f"| {seed} | `{variant}` | `{budget}` | "
                    f"{_metric_cell(role['budgets'][budget])} |"
                )
    lines.extend(
        [
            "",
            "## Protocol Gate A–E",
            "",
            "| Gate | Result |",
            "| --- | --- |",
        ]
    )
    for name, check in report["engineering_gate"]["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if check['passed'] else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            f"- `engineering_gate_passed={str(report['engineering_gate_passed']).lower()}`",
            f"- `ner_stage_authorized={str(report['ner_stage_authorized']).lower()}`",
            "- `mainline_changed=false`",
            "- `paper_core_established=false`",
            "- `stability_claim_supported=false`",
            "",
            "Passing all five gates authorizes only the next NER engineering "
            "stage; it does not replace the existing TPD-v1 mainline.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_report(
    report: Mapping[str, Any], output_dir: Path, *, overwrite: bool
) -> tuple[Path, Path]:
    json_path = output_dir / JSON_OUTPUT_NAME
    markdown_path = output_dir / MARKDOWN_OUTPUT_NAME
    if not overwrite:
        existing = [path for path in (json_path, markdown_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "output exists; pass --overwrite: "
                + ", ".join(str(path) for path in existing)
            )
    _atomic_write(
        json_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(markdown_path, render_markdown(report))
    return json_path, markdown_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit TPD-Clean-v5 screen800 and evaluate Gates A--E"
    )
    parser.add_argument(
        "--candidate-root", type=Path, default=DEFAULT_CANDIDATE_ROOT
    )
    parser.add_argument(
        "--formal-reference-root",
        type=Path,
        default=DEFAULT_FORMAL_REFERENCE_ROOT,
    )
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument(
        "--training-source-lock",
        type=Path,
        default=DEFAULT_TRAINING_SOURCE_LOCK,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(
            args.candidate_root,
            args.formal_reference_root,
            args.smoke_root,
            args.training_source_lock,
        )
    except IncompleteArtifact as exc:
        report = {
            "schema": SCHEMA,
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "incomplete",
            "scope": {
                "dataset": DATASET,
                "candidate_variants": list(VARIANTS),
                "model_seeds": list(SEEDS),
                "official_test_accessed": False,
            },
            "candidate_runs": {},
            "incomplete_reasons": [str(exc)],
            "gate_evaluated": False,
            "engineering_gate_passed": None,
            "mainline_changed": False,
            "paper_core_established": False,
            "stability_claim_supported": False,
            "ner_stage_authorized": False,
            "decision": "INCOMPLETE",
        }
    json_path, markdown_path = write_report(
        report, args.output_dir, overwrite=args.overwrite
    )
    print(
        "TPDCLEANV5_SUMMARY"
        f" status={report['status']}"
        f" gate={report['engineering_gate_passed']}"
        f" json={json_path}"
        f" markdown={markdown_path}",
        flush=True,
    )
    return 2 if args.require_complete and report["status"] != "complete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
