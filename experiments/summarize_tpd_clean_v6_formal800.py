#!/usr/bin/env python3
"""Validate V6 formal800 artifacts and evaluate protocol Gates A--E.

This module is deliberately independent from the frozen training entry.  It
does not train, repair, overwrite, or infer missing results.  Formal gates are
evaluated only after all four 800-epoch runs and all eight closed-interval
sweeps pass the checks below.  ``--preflight`` is read-only and is the only CLI
mode permitted while the four-run matrix is incomplete.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA = "sctransnet_tpd_clean_v6_formal800_comparison_v1"
DATASET = "NUDT-SIRST"
VARIANTS = ("tpd_clean_v6_full", "tpd_clean_v6_phase_capacity")
PRIMARY_VARIANT = VARIANTS[0]
CONTROL_VARIANT = VARIANTS[1]
SEEDS = (42, 3407)
RUN_TAG = "formal800_exact_fp32_2x5090_v1"
EXPECTED_EPOCHS = 800
EXPECTED_TRAIN_COUNT = 530
EXPECTED_VAL_COUNT = 133
EXPECTED_TARGET_COUNT = 189
EXPECTED_SPLIT_SEED = 20260722
EXPECTED_TRAINING_LOCK_SHA256 = (
    "2de1a8f75deb321b5aec4cf5dfa6bc16df8443e858e1d48a3ab6bea34de526d2"
)
EXPECTED_TRAINING_DATA_SHA256 = (
    "39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
)
LAST_FLOAT32_BELOW_ONE = 0.9999999403953552
BUDGET_KEYS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")
ROLE_SPECS: Mapping[str, Mapping[str, str]] = {
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
INTEGRITY_KEYS = (
    "four_runs_contiguous_800_epochs",
    "twelve_checkpoints_present_and_strict_load",
    "eight_closed_interval_sweeps",
    "model_split_protocol_evaluator_hashes_consistent",
    "cpu_and_rtx5090_smoke_passed",
    "fixed_threshold_reproduction_exact",
    "all_five_budgets_available",
    "preregistered_endpoint_provenance",
    "exact_epoch_journals_complete",
    "worker_logs_complete_gpu_mapped",
)

DEFAULT_CANDIDATE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_clean_v6_formal800_2x5090_v1"
)
DEFAULT_REFERENCE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_pe_formal800_4x5090_v1"
)
DEFAULT_SMOKE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_clean_v6_preflight_v1/smoke_reports"
)
DEFAULT_TRAINING_SOURCE_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v6_exact_source_lock.json"
)
DEFAULT_POSTPROCESS_SOURCE_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v6_postprocess_source_lock.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CANDIDATE_ROOT / DATASET / "comparison"
JSON_OUTPUT_NAME = "tpd_clean_v6_formal800_comparison.json"
MARKDOWN_OUTPUT_NAME = "tpd_clean_v6_formal800_comparison.md"
SPD_RUN = (
    DEFAULT_REFERENCE_ROOT
    / DATASET
    / "spd"
    / "seed_42_formal800_pd_fp32_4x5090_v1"
)
SPD_SWEEP = SPD_RUN / "pd_fa_sweep_best.pth.json"
SPD_SPLIT = SPD_RUN / "split.json"
SPD_REFERENCE_FILES = (
    SPD_RUN / "best.pth.tar",
    SPD_SWEEP,
    SPD_RUN / "protocol.json",
    SPD_SPLIT,
    SPD_RUN / "summary.json",
    SPD_RUN / "metrics.jsonl",
)
SMOKE_VERIFICATION = (
    REPO_ROOT
    / "experiments/results/tpd_clean_v6_preflight_v1/v6_smoke_verification.json"
)
GPU_ASSIGNMENTS = {
    (PRIMARY_VARIANT, 42): (
        "2",
        "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    ),
    (CONTROL_VARIANT, 42): (
        "3",
        "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
    ),
    (PRIMARY_VARIANT, 3407): (
        "3",
        "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
    ),
    (CONTROL_VARIANT, 3407): (
        "2",
        "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    ),
}
POSTPROCESS_SOURCE_RELATIVES = frozenset(
    {
        "experiments/summarize_tpd_clean_v6_formal800.py",
        "experiments/run_tpd_clean_v6_formal800_sweeps.py",
        "experiments/validate_tpd_clean_v6_formal800_completion.py",
        "experiments/freeze_tpd_clean_v6_postprocess_source_lock.py",
        "experiments/run_tpd_clean_v6_formal800_finalizer.sh",
        "experiments/launch_tpd_clean_v6_formal800_finalizer.sh",
        "experiments/status_tpd_clean_v6_formal800_finalizer.sh",
        "tests/test_summarize_tpd_clean_v6_formal800.py",
        "tests/test_run_tpd_clean_v6_formal800_sweeps.py",
        "tests/test_validate_tpd_clean_v6_formal800_completion.py",
        "tests/test_freeze_tpd_clean_v6_postprocess_source_lock.py",
        "tests/test_tpd_clean_v6_formal800_finalizer.py",
    }
)
FROZEN_REFERENCE_RELATIVES = frozenset(
    {
        *{str(path.relative_to(REPO_ROOT)) for path in SPD_REFERENCE_FILES},
        *{
            str((DEFAULT_SMOKE_ROOT / name).relative_to(REPO_ROOT))
            for name in ("cpu_all.json", "gpu2_full.json", "gpu3_capacity.json")
        },
        str(SMOKE_VERIFICATION.relative_to(REPO_ROOT)),
    }
)


class IncompleteArtifact(ValueError):
    """A required formal artifact is absent, unfinished, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IncompleteArtifact(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> None:
    _require(
        path.is_file() and not path.is_symlink(),
        f"{label}: missing, linked, or non-regular file: {path}",
    )


def _finite_tree(value: Any, label: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), f"{label}: non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _finite_tree(item, f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{label}[{index}]")
        return
    _require(False, f"{label}: non-JSON value")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _regular(path, label)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise IncompleteArtifact(f"{label}: invalid JSON: {exc}") from exc
    _require(isinstance(payload, dict), f"{label}: expected JSON object")
    _finite_tree(payload, label)
    return payload


def _run_directory(candidate_root: Path, variant: str, seed: int) -> Path:
    return (
        Path(candidate_root)
        / DATASET
        / variant
        / f"seed_{seed}_{RUN_TAG}"
    )


def inspect_training_readiness(
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
) -> dict[str, Any]:
    """Return a read-only four-run readiness report.

    This intentionally does not load checkpoints or require sweep outputs.  It
    is safe while training is live and never evaluates Gates A--E.
    """

    runs: dict[str, Any] = {}
    ready = True
    for seed in SEEDS:
        for variant in VARIANTS:
            directory = _run_directory(candidate_root, variant, seed)
            key = f"{variant}/seed_{seed}"
            metrics_path = directory / "metrics.jsonl"
            summary_path = directory / "summary.json"
            row_count = 0
            contiguous = False
            if metrics_path.is_file() and not metrics_path.is_symlink():
                lines = metrics_path.read_text(encoding="utf-8").splitlines()
                row_count = len(lines)
                if row_count == EXPECTED_EPOCHS:
                    try:
                        contiguous = all(
                            json.loads(line).get("epoch") == index
                            for index, line in enumerate(lines, start=1)
                        )
                    except (json.JSONDecodeError, AttributeError):
                        contiguous = False
            summary_complete = False
            if summary_path.is_file() and not summary_path.is_symlink():
                try:
                    summary_complete = (
                        json.loads(summary_path.read_text(encoding="utf-8")).get(
                            "status"
                        )
                        == "complete"
                    )
                except (json.JSONDecodeError, AttributeError):
                    summary_complete = False
            checkpoints = {
                name: (directory / name).is_file()
                and not (directory / name).is_symlink()
                for name in ("best.pth.tar", "best_miou.pth.tar", "last.pth.tar")
            }
            run_ready = (
                directory.is_dir()
                and row_count == EXPECTED_EPOCHS
                and contiguous
                and summary_complete
                and all(checkpoints.values())
            )
            ready = ready and run_ready
            runs[key] = {
                "run_directory": str(directory.resolve()),
                "metrics_rows": row_count,
                "metrics_contiguous_1_to_800": contiguous,
                "summary_complete": summary_complete,
                "checkpoints_present": checkpoints,
                "ready_for_sweep": run_ready,
            }
    return {
        "schema": "sctransnet_tpd_clean_v6_postprocess_preflight_v1",
        "mode": "preflight",
        "candidate_root": str(Path(candidate_root).resolve()),
        "formal_matrix_complete": ready,
        "gate_evaluated": False,
        "engineering_gate_passed": None,
        "runs": runs,
    }


def _validate_point(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label}: point must be an object")
    point = dict(value)
    for key in ("matched_target_count", "target_count"):
        observed = point.get(key)
        _require(
            isinstance(observed, int) and not isinstance(observed, bool),
            f"{label}: {key} must be an integer",
        )
    _require(
        point["target_count"] == EXPECTED_TARGET_COUNT,
        f"{label}: target_count must be {EXPECTED_TARGET_COUNT}",
    )
    _require(
        0 <= point["matched_target_count"] <= EXPECTED_TARGET_COUNT,
        f"{label}: invalid matched_target_count",
    )
    for key in ("pd", "tiny_pd", "fa", "miou", "threshold"):
        observed = point.get(key)
        _require(
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and math.isfinite(float(observed)),
            f"{label}: {key} must be finite numeric",
        )
    _require(0.0 <= float(point["pd"]) <= 1.0, f"{label}: Pd range")
    _require(
        0.0 <= float(point["tiny_pd"]) <= 1.0,
        f"{label}: tiny-Pd range",
    )
    _require(float(point["fa"]) >= 0.0, f"{label}: negative Fa")
    _require(0.0 <= float(point["miou"]) <= 1.0, f"{label}: mIoU range")
    _require(
        0.0 <= float(point["threshold"]) <= 1.0,
        f"{label}: threshold range",
    )
    expected_pd = point["matched_target_count"] / EXPECTED_TARGET_COUNT
    _require(
        math.isclose(
            float(point["pd"]), expected_pd, rel_tol=0.0, abs_tol=1e-15
        ),
        f"{label}: Pd/count mismatch",
    )
    for key in (
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
    ):
        observed = point.get(key)
        _require(
            isinstance(observed, int)
            and not isinstance(observed, bool)
            and observed >= 0,
            f"{label}: {key} must be a non-negative integer",
        )
    _require(
        point["tiny_target_count"] == 39
        and point["matched_tiny_target_count"] <= point["tiny_target_count"],
        f"{label}: tiny-target counts differ",
    )
    return point


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
        "threshold": float(point["threshold"]),
    }


def evaluate_engineering_gates(
    runs: Mapping[tuple[str, int], Mapping[str, Any]],
    spd_reference: Mapping[str, Any],
    engineering_integrity: Mapping[str, bool],
) -> dict[str, Any]:
    """Evaluate the frozen V6 protocol Gates A--E without fallback values."""

    expected_runs = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    _require(set(runs) == expected_runs, "Gate input run matrix differs")
    _require(
        set(engineering_integrity) == set(INTEGRITY_KEYS),
        "Gate E integrity key set differs",
    )
    full42 = runs[(PRIMARY_VARIANT, 42)]
    full3407 = runs[(PRIMARY_VARIANT, 3407)]

    a_pd = full42["roles"]["pd_primary"]["fixed_threshold_0_5"]
    a_miou = full42["roles"]["miou_primary"]["fixed_threshold_0_5"]
    a_checks = {
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
        "passed": all(a_checks.values()),
        "subchecks": a_checks,
        "observed": {
            "pd_primary": _point_digest(a_pd),
            "miou_primary": _point_digest(a_miou),
        },
    }

    full42_budgets = full42["roles"]["pd_primary"]["budgets"]
    spd_budgets = spd_reference["roles"]["pd_primary"]["budgets"]
    floors: dict[str, Any] = {}
    spd_comparisons: dict[str, Any] = {}
    for budget in BUDGET_KEYS:
        point = full42_budgets[budget]
        minimum = 187 if budget == "1e-06" else 188
        floors[budget] = {
            "passed": int(point["matched_target_count"]) >= minimum
            and float(point["fa"]) <= float(budget) + 1e-15,
            "minimum_matched_target_count": minimum,
            "observed": _point_digest(point),
        }
        spd_point = spd_budgets[budget]
        covered = _covers(spd_point, point)
        spd_comparisons[budget] = {
            "full_not_covered_by_spd": not covered,
            "full": _point_digest(point),
            "spd": _point_digest(spd_point),
        }
    b_checks = {
        "all_five_budget_pd_floors": all(
            item["passed"] for item in floors.values()
        ),
        "at_least_one_budget_not_covered_by_frozen_spd": any(
            item["full_not_covered_by_spd"] for item in spd_comparisons.values()
        ),
    }
    gate_b = {
        "passed": all(b_checks.values()),
        "subchecks": b_checks,
        "budget_floors": floors,
        "frozen_spd_comparisons": spd_comparisons,
    }

    c_pd = full3407["roles"]["pd_primary"]["fixed_threshold_0_5"]
    c_miou = full3407["roles"]["miou_primary"]["fixed_threshold_0_5"]
    stability: dict[str, Any] = {}
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
        stability[budget] = {
            "passed": passed,
            "seed42_matched_target_count": count42,
            "seed3407_matched_target_count": count3407,
            "minimum_seed3407": count42 - 1,
        }
    c_checks = {
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
        "passed": all(c_checks.values()),
        "subchecks": c_checks,
        "budget_stability": stability,
        "budget_stability_pass_count": within_one_count,
        "observed": {
            "pd_primary": _point_digest(c_pd),
            "miou_primary": _point_digest(c_miou),
        },
    }

    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        full = runs[(PRIMARY_VARIANT, seed)]
        capacity = runs[(CONTROL_VARIANT, seed)]
        comparisons: dict[str, Any] = {}
        capacity_dominance: list[str] = []
        full_budget_advantages: list[str] = []
        full_nonempty_advantages: list[str] = []
        for role_name in ROLE_SPECS:
            full_role = full["roles"][role_name]
            capacity_role = capacity["roles"][role_name]
            fixed_label = f"{role_name}.fixed_threshold_0_5"
            capacity_strict = _dominates(
                capacity_role["fixed_threshold_0_5"],
                full_role["fixed_threshold_0_5"],
            )
            full_fixed_strict = _dominates(
                full_role["fixed_threshold_0_5"],
                capacity_role["fixed_threshold_0_5"],
            )
            if capacity_strict:
                capacity_dominance.append(fixed_label)
            comparisons[fixed_label] = {
                "capacity_strictly_covers_full": capacity_strict,
                "full_strictly_covers_capacity": full_fixed_strict,
                "full": _point_digest(full_role["fixed_threshold_0_5"]),
                "capacity": _point_digest(
                    capacity_role["fixed_threshold_0_5"]
                ),
            }
            for budget in BUDGET_KEYS:
                label = f"{role_name}.budget.{budget}"
                full_point = full_role["budgets"][budget]
                capacity_point = capacity_role["budgets"][budget]
                capacity_strict = _dominates(capacity_point, full_point)
                full_strict = _dominates(full_point, capacity_point)
                if capacity_strict:
                    capacity_dominance.append(label)
                if full_strict:
                    full_budget_advantages.append(label)
                    if float(full_point["threshold"]) < 1.0:
                        full_nonempty_advantages.append(label)
                comparisons[label] = {
                    "capacity_strictly_covers_full": capacity_strict,
                    "full_strictly_covers_capacity": full_strict,
                    "full_advantage_not_empty_endpoint": (
                        full_strict and float(full_point["threshold"]) < 1.0
                    ),
                    "full": _point_digest(full_point),
                    "capacity": _point_digest(capacity_point),
                }
        d_checks = {
            "capacity_never_strictly_covers_full": not capacity_dominance,
            "full_strict_at_one_or_more_registered_budgets": bool(
                full_budget_advantages
            ),
            "full_advantage_not_only_threshold_1_empty_endpoint": bool(
                full_nonempty_advantages
            ),
        }
        per_seed[str(seed)] = {
            "passed": all(d_checks.values()),
            "subchecks": d_checks,
            "capacity_strict_coverage_points": capacity_dominance,
            "full_strict_budget_advantages": full_budget_advantages,
            "nonempty_full_strict_budget_advantages": full_nonempty_advantages,
            "comparisons": comparisons,
        }
    gate_d = {
        "passed": all(item["passed"] for item in per_seed.values()),
        "per_seed": per_seed,
        "strict_coverage_definition": (
            "Pd no lower, Fa no higher, mIoU no lower, and at least one strict"
        ),
    }

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
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "protocol": "experiments/TPD_CLEAN_V6_PROTOCOL.md section 6",
    }


def _read_metrics(path: Path, variant: str) -> list[dict[str, Any]]:
    _regular(path, f"{variant} metrics")
    lines = path.read_text(encoding="utf-8").splitlines()
    _require(
        len(lines) == EXPECTED_EPOCHS,
        f"{variant}: metrics rows={len(lines)}, expected {EXPECTED_EPOCHS}",
    )
    rows: list[dict[str, Any]] = []
    for epoch, raw in enumerate(lines, start=1):
        _require(bool(raw.strip()), f"{variant}: blank metrics row {epoch}")
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise IncompleteArtifact(
                f"{variant}: invalid metrics row {epoch}: {exc}"
            ) from exc
        _require(isinstance(row, dict), f"{variant}: metrics row {epoch}")
        _finite_tree(row, f"{variant}.metrics[{epoch}]")
        _require(row.get("epoch") == epoch, f"{variant}: noncontiguous metrics")
        _require(row.get("variant") == variant, f"{variant}: row variant")
        _require(
            row.get("processed_train_samples") == EXPECTED_TRAIN_COUNT,
            f"{variant}: epoch {epoch} processed_train_samples differs",
        )
        rows.append(row)
    return rows


def _selection_key_pd(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["pd"]),
        -float(row["fa"]),
        float(row["tiny_pd"]),
        float(row["miou"]),
        -float(row["val_loss"]),
    )


def _selection_key_miou(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["miou"]),
        float(row["pd"]),
        -float(row["fa"]),
        float(row["tiny_pd"]),
        -float(row["val_loss"]),
    )


def _validate_closed_interval(
    points: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> None:
    thresholds = [float(point["threshold"]) for point in points]
    _require(
        thresholds == sorted(set(thresholds)),
        "sweep thresholds must be sorted and unique",
    )
    _require(0.5 in thresholds, "sweep threshold 0.5 missing")
    _require(
        LAST_FLOAT32_BELOW_ONE in thresholds and thresholds[-1] == 1.0,
        "closed interval endpoints missing",
    )
    # The frozen evaluator computes empirical quantiles in float64.  Legal
    # interpolated thresholds can therefore lie between the last float32 value
    # below one and 1.0; membership, not penultimate position, is the protocol.
    endpoint = points[-1]
    _require(
        endpoint["matched_target_count"] == 0
        and float(endpoint["pd"]) == 0.0
        and float(endpoint["fa"]) == 0.0,
        "threshold 1.0 is not the empty-prediction endpoint",
    )
    _require(
        float(endpoint["miou"]) == 0.0
        and endpoint["predicted_object_count"] == 0
        and endpoint["unmatched_predicted_object_count"] == 0,
        "threshold 1.0 has a non-empty segmentation/object result",
    )
    expected_provenance = {
        "posthoc_endpoint_completion": False,
        "preregistered_endpoint_completion": True,
        "endpoint_protocol_stage": "before_formal_training",
        "closed_probability_interval": True,
        "score_dtype": "float32",
        "last_float32_below_one": LAST_FLOAT32_BELOW_ONE,
        "upper_boundary_threshold": 1.0,
        "upper_boundary_comparison": "prediction > threshold",
        "upper_boundary_semantics": "empty_prediction_pd0_fa0",
    }
    for key, expected in expected_provenance.items():
        _require(
            provenance.get(key) == expected,
            f"sweep endpoint provenance differs: {key}",
        )


def _validate_sweep(
    path: Path,
    *,
    run_dir: Path,
    variant: str,
    seed: int,
    role_name: str,
    checkpoint_sha256: str,
    checkpoint_epoch: int,
    summary_metrics: Mapping[str, Any],
    evaluator_sha256: str,
    validation_split_sha256: str,
) -> dict[str, Any]:
    spec = ROLE_SPECS[role_name]
    payload = _load_json(path, f"{variant}/seed={seed}/{role_name} sweep")
    _require(payload.get("variant") == variant, "sweep variant differs")
    _require(payload.get("seed") == seed, "sweep seed differs")
    _require(payload.get("dataset") == DATASET, "sweep dataset differs")
    _require(payload.get("official_test_accessed") is False, "sweep test access")
    _require(payload.get("checkpoint_role") == spec["checkpoint_role"], "role")
    _require(payload.get("checkpoint_epoch") == checkpoint_epoch, "epoch")
    _require(payload.get("checkpoint_sha256") == checkpoint_sha256, "ckpt hash")
    _require(
        Path(str(payload.get("checkpoint", ""))).resolve()
        == (run_dir / spec["checkpoint"]).resolve(),
        "sweep checkpoint path differs",
    )
    _require(
        payload.get("checkpoint_validation_metrics") == dict(summary_metrics),
        "sweep checkpoint metrics differ",
    )
    _require(
        payload.get("validation_split_sha256") == validation_split_sha256,
        "sweep validation split differs",
    )
    points_raw = payload.get("points")
    _require(isinstance(points_raw, list) and points_raw, "sweep points missing")
    points = [
        _validate_point(point, f"{variant}/seed={seed}/{role_name}/point[{index}]")
        for index, point in enumerate(points_raw)
    ]
    provenance = payload.get("threshold_provenance")
    _require(isinstance(provenance, dict), "threshold provenance missing")
    _validate_closed_interval(points, provenance)
    threshold_configuration = payload.get("threshold_configuration")
    _require(
        isinstance(threshold_configuration, dict)
        and [
            f"{float(value):.10g}"
            for value in threshold_configuration.get("fa_budgets", [])
        ]
        == list(BUDGET_KEYS),
        "sweep registered Fa budgets differ",
    )
    fixed = _validate_point(payload.get("fixed_threshold_0_5"), "fixed 0.5")
    _require(float(fixed["threshold"]) == 0.5, "fixed threshold differs")
    _require(fixed in points, "fixed threshold point absent")
    for key, expected in summary_metrics.items():
        if key in fixed:
            _require(fixed[key] == expected, f"fixed metric differs: {key}")
    fixed_audit = payload.get("fixed_threshold_0_5_checkpoint_audit")
    _require(
        isinstance(fixed_audit, dict)
        and float(
            fixed_audit.get("max_abs_non_strict_numeric_delta", math.inf)
        )
        == 0.0,
        "fixed threshold reproduction is not exact",
    )
    budgets_raw = payload.get("best_points_under_fa_budget")
    _require(
        isinstance(budgets_raw, dict)
        and set(budgets_raw) == set(BUDGET_KEYS),
        "sweep budget keys differ",
    )
    budgets: dict[str, dict[str, Any]] = {}
    for key in BUDGET_KEYS:
        _require(budgets_raw[key] is not None, f"null budget point: {key}")
        point = _validate_point(budgets_raw[key], f"budget {key}")
        optimum = _best_under_budget(points, float(key))
        _require(
            optimum is not None and point == optimum,
            f"budget {key} is not the registered optimum",
        )
        budgets[key] = point
    audit = payload.get("audit")
    _require(isinstance(audit, dict), "sweep audit missing")
    _require(
        audit.get("expected_epochs") == EXPECTED_EPOCHS
        and audit.get("metrics_event_count") == EXPECTED_EPOCHS
        and audit.get("metrics_epoch_range") == [1, EXPECTED_EPOCHS]
        and audit.get("summary_status") == "complete"
        and audit.get("selection_source") == "internal_validation_only",
        "sweep completeness audit differs",
    )
    integrity = audit.get("integrity_checks_passed")
    required_sweep_integrity = {
        "summary_complete",
        "metrics_complete_contiguous_finite",
        "metadata_consistent",
        "official_test_isolated",
        "split_hashes_recomputed_consistent",
        "checkpoint_role_epoch_metrics_consistent",
        "global_selection_keys_recomputed",
        "state_dict_strict_load",
        "fixed_threshold_object_metrics_exact",
    }
    _require(
        isinstance(integrity, dict)
        and set(integrity) == required_sweep_integrity
        and all(value is True for value in integrity.values()),
        "sweep integrity audit failed",
    )
    artifact_hashes = audit.get("artifact_sha256")
    expected_artifacts = {
        "protocol.json": run_dir / "protocol.json",
        "split.json": run_dir / "split.json",
        "summary.json": run_dir / "summary.json",
        "metrics.jsonl": run_dir / "metrics.jsonl",
        "checkpoint": run_dir / spec["checkpoint"],
    }
    _require(
        isinstance(artifact_hashes, dict)
        and set(artifact_hashes)
        == {*expected_artifacts, "evaluator"},
        "sweep artifact hash set differs",
    )
    for name, artifact in expected_artifacts.items():
        _require(
            artifact_hashes[name] == sha256_file(artifact),
            f"sweep artifact digest differs: {name}",
        )
    _require(
        artifact_hashes["evaluator"] == evaluator_sha256,
        "sweep evaluator digest differs",
    )
    return {
        "checkpoint": str((run_dir / spec["checkpoint"]).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint_epoch,
        "sweep": str(path.resolve()),
        "sweep_sha256": sha256_file(path),
        "evaluator_sha256": evaluator_sha256,
        "fixed_threshold_0_5": fixed,
        "budgets": budgets,
        "closed_interval": True,
        "preregistered_endpoint_provenance": True,
        "fixed_threshold_reproduction_exact": True,
    }


def validate_existing_sweep(
    run_dir: Path,
    *,
    variant: str,
    seed: int,
    role_name: str,
    evaluator_sha256: str,
) -> dict[str, Any]:
    """Strictly validate one already-created sweep without replacing it."""

    _require(role_name in ROLE_SPECS, "unknown sweep checkpoint role")
    run_dir = Path(run_dir)
    protocol = _load_json(run_dir / "protocol.json", "existing sweep protocol")
    split = _load_json(run_dir / "split.json", "existing sweep split")
    summary_payload = _load_json(
        run_dir / "summary.json", "existing sweep summary"
    )
    args = protocol.get("arguments")
    _require(
        isinstance(args, dict)
        and args.get("variant") == variant
        and args.get("seed") == seed
        and args.get("dataset") == DATASET
        and args.get("epochs") == EXPECTED_EPOCHS
        and summary_payload.get("status") == "complete",
        "existing sweep run identity/completeness differs",
    )
    spec = ROLE_SPECS[role_name]
    checkpoint_path = run_dir / spec["checkpoint"]
    _regular(checkpoint_path, "existing sweep checkpoint")
    split_hashes = split.get("hashes")
    _require(isinstance(split_hashes, dict), "existing sweep split hashes")
    return _validate_sweep(
        run_dir / spec["sweep"],
        run_dir=run_dir,
        variant=variant,
        seed=seed,
        role_name=role_name,
        checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_epoch=int(summary_payload[spec["summary_epoch"]]),
        summary_metrics=summary_payload[spec["summary_metrics"]],
        evaluator_sha256=evaluator_sha256,
        validation_split_sha256=split_hashes["used_val_sha256"],
    )


def _validate_current_training_contract() -> tuple[dict[str, Any], str]:
    from experiments import train_tpd_clean_v6_exact as exact

    lock = _load_json(DEFAULT_TRAINING_SOURCE_LOCK, "V6 training source lock")
    digest = sha256_file(DEFAULT_TRAINING_SOURCE_LOCK)
    _require(
        digest == EXPECTED_TRAINING_LOCK_SHA256,
        "V6 training source lock SHA-256 differs",
    )
    _require(
        lock.get("training_data_sha256") == EXPECTED_TRAINING_DATA_SHA256,
        "training lock data digest differs",
    )
    exact.source_lock_contract(
        EXPECTED_TRAINING_DATA_SHA256,
        DEFAULT_TRAINING_SOURCE_LOCK,
    )
    dataset_root = REPO_ROOT / "datasets" / DATASET
    index_bytes, identifiers = exact.read_official_training_index(
        dataset_root, DATASET
    )
    actual_data = exact.official_training_data_sha256(
        dataset_root, DATASET, identifiers, index_bytes
    )
    _require(
        actual_data == EXPECTED_TRAINING_DATA_SHA256,
        "current training data differs from the frozen digest",
    )
    return lock, digest


def validate_postprocess_source_lock(
    path: Path = DEFAULT_POSTPROCESS_SOURCE_LOCK,
) -> tuple[dict[str, Any], str]:
    payload = _load_json(path, "V6 postprocess source lock")
    _require(
        payload.get("schema")
        == "sctransnet_tpd_clean_v6_postprocess_source_lock_v1",
        "postprocess source-lock schema differs",
    )
    _require(
        payload.get("training_source_lock_sha256")
        == EXPECTED_TRAINING_LOCK_SHA256,
        "postprocess lock does not bind the V6 training lock",
    )
    _require(
        payload.get("training_data_sha256") == EXPECTED_TRAINING_DATA_SHA256,
        "postprocess lock does not bind the training data",
    )
    training_lock = _load_json(
        DEFAULT_TRAINING_SOURCE_LOCK, "V6 training source lock binding"
    )
    _require(
        payload.get("training_source_lock")
        == str(DEFAULT_TRAINING_SOURCE_LOCK.relative_to(REPO_ROOT))
        and payload.get("candidate_root")
        == str(DEFAULT_CANDIDATE_ROOT.relative_to(REPO_ROOT))
        and payload.get("evaluator_sha256")
        == training_lock["source_sha256"][
            "experiments/evaluate_tpd_clean_v6_pd_fa.py"
        ],
        "postprocess lock fixed path/evaluator binding differs",
    )
    sources = payload.get("source_sha256")
    _require(
        isinstance(sources, dict)
        and set(sources) == set(POSTPROCESS_SOURCE_RELATIVES),
        "postprocess source path set differs",
    )
    for relative, expected in sources.items():
        source = (REPO_ROOT / relative).resolve()
        try:
            canonical = str(source.relative_to(REPO_ROOT))
        except ValueError as exc:
            raise IncompleteArtifact(
                f"postprocess source escapes repository: {relative}"
            ) from exc
        _require(canonical == relative, "noncanonical postprocess source path")
        _regular(source, f"postprocess source {relative}")
        _require(
            sha256_file(source) == expected,
            f"postprocess source digest differs: {relative}",
        )
    references = payload.get("frozen_reference_sha256")
    _require(
        isinstance(references, dict)
        and set(references) == set(FROZEN_REFERENCE_RELATIVES),
        "frozen reference path set differs",
    )
    for relative, expected in references.items():
        reference = (REPO_ROOT / relative).resolve()
        _regular(reference, f"frozen reference {relative}")
        _require(
            sha256_file(reference) == expected,
            f"frozen reference digest differs: {relative}",
        )
    policy = payload.get("policy")
    _require(
        isinstance(policy, dict)
        and policy.get("separate_from_training_source_lock") is True
        and policy.get("does_not_modify_frozen_training_sources") is True
        and policy.get("does_not_modify_training_results") is True
        and policy.get("formal_report_overwrite_forbidden") is True
        and policy.get(
            "gate_evaluation_before_four_complete_runs_forbidden"
        )
        is True
        and policy.get("candidate_null_budget_points_forbidden") is True
        and policy.get("automatic_mainline_replacement") is False,
        "postprocess source-lock policy differs",
    )
    return payload, sha256_file(path)


def _require_finite_state(value: Any, label: str) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            _require(
                bool(torch.isfinite(value).all().item()),
                f"{label}: non-finite tensor",
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_state(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_state(item, f"{label}[{index}]")
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label}: non-finite scalar")


def _strict_checkpoint_load(
    path: Path,
    variant: str,
    seed: int,
    expected_role: str,
    *,
    model: Any,
    protocol_identity: Mapping[str, Any],
    protocol_model: Mapping[str, Any],
    split_hashes: Mapping[str, str],
) -> dict[str, Any]:
    import torch

    from experiments import tpd_exact_runner as exact_runner

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, dict), f"{path.name}: invalid checkpoint")
    expected_metadata = {
        "derived_schema": exact_runner.DERIVED_CHECKPOINT_SCHEMA,
        "variant": variant,
        "dataset": DATASET,
        "seed": seed,
        "split_seed": EXPECTED_SPLIT_SEED,
        "checkpoint_role": expected_role,
        "official_test_accessed": False,
        "selection_source": "internal_validation_only",
    }
    for key, expected in expected_metadata.items():
        _require(
            checkpoint.get(key) == expected,
            f"{path.name}: metadata differs: {key}",
        )
    _require(
        checkpoint.get("run_identity") == dict(protocol_identity),
        f"{path.name}: run identity",
    )
    _require(
        checkpoint.get("split_hashes") == dict(split_hashes),
        f"{path.name}: split hashes",
    )
    _require(
        json.loads(json.dumps(checkpoint.get("model_metadata")))
        == dict(protocol_model),
        f"{path.name}: model metadata",
    )
    state_dict = checkpoint.get("state_dict")
    optimizer = checkpoint.get("optimizer")
    scaler = checkpoint.get("scaler")
    _require(isinstance(state_dict, dict), f"{path.name}: state_dict")
    _require(isinstance(optimizer, dict), f"{path.name}: optimizer")
    _require(isinstance(scaler, dict), f"{path.name}: scaler")
    _require_finite_state(state_dict, f"{path.name}.state_dict")
    _require_finite_state(optimizer, f"{path.name}.optimizer")
    _require_finite_state(scaler, f"{path.name}.scaler")
    model.load_state_dict(state_dict, strict=True)
    digests = {
        "state_dict_sha256": exact_runner._state_content_sha256(
            state_dict, f"{path.name} state"
        ),
        "optimizer_state_sha256": exact_runner._state_content_sha256(
            optimizer, f"{path.name} optimizer"
        ),
        "scaler_state_sha256": exact_runner._state_content_sha256(
            scaler, f"{path.name} scaler"
        ),
    }
    for key, expected in digests.items():
        _require(checkpoint.get(key) == expected, f"{path.name}: {key}")
    source_sha = checkpoint.get("source_exact_checkpoint_sha256")
    _require(
        isinstance(source_sha, str)
        and len(source_sha) == 64
        and all(character in "0123456789abcdef" for character in source_sha),
        f"{path.name}: source exact checkpoint digest",
    )
    return checkpoint


def _validate_rng_state_offline(value: Any) -> dict[str, Any]:
    import random

    import numpy as np
    import torch

    from experiments import tpd_exact_resume as exact

    required = {
        "schema",
        "python_random",
        "numpy_random",
        "torch_cpu",
        "torch_cuda_available",
        "torch_cuda_device_count",
        "torch_cuda",
        "loader_generator_device",
        "loader_generator",
    }
    _require(
        isinstance(value, Mapping) and set(value) == required,
        "exact RNG state key set differs",
    )
    _require(value.get("schema") == exact.RNG_STATE_SCHEMA, "exact RNG schema")
    try:
        random.Random().setstate(value["python_random"])
        np.random.RandomState().set_state(value["numpy_random"])
    except (TypeError, ValueError) as exc:
        raise IncompleteArtifact(f"exact RNG state is invalid: {exc}") from exc
    for key in ("torch_cpu", "loader_generator"):
        tensor = value[key]
        _require(
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.uint8
            and tensor.ndim == 1
            and tensor.numel() > 0,
            f"exact RNG {key} differs",
        )
        torch.Generator(device="cpu").set_state(tensor)
    cuda_states = value["torch_cuda"]
    _require(
        value["torch_cuda_available"] is True
        and value["torch_cuda_device_count"] == 1
        and isinstance(cuda_states, (list, tuple))
        and len(cuda_states) == 1,
        "exact CUDA RNG capture differs",
    )
    _require(
        isinstance(cuda_states[0], torch.Tensor)
        and cuda_states[0].device.type == "cpu"
        and cuda_states[0].dtype == torch.uint8
        and cuda_states[0].ndim == 1
        and cuda_states[0].numel() > 0,
        "exact CUDA RNG tensor differs",
    )
    _require(
        value["loader_generator_device"] == "cpu",
        "exact DataLoader RNG device differs",
    )
    return {
        "schema": value["schema"],
        "cuda_device_count": value["torch_cuda_device_count"],
        "loader_generator_device": value["loader_generator_device"],
    }


def _validate_exact_journal(
    run_dir: Path,
    *,
    protocol_identity: Mapping[str, Any],
    model: Any,
    last_checkpoint: Mapping[str, Any],
    last_event: Mapping[str, Any],
    summary_payload: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    from experiments import tpd_exact_resume as exact
    from experiments import tpd_exact_runner as exact_runner
    from experiments.tpd_exact_epoch_journal import ExactEpochJournal

    state = ExactEpochJournal(run_dir / "exact_journal").load_active()
    _require(state is not None, "exact journal has no active state")
    _require(state.epoch == EXPECTED_EPOCHS, "exact journal epoch is not 800")
    _require(
        state.metrics_path.read_bytes() == (run_dir / "metrics.jsonl").read_bytes(),
        "active journal metrics differ from published metrics",
    )
    root = run_dir / "exact_journal"
    slot_records: dict[str, Any] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for slot in ("slot_a", "slot_b"):
        metrics_path = root / f"{slot}.metrics.jsonl"
        checkpoint_path = root / f"{slot}.exact.pth"
        _regular(metrics_path, f"exact journal {slot} metrics")
        _regular(checkpoint_path, f"exact journal {slot} checkpoint")
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
        try:
            rows = [json.loads(line) for line in lines]
        except json.JSONDecodeError as exc:
            raise IncompleteArtifact(
                f"exact journal {slot} metrics invalid: {exc}"
            ) from exc
        slot_epoch = len(rows)
        _require(
            slot_epoch > 0
            and [row.get("epoch") for row in rows]
            == list(range(1, slot_epoch + 1)),
            f"exact journal {slot} metrics are not contiguous",
        )
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        _require(
            isinstance(payload, dict)
            and set(payload) == set(exact.EXACT_RESUME_REQUIRED_KEYS)
            and payload.get("schema") == exact.EXACT_RESUME_SCHEMA
            and payload.get("mode") == exact.EXACT_RESUME_MODE
            and payload.get("epoch") == slot_epoch,
            f"exact journal {slot} payload contract differs",
        )
        _require(
            payload.get("run_identity") == dict(protocol_identity),
            f"exact journal {slot} run identity differs",
        )
        boundary = exact.metrics_boundary_from_jsonl(
            metrics_path, expected_epoch=slot_epoch
        )
        _require(
            payload.get("metrics_boundary") == boundary,
            f"exact journal {slot} metrics boundary differs",
        )
        _require_finite_state(payload["model"], f"exact journal {slot}.model")
        _require_finite_state(
            payload["optimizer"], f"exact journal {slot}.optimizer"
        )
        _require_finite_state(
            payload["scaler"], f"exact journal {slot}.scaler"
        )
        payloads[slot] = payload
        slot_records[slot] = {
            "epoch": slot_epoch,
            "metrics_file": str(metrics_path.resolve()),
            "metrics_sha256": sha256_file(metrics_path),
            "checkpoint_file": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        }
    _require(
        {record["epoch"] for record in slot_records.values()}
        == {EXPECTED_EPOCHS - 1, EXPECTED_EPOCHS},
        "exact journal slots must bind epochs 799 and 800",
    )
    active = payloads[state.slot]
    _require(active["epoch"] == EXPECTED_EPOCHS, "active exact epoch differs")
    exact_model = active.get("model")
    _require(
        isinstance(exact_model, Mapping)
        and set(exact_model) == {"layout", "state_dict"}
        and exact_model["layout"] == exact.model_layout(model),
        "active exact model component differs",
    )
    model.load_state_dict(exact_model["state_dict"], strict=True)
    _require(
        exact_runner._state_values_equal(
            exact_model["state_dict"], last_checkpoint["state_dict"]
        ),
        "active exact model differs from last checkpoint",
    )
    optimizer = active.get("optimizer")
    _require(
        isinstance(optimizer, Mapping)
        and set(optimizer) == {"class", "parameter_names", "state_dict"},
        "active exact optimizer component differs",
    )
    parameter_names = optimizer["parameter_names"]
    _require(
        isinstance(parameter_names, list)
        and len(parameter_names) == 1
        and parameter_names[0] == [name for name, _ in model.named_parameters()],
        "active exact optimizer parameter binding differs",
    )
    optimizer_state = optimizer["state_dict"]
    _require(
        isinstance(optimizer_state, Mapping)
        and isinstance(optimizer_state.get("state"), Mapping)
        and bool(optimizer_state["state"])
        and isinstance(optimizer_state.get("param_groups"), list)
        and len(optimizer_state["param_groups"]) == 1,
        "active exact optimizer state differs",
    )
    _require(
        exact_runner._state_values_equal(
            optimizer_state, last_checkpoint["optimizer"]
        ),
        "active exact optimizer differs from last checkpoint",
    )
    scaler = active.get("scaler")
    _require(
        isinstance(scaler, Mapping)
        and set(scaler) == {"class", "state_dict"}
        and exact_runner._state_values_equal(
            scaler["state_dict"], last_checkpoint["scaler"]
        ),
        "active exact scaler differs from last checkpoint",
    )
    _require(active.get("scheduler") is None, "active exact scheduler differs")
    rng = _validate_rng_state_offline(active.get("rng_state"))
    selection = exact._validate_best_selection(
        active.get("best_selection"), EXPECTED_EPOCHS
    )
    expected_selection = {
        "primary": (
            int(summary_payload["best_pd_epoch"]),
            summary_payload["best_pd_validation_metrics"],
        ),
        "secondary": (
            int(summary_payload["best_miou_epoch"]),
            summary_payload["best_miou_validation_metrics"],
        ),
    }
    for name, (epoch, metrics) in expected_selection.items():
        _require(
            selection[name]["epoch"] == epoch
            and all(
                selection[name]["metrics"].get(key) == value
                for key, value in metrics.items()
            ),
            f"active exact {name} selection differs",
        )
    _require(
        last_checkpoint.get("epoch") == EXPECTED_EPOCHS
        and last_checkpoint.get("validation_metrics")
        == {
            key: last_event[key]
            for key in last_checkpoint["validation_metrics"]
        },
        "last checkpoint does not match epoch-800 metrics",
    )
    _require(
        last_checkpoint.get("source_exact_checkpoint_sha256")
        == state.checkpoint_sha256,
        "last checkpoint does not bind active exact checkpoint",
    )
    return {
        "active_epoch": state.epoch,
        "active_slot": state.slot,
        "marker_file": str(state.marker_path.resolve()),
        "marker_sha256": state.marker_sha256,
        "checkpoint_sha256": state.checkpoint_sha256,
        "metrics_sha256": state.metrics_boundary["metrics_sha256"],
        "rng": rng,
        "slots": slot_records,
    }


def _parse_key_values(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value
    return values


def _validate_worker_log(
    candidate_root: Path,
    *,
    variant: str,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    physical_gpu, gpu_uuid = GPU_ASSIGNMENTS[(variant, seed)]
    path = Path(candidate_root) / "logs" / f"{variant}_seed{seed}.log"
    _regular(path, f"{variant}/seed={seed} worker log")
    lines = path.read_text(encoding="utf-8").splitlines()
    _require(
        not any("TPDCLEANV6_2X_ABORT" in line for line in lines),
        f"{variant}/seed={seed}: worker log contains abort",
    )
    starts = [
        _parse_key_values(line)
        for line in lines
        if line.startswith("TPDCLEANV6_2X_START ")
    ]
    _require(starts, f"{variant}/seed={seed}: worker start missing")
    for start in starts:
        _require(
            start.get("variant") == variant
            and start.get("seed") == str(seed)
            and start.get("physical_gpu") == physical_gpu
            and start.get("gpu_uuid") == gpu_uuid
            and start.get("mode") in {"--fresh", "--exact-resume"}
            and Path(start.get("run_dir", "")).resolve() == run_dir.resolve(),
            f"{variant}/seed={seed}: worker start identity differs",
        )
    completion_lines = [
        line
        for line in lines
        if line.startswith("TPDCLEANV6_2X_COMPLETE ")
    ]
    _require(
        len(completion_lines) == 1,
        f"{variant}/seed={seed}: expected exactly one worker completion",
    )
    completion = _parse_key_values(completion_lines[0])
    _require(
        completion
        == {
            "variant": variant,
            "seed": str(seed),
            "physical_gpu": physical_gpu,
            "epochs": str(EXPECTED_EPOCHS),
        },
        f"{variant}/seed={seed}: worker completion identity differs",
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "start_count": len(starts),
        "initialization_modes": [start["mode"] for start in starts],
        "physical_gpu": physical_gpu,
        "gpu_uuid": gpu_uuid,
        "completion_count": 1,
    }


def validate_candidate_run(
    candidate_root: Path,
    variant: str,
    seed: int,
    *,
    training_lock: Mapping[str, Any],
    evaluator_sha256: str,
) -> dict[str, Any]:
    run_dir = _run_directory(candidate_root, variant, seed)
    _require(run_dir.is_dir() and not run_dir.is_symlink(), "run directory")
    protocol = _load_json(run_dir / "protocol.json", "protocol")
    split = _load_json(run_dir / "split.json", "split")
    summary = _load_json(run_dir / "summary.json", "summary")
    rows = _read_metrics(run_dir / "metrics.jsonl", variant)

    args = protocol.get("arguments")
    _require(isinstance(args, dict), "protocol arguments missing")
    expected_args = {
        "variant": variant,
        "seed": seed,
        "dataset": DATASET,
        "epochs": EXPECTED_EPOCHS,
        "eval_every": 1,
        "workers": 0,
        "amp": False,
        "eps": 1e-6,
        "patch_size": 256,
        "batch_size": 16,
        "split_seed": EXPECTED_SPLIT_SEED,
        "run_tag": RUN_TAG,
        "max_train_images": None,
        "max_val_images": None,
    }
    for key, expected in expected_args.items():
        _require(args.get(key) == expected, f"protocol argument differs: {key}")
    _require(
        protocol.get("formal_contract") == training_lock.get("formal_contract"),
        "protocol formal contract differs from training lock",
    )
    _require(
        protocol.get("official_test_accessed") is False,
        "protocol official test isolation differs",
    )
    identity = protocol.get("run_identity")
    _require(isinstance(identity, dict), "run identity missing")
    _require(
        identity.get("run_id")
        == f"tpd-clean-v6-exact:{DATASET}:{variant}:seed-{seed}:{RUN_TAG}"
        and identity.get("variant") == variant
        and identity.get("dataset") == DATASET
        and identity.get("seed") == seed
        and identity.get("split_seed") == EXPECTED_SPLIT_SEED,
        "run identity fields differ",
    )
    source_locks = identity.get("source_locks")
    _require(isinstance(source_locks, dict), "run source locks missing")
    expected_source_locks = {
        "tpd_clean_v6_exact_source_lock": EXPECTED_TRAINING_LOCK_SHA256,
        "training_data": EXPECTED_TRAINING_DATA_SHA256,
        **{
            f"exact_source:{relative}": digest
            for relative, digest in training_lock["source_sha256"].items()
        },
    }
    _require(
        source_locks == expected_source_locks,
        "run identity lock/data binding differs",
    )
    physical_gpu, gpu_uuid = GPU_ASSIGNMENTS[(variant, seed)]
    training_contract = identity.get("training_contract")
    environment = (
        training_contract.get("environment")
        if isinstance(training_contract, Mapping)
        else None
    )
    _require(
        isinstance(environment, Mapping)
        and environment.get("device_type") == "cuda"
        and environment.get("logical_device") == "cuda:0"
        and environment.get("visible_cuda_device_count") == 1
        and environment.get("device_name") == "NVIDIA GeForce RTX 5090"
        and environment.get("device_uuid") == gpu_uuid
        and environment.get("cuda_visible_devices") == gpu_uuid
        and environment.get("deterministic_algorithms") is True
        and environment.get("cudnn_deterministic") is True
        and environment.get("cudnn_benchmark") is False,
        f"run GPU/determinism environment differs for physical GPU {physical_gpu}",
    )
    _require(
        summary.get("status") == "complete"
        and summary.get("variant") == variant
        and summary.get("seed") == seed
        and summary.get("dataset") == DATASET
        and summary.get("official_test_accessed") is False
        and summary.get("selection_source") == "internal_validation_only",
        "summary completion metadata differs",
    )
    _require(
        split.get("dataset") == DATASET
        and split.get("split_seed") == EXPECTED_SPLIT_SEED
        and split.get("used_train_count") == EXPECTED_TRAIN_COUNT
        and split.get("used_val_count") == EXPECTED_VAL_COUNT
        and split.get("full_official_train_count")
        == EXPECTED_TRAIN_COUNT + EXPECTED_VAL_COUNT,
        "split contract differs",
    )
    split_hashes = split.get("hashes")
    _require(
        isinstance(split_hashes, dict)
        and summary.get("split_hashes") == split_hashes,
        "split hashes differ",
    )
    selection_fields = ("pd", "fa", "tiny_pd", "miou", "val_loss")
    _require(
        all(all(field in row for field in selection_fields) for row in rows),
        "metrics validation fields missing",
    )
    pd_row = max(rows, key=_selection_key_pd)
    miou_row = max(rows, key=_selection_key_miou)
    _require(
        summary.get("best_pd_epoch") == pd_row["epoch"]
        and summary.get("best_epoch") == pd_row["epoch"]
        and summary.get("best_miou_epoch") == miou_row["epoch"],
        "summary selection epochs differ",
    )
    pd_summary_metrics = summary.get("best_pd_validation_metrics")
    miou_summary_metrics = summary.get("best_miou_validation_metrics")
    _require(
        isinstance(pd_summary_metrics, dict)
        and summary.get("best_validation_metrics") == pd_summary_metrics
        and all(
            pd_row.get(key) == value
            for key, value in pd_summary_metrics.items()
        ),
        "summary Pd-primary metrics differ from the selected metrics row",
    )
    _require(
        isinstance(miou_summary_metrics, dict)
        and all(
            miou_row.get(key) == value
            for key, value in miou_summary_metrics.items()
        ),
        "summary mIoU-primary metrics differ from the selected metrics row",
    )

    from experiments.train_tpd_clean_v6_exact import build_selected_model

    model_metadata = protocol.get("model")
    _require(isinstance(model_metadata, dict), "protocol model metadata missing")
    candidate_model, rebuilt_metadata = build_selected_model(variant, seed)
    _require(
        json.loads(json.dumps(rebuilt_metadata)) == model_metadata,
        "current strict builder metadata differs from protocol",
    )
    checkpoint_records: dict[str, Any] = {}
    roles: dict[str, Any] = {}
    for role_name, spec in ROLE_SPECS.items():
        checkpoint_path = run_dir / spec["checkpoint"]
        _regular(checkpoint_path, f"{variant}/seed={seed}/{role_name}")
        checkpoint = _strict_checkpoint_load(
            checkpoint_path,
            variant,
            seed,
            spec["checkpoint_role"],
            model=candidate_model,
            protocol_identity=identity,
            protocol_model=model_metadata,
            split_hashes=split_hashes,
        )
        expected_epoch = int(summary[spec["summary_epoch"]])
        expected_metrics = summary[spec["summary_metrics"]]
        _require(checkpoint.get("epoch") == expected_epoch, "checkpoint epoch")
        _require(
            checkpoint.get("validation_metrics") == expected_metrics,
            "checkpoint validation metrics",
        )
        checkpoint_sha = sha256_file(checkpoint_path)
        checkpoint_records[spec["checkpoint"]] = {
            "sha256": checkpoint_sha,
            "epoch": expected_epoch,
            "role": spec["checkpoint_role"],
            "strict_load": True,
        }
        roles[role_name] = _validate_sweep(
            run_dir / spec["sweep"],
            run_dir=run_dir,
            variant=variant,
            seed=seed,
            role_name=role_name,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_epoch=expected_epoch,
            summary_metrics=expected_metrics,
            evaluator_sha256=evaluator_sha256,
            validation_split_sha256=split_hashes["used_val_sha256"],
        )

    last_path = run_dir / "last.pth.tar"
    _regular(last_path, f"{variant}/seed={seed}/last")
    last = _strict_checkpoint_load(
        last_path,
        variant,
        seed,
        "last_evaluated_epoch",
        model=candidate_model,
        protocol_identity=identity,
        protocol_model=model_metadata,
        split_hashes=split_hashes,
    )
    _require(last.get("epoch") == EXPECTED_EPOCHS, "last checkpoint epoch")
    checkpoint_records["last.pth.tar"] = {
        "sha256": sha256_file(last_path),
        "epoch": EXPECTED_EPOCHS,
        "role": "last_evaluated_epoch",
        "strict_load": True,
    }
    journal = _validate_exact_journal(
        run_dir,
        protocol_identity=identity,
        model=candidate_model,
        last_checkpoint=last,
        last_event=rows[-1],
        summary_payload=summary,
    )
    worker_log = _validate_worker_log(
        candidate_root,
        variant=variant,
        seed=seed,
        run_dir=run_dir,
    )
    return {
        "run_directory": str(run_dir.resolve()),
        "variant": variant,
        "seed": seed,
        "metrics_event_count": len(rows),
        "split_hashes": copy.deepcopy(split_hashes),
        "model": {
            "full_initialization_sha256": model_metadata.get(
                "full_initialization_sha256"
            ),
            "shared_initialization_sha256": model_metadata.get(
                "shared_initialization_sha256"
            ),
            "total_parameters": model_metadata.get("total_parameters"),
            "shallow_embedding_parameters": model_metadata.get(
                "shallow_embedding_parameters"
            ),
        },
        "checkpoints": checkpoint_records,
        "roles": roles,
        "exact_journal": journal,
        "worker_log": worker_log,
        "artifact_sha256": {
            **{
                name: sha256_file(run_dir / name)
                for name in (
                    "protocol.json",
                    "split.json",
                    "summary.json",
                    "metrics.jsonl",
                )
            },
            **{
                f"exact_journal/{name}": sha256_file(
                    run_dir / "exact_journal" / name
                )
                for name in (
                    "active.json",
                    "slot_a.metrics.jsonl",
                    "slot_a.exact.pth",
                    "slot_b.metrics.jsonl",
                    "slot_b.exact.pth",
                )
            },
            "worker_log": worker_log["sha256"],
        },
    }


def load_spd_reference(
    expected_validation_sha256: str,
    *,
    sweep_path: Path = SPD_SWEEP,
    split_path: Path = SPD_SPLIT,
) -> dict[str, Any]:
    import torch

    from experiments.train_tpd_pilot import build_model

    run_dir = sweep_path.parent
    checkpoint_path = run_dir / "best.pth.tar"
    protocol_path = run_dir / "protocol.json"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    for path, label in (
        (checkpoint_path, "frozen SPD checkpoint"),
        (protocol_path, "frozen SPD protocol"),
        (summary_path, "frozen SPD summary"),
        (metrics_path, "frozen SPD metrics"),
    ):
        _regular(path, label)
    sweep = _load_json(sweep_path, "frozen SPD sweep")
    split = _load_json(split_path, "frozen SPD split")
    protocol = _load_json(protocol_path, "frozen SPD protocol")
    summary_payload = _load_json(summary_path, "frozen SPD summary")
    rows = _read_metrics(metrics_path, "spd")
    args = protocol.get("arguments")
    _require(
        isinstance(args, dict)
        and args.get("variant") == "spd"
        and args.get("dataset") == DATASET
        and args.get("seed") == 42
        and args.get("epochs") == EXPECTED_EPOCHS
        and args.get("eval_every") == 1
        and args.get("split_seed") == EXPECTED_SPLIT_SEED,
        "frozen SPD protocol identity differs",
    )
    _require(
        split.get("hashes", {}).get("used_val_sha256")
        == expected_validation_sha256,
        "frozen SPD validation split differs",
    )
    _require(
        split.get("dataset") == DATASET
        and split.get("split_seed") == EXPECTED_SPLIT_SEED
        and split.get("used_train_count") == EXPECTED_TRAIN_COUNT
        and split.get("used_val_count") == EXPECTED_VAL_COUNT,
        "frozen SPD split contract differs",
    )
    pd_row = max(rows, key=_selection_key_pd)
    _require(
        summary_payload.get("status") == "complete"
        and summary_payload.get("variant") == "spd"
        and summary_payload.get("dataset") == DATASET
        and summary_payload.get("seed") == 42
        and summary_payload.get("best_epoch") == pd_row["epoch"]
        and summary_payload.get("best_pd_epoch") == pd_row["epoch"]
        and summary_payload.get("official_test_accessed") is False
        and summary_payload.get("selection_source")
        == "internal_validation_only",
        "frozen SPD summary differs",
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    _require(
        isinstance(checkpoint, dict)
        and checkpoint.get("variant") == "spd"
        and checkpoint.get("dataset") == DATASET
        and checkpoint.get("seed") == 42
        and checkpoint.get("split_seed") == EXPECTED_SPLIT_SEED
        and checkpoint.get("epoch") == pd_row["epoch"]
        and checkpoint.get("checkpoint_role")
        == "best_validation_pd_primary"
        and checkpoint.get("official_test_accessed") is False
        and checkpoint.get("selection_source")
        == "internal_validation_only",
        "frozen SPD checkpoint identity differs",
    )
    state_dict = checkpoint.get("state_dict")
    _require(isinstance(state_dict, Mapping), "frozen SPD state_dict missing")
    _require_finite_state(state_dict, "frozen SPD state_dict")
    spd_model, _ = build_model("spd", 42)
    spd_model.load_state_dict(state_dict, strict=True)
    summary_metrics = summary_payload.get("best_pd_validation_metrics")
    _require(
        isinstance(summary_metrics, dict)
        and summary_payload.get("best_validation_metrics") == summary_metrics
        and checkpoint.get("validation_metrics") == summary_metrics
        and all(pd_row.get(key) == value for key, value in summary_metrics.items()),
        "frozen SPD selected metrics differ",
    )
    _require(
        sweep.get("variant") == "spd"
        and sweep.get("seed") == 42
        and sweep.get("validation_split_sha256")
        == expected_validation_sha256,
        "frozen SPD identity differs",
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    _require(
        sweep.get("checkpoint_sha256") == checkpoint_sha
        and Path(str(sweep.get("checkpoint", ""))).resolve()
        == checkpoint_path.resolve()
        and sweep.get("checkpoint_role")
        == "best_validation_pd_primary"
        and sweep.get("checkpoint_epoch") == pd_row["epoch"]
        and sweep.get("checkpoint_validation_metrics") == summary_metrics,
        "frozen SPD sweep/checkpoint binding differs",
    )
    points_raw = sweep.get("points")
    _require(isinstance(points_raw, list) and points_raw, "SPD points missing")
    points = [
        _validate_point(point, f"frozen SPD point[{index}]")
        for index, point in enumerate(points_raw)
    ]
    raw_budgets = sweep.get("best_points_under_fa_budget")
    _require(
        isinstance(raw_budgets, dict)
        and set(raw_budgets) == set(BUDGET_KEYS),
        "frozen SPD budget keys differ",
    )
    budgets: dict[str, dict[str, Any]] = {}
    for key in BUDGET_KEYS:
        _require(raw_budgets[key] is not None, f"frozen SPD null budget {key}")
        point = _validate_point(raw_budgets[key], f"frozen SPD budget {key}")
        _require(
            point == _best_under_budget(points, float(key)),
            f"frozen SPD budget optimum differs: {key}",
        )
        budgets[key] = point
    fixed = _validate_point(
        sweep.get("fixed_threshold_0_5"), "frozen SPD fixed threshold"
    )
    _require(
        float(fixed["threshold"]) == 0.5
        and fixed in points
        and all(
            fixed.get(key) == value
            for key, value in summary_metrics.items()
            if key in fixed
        ),
        "frozen SPD fixed-threshold reproduction differs",
    )
    audit = sweep.get("audit")
    _require(
        isinstance(audit, dict)
        and audit.get("expected_epochs") == EXPECTED_EPOCHS
        and audit.get("metrics_event_count") == EXPECTED_EPOCHS
        and audit.get("metrics_epoch_range") == [1, EXPECTED_EPOCHS]
        and audit.get("summary_status") == "complete",
        "frozen SPD sweep audit differs",
    )
    return {
        "variant": "spd",
        "seed": 42,
        "run_directory": str(run_dir.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": pd_row["epoch"],
        "checkpoint_role": "best_validation_pd_primary",
        "strict_load": True,
        "sweep": str(sweep_path.resolve()),
        "sweep_sha256": sha256_file(sweep_path),
        "artifact_sha256": {
            path.name: sha256_file(path)
            for path in (
                checkpoint_path,
                sweep_path,
                protocol_path,
                split_path,
                summary_path,
                metrics_path,
            )
        },
        "roles": {"pd_primary": {"budgets": budgets}},
    }


def _bind_persisted_smoke_verification(
    current: Mapping[str, Any],
    persisted: Mapping[str, Any],
) -> dict[str, Any]:
    current_stable = {
        key: value for key, value in current.items() if key != "verified_at_utc"
    }
    persisted_stable = {
        key: value
        for key, value in persisted.items()
        if key != "verified_at_utc"
    }
    _require(
        current_stable == persisted_stable,
        "persisted smoke verification differs from current re-verification",
    )
    normalized = copy.deepcopy(dict(current))
    normalized["verified_at_utc"] = persisted.get("verified_at_utc")
    normalized["persisted_verification"] = {
        "path": str(SMOKE_VERIFICATION.resolve()),
        "sha256": sha256_file(SMOKE_VERIFICATION),
    }
    return normalized


def build_report(
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    *,
    postprocess_source_lock: Path = DEFAULT_POSTPROCESS_SOURCE_LOCK,
) -> dict[str, Any]:
    """Strictly derive one complete V6 Gate A--E report."""

    training_lock, training_lock_sha = _validate_current_training_contract()
    _, postprocess_lock_sha = validate_postprocess_source_lock(
        postprocess_source_lock
    )
    evaluator_relative = "experiments/evaluate_tpd_clean_v6_pd_fa.py"
    evaluator_sha = training_lock["source_sha256"][evaluator_relative]

    runs: dict[tuple[str, int], dict[str, Any]] = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            runs[(variant, seed)] = validate_candidate_run(
                candidate_root,
                variant,
                seed,
                training_lock=training_lock,
                evaluator_sha256=evaluator_sha,
            )
    split_pairs = {
        (
            run["split_hashes"]["used_train_sha256"],
            run["split_hashes"]["used_val_sha256"],
        )
        for run in runs.values()
    }
    _require(len(split_pairs) == 1, "candidate run splits differ")
    validation_sha = next(iter(split_pairs))[1]
    for seed in SEEDS:
        full = runs[(PRIMARY_VARIANT, seed)]["model"]
        capacity = runs[(CONTROL_VARIANT, seed)]["model"]
        _require(
            full["full_initialization_sha256"]
            == capacity["full_initialization_sha256"],
            f"seed={seed}: paired full initialization differs",
        )
        _require(
            full["shared_initialization_sha256"]
            == capacity["shared_initialization_sha256"],
            f"seed={seed}: paired shared initialization differs",
        )

    from experiments.verify_tpd_clean_v6_smoke_reports import (
        validate_smoke_reports,
    )

    smoke = validate_smoke_reports(DEFAULT_SMOKE_ROOT)
    _require(
        smoke.get("status") == "complete" and smoke.get("passed") is True,
        "CPU/GPU2/GPU3 smoke set failed",
    )
    persisted_smoke = _load_json(
        SMOKE_VERIFICATION, "persisted V6 smoke verification"
    )
    # Re-verification creates a fresh observation timestamp.  The formal report
    # instead binds the already persisted verification time so repeated strict
    # derivations differ only in the report's own generated_at_utc field.
    smoke = _bind_persisted_smoke_verification(smoke, persisted_smoke)
    spd = load_spd_reference(validation_sha)
    integrity = {
        "four_runs_contiguous_800_epochs": all(
            run["metrics_event_count"] == EXPECTED_EPOCHS
            for run in runs.values()
        ),
        "twelve_checkpoints_present_and_strict_load": (
            sum(len(run["checkpoints"]) for run in runs.values()) == 12
            and all(
                item["strict_load"] is True
                for run in runs.values()
                for item in run["checkpoints"].values()
            )
        ),
        "eight_closed_interval_sweeps": (
            sum(len(run["roles"]) for run in runs.values()) == 8
            and all(
                role["closed_interval"] is True
                for run in runs.values()
                for role in run["roles"].values()
            )
        ),
        "model_split_protocol_evaluator_hashes_consistent": all(
            role["evaluator_sha256"] == evaluator_sha
            for run in runs.values()
            for role in run["roles"].values()
        ),
        "cpu_and_rtx5090_smoke_passed": smoke.get("passed") is True,
        "fixed_threshold_reproduction_exact": all(
            role["fixed_threshold_reproduction_exact"] is True
            for run in runs.values()
            for role in run["roles"].values()
        ),
        "all_five_budgets_available": all(
            set(role["budgets"]) == set(BUDGET_KEYS)
            for run in runs.values()
            for role in run["roles"].values()
        ),
        "preregistered_endpoint_provenance": all(
            role["preregistered_endpoint_provenance"] is True
            for run in runs.values()
            for role in run["roles"].values()
        ),
        "exact_epoch_journals_complete": all(
            run["exact_journal"]["active_epoch"] == EXPECTED_EPOCHS
            for run in runs.values()
        ),
        "worker_logs_complete_gpu_mapped": all(
            run["worker_log"]["completion_count"] == 1
            and (
                run["worker_log"]["physical_gpu"],
                run["worker_log"]["gpu_uuid"],
            )
            == GPU_ASSIGNMENTS[(variant, seed)]
            for (variant, seed), run in runs.items()
        ),
    }
    gates = evaluate_engineering_gates(runs, spd, integrity)
    passed = bool(gates["passed"])
    return {
        "schema": SCHEMA,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete",
        "scope": {
            "dataset": DATASET,
            "candidate_root": str(Path(candidate_root).resolve()),
            "variants": list(VARIANTS),
            "seeds": list(SEEDS),
            "epochs": EXPECTED_EPOCHS,
            "fa_budgets": list(BUDGET_KEYS),
            "official_test_accessed": False,
        },
        "training_source_lock_sha256": training_lock_sha,
        "training_data_sha256": EXPECTED_TRAINING_DATA_SHA256,
        "postprocess_source_lock_sha256": postprocess_lock_sha,
        "evaluator_sha256": evaluator_sha,
        "candidate_runs": {
            f"{variant}/seed_{seed}": copy.deepcopy(runs[(variant, seed)])
            for seed in SEEDS
            for variant in VARIANTS
        },
        "frozen_spd_reference": spd,
        "smoke_validation": smoke,
        "engineering_integrity": integrity,
        "engineering_gate": gates,
        "gate_evaluated": True,
        "engineering_gate_passed": passed,
        "ner_stage_authorized": passed,
        "decision": (
            "ENGINEERING_GATE_PASS" if passed else "ENGINEERING_GATE_FAIL"
        ),
        "mainline_changed": False,
        "paper_core_established": False,
        "stability_claim_supported": False,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    gates = report["engineering_gate"]["checks"]
    lines = [
        "# TPD-Clean-v6 formal800 Gates A–E",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Engineering gate passed: `{str(report['engineering_gate_passed']).lower()}`",
        f"- NER engineering stage authorized: `{str(report['ner_stage_authorized']).lower()}`",
        "- Mainline changed: `false`",
        "",
        "## Fixed-threshold checkpoints (8)",
        "",
        "| Seed | Variant | Role | Pd | Fa | mIoU |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        for variant in VARIANTS:
            run = report["candidate_runs"][f"{variant}/seed_{seed}"]
            for role in ROLE_SPECS:
                point = run["roles"][role]["fixed_threshold_0_5"]
                lines.append(
                    f"| {seed} | `{variant}` | `{role}` | "
                    f"{point['matched_target_count']}/{point['target_count']} | "
                    f"{float(point['fa']):.10g} | {float(point['miou']):.9f} |"
                )
    lines.extend(
        [
            "",
            "## Registered Fa-budget operating points (40)",
            "",
            "| Seed | Variant | Role | Budget | Threshold | Pd | Fa | mIoU |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for seed in SEEDS:
        for variant in VARIANTS:
            run = report["candidate_runs"][f"{variant}/seed_{seed}"]
            for role_name in ROLE_SPECS:
                for budget in BUDGET_KEYS:
                    point = run["roles"][role_name]["budgets"][budget]
                    lines.append(
                        f"| {seed} | `{variant}` | `{role_name}` | `{budget}` | "
                        f"{float(point['threshold']):.10g} | "
                        f"{point['matched_target_count']}/{point['target_count']} | "
                        f"{float(point['fa']):.10g} | "
                        f"{float(point['miou']):.9f} |"
                    )

    gate_a = gates["gate_a_seed42_fixed_threshold"]
    lines.extend(
        [
            "",
            "## Gate A — seed 42 fixed threshold",
            "",
            f"Gate A passed: `{str(gate_a['passed']).lower()}`.",
            "",
            "| Subcheck | Passed |",
            "| --- | --- |",
        ]
    )
    for name, passed in gate_a["subchecks"].items():
        lines.append(f"| `{name}` | `{str(passed).lower()}` |")

    gate_b = gates["gate_b_seed42_budget_and_spd"]
    lines.extend(
        [
            "",
            "## Gate B — seed 42 budgets and frozen SPD",
            "",
            f"Gate B passed: `{str(gate_b['passed']).lower()}`.",
            "",
            "| Budget | Full thr | Full Pd | Full Fa | Full mIoU | "
            "SPD thr | SPD Pd | SPD Fa | SPD mIoU | Full not covered |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for budget in BUDGET_KEYS:
        comparison = gate_b["frozen_spd_comparisons"][budget]
        full = comparison["full"]
        spd = comparison["spd"]
        lines.append(
            f"| `{budget}` | {full['threshold']:.10g} | "
            f"{full['matched_target_count']}/{full['target_count']} | "
            f"{full['fa']:.10g} | {full['miou']:.9f} | "
            f"{spd['threshold']:.10g} | "
            f"{spd['matched_target_count']}/{spd['target_count']} | "
            f"{spd['fa']:.10g} | {spd['miou']:.9f} | "
            f"`{str(comparison['full_not_covered_by_spd']).lower()}` |"
        )
    for name, passed in gate_b["subchecks"].items():
        lines.append(f"- `{name}`: `{str(passed).lower()}`")

    gate_c = gates["gate_c_seed3407_stability"]
    lines.extend(
        [
            "",
            "## Gate C — seed 3407 stability",
            "",
            f"Gate C passed: `{str(gate_c['passed']).lower()}`.",
            "",
            "| Budget | seed42 matched | seed3407 matched | Required minimum | Passed |",
            "| ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for budget in BUDGET_KEYS:
        item = gate_c["budget_stability"][budget]
        lines.append(
            f"| `{budget}` | {item['seed42_matched_target_count']} | "
            f"{item['seed3407_matched_target_count']} | "
            f"{item['minimum_seed3407']} | "
            f"`{str(item['passed']).lower()}` |"
        )
    for name, passed in gate_c["subchecks"].items():
        lines.append(f"- `{name}`: `{str(passed).lower()}`")

    gate_d = gates["gate_d_full_vs_capacity"]
    lines.extend(
        [
            "",
            "## Gate D — Full versus capacity (24 comparisons)",
            "",
            f"Gate D passed: `{str(gate_d['passed']).lower()}`.",
            "",
            "| Seed | Work point | Full thr | Full Pd | Full Fa | Full mIoU | "
            "Capacity thr | Capacity Pd | Capacity Fa | Capacity mIoU | Direction |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for seed in SEEDS:
        record = gate_d["per_seed"][str(seed)]
        for label, comparison in record["comparisons"].items():
            full = comparison["full"]
            capacity = comparison["capacity"]
            if comparison.get("capacity_strictly_covers_full"):
                direction = "capacity>full"
            elif comparison.get("full_strictly_covers_capacity"):
                direction = "full>capacity"
            else:
                direction = "tradeoff/equal"
            lines.append(
                f"| {seed} | `{label}` | {full['threshold']:.10g} | "
                f"{full['matched_target_count']}/{full['target_count']} | "
                f"{full['fa']:.10g} | {full['miou']:.9f} | "
                f"{capacity['threshold']:.10g} | "
                f"{capacity['matched_target_count']}/{capacity['target_count']} | "
                f"{capacity['fa']:.10g} | {capacity['miou']:.9f} | "
                f"`{direction}` |"
            )
        for name, passed in record["subchecks"].items():
            lines.append(
                f"- seed {seed} `{name}`: `{str(passed).lower()}`"
            )

    gate_e = gates["gate_e_engineering_integrity"]
    lines.extend(
        [
            "",
            "## Gate E — engineering completeness",
            "",
            f"Gate E passed: `{str(gate_e['passed']).lower()}`.",
            "",
            "| Integrity item | Passed |",
            "| --- | --- |",
        ]
    )
    for name, passed in gate_e["subchecks"].items():
        lines.append(f"| `{name}` | `{str(passed).lower()}` |")

    lines.extend(
        [
            "",
            "## SHA-256 bindings",
            "",
            "| Artifact | SHA-256 |",
            "| --- | --- |",
            f"| Training source lock | `{report['training_source_lock_sha256']}` |",
            f"| Training data | `{report['training_data_sha256']}` |",
            f"| Postprocess source lock | `{report['postprocess_source_lock_sha256']}` |",
            f"| V6 evaluator | `{report['evaluator_sha256']}` |",
            f"| Frozen SPD sweep | `{report['frozen_spd_reference']['sweep_sha256']}` |",
            f"| Persisted smoke verification | "
            f"`{report['smoke_validation']['persisted_verification']['sha256']}` |",
        ]
    )
    for seed in SEEDS:
        for variant in VARIANTS:
            run = report["candidate_runs"][f"{variant}/seed_{seed}"]
            for name, digest in run["artifact_sha256"].items():
                lines.append(
                    f"| `{variant}/seed_{seed}/{name}` | `{digest}` |"
                )
            for role_name, role in run["roles"].items():
                lines.append(
                    f"| `{variant}/seed_{seed}/{role_name}/sweep` | "
                    f"`{role['sweep_sha256']}` |"
                )
                lines.append(
                    f"| `{variant}/seed_{seed}/{role_name}/checkpoint` | "
                    f"`{role['checkpoint_sha256']}` |"
                )
    lines.extend(
        [
            "",
            "Passing all five gates authorizes only the next five-node NER "
            "engineering and tokenizer×relay interaction stage.",
            "",
        ]
    )
    return "\n".join(lines)


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


def write_new(path: Path, content: bytes) -> Path:
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite formal report: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise NotADirectoryError(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def write_report_once(
    report: Mapping[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir).absolute()
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise NotADirectoryError(output_dir)
    json_path = output_dir / JSON_OUTPUT_NAME
    markdown_path = output_dir / MARKDOWN_OUTPUT_NAME
    if any(path.exists() or path.is_symlink() for path in (json_path, markdown_path)):
        raise FileExistsError("refusing to overwrite an existing formal report")
    write_new(json_path, _json_bytes(report))
    try:
        write_new(markdown_path, render_markdown(report).encode("utf-8"))
    except BaseException:
        json_path.unlink(missing_ok=True)
        raise
    return json_path, markdown_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate V6 formal800 and evaluate Gates A--E"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--write", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.preflight:
        print(
            json.dumps(
                inspect_training_readiness(),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            flush=True,
        )
        return
    readiness = inspect_training_readiness()
    if readiness["formal_matrix_complete"] is not True:
        raise SystemExit(
            "V6 formal800 matrix is incomplete; only --preflight is allowed"
        )
    report = build_report()
    paths = write_report_once(report)
    print(
        f"WROTE decision={report['decision']} "
        f"json={paths[0]} markdown={paths[1]}",
        flush=True,
    )


__all__ = [
    "BUDGET_KEYS",
    "CONTROL_VARIANT",
    "DEFAULT_CANDIDATE_ROOT",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_POSTPROCESS_SOURCE_LOCK",
    "EXPECTED_EPOCHS",
    "EXPECTED_TRAINING_DATA_SHA256",
    "EXPECTED_TRAINING_LOCK_SHA256",
    "INTEGRITY_KEYS",
    "IncompleteArtifact",
    "JSON_OUTPUT_NAME",
    "LAST_FLOAT32_BELOW_ONE",
    "MARKDOWN_OUTPUT_NAME",
    "PRIMARY_VARIANT",
    "ROLE_SPECS",
    "SEEDS",
    "VARIANTS",
    "build_report",
    "evaluate_engineering_gates",
    "inspect_training_readiness",
    "load_spd_reference",
    "main",
    "parse_args",
    "render_markdown",
    "sha256_file",
    "validate_postprocess_source_lock",
    "write_new",
    "write_report_once",
]


if __name__ == "__main__":
    main()
