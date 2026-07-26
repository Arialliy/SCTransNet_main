#!/usr/bin/env python3
"""Audit and summarize the isolated TPD-Clean-v3 screen.

This post-training utility is deliberately outside the v3 training source
lock.  It never repairs or infers a missing result: the engineering gate is
evaluated only after all four candidate runs, eight checkpoints, eight
candidate sweeps, and the frozen seed-42 reference sweeps pass validation.
Otherwise it writes an explicit ``status=incomplete`` report.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_clean_v3_screen800_4x5090_v1"
)
DEFAULT_FORMAL_REFERENCE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_pe_formal800_4x5090_v1"
)
DEFAULT_V2_REFERENCE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_clean_screen800_4x5090_v1"
)
DEFAULT_REFERENCE_MIOU_ROOT = (
    DEFAULT_V2_REFERENCE_ROOT / "frozen_reference_miou_runs"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CANDIDATE_ROOT / "NUDT-SIRST/comparison"

JSON_OUTPUT_NAME = "tpd_clean_v3_screen800_comparison.json"
MARKDOWN_OUTPUT_NAME = "tpd_clean_v3_screen800_comparison.md"
SCHEMA = "sctransnet_tpd_clean_v3_screen800_comparison_v1"

DATASET = "NUDT-SIRST"
VARIANTS = (
    "tpd_clean_v3_full",
    "tpd_clean_v3_sal_capacity",
)
PRIMARY_VARIANT = "tpd_clean_v3_full"
CONTROL_VARIANT = "tpd_clean_v3_sal_capacity"
SEEDS = (42, 3407)
RUN_TAG = "screen800_pd_fp32_shared4x5090_v1"
EXPECTED_EPOCHS = 800
EXPECTED_TRAIN_COUNT = 530
EXPECTED_VAL_COUNT = 133
EXPECTED_TARGET_COUNT = 189
EXPECTED_SPLIT_SEED = 20260722
EXPECTED_TOTAL_PARAMETERS = 10_843_475
EXPECTED_SHALLOW_PARAMETERS = 66_496
BUDGET_KEYS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")
LOW_FA_BUDGET_KEYS = ("1e-06", "5e-06")
WIDE_BUDGET_KEYS = ("1e-05", "5e-05", "0.0001")
REFERENCE_GATE_BUDGET_USAGE = {
    **{
        ("spd", "pd_primary", budget): "gate_4_seed42_frozen_references"
        for budget in BUDGET_KEYS
    },
    (
        "v2_sal_only",
        "pd_primary",
        "5e-06",
    ): "gate_4_seed42_frozen_references",
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
POINT_FIELDS = ("matched_target_count", "target_count", "fa", "miou", "pd")


class IncompleteArtifact(ValueError):
    """A required result is missing, unfinished, or fails integrity checks."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IncompleteArtifact(message)


def _require_file(path: Path, label: str) -> None:
    _require(path.is_file() and not path.is_symlink(), f"{label}: missing file {path}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IncompleteArtifact(f"{label}: invalid JSON {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{label}: JSON root must be an object")
    _require_finite_tree(payload, label)
    return payload


def _require_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label}: numeric values must be finite")
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            _require_finite_tree(nested, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _require_finite_tree(nested, f"{label}[{index}]")


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label}: expected lowercase SHA-256",
    )
    return value


def _same_number(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def _require_metric_subset(
    actual: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    for key, expected_value in expected.items():
        if key == "threshold":
            continue
        _require(key in actual, f"{label}: missing metric {key}")
        _require(
            _same_number(actual[key], expected_value),
            f"{label}: metric {key} differs: {actual[key]!r} != {expected_value!r}",
        )


def _validate_point(point: Any, label: str) -> dict[str, Any]:
    if not isinstance(point, dict):
        raise ValueError(f"{label}: point must be an object")
    _require_finite_tree(point, label)
    for field in POINT_FIELDS:
        if field not in point:
            raise ValueError(f"{label}: missing required field {field}")
    matched = point["matched_target_count"]
    target = point["target_count"]
    fa = point["fa"]
    miou = point["miou"]
    pd = point["pd"]
    if isinstance(matched, bool) or not isinstance(matched, int):
        raise ValueError(f"{label}: matched_target_count must be an integer")
    if isinstance(target, bool) or not isinstance(target, int):
        raise ValueError(f"{label}: target_count must be an integer")
    if target != EXPECTED_TARGET_COUNT:
        raise ValueError(
            f"{label}: target_count must be {EXPECTED_TARGET_COUNT}, got {target}"
        )
    if not 0 <= matched <= target:
        raise ValueError(f"{label}: invalid matched_target_count {matched}/{target}")
    for name, value in (("fa", fa), ("miou", miou), ("pd", pd)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}: {name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{label}: {name} must be finite")
    if float(fa) < 0.0:
        raise ValueError(f"{label}: fa cannot be negative")
    if not 0.0 <= float(miou) <= 1.0:
        raise ValueError(f"{label}: miou outside [0, 1]")
    expected_pd = matched / target
    if abs(float(pd) - expected_pd) > 1e-15:
        raise ValueError(
            f"{label}: pd={pd} inconsistent with {matched}/{target}"
        )
    return point


def _validate_role_record(role: Any, label: str) -> dict[str, Any]:
    if not isinstance(role, dict):
        raise ValueError(f"{label}: role must be an object")
    if "fixed_threshold_0_5" not in role:
        raise ValueError(f"{label}: missing fixed_threshold_0_5")
    if "budgets" not in role or not isinstance(role["budgets"], dict):
        raise ValueError(f"{label}: missing budgets")
    _validate_point(role["fixed_threshold_0_5"], f"{label}.fixed_threshold_0_5")
    budgets = role["budgets"]
    for key in BUDGET_KEYS:
        if key not in budgets:
            raise ValueError(f"{label}.budgets: missing {key}")
        point = _validate_point(budgets[key], f"{label}.budgets.{key}")
        if float(point["fa"]) > float(key) + 1e-15:
            raise ValueError(
                f"{label}.budgets.{key}: actual Fa {point['fa']} exceeds budget"
            )
    return role


def _validate_reference_role_record(role: Any, label: str) -> dict[str, Any]:
    """Validate a frozen role while preserving explicit unavailable budgets."""
    if not isinstance(role, dict):
        raise ValueError(f"{label}: role must be an object")
    if "fixed_threshold_0_5" not in role:
        raise ValueError(f"{label}: missing fixed_threshold_0_5")
    if "budgets" not in role or not isinstance(role["budgets"], dict):
        raise ValueError(f"{label}: missing budgets")
    _validate_point(role["fixed_threshold_0_5"], f"{label}.fixed_threshold_0_5")
    budgets = role["budgets"]
    for key in BUDGET_KEYS:
        if key not in budgets:
            raise ValueError(f"{label}.budgets: missing {key}")
        if budgets[key] is None:
            continue
        point = _validate_point(budgets[key], f"{label}.budgets.{key}")
        if float(point["fa"]) > float(key) + 1e-15:
            raise ValueError(
                f"{label}.budgets.{key}: actual Fa {point['fa']} exceeds budget"
            )
    return role


def _get_run(
    runs: Mapping[Any, Any], variant: str, seed: int
) -> dict[str, Any]:
    key = (variant, seed)
    if key not in runs:
        raise ValueError(f"candidate runs missing {variant}/seed={seed}")
    run = runs[key]
    if not isinstance(run, dict) or not isinstance(run.get("roles"), dict):
        raise ValueError(f"candidate run {variant}/seed={seed} lacks roles")
    for role_name in ROLE_SPECS:
        if role_name not in run["roles"]:
            raise ValueError(
                f"candidate run {variant}/seed={seed} missing {role_name}"
            )
        _validate_role_record(
            run["roles"][role_name],
            f"{variant}/seed={seed}.{role_name}",
        )
    return run


def _get_reference(
    references: Mapping[str, Any], name: str
) -> dict[str, Any]:
    if name not in references:
        raise ValueError(f"frozen references missing {name}")
    reference = references[name]
    if not isinstance(reference, dict) or not isinstance(
        reference.get("roles"), dict
    ):
        raise ValueError(f"frozen reference {name} lacks roles")
    for role_name in ROLE_SPECS:
        if role_name not in reference["roles"]:
            raise ValueError(f"frozen reference {name} missing {role_name}")
        _validate_reference_role_record(
            reference["roles"][role_name],
            f"frozen_reference.{name}.{role_name}",
        )
    return reference


def _required_reference_budget_point(
    references: Mapping[str, Any],
    method: str,
    role_name: str,
    budget: str,
) -> dict[str, Any]:
    point = references[method]["roles"][role_name]["budgets"][budget]
    if point is None:
        raise ValueError(
            "frozen_reference."
            f"{method}.{role_name}.budgets.{budget}: "
            "point required by the engineering gate is unavailable"
        )
    return _validate_point(
        point,
        f"frozen_reference.{method}.{role_name}.budgets.{budget}",
    )


def _reference_unavailable_points(
    references: Mapping[str, Any],
) -> list[dict[str, Any]]:
    unavailable: list[dict[str, Any]] = []
    for method in ("spd", "tpd_v1", "v2_sal_only", "v2_full"):
        reference = references.get(method)
        if not isinstance(reference, Mapping):
            continue
        roles = reference.get("roles")
        if not isinstance(roles, Mapping):
            continue
        for role_name in ROLE_SPECS:
            role = roles.get(role_name)
            if not isinstance(role, Mapping):
                continue
            budgets = role.get("budgets")
            if not isinstance(budgets, Mapping):
                continue
            for budget in BUDGET_KEYS:
                if budgets.get(budget) is not None:
                    continue
                usage = REFERENCE_GATE_BUDGET_USAGE.get(
                    (method, role_name, budget)
                )
                unavailable.append(
                    {
                        "method": method,
                        "role": role_name,
                        "budget": budget,
                        "used_by_gates": usage is not None,
                        "gate_usage": usage or "not_used_by_gates",
                    }
                )
    return unavailable


def _operational_compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    """Compare a Pd--Fa point in the registered task-priority order."""
    _validate_point(dict(left), "operational_compare.left")
    _validate_point(dict(right), "operational_compare.right")
    left_key = (
        int(left["matched_target_count"]),
        -float(left["fa"]),
        float(left["miou"]),
    )
    right_key = (
        int(right["matched_target_count"]),
        -float(right["fa"]),
        float(right["miou"]),
    )
    return (left_key > right_key) - (left_key < right_key)


def _dominates(
    candidate: Mapping[str, Any], other: Mapping[str, Any]
) -> bool:
    """Return whether ``candidate`` Pareto-dominates ``other``."""
    _validate_point(dict(candidate), "dominance.candidate")
    _validate_point(dict(other), "dominance.other")
    no_worse = (
        int(candidate["matched_target_count"])
        >= int(other["matched_target_count"])
        and float(candidate["fa"]) <= float(other["fa"])
        and float(candidate["miou"]) >= float(other["miou"])
    )
    strict = (
        int(candidate["matched_target_count"])
        > int(other["matched_target_count"])
        or float(candidate["fa"]) < float(other["fa"])
        or float(candidate["miou"]) > float(other["miou"])
    )
    return no_worse and strict


def _point_digest(point: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "matched_target_count": int(point["matched_target_count"]),
        "target_count": int(point["target_count"]),
        "pd": float(point["pd"]),
        "fa": float(point["fa"]),
        "miou": float(point["miou"]),
    }


def _sweep_budget_selection_key(
    point: Mapping[str, Any],
) -> tuple[float, float, float, float, float]:
    tiny_pd = point.get("tiny_pd")
    tiny_value = -1.0 if tiny_pd is None else float(tiny_pd)
    return (
        float(point["pd"]),
        -float(point["fa"]),
        tiny_value,
        float(point["miou"]),
        -abs(float(point["threshold"]) - 0.5),
    )


def _best_sweep_point_under_fa(
    points: Sequence[Mapping[str, Any]],
    budget: float,
) -> Mapping[str, Any] | None:
    feasible = [point for point in points if float(point["fa"]) <= budget]
    if not feasible:
        return None
    return max(feasible, key=_sweep_budget_selection_key)


def _direction_name(sign: int) -> str:
    return {
        -1: "capacity_control_better",
        0: "tie",
        1: "full_better",
    }[sign]


def _aggregate_direction(signs: Sequence[int]) -> str:
    has_positive = any(sign > 0 for sign in signs)
    has_negative = any(sign < 0 for sign in signs)
    if has_positive and not has_negative:
        return "full_better"
    if has_negative and not has_positive:
        return "capacity_control_better"
    if has_positive and has_negative:
        return "mixed"
    return "tie"


def evaluate_engineering_gate(
    runs: Mapping[Any, Any],
    references: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate all seven checks from ``TPD_CLEAN_V3_PROTOCOL.md`` section 5.

    This function requires complete normalized point records.  A missing or
    non-finite field raises ``ValueError``; callers must report that state as
    incomplete instead of treating it as a failed or passed gate.
    """

    normalized_runs = {
        (variant, seed): _get_run(runs, variant, seed)
        for variant in VARIANTS
        for seed in SEEDS
    }
    normalized_references = {
        name: _get_reference(references, name)
        for name in ("spd", "tpd_v1", "v2_sal_only", "v2_full")
    }
    full_42 = normalized_runs[(PRIMARY_VARIANT, 42)]

    pd_fixed = full_42["roles"]["pd_primary"]["fixed_threshold_0_5"]
    gate_1_subchecks = {
        "matched_target_count_at_least_188": int(
            pd_fixed["matched_target_count"]
        )
        >= 188,
        "fa_at_most_5e_6": float(pd_fixed["fa"]) <= 5e-6,
        "miou_at_least_tpd_v1_anchor": float(pd_fixed["miou"])
        >= 0.9336470588,
    }
    gate_1 = {
        "passed": all(gate_1_subchecks.values()),
        "subchecks": gate_1_subchecks,
        "observed": _point_digest(pd_fixed),
    }

    miou_fixed = full_42["roles"]["miou_primary"]["fixed_threshold_0_5"]
    gate_2_subchecks = {
        "miou_at_least_spd_pd_anchor": float(miou_fixed["miou"]) >= 0.946542,
        "matched_target_count_at_least_187": int(
            miou_fixed["matched_target_count"]
        )
        >= 187,
        "fa_at_most_1e_6": float(miou_fixed["fa"]) <= 1e-6,
    }
    gate_2 = {
        "passed": all(gate_2_subchecks.values()),
        "subchecks": gate_2_subchecks,
        "observed": _point_digest(miou_fixed),
    }

    pd_budgets_42 = full_42["roles"]["pd_primary"]["budgets"]
    gate_3_budget_checks: dict[str, Any] = {}
    for budget in BUDGET_KEYS:
        minimum = 187 if budget == "1e-06" else 188
        point = pd_budgets_42[budget]
        passed = (
            int(point["matched_target_count"]) >= minimum
            and float(point["fa"]) <= float(budget) + 1e-15
        )
        gate_3_budget_checks[budget] = {
            "passed": passed,
            "minimum_matched_target_count": minimum,
            "observed": _point_digest(point),
        }
    gate_3 = {
        "passed": all(
            item["passed"] for item in gate_3_budget_checks.values()
        ),
        "budgets": gate_3_budget_checks,
    }

    strict_spd_comparisons: dict[str, Any] = {}
    for budget in BUDGET_KEYS:
        spd_point = _required_reference_budget_point(
            normalized_references,
            "spd",
            "pd_primary",
            budget,
        )
        sign = _operational_compare(pd_budgets_42[budget], spd_point)
        strict_spd_comparisons[budget] = {
            "full_vs_spd": _direction_name(sign),
            "strictly_better": sign > 0,
            "full": _point_digest(pd_budgets_42[budget]),
            "spd": _point_digest(spd_point),
        }
    v2_sal_5e6 = _required_reference_budget_point(
        normalized_references,
        "v2_sal_only",
        "pd_primary",
        "5e-06",
    )
    v2_sal_sign = _operational_compare(pd_budgets_42["5e-06"], v2_sal_5e6)
    gate_4_subchecks = {
        "strictly_better_than_spd_at_one_or_more_budgets": any(
            item["strictly_better"]
            for item in strict_spd_comparisons.values()
        ),
        "not_worse_than_v2_sal_only_at_5e_6": v2_sal_sign >= 0,
    }
    gate_4 = {
        "passed": all(gate_4_subchecks.values()),
        "subchecks": gate_4_subchecks,
        "spd_budget_comparisons": strict_spd_comparisons,
        "v2_sal_only_5e_6_comparison": {
            "direction": _direction_name(v2_sal_sign),
            "full": _point_digest(pd_budgets_42["5e-06"]),
            "v2_sal_only": _point_digest(v2_sal_5e6),
        },
    }

    gate_5_seeds: dict[str, Any] = {}
    for seed in SEEDS:
        full = normalized_runs[(PRIMARY_VARIANT, seed)]
        control = normalized_runs[(CONTROL_VARIANT, seed)]
        comparisons: dict[str, Any] = {}
        dominated_labels: list[str] = []
        for role_name in ROLE_SPECS:
            full_role = full["roles"][role_name]
            control_role = control["roles"][role_name]
            label = f"{role_name}.fixed_threshold_0_5"
            dominated = _dominates(
                control_role["fixed_threshold_0_5"],
                full_role["fixed_threshold_0_5"],
            )
            if dominated:
                dominated_labels.append(label)
            comparisons[label] = {
                "capacity_control_dominates_full": dominated,
                "full": _point_digest(full_role["fixed_threshold_0_5"]),
                "capacity_control": _point_digest(
                    control_role["fixed_threshold_0_5"]
                ),
            }
            for budget in BUDGET_KEYS:
                label = f"{role_name}.budget.{budget}"
                dominated = _dominates(
                    control_role["budgets"][budget],
                    full_role["budgets"][budget],
                )
                if dominated:
                    dominated_labels.append(label)
                comparisons[label] = {
                    "capacity_control_dominates_full": dominated,
                    "full": _point_digest(full_role["budgets"][budget]),
                    "capacity_control": _point_digest(
                        control_role["budgets"][budget]
                    ),
                }
        gate_5_seeds[str(seed)] = {
            "passed": not dominated_labels,
            "dominated_points": dominated_labels,
            "comparisons": comparisons,
        }
    gate_5 = {
        "passed": all(item["passed"] for item in gate_5_seeds.values()),
        "per_seed": gate_5_seeds,
        "policy": (
            "For each seed, capacity control must not Pareto-dominate Full "
            "at either fixed point or any corresponding budget point from "
            "either checkpoint role."
        ),
    }

    gate_6_seeds: dict[str, Any] = {}
    for seed in SEEDS:
        full = normalized_runs[(PRIMARY_VARIANT, seed)]
        control = normalized_runs[(CONTROL_VARIANT, seed)]
        low_fa_comparisons: dict[str, Any] = {}
        for role_name in ROLE_SPECS:
            for budget in LOW_FA_BUDGET_KEYS:
                full_point = full["roles"][role_name]["budgets"][budget]
                control_point = control["roles"][role_name]["budgets"][budget]
                sign = _operational_compare(full_point, control_point)
                low_fa_comparisons[f"{role_name}.{budget}"] = {
                    "direction": _direction_name(sign),
                    "full_strictly_better": sign > 0,
                }
        full_miou = float(
            full["roles"]["miou_primary"]["fixed_threshold_0_5"]["miou"]
        )
        control_miou = float(
            control["roles"]["miou_primary"]["fixed_threshold_0_5"]["miou"]
        )
        strict_advantage = (
            any(
                item["full_strictly_better"]
                for item in low_fa_comparisons.values()
            )
            or full_miou > control_miou
        )
        wide_budget_checks: dict[str, Any] = {}
        for budget in WIDE_BUDGET_KEYS:
            full_count = int(
                full["roles"]["pd_primary"]["budgets"][budget][
                    "matched_target_count"
                ]
            )
            control_count = int(
                control["roles"]["pd_primary"]["budgets"][budget][
                    "matched_target_count"
                ]
            )
            wide_budget_checks[budget] = {
                "passed": full_count >= control_count - 1,
                "full_matched_target_count": full_count,
                "capacity_matched_target_count": control_count,
                "full_minus_capacity": full_count - control_count,
            }
        wide_pd_passed = all(
            item["passed"] for item in wide_budget_checks.values()
        )
        gate_6_seeds[str(seed)] = {
            "passed": strict_advantage and wide_pd_passed,
            "strict_low_fa_or_miou_advantage": strict_advantage,
            "low_fa_comparisons": low_fa_comparisons,
            "miou_primary_fixed": {
                "full": full_miou,
                "capacity_control": control_miou,
                "full_strictly_better": full_miou > control_miou,
            },
            "wide_budget_pd_not_worse_by_more_than_one": wide_pd_passed,
            "wide_budget_checks": wide_budget_checks,
        }
    gate_6 = {
        "passed": all(item["passed"] for item in gate_6_seeds.values()),
        "per_seed": gate_6_seeds,
    }

    gate_7_seeds: dict[str, Any] = {}
    for seed in SEEDS:
        full = normalized_runs[(PRIMARY_VARIANT, seed)]
        control = normalized_runs[(CONTROL_VARIANT, seed)]
        fixed_signs = [
            _operational_compare(
                full["roles"][role_name]["fixed_threshold_0_5"],
                control["roles"][role_name]["fixed_threshold_0_5"],
            )
            for role_name in ROLE_SPECS
        ]
        sweep_signs = [
            _operational_compare(
                full["roles"][role_name]["budgets"][budget],
                control["roles"][role_name]["budgets"][budget],
            )
            for role_name in ROLE_SPECS
            for budget in BUDGET_KEYS
        ]
        fixed_direction = _aggregate_direction(fixed_signs)
        sweep_direction = _aggregate_direction(sweep_signs)
        opposite_decisive = {
            fixed_direction,
            sweep_direction,
        } == {"full_better", "capacity_control_better"}
        gate_7_seeds[str(seed)] = {
            "passed": not opposite_decisive,
            "fixed_threshold_overall_direction": fixed_direction,
            "budget_sweep_overall_direction": sweep_direction,
            "fixed_point_directions": [
                _direction_name(sign) for sign in fixed_signs
            ],
            "budget_point_directions": [
                _direction_name(sign) for sign in sweep_signs
            ],
        }
    gate_7 = {
        "passed": all(item["passed"] for item in gate_7_seeds.values()),
        "per_seed": gate_7_seeds,
        "policy": (
            "A contradiction is recorded only when all non-tied fixed-point "
            "comparisons give one decisive overall direction while all "
            "non-tied budget comparisons give the opposite decisive direction; "
            "mixed or tied evidence is reported without being invented as a win."
        ),
    }

    checks = {
        "gate_1_seed42_pd_primary_fixed": gate_1,
        "gate_2_seed42_miou_primary_fixed": gate_2,
        "gate_3_seed42_budget_floors": gate_3,
        "gate_4_seed42_frozen_references": gate_4,
        "gate_5_no_capacity_dominance": gate_5,
        "gate_6_paired_advantage_and_wide_pd": gate_6,
        "gate_7_fixed_sweep_direction_coherence": gate_7,
    }
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "comparison_policy": {
            "operational_order": (
                "matched_target_count higher, then actual Fa lower, "
                "then mIoU higher"
            ),
            "joint_dominance": (
                "matched_target_count no lower, actual Fa no higher, "
                "and mIoU no lower, with at least one strict inequality"
            ),
            "same_budget_uses_actual_fa": True,
        },
    }


def _read_metrics(
    path: Path, variant: str, label: str
) -> list[dict[str, Any]]:
    _require_file(path, label)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise IncompleteArtifact(f"{label}: cannot read {path}: {exc}") from exc
    _require(
        len(lines) == EXPECTED_EPOCHS,
        f"{label}: metrics.jsonl must contain {EXPECTED_EPOCHS} rows, got {len(lines)}",
    )
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        _require(bool(line.strip()), f"{label}: empty metrics row {index}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IncompleteArtifact(
                f"{label}: invalid metrics JSON at row {index}: {exc}"
            ) from exc
        _require(isinstance(row, dict), f"{label}: metrics row {index} is not an object")
        try:
            _require_finite_tree(row, f"{label}.metrics[{index}]")
        except ValueError as exc:
            raise IncompleteArtifact(str(exc)) from exc
        _require(
            row.get("epoch") == index,
            f"{label}: metrics epochs must be contiguous 1..800; row {index} has {row.get('epoch')}",
        )
        _require(
            row.get("variant") == variant,
            f"{label}: metrics row {index} variant mismatch",
        )
        _require(
            row.get("processed_train_samples") == EXPECTED_TRAIN_COUNT,
            f"{label}: metrics row {index} processed_train_samples mismatch",
        )
        try:
            _validate_point(row, f"{label}.metrics[{index}]")
        except ValueError as exc:
            raise IncompleteArtifact(str(exc)) from exc
        for required in ("val_loss", "tiny_pd"):
            _require(
                required in row and isinstance(row[required], (int, float)),
                f"{label}: metrics row {index} missing numeric {required}",
            )
        rows.append(row)
    return rows


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


def _validate_protocol(
    protocol: Mapping[str, Any],
    variant: str,
    seed: int,
    label: str,
) -> None:
    arguments = protocol.get("arguments")
    _require(isinstance(arguments, dict), f"{label}: protocol.arguments missing")
    expected_arguments = {
        "variant": variant,
        "dataset": DATASET,
        "epochs": 800,
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
    for key, expected in expected_arguments.items():
        _require(
            key in arguments and _same_number(arguments[key], expected),
            f"{label}: protocol argument {key} mismatch",
        )
    _require(
        protocol.get("official_test_accessed") is False,
        f"{label}: protocol must isolate official test data",
    )
    model = protocol.get("model")
    _require(isinstance(model, dict), f"{label}: protocol.model missing")
    _validate_model_metadata(model, variant, label)


def _validate_model_metadata(
    model: Mapping[str, Any], variant: str, label: str
) -> None:
    expected = {
        "variant": variant,
        "candidate_family": "spd_anchored_tpd_clean_v3_kcs",
        "primary_candidate": variant == PRIMARY_VARIANT,
        "mainline_contract": "Keep-Context-Saliency",
        "fourth_parallel_branch_added": False,
        "total_parameters": EXPECTED_TOTAL_PARAMETERS,
        "trainable_parameters": EXPECTED_TOTAL_PARAMETERS,
        "shallow_embedding_parameters": EXPECTED_SHALLOW_PARAMETERS,
    }
    for key, value in expected.items():
        _require(
            model.get(key) == value,
            f"{label}: model metadata {key} mismatch",
        )
    _require_sha256(
        model.get("shared_initialization_sha256"),
        f"{label}.model.shared_initialization_sha256",
    )
    _require_sha256(
        model.get("full_initialization_sha256"),
        f"{label}.model.full_initialization_sha256",
    )


def _validate_split(
    split: Mapping[str, Any], label: str
) -> dict[str, str]:
    expected = {
        "dataset": DATASET,
        "split_seed": EXPECTED_SPLIT_SEED,
        "used_train_count": EXPECTED_TRAIN_COUNT,
        "used_val_count": EXPECTED_VAL_COUNT,
        "official_test_accessed": False,
        "source": "img_idx/train_NUDT-SIRST.txt",
    }
    for key, value in expected.items():
        _require(split.get(key) == value, f"{label}: split {key} mismatch")
    hashes = split.get("hashes")
    _require(isinstance(hashes, dict), f"{label}: split.hashes missing")
    required_hashes = (
        "full_internal_train_sha256",
        "full_internal_val_sha256",
        "used_train_sha256",
        "used_val_sha256",
    )
    result = {
        key: _require_sha256(hashes.get(key), f"{label}.split.hashes.{key}")
        for key in required_hashes
    }
    _require(
        result["full_internal_train_sha256"] == result["used_train_sha256"],
        f"{label}: training split hash mismatch",
    )
    _require(
        result["full_internal_val_sha256"] == result["used_val_sha256"],
        f"{label}: validation split hash mismatch",
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
        _require(summary.get(key) == value, f"{label}: summary {key} mismatch")
    model = summary.get("model")
    _require(isinstance(model, dict), f"{label}: summary.model missing")
    _validate_model_metadata(model, variant, label)
    _require(
        model == protocol.get("model"),
        f"{label}: protocol/summary model metadata differ",
    )
    summary_hashes = summary.get("split_hashes")
    _require(
        isinstance(summary_hashes, dict)
        and all(summary_hashes.get(key) == value for key, value in split_hashes.items()),
        f"{label}: summary/split hashes differ",
    )

    selected_pd = max(rows, key=_pd_selection_key)
    selected_miou = max(rows, key=_miou_selection_key)
    _require(
        summary.get("best_pd_epoch") == selected_pd["epoch"],
        f"{label}: best_pd_epoch not globally selected",
    )
    _require(
        summary.get("best_epoch") == selected_pd["epoch"],
        f"{label}: best_epoch differs from best_pd_epoch",
    )
    _require(
        summary.get("best_miou_epoch") == selected_miou["epoch"],
        f"{label}: best_miou_epoch not globally selected",
    )
    for field, selected in (
        ("best_pd_validation_metrics", selected_pd),
        ("best_validation_metrics", selected_pd),
        ("best_miou_validation_metrics", selected_miou),
    ):
        metrics = summary.get(field)
        _require(isinstance(metrics, dict), f"{label}: summary.{field} missing")
        _require_metric_subset(selected, metrics, f"{label}.summary.{field}")

    checkpoint_fields = {
        "best_checkpoint": "best.pth.tar",
        "best_miou_checkpoint": "best_miou.pth.tar",
        "last_checkpoint": "last.pth.tar",
    }
    for key, basename in checkpoint_fields.items():
        value = summary.get(key)
        _require(
            isinstance(value, str) and Path(value).name == basename,
            f"{label}: summary.{key} mismatch",
        )


def _validate_sweep(
    run_dir: Path,
    sweep_path: Path,
    checkpoint_path: Path,
    role_name: str,
    expected_variant: str,
    expected_seed: int,
    expected_epoch: int,
    expected_metrics: Mapping[str, Any] | None,
    split_hashes: Mapping[str, str] | None,
    *,
    require_common_artifact_hashes: bool,
    allow_unavailable_budget_points: bool = False,
    label: str,
) -> dict[str, Any]:
    payload = _load_json(sweep_path, label)
    role_spec = ROLE_SPECS[role_name]
    expected = {
        "variant": expected_variant,
        "seed": expected_seed,
        "dataset": DATASET,
        "checkpoint_epoch": expected_epoch,
        "checkpoint_role": role_spec["checkpoint_role"],
        "official_test_accessed": False,
        "validation_count": EXPECTED_VAL_COUNT,
        "split_seed": EXPECTED_SPLIT_SEED,
    }
    for key, value in expected.items():
        _require(payload.get(key) == value, f"{label}: sweep {key} mismatch")
    _require_file(checkpoint_path, label)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    _require(
        payload.get("checkpoint_sha256") == checkpoint_sha256,
        f"{label}: checkpoint SHA-256 mismatch",
    )
    checkpoint_value = payload.get("checkpoint")
    _require(
        isinstance(checkpoint_value, str)
        and Path(checkpoint_value).name == checkpoint_path.name,
        f"{label}: sweep checkpoint path mismatch",
    )
    if split_hashes is not None:
        _require(
            payload.get("validation_split_sha256")
            == split_hashes["used_val_sha256"],
            f"{label}: validation split hash mismatch",
        )
    else:
        _require_sha256(
            payload.get("validation_split_sha256"),
            f"{label}.validation_split_sha256",
        )

    fixed = payload.get("fixed_threshold_0_5")
    try:
        _validate_point(fixed, f"{label}.fixed_threshold_0_5")
    except ValueError as exc:
        raise IncompleteArtifact(str(exc)) from exc
    _require(
        float(fixed.get("threshold", 0.5)) == 0.5,
        f"{label}: fixed threshold is not 0.5",
    )
    if expected_metrics is not None:
        _require_metric_subset(
            fixed, expected_metrics, f"{label}.fixed_threshold_0_5"
        )

    raw_points = payload.get("points")
    _require(
        isinstance(raw_points, list) and bool(raw_points),
        f"{label}: sweep points must be a non-empty list",
    )
    points: list[dict[str, Any]] = []
    previous_threshold = -math.inf
    for index, raw_point in enumerate(raw_points):
        point_label = f"{label}.points[{index}]"
        try:
            point = _validate_point(raw_point, point_label)
        except ValueError as exc:
            raise IncompleteArtifact(str(exc)) from exc
        threshold = point.get("threshold")
        _require(
            not isinstance(threshold, bool)
            and isinstance(threshold, (int, float))
            and math.isfinite(float(threshold))
            and 0.0 <= float(threshold) <= 1.0,
            f"{point_label}: threshold must be finite and inside [0, 1]",
        )
        _require(
            float(threshold) > previous_threshold,
            f"{label}: sweep point thresholds must be strictly increasing",
        )
        previous_threshold = float(threshold)
        tiny_pd = point.get("tiny_pd")
        _require(
            tiny_pd is None
            or (
                not isinstance(tiny_pd, bool)
                and isinstance(tiny_pd, (int, float))
                and math.isfinite(float(tiny_pd))
                and 0.0 <= float(tiny_pd) <= 1.0
            ),
            f"{point_label}: tiny_pd must be null or finite inside [0, 1]",
        )
        points.append(point)
    fixed_from_points = min(
        points,
        key=lambda point: abs(float(point["threshold"]) - 0.5),
    )
    _require(
        float(fixed_from_points["threshold"]) == 0.5,
        f"{label}: sweep points do not contain the exact threshold 0.5",
    )
    _require(
        fixed == fixed_from_points,
        f"{label}: fixed_threshold_0_5 does not match the threshold-0.5 point",
    )

    budgets = payload.get("best_points_under_fa_budget")
    _require(isinstance(budgets, dict), f"{label}: sweep budgets missing")
    _require(
        set(budgets) == set(BUDGET_KEYS),
        f"{label}: sweep budget keys must be {BUDGET_KEYS}",
    )
    for budget in BUDGET_KEYS:
        recomputed = _best_sweep_point_under_fa(points, float(budget))
        if budgets[budget] is None:
            _require(
                allow_unavailable_budget_points,
                f"{label}: budget {budget} is unavailable",
            )
            _require(
                recomputed is None,
                f"{label}: budget {budget} is null despite a feasible sweep point",
            )
            continue
        try:
            point = _validate_point(budgets[budget], f"{label}.budget.{budget}")
        except ValueError as exc:
            raise IncompleteArtifact(str(exc)) from exc
        _require(
            float(point["fa"]) <= float(budget) + 1e-15,
            f"{label}: budget {budget} exceeded by actual Fa {point['fa']}",
        )
        _require(
            recomputed is not None and point == recomputed,
            f"{label}: budget {budget} is not the exact optimum recomputed "
            "from sweep points",
        )

    audit = payload.get("audit")
    _require(isinstance(audit, dict), f"{label}: audit missing")
    _require(
        audit.get("expected_epochs") == EXPECTED_EPOCHS
        and audit.get("metrics_event_count") == EXPECTED_EPOCHS
        and audit.get("metrics_epoch_range") == [1, EXPECTED_EPOCHS],
        f"{label}: audit metrics completeness mismatch",
    )
    _require(
        audit.get("summary_status") == "complete"
        and audit.get("selection_source") == "internal_validation_only",
        f"{label}: audit summary/selection mismatch",
    )
    integrity = audit.get("integrity_checks_passed")
    _require(isinstance(integrity, dict), f"{label}: integrity checks missing")
    for key in REQUIRED_INTEGRITY_CHECKS:
        _require(
            integrity.get(key) is True,
            f"{label}: integrity check {key} did not pass",
        )
    artifact_hashes = audit.get("artifact_sha256")
    _require(isinstance(artifact_hashes, dict), f"{label}: artifact hashes missing")
    _require(
        artifact_hashes.get("checkpoint") == checkpoint_sha256,
        f"{label}: audited checkpoint hash mismatch",
    )
    if require_common_artifact_hashes:
        for filename in ("protocol.json", "split.json", "summary.json", "metrics.jsonl"):
            path = run_dir / filename
            _require_file(path, label)
            _require(
                artifact_hashes.get(filename) == sha256_file(path),
                f"{label}: audited {filename} hash mismatch",
            )
    fixed_audit = payload.get("fixed_threshold_0_5_checkpoint_audit")
    _require(
        isinstance(fixed_audit, dict)
        and float(fixed_audit.get("max_abs_non_strict_numeric_delta", math.inf))
        == 0.0,
        f"{label}: fixed-threshold checkpoint reproduction is not exact",
    )
    return {
        "checkpoint_epoch": expected_epoch,
        "checkpoint_role": role_spec["checkpoint_role"],
        "fixed_threshold_0_5": copy.deepcopy(fixed),
        "budgets": copy.deepcopy(budgets),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "sweep": str(sweep_path.resolve()),
        "sweep_sha256": sha256_file(sweep_path),
        "validation_split_sha256": payload["validation_split_sha256"],
        "integrity_checks_passed": copy.deepcopy(integrity),
    }


def _candidate_run_dir(root: Path, variant: str, seed: int) -> Path:
    return root / DATASET / variant / f"seed_{seed}_{RUN_TAG}"


def load_candidate_run(root: Path, variant: str, seed: int) -> dict[str, Any]:
    label = f"{variant}/seed={seed}"
    run_dir = _candidate_run_dir(root, variant, seed)
    _require(run_dir.is_dir() and not run_dir.is_symlink(), f"{label}: missing run directory {run_dir}")
    protocol_path = run_dir / "protocol.json"
    split_path = run_dir / "split.json"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.jsonl"
    protocol = _load_json(protocol_path, label)
    split = _load_json(split_path, label)
    summary = _load_json(summary_path, label)
    rows = _read_metrics(metrics_path, variant, label)
    _validate_protocol(protocol, variant, seed, label)
    split_hashes = _validate_split(split, label)
    _validate_summary(
        summary, protocol, split_hashes, rows, variant, seed, label
    )

    roles: dict[str, Any] = {}
    for role_name, spec in ROLE_SPECS.items():
        checkpoint_path = run_dir / spec["checkpoint"]
        sweep_path = run_dir / spec["sweep"]
        expected_epoch = int(summary[spec["summary_epoch"]])
        expected_metrics = summary[spec["summary_metrics"]]
        roles[role_name] = _validate_sweep(
            run_dir,
            sweep_path,
            checkpoint_path,
            role_name,
            variant,
            seed,
            expected_epoch,
            expected_metrics,
            split_hashes,
            require_common_artifact_hashes=True,
            label=f"{label}/{role_name}",
        )

    last_path = run_dir / "last.pth.tar"
    _require_file(last_path, label)
    return {
        "variant": variant,
        "seed": seed,
        "run_directory": str(run_dir.resolve()),
        "roles": roles,
        "model": copy.deepcopy(summary["model"]),
        "split_hashes": split_hashes,
        "metrics_event_count": len(rows),
        "artifacts": {
            "protocol.json": sha256_file(protocol_path),
            "split.json": sha256_file(split_path),
            "summary.json": sha256_file(summary_path),
            "metrics.jsonl": sha256_file(metrics_path),
            "last.pth.tar": sha256_file(last_path),
        },
    }


def _reference_paths(
    formal_root: Path,
    v2_root: Path,
    miou_root: Path,
) -> dict[str, dict[str, tuple[Path, str]]]:
    formal_tag = "seed_42_formal800_pd_fp32_4x5090_v1"
    v2_tag = "seed_42_screen800_pd_fp32_shared4x5090_v1"
    return {
        "spd": {
            "pd_primary": (
                formal_root / DATASET / "spd" / formal_tag,
                "spd",
            ),
            "miou_primary": (
                miou_root / DATASET / "spd" / formal_tag,
                "spd",
            ),
        },
        "tpd_v1": {
            "pd_primary": (
                formal_root / DATASET / "tpd" / formal_tag,
                "tpd",
            ),
            "miou_primary": (
                miou_root / DATASET / "tpd" / formal_tag,
                "tpd",
            ),
        },
        "v2_sal_only": {
            role: (
                v2_root / DATASET / "tpd_clean_sal" / v2_tag,
                "tpd_clean_sal",
            )
            for role in ROLE_SPECS
        },
        "v2_full": {
            role: (
                v2_root / DATASET / "tpd_clean_full" / v2_tag,
                "tpd_clean_full",
            )
            for role in ROLE_SPECS
        },
    }


def load_frozen_references(
    formal_root: Path,
    v2_root: Path,
    miou_root: Path,
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for name, roles in _reference_paths(
        formal_root, v2_root, miou_root
    ).items():
        references[name] = {"seed": 42, "roles": {}}
        for role_name, (run_dir, variant) in roles.items():
            spec = ROLE_SPECS[role_name]
            sweep_path = run_dir / spec["sweep"]
            checkpoint_path = run_dir / spec["checkpoint"]
            payload = _load_json(
                sweep_path, f"frozen_reference/{name}/{role_name}"
            )
            expected_epoch = payload.get("checkpoint_epoch")
            _require(
                isinstance(expected_epoch, int),
                f"frozen_reference/{name}/{role_name}: checkpoint epoch missing",
            )
            references[name]["roles"][role_name] = _validate_sweep(
                run_dir,
                sweep_path,
                checkpoint_path,
                role_name,
                variant,
                42,
                expected_epoch,
                None,
                None,
                require_common_artifact_hashes=False,
                allow_unavailable_budget_points=True,
                label=f"frozen_reference/{name}/{role_name}",
            )
    return references


def _json_candidate_runs(
    runs: Mapping[tuple[str, int], Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        f"{variant}/seed_{seed}": copy.deepcopy(record)
        for (variant, seed), record in sorted(
            runs.items(), key=lambda item: (item[0][1], item[0][0])
        )
    }


def build_report(
    candidate_root: Path,
    formal_reference_root: Path,
    v2_reference_root: Path,
    reference_miou_root: Path,
) -> dict[str, Any]:
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    candidate_runs: dict[tuple[str, int], dict[str, Any]] = {}
    reasons: list[str] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            try:
                candidate_runs[(variant, seed)] = load_candidate_run(
                    candidate_root, variant, seed
                )
            except (IncompleteArtifact, ValueError) as exc:
                reasons.append(f"{variant}/seed={seed}: {exc}")

    references: dict[str, dict[str, Any]] = {}
    try:
        references = load_frozen_references(
            formal_reference_root,
            v2_reference_root,
            reference_miou_root,
        )
    except (IncompleteArtifact, ValueError) as exc:
        reasons.append(f"frozen references: {exc}")
    reference_unavailable_points = _reference_unavailable_points(references)

    common = {
        "schema": SCHEMA,
        "generated_at_utc": generated_at,
        "status": "incomplete" if reasons else "complete",
        "scope": {
            "dataset": DATASET,
            "candidate_variants": list(VARIANTS),
            "model_seeds": list(SEEDS),
            "epochs": EXPECTED_EPOCHS,
            "split_seed": EXPECTED_SPLIT_SEED,
            "official_test_accessed": False,
            "candidate_root": str(candidate_root.resolve()),
            "formal_reference_root": str(formal_reference_root.resolve()),
            "v2_reference_root": str(v2_reference_root.resolve()),
            "reference_miou_root": str(reference_miou_root.resolve()),
            "protocol": str(
                (REPO_ROOT / "experiments/TPD_CLEAN_V3_PROTOCOL.md").resolve()
            ),
        },
        "candidate_runs": _json_candidate_runs(candidate_runs),
        "reference_unavailable_points": reference_unavailable_points,
        "gate_evaluated": False,
        "engineering_gate_passed": None,
        "mainline_changed": False,
        "paper_core_established": False,
        "stability_claim_supported": False,
    }
    if reasons:
        return {
            **common,
            "incomplete_reasons": reasons,
            "required_before_gate_evaluation": {
                "candidate_runs": 4,
                "candidate_checkpoints": 8,
                "candidate_sweeps": 8,
                "metrics_rows_per_run": 800,
                "frozen_reference_methods": [
                    "spd",
                    "tpd_v1",
                    "v2_sal_only",
                    "v2_full",
                ],
            },
            "decision": (
                "INCOMPLETE: no engineering-gate conclusion is computed "
                "from missing or invalid artifacts."
            ),
        }

    full_initialization_pairs: dict[str, Any] = {}
    shared_initialization_pairs: dict[str, Any] = {}
    split_fingerprints = set()
    for seed in SEEDS:
        full_model = candidate_runs[(PRIMARY_VARIANT, seed)]["model"]
        control_model = candidate_runs[(CONTROL_VARIANT, seed)]["model"]
        full_equal = (
            full_model["full_initialization_sha256"]
            == control_model["full_initialization_sha256"]
        )
        if not full_equal:
            raise IncompleteArtifact(
                f"seed={seed}: paired candidate initialization differs"
            )
        full_initialization_pairs[str(seed)] = {
            "equal": True,
            "sha256": full_model["full_initialization_sha256"],
        }
        shared_equal = (
            full_model["shared_initialization_sha256"]
            == control_model["shared_initialization_sha256"]
        )
        if not shared_equal:
            raise IncompleteArtifact(
                f"seed={seed}: paired non-shallow initialization differs"
            )
        shared_initialization_pairs[str(seed)] = {
            "equal": True,
            "sha256": full_model["shared_initialization_sha256"],
        }
    for record in candidate_runs.values():
        split_fingerprints.add(
            (
                record["split_hashes"]["used_train_sha256"],
                record["split_hashes"]["used_val_sha256"],
            )
        )
    if len(split_fingerprints) != 1:
        raise IncompleteArtifact("candidate train/validation splits differ")
    candidate_validation_sha256 = next(iter(split_fingerprints))[1]
    for method, reference in references.items():
        for role_name, role in reference["roles"].items():
            if (
                role["validation_split_sha256"]
                != candidate_validation_sha256
            ):
                raise IncompleteArtifact(
                    "frozen reference validation split differs from candidates: "
                    f"{method}.{role_name}"
                )

    try:
        gate = evaluate_engineering_gate(candidate_runs, references)
    except (IncompleteArtifact, ValueError) as exc:
        return {
            **common,
            "status": "incomplete",
            "incomplete_reasons": [f"engineering gate inputs: {exc}"],
            "required_before_gate_evaluation": {
                "candidate_runs": 4,
                "candidate_checkpoints": 8,
                "candidate_sweeps": 8,
                "metrics_rows_per_run": 800,
                "frozen_reference_methods": [
                    "spd",
                    "tpd_v1",
                    "v2_sal_only",
                    "v2_full",
                ],
            },
            "decision": (
                "INCOMPLETE: no engineering-gate conclusion is computed "
                "from missing or invalid artifacts."
            ),
        }
    return {
        **common,
        "status": "complete",
        "frozen_references": copy.deepcopy(references),
        "engineering_gate": gate,
        "gate_evaluated": True,
        "engineering_gate_passed": bool(gate["passed"]),
        "validation": {
            "candidate_run_count": 4,
            "candidate_checkpoint_count": 8,
            "candidate_sweep_count": 8,
            "candidate_metrics_event_count": sum(
                record["metrics_event_count"]
                for record in candidate_runs.values()
            ),
            "metrics_epochs_complete": True,
            "candidate_sweeps_integrity_checked": True,
            "frozen_reference_sweeps_integrity_checked": True,
            "paired_full_initialization_sha256": full_initialization_pairs,
            "paired_shared_non_shallow_initialization_sha256": (
                shared_initialization_pairs
            ),
            "paired_split_fingerprint_equal": True,
        },
        "decision": (
            "ENGINEERING_GATE_PASS"
            if gate["passed"]
            else "ENGINEERING_GATE_FAIL"
        ),
        "decision_boundary": {
            "gate_only_controls_next_engineering_stage": True,
            "automatic_mainline_replacement": False,
            "requires_at_least_three_paired_seeds": True,
            "requires_more_datasets": True,
            "mainline_changed": False,
            "paper_core_established": False,
            "stability_claim_supported": False,
        },
    }


def _metric_cell(point: Mapping[str, Any]) -> str:
    return (
        f"{int(point['matched_target_count'])}/{int(point['target_count'])}; "
        f"Fa={float(point['fa']):.8g}; "
        f"mIoU={float(point['miou']):.6f}"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# TPD-Clean-v3 screen800 comparison",
        "",
        (
            "Status: **INCOMPLETE**"
            if report["status"] == "incomplete"
            else f"Status: **{report['decision']}**"
        ),
        "",
        "NUDT-SIRST official-training-index internal 530/133 split only; "
        "official test data were not accessed.",
        "",
    ]
    if report["status"] == "incomplete":
        lines.extend(
            [
                "## Incomplete artifacts",
                "",
                "工程门槛未计算；缺失字段或结果不会被默认值替代。",
                "",
            ]
        )
        for reason in report.get("incomplete_reasons", []):
            lines.append(f"- {reason}")
        lines.extend(
            [
                "",
                "## Decision boundary",
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
            "## Candidate fixed-threshold results",
            "",
            "| Seed | Variant | Pd-primary | mIoU-primary |",
            "| ---: | --- | --- | --- |",
        ]
    )
    candidate_runs = report["candidate_runs"]
    for seed in SEEDS:
        for variant in VARIANTS:
            record = candidate_runs[f"{variant}/seed_{seed}"]
            pd_point = record["roles"]["pd_primary"]["fixed_threshold_0_5"]
            miou_point = record["roles"]["miou_primary"]["fixed_threshold_0_5"]
            lines.append(
                f"| {seed} | `{variant}` | {_metric_cell(pd_point)} | "
                f"{_metric_cell(miou_point)} |"
            )

    lines.extend(
        [
            "",
            "## Seed-42 frozen references",
            "",
            "| Method | Pd-primary | mIoU-primary |",
            "| --- | --- | --- |",
        ]
    )
    for name in ("spd", "tpd_v1", "v2_sal_only", "v2_full"):
        reference = report["frozen_references"][name]
        lines.append(
            f"| `{name}` | "
            f"{_metric_cell(reference['roles']['pd_primary']['fixed_threshold_0_5'])} | "
            f"{_metric_cell(reference['roles']['miou_primary']['fixed_threshold_0_5'])} |"
        )

    unavailable_points = report.get("reference_unavailable_points", [])
    if unavailable_points:
        lines.extend(
            [
                "",
                "### Explicitly unavailable frozen-reference budget points",
                "",
                "| Method | Role | Fa budget | Gate use |",
                "| --- | --- | ---: | --- |",
            ]
        )
        for point in unavailable_points:
            lines.append(
                f"| `{point['method']}` | `{point['role']}` | "
                f"`{point['budget']}` | `{point['gate_usage']}` |"
            )

    lines.extend(
        [
            "",
            "## Protocol section 5 engineering gate",
            "",
            "| Check | Result |",
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
            "- `mainline_changed=false`",
            "- `paper_core_established=false`",
            "- `stability_claim_supported=false`",
            "",
            "即使工程门槛通过，两个 seed 和一个内部验证划分也不能替换现有 "
            "TPD-v1 主线；仍需至少三个配对 seed、更多数据集和既定统计确认。",
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
    json_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    markdown_text = render_markdown(report)
    _atomic_write(json_path, json_text)
    _atomic_write(markdown_path, markdown_text)
    return json_path, markdown_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit four TPD-Clean-v3 screen800 runs and evaluate the frozen "
            "post-training engineering gate only when every artifact exists."
        )
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT,
    )
    parser.add_argument(
        "--formal-reference-root",
        type=Path,
        default=DEFAULT_FORMAL_REFERENCE_ROOT,
    )
    parser.add_argument(
        "--v2-reference-root",
        type=Path,
        default=DEFAULT_V2_REFERENCE_ROOT,
    )
    parser.add_argument(
        "--reference-miou-root",
        type=Path,
        default=DEFAULT_REFERENCE_MIOU_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(
            args.candidate_root,
            args.formal_reference_root,
            args.v2_reference_root,
            args.reference_miou_root,
        )
    except (IncompleteArtifact, ValueError) as exc:
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
            "decision": (
                "INCOMPLETE: no engineering-gate conclusion is computed "
                "from missing or invalid artifacts."
            ),
        }
    json_path, markdown_path = write_report(
        report, args.output_dir, overwrite=args.overwrite
    )
    print(
        "TPDCLEANV3_SUMMARY"
        f" status={report['status']}"
        f" gate={report['engineering_gate_passed']}"
        f" json={json_path}"
        f" markdown={markdown_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
