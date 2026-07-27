#!/usr/bin/env python3
"""Supplemental strict validation for the eight formal V6 Pd--Fa sweeps.

The frozen formal summarizer remains unchanged.  This independent validator
adds two fail-closed checks before any V6 Gate A--E report is accepted:

1. the complete preregistered threshold configuration and grid provenance;
2. tiny-Pd and object-count identities for every stored operating point.

It is read-only and can be run in ``--preflight`` mode while training or
sweeps are incomplete.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_pd_fa_sweep as sweep_base  # noqa: E402
from experiments import summarize_tpd_clean_v6_formal800 as summary  # noqa: E402


EXPECTED_THRESHOLD_CONFIGURATION = {
    "threshold_min": 0.01,
    "threshold_max": 0.99,
    "threshold_step": 0.01,
    "extra_thresholds": [0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999],
    "tail_logit_step": 0.1,
    "fa_budgets": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
}
EXPECTED_VALIDATION_COUNT = 133
EXPECTED_VALID_PIXEL_COUNT = 8_716_288
EXPECTED_TARGET_COUNT = 189
EXPECTED_TINY_TARGET_COUNT = 39
EXPECTED_QUANTILE_LEVELS = [
    0.9,
    0.95,
    0.98,
    0.99,
    0.995,
    0.999,
    0.9995,
    0.9999,
    0.99995,
    0.99999,
    0.999995,
    0.999999,
]
EXPECTED_QUANTILE_KEYS = [
    f"{level:.9g}" for level in EXPECTED_QUANTILE_LEVELS
]
LAST_FLOAT32_BELOW_ONE = float(
    np.nextafter(np.float32(1.0), np.float32(0.0))
)


class StrictSweepValidationError(ValueError):
    """Raised when a formal sweep violates the supplemental contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StrictSweepValidationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing sweep: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"sweep is not an object: {path}")
    return payload


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _float_equal(left: Any, right: Any, *, atol: float = 1e-15) -> bool:
    return _finite_number(left) and _finite_number(right) and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=atol
    )


def _exact_float_sequence(value: Any, expected: Sequence[float]) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(expected)
        and all(_float_equal(item, target) for item, target in zip(value, expected))
    )


def _expected_base_thresholds() -> list[float]:
    return sweep_base.threshold_grid(
        EXPECTED_THRESHOLD_CONFIGURATION["threshold_min"],
        EXPECTED_THRESHOLD_CONFIGURATION["threshold_max"],
        EXPECTED_THRESHOLD_CONFIGURATION["threshold_step"],
        EXPECTED_THRESHOLD_CONFIGURATION["extra_thresholds"],
    )


def _expected_tail_thresholds() -> tuple[list[float], list[float]]:
    lower_probability = 0.95
    upper_probability = 0.9999
    lower_logit = math.log(lower_probability / (1.0 - lower_probability))
    upper_logit = math.log(upper_probability / (1.0 - upper_probability))
    step = EXPECTED_THRESHOLD_CONFIGURATION["tail_logit_step"]
    logit_values = np.arange(
        lower_logit, upper_logit + step / 2, step
    )
    thresholds = [
        float(1.0 / (1.0 + math.exp(-float(value))))
        for value in logit_values
    ]
    return thresholds, [lower_logit, upper_logit]


def _contains_threshold(
    observed: Sequence[float], expected: float, *, atol: float = 1e-15
) -> bool:
    return any(
        math.isclose(value, expected, rel_tol=0.0, abs_tol=atol)
        for value in observed
    )


def validate_point_identities(point: Any, label: str) -> None:
    _require(isinstance(point, Mapping), f"{label}: point is not an object")
    for key in ("threshold", "pd", "tiny_pd", "fa", "miou"):
        _require(
            _finite_number(point.get(key)),
            f"{label}: {key} must be finite numeric",
        )
    for key in (
        "val_loss",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "false_objects_per_image",
    ):
        _require(
            _finite_number(point.get(key)),
            f"{label}: {key} must be finite numeric",
        )
    _require(
        0.0 <= float(point["threshold"]) <= 1.0,
        f"{label}: threshold range differs",
    )
    for key in ("pd", "tiny_pd", "miou", "fa"):
        _require(
            0.0 <= float(point[key]) <= 1.0,
            f"{label}: {key} range differs",
        )
    for key in ("niou", "pixel_precision", "pixel_recall", "pixel_f1"):
        _require(
            0.0 <= float(point[key]) <= 1.0,
            f"{label}: {key} range differs",
        )
    _require(float(point["val_loss"]) >= 0.0, f"{label}: negative val loss")
    _require(
        float(point["false_objects_per_image"]) >= 0.0,
        f"{label}: negative false-objects/image",
    )
    for key, expected in (
        ("target_count", EXPECTED_TARGET_COUNT),
        ("tiny_target_count", EXPECTED_TINY_TARGET_COUNT),
        ("valid_pixel_count", EXPECTED_VALID_PIXEL_COUNT),
    ):
        value = point.get(key)
        _require(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value == expected,
            f"{label}: {key} differs",
        )
    for key in (
        "matched_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
    ):
        value = point.get(key)
        _require(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0,
            f"{label}: {key} is invalid",
        )
    matched = int(point["matched_target_count"])
    matched_tiny = int(point["matched_tiny_target_count"])
    predicted = int(point["predicted_object_count"])
    unmatched = int(point["unmatched_predicted_object_count"])
    _require(matched <= EXPECTED_TARGET_COUNT, f"{label}: matched target overflow")
    _require(
        matched_tiny <= EXPECTED_TINY_TARGET_COUNT,
        f"{label}: matched tiny-target overflow",
    )
    _require(
        matched_tiny <= matched,
        f"{label}: matched tiny targets exceed all matched targets",
    )
    _require(
        matched - matched_tiny
        <= EXPECTED_TARGET_COUNT - EXPECTED_TINY_TARGET_COUNT,
        f"{label}: matched non-tiny targets exceed available non-tiny targets",
    )
    _require(predicted >= matched, f"{label}: predicted objects below matches")
    _require(
        unmatched == predicted - matched,
        f"{label}: unmatched object identity differs",
    )
    _require(
        _float_equal(point.get("pd"), matched / EXPECTED_TARGET_COUNT),
        f"{label}: Pd/count identity differs",
    )
    _require(
        _float_equal(
            point.get("tiny_pd"),
            matched_tiny / EXPECTED_TINY_TARGET_COUNT,
        ),
        f"{label}: tiny-Pd/count identity differs",
    )
    _require(
        _float_equal(
            point.get("false_objects_per_image"),
            unmatched / EXPECTED_VALIDATION_COUNT,
            atol=1e-12,
        ),
        f"{label}: false-objects/image identity differs",
    )
    false_pixel_count = float(point["fa"]) * EXPECTED_VALID_PIXEL_COUNT
    _require(
        math.isclose(
            false_pixel_count,
            round(false_pixel_count),
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        f"{label}: Fa is not on the valid-pixel lattice",
    )
    false_pixel_count_int = int(round(false_pixel_count))
    _require(
        false_pixel_count_int >= unmatched,
        f"{label}: unmatched-object count exceeds unmatched pixels",
    )
    _require(
        (unmatched == 0) == (false_pixel_count_int == 0),
        f"{label}: Fa/object emptiness identity differs",
    )


def validate_threshold_contract(payload: Mapping[str, Any], label: str) -> None:
    configuration = payload.get("threshold_configuration")
    _require(
        isinstance(configuration, Mapping)
        and set(configuration) == set(EXPECTED_THRESHOLD_CONFIGURATION),
        f"{label}: threshold configuration key set differs",
    )
    for key in ("threshold_min", "threshold_max", "threshold_step", "tail_logit_step"):
        _require(
            _float_equal(
                configuration.get(key), EXPECTED_THRESHOLD_CONFIGURATION[key]
            ),
            f"{label}: {key} differs",
        )
    for key in ("extra_thresholds", "fa_budgets"):
        _require(
            _exact_float_sequence(
                configuration.get(key), EXPECTED_THRESHOLD_CONFIGURATION[key]
            ),
            f"{label}: {key} differs",
        )

    audit = payload.get("audit")
    parsed = audit.get("parsed_arguments") if isinstance(audit, Mapping) else None
    _require(isinstance(parsed, Mapping), f"{label}: parsed arguments missing")
    for key in ("threshold_min", "threshold_max", "threshold_step", "tail_logit_step"):
        _require(
            _float_equal(parsed.get(key), EXPECTED_THRESHOLD_CONFIGURATION[key]),
            f"{label}: parsed {key} differs",
        )
    for key in ("extra_thresholds", "fa_budgets"):
        _require(
            _exact_float_sequence(
                parsed.get(key), EXPECTED_THRESHOLD_CONFIGURATION[key]
            ),
            f"{label}: parsed {key} differs",
        )
    _require(parsed.get("overwrite") is False, f"{label}: overwrite was enabled")
    _require(
        parsed.get("expected_epochs") == summary.EXPECTED_EPOCHS,
        f"{label}: parsed epoch contract differs",
    )
    invocation = audit.get("invocation_argv")
    _require(
        isinstance(invocation, list) and "--overwrite" not in invocation,
        f"{label}: overwrite invocation differs",
    )

    points = payload.get("points")
    _require(isinstance(points, list) and points, f"{label}: points missing")
    thresholds: list[float] = []
    for index, point in enumerate(points):
        _require(
            isinstance(point, Mapping) and _finite_number(point.get("threshold")),
            f"{label}: invalid threshold at point {index}",
        )
        thresholds.append(float(point["threshold"]))
    _require(
        all(left < right for left, right in zip(thresholds, thresholds[1:])),
        f"{label}: thresholds are not strictly increasing and unique",
    )

    provenance = payload.get("threshold_provenance")
    _require(isinstance(provenance, Mapping), f"{label}: provenance missing")
    base_thresholds = _expected_base_thresholds()
    tail_thresholds, tail_logit_range = _expected_tail_thresholds()
    _require(
        provenance.get("uniform_probability_grid_count") == len(base_thresholds),
        f"{label}: uniform grid count differs",
    )
    _require(
        _exact_float_sequence(
            provenance.get("tail_logit_range"), tail_logit_range
        ),
        f"{label}: tail logit range differs",
    )
    _require(
        _float_equal(
            provenance.get("tail_logit_step"),
            EXPECTED_THRESHOLD_CONFIGURATION["tail_logit_step"],
        )
        and provenance.get("tail_logit_threshold_count") == len(tail_thresholds),
        f"{label}: tail grid provenance differs",
    )
    _require(
        provenance.get("total_unique_threshold_count") == len(points),
        f"{label}: total threshold count differs",
    )
    _require(
        provenance.get("score_count") == EXPECTED_VALID_PIXEL_COUNT,
        f"{label}: score count differs",
    )
    _require(
        _exact_float_sequence(
            provenance.get("added_thresholds"),
            [LAST_FLOAT32_BELOW_ONE, 1.0],
        ),
        f"{label}: closed-interval endpoints differ",
    )
    quantiles = provenance.get("empirical_score_quantiles")
    _require(isinstance(quantiles, Mapping), f"{label}: quantiles missing")
    _require(
        set(quantiles).issubset(set(EXPECTED_QUANTILE_KEYS)),
        f"{label}: quantile key set differs",
    )
    exact_one_score_count = provenance.get("exact_one_score_count")
    _require(
        isinstance(exact_one_score_count, int)
        and not isinstance(exact_one_score_count, bool)
        and 0 <= exact_one_score_count <= EXPECTED_VALID_PIXEL_COUNT,
        f"{label}: exact-one score count differs",
    )
    observed_quantile_keys = sorted(quantiles, key=float)
    if observed_quantile_keys:
        start = EXPECTED_QUANTILE_KEYS.index(observed_quantile_keys[0])
        end = EXPECTED_QUANTILE_KEYS.index(observed_quantile_keys[-1])
        _require(
            observed_quantile_keys
            == EXPECTED_QUANTILE_KEYS[start : end + 1],
            f"{label}: quantile keys are not a contiguous registered slice",
        )
        one_plateau_start = (
            (
                EXPECTED_VALID_PIXEL_COUNT - exact_one_score_count
            )
            / (EXPECTED_VALID_PIXEL_COUNT - 1)
            if exact_one_score_count > 0
            else math.inf
        )
        _require(
            all(
                level < one_plateau_start
                for level in EXPECTED_QUANTILE_LEVELS[start : end + 1]
            ),
            f"{label}: unit quantile was incorrectly retained",
        )
        omitted_suffix_levels = EXPECTED_QUANTILE_LEVELS[end + 1 :]
        if omitted_suffix_levels:
            _require(
                exact_one_score_count > 0,
                f"{label}: quantile suffix omitted without exact-one scores",
            )
            _require(
                all(
                    level >= one_plateau_start
                    for level in omitted_suffix_levels
                ),
                f"{label}: non-unit quantile suffix is missing",
            )
    ordered_quantiles = sorted(
        ((float(key), float(value)) for key, value in quantiles.items()),
        key=lambda item: item[0],
    )
    _require(
        all(
            left[1] <= right[1]
            for left, right in zip(ordered_quantiles, ordered_quantiles[1:])
        ),
        f"{label}: empirical quantiles are not monotone",
    )
    for key, value in quantiles.items():
        _require(
            _finite_number(value)
            and 0.0 < float(value) < 1.0
            and _contains_threshold(thresholds, float(value)),
            f"{label}: quantile threshold missing for {key}",
        )
    expected_thresholds = sorted(
        {
            *base_thresholds,
            *tail_thresholds,
            *[float(value) for value in quantiles.values()],
            LAST_FLOAT32_BELOW_ONE,
            1.0,
        }
    )
    _require(
        thresholds == expected_thresholds,
        f"{label}: threshold sequence differs from the registered union",
    )


def validate_sweep_payload(payload: Mapping[str, Any], label: str) -> None:
    _require(
        payload.get("validation_count") == EXPECTED_VALIDATION_COUNT,
        f"{label}: validation count differs",
    )
    _require(_float_equal(payload.get("match_radius"), 3.0), f"{label}: radius")
    _require(payload.get("tiny_area") == 9, f"{label}: tiny area differs")
    validate_threshold_contract(payload, label)
    points = payload["points"]
    for index, point in enumerate(points):
        validate_point_identities(point, f"{label}/point[{index}]")
    fixed = payload.get("fixed_threshold_0_5")
    validate_point_identities(fixed, f"{label}/fixed_threshold_0_5")
    _require(
        _float_equal(fixed.get("threshold"), 0.5),
        f"{label}: fixed threshold is not 0.5",
    )
    _require(fixed in points, f"{label}: fixed threshold point is absent")
    budgets = payload.get("best_points_under_fa_budget")
    expected_budget_keys = {
        f"{budget:.10g}"
        for budget in EXPECTED_THRESHOLD_CONFIGURATION["fa_budgets"]
    }
    _require(
        isinstance(budgets, Mapping)
        and set(budgets) == expected_budget_keys,
        f"{label}: budget point key set differs",
    )
    for budget_key in sorted(expected_budget_keys, key=float):
        point = budgets[budget_key]
        validate_point_identities(point, f"{label}/budget[{budget_key}]")
        _require(point in points, f"{label}: budget point is absent from points")
        feasible = [
            candidate
            for candidate in points
            if float(candidate["fa"]) <= float(budget_key)
        ]
        _require(feasible, f"{label}: budget {budget_key} has no feasible point")
        optimum = max(
            feasible,
            key=lambda candidate: (
                float(candidate["pd"]),
                -float(candidate["fa"]),
                float(candidate["tiny_pd"]),
                float(candidate["miou"]),
                -abs(float(candidate["threshold"]) - 0.5),
            ),
        )
        _require(
            point == optimum,
            f"{label}: budget {budget_key} is not the registered optimum",
        )


def expected_sweep_jobs(candidate_root: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in summary.SEEDS:
        for variant in summary.VARIANTS:
            run_dir = (
                candidate_root
                / summary.DATASET
                / variant
                / f"seed_{seed}_{summary.RUN_TAG}"
            )
            for role_name, spec in summary.ROLE_SPECS.items():
                jobs.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "role": role_name,
                        "run_dir": run_dir,
                        "path": run_dir / spec["sweep"],
                    }
                )
    return jobs


def inspect_strict_sweeps(candidate_root: Path) -> dict[str, Any]:
    jobs = expected_sweep_jobs(candidate_root)
    results: list[dict[str, Any]] = []
    for job in jobs:
        path = job["path"]
        status = "missing"
        error: str | None = None
        if path.is_file() and not path.is_symlink():
            try:
                validate_sweep_payload(
                    _load_json(path),
                    f"{job['variant']}/seed={job['seed']}/{job['role']}",
                )
                status = "strict_valid"
            except Exception as exc:  # report-only preflight boundary
                status = "invalid"
                error = f"{type(exc).__name__}: {exc}"
        elif path.exists() or path.is_symlink():
            status = "invalid"
            error = "sweep path is not a regular file"
        results.append(
            {
                "variant": job["variant"],
                "seed": job["seed"],
                "role": job["role"],
                "path": str(path.resolve()),
                "status": status,
                "error": error,
            }
        )
    valid = sum(item["status"] == "strict_valid" for item in results)
    missing = sum(item["status"] == "missing" for item in results)
    invalid = sum(item["status"] == "invalid" for item in results)
    return {
        "schema": "sctransnet_tpd_clean_v6_strict_sweep_validation_v1",
        "candidate_root": str(candidate_root.resolve()),
        "expected_sweeps": len(results),
        "strict_valid_sweeps": valid,
        "missing_sweeps": missing,
        "invalid_sweeps": invalid,
        "complete_and_strict_valid": valid == len(results),
        "results": results,
    }


def validate_all_strict_sweeps(candidate_root: Path) -> dict[str, Any]:
    training_lock, _ = summary._validate_current_training_contract()
    summary.validate_postprocess_source_lock()
    evaluator_sha = training_lock["source_sha256"][
        "experiments/evaluate_tpd_clean_v6_pd_fa.py"
    ]
    for job in expected_sweep_jobs(candidate_root):
        summary.validate_existing_sweep(
            job["run_dir"],
            variant=job["variant"],
            seed=job["seed"],
            role_name=job["role"],
            evaluator_sha256=evaluator_sha,
        )
        validate_sweep_payload(
            _load_json(job["path"]),
            f"{job['variant']}/seed={job['seed']}/{job['role']}",
        )
    report = inspect_strict_sweeps(candidate_root)
    _require(
        report["complete_and_strict_valid"] is True,
        "strict V6 sweep matrix is incomplete or invalid",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly validate the eight formal V6 Pd--Fa sweeps"
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=summary.DEFAULT_CANDIDATE_ROOT,
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Report missing/invalid sweeps without requiring all eight",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_root = args.candidate_root.resolve()
    report = (
        inspect_strict_sweeps(candidate_root)
        if args.preflight
        else validate_all_strict_sweeps(candidate_root)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.preflight and report["complete_and_strict_valid"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
