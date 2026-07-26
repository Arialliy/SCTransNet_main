#!/usr/bin/env python3
"""Audit the eight post-hoc closed-interval TPD-Clean-v3 sweeps."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = Path(
    "experiments/results/tpd_clean_v3_screen800_4x5090_v1"
)
RUN_TAG = "screen800_pd_fp32_shared4x5090_v1"
LAST_FLOAT32_BELOW_ONE = 0.9999999403953552
UPPER_BOUNDARY_THRESHOLD = 1.0
BUDGETS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")
ROLES = {
    "pd_primary": ("best.pth.tar", "pd_fa_sweep_best.pth.json"),
    "miou_primary": (
        "best_miou.pth.tar",
        "pd_fa_sweep_best_miou.pth.json",
    ),
}
JOBS = (
    ("tpd_clean_v3_full", 42),
    ("tpd_clean_v3_sal_capacity", 42),
    ("tpd_clean_v3_full", 3407),
    ("tpd_clean_v3_sal_capacity", 3407),
)


class RecoveryAuditError(ValueError):
    """A closed-interval recovery invariant failed."""


def _fail(message: str) -> None:
    raise RecoveryAuditError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        _fail(f"missing or linked JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        _fail(f"expected JSON object: {path}")
    return payload


def _best_point(
    points: Sequence[Mapping[str, Any]], budget: float
) -> Mapping[str, Any] | None:
    feasible = [point for point in points if float(point["fa"]) <= budget]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda point: (
            float(point["pd"]),
            -float(point["fa"]),
            (
                float(point["tiny_pd"])
                if point.get("tiny_pd") is not None
                else -1.0
            ),
            float(point["miou"]),
            -abs(float(point["threshold"]) - 0.5),
        ),
    )


def _backup_name(variant: str, seed: int, role: str) -> str:
    return f"{variant}_seed{seed}_{role}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit all TPD-Clean-v3 closed-interval sweep recoveries"
    )
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--candidate-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=(
            RESULT_ROOT
            / "resume_2x5090_v1"
            / "recovery_threshold1_20260726_113926"
            / "original"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optionally create a persistent JSON audit report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve(strict=True)
    candidate_root = (
        args.candidate_root
        if args.candidate_root.is_absolute()
        else repo / args.candidate_root
    ).resolve(strict=True)
    backup_root = (
        args.backup_root
        if args.backup_root.is_absolute()
        else repo / args.backup_root
    ).resolve(strict=True)
    wrapper = (
        repo / "experiments/evaluate_tpd_clean_v3_pd_fa_closed_interval.py"
    ).resolve(strict=True)
    wrapper_sha256 = _sha256(wrapper)
    records: list[dict[str, Any]] = []

    for variant, seed in JOBS:
        run_dir = (
            candidate_root
            / "NUDT-SIRST"
            / variant
            / f"seed_{seed}_{RUN_TAG}"
        )
        for role, (checkpoint_name, sweep_name) in ROLES.items():
            label = f"{variant}/seed={seed}/{role}"
            current_path = run_dir / sweep_name
            backup_path = backup_root / _backup_name(variant, seed, role)
            checkpoint_path = run_dir / checkpoint_name
            current = _load(current_path)
            original = _load(backup_path)

            unchanged_keys = (
                "checkpoint",
                "checkpoint_sha256",
                "checkpoint_epoch",
                "checkpoint_role",
                "checkpoint_validation_metrics",
                "variant",
                "dataset",
                "seed",
                "split_seed",
                "validation_count",
                "validation_split_sha256",
                "official_test_accessed",
                "match_radius",
                "tiny_area",
                "threshold_configuration",
                "fixed_threshold_0_5",
                "fixed_threshold_0_5_checkpoint_audit",
            )
            for key in unchanged_keys:
                if current.get(key) != original.get(key):
                    _fail(f"{label}: recovery changed {key}")
            if current.get("checkpoint_sha256") != _sha256(checkpoint_path):
                _fail(f"{label}: checkpoint SHA-256 mismatch")

            current_points = current.get("points")
            original_points = original.get("points")
            if not isinstance(current_points, list) or not isinstance(
                original_points, list
            ):
                _fail(f"{label}: points are missing")
            current_by_threshold = {
                float(point["threshold"]): point for point in current_points
            }
            original_by_threshold = {
                float(point["threshold"]): point for point in original_points
            }
            if len(current_by_threshold) != len(current_points):
                _fail(f"{label}: current thresholds are not unique")
            if len(original_by_threshold) != len(original_points):
                _fail(f"{label}: original thresholds are not unique")
            for threshold, point in original_by_threshold.items():
                if current_by_threshold.get(threshold) != point:
                    _fail(
                        f"{label}: original point changed at threshold={threshold}"
                    )
            expected_added = {
                threshold
                for threshold in (
                    LAST_FLOAT32_BELOW_ONE,
                    UPPER_BOUNDARY_THRESHOLD,
                )
                if threshold not in original_by_threshold
            }
            actual_added = set(current_by_threshold) - set(original_by_threshold)
            if actual_added != expected_added:
                _fail(
                    f"{label}: added thresholds mismatch; "
                    f"expected={sorted(expected_added)} actual={sorted(actual_added)}"
                )

            provenance = current.get("threshold_provenance")
            if not isinstance(provenance, dict):
                _fail(f"{label}: threshold provenance missing")
            expected_provenance = {
                "posthoc_endpoint_completion": True,
                "closed_probability_interval": True,
                "score_dtype": "float32",
                "added_thresholds": [
                    LAST_FLOAT32_BELOW_ONE,
                    UPPER_BOUNDARY_THRESHOLD,
                ],
                "last_float32_below_one": LAST_FLOAT32_BELOW_ONE,
                "last_float32_semantics": "exact_one_score_plateau",
                "upper_boundary_threshold": UPPER_BOUNDARY_THRESHOLD,
                "upper_boundary_comparison": "prediction > threshold",
                "upper_boundary_semantics": "empty_prediction_pd0_fa0",
                "total_unique_threshold_count": len(current_points),
            }
            for key, value in expected_provenance.items():
                if provenance.get(key) != value:
                    _fail(f"{label}: provenance {key} mismatch")
            score_count = provenance.get("score_count")
            exact_one_count = provenance.get("exact_one_score_count")
            if (
                isinstance(score_count, bool)
                or not isinstance(score_count, int)
                or score_count <= 0
                or isinstance(exact_one_count, bool)
                or not isinstance(exact_one_count, int)
                or not 0 <= exact_one_count <= score_count
            ):
                _fail(f"{label}: invalid score counts")

            endpoint = current_by_threshold.get(UPPER_BOUNDARY_THRESHOLD)
            if endpoint is None:
                _fail(f"{label}: empty-prediction endpoint missing")
            expected_zero_fields = (
                "miou",
                "pixel_precision",
                "pixel_recall",
                "pixel_f1",
                "pd",
                "tiny_pd",
                "fa",
                "false_objects_per_image",
                "matched_target_count",
                "matched_tiny_target_count",
                "predicted_object_count",
                "unmatched_predicted_object_count",
            )
            for key in expected_zero_fields:
                if float(endpoint[key]) != 0.0:
                    _fail(f"{label}: endpoint {key} is not zero")

            budgets = current.get("best_points_under_fa_budget")
            if not isinstance(budgets, dict) or set(budgets) != set(BUDGETS):
                _fail(f"{label}: budget keys mismatch")
            for budget in BUDGETS:
                recomputed = _best_point(current_points, float(budget))
                if budgets[budget] is None or budgets[budget] != recomputed:
                    _fail(f"{label}: budget {budget} is not recomputed optimum")

            audit = current.get("audit")
            if not isinstance(audit, dict):
                _fail(f"{label}: evaluator audit missing")
            artifact_hashes = audit.get("artifact_sha256")
            invocation = audit.get("invocation_argv")
            if (
                not isinstance(artifact_hashes, dict)
                or artifact_hashes.get("evaluator") != wrapper_sha256
                or not isinstance(invocation, list)
                or len(invocation) < 2
                or Path(str(invocation[1])).resolve() != wrapper
            ):
                _fail(f"{label}: wrapper provenance mismatch")

            original_null_budgets = sorted(
                key
                for key, value in original[
                    "best_points_under_fa_budget"
                ].items()
                if value is None
            )
            records.append(
                {
                    "label": label,
                    "checkpoint_sha256": current["checkpoint_sha256"],
                    "original_sweep": str(backup_path),
                    "original_sweep_sha256": _sha256(backup_path),
                    "closed_interval_sweep": str(current_path),
                    "closed_interval_sweep_sha256": _sha256(current_path),
                    "original_point_count": len(original_points),
                    "closed_interval_point_count": len(current_points),
                    "added_thresholds": sorted(actual_added),
                    "exact_one_score_count": exact_one_count,
                    "original_null_budgets": original_null_budgets,
                    "closed_interval_null_budgets": [],
                    "original_points_preserved": True,
                    "fixed_threshold_preserved": True,
                    "budgets_recomputed": True,
                }
            )

    report = {
        "schema": "sctransnet_tpd_clean_v3_closed_interval_sweep_audit_v1",
        "status": "complete",
        "candidate_sweeps": len(records),
        "wrapper": str(wrapper),
        "wrapper_sha256": wrapper_sha256,
        "score_dtype": "float32",
        "comparison_rule": "prediction > threshold",
        "added_thresholds": [
            LAST_FLOAT32_BELOW_ONE,
            UPPER_BOUNDARY_THRESHOLD,
        ],
        "training_rerun": False,
        "historical_reference_modified": False,
        "records": records,
    }
    compact = json.dumps(report, sort_keys=True, separators=(",", ":"))
    if args.output is not None:
        output = (
            args.output if args.output.is_absolute() else repo / args.output
        )
        if output.exists():
            _fail(f"refusing to overwrite audit report: {output}")
        if not output.parent.is_dir() or output.parent.is_symlink():
            _fail(f"audit report parent is unavailable: {output.parent}")
        with output.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n"
            )
    print(compact)


if __name__ == "__main__":
    try:
        main()
    except (RecoveryAuditError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"TPDCLEANV3_CLOSED_INTERVAL_AUDIT_INVALID reason={exc}")
