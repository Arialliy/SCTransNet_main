#!/usr/bin/env python3
"""Frozen CPU-only policy helpers for the paired DLR+ramp100 closure.

The closure compares complete checkpoint-local operating points.  A point is
never assembled from metrics or thresholds belonging to different
checkpoints.  Method selection uses all twelve aligned locations per method:
two validation-selected checkpoints times fixed-0.5 plus five preregistered
Fa budgets.  Deployment then selects one atomic point from the chosen method.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = "NUDT-SIRST"
TRAINING_SEED = 42
SPLIT_SEED = 20260722

TRAINING_LOCK_PATH = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact_source_lock.json"
)
TRAINING_LOCK_SHA256 = (
    "88b4839b40484c881544614e60675c4d2805a4fd6de1cc2f0aad28bdcb1395e8"
)
REFERENCE_CLOSURE_LOCK_PATH = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v4_qfg_v2_croa_posttraining_closure_source_lock.json"
)
REFERENCE_CLOSURE_LOCK_SHA256 = (
    "315f091b75078e65b871946cecae92893e8915bb3951b6fc4dcf3a52c984cbbd"
)

LEGACY_V1_LOCK_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "posttraining_closure_source_lock_v1"
)
LEGACY_V1_LOCK_PATH = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "posttraining_closure_source_lock.json"
)
LEGACY_V1_LOCK_SHA256 = (
    "c0af9b9001f0ed4814424700d3797251dd1cb21832c9e7cd95c3a7736287baf5"
)
SUPERSESSION_REASON = (
    "The checkpoint-local evaluator and its two execution lanes were "
    "hardened after the v1 freeze; v1 therefore binds an earlier evaluator "
    "and is retained only as immutable history."
)

LOCK_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "posttraining_closure_source_lock_v2"
)
LOCK_ACTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "posttraining_closure_lock_action_v2"
)
LOCK_ENVIRONMENT = (
    "TPD_NER_V4_QFG_V2_CROA_DLR_RAMP100_POSTTRAINING_SOURCE_LOCK"
)
DEFAULT_LOCK_PATH = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "posttraining_closure_source_lock_v2.json"
)

FINAL_EVALUATION_SOURCE_SHA256 = {
    (
        "experiments/"
        "evaluate_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_pd_fa.py"
    ): "ae0ffd138e161db1aa20e91c8da5f884b832598454eac9577386331ddd7df90a",
    (
        "experiments/"
        "run_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_eval_lane.sh"
    ): "ec81eca1425e256a546c1f68463a968a5cdfadbec4012f75d846148f3588a588",
    (
        "experiments/"
        "run_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_sweeps_2x5090.sh"
    ): "56c79b29baceb4599f1a7002c3780b5cee8c770bae84e626c98dc4f08cb9850d",
}

CHECKPOINT_ROLE_ORDER = ("pd_primary", "miou_secondary")
CHECKPOINT_FILENAMES = {
    "pd_primary": "best.pth.tar",
    "miou_secondary": "best_miou.pth.tar",
}
CHECKPOINT_ROLES = {
    "best.pth.tar": "best_validation_pd_primary",
    "best_miou.pth.tar": "best_validation_miou_secondary",
}
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
BUDGET_KEYS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")
BUDGET_BY_KEY = dict(zip(BUDGET_KEYS, FA_BUDGETS))
LOCATION_ORDER = (
    "fixed_threshold_0_5",
    *(f"fa_budget:{key}" for key in BUDGET_KEYS),
)
OBJECTIVE_DIRECTIONS = {
    "pd": "maximize",
    "fa": "minimize",
    "miou": "maximize",
    "tiny_pd": "maximize",
    "false_objects_per_image": "minimize",
}
OBJECTIVE_FIELDS = tuple(OBJECTIVE_DIRECTIONS)
FLOAT_ATOL = 1e-12

METHOD_ORDER = (
    "baseline",
    "v4",
    "a_control",
    "b_tss",
    "c_qfg_only",
    "d_tss_qfg",
    "e_qfg_dlr",
    "f_tss_qfg_dlr",
)
METHOD_COMPLEXITY_RANK = {
    "baseline": 0,
    "v4": 1,
    "a_control": 1,
    "b_tss": 2,
    "c_qfg_only": 2,
    "d_tss_qfg": 3,
    "e_qfg_dlr": 2,
    "f_tss_qfg_dlr": 3,
}

DEPLOYMENT_POLICY_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "deployment_point_policy_v1"
)
DEPLOYMENT_SELECTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "deployment_selection_v1"
)
METHOD_SELECTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "aligned_method_selection_v1"
)

# The evaluator is owned independently, but its public interface is frozen
# here so the comparator does not depend on evaluator-private helpers.
DLR_SWEEP_INTERFACE_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "public_sweep_interface_v1"
)
DLR_EVALUATION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "checkpoint_local_pd_fa_v1"
)
DLR_SWEEP_REQUIRED_FIELDS = (
    "schema",
    "dataset",
    "seed",
    "split_seed",
    "variant",
    "run_directory",
    "checkpoint",
    "checkpoint_role",
    "checkpoint_epoch",
    "checkpoint_sha256",
    "validation_count",
    "validation_split_sha256",
    "fixed_threshold_0_5",
    "best_points_under_fa_budget",
    "points",
    "threshold_selection_scope",
    "cross_checkpoint_point_pooling",
    "evaluated_checkpoint_count",
    "official_test_accessed",
    "run_identity",
    "source_checkpoint_identity",
)

# Filled by the write-once freezer.  Keeping this list explicit prevents a
# training lock from being silently repurposed as a post-training lock.
POSTTRAINING_SOURCE_PATHS = (
    (
        "experiments/"
        "tpd_ner_v4_qfg_v2_croa_dlr_ramp100_posttraining_policy.py"
    ),
    (
        "experiments/"
        "freeze_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_posttraining_closure.py"
    ),
    (
        "experiments/"
        "evaluate_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_pd_fa.py"
    ),
    (
        "experiments/"
        "run_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_eval_lane.sh"
    ),
    (
        "experiments/"
        "run_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_sweeps_2x5090.sh"
    ),
    (
        "experiments/"
        "compare_tpd_ner_v4_qfg_v2_croa_dlr_ramp100.py"
    ),
    (
        "experiments/"
        "export_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_to_inference.py"
    ),
    (
        "experiments/"
        "deploy_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800.py"
    ),
    "experiments/postprocess_tpd_ner_v4_qfg_v2_croa_formal800.py",
    "experiments/tpd_ner_v4_qfg_v2_croa_posttraining_policy.py",
    "experiments/export_tpd_ner_v4_qfg_v2_croa_to_inference.py",
    "experiments/export_tpd_ner_v4_survival_to_inference.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def regular_file(path: Path, label: str) -> Path:
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {value}")
    return value


def sha256_file(path: Path) -> str:
    value = regular_file(path, "SHA-256 input")
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


def supersession_summary() -> dict[str, Any]:
    return canonical(
        {
            "supersedes_schema": LEGACY_V1_LOCK_SCHEMA,
            "superseded_path": str(LEGACY_V1_LOCK_PATH.resolve()),
            "superseded_sha256": LEGACY_V1_LOCK_SHA256,
            "reason": SUPERSESSION_REASON,
            "replacement_schema": LOCK_SCHEMA,
            "old_lock_retained": True,
            "old_lock_must_not_be_used": True,
        }
    )


def require_v2_lock_target(path: Path) -> Path:
    target = Path(path).expanduser().resolve()
    if target == LEGACY_V1_LOCK_PATH.resolve():
        raise ValueError(
            "the v1 DLR post-training closure lock is superseded and "
            "cannot be read, verified, or overwritten"
        )
    return target


def verify_final_evaluation_sources() -> None:
    for relative, expected in FINAL_EVALUATION_SOURCE_SHA256.items():
        actual = sha256_file(REPO_ROOT / relative)
        _require(
            actual == expected,
            f"final DLR evaluation source differs: {relative}",
        )


def policy_summary() -> dict[str, Any]:
    return canonical(
        {
            "schema": DEPLOYMENT_POLICY_SCHEMA,
            "dataset": DATASET,
            "training_seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "methods": list(METHOD_ORDER),
            "checkpoint_roles": list(CHECKPOINT_ROLE_ORDER),
            "checkpoint_filenames": dict(CHECKPOINT_FILENAMES),
            "aligned_locations": list(LOCATION_ORDER),
            "operating_point_sources": {
                "fixed_threshold": 0.5,
                "fa_budgets": list(FA_BUDGETS),
                "fa_budget_keys": list(BUDGET_KEYS),
            },
            "objective_priority": [
                "pd:maximize",
                "fa:minimize",
                "miou:maximize",
                "tiny_pd:maximize",
                "false_objects_per_image:minimize",
            ],
            "method_selection": {
                "unit": "same_role_same_location_atomic_point",
                "uses_all_aligned_locations": True,
                "aligned_location_count": (
                    len(CHECKPOINT_ROLE_ORDER) * len(LOCATION_ORDER)
                ),
                "primary": "sum_of_per_location_lexicographic_ranks:minimize",
                "secondary": "aligned_first_place_count:maximize",
                "tertiary": "aligned_pareto_membership_count:maximize",
                "quaternary": "aligned_pairwise_strict_dominance_count:maximize",
                "equivalence_tiebreak": (
                    "inference_complexity_then_frozen_method_order"
                ),
            },
            "deployment_selection": {
                "unit": "one_complete_checkpoint_local_operating_point",
                "pareto_filter": copy.deepcopy(OBJECTIVE_DIRECTIONS),
                "tiebreak": (
                    "objective_priority_then_pd_primary_then_"
                    "fixed0.5_and_ascending_budgets_then_threshold"
                ),
            },
            "cross_checkpoint_metric_stitching": False,
            "cross_checkpoint_threshold_stitching": False,
            "cross_method_metric_stitching": False,
            "training_source_lock_sha256": TRAINING_LOCK_SHA256,
            "reference_closure_lock_sha256": REFERENCE_CLOSURE_LOCK_SHA256,
            "closure_lock_generation": 2,
            "supersession": supersession_summary(),
            "final_evaluation_source_sha256": copy.deepcopy(
                FINAL_EVALUATION_SOURCE_SHA256
            ),
            "official_test_accessed": False,
        }
    )


def policy_summary_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(policy_summary())).hexdigest()


def closure_lock_path(path: Path | None = None) -> Path:
    if path is not None:
        return require_v2_lock_target(path)
    configured = os.environ.get(LOCK_ENVIRONMENT)
    resolved = (
        Path(configured).expanduser().resolve()
        if configured
        else DEFAULT_LOCK_PATH.resolve()
    )
    return require_v2_lock_target(resolved)


def load_closure_lock(
    path: Path | None = None,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = closure_lock_path(path)
    raw = regular_file(lock_path, "DLR post-training closure lock").read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid DLR closure lock: {error}") from error
    _require(isinstance(payload, dict), "DLR closure lock must be one object")
    _require(
        canonical_json_bytes(payload) == raw,
        "DLR closure lock is not canonical",
    )
    if payload.get("schema") == LEGACY_V1_LOCK_SCHEMA:
        raise ValueError(
            "the v1 DLR post-training closure lock is superseded and "
            "cannot be used"
        )
    _require(payload.get("schema") == LOCK_SCHEMA, "DLR lock schema differs")
    _require(payload.get("status") == "complete", "DLR lock is incomplete")
    _require(
        payload.get("lock_kind") == "post_training_closure",
        "DLR lock kind differs",
    )
    expected_paths = list(POSTTRAINING_SOURCE_PATHS)
    source_sha256 = payload.get("source_sha256")
    _require(isinstance(source_sha256, Mapping), "DLR source hashes are missing")
    _require(
        payload.get("source_count") == len(expected_paths),
        "DLR closure source count differs",
    )
    _require(
        set(source_sha256) == set(expected_paths),
        "DLR closure source path set differs",
    )
    _require(
        payload.get("policy_summary") == policy_summary(),
        "DLR policy summary differs",
    )
    _require(
        payload.get("policy_summary_sha256") == policy_summary_sha256(),
        "DLR policy summary SHA differs",
    )
    _require(
        payload.get("supersession") == supersession_summary(),
        "DLR closure-lock supersession record differs",
    )
    _require(
        payload.get("final_evaluation_source_sha256")
        == FINAL_EVALUATION_SOURCE_SHA256,
        "DLR final evaluation-source binding differs",
    )
    training_binding = payload.get("training_source_lock")
    reference_binding = payload.get("reference_closure_source_lock")
    _require(
        isinstance(training_binding, Mapping)
        and training_binding.get("sha256") == TRAINING_LOCK_SHA256,
        "DLR training-lock binding differs",
    )
    _require(
        isinstance(reference_binding, Mapping)
        and reference_binding.get("sha256") == REFERENCE_CLOSURE_LOCK_SHA256,
        "reference closure-lock binding differs",
    )
    if verify_sources:
        for relative, expected in source_sha256.items():
            _require(
                sha256_file(REPO_ROOT / relative) == expected,
                f"DLR post-training source changed: {relative}",
            )
        _require(
            sha256_file(TRAINING_LOCK_PATH) == TRAINING_LOCK_SHA256,
            "DLR training source lock changed",
        )
        _require(
            sha256_file(REFERENCE_CLOSURE_LOCK_PATH)
            == REFERENCE_CLOSURE_LOCK_SHA256,
            "reference closure source lock changed",
        )
        _require(
            sha256_file(LEGACY_V1_LOCK_PATH) == LEGACY_V1_LOCK_SHA256,
            "superseded v1 DLR closure lock changed",
        )
        verify_final_evaluation_sources()
    binding = {
        "schema": LOCK_SCHEMA,
        "path": str(lock_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_count": len(expected_paths),
        "policy_summary_sha256": policy_summary_sha256(),
        "training_source_lock_sha256": TRAINING_LOCK_SHA256,
        "reference_closure_lock_sha256": REFERENCE_CLOSURE_LOCK_SHA256,
        "superseded_lock_sha256": LEGACY_V1_LOCK_SHA256,
        "supersession_reason": SUPERSESSION_REASON,
        "lock_generation": 2,
        "verified_live": bool(verify_sources),
    }
    return copy.deepcopy(payload), binding


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def objective_key(point: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        -_finite(point.get("pd"), "Pd"),
        _finite(point.get("fa"), "Fa"),
        -_finite(point.get("miou"), "mIoU"),
        -_finite(point.get("tiny_pd"), "tiny-Pd"),
        _finite(
            point.get("false_objects_per_image"),
            "false objects per image",
        ),
    )


def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    weak = True
    strict = False
    for field, direction in OBJECTIVE_DIRECTIONS.items():
        left_value = _finite(left.get(field), f"left {field}")
        right_value = _finite(right.get(field), f"right {field}")
        if direction == "maximize":
            weak = weak and left_value >= right_value - FLOAT_ATOL
            strict = strict or left_value > right_value + FLOAT_ATOL
        else:
            weak = weak and left_value <= right_value + FLOAT_ATOL
            strict = strict or left_value < right_value - FLOAT_ATOL
    return weak and strict


def point_for_location(
    method: Mapping[str, Any],
    role_name: str,
    location: str,
) -> Mapping[str, Any]:
    roles = method.get("roles")
    _require(isinstance(roles, Mapping), "method roles are missing")
    _require(
        set(roles) == set(CHECKPOINT_ROLE_ORDER),
        "method checkpoint-role matrix differs",
    )
    role = roles[role_name]
    _require(isinstance(role, Mapping), f"{role_name} is not an object")
    if location == "fixed_threshold_0_5":
        point = role.get("fixed_threshold_0_5")
    else:
        prefix = "fa_budget:"
        _require(location.startswith(prefix), f"unknown location: {location}")
        budgets = role.get("fa_budget_points")
        _require(isinstance(budgets, Mapping), "method Fa budgets are missing")
        _require(
            tuple(budgets) == BUDGET_KEYS,
            "method Fa-budget key order differs",
        )
        point = budgets.get(location[len(prefix) :])
    _require(isinstance(point, Mapping), f"point is missing: {role_name}/{location}")
    for field in ("threshold", *OBJECTIVE_FIELDS):
        _finite(point.get(field), f"{role_name}/{location}/{field}")
    return point


def atomic_candidates(method: Mapping[str, Any]) -> list[dict[str, Any]]:
    method_id = method.get("method_id")
    _require(method_id in METHOD_ORDER, f"unsupported method: {method_id!r}")
    candidates: list[dict[str, Any]] = []
    for role_rank, role_name in enumerate(CHECKPOINT_ROLE_ORDER):
        role = method["roles"][role_name]
        _require(
            role.get("checkpoint") == CHECKPOINT_FILENAMES[role_name],
            "checkpoint filename differs",
        )
        checkpoint_sha = role.get("checkpoint_sha256")
        _require(is_sha256(checkpoint_sha), "checkpoint SHA is invalid")
        for location_rank, location in enumerate(LOCATION_ORDER):
            point = point_for_location(method, role_name, location)
            budget_key = (
                None
                if location == "fixed_threshold_0_5"
                else location.split(":", 1)[1]
            )
            candidates.append(
                {
                    "candidate_id": f"{role_name}:{location}",
                    "method_id": method_id,
                    "variant": method.get("variant"),
                    "checkpoint": role["checkpoint"],
                    "checkpoint_role": role.get("checkpoint_role"),
                    "role_name": role_name,
                    "checkpoint_epoch": role.get("checkpoint_epoch"),
                    "checkpoint_path": str(role.get("checkpoint_path")),
                    "checkpoint_sha256": checkpoint_sha,
                    "operating_point_source": location,
                    "fa_budget_key": budget_key,
                    "fa_budget": (
                        None
                        if budget_key is None
                        else BUDGET_BY_KEY[budget_key]
                    ),
                    "threshold": float(point["threshold"]),
                    "metrics": {
                        field: float(point[field])
                        for field in OBJECTIVE_FIELDS
                    },
                    "checkpoint_local_atomic_point": True,
                    "_role_rank": role_rank,
                    "_location_rank": location_rank,
                }
            )
    _require(len(candidates) == 12, "atomic candidate count differs")
    return candidates


def _deployment_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        *objective_key(candidate["metrics"]),
        int(candidate["_role_rank"]),
        int(candidate["_location_rank"]),
        float(candidate["threshold"]),
    )


def select_deployment_operating_point(
    method: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = atomic_candidates(method)
    frontier = [
        candidate
        for candidate in candidates
        if not any(
            other is not candidate
            and dominates(other["metrics"], candidate["metrics"])
            for other in candidates
        )
    ]
    _require(frontier, "deployment frontier is empty")
    selected = min(frontier, key=_deployment_key)
    public = {
        key: copy.deepcopy(value)
        for key, value in selected.items()
        if not key.startswith("_")
    }
    return {
        "schema": DEPLOYMENT_SELECTION_SCHEMA,
        "policy": policy_summary(),
        "candidate_count": len(candidates),
        "pareto_frontier_candidate_count": len(frontier),
        "pareto_frontier_candidate_ids": [
            candidate["candidate_id"]
            for candidate in sorted(frontier, key=_deployment_key)
        ],
        "selected": public,
        "cross_checkpoint_metric_stitching": False,
        "selected_point_is_checkpoint_local": True,
    }


def _method_score_key(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    method_id = str(summary["method_id"])
    return (
        int(summary["aligned_rank_sum"]),
        -int(summary["aligned_first_place_count"]),
        -int(summary["aligned_pareto_membership_count"]),
        -int(summary["aligned_pairwise_strict_dominance_count"]),
        int(METHOD_COMPLEXITY_RANK[method_id]),
        METHOD_ORDER.index(method_id),
    )


def select_method(
    methods: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(
        tuple(methods) == METHOD_ORDER,
        "method order/matrix differs from the frozen closure",
    )
    summaries = {
        method_id: {
            "method_id": method_id,
            "aligned_rank_sum": 0,
            "aligned_first_place_count": 0,
            "aligned_pareto_membership_count": 0,
            "aligned_pairwise_strict_dominance_count": 0,
            "location_ranks": {},
        }
        for method_id in METHOD_ORDER
    }
    location_reports: dict[str, Any] = {}
    for role_name in CHECKPOINT_ROLE_ORDER:
        for location in LOCATION_ORDER:
            location_id = f"{role_name}:{location}"
            points = {
                method_id: point_for_location(
                    methods[method_id],
                    role_name,
                    location,
                )
                for method_id in METHOD_ORDER
            }
            ranked = sorted(
                METHOD_ORDER,
                key=lambda method_id: (
                    objective_key(points[method_id]),
                    METHOD_COMPLEXITY_RANK[method_id],
                    METHOD_ORDER.index(method_id),
                ),
            )
            frontier = [
                method_id
                for method_id in METHOD_ORDER
                if not any(
                    other != method_id
                    and dominates(points[other], points[method_id])
                    for other in METHOD_ORDER
                )
            ]
            pairwise_dominance = {
                method_id: sum(
                    int(
                        other != method_id
                        and dominates(points[method_id], points[other])
                    )
                    for other in METHOD_ORDER
                )
                for method_id in METHOD_ORDER
            }
            for rank, method_id in enumerate(ranked, start=1):
                summary = summaries[method_id]
                summary["aligned_rank_sum"] += rank
                summary["aligned_first_place_count"] += int(rank == 1)
                summary["aligned_pareto_membership_count"] += int(
                    method_id in frontier
                )
                summary[
                    "aligned_pairwise_strict_dominance_count"
                ] += pairwise_dominance[method_id]
                summary["location_ranks"][location_id] = rank
            location_reports[location_id] = {
                "role_name": role_name,
                "location": location,
                "ranked_method_ids": ranked,
                "pareto_method_ids": frontier,
                "pairwise_strict_dominance_count": pairwise_dominance,
                "atomic_points": {
                    method_id: {
                        "threshold": float(points[method_id]["threshold"]),
                        "objectives": {
                            field: float(points[method_id][field])
                            for field in OBJECTIVE_FIELDS
                        },
                    }
                    for method_id in METHOD_ORDER
                },
            }
    selected_method_id = min(
        METHOD_ORDER,
        key=lambda method_id: _method_score_key(summaries[method_id]),
    )
    ranked_methods = sorted(
        METHOD_ORDER,
        key=lambda method_id: _method_score_key(summaries[method_id]),
    )
    baseline_rank = ranked_methods.index("baseline") + 1
    selected_rank = ranked_methods.index(selected_method_id) + 1
    return {
        "schema": METHOD_SELECTION_SCHEMA,
        "selected_method_id": selected_method_id,
        "selected_variant": methods[selected_method_id].get("variant"),
        "ranked_method_ids": ranked_methods,
        "method_summaries": summaries,
        "aligned_location_reports": location_reports,
        "aligned_location_count": len(location_reports),
        "selected_rank": selected_rank,
        "baseline_rank": baseline_rank,
        "selected_outperforms_baseline_under_frozen_policy": (
            selected_method_id != "baseline"
            and selected_rank < baseline_rank
        ),
        "cross_checkpoint_metric_stitching": False,
        "cross_method_metric_stitching": False,
    }


def interface_summary() -> dict[str, Any]:
    return canonical(
        {
            "schema": DLR_SWEEP_INTERFACE_SCHEMA,
            "accepted_evaluation_schema": DLR_EVALUATION_SCHEMA,
            "required_fields": list(DLR_SWEEP_REQUIRED_FIELDS),
            "variants": ["qfg_dlr", "tss_qfg_dlr"],
            "checkpoints": dict(CHECKPOINT_ROLES),
            "fixed_threshold": 0.5,
            "fa_budget_keys": list(BUDGET_KEYS),
            "fa_budgets": list(FA_BUDGETS),
            "threshold_selection_scope": "single_checkpoint_only",
            "cross_checkpoint_point_pooling": False,
            "evaluated_checkpoint_count": 1,
        }
    )


__all__ = [
    "BUDGET_BY_KEY",
    "BUDGET_KEYS",
    "CHECKPOINT_FILENAMES",
    "CHECKPOINT_ROLE_ORDER",
    "CHECKPOINT_ROLES",
    "DATASET",
    "DEFAULT_LOCK_PATH",
    "DEPLOYMENT_SELECTION_SCHEMA",
    "DLR_EVALUATION_SCHEMA",
    "DLR_SWEEP_INTERFACE_SCHEMA",
    "DLR_SWEEP_REQUIRED_FIELDS",
    "FA_BUDGETS",
    "FINAL_EVALUATION_SOURCE_SHA256",
    "LEGACY_V1_LOCK_PATH",
    "LEGACY_V1_LOCK_SCHEMA",
    "LEGACY_V1_LOCK_SHA256",
    "LOCATION_ORDER",
    "LOCK_ACTION_SCHEMA",
    "LOCK_ENVIRONMENT",
    "LOCK_SCHEMA",
    "METHOD_COMPLEXITY_RANK",
    "METHOD_ORDER",
    "METHOD_SELECTION_SCHEMA",
    "OBJECTIVE_DIRECTIONS",
    "OBJECTIVE_FIELDS",
    "POSTTRAINING_SOURCE_PATHS",
    "REFERENCE_CLOSURE_LOCK_PATH",
    "REFERENCE_CLOSURE_LOCK_SHA256",
    "SPLIT_SEED",
    "TRAINING_LOCK_PATH",
    "TRAINING_LOCK_SHA256",
    "TRAINING_SEED",
    "SUPERSESSION_REASON",
    "atomic_candidates",
    "canonical",
    "canonical_json_bytes",
    "closure_lock_path",
    "dominates",
    "interface_summary",
    "is_sha256",
    "load_closure_lock",
    "objective_key",
    "point_for_location",
    "policy_summary",
    "policy_summary_sha256",
    "regular_file",
    "require_v2_lock_target",
    "select_deployment_operating_point",
    "select_method",
    "sha256_file",
    "supersession_summary",
    "verify_final_evaluation_sources",
]
