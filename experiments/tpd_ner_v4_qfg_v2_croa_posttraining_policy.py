#!/usr/bin/env python3
"""Frozen policy and provenance helpers for the QFG formal800 closure.

This module is deliberately independent of torch.  It owns the exact
post-training source set, the five-objective deployment operating-point
policy, and strict loading of the write-once closure source lock.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCK_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_posttraining_closure_source_lock_v1"
)
LOCK_ACTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_posttraining_closure_lock_action_v1"
)
LOCK_ENVIRONMENT = "TPD_NER_V4_QFG_V2_CROA_POSTTRAINING_SOURCE_LOCK"
DEFAULT_LOCK_PATH = (
    REPO_ROOT
    / "experiments/"
    "tpd_ner_v4_qfg_v2_croa_posttraining_closure_source_lock.json"
)
TRAINING_LOCK_PATH = (
    REPO_ROOT
    / "experiments/tpd_ner_v4_qfg_v2_croa_"
    "exact_source_lock_v2_optimized.json"
)
TRAINING_LOCK_SHA256 = (
    "8d55464851db9441383854189eff64c05daf25e7ff3502c6c67cf06401996478"
)
FROZEN_AUTHORITY_PATH = (
    REPO_ROOT
    / "experiments/results/"
    "tpd_ner_v8_mprs_dch_v4_tail_aware_exact_v1/"
    "NUDT-SIRST/comparison/"
    "tpd_ner_v8_mprs_dch_v4_tail_aware_formal800_comparison.json"
)
FROZEN_AUTHORITY_SHA256 = (
    "fdcb7dd0a1f591fcd6446a806d007ed8f07b1fd9e217549318dbd1ee1a69e968"
)
FROZEN_AUTHORITY_MARKER_PATH = (
    FROZEN_AUTHORITY_PATH.parent / "POSTPROCESS_COMPLETE.json"
)
FROZEN_AUTHORITY_MARKER_SHA256 = (
    "be222b88b22a558ec2c8588d5e863da630f5c6af6d660de2c5b86b55547edc95"
)

POSTTRAINING_SOURCE_PATHS = (
    "experiments/tpd_ner_v4_qfg_v2_croa_posttraining_policy.py",
    "experiments/freeze_tpd_ner_v4_qfg_v2_croa_posttraining_closure.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "experiments/evaluate_tpd_clean_v6_pd_fa.py",
    "experiments/evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa.py",
    (
        "experiments/"
        "evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_pd_fa.py"
    ),
    "experiments/evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa.py",
    "experiments/run_tpd_ner_v4_qfg_v2_croa_formal800_eval_lane.sh",
    "experiments/run_tpd_ner_v4_qfg_v2_croa_formal800_sweeps_2x5090.sh",
    "experiments/compare_tss_qfg_v2_croa_factorial.py",
    "experiments/postprocess_tpd_ner_v4_qfg_v2_croa_formal800.py",
    "experiments/export_tpd_ner_v4_qfg_v2_croa_to_inference.py",
    "experiments/deploy_tpd_ner_v4_qfg_v2_croa_formal800.py",
    "experiments/finalize_tpd_ner_v4_qfg_v2_croa_formal800_2x5090.sh",
    (
        "experiments/"
        "launch_tpd_ner_v4_qfg_v2_croa_formal800_finalizer_2x5090.sh"
    ),
)

CHECKPOINT_ROLE_ORDER = ("pd_primary", "miou_secondary")
CHECKPOINT_FILENAMES = {
    "pd_primary": "best.pth.tar",
    "miou_secondary": "best_miou.pth.tar",
}
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
BUDGET_KEYS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")
OBJECTIVE_DIRECTIONS = {
    "pd": "maximize",
    "fa": "minimize",
    "miou": "maximize",
    "tiny_pd": "maximize",
    "false_objects_per_image": "minimize",
}
DEPLOYMENT_POLICY_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_deployment_point_policy_v1"
)
DEPLOYMENT_SELECTION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_deployment_selection_v1"
)
FLOAT_ATOL = 1e-12


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


def policy_summary() -> dict[str, Any]:
    summary = {
        "schema": DEPLOYMENT_POLICY_SCHEMA,
        "dataset": "NUDT-SIRST",
        "training_seed": 42,
        "split_seed": 20260722,
        "checkpoint_roles": list(CHECKPOINT_ROLE_ORDER),
        "checkpoint_filenames": dict(CHECKPOINT_FILENAMES),
        "operating_point_sources": {
            "fixed_threshold": 0.5,
            "fa_budgets": list(FA_BUDGETS),
            "fa_budget_keys": list(BUDGET_KEYS),
        },
        "candidate_unit": "one_complete_checkpoint_local_operating_point",
        "cross_checkpoint_metric_stitching": False,
        "cross_checkpoint_threshold_stitching": False,
        "checkpoint_candidates_compared_as_atomic_points": True,
        "frontier_filter": {
            "objective_directions": copy.deepcopy(OBJECTIVE_DIRECTIONS),
            "epsilon": FLOAT_ATOL,
        },
        "frontier_tiebreak_order": [
            "pd:maximize",
            "fa:minimize",
            "miou:maximize",
            "tiny_pd:maximize",
            "false_objects_per_image:minimize",
            "checkpoint_role:pd_primary_before_miou_secondary",
            "source:fixed0.5_then_fa_budget_ascending",
            "threshold:minimize",
        ],
        "export_policy": {
            "qfg_methods": "strict_head_free_qfg_export",
            "v4_fallback": "write_once_native_checkpoint_copy",
            "tss_inference_state": "removed_for_qfg_export",
        },
        "training_source_lock_sha256": TRAINING_LOCK_SHA256,
        "official_test_accessed": False,
    }
    return canonical(summary)


def policy_summary_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(policy_summary())).hexdigest()


def closure_lock_path(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    configured = os.environ.get(LOCK_ENVIRONMENT)
    return (
        Path(configured).expanduser().resolve()
        if configured
        else DEFAULT_LOCK_PATH.resolve()
    )


def load_closure_lock(
    path: Path | None = None,
    *,
    verify_sources: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = closure_lock_path(path)
    raw = regular_file(lock_path, "post-training closure source lock").read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid post-training closure source lock: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("post-training closure source lock must contain one object")
    if canonical_json_bytes(payload) != raw:
        raise ValueError("post-training closure source lock is not canonical")
    expected_paths = list(POSTTRAINING_SOURCE_PATHS)
    _require(payload.get("schema") == LOCK_SCHEMA, "closure lock schema differs")
    _require(payload.get("status") == "complete", "closure lock is not complete")
    _require(
        payload.get("lock_kind") == "post_training_closure",
        "closure lock kind differs",
    )
    _require(
        payload.get("source_count") == len(expected_paths),
        "closure source count differs",
    )
    source_sha256 = payload.get("source_sha256")
    _require(isinstance(source_sha256, Mapping), "closure source hashes are missing")
    _require(
        set(source_sha256) == set(expected_paths),
        "closure source path set differs",
    )
    _require(
        payload.get("policy_summary") == policy_summary(),
        "closure policy summary differs",
    )
    _require(
        payload.get("policy_summary_sha256") == policy_summary_sha256(),
        "closure policy summary SHA differs",
    )
    training_binding = payload.get("training_source_lock")
    _require(
        isinstance(training_binding, Mapping)
        and training_binding.get("sha256") == TRAINING_LOCK_SHA256,
        "closure training source-lock binding differs",
    )
    if verify_sources:
        for relative, expected_sha256 in source_sha256.items():
            _require(
                sha256_file(REPO_ROOT / relative) == expected_sha256,
                f"post-training source changed: {relative}",
            )
        _require(
            sha256_file(TRAINING_LOCK_PATH) == TRAINING_LOCK_SHA256,
            "training source lock changed",
        )
        _require(
            sha256_file(FROZEN_AUTHORITY_PATH) == FROZEN_AUTHORITY_SHA256,
            "frozen authority changed",
        )
        _require(
            sha256_file(FROZEN_AUTHORITY_MARKER_PATH)
            == FROZEN_AUTHORITY_MARKER_SHA256,
            "frozen authority marker changed",
        )
    binding = {
        "schema": LOCK_SCHEMA,
        "path": str(lock_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_count": len(expected_paths),
        "policy_summary_sha256": policy_summary_sha256(),
        "training_source_lock_sha256": TRAINING_LOCK_SHA256,
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


def _atomic_candidate(
    method: Mapping[str, Any],
    *,
    role_name: str,
    location: str,
    location_rank: int,
    fa_budget_key: str | None,
    point: Mapping[str, Any],
) -> dict[str, Any]:
    role = method["roles"][role_name]
    metrics = {
        field: _finite(point.get(field), f"{role_name}/{location}/{field}")
        for field in OBJECTIVE_DIRECTIONS
    }
    threshold = _finite(point.get("threshold"), f"{role_name}/{location}/threshold")
    checkpoint_path = str(role.get("checkpoint_path"))
    checkpoint_sha256 = role.get("checkpoint_sha256")
    _require(is_sha256(checkpoint_sha256), "deployment checkpoint SHA is invalid")
    checkpoint = role.get("checkpoint")
    _require(
        checkpoint == CHECKPOINT_FILENAMES[role_name],
        "deployment checkpoint filename differs",
    )
    candidate_id = f"{role_name}:{location}"
    return {
        "candidate_id": candidate_id,
        "method_id": method.get("method_id"),
        "variant": method.get("variant"),
        "checkpoint": checkpoint,
        "checkpoint_role": role.get("checkpoint_role"),
        "role_name": role_name,
        "checkpoint_epoch": role.get("checkpoint_epoch"),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "operating_point_source": location,
        "fa_budget_key": fa_budget_key,
        "fa_budget": (
            None
            if fa_budget_key is None
            else FA_BUDGETS[BUDGET_KEYS.index(fa_budget_key)]
        ),
        "threshold": threshold,
        "metrics": metrics,
        "checkpoint_local_atomic_point": True,
        "_role_rank": CHECKPOINT_ROLE_ORDER.index(role_name),
        "_location_rank": location_rank,
    }


def deployment_candidates(method: Mapping[str, Any]) -> list[dict[str, Any]]:
    roles = method.get("roles")
    _require(isinstance(roles, Mapping), "deployment method roles are missing")
    _require(
        set(roles) == set(CHECKPOINT_ROLE_ORDER),
        "deployment checkpoint role matrix differs",
    )
    candidates: list[dict[str, Any]] = []
    for role_name in CHECKPOINT_ROLE_ORDER:
        role = roles[role_name]
        _require(isinstance(role, Mapping), f"{role_name} is not an object")
        fixed = role.get("fixed_threshold_0_5")
        budgets = role.get("fa_budget_points")
        _require(isinstance(fixed, Mapping), f"{role_name} fixed point is missing")
        _require(isinstance(budgets, Mapping), f"{role_name} budgets are missing")
        _require(
            tuple(budgets) == BUDGET_KEYS,
            f"{role_name} Fa-budget keys differ",
        )
        candidates.append(
            _atomic_candidate(
                method,
                role_name=role_name,
                location="fixed_threshold_0_5",
                location_rank=0,
                fa_budget_key=None,
                point=fixed,
            )
        )
        for offset, key in enumerate(BUDGET_KEYS, start=1):
            candidates.append(
                _atomic_candidate(
                    method,
                    role_name=role_name,
                    location=f"fa_budget:{key}",
                    location_rank=offset,
                    fa_budget_key=key,
                    point=budgets[key],
                )
            )
    _require(len(candidates) == 12, "deployment candidate count differs")
    return candidates


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    weak = True
    strict = False
    for field, direction in OBJECTIVE_DIRECTIONS.items():
        left_value = float(left["metrics"][field])
        right_value = float(right["metrics"][field])
        if direction == "maximize":
            weak = weak and left_value >= right_value - FLOAT_ATOL
            strict = strict or left_value > right_value + FLOAT_ATOL
        else:
            weak = weak and left_value <= right_value + FLOAT_ATOL
            strict = strict or left_value < right_value - FLOAT_ATOL
    return weak and strict


def _deployment_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["metrics"]
    return (
        -float(metrics["pd"]),
        float(metrics["fa"]),
        -float(metrics["miou"]),
        -float(metrics["tiny_pd"]),
        float(metrics["false_objects_per_image"]),
        int(candidate["_role_rank"]),
        int(candidate["_location_rank"]),
        float(candidate["threshold"]),
    )


def _public_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in candidate.items()
        if not key.startswith("_")
    }


def select_deployment_operating_point(
    method: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = deployment_candidates(method)
    frontier = [
        candidate
        for candidate in candidates
        if not any(
            other is not candidate and _dominates(other, candidate)
            for other in candidates
        )
    ]
    _require(frontier, "deployment frontier is empty")
    selected = min(frontier, key=_deployment_key)
    return {
        "schema": DEPLOYMENT_SELECTION_SCHEMA,
        "policy": policy_summary(),
        "candidate_count": len(candidates),
        "pareto_frontier_candidate_count": len(frontier),
        "pareto_frontier_candidate_ids": [
            candidate["candidate_id"]
            for candidate in sorted(frontier, key=_deployment_key)
        ],
        "selected": _public_candidate(selected),
        "cross_checkpoint_metric_stitching": False,
        "selected_point_is_checkpoint_local": True,
    }


__all__ = [
    "BUDGET_KEYS",
    "DEFAULT_LOCK_PATH",
    "DEPLOYMENT_POLICY_SCHEMA",
    "DEPLOYMENT_SELECTION_SCHEMA",
    "FA_BUDGETS",
    "LOCK_ACTION_SCHEMA",
    "LOCK_ENVIRONMENT",
    "LOCK_SCHEMA",
    "OBJECTIVE_DIRECTIONS",
    "POSTTRAINING_SOURCE_PATHS",
    "TRAINING_LOCK_SHA256",
    "canonical",
    "canonical_json_bytes",
    "closure_lock_path",
    "deployment_candidates",
    "is_sha256",
    "load_closure_lock",
    "policy_summary",
    "policy_summary_sha256",
    "regular_file",
    "select_deployment_operating_point",
    "sha256_file",
]
