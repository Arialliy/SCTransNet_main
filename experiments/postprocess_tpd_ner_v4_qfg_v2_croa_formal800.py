#!/usr/bin/env python3
"""Write-once final selector for the formal V4/TSS/QFG experiment.

This module performs no model inference.  It binds the frozen
baseline/V1/V2/V3/V4 comparison authority, validates the four A/B and four
C/D checkpoint-local sweeps through their owning evaluators, and then applies
the preregistered five-objective all-point Pareto policy.

The five objectives are:

* Pd: maximize;
* Fa: minimize;
* mIoU: maximize;
* tiny-Pd: maximize;
* false objects per image: minimize.

One floating-point threshold cannot establish a contribution.  Candidate
evidence is therefore marked non-isolated only when a frontier point has an
adjacent-threshold peer, or when two adjacent preregistered Fa budgets form a
consistent frontier interval.  Exclusive support is recorded separately and
is required before D can be retained over the simpler C recipe.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    tpd_ner_v4_qfg_v2_croa_posttraining_policy as closure_policy,
)

SCHEMA = "sctransnet_tpd_ner_v4_qfg_v2_croa_final_selection_v2"
DATASET = "NUDT-SIRST"
TRAINING_SEED = 42
SPLIT_SEED = 20260722
VALIDATION_COUNT = 133
TARGET_COUNT = 189
TINY_TARGET_COUNT = 39
VALIDATION_SPLIT_SHA256 = (
    "86247e5970f93224c64005e1ac7f3a933bafb37baf279ab71fce5670ae925e06"
)
SCOPE = "single_seed_internal_validation"

CHECKPOINTS = ("best.pth.tar", "best_miou.pth.tar")
CHECKPOINT_ROLES = {
    "best.pth.tar": "best_validation_pd_primary",
    "best_miou.pth.tar": "best_validation_miou_secondary",
}
ROLE_NAMES = {
    "best_validation_pd_primary": "pd_primary",
    "best_validation_miou_secondary": "miou_secondary",
}
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
BUDGET_KEYS = ("1e-06", "5e-06", "1e-05", "5e-05", "0.0001")
BUDGET_BY_KEY = dict(zip(BUDGET_KEYS, FA_BUDGETS))

RELATIVE_IMPROVED = "RELATIVE_IMPROVED"
PARETO_MIXED_TRADEOFF = "PARETO_MIXED_TRADEOFF"
DOMINATED = "DOMINATED"
CANDIDATE_STATES = (
    RELATIVE_IMPROVED,
    PARETO_MIXED_TRADEOFF,
    DOMINATED,
)

BASELINE_VARIANT = "baseline_sctransnet"
V1_VARIANT = "tpd_ner_v8_mprs_dch_full_relay_off"
V2_VARIANT = "tpd_ner_v8_mprs_dch_v2_full_relay_on"
V3_VARIANT = "tpd_ner_v8_mprs_dch_v3_full_relay_on"
V4_VARIANT = "tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on"

AUTHORITY_METHODS = {
    BASELINE_VARIANT: "baseline",
    V1_VARIANT: "v1",
    V2_VARIANT: "v2",
    V3_VARIANT: "v3",
    V4_VARIANT: "v4",
}
AUTHORITY_VARIANTS = tuple(AUTHORITY_METHODS)

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
FROZEN_AUTHORITY_MARKER = FROZEN_AUTHORITY_PATH.parent / "POSTPROCESS_COMPLETE.json"
FROZEN_AUTHORITY_MARKER_SHA256 = (
    "be222b88b22a558ec2c8588d5e863da630f5c6af6d660de2c5b86b55547edc95"
)
FROZEN_AUTHORITY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "posttraining_aggregate_v1"
)
FROZEN_AUTHORITY_MARKER_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_aware_"
    "posttraining_complete_v1"
)
FROZEN_TSS_B_DECISION = PARETO_MIXED_TRADEOFF

TSS_RESULT_ROOT = (
    REPO_ROOT
    / "experiments/results/tpd_ner_v4_survival_exact_v1"
    / DATASET
)
QFG_RESULT_ROOT = (
    REPO_ROOT
    / "experiments/results/"
    "tpd_ner_v4_qfg_v2_croa_exact_v2_optimized"
    / DATASET
)
QFG_OPTIMIZED_SOURCE_LOCK = (
    REPO_ROOT
    / "experiments/tpd_ner_v4_qfg_v2_croa_"
    "exact_source_lock_v2_optimized.json"
)
QFG_OPTIMIZED_SOURCE_LOCK_SHA256 = (
    "8d55464851db9441383854189eff64c05daf25e7ff3502c6c67cf06401996478"
)
QFG_SOURCE_LOCK_KEY = "tpd_ner_v4_qfg_v2_croa_exact_source_lock"
PHYSICAL_GPU_UUIDS = {
    2: "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    3: "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
EXPECTED_EVALUATION_GPU = {
    "a_control": 2,
    "b_tss": 3,
    "c_qfg_only": 2,
    "d_tss_qfg": 3,
}


@dataclass(frozen=True)
class ExtensionSpec:
    method_id: str
    display_name: str
    variant: str
    evaluator_module: str
    run_dir: Path


EXTENSION_SPECS = (
    ExtensionSpec(
        "a_control",
        "A: TSS-control",
        "tss_control",
        (
            "experiments."
            "evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_pd_fa"
        ),
        TSS_RESULT_ROOT / "tss_control" / "seed_42_formal800_control",
    ),
    ExtensionSpec(
        "b_tss",
        "B: TSS-on",
        "tss_on",
        (
            "experiments."
            "evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_pd_fa"
        ),
        TSS_RESULT_ROOT / "tss_on" / "seed_42_formal800_tss",
    ),
    ExtensionSpec(
        "c_qfg_only",
        "C: QFG-only",
        "qfg_only",
        "experiments.evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa",
        QFG_RESULT_ROOT
        / "qfg_only"
        / "seed_42_formal800_qfg_only",
    ),
    ExtensionSpec(
        "d_tss_qfg",
        "D: TSS+QFG",
        "tss_qfg",
        "experiments.evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa",
        QFG_RESULT_ROOT
        / "tss_qfg"
        / "seed_42_formal800_tss_qfg",
    ),
)
EXTENSION_BY_METHOD = {spec.method_id: spec for spec in EXTENSION_SPECS}

OUTPUT_DIR = QFG_RESULT_ROOT / "final_selection"
JSON_OUTPUT = OUTPUT_DIR / "tpd_ner_v4_qfg_v2_croa_formal800_final_selection.json"
MARKDOWN_OUTPUT = OUTPUT_DIR / "tpd_ner_v4_qfg_v2_croa_formal800_final_selection.md"

OBJECTIVE_DIRECTIONS = {
    "pd": "maximize",
    "fa": "minimize",
    "miou": "maximize",
    "tiny_pd": "maximize",
    "false_objects_per_image": "minimize",
}
OBJECTIVE_FIELDS = tuple(OBJECTIVE_DIRECTIONS)
FLOAT_ATOL = 1e-12
QFG_CANDIDATE_GROUP = frozenset(("c_qfg_only", "d_tss_qfg"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix != ".json" or path.name == "metrics.jsonl":
        raise ValueError(f"input is not a JSON evidence file: {path}")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} is not an integer")
    return int(value)


def _close(left: Any, right: Any, *, atol: float = FLOAT_ATOL) -> bool:
    return math.isclose(
        _finite(left, "left comparison value"),
        _finite(right, "right comparison value"),
        rel_tol=0.0,
        abs_tol=atol,
    )


def _canonical_equal(left: Any, right: Any) -> bool:
    def normalize(value: Any) -> Any:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )

    return normalize(left) == normalize(right)


def _checkpoint_filename(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label} is missing")
    filename = Path(value).name
    _require(filename in CHECKPOINTS, f"{label} is not a selector checkpoint")
    return filename


def _objective(point: Mapping[str, Any]) -> dict[str, float]:
    return {
        field: _finite(point[field], f"objective {field}")
        for field in OBJECTIVE_FIELDS
    }


def _point_signature(point: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        point[field]
        for field in (
            "threshold",
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
        )
    )


def normalize_point(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not an object")
    target_count = _integer(value.get("target_count"), f"{label} target_count")
    matched = _integer(
        value.get("matched_target_count"),
        f"{label} matched_target_count",
    )
    tiny_count = _integer(
        value.get("tiny_target_count"),
        f"{label} tiny_target_count",
    )
    tiny_matched = _integer(
        value.get("matched_tiny_target_count"),
        f"{label} matched_tiny_target_count",
    )
    _require(target_count == TARGET_COUNT, f"{label} target count differs")
    _require(tiny_count == TINY_TARGET_COUNT, f"{label} tiny count differs")
    _require(0 <= matched <= target_count, f"{label} matched count is invalid")
    _require(
        0 <= tiny_matched <= tiny_count,
        f"{label} tiny matched count is invalid",
    )

    threshold = _finite(value.get("threshold"), f"{label} threshold")
    pd = _finite(value.get("pd"), f"{label} Pd")
    fa = _finite(value.get("fa"), f"{label} Fa")
    miou = _finite(value.get("miou"), f"{label} mIoU")
    tiny_pd = _finite(value.get("tiny_pd"), f"{label} tiny-Pd")
    false_objects = _finite(
        value.get("false_objects_per_image"),
        f"{label} false objects",
    )
    _require(0.0 <= threshold <= 1.0, f"{label} threshold is out of range")
    _require(_close(pd, matched / target_count), f"{label} Pd/count differs")
    _require(
        _close(tiny_pd, tiny_matched / tiny_count),
        f"{label} tiny-Pd/count differs",
    )
    _require(0.0 <= pd <= 1.0, f"{label} Pd is out of range")
    _require(0.0 <= tiny_pd <= 1.0, f"{label} tiny-Pd is out of range")
    _require(0.0 <= miou <= 1.0, f"{label} mIoU is out of range")
    _require(fa >= 0.0, f"{label} Fa is negative")
    _require(false_objects >= 0.0, f"{label} false objects is negative")

    unmatched_value = value.get("unmatched_predicted_object_count")
    if unmatched_value is None:
        unmatched = int(round(false_objects * VALIDATION_COUNT))
    else:
        unmatched = _integer(
            unmatched_value,
            f"{label} unmatched predicted objects",
        )
        _require(unmatched >= 0, f"{label} unmatched count is negative")
        _require(
            _close(
                false_objects,
                unmatched / VALIDATION_COUNT,
                atol=2e-12,
            ),
            f"{label} false-object count/rate differs",
        )
    return {
        "threshold": threshold,
        "pd": pd,
        "fa": fa,
        "miou": miou,
        "tiny_pd": tiny_pd,
        "false_objects_per_image": false_objects,
        "target_count": target_count,
        "matched_target_count": matched,
        "tiny_target_count": tiny_count,
        "matched_tiny_target_count": tiny_matched,
        "unmatched_predicted_object_count": unmatched,
    }


def normalize_budgets(value: Any, *, label: str) -> dict[str, dict[str, Any]]:
    _require(isinstance(value, Mapping), f"{label} is missing")
    _require(set(value) == set(BUDGET_KEYS), f"{label} budget keys differ")
    normalized: dict[str, dict[str, Any]] = {}
    for key in BUDGET_KEYS:
        point = normalize_point(value[key], label=f"{label}:{key}")
        _require(
            point["fa"] <= BUDGET_BY_KEY[key] + 1e-15,
            f"{label}:{key} exceeds its Fa budget",
        )
        normalized[key] = point
    return normalized


def _sweep_filename(checkpoint: str) -> str:
    return f"pd_fa_sweep_{Path(checkpoint).stem}.json"


def normalize_sweep_payload(
    payload: Mapping[str, Any],
    *,
    method_id: str,
    display_name: str,
    expected_variant: str,
    checkpoint: str,
    sweep_path: Path | None = None,
    sweep_sha256: str | None = None,
    require_checkpoint_local_contract: bool = True,
) -> dict[str, Any]:
    """Normalize one already identity-validated checkpoint-local sweep."""

    _require(checkpoint in CHECKPOINTS, f"unknown checkpoint: {checkpoint}")
    role = CHECKPOINT_ROLES[checkpoint]
    label = f"{method_id}:{checkpoint}"
    for field, expected in {
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "validation_count": VALIDATION_COUNT,
        "validation_split_sha256": VALIDATION_SPLIT_SHA256,
        "official_test_accessed": False,
        "variant": expected_variant,
        "checkpoint_role": role,
    }.items():
        _require(payload.get(field) == expected, f"{label} differs: {field}")
    _require(
        _checkpoint_filename(payload.get("checkpoint"), f"{label} checkpoint")
        == checkpoint,
        f"{label} checkpoint filename differs",
    )
    checkpoint_sha = payload.get("checkpoint_sha256")
    _require(_is_sha256(checkpoint_sha), f"{label} checkpoint SHA is invalid")
    epoch = _integer(payload.get("checkpoint_epoch"), f"{label} checkpoint epoch")
    _require(1 <= epoch <= 800, f"{label} checkpoint epoch is out of range")
    if require_checkpoint_local_contract:
        _require(
            payload.get("threshold_selection_scope")
            == "single_checkpoint_only",
            f"{label} threshold selection scope differs",
        )
        _require(
            payload.get("cross_checkpoint_point_pooling") is False,
            f"{label} cross-checkpoint pooling is not disabled",
        )
        _require(
            payload.get("evaluated_checkpoint_count") == 1,
            f"{label} evaluated checkpoint count differs",
        )

    fixed = normalize_point(
        payload.get("fixed_threshold_0_5"),
        label=f"{label} fixed0.5",
    )
    _require(_close(fixed["threshold"], 0.5), f"{label} fixed point is not 0.5")
    budgets = normalize_budgets(
        payload.get("best_points_under_fa_budget"),
        label=f"{label} budgets",
    )
    raw_value = payload.get("points")
    _require(isinstance(raw_value, list) and raw_value, f"{label} points are missing")
    raw_points = [
        normalize_point(point, label=f"{label} point[{index}]")
        for index, point in enumerate(raw_value)
    ]
    thresholds = [point["threshold"] for point in raw_points]
    _require(
        len(set(thresholds)) == len(thresholds),
        f"{label} contains duplicate thresholds",
    )
    raw_signatures = {_point_signature(point) for point in raw_points}
    _require(
        _point_signature(fixed) in raw_signatures,
        f"{label} fixed point is absent from raw points",
    )
    for key, point in budgets.items():
        _require(
            _point_signature(point) in raw_signatures,
            f"{label} budget {key} is absent from raw points",
        )

    binding: dict[str, Any] = {}
    if sweep_path is not None:
        path = Path(sweep_path).resolve()
        _require(path.name == _sweep_filename(checkpoint), f"{label} sweep name differs")
        observed_sweep_sha = sha256_file(path)
        if sweep_sha256 is not None:
            _require(
                observed_sweep_sha == sweep_sha256,
                f"{label} sweep SHA differs",
            )
        binding = {"path": str(path), "sha256": observed_sweep_sha}

    checkpoint_path_value = payload.get("checkpoint")
    checkpoint_path = Path(str(checkpoint_path_value))
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(str(payload.get("run_directory"))) / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    return {
        "method_id": method_id,
        "display_name": display_name,
        "variant": expected_variant,
        "checkpoint": checkpoint,
        "checkpoint_role": role,
        "role_name": ROLE_NAMES[role],
        "checkpoint_epoch": epoch,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_path": str(checkpoint_path),
        "run_directory": str(Path(str(payload.get("run_directory"))).resolve()),
        "fixed_threshold_0_5": fixed,
        "fa_budget_points": budgets,
        "raw_points": sorted(raw_points, key=lambda point: point["threshold"]),
        "raw_point_count": len(raw_points),
        "sweep_binding": binding,
    }


def _verify_bound_file(
    entry: Any,
    *,
    label: str,
    expected_path: Path | None = None,
) -> dict[str, str]:
    _require(isinstance(entry, Mapping), f"{label} binding is missing")
    path_value = entry.get("path")
    digest = entry.get("sha256")
    _require(isinstance(path_value, str), f"{label} path is missing")
    _require(_is_sha256(digest), f"{label} SHA is invalid")
    path = Path(path_value).resolve()
    if expected_path is not None:
        _require(path == Path(expected_path).resolve(), f"{label} path differs")
    observed = sha256_file(path)
    _require(observed == digest, f"{label} SHA differs")
    return {"path": str(path), "sha256": observed}


def _authority_metrics_match(
    authority_row: Mapping[str, Any],
    raw_row: Mapping[str, Any],
) -> bool:
    fixed = authority_row["fixed_threshold_0_5"]
    raw_fixed = raw_row["fixed_threshold_0_5"]
    budgets = authority_row["pd_at_fa_budget"]
    raw_budgets = raw_row["pd_at_fa_budget"]
    fixed_fields = (
        "threshold",
        "pd",
        "fa",
        "miou",
        "false_objects_per_image",
        "target_count",
        "matched_target_count",
        "tiny_pd",
        "tiny_target_count",
        "matched_tiny_target_count",
    )
    budget_fields = (
        "threshold",
        "pd",
        "fa",
        "miou",
        "false_objects_per_image",
        "target_count",
        "matched_target_count",
    )
    if any(
        field not in fixed
        or field not in raw_fixed
        or fixed[field] != raw_fixed[field]
        for field in fixed_fields
    ):
        return False
    required_budget_fields = (
        "threshold",
        "pd",
        "fa",
        "target_count",
        "matched_target_count",
    )
    for key in BUDGET_KEYS:
        authority_point = budgets[key]
        raw_point = raw_budgets[key]
        if any(field not in authority_point for field in required_budget_fields):
            return False
        for field in budget_fields:
            if (
                field in authority_point
                and (
                    field not in raw_point
                    or authority_point[field] != raw_point[field]
                )
            ):
                return False
    return True


def validate_frozen_authority() -> dict[str, Any]:
    """Validate and expand the frozen ten-row comparison authority."""

    _require(
        sha256_file(FROZEN_AUTHORITY_PATH) == FROZEN_AUTHORITY_SHA256,
        "frozen V4 comparison authority SHA differs",
    )
    _require(
        sha256_file(FROZEN_AUTHORITY_MARKER)
        == FROZEN_AUTHORITY_MARKER_SHA256,
        "frozen V4 comparison marker SHA differs",
    )
    authority = load_json(FROZEN_AUTHORITY_PATH)
    marker = load_json(FROZEN_AUTHORITY_MARKER)
    for field, expected in {
        "schema": FROZEN_AUTHORITY_SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "official_test_accessed": False,
        "scope": SCOPE,
        "row_count": 10,
    }.items():
        _require(authority.get(field) == expected, f"authority differs: {field}")
    _require(
        marker.get("schema") == FROZEN_AUTHORITY_MARKER_SCHEMA,
        "authority marker schema differs",
    )
    _require(marker.get("status") == "complete", "authority marker is incomplete")
    outputs = marker.get("outputs")
    _require(isinstance(outputs, Mapping), "authority marker outputs are missing")
    _require(
        outputs.get(FROZEN_AUTHORITY_PATH.name) == FROZEN_AUTHORITY_SHA256,
        "authority marker does not bind the comparison JSON",
    )

    bindings = authority.get("bindings")
    _require(isinstance(bindings, Mapping), "authority bindings are missing")
    before = bindings.get("input_snapshot_before")
    after = bindings.get("input_snapshot_after")
    _require(
        isinstance(before, Mapping) and dict(before) == dict(after),
        "authority input snapshots differ",
    )

    historical = bindings.get("historical_authority")
    _require(isinstance(historical, Mapping), "historical authority binding missing")
    verified_inputs: dict[str, dict[str, str]] = {
        "frozen_v4_comparison_authority": {
            "path": str(FROZEN_AUTHORITY_PATH),
            "sha256": FROZEN_AUTHORITY_SHA256,
        },
        "frozen_v4_comparison_marker": {
            "path": str(FROZEN_AUTHORITY_MARKER),
            "sha256": FROZEN_AUTHORITY_MARKER_SHA256,
        },
    }
    for name in ("aggregate", "completion_marker"):
        verified_inputs[f"historical_authority:{name}"] = _verify_bound_file(
            historical.get(name),
            label=f"historical authority {name}",
        )

    historical_sweeps = historical.get("sweeps")
    v4_sweeps = bindings.get("v4_sweeps")
    _require(isinstance(historical_sweeps, Mapping), "historical sweeps missing")
    _require(isinstance(v4_sweeps, Mapping), "V4 sweeps missing")
    rows_value = authority.get("rows")
    _require(isinstance(rows_value, list), "authority rows are missing")
    indexed_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    for value in rows_value:
        _require(isinstance(value, Mapping), "authority row is not an object")
        variant = value.get("variant")
        checkpoint = value.get("checkpoint")
        _require(
            variant in AUTHORITY_VARIANTS and checkpoint in CHECKPOINTS,
            "authority row identity differs",
        )
        key = (str(variant), str(checkpoint))
        _require(key not in indexed_rows, f"duplicate authority row: {key}")
        indexed_rows[key] = value
    expected_keys = {
        (variant, checkpoint)
        for variant in AUTHORITY_VARIANTS
        for checkpoint in CHECKPOINTS
    }
    _require(set(indexed_rows) == expected_keys, "authority ten-row matrix differs")

    methods: dict[str, dict[str, Any]] = {}
    for variant in AUTHORITY_VARIANTS:
        method_id = AUTHORITY_METHODS[variant]
        method = {
            "method_id": method_id,
            "display_name": variant,
            "variant": variant,
            "roles": {},
        }
        for checkpoint in CHECKPOINTS:
            row = indexed_rows[(variant, checkpoint)]
            binding_key = f"{variant}:{checkpoint}"
            if variant == V4_VARIANT:
                entry = v4_sweeps.get(binding_key)
            else:
                entry = historical_sweeps.get(binding_key)
            verified_sweep = _verify_bound_file(
                entry,
                label=f"authority sweep {binding_key}",
            )
            checkpoint_entry = entry
            checkpoint_path_value = checkpoint_entry.get("checkpoint_path")
            checkpoint_sha = checkpoint_entry.get("checkpoint_sha256")
            if variant == V4_VARIANT:
                checkpoint_bindings = bindings.get("v4_checkpoints")
                _require(
                    isinstance(checkpoint_bindings, Mapping),
                    "V4 checkpoint bindings missing",
                )
                checkpoint_entry = checkpoint_bindings.get(binding_key)
                checkpoint_path_value = (
                    checkpoint_entry.get("path")
                    if isinstance(checkpoint_entry, Mapping)
                    else None
                )
                checkpoint_sha = (
                    checkpoint_entry.get("sha256")
                    if isinstance(checkpoint_entry, Mapping)
                    else None
                )
            _require(
                isinstance(checkpoint_path_value, str)
                and _is_sha256(checkpoint_sha),
                f"authority checkpoint binding missing: {binding_key}",
            )
            checkpoint_path = Path(checkpoint_path_value).resolve()
            _require(
                sha256_file(checkpoint_path) == checkpoint_sha,
                f"authority checkpoint SHA differs: {binding_key}",
            )
            _require(
                checkpoint_sha == row.get("checkpoint_sha256"),
                f"authority row/checkpoint SHA differs: {binding_key}",
            )
            verified_inputs[f"authority_sweep:{binding_key}"] = verified_sweep
            verified_inputs[f"authority_checkpoint:{binding_key}"] = {
                "path": str(checkpoint_path),
                "sha256": str(checkpoint_sha),
            }

            payload = load_json(Path(verified_sweep["path"]))
            expected_raw_variant = (
                "original" if variant == BASELINE_VARIANT else variant
            )
            normalized = normalize_sweep_payload(
                payload,
                method_id=method_id,
                display_name=variant,
                expected_variant=expected_raw_variant,
                checkpoint=checkpoint,
                sweep_path=Path(verified_sweep["path"]),
                sweep_sha256=verified_sweep["sha256"],
                # The frozen authority predates the explicit fields below;
                # its own write-once SHA, raw-sweep bindings, and ten-row
                # identity are the governing contract.  A/B/C/D must expose
                # the newer checkpoint-local fields.
                require_checkpoint_local_contract=False,
            )
            normalized["variant"] = variant
            _require(
                normalized["checkpoint_epoch"] == row.get("checkpoint_epoch"),
                f"authority row/sweep epoch differs: {binding_key}",
            )
            _require(
                normalized["checkpoint_sha256"] == row.get("checkpoint_sha256"),
                f"authority row/sweep checkpoint SHA differs: {binding_key}",
            )
            raw_projection = {
                "fixed_threshold_0_5": normalized["fixed_threshold_0_5"],
                "pd_at_fa_budget": normalized["fa_budget_points"],
            }
            _require(
                _authority_metrics_match(row, raw_projection),
                f"authority row/raw sweep metrics differ: {binding_key}",
            )
            method["roles"][normalized["role_name"]] = normalized
        _require(
            set(method["roles"]) == set(ROLE_NAMES.values()),
            f"authority roles differ: {variant}",
        )
        methods[method_id] = method

    return {
        "methods": methods,
        "bindings": verified_inputs,
        "authority": authority,
        "authority_binding": {
            "path": str(FROZEN_AUTHORITY_PATH),
            "sha256": FROZEN_AUTHORITY_SHA256,
            "marker_path": str(FROZEN_AUTHORITY_MARKER),
            "marker_sha256": FROZEN_AUTHORITY_MARKER_SHA256,
            "historical_authority_sha_verified": True,
        },
    }


def _evaluator_artifact_audit(
    evaluator: Any,
    *,
    run_dir: Path,
    checkpoint: str,
) -> dict[str, Any]:
    audit = evaluator.validate_run_artifacts(run_dir, checkpoint)
    _require(isinstance(audit, Mapping), "evaluator artifact audit is invalid")
    return copy.deepcopy(dict(audit))


def validate_extension_method(spec: ExtensionSpec) -> dict[str, Any]:
    """Validate both selector sweeps through their owning evaluator."""

    evaluator = importlib.import_module(spec.evaluator_module)
    optimized_qfg = spec.method_id in ("c_qfg_only", "d_tss_qfg")
    if optimized_qfg:
        _require(
            sha256_file(QFG_OPTIMIZED_SOURCE_LOCK)
            == QFG_OPTIMIZED_SOURCE_LOCK_SHA256,
            "optimized QFG source-lock SHA differs",
        )
        freezer = importlib.import_module(
            "experiments."
            "freeze_tpd_ner_v4_qfg_v2_croa_exact_source_lock"
        )
        verified_lock = freezer.verify_source_lock(
            QFG_OPTIMIZED_SOURCE_LOCK,
            dataset_dir=REPO_ROOT / "datasets",
        )
        _require(
            freezer.payload_sha256(verified_lock)
            == QFG_OPTIMIZED_SOURCE_LOCK_SHA256,
            "optimized QFG source-lock live verification differs",
        )
    roles: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        sweep_path = spec.run_dir / _sweep_filename(checkpoint)
        before_sha = sha256_file(sweep_path)
        payload = load_json(sweep_path)
        if optimized_qfg:
            run_identity = payload.get("run_identity")
            source_locks = (
                run_identity.get("source_locks")
                if isinstance(run_identity, Mapping)
                else None
            )
            _require(
                isinstance(source_locks, Mapping)
                and source_locks.get(QFG_SOURCE_LOCK_KEY)
                == QFG_OPTIMIZED_SOURCE_LOCK_SHA256,
                (
                    f"{spec.method_id}:{checkpoint} is not bound to the "
                    "v2_optimized source lock"
                ),
            )
        audit = _evaluator_artifact_audit(
            evaluator,
            run_dir=spec.run_dir,
            checkpoint=checkpoint,
        )
        evaluator.validate_output_identity(payload, artifact_audit=audit)
        _require(
            sha256_file(sweep_path) == before_sha,
            f"{spec.method_id}:{checkpoint} changed during evaluator validation",
        )
        normalized = normalize_sweep_payload(
            payload,
            method_id=spec.method_id,
            display_name=spec.display_name,
            expected_variant=spec.variant,
            checkpoint=checkpoint,
            sweep_path=sweep_path,
            sweep_sha256=before_sha,
        )
        expected_checkpoint_path = (spec.run_dir / checkpoint).resolve()
        for field, expected in {
            "variant": spec.variant,
            "checkpoint_filename": checkpoint,
            "checkpoint_role": CHECKPOINT_ROLES[checkpoint],
            "checkpoint_epoch": normalized["checkpoint_epoch"],
            "checkpoint_sha256": normalized["checkpoint_sha256"],
        }.items():
            _require(
                audit.get(field) == expected,
                f"{spec.method_id}:{checkpoint} evaluator audit differs: {field}",
            )
        _require(
            audit.get("state_dict_strict_load") is True,
            f"{spec.method_id}:{checkpoint} strict state load was not proven",
        )
        _require(
            Path(str(audit.get("checkpoint_path"))).resolve()
            == expected_checkpoint_path,
            f"{spec.method_id}:{checkpoint} evaluator checkpoint path differs",
        )
        _require(
            Path(normalized["checkpoint_path"]) == expected_checkpoint_path,
            f"{spec.method_id}:{checkpoint} sweep checkpoint path differs",
        )
        _require(
            Path(normalized["run_directory"]) == spec.run_dir.resolve(),
            f"{spec.method_id}:{checkpoint} run directory differs",
        )
        _require(
            sha256_file(Path(normalized["checkpoint_path"]))
            == normalized["checkpoint_sha256"],
            f"{spec.method_id}:{checkpoint} checkpoint SHA differs",
        )
        _require(
            _canonical_equal(
                payload.get("evaluation_source_binding"),
                audit.get("source_binding"),
            ),
            f"{spec.method_id}:{checkpoint} evaluation source binding differs",
        )
        _require(
            _canonical_equal(
                payload.get("evaluator_contract"),
                evaluator.evaluator_contract(),
            ),
            f"{spec.method_id}:{checkpoint} evaluator contract differs",
        )
        sweep_audit = payload.get("audit")
        assignment = (
            sweep_audit.get("device_assignment")
            if isinstance(sweep_audit, Mapping)
            else None
        )
        _require(
            isinstance(assignment, Mapping),
            f"{spec.method_id}:{checkpoint} device assignment is missing",
        )
        expected_gpu = EXPECTED_EVALUATION_GPU.get(spec.method_id)
        if expected_gpu is None:
            _require(
                assignment.get("device") in ("cpu", "cuda:0"),
                f"{spec.method_id}:{checkpoint} test device differs",
            )
        else:
            for field, expected in {
                "device": "cuda:0",
                "physical_gpu_index": expected_gpu,
                "physical_gpu_uuid": PHYSICAL_GPU_UUIDS[expected_gpu],
                "cuda_visible_devices": PHYSICAL_GPU_UUIDS[expected_gpu],
                "device_name": "NVIDIA GeForce RTX 5090",
            }.items():
                _require(
                    assignment.get(field) == expected,
                    (
                        f"{spec.method_id}:{checkpoint} device assignment "
                        f"differs: {field}"
                    ),
                )
        normalized["evaluator_validation"] = {
            "module": spec.evaluator_module,
            "validate_run_artifacts_called": True,
            "validate_output_identity_called": True,
            "state_dict_strict_load": True,
            "evaluation_source_binding_current": True,
            "evaluator_contract_current": True,
            "device_assignment_verified": True,
        }
        roles[normalized["role_name"]] = normalized
    _require(
        set(roles) == set(ROLE_NAMES.values()),
        f"{spec.method_id} selector role matrix differs",
    )
    result = {
        "method_id": spec.method_id,
        "display_name": spec.display_name,
        "variant": spec.variant,
        "roles": roles,
    }
    if optimized_qfg:
        result["optimized_source_lock_binding"] = {
            "path": str(QFG_OPTIMIZED_SOURCE_LOCK.resolve()),
            "sha256": QFG_OPTIMIZED_SOURCE_LOCK_SHA256,
            "live_verified": True,
        }
    return result


def objective_dominates(
    candidate: Mapping[str, Any],
    other: Mapping[str, Any],
) -> bool:
    """Return strict weak Pareto dominance under the frozen directions."""

    candidate_objective = _objective(candidate)
    other_objective = _objective(other)
    no_worse = True
    strictly_better = False
    for field, direction in OBJECTIVE_DIRECTIONS.items():
        left = candidate_objective[field]
        right = other_objective[field]
        if direction == "maximize":
            if left < right - FLOAT_ATOL:
                no_worse = False
                break
            strictly_better |= left > right + FLOAT_ATOL
        else:
            if left > right + FLOAT_ATOL:
                no_worse = False
                break
            strictly_better |= left < right - FLOAT_ATOL
    return no_worse and strictly_better


def objectives_equivalent(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    left_objective = _objective(left)
    right_objective = _objective(right)
    return all(
        _close(left_objective[field], right_objective[field])
        for field in OBJECTIVE_FIELDS
    )


def _all_point_refs(
    methods: Mapping[str, Mapping[str, Any]],
    *,
    method_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    selected = list(method_ids) if method_ids is not None else list(methods)
    refs: list[dict[str, Any]] = []
    for method_id in selected:
        method = methods[method_id]
        for role_name in ("pd_primary", "miou_secondary"):
            role = method["roles"][role_name]
            for index, point in enumerate(role["raw_points"]):
                refs.append(
                    {
                        "point_id": f"{method_id}:{role_name}:{index}",
                        "method_id": method_id,
                        "role_name": role_name,
                        "checkpoint": role["checkpoint"],
                        "point_index": index,
                        "threshold": point["threshold"],
                        "point": point,
                    }
                )
    return refs


def _candidate_frontier_membership(
    methods: Mapping[str, Mapping[str, Any]],
    candidate_id: str,
    *,
    comparison_method_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    comparison_ids = (
        list(comparison_method_ids)
        if comparison_method_ids is not None
        else list(methods)
    )
    _require(candidate_id in comparison_ids, "candidate is outside comparison pool")
    pool = _all_point_refs(methods, method_ids=comparison_ids)
    candidates = [ref for ref in pool if ref["method_id"] == candidate_id]
    membership: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        dominated_by: list[str] = []
        equivalent_other_methods: list[str] = []
        for other in pool:
            if other["point_id"] == candidate["point_id"]:
                continue
            if objective_dominates(other["point"], candidate["point"]):
                dominated_by.append(other["point_id"])
            elif (
                other["method_id"] != candidate_id
                and objectives_equivalent(other["point"], candidate["point"])
            ):
                equivalent_other_methods.append(other["point_id"])
        non_dominated = not dominated_by
        exclusive = non_dominated and not equivalent_other_methods
        equivalent_outside_qfg_group = [
            point_id
            for point_id in equivalent_other_methods
            if point_id.split(":", 1)[0] not in QFG_CANDIDATE_GROUP
        ]
        contributes_qfg_frontier = (
            non_dominated and not equivalent_outside_qfg_group
        )
        membership[candidate["point_id"]] = {
            **candidate,
            "is_joint_non_dominated": non_dominated,
            "is_exclusive": exclusive,
            "contributes_qfg_frontier": contributes_qfg_frontier,
            "dominated_by": dominated_by,
            "equivalent_other_method_points": equivalent_other_methods,
            "equivalent_outside_qfg_group": (
                equivalent_outside_qfg_group
            ),
        }
    return {
        "pool_point_count": len(pool),
        "candidate_point_count": len(candidates),
        "membership": membership,
    }


def _threshold_support(
    membership: Mapping[str, Mapping[str, Any]],
    candidate_id: str,
    *,
    exclusive_only: bool,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    by_role: dict[str, list[Mapping[str, Any]]] = {}
    for value in membership.values():
        by_role.setdefault(str(value["role_name"]), []).append(value)
    for role_name, values in by_role.items():
        ordered = sorted(values, key=lambda value: int(value["point_index"]))
        for left, right in zip(ordered, ordered[1:]):
            left_qualified = (
                left["is_exclusive"]
                if exclusive_only
                else left["contributes_qfg_frontier"]
            )
            right_qualified = (
                right["is_exclusive"]
                if exclusive_only
                else right["contributes_qfg_frontier"]
            )
            if left_qualified and right_qualified:
                evidence.append(
                    {
                        "support_type": "adjacent_threshold_points",
                        "exclusive_only": exclusive_only,
                        "all_points_exclusive": (
                            left["is_exclusive"] and right["is_exclusive"]
                        ),
                        "method_id": candidate_id,
                        "role_name": role_name,
                        "point_ids": [left["point_id"], right["point_id"]],
                        "thresholds": [left["threshold"], right["threshold"]],
                        "objectives": [
                            _objective(left["point"]),
                            _objective(right["point"]),
                        ],
                    }
                )
    return evidence


def _point_membership_against_pool(
    point: Mapping[str, Any],
    *,
    candidate_id: str,
    pool: Sequence[Mapping[str, Any]],
) -> tuple[bool, bool, bool]:
    if any(objective_dominates(other["point"], point) for other in pool):
        return False, False, False
    equivalent_methods = {
        other["method_id"]
        for other in pool
        if (
            other["method_id"] != candidate_id
            and objectives_equivalent(other["point"], point)
        )
    }
    exclusive = not equivalent_methods
    contributes_qfg_frontier = not (
        equivalent_methods - QFG_CANDIDATE_GROUP
    )
    return True, exclusive, contributes_qfg_frontier


def _budget_support(
    methods: Mapping[str, Mapping[str, Any]],
    candidate_id: str,
    *,
    comparison_method_ids: Iterable[str],
    exclusive_only: bool,
) -> list[dict[str, Any]]:
    comparison_ids = list(comparison_method_ids)
    pool = _all_point_refs(methods, method_ids=comparison_ids)
    evidence: list[dict[str, Any]] = []
    candidate = methods[candidate_id]
    for role_name in ("pd_primary", "miou_secondary"):
        budgets = candidate["roles"][role_name]["fa_budget_points"]
        membership = {}
        for key in BUDGET_KEYS:
            (
                non_dominated,
                exclusive,
                contributes_qfg_frontier,
            ) = _point_membership_against_pool(
                budgets[key],
                candidate_id=candidate_id,
                pool=pool,
            )
            membership[key] = {
                "non_dominated": non_dominated,
                "exclusive": exclusive,
                "contributes_qfg_frontier": contributes_qfg_frontier,
            }
        for left_key, right_key in zip(BUDGET_KEYS, BUDGET_KEYS[1:]):
            left_qualified = membership[left_key][
                (
                    "exclusive"
                    if exclusive_only
                    else "contributes_qfg_frontier"
                )
            ]
            right_qualified = membership[right_key][
                (
                    "exclusive"
                    if exclusive_only
                    else "contributes_qfg_frontier"
                )
            ]
            distinct_thresholds = not _close(
                budgets[left_key]["threshold"],
                budgets[right_key]["threshold"],
            )
            if left_qualified and right_qualified and distinct_thresholds:
                evidence.append(
                    {
                        "support_type": "adjacent_fa_budget_interval",
                        "exclusive_only": exclusive_only,
                        "all_points_exclusive": (
                            membership[left_key]["exclusive"]
                            and membership[right_key]["exclusive"]
                        ),
                        "method_id": candidate_id,
                        "role_name": role_name,
                        "fa_budget_keys": [left_key, right_key],
                        "fa_budgets": [
                            BUDGET_BY_KEY[left_key],
                            BUDGET_BY_KEY[right_key],
                        ],
                        "objectives": [
                            _objective(budgets[left_key]),
                            _objective(budgets[right_key]),
                        ],
                    }
                )
    return evidence


def aligned_strict_improvement(
    methods: Mapping[str, Mapping[str, Any]],
    *,
    candidate_id: str,
    reference_id: str,
) -> dict[str, Any]:
    """Assess full-five-objective dominance at aligned selector locations."""

    candidate = methods[candidate_id]
    reference = methods[reference_id]
    locations: dict[str, dict[str, Any]] = {}
    for role_name in ("pd_primary", "miou_secondary"):
        candidate_role = candidate["roles"][role_name]
        reference_role = reference["roles"][role_name]
        fixed_key = f"{role_name}:fixed0.5"
        fixed_dominates = objective_dominates(
            candidate_role["fixed_threshold_0_5"],
            reference_role["fixed_threshold_0_5"],
        )
        locations[fixed_key] = {
            "role_name": role_name,
            "location": "fixed0.5",
            "candidate_dominates": fixed_dominates,
            "candidate_objectives": _objective(
                candidate_role["fixed_threshold_0_5"]
            ),
            "reference_objectives": _objective(
                reference_role["fixed_threshold_0_5"]
            ),
        }
        for budget_key in BUDGET_KEYS:
            location_key = f"{role_name}:fa_budget:{budget_key}"
            candidate_point = candidate_role["fa_budget_points"][budget_key]
            reference_point = reference_role["fa_budget_points"][budget_key]
            locations[location_key] = {
                "role_name": role_name,
                "location": "fa_budget",
                "fa_budget_key": budget_key,
                "fa_budget": BUDGET_BY_KEY[budget_key],
                "candidate_dominates": objective_dominates(
                    candidate_point,
                    reference_point,
                ),
                "candidate_objectives": _objective(candidate_point),
                "reference_objectives": _objective(reference_point),
            }

    support: list[dict[str, Any]] = []
    for role_name in ("pd_primary", "miou_secondary"):
        for left_key, right_key in zip(BUDGET_KEYS, BUDGET_KEYS[1:]):
            left_location = locations[
                f"{role_name}:fa_budget:{left_key}"
            ]
            right_location = locations[
                f"{role_name}:fa_budget:{right_key}"
            ]
            if (
                left_location["candidate_dominates"]
                and right_location["candidate_dominates"]
            ):
                support.append(
                    {
                        "support_type": "adjacent_fa_budget_strict_improvement",
                        "role_name": role_name,
                        "fa_budget_keys": [left_key, right_key],
                    }
                )
    for location in ("fixed0.5", *BUDGET_KEYS):
        if location == "fixed0.5":
            keys = [
                "pd_primary:fixed0.5",
                "miou_secondary:fixed0.5",
            ]
        else:
            keys = [
                f"pd_primary:fa_budget:{location}",
                f"miou_secondary:fa_budget:{location}",
            ]
        if all(locations[key]["candidate_dominates"] for key in keys):
            support.append(
                {
                    "support_type": "both_selector_roles_strict_improvement",
                    "location": location,
                }
            )
    return {
        "candidate_id": candidate_id,
        "reference_id": reference_id,
        "locations": locations,
        "strict_location_count": sum(
            int(value["candidate_dominates"])
            for value in locations.values()
        ),
        "non_isolated_support": support,
        "non_isolated_strict_improvement": bool(support),
    }


def analyze_candidate(
    methods: Mapping[str, Mapping[str, Any]],
    *,
    candidate_id: str,
    direct_reference_id: str,
    comparison_method_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    comparison_ids = (
        list(comparison_method_ids)
        if comparison_method_ids is not None
        else list(methods)
    )
    membership_result = _candidate_frontier_membership(
        methods,
        candidate_id,
        comparison_method_ids=comparison_ids,
    )
    membership = membership_result["membership"]
    threshold_support = _threshold_support(
        membership,
        candidate_id,
        exclusive_only=False,
    )
    budget_support = _budget_support(
        methods,
        candidate_id,
        comparison_method_ids=comparison_ids,
        exclusive_only=False,
    )
    non_isolated_support = [*threshold_support, *budget_support]
    exclusive_threshold_support = _threshold_support(
        membership,
        candidate_id,
        exclusive_only=True,
    )
    exclusive_budget_support = _budget_support(
        methods,
        candidate_id,
        comparison_method_ids=comparison_ids,
        exclusive_only=True,
    )
    exclusive_non_isolated_support = [
        *exclusive_threshold_support,
        *exclusive_budget_support,
    ]
    aligned = aligned_strict_improvement(
        methods,
        candidate_id=candidate_id,
        reference_id=direct_reference_id,
    )
    exclusive_points = [
        {
            "point_id": value["point_id"],
            "role_name": value["role_name"],
            "checkpoint": value["checkpoint"],
            "point_index": value["point_index"],
            "threshold": value["threshold"],
            "objectives": _objective(value["point"]),
        }
        for value in membership.values()
        if value["is_exclusive"]
    ]
    isolated_only = bool(exclusive_points) and not non_isolated_support
    if non_isolated_support and aligned["non_isolated_strict_improvement"]:
        status = RELATIVE_IMPROVED
        rationale = (
            "candidate has a non-isolated joint-frontier interval and a "
            "non-isolated aligned strict improvement over its direct "
            "factorial reference"
        )
    elif exclusive_non_isolated_support:
        status = PARETO_MIXED_TRADEOFF
        rationale = (
            "candidate contributes a non-isolated exclusive joint-frontier "
            "interval without non-isolated aligned strict dominance"
        )
    else:
        status = DOMINATED
        rationale = (
            "candidate has neither non-isolated aligned strict improvement "
            "nor a non-isolated exclusive contribution across the complete "
            "selector sweeps or Fa-budget envelopes"
        )
    _require(status in CANDIDATE_STATES, "invalid candidate status")
    return {
        "candidate_id": candidate_id,
        "direct_reference_id": direct_reference_id,
        "status": status,
        "rationale": rationale,
        "objective_directions": copy.deepcopy(OBJECTIVE_DIRECTIONS),
        "comparison_method_ids": comparison_ids,
        "joint_pool_point_count": membership_result["pool_point_count"],
        "candidate_point_count": membership_result["candidate_point_count"],
        "joint_non_dominated_point_count": sum(
            int(value["is_joint_non_dominated"])
            for value in membership.values()
        ),
        "qfg_frontier_contribution_point_count": sum(
            int(value["contributes_qfg_frontier"])
            for value in membership.values()
        ),
        "exclusive_joint_frontier_point_count": len(exclusive_points),
        "exclusive_joint_frontier_points": exclusive_points,
        "non_isolated_support_count": len(non_isolated_support),
        "non_isolated_support": non_isolated_support,
        "exclusive_non_isolated_support_count": len(
            exclusive_non_isolated_support
        ),
        "exclusive_non_isolated_support": exclusive_non_isolated_support,
        "isolated_exclusive_points_only": isolated_only,
        "aligned_strict_improvement": aligned,
    }


def decide_final_recipe(
    methods: Mapping[str, Mapping[str, Any]],
    c_analysis: Mapping[str, Any],
    d_analysis: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply Decision F-F, with C preferred unless D adds unique value."""

    c_status = c_analysis["status"]
    d_status = d_analysis["status"]
    if c_status == DOMINATED and d_status == DOMINATED:
        return {
            "decision": "FALLBACK_TO_FROZEN_V4",
            "selected_method_id": "v4",
            "selected_variant": methods["v4"]["variant"],
            "final_training_uses_tss": False,
            "final_inference_uses_tss": False,
            "query_fg_stage_success": False,
            "reason": "both C and D are DOMINATED; return to frozen V4",
            "d_unique_over_c": None,
        }
    if c_status == DOMINATED and d_status != DOMINATED:
        return {
            "decision": "SELECT_D_TSS_QFG",
            "selected_method_id": "d_tss_qfg",
            "selected_variant": methods["d_tss_qfg"]["variant"],
            "final_training_uses_tss": True,
            "final_inference_uses_tss": False,
            "query_fg_stage_success": True,
            "reason": "D contributes non-isolated value while C is DOMINATED",
            "d_unique_over_c": True,
        }
    if c_status != DOMINATED and d_status == DOMINATED:
        return {
            "decision": "SELECT_C_QFG_ONLY",
            "selected_method_id": "c_qfg_only",
            "selected_variant": methods["c_qfg_only"]["variant"],
            "final_training_uses_tss": False,
            "final_inference_uses_tss": False,
            "query_fg_stage_success": True,
            "reason": "C contributes non-isolated value and D is DOMINATED",
            "d_unique_over_c": False,
        }

    d_vs_c = analyze_candidate(
        methods,
        candidate_id="d_tss_qfg",
        direct_reference_id="c_qfg_only",
        comparison_method_ids=("c_qfg_only", "d_tss_qfg"),
    )
    d_has_unique_over_c = (
        d_vs_c["exclusive_non_isolated_support_count"] > 0
    )
    d_minus_c_strict = d_vs_c["aligned_strict_improvement"][
        "non_isolated_strict_improvement"
    ]
    b_qualifies = FROZEN_TSS_B_DECISION in (
        RELATIVE_IMPROVED,
        PARETO_MIXED_TRADEOFF,
    )
    condition_1 = (
        d_analysis["exclusive_non_isolated_support_count"] > 0
    )
    condition_2 = (
        d_status == RELATIVE_IMPROVED and d_minus_c_strict
    )
    condition_3 = b_qualifies and d_has_unique_over_c
    d_retained = condition_1 or condition_2 or condition_3
    if d_retained:
        decision = "SELECT_D_TSS_QFG"
        selected = "d_tss_qfg"
        reason = (
            "D provides a non-isolated strict or Pareto contribution that C "
            "does not provide; retain TSS for training only"
        )
    else:
        decision = "SELECT_C_QFG_ONLY"
        selected = "c_qfg_only"
        reason = (
            "C and D are approximately equivalent under Decision F-F, or D "
            "adds no non-isolated contribution unavailable from C; prefer C"
        )
    return {
        "decision": decision,
        "selected_method_id": selected,
        "selected_variant": methods[selected]["variant"],
        "final_training_uses_tss": d_retained,
        "final_inference_uses_tss": False,
        "query_fg_stage_success": True,
        "reason": reason,
        "d_unique_over_c": d_retained,
        "d_vs_c_analysis": d_vs_c,
        "frozen_b_status": FROZEN_TSS_B_DECISION,
        "decision_f_f_conditions": {
            "condition_1_d_unique_global_pareto_interval": condition_1,
            "condition_2_d_relative_improved_and_d_minus_c": condition_2,
            "condition_3_b_qualifies_and_d_unique_on_cd_frontier": (
                condition_3
            ),
            "d_unique_non_isolated_interval_over_c": d_has_unique_over_c,
            "d_minus_c_non_isolated_strict_improvement": d_minus_c_strict,
            "b_has_frozen_relative_or_pareto_status": b_qualifies,
        },
    }


def _public_role(role: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(role[key])
        for key in (
            "checkpoint",
            "checkpoint_role",
            "role_name",
            "checkpoint_epoch",
            "checkpoint_sha256",
            "checkpoint_path",
            "run_directory",
            "fixed_threshold_0_5",
            "fa_budget_points",
            "raw_point_count",
            "sweep_binding",
        )
    } | (
        {"evaluator_validation": copy.deepcopy(role["evaluator_validation"])}
        if "evaluator_validation" in role
        else {}
    )


def _public_method(method: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "method_id": method["method_id"],
        "display_name": method["display_name"],
        "variant": method["variant"],
        "roles": {
            role_name: _public_role(method["roles"][role_name])
            for role_name in ("pd_primary", "miou_secondary")
        },
    }
    if "optimized_source_lock_binding" in method:
        result["optimized_source_lock_binding"] = copy.deepcopy(
            method["optimized_source_lock_binding"]
        )
    return result


def _snapshot_bindings(
    methods: Mapping[str, Mapping[str, Any]],
    authority_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    snapshot = {
        key: copy.deepcopy(dict(value))
        for key, value in authority_bindings.items()
    }
    for method_id in ("a_control", "b_tss", "c_qfg_only", "d_tss_qfg"):
        method = methods[method_id]
        if "optimized_source_lock_binding" in method:
            snapshot[f"source_lock:{method_id}"] = copy.deepcopy(
                method["optimized_source_lock_binding"]
            )
        for role_name in ("pd_primary", "miou_secondary"):
            role = method["roles"][role_name]
            snapshot[f"sweep:{method_id}:{role_name}"] = copy.deepcopy(
                role["sweep_binding"]
            )
            snapshot[f"checkpoint:{method_id}:{role_name}"] = {
                "path": role["checkpoint_path"],
                "sha256": role["checkpoint_sha256"],
            }
    return snapshot


def verify_snapshot(snapshot: Mapping[str, Mapping[str, str]]) -> None:
    for label, binding in snapshot.items():
        _require(
            sha256_file(Path(binding["path"])) == binding["sha256"],
            f"input changed during final selection: {label}",
        )


def build_report(
    methods: Mapping[str, Mapping[str, Any]],
    *,
    authority_binding: Mapping[str, Any],
    input_bindings: Mapping[str, Mapping[str, str]],
    posttraining_closure_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    required_methods = {
        "baseline",
        "v1",
        "v2",
        "v3",
        "v4",
        "a_control",
        "b_tss",
        "c_qfg_only",
        "d_tss_qfg",
    }
    _require(set(methods) == required_methods, "final method matrix differs")
    c_analysis = analyze_candidate(
        methods,
        candidate_id="c_qfg_only",
        direct_reference_id="a_control",
    )
    d_analysis = analyze_candidate(
        methods,
        candidate_id="d_tss_qfg",
        direct_reference_id="b_tss",
    )
    selection = decide_final_recipe(methods, c_analysis, d_analysis)
    deployment_selection = closure_policy.select_deployment_operating_point(
        methods[selection["selected_method_id"]]
    )
    selection = copy.deepcopy(dict(selection))
    selection["deployment"] = copy.deepcopy(deployment_selection)
    closure_binding = (
        closure_policy.load_closure_lock(verify_sources=True)[1]
        if posttraining_closure_binding is None
        else copy.deepcopy(dict(posttraining_closure_binding))
    )
    _require(
        closure_policy.is_sha256(closure_binding.get("sha256")),
        "post-training closure source-lock SHA is invalid",
    )
    _require(
        closure_binding.get("schema") == closure_policy.LOCK_SCHEMA
        and closure_binding.get("source_count")
        == len(closure_policy.POSTTRAINING_SOURCE_PATHS)
        and closure_binding.get("training_source_lock_sha256")
        == QFG_OPTIMIZED_SOURCE_LOCK_SHA256
        and closure_binding.get("verified_live") is True,
        "post-training closure source-lock identity differs",
    )
    _require(
        closure_binding.get("policy_summary_sha256")
        == closure_policy.policy_summary_sha256(),
        "post-training deployment policy binding differs",
    )
    verify_snapshot(input_bindings)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "scope": SCOPE,
        "official_test_accessed": False,
        "validation_count": VALIDATION_COUNT,
        "target_count": TARGET_COUNT,
        "tiny_target_count": TINY_TARGET_COUNT,
        "validation_split_sha256": VALIDATION_SPLIT_SHA256,
        "authority_binding": copy.deepcopy(dict(authority_binding)),
        "input_bindings": {
            key: copy.deepcopy(dict(value))
            for key, value in sorted(input_bindings.items())
        },
        "metric_contract": {
            "fixed_threshold": 0.5,
            "fa_budgets": list(FA_BUDGETS),
            "objective_directions": copy.deepcopy(OBJECTIVE_DIRECTIONS),
            "all_selector_sweep_points_used": True,
            "cross_checkpoint_point_pooling_for_metric_selection": False,
            "joint_pooling_for_posthoc_pareto_analysis_only": True,
            "non_isolated_support_required": True,
            "non_isolated_support_modes": [
                "adjacent_threshold_points",
                "adjacent_fa_budget_interval",
            ],
            "shared_point_policy": (
                "C/D-equivalent frontier intervals may establish the QFG "
                "stage jointly, but points shared with any pre-QFG method "
                "do not establish a C/D contribution"
            ),
            "relative_improved_requires": (
                "non-isolated joint-frontier support plus "
                "non-isolated aligned strict dominance over the direct "
                "factorial reference"
            ),
            "pareto_mixed_tradeoff_requires": (
                "a non-isolated exclusive joint-frontier interval"
            ),
            "deployment_point_policy": closure_policy.policy_summary(),
        },
        "methods": {
            method_id: _public_method(methods[method_id])
            for method_id in (
                "baseline",
                "v1",
                "v2",
                "v3",
                "v4",
                "a_control",
                "b_tss",
                "c_qfg_only",
                "d_tss_qfg",
            )
        },
        "candidate_assessments": {
            "c_qfg_only": c_analysis,
            "d_tss_qfg": d_analysis,
        },
        "selection": selection,
        "deployment_selection": copy.deepcopy(deployment_selection),
        "decision": selection["decision"],
        "query_fg_stage_success": selection["query_fg_stage_success"],
        "final_model_engineering_selected": selection[
            "query_fg_stage_success"
        ],
        "final_model_established": selection["query_fg_stage_success"],
        "final_training_uses_tss": selection["final_training_uses_tss"],
        "final_inference_uses_tss": False,
        "frozen_tss_b_decision": FROZEN_TSS_B_DECISION,
        "paper_core_established": False,
        "stability_claim_supported": False,
        "claim_boundary": {
            "single_seed_only": True,
            "internal_validation_only": True,
            "official_test_claim": False,
            "cross_seed_stability_claim": False,
            "cross_dataset_claim": False,
            "universal_dominance_claim": False,
        },
        "posttraining_closure_source_lock": closure_binding,
        "write_once": True,
    }


def build_formal_report(
    *,
    posttraining_closure_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    closure_binding = (
        closure_policy.load_closure_lock(verify_sources=True)[1]
        if posttraining_closure_binding is None
        else copy.deepcopy(dict(posttraining_closure_binding))
    )
    authority = validate_frozen_authority()
    methods = dict(authority["methods"])
    for spec in EXTENSION_SPECS:
        methods[spec.method_id] = validate_extension_method(spec)
    snapshot = _snapshot_bindings(methods, authority["bindings"])
    verify_snapshot(snapshot)
    return build_report(
        methods,
        authority_binding=authority["authority_binding"],
        input_bindings=snapshot,
        posttraining_closure_binding=closure_binding,
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    selection = report["selection"]
    lines = [
        "# TPD-NER-V4 + QFG-V2-CROA formal800 最终选择",
        "",
        f"- Decision：`{report['decision']}`",
        f"- 最终选择：`{selection['selected_method_id']}`",
        (
            "- 部署 checkpoint："
            f"`{report['deployment_selection']['selected']['checkpoint']}` / "
            f"`{report['deployment_selection']['selected']['checkpoint_role']}`，"
            f"epoch={report['deployment_selection']['selected']['checkpoint_epoch']}"
        ),
        (
            "- 部署工作点："
            f"`{report['deployment_selection']['selected']['operating_point_source']}`，"
            f"threshold={report['deployment_selection']['selected']['threshold']:.10g}"
        ),
        (
            "- Post-training closure lock："
            f"`{report['posttraining_closure_source_lock']['sha256']}`"
        ),
        f"- C 状态：`{report['candidate_assessments']['c_qfg_only']['status']}`",
        f"- D 状态：`{report['candidate_assessments']['d_tss_qfg']['status']}`",
        (
            "- 最终训练使用 TSS："
            f"`{str(report['final_training_uses_tss']).lower()}`"
        ),
        "- 最终推理使用 TSS：`false`",
        "- 数据边界：NUDT-SIRST 530/133 内部分割，seed 42；未访问官方测试集。",
        "- `paper_core_established=false`",
        "- `stability_claim_supported=false`",
        "",
        "## 固定阈值 0.5",
        "",
        (
            "| 方法 | selector | epoch | role | checkpoint SHA256 | "
            "Pd | Fa | mIoU | tiny-Pd | 错误目标/图 |"
        ),
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in report["methods"].values():
        for role_name in ("pd_primary", "miou_secondary"):
            role = method["roles"][role_name]
            point = role["fixed_threshold_0_5"]
            lines.append(
                "| {method} | {checkpoint} | {epoch} | `{role}` | "
                "`{sha}` | {pd:.9f} | {fa:.9g} | {miou:.9f} | "
                "{tiny:.9f} | {errors:.9f} |".format(
                    method=method["method_id"],
                    checkpoint=role["checkpoint"],
                    epoch=role["checkpoint_epoch"],
                    role=role["checkpoint_role"],
                    sha=role["checkpoint_sha256"],
                    pd=point["pd"],
                    fa=point["fa"],
                    miou=point["miou"],
                    tiny=point["tiny_pd"],
                    errors=point["false_objects_per_image"],
                )
            )

    lines.extend(
        [
            "",
            "## 五个 Fa budget",
            "",
            (
                "| 方法 | selector | Fa budget | Pd | achieved Fa | "
                "mIoU | tiny-Pd | 错误目标/图 |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in report["methods"].values():
        for role_name in ("pd_primary", "miou_secondary"):
            role = method["roles"][role_name]
            for key in BUDGET_KEYS:
                point = role["fa_budget_points"][key]
                lines.append(
                    "| {method} | {checkpoint} | {budget:.9g} | "
                    "{pd:.9f} | {fa:.9g} | {miou:.9f} | "
                    "{tiny:.9f} | {errors:.9f} |".format(
                        method=method["method_id"],
                        checkpoint=role["checkpoint"],
                        budget=BUDGET_BY_KEY[key],
                        pd=point["pd"],
                        fa=point["fa"],
                        miou=point["miou"],
                        tiny=point["tiny_pd"],
                        errors=point["false_objects_per_image"],
                    )
                )

    lines.extend(["", "## 联合非支配分析", ""])
    for candidate_id in ("c_qfg_only", "d_tss_qfg"):
        assessment = report["candidate_assessments"][candidate_id]
        strict_supported = assessment["aligned_strict_improvement"][
            "non_isolated_strict_improvement"
        ]
        lines.extend(
            [
                f"### {candidate_id}",
                "",
                f"- 状态：`{assessment['status']}`",
                (
                    "- 联合池/候选点："
                    f"{assessment['joint_pool_point_count']}/"
                    f"{assessment['candidate_point_count']}"
                ),
                (
                    "- 独有非支配点："
                    f"{assessment['exclusive_joint_frontier_point_count']}"
                ),
                (
                    "- 非孤立支持："
                    f"{assessment['non_isolated_support_count']}"
                ),
                (
                    "- 其中独有非孤立支持："
                    f"{assessment['exclusive_non_isolated_support_count']}"
                ),
                (
                    "- 对直接因子对照的非孤立严格改善："
                    f"`{str(strict_supported).lower()}`"
                ),
                f"- 解释：{assessment['rationale']}",
            ]
        )
        for evidence in assessment["non_isolated_support"]:
            if evidence["support_type"] == "adjacent_threshold_points":
                description = (
                    f"role={evidence['role_name']}, "
                    f"thresholds={evidence['thresholds']}, "
                    f"exclusive={evidence['all_points_exclusive']}"
                )
            else:
                description = (
                    f"role={evidence['role_name']}, "
                    f"Fa budgets={evidence['fa_budgets']}, "
                    f"exclusive={evidence['all_points_exclusive']}"
                )
            lines.append(
                f"- 非孤立证据 `{evidence['support_type']}`：{description}"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision F-F",
            "",
            selection["reason"],
            "",
            (
                "该输出只建立 seed 42 当前内部验证划分上的工程选择，"
                "不建立跨随机性、跨数据集或官方测试结论。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_temporary(path: Path, content: bytes) -> Path:
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
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _unlink_if_same_file(path: Path, owned_link: Path) -> bool:
    """Remove ``path`` only while it still names this process's inode."""

    try:
        path_stat = path.stat(follow_symlinks=False)
        owned_stat = owned_link.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (
        path_stat.st_dev != owned_stat.st_dev
        or path_stat.st_ino != owned_stat.st_ino
    ):
        return False
    path.unlink()
    return True


def write_outputs_once(
    report: Mapping[str, Any],
    *,
    json_output: Path = JSON_OUTPUT,
    markdown_output: Path = MARKDOWN_OUTPUT,
) -> tuple[Path, Path]:
    """Create JSON and Markdown without replacing either existing output."""

    json_output = Path(json_output).resolve()
    markdown_output = Path(markdown_output).resolve()
    _require(json_output != markdown_output, "JSON and Markdown paths collide")
    for path in (json_output, markdown_output):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to replace final output: {path}")
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_bytes = (
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    markdown_bytes = render_markdown(report).encode("utf-8")
    json_temporary = _write_temporary(json_output, json_bytes)
    markdown_temporary = _write_temporary(markdown_output, markdown_bytes)
    json_created = False
    markdown_created = False
    try:
        try:
            os.link(json_temporary, json_output)
            json_created = True
            os.link(markdown_temporary, markdown_output)
            markdown_created = True
        except BaseException as exc:
            if markdown_created:
                _unlink_if_same_file(
                    markdown_output,
                    markdown_temporary,
                )
            if json_created:
                _unlink_if_same_file(json_output, json_temporary)
            if isinstance(exc, FileExistsError):
                raise FileExistsError(
                    "refusing to replace an existing final-selection output"
                ) from exc
            raise
        directory_descriptors = {
            os.open(str(json_output.parent), os.O_RDONLY),
            os.open(str(markdown_output.parent), os.O_RDONLY),
        }
        try:
            for descriptor in directory_descriptors:
                os.fsync(descriptor)
        finally:
            for descriptor in directory_descriptors:
                os.close(descriptor)
    finally:
        json_temporary.unlink(missing_ok=True)
        markdown_temporary.unlink(missing_ok=True)
    return json_output, markdown_output


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=MARKDOWN_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _argument_parser().parse_args(argv)
    report = build_formal_report()
    json_path, markdown_path = write_outputs_once(
        report,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
    )
    print(f"FINAL_SELECTION_JSON={json_path}")
    print(f"FINAL_SELECTION_MARKDOWN={markdown_path}")
    print(f"FINAL_SELECTION_DECISION={report['decision']}")


__all__ = [
    "BUDGET_KEYS",
    "CANDIDATE_STATES",
    "DOMINATED",
    "FA_BUDGETS",
    "FROZEN_AUTHORITY_PATH",
    "FROZEN_AUTHORITY_SHA256",
    "JSON_OUTPUT",
    "MARKDOWN_OUTPUT",
    "OBJECTIVE_DIRECTIONS",
    "PARETO_MIXED_TRADEOFF",
    "RELATIVE_IMPROVED",
    "aligned_strict_improvement",
    "analyze_candidate",
    "build_formal_report",
    "build_report",
    "decide_final_recipe",
    "main",
    "normalize_point",
    "normalize_sweep_payload",
    "objective_dominates",
    "objectives_equivalent",
    "render_markdown",
    "sha256_file",
    "validate_extension_method",
    "validate_frozen_authority",
    "write_outputs_once",
]


if __name__ == "__main__":
    main()
