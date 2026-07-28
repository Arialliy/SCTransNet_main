#!/usr/bin/env python3
"""Read-only aggregate for the two V4 tail-formula counterfactual artifacts.

This postprocessor does not run inference, alter a checkpoint, or revise the
formal V3 decision.  It consumes exactly the fixed ``best`` and ``best_miou``
counterfactual JSON files, revalidates their immutable inputs and source hashes,
and applies the formula-selection rule preregistered in
``SCTransNet_NER_V3失败复盘与V4_Tail_Aware修改方案.md``.

Publication is append-only.  The JSON and Markdown report are written first,
and the completion marker is written last.  Existing paths are never replaced.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_v4_tail_formula_counterfactual as evaluator,
)
from experiments import evaluate_pd_fa_sweep as metric_core  # noqa: E402


AGGREGATE_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_formula_"
    "counterfactual_aggregate_v1"
)
MARKER_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_tail_formula_"
    "counterfactual_postprocess_complete_v1"
)
POSTPROCESS_PATH = Path(__file__).resolve()
LOCAL_FORMULA_MODES = ("direct_tail", "complement_tail")
ALL_FORMULA_MODES = (
    "legacy_global",
    "direct_tail",
    "complement_tail",
)
QUALIFYING_BUDGETS_PER_ROLE = 4
REQUIRED_STRICT_BUDGET_IMPROVEMENTS = 1
INPUT_HASH_KEYS = (
    "source_checkpoint",
    "canonical_v3_sweep",
    "protocol.json",
    "split.json",
    "summary.json",
    "metrics.jsonl",
)
SHARED_INPUT_HASH_KEYS = (
    "protocol.json",
    "split.json",
    "summary.json",
    "metrics.jsonl",
)
FIXED_METRIC_NAMES = (
    "matched_target_count",
    "fa",
    "miou",
)
AGGREGATE_DIR = evaluator.DEFAULT_OUTPUT_ROOT / evaluator.DATASET
AGGREGATE_JSON_PATH = (
    AGGREGATE_DIR
    / "tpd_ner_v8_mprs_dch_v4_tail_formula_counterfactual_aggregate.json"
)
AGGREGATE_MD_PATH = (
    AGGREGATE_DIR
    / "tpd_ner_v8_mprs_dch_v4_tail_formula_counterfactual_aggregate.md"
)
COMPLETION_MARKER_PATH = AGGREGATE_DIR / "POSTPROCESS_COMPLETE.json"


def _require(condition: bool, message: str) -> None:
    """Optimization-independent invariant check."""

    if not condition:
        raise ValueError(message)


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, observed={observed!r}"
        )


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return dict(value)


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    return value


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{location} must be finite")
    return normalized


def _nonnegative_integer(value: Any, location: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{location} must be a non-negative integer")
    return value


def _positive_integer(value: Any, location: str) -> int:
    normalized = _nonnegative_integer(value, location)
    if normalized < 1:
        raise ValueError(f"{location} must be positive")
    return normalized


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    value = Path(path)
    if not value.is_file() or value.is_symlink():
        raise ValueError(f"expected a regular non-symlink file: {value}")
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, location: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{location} is not a lowercase SHA-256 digest")
    return str(value)


def _normalize_tail_thresholds(
    value: Any,
    *,
    location: str,
) -> dict[int, float]:
    """Normalize JSON-stringified stage keys and require the frozen values."""

    ready = _require_mapping(value, location)
    normalized: dict[int, float] = {}
    for key, raw_threshold in ready.items():
        if type(key) is int:
            stage = key
        elif isinstance(key, str) and key in {"4", "3", "2"}:
            stage = int(key)
        else:
            raise ValueError(f"{location} has an invalid stage key: {key!r}")
        if stage in normalized:
            raise ValueError(f"{location} duplicates stage {stage}")
        normalized[stage] = _finite_number(
            raw_threshold,
            f"{location}[{stage}]",
        )
    expected = dict(evaluator.v4_model_source.DEFAULT_TAIL_Z_THRESHOLDS)
    _require_equal(location, normalized, expected)
    return normalized


def _canonical_json_bytes(value: Any) -> bytes:
    ready = metric_core.json_ready(value)
    return (
        json.dumps(
            ready,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    ready = metric_core.json_ready(value)
    return (
        json.dumps(
            ready,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = Path(path)
    if not value.is_file() or value.is_symlink():
        raise FileNotFoundError(value)
    payload = json.loads(value.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {value}")
    return payload


def _fixed_dependency_paths(checkpoint: str) -> dict[str, Path]:
    if checkpoint not in evaluator.CHECKPOINTS:
        raise ValueError(f"unsupported checkpoint: {checkpoint}")
    return {
        "source_checkpoint": (
            evaluator.FORMAL_RUN_DIR / checkpoint
        ).resolve(),
        "canonical_v3_sweep": evaluator.canonical_sweep_path(checkpoint),
        "protocol.json": evaluator.FORMAL_RUN_DIR / "protocol.json",
        "split.json": evaluator.FORMAL_RUN_DIR / "split.json",
        "summary.json": evaluator.FORMAL_RUN_DIR / "summary.json",
        "metrics.jsonl": evaluator.FORMAL_RUN_DIR / "metrics.jsonl",
    }


def _validate_hash_record(
    record: Any,
    *,
    expected_path: Path,
    location: str,
    verify_live: bool,
) -> dict[str, str]:
    ready = _require_mapping(record, location)
    _require_equal(f"{location} fields", set(ready), {"path", "sha256"})
    path = Path(str(ready.get("path", "")))
    _require(path.is_absolute(), f"{location}.path must be absolute")
    _require_equal(
        f"{location}.path",
        path.resolve(),
        expected_path.resolve(),
    )
    digest = _require_sha256(ready.get("sha256"), f"{location}.sha256")
    if verify_live:
        _require_equal(
            f"{location} live SHA-256",
            _sha256_file(expected_path),
            digest,
        )
    return {"path": str(expected_path.resolve()), "sha256": digest}


def _validate_implementation_hashes(
    payload: Mapping[str, Any],
    *,
    verify_live: bool,
) -> dict[str, dict[str, str]]:
    before = _require_mapping(
        payload.get("implementation_hashes_before"),
        "implementation_hashes_before",
    )
    after = _require_mapping(
        payload.get("implementation_hashes_after"),
        "implementation_hashes_after",
    )
    _require_equal("implementation hashes before/after", after, before)
    expected_paths = evaluator._source_paths()
    _require_equal(
        "implementation hash keys",
        set(before),
        set(expected_paths),
    )
    return {
        name: _validate_hash_record(
            before[name],
            expected_path=path,
            location=f"implementation_hashes[{name}]",
            verify_live=verify_live,
        )
        for name, path in expected_paths.items()
    }


def _validate_input_hashes(
    payload: Mapping[str, Any],
    *,
    checkpoint: str,
    verify_live: bool,
) -> dict[str, str]:
    before = _require_mapping(
        payload.get("input_hashes_before"),
        "input_hashes_before",
    )
    after = _require_mapping(
        payload.get("input_hashes_after"),
        "input_hashes_after",
    )
    _require_equal("input hashes before/after", after, before)
    _require_equal("input hash keys", tuple(before), INPUT_HASH_KEYS)
    dependency_paths = _fixed_dependency_paths(checkpoint)
    normalized: dict[str, str] = {}
    for name in INPUT_HASH_KEYS:
        digest = _require_sha256(before.get(name), f"input_hashes[{name}]")
        if verify_live:
            _require_equal(
                f"input_hashes[{name}] live SHA-256",
                _sha256_file(dependency_paths[name]),
                digest,
            )
        normalized[name] = digest
    return normalized


def _validate_mode_model(
    evaluation: Mapping[str, Any],
    *,
    mode: str,
    source_state_sha256: str,
) -> dict[str, Any]:
    model = _require_mapping(evaluation.get("model"), f"{mode}.model")
    for name, expected in {
        "formula_mode": mode,
        "formula_expression": evaluator.FORMULA_EXPRESSIONS[mode],
        "relay_parameters": (
            evaluator.v4_model_source.PRODUCTION_V4_RELAY_PARAMETERS
        ),
        "total_parameters": (
            evaluator.v4_model_source.PRODUCTION_V4_RELAY_ON_PARAMETERS
        ),
        "state_dict_sha256": source_state_sha256,
    }.items():
        _require_equal(f"{mode}.model.{name}", model.get(name), expected)
    state_key_count = _nonnegative_integer(
        model.get("state_key_count"),
        f"{mode}.model.state_key_count",
    )
    _require(state_key_count > 0, f"{mode} state key count must be positive")
    manifest = _require_mapping(
        model.get("architecture_manifest"),
        f"{mode}.model.architecture_manifest",
    )
    _require_equal(
        f"{mode} manifest mode",
        manifest.get("ner_dc_offset_support_mode"),
        mode,
    )
    _require_equal(
        f"{mode} manifest thresholds",
        _normalize_tail_thresholds(
            manifest.get("tail_z_thresholds"),
            location=f"{mode} manifest thresholds",
        ),
        dict(evaluator.v4_model_source.DEFAULT_TAIL_Z_THRESHOLDS),
    )
    _require_equal(
        f"{mode} manifest thresholds frozen",
        manifest.get("tail_z_thresholds_frozen"),
        True,
    )
    manifest_sha256 = _require_sha256(
        model.get("architecture_manifest_sha256"),
        f"{mode}.model.architecture_manifest_sha256",
    )
    _require_equal(
        f"{mode} architecture manifest SHA-256",
        manifest_sha256,
        evaluator._canonical_sha256(manifest),
    )
    return {
        "model_class": model.get("model_class"),
        "relay_parameters": model["relay_parameters"],
        "total_parameters": model["total_parameters"],
        "state_key_count": state_key_count,
        "state_dict_sha256": source_state_sha256,
        "architecture_manifest_sha256": manifest_sha256,
    }


def _extract_metric_point(
    point: Any,
    *,
    location: str,
) -> dict[str, Any]:
    ready = _require_mapping(point, location)
    matched = _nonnegative_integer(
        ready.get("matched_target_count"),
        f"{location}.matched_target_count",
    )
    target_count = _nonnegative_integer(
        ready.get("target_count"),
        f"{location}.target_count",
    )
    _require(target_count > 0, f"{location}.target_count must be positive")
    matched_tiny = _nonnegative_integer(
        ready.get("matched_tiny_target_count"),
        f"{location}.matched_tiny_target_count",
    )
    tiny_count = _nonnegative_integer(
        ready.get("tiny_target_count"),
        f"{location}.tiny_target_count",
    )
    _require(
        tiny_count > 0,
        f"{location}.tiny_target_count must be positive",
    )
    _require(matched <= target_count, f"{location} matched exceeds targets")
    _require(
        matched_tiny <= tiny_count,
        f"{location} matched tiny exceeds tiny targets",
    )
    fa = _finite_number(ready.get("fa"), f"{location}.fa")
    miou = _finite_number(ready.get("miou"), f"{location}.miou")
    threshold = _finite_number(
        ready.get("threshold"),
        f"{location}.threshold",
    )
    pd_value = _finite_number(ready.get("pd"), f"{location}.pd")
    tiny_pd_value = _finite_number(
        ready.get("tiny_pd"),
        f"{location}.tiny_pd",
    )
    _require(fa >= 0.0, f"{location}.fa must be non-negative")
    _require(0.0 <= miou <= 1.0, f"{location}.miou is out of range")
    _require(0.0 <= pd_value <= 1.0, f"{location}.pd is out of range")
    _require(
        0.0 <= tiny_pd_value <= 1.0,
        f"{location}.tiny_pd is out of range",
    )
    _require_equal(
        f"{location}.pd/count consistency",
        pd_value,
        matched / target_count,
    )
    _require_equal(
        f"{location}.tiny_pd/count consistency",
        tiny_pd_value,
        matched_tiny / tiny_count,
    )
    _require(
        0.0 <= threshold <= 1.0,
        f"{location}.threshold is out of range",
    )
    return {
        "matched_target_count": matched,
        "target_count": target_count,
        "pd": pd_value,
        "fa": fa,
        "miou": miou,
        "matched_tiny_target_count": matched_tiny,
        "tiny_target_count": tiny_count,
        "tiny_pd": tiny_pd_value,
        "threshold": threshold,
    }


def _extract_mode_metrics(
    evaluation: Mapping[str, Any],
    *,
    location: str,
) -> dict[str, Any]:
    fixed = _extract_metric_point(
        evaluation.get("fixed_threshold_0_5"),
        location=f"{location}.fixed_threshold_0_5",
    )
    _require_equal(
        f"{location} fixed threshold",
        fixed["threshold"],
        evaluator.FIXED_THRESHOLD,
    )
    raw_budgets = _require_mapping(
        evaluation.get("best_points_under_fa_budget"),
        f"{location}.best_points_under_fa_budget",
    )
    _require_equal(
        f"{location} budget keys",
        tuple(raw_budgets),
        evaluator.BUDGET_KEYS,
    )
    budgets = {
        key: _extract_metric_point(
            raw_budgets[key],
            location=f"{location}.budgets[{key}]",
        )
        for key in evaluator.BUDGET_KEYS
    }
    for key, limit in zip(evaluator.BUDGET_KEYS, evaluator.FA_BUDGETS):
        _require(
            budgets[key]["fa"] <= float(limit),
            f"{location}.budgets[{key}] exceeds its Fa budget",
        )
    return {"fixed_threshold_0_5": fixed, "budgets": budgets}


def validate_counterfactual_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint: str,
    input_path: Path,
    canonical_payload: Mapping[str, Any],
    verify_live_dependencies: bool,
) -> dict[str, Any]:
    """Validate one evaluator artifact and return its gate-relevant projection."""

    if checkpoint not in evaluator.CHECKPOINTS:
        raise ValueError(f"unsupported checkpoint: {checkpoint}")
    ready = evaluator.validate_output_payload(payload, checkpoint=checkpoint)
    for name, expected in {
        "scope": "same_v3_checkpoint_three_v4_forward_formulas",
        "affects_v3_formal_decision": False,
        "formal_training_authorized_by_this_artifact": False,
        "dataset": evaluator.DATASET,
        "variant": evaluator.VARIANT,
        "training_seed": evaluator.TRAINING_SEED,
        "split_seed": evaluator.SPLIT_SEED,
        "expected_epochs": evaluator.EXPECTED_EPOCHS,
        "validation_count": evaluator.VALIDATION_COUNT,
        "formal_default_mode": (
            evaluator.v4_model_source.DEFAULT_DC_SUPPORT_MODE
        ),
        "tail_z_thresholds_frozen": True,
        "fixed_threshold": evaluator.FIXED_THRESHOLD,
        "fa_budgets": list(evaluator.FA_BUDGETS),
    }.items():
        _require_equal(f"payload {name}", ready.get(name), expected)
    _normalize_tail_thresholds(
        ready.get("tail_z_thresholds"),
        location="payload tail_z_thresholds",
    )

    expected_input_path = Path(input_path).resolve()
    audit = _require_mapping(ready.get("audit"), "payload.audit")
    _require_equal(
        "payload audit output path",
        Path(str(audit.get("output_path", ""))).resolve(),
        expected_input_path,
    )
    source_state_sha = _require_sha256(
        ready.get("source_state_dict_sha256"),
        "source_state_dict_sha256",
    )
    source_checkpoint = _require_mapping(
        ready.get("source_checkpoint"),
        "source_checkpoint",
    )
    dependency_paths = _fixed_dependency_paths(checkpoint)
    _require_equal(
        "source checkpoint path",
        Path(str(source_checkpoint.get("path", ""))).resolve(),
        dependency_paths["source_checkpoint"].resolve(),
    )
    source_checkpoint_sha = _require_sha256(
        source_checkpoint.get("sha256"),
        "source_checkpoint.sha256",
    )
    _require_equal(
        "source checkpoint state SHA",
        source_checkpoint.get("state_dict_sha256"),
        source_state_sha,
    )
    _require_equal(
        "checkpoint payload state SHA",
        source_checkpoint.get("checkpoint_payload_state_dict_sha256"),
        source_state_sha,
    )

    canonical_record = _require_mapping(
        ready.get("canonical_v3_sweep"),
        "canonical_v3_sweep",
    )
    _require_equal(
        "canonical V3 sweep path",
        Path(str(canonical_record.get("path", ""))).resolve(),
        dependency_paths["canonical_v3_sweep"].resolve(),
    )
    canonical_sha = _require_sha256(
        canonical_record.get("sha256"),
        "canonical_v3_sweep.sha256",
    )
    implementation_hashes = _validate_implementation_hashes(
        ready,
        verify_live=verify_live_dependencies,
    )
    input_hashes = _validate_input_hashes(
        ready,
        checkpoint=checkpoint,
        verify_live=verify_live_dependencies,
    )
    _require_equal(
        "source checkpoint/input hash",
        source_checkpoint_sha,
        input_hashes["source_checkpoint"],
    )
    _require_equal(
        "canonical sweep/input hash",
        canonical_sha,
        input_hashes["canonical_v3_sweep"],
    )

    evaluations = _require_list(ready.get("evaluations"), "evaluations")
    modes: dict[str, Any] = {}
    common_state_key_count: int | None = None
    for index, mode in enumerate(ALL_FORMULA_MODES):
        evaluation = _require_mapping(
            evaluations[index],
            f"evaluations[{index}]",
        )
        for name, expected in {
            "formula_mode": mode,
            "formula_expression": evaluator.FORMULA_EXPRESSIONS[mode],
            "formula_index": index,
            "source_checkpoint_sha256_before": source_checkpoint_sha,
            "source_checkpoint_sha256_after": source_checkpoint_sha,
            "source_state_dict_sha256": source_state_sha,
            "evaluated_state_dict_sha256": source_state_sha,
            "strict_v3_state_load": True,
            "state_changed": False,
            "derived_checkpoint_written": False,
            "validation_count": evaluator.VALIDATION_COUNT,
            "fixed_threshold": evaluator.FIXED_THRESHOLD,
            "fa_budgets": list(evaluator.FA_BUDGETS),
        }.items():
            _require_equal(
                f"evaluations[{index}].{name}",
                evaluation.get(name),
                expected,
            )
        _normalize_tail_thresholds(
            evaluation.get("tail_z_thresholds"),
            location=f"evaluations[{index}].tail_z_thresholds",
        )
        model = _validate_mode_model(
            evaluation,
            mode=mode,
            source_state_sha256=source_state_sha,
        )
        if common_state_key_count is None:
            common_state_key_count = model["state_key_count"]
        _require_equal(
            f"{mode} common state key count",
            model["state_key_count"],
            common_state_key_count,
        )
        modes[mode] = {
            "model": model,
            **_extract_mode_metrics(
                evaluation,
                location=f"{checkpoint}.{mode}",
            ),
        }

    recomputed_equivalence = evaluator.require_legacy_canonical_exact(
        evaluations[0],
        canonical_payload,
    )
    _require_equal(
        "legacy canonical equivalence record",
        ready.get("legacy_canonical_equivalence"),
        recomputed_equivalence,
    )
    return {
        "checkpoint_filename": checkpoint,
        "checkpoint_role": evaluator.CHECKPOINT_ROLES[checkpoint],
        "checkpoint_epoch": _positive_integer(
            ready.get("checkpoint_epoch"),
            "checkpoint_epoch",
        ),
        "source_checkpoint_sha256": source_checkpoint_sha,
        "source_state_dict_sha256": source_state_sha,
        "counterfactual_artifact": {
            "path": str(expected_input_path),
            "sha256": _sha256_file(expected_input_path),
            "schema": evaluator.EVALUATION_SCHEMA,
        },
        "canonical_v3_sweep": {
            "path": str(dependency_paths["canonical_v3_sweep"].resolve()),
            "sha256": canonical_sha,
            "projection_sha256": recomputed_equivalence[
                "canonical_projection_sha256"
            ],
        },
        "implementation_hashes": implementation_hashes,
        "input_hashes": input_hashes,
        "legacy_canonical_equivalence": recomputed_equivalence,
        "all_modes_same_state": True,
        "modes": modes,
    }


def _fixed_pareto_dominates(
    candidate: Mapping[str, Any],
    other: Mapping[str, Any],
) -> bool:
    nonworse = (
        candidate["matched_target_count"] >= other["matched_target_count"]
        and candidate["fa"] <= other["fa"]
        and candidate["miou"] >= other["miou"]
    )
    strict = (
        candidate["matched_target_count"] > other["matched_target_count"]
        or candidate["fa"] < other["fa"]
        or candidate["miou"] > other["miou"]
    )
    return bool(nonworse and strict)


def assess_local_formula(
    role_rows: Mapping[str, Mapping[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    """Apply the preregistered candidate-vs-legacy qualification gate."""

    if mode not in LOCAL_FORMULA_MODES:
        raise ValueError(f"not a local formula mode: {mode}")
    _require_equal(
        "qualification role set",
        set(role_rows),
        set(evaluator.CHECKPOINT_ROLES.values()),
    )
    roles: dict[str, Any] = {}
    strict_total = 0
    fixed_legacy_dominance_count = 0
    for role in evaluator.CHECKPOINT_ROLES.values():
        row = role_rows[role]
        legacy = row["modes"]["legacy_global"]
        candidate = row["modes"][mode]
        budget_comparisons: dict[str, Any] = {}
        noninferior_count = 0
        strict_count = 0
        for key in evaluator.BUDGET_KEYS:
            legacy_count = legacy["budgets"][key]["matched_target_count"]
            candidate_count = candidate["budgets"][key][
                "matched_target_count"
            ]
            noninferior = candidate_count >= legacy_count
            strict = candidate_count > legacy_count
            noninferior_count += int(noninferior)
            strict_count += int(strict)
            budget_comparisons[key] = {
                "legacy_matched_target_count": legacy_count,
                "candidate_matched_target_count": candidate_count,
                "delta": candidate_count - legacy_count,
                "noninferior": noninferior,
                "strictly_better": strict,
            }
        strict_total += strict_count
        legacy_dominates = _fixed_pareto_dominates(
            legacy["fixed_threshold_0_5"],
            candidate["fixed_threshold_0_5"],
        )
        fixed_legacy_dominance_count += int(legacy_dominates)
        roles[role] = {
            "noninferior_budget_count": noninferior_count,
            "required_noninferior_budget_count": (
                QUALIFYING_BUDGETS_PER_ROLE
            ),
            "budget_noninferiority_passed": (
                noninferior_count >= QUALIFYING_BUDGETS_PER_ROLE
            ),
            "strict_budget_improvement_count": strict_count,
            "legacy_fixed_pareto_dominates_candidate": legacy_dominates,
            "fixed_pareto_gate_passed": not legacy_dominates,
            "budget_comparisons": budget_comparisons,
        }
    qualifies = (
        all(
            item["budget_noninferiority_passed"]
            and item["fixed_pareto_gate_passed"]
            for item in roles.values()
        )
        and strict_total >= REQUIRED_STRICT_BUDGET_IMPROVEMENTS
    )
    return {
        "formula_mode": mode,
        "qualifies": qualifies,
        "per_role": roles,
        "strict_budget_improvement_count_across_roles": strict_total,
        "required_strict_budget_improvement_count_across_roles": (
            REQUIRED_STRICT_BUDGET_IMPROVEMENTS
        ),
        "strict_budget_improvement_gate_passed": (
            strict_total >= REQUIRED_STRICT_BUDGET_IMPROVEMENTS
        ),
        "legacy_fixed_pareto_dominance_count": (
            fixed_legacy_dominance_count
        ),
    }


def formula_jointly_dominates(
    role_rows: Mapping[str, Mapping[str, Any]],
    *,
    candidate_mode: str,
    other_mode: str,
) -> bool:
    """Return strict joint dominance across both roles and all gate metrics."""

    if candidate_mode not in LOCAL_FORMULA_MODES:
        raise ValueError(f"not a local formula mode: {candidate_mode}")
    if other_mode not in LOCAL_FORMULA_MODES:
        raise ValueError(f"not a local formula mode: {other_mode}")
    strict = False
    for role in evaluator.CHECKPOINT_ROLES.values():
        candidate = role_rows[role]["modes"][candidate_mode]
        other = role_rows[role]["modes"][other_mode]
        candidate_fixed = candidate["fixed_threshold_0_5"]
        other_fixed = other["fixed_threshold_0_5"]
        fixed_nonworse = (
            candidate_fixed["matched_target_count"]
            >= other_fixed["matched_target_count"]
            and candidate_fixed["fa"] <= other_fixed["fa"]
            and candidate_fixed["miou"] >= other_fixed["miou"]
        )
        if not fixed_nonworse:
            return False
        strict = strict or (
            candidate_fixed["matched_target_count"]
            > other_fixed["matched_target_count"]
            or candidate_fixed["fa"] < other_fixed["fa"]
            or candidate_fixed["miou"] > other_fixed["miou"]
        )
        for key in evaluator.BUDGET_KEYS:
            candidate_count = candidate["budgets"][key][
                "matched_target_count"
            ]
            other_count = other["budgets"][key]["matched_target_count"]
            if candidate_count < other_count:
                return False
            strict = strict or candidate_count > other_count
    return bool(strict)


def select_formula(
    role_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    assessments = {
        mode: assess_local_formula(role_rows, mode=mode)
        for mode in LOCAL_FORMULA_MODES
    }
    qualifying = [
        mode for mode in LOCAL_FORMULA_MODES
        if assessments[mode]["qualifies"]
    ]
    direct_dominates = formula_jointly_dominates(
        role_rows,
        candidate_mode="direct_tail",
        other_mode="complement_tail",
    )
    complement_dominates = formula_jointly_dominates(
        role_rows,
        candidate_mode="complement_tail",
        other_mode="direct_tail",
    )
    selected: str | None = None
    if len(qualifying) == 1:
        selected = qualifying[0]
        decision = (
            "DIRECT_TAIL_SELECTED"
            if selected == "direct_tail"
            else "COMPLEMENT_TAIL_SELECTED"
        )
        basis = "only_one_local_formula_qualified"
    elif len(qualifying) == 2:
        if direct_dominates and not complement_dominates:
            selected = "direct_tail"
            decision = "DIRECT_TAIL_SELECTED"
            basis = "qualified_direct_jointly_dominates_qualified_complement"
        elif complement_dominates and not direct_dominates:
            selected = "complement_tail"
            decision = "COMPLEMENT_TAIL_SELECTED"
            basis = "qualified_complement_jointly_dominates_qualified_direct"
        else:
            decision = "FORMULA_INCONCLUSIVE"
            basis = "both_local_formulas_qualified_with_mixed_tradeoff"
    else:
        decision = "LOCAL_SCOPE_REJECTED"
        basis = "no_local_formula_qualified"
    return {
        "decision": decision,
        "decision_basis": basis,
        "selected_formula_mode": selected,
        "formal_v4_formula_selected": selected is not None,
        "qualifying_local_formula_modes": qualifying,
        "candidate_assessments": assessments,
        "joint_dominance": {
            "direct_tail_over_complement_tail": direct_dominates,
            "complement_tail_over_direct_tail": complement_dominates,
        },
    }


def _validate_pair(
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    _require_equal(
        "counterfactual checkpoint set",
        set(records),
        set(evaluator.CHECKPOINTS),
    )
    implementation_reference: Mapping[str, Any] | None = None
    shared_input_reference: dict[str, str] | None = None
    for checkpoint in evaluator.CHECKPOINTS:
        record = records[checkpoint]
        _require_equal(
            f"{checkpoint} filename",
            record.get("checkpoint_filename"),
            checkpoint,
        )
        _require_equal(
            f"{checkpoint} role",
            record.get("checkpoint_role"),
            evaluator.CHECKPOINT_ROLES[checkpoint],
        )
        _require_equal(
            f"{checkpoint} mode set",
            tuple(record["modes"]),
            ALL_FORMULA_MODES,
        )
        if implementation_reference is None:
            implementation_reference = record["implementation_hashes"]
        _require_equal(
            f"{checkpoint} implementation hashes",
            record["implementation_hashes"],
            implementation_reference,
        )
        shared_inputs = {
            name: record["input_hashes"][name]
            for name in SHARED_INPUT_HASH_KEYS
        }
        if shared_input_reference is None:
            shared_input_reference = shared_inputs
        _require_equal(
            f"{checkpoint} shared input hashes",
            shared_inputs,
            shared_input_reference,
        )


def build_aggregate_report(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the immutable report in memory without publishing it."""

    _validate_pair(records)
    role_rows = {
        records[checkpoint]["checkpoint_role"]: {
            "checkpoint_filename": checkpoint,
            "checkpoint_epoch": records[checkpoint]["checkpoint_epoch"],
            "source_checkpoint_sha256": records[checkpoint][
                "source_checkpoint_sha256"
            ],
            "source_state_dict_sha256": records[checkpoint][
                "source_state_dict_sha256"
            ],
            "modes": copy.deepcopy(records[checkpoint]["modes"]),
        }
        for checkpoint in evaluator.CHECKPOINTS
    }
    selection = select_formula(role_rows)
    source_hashes = {
        "postprocessor": {
            "path": str(POSTPROCESS_PATH),
            "sha256": _sha256_file(POSTPROCESS_PATH),
        },
        **copy.deepcopy(
            records[evaluator.CHECKPOINTS[0]]["implementation_hashes"]
        ),
    }
    report = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete",
        "artifact_kind": "v4_tail_formula_counterfactual_aggregate",
        "scope": "local_formula_freeze_from_same_v3_state_counterfactual",
        "diagnostic_only": True,
        "zero_training": True,
        "official_test_accessed": False,
        "affects_v3_formal_decision": False,
        "v3_formal_decision_modified": False,
        "formal_training_authorized_by_this_artifact": False,
        "dataset": evaluator.DATASET,
        "variant": evaluator.VARIANT,
        "training_seed": evaluator.TRAINING_SEED,
        "split_seed": evaluator.SPLIT_SEED,
        "checkpoint_count": len(evaluator.CHECKPOINTS),
        "checkpoint_roles": dict(evaluator.CHECKPOINT_ROLES),
        "formula_modes": list(ALL_FORMULA_MODES),
        "local_formula_modes": list(LOCAL_FORMULA_MODES),
        "fa_budgets": list(evaluator.FA_BUDGETS),
        "budget_keys": list(evaluator.BUDGET_KEYS),
        "fixed_threshold": evaluator.FIXED_THRESHOLD,
        "selection_rule": {
            "per_role_required_noninferior_budgets": (
                QUALIFYING_BUDGETS_PER_ROLE
            ),
            "across_roles_required_strict_budget_improvements": (
                REQUIRED_STRICT_BUDGET_IMPROVEMENTS
            ),
            "fixed_threshold_legacy_pareto_dominance_forbidden": True,
            "fixed_pareto_metrics": list(FIXED_METRIC_NAMES),
            "joint_dominance_requires_both_roles_all_fixed_metrics_and_budgets": (
                True
            ),
            "tiny_pd_is_report_only": True,
        },
        **selection,
        "role_results": role_rows,
        "tiny_pd_disclosure": {
            role: {
                mode: {
                    "fixed_matched_tiny_target_count": row["modes"][mode][
                        "fixed_threshold_0_5"
                    ]["matched_tiny_target_count"],
                    "fixed_tiny_target_count": row["modes"][mode][
                        "fixed_threshold_0_5"
                    ]["tiny_target_count"],
                    "budget_matched_tiny_target_counts": {
                        key: row["modes"][mode]["budgets"][key][
                            "matched_tiny_target_count"
                        ]
                        for key in evaluator.BUDGET_KEYS
                    },
                }
                for mode in ALL_FORMULA_MODES
            }
            for role, row in role_rows.items()
        },
        "input_artifacts": {
            checkpoint: copy.deepcopy(
                records[checkpoint]["counterfactual_artifact"]
            )
            for checkpoint in evaluator.CHECKPOINTS
        },
        "canonical_v3_sweeps": {
            checkpoint: copy.deepcopy(
                records[checkpoint]["canonical_v3_sweep"]
            )
            for checkpoint in evaluator.CHECKPOINTS
        },
        "source_checkpoints": {
            checkpoint: {
                "sha256": records[checkpoint][
                    "source_checkpoint_sha256"
                ],
                "state_dict_sha256": records[checkpoint][
                    "source_state_dict_sha256"
                ],
            }
            for checkpoint in evaluator.CHECKPOINTS
        },
        "source_hashes": source_hashes,
        "audit": {
            "fixed_two_checkpoint_outputs_consumed": True,
            "three_modes_per_checkpoint_validated": True,
            "legacy_canonical_exact_both_checkpoints": True,
            "same_pristine_state_within_each_checkpoint": True,
            "input_and_source_hashes_recomputed": True,
            "no_checkpoint_written": True,
            "no_inference_performed": True,
            "no_threshold_search": True,
            "v3_decision_unchanged": True,
            "formal_v4_formula_selected_only_if_unique": (
                selection["formal_v4_formula_selected"]
                == (selection["selected_formula_mode"] is not None)
            ),
        },
    }
    metric_core.assert_finite_numbers(report, "counterfactual aggregate")
    return metric_core.json_ready(report)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V4 Tail Formula Counterfactual Aggregate",
        "",
        f"- Decision: `{report['decision']}`",
        (
            "- Selected local formula: "
            f"`{report['selected_formula_mode']}`"
        ),
        (
            "- `formal_v4_formula_selected`: "
            f"`{str(report['formal_v4_formula_selected']).lower()}`"
        ),
        "- Scope: zero-training formula freeze only; V3 decision is unchanged.",
        "- Formal training authorization from this artifact: `false`.",
        "",
        "## Fixed threshold 0.5",
        "",
        "| role | mode | matched | Fa | mIoU | tiny |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for role, row in report["role_results"].items():
        for mode in ALL_FORMULA_MODES:
            fixed = row["modes"][mode]["fixed_threshold_0_5"]
            lines.append(
                "| "
                f"{role} | {mode} | "
                f"{fixed['matched_target_count']}/{fixed['target_count']} | "
                f"{fixed['fa']:.12g} | {fixed['miou']:.12g} | "
                f"{fixed['matched_tiny_target_count']}/"
                f"{fixed['tiny_target_count']} |"
            )
    lines.extend(
        [
            "",
            "## Preregistered qualification",
            "",
            (
                "| formula | qualifies | non-inferior budgets by role | "
                "strict gains across roles | legacy fixed dominance count |"
            ),
            "|---|---:|---|---:|---:|",
        ]
    )
    assessments = report["candidate_assessments"]
    for mode in LOCAL_FORMULA_MODES:
        assessment = assessments[mode]
        role_counts = ", ".join(
            f"{role}={item['noninferior_budget_count']}/5"
            for role, item in assessment["per_role"].items()
        )
        lines.append(
            "| "
            f"{mode} | {assessment['qualifies']} | {role_counts} | "
            f"{assessment['strict_budget_improvement_count_across_roles']} | "
            f"{assessment['legacy_fixed_pareto_dominance_count']} |"
        )
    lines.extend(
        [
            "",
            "## Five Fa budgets",
            "",
            "| role | budget | legacy | direct | complement |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for role, row in report["role_results"].items():
        for key in evaluator.BUDGET_KEYS:
            counts = {
                mode: row["modes"][mode]["budgets"][key][
                    "matched_target_count"
                ]
                for mode in ALL_FORMULA_MODES
            }
            lines.append(
                "| "
                f"{role} | {key} | {counts['legacy_global']} | "
                f"{counts['direct_tail']} | {counts['complement_tail']} |"
            )
    lines.extend(
        [
            "",
            "Tiny-Pd is disclosed in the JSON report and is not used by the gate.",
            "",
        ]
    )
    return "\n".join(lines)


def load_and_validate_fixed_inputs() -> dict[str, dict[str, Any]]:
    """Read and fully revalidate the two fixed evaluator outputs."""

    records: dict[str, dict[str, Any]] = {}
    for checkpoint in evaluator.CHECKPOINTS:
        input_path = evaluator.output_path(checkpoint).resolve()
        canonical_path = evaluator.canonical_sweep_path(checkpoint)
        payload = _load_json(input_path)
        canonical = _load_json(canonical_path)
        records[checkpoint] = validate_counterfactual_payload(
            payload,
            checkpoint=checkpoint,
            input_path=input_path,
            canonical_payload=canonical,
            verify_live_dependencies=True,
        )
    _validate_pair(records)
    return records


def _atomic_publish_bytes_new(path: Path, payload: bytes) -> None:
    """Atomically link a complete temporary file into a previously absent path."""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def publish_bundle(
    report: Mapping[str, Any],
    *,
    json_path: Path,
    markdown_path: Path,
    marker_path: Path,
) -> dict[str, Any]:
    destinations = (
        Path(json_path),
        Path(markdown_path),
        Path(marker_path),
    )
    _require_equal(
        "publication destination uniqueness",
        len(set(path.resolve() for path in destinations)),
        3,
    )
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite output: {destination}"
            )

    json_bytes = _pretty_json_bytes(report)
    markdown_bytes = render_markdown(report).encode("utf-8")
    marker = {
        "schema": MARKER_SCHEMA,
        "status": "complete",
        "decision": report["decision"],
        "selected_formula_mode": report["selected_formula_mode"],
        "formal_v4_formula_selected": report[
            "formal_v4_formula_selected"
        ],
        "affects_v3_formal_decision": False,
        "v3_formal_decision_modified": False,
        "formal_training_authorized_by_this_artifact": False,
        "aggregate_json": {
            "path": str(Path(json_path).resolve()),
            "sha256": _sha256_bytes(json_bytes),
        },
        "aggregate_markdown": {
            "path": str(Path(markdown_path).resolve()),
            "sha256": _sha256_bytes(markdown_bytes),
        },
        "input_artifacts": copy.deepcopy(report["input_artifacts"]),
        "source_hashes": copy.deepcopy(report["source_hashes"]),
        "marker_written_last": True,
        "output_overwrite_forbidden": True,
    }
    marker_bytes = _pretty_json_bytes(marker)
    _atomic_publish_bytes_new(Path(json_path), json_bytes)
    _atomic_publish_bytes_new(Path(markdown_path), markdown_bytes)
    _atomic_publish_bytes_new(Path(marker_path), marker_bytes)
    return metric_core.json_ready(marker)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and aggregate the fixed two-checkpoint V4 tail-formula "
            "counterfactual without running inference"
        )
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--verify-inputs",
        action="store_true",
        help="validate both fixed inputs and print the prospective decision",
    )
    action.add_argument(
        "--aggregate",
        action="store_true",
        help="validate fixed inputs and publish the non-overwritable bundle",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_and_validate_fixed_inputs()
    report = build_aggregate_report(records)
    if args.verify_inputs:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "decision": report["decision"],
                    "selected_formula_mode": report[
                        "selected_formula_mode"
                    ],
                    "formal_v4_formula_selected": report[
                        "formal_v4_formula_selected"
                    ],
                    "writes_performed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    marker = publish_bundle(
        report,
        json_path=AGGREGATE_JSON_PATH,
        markdown_path=AGGREGATE_MD_PATH,
        marker_path=COMPLETION_MARKER_PATH,
    )
    print(json.dumps(marker, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
