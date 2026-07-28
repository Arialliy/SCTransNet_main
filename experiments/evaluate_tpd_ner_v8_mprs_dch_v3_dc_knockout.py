#!/usr/bin/env python3
"""Independent, evaluation-only V3 DC-offset knockout evaluator.

The evaluator consumes the two immutable formal V3 compatibility
checkpoints, applies one of four fixed counterfactual state transforms in
memory, and evaluates all four transforms for one checkpoint in a single
publication.  It never writes a derived checkpoint and has no authority over
the formal V3 gate.
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

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_pd_fa_sweep as metric_core  # noqa: E402
from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_v3_pd_fa as formal_evaluator,
)
from experiments import (  # noqa: E402
    freeze_tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock as source_freezer,
)
from experiments import tpd_exact_runner as exact_runner  # noqa: E402
from experiments import (  # noqa: E402
    tpd_ner_v8_mprs_dch_v3_dc_knockout_spec as spec,
)
from experiments import (  # noqa: E402
    train_tpd_ner_v8_mprs_dch_v3_exact as exact,
)
from experiments.evaluate_tpd_clean_v8_mprs_dch_pd_fa import (  # noqa: E402
    CUBLAS_WORKSPACE_CONFIG,
    DETERMINISM_SETTINGS,
    LAST_FLOAT32_BELOW_ONE,
    UPPER_BOUNDARY_THRESHOLD,
    adaptive_thresholds_closed_interval,
    configure_v8_inference,
)


EVALUATION_SCHEMA = spec.EVALUATION_SCHEMA
FINAL_METRIC_COVERAGE_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_dc_knockout_"
    "final_metric_coverage_v1"
)
STATE_TRANSFORM_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v3_dc_knockout_"
    "state_transform_v1"
)
EVALUATOR_PATH = Path(__file__).resolve()
FORMAL_EVALUATOR_PATH = Path(formal_evaluator.__file__).resolve()
METRIC_CORE_PATH = Path(metric_core.__file__).resolve()
CLOSED_INTERVAL_CORE_PATH = (
    REPO_ROOT / "experiments/evaluate_tpd_clean_v6_pd_fa.py"
)
DETERMINISM_CORE_PATH = (
    REPO_ROOT / "experiments/evaluate_tpd_clean_v8_mprs_dch_pd_fa.py"
)
FORMAL_MATCH_RADIUS = 3.0
FORMAL_TINY_AREA = 9
THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.99
THRESHOLD_STEP = 0.01
TAIL_LOGIT_STEP = 0.1
CUDA_DEVICE_ORDER = spec.CUDA_DEVICE_ORDER
CUBLAS_WORKSPACE_CONFIG_ENV = spec.CUBLAS_WORKSPACE_CONFIG_ENV
CUBLAS_WORKSPACE_CONFIG_VALUE = spec.CUBLAS_WORKSPACE_CONFIG
PYTHONHASHSEED_ENV = spec.PYTHONHASHSEED_ENV
PYTHONHASHSEED_VALUE = spec.PYTHONHASHSEED
PHYSICAL_GPU_INDEX_ENV = spec.PHYSICAL_GPU_INDEX_ENV
PHYSICAL_GPU_UUID_ENV = spec.PHYSICAL_GPU_UUID_ENV
CHECKPOINT_GPU_LANES = spec.CHECKPOINT_GPU_LANES


def _require(condition: bool, message: str) -> None:
    """Optimization-safe invariant check."""

    if not condition:
        raise ValueError(message)


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, observed={observed!r}"
        )


def _sha256_file(path: Path) -> str:
    value = Path(path)
    if not value.is_file() or value.is_symlink():
        raise ValueError(f"expected a regular non-symlink file: {value}")
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = Path(path)
    if not value.is_file() or value.is_symlink():
        raise FileNotFoundError(value)
    payload = json.loads(value.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {value}")
    return payload


def _canonical_json_equal(
    location: str,
    observed: Any,
    expected: Any,
) -> None:
    observed_bytes = spec.canonical_json_bytes(metric_core.json_ready(observed))
    expected_bytes = spec.canonical_json_bytes(metric_core.json_ready(expected))
    if observed_bytes != expected_bytes:
        raise ValueError(f"{location} differs")


def state_content_sha256(state: Mapping[str, Any]) -> str:
    """Return the exact-runner tensor-content identity for a state mapping."""

    if not isinstance(state, Mapping) or not state:
        raise ValueError("state must be a non-empty mapping")
    try:
        return exact_runner._state_content_sha256(
            state,
            "V3 DC-knockout state",
        )
    except exact_runner.ExactRunnerError as exc:
        raise ValueError(f"invalid state mapping: {exc}") from exc


def _validate_dc_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("state must be a non-empty mapping")
    observed = {
        str(name)
        for name in state
        if str(name).startswith("tpd_ner.dc_offsets.")
    }
    _require_equal(
        "DC-offset state-key set",
        observed,
        set(spec.DC_OFFSET_KEYS),
    )
    for key in spec.DC_OFFSET_KEYS:
        tensor = state.get(key)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{key} must be a tensor")
        _require_equal(f"{key} shape", tuple(tensor.shape), (1,))
        _require_equal(f"{key} dtype", tensor.dtype, torch.float32)
        if not bool(torch.isfinite(tensor.detach()).all().item()):
            raise ValueError(f"{key} must be finite")


def non_dc_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash every state entry except the three declared DC offsets."""

    _validate_dc_state(state)
    non_dc = {
        str(name): value
        for name, value in state.items()
        if str(name) not in spec.DC_OFFSET_KEYS
    }
    _require(bool(non_dc), "state has no non-DC entries")
    return state_content_sha256(non_dc)


def dc_offset_records(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return JSON-native scalar and tensor identities for all DC offsets."""

    _validate_dc_state(state)
    records: dict[str, dict[str, Any]] = {}
    for key in spec.DC_OFFSET_KEYS:
        tensor = state[key].detach()
        records[key] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "value": float(tensor.reshape(-1)[0].cpu().item()),
            "tensor_sha256": state_content_sha256({key: tensor}),
        }
    return records


def _clone_state(state: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("model state must map string keys to tensors")
        cloned[name] = value.detach().clone()
    return cloned


def _changed_state_keys(
    source: Mapping[str, torch.Tensor],
    transformed: Mapping[str, torch.Tensor],
) -> list[str]:
    _require_equal(
        "transformed state-key set",
        set(transformed),
        set(source),
    )
    return sorted(
        name
        for name in source
        if (
            source[name].dtype != transformed[name].dtype
            or tuple(source[name].shape) != tuple(transformed[name].shape)
            or not torch.equal(source[name], transformed[name])
        )
    )


def transform_state_dict(
    state: Mapping[str, Any],
    mode: str,
) -> dict[str, torch.Tensor]:
    """Clone a pristine source state and zero only the mode's DC offsets."""

    if mode not in spec.KNOCKOUT_MODES:
        raise ValueError(f"unsupported knockout mode: {mode}")
    _validate_dc_state(state)
    source_sha = state_content_sha256(state)
    transformed = _clone_state(state)
    for key in spec.KNOCKOUT_ZERO_KEYS[mode]:
        transformed[key] = torch.zeros_like(transformed[key])
    _validate_dc_state(transformed)
    _require_equal(
        "source state after transform",
        state_content_sha256(state),
        source_sha,
    )
    changed = set(_changed_state_keys(state, transformed))
    requested = set(spec.KNOCKOUT_ZERO_KEYS[mode])
    if not changed <= requested:
        raise ValueError(
            f"knockout changed non-requested state keys: "
            f"{sorted(changed - requested)}"
        )
    for key in requested:
        if not bool(torch.count_nonzero(transformed[key]).item() == 0):
            raise ValueError(f"knockout failed to zero {key}")
    for key in set(spec.DC_OFFSET_KEYS) - requested:
        if not torch.equal(transformed[key], state[key]):
            raise ValueError(f"knockout changed non-requested DC offset {key}")
    _require_equal(
        "non-DC state after transform",
        non_dc_state_sha256(transformed),
        non_dc_state_sha256(state),
    )
    return transformed


def _source_checkpoint(
    checkpoint_name: str,
    artifact_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    checkpoint_path = (
        Path(str(artifact_audit["run_directory"])) / checkpoint_name
    ).resolve()
    _require_equal(
        "canonical checkpoint path",
        checkpoint_path.parent,
        spec.FORMAL_RUN_DIR.resolve(),
    )
    _require_equal(
        "checkpoint file SHA before load",
        _sha256_file(checkpoint_path),
        artifact_audit["checkpoint_sha256"],
    )
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("source checkpoint must be an object")
    checkpoint = exact.require_evaluator_checkpoint_payload(
        payload,
        expected_variant=spec.VARIANT,
    )
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("source checkpoint has no state_dict")
    _validate_dc_state(state)
    state_sha = state_content_sha256(state)
    _require_equal(
        "source state_dict SHA",
        state_sha,
        checkpoint.get("state_dict_sha256"),
    )
    _require_equal(
        "preflight/source checkpoint identity",
        checkpoint.get("checkpoint_identity"),
        artifact_audit.get("checkpoint_identity"),
    )
    _require_equal(
        "preflight/source checkpoint epoch",
        checkpoint.get("epoch"),
        artifact_audit.get("checkpoint_epoch"),
    )
    _require_equal(
        "preflight/source checkpoint role",
        checkpoint.get("checkpoint_role"),
        spec.CHECKPOINT_ROLES[checkpoint_name],
    )
    _require_equal(
        "checkpoint file SHA after load",
        _sha256_file(checkpoint_path),
        artifact_audit["checkpoint_sha256"],
    )
    return dict(checkpoint), checkpoint_path


def _evaluation_data(
    artifact_audit: Mapping[str, Any],
) -> tuple[DataLoader, dict[str, Any]]:
    run_dir = Path(str(artifact_audit["run_directory"])).resolve()
    protocol = _load_json(run_dir / "protocol.json")
    split = _load_json(run_dir / "split.json")
    arguments = protocol.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("protocol.arguments must be an object")
    _require_equal(
        "protocol match radius",
        arguments.get("match_radius"),
        FORMAL_MATCH_RADIUS,
    )
    _require_equal(
        "protocol tiny area",
        arguments.get("tiny_area"),
        FORMAL_TINY_AREA,
    )
    validation_ids = split.get("used_val_ids")
    if not isinstance(validation_ids, list) or not all(
        isinstance(identifier, str) for identifier in validation_ids
    ):
        raise ValueError("split.used_val_ids must be a string list")
    _require_equal(
        "validation identifier count",
        len(validation_ids),
        spec.VALIDATION_COUNT,
    )
    _require_equal(
        "preflight validation count",
        artifact_audit.get("validation_count"),
        spec.VALIDATION_COUNT,
    )
    _require_equal(
        "validation split SHA",
        metric_core.identifier_sha256(validation_ids),
        artifact_audit.get("validation_split_sha256"),
    )
    normalization = protocol.get("normalization")
    if not isinstance(normalization, Mapping) or not normalization:
        raise ValueError("protocol.normalization must be an object")
    normalization_ready = {
        str(name): float(value) for name, value in normalization.items()
    }
    if not all(
        math.isfinite(value) for value in normalization_ready.values()
    ):
        raise ValueError("protocol normalization must be finite")
    dataset_dir = Path(str(arguments.get("dataset_dir")))
    if not dataset_dir.is_absolute():
        dataset_dir = (REPO_ROOT / dataset_dir).resolve()
    validation_set = metric_core.ValidationSubset(
        dataset_dir / spec.DATASET,
        validation_ids,
        normalization_ready,
    )
    _require_equal(
        "validation dataset count",
        len(validation_set),
        spec.VALIDATION_COUNT,
    )
    loader = DataLoader(
        validation_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    return loader, {
        "dataset_directory": str(dataset_dir),
        "validation_ids": list(validation_ids),
        "normalization": normalization_ready,
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
    }


def _threshold_configuration() -> dict[str, Any]:
    return {
        "threshold_min": THRESHOLD_MIN,
        "threshold_max": THRESHOLD_MAX,
        "threshold_step": THRESHOLD_STEP,
        "extra_thresholds": list(spec.EXTRA_THRESHOLDS),
        "tail_logit_step": TAIL_LOGIT_STEP,
        "fa_budgets": list(spec.FA_BUDGETS),
        "prediction_comparison": "prediction > threshold",
        "score_dtype": "float32",
        "include_zero": False,
        "include_last_float32_below_one": True,
        "include_one": True,
    }


def _metric_coverage(
    fixed: Mapping[str, Any],
    normalized_budgets: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_fixed = formal_evaluator._normalize_fixed(fixed)
    coverage = {
        "schema": FINAL_METRIC_COVERAGE_SCHEMA,
        "fixed_threshold": 0.5,
        "required_fixed_threshold_fields": list(
            spec.FIXED_THRESHOLD_FIELDS
        ),
        "fixed_threshold_0_5": {
            name: normalized_fixed[name]
            for name in spec.FIXED_THRESHOLD_FIELDS
        },
        "required_fa_budget_keys": list(spec.BUDGET_KEYS),
        "pd_at_fa_budget": copy.deepcopy(dict(normalized_budgets)),
        "all_required_metrics_present": True,
    }
    _require_equal(
        "fixed metric coverage field set",
        set(coverage["fixed_threshold_0_5"]),
        set(spec.FIXED_THRESHOLD_FIELDS),
    )
    _require_equal(
        "budget metric coverage key set",
        set(coverage["pd_at_fa_budget"]),
        set(spec.BUDGET_KEYS),
    )
    return coverage


def _validate_sweep_payload(payload: Mapping[str, Any]) -> None:
    _require_equal(
        "evaluation validation_count",
        payload.get("validation_count"),
        spec.VALIDATION_COUNT,
    )
    points = payload.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("evaluation points must be a non-empty list")
    first = points[0]
    if not isinstance(first, Mapping):
        raise ValueError("evaluation point must be an object")
    invariant_counts = {
        "target_count": first.get("target_count"),
        "tiny_target_count": first.get("tiny_target_count"),
        "valid_pixel_count": first.get("valid_pixel_count"),
    }
    _require_equal(
        "evaluation target_count",
        invariant_counts["target_count"],
        spec.TARGET_COUNT,
    )
    _require_equal(
        "evaluation tiny_target_count",
        invariant_counts["tiny_target_count"],
        spec.TINY_TARGET_COUNT,
    )
    if (
        type(invariant_counts["valid_pixel_count"]) is not int
        or invariant_counts["valid_pixel_count"] < 1
    ):
        raise ValueError("evaluation valid_pixel_count must be positive")
    validated = [
        formal_evaluator._validate_raw_point(
            point,
            index=index,
            validation_count=spec.VALIDATION_COUNT,
            invariant_counts=invariant_counts,
        )
        for index, point in enumerate(points)
    ]
    thresholds = [float(point["threshold"]) for point in validated]
    if thresholds != sorted(thresholds) or len(thresholds) != len(
        set(thresholds)
    ):
        raise ValueError("evaluation thresholds must be unique and sorted")
    if 0.0 in thresholds:
        raise ValueError("locked formal threshold set does not include zero")
    provenance = payload.get("threshold_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("threshold_provenance must be an object")
    expected = formal_evaluator._expected_raw_thresholds(provenance)
    _require_equal(
        "threshold point count",
        len(validated),
        len(expected),
    )
    if any(
        not math.isclose(observed, required, rel_tol=0.0, abs_tol=2e-15)
        for observed, required in zip(thresholds, expected)
    ):
        raise ValueError("evaluation threshold set differs from formal core")
    _require_equal(
        "threshold provenance total count",
        provenance.get("total_unique_threshold_count"),
        len(validated),
    )
    _require_equal(
        "threshold provenance score count",
        provenance.get("score_count"),
        invariant_counts["valid_pixel_count"],
    )
    fixed_points = [
        point for point in validated if float(point["threshold"]) == 0.5
    ]
    _require_equal(
        "threshold=0.5 raw point count",
        len(fixed_points),
        1,
    )
    fixed = payload.get("fixed_threshold_0_5")
    if not isinstance(fixed, Mapping):
        raise ValueError("fixed_threshold_0_5 must be an object")
    _canonical_json_equal(
        "fixed threshold/raw point",
        fixed,
        fixed_points[0],
    )
    budgets = payload.get("best_points_under_fa_budget")
    if not isinstance(budgets, Mapping):
        raise ValueError("best_points_under_fa_budget must be an object")
    _require_equal("FA budget keys", set(budgets), set(spec.BUDGET_KEYS))
    for budget, key in zip(spec.FA_BUDGETS, spec.BUDGET_KEYS):
        expected_point = metric_core.best_point_under_fa(validated, budget)
        if expected_point is None:
            raise ValueError(f"no point satisfies FA budget {key}")
        _canonical_json_equal(
            f"FA budget {key}/raw optimum",
            budgets[key],
            expected_point,
        )
    normalized_budgets = formal_evaluator._normalize_budgets(payload)
    _canonical_json_equal(
        "final metric coverage",
        payload.get("final_metric_coverage"),
        _metric_coverage(fixed, normalized_budgets),
    )
    formal_evaluator._validate_closed_interval(payload)


def sweep_predictions(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    losses: Sequence[float],
    *,
    validation_count: int,
) -> dict[str, Any]:
    """Evaluate one counterfactual prediction set with the formal metric core."""

    _require_equal(
        "prediction count",
        len(probabilities),
        validation_count,
    )
    _require_equal("target count", len(targets), validation_count)
    _require_equal("loss count", len(losses), validation_count)
    _require_equal(
        "canonical validation count",
        validation_count,
        spec.VALIDATION_COUNT,
    )
    for index, (probability, target, loss) in enumerate(
        zip(probabilities, targets, losses)
    ):
        if (
            not isinstance(probability, np.ndarray)
            or probability.dtype != np.float32
            or not np.isfinite(probability).all()
        ):
            raise ValueError(
                f"prediction[{index}] must be a finite FP32 array"
            )
        if not isinstance(target, np.ndarray) or not np.isfinite(target).all():
            raise ValueError(f"target[{index}] must be a finite array")
        if not math.isfinite(float(loss)):
            raise ValueError(f"loss[{index}] must be finite")
    base_thresholds = metric_core.threshold_grid(
        THRESHOLD_MIN,
        THRESHOLD_MAX,
        THRESHOLD_STEP,
        spec.EXTRA_THRESHOLDS,
    )
    thresholds, provenance = adaptive_thresholds_closed_interval(
        probabilities,
        base_thresholds,
        TAIL_LOGIT_STEP,
    )
    points: list[dict[str, Any]] = []
    for threshold in thresholds:
        accumulator = metric_core.ValidationMetrics(
            float(threshold),
            FORMAL_MATCH_RADIUS,
            FORMAL_TINY_AREA,
        )
        for probability, target, loss in zip(
            probabilities,
            targets,
            losses,
        ):
            accumulator.update(probability, target, float(loss))
        point = metric_core.json_ready(accumulator.compute())
        point["threshold"] = float(threshold)
        metric_core.assert_finite_numbers(
            point,
            f"knockout sweep threshold {threshold}",
        )
        points.append(point)
    fixed_matches = [
        point for point in points if float(point["threshold"]) == 0.5
    ]
    _require_equal("computed threshold=0.5 point count", len(fixed_matches), 1)
    fixed = fixed_matches[0]
    budget_points: dict[str, Any] = {}
    for budget, key in zip(spec.FA_BUDGETS, spec.BUDGET_KEYS):
        point = metric_core.best_point_under_fa(points, budget)
        if point is None:
            raise ValueError(f"no threshold satisfies FA budget {key}")
        budget_points[key] = copy.deepcopy(point)
    interim = {
        "validation_count": validation_count,
        "threshold_configuration": _threshold_configuration(),
        "threshold_provenance": provenance,
        "fixed_threshold_0_5": fixed,
        "best_points_under_fa_budget": budget_points,
        "points": points,
    }
    normalized = formal_evaluator._normalize_budgets(interim)
    interim["final_metric_coverage"] = _metric_coverage(fixed, normalized)
    _validate_sweep_payload(interim)
    return interim


def _gpu_memory_record(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "device_type": device.type,
            "max_memory_allocated_bytes": None,
            "max_memory_reserved_bytes": None,
        }
    return {
        "device_type": "cuda",
        "max_memory_allocated_bytes": int(
            torch.cuda.max_memory_allocated(device)
        ),
        "max_memory_reserved_bytes": int(
            torch.cuda.max_memory_reserved(device)
        ),
    }


def _evaluate_mode(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    source_state: Mapping[str, torch.Tensor],
    source_checkpoint_sha256: str,
    mode: str,
) -> dict[str, Any]:
    source_state_sha = state_content_sha256(source_state)
    source_non_dc_sha = non_dc_state_sha256(source_state)
    transformed = transform_state_dict(source_state, mode)
    changed = _changed_state_keys(source_state, transformed)
    transformed_sha = state_content_sha256(transformed)
    transformed_non_dc_sha = non_dc_state_sha256(transformed)
    load_result = model.load_state_dict(transformed, strict=True)
    _require_equal("strict-load missing keys", list(load_result.missing_keys), [])
    _require_equal(
        "strict-load unexpected keys",
        list(load_result.unexpected_keys),
        [],
    )
    model_state_before = model.state_dict()
    _require_equal(
        "loaded transformed state SHA",
        state_content_sha256(model_state_before),
        transformed_sha,
    )
    _require_equal(
        "loaded non-DC state SHA",
        non_dc_state_sha256(model_state_before),
        source_non_dc_sha,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    probabilities, targets, losses = metric_core.collect_predictions(
        model,
        loader,
        device,
    )
    memory = _gpu_memory_record(device)
    _require_equal(
        "post-inference transformed state SHA",
        state_content_sha256(model.state_dict()),
        transformed_sha,
    )
    _require_equal(
        "post-inference non-DC state SHA",
        non_dc_state_sha256(model.state_dict()),
        source_non_dc_sha,
    )
    _require_equal(
        "source state SHA after inference",
        state_content_sha256(source_state),
        source_state_sha,
    )
    sweep = sweep_predictions(
        probabilities,
        targets,
        losses,
        validation_count=spec.VALIDATION_COUNT,
    )
    requested = list(spec.KNOCKOUT_ZERO_KEYS[mode])
    evaluation = {
        "schema": STATE_TRANSFORM_SCHEMA,
        "status": "complete",
        "knockout_mode": mode,
        "zeroed_state_keys": requested,
        "effective_changed_state_keys": changed,
        "source_dc_offsets": dc_offset_records(source_state),
        "evaluated_dc_offsets": dc_offset_records(transformed),
        "source_state_dict_sha256": source_state_sha,
        "evaluated_state_dict_sha256": transformed_sha,
        "source_checkpoint_sha256_before": source_checkpoint_sha256,
        "source_checkpoint_sha256_after": source_checkpoint_sha256,
        "non_dc_state_sha256_before": source_non_dc_sha,
        "non_dc_state_sha256_after": transformed_non_dc_sha,
        "validation_count": spec.VALIDATION_COUNT,
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "formal_gate_eligible": False,
        **sweep,
        "audit": {
            "source_state_strict_load": True,
            "only_requested_dc_state_keys_changed": (
                set(changed) <= set(requested)
            ),
            "requested_dc_state_keys_zero": all(
                evaluation_record["value"] == 0.0
                for key, evaluation_record in dc_offset_records(
                    transformed
                ).items()
                if key in requested
            ),
            "non_zeroed_dc_offsets_preserved": all(
                torch.equal(transformed[key], source_state[key])
                for key in set(spec.DC_OFFSET_KEYS) - set(requested)
            ),
            "source_state_unchanged": (
                state_content_sha256(source_state) == source_state_sha
            ),
            "transformed_state_stable_during_inference": True,
            "non_dc_state_unchanged": (
                transformed_non_dc_sha == source_non_dc_sha
            ),
            "closed_interval_validated": True,
            "derived_checkpoint_written": False,
            "gpu_memory": memory,
        },
    }
    return evaluation


def _artifact_hashes(
    *,
    checkpoint_path: Path,
    artifact_audit: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> dict[str, str]:
    formal_binding = artifact_audit.get("evaluation_source_binding")
    if not isinstance(formal_binding, Mapping):
        raise ValueError("formal evaluation source binding is missing")
    run_dir = Path(str(artifact_audit["run_directory"])).resolve()
    expected = {
        "diagnostic_evaluator": _sha256_file(EVALUATOR_PATH),
        "diagnostic_source_lock": source_binding[
            "diagnostic_source_lock"
        ]["sha256"],
        "knockout_spec_source": _sha256_file(Path(spec.__file__).resolve()),
        "formal_evaluator": formal_binding["evaluator"]["sha256"],
        "formal_training_source_lock": formal_binding[
            "training_source_lock"
        ]["sha256"],
        "formal_acceptance_source_lock": formal_binding[
            "acceptance_source_lock"
        ]["sha256"],
        "shared_metric_core": formal_binding[
            "shared_metric_core"
        ]["sha256"],
        "closed_interval_core": formal_binding[
            "closed_interval_core"
        ]["sha256"],
        "determinism_core": formal_binding[
            "determinism_core"
        ]["sha256"],
        "protocol.json": _sha256_file(run_dir / "protocol.json"),
        "split.json": _sha256_file(run_dir / "split.json"),
        "summary.json": _sha256_file(run_dir / "summary.json"),
        "metrics.jsonl": _sha256_file(run_dir / "metrics.jsonl"),
        "source_checkpoint": _sha256_file(checkpoint_path),
    }
    for key in (
        "evaluator",
        "shared_metric_core",
        "closed_interval_core",
        "determinism_core",
    ):
        path = Path(str(formal_binding[key]["path"])).resolve()
        _require_equal(
            f"formal source binding {key}",
            _sha256_file(path),
            expected[
                "formal_evaluator" if key == "evaluator" else key
            ],
        )
    return expected


def _validated_cuda_lane(
    checkpoint: str,
    device_name: str,
) -> dict[str, Any]:
    """Prove that logical cuda:0 is the checkpoint's fixed GPU2/3 lane."""

    _require_equal("knockout CUDA device", device_name, "cuda:0")
    expected = CHECKPOINT_GPU_LANES.get(checkpoint)
    if expected is None:
        raise ValueError(f"unsupported checkpoint GPU lane: {checkpoint}")
    expected_index = int(expected["physical_gpu_index"])
    expected_uuid = str(expected["physical_gpu_uuid"])
    _require_equal(
        "inherited evaluator CUBLAS workspace contract",
        CUBLAS_WORKSPACE_CONFIG,
        CUBLAS_WORKSPACE_CONFIG_VALUE,
    )
    _require_equal(
        "CUDA_DEVICE_ORDER",
        os.environ.get("CUDA_DEVICE_ORDER"),
        CUDA_DEVICE_ORDER,
    )
    _require_equal(
        "CUDA_VISIBLE_DEVICES",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        expected_uuid,
    )
    _require_equal(
        CUBLAS_WORKSPACE_CONFIG_ENV,
        os.environ.get(CUBLAS_WORKSPACE_CONFIG_ENV),
        CUBLAS_WORKSPACE_CONFIG_VALUE,
    )
    _require_equal(
        PYTHONHASHSEED_ENV,
        os.environ.get(PYTHONHASHSEED_ENV),
        PYTHONHASHSEED_VALUE,
    )
    _require_equal(
        PHYSICAL_GPU_INDEX_ENV,
        os.environ.get(PHYSICAL_GPU_INDEX_ENV),
        str(expected_index),
    )
    _require_equal(
        PHYSICAL_GPU_UUID_ENV,
        os.environ.get(PHYSICAL_GPU_UUID_ENV),
        expected_uuid,
    )
    return {
        "logical_device": "cuda:0",
        "physical_gpu_index": expected_index,
        "physical_gpu_uuid": expected_uuid,
        "cuda_device_order": CUDA_DEVICE_ORDER,
        "cuda_visible_devices": expected_uuid,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG_VALUE,
        "pythonhashseed": PYTHONHASHSEED_VALUE,
        "checkpoint": checkpoint,
    }


def validate_evaluation_payload(
    payload: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    artifact_audit: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete no-authority diagnostic publication."""

    if not isinstance(payload, Mapping):
        raise ValueError("evaluation payload must be an object")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be an object")
    if not isinstance(artifact_audit, Mapping) or not artifact_audit:
        raise ValueError("artifact_audit must be a non-empty object")
    if not isinstance(source_binding, Mapping) or not source_binding:
        raise ValueError("source_binding must be a non-empty object")
    diagnostic_lock = source_binding.get("diagnostic_source_lock")
    if not isinstance(diagnostic_lock, Mapping):
        raise ValueError(
            "source_binding.diagnostic_source_lock must be an object"
        )
    required_audit_fields = (
        "checkpoint_filename",
        "checkpoint_role",
        "checkpoint_epoch",
        "checkpoint_sha256",
        "run_identity",
        "checkpoint_identity",
    )
    missing_audit = [
        name for name in required_audit_fields if name not in artifact_audit
    ]
    if missing_audit:
        raise ValueError(
            f"artifact_audit is incomplete: {missing_audit}"
        )
    for name in ("sha256",):
        if name not in diagnostic_lock:
            raise ValueError(
                f"source_binding.diagnostic_source_lock.{name} is missing"
            )
    if "knockout_spec_sha256" not in source_binding:
        raise ValueError("source_binding.knockout_spec_sha256 is missing")
    ready = copy.deepcopy(dict(payload))
    required_scalars = {
        "schema": EVALUATION_SCHEMA,
        "status": "complete",
        "artifact_kind": spec.ARTIFACT_KIND,
        "scope": "evaluation_only_same_checkpoint_counterfactual",
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "formal_gate_eligible": False,
        "official_test_accessed": False,
        "dataset": spec.DATASET,
        "variant": spec.VARIANT,
        "training_seed": spec.TRAINING_SEED,
        "split_seed": spec.SPLIT_SEED,
        "expected_epochs": spec.EXPECTED_EPOCHS,
        "validation_count": spec.VALIDATION_COUNT,
        "checkpoint_filename": artifact_audit["checkpoint_filename"],
        "checkpoint_role": artifact_audit["checkpoint_role"],
        "checkpoint_epoch": artifact_audit["checkpoint_epoch"],
        "diagnostic_source_lock_sha256": diagnostic_lock["sha256"],
        "knockout_spec_sha256": source_binding[
            "knockout_spec_sha256"
        ],
    }
    for name, expected in required_scalars.items():
        _require_equal(f"payload.{name}", ready.get(name), expected)
    _require_equal(
        "payload source_binding",
        ready.get("source_binding"),
        source_binding,
    )
    _require_equal(
        "payload run_identity",
        ready.get("run_identity"),
        artifact_audit.get("run_identity"),
    )
    _require_equal(
        "payload source checkpoint identity",
        ready.get("source_checkpoint_identity"),
        artifact_audit.get("checkpoint_identity"),
    )
    run_directory = Path(str(ready.get("run_directory"))).resolve()
    expected_run_directory = Path(
        str(artifact_audit.get("run_directory"))
    ).resolve()
    _require_equal(
        "payload run directory",
        run_directory,
        expected_run_directory,
    )
    _require_equal(
        "payload canonical run directory",
        run_directory,
        spec.FORMAL_RUN_DIR.resolve(),
    )
    device_lane = ready.get("device_lane")
    if not isinstance(device_lane, Mapping):
        raise ValueError("payload.device_lane must be an object")
    expected_lane = CHECKPOINT_GPU_LANES[
        str(artifact_audit["checkpoint_filename"])
    ]
    for name, expected in {
        "logical_device": "cuda:0",
        "physical_gpu_index": expected_lane["physical_gpu_index"],
        "physical_gpu_uuid": expected_lane["physical_gpu_uuid"],
        "cuda_device_order": CUDA_DEVICE_ORDER,
        "cuda_visible_devices": expected_lane["physical_gpu_uuid"],
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG_VALUE,
        "pythonhashseed": PYTHONHASHSEED_VALUE,
        "checkpoint": artifact_audit["checkpoint_filename"],
    }.items():
        _require_equal(
            f"payload.device_lane.{name}",
            device_lane.get(name),
            expected,
        )
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("validation checkpoint has no state_dict")
    source_state_sha = state_content_sha256(state)
    _require_equal(
        "payload source state SHA",
        ready.get("source_state_dict_sha256"),
        source_state_sha,
    )
    _canonical_json_equal(
        "payload original DC offsets",
        ready.get("original_dc_offsets"),
        dc_offset_records(state),
    )
    evaluations = ready.get("evaluations")
    if not isinstance(evaluations, list):
        raise ValueError("payload.evaluations must be a list")
    _require_equal(
        "evaluation mode order",
        [
            item.get("knockout_mode")
            if isinstance(item, Mapping)
            else None
            for item in evaluations
        ],
        list(spec.KNOCKOUT_MODES),
    )
    _require_equal(
        "evaluation count",
        len(evaluations),
        len(spec.KNOCKOUT_MODES),
    )
    source_non_dc_sha = non_dc_state_sha256(state)
    for mode, evaluation in zip(spec.KNOCKOUT_MODES, evaluations):
        if not isinstance(evaluation, Mapping):
            raise ValueError(f"evaluation {mode} must be an object")
        for name, expected in {
            "schema": STATE_TRANSFORM_SCHEMA,
            "status": "complete",
            "knockout_mode": mode,
            "zeroed_state_keys": list(spec.KNOCKOUT_ZERO_KEYS[mode]),
            "source_state_dict_sha256": source_state_sha,
            "source_checkpoint_sha256_before": artifact_audit[
                "checkpoint_sha256"
            ],
            "source_checkpoint_sha256_after": artifact_audit[
                "checkpoint_sha256"
            ],
            "non_dc_state_sha256_before": source_non_dc_sha,
            "non_dc_state_sha256_after": source_non_dc_sha,
            "validation_count": spec.VALIDATION_COUNT,
            "diagnostic_only": True,
            "affects_formal_gate": False,
            "formal_decision_authority": False,
            "formal_gate_eligible": False,
        }.items():
            _require_equal(f"evaluation {mode}.{name}", evaluation.get(name), expected)
        changed = evaluation.get("effective_changed_state_keys")
        if not isinstance(changed, list) or not set(changed) <= set(
            spec.KNOCKOUT_ZERO_KEYS[mode]
        ):
            raise ValueError(f"evaluation {mode} changed invalid state keys")
        transformed = transform_state_dict(state, mode)
        _require_equal(
            f"evaluation {mode} transformed state SHA",
            evaluation.get("evaluated_state_dict_sha256"),
            state_content_sha256(transformed),
        )
        _canonical_json_equal(
            f"evaluation {mode} source DC offsets",
            evaluation.get("source_dc_offsets"),
            dc_offset_records(state),
        )
        _canonical_json_equal(
            f"evaluation {mode} evaluated DC offsets",
            evaluation.get("evaluated_dc_offsets"),
            dc_offset_records(transformed),
        )
        audit = evaluation.get("audit")
        if not isinstance(audit, Mapping):
            raise ValueError(f"evaluation {mode}.audit must be an object")
        required_checks = (
            "source_state_strict_load",
            "only_requested_dc_state_keys_changed",
            "requested_dc_state_keys_zero",
            "non_zeroed_dc_offsets_preserved",
            "source_state_unchanged",
            "transformed_state_stable_during_inference",
            "non_dc_state_unchanged",
            "closed_interval_validated",
        )
        if any(audit.get(name) is not True for name in required_checks):
            raise ValueError(f"evaluation {mode} integrity checks are incomplete")
        _require_equal(
            f"evaluation {mode} derived checkpoint",
            audit.get("derived_checkpoint_written"),
            False,
        )
        _validate_sweep_payload(evaluation)
    checkpoint_path = (
        run_directory / str(artifact_audit["checkpoint_filename"])
    )
    expected_artifact_hashes = _artifact_hashes(
        checkpoint_path=checkpoint_path,
        artifact_audit=artifact_audit,
        source_binding=source_binding,
    )
    _require_equal(
        "payload artifact SHA registry",
        ready.get("artifact_sha256"),
        expected_artifact_hashes,
    )
    forbidden = {"decision", "performance_gate_assessment"}

    def _walk(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            overlap = forbidden & set(value)
            if overlap:
                raise ValueError(
                    f"formal decision fields forbidden at {location}: "
                    f"{sorted(overlap)}"
                )
            for name, item in value.items():
                _walk(item, f"{location}.{name}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                _walk(item, f"{location}[{index}]")

    _walk(ready, "payload")
    metric_core.assert_finite_numbers(ready, "knockout evaluation payload")
    return metric_core.json_ready(ready)


def atomic_publish_new(
    path: Path,
    payload: Mapping[str, Any],
) -> Path:
    """Atomically publish a new JSON artifact without an overwrite path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    ready = metric_core.json_ready(payload)
    content = (
        json.dumps(
            ready,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite diagnostic output: {output}"
            ) from exc
        directory_descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _build_plan_from_preflight(
    checkpoint: str,
    *,
    source_binding: Mapping[str, Any],
    artifact_audit: Mapping[str, Any],
) -> dict[str, Any]:
    output = spec.sweep_path(checkpoint)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    return {
        "schema": (
            "sctransnet_tpd_ner_v8_mprs_dch_v3_dc_knockout_plan_v1"
        ),
        "status": "ready",
        "artifact_kind": spec.ARTIFACT_KIND,
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "formal_gate_eligible": False,
        "official_test_accessed": False,
        "checkpoint": checkpoint,
        "checkpoint_role": spec.CHECKPOINT_ROLES[checkpoint],
        "checkpoint_epoch": artifact_audit["checkpoint_epoch"],
        "source_checkpoint_sha256": artifact_audit["checkpoint_sha256"],
        "source_state_dict_sha256": None,
        "source_binding": copy.deepcopy(dict(source_binding)),
        "diagnostic_source_lock_sha256": source_binding[
            "diagnostic_source_lock"
        ]["sha256"],
        "knockout_spec_sha256": source_binding[
            "knockout_spec_sha256"
        ],
        "knockout_modes": list(spec.KNOCKOUT_MODES),
        "mode_count": len(spec.KNOCKOUT_MODES),
        "output": str(output.resolve()),
        "output_overwrite_forbidden": True,
        "derived_checkpoint_written": False,
        "resource_policy": {
            "modes_evaluated_sequentially": True,
            "checkpoint_scope_per_invocation": 1,
            "cross_checkpoint_scheduling": "parallel_fixed_gpu2_gpu3",
            "batch_size": 1,
            "num_workers": 0,
            "fp32_inference": True,
            "minimum_recommended_free_gpu_memory_gib": 8,
            "preferred_free_gpu_memory_gib": 12,
            "memory_values_are_conservative_estimates_not_measurements": True,
            "required_process_environment": {
                CUBLAS_WORKSPACE_CONFIG_ENV: (
                    CUBLAS_WORKSPACE_CONFIG_VALUE
                ),
                PYTHONHASHSEED_ENV: PYTHONHASHSEED_VALUE,
            },
        },
    }


def build_plan(checkpoint: str) -> dict[str, Any]:
    """Fail-closed, CPU-only preflight for one fixed checkpoint."""

    if checkpoint not in spec.CHECKPOINTS:
        raise ValueError(f"unsupported checkpoint: {checkpoint}")
    source_binding = source_freezer.current_source_binding()
    artifact_audit = formal_evaluator.validate_run_artifacts(
        spec.FORMAL_RUN_DIR,
        checkpoint,
    )
    return _build_plan_from_preflight(
        checkpoint,
        source_binding=source_binding,
        artifact_audit=artifact_audit,
    )


def run_checkpoint(checkpoint: str, device_name: str) -> dict[str, Any]:
    """Evaluate all four fixed modes and publish one checkpoint artifact."""

    if checkpoint not in spec.CHECKPOINTS:
        raise ValueError(f"unsupported checkpoint: {checkpoint}")
    if device_name != "cuda:0":
        raise ValueError("formal knockout execution requires logical cuda:0")

    # All mutable runtime/device setup deliberately follows immutable preflight.
    source_binding_before = source_freezer.current_source_binding()
    artifact_audit = formal_evaluator.validate_run_artifacts(
        spec.FORMAL_RUN_DIR,
        checkpoint,
    )
    plan = _build_plan_from_preflight(
        checkpoint,
        source_binding=source_binding_before,
        artifact_audit=artifact_audit,
    )
    checkpoint_payload, checkpoint_path = _source_checkpoint(
        checkpoint,
        artifact_audit,
    )
    source_state = checkpoint_payload["state_dict"]
    source_state_sha = state_content_sha256(source_state)
    source_non_dc_sha = non_dc_state_sha256(source_state)
    source_checkpoint_sha = _sha256_file(checkpoint_path)

    device_lane = _validated_cuda_lane(checkpoint, device_name)
    determinism = configure_v8_inference(device_name)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, model_metadata = formal_evaluator.build_model(
        spec.VARIANT,
        spec.TRAINING_SEED,
    )
    initial_load = model.load_state_dict(source_state, strict=True)
    _require_equal(
        "initial strict-load missing keys",
        list(initial_load.missing_keys),
        [],
    )
    _require_equal(
        "initial strict-load unexpected keys",
        list(initial_load.unexpected_keys),
        [],
    )
    _require_equal(
        "initial model state SHA",
        state_content_sha256(model.state_dict()),
        source_state_sha,
    )
    model.to(device)
    loader, data_contract = _evaluation_data(artifact_audit)
    evaluations: list[dict[str, Any]] = []
    for mode in spec.KNOCKOUT_MODES:
        _require_equal(
            f"source checkpoint SHA before {mode}",
            _sha256_file(checkpoint_path),
            source_checkpoint_sha,
        )
        evaluations.append(
            _evaluate_mode(
                model=model,
                loader=loader,
                device=device,
                source_state=source_state,
                source_checkpoint_sha256=source_checkpoint_sha,
                mode=mode,
            )
        )
        _require_equal(
            f"source checkpoint SHA after {mode}",
            _sha256_file(checkpoint_path),
            source_checkpoint_sha,
        )
    _require_equal(
        "source state SHA after all modes",
        state_content_sha256(source_state),
        source_state_sha,
    )
    _require_equal(
        "source non-DC state SHA after all modes",
        non_dc_state_sha256(source_state),
        source_non_dc_sha,
    )
    source_binding_after = source_freezer.current_source_binding()
    _require_equal(
        "diagnostic source binding before/after",
        source_binding_after,
        source_binding_before,
    )
    _require_equal(
        "source checkpoint SHA before publication",
        _sha256_file(checkpoint_path),
        source_checkpoint_sha,
    )
    artifact_hashes = _artifact_hashes(
        checkpoint_path=checkpoint_path,
        artifact_audit=artifact_audit,
        source_binding=source_binding_before,
    )
    output = {
        "schema": EVALUATION_SCHEMA,
        "status": "complete",
        "artifact_kind": spec.ARTIFACT_KIND,
        "scope": "evaluation_only_same_checkpoint_counterfactual",
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "formal_gate_eligible": False,
        "formal_gate_components": [],
        "official_test_accessed": False,
        "dataset": spec.DATASET,
        "variant": spec.VARIANT,
        "training_seed": spec.TRAINING_SEED,
        "split_seed": spec.SPLIT_SEED,
        "expected_epochs": spec.EXPECTED_EPOCHS,
        "validation_count": spec.VALIDATION_COUNT,
        "validation_split_sha256": artifact_audit[
            "validation_split_sha256"
        ],
        "run_directory": artifact_audit["run_directory"],
        "run_identity": copy.deepcopy(artifact_audit["run_identity"]),
        "checkpoint_filename": checkpoint,
        "checkpoint_role": artifact_audit["checkpoint_role"],
        "checkpoint_epoch": artifact_audit["checkpoint_epoch"],
        "checkpoint_validation_metrics": copy.deepcopy(
            artifact_audit["checkpoint_validation_metrics"]
        ),
        "source_checkpoint_identity": copy.deepcopy(
            artifact_audit["checkpoint_identity"]
        ),
        "source_checkpoint": {
            "path": str(checkpoint_path),
            "filename": checkpoint,
            "role": artifact_audit["checkpoint_role"],
            "epoch": artifact_audit["checkpoint_epoch"],
            "sha256": source_checkpoint_sha,
            "state_dict_sha256": source_state_sha,
            "checkpoint_identity": copy.deepcopy(
                artifact_audit["checkpoint_identity"]
            ),
            "validation_metrics": copy.deepcopy(
                artifact_audit["checkpoint_validation_metrics"]
            ),
        },
        "source_state_dict_sha256": source_state_sha,
        "source_non_dc_state_sha256": source_non_dc_sha,
        "original_dc_offsets": dc_offset_records(source_state),
        "knockout_modes": list(spec.KNOCKOUT_MODES),
        "knockout_specification": spec.fixed_specification(),
        "knockout_spec_sha256": source_binding_before[
            "knockout_spec_sha256"
        ],
        "source_binding": copy.deepcopy(source_binding_before),
        "diagnostic_source_lock_sha256": source_binding_before[
            "diagnostic_source_lock"
        ]["sha256"],
        "threshold_contract": spec.threshold_contract(),
        "data_contract": data_contract,
        "device_lane": device_lane,
        "model_metadata": copy.deepcopy(model_metadata),
        "determinism": determinism,
        "evaluations": evaluations,
        "artifact_sha256": artifact_hashes,
        "audit": {
            "plan": plan,
            "source_checkpoint_sha256_before": source_checkpoint_sha,
            "source_checkpoint_sha256_after": _sha256_file(
                checkpoint_path
            ),
            "source_state_dict_sha256_before": source_state_sha,
            "source_state_dict_sha256_after": state_content_sha256(
                source_state
            ),
            "non_dc_state_sha256_before": source_non_dc_sha,
            "non_dc_state_sha256_after": non_dc_state_sha256(
                source_state
            ),
            "diagnostic_source_binding_before": copy.deepcopy(
                source_binding_before
            ),
            "diagnostic_source_binding_after": copy.deepcopy(
                source_binding_after
            ),
            "formal_artifacts_read_only": True,
            "formal_artifacts_unchanged": True,
            "all_modes_from_pristine_source_state": True,
            "modes_evaluated_sequentially": True,
            "derived_checkpoint_written": False,
            "output_overwrite_forbidden": True,
            "invocation_argv": [
                sys.executable,
                str(EVALUATOR_PATH),
                *sys.argv[1:],
            ],
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES"
            ),
            "cublas_workspace_config": os.environ.get(
                CUBLAS_WORKSPACE_CONFIG_ENV
            ),
            "pythonhashseed": os.environ.get(PYTHONHASHSEED_ENV),
        },
    }
    ready = validate_evaluation_payload(
        output,
        checkpoint_payload,
        artifact_audit,
        source_binding_before,
    )
    _require_equal(
        "source binding immediately before publication",
        source_freezer.current_source_binding(),
        source_binding_before,
    )
    _require_equal(
        "checkpoint immediately before publication",
        _sha256_file(checkpoint_path),
        source_checkpoint_sha,
    )
    output_path = spec.sweep_path(checkpoint)
    atomic_publish_new(output_path, ready)
    return ready


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run the independent V3 DC-offset knockout diagnostic"
        )
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--plan",
        action="store_true",
        help="verify frozen inputs and print the fixed plan without inference",
    )
    action.add_argument(
        "--run",
        action="store_true",
        help="evaluate all four fixed knockout modes",
    )
    parser.add_argument(
        "--checkpoint",
        choices=spec.CHECKPOINTS,
        required=True,
    )
    parser.add_argument(
        "--device",
        choices=("cuda:0",),
        default="cuda:0",
    )
    return parser.parse_args(
        list(sys.argv[1:] if argv is None else argv)
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.plan:
        output = build_plan(args.checkpoint)
    else:
        output = run_checkpoint(args.checkpoint, args.device)
    print(
        json.dumps(
            metric_core.json_ready(output),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "EVALUATION_SCHEMA",
    "FINAL_METRIC_COVERAGE_SCHEMA",
    "STATE_TRANSFORM_SCHEMA",
    "CUBLAS_WORKSPACE_CONFIG_ENV",
    "CUBLAS_WORKSPACE_CONFIG_VALUE",
    "PYTHONHASHSEED_ENV",
    "PYTHONHASHSEED_VALUE",
    "atomic_publish_new",
    "build_plan",
    "dc_offset_records",
    "main",
    "non_dc_state_sha256",
    "parse_args",
    "run_checkpoint",
    "state_content_sha256",
    "sweep_predictions",
    "transform_state_dict",
    "validate_evaluation_payload",
]
