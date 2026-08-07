#!/usr/bin/env python3
"""Exhaustive, zero-margin PBDR-V3 residual-calibration sweep.

Only a validated ``internal_validation`` raw-logit cache is accepted.  The
fixed 378-entry grid is evaluated at ``sigmoid(logit) > 0.5`` with the V4
metric core.  Current and PBDR-V3 grid anchors must replay the corresponding
cached logits bit for bit before any selection result can be produced.

The output is deterministic and self-hashed.  Writing is single-use via
``O_EXCL``; reading performs a full cache-backed recomputation rather than
trusting the serialized winner or metric values.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from experiments import pbdr_v3_residual_calibration as calibration
from experiments import pbdr_v4_internal_cache as cache_io
from experiments import pbdr_v4_metric_core as metric_core
from experiments import pbdr_v4_zero_margin_selector as selector


SCHEMA = "sctransnet_pbdr_v3_residual_calibration_sweep_v1/v1"
GRID_SCHEMA = "sctransnet_pbdr_v3_residual_calibration_grid_v1/v1"
EXPECTED_GRID_SIZE = 378
ROLE_KEY_FIELDS: Mapping[str, tuple[str, ...]] = {
    "best_miou": (
        "miou",
        "pd",
        "negative_fa",
        "niou",
        "tiny_pd",
        "negative_loss",
    ),
    "best_pd": (
        "pd",
        "negative_fa",
        "tiny_pd",
        "miou",
        "niou",
        "negative_loss",
    ),
}


class PBDRV3ResidualSweepError(ValueError):
    """The cache, frozen sweep, result, or immutable output is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV3ResidualSweepError(message)


def _source_path(module: Any, label: str) -> Path:
    raw = getattr(module, "__file__", None)
    _require(type(raw) is str and bool(raw), f"{label} source path is unavailable")
    supplied = Path(raw)
    _require(
        not supplied.is_symlink() and supplied.is_file(),
        f"{label} source must be a regular non-symlink file",
    )
    return supplied.resolve(strict=True)


def _source_sha256(module: Any, label: str) -> str:
    return cache_io.file_sha256(_source_path(module, label))


def _assert_tf32_disabled() -> None:
    _require(
        torch.backends.cuda.matmul.allow_tf32 is False
        and torch.backends.cudnn.allow_tf32 is False,
        "both live TF32 switches must be false",
    )


def _read_json_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    candidate = Path(path)
    _require(
        not candidate.is_symlink() and candidate.is_file(),
        f"{label} must be a regular non-symlink file",
    )
    try:
        raw = candidate.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV3ResidualSweepError(f"cannot read {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    return value, raw


def _cache_binding(
    cache: cache_io.ValidatedInternalRawLogitCache,
) -> dict[str, Any]:
    commit_path = cache.path / cache_io.COMMIT_NAME
    manifest_path = cache.path / cache_io.MANIFEST_NAME
    commit, commit_bytes = _read_json_object(commit_path, "cache commit")
    manifest, manifest_bytes = _read_json_object(manifest_path, "cache manifest")
    _require(manifest == dict(cache.manifest), "validated cache manifest changed")
    identity = manifest.get("identity")
    _require(isinstance(identity, Mapping), "cache identity is malformed")
    return {
        "cache_path": str(cache.path),
        "commit_sha256": commit.get("commit_sha256"),
        "commit_file_bytes": len(commit_bytes),
        "commit_file_sha256": cache_io.file_sha256(commit_path),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "manifest_file_bytes": len(manifest_bytes),
        "manifest_file_sha256": cache_io.file_sha256(manifest_path),
        "identity_sha256": identity.get("identity_sha256"),
    }


@dataclass(frozen=True, slots=True)
class _GridBinding:
    entries: tuple[calibration.ResidualCalibration, ...]
    current_index: int
    v3_index: int
    payload: Mapping[str, Any]


def _frozen_grid() -> _GridBinding:
    entries = calibration.calibration_grid()
    _require(len(entries) == EXPECTED_GRID_SIZE, "calibration grid is not 378 entries")
    _require(len(set(entries)) == len(entries), "calibration grid contains duplicates")
    current_indices = [
        index for index, config in enumerate(entries)
        if config == calibration.CURRENT_ANCHOR
    ]
    v3_indices = [
        index for index, config in enumerate(entries)
        if config == calibration.PBDR_V3_ANCHOR
    ]
    _require(len(current_indices) == 1, "Current anchor is not unique in the grid")
    _require(len(v3_indices) == 1, "PBDR-V3 anchor is not unique in the grid")
    configs = [config.as_dict() for config in entries]
    payload: dict[str, Any] = {
        "schema": GRID_SCHEMA,
        "entry_count": len(entries),
        "iteration_order": "positive_scale_then_negative_scale_then_bias",
        "positive_scales": [float(value) for value in calibration.POSITIVE_SCALES],
        "negative_scales": [float(value) for value in calibration.NEGATIVE_SCALES],
        "biases": [float(value) for value in calibration.BIASES],
        "configs_canonical_sha256": cache_io.canonical_sha256(configs),
        "current_anchor_grid_index": current_indices[0],
        "v3_anchor_grid_index": v3_indices[0],
        "calibration_source_sha256": _source_sha256(
            calibration,
            "residual calibration",
        ),
    }
    payload["grid_binding_sha256"] = cache_io.canonical_sha256(payload)
    return _GridBinding(
        entries=entries,
        current_index=current_indices[0],
        v3_index=v3_indices[0],
        payload=payload,
    )


@dataclass(frozen=True, slots=True)
class _PreparedSample:
    sample_id: str
    base_logits: torch.Tensor
    delta_logits: torch.Tensor
    routed_logits: torch.Tensor
    current_logits: torch.Tensor
    original_logits: torch.Tensor
    target: torch.Tensor
    cached_routed: np.ndarray
    cached_current: np.ndarray
    target_array: np.ndarray


def _as_bchw(value: np.ndarray) -> torch.Tensor:
    copied = np.array(value, dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(copied).unsqueeze(0).unsqueeze(0)


def _prepare_samples(
    cache: cache_io.ValidatedInternalRawLogitCache,
) -> tuple[_PreparedSample, ...]:
    prepared: list[_PreparedSample] = []
    for sample in cache.samples:
        arrays = sample.arrays
        prepared.append(
            _PreparedSample(
                sample_id=sample.sample_id,
                base_logits=_as_bchw(arrays["base_logits"]),
                delta_logits=_as_bchw(arrays["delta_logits"]),
                routed_logits=_as_bchw(arrays["routed_logits"]),
                current_logits=_as_bchw(arrays["current_logits"]),
                original_logits=_as_bchw(arrays["original_logits"]),
                target=_as_bchw(arrays["target"]),
                cached_routed=arrays["routed_logits"],
                cached_current=arrays["current_logits"],
                target_array=arrays["target"],
            )
        )
    _require(bool(prepared), "internal-validation cache cannot be empty")
    return tuple(prepared)


def _tensor_map_bytes(value: torch.Tensor) -> bytes:
    _require(
        value.device.type == "cpu"
        and value.dtype == torch.float32
        and value.ndim == 4
        and value.shape[0] == 1
        and value.shape[1] == 1,
        "calibrated logits violate the CPU FP32 BCHW contract",
    )
    ready = np.ascontiguousarray(value.detach()[0, 0].numpy())
    return ready.tobytes(order="C")


def _verify_anchor_replay(
    samples: Sequence[_PreparedSample],
    grid: _GridBinding,
) -> dict[str, Any]:
    with torch.no_grad():
        for sample in samples:
            current = calibration.apply_residual_calibration(
                sample.base_logits,
                sample.delta_logits,
                calibration.CURRENT_ANCHOR,
            )
            _require(
                _tensor_map_bytes(current)
                == sample.cached_current.tobytes(order="C"),
                f"Current anchor is not byte-exact for sample {sample.sample_id!r}",
            )
            routed = calibration.apply_residual_calibration(
                sample.base_logits,
                sample.delta_logits,
                calibration.PBDR_V3_ANCHOR,
            )
            _require(
                _tensor_map_bytes(routed)
                == sample.cached_routed.tobytes(order="C"),
                f"PBDR-V3 anchor is not byte-exact for sample {sample.sample_id!r}",
            )
    return {
        "current": {
            "grid_index": grid.current_index,
            "config": calibration.CURRENT_ANCHOR.as_dict(),
            "cache_tensor": "current_logits",
            "sample_count": len(samples),
            "byte_exact_for_every_sample": True,
        },
        "v3": {
            "grid_index": grid.v3_index,
            "config": calibration.PBDR_V3_ANCHOR.as_dict(),
            "cache_tensor": "routed_logits",
            "sample_count": len(samples),
            "byte_exact_for_every_sample": True,
        },
    }


def _evaluate_logits(
    samples: Sequence[_PreparedSample],
    *,
    cached_field: str | None = None,
    config: calibration.ResidualCalibration | None = None,
) -> dict[str, object]:
    _require(
        (cached_field is None) != (config is None),
        "evaluation must specify exactly one logit source",
    )
    accumulator = metric_core.PBDRV4MetricAccumulator()
    with torch.no_grad():
        for sample in samples:
            if config is not None:
                logits = calibration.apply_residual_calibration(
                    sample.base_logits,
                    sample.delta_logits,
                    config,
                )
            else:
                _require(
                    cached_field in {
                        "original_logits",
                        "current_logits",
                        "routed_logits",
                    },
                    "unsupported cached logit field",
                )
                logits = getattr(sample, cached_field)
            _require(
                logits.dtype == torch.float32 and logits.device.type == "cpu",
                "sweep logits must remain CPU FP32",
            )
            probability = torch.sigmoid(logits)
            loss = F.binary_cross_entropy(
                probability,
                sample.target,
                reduction="mean",
            )
            probability_map = np.ascontiguousarray(
                probability[0, 0].detach().numpy(),
                dtype=np.float32,
            )
            accumulator.update(
                probability=probability_map,
                target=sample.target_array,
                loss=float(loss.item()),
                identifier=sample.sample_id,
            )
    return accumulator.compute()


def _evaluation_context_sha256(
    *,
    cache_binding: Mapping[str, Any],
    grid_binding: Mapping[str, Any],
    metric_core_sha256: str,
    selector_sha256: str,
    source_lock_sha256: str,
) -> str:
    return cache_io.canonical_sha256(
        {
            "schema": "sctransnet_pbdr_v3_residual_sweep_context_v1/v1",
            "cache_commit_sha256": cache_binding["commit_sha256"],
            "cache_manifest_sha256": cache_binding["manifest_sha256"],
            "grid_binding_sha256": grid_binding["grid_binding_sha256"],
            "metric_core_source_sha256": metric_core_sha256,
            "selector_source_sha256": selector_sha256,
            "source_lock_sha256": source_lock_sha256,
            "probability_transform": "torch_sigmoid_float32",
            "probability_comparison": "strict_greater_than",
            "threshold_numerator": 1,
            "threshold_denominator": 2,
            "loss": "torch_bce_on_float32_probability_mean_per_sample",
        }
    )


def _serialized_role_key(
    role: str,
    record: selector.MetricRecord,
) -> dict[str, Any]:
    fields = ROLE_KEY_FIELDS.get(role)
    _require(fields is not None, "unsupported sweep role")
    key = selector.role_key(role, record)  # type: ignore[arg-type]
    _require(len(key) == len(fields), "selector role-key arity differs")
    components: list[dict[str, Any]] = []
    for field, value in zip(fields, key):
        if isinstance(value, Fraction):
            components.append(
                {
                    "field": field,
                    "representation": "exact_fraction",
                    "numerator": value.numerator,
                    "denominator": value.denominator,
                }
            )
        else:
            numeric = float(value)
            _require(math.isfinite(numeric), "role-key float is non-finite")
            components.append(
                {
                    "field": field,
                    "representation": "binary64_hex",
                    "hex": numeric.hex(),
                }
            )
    return {
        "comparison": "lexicographic_maximum",
        "components": components,
    }


def _record_and_payload(
    *,
    name: str,
    family: selector.CandidateFamily,
    binding: selector.EvaluationBinding,
    metrics: Mapping[str, object],
) -> tuple[selector.MetricRecord, dict[str, Any]]:
    record = selector.MetricRecord.from_mapping(
        name=name,
        family=family,
        binding=binding,
        value=metrics,
    )
    return record, {
        "name": name,
        "family": family,
        "metrics": dict(metrics),
        "exact_sufficient_statistics": metric_core.exact_statistics(metrics),
        "role_key": _serialized_role_key(binding.role, record),
    }


def _assert_same_evaluation_targets(
    expected_sample_hash: str,
    expected_target_hash: str,
    metrics: Mapping[str, object],
    label: str,
) -> None:
    _require(
        metrics.get("sample_id_order_sha256") == expected_sample_hash,
        f"{label} sample-ID order hash differs",
    )
    _require(
        metrics.get("target_sha256") == expected_target_hash,
        f"{label} target hash differs",
    )


def compute_sweep_result(
    cache: cache_io.ValidatedInternalRawLogitCache,
) -> dict[str, Any]:
    """Compute all baselines and all 378 configs without touching a dataset."""

    _require(
        isinstance(cache, cache_io.ValidatedInternalRawLogitCache),
        "cache must be a ValidatedInternalRawLogitCache",
    )
    _assert_tf32_disabled()
    identity = cache.manifest.get("identity")
    _require(isinstance(identity, Mapping), "cache identity is malformed")
    _require(
        identity.get("partition") == "internal_validation",
        "residual sweep accepts internal_validation only",
    )
    _require(
        identity.get("official_test_accessed") is False
        and cache.manifest.get("official_test_accessed") is False,
        "cache claims official-test access",
    )
    role = identity.get("parent_role")
    _require(role in ROLE_KEY_FIELDS, "cache parent role is unsupported")

    metric_core_sha = _source_sha256(metric_core, "V4 metric core")
    selector_sha = _source_sha256(selector, "zero-margin selector")
    source_lock_sha = identity.get("source_lock_sha256")
    _require(
        type(source_lock_sha) is str and len(source_lock_sha) == 64,
        "cache source-lock SHA-256 is malformed",
    )
    _require(
        identity.get("metric_core_sha256") == metric_core_sha,
        "cache metric-core SHA-256 differs from the live V4 metric core",
    )
    runtime = identity.get("runtime")
    _require(isinstance(runtime, Mapping), "cache runtime binding is malformed")
    _require(
        runtime.get("cuda_matmul_allow_tf32") is False
        and runtime.get("cudnn_allow_tf32") is False,
        "cache runtime does not bind both TF32 switches to false",
    )

    cache_binding = _cache_binding(cache)
    grid = _frozen_grid()
    samples = _prepare_samples(cache)
    anchor_replay = _verify_anchor_replay(samples, grid)

    original_metrics = _evaluate_logits(samples, cached_field="original_logits")
    sample_hash = original_metrics.get("sample_id_order_sha256")
    target_hash = original_metrics.get("target_sha256")
    _require(type(sample_hash) is str, "metric sample-ID hash is malformed")
    _require(type(target_hash) is str, "metric target hash is malformed")
    context_sha = _evaluation_context_sha256(
        cache_binding=cache_binding,
        grid_binding=grid.payload,
        metric_core_sha256=metric_core_sha,
        selector_sha256=selector_sha,
        source_lock_sha256=source_lock_sha,
    )
    binding = selector.EvaluationBinding(
        dataset=identity["dataset"],
        role=role,
        evaluation_context_sha256=context_sha,
        sample_id_order_sha256=sample_hash,
        target_sha256=target_hash,
        metric_core_sha256=metric_core_sha,
    )
    _, original_payload = _record_and_payload(
        name="Original",
        family="Original",
        binding=binding,
        metrics=original_metrics,
    )

    current_metrics = _evaluate_logits(samples, cached_field="current_logits")
    v3_metrics = _evaluate_logits(samples, cached_field="routed_logits")
    _assert_same_evaluation_targets(sample_hash, target_hash, current_metrics, "Current")
    _assert_same_evaluation_targets(sample_hash, target_hash, v3_metrics, "PBDR-V3")
    _, current_payload = _record_and_payload(
        name="Current",
        family="Current",
        binding=binding,
        metrics=current_metrics,
    )
    _, v3_payload = _record_and_payload(
        name="PBDR-V3-anchor",
        family="V3-calibrated",
        binding=binding,
        metrics=v3_metrics,
    )
    current_payload.update(
        {
            "grid_index": grid.current_index,
            "config": calibration.CURRENT_ANCHOR.as_dict(),
            "cache_tensor": "current_logits",
        }
    )
    v3_payload.update(
        {
            "grid_index": grid.v3_index,
            "config": calibration.PBDR_V3_ANCHOR.as_dict(),
            "cache_tensor": "routed_logits",
        }
    )

    candidates: list[dict[str, Any]] = []
    selected_index = 0
    selected_key: tuple[object, ...] | None = None
    for index, config in enumerate(grid.entries):
        metrics = _evaluate_logits(samples, config=config)
        _assert_same_evaluation_targets(
            sample_hash,
            target_hash,
            metrics,
            f"grid[{index}]",
        )
        record, payload = _record_and_payload(
            name=f"grid-{index:03d}-{config.name}",
            family="V3-calibrated",
            binding=binding,
            metrics=metrics,
        )
        key = selector.role_key(role, record)  # type: ignore[arg-type]
        if selected_key is None or key > selected_key:
            selected_index = index
            selected_key = key
        payload.update(
            {
                "grid_index": index,
                "config": config.as_dict(),
            }
        )
        candidates.append(payload)

    _require(len(candidates) == EXPECTED_GRID_SIZE, "candidate count differs")
    _require(
        candidates[grid.current_index]["metrics"] == current_metrics,
        "Current anchor metrics do not replay cached Current metrics",
    )
    _require(
        candidates[grid.v3_index]["metrics"] == v3_metrics,
        "PBDR-V3 anchor metrics do not replay cached routed metrics",
    )
    selected = candidates[selected_index]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_internal_validation_sweep",
        "dataset": identity["dataset"],
        "role": role,
        "selection_scope": "internal_validation_only",
        "probability_contract": {
            "logit_transform": "torch_sigmoid_float32",
            "comparison": "strict_greater_than",
            "threshold": {"numerator": 1, "denominator": 2},
            "loss": "torch_bce_on_float32_probability_mean_per_sample",
        },
        "selection_policy": {
            "comparison": "strict_lexicographic_full_role_key_no_positive_margin",
            "exact_tie_break": "earlier_grid_index",
            "pool": "all_378_grid_entries",
        },
        "cache_binding": cache_binding,
        "grid_binding": dict(grid.payload),
        "source_binding": {
            "metric_core_source_sha256": metric_core_sha,
            "selector_source_sha256": selector_sha,
            "source_lock_sha256": source_lock_sha,
        },
        "split_and_target_binding": {
            "split_projection_sha256": identity["split_projection_sha256"],
            "canonical_split_sha256": identity["canonical_split_sha256"],
            "cache_ordered_sample_ids_sha256": identity[
                "ordered_sample_ids_sha256"
            ],
            "metric_sample_id_order_sha256": sample_hash,
            "target_sha256": target_hash,
            "sample_count": len(samples),
        },
        "runtime_binding": {
            "cache_runtime": dict(runtime),
            "live_cuda_matmul_allow_tf32": False,
            "live_cudnn_allow_tf32": False,
        },
        "evaluation_binding": binding.as_dict(),
        "anchor_replay": anchor_replay,
        "baselines": {
            "original": original_payload,
            "current": current_payload,
            "v3_anchor": v3_payload,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": {
            "grid_index": selected_index,
            "name": selected["name"],
            "config": selected["config"],
            "role_key": selected["role_key"],
            "exact_sufficient_statistics": selected[
                "exact_sufficient_statistics"
            ],
        },
        "official_test_accessed": False,
    }
    result["result_sha256"] = cache_io.canonical_sha256(result)
    return result


def _validate_result_self_hash(result: Mapping[str, Any]) -> None:
    _require(isinstance(result, Mapping), "sweep result must be a mapping")
    declared = result.get("result_sha256")
    _require(
        type(declared) is str and len(declared) == 64,
        "sweep result SHA-256 is malformed",
    )
    unsigned = dict(result)
    del unsigned["result_sha256"]
    _require(
        cache_io.canonical_sha256(unsigned) == declared,
        "sweep result SHA-256 differs",
    )


def validate_sweep_result(
    result: Mapping[str, Any],
    cache: cache_io.ValidatedInternalRawLogitCache,
) -> dict[str, Any]:
    """Fully replay all 381 evaluations and compare every serialized field."""

    _validate_result_self_hash(result)
    expected = compute_sweep_result(cache)
    _require(
        cache_io.canonical_json_bytes(result)
        == cache_io.canonical_json_bytes(expected),
        "sweep result differs from a full cache-backed replay",
    )
    return dict(result)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_result_exclusive(path: Path, result: Mapping[str, Any]) -> Path:
    supplied = Path(path)
    supplied.parent.mkdir(parents=True, exist_ok=True)
    _require(not supplied.parent.is_symlink(), "result parent cannot be a symlink")
    parent = supplied.parent.resolve(strict=True)
    destination = parent / supplied.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"sweep result already exists: {destination}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(cache_io.canonical_json_bytes(result, trailing_newline=True))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(parent)
    return destination.resolve(strict=True)


def write_sweep_result_once(
    path: Path,
    *,
    result: Mapping[str, Any],
    cache: cache_io.ValidatedInternalRawLogitCache,
) -> Path:
    """Full-replay a result, then commit one immutable canonical JSON file."""

    supplied = Path(path)
    if supplied.exists() or supplied.is_symlink():
        raise FileExistsError(f"sweep result already exists: {supplied}")
    validated = validate_sweep_result(result, cache)
    return _write_result_exclusive(supplied, validated)


def run_sweep(
    *,
    cache_path: Path,
    split_projection: Mapping[str, Any],
    output_path: Path,
) -> Path:
    """Read a committed cache, sweep, fully replay, and commit once."""

    validated_cache = cache_io.read_cache(
        cache_path,
        split_projection=split_projection,
    )
    result = compute_sweep_result(validated_cache)
    return write_sweep_result_once(
        output_path,
        result=result,
        cache=validated_cache,
    )


def read_sweep_result(
    path: Path,
    *,
    cache_path: Path,
    split_projection: Mapping[str, Any],
) -> dict[str, Any]:
    """Read canonical JSON and fully revalidate it against the raw-logit cache."""

    result, raw = _read_json_object(Path(path), "sweep result")
    _require(
        raw == cache_io.canonical_json_bytes(result, trailing_newline=True),
        "sweep result file is not canonical JSON",
    )
    validated_cache = cache_io.read_cache(
        cache_path,
        split_projection=split_projection,
    )
    return validate_sweep_result(result, validated_cache)


def _load_projection(path: Path) -> dict[str, Any]:
    projection, _ = _read_json_object(path, "split projection")
    return projection


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split-projection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    destination = run_sweep(
        cache_path=arguments.cache,
        split_projection=_load_projection(arguments.split_projection),
        output_path=arguments.output,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_GRID_SIZE",
    "GRID_SCHEMA",
    "PBDRV3ResidualSweepError",
    "ROLE_KEY_FIELDS",
    "SCHEMA",
    "compute_sweep_result",
    "read_sweep_result",
    "run_sweep",
    "validate_sweep_result",
    "write_sweep_result_once",
]
