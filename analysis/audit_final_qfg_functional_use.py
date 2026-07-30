#!/usr/bin/env python3
"""Evaluation-only QFG alpha knockout and functional-audit primitives.

The context manager changes only the four scalar QFG alpha parameters in
memory and restores all four exactly, including when the caller raises.  This
module never writes a checkpoint and never selects a deployment model.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import functools
import hashlib
import math
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from analysis import collect_final_model_validation_statistics as cache_core


AUDIT_SCHEMA = "sctransnet_final_model_qfg_functional_audit_v1"
ALPHA_KNOCKOUT_SCHEMA = "sctransnet_final_model_qfg_alpha_knockout_v1"
FACTOR_SUMMARY_SCHEMA = "sctransnet_final_model_qfg_factor_summary_v1"
OUTPUT_EQUIVALENCE_MAX_ABS = 1e-7
OUTPUT_EQUIVALENCE_MEAN_ABS = 1e-8
NONTRIVIAL_FACTOR_MEAN_ABS = 1e-4
QFG_LEVEL_COUNT = 4


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolve_qfg(model_or_qfg: nn.Module) -> nn.Module:
    if not isinstance(model_or_qfg, nn.Module):
        raise TypeError("model_or_qfg must be a torch module")
    qfg = getattr(model_or_qfg, "tpd_qfg", model_or_qfg)
    if not isinstance(qfg, nn.Module):
        raise TypeError("model.tpd_qfg must be a torch module")
    levels = getattr(qfg, "levels", None)
    _require(
        isinstance(levels, (nn.ModuleList, list, tuple)),
        "QFG levels are missing",
    )
    _require(len(levels) == QFG_LEVEL_COUNT, "QFG must contain four levels")
    for index, level in enumerate(levels):
        alpha = getattr(level, "alpha", None)
        _require(
            isinstance(alpha, nn.Parameter),
            f"QFG level {index} alpha must be a parameter",
        )
        _require(
            alpha.numel() == 1 and bool(torch.isfinite(alpha).all().item()),
            f"QFG level {index} alpha must be one finite scalar",
        )
    return qfg


def _alpha_parameters(qfg: nn.Module) -> tuple[nn.Parameter, ...]:
    return tuple(level.alpha for level in qfg.levels)


def alpha_state_sha256(model_or_qfg: nn.Module) -> str:
    qfg = _resolve_qfg(model_or_qfg)
    digest = hashlib.sha256()
    for index, alpha in enumerate(_alpha_parameters(qfg)):
        tensor = alpha.detach().cpu().contiguous()
        digest.update(f"levels.{index}.alpha".encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(
            np.asarray(tuple(tensor.shape), dtype="<i8").tobytes()
        )
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def alpha_records(model_or_qfg: nn.Module) -> list[dict[str, Any]]:
    qfg = _resolve_qfg(model_or_qfg)
    return [
        {
            "level": index + 1,
            "parameter_value": float(alpha.detach().double().item()),
            "effective_value": float(
                torch.tanh(alpha.detach().double()).item()
            ),
            "dtype": str(alpha.dtype).removeprefix("torch."),
            "device": str(alpha.device),
            "requires_grad": bool(alpha.requires_grad),
        }
        for index, alpha in enumerate(_alpha_parameters(qfg))
    ]


@contextmanager
def temporary_qfg_alpha_knockout(
    model_or_qfg: nn.Module,
    mode: str,
) -> Iterator[dict[str, Any]]:
    """Temporarily zero the requested QFG alpha scalars and restore all four."""

    qfg = _resolve_qfg(model_or_qfg)
    _require(
        not model_or_qfg.training and not qfg.training,
        "QFG knockout audit requires model.eval()",
    )
    descriptor = cache_core.normalize_mode(mode)
    parameters = _alpha_parameters(qfg)
    snapshots = tuple(
        parameter.detach().clone(memory_format=torch.preserve_format)
        for parameter in parameters
    )
    before_sha256 = alpha_state_sha256(qfg)
    selected = tuple(
        int(value)
        for value in descriptor["knockout_level_indices_zero_based"]
    )
    with torch.no_grad():
        for index in selected:
            parameters[index].zero_()
    inside_sha256 = alpha_state_sha256(qfg)
    for index, (parameter, snapshot) in enumerate(
        zip(parameters, snapshots)
    ):
        if index in selected:
            _require(
                bool(torch.count_nonzero(parameter).item()) is False,
                f"QFG level {index} alpha was not zeroed",
            )
        else:
            _require(
                torch.equal(parameter.detach(), snapshot),
                f"QFG level {index} alpha changed unexpectedly",
            )
    audit = {
        "schema": ALPHA_KNOCKOUT_SCHEMA,
        "mode": descriptor,
        "source_alpha_sha256": before_sha256,
        "active_alpha_sha256": inside_sha256,
        "selected_level_indices_zero_based": list(selected),
        "derived_checkpoint_written": False,
        "diagnostic_only": mode != "full",
    }
    try:
        yield audit
    finally:
        with torch.no_grad():
            for parameter, snapshot in zip(parameters, snapshots):
                parameter.copy_(snapshot)
        after_sha256 = alpha_state_sha256(qfg)
        if after_sha256 != before_sha256:
            raise RuntimeError("QFG alpha parameters were not restored exactly")


def _factor_tensor(prepared_level: Any, index: int) -> torch.Tensor:
    factor = getattr(prepared_level, "factor", None)
    if not isinstance(factor, torch.Tensor):
        raise ValueError(f"prepared QFG level {index} has no factor tensor")
    if factor.numel() < 1 or not bool(torch.isfinite(factor).all().item()):
        raise ValueError(f"prepared QFG level {index} factor is invalid")
    return factor.detach().to(device="cpu", dtype=torch.float64).reshape(-1)


def _factor_record(values: torch.Tensor, level_index: int) -> dict[str, Any]:
    delta = torch.abs(values - 1.0)
    quantiles = torch.quantile(
        values,
        torch.tensor((0.05, 0.5, 0.95), dtype=torch.float64),
    )
    return {
        "level": level_index + 1,
        "element_count": values.numel(),
        "mean": float(values.mean().item()),
        "rms": float(torch.sqrt(torch.mean(values.square())).item()),
        "p5": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p95": float(quantiles[2].item()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
        "mean_abs_factor_minus_one": float(delta.mean().item()),
        "max_abs_factor_minus_one": float(delta.max().item()),
    }


def summarize_prepared_factors(prepared: Any) -> dict[str, Any]:
    levels = getattr(prepared, "levels", None)
    _require(
        isinstance(levels, (tuple, list))
        and len(levels) == QFG_LEVEL_COUNT,
        "prepared QFG object must contain four levels",
    )
    records = [
        _factor_record(_factor_tensor(level, index), index)
        for index, level in enumerate(levels)
    ]
    maximum = max(
        record["mean_abs_factor_minus_one"] for record in records
    )
    return {
        "schema": FACTOR_SUMMARY_SCHEMA,
        "forward_count": 1,
        "levels": records,
        "maximum_level_mean_abs_factor_minus_one": maximum,
        "nontrivial_factor_use": maximum > NONTRIVIAL_FACTOR_MEAN_ABS,
        "nontrivial_factor_threshold": NONTRIVIAL_FACTOR_MEAN_ABS,
    }


@dataclass(slots=True)
class PreparedFactorCapture:
    """Capture prepared factors without retaining accelerator tensors."""

    _level_values: list[list[torch.Tensor]] = field(
        default_factory=lambda: [[] for _ in range(QFG_LEVEL_COUNT)]
    )
    forward_count: int = 0

    def append(self, prepared: Any) -> None:
        levels = getattr(prepared, "levels", None)
        _require(
            isinstance(levels, (tuple, list))
            and len(levels) == QFG_LEVEL_COUNT,
            "prepared QFG object must contain four levels",
        )
        for index, level in enumerate(levels):
            self._level_values[index].append(_factor_tensor(level, index))
        self.forward_count += 1

    def summary(self) -> dict[str, Any]:
        _require(self.forward_count > 0, "no QFG prepared factors captured")
        records = [
            _factor_record(torch.cat(values), index)
            for index, values in enumerate(self._level_values)
        ]
        maximum = max(
            record["mean_abs_factor_minus_one"] for record in records
        )
        return {
            "schema": FACTOR_SUMMARY_SCHEMA,
            "forward_count": self.forward_count,
            "levels": records,
            "maximum_level_mean_abs_factor_minus_one": maximum,
            "nontrivial_factor_use": (
                maximum > NONTRIVIAL_FACTOR_MEAN_ABS
            ),
            "nontrivial_factor_threshold": NONTRIVIAL_FACTOR_MEAN_ABS,
        }


@contextmanager
def capture_qfg_prepared_factors(
    model_or_qfg: nn.Module,
) -> Iterator[PreparedFactorCapture]:
    """Wrap ``qfg.prepare`` for one scope and restore method lookup exactly."""

    qfg = _resolve_qfg(model_or_qfg)
    _require(
        not model_or_qfg.training and not qfg.training,
        "QFG factor audit requires model.eval()",
    )
    original = qfg.prepare
    capture = PreparedFactorCapture()
    had_instance_attribute = "prepare" in qfg.__dict__
    previous_instance_attribute = qfg.__dict__.get("prepare")

    @functools.wraps(original)
    def wrapped_prepare(*args: Any, **kwargs: Any) -> Any:
        prepared = original(*args, **kwargs)
        capture.append(prepared)
        return prepared

    object.__setattr__(qfg, "prepare", wrapped_prepare)
    try:
        yield capture
    finally:
        if had_instance_attribute:
            object.__setattr__(
                qfg,
                "prepare",
                previous_instance_attribute,
            )
        else:
            object.__delattr__(qfg, "prepare")


def _validate_cache_pair(
    full: cache_core.PredictionCache,
    counterfactual: cache_core.PredictionCache,
) -> None:
    _require(
        full.identity["mode"]["name"] == "full",
        "reference cache mode must be full",
    )
    _require(
        counterfactual.identity["mode"]["name"] != "full",
        "counterfactual cache mode must be a knockout",
    )
    _require(
        full.identity["compatibility_sha256"]
        == counterfactual.identity["compatibility_sha256"],
        "full/counterfactual cache compatibility differs",
    )
    _require(
        len(full.records) == len(counterfactual.records),
        "full/counterfactual image count differs",
    )
    for index, (left, right) in enumerate(
        zip(full.records, counterfactual.records)
    ):
        _require(left.image_id == right.image_id, f"image[{index}] ID differs")
        _require(
            left.probability.shape == right.probability.shape,
            f"image[{index}] shape differs",
        )
        _require(
            np.array_equal(left.target, right.target),
            f"image[{index}] target differs",
        )
    _require(
        full.match_radius == counterfactual.match_radius
        and full.tiny_area == counterfactual.tiny_area,
        "full/counterfactual evaluation contract differs",
    )


def audit_probability_caches(
    full: cache_core.PredictionCache,
    counterfactual: cache_core.PredictionCache,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare two compatible caches and recompute both fixed-point metrics."""

    _validate_cache_pair(full, counterfactual)
    per_image: list[dict[str, Any]] = []
    total_abs = 0.0
    total_count = 0
    maximum = 0.0
    for left, right in zip(full.records, counterfactual.records):
        difference = np.abs(
            left.probability.astype(np.float64)
            - right.probability.astype(np.float64)
        )
        image_max = float(difference.max())
        image_mean = float(difference.mean())
        maximum = max(maximum, image_max)
        total_abs += float(difference.sum())
        total_count += difference.size
        per_image.append(
            {
                "image_id": left.image_id,
                "max_abs": image_max,
                "mean_abs": image_mean,
            }
        )
    mean_abs = total_abs / total_count
    equivalent = (
        maximum <= OUTPUT_EQUIVALENCE_MAX_ABS
        and mean_abs <= OUTPUT_EQUIVALENCE_MEAN_ABS
    )
    return {
        "schema": AUDIT_SCHEMA,
        "status": "complete",
        "scope": "internal_validation_same_checkpoint_counterfactual",
        "official_test_accessed": False,
        "checkpoint_sha256": full.identity["checkpoint_sha256"],
        "dataset_sha256": full.identity["dataset_sha256"],
        "evaluator_sha256": full.identity["evaluator_sha256"],
        "full_cache_key_sha256": full.identity["cache_key_sha256"],
        "counterfactual_cache_key_sha256": (
            counterfactual.identity["cache_key_sha256"]
        ),
        "counterfactual_mode": counterfactual.identity["mode"],
        "output_difference": {
            "max_abs": maximum,
            "mean_abs": mean_abs,
            "equivalent": equivalent,
            "functionally_different": not equivalent,
            "equivalence_max_abs_threshold": (
                OUTPUT_EQUIVALENCE_MAX_ABS
            ),
            "equivalence_mean_abs_threshold": (
                OUTPUT_EQUIVALENCE_MEAN_ABS
            ),
            "per_image": per_image,
        },
        "fixed_threshold": float(threshold),
        "full_metrics": cache_core.recompute_metrics(
            full,
            threshold=threshold,
        ),
        "counterfactual_metrics": cache_core.recompute_metrics(
            counterfactual,
            threshold=threshold,
        ),
        "diagnostic_only": True,
        "derived_checkpoint_written": False,
    }


__all__ = [
    "ALPHA_KNOCKOUT_SCHEMA",
    "AUDIT_SCHEMA",
    "FACTOR_SUMMARY_SCHEMA",
    "NONTRIVIAL_FACTOR_MEAN_ABS",
    "OUTPUT_EQUIVALENCE_MAX_ABS",
    "OUTPUT_EQUIVALENCE_MEAN_ABS",
    "PreparedFactorCapture",
    "alpha_records",
    "alpha_state_sha256",
    "audit_probability_caches",
    "capture_qfg_prepared_factors",
    "summarize_prepared_factors",
    "temporary_qfg_alpha_knockout",
]
