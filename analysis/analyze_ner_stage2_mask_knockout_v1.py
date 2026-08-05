#!/usr/bin/env python3
"""Evaluate a fixed-weight NER stage-2 mask knockout on one frozen run.

This is a deliberately thin diagnostic.  It reuses the frozen three-dataset
loader, model builder, inference collector, and metric implementation.  The
only graph intervention is a temporary ``MethodType`` wrapper around
``tpd_ner.forward_stage``: stages 4/3 are returned unchanged, while stage 2
returns a zero mask after recording the original V4 mask and persistent-tail
support ``P2``.

The historical reference metrics are read from the completed TSS-off
evaluation.  They are not replayed, and the checkpoint's stored test metrics
are intentionally not compared with the counterfactual output.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import sys
import tempfile
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from skimage import measure
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_three_dataset_tss_off_seed42_v1 as adapter  # noqa: E402
from experiments import evaluate_three_dataset_v2 as core  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import (  # noqa: E402
    train_four_dataset_original_final_seed42_exact_v1 as training_engine,
)
from model.tpd_ner_v8_mprs_dch_v2 import (  # noqa: E402
    spatially_center_gate_logits,
)


SCHEMA = "sctransnet_ner_stage2_mask_knockout_v1/v1"
INTERVENTION = "stage2_mask_off"
REFERENCE_METHOD = "final_tss_off"
TRAINING_MODEL_METHOD = "final"
SEED = 42
FIXED_THRESHOLD = 0.5
LOW_P2_THRESHOLD = 0.25
STAGE2_SCALE = 2

DEFAULT_TSS_OFF_ROOT = REPO_ROOT / "results" / "three_dataset_tss_off_seed42_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "ner_stage2_mask_knockout_v1"

HISTOGRAM_EDGES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
HISTOGRAM_LABELS = (
    "[0,0.1)",
    "[0.1,0.25)",
    "[0.25,0.5)",
    "[0.5,0.75)",
    "[0.75,0.9)",
    "[0.9,1]",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    """Hash state values canonically, independent of their current device."""

    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        value = state[key]
        _require(isinstance(value, torch.Tensor), f"non-tensor state: {key}")
        ready = value.detach().cpu().contiguous().reshape(-1)
        raw = ready.view(torch.uint8).numpy().tobytes()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(raw)
    return digest.hexdigest()


def _validate_historical_runtime_sources_allow_additions(
    run_protocol: Mapping[str, Any],
) -> dict[str, str]:
    """Validate every frozen dependency without rejecting new unrelated files.

    The old core validator compares the complete current ``model/*.py`` set to
    the historical set.  That is appropriate for an untouched runtime tree,
    but it makes a read-only historical diagnostic fail merely because a new
    V5 source file was added.  Here every source named by the historical run is
    still path- and byte-locked; only additional current files are ignored.
    """

    frozen = run_protocol.get("runtime_sources")
    _require(isinstance(frozen, Mapping), "run protocol lacks runtime_sources")
    architecture_keys = {
        key
        for key in frozen
        if isinstance(key, str) and key.startswith("architecture::")
    }
    _require(bool(architecture_keys), "historical architecture source set is empty")
    non_architecture_keys = set(frozen) - architecture_keys
    _require(
        {"model_builder", "training_metrics_and_schedule"}.issubset(
            non_architecture_keys
        ),
        "historical runtime sources lack core model/metric dependencies",
    )

    verified: dict[str, str] = {}
    for key in sorted(frozen):
        _require(isinstance(key, str) and bool(key), "malformed runtime source key")
        entry = frozen[key]
        _require(isinstance(entry, Mapping), f"malformed runtime source: {key}")
        frozen_path = entry.get("path")
        frozen_sha = entry.get("sha256")
        _require(
            isinstance(frozen_path, str) and bool(frozen_path),
            f"malformed runtime source path: {key}",
        )
        _require(
            isinstance(frozen_sha, str) and len(frozen_sha) == 64,
            f"malformed runtime source SHA: {key}",
        )
        frozen_path_object = Path(frozen_path)
        _require(
            frozen_path_object.is_absolute(),
            f"historical runtime source path is not absolute: {key}",
        )
        _require(
            not frozen_path_object.is_symlink(),
            f"historical runtime source path is a symlink: {key}",
        )
        observed_path = frozen_path_object.resolve(strict=True)
        _require(
            observed_path.is_relative_to(REPO_ROOT.resolve()),
            f"historical runtime source path escapes repository: {key}",
        )
        if key in architecture_keys:
            relative = key.removeprefix("architecture::")
            _require(
                relative.startswith("model/") and relative.endswith(".py"),
                f"malformed historical architecture key: {key}",
            )
            expected_path = (REPO_ROOT / relative).resolve()
            _require(
                expected_path.is_relative_to((REPO_ROOT / "model").resolve()),
                f"historical architecture path escapes model/: {key}",
            )
            _require(
                observed_path == expected_path,
                f"runtime source path differs from architecture key: {key}",
            )
        # For non-architecture entries the immutable, protocol-hash-bound
        # absolute path is itself the historical identity.  Validate every
        # such recorded file rather than assuming only two legacy keys exist.
        observed_sha = file_sha256(observed_path)
        _require(observed_sha == frozen_sha, f"runtime source SHA differs: {key}")
        verified[str(key)] = observed_sha
    return verified


@contextlib.contextmanager
def historical_checkpoint_loader_compatibility() -> Iterator[None]:
    """Let ``core.load_checkpoint`` admit added files, then restore the core."""

    original = core._validate_training_runtime_sources
    core._validate_training_runtime_sources = (  # type: ignore[attr-defined]
        _validate_historical_runtime_sources_allow_additions
    )
    try:
        yield
    finally:
        core._validate_training_runtime_sources = original  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Stage2Observation:
    original_mask: np.ndarray
    persistent_support_p2: np.ndarray
    centered_local_logits: np.ndarray
    output_size: tuple[int, int]


class Stage2KnockoutRecorder:
    """In-memory hook cache; no full probability or feature map is persisted."""

    def __init__(self) -> None:
        self.stage_call_counts: Counter[int] = Counter()
        self.observations: list[Stage2Observation] = []
        self.returned_stage2_mask_abs_max = 0.0

    def observe_call(self, stage: int) -> None:
        self.stage_call_counts[int(stage)] += 1

    def record_stage2(
        self,
        original_mask: torch.Tensor,
        p2: torch.Tensor,
        centered_local_logits: torch.Tensor,
        output_size: Sequence[int],
    ) -> None:
        _require(original_mask.ndim == 4, "stage2 mask must be BCHW")
        _require(p2.shape == original_mask.shape, "stage2 P2/mask shape differs")
        _require(
            centered_local_logits.shape == original_mask.shape,
            "stage2 centered-local-logit/mask shape differs",
        )
        _require(
            int(original_mask.shape[0]) == 1 and int(original_mask.shape[1]) == 1,
            "stage2 diagnostic requires batch=1 and one mask channel",
        )
        _require(bool(torch.isfinite(original_mask).all()), "non-finite stage2 mask")
        _require(bool(torch.isfinite(p2).all()), "non-finite stage2 P2")
        _require(
            bool(torch.isfinite(centered_local_logits).all()),
            "non-finite stage2 centered local logits",
        )
        self.observations.append(
            Stage2Observation(
                original_mask=np.array(
                    original_mask[0, 0].detach().float().cpu().numpy(), copy=True
                ),
                persistent_support_p2=np.array(
                    p2[0, 0].detach().float().cpu().numpy(), copy=True
                ),
                centered_local_logits=np.array(
                    centered_local_logits[0, 0]
                    .detach()
                    .float()
                    .cpu()
                    .numpy(),
                    copy=True,
                ),
                output_size=(int(output_size[0]), int(output_size[1])),
            )
        )


@contextlib.contextmanager
def temporary_stage2_mask_knockout(
    relay: nn.Module,
    recorder: Stage2KnockoutRecorder,
) -> Iterator[None]:
    """Temporarily zero only the stage-2 returned mask using ``MethodType``."""

    _require(hasattr(relay, "forward_stage"), "relay lacks forward_stage")
    _require(hasattr(relay, "dc_support"), "relay lacks dc_support")
    had_instance_override = "forward_stage" in relay.__dict__
    prior_instance_value = relay.__dict__.get("forward_stage")
    original_forward_stage = relay.forward_stage

    def knocked_forward_stage(
        self: nn.Module,
        stage: int,
        sources: Sequence[torch.Tensor],
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value, mask = original_forward_stage(stage, sources, output_size)
        recorder.observe_call(stage)
        if int(stage) != 2:
            return value, mask

        mode = getattr(self, "dc_support_mode", None)
        mode = getattr(mode, "value", mode)
        _require(
            mode == "complement_tail",
            f"stage2 P2 reconstruction requires complement_tail, got {mode!r}",
        )
        background_support = self.dc_support(
            2, value, sources, output_size
        ).detach()
        p2 = torch.clamp(
            background_support.new_tensor(1.0) - background_support,
            min=0.0,
            max=1.0,
        )
        gates = getattr(self, "gates", None)
        _require(gates is not None and "2" in gates, "relay lacks stage2 gate")
        centered_local_logits = spatially_center_gate_logits(gates["2"](value))
        recorder.record_stage2(
            mask,
            p2,
            centered_local_logits,
            output_size,
        )
        zero_mask = torch.zeros_like(mask)
        recorder.returned_stage2_mask_abs_max = max(
            recorder.returned_stage2_mask_abs_max,
            float(zero_mask.detach().abs().max().cpu()),
        )
        return value, zero_mask

    relay.forward_stage = types.MethodType(  # type: ignore[method-assign]
        knocked_forward_stage,
        relay,
    )
    try:
        yield
    finally:
        if had_instance_override:
            relay.forward_stage = prior_instance_value  # type: ignore[method-assign]
        elif "forward_stage" in relay.__dict__:
            delattr(relay, "forward_stage")


def _match_regions(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    match_radius: float = core.MATCH_RADIUS,
) -> tuple[list[Any], list[Any], set[int], set[int]]:
    """Mirror ``ValidationMetrics`` one-to-one component matching exactly."""

    predicted_regions = measure.regionprops(
        measure.label(prediction, connectivity=2)
    )
    target_regions = measure.regionprops(measure.label(target, connectivity=2))
    matched_targets: set[int] = set()
    matched_predictions: set[int] = set()
    if target_regions and predicted_regions:
        distances = np.empty(
            (len(target_regions), len(predicted_regions)), dtype=np.float64
        )
        for target_index, target_region in enumerate(target_regions):
            target_centroid = np.asarray(target_region.centroid)
            for prediction_index, prediction_region in enumerate(predicted_regions):
                distances[target_index, prediction_index] = np.linalg.norm(
                    np.asarray(prediction_region.centroid) - target_centroid
                )
        cardinality_reward = (
            (min(len(target_regions), len(predicted_regions)) + 1)
            * max(1.0, float(match_radius))
        )
        real_cost = np.where(
            distances < match_radius,
            distances - cardinality_reward,
            cardinality_reward,
        )
        assignment_cost = np.concatenate(
            (real_cost, np.zeros((len(target_regions), len(target_regions)))),
            axis=1,
        )
        assigned_targets, assigned_columns = linear_sum_assignment(assignment_cost)
        for target_index, column_index in zip(assigned_targets, assigned_columns):
            if (
                column_index < len(predicted_regions)
                and distances[target_index, column_index] < match_radius
            ):
                matched_targets.add(int(target_index))
                matched_predictions.add(int(column_index))
    return predicted_regions, target_regions, matched_targets, matched_predictions


def unmatched_false_component_mask(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float = FIXED_THRESHOLD,
) -> tuple[np.ndarray, dict[str, int]]:
    prediction = np.asarray(probability) > threshold
    target_binary = np.asarray(target) > 0.5
    predicted, targets, matched_targets, matched_predictions = _match_regions(
        prediction, target_binary
    )
    unmatched = np.zeros_like(prediction, dtype=bool)
    for index, region in enumerate(predicted):
        if index not in matched_predictions:
            coordinates = region.coords
            unmatched[coordinates[:, 0], coordinates[:, 1]] = True
    return unmatched, {
        "predicted_object_count": len(predicted),
        "target_count": len(targets),
        "matched_target_count": len(matched_targets),
        "unmatched_predicted_object_count": len(predicted) - len(matched_predictions),
        "unmatched_predicted_pixels": int(unmatched.sum()),
    }


def pixel_confusion(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    *,
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, int | float]:
    _require(len(probabilities) == len(targets), "probability/target count differs")
    tp = fp = fn = tn = 0
    for probability, target in zip(probabilities, targets):
        prediction = np.asarray(probability) > threshold
        target_binary = np.asarray(target) > 0.5
        _require(prediction.shape == target_binary.shape, "pixel arrays differ")
        tp += int(np.logical_and(prediction, target_binary).sum())
        fp += int(np.logical_and(prediction, ~target_binary).sum())
        fn += int(np.logical_and(~prediction, target_binary).sum())
        tn += int(np.logical_and(~prediction, ~target_binary).sum())
    valid = tp + fp + fn + tn
    return {
        "true_positive_pixels": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "true_negative_pixels": tn,
        "target_positive_pixels": tp + fn,
        "target_background_pixels": fp + tn,
        "valid_pixel_count": valid,
        "false_positive_per_valid_pixel": fp / max(1, valid),
        "false_positive_per_background_pixel": fp / max(1, fp + tn),
    }


def recover_reference_pixel_confusion(
    fixed_point: Mapping[str, Any],
    target_positive_pixels: int,
    valid_pixel_count: int,
) -> dict[str, Any]:
    """Recover integer reference FP from exact aggregate precision/recall."""

    precision = float(fixed_point["pixel_precision"])
    recall = float(fixed_point["pixel_recall"])
    _require(target_positive_pixels >= 0, "negative target-positive count")
    true_positive_float = recall * target_positive_pixels
    true_positive = int(round(true_positive_float))
    _require(
        math.isclose(true_positive_float, true_positive, abs_tol=1e-6),
        "reference recall does not map to an integer TP count",
    )
    false_negative = target_positive_pixels - true_positive
    if precision == 0.0:
        _require(true_positive == 0, "zero precision with positive TP")
        return {
            "recoverable": False,
            "reason": "precision_and_true_positive_are_zero",
            "target_positive_pixels": target_positive_pixels,
            "valid_pixel_count": valid_pixel_count,
        }
    predicted_positive_float = true_positive / precision
    predicted_positive = int(round(predicted_positive_float))
    _require(
        math.isclose(predicted_positive_float, predicted_positive, abs_tol=1e-5),
        "reference precision does not map to an integer prediction count",
    )
    false_positive = predicted_positive - true_positive
    true_negative = valid_pixel_count - true_positive - false_positive - false_negative
    _require(min(false_positive, true_negative) >= 0, "invalid recovered confusion")
    return {
        "recoverable": True,
        "derivation": "integer counts recovered from frozen precision/recall and targets",
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "true_negative_pixels": true_negative,
        "target_positive_pixels": target_positive_pixels,
        "target_background_pixels": false_positive + true_negative,
        "valid_pixel_count": valid_pixel_count,
        "false_positive_per_valid_pixel": false_positive / max(1, valid_pixel_count),
        "false_positive_per_background_pixel": false_positive
        / max(1, false_positive + true_negative),
    }


def _full_mask_to_stage2_cells(
    full_mask: np.ndarray,
    stage2_shape: tuple[int, int],
) -> np.ndarray:
    """Map full-resolution membership to stage-2 cells by 2x2 any-pooling."""

    full_mask = np.asarray(full_mask, dtype=bool)
    target_h = stage2_shape[0] * STAGE2_SCALE
    target_w = stage2_shape[1] * STAGE2_SCALE
    _require(
        full_mask.shape[0] <= target_h and full_mask.shape[1] <= target_w,
        "stage2 observation is smaller than the valid full-resolution image",
    )
    padded = np.zeros((target_h, target_w), dtype=bool)
    padded[: full_mask.shape[0], : full_mask.shape[1]] = full_mask
    return padded.reshape(
        stage2_shape[0], STAGE2_SCALE, stage2_shape[1], STAGE2_SCALE
    ).any(axis=(1, 3))


def _new_region_accumulator() -> dict[str, Any]:
    return {
        "cell_count": 0,
        "image_count_with_cells": 0,
        "raw_centered_local_logit_mass": 0.0,
        "positive_centered_local_logit_mass": 0.0,
        "negative_centered_local_logit_mass": 0.0,
        "positive_centered_local_logit_active_cells": 0,
        "p2_mass": 0.0,
        "p2_min": None,
        "p2_max": None,
        "p2_histogram_counts": [0] * len(HISTOGRAM_LABELS),
    }


def _accumulate_region(
    accumulator: dict[str, Any],
    region: np.ndarray,
    centered_local_logits: np.ndarray,
    p2: np.ndarray,
) -> None:
    count = int(region.sum())
    if count == 0:
        return
    local_values = centered_local_logits[region].astype(np.float64, copy=False)
    p2_values = p2[region].astype(np.float64, copy=False)
    positive = np.maximum(local_values, 0.0)
    negative = np.maximum(-local_values, 0.0)
    accumulator["cell_count"] += count
    accumulator["image_count_with_cells"] += 1
    accumulator["raw_centered_local_logit_mass"] += float(local_values.sum())
    accumulator["positive_centered_local_logit_mass"] += float(positive.sum())
    accumulator["negative_centered_local_logit_mass"] += float(negative.sum())
    accumulator["positive_centered_local_logit_active_cells"] += int(
        (local_values > 0.0).sum()
    )
    accumulator["p2_mass"] += float(p2_values.sum())
    observed_min = float(p2_values.min())
    observed_max = float(p2_values.max())
    current_min = accumulator["p2_min"]
    current_max = accumulator["p2_max"]
    accumulator["p2_min"] = observed_min if current_min is None else min(current_min, observed_min)
    accumulator["p2_max"] = observed_max if current_max is None else max(current_max, observed_max)
    histogram, _ = np.histogram(p2_values, bins=np.asarray(HISTOGRAM_EDGES))
    for index, value in enumerate(histogram.tolist()):
        accumulator["p2_histogram_counts"][index] += int(value)


def _finish_region(accumulator: Mapping[str, Any]) -> dict[str, Any]:
    count = int(accumulator["cell_count"])
    positive_mass = float(accumulator["positive_centered_local_logit_mass"])
    return {
        **dict(accumulator),
        "empty": count == 0,
        "raw_centered_local_logit_mean": float(
            accumulator["raw_centered_local_logit_mass"]
        )
        / count
        if count
        else None,
        "positive_centered_local_logit_mass_per_cell": positive_mass / count
        if count
        else None,
        "negative_centered_local_logit_mass_per_cell": float(
            accumulator["negative_centered_local_logit_mass"]
        )
        / count
        if count
        else None,
        "positive_centered_local_logit_active_fraction": int(
            accumulator["positive_centered_local_logit_active_cells"]
        )
        / count
        if count
        else None,
        "p2_mean": float(accumulator["p2_mass"]) / count if count else None,
        "p2_histogram": {
            label: int(value)
            for label, value in zip(
                HISTOGRAM_LABELS, accumulator["p2_histogram_counts"]
            )
        },
    }


def _correlation(x_values: Sequence[float], y_values: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    _require(x.shape == y.shape, "correlation vectors differ")
    if x.size < 2:
        return {"available": False, "reason": "fewer_than_two_images", "count": int(x.size)}
    if float(np.ptp(x)) == 0.0 or float(np.ptp(y)) == 0.0:
        return {"available": False, "reason": "zero_variance", "count": int(x.size)}
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    _require(math.isfinite(pearson) and math.isfinite(spearman), "non-finite correlation")
    return {
        "available": True,
        "count": int(x.size),
        "pearson": pearson,
        "spearman": spearman,
    }


def analyze_stage2_observations(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    observations: Sequence[Stage2Observation],
) -> dict[str, Any]:
    _require(
        len(probabilities) == len(targets) == len(observations),
        "prediction/target/stage2 observation count differs",
    )
    region_names = (
        "gt_target_cells",
        "knockout_unmatched_false_component_cells",
        "knockout_normal_background_cells",
        "all_background_cells",
        "low_p2_background_cells",
    )
    accumulators = {name: _new_region_accumulator() for name in region_names}
    background_positive_mass_per_cell_by_image: list[float] = []
    pixel_fp_per_background_pixel_by_image: list[float] = []
    component_fa_by_image: list[float] = []
    observed_mask_abs_max = 0.0
    final_mask_cell_count = 0
    final_mask_raw_mass = 0.0
    final_mask_positive_mass = 0.0
    final_mask_negative_mass = 0.0

    for probability, target, observation in zip(probabilities, targets, observations):
        probability = np.asarray(probability)
        target = np.asarray(target)
        _require(probability.shape == target.shape, "probability/target shape differs")
        full_prediction = probability > FIXED_THRESHOLD
        full_target = target > 0.5
        false_component, object_counts = unmatched_false_component_mask(
            probability, target
        )

        valid_stage_h = (full_target.shape[0] + STAGE2_SCALE - 1) // STAGE2_SCALE
        valid_stage_w = (full_target.shape[1] + STAGE2_SCALE - 1) // STAGE2_SCALE
        _require(
            observation.original_mask.shape
            == observation.persistent_support_p2.shape
            == observation.centered_local_logits.shape,
            "recorded mask/P2/centered-local-logit shape differs",
        )
        _require(
            observation.output_size == observation.original_mask.shape,
            "recorded stage2 output_size differs from tensor shape",
        )
        _require(
            observation.original_mask.shape[0] >= valid_stage_h
            and observation.original_mask.shape[1] >= valid_stage_w,
            "recorded stage2 tensor does not cover valid image",
        )
        mask = observation.original_mask[:valid_stage_h, :valid_stage_w]
        p2 = observation.persistent_support_p2[:valid_stage_h, :valid_stage_w]
        centered_local = observation.centered_local_logits[
            :valid_stage_h, :valid_stage_w
        ]
        _require(
            bool(np.isfinite(mask).all())
            and bool(np.isfinite(p2).all())
            and bool(np.isfinite(centered_local).all()),
            "non-finite recorded stage2 values",
        )
        _require(
            bool((p2 >= 0.0).all()) and bool((p2 <= 1.0).all()),
            "recorded P2 is outside [0,1]",
        )
        observed_mask_abs_max = max(observed_mask_abs_max, float(np.abs(mask).max()))
        final_mask_cell_count += int(mask.size)
        final_mask_raw_mass += float(mask.astype(np.float64, copy=False).sum())
        final_mask_positive_mass += float(
            np.maximum(mask.astype(np.float64, copy=False), 0.0).sum()
        )
        final_mask_negative_mass += float(
            np.maximum(-mask.astype(np.float64, copy=False), 0.0).sum()
        )

        stage_shape = (valid_stage_h, valid_stage_w)
        target_cells = _full_mask_to_stage2_cells(full_target, stage_shape)
        prediction_cells = _full_mask_to_stage2_cells(full_prediction, stage_shape)
        false_component_cells = _full_mask_to_stage2_cells(false_component, stage_shape)
        # "Normal background" is frozen as a strict true-negative cell: no GT
        # and no threshold-0.5 predicted foreground in its 2x2 footprint.
        normal_background_cells = ~(target_cells | prediction_cells)
        all_background_cells = ~target_cells
        low_p2_background_cells = all_background_cells & (p2 <= LOW_P2_THRESHOLD)
        regions = {
            "gt_target_cells": target_cells,
            "knockout_unmatched_false_component_cells": false_component_cells,
            "knockout_normal_background_cells": normal_background_cells,
            "all_background_cells": all_background_cells,
            "low_p2_background_cells": low_p2_background_cells,
        }
        for name, region in regions.items():
            _accumulate_region(
                accumulators[name], region, centered_local, p2
            )

        background_count = int(all_background_cells.sum())
        background_mass = float(
            np.maximum(centered_local[all_background_cells], 0.0).sum()
        )
        background_positive_mass_per_cell_by_image.append(
            background_mass / max(1, background_count)
        )
        pixel_fp = int(np.logical_and(full_prediction, ~full_target).sum())
        background_pixels = int((~full_target).sum())
        pixel_fp_per_background_pixel_by_image.append(pixel_fp / max(1, background_pixels))
        component_fa_by_image.append(
            int(object_counts["unmatched_predicted_pixels"]) / max(1, full_target.size)
        )

    regions = {name: _finish_region(value) for name, value in accumulators.items()}
    false_region = regions["knockout_unmatched_false_component_cells"]
    normal_region = regions["knockout_normal_background_cells"]
    background_region = regions["all_background_cells"]
    low_p2_region = regions["low_p2_background_cells"]

    false_density = false_region[
        "positive_centered_local_logit_mass_per_cell"
    ]
    normal_density = normal_region[
        "positive_centered_local_logit_mass_per_cell"
    ]
    b_ratio = None
    if false_density is not None and normal_density not in (None, 0.0):
        b_ratio = float(false_density) / float(normal_density)
    background_mass = float(
        background_region["positive_centered_local_logit_mass"]
    )
    low_p2_mass = float(
        low_p2_region["positive_centered_local_logit_mass"]
    )
    c_share = low_p2_mass / background_mass if background_mass > 0.0 else None

    return {
        "stage2_cell_mapping": {
            "scale_to_full_resolution": STAGE2_SCALE,
            "membership_pooling": "2x2_any",
            "padding": "bottom_right_cells_outside_original_image_excluded",
            "normal_background_definition": (
                "no_gt_and_no_threshold_0_5_prediction_in_cell"
            ),
        },
        "p2_definition": "1-dc_support_under_complement_tail",
        "local_signal_definition": (
            "spatially_center_gate_logits(tpd_ner.gates['2'](relay_value))"
        ),
        "positive_local_signal_definition": "relu(centered_local_logits)",
        "low_p2_comparison": f"P2 <= {LOW_P2_THRESHOLD}",
        "histogram_edges": list(HISTOGRAM_EDGES),
        "regions": regions,
        "original_final_mask_descriptive": {
            "cell_count": final_mask_cell_count,
            "raw_mass": final_mask_raw_mass,
            "positive_mass": final_mask_positive_mass,
            "negative_mass": final_mask_negative_mass,
            "abs_max": observed_mask_abs_max,
            "excluded_from_gate_b_and_c": True,
            "reason": "final_arctangent_mask_includes_dc_offset",
        },
        "gate_b_raw": {
            "available": False,
            "reason": "reference_probability_cache_absent",
            "required_semantics": (
                "reference_V4_false_components_aligned_with_reference_V4_local_logits"
            ),
            "knockout_region_descriptive_only": {
                "numerator": (
                    "knockout_false_component_positive_centered_local_logit_mass_per_cell"
                ),
                "denominator": (
                    "knockout_normal_background_positive_centered_local_logit_mass_per_cell"
                ),
                "false_component_density": false_density,
                "normal_background_density": normal_density,
                "density_ratio": b_ratio,
                "denominator_is_zero": normal_density == 0.0,
            },
        },
        "gate_c_raw": {
            "numerator": (
                "low_p2_background_positive_centered_local_logit_mass"
            ),
            "denominator": (
                "all_background_positive_centered_local_logit_mass"
            ),
            "low_p2_threshold": LOW_P2_THRESHOLD,
            "low_p2_background_positive_centered_local_logit_mass": low_p2_mass,
            "all_background_positive_centered_local_logit_mass": background_mass,
            "mass_share": c_share,
            "denominator_is_zero": background_mass == 0.0,
            "available": c_share is not None,
        },
        "correlations_across_images": {
            "x": (
                "stage2_background_positive_centered_local_logit_mass_per_cell"
            ),
            "region_semantics": "knockout_predictions_descriptive_only",
            "with_pixel_fp_per_background_pixel": _correlation(
                background_positive_mass_per_cell_by_image,
                pixel_fp_per_background_pixel_by_image,
            ),
            "with_component_fa": _correlation(
                background_positive_mass_per_cell_by_image,
                component_fa_by_image,
            ),
        },
        "full_probability_arrays_persisted": False,
        "full_stage2_arrays_persisted": False,
    }


def _relative_reduction(reference: int | float, candidate: int | float) -> float | None:
    reference = float(reference)
    candidate = float(candidate)
    if reference == 0.0:
        return None
    return (reference - candidate) / reference


def build_gate_a_raw(
    reference_fixed: Mapping[str, Any],
    knockout_fixed: Mapping[str, Any],
    reference_pixel: Mapping[str, Any],
    knockout_pixel: Mapping[str, Any],
) -> dict[str, Any]:
    reference_component = int(reference_fixed["unmatched_predicted_pixels"])
    knockout_component = int(knockout_fixed["unmatched_predicted_pixels"])
    component_reduction = _relative_reduction(reference_component, knockout_component)
    pixel_reduction = None
    if reference_pixel.get("recoverable") is not False:
        pixel_reduction = _relative_reduction(
            int(reference_pixel["false_positive_pixels"]),
            int(knockout_pixel["false_positive_pixels"]),
        )
    return {
        "reference_component_false_positive_pixels": reference_component,
        "knockout_component_false_positive_pixels": knockout_component,
        "component_fa_relative_reduction": component_reduction,
        "reference_all_background_pixel_fp": reference_pixel.get(
            "false_positive_pixels"
        ),
        "knockout_all_background_pixel_fp": knockout_pixel[
            "false_positive_pixels"
        ],
        "all_background_pixel_fp_relative_reduction": pixel_reduction,
        "matched_target_drop": int(reference_fixed["matched_target_count"])
        - int(knockout_fixed["matched_target_count"]),
        "matched_tiny_target_drop": int(
            reference_fixed["matched_tiny_target_count"]
        )
        - int(knockout_fixed["matched_tiny_target_count"]),
        "miou_drop": float(reference_fixed["miou"])
        - float(knockout_fixed["miou"]),
        "niou_drop": float(reference_fixed["niou"])
        - float(knockout_fixed["niou"]),
    }


def _load_checkpoint_allowing_added_sources(
    request: core.EvaluationRequest,
    run_dir: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    with historical_checkpoint_loader_compatibility():
        return core.load_checkpoint(
            request,
            run_dir,
            manifest_path=manifest_path,
            manifest=manifest,
        )


def analyze_run(
    *,
    dataset: str,
    checkpoint_role: str,
    run_dir: Path,
    dataset_root: Path,
    data_protocol_manifest: Path,
    reference_evaluation: Path,
    device_name: str,
    workers: int,
) -> dict[str, Any]:
    adapter.configure_core()
    training_engine.configure_determinism()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    manifest_path = Path(data_protocol_manifest).resolve(strict=True)
    manifest = data_protocol.load_protocol_manifest(
        manifest_path, dataset_root=dataset_root
    )
    request = core.EvaluationRequest(
        dataset=dataset,
        method=TRAINING_MODEL_METHOD,
        checkpoint_role=checkpoint_role,
        requested_tss_weight=adapter.REQUESTED_TSS_WEIGHT,
    )
    request.validate()
    checkpoint_payload, checkpoint_binding = _load_checkpoint_allowing_added_sources(
        request, Path(run_dir), manifest_path, manifest
    )
    reference = adapter.validate_completed_output(
        Path(reference_evaluation),
        dataset=dataset,
        checkpoint_role=checkpoint_role,
    )
    _require(
        reference["checkpoint_binding"]["checkpoint"]["sha256"]
        == checkpoint_binding["checkpoint"]["sha256"],
        "reference evaluation/checkpoint SHA differs",
    )

    model, model_metadata = core.build_inference_model(
        request, checkpoint_payload["state_dict"]
    )
    _require(hasattr(model, "tpd_ner"), "inference model lacks tpd_ner")
    model.to(device)
    model.eval()
    state_before = module_state_sha256(model)
    recorder = Stage2KnockoutRecorder()

    dataset_object = core.ThreeDatasetTestDataset(
        dataset_root, dataset, manifest_path
    )
    loader = DataLoader(
        dataset_object,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    with temporary_stage2_mask_knockout(model.tpd_ner, recorder):
        probabilities, targets, losses, identifiers = core.collect_predictions(
            model, loader, device
        )
    state_after = module_state_sha256(model)
    _require(state_before == state_after, "stage2 intervention changed model state")
    _require(
        identifiers == list(dataset_object.sample_ids),
        "inference order differs from frozen img_idx/test order",
    )
    expected_count = len(identifiers)
    _require(
        len(recorder.observations) == expected_count,
        "stage2 observation count differs from test count",
    )
    for stage in (4, 3, 2):
        _require(
            recorder.stage_call_counts[stage] == expected_count,
            f"stage {stage} call count differs from test count",
        )
    _require(
        set(recorder.stage_call_counts) == {2, 3, 4},
        "unexpected relay stage observed",
    )
    _require(
        recorder.returned_stage2_mask_abs_max == 0.0,
        "returned stage2 knockout mask was not exactly zero",
    )

    evaluated = core.evaluate_probability_arrays(probabilities, targets, losses)
    knockout_fixed = evaluated["fixed_threshold_0_5"]
    knockout_pixel = pixel_confusion(probabilities, targets)
    _require(
        knockout_pixel["valid_pixel_count"] == knockout_fixed["valid_pixel_count"],
        "pixel-confusion valid count differs from frozen metric",
    )
    mechanism = analyze_stage2_observations(
        probabilities, targets, recorder.observations
    )
    reference_fixed = reference["fixed_threshold_0_5"]
    reference_pixel = recover_reference_pixel_confusion(
        reference_fixed,
        int(knockout_pixel["target_positive_pixels"]),
        int(knockout_pixel["valid_pixel_count"]),
    )
    gate_a_raw = build_gate_a_raw(
        reference_fixed, knockout_fixed, reference_pixel, knockout_pixel
    )

    ordered_id_sha = hashlib.sha256(
        ("\n".join(identifiers) + "\n").encode("utf-8")
    ).hexdigest()
    output = {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": dataset,
        "method": REFERENCE_METHOD,
        "training_model_method": TRAINING_MODEL_METHOD,
        "intervention": INTERVENTION,
        "checkpoint_role": checkpoint_role,
        "seed": SEED,
        "test_selected": True,
        "selection_is_optimistic": True,
        **evaluated,
        "additive_pixel_confusion": {
            "reference": reference_pixel,
            "knockout": knockout_pixel,
        },
        "mechanism_statistics": mechanism,
        "gate_inputs": {
            "A": gate_a_raw,
            "B": mechanism["gate_b_raw"],
            "C": mechanism["gate_c_raw"],
            "decision_deferred_to": (
                "analysis/compare_ner_stage2_mask_knockout_v1.py"
            ),
        },
        "reference_reuse": {
            "path": str(Path(reference_evaluation).resolve(strict=True)),
            "sha256": file_sha256(Path(reference_evaluation).resolve(strict=True)),
            "fixed_threshold_0_5": reference_fixed,
            "new_reference_inference_performed": False,
        },
        "intervention_audit": {
            "implementation": "temporary_MethodType_forward_stage_wrapper",
            "stage4_return_modified": False,
            "stage3_return_modified": False,
            "stage2_relay_value_modified": False,
            "stage2_original_mask_recorded_before_replacement": True,
            "stage2_centered_local_logits_recomputed_and_recorded": True,
            "stage2_gate_statistics_exclude_dc_offset": True,
            "stage2_returned_mask": "zeros_like(original_mask)",
            "returned_stage2_mask_abs_max": recorder.returned_stage2_mask_abs_max,
            "stage_call_counts": {
                str(stage): int(recorder.stage_call_counts[stage])
                for stage in (4, 3, 2)
            },
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_state_unchanged": state_before == state_after,
            "checkpoint_metric_replay_skipped": True,
            "checkpoint_metric_replay_skip_reason": (
                "counterfactual output must differ from stored reference metrics"
            ),
            "checkpoint_written": False,
        },
        "checkpoint_binding": checkpoint_binding,
        "model": model_metadata,
        "data": {
            "dataset_root": str(Path(dataset_root).resolve()),
            "protocol_manifest": {
                "path": str(manifest_path),
                "sha256": file_sha256(manifest_path),
                "schema": manifest.get("schema"),
                "manifest_id": manifest.get("manifest_id"),
            },
            "split": "img_idx/test",
            "test_count": expected_count,
            "inference_order_newline_sha256": ordered_id_sha,
            "normalization": core.NORMALIZATION[dataset],
            "sirst3_in_formal_matrix": False,
        },
        "metric_protocol": {
            "implementation": "experiments.train_tpd_pilot.ValidationMetrics",
            "threshold": FIXED_THRESHOLD,
            "prediction_comparison": "probability > threshold",
            "connectivity": 8,
            "matching": "one_to_one_max_cardinality_min_distance",
            "centroid_radius_comparison": "distance < 3",
            "match_radius": core.MATCH_RADIUS,
            "tiny_area": core.TINY_AREA,
        },
        "source_lock_policy": {
            "historical_frozen_dependencies_verified_by_path_and_sha256": True,
            "new_unlisted_model_sources_allowed": True,
            "reason": "new V5 files are not dependencies of the frozen V4 checkpoint",
            "verified_historical_runtime_sources": checkpoint_binding[
                "training_runtime_sources"
            ]["source_sha256"],
        },
        "source_sha256": {
            "analysis/analyze_ner_stage2_mask_knockout_v1.py": file_sha256(
                Path(__file__)
            ),
            "experiments/evaluate_three_dataset_v2.py": file_sha256(
                Path(core.__file__)
            ),
            "experiments/evaluate_three_dataset_tss_off_seed42_v1.py": file_sha256(
                Path(adapter.__file__)
            ),
        },
        "probability_cache_written": False,
        "stage2_tensor_cache_written": False,
        "no_fabricated_results": True,
        "stability_claim_supported": False,
    }

    del model, loader, probabilities, targets, losses
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def _default_run_dir(dataset: str) -> Path:
    return (
        DEFAULT_TSS_OFF_ROOT
        / "runs"
        / dataset
        / "final_tss_off"
        / "seed_42"
    )


def _default_reference(run_dir: Path, checkpoint_role: str) -> Path:
    return Path(run_dir) / "evaluations" / f"{checkpoint_role}.json"


def _default_output(dataset: str, checkpoint_role: str) -> Path:
    return (
        DEFAULT_OUTPUT_ROOT
        / "runs"
        / dataset
        / f"v4_tss_off_{checkpoint_role}_seed42"
        / INTERVENTION
        / "evaluation.json"
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=data_protocol.DATASETS, required=True)
    parser.add_argument(
        "--checkpoint-role", choices=core.CHECKPOINT_ROLES, default="best_miou"
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--reference-evaluation", type=Path)
    parser.add_argument(
        "--dataset-root", type=Path, default=data_protocol.DEFAULT_DATASET_ROOT
    )
    parser.add_argument(
        "--data-protocol-manifest",
        type=Path,
        default=data_protocol.DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if not (args.device == "cpu" or args.device.startswith("cuda:")):
        parser.error("--device must be cpu or cuda:N")
    args.run_dir = args.run_dir or _default_run_dir(args.dataset)
    args.reference_evaluation = args.reference_evaluation or _default_reference(
        args.run_dir, args.checkpoint_role
    )
    args.output = args.output or _default_output(args.dataset, args.checkpoint_role)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = analyze_run(
        dataset=args.dataset,
        checkpoint_role=args.checkpoint_role,
        run_dir=args.run_dir,
        dataset_root=args.dataset_root,
        data_protocol_manifest=args.data_protocol_manifest,
        reference_evaluation=args.reference_evaluation,
        device_name=args.device,
        workers=args.workers,
    )
    atomic_write_json(args.output, output, overwrite=args.overwrite)
    print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__":
    main()
