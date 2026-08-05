#!/usr/bin/env python3
"""Run the frozen three-dataset QFG alpha-knockout diagnostic.

The diagnostic is evaluation-only.  For one seed-42 TSS-off ``best_miou``
checkpoint it evaluates ``full``, four zero-based single-level knockouts, and
``all_off`` on the frozen ``img_idx/test`` split.  Checkpoint selection and the
fixed operating threshold remain unchanged; the threshold sweep is descriptive.

No derived checkpoint and no full probability cache is written.  Probabilities
are retained only long enough to compute exact metrics and paired, unpadded
full-versus-counterfactual output differences.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import audit_final_qfg_functional_use as qfg_audit  # noqa: E402
from analysis import analyze_ner_stage2_mask_knockout_v1 as stage2_audit  # noqa: E402
from analysis import run_final_qfg_six_mode_audit as legacy_qfg_audit  # noqa: E402
from experiments import evaluate_three_dataset_tss_off_seed42_v1 as adapter  # noqa: E402
from experiments import evaluate_three_dataset_v2 as core  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import (  # noqa: E402
    train_four_dataset_original_final_seed42_exact_v1 as training_engine,
)


SCHEMA = "sctransnet_three_dataset_qfg_level_knockout_v1/v1"
REFERENCE_METHOD = "final_tss_off"
TRAINING_MODEL_METHOD = "final"
CHECKPOINT_ROLE = "best_miou"
SEED = 42
FIXED_THRESHOLD = 0.5
EVALUATION_PROTOCOL = "img_idx_test_selected_development"

PUBLIC_TO_PRIMITIVE_MODE = {
    "full": "full",
    "level0_off": "level_1_off",
    "level1_off": "level_2_off",
    "level2_off": "level_3_off",
    "level3_off": "level_4_off",
    "all_off": "all_off",
}
PUBLIC_MODES = tuple(PUBLIC_TO_PRIMITIVE_MODE)

OUTPUT_EQUIVALENCE_MAX_ABS = qfg_audit.OUTPUT_EQUIVALENCE_MAX_ABS
OUTPUT_EQUIVALENCE_MEAN_ABS = qfg_audit.OUTPUT_EQUIVALENCE_MEAN_ABS

DEFAULT_TSS_OFF_ROOT = REPO_ROOT / "results" / "three_dataset_tss_off_seed42_v1"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "three_dataset_qfg_level_knockout_v1"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    return stage2_audit.file_sha256(Path(path))


def normalize_public_mode(public_mode: str) -> dict[str, Any]:
    """Return an explicit zero-based/public to one-based/primitive binding."""

    _require(
        public_mode in PUBLIC_TO_PRIMITIVE_MODE,
        f"unknown QFG public mode: {public_mode!r}",
    )
    primitive = PUBLIC_TO_PRIMITIVE_MODE[public_mode]
    descriptor = qfg_audit.cache_core.normalize_mode(primitive)
    selected = list(descriptor["knockout_level_indices_zero_based"])
    expected = {
        "full": [],
        "level0_off": [0],
        "level1_off": [1],
        "level2_off": [2],
        "level3_off": [3],
        "all_off": [0, 1, 2, 3],
    }[public_mode]
    _require(selected == expected, "public/primitive QFG level mapping differs")
    return {
        "public_mode": public_mode,
        "primitive_mode": primitive,
        "knockout_level_indices_zero_based": selected,
        "diagnostic_only": public_mode != "full",
    }


@dataclass(slots=True)
class _LevelQuerySums:
    call_count: int = 0
    query_element_count: int = 0
    query_input_square_sum: float = 0.0
    query_delta_square_sum: float = 0.0
    query_delta_abs_max: float = 0.0
    factor_element_count: int = 0
    factor_delta_square_sum: float = 0.0
    factor_minimum: float = math.inf
    factor_maximum: float = -math.inf
    gate_element_count: int = 0
    gate_square_sum: float = 0.0


@dataclass(slots=True)
class QueryPerturbationRecorder:
    """Accumulate direct ``apply_prepared`` input/output Query differences."""

    levels: list[_LevelQuerySums] = field(
        default_factory=lambda: [_LevelQuerySums() for _ in range(4)]
    )
    wrapper_call_count: int = 0

    def append(
        self,
        input_queries: Sequence[torch.Tensor],
        output_queries: Sequence[torch.Tensor],
        factors: Sequence[torch.Tensor],
        gates: Sequence[torch.Tensor],
    ) -> None:
        _require(
            len(input_queries) == len(output_queries) == len(factors) == len(gates) == 4,
            "QFG apply_prepared audit requires four levels",
        )
        for index, (input_query, output_query, factor, gate) in enumerate(
            zip(input_queries, output_queries, factors, gates)
        ):
            _require(
                isinstance(input_query, torch.Tensor)
                and isinstance(output_query, torch.Tensor)
                and input_query.shape == output_query.shape,
                f"QFG level {index} Query input/output differs",
            )
            _require(
                isinstance(factor, torch.Tensor) and isinstance(gate, torch.Tensor),
                f"QFG level {index} factor/gate is missing",
            )
            input_ready = input_query.detach().float()
            output_ready = output_query.detach().float()
            delta = output_ready - input_ready
            factor_ready = factor.detach().float()
            gate_ready = gate.detach().float()
            _require(
                bool(torch.isfinite(delta).all())
                and bool(torch.isfinite(factor_ready).all())
                and bool(torch.isfinite(gate_ready).all()),
                f"QFG level {index} perturbation statistics are non-finite",
            )
            level = self.levels[index]
            level.call_count += 1
            level.query_element_count += int(delta.numel())
            level.query_input_square_sum += float(
                torch.sum(input_ready.square()).item()
            )
            level.query_delta_square_sum += float(torch.sum(delta.square()).item())
            level.query_delta_abs_max = max(
                level.query_delta_abs_max,
                float(torch.max(torch.abs(delta)).item()),
            )
            factor_delta = factor_ready - 1.0
            level.factor_element_count += int(factor_delta.numel())
            level.factor_delta_square_sum += float(
                torch.sum(factor_delta.square()).item()
            )
            level.factor_minimum = min(
                level.factor_minimum, float(torch.min(factor_ready).item())
            )
            level.factor_maximum = max(
                level.factor_maximum, float(torch.max(factor_ready).item())
            )
            level.gate_element_count += int(gate_ready.numel())
            level.gate_square_sum += float(torch.sum(gate_ready.square()).item())
        self.wrapper_call_count += 1

    def summary(self) -> dict[str, Any]:
        _require(self.wrapper_call_count > 0, "no QFG apply_prepared call captured")
        records: list[dict[str, Any]] = []
        for index, level in enumerate(self.levels):
            _require(
                level.call_count == self.wrapper_call_count
                and level.query_element_count > 0
                and level.factor_element_count > 0
                and level.gate_element_count > 0,
                f"QFG level {index} capture is incomplete",
            )
            query_rms = math.sqrt(
                level.query_delta_square_sum / level.query_element_count
            )
            input_rms = math.sqrt(
                level.query_input_square_sum / level.query_element_count
            )
            relative = (
                query_rms / input_rms
                if input_rms > 0.0
                else (0.0 if query_rms == 0.0 else None)
            )
            records.append(
                {
                    "level_index_zero_based": index,
                    "implementation_level_one_based": index + 1,
                    "apply_prepared_call_count": level.call_count,
                    "query_element_count": level.query_element_count,
                    "query_input_rms": input_rms,
                    "query_perturbation_rms": query_rms,
                    "relative_query_perturbation_rms": relative,
                    "query_perturbation_max_abs": level.query_delta_abs_max,
                    "factor_element_count": level.factor_element_count,
                    "factor_minus_one_rms": math.sqrt(
                        level.factor_delta_square_sum / level.factor_element_count
                    ),
                    "factor_minimum": level.factor_minimum,
                    "factor_maximum": level.factor_maximum,
                    "gate_element_count": level.gate_element_count,
                    "gate_rms": math.sqrt(
                        level.gate_square_sum / level.gate_element_count
                    ),
                }
            )
        return {
            "implementation": "direct_apply_prepared_output_minus_input",
            "factor_minus_one_is_not_query_perturbation": True,
            "wrapper_call_count": self.wrapper_call_count,
            "levels": records,
        }


@contextlib.contextmanager
def capture_query_perturbation(
    model_or_qfg: nn.Module,
) -> Iterator[QueryPerturbationRecorder]:
    """Temporarily wrap QFG ``apply_prepared`` and restore lookup exactly."""

    qfg = qfg_audit._resolve_qfg(model_or_qfg)
    _require(not qfg.training, "query perturbation capture requires eval mode")
    original = qfg.apply_prepared
    recorder = QueryPerturbationRecorder()
    had_instance_attribute = "apply_prepared" in qfg.__dict__
    previous_instance_attribute = qfg.__dict__.get("apply_prepared")

    @functools.wraps(original)
    def wrapped_apply_prepared(*args: Any, **kwargs: Any) -> Any:
        _require(bool(args), "QFG apply_prepared Query argument is missing")
        input_queries = tuple(args[0])
        output = original(*args, **kwargs)
        output_queries = getattr(output, "queries", None)
        factors = getattr(output, "factors", None)
        gates = getattr(output, "gates", None)
        _require(
            isinstance(output_queries, (tuple, list))
            and isinstance(factors, (tuple, list))
            and isinstance(gates, (tuple, list)),
            "QFG apply_prepared output contract differs",
        )
        recorder.append(input_queries, output_queries, factors, gates)
        return output

    object.__setattr__(qfg, "apply_prepared", wrapped_apply_prepared)
    try:
        yield recorder
    finally:
        if had_instance_attribute:
            object.__setattr__(qfg, "apply_prepared", previous_instance_attribute)
        else:
            object.__delattr__(qfg, "apply_prepared")


def _all_background_false_positive_pixels(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    *,
    threshold: float = FIXED_THRESHOLD,
) -> int:
    _require(
        len(probabilities) == len(targets) and bool(probabilities),
        "probability/target collection differs",
    )
    total = 0
    for probability, target in zip(probabilities, targets):
        probability_array = np.asarray(probability)
        target_array = np.asarray(target)
        _require(
            probability_array.shape == target_array.shape,
            "probability/target shape differs",
        )
        total += int(
            np.count_nonzero(
                np.logical_and(
                    probability_array > float(threshold),
                    target_array <= 0.5,
                )
            )
        )
    return total


def probability_difference(
    full_probabilities: Sequence[np.ndarray],
    other_probabilities: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Aggregate differences over every original, unpadded test pixel."""

    _require(
        len(full_probabilities) == len(other_probabilities)
        and bool(full_probabilities),
        "full/counterfactual probability collection differs",
    )
    maximum = 0.0
    absolute_sum = 0.0
    element_count = 0
    for full, other in zip(full_probabilities, other_probabilities):
        full_array = np.asarray(full, dtype=np.float32)
        other_array = np.asarray(other, dtype=np.float32)
        _require(full_array.shape == other_array.shape, "probability shape differs")
        difference = np.abs(
            full_array.astype(np.float64) - other_array.astype(np.float64)
        )
        maximum = max(maximum, float(difference.max()))
        absolute_sum += float(difference.sum())
        element_count += int(difference.size)
    mean_abs = absolute_sum / element_count
    equivalent = (
        maximum <= OUTPUT_EQUIVALENCE_MAX_ABS
        and mean_abs <= OUTPUT_EQUIVALENCE_MEAN_ABS
    )
    return {
        "scope": "all_original_unpadded_test_pixels",
        "element_count": element_count,
        "absolute_difference_sum": absolute_sum,
        "max_abs": maximum,
        "mean_abs": mean_abs,
        "equivalent": equivalent,
        "functionally_different": not equivalent,
        "equivalence_max_abs_threshold": OUTPUT_EQUIVALENCE_MAX_ABS,
        "equivalence_mean_abs_threshold": OUTPUT_EQUIVALENCE_MEAN_ABS,
    }


def _zero_based_spatial_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    ready = dict(summary)
    converted: list[dict[str, Any]] = []
    for raw in summary.get("levels", []):
        record = dict(raw)
        one_based = int(record.pop("level"))
        record["implementation_level_one_based"] = one_based
        record["level_index_zero_based"] = one_based - 1
        regions = record.get("regions", {})
        target_gate = regions.get("target", {}).get("gate", {}).get("mean")
        hard_gate = regions.get("hard_negative", {}).get("gate", {}).get("mean")
        target_factor = regions.get("target", {}).get("factor", {}).get("mean")
        hard_factor = regions.get("hard_negative", {}).get("factor", {}).get("mean")
        record.setdefault("gate_contrasts", {})[
            "target_minus_hard_negative"
        ] = (
            None
            if target_gate is None or hard_gate is None
            else float(target_gate) - float(hard_gate)
        )
        record.setdefault("factor_contrasts", {})[
            "target_minus_hard_negative"
        ] = (
            None
            if target_factor is None or hard_factor is None
            else float(target_factor) - float(hard_factor)
        )
        converted.append(record)
    ready["levels"] = sorted(
        converted, key=lambda value: value["level_index_zero_based"]
    )
    return ready


def _zero_based_factor_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    ready = dict(summary)
    levels: list[dict[str, Any]] = []
    for raw in summary.get("levels", []):
        record = dict(raw)
        one_based = int(record.pop("level"))
        record["implementation_level_one_based"] = one_based
        record["level_index_zero_based"] = one_based - 1
        levels.append(record)
    ready["levels"] = sorted(
        levels, key=lambda value: value["level_index_zero_based"]
    )
    return ready


@torch.inference_mode()
def _collect_one_mode(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    public_mode: str,
    *,
    full_reference: Sequence[np.ndarray] | None,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[float],
    list[str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    model.eval()
    primitive_mode = PUBLIC_TO_PRIMITIVE_MODE[public_mode]
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    identifiers: list[str] = []
    criterion = nn.BCELoss(reduction="mean")

    with qfg_audit.temporary_qfg_alpha_knockout(
        model, primitive_mode
    ) as alpha_knockout:
        with qfg_audit.capture_qfg_prepared_factors(model) as factor_capture:
            with legacy_qfg_audit.capture_spatial_qfg(model) as spatial_capture:
                with capture_query_perturbation(model) as query_capture:
                    for index, (images, masks, sizes, sample_ids) in enumerate(loader):
                        _require(
                            int(images.shape[0]) == int(masks.shape[0]) == 1,
                            "QFG diagnostic requires batch_size=1",
                        )
                        images = images.to(device, non_blocking=device.type == "cuda")
                        masks = masks.to(device, non_blocking=device.type == "cuda")
                        height, width = core._extract_hw(sizes)
                        prediction = core._final_prediction(model(images))[
                            :, :, :height, :width
                        ]
                        target = masks[:, :, :height, :width]
                        _require(
                            prediction.shape == target.shape,
                            "prediction/target shape differs",
                        )
                        _require(
                            bool(torch.isfinite(prediction).all()),
                            "prediction contains non-finite values",
                        )
                        loss = criterion(prediction.float(), target.float())
                        _require(
                            math.isfinite(float(loss.item())),
                            "evaluation loss is non-finite",
                        )
                        probability = (
                            prediction[0, 0]
                            .float()
                            .cpu()
                            .contiguous()
                            .numpy()
                            .astype(np.float32, copy=False)
                        )
                        target_array = (
                            target[0, 0].float().cpu().contiguous().numpy()
                        )
                        _require(
                            isinstance(sample_ids, (tuple, list))
                            and len(sample_ids) == 1,
                            "QFG diagnostic requires one sample ID per batch",
                        )
                        image_id = str(sample_ids[0])
                        probabilities.append(probability)
                        targets.append(target_array)
                        losses.append(float(loss.item()))
                        identifiers.append(image_id)
                        reference_probability = (
                            probability
                            if full_reference is None
                            else np.asarray(full_reference[index])
                        )
                        spatial_capture.consume(target_array, reference_probability)

    _require(
        len(probabilities) == len(loader.dataset),
        f"{public_mode} inference count differs",
    )
    return (
        probabilities,
        targets,
        losses,
        identifiers,
        dict(alpha_knockout),
        _zero_based_factor_summary(factor_capture.summary()),
        _zero_based_spatial_summary(spatial_capture.summary()),
        query_capture.summary(),
    )


def analyze_loaded_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    expected_identifiers: Sequence[str],
    *,
    sweep_thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Evaluate all six modes on an already strict-loaded model and loader."""

    model.eval()
    _require(not model.training, "QFG diagnostic model must be in eval mode")
    qfg_audit._resolve_qfg(model)
    state_before = stage2_audit.module_state_sha256(model)
    alpha_before = qfg_audit.alpha_state_sha256(model)
    full_probabilities: list[np.ndarray] | None = None
    full_targets: list[np.ndarray] | None = None
    full_identifiers: list[str] | None = None
    modes: dict[str, Any] = {}

    for public_mode in PUBLIC_MODES:
        mapping = normalize_public_mode(public_mode)
        collected = _collect_one_mode(
            model,
            loader,
            device,
            public_mode,
            full_reference=full_probabilities,
        )
        (
            probabilities,
            targets,
            losses,
            identifiers,
            alpha_knockout,
            factor_summary,
            spatial_summary,
            query_summary,
        ) = collected
        _require(
            identifiers == list(expected_identifiers),
            f"{public_mode} inference order differs from img_idx/test",
        )
        if full_probabilities is None:
            _require(public_mode == "full", "full mode must execute first")
            full_probabilities = probabilities
            full_targets = targets
            full_identifiers = identifiers
        else:
            _require(
                identifiers == full_identifiers,
                f"{public_mode} identifiers differ from full",
            )
            _require(
                all(
                    np.array_equal(left, right)
                    for left, right in zip(targets, full_targets or [])
                ),
                f"{public_mode} targets differ from full",
            )

        evaluated = core.evaluate_probability_arrays(
            probabilities,
            targets,
            losses,
            sweep_thresholds=sweep_thresholds,
        )
        fixed = dict(evaluated["fixed_threshold_0_5"])
        false_positive_pixels = _all_background_false_positive_pixels(
            probabilities, targets
        )
        _require(
            0 <= false_positive_pixels <= int(fixed["valid_pixel_count"]),
            "all-background false-positive count is invalid",
        )
        fixed["false_positive_pixels"] = false_positive_pixels
        difference = probability_difference(full_probabilities, probabilities)
        alpha_after_mode = qfg_audit.alpha_state_sha256(model)
        _require(alpha_after_mode == alpha_before, "QFG alpha state was not restored")
        modes[public_mode] = {
            **mapping,
            "alpha_knockout": alpha_knockout,
            "fixed_threshold_0_5": fixed,
            "descriptive_pd_fa": evaluated["descriptive_pd_fa"],
            "threshold_roles": evaluated["threshold_roles"],
            "probability_difference_to_full": difference,
            "query_perturbation": query_summary,
            "factor_summary": factor_summary,
            "spatial_gate_factor_statistics": spatial_summary,
            "restoration_audit": {
                "alpha_state_sha256_expected": alpha_before,
                "alpha_state_sha256_after_mode": alpha_after_mode,
                "alpha_state_unchanged": alpha_after_mode == alpha_before,
            },
        }
        if public_mode != "full":
            del probabilities, targets, losses

    _require(full_probabilities is not None, "full probabilities are missing")
    state_after = stage2_audit.module_state_sha256(model)
    alpha_after = qfg_audit.alpha_state_sha256(model)
    _require(state_after == state_before, "QFG diagnostic changed model state")
    _require(alpha_after == alpha_before, "QFG diagnostic changed alpha state")
    return {
        "modes": modes,
        "restoration_audit": {
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_state_unchanged": state_after == state_before,
            "alpha_state_sha256_before": alpha_before,
            "alpha_state_sha256_after": alpha_after,
            "alpha_state_unchanged": alpha_after == alpha_before,
            "temporary_hooks_restored": True,
        },
        "probability_arrays_persisted": False,
    }


_REFERENCE_EXACT_COUNT_KEYS = (
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "unmatched_predicted_pixels",
    "valid_pixel_count",
)


def reference_replay_audit(
    observed: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the full-mode fixed point to replay the historical evaluation."""

    required = {
        "threshold",
        "miou",
        "niou",
        "pd",
        "fa",
        *_REFERENCE_EXACT_COUNT_KEYS,
    }
    _require(required <= set(observed), "full replay metrics are incomplete")
    _require(required <= set(reference), "reference metrics are incomplete")
    compared: dict[str, Any] = {}
    for key in sorted(reference):
        if key not in observed or key == "false_positive_pixels":
            continue
        expected = reference[key]
        actual = observed[key]
        if key in _REFERENCE_EXACT_COUNT_KEYS:
            _require(actual == expected, f"reference replay count differs: {key}")
            tolerance = 0.0
            absolute_difference = 0.0
        elif expected is None:
            _require(actual is None, f"reference replay null differs: {key}")
            tolerance = 0.0
            absolute_difference = 0.0
        else:
            tolerance = (
                1e-4
                if key
                in {"miou", "niou", "pixel_precision", "pixel_recall", "pixel_f1"}
                else 1e-7
                if key == "test_loss"
                else 1e-15
            )
            absolute_difference = abs(float(actual) - float(expected))
            _require(
                math.isclose(
                    float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance
                ),
                f"reference replay metric differs: {key}",
            )
        compared[key] = {
            "absolute_difference": absolute_difference,
            "absolute_tolerance": tolerance,
        }
    return {
        "passed": True,
        "comparison": "full_mode_fixed_threshold_0_5_vs_existing_best_miou",
        "compared": compared,
    }


def analyze_run(
    *,
    dataset: str,
    run_dir: Path,
    dataset_root: Path,
    data_protocol_manifest: Path,
    reference_evaluation: Path,
    device_name: str,
    workers: int,
) -> dict[str, Any]:
    _require(dataset in data_protocol.DATASETS, "dataset is outside formal scope")
    _require(workers >= 0, "workers must be non-negative")
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
        checkpoint_role=CHECKPOINT_ROLE,
        requested_tss_weight=adapter.REQUESTED_TSS_WEIGHT,
    )
    request.validate()
    checkpoint_payload, checkpoint_binding = (
        stage2_audit._load_checkpoint_allowing_added_sources(
            request, Path(run_dir), manifest_path, manifest
        )
    )
    reference_path = Path(reference_evaluation).resolve(strict=True)
    reference = adapter.validate_completed_output(
        reference_path, dataset=dataset, checkpoint_role=CHECKPOINT_ROLE
    )
    _require(
        reference["checkpoint_binding"]["checkpoint"]["sha256"]
        == checkpoint_binding["checkpoint"]["sha256"],
        "reference evaluation/checkpoint SHA differs",
    )

    model, model_metadata = core.build_inference_model(
        request, checkpoint_payload["state_dict"]
    )
    model.to(device)
    model.eval()
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
    analyzed = analyze_loaded_model(
        model, loader, device, dataset_object.sample_ids
    )
    replay = reference_replay_audit(
        analyzed["modes"]["full"]["fixed_threshold_0_5"],
        reference["fixed_threshold_0_5"],
    )
    _require(replay["passed"] is True, "full reference replay did not pass")
    ordered_id_sha = hashlib.sha256(
        ("\n".join(dataset_object.sample_ids) + "\n").encode("utf-8")
    ).hexdigest()

    output = {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": dataset,
        "method": REFERENCE_METHOD,
        "training_model_method": TRAINING_MODEL_METHOD,
        "checkpoint_role": CHECKPOINT_ROLE,
        "seed": SEED,
        "test_selected": True,
        "selection_is_optimistic": True,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "fixed_threshold": FIXED_THRESHOLD,
        "mode_order": list(PUBLIC_MODES),
        **analyzed,
        "reference_replay_audit": replay,
        "reference_reuse": {
            "path": str(reference_path),
            "sha256": file_sha256(reference_path),
            "checkpoint_role": CHECKPOINT_ROLE,
            "fixed_threshold_0_5": reference["fixed_threshold_0_5"],
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
            "test_count": len(dataset_object.sample_ids),
            "inference_order_newline_sha256": ordered_id_sha,
            "normalization": core.NORMALIZATION[dataset],
            "sirst3_in_formal_matrix": False,
        },
        "metric_protocol": {
            "implementation": "experiments.train_tpd_pilot.ValidationMetrics",
            "fixed_threshold": FIXED_THRESHOLD,
            "prediction_comparison": "probability > threshold",
            "match_radius": core.MATCH_RADIUS,
            "tiny_area": core.TINY_AREA,
            "descriptive_sweep_only": True,
            "sweep_reselects_checkpoint": False,
        },
        "source_lock_policy": {
            "historical_frozen_dependencies_verified_by_path_and_sha256": True,
            "new_unlisted_model_sources_allowed": True,
            "verified_historical_runtime_sources": checkpoint_binding[
                "training_runtime_sources"
            ]["source_sha256"],
        },
        "source_sha256": {
            "analysis/analyze_three_dataset_qfg_level_knockout_v1.py": file_sha256(
                Path(__file__)
            ),
            "analysis/audit_final_qfg_functional_use.py": file_sha256(
                Path(qfg_audit.__file__)
            ),
            "analysis/run_final_qfg_six_mode_audit.py": file_sha256(
                Path(legacy_qfg_audit.__file__)
            ),
            "analysis/analyze_ner_stage2_mask_knockout_v1.py": file_sha256(
                Path(stage2_audit.__file__)
            ),
            "experiments/evaluate_three_dataset_v2.py": file_sha256(
                Path(core.__file__)
            ),
            "experiments/evaluate_three_dataset_tss_off_seed42_v1.py": file_sha256(
                Path(adapter.__file__)
            ),
        },
        "derived_checkpoint_written": False,
        "probability_cache_written": False,
        "no_fabricated_results": True,
        "stability_claim_supported": False,
    }
    validate_output_payload(output)
    del model, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output


def validate_output_payload(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema") == SCHEMA, "QFG analyzer schema differs")
    _require(payload.get("status") == "complete", "QFG analyzer is incomplete")
    _require(payload.get("dataset") in data_protocol.DATASETS, "dataset differs")
    _require(payload.get("checkpoint_role") == CHECKPOINT_ROLE, "role differs")
    _require(payload.get("seed") == SEED, "seed differs")
    _require(payload.get("test_selected") is True, "test_selected differs")
    _require(
        payload.get("evaluation_protocol") == EVALUATION_PROTOCOL,
        "evaluation protocol differs",
    )
    _require(
        payload.get("reference_replay_audit", {}).get("passed") is True,
        "full reference replay did not pass",
    )
    modes = payload.get("modes")
    _require(isinstance(modes, Mapping), "QFG analyzer modes are missing")
    _require(set(modes) == set(PUBLIC_MODES), "QFG analyzer mode set differs")
    declared_order = payload.get("mode_order")
    if declared_order is not None:
        _require(
            declared_order == list(PUBLIC_MODES),
            "QFG analyzer declared mode order differs",
        )
    required_fixed = {
        "matched_target_count",
        "matched_tiny_target_count",
        "miou",
        "niou",
        "unmatched_predicted_pixels",
        "false_positive_pixels",
    }
    for public_mode in PUBLIC_MODES:
        mode = modes[public_mode]
        _require(
            mode.get("public_mode") == public_mode,
            f"{public_mode} public mode differs",
        )
        fixed = mode.get("fixed_threshold_0_5")
        _require(isinstance(fixed, Mapping), f"{public_mode} fixed point missing")
        _require(required_fixed <= set(fixed), f"{public_mode} fixed fields missing")
        _require(
            fixed.get("threshold") == FIXED_THRESHOLD,
            f"{public_mode} fixed threshold differs",
        )
        difference = mode.get("probability_difference_to_full")
        _require(
            isinstance(difference, Mapping)
            and int(difference.get("element_count", 0)) > 0,
            f"{public_mode} probability difference missing",
        )
        _require(
            mode.get("restoration_audit", {}).get("alpha_state_unchanged") is True,
            f"{public_mode} alpha restoration failed",
        )
    restoration = payload.get("restoration_audit", {})
    _require(
        restoration.get("model_state_unchanged") is True
        and restoration.get("alpha_state_unchanged") is True,
        "QFG analyzer restoration audit failed",
    )
    _require(
        payload.get("probability_cache_written") is False
        and payload.get("derived_checkpoint_written") is False,
        "QFG analyzer wrote a forbidden artifact",
    )


def _default_run_dir(dataset: str) -> Path:
    return (
        DEFAULT_TSS_OFF_ROOT
        / "runs"
        / dataset
        / "final_tss_off"
        / "seed_42"
    )


def _default_reference(run_dir: Path) -> Path:
    return Path(run_dir) / "evaluations" / "best_miou.json"


def _default_output(dataset: str) -> Path:
    return (
        DEFAULT_OUTPUT_ROOT
        / "runs"
        / dataset
        / "v4_tss_off_best_miou_seed42"
        / "evaluation.json"
    )


def atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one JSON file exactly once, including concurrent writers."""

    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=data_protocol.DATASETS, required=True)
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
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if not (args.device == "cpu" or args.device.startswith("cuda:")):
        parser.error("--device must be cpu or cuda:N")
    args.run_dir = args.run_dir or _default_run_dir(args.dataset)
    args.reference_evaluation = args.reference_evaluation or _default_reference(
        args.run_dir
    )
    args.output = args.output or _default_output(args.dataset)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing existing output before inference: {args.output}")
    output = analyze_run(
        dataset=args.dataset,
        run_dir=args.run_dir,
        dataset_root=args.dataset_root,
        data_protocol_manifest=args.data_protocol_manifest,
        reference_evaluation=args.reference_evaluation,
        device_name=args.device,
        workers=args.workers,
    )
    atomic_create_json(args.output, output)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "complete",
                "output": str(args.output.resolve()),
                "sha256": file_sha256(args.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


__all__ = [
    "CHECKPOINT_ROLE",
    "EVALUATION_PROTOCOL",
    "FIXED_THRESHOLD",
    "PUBLIC_MODES",
    "PUBLIC_TO_PRIMITIVE_MODE",
    "QueryPerturbationRecorder",
    "SCHEMA",
    "analyze_loaded_model",
    "analyze_run",
    "atomic_create_json",
    "capture_query_perturbation",
    "main",
    "normalize_public_mode",
    "parse_args",
    "probability_difference",
    "reference_replay_audit",
    "validate_output_payload",
]


if __name__ == "__main__":
    main()
