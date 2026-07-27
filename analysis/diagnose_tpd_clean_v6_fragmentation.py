#!/usr/bin/env python3
"""Read-only failure diagnosis for the completed TPD-Clean-v6 matrix.

This tool never trains, selects a new checkpoint, or writes into a formal run
directory.  It strict-loads the existing checkpoints, evaluates the frozen
internal validation split, and writes new diagnostic JSON files below an
independent analysis directory.

The diagnostic has two layers:

* the repository's established Pd/Fa/mIoU metrics at fixed thresholds and
  independently selected Pd-at-Fa-budget operating points;
* a mutually exclusive taxonomy of *unmatched* predicted components plus
  per-GT fragmentation measurements.

Three zero-training conditions are supported:

``as_trained``
    Evaluate the checkpoint without changing its forward path.
``same_weights_context_off``
    Temporarily set every V6 block's Context headroom to neutral one.
``same_weights_residual_off``
    Temporarily replace every V6 block output with its Keep-only output by a
    forward hook.

All temporary changes are restored in ``finally`` blocks.  A model-state
SHA256 is checked before and after every condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from skimage import measure, morphology
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_pd_fa_sweep as sweep_base  # noqa: E402
from experiments import summarize_tpd_clean_v6_formal800 as formal  # noqa: E402
from experiments.evaluate_tpd_clean_v6_pd_fa import (  # noqa: E402
    adaptive_thresholds_closed_interval,
)
from experiments.train_tpd_clean_v6 import build_clean_v6_model  # noqa: E402
from experiments.train_tpd_pilot import (  # noqa: E402
    ValidationMetrics,
    ValidationSubset,
    final_prediction,
    json_ready,
)
from model.tpd_clean_v6 import TPDCleanV6Block  # noqa: E402


SCHEMA = "sctransnet_tpd_clean_v6_frozen_failure_diagnostic_v1"
MATRIX_SCHEMA = "sctransnet_tpd_clean_v6_frozen_failure_matrix_v1"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "analysis/results/tpd_clean_v6_frozen_failure_diagnostic_v1"
)
DEFAULT_FIXED_THRESHOLDS = (0.5, 0.58, 0.999)
DEFAULT_FA_BUDGETS = tuple(float(value) for value in formal.BUDGET_KEYS)
DEFAULT_MODES = (
    "as_trained",
    "same_weights_context_off",
    "same_weights_residual_off",
)
POSTPROCESS_GPUS = {
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
UNMATCHED_CLASSES = (
    "in_gt_fragment",
    "near_gt_duplicate",
    "attached_or_near_gt",
    "background_false_object",
)
TAXONOMY_DEFINITIONS = {
    "matched_primary": (
        "Predicted component selected by the repository's maximum-cardinality "
        "then minimum-centroid-distance one-to-one matching."
    ),
    "in_gt_fragment": (
        "Unmatched component with at least one pixel overlapping a GT component."
    ),
    "near_gt_duplicate": (
        "Unmatched non-overlapping component whose centroid is at distance "
        "strictly less than match_radius from at least one GT centroid."
    ),
    "attached_or_near_gt": (
        "Remaining unmatched component that intersects the binary GT dilation; "
        "the default Euclidean disk radius is three pixels."
    ),
    "background_false_object": (
        "Remaining unmatched component unrelated to the GT mask or its "
        "registered dilation neighbourhood."
    ),
}
METRIC_KEYS = (
    "val_loss",
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pd",
    "tiny_pd",
    "fa",
    "false_objects_per_image",
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)


def configure_formal_inference_determinism(device: str) -> Dict[str, Any]:
    """Match the frozen V6 FP32 inference controls before loading a model."""

    if device == "cuda:0" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "CUDA diagnosis requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def load_json_object(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected regular JSON file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _finite_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    thresholds = tuple(float(value) for value in values)
    if not thresholds:
        raise ValueError("At least one fixed threshold is required")
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in thresholds
    ):
        raise ValueError("Fixed thresholds must be finite values in [0, 1]")
    if len(thresholds) != len(set(thresholds)):
        raise ValueError("Fixed thresholds must be unique")
    return thresholds


def _finite_budgets(values: Sequence[float]) -> tuple[float, ...]:
    budgets = tuple(float(value) for value in values)
    if not budgets:
        raise ValueError("At least one Fa budget is required")
    if any(not math.isfinite(value) or value < 0.0 for value in budgets):
        raise ValueError("Fa budgets must be finite and non-negative")
    if len(budgets) != len(set(budgets)):
        raise ValueError("Fa budgets must be unique")
    return budgets


def run_directory(
    results_root: Path,
    variant: str,
    seed: int,
) -> Path:
    return (
        results_root
        / formal.DATASET
        / variant
        / f"seed_{seed}_{formal.RUN_TAG}"
    )


def diagnostic_output_path(
    output_dir: Path,
    variant: str,
    seed: int,
    role: str,
) -> Path:
    return output_dir / variant / f"seed_{seed}" / f"{role}.json"


def requested_jobs(args: argparse.Namespace) -> list[Dict[str, Any]]:
    jobs: list[Dict[str, Any]] = []
    for variant in args.variant:
        for seed in args.seed:
            directory = run_directory(args.results_root, variant, seed)
            for role in args.checkpoint_role:
                spec = formal.ROLE_SPECS[role]
                jobs.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "role": role,
                        "run_dir": directory,
                        "checkpoint": directory / spec["checkpoint"],
                        "formal_sweep": directory / spec["sweep"],
                        "output": diagnostic_output_path(
                            args.output_dir,
                            variant,
                            seed,
                            role,
                        ),
                    }
                )
    return jobs


def validate_job_artifacts(job: Mapping[str, Any]) -> Dict[str, Any]:
    run_dir = Path(job["run_dir"])
    required = {
        "protocol": run_dir / "protocol.json",
        "split": run_dir / "split.json",
        "summary": run_dir / "summary.json",
        "metrics": run_dir / "metrics.jsonl",
        "checkpoint": Path(job["checkpoint"]),
        "formal_sweep": Path(job["formal_sweep"]),
    }
    for label, path in required.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{label} is not a regular file: {path}")

    protocol = load_json_object(required["protocol"])
    split = load_json_object(required["split"])
    summary = load_json_object(required["summary"])
    checkpoint = torch.load(
        required["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {required['checkpoint']}")
    arguments = protocol.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError(f"protocol.json lacks arguments: {run_dir}")

    expected = {
        "dataset": formal.DATASET,
        "variant": str(job["variant"]),
        "seed": int(job["seed"]),
    }
    observed_sources = {
        "protocol": {
            "dataset": arguments.get("dataset"),
            "variant": arguments.get("variant"),
            "seed": arguments.get("seed"),
        },
        "summary": {
            "dataset": summary.get("dataset"),
            "variant": summary.get("variant"),
            "seed": summary.get("seed"),
        },
        "checkpoint": {
            "dataset": checkpoint.get("dataset"),
            "variant": checkpoint.get("variant"),
            "seed": checkpoint.get("seed"),
        },
    }
    for source, observed in observed_sources.items():
        if observed != expected:
            raise ValueError(
                f"{source} metadata mismatch in {run_dir}: "
                f"expected={expected}, observed={observed}"
            )
    if summary.get("status") != "complete":
        raise ValueError(f"Run is not complete: {run_dir}")
    if int(arguments.get("epochs", -1)) != formal.EXPECTED_EPOCHS:
        raise ValueError(f"Unexpected epoch contract: {run_dir}")
    if split.get("used_val_count") != len(split.get("used_val_ids", [])):
        raise ValueError(f"Validation manifest count mismatch: {run_dir}")
    if split.get("used_val_count") != formal.EXPECTED_VAL_COUNT:
        raise ValueError(f"Unexpected validation count: {run_dir}")
    for label, payload in {
        "protocol": protocol,
        "split": split,
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        if payload.get("official_test_accessed") is not False:
            raise ValueError(
                f"{label} does not assert official_test_accessed=false: {run_dir}"
            )
    if checkpoint.get("checkpoint_role") != formal.ROLE_SPECS[job["role"]][
        "checkpoint_role"
    ]:
        raise ValueError(f"Checkpoint role mismatch: {required['checkpoint']}")

    return {
        "paths": required,
        "protocol": protocol,
        "split": split,
        "summary": summary,
        "checkpoint": checkpoint,
        "formal_sweep": load_json_object(required["formal_sweep"]),
        "input_sha256": {
            label: file_sha256(path) for label, path in required.items()
        },
    }


def _bind_requested_gpu(device: str, physical_gpu: str | None) -> Dict[str, Any]:
    if device == "cpu":
        if physical_gpu is not None:
            raise ValueError("--physical-gpu is only valid with --device cuda:0")
        return {"device": "cpu", "physical_gpu": None, "gpu_uuid": None}
    if device != "cuda:0" or physical_gpu not in POSTPROCESS_GPUS:
        raise ValueError(
            "CUDA diagnosis requires --device cuda:0 and --physical-gpu 2 or 3"
        )
    gpu_uuid = POSTPROCESS_GPUS[physical_gpu]
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu_uuid,
            "--query-gpu=index,name,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fields = [field.strip() for field in query.split(",")]
    if fields != [physical_gpu, "NVIDIA GeForce RTX 5090", gpu_uuid]:
        raise RuntimeError(
            "Requested physical GPU identity differs: "
            f"expected={[physical_gpu, 'NVIDIA GeForce RTX 5090', gpu_uuid]}, "
            f"observed={fields}"
        )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable after binding the requested GPU")
    visible_name = torch.cuda.get_device_name(0)
    if visible_name != "NVIDIA GeForce RTX 5090":
        raise RuntimeError(f"Unexpected visible CUDA device: {visible_name!r}")
    return {
        "device": "cuda:0",
        "physical_gpu": physical_gpu,
        "gpu_uuid": gpu_uuid,
        "visible_device_name": visible_name,
    }


def v6_blocks(model: nn.Module) -> list[TPDCleanV6Block]:
    blocks = [
        module for module in model.modules() if isinstance(module, TPDCleanV6Block)
    ]
    if len(blocks) != 7:
        raise RuntimeError(f"Expected seven V6 blocks (4+3), found {len(blocks)}")
    return blocks


def _distribution_summary(values: np.ndarray) -> Dict[str, float]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if flattened.size == 0:
        return {"median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "median": float(np.median(flattened)),
        "p90": float(np.quantile(flattened, 0.9)),
        "max": float(np.max(flattened)),
    }


def static_checkpoint_diagnostics(model: nn.Module) -> Dict[str, Any]:
    """Measure learned scale and phase-sum cancellation without activations.

    For phase weights ``W_p``, both cancellation ratios use a triangle-
    inequality denominator and therefore lie in ``[0, 1]``:

    ``rho_L1 = ||sum_p W_p||_1 / sum_p ||W_p||_1``
    ``rho_L2 = ||sum_p W_p||_2 / (sqrt(4) * ||W||_2 + eps)``.

    A smaller ratio means stronger signed cancellation in the phase-tied
    projection.  These static weight summaries are background evidence only;
    they do not measure activation energy or spatial total variation.
    """

    rows: list[Dict[str, Any]] = []
    all_effective_scales: list[np.ndarray] = []
    global_l1_numerator = 0.0
    global_l1_denominator = 0.0
    global_l2_numerator_squared = 0.0
    global_weight_l2_squared = 0.0
    named_blocks = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, TPDCleanV6Block)
    ]
    if len(named_blocks) != 7:
        raise RuntimeError(
            f"Expected seven named V6 blocks (4+3), found {len(named_blocks)}"
        )
    for name, block in named_blocks:
        raw_scale = block.saliency_scale.detach().float().cpu().numpy()
        effective_abs_scale = np.abs(np.tanh(raw_scale))
        all_effective_scales.append(effective_abs_scale)

        weight = (
            block.phase_compress.weight.detach()
            .float()
            .cpu()
            .reshape(
                block.phase_compress.out_channels,
                block.channels,
                4,
            )
            .numpy()
        )
        phase_sum = weight.sum(axis=2)
        l1_numerator = float(np.abs(phase_sum).sum())
        l1_denominator = float(np.abs(weight).sum())
        l2_numerator = float(np.linalg.norm(phase_sum.reshape(-1), ord=2))
        weight_l2 = float(np.linalg.norm(weight.reshape(-1), ord=2))
        l2_denominator = (
            math.sqrt(4.0) * weight_l2 + np.finfo(np.float64).eps
        )
        rho_l1 = l1_numerator / l1_denominator if l1_denominator else 0.0
        rho_l2 = l2_numerator / l2_denominator if l2_denominator else 0.0
        if not 0.0 <= rho_l1 <= 1.0 + 1e-7:
            raise RuntimeError(f"Invalid rho_L1 for {name}: {rho_l1}")
        if not 0.0 <= rho_l2 <= 1.0 + 1e-7:
            raise RuntimeError(f"Invalid rho_L2 for {name}: {rho_l2}")
        global_l1_numerator += l1_numerator
        global_l1_denominator += l1_denominator
        global_l2_numerator_squared += l2_numerator**2
        global_weight_l2_squared += weight_l2**2
        rows.append(
            {
                "block": name,
                "channels": block.channels,
                "use_context_headroom": block.use_context_headroom,
                "saliency_scale_raw": _distribution_summary(
                    np.abs(raw_scale)
                ),
                "saliency_scale_effective_abs_tanh": _distribution_summary(
                    effective_abs_scale
                ),
                "phase_sum_cancellation": {
                    "rho_l1": rho_l1,
                    "rho_l2": rho_l2,
                    "l1_numerator": l1_numerator,
                    "l1_denominator": l1_denominator,
                    "l2_numerator": l2_numerator,
                    "l2_denominator_sqrt4_weight_norm": l2_denominator,
                },
            }
        )
    global_effective = np.concatenate(all_effective_scales)
    return {
        "block_count": len(rows),
        "blocks": rows,
        "aggregate": {
            "saliency_scale_effective_abs_tanh": _distribution_summary(
                global_effective
            ),
            "phase_sum_cancellation": {
                "rho_l1": (
                    global_l1_numerator / global_l1_denominator
                    if global_l1_denominator
                    else 0.0
                ),
                "rho_l2": (
                    math.sqrt(global_l2_numerator_squared)
                    / (
                        math.sqrt(4.0)
                        * math.sqrt(global_weight_l2_squared)
                        + np.finfo(np.float64).eps
                    )
                    if global_weight_l2_squared
                    else 0.0
                ),
            },
        },
        "definitions": {
            "saliency_scale": "absolute tanh(saliency_scale parameter)",
            "rho_l1": (
                "L1 norm of phase-summed Keep weights divided by the sum of "
                "the four phase L1 norms"
            ),
            "rho_l2": (
                "L2 norm of phase-summed Keep weights divided by sqrt(4) "
                "times the L2 norm of the full four-phase Keep tensor plus eps"
            ),
            "direction": (
                "smaller rho means stronger signed phase cancellation"
            ),
        },
        "scope_limit": (
            "Static checkpoint weights only; activation-energy and spatial-TV "
            "diagnostics require a separate registered activation audit."
        ),
    }


@contextmanager
def temporary_counterfactual(
    model: nn.Module,
    mode: str,
) -> Iterator[Dict[str, Any]]:
    """Apply one zero-training condition and restore it unconditionally."""

    if mode not in DEFAULT_MODES:
        raise ValueError(f"Unsupported diagnostic mode: {mode}")
    blocks = v6_blocks(model)
    state_before = model_state_sha256(model)
    original_context_flags = [block.use_context_headroom for block in blocks]
    handles: list[Any] = []
    implementation = "unchanged_forward"
    try:
        if mode == "same_weights_context_off":
            implementation = "temporary_use_context_headroom_false"
            for block in blocks:
                block.use_context_headroom = False
        elif mode == "same_weights_residual_off":
            implementation = "temporary_keep_only_forward_hooks"

            def keep_only_hook(
                module: nn.Module,
                inputs: tuple[torch.Tensor, ...],
                _output: torch.Tensor,
            ) -> torch.Tensor:
                if not isinstance(module, TPDCleanV6Block):
                    raise TypeError("Keep-only hook attached to non-V6 block")
                keep, _, _ = module.branches(inputs[0])
                return module.activation(keep)

            handles = [
                block.register_forward_hook(keep_only_hook) for block in blocks
            ]
        yield {
            "mode": mode,
            "implementation": implementation,
            "block_count": len(blocks),
            "state_sha256_before": state_before,
        }
    finally:
        for handle in handles:
            handle.remove()
        for block, original in zip(blocks, original_context_flags):
            block.use_context_headroom = original
        state_after = model_state_sha256(model)
        restored_flags = [
            block.use_context_headroom for block in blocks
        ] == original_context_flags
        if state_after != state_before or not restored_flags:
            raise RuntimeError(
                f"Diagnostic mode {mode!r} failed to restore model state"
            )


def centroid_matching(
    target_regions: Sequence[Any],
    predicted_regions: Sequence[Any],
    match_radius: float,
) -> tuple[Dict[int, int], set[int]]:
    """Replay the repository's cardinality-first centroid matching exactly."""

    target_to_prediction: Dict[int, int] = {}
    matched_predictions: set[int] = set()
    if not target_regions or not predicted_regions:
        return target_to_prediction, matched_predictions
    distances = np.empty(
        (len(target_regions), len(predicted_regions)),
        dtype=np.float64,
    )
    for target_index, target_region in enumerate(target_regions):
        target_centroid = np.asarray(target_region.centroid)
        for predicted_index, predicted_region in enumerate(predicted_regions):
            distances[target_index, predicted_index] = np.linalg.norm(
                np.asarray(predicted_region.centroid) - target_centroid
            )
    cardinality_reward = (
        min(len(target_regions), len(predicted_regions)) + 1
    ) * max(1.0, match_radius)
    real_cost = np.where(
        distances < match_radius,
        distances - cardinality_reward,
        cardinality_reward,
    )
    assignment_cost = np.concatenate(
        (
            real_cost,
            np.zeros((len(target_regions), len(target_regions))),
        ),
        axis=1,
    )
    assigned_targets, assigned_columns = linear_sum_assignment(assignment_cost)
    for target_index, column_index in zip(
        assigned_targets,
        assigned_columns,
    ):
        if (
            column_index < len(predicted_regions)
            and distances[target_index, column_index] < match_radius
        ):
            target_to_prediction[int(target_index)] = int(column_index)
            matched_predictions.add(int(column_index))
    return target_to_prediction, matched_predictions


def component_diagnostics(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
    match_radius: float,
    dilation_radius: int,
) -> Dict[str, Any]:
    """Classify all unmatched components and return per-GT topology records."""

    prediction = np.asarray(probability) > float(threshold)
    target_binary = np.asarray(target) > 0.5
    predicted_labels = measure.label(prediction, connectivity=2)
    target_labels = measure.label(target_binary, connectivity=2)
    predicted_regions = measure.regionprops(predicted_labels)
    target_regions = measure.regionprops(target_labels)
    target_to_prediction, matched_predictions = centroid_matching(
        target_regions,
        predicted_regions,
        match_radius,
    )

    overlap = np.zeros(
        (len(target_regions), len(predicted_regions)),
        dtype=np.int64,
    )
    for target_index, region in enumerate(target_regions):
        rows, columns = region.coords.T
        labels, counts = np.unique(
            predicted_labels[rows, columns],
            return_counts=True,
        )
        for label, count in zip(labels, counts):
            if int(label) > 0:
                overlap[target_index, int(label) - 1] = int(count)

    if target_binary.any():
        dilated_target = morphology.binary_dilation(
            target_binary,
            footprint=morphology.disk(dilation_radius),
        )
    else:
        dilated_target = target_binary.copy()

    class_counts = {name: 0 for name in UNMATCHED_CLASSES}
    class_pixels = {name: 0 for name in UNMATCHED_CLASSES}
    literal_gt_overlap_pixels = 0
    for predicted_index, region in enumerate(predicted_regions):
        if predicted_index in matched_predictions:
            continue
        overlapping_targets = (
            np.flatnonzero(overlap[:, predicted_index])
            if len(target_regions)
            else np.asarray([], dtype=np.int64)
        )
        if overlapping_targets.size:
            category = "in_gt_fragment"
            literal_gt_overlap_pixels += int(
                overlap[:, predicted_index].sum()
            )
        else:
            near_centroid = any(
                np.linalg.norm(
                    np.asarray(region.centroid)
                    - np.asarray(target_region.centroid)
                )
                < match_radius
                for target_region in target_regions
            )
            if near_centroid:
                category = "near_gt_duplicate"
            else:
                rows, columns = region.coords.T
                if bool(dilated_target[rows, columns].any()):
                    category = "attached_or_near_gt"
                else:
                    category = "background_false_object"
        class_counts[category] += 1
        class_pixels[category] += int(region.area)

    per_gt: list[Dict[str, Any]] = []
    for target_index, target_region in enumerate(target_regions):
        overlaps = overlap[target_index]
        overlapping_indices = np.flatnonzero(overlaps)
        in_gt_prediction_area = int(overlaps.sum())
        largest_overlap = (
            int(overlaps[overlapping_indices].max())
            if overlapping_indices.size
            else 0
        )
        matched_prediction = target_to_prediction.get(target_index)
        matched_component_area = (
            int(predicted_regions[matched_prediction].area)
            if matched_prediction is not None
            else 0
        )
        per_gt.append(
            {
                "gt_index": target_index,
                "gt_area": int(target_region.area),
                "gt_centroid": [
                    float(target_region.centroid[0]),
                    float(target_region.centroid[1]),
                ],
                "formally_matched": matched_prediction is not None,
                "matched_prediction_index": matched_prediction,
                "overlapping_prediction_components": int(
                    overlapping_indices.size
                ),
                "matched_component_area": matched_component_area,
                "all_in_gt_prediction_area": in_gt_prediction_area,
                "largest_in_gt_fragment_area": largest_overlap,
                "largest_fragment_fraction": (
                    largest_overlap / in_gt_prediction_area
                    if in_gt_prediction_area
                    else 0.0
                ),
                "covered_gt_fraction": (
                    in_gt_prediction_area / int(target_region.area)
                    if target_region.area
                    else 0.0
                ),
                "fragment_excess": max(
                    0,
                    int(overlapping_indices.size) - 1,
                ),
            }
        )

    unmatched_pixels_total = sum(class_pixels.values())
    unmatched_components_total = sum(class_counts.values())
    matched_primary_pixels = sum(
        int(predicted_regions[index].area) for index in matched_predictions
    )
    target_associated_pixels = (
        class_pixels["in_gt_fragment"]
        + class_pixels["near_gt_duplicate"]
        + class_pixels["attached_or_near_gt"]
    )
    return {
        "predicted_component_count": len(predicted_regions),
        "target_component_count": len(target_regions),
        "matched_primary_component_count": len(matched_predictions),
        "matched_primary_pixels": matched_primary_pixels,
        "unmatched_component_count": unmatched_components_total,
        "unmatched_pixels_total": unmatched_pixels_total,
        "unmatched_component_count_by_class": class_counts,
        "unmatched_component_pixels_by_class": class_pixels,
        "literal_gt_overlap_pixels_in_unmatched_components": (
            literal_gt_overlap_pixels
        ),
        "unmatched_pixels_in_gt": class_pixels["in_gt_fragment"],
        "unmatched_pixels_near_gt": (
            class_pixels["near_gt_duplicate"]
            + class_pixels["attached_or_near_gt"]
        ),
        "unmatched_pixels_background": class_pixels[
            "background_false_object"
        ],
        "fragment_fa_fraction": (
            target_associated_pixels / unmatched_pixels_total
            if unmatched_pixels_total
            else 0.0
        ),
        "background_fa_fraction": (
            class_pixels["background_false_object"]
            / unmatched_pixels_total
            if unmatched_pixels_total
            else 0.0
        ),
        "per_gt": per_gt,
    }


def aggregate_component_diagnostics(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    identifiers: Sequence[str],
    threshold: float,
    match_radius: float,
    dilation_radius: int,
) -> Dict[str, Any]:
    class_counts = {name: 0 for name in UNMATCHED_CLASSES}
    class_pixels = {name: 0 for name in UNMATCHED_CLASSES}
    totals = {
        "predicted_component_count": 0,
        "target_component_count": 0,
        "matched_primary_component_count": 0,
        "matched_primary_pixels": 0,
        "unmatched_component_count": 0,
        "unmatched_pixels_total": 0,
        "literal_gt_overlap_pixels_in_unmatched_components": 0,
    }
    per_gt: list[Dict[str, Any]] = []
    for probability, target, identifier in zip(
        probabilities,
        targets,
        identifiers,
    ):
        image_result = component_diagnostics(
            probability,
            target,
            threshold,
            match_radius,
            dilation_radius,
        )
        for key in totals:
            totals[key] += int(image_result[key])
        for name in UNMATCHED_CLASSES:
            class_counts[name] += int(
                image_result["unmatched_component_count_by_class"][name]
            )
            class_pixels[name] += int(
                image_result["unmatched_component_pixels_by_class"][name]
            )
        for record in image_result["per_gt"]:
            per_gt.append({"identifier": identifier, **record})

    unmatched_pixels_total = totals["unmatched_pixels_total"]
    target_associated_pixels = (
        class_pixels["in_gt_fragment"]
        + class_pixels["near_gt_duplicate"]
        + class_pixels["attached_or_near_gt"]
    )
    formally_detected = [
        record for record in per_gt if record["formally_matched"]
    ]
    overlap_covered = [
        record
        for record in per_gt
        if record["overlapping_prediction_components"] > 0
    ]
    components_per_detected = [
        int(record["overlapping_prediction_components"])
        for record in formally_detected
    ]
    largest_fractions = [
        float(record["largest_fragment_fraction"])
        for record in overlap_covered
    ]
    split_target_count = sum(
        record["overlapping_prediction_components"] >= 2
        for record in per_gt
    )
    fragment_excess_total = sum(
        int(record["fragment_excess"]) for record in per_gt
    )
    result = {
        **totals,
        "unmatched_component_count_by_class": class_counts,
        "unmatched_component_pixels_by_class": class_pixels,
        "unmatched_pixels_in_gt": class_pixels["in_gt_fragment"],
        "unmatched_pixels_near_gt": (
            class_pixels["near_gt_duplicate"]
            + class_pixels["attached_or_near_gt"]
        ),
        "unmatched_pixels_background": class_pixels[
            "background_false_object"
        ],
        "fragment_fa_fraction": (
            target_associated_pixels / unmatched_pixels_total
            if unmatched_pixels_total
            else 0.0
        ),
        "background_fa_fraction": (
            class_pixels["background_false_object"]
            / unmatched_pixels_total
            if unmatched_pixels_total
            else 0.0
        ),
        "fragmented_gt_count": int(split_target_count),
        "split_target_count": int(split_target_count),
        "extra_fragments": int(fragment_excess_total),
        "fragment_excess_total": int(fragment_excess_total),
        "formally_detected_gt_count": len(formally_detected),
        "overlap_covered_gt_count": len(overlap_covered),
        "mean_components_per_detected_gt": (
            float(np.mean(components_per_detected))
            if components_per_detected
            else 0.0
        ),
        "p90_components_per_detected_gt": (
            float(np.quantile(components_per_detected, 0.9))
            if components_per_detected
            else 0.0
        ),
        "largest_fragment_fraction_mean": (
            float(np.mean(largest_fractions))
            if largest_fractions
            else 0.0
        ),
        "largest_fragment_fraction_p10": (
            float(np.quantile(largest_fractions, 0.1))
            if largest_fractions
            else 0.0
        ),
        "per_gt": per_gt,
    }
    if sum(class_counts.values()) != totals["unmatched_component_count"]:
        raise RuntimeError("Unmatched component taxonomy count does not close")
    if sum(class_pixels.values()) != unmatched_pixels_total:
        raise RuntimeError("Unmatched component taxonomy pixels do not close")
    if not math.isclose(
        result["fragment_fa_fraction"] + result["background_fa_fraction"],
        1.0 if unmatched_pixels_total else 0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Fragment/background Fa fractions do not close")
    return result


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    model.eval()
    criterion = nn.BCELoss(reduction="mean")
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    for images, masks, sizes, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        height = int(sizes[0, 0].item())
        width = int(sizes[0, 1].item())
        prediction = final_prediction(model(images))[:, :, :height, :width]
        target = masks[:, :, :height, :width]
        losses.append(
            float(criterion(prediction.float(), target.float()).item())
        )
        probabilities.append(prediction[0, 0].float().cpu().numpy())
        targets.append(target[0, 0].float().cpu().numpy())
    return probabilities, targets, losses


def metric_point(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    losses: Sequence[float],
    threshold: float,
    match_radius: float,
    tiny_area: int,
) -> Dict[str, Any]:
    accumulator = ValidationMetrics(threshold, match_radius, tiny_area)
    for probability, target, loss in zip(probabilities, targets, losses):
        accumulator.update(probability, target, loss)
    ready = json_ready(accumulator.compute())
    ready["threshold"] = float(threshold)
    return ready


def decorated_point(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    losses: Sequence[float],
    identifiers: Sequence[str],
    threshold: float,
    match_radius: float,
    tiny_area: int,
    dilation_radius: int,
) -> Dict[str, Any]:
    metrics = metric_point(
        probabilities,
        targets,
        losses,
        threshold,
        match_radius,
        tiny_area,
    )
    topology = aggregate_component_diagnostics(
        probabilities,
        targets,
        identifiers,
        threshold,
        match_radius,
        dilation_radius,
    )
    if topology["target_component_count"] != metrics["target_count"]:
        raise RuntimeError("Topology target count differs from formal metric")
    if (
        topology["matched_primary_component_count"]
        != metrics["matched_target_count"]
    ):
        raise RuntimeError("Topology matching differs from formal metric")
    if (
        topology["unmatched_component_count"]
        != metrics["unmatched_predicted_object_count"]
    ):
        raise RuntimeError("Topology unmatched count differs from formal metric")
    expected_unmatched_pixels = int(
        round(float(metrics["fa"]) * int(metrics["valid_pixel_count"]))
    )
    if topology["unmatched_pixels_total"] != expected_unmatched_pixels:
        raise RuntimeError("Topology unmatched pixels differ from Fa numerator")
    return {
        **metrics,
        "component_taxonomy": {
            key: value for key, value in topology.items() if key != "per_gt"
        },
        "gt_topology": {
            "fragmented_gt_count": topology["fragmented_gt_count"],
            "split_target_count": topology["split_target_count"],
            "extra_fragments": topology["extra_fragments"],
            "fragment_excess_total": topology["fragment_excess_total"],
            "formally_detected_gt_count": topology[
                "formally_detected_gt_count"
            ],
            "overlap_covered_gt_count": topology[
                "overlap_covered_gt_count"
            ],
            "mean_components_per_detected_gt": topology[
                "mean_components_per_detected_gt"
            ],
            "p90_components_per_detected_gt": topology[
                "p90_components_per_detected_gt"
            ],
            "largest_fragment_fraction_mean": topology[
                "largest_fragment_fraction_mean"
            ],
            "largest_fragment_fraction_p10": topology[
                "largest_fragment_fraction_p10"
            ],
            "per_gt": topology["per_gt"],
        },
    }


def threshold_key(value: float) -> str:
    return f"{float(value):.10g}"


def evaluate_mode(
    model: nn.Module,
    loader: DataLoader,
    identifiers: Sequence[str],
    device: torch.device,
    mode: str,
    fixed_thresholds: Sequence[float],
    fa_budgets: Sequence[float],
    match_radius: float,
    tiny_area: int,
    dilation_radius: int,
    include_budget_thresholds: bool,
) -> Dict[str, Any]:
    with temporary_counterfactual(model, mode) as provenance:
        probabilities, targets, losses = collect_predictions(
            model,
            loader,
            device,
        )
    for index, (probability, target, loss) in enumerate(
        zip(probabilities, targets, losses)
    ):
        if (
            not np.isfinite(probability).all()
            or not np.isfinite(target).all()
            or not math.isfinite(loss)
        ):
            raise ValueError(f"Non-finite inference output at validation index {index}")

    base_thresholds = sweep_base.threshold_grid(
        0.01,
        0.99,
        0.01,
        (0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999),
    )
    base_thresholds = sorted(
        set(base_thresholds).union(float(value) for value in fixed_thresholds)
    )
    thresholds, adaptive_provenance = adaptive_thresholds_closed_interval(
        probabilities,
        base_thresholds,
        0.1,
    )
    sweep_points = [
        metric_point(
            probabilities,
            targets,
            losses,
            threshold,
            match_radius,
            tiny_area,
        )
        for threshold in thresholds
    ]
    selected_budget_points = {
        threshold_key(budget): sweep_base.best_point_under_fa(
            sweep_points,
            budget,
        )
        for budget in fa_budgets
    }

    point_cache: Dict[float, Dict[str, Any]] = {}

    def topology_point(threshold: float) -> Dict[str, Any]:
        exact = float(threshold)
        if exact not in point_cache:
            point_cache[exact] = decorated_point(
                probabilities,
                targets,
                losses,
                identifiers,
                exact,
                match_radius,
                tiny_area,
                dilation_radius,
            )
        return point_cache[exact]

    fixed_points = {
        threshold_key(threshold): topology_point(threshold)
        for threshold in fixed_thresholds
    }
    budget_points: Dict[str, Any] = {}
    if include_budget_thresholds:
        for budget_key, point in selected_budget_points.items():
            budget_points[budget_key] = (
                topology_point(float(point["threshold"]))
                if point is not None
                else None
            )
    else:
        budget_points = selected_budget_points

    return {
        "mode": mode,
        "counterfactual_provenance": {
            **provenance,
            "state_sha256_after_restore": model_state_sha256(model),
            "state_restored_exactly": True,
            "zero_training": True,
        },
        "validation_count": len(probabilities),
        "fixed_threshold_points": fixed_points,
        "best_points_under_fa_budget": budget_points,
        "threshold_sweep": {
            "configuration": {
                "minimum": 0.01,
                "maximum": 0.99,
                "step": 0.01,
                "extras": [0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999],
                "tail_logit_step": 0.1,
                "closed_probability_interval": True,
                "fa_budgets": list(fa_budgets),
            },
            "provenance": adaptive_provenance,
            "points": sweep_points,
        },
    }


def formal_sweep_consistency(
    mode_result: Mapping[str, Any],
    formal_sweep: Mapping[str, Any],
) -> Dict[str, Any]:
    fixed = mode_result["fixed_threshold_points"].get("0.5")
    reference = formal_sweep.get("fixed_threshold_0_5")
    if not isinstance(fixed, dict) or not isinstance(reference, dict):
        raise ValueError("Cannot compare as-trained fixed threshold with formal sweep")
    deltas: Dict[str, Any] = {}
    exact_count_matches: Dict[str, bool] = {}
    for key in METRIC_KEYS:
        if key not in fixed or key not in reference:
            continue
        left, right = fixed[key], reference[key]
        if key.endswith("_count"):
            exact_count_matches[key] = left == right
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            deltas[key] = float(left) - float(right)
    return {
        "formal_sweep_checkpoint_sha256": formal_sweep.get(
            "checkpoint_sha256"
        ),
        "fixed_threshold_0_5_numeric_deltas_diagnostic_minus_formal": deltas,
        "fixed_threshold_0_5_exact_count_matches": exact_count_matches,
        "all_count_fields_match": all(exact_count_matches.values()),
        "max_abs_numeric_delta": max(
            (abs(value) for value in deltas.values()),
            default=0.0,
        ),
    }


def evaluate_job(
    job: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    device_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    artifacts = validate_job_artifacts(job)
    checkpoint = artifacts["checkpoint"]
    protocol = artifacts["protocol"]
    split = artifacts["split"]
    arguments = protocol["arguments"]

    model, model_metadata = build_clean_v6_model(
        str(job["variant"]),
        int(job["seed"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    loaded_state_sha256 = model_state_sha256(model)
    checkpoint_static_diagnostics = static_checkpoint_diagnostics(model)

    validation_ids = list(split["used_val_ids"])
    if args.max_validation_images is not None:
        validation_ids = validation_ids[: args.max_validation_images]
    dataset_dir = Path(arguments["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = (REPO_ROOT / dataset_dir).resolve()
    validation_set = ValidationSubset(
        dataset_dir / formal.DATASET,
        validation_ids,
        {
            key: float(value)
            for key, value in protocol["normalization"].items()
        },
    )
    loader = DataLoader(
        validation_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    match_radius = float(arguments["match_radius"])
    tiny_area = int(arguments["tiny_area"])
    modes: Dict[str, Any] = {}
    for mode in args.modes:
        modes[mode] = evaluate_mode(
            model=model,
            loader=loader,
            identifiers=validation_ids,
            device=device,
            mode=mode,
            fixed_thresholds=args.thresholds,
            fa_budgets=args.fa_budgets,
            match_radius=match_radius,
            tiny_area=tiny_area,
            dilation_radius=args.dilation_radius,
            include_budget_thresholds=args.include_budget_thresholds,
        )

    as_trained_consistency = None
    if (
        "as_trained" in modes
        and args.max_validation_images is None
        and 0.5 in args.thresholds
    ):
        as_trained_consistency = formal_sweep_consistency(
            modes["as_trained"],
            artifacts["formal_sweep"],
        )
    input_sha256_after = {
        label: file_sha256(path)
        for label, path in artifacts["paths"].items()
    }
    if input_sha256_after != artifacts["input_sha256"]:
        raise RuntimeError("A formal input artifact changed during diagnosis")
    if model_state_sha256(model) != loaded_state_sha256:
        raise RuntimeError("Loaded model state changed during diagnosis")

    payload = {
        "schema": SCHEMA,
        "diagnostic_scope": (
            "frozen_internal_validation_counterfactual_only"
        ),
        "formal_gate_replacement": False,
        "checkpoint_reselection_permitted": False,
        "training_performed": False,
        "official_test_accessed": False,
        "complete_validation_split": args.max_validation_images is None,
        "variant": job["variant"],
        "seed": job["seed"],
        "checkpoint_role": job["role"],
        "checkpoint_epoch": checkpoint.get("epoch"),
        "run_directory": str(Path(job["run_dir"]).resolve()),
        "checkpoint": str(Path(job["checkpoint"]).resolve()),
        "formal_sweep": str(Path(job["formal_sweep"]).resolve()),
        "output": str(Path(job["output"]).resolve()),
        "model_metadata": json_ready(model_metadata),
        "loaded_model_state_sha256": loaded_state_sha256,
        "checkpoint_static_diagnostics": checkpoint_static_diagnostics,
        "validation": {
            "dataset": formal.DATASET,
            "validation_count": len(validation_ids),
            "formal_validation_count": split["used_val_count"],
            "validation_ids": validation_ids,
            "validation_split_sha256": split["hashes"]["used_val_sha256"],
            "match_radius": match_radius,
            "tiny_area": tiny_area,
            "component_connectivity": 2,
            "dilation_radius_pixels": args.dilation_radius,
            "prediction_comparison": "probability > threshold",
        },
        "component_taxonomy_contract": {
            "scope": "unmatched_predicted_components_only",
            "priority_order": list(UNMATCHED_CLASSES),
            "definitions": TAXONOMY_DEFINITIONS,
            "whole_component_area_assignment": True,
            "mutually_exclusive": True,
            "exhaustive": True,
            "fragment_fa_fraction": (
                "(in_gt_fragment + near_gt_duplicate + "
                "attached_or_near_gt unmatched component pixels) / "
                "all unmatched component pixels"
            ),
            "largest_fragment_fraction": (
                "largest predicted-component intersection area inside one GT / "
                "all predicted-positive pixels inside that GT"
            ),
            "fragmented_gt_count": (
                "number of GT components overlapped by at least two predicted "
                "components"
            ),
            "extra_fragments": (
                "sum over GT of max(0, overlapping_prediction_components - 1)"
            ),
        },
        "device": dict(device_provenance),
        "fixed_thresholds": list(args.thresholds),
        "fa_budgets": list(args.fa_budgets),
        "modes": modes,
        "as_trained_formal_sweep_consistency": as_trained_consistency,
        "input_sha256_before": artifacts["input_sha256"],
        "input_sha256_after": input_sha256_after,
        "formal_inputs_unchanged": True,
        "limitations": [
            "These are immediate zero-training counterfactuals, not retrained models.",
            "Independent Fa-budget point selection may choose different thresholds "
            "for different modes; fixed-threshold comparisons are the primary "
            "mechanism screen.",
            "Component taxonomy is descriptive and does not replace Pd, Fa, mIoU "
            "or formal Gates A-E.",
            "A two-seed internal validation diagnosis cannot establish a stability "
            "or causal mechanism claim.",
        ],
    }
    return json_ready(payload)


def write_json(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite diagnostic output: {path}; pass --overwrite"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _point_delta(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate_topology = candidate["gt_topology"]
    reference_topology = reference["gt_topology"]
    return {
        "pd": float(candidate["pd"]) - float(reference["pd"]),
        "fa": float(candidate["fa"]) - float(reference["fa"]),
        "miou": float(candidate["miou"]) - float(reference["miou"]),
        "fragmented_gt_count": int(
            candidate_topology["fragmented_gt_count"]
        )
        - int(reference_topology["fragmented_gt_count"]),
        "extra_fragments": int(candidate_topology["extra_fragments"])
        - int(reference_topology["extra_fragments"]),
        "pixel_precision": float(candidate["pixel_precision"])
        - float(reference["pixel_precision"]),
    }


def point_supports_context(delta: Mapping[str, Any]) -> bool:
    pd_not_worse = float(delta["pd"]) >= -1e-12
    task_improved = (
        float(delta["fa"]) < -1e-15
        or float(delta["miou"]) > 1e-12
    )
    topology_not_worse = (
        int(delta["fragmented_gt_count"]) <= 0
        and int(delta["extra_fragments"]) <= 0
    )
    topology_strictly_improved = (
        int(delta["fragmented_gt_count"]) < 0
        or int(delta["extra_fragments"]) < 0
    )
    return (
        pd_not_worse
        and task_improved
        and topology_not_worse
        and topology_strictly_improved
    )


def build_decision_inputs(
    payloads: Mapping[tuple[str, int, str], Mapping[str, Any]],
    fixed_thresholds: Sequence[float],
) -> Dict[str, Any]:
    """Return a conservative screening signal; never replace formal gates."""

    required_keys = {
        (formal.PRIMARY_VARIANT, 3407, role)
        for role in ("pd_primary", "miou_primary")
    }
    required_modes = set(DEFAULT_MODES)
    default_thresholds_present = set(DEFAULT_FIXED_THRESHOLDS).issubset(
        set(float(value) for value in fixed_thresholds)
    )
    complete_inputs = (
        required_keys.issubset(payloads)
        and default_thresholds_present
        and all(
            required_modes.issubset(payloads[key]["modes"])
            and payloads[key]["complete_validation_split"] is True
            for key in required_keys
            if key in payloads
        )
    )
    rule = {
        "primary_comparison": (
            "same fixed threshold; independently selected Fa-budget points are "
            "reported but not used by this screen"
        ),
        "required_checkpoints": [
            "tpd_clean_v6_full seed3407 pd_primary",
            "tpd_clean_v6_full seed3407 miou_primary",
        ],
        "checkpoint_support": (
            "at least one preregistered fixed point has non-worse Pd, lower Fa "
            "or higher mIoU, non-worse fragmented_gt_count and extra_fragments, "
            "and a strict improvement in at least one of those topology counts"
        ),
        "context_support": "checkpoint_support must hold for both checkpoints",
        "formal_gate_replacement": False,
        "causal_claim_permitted": False,
        "ner_authorization_permitted": False,
    }
    if not complete_inputs:
        return {
            "status": "INCOMPLETE_DIAGNOSTIC",
            "dch_supported_by_context_screen": False,
            "rule": rule,
            "comparisons": {},
        }

    comparisons: Dict[str, Any] = {}
    context_checkpoint_support: list[bool] = []
    residual_checkpoint_support: list[bool] = []
    for key in sorted(required_keys):
        payload = payloads[key]
        reference = payload["modes"]["as_trained"]["fixed_threshold_points"]
        role_comparison: Dict[str, Any] = {}
        for mode in (
            "same_weights_context_off",
            "same_weights_residual_off",
        ):
            candidate = payload["modes"][mode]["fixed_threshold_points"]
            points: Dict[str, Any] = {}
            for threshold in DEFAULT_FIXED_THRESHOLDS:
                key_name = threshold_key(threshold)
                delta = _point_delta(
                    candidate[key_name],
                    reference[key_name],
                )
                points[key_name] = {
                    "delta_candidate_minus_as_trained": delta,
                    "supports_registered_condition": point_supports_context(
                        delta
                    ),
                }
            role_comparison[mode] = {
                "points": points,
                "checkpoint_support": any(
                    point["supports_registered_condition"]
                    for point in points.values()
                ),
            }
        context_checkpoint_support.append(
            role_comparison["same_weights_context_off"][
                "checkpoint_support"
            ]
        )
        residual_checkpoint_support.append(
            role_comparison["same_weights_residual_off"][
                "checkpoint_support"
            ]
        )
        comparisons[key[2]] = role_comparison

    context_supported = all(context_checkpoint_support)
    residual_supported = all(residual_checkpoint_support)
    if context_supported:
        status = "DCH_CONTEXT_SCREEN_SUPPORTED"
    elif residual_supported:
        status = "PHASE_OR_OTHER_SIGNAL"
    else:
        status = "INCONCLUSIVE"
    return {
        "status": status,
        "dch_supported_by_context_screen": context_supported,
        "residual_or_other_signal": residual_supported,
        "rule": rule,
        "comparisons": comparisons,
        "interpretation_boundary": (
            "This status only prioritizes the next K/C/S implementation test. "
            "It is not a causal conclusion and does not alter Gates A-E."
        ),
    }


def preflight(args: argparse.Namespace) -> Dict[str, Any]:
    jobs = requested_jobs(args)
    inspected = []
    for job in jobs:
        artifacts = validate_job_artifacts(job)
        inspected.append(
            {
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in job.items()
                },
                "input_sha256": artifacts["input_sha256"],
                "ready": True,
            }
        )
    return {
        "schema": MATRIX_SCHEMA,
        "mode": "preflight",
        "jobs": inspected,
        "job_count": len(inspected),
        "training_performed": False,
        "outputs_written": 0,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    determinism = configure_formal_inference_determinism(args.device)
    device_provenance = _bind_requested_gpu(
        args.device,
        args.physical_gpu,
    )
    device_provenance["determinism"] = determinism
    device = torch.device(args.device)
    payloads: Dict[tuple[str, int, str], Dict[str, Any]] = {}
    outputs: list[str] = []
    for job in requested_jobs(args):
        output_path = Path(job["output"])
        payload = evaluate_job(
            job,
            args,
            device,
            device_provenance,
        )
        write_json(output_path, payload, args.overwrite)
        payloads[
            (str(job["variant"]), int(job["seed"]), str(job["role"]))
        ] = payload
        outputs.append(str(output_path.resolve()))
        print(
            f"COMPLETE variant={job['variant']} seed={job['seed']} "
            f"role={job['role']} output={output_path}",
            flush=True,
        )

    decision_inputs = build_decision_inputs(payloads, args.thresholds)
    matrix = json_ready(
        {
            "schema": MATRIX_SCHEMA,
            "mode": "run",
            "diagnostic_scope": (
                "frozen_internal_validation_counterfactual_only"
            ),
            "training_performed": False,
            "official_test_accessed": False,
            "formal_gate_replacement": False,
            "device": device_provenance,
            "results_root": str(args.results_root.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "outputs": outputs,
            "output_count": len(outputs),
            "requested_variants": list(args.variant),
            "requested_seeds": list(args.seed),
            "requested_checkpoint_roles": list(args.checkpoint_role),
            "requested_modes": list(args.modes),
            "fixed_thresholds": list(args.thresholds),
            "fa_budgets": list(args.fa_budgets),
            "complete_validation_split": args.max_validation_images is None,
            "decision_inputs": decision_inputs,
        }
    )
    matrix_path = args.output_dir / "matrix_summary.json"
    write_json(matrix_path, matrix, args.overwrite)
    return matrix


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose frozen TPD-Clean-v6 checkpoint failure modes"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=formal.DEFAULT_CANDIDATE_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        choices=formal.VARIANTS,
        default=list(formal.VARIANTS),
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        choices=formal.SEEDS,
        default=list(formal.SEEDS),
    )
    parser.add_argument(
        "--checkpoint-role",
        nargs="+",
        choices=tuple(formal.ROLE_SPECS),
        default=list(formal.ROLE_SPECS),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=list(DEFAULT_MODES),
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_FIXED_THRESHOLDS),
    )
    parser.add_argument(
        "--fa-budgets",
        nargs="+",
        type=float,
        default=list(DEFAULT_FA_BUDGETS),
    )
    parser.add_argument(
        "--include-budget-thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dilation-radius", type=int, default=3)
    parser.add_argument(
        "--max-validation-images",
        type=int,
        default=None,
        help="Smoke-only prefix length; omitted means all 133 validation images",
    )
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--physical-gpu", choices=tuple(POSTPROCESS_GPUS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.results_root = args.results_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.thresholds = _finite_thresholds(args.thresholds)
    args.fa_budgets = _finite_budgets(args.fa_budgets)
    args.variant = tuple(dict.fromkeys(args.variant))
    args.seed = tuple(dict.fromkeys(args.seed))
    args.checkpoint_role = tuple(dict.fromkeys(args.checkpoint_role))
    args.modes = tuple(dict.fromkeys(args.modes))
    if args.dilation_radius < 1:
        parser.error("--dilation-radius must be >= 1")
    if (
        args.max_validation_images is not None
        and args.max_validation_images < 1
    ):
        parser.error("--max-validation-images must be >= 1")
    if args.device == "cuda:0" and args.physical_gpu is None:
        parser.error("CUDA diagnosis requires --physical-gpu 2 or 3")
    if args.device == "cpu" and args.physical_gpu is not None:
        parser.error("--physical-gpu is only valid with --device cuda:0")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.preflight:
        print(
            json.dumps(
                preflight(args),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    matrix = run(args)
    print(
        f"MATRIX_COMPLETE outputs={matrix['output_count']} "
        f"decision={matrix['decision_inputs']['status']} "
        f"summary={args.output_dir / 'matrix_summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
