#!/usr/bin/env python3
"""Run the frozen PBDR-V1 zero-training audit on one Current checkpoint.

One unchanged model forward captures the raw ``q4``, ``out`` and ``d0``
tensors.  The identity point, four authorization candidates and the g=1
oracle are then evaluated from that shared forward-local state.  The script
never writes a checkpoint or a tensor/probability cache.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import analyze_ner_stage2_mask_knockout_v1 as component_core  # noqa: E402
from analysis import analyze_three_dataset_dorf_v1 as dorf_core  # noqa: E402
from analysis import analyze_three_dataset_ner_l4_tpr_v1 as tpr_screen  # noqa: E402
from analysis import diagnose_tpd_clean_v6_fragmentation as topology_core  # noqa: E402
from experiments import evaluate_three_dataset_tss_off_seed42_v1 as adapter  # noqa: E402
from experiments import evaluate_three_dataset_v2 as metric_core  # noqa: E402
from experiments import four_dataset_models_seed42_v1 as model_builder  # noqa: E402
from experiments import ner_l4_tpr_strict_migration_v1 as input_core  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import (  # noqa: E402
    train_four_dataset_original_final_seed42_exact_v1 as training_engine,
)
from model.tpd_ner_l4_target_protected_reallocation import (  # noqa: E402
    FORMAL_L4_PROTECTION_DILATION_KERNEL,
    FORMAL_L4_TAIL_Z_THRESHOLD,
    FORMAL_Q4_RELAY_CHANNELS,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (  # noqa: E402
    relay_spatial_tail_support,
)


SCHEMA = "sctransnet_three_dataset_pbdr_zero_training_v1/v1"
SEED = 42
FIXED_THRESHOLD = 0.5
SWEEP_THRESHOLDS = (0.5, 1.0)
DATASETS = tuple(data_protocol.DATASETS)
CHECKPOINT_ROLES = tuple(metric_core.CHECKPOINT_ROLES)
G_EIGHTHS_ORDER = (0, 1, 2, 4, 6, 8)
AUTHORIZATION_G_EIGHTHS = (1, 2, 4, 6)
IDENTITY_G_EIGHTHS = 0
ORACLE_G_EIGHTHS = 8
MODE_BY_G_EIGHTHS = {
    0: "current_g0",
    1: "pbdr_g0125",
    2: "pbdr_g0250",
    4: "pbdr_g0500",
    6: "pbdr_g0750",
    8: "oracle_g1000",
}
MODE_ORDER = tuple(MODE_BY_G_EIGHTHS[value] for value in G_EIGHTHS_ORDER)
CURRENT_MODE = MODE_BY_G_EIGHTHS[IDENTITY_G_EIGHTHS]
ORACLE_MODE = MODE_BY_G_EIGHTHS[ORACLE_G_EIGHTHS]
FLOAT_EQ_ATOL = 1e-12
FLOAT_EQ_RTOL = 0.0
HISTORICAL_COUNT_FIELDS = (
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "unmatched_predicted_pixels",
    "valid_pixel_count",
)

DEFAULT_IDENTITY_MANIFEST = tpr_screen.DEFAULT_IDENTITY_MANIFEST
FROZEN_IDENTITY_MANIFEST_SHA256 = tpr_screen.FROZEN_IDENTITY_MANIFEST_SHA256
DEFAULT_DORF_INPUT_MANIFEST = dorf_core.DEFAULT_INPUT_MANIFEST
FROZEN_DORF_INPUT_MANIFEST_SHA256 = dorf_core.FROZEN_INPUT_MANIFEST_SHA256
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/three_dataset_pbdr_zero_training_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def configure_inference_math() -> dict[str, Any]:
    """Freeze the historical Current checkpoint-selection math contract."""

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    contract = {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    _require(
        contract
        == {
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms": True,
        },
        "inference math contract differs",
    )
    return contract


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def file_sha256(path: Path) -> str:
    return input_core.file_sha256(Path(path))


def historical_reference_drift_audit(
    observed: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    expected_background_false_positive_pixels: int,
) -> dict[str, Any]:
    """Report old-evaluation drift without changing same-forward PBDR gates."""

    comparisons: dict[str, Any] = {}
    all_exact = True
    all_within_dorf_tolerance = True
    exact_count_fields = True
    for key in sorted(reference):
        if key not in observed:
            continue
        actual = observed[key]
        expected = reference[key]
        if key in HISTORICAL_COUNT_FIELDS:
            tolerance = 0.0
            exact = actual == expected
            within = exact
            difference: int | float | None = abs(int(actual) - int(expected))
            exact_count_fields = exact_count_fields and exact
        elif expected is None:
            tolerance = 0.0
            exact = actual is None
            within = exact
            difference = None
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            tolerance = (
                1e-4
                if key
                in {
                    "miou",
                    "niou",
                    "pixel_precision",
                    "pixel_recall",
                    "pixel_f1",
                }
                else 1e-7
                if key == "test_loss"
                else 1e-15
            )
            exact = float(actual) == float(expected)
            difference = abs(float(actual) - float(expected))
            within = math.isclose(
                float(actual),
                float(expected),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        else:
            tolerance = 0.0
            exact = actual == expected
            within = exact
            difference = None
        all_exact = all_exact and exact
        all_within_dorf_tolerance = all_within_dorf_tolerance and within
        comparisons[key] = {
            "observed": actual,
            "historical": expected,
            "absolute_difference": difference,
            "dorf_absolute_tolerance": tolerance,
            "exact": exact,
            "within_dorf_tolerance": within,
        }
    observed_background = int(observed["background_false_positive_pixels"])
    expected_background = int(expected_background_false_positive_pixels)
    background_exact = observed_background == expected_background
    return {
        "same_forward_authorization_gate": False,
        "scope": "descriptive_bound_historical_execution_drift",
        "historical_exact": all_exact and background_exact,
        "historical_within_frozen_dorf_tolerance": (
            all_within_dorf_tolerance and background_exact
        ),
        "historical_count_fields_exact": exact_count_fields,
        "background_false_positive_pixels_exact": background_exact,
        "observed_background_false_positive_pixels": observed_background,
        "historical_background_false_positive_pixels": expected_background,
        "background_false_positive_pixel_delta": (
            observed_background - expected_background
        ),
        "comparisons": comparisons,
        "reason_not_hard_gate": (
            "T1/T2 compare every routed point with the exact g0 output from "
            "the same forward and frozen uniform inference-math contract"
        ),
    }


def _source_sha256() -> dict[str, str]:
    sources = {
        "analysis/analyze_three_dataset_pbdr_zero_training_v1.py": Path(__file__),
        "analysis/analyze_three_dataset_dorf_v1.py": Path(dorf_core.__file__),
        "analysis/analyze_three_dataset_ner_l4_tpr_v1.py": Path(
            tpr_screen.__file__
        ),
        "analysis/analyze_ner_stage2_mask_knockout_v1.py": Path(
            component_core.__file__
        ),
        "analysis/diagnose_tpd_clean_v6_fragmentation.py": Path(
            topology_core.__file__
        ),
        "experiments/evaluate_three_dataset_v2.py": Path(metric_core.__file__),
        "experiments/evaluate_three_dataset_tss_off_seed42_v1.py": Path(
            adapter.__file__
        ),
        "experiments/four_dataset_models_seed42_v1.py": Path(
            model_builder.__file__
        ),
        "experiments/ner_l4_tpr_strict_migration_v1.py": Path(
            input_core.__file__
        ),
        "experiments/three_dataset_v2_protocol.py": Path(data_protocol.__file__),
        "model/tpd_ner_l4_target_protected_reallocation.py": REPO_ROOT
        / "model/tpd_ner_l4_target_protected_reallocation.py",
        "model/tpd_ner_v8_mprs_dch_v4_tail_aware.py": REPO_ROOT
        / "model/tpd_ner_v8_mprs_dch_v4_tail_aware.py",
    }
    return {
        relative: file_sha256(path.resolve(strict=True))
        for relative, path in sorted(sources.items())
    }


def point_kind(g_eighths: int) -> str:
    _require(g_eighths in G_EIGHTHS_ORDER, "unknown PBDR gate")
    if g_eighths == IDENTITY_G_EIGHTHS:
        return "identity"
    if g_eighths == ORACLE_G_EIGHTHS:
        return "oracle"
    return "authorization_candidate"


def route_with_gate(
    z_out: torch.Tensor,
    z_d0: torch.Tensor,
    protection: torch.Tensor,
    g_eighths: int,
) -> torch.Tensor:
    """Apply the preregistered PBDR raw-logit formula."""

    _require(g_eighths in G_EIGHTHS_ORDER, "gate is outside the frozen grid")
    _require(z_out.shape == z_d0.shape, "out/d0 shapes differ")
    _require(z_out.ndim == 4 and z_out.shape[1] == 1, "logits must be Bx1xHxW")
    _require(
        tuple(protection.shape) == tuple(z_out.shape),
        "protection/logit shapes differ",
    )
    _require(
        z_out.device == z_d0.device == protection.device,
        "routing tensors are on different devices",
    )
    _require(
        z_out.dtype == z_d0.dtype == protection.dtype,
        "routing tensors have different dtypes",
    )
    if g_eighths == 0:
        return z_out
    if g_eighths == ORACLE_G_EIGHTHS:
        return torch.where(
            protection.bool(),
            torch.maximum(z_out, z_d0),
            torch.minimum(z_out, z_d0),
        )
    disagreement = z_d0 - z_out
    target_rescue = protection * F.relu(disagreement)
    background_suppression = (1.0 - protection) * F.relu(-disagreement)
    gate = z_out.new_tensor(g_eighths / 8.0)
    routed = z_out + gate * (target_rescue - background_suppression)
    _require(bool(torch.isfinite(routed).all()), "routed logits are non-finite")
    return routed


def build_protection(
    q4: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    """Reproduce the frozen detached q4 tail-protection contract."""

    _require(
        q4.ndim == 4 and q4.shape[1] == FORMAL_Q4_RELAY_CHANNELS,
        "q4 must be Bx8xHxW",
    )
    _require(q4.is_floating_point(), "q4 must be floating point")
    _require(bool(torch.isfinite(q4).all()), "q4 contains non-finite values")
    with torch.no_grad():
        tail = relay_spatial_tail_support(
            q4.detach(),
            z_threshold=FORMAL_L4_TAIL_Z_THRESHOLD,
        )
        binary = tail.gt(0.0).to(dtype=q4.dtype)
        protection = F.max_pool2d(
            binary,
            kernel_size=FORMAL_L4_PROTECTION_DILATION_KERNEL,
            stride=1,
            padding=FORMAL_L4_PROTECTION_DILATION_KERNEL // 2,
        )
        protection = F.interpolate(
            protection,
            size=output_size,
            mode="nearest",
        )
    protection = protection.detach()
    _require(not protection.requires_grad, "protection must be detached")
    _require(
        set(float(value) for value in torch.unique(protection)) <= {0.0, 1.0},
        "protection must be binary",
    )
    return protection


class RawQ4D0OutHookCapture:
    """Capture q4/out/d0 exactly once from one unchanged model forward."""

    def __init__(self, model: nn.Module) -> None:
        self.out_module = getattr(model, "outc", None)
        self.d0_module = getattr(model, "outconv", None)
        relay = getattr(model, "tpd_ner", None)
        fusions = getattr(relay, "fusions", None)
        self.q4_module = fusions["4"] if fusions is not None and "4" in fusions else None
        for module, name in (
            (self.out_module, "outc"),
            (self.d0_module, "outconv"),
            (self.q4_module, "tpd_ner.fusions[4]"),
        ):
            _require(isinstance(module, nn.Module), f"model lacks {name}")
        self._modules = {
            "q4": self.q4_module,
            "out": self.out_module,
            "d0": self.d0_module,
        }
        self._hook_ids_before = {
            name: tuple(module._forward_hooks)
            for name, module in self._modules.items()
        }
        self._handles: list[Any] = []
        self._active = False
        self._current: dict[str, torch.Tensor] = {}
        self.total_counts = {name: 0 for name in self._modules}
        self.batch_count = 0
        self.temporary_hooks_restored = False

    def _hook(self, name: str):
        def record(_module: nn.Module, _inputs: Any, output: Any) -> None:
            _require(self._active, f"{name} hook fired outside active batch")
            _require(name not in self._current, f"{name} executed twice in one batch")
            _require(isinstance(output, torch.Tensor), f"{name} output is not a tensor")
            _require(output.ndim == 4, f"{name} output is not BCHW")
            _require(bool(torch.isfinite(output).all()), f"{name} output is non-finite")
            self._current[name] = output
            self.total_counts[name] += 1

        return record

    def __enter__(self) -> "RawQ4D0OutHookCapture":
        _require(not self._handles, "capture hooks are already installed")
        self._handles = [
            module.register_forward_hook(self._hook(name))
            for name, module in self._modules.items()
        ]
        return self

    def begin_batch(self) -> None:
        _require(not self._active, "previous capture batch is active")
        self._active = True
        self._current = {}

    def finish_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _require(self._active, "no active capture batch")
        self._active = False
        _require(set(self._current) == set(self._modules), "capture tensor set differs")
        q4 = self._current["q4"]
        z_out = self._current["out"]
        z_d0 = self._current["d0"]
        _require(z_out.shape == z_d0.shape, "captured out/d0 shapes differ")
        _require(q4.shape[0] == z_out.shape[0], "captured batch sizes differ")
        _require(q4.shape[1] == FORMAL_Q4_RELAY_CHANNELS, "captured q4 width differs")
        _require(
            q4.device == z_out.device == z_d0.device,
            "captured devices differ",
        )
        _require(q4.dtype == z_out.dtype == z_d0.dtype, "captured dtypes differ")
        self.batch_count += 1
        self._current = {}
        return q4, z_out, z_d0

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self._active = False
        self._current = {}
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.temporary_hooks_restored = all(
            tuple(module._forward_hooks) == self._hook_ids_before[name]
            for name, module in self._modules.items()
        )


def _ratio(numerator: int, denominator: int) -> tuple[float | None, bool]:
    _require(numerator >= 0 and denominator >= 0, "ratio counts must be non-negative")
    if denominator == 0:
        return None, True
    return numerator / denominator, False


def _compact_topology(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    identifiers: Sequence[str],
) -> dict[str, Any]:
    full = topology_core.aggregate_component_diagnostics(
        probabilities,
        targets,
        identifiers,
        FIXED_THRESHOLD,
        metric_core.MATCH_RADIUS,
        3,
    )
    keys = (
        "predicted_component_count",
        "target_component_count",
        "matched_primary_component_count",
        "unmatched_component_count",
        "unmatched_pixels_total",
        "unmatched_component_count_by_class",
        "unmatched_component_pixels_by_class",
        "fragmented_gt_count",
        "extra_fragments",
        "fragment_fa_fraction",
        "background_fa_fraction",
    )
    return {
        **{key: full[key] for key in keys},
        "per_gt_persisted": False,
        "dilation_radius": 3,
    }


def _new_signal_totals() -> dict[str, int | float]:
    return {
        "current_missed_gt_object_count": 0,
        "current_missed_gt_pixel_count": 0,
        "missed_gt_pixels_with_d0_gt_out": 0,
        "missed_gt_objects_with_protected_pixels": 0,
        "missed_gt_objects_with_d0_gt_out_pixels": 0,
        "missed_gt_objects_with_protected_rescue_pixels": 0,
        "protected_rescue_pixels_inside_missed_gt": 0,
        "target_rescue_nonzero_pixels_all_valid": 0,
        "target_rescue_abs_sum_all_valid": 0.0,
        "current_unmatched_predicted_object_count": 0,
        "current_unmatched_predicted_pixel_count": 0,
        "current_unmatched_pixels_with_p_zero": 0,
        "current_unmatched_pixels_with_out_gt_d0": 0,
        "unmatched_fp_pixels_with_unprotected_suppression": 0,
        "background_suppression_nonzero_pixels_all_valid": 0,
        "background_suppression_abs_sum_all_valid": 0.0,
    }


@torch.inference_mode()
def analyze_loaded_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    expected_identifiers: Sequence[str],
) -> dict[str, Any]:
    """Evaluate all six fixed gates from one forward per test image."""

    model.eval()
    model.mode = "test"
    _require(not model.training, "PBDR audit requires model.eval()")
    state_before = component_core.module_state_sha256(model)
    probabilities: dict[str, list[np.ndarray]] = {mode: [] for mode in MODE_ORDER}
    losses: dict[str, list[float]] = {mode: [] for mode in MODE_ORDER}
    routing_totals: dict[str, dict[str, int | float]] = {
        mode: {
            "actual_target_rescue_nonzero_pixels": 0,
            "actual_target_rescue_abs_sum": 0.0,
            "actual_background_suppression_nonzero_pixels": 0,
            "actual_background_suppression_abs_sum": 0.0,
            "prediction_changed_pixel_count_vs_current": 0,
        }
        for mode in MODE_ORDER
    }
    targets: list[np.ndarray] = []
    identifiers: list[str] = []
    criterion = nn.BCELoss(reduction="mean")
    tensor_stream_sha = hashlib.sha256()
    signal_totals = _new_signal_totals()
    protection_counts = {
        "protected_pixel_count": 0,
        "valid_pixel_count": 0,
        "gt_pixel_count": 0,
        "protected_gt_pixel_count": 0,
        "background_pixel_count": 0,
        "protected_background_pixel_count": 0,
        "current_unmatched_predicted_pixel_count": 0,
        "protected_current_unmatched_predicted_pixel_count": 0,
    }
    returned_probability_exact = True
    g0_logit_exact = True
    oracle_formula_exact = True
    batch_count = 0

    with RawQ4D0OutHookCapture(model) as capture:
        for images, masks, sizes, sample_ids in loader:
            _require(
                int(images.shape[0]) == int(masks.shape[0]) == 1,
                "PBDR audit requires batch_size=1",
            )
            _require(
                isinstance(sample_ids, (tuple, list)) and len(sample_ids) == 1,
                "PBDR audit requires one sample ID per batch",
            )
            identifier = str(sample_ids[0])
            height, width = metric_core._extract_hw(sizes)
            images_device = images.to(device, non_blocking=device.type == "cuda")
            masks_device = masks.to(device, non_blocking=device.type == "cuda")
            capture.begin_batch()
            returned = model(images_device)
            q4, z_out, z_d0 = capture.finish_batch()
            returned_probability = metric_core._final_prediction(returned)
            expected_probability = torch.sigmoid(z_out)
            this_return_exact = torch.equal(returned_probability, expected_probability)
            returned_probability_exact = returned_probability_exact and this_return_exact
            _require(this_return_exact, "model return is not sigmoid(raw out)")
            protection = build_protection(q4, tuple(z_out.shape[-2:]))
            _require(protection.shape == z_out.shape, "protection output shape differs")
            g0_logit_exact = g0_logit_exact and torch.equal(
                route_with_gate(z_out, z_d0, protection, 0),
                z_out,
            )
            oracle = route_with_gate(z_out, z_d0, protection, 8)
            oracle_expected = torch.where(
                protection.bool(),
                torch.maximum(z_out, z_d0),
                torch.minimum(z_out, z_d0),
            )
            oracle_formula_exact = oracle_formula_exact and torch.equal(
                oracle, oracle_expected
            )

            for name, tensor in (("q4", q4), ("z_out", z_out), ("z_d0", z_d0)):
                dorf_core._update_tensor_digest(tensor_stream_sha, name, tensor)

            valid_slice = (slice(None), slice(None), slice(0, height), slice(0, width))
            target = masks_device[valid_slice]
            current_probability = returned_probability[valid_slice]
            protection_valid = protection[valid_slice]
            out_valid = z_out[valid_slice]
            d0_valid = z_d0[valid_slice]
            disagreement = d0_valid - out_valid
            target_rescue = protection_valid * F.relu(disagreement)
            background_suppression = (1.0 - protection_valid) * F.relu(-disagreement)
            target_rescue_mask = target_rescue.gt(0.0)
            background_suppression_mask = background_suppression.gt(0.0)
            target_array = target[0, 0].float().cpu().numpy().copy()
            current_array = current_probability[0, 0].float().cpu().numpy().copy()
            protection_array = protection_valid[0, 0].bool().cpu().numpy().copy()
            rescue_array = target_rescue_mask[0, 0].cpu().numpy().copy()
            suppression_array = background_suppression_mask[0, 0].cpu().numpy().copy()
            d0_gt_out_array = disagreement[0, 0].gt(0.0).cpu().numpy().copy()
            out_gt_d0_array = disagreement[0, 0].lt(0.0).cpu().numpy().copy()
            target_binary = target_array > 0.5
            prediction_binary = current_array > FIXED_THRESHOLD

            predicted_regions, target_regions, matched_targets, matched_predictions = (
                component_core._match_regions(prediction_binary, target_binary)
            )
            missed_targets = [
                (index, region)
                for index, region in enumerate(target_regions)
                if index not in matched_targets
            ]
            signal_totals["current_missed_gt_object_count"] += len(missed_targets)
            for _index, region in missed_targets:
                rows, columns = region.coords.T
                signal_totals["current_missed_gt_pixel_count"] += int(region.area)
                has_protection = bool(protection_array[rows, columns].any())
                has_d0_gt_out = bool(d0_gt_out_array[rows, columns].any())
                signal_totals["missed_gt_pixels_with_d0_gt_out"] += int(
                    d0_gt_out_array[rows, columns].sum()
                )
                rescue_pixels = int(rescue_array[rows, columns].sum())
                signal_totals["missed_gt_objects_with_protected_pixels"] += int(
                    has_protection
                )
                signal_totals["missed_gt_objects_with_d0_gt_out_pixels"] += int(
                    has_d0_gt_out
                )
                signal_totals[
                    "missed_gt_objects_with_protected_rescue_pixels"
                ] += int(rescue_pixels > 0)
                signal_totals["protected_rescue_pixels_inside_missed_gt"] += (
                    rescue_pixels
                )

            unmatched_mask = np.zeros_like(prediction_binary, dtype=bool)
            unmatched_object_count = 0
            for index, region in enumerate(predicted_regions):
                if index in matched_predictions:
                    continue
                rows, columns = region.coords.T
                unmatched_mask[rows, columns] = True
                unmatched_object_count += 1
            unmatched_pixels = int(unmatched_mask.sum())
            signal_totals["current_unmatched_predicted_object_count"] += (
                unmatched_object_count
            )
            signal_totals["current_unmatched_predicted_pixel_count"] += unmatched_pixels
            signal_totals["current_unmatched_pixels_with_p_zero"] += int(
                np.logical_and(unmatched_mask, ~protection_array).sum()
            )
            signal_totals["current_unmatched_pixels_with_out_gt_d0"] += int(
                np.logical_and(unmatched_mask, out_gt_d0_array).sum()
            )
            signal_totals[
                "unmatched_fp_pixels_with_unprotected_suppression"
            ] += int(np.logical_and(unmatched_mask, suppression_array).sum())

            valid_pixels = int(target_binary.size)
            gt_pixels = int(target_binary.sum())
            background_pixels = valid_pixels - gt_pixels
            protection_counts["protected_pixel_count"] += int(protection_array.sum())
            protection_counts["valid_pixel_count"] += valid_pixels
            protection_counts["gt_pixel_count"] += gt_pixels
            protection_counts["protected_gt_pixel_count"] += int(
                np.logical_and(protection_array, target_binary).sum()
            )
            protection_counts["background_pixel_count"] += background_pixels
            protection_counts["protected_background_pixel_count"] += int(
                np.logical_and(protection_array, ~target_binary).sum()
            )
            protection_counts[
                "current_unmatched_predicted_pixel_count"
            ] += unmatched_pixels
            protection_counts[
                "protected_current_unmatched_predicted_pixel_count"
            ] += int(np.logical_and(unmatched_mask, protection_array).sum())

            raw_target_count = int(torch.count_nonzero(target_rescue).item())
            raw_background_count = int(torch.count_nonzero(background_suppression).item())
            raw_target_abs_sum = float(target_rescue.double().sum().item())
            raw_background_abs_sum = float(background_suppression.double().sum().item())
            signal_totals["target_rescue_nonzero_pixels_all_valid"] += raw_target_count
            signal_totals["target_rescue_abs_sum_all_valid"] += raw_target_abs_sum
            signal_totals[
                "background_suppression_nonzero_pixels_all_valid"
            ] += raw_background_count
            signal_totals[
                "background_suppression_abs_sum_all_valid"
            ] += raw_background_abs_sum

            for g_eighths in G_EIGHTHS_ORDER:
                mode = MODE_BY_G_EIGHTHS[g_eighths]
                if g_eighths == 0:
                    probability_tensor = current_probability
                else:
                    routed = route_with_gate(z_out, z_d0, protection, g_eighths)
                    probability_tensor = torch.sigmoid(routed)[valid_slice]
                _require(probability_tensor.shape == target.shape, "point shape differs")
                one_loss = criterion(probability_tensor.float(), target.float())
                _require(math.isfinite(float(one_loss.item())), "point loss is non-finite")
                probability_array = (
                    probability_tensor[0, 0].float().cpu().numpy().copy()
                )
                probabilities[mode].append(probability_array)
                losses[mode].append(float(one_loss.item()))
                gate = g_eighths / 8.0
                routing_totals[mode][
                    "actual_target_rescue_nonzero_pixels"
                ] += raw_target_count if g_eighths else 0
                routing_totals[mode]["actual_target_rescue_abs_sum"] += (
                    gate * raw_target_abs_sum
                )
                routing_totals[mode][
                    "actual_background_suppression_nonzero_pixels"
                ] += raw_background_count if g_eighths else 0
                routing_totals[mode]["actual_background_suppression_abs_sum"] += (
                    gate * raw_background_abs_sum
                )
                routing_totals[mode][
                    "prediction_changed_pixel_count_vs_current"
                ] += int(
                    np.count_nonzero(
                        (probability_array > FIXED_THRESHOLD) != prediction_binary
                    )
                )

            targets.append(target_array)
            identifiers.append(identifier)
            batch_count += 1

    _require(capture.temporary_hooks_restored, "temporary capture hooks were not restored")
    _require(identifiers == list(expected_identifiers), "img_idx/test order differs")
    _require(batch_count == len(expected_identifiers), "inference count differs")
    _require(
        capture.batch_count == batch_count
        and capture.total_counts == {
            "q4": batch_count,
            "out": batch_count,
            "d0": batch_count,
        },
        "capture count differs",
    )
    _require(g0_logit_exact, "g=0 is not an exact raw-logit identity")
    _require(returned_probability_exact, "g=0 probability identity differs")
    _require(oracle_formula_exact, "g=1 max/min oracle formula differs")

    raw_target_count = int(signal_totals["target_rescue_nonzero_pixels_all_valid"])
    raw_target_abs_sum = float(signal_totals["target_rescue_abs_sum_all_valid"])
    raw_background_count = int(
        signal_totals["background_suppression_nonzero_pixels_all_valid"]
    )
    raw_background_abs_sum = float(
        signal_totals["background_suppression_abs_sum_all_valid"]
    )
    points: list[dict[str, Any]] = []
    modes: dict[str, Any] = {}
    current_probabilities = probabilities[CURRENT_MODE]
    for g_eighths in G_EIGHTHS_ORDER:
        mode = MODE_BY_G_EIGHTHS[g_eighths]
        evaluated = metric_core.evaluate_probability_arrays(
            probabilities[mode],
            targets,
            losses[mode],
            sweep_thresholds=SWEEP_THRESHOLDS,
        )
        evaluated = dorf_core._annotate_two_point_sweep(evaluated)
        fixed = tpr_screen._annotate_fixed_metrics(
            evaluated["fixed_threshold_0_5"],
            probabilities[mode],
            targets,
        )
        topology = _compact_topology(
            probabilities[mode],
            targets,
            identifiers,
        )
        _require(
            int(topology["unmatched_pixels_total"])
            == int(fixed["unmatched_predicted_pixels"]),
            "fragmentation/component FP total differs",
        )
        fixed["fragmented_gt_count"] = int(topology["fragmented_gt_count"])
        fixed["extra_fragments"] = int(topology["extra_fragments"])
        routing = {
            "raw_target_rescue_nonzero_pixels": raw_target_count,
            "raw_target_rescue_abs_sum": raw_target_abs_sum,
            "raw_background_suppression_nonzero_pixels": raw_background_count,
            "raw_background_suppression_abs_sum": raw_background_abs_sum,
            **routing_totals[mode],
        }
        difference = tpr_screen.probability_difference(
            current_probabilities,
            probabilities[mode],
        )
        point = {
            "mode": mode,
            "g_eighths": g_eighths,
            "g": g_eighths / 8.0,
            "point_kind": point_kind(g_eighths),
            "authorization_eligible": g_eighths in AUTHORIZATION_G_EIGHTHS,
            "fixed_threshold_0_5": fixed,
            "routing": routing,
            "fragmentation": topology,
            "probability_difference_to_current": difference,
            "descriptive_pd_fa": evaluated["descriptive_pd_fa"],
        }
        points.append(point)
        modes[mode] = point

    current_difference = modes[CURRENT_MODE]["probability_difference_to_current"]
    _require(
        float(current_difference["max_abs"]) == 0.0
        and float(current_difference["absolute_difference_sum"]) == 0.0,
        "g=0 self-difference is nonzero",
    )
    current_fixed = modes[CURRENT_MODE]["fixed_threshold_0_5"]
    _require(
        int(signal_totals["current_missed_gt_object_count"])
        == int(current_fixed["target_count"])
        - int(current_fixed["matched_target_count"]),
        "missed-target signal count differs from evaluator",
    )
    _require(
        int(signal_totals["current_unmatched_predicted_object_count"])
        == int(current_fixed["unmatched_predicted_object_count"]),
        "unmatched-object signal count differs from evaluator",
    )
    _require(
        int(signal_totals["current_unmatched_predicted_pixel_count"])
        == int(current_fixed["unmatched_predicted_pixels"]),
        "unmatched-pixel signal count differs from evaluator",
    )

    occupancy, occupancy_zero = _ratio(
        int(protection_counts["protected_pixel_count"]),
        int(protection_counts["valid_pixel_count"]),
    )
    gt_coverage, gt_zero = _ratio(
        int(protection_counts["protected_gt_pixel_count"]),
        int(protection_counts["gt_pixel_count"]),
    )
    background_fraction, background_zero = _ratio(
        int(protection_counts["protected_background_pixel_count"]),
        int(protection_counts["background_pixel_count"]),
    )
    false_component_fraction, false_component_zero = _ratio(
        int(protection_counts["protected_current_unmatched_predicted_pixel_count"]),
        int(protection_counts["current_unmatched_predicted_pixel_count"]),
    )
    _require(not occupancy_zero, "valid-pixel denominator is zero")
    _require(not background_zero, "background-pixel denominator is zero")
    protection_statistics = {
        **protection_counts,
        "protection_occupancy": occupancy,
        "protection_occupancy_denominator_zero": occupancy_zero,
        "protected_gt_coverage": gt_coverage,
        "protected_gt_coverage_denominator_zero": gt_zero,
        "protected_background_fraction": background_fraction,
        "protected_background_fraction_denominator_zero": background_zero,
        "false_component_protected_fraction": false_component_fraction,
        "false_component_protected_fraction_denominator_zero": false_component_zero,
        "binary": True,
        "detached": True,
        "protection_flip_rate": None,
        "protection_flip_rate_scope": "training_evaluation_only_not_zero_training",
    }

    state_after = component_core.module_state_sha256(model)
    _require(state_after == state_before, "model state changed during PBDR audit")
    target_signal = {
        key: value
        for key, value in signal_totals.items()
        if "rescue" in key or "missed_gt" in key
    }
    protected_object_fraction, protected_object_zero = _ratio(
        int(target_signal["missed_gt_objects_with_protected_pixels"]),
        int(target_signal["current_missed_gt_object_count"]),
    )
    d0_pixel_fraction, d0_pixel_zero = _ratio(
        int(target_signal["missed_gt_pixels_with_d0_gt_out"]),
        int(target_signal["current_missed_gt_pixel_count"]),
    )
    target_signal.update(
        {
            "missed_gt_objects_with_protected_pixels_fraction": (
                protected_object_fraction
            ),
            "missed_gt_objects_with_protected_pixels_fraction_denominator_zero": (
                protected_object_zero
            ),
            "missed_gt_pixels_with_d0_gt_out_fraction": d0_pixel_fraction,
            "missed_gt_pixels_with_d0_gt_out_fraction_denominator_zero": (
                d0_pixel_zero
            ),
        }
    )
    background_signal = {
        key: value
        for key, value in signal_totals.items()
        if "suppression" in key or "current_unmatched" in key
    }
    p_zero_fraction, p_zero_zero = _ratio(
        int(background_signal["current_unmatched_pixels_with_p_zero"]),
        int(background_signal["current_unmatched_predicted_pixel_count"]),
    )
    out_gt_d0_fraction, out_gt_d0_zero = _ratio(
        int(background_signal["current_unmatched_pixels_with_out_gt_d0"]),
        int(background_signal["current_unmatched_predicted_pixel_count"]),
    )
    background_signal.update(
        {
            "current_unmatched_pixels_with_p_zero_fraction": p_zero_fraction,
            "current_unmatched_pixels_with_p_zero_fraction_denominator_zero": (
                p_zero_zero
            ),
            "current_unmatched_pixels_with_out_gt_d0_fraction": (
                out_gt_d0_fraction
            ),
            "current_unmatched_pixels_with_out_gt_d0_fraction_denominator_zero": (
                out_gt_d0_zero
            ),
        }
    )
    return {
        "points": points,
        "modes": modes,
        "signals": {
            "target_rescue": target_signal,
            "background_suppression": background_signal,
        },
        "protection": protection_statistics,
        "capture": {
            "image_count": batch_count,
            "batch_count": batch_count,
            "q4_capture_count": capture.total_counts["q4"],
            "z_out_capture_count": capture.total_counts["out"],
            "z_d0_capture_count": capture.total_counts["d0"],
            "each_tensor_captured_once_per_batch": True,
            "q4_channels": FORMAL_Q4_RELAY_CHANNELS,
            "captured_values_are_raw_logits": True,
            "temporary_hooks_restored": capture.temporary_hooks_restored,
            "tensor_stream_sha256": tensor_stream_sha.hexdigest(),
            "image_order_newline_sha256": hashlib.sha256(
                ("\n".join(identifiers) + "\n").encode("utf-8")
            ).hexdigest(),
            "one_model_forward_per_batch": True,
            "same_captured_tensors_reused_for_all_gates": True,
        },
        "identity": {
            "g0_raw_logit_bitwise_equal": g0_logit_exact,
            "g0_returned_probability_bitwise_equal": returned_probability_exact,
            "g0_binary_prediction_equal": True,
            "oracle_g1_max_min_formula_bitwise_equal": oracle_formula_exact,
        },
        "restoration_audit": {
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_state_unchanged": state_after == state_before,
        },
        "probability_cache_written": False,
        "feature_cache_written": False,
        "per_image_artifact_written": False,
    }


def analyze_run(
    *,
    dataset: str,
    checkpoint_role: str,
    identity_manifest: Path,
    identity_manifest_sha256: str,
    dorf_input_manifest: Path,
    dataset_root: Path,
    device_name: str,
    workers: int,
) -> dict[str, Any]:
    _require(dataset in DATASETS, "dataset is outside the formal matrix")
    _require(checkpoint_role in CHECKPOINT_ROLES, "checkpoint role differs")
    _require(workers >= 0, "workers must be non-negative")
    _require(
        identity_manifest_sha256 == FROZEN_IDENTITY_MANIFEST_SHA256,
        "identity manifest SHA differs",
    )
    _require(
        file_sha256(Path(dorf_input_manifest).resolve(strict=True))
        == FROZEN_DORF_INPUT_MANIFEST_SHA256,
        "DORF authority manifest SHA differs",
    )
    sources_before = _source_sha256()
    adapter.configure_core()
    training_engine.configure_determinism()
    inference_math = configure_inference_math()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    identity_binding = input_core.load_manifest_binding(
        identity_manifest,
        expected_manifest_sha256=identity_manifest_sha256,
        dataset=dataset,
        checkpoint_role=checkpoint_role,
    )
    dorf_binding, dorf_paths, background_record = dorf_core.load_input_manifest_binding(
        input_manifest=dorf_input_manifest,
        method="final_tss_off",
        dataset=dataset,
        checkpoint_role=checkpoint_role,
        run_dir=None,
        reference_evaluation=None,
        data_protocol_manifest=Path(identity_binding["data_protocol_manifest"]["path"]),
    )
    _require(
        identity_binding["checkpoint_sha256"]
        == dorf_binding["entry"]["checkpoint_sha256"],
        "two frozen authorities bind different checkpoints",
    )
    _require(
        identity_binding["reference_evaluation_sha256"]
        == dorf_binding["entry"]["evaluation_sha256"],
        "two frozen authorities bind different evaluations",
    )
    checkpoint = input_core.load_bound_parent_checkpoint(identity_binding)
    reference_path = Path(identity_binding["reference_evaluation_path"])
    reference = adapter.validate_completed_output(
        reference_path,
        dataset=dataset,
        checkpoint_role=checkpoint_role,
    )
    model, model_metadata = (
        model_builder.build_final_inference_model_from_training_state_dict(
            checkpoint["state_dict"],
            dataset_name=dataset,
            seed=SEED,
        )
    )
    _require(len(model.state_dict()) == 564, "Current inference state-key count differs")
    _require(model.training is False, "Current inference loader did not return eval model")
    _require(getattr(model, "mode", None) == "test", "Current model mode differs")
    model.to(device)
    manifest_path = Path(identity_binding["data_protocol_manifest"]["path"])
    manifest = data_protocol.load_protocol_manifest(
        manifest_path,
        dataset_root=dataset_root,
    )
    dataset_object = metric_core.ThreeDatasetTestDataset(
        dataset_root,
        dataset,
        manifest_path,
    )
    loader = DataLoader(
        dataset_object,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    analyzed = analyze_loaded_model(model, loader, device, dataset_object.sample_ids)
    current_fixed = analyzed["modes"][CURRENT_MODE]["fixed_threshold_0_5"]
    replay = historical_reference_drift_audit(
        current_fixed,
        reference["fixed_threshold_0_5"],
        expected_background_false_positive_pixels=int(
            background_record["false_positive_pixels"]
        ),
    )
    _require(_source_sha256() == sources_before, "runtime source changed")
    dorf_core.verify_bound_input_artifacts(
        dorf_binding["entry"],
        dorf_paths,
        {"sha256": dorf_binding["data_protocol_manifest"]["sha256"]},
        {"sha256": dorf_binding["background_pixel_authority"]["sha256"]},
    )
    _require(
        file_sha256(Path(identity_manifest).resolve(strict=True))
        == identity_manifest_sha256,
        "identity manifest changed during inference",
    )

    output = {
        "schema": SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "method": "current_tss_off",
        "dataset": dataset,
        "checkpoint_role": checkpoint_role,
        "seed": SEED,
        "split": "img_idx/test",
        "test_selected": True,
        "selection_is_optimistic": True,
        "protocol": {
            "threshold": FIXED_THRESHOLD,
            "threshold_operator": ">",
            "connectivity": 2,
            "object_match_rule": (
                "hungarian_max_cardinality_then_min_centroid_distance"
            ),
            "match_radius": metric_core.MATCH_RADIUS,
            "match_operator": "<",
            "tiny_area": metric_core.TINY_AREA,
            "tiny_operator": "<=",
            "float_eq_atol": FLOAT_EQ_ATOL,
            "float_eq_rtol": FLOAT_EQ_RTOL,
            "g_eighths_order": list(G_EIGHTHS_ORDER),
            "authorization_g_eighths": list(AUTHORIZATION_G_EIGHTHS),
            "identity_g_eighths": IDENTITY_G_EIGHTHS,
            "oracle_g_eighths": ORACLE_G_EIGHTHS,
        },
        **analyzed,
        "current_reference": {
            "metrics": current_fixed,
            "authority_metrics": reference["fixed_threshold_0_5"],
            "g0_historical_drift_audit": replay,
        },
        "bindings": {
            "identity_manifest": {
                "path": str(Path(identity_manifest).resolve(strict=True)),
                "sha256": identity_manifest_sha256,
            },
            "dorf_input_manifest": {
                "path": str(Path(dorf_input_manifest).resolve(strict=True)),
                "sha256": FROZEN_DORF_INPUT_MANIFEST_SHA256,
            },
            "checkpoint": {
                "path": identity_binding["checkpoint_path"],
                "sha256": identity_binding["checkpoint_sha256"],
                "epoch": identity_binding["epoch"],
            },
            "reference_evaluation": {
                "path": str(reference_path),
                "sha256": identity_binding["reference_evaluation_sha256"],
            },
            "data_protocol_manifest": {
                "path": str(manifest_path),
                "sha256": identity_binding["data_protocol_manifest"]["sha256"],
                "schema": manifest.get("schema"),
                "manifest_id": manifest.get("manifest_id"),
            },
            "background_pixel_authority": dorf_binding[
                "background_pixel_authority"
            ],
        },
        "model": model_metadata,
        "inference_math": inference_math,
        "source_sha256": sources_before,
        "intervention_contract": {
            "family": "PBDR_V1_zero_training_shared_forward",
            "formula": (
                "z_out+g*(P*relu(z_d0-z_out)-(1-P)*relu(z_out-z_d0))"
            ),
            "protection": (
                "q4_tail_z1.5_binary_dilate3_nearest_detached"
            ),
            "one_forward_per_image": True,
            "model_state_modified": False,
            "derived_checkpoint_written": False,
            "formal_training_started": False,
        },
        "engineering_valid": True,
        "derived_checkpoint_written": False,
        "formal_training_started": False,
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
    _require(payload.get("schema") == SCHEMA, "analyzer schema differs")
    _require(payload.get("status") == "complete", "analyzer is incomplete")
    _require(payload.get("dataset") in DATASETS, "dataset differs")
    _require(payload.get("checkpoint_role") in CHECKPOINT_ROLES, "role differs")
    _require(payload.get("seed") == SEED, "seed differs")
    _require(payload.get("split") == "img_idx/test", "split differs")
    protocol = payload.get("protocol")
    _require(isinstance(protocol, Mapping), "protocol is missing")
    _require(
        protocol.get("g_eighths_order") == list(G_EIGHTHS_ORDER)
        and protocol.get("authorization_g_eighths")
        == list(AUTHORIZATION_G_EIGHTHS),
        "gate grid differs",
    )
    points = payload.get("points")
    _require(isinstance(points, list) and len(points) == len(G_EIGHTHS_ORDER), "point count differs")
    _require(
        [point.get("g_eighths") for point in points] == list(G_EIGHTHS_ORDER),
        "point order differs",
    )
    modes = payload.get("modes")
    _require(
        isinstance(modes, Mapping) and tuple(modes) == MODE_ORDER,
        "mode mapping differs",
    )
    _require(
        all(modes[point["mode"]] == point for point in points),
        "points/modes mapping differs",
    )
    invariant: tuple[int, int, int] | None = None
    for point, g_eighths in zip(points, G_EIGHTHS_ORDER):
        _require(point.get("mode") == MODE_BY_G_EIGHTHS[g_eighths], "mode differs")
        _require(point.get("point_kind") == point_kind(g_eighths), "point kind differs")
        _require(
            point.get("authorization_eligible")
            is (g_eighths in AUTHORIZATION_G_EIGHTHS),
            "authorization eligibility differs",
        )
        fixed = point.get("fixed_threshold_0_5")
        _require(isinstance(fixed, Mapping), "fixed metrics are missing")
        target_count = int(fixed.get("target_count", -1))
        tiny_count = int(fixed.get("tiny_target_count", -1))
        valid_pixels = int(fixed.get("valid_pixel_count", -1))
        current_invariant = (target_count, tiny_count, valid_pixels)
        invariant = invariant or current_invariant
        _require(current_invariant == invariant, "point denominators differ")
        component_fp = int(fixed.get("component_false_positive_pixels", -1))
        _require(
            component_fp == int(fixed.get("unmatched_predicted_pixels", -2)),
            "component FP alias differs",
        )
        _require(
            math.isclose(
                float(fixed.get("fa")),
                component_fp / valid_pixels,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "Fa integer identity differs",
        )
        for name in ("miou", "niou", "pd", "tiny_pd", "pixel_f1"):
            value = fixed.get(name)
            _require(
                value is None or math.isfinite(float(value)),
                f"{name} is non-finite",
            )
    identity = payload.get("identity")
    _require(
        isinstance(identity, Mapping)
        and identity.get("g0_raw_logit_bitwise_equal") is True
        and identity.get("g0_returned_probability_bitwise_equal") is True
        and identity.get("oracle_g1_max_min_formula_bitwise_equal") is True,
        "identity/oracle audit differs",
    )
    capture = payload.get("capture")
    _require(
        isinstance(capture, Mapping)
        and capture.get("each_tensor_captured_once_per_batch") is True
        and capture.get("temporary_hooks_restored") is True
        and capture.get("one_model_forward_per_batch") is True,
        "capture audit differs",
    )
    capture_batches = int(capture.get("batch_count", -1))
    _require(
        capture_batches > 0
        and int(capture.get("image_count", -2)) == capture_batches
        and int(capture.get("q4_capture_count", -3)) == capture_batches
        and int(capture.get("z_out_capture_count", -4)) == capture_batches
        and int(capture.get("z_d0_capture_count", -5)) == capture_batches,
        "capture count closure differs",
    )
    protection = payload.get("protection")
    _require(isinstance(protection, Mapping), "protection statistics are missing")
    _require(
        protection.get("protected_background_fraction_denominator_zero") is False
        and math.isfinite(float(protection["protected_background_fraction"])),
        "protected background fraction differs",
    )
    current = payload.get("current_reference")
    _require(
        isinstance(current, Mapping)
        and current.get("metrics") == modes[CURRENT_MODE]["fixed_threshold_0_5"]
        and isinstance(current.get("g0_historical_drift_audit"), Mapping)
        and current["g0_historical_drift_audit"].get(
            "same_forward_authorization_gate"
        )
        is False,
        "g0 current/historical audit differs",
    )
    _require(
        payload.get("inference_math")
        == {
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "deterministic_algorithms": True,
        },
        "inference math binding differs",
    )
    _require(
        payload.get("engineering_valid") is True
        and payload.get("derived_checkpoint_written") is False
        and payload.get("formal_training_started") is False,
        "engineering/artifact flags differ",
    )
    json.dumps(payload, allow_nan=False)


def _default_output(dataset: str, checkpoint_role: str) -> Path:
    return (
        DEFAULT_OUTPUT_ROOT
        / "runs"
        / dataset
        / f"final_tss_off_{checkpoint_role}_seed42"
        / "evaluation.json"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--checkpoint-role", choices=CHECKPOINT_ROLES, required=True
    )
    parser.add_argument(
        "--identity-manifest", type=Path, default=DEFAULT_IDENTITY_MANIFEST
    )
    parser.add_argument(
        "--identity-manifest-sha256",
        default=FROZEN_IDENTITY_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--dorf-input-manifest",
        type=Path,
        default=DEFAULT_DORF_INPUT_MANIFEST,
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=data_protocol.DEFAULT_DATASET_ROOT
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if not (args.device == "cpu" or args.device.startswith("cuda:")):
        parser.error("--device must be cpu or cuda:N")
    args.output = args.output or _default_output(args.dataset, args.checkpoint_role)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing existing output: {args.output}")
    output = analyze_run(
        dataset=args.dataset,
        checkpoint_role=args.checkpoint_role,
        identity_manifest=args.identity_manifest,
        identity_manifest_sha256=args.identity_manifest_sha256,
        dorf_input_manifest=args.dorf_input_manifest,
        dataset_root=args.dataset_root,
        device_name=args.device,
        workers=args.workers,
    )
    dorf_core.atomic_create_json(args.output, output)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output.resolve()),
                "sha256": file_sha256(args.output.resolve()),
                "dataset": args.dataset,
                "checkpoint_role": args.checkpoint_role,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZATION_G_EIGHTHS",
    "CHECKPOINT_ROLES",
    "CURRENT_MODE",
    "DATASETS",
    "DEFAULT_OUTPUT_ROOT",
    "G_EIGHTHS_ORDER",
    "MODE_BY_G_EIGHTHS",
    "MODE_ORDER",
    "ORACLE_MODE",
    "RawQ4D0OutHookCapture",
    "SCHEMA",
    "analyze_loaded_model",
    "analyze_run",
    "build_protection",
    "configure_inference_math",
    "historical_reference_drift_audit",
    "point_kind",
    "route_with_gate",
    "validate_output_payload",
]
