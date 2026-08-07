#!/usr/bin/env python3
"""Read-only NUAA PBDR-V2 A0--A8 failure-attribution evaluator.

The evaluator pairs the two independently trained checkpoints for one formal
role (``best_miou`` or ``best_pd``):

* A0 is the Current checkpoint's deployed ``out`` probability;
* A1--A8 are derived from one PBDR-V2 backbone forward per test image by
  reusing its raw ``q4``, ``out`` and ``d0`` tensors.

Every point is evaluated at the frozen threshold 0.5 and on a descriptive
0.20--0.80 threshold grid.  The repository evaluator's threshold-1.0 empty
endpoint is retained as an explicit control, never as a selected attribution
point.  Checkpoint and source bindings are validated before inference and
re-hashed afterwards.  No optimizer, backward pass, checkpoint, feature map,
logit map, or probability cache is created.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np
import torch
import torch.nn as nn
from skimage import measure
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import analyze_ner_stage2_mask_knockout_v1 as state_core  # noqa: E402
from experiments import evaluate_three_dataset_pbdr_v2 as pbdr_evaluator  # noqa: E402
from experiments import evaluate_three_dataset_tss_off_seed42_v1 as current_evaluator  # noqa: E402
from experiments import evaluate_three_dataset_v2 as metric_core  # noqa: E402
from experiments import four_dataset_evaluation_protocol_v1 as protocol_metrics  # noqa: E402
from experiments import three_dataset_pbdr_v2_models_seed42_v1 as pbdr_models  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import (  # noqa: E402
    train_four_dataset_original_final_seed42_exact_v1 as training_engine,
)
from model.tpd_persistent_evidence_residual_router_v2 import (  # noqa: E402
    PBDRV2RoutingOutput,
    PersistentEvidenceResidualRouterV2,
)


SCHEMA = "sctransnet_nuaa_pbdr_v2_failure_attribution_v1/v1"
DATASET = "NUAA-SIRST"
SEED = 42
FIXED_THRESHOLD = 0.5
THRESHOLD_START = 0.20
THRESHOLD_STOP = 0.80
THRESHOLD_STEP = 0.005
EMPTY_ENDPOINT = 1.0
CHECKPOINT_ROLES = tuple(metric_core.CHECKPOINT_ROLES)
STRICT_MATH_SOFT_METRIC_TOLERANCE = 1.0e-3
STRICT_MATH_TEST_LOSS_TOLERANCE = 1.0e-6
STRICT_MATH_MAX_FA_PIXEL_DRIFT = 8

CURRENT_INFERENCE_STATE_KEY_COUNT = 564
PBDR_INFERENCE_STATE_KEY_COUNT = pbdr_models.INFERENCE_STATE_KEY_COUNT

DEFAULT_CURRENT_RUN_DIR = (
    REPO_ROOT
    / "results/three_dataset_tss_off_seed42_v1/runs"
    / DATASET
    / "final_tss_off/seed_42"
)
DEFAULT_PBDR_RUN_DIR = (
    REPO_ROOT
    / "results/three_dataset_pbdr_v2_tss_off_seed42_v1/runs"
    / DATASET
    / "pbdr_v2_tss_off/seed_42"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/nuaa_pbdr_v2_failure_attribution_v1"

AblationMode = Literal[
    "identity",
    "full",
    "direct_only",
    "disagreement_only",
    "rescue_only",
    "suppression_only",
    "nonnegative_strengths",
    "auxiliary_d0",
]

PBDR_ABLATION_MODES: tuple[AblationMode, ...] = (
    "identity",
    "full",
    "direct_only",
    "disagreement_only",
    "rescue_only",
    "suppression_only",
    "nonnegative_strengths",
    "auxiliary_d0",
)

MODE_ORDER = (
    "A0_current_out",
    "A1_pbdr_router_bypass_out",
    "A2_pbdr_full_routed",
    "A3_pbdr_direct_only",
    "A4_pbdr_disagreement_only",
    "A5_pbdr_rescue_only",
    "A6_pbdr_suppression_only",
    "A7_pbdr_nonnegative_strengths",
    "A8_pbdr_auxiliary_d0",
)

PBDR_PUBLIC_TO_ABLATION: dict[str, AblationMode] = {
    "A1_pbdr_router_bypass_out": "identity",
    "A2_pbdr_full_routed": "full",
    "A3_pbdr_direct_only": "direct_only",
    "A4_pbdr_disagreement_only": "disagreement_only",
    "A5_pbdr_rescue_only": "rescue_only",
    "A6_pbdr_suppression_only": "suppression_only",
    "A7_pbdr_nonnegative_strengths": "nonnegative_strengths",
    "A8_pbdr_auxiliary_d0": "auxiliary_d0",
}

MODE_FORMULAS = {
    "A0_current_out": "Current checkpoint: sigmoid(z_out)",
    "A1_pbdr_router_bypass_out": "PBDR checkpoint: sigmoid(z_out)",
    "A2_pbdr_full_routed": "sigmoid(z_out+Q+g_r*R_plus-g_s*R_minus)",
    "A3_pbdr_direct_only": "sigmoid(z_out+Q)",
    "A4_pbdr_disagreement_only": "sigmoid(z_out+g_r*R_plus-g_s*R_minus)",
    "A5_pbdr_rescue_only": "sigmoid(z_out+g_r*R_plus)",
    "A6_pbdr_suppression_only": "sigmoid(z_out-g_s*R_minus)",
    "A7_pbdr_nonnegative_strengths": (
        "sigmoid(z_out+Q+max(g_r,0)*R_plus-max(g_s,0)*R_minus)"
    ),
    "A8_pbdr_auxiliary_d0": "PBDR checkpoint: sigmoid(z_d0)",
}

DIAGNOSTIC_QUANTILES = (1, 5, 50, 95, 99)


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
    """Hash tensor state independently of its current device."""

    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        _require(isinstance(value, torch.Tensor), f"non-tensor state: {key}")
        ready = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(ready.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(ready.shape)).encode("ascii"))
        digest.update(b"\0")
        # Multi-byte zero-dimensional buffers (for example BatchNorm's
        # ``num_batches_tracked``) cannot be dtype-viewed directly on recent
        # PyTorch.  Flattening preserves the exact dense byte stream.
        digest.update(ready.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _source_sha256() -> dict[str, str]:
    sources = {
        "analysis/analyze_nuaa_pbdr_v2_failure_attribution_v1.py": Path(__file__),
        "analysis/analyze_ner_stage2_mask_knockout_v1.py": Path(
            state_core.__file__
        ),
        "experiments/evaluate_three_dataset_pbdr_v2.py": Path(
            pbdr_evaluator.__file__
        ),
        "experiments/evaluate_three_dataset_tss_off_seed42_v1.py": Path(
            current_evaluator.__file__
        ),
        "experiments/evaluate_three_dataset_v2.py": Path(metric_core.__file__),
        "experiments/four_dataset_evaluation_protocol_v1.py": Path(
            protocol_metrics.__file__
        ),
        "experiments/three_dataset_pbdr_v2_models_seed42_v1.py": Path(
            pbdr_models.__file__
        ),
        "experiments/three_dataset_v2_protocol.py": Path(data_protocol.__file__),
        "experiments/train_four_dataset_original_final_seed42_exact_v1.py": Path(
            training_engine.__file__
        ),
        "model/tpd_persistent_evidence_residual_router_v2.py": (
            REPO_ROOT / "model/tpd_persistent_evidence_residual_router_v2.py"
        ),
        "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v2.py": (
            REPO_ROOT
            / "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v2.py"
        ),
    }
    return {
        relative: file_sha256(path.resolve(strict=True))
        for relative, path in sorted(sources.items())
    }


def configure_inference_math() -> dict[str, Any]:
    """Reapply deterministic inference and explicitly prohibit TF32."""

    training_engine.configure_determinism()
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
    expected = {
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
    }
    _require(contract == expected, "inference math contract differs")
    return contract


def _scalar_map(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(1, 1, 1, 1)


def ablation_logits_from_diagnostics(
    z_out: torch.Tensor,
    z_d0: torch.Tensor,
    diagnostics: PBDRV2RoutingOutput,
    mode: AblationMode,
) -> torch.Tensor:
    """Return one exact V2 counterfactual from shared forward-local tensors."""

    if mode not in PBDR_ABLATION_MODES:
        raise ValueError(f"unsupported PBDR-V2 ablation mode: {mode}")
    _require(z_out.shape == z_d0.shape, "z_out/z_d0 shapes differ")
    rescue_strength = _scalar_map(diagnostics.rescue_strength)
    suppression_strength = _scalar_map(diagnostics.suppression_strength)
    if mode == "identity":
        output = z_out
    elif mode == "full":
        output = diagnostics.routed_logits
    elif mode == "direct_only":
        output = z_out + diagnostics.direct_residual
    elif mode == "disagreement_only":
        output = (
            z_out
            + rescue_strength * diagnostics.target_rescue
            - suppression_strength * diagnostics.background_suppression
        )
    elif mode == "rescue_only":
        output = z_out + rescue_strength * diagnostics.target_rescue
    elif mode == "suppression_only":
        output = (
            z_out
            - suppression_strength * diagnostics.background_suppression
        )
    elif mode == "nonnegative_strengths":
        output = (
            z_out
            + diagnostics.direct_residual
            + rescue_strength.clamp_min(0.0) * diagnostics.target_rescue
            - suppression_strength.clamp_min(0.0)
            * diagnostics.background_suppression
        )
    else:
        _require(mode == "auxiliary_d0", "ablation dispatch differs")
        output = z_d0
    _require(output.shape == z_out.shape, "ablation output shape differs")
    _require(bool(torch.isfinite(output).all()), "ablation output is non-finite")
    return output


class PBDRV2AblationWrapper(nn.Module):
    """Evaluation-only wrapper; install only after strict checkpoint loading.

    Wrapping introduces the ``router.`` state-key prefix, so callers must not
    save the wrapped state dict.  The module holds no forward cache.
    """

    def __init__(
        self,
        router: PersistentEvidenceResidualRouterV2,
        mode: AblationMode,
    ) -> None:
        super().__init__()
        if type(router) is not PersistentEvidenceResidualRouterV2:
            raise TypeError("router must be the exact formal PBDR-V2 class")
        if mode not in PBDR_ABLATION_MODES:
            raise ValueError(f"unsupported PBDR-V2 ablation mode: {mode}")
        self.router = router
        self.mode = mode

    def forward(
        self,
        z_out: torch.Tensor,
        z_d0: torch.Tensor,
        q4: torch.Tensor,
    ) -> torch.Tensor:
        diagnostics = self.router.forward_with_diagnostics(z_out, z_d0, q4)
        return ablation_logits_from_diagnostics(
            z_out,
            z_d0,
            diagnostics,
            cast(AblationMode, self.mode),
        )


class RawQ4D0OutCapture:
    """Capture raw q4/out/d0 exactly once during one unchanged forward."""

    def __init__(self, model: nn.Module) -> None:
        relay = getattr(model, "tpd_ner", None)
        fusions = getattr(relay, "fusions", None)
        self._modules = {
            "q4": fusions["4"] if fusions is not None and "4" in fusions else None,
            "z_out": getattr(model, "outc", None),
            "z_d0": getattr(model, "outconv", None),
        }
        for name, module in self._modules.items():
            _require(isinstance(module, nn.Module), f"model lacks capture module {name}")
        self._modules = cast(dict[str, nn.Module], self._modules)
        self._prior_hook_ids = {
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
            _require(self._active, f"{name} capture fired outside an active batch")
            _require(name not in self._current, f"{name} executed twice in one batch")
            _require(isinstance(output, torch.Tensor), f"{name} is not a tensor")
            _require(output.ndim == 4, f"{name} is not BCHW")
            _require(bool(torch.isfinite(output).all()), f"{name} is non-finite")
            self._current[name] = output
            self.total_counts[name] += 1

        return record

    def __enter__(self) -> "RawQ4D0OutCapture":
        _require(not self._handles, "capture hooks are already installed")
        self._handles = [
            module.register_forward_hook(self._hook(name))
            for name, module in self._modules.items()
        ]
        return self

    def begin_batch(self) -> None:
        _require(not self._active, "previous capture batch is still active")
        self._active = True
        self._current = {}

    def finish_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _require(self._active, "no capture batch is active")
        self._active = False
        _require(set(self._current) == set(self._modules), "capture tensor set differs")
        q4 = self._current["q4"]
        z_out = self._current["z_out"]
        z_d0 = self._current["z_d0"]
        _require(z_out.shape == z_d0.shape, "captured readout shapes differ")
        _require(q4.shape[0] == z_out.shape[0], "captured batch sizes differ")
        _require(q4.shape[1] == 8, "captured q4 width differs")
        _require(
            q4.device == z_out.device == z_d0.device,
            "captured tensors use different devices",
        )
        _require(
            q4.dtype == z_out.dtype == z_d0.dtype,
            "captured tensors use different dtypes",
        )
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
            tuple(module._forward_hooks) == self._prior_hook_ids[name]
            for name, module in self._modules.items()
        )


def build_threshold_grid(
    start: float = THRESHOLD_START,
    stop: float = THRESHOLD_STOP,
    step: float = THRESHOLD_STEP,
) -> tuple[float, ...]:
    """Build an inclusive, decimal-stable registered threshold grid."""

    for value, label in ((start, "start"), (stop, "stop"), (step, "step")):
        _require(math.isfinite(float(value)), f"threshold {label} must be finite")
    _require(0.0 <= start <= stop < EMPTY_ENDPOINT, "threshold interval differs")
    _require(step > 0.0, "threshold step must be positive")
    raw_steps = (stop - start) / step
    count = int(round(raw_steps))
    _require(
        math.isclose(raw_steps, count, rel_tol=0.0, abs_tol=1e-9),
        "threshold step does not close the interval",
    )
    values = tuple(round(start + index * step, 12) for index in range(count + 1))
    _require(values[0] == start and values[-1] == stop, "threshold endpoints differ")
    _require(FIXED_THRESHOLD in values, "threshold grid omits 0.5")
    _require(len(values) == len(set(values)), "threshold grid contains duplicates")
    return values


def _update_tensor_digest(
    digest: "hashlib._Hash",
    name: str,
    tensor: torch.Tensor,
) -> None:
    ready = tensor.detach().float().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(ready.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(ready.numpy().tobytes())


def _distribution_summary(
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    ready = value.detach().float()
    if mask is not None:
        _require(mask.shape == ready.shape, "distribution mask shape differs")
        ready = ready[mask]
    else:
        ready = ready.reshape(-1)
    count = int(ready.numel())
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "absolute_mean": None,
            "minimum": None,
            "maximum": None,
            "positive_fraction": None,
            "quantiles": {f"p{value}": None for value in DIAGNOSTIC_QUANTILES},
        }
    array = ready.cpu().numpy().astype(np.float64, copy=False)
    quantiles = np.percentile(array, DIAGNOSTIC_QUANTILES)
    output = {
        "count": count,
        "mean": float(array.mean(dtype=np.float64)),
        "absolute_mean": float(np.abs(array).mean(dtype=np.float64)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "positive_fraction": float(np.count_nonzero(array > 0.0) / count),
        "quantiles": {
            f"p{percentile}": float(result)
            for percentile, result in zip(DIAGNOSTIC_QUANTILES, quantiles)
        },
    }
    _require(
        all(
            result is None or math.isfinite(float(result))
            for result in _walk_scalars(output)
        ),
        "distribution summary is non-finite",
    )
    return output


def _walk_scalars(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        output: list[Any] = []
        for item in value.values():
            output.extend(_walk_scalars(item))
        return output
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(_walk_scalars(item))
        return output
    return [value]


def q4_pre_normalization_statistics(q4: torch.Tensor) -> list[dict[str, float | int]]:
    """Return per-channel spatial statistics before V2 global RMS scaling."""

    _require(q4.ndim == 4 and q4.shape[0] == 1 and q4.shape[1] == 8, "q4 shape differs")
    working = q4.detach().float()[0]
    output: list[dict[str, float | int]] = []
    for channel in range(working.shape[0]):
        values = working[channel]
        output.append(
            {
                "channel": channel,
                "spatial_mean": float(values.double().mean().item()),
                "rms": float(torch.sqrt(values.double().square().mean()).item()),
                "maximum": float(values.max().item()),
                "minimum": float(values.min().item()),
                "absolute_maximum": float(values.abs().max().item()),
            }
        )
    return output


def router_parameter_audit(router: PersistentEvidenceResidualRouterV2) -> dict[str, Any]:
    """Expose the learned scalar signs and all 17 projection parameters."""

    if type(router) is not PersistentEvidenceResidualRouterV2:
        raise TypeError("router must be the exact formal PBDR-V2 class")
    rescue, suppression = router.strengths()
    parameters: dict[str, Any] = {}
    total = 0
    for name, parameter in router.named_parameters():
        values = parameter.detach().float().cpu().reshape(-1).numpy()
        total += int(values.size)
        parameters[name] = {
            "shape": list(parameter.shape),
            "values": [float(value) for value in values],
            "l1_norm": float(np.abs(values).sum(dtype=np.float64)),
            "l2_norm": float(np.sqrt(np.square(values.astype(np.float64)).sum())),
            "positive_count": int(np.count_nonzero(values > 0.0)),
            "negative_count": int(np.count_nonzero(values < 0.0)),
            "zero_count": int(np.count_nonzero(values == 0.0)),
        }
    _require(total == 19, "router parameter count differs")
    return {
        "parameter_count": total,
        "rescue_strength_raw": float(router.rescue_strength_raw.detach().item()),
        "suppression_strength_raw": float(
            router.suppression_strength_raw.detach().item()
        ),
        "rescue_strength_mapped": float(rescue.detach().item()),
        "suppression_strength_mapped": float(suppression.detach().item()),
        "rescue_semantics_reversed": bool(float(rescue.detach().item()) < 0.0),
        "suppression_semantics_reversed": bool(
            float(suppression.detach().item()) < 0.0
        ),
        "parameters": parameters,
    }


def binary_transition_counts(
    current_probability: np.ndarray,
    candidate_probability: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, int]:
    _require(current_probability.shape == candidate_probability.shape == target.shape, "transition arrays differ")
    current = np.asarray(current_probability) > threshold
    candidate = np.asarray(candidate_probability) > threshold
    foreground = np.asarray(target) > FIXED_THRESHOLD
    background = ~foreground
    current_components = measure.regionprops(measure.label(current, connectivity=2))
    candidate_components = measure.regionprops(measure.label(candidate, connectivity=2))
    disjoint_new_components = 0
    components_with_new_pixels = 0
    for region in candidate_components:
        rows, columns = region.coords.T
        overlap = int(current[rows, columns].sum())
        novel = int((~current[rows, columns]).sum())
        disjoint_new_components += int(overlap == 0)
        components_with_new_pixels += int(novel > 0)
    return {
        "background_off_to_on_pixels": int((~current & candidate & background).sum()),
        "background_on_to_off_pixels": int((current & ~candidate & background).sum()),
        "foreground_off_to_on_pixels": int((~current & candidate & foreground).sum()),
        "foreground_on_to_off_pixels": int((current & ~candidate & foreground).sum()),
        "binary_changed_pixels": int(np.count_nonzero(current != candidate)),
        "current_component_count": len(current_components),
        "candidate_component_count": len(candidate_components),
        "candidate_components_with_new_pixels": components_with_new_pixels,
        "disjoint_new_candidate_components": disjoint_new_components,
    }


def _dominant_branch(values: Mapping[str, float]) -> str:
    maximum = max(values.values())
    winners = [
        name
        for name, value in values.items()
        if math.isclose(value, maximum, rel_tol=0.0, abs_tol=1e-12)
    ]
    return winners[0] if len(winners) == 1 else "tie"


def attribute_new_unmatched_components(
    *,
    identifier: str,
    current_probability: np.ndarray,
    full_probability: np.ndarray,
    target: np.ndarray,
    direct_contribution: np.ndarray,
    rescue_contribution: np.ndarray,
    suppression_contribution: np.ndarray,
) -> list[dict[str, Any]]:
    """Attribute A2 unmatched components containing pixels absent from A0."""

    shape = current_probability.shape
    for value in (
        full_probability,
        target,
        direct_contribution,
        rescue_contribution,
        suppression_contribution,
    ):
        _require(value.shape == shape, "component-attribution array shape differs")
    current = current_probability > FIXED_THRESHOLD
    candidate = full_probability > FIXED_THRESHOLD
    target_binary = target > FIXED_THRESHOLD
    predicted_regions, _targets, _matched_targets, matched_predictions = (
        state_core._match_regions(candidate, target_binary)
    )
    output: list[dict[str, Any]] = []
    for index, region in enumerate(predicted_regions):
        if index in matched_predictions:
            continue
        rows, columns = region.coords.T
        current_overlap = int(current[rows, columns].sum())
        novel_pixels = int((~current[rows, columns]).sum())
        if novel_pixels == 0:
            continue
        branch_abs_means = {
            "direct": float(np.abs(direct_contribution[rows, columns]).mean()),
            "rescue": float(np.abs(rescue_contribution[rows, columns]).mean()),
            "suppression": float(
                np.abs(suppression_contribution[rows, columns]).mean()
            ),
        }
        output.append(
            {
                "identifier": identifier,
                "predicted_component_index": index,
                "area": int(region.area),
                "centroid_row": float(region.centroid[0]),
                "centroid_column": float(region.centroid[1]),
                "current_overlap_pixels": current_overlap,
                "novel_pixels": novel_pixels,
                "disjoint_from_current": current_overlap == 0,
                "dominant_branch_by_absolute_mean": _dominant_branch(
                    branch_abs_means
                ),
                "branch_absolute_means": branch_abs_means,
            }
        )
    return output


def _empty_transition_totals() -> dict[str, int]:
    return {
        "background_off_to_on_pixels": 0,
        "background_on_to_off_pixels": 0,
        "foreground_off_to_on_pixels": 0,
        "foreground_on_to_off_pixels": 0,
        "binary_changed_pixels": 0,
        "current_component_count": 0,
        "candidate_component_count": 0,
        "candidate_components_with_new_pixels": 0,
        "disjoint_new_candidate_components": 0,
    }


def _add_counts(destination: dict[str, int], source: Mapping[str, int]) -> None:
    _require(set(destination) == set(source), "count fields differ")
    for key, value in source.items():
        destination[key] += int(value)


def _collated_identifier(sample_ids: Any) -> str:
    _require(
        isinstance(sample_ids, (tuple, list)) and len(sample_ids) == 1,
        "loader must collate exactly one sample ID",
    )
    return str(sample_ids[0])


@torch.inference_mode()
def collect_current_a0(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    expected_identifiers: Sequence[str],
) -> dict[str, Any]:
    """Collect A0 and verify that Current returns sigmoid(raw out)."""

    model.eval()
    model.mode = "test"
    _require(not model.training, "Current model is not in evaluation mode")
    state_before = module_state_sha256(model)
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    identifiers: list[str] = []
    criterion = nn.BCELoss(reduction="mean")
    stream = hashlib.sha256()
    returned_exact = True
    with RawQ4D0OutCapture(model) as capture:
        for images, masks, sizes, sample_ids in loader:
            _require(images.shape[0] == masks.shape[0] == 1, "batch size differs")
            identifier = _collated_identifier(sample_ids)
            height, width = metric_core._extract_hw(sizes)
            images = images.to(device, non_blocking=device.type == "cuda")
            masks = masks.to(device, non_blocking=device.type == "cuda")
            capture.begin_batch()
            returned = model(images)
            q4, z_out, z_d0 = capture.finish_batch()
            returned_probability = metric_core._final_prediction(returned)
            exact = torch.equal(returned_probability, torch.sigmoid(z_out))
            returned_exact = returned_exact and exact
            _require(exact, "Current return is not sigmoid(raw out)")
            for name, value in (("q4", q4), ("z_out", z_out), ("z_d0", z_d0)):
                _update_tensor_digest(stream, name, value)
            probability = returned_probability[:, :, :height, :width]
            target = masks[:, :, :height, :width]
            _require(probability.shape == target.shape, "A0 prediction/target differ")
            loss = criterion(probability.float(), target.float())
            _require(math.isfinite(float(loss.item())), "A0 loss is non-finite")
            probabilities.append(
                np.array(probability[0, 0].float().cpu().numpy(), copy=True)
            )
            targets.append(np.array(target[0, 0].float().cpu().numpy(), copy=True))
            losses.append(float(loss.item()))
            identifiers.append(identifier)
    _require(capture.temporary_hooks_restored, "Current hooks were not restored")
    _require(identifiers == list(expected_identifiers), "Current inference order differs")
    _require(
        capture.total_counts
        == {"q4": len(identifiers), "z_out": len(identifiers), "z_d0": len(identifiers)},
        "Current capture counts differ",
    )
    state_after = module_state_sha256(model)
    _require(state_before == state_after, "Current model state changed")
    return {
        "probabilities": probabilities,
        "targets": targets,
        "losses": losses,
        "identifiers": identifiers,
        "capture": {
            "image_count": len(identifiers),
            "one_model_forward_per_image": True,
            "q4_capture_count": capture.total_counts["q4"],
            "z_out_capture_count": capture.total_counts["z_out"],
            "z_d0_capture_count": capture.total_counts["z_d0"],
            "temporary_hooks_restored": capture.temporary_hooks_restored,
            "returned_probability_equals_sigmoid_raw_out_bitwise": returned_exact,
            "raw_tensor_stream_sha256": stream.hexdigest(),
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_state_unchanged": state_before == state_after,
        },
    }


@torch.inference_mode()
def collect_pbdr_a1_a8(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    expected_identifiers: Sequence[str],
    current_probabilities: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Collect A1--A8 from one PBDR backbone forward per image."""

    model.eval()
    model.mode = "test"
    _require(not model.training, "PBDR model is not in evaluation mode")
    router = getattr(model, "pbdr_v2", None)
    if type(router) is not PersistentEvidenceResidualRouterV2:
        raise TypeError("PBDR model lacks the exact formal V2 router")
    router = cast(PersistentEvidenceResidualRouterV2, router)
    state_before = module_state_sha256(model)
    criterion = nn.BCELoss(reduction="mean")
    probabilities = {mode: [] for mode in PBDR_PUBLIC_TO_ABLATION}
    losses = {mode: [] for mode in PBDR_PUBLIC_TO_ABLATION}
    targets: list[np.ndarray] = []
    identifiers: list[str] = []
    transitions = {
        mode: _empty_transition_totals() for mode in PBDR_PUBLIC_TO_ABLATION
    }
    per_image: list[dict[str, Any]] = []
    full_component_records: list[dict[str, Any]] = []
    stream = hashlib.sha256()
    returned_full_exact = True
    with RawQ4D0OutCapture(model) as capture:
        for batch_index, (images, masks, sizes, sample_ids) in enumerate(loader):
            _require(images.shape[0] == masks.shape[0] == 1, "batch size differs")
            _require(batch_index < len(current_probabilities), "Current cache is short")
            identifier = _collated_identifier(sample_ids)
            height, width = metric_core._extract_hw(sizes)
            images = images.to(device, non_blocking=device.type == "cuda")
            masks = masks.to(device, non_blocking=device.type == "cuda")
            capture.begin_batch()
            returned = model(images)
            q4, z_out, z_d0 = capture.finish_batch()
            diagnostics = router.forward_with_diagnostics(z_out, z_d0, q4)
            returned_probability = metric_core._final_prediction(returned)
            exact = torch.equal(
                returned_probability,
                torch.sigmoid(diagnostics.routed_logits),
            )
            returned_full_exact = returned_full_exact and exact
            _require(exact, "PBDR return differs from recomputed full route")
            for name, value in (("q4", q4), ("z_out", z_out), ("z_d0", z_d0)):
                _update_tensor_digest(stream, name, value)

            valid = (slice(None), slice(None), slice(0, height), slice(0, width))
            target_tensor = masks[valid]
            target_array = np.array(
                target_tensor[0, 0].float().cpu().numpy(), copy=True
            )
            current_array = np.asarray(current_probabilities[batch_index])
            _require(current_array.shape == target_array.shape, "Current/PBDR shape differs")
            foreground = target_tensor > FIXED_THRESHOLD
            background = ~foreground
            rescue_strength = _scalar_map(diagnostics.rescue_strength)
            suppression_strength = _scalar_map(diagnostics.suppression_strength)
            direct = diagnostics.direct_residual[valid]
            rescue = (rescue_strength * diagnostics.target_rescue)[valid]
            suppression = (
                -suppression_strength * diagnostics.background_suppression
            )[valid]
            delta = direct + rescue + suppression
            branch_tensors = {
                "confidence_C": diagnostics.confidence[valid],
                "direct_Q": direct,
                "target_rescue_R_plus": diagnostics.target_rescue[valid],
                "background_suppression_R_minus": diagnostics.background_suppression[
                    valid
                ],
                "applied_rescue": rescue,
                "applied_suppression": suppression,
                "total_delta_logit": delta,
            }
            branch_statistics = {
                name: {
                    "all_valid": _distribution_summary(value),
                    "gt_foreground": _distribution_summary(value, foreground),
                    "gt_background": _distribution_summary(value, background),
                }
                for name, value in branch_tensors.items()
            }

            image_mode_transitions: dict[str, Any] = {}
            image_probabilities: dict[str, np.ndarray] = {}
            for public_mode, ablation_mode in PBDR_PUBLIC_TO_ABLATION.items():
                logits = ablation_logits_from_diagnostics(
                    z_out,
                    z_d0,
                    diagnostics,
                    ablation_mode,
                )
                probability_tensor = torch.sigmoid(logits)[valid]
                _require(
                    probability_tensor.shape == target_tensor.shape,
                    f"{public_mode} prediction shape differs",
                )
                loss = criterion(probability_tensor.float(), target_tensor.float())
                _require(math.isfinite(float(loss.item())), f"{public_mode} loss non-finite")
                probability_array = np.array(
                    probability_tensor[0, 0].float().cpu().numpy(), copy=True
                )
                probabilities[public_mode].append(probability_array)
                losses[public_mode].append(float(loss.item()))
                image_probabilities[public_mode] = probability_array
                counts = binary_transition_counts(
                    current_array,
                    probability_array,
                    target_array,
                )
                image_mode_transitions[public_mode] = counts
                _add_counts(transitions[public_mode], counts)

            direct_array = np.array(direct[0, 0].float().cpu().numpy(), copy=True)
            rescue_array = np.array(rescue[0, 0].float().cpu().numpy(), copy=True)
            suppression_array = np.array(
                suppression[0, 0].float().cpu().numpy(), copy=True
            )
            component_records = attribute_new_unmatched_components(
                identifier=identifier,
                current_probability=current_array,
                full_probability=image_probabilities["A2_pbdr_full_routed"],
                target=target_array,
                direct_contribution=direct_array,
                rescue_contribution=rescue_array,
                suppression_contribution=suppression_array,
            )
            full_component_records.extend(component_records)
            per_image.append(
                {
                    "identifier": identifier,
                    "height": height,
                    "width": width,
                    "target_pixel_count": int(np.count_nonzero(target_array > 0.5)),
                    "empty_target_image": bool(not np.any(target_array > 0.5)),
                    "q4_pre_normalization": q4_pre_normalization_statistics(q4),
                    "branches": branch_statistics,
                    "transitions_vs_A0": image_mode_transitions,
                    "A2_new_unmatched_component_count": len(component_records),
                }
            )
            targets.append(target_array)
            identifiers.append(identifier)

    _require(capture.temporary_hooks_restored, "PBDR hooks were not restored")
    _require(identifiers == list(expected_identifiers), "PBDR inference order differs")
    _require(len(identifiers) == len(current_probabilities), "PBDR cache length differs")
    expected_counts = {
        "q4": len(identifiers),
        "z_out": len(identifiers),
        "z_d0": len(identifiers),
    }
    _require(capture.total_counts == expected_counts, "PBDR capture counts differ")
    state_after = module_state_sha256(model)
    _require(state_before == state_after, "PBDR model state changed")
    dominance_counts = {"direct": 0, "rescue": 0, "suppression": 0, "tie": 0}
    for record in full_component_records:
        dominance_counts[str(record["dominant_branch_by_absolute_mean"])] += 1
    return {
        "probabilities": probabilities,
        "targets": targets,
        "losses": losses,
        "identifiers": identifiers,
        "transitions": transitions,
        "per_image_diagnostics": per_image,
        "A2_new_unmatched_component_attribution": {
            "component_count": len(full_component_records),
            "disjoint_from_current_count": sum(
                int(record["disjoint_from_current"])
                for record in full_component_records
            ),
            "dominance_counts": dominance_counts,
            "records": full_component_records,
        },
        "router": router_parameter_audit(router),
        "capture": {
            "image_count": len(identifiers),
            "one_backbone_forward_per_image": True,
            "same_raw_tensors_reused_for_A1_through_A8": True,
            "q4_capture_count": capture.total_counts["q4"],
            "z_out_capture_count": capture.total_counts["z_out"],
            "z_d0_capture_count": capture.total_counts["z_d0"],
            "temporary_hooks_restored": capture.temporary_hooks_restored,
            "returned_full_probability_recomputed_bitwise": returned_full_exact,
            "raw_tensor_stream_sha256": stream.hexdigest(),
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_state_unchanged": state_before == state_after,
        },
    }


def _point_delta(
    point: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, int | float | None]:
    output: dict[str, int | float | None] = {}
    for key in (
        "matched_target_count",
        "matched_tiny_target_count",
        "unmatched_predicted_pixels",
        "unmatched_predicted_object_count",
        "miou",
        "niou",
        "pd",
        "tiny_pd",
        "fa",
        "false_objects_per_image",
    ):
        left = point.get(key)
        right = current.get(key)
        output[key] = None if left is None or right is None else left - right
    return output


def matched_working_points(
    points: Sequence[Mapping[str, Any]],
    current_fixed: Mapping[str, Any],
) -> dict[str, Any]:
    """Select exact-Pd and no-higher-Fa descriptive comparison points."""

    _require(bool(points), "working-point grid is empty")
    current_targets = int(current_fixed["target_count"])
    current_matched = int(current_fixed["matched_target_count"])
    current_fa = float(current_fixed["fa"])
    for point in points:
        _require(int(point["target_count"]) == current_targets, "target denominator differs")
    same_pd = [
        point for point in points if int(point["matched_target_count"]) == current_matched
    ]
    fa_feasible = [point for point in points if float(point["fa"]) <= current_fa]

    def ready(point: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if point is None:
            return None
        value = dict(point)
        value["delta_vs_A0_fixed_0_5"] = _point_delta(point, current_fixed)
        return value

    same_pd_min_fa = (
        min(
            same_pd,
            key=lambda point: (
                float(point["fa"]),
                -float(point["miou"]),
                -float(point["niou"]),
                abs(float(point["threshold"]) - FIXED_THRESHOLD),
            ),
        )
        if same_pd
        else None
    )
    same_pd_max_miou = (
        max(
            same_pd,
            key=lambda point: (
                float(point["miou"]),
                -float(point["fa"]),
                float(point["niou"]),
                -abs(float(point["threshold"]) - FIXED_THRESHOLD),
            ),
        )
        if same_pd
        else None
    )
    fa_feasible_max_pd = (
        max(
            fa_feasible,
            key=lambda point: (
                int(point["matched_target_count"]),
                -float(point["fa"]),
                float(point["miou"]),
                float(point["niou"]),
            ),
        )
        if fa_feasible
        else None
    )
    fa_feasible_max_miou = (
        max(
            fa_feasible,
            key=lambda point: (
                float(point["miou"]),
                int(point["matched_target_count"]),
                -float(point["fa"]),
                float(point["niou"]),
            ),
        )
        if fa_feasible
        else None
    )
    dominating = [
        point
        for point in points
        if int(point["matched_target_count"]) >= current_matched
        and float(point["fa"]) <= current_fa
        and float(point["miou"]) >= float(current_fixed["miou"])
        and float(point["niou"]) >= float(current_fixed["niou"])
    ]
    return {
        "same_Pd_definition": "exact matched_target_count with invariant target_count",
        "same_Pd_registered_point_count": len(same_pd),
        "same_Pd_minimum_Fa": ready(same_pd_min_fa),
        "same_Pd_maximum_mIoU": ready(same_pd_max_miou),
        "same_Fa_definition": "candidate Fa <= A0 fixed-threshold Fa",
        "same_Fa_registered_point_count": len(fa_feasible),
        "same_Fa_maximum_Pd": ready(fa_feasible_max_pd),
        "same_Fa_maximum_mIoU": ready(fa_feasible_max_miou),
        "four_metric_non_regression_registered_points": [
            ready(point) for point in dominating
        ],
        "threshold_grid_can_fully_restore_A0": bool(dominating),
    }


def evaluate_attribution_mode(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    losses: Sequence[float],
    *,
    threshold_grid: Sequence[float],
    current_fixed: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reuse the frozen metric evaluator for fixed, sweep, and FROC points."""

    registered = tuple(float(value) for value in threshold_grid)
    _require(EMPTY_ENDPOINT not in registered, "analysis grid contains empty endpoint")
    evaluated = metric_core.evaluate_probability_arrays(
        probabilities,
        targets,
        losses,
        sweep_thresholds=(*registered, EMPTY_ENDPOINT),
    )
    fixed = evaluated["fixed_threshold_0_5"]
    all_points = evaluated["descriptive_pd_fa"]["points"]
    points = [point for point in all_points if float(point["threshold"]) in registered]
    endpoint = [
        point for point in all_points if float(point["threshold"]) == EMPTY_ENDPOINT
    ]
    _require(len(points) == len(registered), "registered sweep point count differs")
    _require(len(endpoint) == 1, "empty endpoint count differs")
    output = {
        "fixed_threshold_0_5": fixed,
        "threshold_sweep": {
            "role": "descriptive_failure_attribution_only",
            "selection_effect": "none",
            "registered_grid": list(registered),
            "registered_point_count": len(points),
            "points": points,
            "froc": {
                "x": "fa",
                "y": "pd",
                "points": [
                    {
                        "threshold": point["threshold"],
                        "fa": point["fa"],
                        "pd": point["pd"],
                        "matched_target_count": point["matched_target_count"],
                        "target_count": point["target_count"],
                    }
                    for point in points
                ],
                "pareto_frontier": protocol_metrics.pareto_frontier(points),
            },
            "miou_fa_curve": [
                {
                    "threshold": point["threshold"],
                    "fa": point["fa"],
                    "miou": point["miou"],
                }
                for point in points
            ],
            "niou_fa_curve": [
                {
                    "threshold": point["threshold"],
                    "fa": point["fa"],
                    "niou": point["niou"],
                }
                for point in points
            ],
            "threshold_1_0_empty_control": endpoint[0],
            "fa_budget_controls": evaluated["descriptive_pd_fa"][
                "best_points_under_fa_budget"
            ],
        },
    }
    if current_fixed is not None:
        output["fixed_delta_vs_A0"] = _point_delta(fixed, current_fixed)
        output["matched_working_points_vs_A0"] = matched_working_points(
            points,
            current_fixed,
        )
    return output


def checkpoint_metric_audit_under_tf32_off(
    checkpoint_payload: Mapping[str, Any],
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit an old cuDNN-TF32 checkpoint under the stricter A0--A8 math.

    The frozen Current/PBDR-V2 evaluators reproduced checkpoint selection with
    their legacy cuDNN setting, while this attribution deliberately disables
    both cuDNN and matmul TF32.  Discrete identity/count fields must remain
    exact; overlap metrics receive only a bounded numerical-kernel tolerance.
    """

    raw = checkpoint_payload.get("test_metrics")
    _require(isinstance(raw, Mapping), "checkpoint lacks test_metrics")
    missing = [
        key
        for key in metric_core.REQUIRED_CHECKPOINT_METRICS
        if key not in raw or key not in fixed
    ]
    _require(not missing, f"checkpoint metric audit lacks fields: {missing}")
    count_keys = {
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    }
    soft_keys = {
        "miou",
        "niou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for key, expected in raw.items():
        if key not in fixed:
            continue
        observed = fixed[key]
        if key in count_keys:
            _require(observed == expected, f"strict-math checkpoint count differs: {key}")
            comparisons[key] = {
                "checkpoint": int(expected),
                "observed_tf32_off": int(observed),
                "absolute_delta": 0,
                "absolute_tolerance": 0,
            }
            continue
        if expected is None:
            _require(observed is None, f"strict-math checkpoint null differs: {key}")
            comparisons[key] = {
                "checkpoint": None,
                "observed_tf32_off": None,
                "absolute_delta": 0.0,
                "absolute_tolerance": 0.0,
            }
            continue
        if key == "fa":
            valid_pixels = int(fixed["valid_pixel_count"])
            _require(valid_pixels > 0, "strict-math valid pixel count differs")
            tolerance = STRICT_MATH_MAX_FA_PIXEL_DRIFT / valid_pixels
        else:
            tolerance = (
                STRICT_MATH_SOFT_METRIC_TOLERANCE
                if key in soft_keys
                else STRICT_MATH_TEST_LOSS_TOLERANCE
                if key == "test_loss"
                else 1.0e-15
            )
        delta = float(observed) - float(expected)
        _require(
            math.isclose(
                float(observed),
                float(expected),
                rel_tol=0.0,
                abs_tol=tolerance,
            ),
            f"strict-math checkpoint metric differs beyond tolerance: {key}",
        )
        comparisons[key] = {
            "checkpoint": float(expected),
            "observed_tf32_off": float(observed),
            "absolute_delta": abs(delta),
            "signed_delta": delta,
            "absolute_tolerance": tolerance,
        }
        if key == "fa":
            comparisons[key]["equivalent_absolute_pixel_delta"] = (
                abs(delta) * int(fixed["valid_pixel_count"])
            )
            comparisons[key]["maximum_absolute_pixel_drift"] = (
                STRICT_MATH_MAX_FA_PIXEL_DRIFT
            )
    return {
        "passed": True,
        "audit_kind": "legacy_checkpoint_vs_explicit_tf32_off_replay",
        "discrete_counts_required_exact": True,
        "analysis_cuda_matmul_tf32": False,
        "analysis_cudnn_tf32": False,
        "legacy_checkpoint_evaluator_explicitly_disabled_cudnn_tf32": False,
        "soft_metric_absolute_tolerance": STRICT_MATH_SOFT_METRIC_TOLERANCE,
        "test_loss_absolute_tolerance": STRICT_MATH_TEST_LOSS_TOLERANCE,
        "maximum_fa_pixel_drift": STRICT_MATH_MAX_FA_PIXEL_DRIFT,
        "comparisons": comparisons,
    }


def _artifact_snapshot(binding: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for key in ("summary", "protocol", "checkpoint"):
        record = binding.get(key)
        _require(isinstance(record, Mapping), f"binding lacks {key}")
        path = Path(str(record.get("path", ""))).resolve(strict=True)
        output[key] = {"path": str(path), "sha256": file_sha256(path)}
    return output


def _make_loader(
    dataset: metric_core.ThreeDatasetTestDataset,
    device: torch.device,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def _load_current(
    *,
    checkpoint_role: str,
    run_dir: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], nn.Module, dict[str, Any]]:
    current_evaluator.configure_core()
    request = metric_core.EvaluationRequest(
        dataset=DATASET,
        method="final",
        checkpoint_role=checkpoint_role,
        requested_tss_weight=0.0,
    )
    request.validate()
    # Current predates additive model files.  This context still validates
    # every source recorded in its protocol by absolute path and SHA; it only
    # ignores unrelated files added later to model/.
    with state_core.historical_checkpoint_loader_compatibility():
        checkpoint, binding = metric_core.load_checkpoint(
            request,
            run_dir,
            manifest_path=manifest_path,
            manifest=manifest,
        )
    model, metadata = metric_core.build_inference_model(
        request,
        checkpoint["state_dict"],
    )
    _require(len(model.state_dict()) == CURRENT_INFERENCE_STATE_KEY_COUNT, "Current inference state-key count differs")
    _require(metadata.get("strict_load") is True, "Current checkpoint load was not strict")
    return checkpoint, binding, model, metadata


def _load_pbdr(
    *,
    checkpoint_role: str,
    run_dir: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], nn.Module, dict[str, Any], dict[str, Any]]:
    identity = pbdr_evaluator._validate_training_identity(run_dir, DATASET)
    request = metric_core.EvaluationRequest(
        dataset=DATASET,
        method="final",
        checkpoint_role=checkpoint_role,
        requested_tss_weight=0.0,
    )
    with pbdr_evaluator._configured_core():
        request.validate()
        checkpoint, binding = metric_core.load_checkpoint(
            request,
            run_dir,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        model, metadata = pbdr_evaluator._build_inference_model(
            request,
            checkpoint["state_dict"],
        )
    _require(len(model.state_dict()) == PBDR_INFERENCE_STATE_KEY_COUNT, "PBDR inference state-key count differs")
    _require(metadata.get("strict_load") is True, "PBDR checkpoint load was not strict")
    return checkpoint, binding, model, metadata, identity


def analyze_run(
    *,
    checkpoint_role: str,
    current_run_dir: Path = DEFAULT_CURRENT_RUN_DIR,
    pbdr_run_dir: Path = DEFAULT_PBDR_RUN_DIR,
    dataset_root: Path = data_protocol.DEFAULT_DATASET_ROOT,
    data_protocol_manifest: Path = data_protocol.DEFAULT_MANIFEST_PATH,
    device_name: str = "cuda:0",
    workers: int = 0,
    threshold_start: float = THRESHOLD_START,
    threshold_stop: float = THRESHOLD_STOP,
    threshold_step: float = THRESHOLD_STEP,
) -> dict[str, Any]:
    """Execute one paired-role A0--A8 attribution without model mutation."""

    _require(checkpoint_role in CHECKPOINT_ROLES, "checkpoint role differs")
    _require(workers >= 0, "workers must be non-negative")
    device = torch.device(device_name)
    _require(device.type in {"cpu", "cuda"}, "device must be CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    thresholds = build_threshold_grid(threshold_start, threshold_stop, threshold_step)
    sources_before = _source_sha256()
    inference_math = configure_inference_math()
    manifest_path = Path(data_protocol_manifest).resolve(strict=True)
    manifest = data_protocol.load_protocol_manifest(
        manifest_path,
        dataset_root=dataset_root,
    )
    dataset = metric_core.ThreeDatasetTestDataset(
        dataset_root,
        DATASET,
        manifest_path,
    )

    current_checkpoint, current_binding, current_model, current_metadata = (
        _load_current(
            checkpoint_role=checkpoint_role,
            run_dir=Path(current_run_dir),
            manifest_path=manifest_path,
            manifest=manifest,
        )
    )
    current_artifacts_before = _artifact_snapshot(current_binding)
    current_model.to(device)
    current_collected = collect_current_a0(
        current_model,
        _make_loader(dataset, device, workers),
        device,
        dataset.sample_ids,
    )
    current_test_metrics = dict(current_checkpoint["test_metrics"])
    del current_model, current_checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    pbdr_checkpoint, pbdr_binding, pbdr_model, pbdr_metadata, pbdr_identity = (
        _load_pbdr(
            checkpoint_role=checkpoint_role,
            run_dir=Path(pbdr_run_dir),
            manifest_path=manifest_path,
            manifest=manifest,
        )
    )
    pbdr_artifacts_before = _artifact_snapshot(pbdr_binding)
    pbdr_model.to(device)
    pbdr_collected = collect_pbdr_a1_a8(
        pbdr_model,
        _make_loader(dataset, device, workers),
        device,
        dataset.sample_ids,
        current_collected["probabilities"],
    )
    pbdr_test_metrics = dict(pbdr_checkpoint["test_metrics"])
    del pbdr_model, pbdr_checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    _require(
        current_collected["identifiers"] == pbdr_collected["identifiers"],
        "A0/PBDR inference order differs",
    )
    _require(
        all(
            np.array_equal(current_target, pbdr_target)
            for current_target, pbdr_target in zip(
                current_collected["targets"], pbdr_collected["targets"]
            )
        ),
        "A0/PBDR target arrays differ",
    )

    a0 = evaluate_attribution_mode(
        current_collected["probabilities"],
        current_collected["targets"],
        current_collected["losses"],
        threshold_grid=thresholds,
        current_fixed=None,
    )
    current_fixed = a0["fixed_threshold_0_5"]
    current_checkpoint_audit = checkpoint_metric_audit_under_tf32_off(
        {"test_metrics": current_test_metrics},
        current_fixed,
    )
    mode_results: dict[str, Any] = {
        "A0_current_out": {
            "mode": "A0_current_out",
            "checkpoint_family": "Current",
            "formula": MODE_FORMULAS["A0_current_out"],
            **a0,
            "fixed_delta_vs_A0": _point_delta(current_fixed, current_fixed),
            "matched_working_points_vs_A0": matched_working_points(
                a0["threshold_sweep"]["points"], current_fixed
            ),
            "transitions_vs_A0": _empty_transition_totals(),
        }
    }
    for mode in PBDR_PUBLIC_TO_ABLATION:
        evaluated = evaluate_attribution_mode(
            pbdr_collected["probabilities"][mode],
            pbdr_collected["targets"],
            pbdr_collected["losses"][mode],
            threshold_grid=thresholds,
            current_fixed=current_fixed,
        )
        mode_results[mode] = {
            "mode": mode,
            "checkpoint_family": "PBDR-V2",
            "ablation_mode": PBDR_PUBLIC_TO_ABLATION[mode],
            "formula": MODE_FORMULAS[mode],
            **evaluated,
            "transitions_vs_A0": pbdr_collected["transitions"][mode],
        }
    _require(tuple(mode_results) == MODE_ORDER, "A0--A8 result order differs")
    pbdr_checkpoint_audit = checkpoint_metric_audit_under_tf32_off(
        {"test_metrics": pbdr_test_metrics},
        mode_results["A2_pbdr_full_routed"]["fixed_threshold_0_5"],
    )

    current_artifacts_after = _artifact_snapshot(current_binding)
    pbdr_artifacts_after = _artifact_snapshot(pbdr_binding)
    _require(current_artifacts_after == current_artifacts_before, "Current artifacts changed")
    _require(pbdr_artifacts_after == pbdr_artifacts_before, "PBDR artifacts changed")
    _require(_source_sha256() == sources_before, "runtime source changed")

    output = {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "checkpoint_role": checkpoint_role,
        "seed": SEED,
        "split": "img_idx/test",
        "test_selected": True,
        "selection_is_optimistic": True,
        "mode_order": list(MODE_ORDER),
        "modes": mode_results,
        "protocol": {
            "fixed_threshold": FIXED_THRESHOLD,
            "prediction_operator": ">",
            "threshold_grid_start": threshold_start,
            "threshold_grid_stop": threshold_stop,
            "threshold_grid_step": threshold_step,
            "threshold_grid": list(thresholds),
            "threshold_grid_point_count": len(thresholds),
            "threshold_1_0_empty_control": True,
            "sweep_reselects_checkpoint": False,
            "match_radius": metric_core.MATCH_RADIUS,
            "tiny_area": metric_core.TINY_AREA,
        },
        "formal_replay_audits": {
            "A0_current_checkpoint_metrics": current_checkpoint_audit,
            "A2_pbdr_checkpoint_metrics": pbdr_checkpoint_audit,
        },
        "router_parameter_audit": pbdr_collected["router"],
        "per_image_diagnostics": pbdr_collected["per_image_diagnostics"],
        "A2_new_unmatched_component_attribution": pbdr_collected[
            "A2_new_unmatched_component_attribution"
        ],
        "capture": {
            "A0_current": current_collected["capture"],
            "A1_through_A8_pbdr": pbdr_collected["capture"],
            "image_order_newline_sha256": hashlib.sha256(
                ("\n".join(current_collected["identifiers"]) + "\n").encode(
                    "utf-8"
                )
            ).hexdigest(),
        },
        "bindings": {
            "current": {
                "artifacts": current_artifacts_before,
                "checkpoint_binding": current_binding,
            },
            "pbdr_v2": {
                "artifacts": pbdr_artifacts_before,
                "checkpoint_binding": pbdr_binding,
                "training_identity": pbdr_identity,
            },
            "data_protocol_manifest": {
                "path": str(manifest_path),
                "sha256": file_sha256(manifest_path),
                "schema": manifest.get("schema"),
                "manifest_id": manifest.get("manifest_id"),
            },
        },
        "models": {"Current": current_metadata, "PBDR-V2": pbdr_metadata},
        "inference_math": inference_math,
        "source_sha256": sources_before,
        "read_only_contract": {
            "torch_inference_mode": True,
            "optimizer_constructed": False,
            "backward_called": False,
            "model_eval_called": True,
            "training_mode_enabled": False,
            "model_state_modified": False,
            "checkpoint_written": False,
            "derived_checkpoint_written": False,
            "raw_logit_cache_written": False,
            "probability_cache_written": False,
            "only_json_summary_may_be_written_by_cli": True,
        },
        "no_fabricated_results": True,
        "training_started": False,
    }
    validate_output_payload(output)
    return output


def validate_output_payload(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema") == SCHEMA, "output schema differs")
    _require(payload.get("status") == "complete", "output is incomplete")
    _require(payload.get("dataset") == DATASET, "output dataset differs")
    _require(payload.get("checkpoint_role") in CHECKPOINT_ROLES, "output role differs")
    _require(payload.get("mode_order") == list(MODE_ORDER), "mode order differs")
    modes = payload.get("modes")
    _require(isinstance(modes, Mapping), "output lacks modes")
    _require(tuple(modes) == MODE_ORDER, "mode mapping order differs")
    invariant: tuple[int, int] | None = None
    for mode in MODE_ORDER:
        point = modes[mode]
        _require(point.get("mode") == mode, f"mode identity differs: {mode}")
        fixed = point.get("fixed_threshold_0_5")
        _require(isinstance(fixed, Mapping), f"{mode} lacks fixed metrics")
        this_invariant = (int(fixed["target_count"]), int(fixed["valid_pixel_count"]))
        invariant = invariant or this_invariant
        _require(this_invariant == invariant, "mode metric denominators differ")
        _require(float(fixed["threshold"]) == FIXED_THRESHOLD, "fixed threshold differs")
        sweep = point.get("threshold_sweep")
        _require(isinstance(sweep, Mapping), f"{mode} lacks threshold sweep")
        _require(
            int(sweep.get("registered_point_count", -1))
            == len(payload["protocol"]["threshold_grid"]),
            "threshold sweep count differs",
        )
        endpoint = sweep.get("threshold_1_0_empty_control")
        _require(
            isinstance(endpoint, Mapping)
            and float(endpoint.get("threshold", -1.0)) == EMPTY_ENDPOINT
            and int(endpoint.get("predicted_object_count", -1)) == 0
            and float(endpoint.get("pd", -1.0)) == 0.0
            and float(endpoint.get("fa", -1.0)) == 0.0,
            "empty endpoint differs",
        )
    audits = payload.get("formal_replay_audits")
    _require(
        isinstance(audits, Mapping)
        and audits.get("A0_current_checkpoint_metrics", {}).get("passed") is True
        and audits.get("A2_pbdr_checkpoint_metrics", {}).get("passed") is True,
        "formal replay audit differs",
    )
    capture = payload.get("capture")
    _require(isinstance(capture, Mapping), "capture audit is missing")
    for key in ("A0_current", "A1_through_A8_pbdr"):
        value = capture.get(key)
        _require(
            isinstance(value, Mapping)
            and value.get("temporary_hooks_restored") is True
            and value.get("model_state_unchanged") is True,
            f"{key} capture/restoration differs",
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
        "inference math differs",
    )
    read_only = payload.get("read_only_contract")
    _require(
        isinstance(read_only, Mapping)
        and read_only.get("optimizer_constructed") is False
        and read_only.get("backward_called") is False
        and read_only.get("model_state_modified") is False
        and read_only.get("checkpoint_written") is False
        and read_only.get("raw_logit_cache_written") is False,
        "read-only contract differs",
    )
    _require(payload.get("training_started") is False, "training flag differs")
    json.dumps(payload, ensure_ascii=False, allow_nan=False)


def _default_output(checkpoint_role: str) -> Path:
    return (
        DEFAULT_OUTPUT_ROOT
        / "runs"
        / DATASET
        / checkpoint_role
        / "attribution.json"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-role", choices=CHECKPOINT_ROLES, required=True)
    parser.add_argument("--current-run-dir", type=Path, default=DEFAULT_CURRENT_RUN_DIR)
    parser.add_argument("--pbdr-run-dir", type=Path, default=DEFAULT_PBDR_RUN_DIR)
    parser.add_argument(
        "--dataset-root", type=Path, default=data_protocol.DEFAULT_DATASET_ROOT
    )
    parser.add_argument(
        "--data-protocol-manifest",
        type=Path,
        default=data_protocol.DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument("--threshold-start", type=float, default=THRESHOLD_START)
    parser.add_argument("--threshold-stop", type=float, default=THRESHOLD_STOP)
    parser.add_argument("--threshold-step", type=float, default=THRESHOLD_STEP)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if not (args.device == "cpu" or args.device.startswith("cuda:")):
        parser.error("--device must be cpu or cuda:N")
    try:
        build_threshold_grid(
            args.threshold_start,
            args.threshold_stop,
            args.threshold_step,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    args.output = args.output or _default_output(args.checkpoint_role)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing existing output: {args.output}")
    output = analyze_run(
        checkpoint_role=args.checkpoint_role,
        current_run_dir=args.current_run_dir,
        pbdr_run_dir=args.pbdr_run_dir,
        dataset_root=args.dataset_root,
        data_protocol_manifest=args.data_protocol_manifest,
        device_name=args.device,
        workers=args.workers,
        threshold_start=args.threshold_start,
        threshold_stop=args.threshold_stop,
        threshold_step=args.threshold_step,
    )
    metric_core._atomic_write_json(args.output, output, overwrite=False)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output.resolve()),
                "sha256": file_sha256(args.output.resolve()),
                "dataset": DATASET,
                "checkpoint_role": args.checkpoint_role,
                "training_started": False,
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
    "AblationMode",
    "CHECKPOINT_ROLES",
    "DATASET",
    "MODE_ORDER",
    "PBDRV2AblationWrapper",
    "PBDR_ABLATION_MODES",
    "SCHEMA",
    "RawQ4D0OutCapture",
    "ablation_logits_from_diagnostics",
    "analyze_run",
    "attribute_new_unmatched_components",
    "binary_transition_counts",
    "build_threshold_grid",
    "collect_current_a0",
    "collect_pbdr_a1_a8",
    "configure_inference_math",
    "matched_working_points",
    "q4_pre_normalization_statistics",
    "router_parameter_audit",
    "validate_output_payload",
]
