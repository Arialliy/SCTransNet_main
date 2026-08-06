#!/usr/bin/env python3
"""Zero-training branch audit for the proposed GCSF skip fusion.

The audit consumes one frozen seed-42 TSS-off checkpoint at a time.  For each
test batch the CNN encoder, TPD8 tokenizers, QFG2 bridge, and SCTB encoder run
exactly once.  Their forward-local encoder branches ``E`` and reconstructed
transformed branches ``T`` are then reused by eleven decoder-only modes:

* the production-order current fusion ``(T + E) + E``;
* ``g=-0.25`` at L1, L2, L3, L4, or all four levels;
* ``g=+0.25`` at L1, L2, L3, L4, or all four levels.

For a selected level a mode appends the order-preserving correction
``g*T - g*E`` to the already-computed production baseline.  Thus all tested
counterfactuals are representable by the proposed constant-sum family, while
the current mode is bitwise anchored to the production operation order.  No
feature/probability cache or derived checkpoint is written.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import analyze_ner_stage2_mask_knockout_v1 as stage2_audit  # noqa: E402
from analysis import analyze_three_dataset_qfg_level_knockout_v1 as qfg_audit  # noqa: E402
from experiments import evaluate_three_dataset_tss_off_seed42_v1 as adapter  # noqa: E402
from experiments import evaluate_three_dataset_v2 as core  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import (  # noqa: E402
    train_four_dataset_original_final_seed42_exact_v1 as training_engine,
)
from model import (  # noqa: E402
    tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival as production_model,
)
from model import tpd_query_frequency_bridge as frequency_bridge  # noqa: E402


SCHEMA = "sctransnet_three_dataset_gcsf_branch_audit_v1/v1"
STATISTICS_SCHEMA = "sctransnet_gcsf_branch_statistics_v1/v1"
REFERENCE_METHOD = "final_tss_off"
TRAINING_MODEL_METHOD = "final"
SEED = 42
FIXED_THRESHOLD = 0.5
SWEEP_THRESHOLDS = (0.5, 1.0)
EVALUATION_PROTOCOL = "img_idx_test_selected_development"
CHECKPOINT_ROLES = tuple(core.CHECKPOINT_ROLES)
LEVEL_CHANNELS = (32, 64, 128, 256)
LEVEL_NAMES = ("L1", "L2", "L3", "L4")
GATE_MAGNITUDE = 0.25
AMPLITUDE_EPSILON = 1e-12

MODE_SPECS: dict[str, tuple[float, tuple[int, ...]]] = {
    "current_g0": (0.0, ()),
    "gneg025_l1_only": (-GATE_MAGNITUDE, (0,)),
    "gneg025_l2_only": (-GATE_MAGNITUDE, (1,)),
    "gneg025_l3_only": (-GATE_MAGNITUDE, (2,)),
    "gneg025_l4_only": (-GATE_MAGNITUDE, (3,)),
    "gneg025_all_levels": (-GATE_MAGNITUDE, (0, 1, 2, 3)),
    "gpos025_l1_only": (GATE_MAGNITUDE, (0,)),
    "gpos025_l2_only": (GATE_MAGNITUDE, (1,)),
    "gpos025_l3_only": (GATE_MAGNITUDE, (2,)),
    "gpos025_l4_only": (GATE_MAGNITUDE, (3,)),
    "gpos025_all_levels": (GATE_MAGNITUDE, (0, 1, 2, 3)),
}
PUBLIC_MODES = tuple(MODE_SPECS)
CURRENT_MODE = PUBLIC_MODES[0]

DEFAULT_TSS_OFF_ROOT = REPO_ROOT / "results" / "three_dataset_tss_off_seed42_v1"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "three_dataset_gcsf_branch_audit_v1"

probability_difference = qfg_audit.probability_difference
reference_replay_audit = qfg_audit.reference_replay_audit
atomic_create_json = qfg_audit.atomic_create_json


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    return stage2_audit.file_sha256(Path(path))


def normalize_public_mode(public_mode: str) -> dict[str, Any]:
    _require(public_mode in MODE_SPECS, f"unknown GCSF audit mode: {public_mode!r}")
    gate, levels = MODE_SPECS[public_mode]
    return {
        "public_mode": public_mode,
        "gate_value": gate,
        "selected_level_indices_zero_based": list(levels),
        "selected_level_names": [LEVEL_NAMES[index] for index in levels],
        "transformed_coefficient_selected": 1.0 + gate,
        "encoder_coefficient_selected": 2.0 - gate,
        "coefficient_sum": 3.0,
        "diagnostic_only": public_mode != CURRENT_MODE,
    }


@dataclass(frozen=True, slots=True)
class ForwardLocalBranches:
    """Forward-local tensors retained only until eleven decoders finish."""

    encoder: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    transformed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    d5: torch.Tensor
    evidence1: tuple[torch.Tensor, ...]
    evidence2: tuple[torch.Tensor, ...]


def _validate_branches(
    transformed: Sequence[torch.Tensor],
    encoder: Sequence[torch.Tensor],
) -> None:
    _require(len(transformed) == len(encoder) == 4, "GCSF audit requires four levels")
    for index, (one_t, one_e) in enumerate(zip(transformed, encoder)):
        _require(
            isinstance(one_t, torch.Tensor) and isinstance(one_e, torch.Tensor),
            f"level {index} branch is not a tensor",
        )
        _require(one_t.ndim == 4, f"level {index} branch is not BCHW")
        _require(one_t.shape == one_e.shape, f"level {index} T/E shape differs")
        _require(one_t.device == one_e.device, f"level {index} T/E device differs")
        _require(one_t.dtype == one_e.dtype, f"level {index} T/E dtype differs")
        _require(
            bool(torch.isfinite(one_t).all()) and bool(torch.isfinite(one_e).all()),
            f"level {index} T/E contains non-finite values",
        )


def production_order_baseline(
    transformed: Sequence[torch.Tensor],
    encoder: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    """Reproduce the current ``(T + E) + E`` operation order exactly."""

    _validate_branches(transformed, encoder)
    return tuple((one_t + one_e) + one_e for one_t, one_e in zip(transformed, encoder))


def fuse_public_mode(
    transformed: Sequence[torch.Tensor],
    encoder: Sequence[torch.Tensor],
    public_mode: str,
) -> tuple[torch.Tensor, ...]:
    """Apply one representable constant-sum counterfactual to cached T/E."""

    binding = normalize_public_mode(public_mode)
    baseline = production_order_baseline(transformed, encoder)
    gate = float(binding["gate_value"])
    selected = frozenset(binding["selected_level_indices_zero_based"])
    if not selected:
        _require(gate == 0.0, "empty GCSF selection must be current g=0")
        return baseline
    _require(gate in (-GATE_MAGNITUDE, GATE_MAGNITUDE), "nonzero gate differs")
    ready: list[torch.Tensor] = []
    for index, (current, one_t, one_e) in enumerate(
        zip(baseline, transformed, encoder)
    ):
        if index not in selected:
            ready.append(current)
            continue
        scalar = one_t.new_tensor(gate)
        # Preserve the already-computed production baseline.  Do not regroup
        # the current expression into coefficient multiplications.
        correction = scalar * one_t - scalar * one_e
        ready.append(current + correction)
    return tuple(ready)


def prepare_forward_local_branches(
    model: nn.Module,
    images: torch.Tensor,
) -> ForwardLocalBranches:
    """Run encoder/TPD/QFG once and expose decoder-independent branches."""

    _require(not model.training, "GCSF branch audit requires model.eval()")
    x1 = model.inc(images)
    x2 = model.down_encoder1(model.pool(x1))
    x3 = model.down_encoder2(model.pool(x2))
    x4 = model.down_encoder3(model.pool(x3))
    d5 = model.down_encoder4(model.pool(x4))
    encoder = (x1, x2, x3, x4)

    emb1, emb2, emb3, emb4, evidence1, evidence2 = model.explicit_embeddings(
        x1, x2, x3, x4
    )
    _require(
        isinstance(evidence1, tuple)
        and isinstance(evidence2, tuple)
        and len(evidence1) == 3
        and len(evidence2) == 2,
        "formal TPD evidence layout differs from 3+2",
    )
    prepared_qfg = model.tpd_qfg.prepare(
        encoder,
        tuple(tuple(embedding.shape[-2:]) for embedding in (emb1, emb2, emb3, emb4)),
    )
    encoded1, encoded2, encoded3, encoded4, _ = (
        frequency_bridge.frequency_encoder_forward(
            model.mtc.encoder,
            emb1,
            emb2,
            emb3,
            emb4,
            model.tpd_qfg,
            prepared_qfg,
        )
    )
    transformed = (
        model.mtc.reconstruct_1(encoded1),
        model.mtc.reconstruct_2(encoded2),
        model.mtc.reconstruct_3(encoded3),
        model.mtc.reconstruct_4(encoded4),
    )
    _validate_branches(transformed, encoder)
    return ForwardLocalBranches(
        encoder=encoder,
        transformed=transformed,
        d5=d5,
        evidence1=evidence1,
        evidence2=evidence2,
    )


def decode_forward_local_mode(
    model: nn.Module,
    prepared: ForwardLocalBranches,
    public_mode: str,
) -> torch.Tensor:
    """Rerun only CCA/NER/decoder for one frozen fusion mode."""

    x1, x2, x3, x4 = fuse_public_mode(
        prepared.transformed, prepared.encoder, public_mode
    )
    h11, h12, h13 = prepared.evidence1
    h21, h22 = prepared.evidence2

    up4, skip4 = model.up_decoder4.prepare(prepared.d5, x4)
    q4, mask4 = model.tpd_ner.forward_stage(
        4, (h13, h22, up4), tuple(up4.shape[-2:])
    )
    d4 = model.up_decoder4.finish(up4, skip4, mask4)
    up3, skip3 = model.up_decoder3.prepare(d4, x3)
    q3, mask3 = model.tpd_ner.forward_stage(
        3, (h12, h21, q4, up3), tuple(up3.shape[-2:])
    )
    d3 = model.up_decoder3.finish(up3, skip3, mask3)
    up2, skip2 = model.up_decoder2.prepare(d3, x2)
    _, mask2 = model.tpd_ner.forward_stage(
        2, (h11, q3, up2), tuple(up2.shape[-2:])
    )
    d2 = model.up_decoder2.finish(up2, skip2, mask2)
    output = torch.sigmoid(model.outc(model.up_decoder1(d2, x1)))
    _require(output.ndim == 4 and output.shape[1] == 1, "decoder output differs")
    _require(bool(torch.isfinite(output).all()), "decoder output is non-finite")
    return output


@dataclass(slots=True)
class _BranchLevelAccumulator:
    valid_location_count: int = 0
    target_location_count: int = 0
    background_location_count: int = 0
    transformed_square_sum: float = 0.0
    encoder_square_sum: float = 0.0
    dot_sum: float = 0.0
    target_transformed_square_sum: float = 0.0
    target_encoder_square_sum: float = 0.0
    background_transformed_square_sum: float = 0.0
    background_encoder_square_sum: float = 0.0
    observed_shapes_chw: set[tuple[int, int, int]] = field(default_factory=set)


class BranchStatisticsAccumulator:
    """Padding-aware aggregate T/E statistics; never retains feature tensors."""

    def __init__(self) -> None:
        self.levels = tuple(_BranchLevelAccumulator() for _ in range(4))
        self.batch_count = 0

    @staticmethod
    def _masked_square_sum(value: torch.Tensor, mask: torch.Tensor) -> float:
        return float((value.detach().double().square() * mask.double()).sum().item())

    def append(
        self,
        transformed: Sequence[torch.Tensor],
        encoder: Sequence[torch.Tensor],
        target: torch.Tensor,
        original_size: tuple[int, int],
    ) -> None:
        _validate_branches(transformed, encoder)
        _require(target.ndim == 4 and target.shape[1] == 1, "target must be Bx1xHxW")
        height, width = (int(original_size[0]), int(original_size[1]))
        _require(
            0 < height <= target.shape[-2] and 0 < width <= target.shape[-1],
            "original size lies outside padded target",
        )
        valid = torch.zeros_like(target, dtype=torch.bool)
        valid[..., :height, :width] = True
        target_binary = target > 0.5
        for index, (one_t, one_e, accumulator) in enumerate(
            zip(transformed, encoder, self.levels)
        ):
            output_size = tuple(one_t.shape[-2:])
            pooled_valid = F.adaptive_max_pool2d(valid.float(), output_size) > 0.5
            pooled_target = F.adaptive_max_pool2d(
                target_binary.float() * valid.float(), output_size
            ) > 0.5
            target_region = torch.logical_and(pooled_valid, pooled_target)
            background_region = torch.logical_and(
                pooled_valid, torch.logical_not(pooled_target)
            )
            channels = int(one_t.shape[1])
            accumulator.valid_location_count += int(pooled_valid.sum().item())
            accumulator.target_location_count += int(target_region.sum().item())
            accumulator.background_location_count += int(background_region.sum().item())
            accumulator.transformed_square_sum += self._masked_square_sum(one_t, pooled_valid)
            accumulator.encoder_square_sum += self._masked_square_sum(one_e, pooled_valid)
            accumulator.dot_sum += float(
                (one_t.detach().double() * one_e.detach().double() * pooled_valid.double())
                .sum()
                .item()
            )
            accumulator.target_transformed_square_sum += self._masked_square_sum(
                one_t, target_region
            )
            accumulator.target_encoder_square_sum += self._masked_square_sum(
                one_e, target_region
            )
            accumulator.background_transformed_square_sum += self._masked_square_sum(
                one_t, background_region
            )
            accumulator.background_encoder_square_sum += self._masked_square_sum(
                one_e, background_region
            )
            accumulator.observed_shapes_chw.add(
                (channels, int(one_t.shape[-2]), int(one_t.shape[-1]))
            )
        self.batch_count += 1

    @staticmethod
    def _rms(square_sum: float, spatial_count: int, channels: int) -> float | None:
        element_count = spatial_count * channels
        return None if element_count == 0 else math.sqrt(square_sum / element_count)

    def summary(self) -> dict[str, Any]:
        _require(self.batch_count > 0, "branch statistics have no batches")
        rows: list[dict[str, Any]] = []
        for index, accumulator in enumerate(self.levels):
            _require(bool(accumulator.observed_shapes_chw), f"L{index + 1} was not observed")
            channels_set = {shape[0] for shape in accumulator.observed_shapes_chw}
            _require(len(channels_set) == 1, f"L{index + 1} channel count changed")
            channels = next(iter(channels_set))
            rms_t = self._rms(
                accumulator.transformed_square_sum,
                accumulator.valid_location_count,
                channels,
            )
            rms_e = self._rms(
                accumulator.encoder_square_sum,
                accumulator.valid_location_count,
                channels,
            )
            target_t = self._rms(
                accumulator.target_transformed_square_sum,
                accumulator.target_location_count,
                channels,
            )
            target_e = self._rms(
                accumulator.target_encoder_square_sum,
                accumulator.target_location_count,
                channels,
            )
            background_t = self._rms(
                accumulator.background_transformed_square_sum,
                accumulator.background_location_count,
                channels,
            )
            background_e = self._rms(
                accumulator.background_encoder_square_sum,
                accumulator.background_location_count,
                channels,
            )
            _require(rms_t is not None and rms_e is not None, "valid branch RMS is absent")
            cosine_denominator = math.sqrt(
                accumulator.transformed_square_sum * accumulator.encoder_square_sum
            )
            cosine = (
                None
                if cosine_denominator == 0.0
                else accumulator.dot_sum / cosine_denominator
            )
            if cosine is not None:
                cosine = max(-1.0, min(1.0, cosine))
            rows.append(
                {
                    "level_index_zero_based": index,
                    "level_name": LEVEL_NAMES[index],
                    "channels": channels,
                    "observed_shapes_chw": [
                        list(shape) for shape in sorted(accumulator.observed_shapes_chw)
                    ],
                    "valid_spatial_location_count": accumulator.valid_location_count,
                    "target_spatial_location_count": accumulator.target_location_count,
                    "background_spatial_location_count": accumulator.background_location_count,
                    "transformed_rms": rms_t,
                    "encoder_rms": rms_e,
                    "transformed_to_encoder_rms_ratio": rms_t / (rms_e + AMPLITUDE_EPSILON),
                    "transformed_encoder_cosine": cosine,
                    "target_transformed_rms": target_t,
                    "target_encoder_rms": target_e,
                    "background_transformed_rms": background_t,
                    "background_encoder_rms": background_e,
                    "transformed_target_to_background_rms_ratio": (
                        None
                        if target_t is None or background_t is None
                        else target_t / (background_t + AMPLITUDE_EPSILON)
                    ),
                    "encoder_target_to_background_rms_ratio": (
                        None
                        if target_e is None or background_e is None
                        else target_e / (background_e + AMPLITUDE_EPSILON)
                    ),
                    "current_transformed_amplitude_share_proxy": (
                        rms_t / (rms_t + 2.0 * rms_e + AMPLITUDE_EPSILON)
                    ),
                }
            )
        return {
            "schema": STATISTICS_SCHEMA,
            "batch_count": self.batch_count,
            "level_count": 4,
            "level_order": list(LEVEL_NAMES),
            "target_projection": "adaptive_max_pool2d_binary_presence",
            "valid_projection": "adaptive_max_pool2d_any_original_support",
            "background_region": "pooled_valid_and_not_pooled_target",
            "padding_policy": "exclude_bins_with_no_original_pixel_support",
            "cosine_aggregation": "global_masked_dot_over_global_masked_l2_product",
            "amplitude_share_proxy_formula": "RMS(T)/(RMS(T)+2*RMS(E)+1e-12)",
            "feature_tensors_retained_after_batch": False,
            "levels": rows,
        }


def _annotate_two_point_sweep(evaluated: Mapping[str, Any]) -> None:
    descriptive = evaluated.get("descriptive_pd_fa")
    _require(isinstance(descriptive, Mapping), "descriptive Pd-Fa result missing")
    points = descriptive.get("points")
    _require(isinstance(points, list) and len(points) == 2, "sweep must have two points")
    _require(
        [float(point.get("threshold")) for point in points] == list(SWEEP_THRESHOLDS),
        "sweep thresholds differ",
    )
    empty = points[1]
    empty["selected_point_is_empty"] = core._point_is_empty(empty)
    _require(
        empty["selected_point_is_empty"] is True
        and float(empty.get("pd")) == 0.0
        and float(empty.get("fa")) == 0.0,
        "threshold=1.0 is not the legal empty endpoint",
    )


@torch.inference_mode()
def analyze_loaded_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    expected_identifiers: Sequence[str],
) -> dict[str, Any]:
    """Run eleven decoder modes while preparing T/E once per batch."""

    model.eval()
    _require(not model.training, "GCSF branch audit model must be eval")
    state_before = stage2_audit.module_state_sha256(model)
    probabilities: dict[str, list[np.ndarray]] = {mode: [] for mode in PUBLIC_MODES}
    losses: dict[str, list[float]] = {mode: [] for mode in PUBLIC_MODES}
    targets: list[np.ndarray] = []
    identifiers: list[str] = []
    criterion = nn.BCELoss(reduction="mean")
    statistics = BranchStatisticsAccumulator()
    batch_count = 0

    for images, masks, sizes, sample_ids in loader:
        _require(
            int(images.shape[0]) == int(masks.shape[0]) == 1,
            "GCSF audit requires batch_size=1",
        )
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        height, width = core._extract_hw(sizes)
        prepared = prepare_forward_local_branches(model, images)
        statistics.append(
            prepared.transformed,
            prepared.encoder,
            masks,
            (height, width),
        )
        target = masks[:, :, :height, :width]
        for public_mode in PUBLIC_MODES:
            prediction = decode_forward_local_mode(model, prepared, public_mode)[
                :, :, :height, :width
            ]
            _require(prediction.shape == target.shape, "prediction/target shape differs")
            one_loss = criterion(prediction.float(), target.float())
            _require(math.isfinite(float(one_loss.item())), "loss is non-finite")
            probabilities[public_mode].append(
                prediction[0, 0].float().cpu().contiguous().numpy().astype(np.float32, copy=False)
            )
            losses[public_mode].append(float(one_loss.item()))
        targets.append(target[0, 0].float().cpu().contiguous().numpy())
        _require(
            isinstance(sample_ids, (tuple, list)) and len(sample_ids) == 1,
            "GCSF audit requires one sample ID per batch",
        )
        identifiers.append(str(sample_ids[0]))
        batch_count += 1
        del prepared

    _require(identifiers == list(expected_identifiers), "inference order differs")
    _require(batch_count == len(loader.dataset), "inference count differs")
    current_probabilities = probabilities[CURRENT_MODE]
    modes: dict[str, Any] = {}
    for public_mode in PUBLIC_MODES:
        evaluated = core.evaluate_probability_arrays(
            probabilities[public_mode],
            targets,
            losses[public_mode],
            sweep_thresholds=SWEEP_THRESHOLDS,
        )
        _annotate_two_point_sweep(evaluated)
        fixed = dict(evaluated["fixed_threshold_0_5"])
        fixed["false_positive_pixels"] = qfg_audit._all_background_false_positive_pixels(
            probabilities[public_mode], targets, threshold=FIXED_THRESHOLD
        )
        modes[public_mode] = {
            **normalize_public_mode(public_mode),
            "fixed_threshold_0_5": fixed,
            "descriptive_pd_fa": evaluated["descriptive_pd_fa"],
            "threshold_roles": evaluated["threshold_roles"],
            "sweep_thresholds": list(SWEEP_THRESHOLDS),
            "probability_difference_to_current": probability_difference(
                current_probabilities, probabilities[public_mode]
            ),
        }

    state_after = stage2_audit.module_state_sha256(model)
    _require(state_after == state_before, "GCSF audit changed model state")
    return {
        "modes": modes,
        "branch_statistics": statistics.summary(),
        "execution_audit": {
            "batch_count": batch_count,
            "encoder_tpd_qfg_prepare_count": batch_count,
            "decoder_mode_count_per_batch": len(PUBLIC_MODES),
            "decoder_execution_count": batch_count * len(PUBLIC_MODES),
            "encoder_tpd_qfg_recomputed_per_mode": False,
            "forward_local_feature_reuse_only": True,
        },
        "restoration_audit": {
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_state_unchanged": state_after == state_before,
        },
        "probability_arrays_persisted": False,
        "feature_tensors_persisted": False,
    }


def _frozen_source_sha256() -> dict[str, str]:
    sources = {
        "analysis/analyze_three_dataset_gcsf_branch_audit_v1.py": Path(__file__),
        "analysis/analyze_ner_stage2_mask_knockout_v1.py": Path(stage2_audit.__file__),
        "analysis/analyze_three_dataset_qfg_level_knockout_v1.py": Path(qfg_audit.__file__),
        "experiments/evaluate_three_dataset_v2.py": Path(core.__file__),
        "experiments/evaluate_three_dataset_tss_off_seed42_v1.py": Path(adapter.__file__),
        "model/tpd_query_frequency_bridge.py": Path(frequency_bridge.__file__),
        (
            "model/tpd_ner_v8_mprs_dch_v4_tail_aware_"
            "qfg_v2_croa_survival.py"
        ): Path(production_model.__file__),
    }
    return {name: file_sha256(path.resolve(strict=True)) for name, path in sources.items()}


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
    _require(dataset in data_protocol.DATASETS, "dataset is outside formal scope")
    _require(checkpoint_role in CHECKPOINT_ROLES, "checkpoint role differs")
    _require(workers >= 0, "workers must be non-negative")
    frozen_sources = _frozen_source_sha256()
    adapter.configure_core()
    training_engine.configure_determinism()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    manifest_path = Path(data_protocol_manifest).resolve(strict=True)
    manifest = data_protocol.load_protocol_manifest(manifest_path, dataset_root=dataset_root)
    request = core.EvaluationRequest(
        dataset=dataset,
        method=TRAINING_MODEL_METHOD,
        checkpoint_role=checkpoint_role,
        requested_tss_weight=adapter.REQUESTED_TSS_WEIGHT,
    )
    request.validate()
    checkpoint_payload, checkpoint_binding = stage2_audit._load_checkpoint_allowing_added_sources(
        request, Path(run_dir), manifest_path, manifest
    )
    reference_path = Path(reference_evaluation).resolve(strict=True)
    reference = adapter.validate_completed_output(
        reference_path, dataset=dataset, checkpoint_role=checkpoint_role
    )
    _require(
        reference["checkpoint_binding"]["checkpoint"]["sha256"]
        == checkpoint_binding["checkpoint"]["sha256"],
        "reference evaluation/checkpoint SHA differs",
    )

    model, model_metadata = core.build_inference_model(request, checkpoint_payload["state_dict"])
    model.to(device)
    model.eval()
    dataset_object = core.ThreeDatasetTestDataset(dataset_root, dataset, manifest_path)
    loader = DataLoader(
        dataset_object,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    analyzed = analyze_loaded_model(model, loader, device, dataset_object.sample_ids)
    replay = reference_replay_audit(
        analyzed["modes"][CURRENT_MODE]["fixed_threshold_0_5"],
        reference["fixed_threshold_0_5"],
    )
    replay["comparison"] = (
        f"current_g0_fixed_threshold_0_5_vs_existing_{checkpoint_role}"
    )
    _require(replay["passed"] is True, "current reference replay failed")
    ordered_id_sha = hashlib.sha256(
        ("\n".join(dataset_object.sample_ids) + "\n").encode("utf-8")
    ).hexdigest()
    _require(_frozen_source_sha256() == frozen_sources, "runtime source changed after freeze")

    output = {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": dataset,
        "method": REFERENCE_METHOD,
        "training_model_method": TRAINING_MODEL_METHOD,
        "checkpoint_role": checkpoint_role,
        "seed": SEED,
        "test_selected": True,
        "selection_is_optimistic": True,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "fixed_threshold": FIXED_THRESHOLD,
        "sweep_thresholds": list(SWEEP_THRESHOLDS),
        "mode_order": list(PUBLIC_MODES),
        **analyzed,
        "reference_replay_audit": replay,
        "reference_reuse": {
            "path": str(reference_path),
            "sha256": file_sha256(reference_path),
            "checkpoint_role": checkpoint_role,
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
            "descriptive_sweep_thresholds": list(SWEEP_THRESHOLDS),
            "sweep_reselects_checkpoint": False,
        },
        "intervention_contract": {
            "family": "GCSF_constant_sum_representable_counterfactual",
            "current_formula_operation_order": "(T+E)+E",
            "selected_correction_operation_order": "baseline+(g*T-g*E)",
            "tested_nonzero_gate_values": [-GATE_MAGNITUDE, GATE_MAGNITUDE],
            "unrepresentable_f1_t_plus_e_used_for_trigger": False,
            "unrepresentable_f3_2t_plus_e_used_for_trigger": False,
            "model_state_modified": False,
            "derived_checkpoint_written": False,
        },
        "source_lock_policy": {
            "source_hash_frozen_before_inference": True,
            "source_hash_reverified_before_output": True,
            "historical_frozen_dependencies_verified_by_path_and_sha256": True,
            "new_unlisted_model_sources_allowed": True,
            "verified_historical_runtime_sources": checkpoint_binding[
                "training_runtime_sources"
            ]["source_sha256"],
        },
        "source_sha256": frozen_sources,
        "derived_checkpoint_written": False,
        "probability_cache_written": False,
        "feature_cache_written": False,
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
    _require(payload.get("schema") == SCHEMA, "GCSF analyzer schema differs")
    _require(payload.get("status") == "complete", "GCSF analyzer is incomplete")
    _require(payload.get("dataset") in data_protocol.DATASETS, "dataset differs")
    role = payload.get("checkpoint_role")
    _require(role in CHECKPOINT_ROLES, "checkpoint role differs")
    _require(payload.get("method") == REFERENCE_METHOD, "method differs")
    _require(
        payload.get("training_model_method") == TRAINING_MODEL_METHOD,
        "training model method differs",
    )
    _require(payload.get("seed") == SEED, "seed differs")
    _require(payload.get("test_selected") is True, "test_selected differs")
    _require(
        payload.get("selection_is_optimistic") is True,
        "selection policy differs",
    )
    _require(payload.get("evaluation_protocol") == EVALUATION_PROTOCOL, "protocol differs")
    _require(payload.get("sweep_thresholds") == list(SWEEP_THRESHOLDS), "sweep differs")
    _require(payload.get("mode_order") == list(PUBLIC_MODES), "mode order differs")
    replay = payload.get("reference_replay_audit")
    _require(
        isinstance(replay, Mapping)
        and replay.get("passed") is True
        and replay.get("comparison")
        == f"current_g0_fixed_threshold_0_5_vs_existing_{role}",
        "reference replay differs",
    )
    checkpoint_binding = payload.get("checkpoint_binding")
    _require(isinstance(checkpoint_binding, Mapping), "checkpoint binding missing")
    checkpoint = checkpoint_binding.get("checkpoint")
    protocol = checkpoint_binding.get("protocol")
    _require(
        isinstance(checkpoint, Mapping)
        and checkpoint.get("role") == role
        and isinstance(checkpoint.get("sha256"), str)
        and len(checkpoint["sha256"]) == 64,
        "checkpoint file binding differs",
    )
    _require(
        isinstance(protocol, Mapping)
        and isinstance(protocol.get("payload_sha256"), str)
        and len(protocol["payload_sha256"]) == 64,
        "checkpoint protocol binding differs",
    )
    reference = payload.get("reference_reuse")
    _require(
        isinstance(reference, Mapping)
        and reference.get("checkpoint_role") == role
        and isinstance(reference.get("sha256"), str)
        and len(reference["sha256"]) == 64,
        "reference binding differs",
    )
    data = payload.get("data")
    _require(isinstance(data, Mapping) and data.get("split") == "img_idx/test", "data differs")
    manifest = data.get("protocol_manifest")
    _require(
        isinstance(manifest, Mapping)
        and isinstance(manifest.get("sha256"), str)
        and len(manifest["sha256"]) == 64
        and isinstance(data.get("inference_order_newline_sha256"), str)
        and len(data["inference_order_newline_sha256"]) == 64,
        "data identity SHA differs",
    )
    sources = payload.get("source_sha256")
    _require(
        isinstance(sources, Mapping)
        and bool(sources)
        and all(isinstance(value, str) and len(value) == 64 for value in sources.values()),
        "source SHA map differs",
    )
    intervention = payload.get("intervention_contract")
    _require(
        isinstance(intervention, Mapping)
        and intervention.get("family")
        == "GCSF_constant_sum_representable_counterfactual"
        and intervention.get("current_formula_operation_order") == "(T+E)+E"
        and intervention.get("selected_correction_operation_order")
        == "baseline+(g*T-g*E)"
        and intervention.get("unrepresentable_f1_t_plus_e_used_for_trigger") is False
        and intervention.get("unrepresentable_f3_2t_plus_e_used_for_trigger") is False
        and intervention.get("model_state_modified") is False
        and intervention.get("derived_checkpoint_written") is False,
        "intervention contract differs",
    )
    modes = payload.get("modes")
    _require(isinstance(modes, Mapping) and set(modes) == set(PUBLIC_MODES), "modes differ")
    invariant: tuple[int, int, int] | None = None
    required_fixed = {
        "test_loss",
        "matched_target_count",
        "matched_tiny_target_count",
        "miou",
        "niou",
        "unmatched_predicted_pixels",
        "false_positive_pixels",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "valid_pixel_count",
    }
    for public_mode in PUBLIC_MODES:
        mode = modes[public_mode]
        _require(isinstance(mode, Mapping), f"{public_mode} is malformed")
        _require(mode.get("public_mode") == public_mode, f"{public_mode} binding differs")
        binding = normalize_public_mode(public_mode)
        for key in (
            "gate_value",
            "selected_level_indices_zero_based",
            "transformed_coefficient_selected",
            "encoder_coefficient_selected",
            "coefficient_sum",
        ):
            _require(mode.get(key) == binding[key], f"{public_mode}.{key} differs")
        fixed = mode.get("fixed_threshold_0_5")
        _require(isinstance(fixed, Mapping) and required_fixed <= set(fixed), "fixed fields missing")
        _require(float(fixed.get("threshold")) == FIXED_THRESHOLD, "threshold differs")
        current_invariant = (
            int(fixed.get("target_count")),
            int(fixed.get("tiny_target_count")),
            int(fixed.get("valid_pixel_count")),
        )
        invariant = invariant or current_invariant
        _require(current_invariant == invariant, "mode target/pixel totals differ")
        descriptive = mode.get("descriptive_pd_fa")
        _require(isinstance(descriptive, Mapping), "descriptive sweep missing")
        points = descriptive.get("points")
        _require(isinstance(points, list) and len(points) == 2, "two points required")
        _require([float(point["threshold"]) for point in points] == [0.5, 1.0], "points differ")
        _require(
            points[1].get("selected_point_is_empty") is True
            and float(points[1].get("pd")) == 0.0
            and float(points[1].get("fa")) == 0.0,
            "empty endpoint differs",
        )
        difference = mode.get("probability_difference_to_current")
        _require(isinstance(difference, Mapping), "probability difference missing")
        element_count = int(difference.get("element_count", 0))
        _require(element_count == current_invariant[2] and element_count > 0, "difference count differs")
        absolute_sum = float(difference.get("absolute_difference_sum"))
        _require(
            float(difference.get("mean_abs")) == absolute_sum / element_count,
            "difference mean identity differs",
        )
        if public_mode == CURRENT_MODE:
            _require(
                float(difference.get("max_abs")) == 0.0 and absolute_sum == 0.0,
                "current self-difference is nonzero",
            )
    statistics = payload.get("branch_statistics")
    _require(isinstance(statistics, Mapping), "branch statistics missing")
    _require(statistics.get("schema") == STATISTICS_SCHEMA, "statistics schema differs")
    _require(statistics.get("level_order") == list(LEVEL_NAMES), "statistics order differs")
    rows = statistics.get("levels")
    _require(isinstance(rows, list) and len(rows) == 4, "statistics require four levels")
    for index, row in enumerate(rows):
        _require(row.get("level_index_zero_based") == index, "statistics level differs")
        _require(int(row.get("valid_spatial_location_count", 0)) > 0, "no valid locations")
        for key in ("transformed_rms", "encoder_rms", "current_transformed_amplitude_share_proxy"):
            value = float(row.get(key))
            _require(math.isfinite(value) and value >= 0.0, f"statistics {key} differs")
    execution = payload.get("execution_audit")
    _require(isinstance(execution, Mapping), "execution audit missing")
    batch_count = int(execution.get("batch_count", 0))
    _require(batch_count > 0, "batch count differs")
    _require(
        execution.get("encoder_tpd_qfg_prepare_count") == batch_count
        and execution.get("decoder_execution_count") == batch_count * len(PUBLIC_MODES)
        and execution.get("encoder_tpd_qfg_recomputed_per_mode") is False,
        "prepare-once execution contract differs",
    )
    restoration = payload.get("restoration_audit")
    _require(
        isinstance(restoration, Mapping)
        and restoration.get("model_state_unchanged") is True
        and restoration.get("model_state_sha256_before")
        == restoration.get("model_state_sha256_after"),
        "model state restoration differs",
    )
    _require(
        payload.get("derived_checkpoint_written") is False
        and payload.get("probability_cache_written") is False
        and payload.get("feature_cache_written") is False,
        "forbidden artifact flag differs",
    )


def _default_run_dir(dataset: str) -> Path:
    return DEFAULT_TSS_OFF_ROOT / "runs" / dataset / "final_tss_off" / "seed_42"


def _default_reference(run_dir: Path, checkpoint_role: str) -> Path:
    return Path(run_dir) / "evaluations" / f"{checkpoint_role}.json"


def _default_output(dataset: str, checkpoint_role: str) -> Path:
    return (
        DEFAULT_OUTPUT_ROOT
        / "runs"
        / dataset
        / f"v4_tss_off_{checkpoint_role}_seed42"
        / "evaluation.json"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=data_protocol.DATASETS, required=True)
    parser.add_argument("--checkpoint-role", choices=CHECKPOINT_ROLES, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--reference-evaluation", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=data_protocol.DEFAULT_DATASET_ROOT)
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
        args.run_dir, args.checkpoint_role
    )
    args.output = args.output or _default_output(args.dataset, args.checkpoint_role)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing existing output before inference: {args.output}")
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
    atomic_create_json(args.output, output)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
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


if __name__ == "__main__":
    main()
