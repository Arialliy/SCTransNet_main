#!/usr/bin/env python3
"""Evaluate frozen TPD8 block-residual knockouts on the three datasets.

For each seed-42 TSS-off ``best_miou`` checkpoint this evaluation-only audit
runs the unchanged model, seven single-block counterfactuals, and an all-seven
counterfactual.  A counterfactual changes only the selected block's learned
``saliency_scale`` in memory, restores every value exactly, and never writes a
derived checkpoint.

The ``full`` pass additionally wraps the seven block forwards and their
production ``aligned_mprs_terms`` calls.  Each wrapper returns the original
object unchanged; the terms already computed by production are observed once,
and only headroom/residual summaries are derived under ``no_grad``.  The
pixel-unshuffle, pooling, and projections are not rerun for statistics.  No
probability cache or feature cache is written.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import analyze_ner_stage2_mask_knockout_v1 as stage2_audit  # noqa: E402
from analysis import analyze_three_dataset_qfg_level_knockout_v1 as qfg_skeleton  # noqa: E402
from experiments import evaluate_three_dataset_tss_off_seed42_v1 as adapter  # noqa: E402
from experiments import evaluate_three_dataset_v2 as core  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import (  # noqa: E402
    train_four_dataset_original_final_seed42_exact_v1 as training_engine,
)
from model import tpd_clean_v8_mprs_dch as tpd8_source  # noqa: E402
from model.tpd_clean_v8_mprs_dch import (  # noqa: E402
    TPDCleanV8MPRSDCHBlock,
)


SCHEMA = "sctransnet_three_dataset_tpd8_block_residual_knockout_v1/v1"
KNOCKOUT_SCHEMA = "sctransnet_tpd8_saliency_scale_knockout_v1/v1"
MECHANISM_SCHEMA = "sctransnet_tpd8_full_mprs_statistics_v1/v1"
REFERENCE_METHOD = "final_tss_off"
TRAINING_MODEL_METHOD = "final"
CHECKPOINT_ROLE = "best_miou"
SEED = 42
FIXED_THRESHOLD = 0.5
SWEEP_THRESHOLDS = (0.5, 1.0)
EVALUATION_PROTOCOL = "img_idx_test_selected_development"

BLOCK_PATHS = (
    "mtc.embeddings_1.blocks.0",
    "mtc.embeddings_1.blocks.1",
    "mtc.embeddings_1.blocks.2",
    "mtc.embeddings_1.blocks.3",
    "mtc.embeddings_2.blocks.0",
    "mtc.embeddings_2.blocks.1",
    "mtc.embeddings_2.blocks.2",
)
BLOCK_IDS = (
    "E1.B0",
    "E1.B1",
    "E1.B2",
    "E1.B3",
    "E2.B0",
    "E2.B1",
    "E2.B2",
)
MODE_TO_BLOCK_PATHS = {
    "full": (),
    "e1b0_off": (BLOCK_PATHS[0],),
    "e1b1_off": (BLOCK_PATHS[1],),
    "e1b2_off": (BLOCK_PATHS[2],),
    "e1b3_off": (BLOCK_PATHS[3],),
    "e2b0_off": (BLOCK_PATHS[4],),
    "e2b1_off": (BLOCK_PATHS[5],),
    "e2b2_off": (BLOCK_PATHS[6],),
    "all7_off": BLOCK_PATHS,
}
PUBLIC_MODES = tuple(MODE_TO_BLOCK_PATHS)

DEFAULT_TSS_OFF_ROOT = REPO_ROOT / "results" / "three_dataset_tss_off_seed42_v1"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results" / "three_dataset_tpd8_block_residual_knockout_v1"
)

# Reuse the already-tested all-original-pixel comparison, reference replay,
# and write-once publication primitives without copying their implementations.
probability_difference = qfg_skeleton.probability_difference
reference_replay_audit = qfg_skeleton.reference_replay_audit
atomic_create_json = qfg_skeleton.atomic_create_json


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    return stage2_audit.file_sha256(Path(path))


def normalize_public_mode(public_mode: str) -> dict[str, Any]:
    _require(public_mode in MODE_TO_BLOCK_PATHS, f"unknown TPD8 mode: {public_mode!r}")
    selected = list(MODE_TO_BLOCK_PATHS[public_mode])
    return {
        "public_mode": public_mode,
        "knockout_block_paths": selected,
        "knockout_block_indices_zero_based": [
            BLOCK_PATHS.index(path) for path in selected
        ],
        "diagnostic_only": public_mode != "full",
    }


def _resolve_tpd8_blocks(
    model: nn.Module,
) -> tuple[tuple[str, TPDCleanV8MPRSDCHBlock], ...]:
    _require(isinstance(model, nn.Module), "model must be a torch module")
    resolved: list[tuple[str, TPDCleanV8MPRSDCHBlock]] = []
    for path in BLOCK_PATHS:
        try:
            block = model.get_submodule(path)
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"TPD8 block is missing: {path}") from exc
        _require(
            isinstance(block, TPDCleanV8MPRSDCHBlock),
            f"unexpected TPD8 block type at {path}",
        )
        _require(block.context_gate == 1.0, f"formal TPD8 context gate differs: {path}")
        projection = block.phase_compress
        _require(
            projection.in_channels == 4 * block.channels
            and projection.out_channels == block.channels,
            f"TPD8 phase projection shape differs: {path}",
        )
        _require(
            block.saliency_scale.shape == (block.channels,),
            f"TPD8 saliency scale shape differs: {path}",
        )
        resolved.append((path, block))
    return tuple(resolved)


def _validate_formal_channels(model: nn.Module) -> None:
    for index, (path, block) in enumerate(_resolve_tpd8_blocks(model)):
        expected = 32 if index < 4 else 64
        _require(block.channels == expected, f"formal channel count differs: {path}")


def _tensor_sha_update(digest: Any, name: str, value: torch.Tensor) -> None:
    ready = value.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(ready.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(ready.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(ready.reshape(-1).view(torch.uint8).numpy().tobytes())


def saliency_scale_state_sha256(model: nn.Module) -> str:
    """Hash the seven learned scale vectors in exact execution order."""

    digest = hashlib.sha256()
    for path, block in _resolve_tpd8_blocks(model):
        _tensor_sha_update(digest, f"{path}.saliency_scale", block.saliency_scale)
    return digest.hexdigest()


def saliency_scale_records(model: nn.Module) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (path, block) in enumerate(_resolve_tpd8_blocks(model)):
        scale = block.saliency_scale.detach().float()
        effective = torch.tanh(scale)
        records.append(
            {
                "block_index_zero_based": index,
                "block_path": path,
                "channels": block.channels,
                "parameter_minimum": float(scale.min().item()),
                "parameter_maximum": float(scale.max().item()),
                "parameter_mean": float(scale.mean().item()),
                "parameter_rms": float(scale.square().mean().sqrt().item()),
                "effective_tanh_minimum": float(effective.min().item()),
                "effective_tanh_maximum": float(effective.max().item()),
                "effective_tanh_mean": float(effective.mean().item()),
                "effective_tanh_rms": float(
                    effective.square().mean().sqrt().item()
                ),
                "nonzero_count": int(torch.count_nonzero(scale).item()),
                "element_count": int(scale.numel()),
                "dtype": str(block.saliency_scale.dtype).removeprefix("torch."),
                "requires_grad": bool(block.saliency_scale.requires_grad),
            }
        )
    return records


@contextlib.contextmanager
def temporary_saliency_scale_knockout(
    model: nn.Module,
    public_mode: str,
) -> Iterator[dict[str, Any]]:
    """Zero selected scale vectors in memory and restore every value exactly."""

    _require(not model.training, "TPD8 knockout requires model.eval()")
    binding = normalize_public_mode(public_mode)
    resolved = _resolve_tpd8_blocks(model)
    selected_paths = tuple(binding["knockout_block_paths"])
    selected = tuple(BLOCK_PATHS.index(path) for path in selected_paths)
    snapshots = tuple(
        block.saliency_scale.detach().clone(memory_format=torch.preserve_format)
        for _, block in resolved
    )
    source_scale_sha = saliency_scale_state_sha256(model)
    source_model_sha = stage2_audit.module_state_sha256(model)
    audit: dict[str, Any] = {
        "schema": KNOCKOUT_SCHEMA,
        "public_mode": public_mode,
        "selected_block_paths": list(selected_paths),
        "selected_block_indices_zero_based": list(selected),
        "source_saliency_scale_sha256": source_scale_sha,
        "source_model_state_sha256": source_model_sha,
        "derived_checkpoint_written": False,
        "diagnostic_only": public_mode != "full",
    }
    try:
        with torch.no_grad():
            for index in selected:
                resolved[index][1].saliency_scale.zero_()
        for index, ((path, block), snapshot) in enumerate(zip(resolved, snapshots)):
            if index in selected:
                _require(
                    int(torch.count_nonzero(block.saliency_scale).item()) == 0,
                    f"selected TPD8 scale was not zeroed: {path}",
                )
            else:
                _require(
                    torch.equal(block.saliency_scale.detach(), snapshot),
                    f"unselected TPD8 scale changed: {path}",
                )
        selected_vectors = []
        for index in selected:
            path, block = resolved[index]
            source_nonzero = int(torch.count_nonzero(snapshots[index]).item())
            active_nonzero = int(
                torch.count_nonzero(block.saliency_scale.detach()).item()
            )
            selected_vectors.append(
                {
                    "block_index_zero_based": index,
                    "block_id": BLOCK_IDS[index],
                    "block_path": path,
                    "element_count": int(snapshots[index].numel()),
                    "source_nonzero_count": source_nonzero,
                    "active_nonzero_count": active_nonzero,
                }
            )
        audit["selected_vectors"] = selected_vectors
        audit["selected_source_nonzero_count"] = sum(
            int(record["source_nonzero_count"]) for record in selected_vectors
        )
        audit["selected_active_nonzero_count"] = sum(
            int(record["active_nonzero_count"]) for record in selected_vectors
        )
        _require(
            audit["selected_active_nonzero_count"] == 0,
            "selected TPD8 scale vectors retain active nonzero values",
        )
        audit["active_saliency_scale_sha256"] = saliency_scale_state_sha256(model)
        audit["active_model_state_sha256"] = stage2_audit.module_state_sha256(model)
        yield audit
    finally:
        with torch.no_grad():
            for index in selected:
                resolved[index][1].saliency_scale.copy_(snapshots[index])
        restored_scale_sha = saliency_scale_state_sha256(model)
        restored_model_sha = stage2_audit.module_state_sha256(model)
        audit.update(
            {
                "restored_saliency_scale_sha256": restored_scale_sha,
                "restored_model_state_sha256": restored_model_sha,
                "saliency_scale_restored_exactly": (
                    restored_scale_sha == source_scale_sha
                ),
                "model_state_restored_exactly": restored_model_sha == source_model_sha,
            }
        )
        if restored_scale_sha != source_scale_sha:
            raise RuntimeError("TPD8 saliency scales were not restored exactly")
        if restored_model_sha != source_model_sha:
            raise RuntimeError("TPD8 model state was not restored exactly")


@dataclass(slots=True)
class _RMSAccumulator:
    element_count: int = 0
    dynamic_element_count: torch.Tensor | None = None
    square_sum: torch.Tensor | None = None

    def _append_square_sum(self, value: torch.Tensor) -> None:
        scalar = value.detach().float()
        self.square_sum = scalar if self.square_sum is None else self.square_sum + scalar

    def append(self, tensor: torch.Tensor) -> None:
        ready = tensor.detach().float()
        self.element_count += int(ready.numel())
        self._append_square_sum(ready.square().sum())

    def append_spatial_mask(
        self,
        tensor: torch.Tensor,
        spatial_mask: torch.Tensor,
    ) -> None:
        ready = tensor.detach().float()
        mask = spatial_mask.detach().to(device=ready.device, dtype=ready.dtype)
        _require(
            mask.ndim == 4
            and mask.shape[0] == ready.shape[0]
            and mask.shape[1] == 1
            and mask.shape[-2:] == ready.shape[-2:],
            "TPD8 statistic mask shape differs",
        )
        self._append_square_sum((ready.square() * mask).sum())
        count = mask.sum() * float(ready.shape[1])
        self.dynamic_element_count = (
            count
            if self.dynamic_element_count is None
            else self.dynamic_element_count + count
        )

    def summary(self) -> dict[str, Any]:
        dynamic = (
            0
            if self.dynamic_element_count is None
            else int(round(float(self.dynamic_element_count.detach().cpu().item())))
        )
        count = self.element_count + dynamic
        square_sum = (
            0.0
            if self.square_sum is None
            else float(self.square_sum.detach().cpu().item())
        )
        _require(count >= 0 and square_sum >= 0.0, "invalid TPD8 RMS state")
        return {
            "element_count": count,
            "square_sum": square_sum,
            "rms": None if count == 0 else math.sqrt(square_sum / count),
        }


_TERM_NAMES = (
    "keep_K",
    "context_aligned_Ca",
    "saliency_v8_Sa8",
    "phase_correction_P",
    "residual_R",
    "modulation_V",
    "headroom_minus_one",
    "target_residual_R",
    "background_residual_R",
)


@dataclass(slots=True)
class _BlockStatistics:
    call_count: int = 0
    output_shapes: set[tuple[int, int, int]] = field(default_factory=set)
    terms: dict[str, _RMSAccumulator] = field(
        default_factory=lambda: {name: _RMSAccumulator() for name in _TERM_NAMES}
    )


class FullMPRSStatisticsRecorder:
    """Aggregate seven-block full-model MPRS statistics without feature caches."""

    def __init__(
        self,
        blocks: Sequence[tuple[str, TPDCleanV8MPRSDCHBlock]],
    ) -> None:
        self.blocks = tuple(blocks)
        self.records = {path: _BlockStatistics() for path, _ in self.blocks}
        self.current_target: torch.Tensor | None = None
        self.current_valid: torch.Tensor | None = None
        self._batch_call_counts: dict[str, int] | None = None
        self.batch_count = 0
        self.aborted_batch_count = 0
        self.hooks_restored = False

    def begin_batch(
        self,
        target: torch.Tensor,
        original_size: tuple[int, int],
    ) -> None:
        _require(self.current_target is None, "TPD8 batch capture is already active")
        _require(
            isinstance(target, torch.Tensor)
            and target.ndim == 4
            and target.shape[1] == 1,
            "TPD8 capture target must be Bx1xHxW",
        )
        _require(
            len(original_size) == 2
            and 0 < int(original_size[0]) <= int(target.shape[-2])
            and 0 < int(original_size[1]) <= int(target.shape[-1]),
            "TPD8 original size is outside the padded target",
        )
        self.current_target = target.detach()
        valid = torch.zeros_like(target, dtype=torch.bool)
        valid[..., : int(original_size[0]), : int(original_size[1])] = True
        self.current_valid = valid
        self._batch_call_counts = {
            path: record.call_count for path, record in self.records.items()
        }

    def end_batch(self) -> None:
        _require(self.current_target is not None, "TPD8 batch capture is inactive")
        _require(self._batch_call_counts is not None, "TPD8 batch counters are absent")
        for path, before in self._batch_call_counts.items():
            _require(
                self.records[path].call_count == before + 1,
                f"TPD8 block did not execute exactly once in batch: {path}",
            )
        self.current_target = None
        self.current_valid = None
        self._batch_call_counts = None
        self.batch_count += 1

    def abort_batch(self) -> None:
        """Clear transient target state when a model forward raises."""

        if self.current_target is None:
            return
        self.current_target = None
        self.current_valid = None
        self._batch_call_counts = None
        self.aborted_batch_count += 1

    def record(
        self,
        path: str,
        *,
        keep: torch.Tensor,
        context_aligned: torch.Tensor,
        saliency_v8: torch.Tensor,
        phase_correction: torch.Tensor,
        residual: torch.Tensor,
        modulation: torch.Tensor,
        headroom: torch.Tensor,
    ) -> None:
        _require(
            self.current_target is not None and self.current_valid is not None,
            "TPD8 target/valid mask was not set",
        )
        record = self.records[path]
        record.call_count += 1
        record.output_shapes.add(
            (int(keep.shape[1]), int(keep.shape[2]), int(keep.shape[3]))
        )
        tensors = {
            "keep_K": keep,
            "context_aligned_Ca": context_aligned,
            "saliency_v8_Sa8": saliency_v8,
            "phase_correction_P": phase_correction,
            "residual_R": residual,
            "modulation_V": modulation,
            "headroom_minus_one": headroom - 1.0,
        }
        _require(
            bool(
                torch.stack(
                    [torch.isfinite(tensor).all() for tensor in tensors.values()]
                ).all()
            ),
            f"non-finite TPD8 mechanism statistic: {path}",
        )
        for name, tensor in tensors.items():
            record.terms[name].append(tensor)

        output_size = tuple(residual.shape[-2:])
        target = F.adaptive_max_pool2d(
            self.current_target.float() * self.current_valid.float(),
            output_size=output_size,
        ) > 0.5
        # A block location is valid when its adaptive bin contains at least one
        # original pixel.  Only bins made entirely from bottom/right padding
        # are excluded; real boundary targets remain represented.
        pooled_valid = F.adaptive_max_pool2d(
            self.current_valid.float(), output_size=output_size
        ) > 0.5
        target_region = torch.logical_and(pooled_valid, target)
        background_region = torch.logical_and(pooled_valid, torch.logical_not(target))
        record.terms["target_residual_R"].append_spatial_mask(
            residual, target_region
        )
        record.terms["background_residual_R"].append_spatial_mask(
            residual, background_region
        )

    def summary(self) -> dict[str, Any]:
        _require(self.current_target is None, "TPD8 summary requested mid-batch")
        _require(self.batch_count > 0, "no TPD8 full-mode batch was captured")
        rows: list[dict[str, Any]] = []
        for index, (path, block) in enumerate(self.blocks):
            record = self.records[path]
            _require(
                record.call_count == self.batch_count,
                f"TPD8 block capture count differs: {path}",
            )
            term_summary = {
                name: record.terms[name].summary() for name in _TERM_NAMES
            }
            target_rms = term_summary["target_residual_R"]["rms"]
            background_rms = term_summary["background_residual_R"]["rms"]
            rows.append(
                {
                    "block_index_zero_based": index,
                    "block_path": path,
                    "embedding": "embeddings_1" if index < 4 else "embeddings_2",
                    "embedding_block_index_zero_based": index if index < 4 else index - 4,
                    "channels": block.channels,
                    "activation": block.activation.__class__.__name__,
                    "forward_call_count": record.call_count,
                    "observed_output_shapes_chw": [
                        list(shape) for shape in sorted(record.output_shapes)
                    ],
                    "rms_statistics": term_summary,
                    "target_minus_background_residual_rms": (
                        None
                        if target_rms is None or background_rms is None
                        else float(target_rms) - float(background_rms)
                    ),
                }
            )
        return {
            "schema": MECHANISM_SCHEMA,
            "production_output_policy": "return_original_forward_output_unchanged",
            "diagnostic_execution": (
                "in_production_aligned_mprs_terms_capture_plus_no_grad_headroom"
            ),
            "branch_projection_recomputed_for_statistics": False,
            "feature_cache_written": False,
            "batch_count": self.batch_count,
            "aborted_batch_count": self.aborted_batch_count,
            "block_count": len(rows),
            "block_order": list(BLOCK_PATHS),
            "temporary_forward_wrappers_restored": self.hooks_restored,
            "target_projection": "adaptive_max_pool2d_binary_presence",
            "valid_projection": "adaptive_max_pool2d_any_original_support",
            "background_region": "pooled_valid_and_not_pooled_target",
            "blocks": rows,
        }


@contextlib.contextmanager
def capture_full_mprs_statistics(
    model: nn.Module,
) -> Iterator[FullMPRSStatisticsRecorder]:
    """Capture production MPRS terms while returning every object unchanged."""

    _require(not model.training, "TPD8 mechanism capture requires model.eval()")
    blocks = _resolve_tpd8_blocks(model)
    recorder = FullMPRSStatisticsRecorder(blocks)
    lookup_state: list[tuple[TPDCleanV8MPRSDCHBlock, str, bool, Any]] = []
    try:
        for path, block in blocks:
            had_instance_aligned = "aligned_mprs_terms" in block.__dict__
            prior_instance_aligned = block.__dict__.get("aligned_mprs_terms")
            original_aligned = block.aligned_mprs_terms

            def wrapped_aligned_mprs_terms(
                self: TPDCleanV8MPRSDCHBlock,
                x: torch.Tensor,
                *,
                block_path: str = path,
                formal_aligned: Any = original_aligned,
            ) -> tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]:
                terms = formal_aligned(x)
                keep, context_aligned, _, phase_correction, saliency_v8 = terms
                with torch.no_grad():
                    scale, modulation, headroom = self.headroom(context_aligned)
                    residual = (
                        saliency_v8 * (scale * headroom)
                    ).to(dtype=keep.dtype)
                    recorder.record(
                        block_path,
                        keep=keep,
                        context_aligned=context_aligned,
                        saliency_v8=saliency_v8,
                        phase_correction=phase_correction,
                        residual=residual,
                        modulation=modulation,
                        headroom=headroom,
                    )
                return terms

            lookup_state.append(
                (
                    block,
                    "aligned_mprs_terms",
                    had_instance_aligned,
                    prior_instance_aligned,
                )
            )
            block.aligned_mprs_terms = types.MethodType(  # type: ignore[method-assign]
                wrapped_aligned_mprs_terms, block
            )

            had_instance_forward = "forward" in block.__dict__
            prior_instance_forward = block.__dict__.get("forward")
            original_forward = block.forward

            def wrapped_forward(
                self: TPDCleanV8MPRSDCHBlock,
                x: torch.Tensor,
                *,
                block_path: str = path,
                formal_forward: Any = original_forward,
            ) -> torch.Tensor:
                del self, block_path
                return formal_forward(x)

            lookup_state.append(
                (block, "forward", had_instance_forward, prior_instance_forward)
            )
            block.forward = types.MethodType(  # type: ignore[method-assign]
                wrapped_forward, block
            )
        yield recorder
    finally:
        recorder.abort_batch()
        for block, method_name, had_instance_value, prior_instance_value in reversed(
            lookup_state
        ):
            if had_instance_value:
                setattr(block, method_name, prior_instance_value)
            elif method_name in block.__dict__:
                delattr(block, method_name)
        recorder.hooks_restored = True


@torch.inference_mode()
def _collect_one_mode(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    public_mode: str,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[float],
    list[str],
    dict[str, Any],
    dict[str, Any] | None,
]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []
    identifiers: list[str] = []
    criterion = nn.BCELoss(reduction="mean")

    with temporary_saliency_scale_knockout(model, public_mode) as knockout:
        diagnostic_context: Any = (
            capture_full_mprs_statistics(model)
            if public_mode == "full"
            else contextlib.nullcontext(None)
        )
        with diagnostic_context as recorder:
            for images, masks, sizes, sample_ids in loader:
                _require(
                    int(images.shape[0]) == int(masks.shape[0]) == 1,
                    "TPD8 diagnostic requires batch_size=1",
                )
                images = images.to(device, non_blocking=device.type == "cuda")
                masks = masks.to(device, non_blocking=device.type == "cuda")
                height, width = core._extract_hw(sizes)
                if recorder is not None:
                    recorder.begin_batch(masks, (height, width))
                try:
                    prediction = core._final_prediction(model(images))[
                        :, :, :height, :width
                    ]
                except BaseException:
                    if recorder is not None:
                        recorder.abort_batch()
                    raise
                else:
                    if recorder is not None:
                        recorder.end_batch()
                target = masks[:, :, :height, :width]
                _require(prediction.shape == target.shape, "prediction/target differs")
                _require(
                    bool(torch.isfinite(prediction).all()),
                    "prediction contains non-finite values",
                )
                loss = criterion(prediction.float(), target.float())
                _require(math.isfinite(float(loss.item())), "loss is non-finite")
                probabilities.append(
                    prediction[0, 0]
                    .float()
                    .cpu()
                    .contiguous()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                targets.append(target[0, 0].float().cpu().contiguous().numpy())
                losses.append(float(loss.item()))
                _require(
                    isinstance(sample_ids, (tuple, list)) and len(sample_ids) == 1,
                    "TPD8 diagnostic requires one sample ID per batch",
                )
                identifiers.append(str(sample_ids[0]))
        mechanism = None if recorder is None else recorder.summary()

    _require(
        len(probabilities) == len(loader.dataset),
        f"{public_mode} inference count differs",
    )
    return probabilities, targets, losses, identifiers, dict(knockout), mechanism


def _validate_and_annotate_two_point_sweep(
    evaluated: Mapping[str, Any],
) -> None:
    """Lock the descriptive sweep to 0.5 plus the legal empty 1.0 endpoint."""

    descriptive = evaluated.get("descriptive_pd_fa")
    _require(isinstance(descriptive, Mapping), "descriptive Pd-Fa result is missing")
    points = descriptive.get("points")
    _require(isinstance(points, list) and len(points) == 2, "sweep must have two points")
    thresholds = [float(point.get("threshold")) for point in points]
    _require(thresholds == list(SWEEP_THRESHOLDS), "sweep points differ from 0.5/1.0")
    empty = points[1]
    empty["selected_point_is_empty"] = core._point_is_empty(empty)
    _require(
        empty["selected_point_is_empty"] is True,
        "threshold=1.0 point is not empty",
    )
    _require(float(empty.get("pd")) == 0.0, "threshold=1.0 Pd must be zero")
    _require(float(empty.get("fa")) == 0.0, "threshold=1.0 Fa must be zero")


def analyze_loaded_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    expected_identifiers: Sequence[str],
) -> dict[str, Any]:
    """Evaluate the frozen nine-mode matrix on an already loaded model."""

    model.eval()
    _require(not model.training, "TPD8 diagnostic model must be in eval mode")
    _resolve_tpd8_blocks(model)
    state_before = stage2_audit.module_state_sha256(model)
    scale_before = saliency_scale_state_sha256(model)
    scale_values = saliency_scale_records(model)
    full_probabilities: list[np.ndarray] | None = None
    full_targets: list[np.ndarray] | None = None
    full_identifiers: list[str] | None = None
    modes: dict[str, Any] = {}

    for public_mode in PUBLIC_MODES:
        probabilities, targets, losses, identifiers, knockout, mechanism = (
            _collect_one_mode(model, loader, device, public_mode)
        )
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
            _require(identifiers == full_identifiers, "mode identifiers differ")
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
            sweep_thresholds=SWEEP_THRESHOLDS,
        )
        _validate_and_annotate_two_point_sweep(evaluated)
        fixed = dict(evaluated["fixed_threshold_0_5"])
        fixed["false_positive_pixels"] = (
            qfg_skeleton._all_background_false_positive_pixels(
                probabilities, targets, threshold=FIXED_THRESHOLD
            )
        )
        difference = probability_difference(full_probabilities, probabilities)
        scale_after_mode = saliency_scale_state_sha256(model)
        state_after_mode = stage2_audit.module_state_sha256(model)
        _require(scale_after_mode == scale_before, "TPD8 scale state was not restored")
        _require(state_after_mode == state_before, "TPD8 model state was not restored")
        if public_mode == "full":
            _require(mechanism is not None, "full TPD8 mechanism statistics missing")
        else:
            _require(mechanism is None, "off mode unexpectedly collected mechanism stats")
        modes[public_mode] = {
            **normalize_public_mode(public_mode),
            "saliency_scale_knockout": knockout,
            "fixed_threshold_0_5": fixed,
            "descriptive_pd_fa": evaluated["descriptive_pd_fa"],
            "threshold_roles": evaluated["threshold_roles"],
            "sweep_thresholds": list(SWEEP_THRESHOLDS),
            "probability_difference_to_full": difference,
            "full_mprs_statistics": mechanism,
            "restoration_audit": {
                "saliency_scale_sha256_expected": scale_before,
                "saliency_scale_sha256_after_mode": scale_after_mode,
                "saliency_scale_unchanged": scale_after_mode == scale_before,
                "model_state_sha256_expected": state_before,
                "model_state_sha256_after_mode": state_after_mode,
                "model_state_unchanged": state_after_mode == state_before,
            },
        }
        if public_mode != "full":
            del probabilities, targets, losses

    _require(full_probabilities is not None, "full probabilities are missing")
    state_after = stage2_audit.module_state_sha256(model)
    scale_after = saliency_scale_state_sha256(model)
    _require(state_after == state_before, "TPD8 diagnostic changed model state")
    _require(scale_after == scale_before, "TPD8 diagnostic changed scale state")
    return {
        "modes": modes,
        "source_saliency_scale_records": scale_values,
        "restoration_audit": {
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "model_state_unchanged": state_after == state_before,
            "saliency_scale_sha256_before": scale_before,
            "saliency_scale_sha256_after": scale_after,
            "saliency_scale_unchanged": scale_after == scale_before,
            "temporary_forward_wrappers_restored": True,
        },
        "probability_arrays_persisted": False,
    }


def _frozen_source_sha256() -> dict[str, str]:
    sources = {
        "analysis/analyze_three_dataset_tpd8_block_residual_knockout_v1.py": Path(
            __file__
        ),
        "analysis/analyze_three_dataset_qfg_level_knockout_v1.py": Path(
            qfg_skeleton.__file__
        ),
        "analysis/analyze_ner_stage2_mask_knockout_v1.py": Path(
            stage2_audit.__file__
        ),
        "model/tpd_clean_v8_mprs_dch.py": Path(tpd8_source.__file__),
        "experiments/evaluate_three_dataset_v2.py": Path(core.__file__),
        "experiments/evaluate_three_dataset_tss_off_seed42_v1.py": Path(
            adapter.__file__
        ),
    }
    return {name: file_sha256(path.resolve(strict=True)) for name, path in sources.items()}


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
    frozen_sources = _frozen_source_sha256()
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
    _validate_formal_channels(model)
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
    _require(
        _frozen_source_sha256() == frozen_sources,
        "runtime source changed after source-hash freeze",
    )

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
        "sweep_thresholds": list(SWEEP_THRESHOLDS),
        "mode_order": list(PUBLIC_MODES),
        "block_order": list(BLOCK_PATHS),
        "optional_combination_mode": None,
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
            "descriptive_sweep_thresholds": list(SWEEP_THRESHOLDS),
            "sweep_reselects_checkpoint": False,
        },
        "intervention_contract": {
            "parameter": "selected TPD8 block saliency_scale vector",
            "active_value": "exact_zero",
            "weights_saved_valuewise_and_restored": True,
            "phase_compress_modified": False,
            "other_model_state_modified": False,
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
    _require(payload.get("schema") == SCHEMA, "TPD8 analyzer schema differs")
    _require(payload.get("status") == "complete", "TPD8 analyzer is incomplete")
    _require(payload.get("dataset") in data_protocol.DATASETS, "dataset differs")
    _require(payload.get("checkpoint_role") == CHECKPOINT_ROLE, "role differs")
    _require(payload.get("seed") == SEED, "seed differs")
    _require(payload.get("test_selected") is True, "test_selected differs")
    _require(
        payload.get("evaluation_protocol") == EVALUATION_PROTOCOL,
        "evaluation protocol differs",
    )
    _require(payload.get("sweep_thresholds") == list(SWEEP_THRESHOLDS), "sweep differs")
    _require(
        payload.get("reference_replay_audit", {}).get("passed") is True,
        "full reference replay did not pass",
    )
    modes = payload.get("modes")
    _require(isinstance(modes, Mapping), "TPD8 analyzer modes are missing")
    _require(tuple(modes) == PUBLIC_MODES, "TPD8 analyzer mode order differs")
    _require(payload.get("mode_order") in (None, list(PUBLIC_MODES)), "mode declaration differs")
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
        _require(mode.get("public_mode") == public_mode, "public mode differs")
        _require(
            mode.get("knockout_block_paths")
            == list(MODE_TO_BLOCK_PATHS[public_mode]),
            f"{public_mode} block selection differs",
        )
        fixed = mode.get("fixed_threshold_0_5")
        _require(isinstance(fixed, Mapping), f"{public_mode} fixed point missing")
        _require(required_fixed <= set(fixed), f"{public_mode} fixed fields missing")
        _require(fixed.get("threshold") == FIXED_THRESHOLD, "fixed threshold differs")
        descriptive = mode.get("descriptive_pd_fa")
        _require(isinstance(descriptive, Mapping), "descriptive sweep missing")
        points = descriptive.get("points")
        _require(
            isinstance(points, list) and len(points) == 2,
            f"{public_mode} sweep must contain exactly two points",
        )
        _require(
            [float(point.get("threshold")) for point in points]
            == list(SWEEP_THRESHOLDS),
            f"{public_mode} sweep thresholds differ",
        )
        empty = points[1]
        _require(
            empty.get("selected_point_is_empty") is True
            and float(empty.get("pd")) == 0.0
            and float(empty.get("fa")) == 0.0,
            f"{public_mode} threshold=1.0 empty endpoint differs",
        )
        difference = mode.get("probability_difference_to_full")
        _require(
            isinstance(difference, Mapping)
            and int(difference.get("element_count", 0)) > 0,
            f"{public_mode} probability difference missing",
        )
        _require(
            int(difference["element_count"]) == int(fixed["valid_pixel_count"]),
            f"{public_mode} difference count differs from valid pixels",
        )
        absolute_sum = float(difference.get("absolute_difference_sum"))
        _require(absolute_sum >= 0.0, f"{public_mode} difference sum is negative")
        _require(
            float(difference.get("mean_abs"))
            == absolute_sum / int(difference["element_count"]),
            f"{public_mode} difference mean identity differs",
        )
        if public_mode == "full":
            _require(
                float(difference.get("max_abs")) == 0.0
                and absolute_sum == 0.0
                and float(difference.get("mean_abs")) == 0.0,
                "full probability self-difference is not exact zero",
            )
        knockout = mode.get("saliency_scale_knockout", {})
        _require(
            knockout.get("saliency_scale_restored_exactly") is True
            and knockout.get("model_state_restored_exactly") is True,
            f"{public_mode} knockout restoration failed",
        )
        selected_vectors = knockout.get("selected_vectors")
        _require(
            isinstance(selected_vectors, list)
            and len(selected_vectors) == len(MODE_TO_BLOCK_PATHS[public_mode]),
            f"{public_mode} selected-vector audit differs",
        )
        _require(
            all(int(record.get("active_nonzero_count", -1)) == 0 for record in selected_vectors)
            and int(knockout.get("selected_active_nonzero_count", -1)) == 0,
            f"{public_mode} selected scale is not exactly zero",
        )
        for sha_field in (
            "source_saliency_scale_sha256",
            "active_saliency_scale_sha256",
            "restored_saliency_scale_sha256",
            "source_model_state_sha256",
            "active_model_state_sha256",
            "restored_model_state_sha256",
        ):
            sha = knockout.get(sha_field)
            _require(
                isinstance(sha, str) and len(sha) == 64,
                f"{public_mode} {sha_field} is invalid",
            )
        source_scale_sha = knockout["source_saliency_scale_sha256"]
        active_scale_sha = knockout["active_saliency_scale_sha256"]
        _require(
            knockout["restored_saliency_scale_sha256"] == source_scale_sha,
            f"{public_mode} restored scale SHA differs",
        )
        if public_mode == "full":
            _require(active_scale_sha == source_scale_sha, "full active scale SHA differs")
        elif int(knockout.get("selected_source_nonzero_count", 0)) > 0:
            _require(
                active_scale_sha != source_scale_sha,
                f"{public_mode} active scale SHA did not change",
            )
        restoration = mode.get("restoration_audit", {})
        _require(
            restoration.get("saliency_scale_unchanged") is True
            and restoration.get("model_state_unchanged") is True,
            f"{public_mode} post-mode restoration failed",
        )
        mechanism = mode.get("full_mprs_statistics")
        if public_mode == "full":
            _require(isinstance(mechanism, Mapping), "full mechanism stats missing")
            _require(mechanism.get("block_order") == list(BLOCK_PATHS), "block order differs")
            _require(mechanism.get("block_count") == 7, "block count differs")
            _require(
                mechanism.get("temporary_forward_wrappers_restored") is True,
                "full wrappers were not restored",
            )
            blocks = mechanism.get("blocks")
            _require(isinstance(blocks, list) and len(blocks) == 7, "stats rows differ")
            _require(
                all(
                    isinstance(row.get("rms_statistics"), Mapping)
                    and set(row["rms_statistics"]) == set(_TERM_NAMES)
                    for row in blocks
                ),
                "full block RMS fields differ",
            )
            for row in blocks:
                rms = row["rms_statistics"]
                _require(
                    all(
                        int(rms[name].get("element_count", 0)) > 0
                        and rms[name].get("rms") is not None
                        for name in _TERM_NAMES
                    ),
                    "full block RMS statistic is empty",
                )
                expected_contrast = float(rms["target_residual_R"]["rms"]) - float(
                    rms["background_residual_R"]["rms"]
                )
                _require(
                    float(row.get("target_minus_background_residual_rms"))
                    == expected_contrast,
                    "target/background residual RMS contrast differs",
                )
        else:
            _require(mechanism is None, "off mode mechanism stats must be null")
    restoration = payload.get("restoration_audit", {})
    _require(
        restoration.get("model_state_unchanged") is True
        and restoration.get("saliency_scale_unchanged") is True
        and restoration.get("temporary_forward_wrappers_restored") is True,
        "TPD8 analyzer restoration audit failed",
    )
    _require(
        payload.get("probability_cache_written") is False
        and payload.get("derived_checkpoint_written") is False,
        "TPD8 analyzer wrote a forbidden artifact",
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
    "BLOCK_PATHS",
    "CHECKPOINT_ROLE",
    "EVALUATION_PROTOCOL",
    "FIXED_THRESHOLD",
    "MODE_TO_BLOCK_PATHS",
    "PUBLIC_MODES",
    "SCHEMA",
    "SWEEP_THRESHOLDS",
    "FullMPRSStatisticsRecorder",
    "analyze_loaded_model",
    "analyze_run",
    "atomic_create_json",
    "capture_full_mprs_statistics",
    "main",
    "normalize_public_mode",
    "parse_args",
    "probability_difference",
    "reference_replay_audit",
    "saliency_scale_records",
    "saliency_scale_state_sha256",
    "temporary_saliency_scale_knockout",
    "validate_output_payload",
]


if __name__ == "__main__":
    main()
