#!/usr/bin/env python3
"""Seed-42 Final-model retraining with a bounded TSS loss contribution.

This is a deliberately small revision of
``train_four_dataset_original_final_seed42_exact_v1.py``.  The model and
inference graph are unchanged.  Only the training-time TSS coefficient is
changed from a fixed 0.005 to

    min(0.005, 0.10 * stopgrad(L_seg) / max(stopgrad(L_tss), eps)).

The historical V1 entry point and all V1 results remain untouched.  V2 writes
to its own result root and reuses the frozen V1 data manifests.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import train_four_dataset_original_final_seed42_exact_v1 as base
from experiments.tpd_training_loss import (
    TPDTrainingLoss,
    compute_tpd_training_loss as _compute_tpd_training_loss,
)


SCHEMA = "sctransnet_four_dataset_seed42_tss_cap_v2"
REQUESTED_TSS_WEIGHT = 0.005
TSS_RATIO_CAP = 0.10
DEFAULT_RESULTS_ROOT = base.REPO_ROOT / "results" / "four_dataset_seed42_tss_cap_v2"
PROTOCOL_DOCUMENT = base.REPO_ROOT / "SCTransNet_TSS动态约束性能优化V2执行记录.md"
WRAPPER_SOURCE = Path(__file__).resolve()
LOSS_SOURCE = base.REPO_ROOT / "experiments" / "tpd_training_loss.py"


_original_validate_args = base.validate_args
_original_protocol_payload = base._protocol_payload
_original_latest_checkpoint_payload = base._latest_checkpoint_payload


class _EpochTSSAudit:
    """Accumulate sample-weighted diagnostics without changing optimization."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.samples = 0
        self.effective_weight_sum = 0.0
        self.weighted_survival_sum = 0.0
        self.segmentation_sum = 0.0
        self.cap_active_samples = 0

    def add(self, losses: TPDTrainingLoss, samples: int) -> None:
        effective = float(losses.effective_survival_weight.detach().item())
        weighted = float(losses.weighted_survival.detach().item())
        segmentation = float(losses.segmentation.detach().item())
        if not all(math.isfinite(value) for value in (effective, weighted, segmentation)):
            raise FloatingPointError("non-finite TSS audit value")
        self.samples += samples
        self.effective_weight_sum += effective * samples
        self.weighted_survival_sum += weighted * samples
        self.segmentation_sum += segmentation * samples
        # Ignore the normal FP32 representation difference between Python's
        # 0.005 and the stored scalar tensor.  Count only a material clamp.
        if effective < REQUESTED_TSS_WEIGHT * (1.0 - 1e-6):
            self.cap_active_samples += samples

    def payload(self) -> dict[str, float]:
        if self.samples <= 0:
            raise RuntimeError("TSS audit has no training samples")
        return {
            "train_tss_requested_weight": REQUESTED_TSS_WEIGHT,
            "train_tss_ratio_cap": TSS_RATIO_CAP,
            "train_tss_effective_weight_mean": (
                self.effective_weight_sum / self.samples
            ),
            "train_tss_weighted_loss": self.weighted_survival_sum / self.samples,
            "train_tss_weighted_to_segmentation_ratio": (
                self.weighted_survival_sum / max(self.segmentation_sum, 1e-12)
            ),
            "train_tss_cap_active_sample_fraction": (
                self.cap_active_samples / self.samples
            ),
        }


_audit = _EpochTSSAudit()


def _compute_loss_v2(
    output: Any,
    segmentation_target: torch.Tensor,
    segmentation_criterion: nn.Module,
    *,
    survival_weight: float = 0.0,
    survival_pos_weight: float | torch.Tensor = 1.0,
) -> TPDTrainingLoss:
    if float(survival_weight) != REQUESTED_TSS_WEIGHT:
        raise ValueError(
            "V2 Final training requires requested TSS weight "
            f"{REQUESTED_TSS_WEIGHT}, got {survival_weight}"
        )
    losses = _compute_tpd_training_loss(
        output,
        segmentation_target,
        segmentation_criterion,
        survival_weight=survival_weight,
        survival_pos_weight=survival_pos_weight,
        survival_ratio_cap=TSS_RATIO_CAP,
    )
    _audit.add(losses, int(segmentation_target.shape[0]))
    return losses


def _validate_args_v2(args: Any) -> None:
    if args.method != "final":
        raise ValueError("the TSS-cap V2 runner trains only the Final model")
    if args.smoke:
        _original_validate_args(args)
        return
    if args.physical_gpu_index not in base.GPU_UUIDS:
        raise ValueError("formal V2 training requires physical GPU 2 or 3")
    expected_uuid = base.GPU_UUIDS[args.physical_gpu_index]
    if args.expected_gpu_uuid != expected_uuid:
        raise ValueError(
            "expected GPU UUID differs: "
            f"{args.expected_gpu_uuid!r} != {expected_uuid!r}"
        )
    # Reuse every formal V1 invariant while permitting either approved GPU for
    # the Final-only V2 jobs.  This process-local dictionary change cannot
    # affect another training process.
    previous = base.METHOD_GPU["final"]
    base.METHOD_GPU["final"] = args.physical_gpu_index
    try:
        _original_validate_args(args)
    finally:
        base.METHOD_GPU["final"] = previous


def _protocol_payload_v2(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = _original_protocol_payload(*args, **kwargs)
    training = payload["training"]
    training.update(
        {
            "tss_weight": REQUESTED_TSS_WEIGHT,
            "tss_weight_policy": "per_minibatch_loss_ratio_cap",
            "tss_requested_weight": REQUESTED_TSS_WEIGHT,
            "tss_ratio_cap": TSS_RATIO_CAP,
            "tss_effective_weight_formula": (
                "min(0.005, 0.10*stopgrad(L_seg)/"
                "max(stopgrad(L_tss),float32_eps))"
            ),
            "model_graph_changed": False,
            "inference_graph_changed": False,
        }
    )
    runtime_sources = payload["runtime_sources"]
    runtime_sources["base_runner"] = runtime_sources.pop("runner")
    runtime_sources["v2_runner"] = base.file_sha256(WRAPPER_SOURCE)
    runtime_sources["training_loss"] = base.file_sha256(LOSS_SOURCE)
    payload["revision_scope"] = "training_time_tss_weight_only"
    payload["comparison_baseline"] = "four_dataset_seed42_v1"
    return payload


def _latest_checkpoint_payload_v2(*args: Any, **kwargs: Any) -> dict[str, Any]:
    event = kwargs.get("event")
    if not isinstance(event, dict):
        raise TypeError("latest-checkpoint event must be a mutable dictionary")
    event.update(_audit.payload())
    payload = _original_latest_checkpoint_payload(*args, **kwargs)
    _audit.reset()
    return payload


def parse_args(argv: list[str] | None = None) -> Any:
    args = base.parse_args(argv)
    if args.results_root == base.DEFAULT_RESULTS_ROOT:
        args.results_root = DEFAULT_RESULTS_ROOT
    return args


def _install_v2_contract() -> None:
    base.SCHEMA = SCHEMA
    base.PROTOCOL_DOCUMENT = PROTOCOL_DOCUMENT
    base.validate_args = _validate_args_v2
    base.compute_tpd_training_loss = _compute_loss_v2
    base._protocol_payload = _protocol_payload_v2
    base._latest_checkpoint_payload = _latest_checkpoint_payload_v2


def main(argv: list[str] | None = None) -> None:
    _install_v2_contract()
    args = parse_args(argv)
    base.run(args)


if __name__ == "__main__":
    main()
