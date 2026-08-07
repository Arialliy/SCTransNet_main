"""Deterministic optimizer, epoch selection, and checkpoint core for V4."""

from __future__ import annotations

from fractions import Fraction
import math
import os
import random
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from experiments.pbdr_v4_state_contract import (
    PBDR_PREFIX,
    Stage,
    audit_training_modes,
    state_semantic_sha256,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v4 import (
    PBDR_V4_CANDIDATE_CHECKPOINT_SCHEMA,
)


TRAINING_SEED = 42
STAGE_EPOCHS: Mapping[Stage, int] = {"stage1": 150, "stage2": 50}
EVAL_EVERY = 5
TRAIN_BATCH_SIZE = 16
WEIGHT_DECAY = 1.0e-4
STAGE1_LR = 1.0e-4
STAGE2_ROUTER_LR = 1.0e-4
STAGE2_OUTC_LR = 2.0e-6
STAGE2_UP_DECODER1_LR = 1.0e-6
STAGE2_L2SP_WEIGHT = 1.0e-4


class PBDRV4TrainingCoreError(RuntimeError):
    """A deterministic recipe, metric, or checkpoint contract is invalid."""


def configure_determinism(seed: int = TRAINING_SEED) -> dict[str, object]:
    if type(seed) is not int or seed != TRAINING_SEED:
        raise PBDRV4TrainingCoreError("formal PBDR-V4 seed must be 42")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {
        "seed": seed,
        "precision": "fp32",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_rng_state(value: Mapping[str, Any]) -> None:
    if set(value) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise PBDRV4TrainingCoreError("RNG state keys differ")
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available():
        cuda_state = value["torch_cuda"]
        if not isinstance(cuda_state, list) or len(cuda_state) != torch.cuda.device_count():
            raise PBDRV4TrainingCoreError("CUDA RNG state count differs")
        torch.cuda.set_rng_state_all(cuda_state)


def _integer(metrics: Mapping[str, object], name: str, *, positive: bool = False) -> int:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise PBDRV4TrainingCoreError(f"metric {name} must be an integer")
    ready = int(value)
    if ready < 0 or (positive and ready <= 0):
        raise PBDRV4TrainingCoreError(f"metric {name} has an invalid count")
    return ready


def _float(metrics: Mapping[str, object], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool):
        raise PBDRV4TrainingCoreError(f"metric {name} must be real")
    try:
        ready = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise PBDRV4TrainingCoreError(f"metric {name} must be real") from error
    if not math.isfinite(ready):
        raise PBDRV4TrainingCoreError(f"metric {name} is non-finite")
    return ready


def checkpoint_epoch_key(
    role: str,
    metrics: Mapping[str, object],
    epoch: int,
) -> tuple[object, ...]:
    """Complete fixed-0.5 role key followed only by earlier-epoch tie-break."""

    if role not in ("best_miou", "best_pd"):
        raise PBDRV4TrainingCoreError("unsupported role")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise PBDRV4TrainingCoreError("epoch must be positive")
    intersection = _integer(metrics, "intersection_pixels")
    union = _integer(metrics, "union_pixels", positive=True)
    matched = _integer(metrics, "matched_target_count")
    targets = _integer(metrics, "target_count", positive=True)
    unmatched_pixels = _integer(metrics, "unmatched_component_pixels")
    valid_pixels = _integer(metrics, "valid_pixel_count", positive=True)
    matched_tiny = _integer(metrics, "matched_tiny_target_count")
    tiny = _integer(metrics, "tiny_target_count")
    if intersection > union or matched > targets or matched_tiny > tiny or matched_tiny > matched:
        raise PBDRV4TrainingCoreError("metric sufficient statistics are inconsistent")
    miou = Fraction(intersection, union)
    pd = Fraction(matched, targets)
    fa = Fraction(unmatched_pixels, valid_pixels)
    tiny_pd = Fraction(matched_tiny, tiny) if tiny else Fraction(0, 1)
    niou = _float(metrics, "niou")
    loss = _float(metrics, "test_loss")
    if not 0.0 <= niou <= 1.0 or loss < 0.0:
        raise PBDRV4TrainingCoreError("nIoU/loss is outside its valid range")
    tail = (niou, tiny_pd, -loss, -epoch)
    if role == "best_miou":
        return (miou, pd, -fa, *tail)
    return (pd, -fa, tiny_pd, miou, niou, -loss, -epoch)


def _named_parameters_with_prefix(
    model: nn.Module,
    prefix: str,
) -> list[nn.Parameter]:
    return [parameter for name, parameter in model.named_parameters() if name.startswith(prefix)]


def build_optimizer(model: nn.Module, stage: Stage) -> torch.optim.AdamW:
    """Build the exact pre-registered parameter groups for one stage."""

    audit_training_modes(model, stage)
    if stage == "stage1":
        groups = [
            {
                "name": "pbdr_v4",
                "params": _named_parameters_with_prefix(model, PBDR_PREFIX),
                "lr": STAGE1_LR,
            }
        ]
    elif stage == "stage2":
        groups = [
            {
                "name": "pbdr_v4",
                "params": _named_parameters_with_prefix(model, PBDR_PREFIX),
                "lr": STAGE2_ROUTER_LR,
            },
            {
                "name": "outc",
                "params": _named_parameters_with_prefix(model, "outc."),
                "lr": STAGE2_OUTC_LR,
            },
            {
                "name": "up_decoder1",
                "params": _named_parameters_with_prefix(model, "up_decoder1."),
                "lr": STAGE2_UP_DECODER1_LR,
            },
        ]
    else:
        raise PBDRV4TrainingCoreError(f"unsupported stage: {stage!r}")
    if any(not group["params"] for group in groups):
        raise PBDRV4TrainingCoreError("one optimizer parameter group is empty")
    flattened = [parameter for group in groups for parameter in group["params"]]
    if len({id(parameter) for parameter in flattened}) != len(flattened):
        raise PBDRV4TrainingCoreError("optimizer parameter groups overlap")
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if {id(parameter) for parameter in flattened} != expected:
        raise PBDRV4TrainingCoreError("optimizer groups differ from trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def training_recipe(stage: Stage) -> dict[str, object]:
    if stage not in STAGE_EPOCHS:
        raise PBDRV4TrainingCoreError("unsupported stage")
    groups = (
        [{"name": "pbdr_v4", "lr": STAGE1_LR}]
        if stage == "stage1"
        else [
            {"name": "pbdr_v4", "lr": STAGE2_ROUTER_LR},
            {"name": "outc", "lr": STAGE2_OUTC_LR},
            {"name": "up_decoder1", "lr": STAGE2_UP_DECODER1_LR},
        ]
    )
    return {
        "stage": stage,
        "epochs": STAGE_EPOCHS[stage],
        "eval_every": EVAL_EVERY,
        "batch_size": TRAIN_BATCH_SIZE,
        "optimizer": "AdamW",
        "parameter_groups": groups,
        "weight_decay": WEIGHT_DECAY,
        "l2sp_weight": 0.0 if stage == "stage1" else STAGE2_L2SP_WEIGHT,
        "fixed_probability_comparison": ">",
        "fixed_probability_threshold": 0.5,
        "performance_acceptance_margin": None,
    }


def build_candidate_checkpoint(
    *,
    dataset: str,
    role: str,
    stage: Stage,
    epoch: int,
    architecture_manifest: Mapping[str, object],
    state_dict: Mapping[str, torch.Tensor],
    validation_metrics: Mapping[str, object],
    selection_key: tuple[object, ...],
    parent_checkpoint_sha256: str,
    parent_state_sha256: str,
    split_projection_sha256: str,
    atlas_manifest_sha256: str,
    source_lock_sha256: str,
    initialization_checkpoint_sha256: str | None,
) -> dict[str, object]:
    if dataset not in ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"):
        raise PBDRV4TrainingCoreError("unsupported dataset")
    if role not in ("best_miou", "best_pd") or stage not in STAGE_EPOCHS:
        raise PBDRV4TrainingCoreError("role/stage differs")
    expected_key = checkpoint_epoch_key(role, validation_metrics, epoch)
    if tuple(selection_key) != expected_key:
        raise PBDRV4TrainingCoreError("selection key does not replay")
    for name, value in (
        ("parent_checkpoint_sha256", parent_checkpoint_sha256),
        ("parent_state_sha256", parent_state_sha256),
        ("split_projection_sha256", split_projection_sha256),
        ("atlas_manifest_sha256", atlas_manifest_sha256),
        ("source_lock_sha256", source_lock_sha256),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise PBDRV4TrainingCoreError(f"{name} differs")
    if (stage == "stage1") != (initialization_checkpoint_sha256 is None):
        raise PBDRV4TrainingCoreError("stage initialization checkpoint binding differs")
    state = {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}
    if not state or not all(isinstance(name, str) for name in state):
        raise PBDRV4TrainingCoreError("candidate state differs")
    payload: dict[str, object] = {
        "schema": PBDR_V4_CANDIDATE_CHECKPOINT_SCHEMA,
        "dataset": dataset,
        "role": role,
        "stage": stage,
        "epoch": epoch,
        "architecture_manifest": dict(architecture_manifest),
        "state_dict": state,
        "state_sha256": state_semantic_sha256(state),
        "validation_metrics": dict(validation_metrics),
        "selection_key": selection_key,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_state_sha256": parent_state_sha256,
        "split_projection_sha256": split_projection_sha256,
        "atlas_manifest_sha256": atlas_manifest_sha256,
        "source_lock_sha256": source_lock_sha256,
        "initialization_checkpoint_sha256": initialization_checkpoint_sha256,
        "fixed_probability_rule": "strict_greater_than_0.5",
        "performance_acceptance_margin": None,
        "official_test_accessed": False,
    }
    return payload


def validate_candidate_checkpoint(
    payload: Mapping[str, object],
    *,
    dataset: str,
    role: str,
    stage: Stage,
) -> dict[str, object]:
    if payload.get("schema") != PBDR_V4_CANDIDATE_CHECKPOINT_SCHEMA:
        raise PBDRV4TrainingCoreError("candidate schema differs")
    if (payload.get("dataset"), payload.get("role"), payload.get("stage")) != (
        dataset,
        role,
        stage,
    ):
        raise PBDRV4TrainingCoreError("candidate dataset/role/stage differs")
    if payload.get("official_test_accessed") is not False or payload.get(
        "performance_acceptance_margin"
    ) is not None:
        raise PBDRV4TrainingCoreError("candidate scope/margin differs")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise PBDRV4TrainingCoreError("candidate state differs")
    if payload.get("state_sha256") != state_semantic_sha256(state):  # type: ignore[arg-type]
        raise PBDRV4TrainingCoreError("candidate state SHA differs")
    metrics = payload.get("validation_metrics")
    epoch = payload.get("epoch")
    key = payload.get("selection_key")
    if not isinstance(metrics, Mapping) or not isinstance(key, (tuple, list)):
        raise PBDRV4TrainingCoreError("candidate selection fields differ")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or tuple(key) != checkpoint_epoch_key(role, metrics, epoch):
        raise PBDRV4TrainingCoreError("candidate selection key differs")
    return dict(payload)


__all__ = [
    "EVAL_EVERY",
    "PBDRV4TrainingCoreError",
    "STAGE2_L2SP_WEIGHT",
    "STAGE_EPOCHS",
    "TRAINING_SEED",
    "TRAIN_BATCH_SIZE",
    "build_candidate_checkpoint",
    "build_optimizer",
    "capture_rng_state",
    "checkpoint_epoch_key",
    "configure_determinism",
    "restore_rng_state",
    "training_recipe",
    "validate_candidate_checkpoint",
]
