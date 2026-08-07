#!/usr/bin/env python3
"""E1: two-scalar Current calibration control on NUAA internal data only.

The control deliberately uses the formal PBDR-V3 warm-start shell only as a
source of the immutable Current ``base_logits``.  The complete shell,
including ``pbdr_v3``, remains frozen and in evaluation mode.  Exactly two
standalone parameters are optimized::

    calibrated = base_logits / exp(log_temperature) + bias

Both scalars start at zero, giving an exact identity.  Optimization uses only
the deterministic internal-training subset of the official NUAA train index;
epoch and threshold selection use only the paired internal-validation subset.
The official test index is never opened by this program.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any, Mapping, Sequence, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_three_dataset_v2 as metric_runner  # noqa: E402
from experiments import pbdr_v3_non_regression_gate as gate_core  # noqa: E402
from experiments import three_dataset_pbdr_v3_models_seed42_v1 as v3_registry  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import (  # noqa: E402
    train_four_dataset_original_final_seed42_exact_v1 as determinism_core,
)


SCHEMA = "sctransnet_nuaa_current_scalar_calibration_control_v1/v1"
RESUME_SCHEMA = "sctransnet_nuaa_current_scalar_calibration_resume_v1/v1"
BEST_CHECKPOINT_SCHEMA = (
    "sctransnet_nuaa_current_scalar_calibration_best_internal_val_v1/v1"
)
DATASET = "NUAA-SIRST"
SEED = 42
SPLIT_SEED = 20260722
VALIDATION_FRACTION = 0.20
PARENT_ROLES = tuple(v3_registry.PARENT_ROLES)
MAX_EPOCHS = 100
DEFAULT_EPOCHS = 100
EVAL_EVERY = 5
TRAIN_BATCH_SIZE = 16
LEARNING_RATE = 1.0e-2
WEIGHT_DECAY = 0.0
FIXED_THRESHOLD = 0.5
THRESHOLD_START = 0.20
THRESHOLD_STOP = 0.80
THRESHOLD_STEP = 0.01
EMPTY_ENDPOINT = 1.0
MINIMUM_MIOU_GAIN = 0.002
MAXIMUM_FA_RATIO = 1.0
REQUIRE_NIOU_NON_DECREASE = True

DEFAULT_RESULTS_ROOT = REPO_ROOT / (
    "results/nuaa_current_scalar_calibration_control_v1"
)
STAGE1_RUNNER_RELATIVE = "experiments/train_nuaa_pbdr_v3_stage1_v1.py"
RELATIVE_SOURCE = "analysis/train_nuaa_current_scalar_calibration_control_v1.py"

RUNTIME_DEPENDENCY_RELATIVE_PATHS = (
    RELATIVE_SOURCE,
    STAGE1_RUNNER_RELATIVE,
    "experiments/three_dataset_pbdr_v3_models_seed42_v1.py",
    "experiments/pbdr_v3_non_regression_gate.py",
    "experiments/evaluate_three_dataset_v2.py",
    "experiments/four_dataset_evaluation_protocol_v1.py",
    "experiments/three_dataset_v2_protocol.py",
    "experiments/paper_three_dataset_v2.py",
    "experiments/train_tpd_pilot.py",
    "experiments/train_four_dataset_original_final_seed42_exact_v1.py",
    "experiments/PBDR_V3_PROTOCOL.md",
    "model/Config.py",
    "model/SCTransNet.py",
    "model/tpd_conservative_residual_calibrator_v3.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v3.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py",
    "model/tpd_frequency_gate_v2_croa.py",
    "model/tpd_query_frequency_bridge.py",
    "model/tpd_survival.py",
)


class ScalarCalibrationProtocolError(ValueError):
    """The E1 identity, split, parent, resume, or metric contract differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScalarCalibrationProtocolError(message)


def file_sha256(path: Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def runtime_source_records() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for relative in RUNTIME_DEPENDENCY_RELATIVE_PATHS:
        path = (REPO_ROOT / relative).resolve(strict=True)
        _require(path.is_relative_to(REPO_ROOT), "runtime source escapes repository")
        output[relative] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    return output


def build_threshold_grid(
    start: float = THRESHOLD_START,
    stop: float = THRESHOLD_STOP,
    step: float = THRESHOLD_STEP,
) -> tuple[float, ...]:
    for value, name in ((start, "start"), (stop, "stop"), (step, "step")):
        _require(math.isfinite(float(value)), f"threshold {name} is non-finite")
    _require(0.0 <= start <= stop < EMPTY_ENDPOINT, "threshold interval differs")
    _require(step > 0.0, "threshold step must be positive")
    raw_count = (stop - start) / step
    count = int(round(raw_count))
    _require(
        math.isclose(raw_count, count, rel_tol=0.0, abs_tol=1.0e-9),
        "threshold step does not close interval",
    )
    values = tuple(round(start + index * step, 12) for index in range(count + 1))
    _require(values[0] == start and values[-1] == stop, "threshold endpoints differ")
    _require(FIXED_THRESHOLD in values, "threshold grid omits fixed 0.5")
    return values


def configure_inference_math() -> dict[str, Any]:
    determinism_core.configure_determinism()
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


class ScalarTemperatureBiasCalibrator(nn.Module):
    """Exactly two scalar parameters with an exact identity initialization."""

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("scalar calibrator requires floating-point dtype")
        self.log_temperature = nn.Parameter(
            torch.zeros((), device=device, dtype=dtype)
        )
        self.bias = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        _require(sum(value.numel() for value in self.parameters()) == 2, "scalar parameter count differs")
        _require(tuple(self.state_dict()) == ("log_temperature", "bias"), "scalar state keys differ")

    @property
    def temperature(self) -> torch.Tensor:
        value = torch.exp(self.log_temperature)
        if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
            raise FloatingPointError("temperature is non-finite or non-positive")
        return value

    def forward(self, base_logits: torch.Tensor) -> torch.Tensor:
        if not isinstance(base_logits, torch.Tensor):
            raise TypeError("base_logits must be a tensor")
        if base_logits.ndim != 4 or base_logits.shape[1] != 1:
            raise ValueError("base_logits must have shape N,1,H,W")
        if not base_logits.is_floating_point():
            raise TypeError("base_logits must have floating dtype")
        if not bool(torch.isfinite(base_logits).all()):
            raise FloatingPointError("base_logits are non-finite")
        output = base_logits / self.temperature.to(dtype=base_logits.dtype)
        output = output + self.bias.to(dtype=base_logits.dtype)
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("calibrated logits are non-finite")
        return output

    def scalar_values(self) -> dict[str, float]:
        return {
            "log_temperature": float(self.log_temperature.detach().cpu()),
            "temperature": float(self.temperature.detach().cpu()),
            "bias": float(self.bias.detach().cpu()),
        }


def freeze_v3_shell_for_scalar_control(model: nn.Module) -> dict[str, Any]:
    """Freeze Current and pbdr_v3; no shell tensor may receive a gradient."""

    _require(hasattr(model, "pbdr_v3"), "warm-start shell lacks pbdr_v3")
    _require(
        callable(getattr(model, "forward_for_pbdr_v3_training", None)),
        "warm-start shell lacks explicit base-logit interface",
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    bad_parameters = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    bad_modules = [name for name, module in model.named_modules() if module.training]
    _require(not bad_parameters, f"shell parameters remain trainable: {bad_parameters}")
    _require(not bad_modules, f"shell modules remain in training mode: {bad_modules}")
    return {
        "shell_trainable_parameter_names": bad_parameters,
        "shell_trainable_parameter_count": 0,
        "pbdr_v3_frozen": all(
            not parameter.requires_grad for parameter in model.pbdr_v3.parameters()
        ),
        "shell_training": model.training,
        "pbdr_v3_training": model.pbdr_v3.training,
        "base_state_sha256": v3_registry.base_state_sha256(model),
        "batchnorm_buffer_sha256": v3_registry.batchnorm_buffer_sha256(model),
        "full_shell_state_sha256": v3_registry.tensor_mapping_sha256(
            model.state_dict()
        ),
    }


def build_optimizer(
    calibrator: ScalarTemperatureBiasCalibrator,
) -> torch.optim.Optimizer:
    named = tuple(calibrator.named_parameters())
    _require(
        tuple(name for name, _ in named) == ("log_temperature", "bias"),
        "optimizer scalar names differ",
    )
    _require(
        all(parameter.requires_grad for _, parameter in named),
        "scalar parameters are frozen",
    )
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    optimized = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    _require(optimized == 2, "optimizer does not own exactly two scalars")
    return optimizer


def _extract_image_mask(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, Mapping):
        images, masks = batch.get("image"), batch.get("mask")
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        images, masks = batch[0], batch[1]
    else:
        raise TypeError("unsupported internal-training batch")
    if not isinstance(images, torch.Tensor) or not isinstance(masks, torch.Tensor):
        raise TypeError("internal-training image/mask must be tensors")
    _require(images.ndim == masks.ndim == 4, "training tensors must be NCHW")
    _require(images.shape == masks.shape, "training image/mask shapes differ")
    return images, masks


def set_internal_train_epoch(dataset: Any, epoch: int) -> None:
    target = getattr(dataset, "dataset", dataset)
    setter = getattr(target, "set_epoch", None)
    _require(callable(setter), "internal train dataset lacks set_epoch")
    setter(epoch)


def train_scalar_epoch(
    *,
    model: nn.Module,
    calibrator: ScalarTemperatureBiasCalibrator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    """Train only the two scalars using frozen same-forward Current logits."""

    _require(1 <= epoch <= MAX_EPOCHS, "scalar epoch is outside [1,100]")
    _require(not model.training and not model.pbdr_v3.training, "shell left eval mode")
    _require(
        not any(parameter.requires_grad for parameter in model.parameters()),
        "shell has trainable parameters",
    )
    set_internal_train_epoch(loader.dataset, epoch)
    calibrator.train()
    total_loss = 0.0
    total_samples = 0
    batches = 0
    for batch in loader:
        images, masks = _extract_image_mask(batch)
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        with torch.no_grad():
            _probabilities, auxiliary = model.forward_for_pbdr_v3_training(images)
            base_logits = auxiliary.base_logits.detach()
        optimizer.zero_grad(set_to_none=True)
        calibrated = calibrator(base_logits)
        loss = F.binary_cross_entropy_with_logits(
            calibrated.float(), masks.float(), reduction="mean"
        )
        _require(bool(torch.isfinite(loss)), "scalar training loss is non-finite")
        loss.backward()
        gradients = [calibrator.log_temperature.grad, calibrator.bias.grad]
        _require(all(value is not None for value in gradients), "scalar gradient missing")
        _require(
            all(bool(torch.isfinite(cast(torch.Tensor, value)).all()) for value in gradients),
            "scalar gradient is non-finite",
        )
        optimizer.step()
        for name, parameter in calibrator.named_parameters():
            _require(bool(torch.isfinite(parameter).all()), f"scalar {name} is non-finite")
        count = int(images.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_samples += count
        batches += 1
    _require(total_samples > 0 and batches > 0, "internal train loader is empty")
    return {
        "epoch": epoch,
        "mean_bce_with_logits": total_loss / total_samples,
        "sample_count": total_samples,
        "batch_count": batches,
        "scalars": calibrator.scalar_values(),
    }


def certification_payload(
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    decision = gate_core.certify(
        gate_core.CertificationMetrics.from_mapping(current),
        gate_core.CertificationMetrics.from_mapping(candidate),
        minimum_miou_gain=MINIMUM_MIOU_GAIN,
        maximum_fa_ratio=MAXIMUM_FA_RATIO,
        require_niou_non_decrease=REQUIRE_NIOU_NON_DECREASE,
    )
    return {
        "passed": decision.passed,
        "selected": decision.selected,
        "checks": dict(decision.checks),
        "current": asdict(decision.current),
        "candidate": asdict(decision.candidate),
        "scope": "frozen_nuaa_internal_validation_only",
    }


def point_selection_key(point: Mapping[str, Any]) -> tuple[Any, ...]:
    certification = point.get("certification")
    _require(isinstance(certification, Mapping), "point lacks certification")
    checks = certification.get("checks")
    _require(isinstance(checks, Mapping), "point certification lacks checks")
    return (
        int(bool(certification.get("passed"))),
        sum(int(bool(value)) for value in checks.values()),
        int(point["matched_target_count"]),
        -float(point["fa"]),
        float(point["miou"]),
        float(point["niou"]),
        -abs(float(point["threshold"]) - FIXED_THRESHOLD),
    )


def select_validation_point(
    points: Sequence[Mapping[str, Any]],
    current_fixed: Mapping[str, Any],
) -> dict[str, Any]:
    """Select one registered validation threshold, passing gates first."""

    _require(bool(points), "validation threshold point set is empty")
    annotated: list[dict[str, Any]] = []
    for point in points:
        ready = dict(point)
        ready["certification"] = certification_payload(current_fixed, ready)
        annotated.append(ready)
    selected = max(annotated, key=point_selection_key)
    return {
        "selected": selected,
        "registered_point_count": len(annotated),
        "passing_point_count": sum(
            int(point["certification"]["passed"]) for point in annotated
        ),
        "all_points": annotated,
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


def restore_rng_state(state: Mapping[str, Any]) -> None:
    _require(
        set(state) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "resume RNG state fields differ",
    )
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    cuda_state = state["torch_cuda"]
    if torch.cuda.is_available() and cuda_state:
        torch.cuda.set_rng_state_all(
            [value.detach().cpu() for value in cuda_state]
        )


def build_run_identity(
    *,
    parent_role: str,
    parent_record: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    source_records: Mapping[str, Any],
    epochs: int,
) -> dict[str, Any]:
    _require(parent_role in PARENT_ROLES, "parent role differs")
    _require(type(epochs) is int and 1 <= epochs <= MAX_EPOCHS, "epochs must be in [1,100]")
    _require(parent_record.get("checkpoint_role") == parent_role, "parent record role differs")
    parent_sha = parent_record.get("sha256")
    _require(isinstance(parent_sha, str) and len(parent_sha) == 64, "parent SHA differs")
    validate_internal_split_manifest(split_manifest)
    identity = {
        "schema": SCHEMA,
        "dataset": DATASET,
        "seed": SEED,
        "parent_role": parent_role,
        "parent_checkpoint": dict(parent_record),
        "internal_split_sha256": canonical_sha256(split_manifest),
        "internal_split": dict(split_manifest),
        "runtime_sources": dict(source_records),
        "runtime_sources_sha256": canonical_sha256(source_records),
        "epochs": epochs,
        "maximum_epochs": MAX_EPOCHS,
        "eval_every": EVAL_EVERY,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "parameter_names": ["log_temperature", "bias"],
            "parameter_count": 2,
        },
        "thresholds": {
            "fixed": FIXED_THRESHOLD,
            "validation_grid": list(build_threshold_grid()),
            "empty_control": EMPTY_ENDPOINT,
        },
        "official_test_access_authorized": False,
    }
    identity["identity_sha256"] = canonical_sha256(identity)
    return identity


def validate_internal_split_manifest(split_manifest: Mapping[str, Any]) -> None:
    """Validate the exact shared Stage-1 official-train-only split."""

    _require(
        split_manifest.get("schema")
        == "sctransnet_nuaa_pbdr_v3_internal_split/v1",
        "split schema differs",
    )
    _require(split_manifest.get("dataset") == DATASET, "split dataset differs")
    _require(
        split_manifest.get("source_split") == "official_train_only",
        "split source is not official train only",
    )
    _require(
        split_manifest.get("official_test_index_opened") is False,
        "official test index was opened",
    )
    _require(split_manifest.get("split_seed") == SPLIT_SEED, "split seed differs")
    _require(
        float(split_manifest.get("val_fraction")) == VALIDATION_FRACTION,
        "split validation fraction differs",
    )
    official = split_manifest.get("official_train_ids")
    training = split_manifest.get("development_train_ids")
    validation = split_manifest.get("internal_validation_ids")
    _require(
        isinstance(official, list)
        and isinstance(training, list)
        and isinstance(validation, list),
        "split ID lists differ",
    )
    _require(len(official) == 213, "official NUAA train count differs")
    _require(len(validation) == 43 and len(training) == 170, "internal split counts differ")
    _require(len(set(official)) == len(official), "official train IDs repeat")
    _require(not (set(training) & set(validation)), "internal split overlaps")
    _require(
        set(training) | set(validation) == set(official),
        "internal split does not partition official train",
    )
    expected_index_sha = data_protocol.EXPECTED_SPLITS[DATASET]["train"][
        "file_sha256"
    ]
    _require(
        split_manifest.get("official_train_index_sha256") == expected_index_sha,
        "official train index SHA differs",
    )
    declared = split_manifest.get("split_sha256")
    _require(isinstance(declared, str) and len(declared) == 64, "split SHA differs")
    unsigned = dict(split_manifest)
    del unsigned["split_sha256"]
    _require(
        v3_registry.canonical_sha256(unsigned) == declared,
        "split manifest canonical SHA differs",
    )


def validate_resume_payload(
    payload: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    maximum_completed_epoch: int,
) -> None:
    _require(payload.get("schema") == RESUME_SCHEMA, "resume schema differs")
    _require(
        payload.get("identity_sha256") == expected_identity_sha256,
        "resume source/parent/split identity differs",
    )
    epoch = payload.get("completed_epoch")
    _require(
        type(epoch) is int and 0 <= epoch <= maximum_completed_epoch <= MAX_EPOCHS,
        "resume completed epoch differs",
    )
    state = payload.get("calibrator_state")
    _require(isinstance(state, Mapping), "resume lacks scalar state")
    _require(tuple(state) == ("log_temperature", "bias"), "resume scalar state keys differ")
    for name, value in state.items():
        _require(
            isinstance(value, torch.Tensor)
            and value.numel() == 1
            and bool(torch.isfinite(value).all()),
            f"resume scalar {name} differs",
        )
    optimizer_state = payload.get("optimizer_state")
    _require(isinstance(optimizer_state, Mapping), "resume lacks optimizer state")
    _require(
        set(optimizer_state) == {"state", "param_groups"},
        "resume optimizer fields differ",
    )
    param_groups = optimizer_state["param_groups"]
    _require(
        isinstance(param_groups, list)
        and len(param_groups) == 1
        and param_groups[0].get("params") == [0, 1],
        "resume optimizer does not own exactly two scalars",
    )
    rng_state = payload.get("rng_state")
    _require(
        isinstance(rng_state, Mapping)
        and set(rng_state) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "resume RNG state differs",
    )
    history = payload.get("history")
    _require(
        isinstance(history, list) and len(history) == epoch,
        "resume history length differs from completed epoch",
    )
    _require(
        all(
            isinstance(event, Mapping) and event.get("epoch") == index
            for index, event in enumerate(history, start=1)
        ),
        "resume history epoch sequence differs",
    )
    _require(isinstance(payload.get("selection"), Mapping), "resume selection differs")


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ScalarCalibrationProtocolError("JSON value is non-finite")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any], *, replace: bool) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ready = _json_ready(payload)
    if destination.exists() and not replace:
        observed = json.loads(destination.read_text(encoding="utf-8"))
        _require(observed == ready, f"write-once JSON conflicts: {destination}")
        return
    content = (
        json.dumps(ready, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _stage1_runner() -> Any:
    """Import only at execution time so pure CPU unit tests stay lightweight."""

    return importlib.import_module(
        "experiments.train_nuaa_pbdr_v3_stage1_v1"
    )


def _extract_validation_batch(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor, Any, str]:
    if isinstance(batch, Mapping):
        images = batch.get("image")
        masks = batch.get("mask")
        sizes = batch.get("original_hw")
        sample_ids = batch.get("sample_id")
    elif isinstance(batch, (tuple, list)) and len(batch) >= 4:
        images, masks, sizes, sample_ids = batch[:4]
    else:
        raise TypeError("unsupported internal-validation batch")
    if not isinstance(images, torch.Tensor) or not isinstance(masks, torch.Tensor):
        raise TypeError("internal-validation image/mask must be tensors")
    _require(images.shape[0] == masks.shape[0] == 1, "validation requires batch size 1")
    if isinstance(sample_ids, (tuple, list)):
        _require(len(sample_ids) == 1, "validation sample ID batch differs")
        identifier = str(sample_ids[0])
    else:
        identifier = str(sample_ids)
    return images, masks, sizes, identifier


@torch.inference_mode()
def collect_internal_validation_base_logits(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    expected_identifiers: Sequence[str],
) -> dict[str, Any]:
    """Run the frozen shell once over internal val; never opens test data."""

    _require(not model.training and not model.pbdr_v3.training, "shell left eval mode")
    state_before = v3_registry.tensor_mapping_sha256(model.state_dict())
    base_logits: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    identifiers: list[str] = []
    digest = hashlib.sha256()
    for batch in loader:
        images, masks, sizes, identifier = _extract_validation_batch(batch)
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        _probabilities, auxiliary = model.forward_for_pbdr_v3_training(images)
        logits = auxiliary.base_logits
        height, width = metric_runner._extract_hw(sizes)
        logits = logits[:, :, :height, :width]
        target = masks[:, :, :height, :width]
        _require(logits.shape == target.shape, "validation logits/target shape differs")
        _require(bool(torch.isfinite(logits).all()), "validation base logits non-finite")
        logit_array = np.array(logits[0, 0].float().cpu().numpy(), copy=True)
        target_array = np.array(target[0, 0].float().cpu().numpy(), copy=True)
        base_logits.append(logit_array)
        targets.append(target_array)
        identifiers.append(identifier)
        digest.update(identifier.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(logit_array).tobytes())
        digest.update(np.ascontiguousarray(target_array).tobytes())
    _require(identifiers == list(expected_identifiers), "internal val order differs")
    _require(bool(identifiers), "internal validation set is empty")
    state_after = v3_registry.tensor_mapping_sha256(model.state_dict())
    _require(state_after == state_before, "shell changed during validation caching")
    return {
        "base_logits": base_logits,
        "targets": targets,
        "identifiers": identifiers,
        "sample_count": len(identifiers),
        "cache_scope": "memory_only_current_process",
        "cache_written_to_disk": False,
        "tensor_stream_sha256": digest.hexdigest(),
        "shell_state_sha256_before": state_before,
        "shell_state_sha256_after": state_after,
        "shell_state_unchanged": True,
    }


def evaluate_scalar_values_on_cache(
    *,
    cache: Mapping[str, Any],
    log_temperature: float,
    bias: float,
    threshold_grid: Sequence[float] | None,
) -> dict[str, Any]:
    """Evaluate fixed 0.5 and optional internal-val threshold sweep."""

    _require(math.isfinite(log_temperature), "log_temperature is non-finite")
    _require(math.isfinite(bias), "bias is non-finite")
    temperature = math.exp(log_temperature)
    _require(math.isfinite(temperature) and temperature > 0.0, "temperature differs")
    raw_logits = cache.get("base_logits")
    targets = cache.get("targets")
    _require(isinstance(raw_logits, list) and isinstance(targets, list), "validation cache differs")
    _require(bool(raw_logits) and len(raw_logits) == len(targets), "validation cache length differs")
    probabilities: list[np.ndarray] = []
    losses: list[float] = []
    for logit_array, target_array in zip(raw_logits, targets):
        logits = torch.from_numpy(np.asarray(logit_array, dtype=np.float32))
        target = torch.from_numpy(np.asarray(target_array, dtype=np.float32))
        calibrated = logits / np.float32(temperature) + np.float32(bias)
        probability = torch.sigmoid(calibrated)
        loss = F.binary_cross_entropy_with_logits(
            calibrated.float(), target.float(), reduction="mean"
        )
        _require(
            bool(torch.isfinite(probability).all()) and bool(torch.isfinite(loss)),
            "validation calibration is non-finite",
        )
        probabilities.append(np.array(probability.numpy(), copy=True))
        losses.append(float(loss))
    if threshold_grid is None:
        registered = (FIXED_THRESHOLD, EMPTY_ENDPOINT)
    else:
        grid = tuple(float(value) for value in threshold_grid)
        _require(FIXED_THRESHOLD in grid and EMPTY_ENDPOINT not in grid, "validation grid differs")
        registered = (*grid, EMPTY_ENDPOINT)
    evaluated = metric_runner.evaluate_probability_arrays(
        probabilities,
        targets,
        losses,
        sweep_thresholds=registered,
    )
    all_points = evaluated["descriptive_pd_fa"]["points"]
    points = [
        point for point in all_points if float(point["threshold"]) != EMPTY_ENDPOINT
    ]
    endpoint = [
        point for point in all_points if float(point["threshold"]) == EMPTY_ENDPOINT
    ]
    _require(len(endpoint) == 1, "validation empty endpoint differs")
    return {
        "scalars": {
            "log_temperature": log_temperature,
            "temperature": temperature,
            "bias": bias,
        },
        "fixed_threshold_0_5": evaluated["fixed_threshold_0_5"],
        "registered_points": points,
        "threshold_1_0_empty_control": endpoint[0],
        "probability_cache_written": False,
    }


def evaluate_validation_epoch(
    *,
    cache: Mapping[str, Any],
    calibrator: ScalarTemperatureBiasCalibrator,
    current_fixed: Mapping[str, Any],
    threshold_grid: Sequence[float],
    epoch: int,
) -> dict[str, Any]:
    values = calibrator.scalar_values()
    evaluated = evaluate_scalar_values_on_cache(
        cache=cache,
        log_temperature=values["log_temperature"],
        bias=values["bias"],
        threshold_grid=threshold_grid,
    )
    fixed = dict(evaluated["fixed_threshold_0_5"])
    fixed["certification"] = certification_payload(current_fixed, fixed)
    threshold_selection = select_validation_point(
        evaluated["registered_points"], current_fixed
    )
    return {
        "epoch": epoch,
        "scalars": evaluated["scalars"],
        "fixed_threshold_0_5": fixed,
        "val_selected_threshold": threshold_selection["selected"],
        "registered_threshold_point_count": threshold_selection[
            "registered_point_count"
        ],
        "passing_threshold_point_count": threshold_selection[
            "passing_point_count"
        ],
        "threshold_1_0_empty_control": evaluated[
            "threshold_1_0_empty_control"
        ],
    }


def _clone_scalar_state(
    calibrator: ScalarTemperatureBiasCalibrator,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in calibrator.state_dict().items()
    }


def _selection_candidate(
    validation: Mapping[str, Any],
    *,
    point_field: str,
    calibrator: ScalarTemperatureBiasCalibrator,
) -> dict[str, Any]:
    point = validation.get(point_field)
    _require(isinstance(point, Mapping), f"validation lacks {point_field}")
    return {
        "epoch": int(validation["epoch"]),
        "point_field": point_field,
        "point": dict(point),
        "scalars": dict(validation["scalars"]),
        "calibrator_state": _clone_scalar_state(calibrator),
    }


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (*point_selection_key(candidate["point"]), -int(candidate["epoch"]))


def update_selection_state(
    selection: Mapping[str, Any],
    *,
    validation: Mapping[str, Any],
    calibrator: ScalarTemperatureBiasCalibrator,
) -> tuple[dict[str, Any], bool]:
    ready = dict(selection)
    threshold_candidate = _selection_candidate(
        validation,
        point_field="val_selected_threshold",
        calibrator=calibrator,
    )
    fixed_candidate = _selection_candidate(
        validation,
        point_field="fixed_threshold_0_5",
        calibrator=calibrator,
    )
    prior_threshold = ready.get("best_val_selected_threshold")
    threshold_improved = (
        not isinstance(prior_threshold, Mapping)
        or _candidate_key(threshold_candidate) > _candidate_key(prior_threshold)
    )
    if threshold_improved:
        ready["best_val_selected_threshold"] = threshold_candidate
    prior_fixed = ready.get("best_fixed_threshold_0_5")
    if (
        not isinstance(prior_fixed, Mapping)
        or _candidate_key(fixed_candidate) > _candidate_key(prior_fixed)
    ):
        ready["best_fixed_threshold_0_5"] = fixed_candidate
    return ready, threshold_improved


def _json_selection(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key != "calibrator_state"
    }


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for name, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def run_directory(results_root: Path, parent_role: str) -> Path:
    _require(parent_role in PARENT_ROLES, "parent role differs")
    return (
        Path(results_root).resolve()
        / "formal"
        / parent_role
        / "scalar_temperature_bias"
    )


def _resolve_device(device_name: str, stage1: Any) -> torch.device:
    _require(device_name in ("cpu", "cuda:0"), "device must be cpu or cuda:0")
    device = torch.device(device_name)
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
        _require(torch.cuda.device_count() == 1, "exactly one GPU must be visible")
        expected_uuid = str(stage1.GPU0_UUID)
        _require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == expected_uuid,
            "GPU0 UUID visibility binding differs",
        )
        actual = str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
        if not actual.startswith("GPU-"):
            actual = f"GPU-{actual}"
        _require(actual == expected_uuid, "visible GPU UUID differs")
    return device


def _resume_payload(
    *,
    identity_sha256: str,
    epoch: int,
    calibrator: ScalarTemperatureBiasCalibrator,
    optimizer: torch.optim.Optimizer,
    history: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RESUME_SCHEMA,
        "identity_sha256": identity_sha256,
        "completed_epoch": epoch,
        "calibrator_state": _clone_scalar_state(calibrator),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "history": list(history),
        "selection": dict(selection),
    }


def _summary_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    ready = _json_selection(candidate)
    point = ready.get("point")
    _require(isinstance(point, Mapping), "summary candidate point differs")
    return ready


def validate_completed_summary(
    payload: Mapping[str, Any],
    *,
    identity_sha256: str,
) -> None:
    _require(payload.get("schema") == SCHEMA, "summary schema differs")
    _require(payload.get("status") == "complete", "summary is incomplete")
    _require(payload.get("dataset") == DATASET, "summary dataset differs")
    _require(payload.get("parent_role") in PARENT_ROLES, "summary role differs")
    _require(payload.get("identity_sha256") == identity_sha256, "summary identity differs")
    _require(payload.get("epochs", 0) <= MAX_EPOCHS, "summary exceeds 100 epochs")
    _require(payload.get("official_test_accessed") is False, "official test was accessed")
    result = payload.get("calibration_alone_restores_internal_current_gate")
    _require(isinstance(result, Mapping), "summary lacks calibration conclusion")
    _require(
        isinstance(result.get("fixed_threshold_0_5"), bool)
        and isinstance(result.get("val_selected_threshold"), bool),
        "summary calibration conclusion differs",
    )


def train_control(
    *,
    parent_role: str,
    parent_checkpoint: Path | None = None,
    data_root: Path = data_protocol.DEFAULT_DATASET_ROOT,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    epochs: int = DEFAULT_EPOCHS,
    workers: int = 0,
    device_name: str = "cuda:0",
    resume: str = "auto",
) -> Path:
    """Run E1 with exact resume and no official-test construction."""

    _require(parent_role in PARENT_ROLES, "parent role differs")
    _require(type(epochs) is int and 1 <= epochs <= MAX_EPOCHS, "epochs must be in [1,100]")
    _require(type(workers) is int and workers >= 0, "workers must be non-negative")
    _require(resume in ("auto", "never", "required"), "resume mode differs")
    stage1 = _stage1_runner()
    _require(stage1.SPLIT_SEED == SPLIT_SEED, "shared split seed differs")
    _require(stage1.VAL_FRACTION == VALIDATION_FRACTION, "shared val fraction differs")
    _require(tuple(stage1.THRESHOLDS) == build_threshold_grid(), "shared threshold grid differs")
    inference_math = configure_inference_math()
    determinism_core.seed_everything(SEED)
    device = _resolve_device(device_name, stage1)
    root = Path(data_root).resolve(strict=True)
    output_dir = run_directory(Path(results_root), parent_role)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (output_dir / "run.lock").open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(
                lock_handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            raise RuntimeError("another scalar-control run holds the lock") from error

        split = stage1.build_internal_split_manifest(
            data_root=root,
            split_seed=SPLIT_SEED,
            val_fraction=VALIDATION_FRACTION,
        )
        validate_internal_split_manifest(split)
        sources = runtime_source_records()
        model, model_metadata = v3_registry.build_stage1_training_model(
            parent_role,
            parent_checkpoint=parent_checkpoint,
            seed=SEED,
        )
        model.to(device)
        shell_freeze = freeze_v3_shell_for_scalar_control(model)
        parent_record = model_metadata["parent_checkpoint"]
        identity = build_run_identity(
            parent_role=parent_role,
            parent_record=parent_record,
            split_manifest=split,
            source_records=sources,
            epochs=epochs,
        )
        identity_sha256 = str(identity["identity_sha256"])
        protocol = {
            "schema": SCHEMA,
            "run_identity": identity,
            "identity_sha256": identity_sha256,
            "model_registry_schema": model_metadata.get("schema"),
            "model_architecture_id": model_metadata.get("architecture_id"),
            "warm_start_used": model_metadata.get("warm_start_used"),
            "all_current_tensors_bitwise_equal_after_load": model_metadata.get(
                "all_current_tensors_bitwise_equal_after_load"
            ),
            "shell_freeze": shell_freeze,
            "inference_math": inference_math,
            "official_test_accessed": False,
            "official_test_access_authorized": False,
        }
        _atomic_json(output_dir / "split_manifest.json", split, replace=False)
        _atomic_json(output_dir / "protocol.json", protocol, replace=False)

        summary_path = output_dir / "summary.json"
        if summary_path.is_file():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            validate_completed_summary(existing, identity_sha256=identity_sha256)
            _require(runtime_source_records() == sources, "sources changed")
            _require(
                file_sha256(Path(parent_record["path"]))
                == parent_record["sha256"],
                "parent checkpoint changed",
            )
            return summary_path

        train_dataset = stage1.NUAAInternalTrainDataset(
            split["development_train_ids"],
            data_root=root,
            seed=SEED,
        )
        validation_dataset = stage1.NUAAInternalValidationDataset(
            split["internal_validation_ids"],
            data_root=root,
        )
        _require(
            list(train_dataset.sample_ids) == split["development_train_ids"],
            "shared internal train IDs differ",
        )
        _require(
            list(validation_dataset.sample_ids)
            == split["internal_validation_ids"],
            "shared internal val IDs differ",
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        validation_cache = collect_internal_validation_base_logits(
            model=model,
            loader=validation_loader,
            device=device,
            expected_identifiers=split["internal_validation_ids"],
        )
        current_evaluation = evaluate_scalar_values_on_cache(
            cache=validation_cache,
            log_temperature=0.0,
            bias=0.0,
            threshold_grid=None,
        )
        current_fixed = current_evaluation["fixed_threshold_0_5"]

        calibrator = ScalarTemperatureBiasCalibrator(device=device)
        optimizer = build_optimizer(calibrator)
        rolling_path = output_dir / "rolling_state.pth.tar"
        selected_path = output_dir / "selected_scalar_control.pth.tar"
        start_epoch = 1
        history: list[dict[str, Any]] = []
        selection: dict[str, Any] = {}
        if rolling_path.exists():
            _require(resume != "never", "rolling state exists but resume=never")
            resume_payload = torch.load(
                rolling_path,
                map_location="cpu",
                weights_only=False,
            )
            _require(isinstance(resume_payload, Mapping), "resume payload differs")
            validate_resume_payload(
                resume_payload,
                expected_identity_sha256=identity_sha256,
                maximum_completed_epoch=epochs,
            )
            incompatible = calibrator.load_state_dict(
                resume_payload["calibrator_state"], strict=True
            )
            _require(
                not incompatible.missing_keys and not incompatible.unexpected_keys,
                "resume scalar strict load differs",
            )
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            _optimizer_to_device(optimizer, device)
            history = list(resume_payload["history"])
            selection = dict(resume_payload["selection"])
            start_epoch = int(resume_payload["completed_epoch"]) + 1
            # Validation caching may consume framework iterator RNG.  Restore
            # only after rebuilding that deterministic, in-memory cache.
            restore_rng_state(resume_payload["rng_state"])
        else:
            _require(resume != "required", "resume=required but rolling state is missing")
            determinism_core.seed_everything(SEED)

        for epoch in range(start_epoch, epochs + 1):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                data_protocol.stable_sha256_uint64(SEED, "pbdr_v3", epoch)
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=TRAIN_BATCH_SIZE,
                shuffle=True,
                generator=generator,
                num_workers=workers,
                pin_memory=device.type == "cuda",
                drop_last=False,
            )
            training = train_scalar_epoch(
                model=model,
                calibrator=calibrator,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
            )
            validation: dict[str, Any] | None = None
            improved = False
            if epoch % EVAL_EVERY == 0 or epoch == epochs:
                validation = evaluate_validation_epoch(
                    cache=validation_cache,
                    calibrator=calibrator,
                    current_fixed=current_fixed,
                    threshold_grid=build_threshold_grid(),
                    epoch=epoch,
                )
                selection, improved = update_selection_state(
                    selection,
                    validation=validation,
                    calibrator=calibrator,
                )
                if improved:
                    checkpoint = {
                        "schema": BEST_CHECKPOINT_SCHEMA,
                        "identity_sha256": identity_sha256,
                        "parent_role": parent_role,
                        "parent_checkpoint": dict(parent_record),
                        "internal_split_sha256": split["split_sha256"],
                        "epoch": epoch,
                        "calibrator_state": _clone_scalar_state(calibrator),
                        "scalars": calibrator.scalar_values(),
                        "fixed_threshold_0_5": validation[
                            "fixed_threshold_0_5"
                        ],
                        "val_selected_threshold": validation[
                            "val_selected_threshold"
                        ],
                        "official_test_accessed": False,
                    }
                    _atomic_torch_save(selected_path, checkpoint)
                    selection["selected_checkpoint"] = {
                        "path": str(selected_path.resolve()),
                        "sha256": file_sha256(selected_path),
                        "bytes": selected_path.stat().st_size,
                        "epoch": epoch,
                    }
            event = {
                "epoch": epoch,
                "training": training,
                "validation": (
                    None
                    if validation is None
                    else {
                        "epoch": epoch,
                        "scalars": validation["scalars"],
                        "fixed_threshold_0_5": validation[
                            "fixed_threshold_0_5"
                        ],
                        "val_selected_threshold": validation[
                            "val_selected_threshold"
                        ],
                        "passing_threshold_point_count": validation[
                            "passing_threshold_point_count"
                        ],
                    }
                ),
                "selection_improved": improved,
            }
            history.append(event)
            rolling = _resume_payload(
                identity_sha256=identity_sha256,
                epoch=epoch,
                calibrator=calibrator,
                optimizer=optimizer,
                history=history,
                selection=selection,
            )
            _atomic_torch_save(rolling_path, rolling)
            _atomic_json(
                output_dir / "progress.json",
                {
                    "schema": SCHEMA,
                    "status": "running" if epoch < epochs else "training_complete",
                    "identity_sha256": identity_sha256,
                    **event,
                },
                replace=True,
            )

        _require(
            isinstance(selection.get("best_val_selected_threshold"), Mapping),
            "no internal-validation selection was produced",
        )
        _require(
            isinstance(selection.get("best_fixed_threshold_0_5"), Mapping),
            "no fixed-threshold selection was produced",
        )
        selected_record = selection.get("selected_checkpoint")
        _require(isinstance(selected_record, Mapping), "selected checkpoint is missing")
        _require(selected_path.is_file(), "selected checkpoint file is missing")
        _require(
            file_sha256(selected_path) == selected_record.get("sha256"),
            "selected checkpoint SHA differs",
        )
        shell_state_after = v3_registry.tensor_mapping_sha256(model.state_dict())
        base_after = v3_registry.base_state_sha256(model)
        batchnorm_after = v3_registry.batchnorm_buffer_sha256(model)
        _require(
            shell_state_after == shell_freeze["full_shell_state_sha256"],
            "warm-start shell changed during scalar control",
        )
        _require(
            base_after == shell_freeze["base_state_sha256"],
            "Current base changed during scalar control",
        )
        _require(
            batchnorm_after == shell_freeze["batchnorm_buffer_sha256"],
            "Current BatchNorm buffers changed during scalar control",
        )
        _require(runtime_source_records() == sources, "runtime sources changed")
        _require(
            file_sha256(Path(parent_record["path"])) == parent_record["sha256"],
            "parent checkpoint changed",
        )
        validate_internal_split_manifest(split)

        best_threshold = cast(
            Mapping[str, Any], selection["best_val_selected_threshold"]
        )
        best_fixed = cast(
            Mapping[str, Any], selection["best_fixed_threshold_0_5"]
        )
        threshold_passed = bool(
            best_threshold["point"]["certification"]["passed"]
        )
        fixed_passed = bool(best_fixed["point"]["certification"]["passed"])
        summary = {
            "schema": SCHEMA,
            "status": "complete",
            "dataset": DATASET,
            "parent_role": parent_role,
            "seed": SEED,
            "epochs": epochs,
            "identity_sha256": identity_sha256,
            "run_identity": identity,
            "current_internal_validation_fixed_threshold_0_5": current_fixed,
            "best_fixed_threshold_0_5": _summary_candidate(best_fixed),
            "best_val_selected_threshold": _summary_candidate(best_threshold),
            "calibration_alone_restores_internal_current_gate": {
                "fixed_threshold_0_5": fixed_passed,
                "val_selected_threshold": threshold_passed,
                "conclusion": (
                    "calibration_alone_restores_internal_current_gate"
                    if threshold_passed
                    else "calibration_alone_does_not_restore_internal_current_gate"
                ),
            },
            "selected": "scalar_control" if threshold_passed else "current",
            "selected_checkpoint": dict(selected_record),
            "rolling_state": {
                "path": str(rolling_path.resolve()),
                "sha256": file_sha256(rolling_path),
                "bytes": rolling_path.stat().st_size,
            },
            "internal_validation_cache": {
                key: value
                for key, value in validation_cache.items()
                if key not in ("base_logits", "targets")
            },
            "shell_state_sha256_before_after": [
                shell_freeze["full_shell_state_sha256"],
                shell_state_after,
            ],
            "base_state_sha256_before_after": [
                shell_freeze["base_state_sha256"],
                base_after,
            ],
            "batchnorm_buffer_sha256_before_after": [
                shell_freeze["batchnorm_buffer_sha256"],
                batchnorm_after,
            ],
            "optimizer_parameter_names": ["log_temperature", "bias"],
            "optimizer_parameter_count": 2,
            "official_test_accessed": False,
            "official_test_index_opened": False,
            "official_test_metrics": None,
            "internal_validation_selection_only": True,
            "training_history": history,
            "source_sha256": sources,
            "inference_math": inference_math,
            "training_started": True,
            "no_fabricated_results": True,
        }
        validate_completed_summary(summary, identity_sha256=identity_sha256)
        _atomic_json(summary_path, summary, replace=False)
        _atomic_json(
            output_dir / "progress.json",
            {
                "schema": SCHEMA,
                "status": "complete",
                "identity_sha256": identity_sha256,
                "summary": str(summary_path.resolve()),
                "calibration_alone_restores_internal_current_gate": threshold_passed,
                "official_test_accessed": False,
            },
            replace=True,
        )
        del model, calibrator, optimizer, validation_cache
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return summary_path
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-role", choices=PARENT_ROLES, required=True)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=data_protocol.DEFAULT_DATASET_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= MAX_EPOCHS:
        parser.error("--epochs must be in [1, 100]")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = train_control(
        parent_role=args.parent_role,
        parent_checkpoint=args.parent_checkpoint,
        data_root=args.data_root,
        results_root=args.results_root,
        epochs=args.epochs,
        workers=args.workers,
        device_name=args.device,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "summary": str(summary.resolve()),
                "sha256": file_sha256(summary),
                "official_test_accessed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BEST_CHECKPOINT_SCHEMA",
    "DATASET",
    "MAX_EPOCHS",
    "PARENT_ROLES",
    "RESUME_SCHEMA",
    "SCHEMA",
    "ScalarCalibrationProtocolError",
    "ScalarTemperatureBiasCalibrator",
    "build_optimizer",
    "build_run_identity",
    "build_threshold_grid",
    "capture_rng_state",
    "certification_payload",
    "collect_internal_validation_base_logits",
    "configure_inference_math",
    "evaluate_scalar_values_on_cache",
    "evaluate_validation_epoch",
    "freeze_v3_shell_for_scalar_control",
    "parse_args",
    "restore_rng_state",
    "run_directory",
    "select_validation_point",
    "train_control",
    "train_scalar_epoch",
    "update_selection_state",
    "validate_completed_summary",
    "validate_internal_split_manifest",
    "validate_resume_payload",
]
