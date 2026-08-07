#!/usr/bin/env python3
"""One-claim, one-loader-pass evaluator for both frozen PBDR-V4 role pools.

All source, split, candidate-pool, role, path, file, state, and configuration
bindings are checked before
:func:`pbdr_v4_official_once.execute_official_joint_once` commits its exclusive
claim.  The official index and loader are constructed
only by the supplied ``loader_factory``, which the dataset-level one-pass
boundary invokes after that claim.  Every batch is forwarded through all ten
role/family bindings and updates online metrics; no official probability or
logit cache is retained.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments import pbdr_v3_residual_calibration as residual_calibration
from experiments import pbdr_v4_candidate_pool as candidate_pool_module
from experiments import pbdr_v4_metric_core as metric_core
from experiments import pbdr_v4_models_seed42_v1 as v4_models
from experiments import pbdr_v4_official_once as official_once
from experiments import pbdr_v4_original_models as original_models
from experiments import pbdr_v4_source_lock as source_lock_module
from experiments import pbdr_v4_split_authority as split_authority
from experiments import pbdr_v4_training_core as training_core
from experiments import pbdr_v4_zero_margin_selector as zero_margin_selector
from experiments import three_dataset_pbdr_v3_models_seed42_v1 as nuaa_v3_models
from experiments import three_dataset_v2_protocol as data_protocol
from experiments import two_dataset_pbdr_v3_models_seed42_v1 as cross_v3_models


SCHEMA = "sctransnet_three_dataset_pbdr_v4_evaluator_v1/v1"
V3_CALIBRATED_ARTIFACT_SCHEMA = (
    "sctransnet_pbdr_v4_v3_calibrated_candidate/v1"
)
DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
ROLES = ("best_miou", "best_pd")
FAMILY_ORDER = tuple(zero_margin_selector.FROZEN_TIE_ORDER)
SYNTHETIC_PREFLIGHT_SIZE = (512, 512)
FORMAL_GPU_UUIDS: Mapping[str, str] = {
    "NUDT-SIRST": "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    "IRSTD-1K": "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
    "NUAA-SIRST": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "pbdr_v4_v1" / "official"
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
REQUIRED_SOURCE_LOCK_PATHS = (
    "experiments/PBDR_V4_PROTOCOL.md",
    "experiments/build_pbdr_v4_component_atlas.py",
    "experiments/evaluate_three_dataset_pbdr_v4_v1.py",
    "experiments/freeze_pbdr_v4_protocol.py",
    "experiments/launch_three_dataset_pbdr_v4_v1.py",
    "experiments/component_matching_v2.py",
    "experiments/pbdr_v3_residual_calibration.py",
    "experiments/pbdr_v4_candidate_pool.py",
    "experiments/pbdr_v4_metric_core.py",
    "experiments/pbdr_v4_models_seed42_v1.py",
    "experiments/pbdr_v4_official_once.py",
    "experiments/pbdr_v4_original_models.py",
    "experiments/pbdr_v4_source_lock.py",
    "experiments/pbdr_v4_split_authority.py",
    "experiments/pbdr_v4_zero_margin_selector.py",
    "experiments/prepare_pbdr_v4_internal_artifacts.py",
    "experiments/sweep_pbdr_v3_residual_calibration.py",
    "experiments/three_dataset_v2_protocol.py",
    "experiments/train_three_dataset_pbdr_v4_v1.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v4.py",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PBDRV4EvaluationError(RuntimeError):
    """A preclaim binding or one-pass inference contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV4EvaluationError(message)


def _require_dataset(value: str) -> str:
    _require(type(value) is str and value in DATASETS, f"dataset must be one of {DATASETS}")
    return value


def _require_role(value: str) -> str:
    _require(type(value) is str and value in ROLES, f"role must be one of {ROLES}")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{name} must be lowercase SHA-256",
    )
    return value


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    _require(all(type(key) is str for key in value), f"{name} keys must be strings")
    return value


def validate_evaluation_device(
    *,
    dataset: str,
    device: torch.device,
    expected_gpu_uuid: str | None,
) -> str | None:
    """Bind the formal logical CUDA device to the dataset's physical GPU."""

    ready_dataset = _require_dataset(dataset)
    _require(isinstance(device, torch.device), "device must be torch.device")
    if device.type != "cuda":
        _require(
            expected_gpu_uuid is None,
            "CPU evaluation cannot bind an expected GPU UUID",
        )
        return None
    fixed_uuid = FORMAL_GPU_UUIDS[ready_dataset]
    _require(
        expected_gpu_uuid == fixed_uuid,
        f"{ready_dataset} expected GPU must equal its fixed UUID; GPU2 is forbidden",
    )
    _require(
        device.index in (None, 0),
        "formal evaluator must use logical cuda:0 after UUID isolation",
    )
    properties = torch.cuda.get_device_properties(0)
    raw_uuid = getattr(properties, "uuid", None)
    _require(raw_uuid is not None, "CUDA runtime did not expose a device UUID")
    observed = str(raw_uuid)
    if not observed.startswith("GPU-"):
        observed = f"GPU-{observed}"
    _require(observed == fixed_uuid, f"CUDA device UUID differs: {observed}")
    return observed


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    candidate = Path(path)
    _require(candidate.is_file() and not candidate.is_symlink(), f"{name} is missing or unsafe")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PBDRV4EvaluationError(f"could not parse {name}: {error}") from error
    return dict(_require_mapping(value, name=name))


def _torch_payload(path: Path, *, name: str) -> dict[str, Any]:
    candidate = Path(path)
    _require(candidate.is_file() and not candidate.is_symlink(), f"{name} is missing or unsafe")
    try:
        value = torch.load(candidate, map_location="cpu", weights_only=False)
    except Exception as error:
        raise PBDRV4EvaluationError(f"could not load {name}: {error}") from error
    return dict(_require_mapping(value, name=name))


@dataclass(frozen=True, slots=True)
class CandidateRuntime:
    """One preclaim-built model with the exact candidate-pool binding."""

    family: str
    name: str
    model: nn.Module
    artifact_path: str
    artifact_sha256: str
    state_sha256: str
    configuration_sha256: str
    calibration: residual_calibration.ResidualCalibration | None
    checkpoint_binding: Mapping[str, object]

    def __post_init__(self) -> None:
        _require(self.family in FAMILY_ORDER, f"unsupported candidate family: {self.family!r}")
        _require(isinstance(self.name, str) and bool(self.name), "candidate name is empty")
        _require(isinstance(self.model, nn.Module), "candidate model must be a Module")
        path = Path(self.artifact_path)
        _require(path.is_absolute(), "candidate runtime artifact path must be absolute")
        _require_sha256(self.artifact_sha256, name="candidate artifact_sha256")
        _require_sha256(self.state_sha256, name="candidate state_sha256")
        _require_sha256(self.configuration_sha256, name="candidate configuration_sha256")
        _require_mapping(self.checkpoint_binding, name="candidate checkpoint binding")
        if self.family == "V3-calibrated":
            _require(
                isinstance(self.calibration, residual_calibration.ResidualCalibration),
                "V3-calibrated runtime lacks frozen calibration",
            )
        else:
            _require(self.calibration is None, f"{self.family} cannot carry V3 calibration")


@dataclass(slots=True)
class PreparedEvaluation:
    dataset: str
    candidate_pools: dict[str, dict[str, object]]
    joint_candidate_pool: dict[str, object]
    runtimes: dict[str, dict[str, CandidateRuntime]]
    audit: dict[str, object]


CandidateFactory = Callable[..., Mapping[str, CandidateRuntime]]


def candidate_configuration_sha256(
    *,
    family: str,
    dataset: str,
    role: str,
    details: Mapping[str, object],
) -> str:
    """Canonical configuration identity shared by pool freezing and loading."""

    _require(family in FAMILY_ORDER, "configuration family differs")
    ready_dataset = _require_dataset(dataset)
    ready_role = _require_role(role)
    return candidate_pool_module.canonical_sha256(
        {
            "schema": SCHEMA,
            "family": family,
            "dataset": ready_dataset,
            "role": ready_role,
            "fixed_probability_rule": "strict_greater_than_0.5",
            "details": dict(details),
        }
    )


def _validate_source_split_pool(
    *,
    dataset: str,
    role: str,
    source_lock: Mapping[str, object],
    split_projection: Mapping[str, object],
    candidate_pool: Mapping[str, object],
    check_environment: bool,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    ready_dataset = _require_dataset(dataset)
    ready_role = _require_role(role)
    locked = source_lock_module.validate_source_lock(
        source_lock,
        check_environment=check_environment,
    )
    _require(locked.get("official_test_accessed") is False, "source lock crossed official boundary")
    sources = _require_mapping(locked.get("sources"), name="source-lock sources")
    missing_sources = sorted(set(REQUIRED_SOURCE_LOCK_PATHS) - set(sources))
    _require(not missing_sources, f"source lock is incomplete: {missing_sources}")
    source_sha = _require_sha256(locked.get("source_lock_sha256"), name="source_lock_sha256")

    projection = dict(_require_mapping(split_projection, name="split projection"))
    live_projection = split_authority.build_projection()
    _require(projection == live_projection, "split projection differs from frozen authority")
    declared_projection_sha = _require_sha256(
        projection.get("projection_sha256"), name="split projection SHA-256"
    )
    unsigned_projection = dict(projection)
    unsigned_projection.pop("projection_sha256", None)
    _require(
        declared_projection_sha == split_authority.canonical_sha256(unsigned_projection),
        "split projection canonical SHA-256 differs",
    )
    _require(
        projection.get("official_test_accessed") is False
        and projection.get("split_reconstruction_performed") is False,
        "split projection crossed or reconstructed the official boundary",
    )
    projection_datasets = _require_mapping(
        projection.get("datasets"), name="split projection datasets"
    )
    split_record = _require_mapping(
        projection_datasets.get(ready_dataset),
        name=f"split projection {ready_dataset}",
    )
    _require(split_record.get("dataset") == ready_dataset, "split record dataset differs")
    _require(split_record.get("official_test_accessed") is False, "split record crossed official boundary")

    pool = candidate_pool_module.validate_candidate_pool(candidate_pool)
    _require(pool.get("dataset") == ready_dataset, "candidate pool dataset differs")
    _require(pool.get("role") == ready_role, "candidate pool role differs")
    _require(pool.get("source_lock_sha256") == source_sha, "candidate pool source-lock binding differs")
    _require(
        pool.get("split_projection_sha256") == declared_projection_sha,
        "candidate pool split binding differs",
    )
    _require(pool.get("family_order") == list(FAMILY_ORDER), "candidate family order differs")
    return locked, projection, pool


def _model_fp32_finite(model: nn.Module, *, family: str) -> None:
    tensors = (*model.named_parameters(), *model.named_buffers())
    _require(bool(tensors), f"{family} model has no tensor state")
    for name, tensor in tensors:
        if tensor.is_floating_point():
            _require(tensor.dtype == torch.float32, f"{family} tensor {name!r} is not FP32")
            _require(bool(torch.isfinite(tensor).all()), f"{family} tensor {name!r} is non-finite")


def prepare_evaluation(
    *,
    dataset: str,
    source_lock: Mapping[str, object],
    split_projection: Mapping[str, object],
    candidate_pools: Mapping[str, Mapping[str, object]],
    candidate_factory: CandidateFactory,
    device: torch.device,
    expected_gpu_uuid: str | None = None,
    check_environment: bool = True,
) -> PreparedEvaluation:
    """Complete both roles' artifact/model checks without constructing data."""

    _require(isinstance(device, torch.device), "device must be torch.device")
    _require(callable(candidate_factory), "candidate_factory must be callable")
    _require(
        isinstance(candidate_pools, Mapping) and tuple(candidate_pools) == ROLES,
        "candidate-pool role order differs",
    )
    determinism = training_core.configure_determinism(
        seed=training_core.TRAINING_SEED
    )
    observed_gpu_uuid = validate_evaluation_device(
        dataset=dataset,
        device=device,
        expected_gpu_uuid=expected_gpu_uuid,
    )
    validated_pools: dict[str, dict[str, object]] = {}
    locked: dict[str, object] | None = None
    projection: dict[str, object] | None = None
    for role in ROLES:
        role_lock, role_projection, pool = _validate_source_split_pool(
            dataset=dataset,
            role=role,
            source_lock=source_lock,
            split_projection=split_projection,
            candidate_pool=candidate_pools[role],
            check_environment=check_environment,
        )
        if locked is None:
            locked = role_lock
            projection = role_projection
        else:
            _require(role_lock == locked, "role source locks differ")
            _require(role_projection == projection, "role split projections differ")
        validated_pools[role] = pool
    assert locked is not None and projection is not None
    joint_pool = official_once.build_joint_candidate_pool(validated_pools)

    runtimes: dict[str, dict[str, CandidateRuntime]] = {}
    model_ids: set[int] = set()
    audit_by_role: dict[str, list[dict[str, object]]] = {}
    for role in ROLES:
        pool = validated_pools[role]
        raw_runtimes = candidate_factory(
            candidate_pool=pool,
            dataset=dataset,
            role=role,
            device=device,
        )
        runtimes_mapping = _require_mapping(
            raw_runtimes, name=f"{role} candidate runtimes"
        )
        _require(
            set(runtimes_mapping) == set(FAMILY_ORDER),
            f"{role} candidate runtime families differ",
        )
        records = {
            str(record["family"]): record
            for record in pool["candidates"]  # type: ignore[index]
        }
        role_runtimes: dict[str, CandidateRuntime] = {}
        audit_records: list[dict[str, object]] = []
        for family in FAMILY_ORDER:
            runtime = runtimes_mapping[family]
            label = f"{role}::{family}"
            _require(
                isinstance(runtime, CandidateRuntime),
                f"{label} candidate runtime is invalid",
            )
            record = _require_mapping(records[family], name=f"{label} pool record")
            expected = {
                "family": family,
                "name": record["name"],
                "artifact_path": record["artifact_path"],
                "artifact_sha256": record["artifact_sha256"],
                "state_sha256": record["state_sha256"],
                "configuration_sha256": record["configuration_sha256"],
            }
            for name, expected_value in expected.items():
                _require(
                    getattr(runtime, name) == expected_value,
                    f"{label} runtime {name} differs",
                )
            binding = _require_mapping(
                runtime.checkpoint_binding,
                name=f"{label} checkpoint binding",
            )
            _require(
                binding.get("dataset") == dataset,
                f"{label} checkpoint dataset differs",
            )
            _require(
                binding.get(
                    "role", binding.get("checkpoint_role", binding.get("parent_role"))
                )
                == role,
                f"{label} checkpoint role differs",
            )
            _require(
                binding.get("official_test_accessed") is False,
                f"{label} checkpoint binding crossed official boundary",
            )
            _require(
                id(runtime.model) not in model_ids,
                "role/family candidates reuse one model object",
            )
            model_ids.add(id(runtime.model))
            runtime.model.to(device)
            runtime.model.eval()
            runtime.model.mode = "test"
            _require(not runtime.model.training, f"{label} model is not in eval mode")
            _require(
                getattr(runtime.model, "mode", None) == "test",
                f"{label} model mode differs",
            )
            _model_fp32_finite(runtime.model, family=label)
            role_runtimes[family] = runtime
            audit_records.append(
                {
                    **expected,
                    "checkpoint_dataset": binding.get("dataset"),
                    "checkpoint_role": role,
                    "official_test_accessed": False,
                }
            )
        runtimes[role] = role_runtimes
        audit_by_role[role] = audit_records

    synthetic = torch.zeros(
        (1, 1, *SYNTHETIC_PREFLIGHT_SIZE),
        dtype=torch.float32,
        device=device,
    )
    synthetic_forward_counts: dict[str, int] = {}
    with torch.inference_mode():
        for role in ROLES:
            for family in FAMILY_ORDER:
                key = f"{role}::{family}"
                logits = forward_candidate_logits(runtimes[role][family], synthetic)
                _require(
                    tuple(logits.shape) == (1, 1, *SYNTHETIC_PREFLIGHT_SIZE),
                    f"{key} maximum-size preflight shape differs",
                )
                _require(
                    bool(torch.isfinite(logits).all()),
                    f"{key} maximum-size preflight is non-finite",
                )
                synthetic_forward_counts[key] = 1
                del logits
    del synthetic

    audit = {
        "schema": SCHEMA,
        "status": "complete_preclaim",
        "dataset": dataset,
        "role_order": list(ROLES),
        "source_lock_sha256": locked["source_lock_sha256"],
        "split_projection_sha256": projection["projection_sha256"],
        "candidate_pool_sha256_by_role": {
            role: validated_pools[role]["candidate_pool_sha256"] for role in ROLES
        },
        "joint_candidate_pool_sha256": joint_pool["joint_candidate_pool_sha256"],
        "candidate_count": len(ROLES) * len(FAMILY_ORDER),
        "family_order": list(FAMILY_ORDER),
        "execution_keys": list(joint_pool["execution_keys"]),
        "candidates_by_role": audit_by_role,
        "metric_manifest": metric_core.MetricCoreManifest().as_dict(),
        "determinism": dict(determinism),
        "expected_gpu_uuid": expected_gpu_uuid,
        "observed_gpu_uuid": observed_gpu_uuid,
        "synthetic_preflight_size": list(SYNTHETIC_PREFLIGHT_SIZE),
        "synthetic_preflight_forward_counts": synthetic_forward_counts,
        "synthetic_preflight_excluded_from_official_forward_counts": True,
        "models_prepared_before_claim": True,
        "dataset_or_loader_constructed_before_claim": False,
        "official_test_accessed": False,
    }
    return PreparedEvaluation(
        dataset=dataset,
        candidate_pools=validated_pools,
        joint_candidate_pool=joint_pool,
        runtimes=runtimes,
        audit=audit,
    )


@torch.inference_mode()
def forward_candidate_logits(
    runtime: CandidateRuntime,
    images: torch.Tensor,
) -> torch.Tensor:
    """Return one family's raw final logits from exactly one model forward."""

    _require(isinstance(runtime, CandidateRuntime), "runtime must be CandidateRuntime")
    _require(
        isinstance(images, torch.Tensor)
        and images.ndim == 4
        and images.shape[1] == 1
        and images.dtype == torch.float32,
        "images must be FP32 BCHW with C=1",
    )
    model = runtime.model
    if runtime.family == "Original":
        outc = getattr(model, "outc", None)
        _require(isinstance(outc, nn.Module), "Original model lacks outc hook point")
        captured: list[torch.Tensor] = []

        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            _require(isinstance(output, torch.Tensor), "Original outc output is not a tensor")
            captured.append(output)

        handle = outc.register_forward_hook(capture)
        try:
            probability = model(images)
        finally:
            handle.remove()
        _require(len(captured) == 1, "Original outc must execute exactly once")
        logits = captured[0]
        _require(
            isinstance(probability, torch.Tensor),
            "Original public probability output is not a tensor",
        )
        _require(
            probability.shape == logits.shape
            and probability.dtype == logits.dtype
            and probability.device == logits.device,
            "Original public probability shape/dtype/device differs from raw logits",
        )
        _require(
            torch.equal(torch.sigmoid(logits), probability),
            "Original probability is not bitwise sigmoid of captured raw logits",
        )
    elif runtime.family == "Current":
        method = getattr(model, "forward_for_pbdr_v4_training", None)
        _require(callable(method), "Current model lacks raw-logit forward")
        _, auxiliary = method(images)
        logits = getattr(auxiliary, "candidate_base_logits", None)
    elif runtime.family == "V3-calibrated":
        method = getattr(model, "forward_for_pbdr_v3_training", None)
        _require(callable(method), "V3 model lacks raw-logit forward")
        _, auxiliary = method(images)
        base_logits = getattr(auxiliary, "base_logits", None)
        routed_logits = getattr(auxiliary, "routed_logits", None)
        _require(
            isinstance(base_logits, torch.Tensor)
            and isinstance(routed_logits, torch.Tensor),
            "V3 raw logits are missing",
        )
        assert runtime.calibration is not None
        logits = residual_calibration.apply_residual_calibration(
            base_logits,
            routed_logits - base_logits,
            runtime.calibration,
        )
    else:
        method = getattr(model, "forward_for_pbdr_v4_training", None)
        _require(callable(method), f"{runtime.family} model lacks raw-logit forward")
        _, auxiliary = method(images)
        logits = getattr(auxiliary, "routed_logits", None)

    _require(isinstance(logits, torch.Tensor), f"{runtime.family} logits are missing")
    _require(
        logits.ndim == 4
        and logits.shape[0] == images.shape[0]
        and logits.shape[1] == 1
        and logits.shape[-2:] == images.shape[-2:],
        f"{runtime.family} logit shape differs",
    )
    _require(logits.dtype == torch.float32, f"{runtime.family} logits are not FP32")
    _require(logits.device == images.device, f"{runtime.family} logit device differs")
    _require(bool(torch.isfinite(logits).all()), f"{runtime.family} logits are non-finite")
    return logits


def _batch_sizes(value: Any, batch_size: int) -> list[tuple[int, int]]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 1 and batch_size == 1 and tensor.numel() == 2:
            return [(int(tensor[0]), int(tensor[1]))]
        if tensor.ndim == 2 and tuple(tensor.shape) == (batch_size, 2):
            return [(int(row[0]), int(row[1])) for row in tensor]
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(
        isinstance(item, torch.Tensor) for item in value
    ):
        heights = value[0].detach().cpu().reshape(-1)
        widths = value[1].detach().cpu().reshape(-1)
        if len(heights) == len(widths) == batch_size:
            return [(int(heights[index]), int(widths[index])) for index in range(batch_size)]
    if isinstance(value, (tuple, list)) and len(value) == batch_size:
        result: list[tuple[int, int]] = []
        for item in value:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                break
            result.append((int(item[0]), int(item[1])))
        if len(result) == batch_size:
            return result
    raise PBDRV4EvaluationError("collated original sizes are malformed")


class _OnlineEvaluation:
    def __init__(
        self,
        prepared: PreparedEvaluation,
        *,
        device: torch.device,
        operational_test_selected: bool,
    ) -> None:
        self.prepared = prepared
        self.device = device
        self.operational_test_selected = bool(operational_test_selected)
        self.accumulators = {
            role: {
                family: metric_core.PBDRV4MetricAccumulator()
                for family in FAMILY_ORDER
            }
            for role in ROLES
        }

    @torch.inference_mode()
    def consume_batch(self, batch: Any) -> dict[str, object]:
        _require(isinstance(batch, (tuple, list)) and len(batch) == 4, "official batch must have four fields")
        images, targets, sizes, identifiers = batch
        _require(
            isinstance(images, torch.Tensor)
            and isinstance(targets, torch.Tensor)
            and images.ndim == targets.ndim == 4
            and images.shape == targets.shape
            and images.shape[1] == 1,
            "official image/target batch shape differs",
        )
        _require(images.dtype == targets.dtype == torch.float32, "official batch is not FP32")
        _require(bool(torch.isfinite(images).all()) and bool(torch.isfinite(targets).all()), "official batch is non-finite")
        _require(
            bool((targets >= 0.0).all()) and bool((targets <= 1.0).all()),
            "official target is outside [0,1]",
        )
        batch_size = int(images.shape[0])
        _require(batch_size > 0, "official batch is empty")
        ready_sizes = _batch_sizes(sizes, batch_size)
        _require(
            isinstance(identifiers, (tuple, list))
            and len(identifiers) == batch_size
            and all(isinstance(item, str) and item for item in identifiers),
            "official identifiers differ",
        )
        images = images.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        for height, width in ready_sizes:
            _require(
                0 < height <= images.shape[-2] and 0 < width <= images.shape[-1],
                "original sample size exceeds padded batch",
            )

        for role in ROLES:
            for family in FAMILY_ORDER:
                key = f"{role}::{family}"
                logits = forward_candidate_logits(
                    self.prepared.runtimes[role][family], images
                )
                for index, ((height, width), identifier) in enumerate(
                    zip(ready_sizes, identifiers, strict=True)
                ):
                    sample_logits = logits[index : index + 1, :, :height, :width]
                    sample_target = targets[index : index + 1, :, :height, :width]
                    loss = F.binary_cross_entropy_with_logits(
                        sample_logits,
                        sample_target,
                        reduction="mean",
                    )
                    _require(
                        math.isfinite(float(loss.item())), f"{key} loss is non-finite"
                    )
                    probability = torch.sigmoid(sample_logits)[0, 0].cpu().numpy()
                    target = sample_target[0, 0].cpu().numpy()
                    self.accumulators[role][family].update(
                        probability=np.ascontiguousarray(
                            probability, dtype=np.float32
                        ),
                        target=np.ascontiguousarray(target, dtype=np.float32),
                        loss=float(loss.item()),
                        identifier=identifier,
                    )
        return {
            "sample_count": batch_size,
            "forward_counts": {
                f"{role}::{family}": batch_size
                for role in ROLES
                for family in FAMILY_ORDER
            },
        }

    def finalize(self) -> dict[str, object]:
        performance = {
            role: {
                family: self.accumulators[role][family].compute()
                for family in FAMILY_ORDER
            }
            for role in ROLES
        }
        flat_performance = [
            performance[role][family]
            for role in ROLES
            for family in FAMILY_ORDER
        ]
        sample_hashes = {
            value["sample_id_order_sha256"] for value in flat_performance
        }
        target_hashes = {value["target_sha256"] for value in flat_performance}
        count_bindings = {
            (
                value["sample_count"],
                value["target_count"],
                value["tiny_target_count"],
                value["valid_pixel_count"],
            )
            for value in flat_performance
        }
        _require(len(sample_hashes) == len(target_hashes) == len(count_bindings) == 1, "candidate metric contexts differ")
        metric_source_sha = source_lock_module.file_sha256(Path(metric_core.__file__).resolve())
        shared_context_sha = candidate_pool_module.canonical_sha256(
            {
                "schema": SCHEMA,
                "dataset": self.prepared.dataset,
                "role_order": list(ROLES),
                "source_lock_sha256": self.prepared.audit["source_lock_sha256"],
                "split_projection_sha256": self.prepared.audit["split_projection_sha256"],
                "joint_candidate_pool_sha256": self.prepared.audit[
                    "joint_candidate_pool_sha256"
                ],
                "metric_core_sha256": metric_source_sha,
                "metric_manifest": metric_core.MetricCoreManifest().as_dict(),
            }
        )
        bindings: dict[str, zero_margin_selector.EvaluationBinding] = {}
        selections: dict[str, dict[str, object]] = {}
        winners: dict[str, str] = {}
        winner_families: dict[str, str] = {}
        for role in ROLES:
            role_context_sha = candidate_pool_module.canonical_sha256(
                {
                    "shared_evaluation_context_sha256": shared_context_sha,
                    "role": role,
                    "candidate_pool_sha256": self.prepared.candidate_pools[role][
                        "candidate_pool_sha256"
                    ],
                }
            )
            binding = zero_margin_selector.EvaluationBinding(
                dataset=self.prepared.dataset,
                role=role,  # type: ignore[arg-type]
                evaluation_context_sha256=role_context_sha,
                sample_id_order_sha256=next(iter(sample_hashes)),
                target_sha256=next(iter(target_hashes)),
                metric_core_sha256=metric_source_sha,
            )
            bindings[role] = binding
            records = {
                family: zero_margin_selector.MetricRecord.from_mapping(
                    name=self.prepared.runtimes[role][family].name,
                    family=family,  # type: ignore[arg-type]
                    binding=binding,
                    value=performance[role][family],
                )
                for family in FAMILY_ORDER
            }
            selection = zero_margin_selector.selection_report(
                role,  # type: ignore[arg-type]
                original=records["Original"],
                current=records["Current"],
                candidates=tuple(records[family] for family in FAMILY_ORDER[2:]),
                operational_test_selected=self.operational_test_selected,
            )
            selections[role] = selection
            winners[role] = str(selection["winner"])
            winner_families[role] = str(selection["winner_family"])
        return {
            "schema": SCHEMA,
            "dataset": self.prepared.dataset,
            "role_order": list(ROLES),
            "all_performance": performance,
            "zero_margin_selection": selections,
            "winner": winners,
            "winner_family": winner_families,
            "operational_test_selected": self.operational_test_selected,
            "selection_is_optimistic": self.operational_test_selected,
            "performance_acceptance_margin": None,
            "metric_manifest": metric_core.MetricCoreManifest().as_dict(),
            "shared_evaluation_context_sha256": shared_context_sha,
            "evaluation_binding": {
                role: bindings[role].as_dict() for role in ROLES
            },
            "official_probability_or_logit_cache_written": False,
            "official_sweep_performed": False,
        }


def evaluate_official_once(
    *,
    run_dir: Path,
    dataset: str,
    source_lock: Mapping[str, object],
    split_projection: Mapping[str, object],
    candidate_pools: Mapping[str, Mapping[str, object]],
    candidate_factory: CandidateFactory,
    loader_factory: Callable[[], Iterable[Any]],
    device: torch.device,
    expected_gpu_uuid: str | None,
    operational_test_selected: bool,
    check_environment: bool = True,
    materialize_views: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Preflight, claim, and consume one official pass or replay its bundle."""

    ready_dataset = _require_dataset(dataset)
    _require(type(operational_test_selected) is bool, "operational_test_selected must be bool")
    _require(callable(loader_factory), "loader_factory must be callable")
    _require(
        isinstance(candidate_pools, Mapping) and tuple(candidate_pools) == ROLES,
        "candidate-pool role order differs",
    )
    joint_pool = official_once.build_joint_candidate_pool(candidate_pools)
    holder: dict[str, Any] = {}

    def preflight() -> Mapping[str, object]:
        prepared = prepare_evaluation(
            dataset=ready_dataset,
            source_lock=source_lock,
            split_projection=split_projection,
            candidate_pools=candidate_pools,
            candidate_factory=candidate_factory,
            device=device,
            expected_gpu_uuid=expected_gpu_uuid,
            check_environment=check_environment,
        )
        holder["session"] = _OnlineEvaluation(
            prepared,
            device=device,
            operational_test_selected=operational_test_selected,
        )
        return prepared.audit

    def consume_batch(batch: Any) -> Mapping[str, object]:
        session = holder.get("session")
        _require(isinstance(session, _OnlineEvaluation), "official session was not prepared")
        return session.consume_batch(batch)

    def finalize_metrics() -> Mapping[str, object]:
        session = holder.get("session")
        _require(isinstance(session, _OnlineEvaluation), "official session was not prepared")
        return session.finalize()

    return official_once.execute_official_joint_once(
        run_dir=Path(run_dir),
        joint_candidate_pool=joint_pool,
        preflight=preflight,
        loader_factory=loader_factory,
        consume_batch=consume_batch,
        finalize_metrics=finalize_metrics,
        materialize_views=materialize_views,
    )


def preflight_artifacts_only(
    *,
    dataset: str,
    role: str,
    source_lock_path: Path,
    split_projection_path: Path,
    candidate_pool_path: Path,
    check_environment: bool = True,
) -> dict[str, object]:
    """Launcher-safe artifact preflight; it never builds a dataset or loader."""

    locked = _load_json_object(Path(source_lock_path), name="source lock")
    projection = _load_json_object(Path(split_projection_path), name="split projection")
    pool = candidate_pool_module.load_candidate_pool(Path(candidate_pool_path))
    ready_lock, ready_projection, ready_pool = _validate_source_split_pool(
        dataset=dataset,
        role=role,
        source_lock=locked,
        split_projection=projection,
        candidate_pool=pool,
        check_environment=check_environment,
    )
    return {
        "schema": SCHEMA,
        "status": "launcher_artifact_preflight_complete",
        "dataset": dataset,
        "role": role,
        "source_lock_sha256": ready_lock["source_lock_sha256"],
        "split_projection_sha256": ready_projection["projection_sha256"],
        "candidate_pool_sha256": ready_pool["candidate_pool_sha256"],
        "candidate_count": ready_pool["candidate_count"],
        "official_test_accessed": False,
        "dataset_or_loader_constructed": False,
    }


def _configuration_for_baseline(
    family: str,
    *,
    dataset: str,
    role: str,
    state_sha256: str,
) -> str:
    return candidate_configuration_sha256(
        family=family,
        dataset=dataset,
        role=role,
        details={"state_sha256": state_sha256, "inference": "raw_final_logits"},
    )


def _runtime_from_baseline(
    record: Mapping[str, Any],
    *,
    dataset: str,
    role: str,
) -> CandidateRuntime:
    family = str(record["family"])
    if family == "Original":
        model, metadata = original_models.build_original_inference_model(dataset, role)
        binding = _require_mapping(metadata.get("original_checkpoint"), name="Original checkpoint binding")
    else:
        model, metadata = v4_models.build_frozen_current_reference_model(dataset, role, "stage1")
        binding = _require_mapping(metadata.get("parent_checkpoint"), name="Current checkpoint binding")
    _require(binding.get("path") == record["artifact_path"], f"{family} artifact path differs")
    _require(binding.get("sha256") == record["artifact_sha256"], f"{family} artifact SHA differs")
    _require(binding.get("state_sha256") == record["state_sha256"], f"{family} state SHA differs")
    configuration = _configuration_for_baseline(
        family,
        dataset=dataset,
        role=role,
        state_sha256=str(binding["state_sha256"]),
    )
    _require(configuration == record["configuration_sha256"], f"{family} configuration SHA differs")
    return CandidateRuntime(
        family=family,
        name=str(record["name"]),
        model=model,
        artifact_path=str(record["artifact_path"]),
        artifact_sha256=str(record["artifact_sha256"]),
        state_sha256=str(record["state_sha256"]),
        configuration_sha256=str(record["configuration_sha256"]),
        calibration=None,
        checkpoint_binding={**dict(binding), "dataset": dataset, "role": role, "official_test_accessed": False},
    )


def v3_artifact_manifest_sha256(artifact: Mapping[str, Any]) -> str:
    """Replay the freezer's V3-calibrated artifact manifest hash."""

    payload = dict(_require_mapping(artifact, name="V3 calibrated artifact"))
    payload.pop("state_dict", None)
    payload.pop("artifact_manifest_sha256", None)
    return candidate_pool_module.canonical_sha256(payload)


def validate_v3_candidate_envelope(
    artifact: Mapping[str, Any],
    *,
    dataset: str,
    role: str,
    pool: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently replay the calibrated V3 scope and provenance envelope."""

    payload = dict(_require_mapping(artifact, name="V3 calibrated artifact"))
    pool_binding = _require_mapping(pool, name="V3 candidate pool")
    _require(
        payload.get("schema") == V3_CALIBRATED_ARTIFACT_SCHEMA
        and payload.get("family") == "V3-calibrated",
        "V3 calibrated artifact identity differs",
    )
    _require(
        payload.get("dataset") == dataset and payload.get("role") == role,
        "V3 calibrated role binding differs",
    )
    _require(
        payload.get("selected_on") == "internal_validation",
        "V3 calibration was not frozen internally",
    )
    _require(
        payload.get("official_test_accessed") is False,
        "V3 calibration crossed official boundary",
    )
    _require(
        payload.get("performance_acceptance_margin") is None,
        "V3 calibration carries a performance acceptance margin",
    )
    _require(
        payload.get("fixed_probability_rule") == "strict_greater_than_0.5",
        "V3 calibration probability rule differs",
    )
    _require(
        _require_sha256(
            payload.get("source_lock_sha256"),
            name="V3 calibration source-lock SHA-256",
        )
        == _require_sha256(
            pool_binding.get("source_lock_sha256"),
            name="V3 pool source-lock SHA-256",
        ),
        "V3 calibration source-lock binding differs",
    )
    _require(
        _require_sha256(
            payload.get("split_projection_sha256"),
            name="V3 calibration split SHA-256",
        )
        == _require_sha256(
            pool_binding.get("split_projection_sha256"),
            name="V3 pool split SHA-256",
        ),
        "V3 calibration split binding differs",
    )
    state = _require_mapping(payload.get("state_dict"), name="V3 calibrated state")
    _require(
        bool(state)
        and all(
            type(name) is str and isinstance(value, torch.Tensor)
            for name, value in state.items()
        ),
        "V3 calibrated state differs",
    )
    state_sha = v4_models.state_semantic_sha256(state)  # type: ignore[arg-type]
    _require(
        payload.get("state_key_count") == len(state)
        and payload.get("state_sha256") == state_sha
        and payload.get("state_semantic_sha256") == state_sha,
        "V3 calibrated semantic state binding differs",
    )
    _require(
        all(
            isinstance(payload.get(name), Mapping)
            for name in ("sweep_binding", "cache_binding", "v3_candidate_binding")
        ),
        "V3 calibrated provenance bindings differ",
    )
    declared = _require_sha256(
        payload.get("artifact_manifest_sha256"),
        name="V3 artifact manifest SHA-256",
    )
    _require(
        declared == v3_artifact_manifest_sha256(payload),
        "V3 artifact manifest SHA-256 differs",
    )
    return payload


def _runtime_from_v3(
    record: Mapping[str, Any],
    *,
    dataset: str,
    role: str,
    pool: Mapping[str, Any],
) -> CandidateRuntime:
    artifact = validate_v3_candidate_envelope(
        _torch_payload(
            Path(str(record["artifact_path"])),
            name="V3 calibrated artifact",
        ),
        dataset=dataset,
        role=role,
        pool=pool,
    )
    state = _require_mapping(artifact.get("state_dict"), name="V3 calibrated state")
    state_sha = v4_models.state_semantic_sha256(state)  # type: ignore[arg-type]
    _require(state_sha == artifact.get("state_sha256") == record["state_sha256"], "V3 calibrated state SHA differs")
    config_payload = _require_mapping(artifact.get("calibration"), name="V3 calibration")
    _require(set(config_payload) == {"positive_scale", "negative_scale", "bias"}, "V3 calibration keys differ")
    config = residual_calibration.ResidualCalibration(
        positive_scale=float(config_payload["positive_scale"]),
        negative_scale=float(config_payload["negative_scale"]),
        bias=float(config_payload["bias"]),
    )
    configuration = candidate_configuration_sha256(
        family="V3-calibrated",
        dataset=dataset,
        role=role,
        details={"state_sha256": state_sha, "calibration": config.as_dict(), "selected_on": "internal_validation"},
    )
    _require(configuration == artifact.get("configuration_sha256") == record["configuration_sha256"], "V3 configuration SHA differs")
    if dataset == "NUAA-SIRST":
        model, metadata = nuaa_v3_models.build_inference_model_from_candidate_state(
            state,  # type: ignore[arg-type]
            parent_role=role,
        )
    else:
        model, metadata = cross_v3_models.build_inference_model_from_candidate_state(
            state,  # type: ignore[arg-type]
            dataset_name=dataset,
            parent_role=role,
        )
    _require(metadata.get("strict_load") is True, "V3 inference load was not strict")
    return CandidateRuntime(
        family="V3-calibrated",
        name=str(record["name"]),
        model=model,
        artifact_path=str(record["artifact_path"]),
        artifact_sha256=str(record["artifact_sha256"]),
        state_sha256=str(record["state_sha256"]),
        configuration_sha256=str(record["configuration_sha256"]),
        calibration=config,
        checkpoint_binding={
            "dataset": dataset,
            "role": role,
            "parent_role": role,
            "official_test_accessed": False,
        },
    )


def v4_candidate_manifest_sha256(checkpoint: Mapping[str, Any]) -> str:
    """Replay the trainer's manifest hash without trusting a pool freezer."""

    payload = dict(_require_mapping(checkpoint, name="V4 candidate checkpoint"))
    payload.pop("state_dict", None)
    payload.pop("candidate_manifest_sha256", None)
    return candidate_pool_module.canonical_sha256(payload)


def validate_v4_candidate_envelope(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject smoke/test-touched/thresholded V4 candidates independently."""

    payload = dict(_require_mapping(checkpoint, name="V4 candidate checkpoint"))
    _require(payload.get("smoke") is False, "V4 candidate is a smoke artifact")
    _require(
        payload.get("official_test_accessed") is False,
        "V4 candidate crossed the official-test boundary",
    )
    _require(
        payload.get("official_test_data_accessed") is False,
        "V4 candidate checkpoint data crossed the official-test boundary",
    )
    _require(
        payload.get("performance_acceptance_margin") is None,
        "V4 candidate carries a performance acceptance margin",
    )
    declared = _require_sha256(
        payload.get("candidate_manifest_sha256"),
        name="V4 candidate manifest SHA-256",
    )
    _require(
        declared == v4_candidate_manifest_sha256(payload),
        "V4 candidate manifest SHA-256 differs",
    )
    return payload


def _runtime_from_v4(
    record: Mapping[str, Any],
    *,
    dataset: str,
    role: str,
    pool: Mapping[str, Any],
) -> CandidateRuntime:
    family = str(record["family"])
    stage = "stage1" if family == "V4-Stage1" else "stage2"
    artifact = _torch_payload(Path(str(record["artifact_path"])), name=f"{family} checkpoint")
    checkpoint_value = artifact.get("candidate_checkpoint", artifact)
    checkpoint = validate_v4_candidate_envelope(
        _require_mapping(checkpoint_value, name=f"{family} candidate checkpoint")
    )
    _require(checkpoint.get("dataset") == dataset, f"{family} dataset differs")
    _require(checkpoint.get("role") == role and checkpoint.get("parent_role") == role, f"{family} role differs")
    _require(checkpoint.get("stage") == stage, f"{family} stage differs")
    _require(checkpoint.get("state_sha256") == record["state_sha256"], f"{family} state SHA differs")
    _require(checkpoint.get("source_sha256") == pool["source_lock_sha256"], f"{family} source binding differs")
    _require(checkpoint.get("split_sha256") == pool["split_projection_sha256"], f"{family} split binding differs")
    details = {
        "stage": stage,
        "source_sha256": checkpoint.get("source_sha256"),
        "split_sha256": checkpoint.get("split_sha256"),
        "atlas_sha256": checkpoint.get("atlas_sha256"),
        "initialization_sha256": checkpoint.get("initialization_sha256"),
        "state_sha256": checkpoint.get("state_sha256"),
    }
    configuration = candidate_configuration_sha256(
        family=family,
        dataset=dataset,
        role=role,
        details=details,
    )
    _require(configuration == record["configuration_sha256"], f"{family} configuration SHA differs")
    model, metadata = v4_models.build_candidate_inference_model(
        checkpoint,
        dataset_name=dataset,
        role=role,
        stage=stage,
        expected_source_sha256=str(checkpoint["source_sha256"]),
        expected_split_sha256=str(checkpoint["split_sha256"]),
        expected_atlas_sha256=str(checkpoint["atlas_sha256"]),
        expected_initialization_sha256=str(checkpoint["initialization_sha256"]),
        expected_state_sha256=str(record["state_sha256"]),
    )
    _require(metadata.get("strict_complete_payload") is True, f"{family} inference load was not strict")
    return CandidateRuntime(
        family=family,
        name=str(record["name"]),
        model=model,
        artifact_path=str(record["artifact_path"]),
        artifact_sha256=str(record["artifact_sha256"]),
        state_sha256=str(record["state_sha256"]),
        configuration_sha256=str(record["configuration_sha256"]),
        calibration=None,
        checkpoint_binding={
            "dataset": dataset,
            "role": role,
            "parent_role": role,
            "stage": stage,
            "official_test_accessed": False,
        },
    )


def default_candidate_factory(
    *,
    candidate_pool: Mapping[str, object],
    dataset: str,
    role: str,
    device: torch.device,
) -> Mapping[str, CandidateRuntime]:
    """Build the five frozen families from one validated candidate pool."""

    del device
    pool = candidate_pool_module.validate_candidate_pool(candidate_pool)
    records = {
        str(record["family"]): _require_mapping(record, name="candidate record")
        for record in pool["candidates"]  # type: ignore[index]
    }
    return {
        "Original": _runtime_from_baseline(records["Original"], dataset=dataset, role=role),
        "Current": _runtime_from_baseline(records["Current"], dataset=dataset, role=role),
        "V3-calibrated": _runtime_from_v3(
            records["V3-calibrated"],
            dataset=dataset,
            role=role,
            pool=pool,
        ),
        "V4-Stage1": _runtime_from_v4(records["V4-Stage1"], dataset=dataset, role=role, pool=pool),
        "V4-Stage2": _runtime_from_v4(records["V4-Stage2"], dataset=dataset, role=role, pool=pool),
    }


class OfficialTestDataset(Dataset):
    """Official dataset whose constructor is legal only inside loader_factory."""

    def __init__(self, data_root: Path, dataset: str) -> None:
        super().__init__()
        self.dataset = _require_dataset(dataset)
        root = Path(data_root)
        _require(root.is_dir() and not root.is_symlink(), "data root is missing or unsafe")
        self.data_root = root.resolve(strict=True)
        self.sample_ids = data_protocol.load_index(self.data_root, self.dataset, "test")
        expected = data_protocol.EXPECTED_SPLITS[self.dataset]["test"]
        _require(len(self.sample_ids) == expected["count"], "official test count differs")
        _require(
            data_protocol.ordered_ids_sha256(self.sample_ids) == expected["ordered_ids_sha256"],
            "official test ordered-ID SHA differs",
        )
        self._known_ids = frozenset(self.sample_ids)
        self.normalization = data_protocol.get_legacy_normalization(self.dataset)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], str]:
        identifier = self.sample_ids[index]
        sample = data_protocol.resolve_sample(
            self.data_root,
            self.dataset,
            identifier,
            split="test",
            known_ids=self._known_ids,
        )
        with Image.open(sample.image_path) as handle:
            image = np.asarray(handle.convert("I"), dtype=np.float32)
        with Image.open(sample.mask_path) as handle:
            target = np.asarray(handle, dtype=np.float32)
        if target.ndim > 2:
            target = target[:, :, 0]
        _require(image.ndim == target.ndim == 2 and image.shape == target.shape, f"official sample dimensions differ: {identifier}")
        _require(bool(np.isfinite(image).all()) and bool(np.isfinite(target).all()), f"official sample is non-finite: {identifier}")
        height, width = image.shape
        image = (image - np.float32(self.normalization["mean"])) / np.float32(self.normalization["std"])
        target = target / np.float32(255.0)
        padded_height = ((height + 31) // 32) * 32
        padded_width = ((width + 31) // 32) * 32
        image = np.pad(image, ((0, padded_height - height), (0, padded_width - width)))
        target = np.pad(target, ((0, padded_height - height), (0, padded_width - width)))
        return (
            torch.from_numpy(np.ascontiguousarray(image[None], dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(target[None], dtype=np.float32)),
            (height, width),
            identifier,
        )


def make_official_loader_factory(
    *,
    data_root: Path,
    dataset: str,
    device: torch.device,
    workers: int,
) -> Callable[[], DataLoader]:
    """Return a closure; creating it does not read an index or build a loader."""

    ready_dataset = _require_dataset(dataset)
    _require(type(workers) is int and workers >= 0, "workers must be non-negative")
    root = Path(data_root)

    def factory() -> DataLoader:
        official_dataset = OfficialTestDataset(root, ready_dataset)
        return DataLoader(
            official_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

    return factory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--split-projection", type=Path, required=True)
    parser.add_argument("--best-miou-candidate-pool", type=Path, required=True)
    parser.add_argument("--best-pd-candidate-pool", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--workers", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--operational-test-selected", dest="operational_test_selected", action="store_true")
    selection.add_argument("--report-only", dest="operational_test_selected", action="store_false")
    parser.set_defaults(operational_test_selected=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = parse_args(argv)
    training_core.configure_determinism(seed=training_core.TRAINING_SEED)
    device = torch.device(arguments.device)
    validate_evaluation_device(
        dataset=arguments.dataset,
        device=device,
        expected_gpu_uuid=arguments.expected_gpu_uuid,
    )
    locked = _load_json_object(arguments.source_lock, name="source lock")
    projection = _load_json_object(arguments.split_projection, name="split projection")
    pools = {
        "best_miou": candidate_pool_module.load_candidate_pool(
            arguments.best_miou_candidate_pool
        ),
        "best_pd": candidate_pool_module.load_candidate_pool(
            arguments.best_pd_candidate_pool
        ),
    }
    loader_factory = make_official_loader_factory(
        data_root=arguments.data_root,
        dataset=arguments.dataset,
        device=device,
        workers=arguments.workers,
    )
    bundle = evaluate_official_once(
        run_dir=arguments.run_dir,
        dataset=arguments.dataset,
        source_lock=locked,
        split_projection=projection,
        candidate_pools=pools,
        candidate_factory=default_candidate_factory,
        loader_factory=loader_factory,
        device=device,
        expected_gpu_uuid=arguments.expected_gpu_uuid,
        operational_test_selected=arguments.operational_test_selected,
    )
    print(json.dumps(bundle, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()


__all__ = [
    "CandidateRuntime",
    "DATASETS",
    "FAMILY_ORDER",
    "FORMAL_GPU_UUIDS",
    "OfficialTestDataset",
    "PBDRV4EvaluationError",
    "REQUIRED_SOURCE_LOCK_PATHS",
    "ROLES",
    "SCHEMA",
    "SYNTHETIC_PREFLIGHT_SIZE",
    "V3_CALIBRATED_ARTIFACT_SCHEMA",
    "candidate_configuration_sha256",
    "default_candidate_factory",
    "evaluate_official_once",
    "forward_candidate_logits",
    "make_official_loader_factory",
    "parse_args",
    "preflight_artifacts_only",
    "prepare_evaluation",
    "v3_artifact_manifest_sha256",
    "validate_v3_candidate_envelope",
    "v4_candidate_manifest_sha256",
    "validate_evaluation_device",
    "validate_v4_candidate_envelope",
]
