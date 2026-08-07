#!/usr/bin/env python3
"""Prepare leakage-closed PBDR-V4 internal caches and component atlas.

One invocation is bound to one dataset and one checkpoint role.  It validates
the frozen V3 split projection, strictly loads the completed PBDR-V3
Candidate, its Current parent, and the role-matched Original, then visits the
two disjoint projections of official train exactly once.  Every sample gets
one forward through each model.  The resulting development-train and
internal-validation raw-logit caches are immutable, and the component atlas
is derived from the cached development Current logits rather than a repeated
model pass.

Existing artifacts are never replaced.  Replay mode reloads live authorities
and models and fully validates existing cache/atlas bytes without inference or
writes.  No function in this module reads or reconstructs an official-test
index.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from experiments import build_pbdr_v4_component_atlas as atlas_builder
from experiments import pbdr_v4_atlas_dataset as atlas_dataset
from experiments import pbdr_v4_internal_cache as cache_io
from experiments import pbdr_v4_internal_dataset as internal_dataset
from experiments import pbdr_v4_metric_core as metric_core
from experiments import pbdr_v4_models_seed42_v1 as current_registry
from experiments import pbdr_v4_original_models as original_registry
from experiments import pbdr_v4_source_lock as source_lock_io
from experiments import pbdr_v4_split_authority as split_authority


SCHEMA = "sctransnet_prepare_pbdr_v4_internal_artifacts/v1"
MANIFEST_NAME = "preparation_manifest.json"
DEVELOPMENT_CACHE_NAME = "development_train_cache"
VALIDATION_CACHE_NAME = "internal_validation_cache"
ATLAS_NAME = "component_atlas"
DATASETS = split_authority.DATASETS
ROLES = cache_io.ROLES
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REPO_ROOT = Path(__file__).resolve().parents[1]
V3_RUN_DIRECTORIES: Mapping[str, Mapping[str, Path]] = {
    "NUAA-SIRST": {
        role: REPO_ROOT
        / "results/nuaa_pbdr_v3_stage1_v1/formal"
        / role
        / "core"
        for role in ROLES
    },
    **{
        dataset: {
            role: REPO_ROOT
            / "results/two_dataset_pbdr_v3_stage1_v1/runs"
            / dataset
            / "formal"
            / role
            / "core"
            for role in ROLES
        }
        for dataset in ("NUDT-SIRST", "IRSTD-1K")
    },
}


class PBDRV4InternalPreparationError(ValueError):
    """A source, split, checkpoint, inference, or artifact binding differs."""


@dataclass(frozen=True, slots=True)
class ProjectionIds:
    official_train: tuple[str, ...]
    development_train: tuple[str, ...]
    internal_validation: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "official_train",
            "development_train",
            "internal_validation",
        ):
            value = getattr(self, name)
            _require(bool(value), f"{name} cannot be empty")
            _require(
                all(type(identifier) is str and identifier for identifier in value),
                f"{name} contains an invalid identifier",
            )
            _require(len(value) == len(set(value)), f"{name} contains duplicates")
        _require(
            not set(self.development_train) & set(self.internal_validation),
            "internal projections overlap",
        )
        _require(
            set(self.development_train) | set(self.internal_validation)
            == set(self.official_train),
            "internal projections do not partition official train",
        )


@dataclass(frozen=True, slots=True)
class SourceContext:
    path: Path
    bytes: int
    file_sha256: str
    source_lock_sha256: str
    metric_source_sha256: str
    matcher_source_sha256: str


@dataclass(slots=True)
class StrictModelBundle:
    v3_candidate: nn.Module
    current: nn.Module
    original: nn.Module
    v3_checkpoint: cache_io.CheckpointBinding
    current_checkpoint: cache_io.CheckpointBinding
    original_checkpoint: cache_io.CheckpointBinding
    data_root: Path
    candidate_split_sha256: str
    attestations: Mapping[str, Any]

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "v3_candidate": self.v3_checkpoint.as_dict(),
            "current": self.current_checkpoint.as_dict(),
            "original": self.original_checkpoint.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class PreparedInternalArtifacts:
    root: Path
    development_cache: Path
    validation_cache: Path
    atlas: Path
    manifest_path: Path
    manifest_sha256: str
    replayed: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV4InternalPreparationError(message)


def _require_dataset(dataset_name: str) -> str:
    _require(
        type(dataset_name) is str and dataset_name in DATASETS,
        f"dataset_name must be one of {DATASETS}",
    )
    return dataset_name


def _require_role(role: str) -> str:
    _require(
        type(role) is str and role in ROLES,
        f"role must be one of {ROLES}",
    )
    return role


def _sha256(value: Any, *, name: str) -> str:
    _require(
        type(value) is str and _SHA256_RE.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256 digest",
    )
    return value


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    candidate = Path(path)
    _require(
        not candidate.is_symlink() and candidate.is_file(),
        f"{name} must be a regular non-symlink file",
    )
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV4InternalPreparationError(f"cannot read {name}: {error}") from error
    _require(isinstance(value, dict), f"{name} must contain one object")
    return value


def configure_formal_inference() -> None:
    """Apply the deterministic controls bound by the V4 source lock."""

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _module_source_path(module: Any, *, name: str) -> Path:
    value = getattr(module, "__file__", None)
    _require(type(value) is str and bool(value), f"{name} source path is unavailable")
    path = Path(value)
    _require(
        not path.is_symlink() and path.is_file(),
        f"{name} source must be a regular non-symlink file",
    )
    return path.resolve(strict=True)


def load_source_context(path: Path) -> SourceContext:
    """Validate one shared lock and its live metric/matcher source records."""

    lock_path = Path(path)
    payload = source_lock_io.load_source_lock(lock_path, check_environment=True)
    lock_sha = _sha256(
        payload.get("source_lock_sha256"),
        name="source_lock_sha256",
    )
    sources = payload.get("sources")
    _require(isinstance(sources, Mapping), "source lock lacks source records")
    metric_path = _module_source_path(metric_core, name="metric core")
    metric_sha = source_lock_io.file_sha256(metric_path)
    matcher_sha = atlas_dataset.matcher_source_sha256()
    expected = {
        "experiments/pbdr_v4_metric_core.py": metric_sha,
        "experiments/component_matching_v2.py": matcher_sha,
    }
    for relative, observed_sha in expected.items():
        record = sources.get(relative)
        _require(isinstance(record, Mapping), f"source lock lacks {relative}")
        _require(
            record.get("sha256") == observed_sha,
            f"source lock {relative} SHA differs",
        )
    resolved = lock_path.resolve(strict=True)
    return SourceContext(
        path=resolved,
        bytes=resolved.stat().st_size,
        file_sha256=source_lock_io.file_sha256(resolved),
        source_lock_sha256=lock_sha,
        metric_source_sha256=metric_sha,
        matcher_source_sha256=matcher_sha,
    )


def load_projection_ids(
    dataset_name: str,
    split_projection: Mapping[str, Any],
) -> ProjectionIds:
    """Read IDs only from the verified V3 official-train split manifest."""

    dataset = _require_dataset(dataset_name)
    _require(
        split_projection.get("schema") == split_authority.SCHEMA,
        "split projection schema differs",
    )
    _require(
        split_projection.get("official_test_accessed") is False
        and split_projection.get("model_selection_only") is True,
        "split projection scope differs",
    )
    declared = _sha256(
        split_projection.get("projection_sha256"),
        name="split projection SHA-256",
    )
    unsigned = dict(split_projection)
    del unsigned["projection_sha256"]
    _require(
        split_authority.canonical_sha256(unsigned) == declared,
        "split projection SHA-256 does not replay",
    )
    datasets = split_projection.get("datasets")
    _require(isinstance(datasets, Mapping), "split projection lacks datasets")
    projected = datasets.get(dataset)
    _require(isinstance(projected, Mapping), "dataset is absent from split projection")
    source_path = Path(str(projected.get("source_path")))
    expected_path = split_authority.source_manifest_path(dataset)
    _require(
        source_path.resolve(strict=True) == expected_path,
        "projected source manifest path differs",
    )
    _require(
        source_path.stat().st_size == projected.get("source_bytes")
        and split_authority.file_sha256(source_path)
        == projected.get("source_file_sha256"),
        "projected source manifest bytes differ",
    )
    payload = _read_json_object(source_path, name="official-train split manifest")
    validated = split_authority.validate_split_payload(dataset, payload)
    _require(
        validated["canonical_split_sha256"]
        == projected.get("canonical_split_sha256")
        and validated["counts"] == projected.get("counts")
        and validated["ordered_id_sha256"] == projected.get("ordered_id_sha256"),
        "projected split identity differs from source manifest",
    )
    identifiers = ProjectionIds(
        official_train=tuple(payload["official_train_ids"]),
        development_train=tuple(payload["development_train_ids"]),
        internal_validation=tuple(payload["internal_validation_ids"]),
    )
    _require(
        split_authority.ordered_ids_sha256(identifiers.official_train)
        == projected["ordered_id_sha256"]["official_train_ids"],
        "official-train ordered-ID SHA differs",
    )
    return identifiers


def projection_and_ids(dataset_name: str) -> tuple[dict[str, Any], ProjectionIds]:
    projection = split_authority.build_projection()
    return projection, load_projection_ids(dataset_name, projection)


def _checkpoint_binding(
    record: Mapping[str, Any],
    *,
    state_sha256: str | None = None,
) -> cache_io.CheckpointBinding:
    return cache_io.CheckpointBinding(
        path=str(Path(str(record["path"])).resolve(strict=True)),
        bytes=int(record["bytes"]),
        file_sha256=_sha256(record["sha256"], name="checkpoint file SHA-256"),
        state_sha256=_sha256(
            record["state_sha256"] if state_sha256 is None else state_sha256,
            name="checkpoint state SHA-256",
        ),
    )


def _freeze_eval_model(model: nn.Module, *, name: str) -> nn.Module:
    _require(isinstance(model, nn.Module), f"{name} is not a Module")
    model.eval()
    if hasattr(model, "mode"):
        model.mode = "test"
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _require(not model.training, f"{name} is not in eval mode")
    _require(
        not any(parameter.requires_grad for parameter in model.parameters()),
        f"{name} exposes trainable parameters",
    )
    return model


def load_strict_models(
    dataset_name: str,
    role: str,
) -> StrictModelBundle:
    """Load completed V3, Current, and Original through audited registries."""

    dataset = _require_dataset(dataset_name)
    ready_role = _require_role(role)
    run_dir = V3_RUN_DIRECTORIES[dataset][ready_role]
    if dataset == "NUAA-SIRST":
        from experiments import evaluate_nuaa_pbdr_v3_stage1_v1 as evaluator
        from experiments import three_dataset_pbdr_v3_models_seed42_v1 as v3_models

        run = evaluator.validate_completed_run(run_dir)
        _require(run.parent_role == ready_role, "completed V3 role differs")
        _require(run.recipe == "core", "completed NUAA V3 recipe is not core")
        v3_model, v3_metadata = v3_models.build_inference_model_from_candidate_state(
            run.candidate_state,
            parent_role=ready_role,
            parent_checkpoint=Path(str(run.parent_checkpoint["path"])),
        )
        v3_state_sha = v3_models.tensor_mapping_sha256(run.candidate_state)
    else:
        from experiments import evaluate_two_dataset_pbdr_v3_stage1_v1 as evaluator
        from experiments import two_dataset_pbdr_v3_models_seed42_v1 as v3_models

        run = evaluator.validate_completed_run(dataset, run_dir)
        _require(run.parent_role == ready_role, "completed V3 role differs")
        v3_model, v3_metadata = v3_models.build_inference_model_from_candidate_state(
            run.candidate_state,
            dataset_name=dataset,
            parent_role=ready_role,
        )
        v3_state_sha = v3_models.tensor_mapping_sha256(run.candidate_state)
    _require(v3_metadata.get("strict_load") is True, "V3 load is not strict")
    _require(
        v3_metadata.get("base_bitwise_equal_to_parent") is True,
        "V3 inference base is not Current",
    )

    _, current_state, current_record = current_registry.load_current_checkpoint(
        dataset,
        ready_role,
    )
    current_model, current_metadata = (
        current_registry.build_frozen_current_reference_model(
            dataset,
            ready_role,
            "stage1",
        )
    )
    _require(
        current_metadata.get("base_logits_are_current") is True,
        "Current reference logits are not checkpoint-bound",
    )
    candidate_base = {
        name: tensor
        for name, tensor in run.candidate_state.items()
        if not name.startswith("pbdr_v3.")
    }
    _require(set(candidate_base) == set(current_state), "V3/Current base keys differ")
    changed = [
        name
        for name, expected in current_state.items()
        if not torch.equal(candidate_base[name].detach().cpu(), expected.detach().cpu())
    ]
    _require(not changed, f"V3 base differs bitwise from Current: {changed[:5]}")

    _, _, original_record = original_registry.load_original_checkpoint(
        dataset,
        ready_role,
    )
    original_model, original_metadata = original_registry.build_original_inference_model(
        dataset,
        ready_role,
    )
    _require(original_metadata.get("strict_load") is True, "Original load is not strict")

    split_sha = _sha256(
        run.split_manifest.get("split_sha256"),
        name="candidate split SHA-256",
    )
    data_root = Path(str(run.protocol.get("data_root")))
    _require(
        data_root.is_absolute() and data_root.is_dir() and not data_root.is_symlink(),
        "completed V3 data-root binding differs",
    )
    v3_record = {
        "path": str(run.candidate_path),
        "bytes": run.candidate_path.stat().st_size,
        "sha256": run.candidate_sha256,
        "state_sha256": v3_state_sha,
    }
    return StrictModelBundle(
        v3_candidate=_freeze_eval_model(v3_model, name="V3 Candidate"),
        current=_freeze_eval_model(current_model, name="Current"),
        original=_freeze_eval_model(original_model, name="Original"),
        v3_checkpoint=_checkpoint_binding(v3_record),
        current_checkpoint=_checkpoint_binding(current_record),
        original_checkpoint=_checkpoint_binding(original_record),
        data_root=data_root.resolve(strict=True),
        candidate_split_sha256=split_sha,
        attestations={
            "v3_strict_load": True,
            "current_strict_load": True,
            "original_strict_load": True,
            "v3_base_bitwise_current_state": True,
            "current_base_logits_from_current": True,
            "official_test_accessed": False,
        },
    )


def build_partition_dataset(
    *,
    dataset_name: str,
    data_root: Path,
    projection: Mapping[str, Any],
    identifiers: ProjectionIds,
    partition: str,
) -> internal_dataset.PBDRV4InternalInferenceDataset:
    projected = projection["datasets"][dataset_name]
    if partition == "development_train":
        selected = identifiers.development_train
        scope = "development_train_ids"
    elif partition == "internal_validation":
        selected = identifiers.internal_validation
        scope = "internal_validation_ids"
    else:
        raise PBDRV4InternalPreparationError("unsupported internal partition")
    return internal_dataset.PBDRV4InternalInferenceDataset(
        selected_ids=selected,
        known_official_train_ids=identifiers.official_train,
        manifest_scope=scope,
        selected_ids_ordered_sha256=projected["ordered_id_sha256"][scope],
        official_train_count=projected["counts"]["official_train"],
        official_train_ordered_ids_sha256=projected["ordered_id_sha256"][
            "official_train_ids"
        ],
        dataset_name=dataset_name,
        data_root=data_root,
    )


def _bchw(value: Any, *, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} is not a tensor")
    _require(
        value.dtype == torch.float32
        and value.ndim == 4
        and value.shape[0] == value.shape[1] == 1,
        f"{name} must be FP32 1x1xHxW",
    )
    _require(bool(torch.isfinite(value).all()), f"{name} is non-finite")
    return value


def _original_raw_logits(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Capture Original raw logits at ``outc`` and verify its probability API."""

    outc = getattr(model, "outc", None)
    _require(isinstance(outc, nn.Module), "Original model lacks outc")
    captured: list[torch.Tensor] = []

    def capture(_module: nn.Module, _inputs: Any, output: Any) -> None:
        captured.append(_bchw(output, name="Original outc logits"))

    handle = outc.register_forward_hook(capture)
    try:
        probability = model(image)
    finally:
        handle.remove()
    _require(len(captured) == 1, "Original outc hook did not fire exactly once")
    probability_tensor = _bchw(probability, name="Original probability")
    raw = captured[0]
    _require(
        torch.equal(torch.sigmoid(raw), probability_tensor),
        "Original test forward is not sigmoid(outc raw logits)",
    )
    return raw


def _crop_numpy(
    value: torch.Tensor,
    *,
    height: int,
    width: int,
    name: str,
) -> np.ndarray:
    ready = _bchw(value, name=name)
    _require(
        ready.shape[-2] >= height and ready.shape[-1] >= width,
        f"{name} is smaller than original sample size",
    )
    return np.ascontiguousarray(
        ready.detach().cpu()[0, 0, :height, :width].numpy(),
        dtype=np.float32,
    )


def write_partition_cache(
    destination: Path,
    *,
    dataset_name: str,
    role: str,
    partition: str,
    dataset: Any,
    split_projection: Mapping[str, Any],
    models: StrictModelBundle,
    device: torch.device,
    source: SourceContext,
) -> Path:
    """Infer every projected sample once and commit one raw-logit cache."""

    expected_ids = tuple(getattr(dataset, "sample_ids", ()))
    _require(bool(expected_ids), "partition dataset has no ordered sample IDs")
    normalization = getattr(dataset, "normalization", None)
    _require(isinstance(normalization, Mapping), "partition normalization is missing")
    writer = cache_io.InternalRawLogitCacheWriter(
        destination,
        dataset_name=dataset_name,
        parent_role=role,
        partition=partition,
        split_projection=split_projection,
        ordered_sample_ids=expected_ids,
        v3_checkpoint=models.v3_checkpoint,
        current_checkpoint=models.current_checkpoint,
        original_checkpoint=models.original_checkpoint,
        normalization=normalization,
        metric_core_sha256=source.metric_source_sha256,
        source_lock_sha256=source.source_lock_sha256,
    )
    try:
        with torch.no_grad():
            for index, expected_id in enumerate(expected_ids):
                image, target, original_size, sample_id = dataset[index]
                _require(sample_id == expected_id, "dataset sample order differs")
                _require(
                    isinstance(original_size, (tuple, list))
                    and len(original_size) == 2,
                    "sample original size is invalid",
                )
                height, width = (int(original_size[0]), int(original_size[1]))
                _require(
                    isinstance(image, torch.Tensor)
                    and isinstance(target, torch.Tensor)
                    and image.dtype == target.dtype == torch.float32
                    and image.ndim == target.ndim == 3
                    and image.shape[0] == target.shape[0] == 1,
                    "internal dataset tensor contract differs",
                )
                _require(
                    image.shape == target.shape,
                    "internal image/target padded shapes differ",
                )
                batch = image.unsqueeze(0).to(device=device, dtype=torch.float32)
                _, v3_aux = models.v3_candidate.forward_for_pbdr_v3_training(batch)
                _, current_aux = models.current.forward_for_pbdr_v4_training(batch)
                base = _bchw(v3_aux.base_logits, name="V3 base logits")
                delta = _bchw(v3_aux.routing.delta_logits, name="V3 delta logits")
                routed = _bchw(v3_aux.routed_logits, name="V3 routed logits")
                current = _bchw(
                    current_aux.candidate_base_logits,
                    name="Current raw logits",
                )
                _require(
                    torch.equal(base, current),
                    f"V3 base is not bitwise Current for {sample_id}",
                )
                _require(
                    torch.equal(torch.add(base, delta), routed),
                    f"V3 routed logits are not exact base plus delta for {sample_id}",
                )
                original = _original_raw_logits(models.original, batch)
                arrays = {
                    "base_logits": _crop_numpy(
                        base, height=height, width=width, name="base_logits"
                    ),
                    "delta_logits": _crop_numpy(
                        delta, height=height, width=width, name="delta_logits"
                    ),
                    "routed_logits": _crop_numpy(
                        routed, height=height, width=width, name="routed_logits"
                    ),
                    "current_logits": _crop_numpy(
                        current, height=height, width=width, name="current_logits"
                    ),
                    "original_logits": _crop_numpy(
                        original, height=height, width=width, name="original_logits"
                    ),
                    "target": np.ascontiguousarray(
                        target[0, :height, :width].detach().cpu().numpy(),
                        dtype=np.float32,
                    ),
                }
                writer.append_sample(
                    sample_id=sample_id,
                    height=height,
                    width=width,
                    **arrays,
                )
        return writer.finalize()
    except Exception:
        writer.abort()
        raise


def _sigmoid_float32(value: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.array(value, dtype=np.float32, order="C", copy=True))
    return np.ascontiguousarray(torch.sigmoid(tensor).numpy(), dtype=np.float32)


def _build_atlas_from_development_cache(
    destination: Path,
    *,
    dataset_name: str,
    role: str,
    development_cache: cache_io.ValidatedInternalRawLogitCache,
    projection: Mapping[str, Any],
    identifiers: ProjectionIds,
    models: StrictModelBundle,
    source: SourceContext,
) -> atlas_builder.AtlasBuildResult:
    frozen = {
        sample.sample_id: atlas_builder.FrozenCurrentSample(
            probability=_sigmoid_float32(sample.arrays["current_logits"]),
            target=np.ascontiguousarray(sample.arrays["target"], dtype=np.float32),
        )
        for sample in development_cache.samples
    }
    projected = projection["datasets"][dataset_name]
    return atlas_builder.build_pbdr_v4_component_atlas(
        dataset_name=dataset_name,
        role=role,
        development_train_ids=identifiers.development_train,
        frozen_samples=frozen,
        parent_checkpoint_sha256=models.current_checkpoint.file_sha256,
        parent_state_sha256=models.current_checkpoint.state_sha256,
        split_projection_sha256=projection["projection_sha256"],
        official_train_ids_sha256=projected["ordered_id_sha256"][
            "official_train_ids"
        ],
        metric_source_sha256=source.metric_source_sha256,
        matcher_source_sha256=source.matcher_source_sha256,
        source_lock_sha256=source.source_lock_sha256,
        output_root=destination,
    )


def _source_payload(source: SourceContext) -> dict[str, Any]:
    return {
        "path": str(source.path),
        "bytes": source.bytes,
        "file_sha256": source.file_sha256,
        "source_lock_sha256": source.source_lock_sha256,
        "metric_source_sha256": source.metric_source_sha256,
        "matcher_source_sha256": source.matcher_source_sha256,
    }


def _artifact_summary(
    *,
    dataset_name: str,
    role: str,
    projection: Mapping[str, Any],
    identifiers: ProjectionIds,
    models: StrictModelBundle,
    source: SourceContext,
    development: cache_io.ValidatedInternalRawLogitCache,
    validation: cache_io.ValidatedInternalRawLogitCache,
    atlas_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    dev_identity = development.manifest["identity"]
    val_identity = validation.manifest["identity"]
    normalization = dict(dev_identity["normalization"])
    _require(
        normalization == dict(val_identity["normalization"]),
        "cache normalizations differ",
    )
    checkpoints = models.checkpoint_payload()
    for identity, partition in (
        (dev_identity, "development_train"),
        (val_identity, "internal_validation"),
    ):
        _require(identity["partition"] == partition, "cache partition differs")
        _require(identity["checkpoints"] == checkpoints, "cache checkpoints differ")
        _require(
            identity["source_lock_sha256"] == source.source_lock_sha256
            and identity["metric_core_sha256"] == source.metric_source_sha256,
            "cache source binding differs",
        )
    _require(
        atlas_manifest["source_lock_sha256"] == source.source_lock_sha256
        and atlas_manifest["metric_source_sha256"]
        == source.metric_source_sha256
        and atlas_manifest["matcher_source_sha256"]
        == source.matcher_source_sha256,
        "atlas source binding differs",
    )
    _require(
        atlas_manifest["parent_checkpoint_sha256"]
        == models.current_checkpoint.file_sha256
        and atlas_manifest["parent_state_sha256"]
        == models.current_checkpoint.state_sha256,
        "atlas Current binding differs",
    )
    projected = projection["datasets"][dataset_name]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_internal_artifacts",
        "dataset": dataset_name,
        "role": role,
        "scope": "official_train_projection_only",
        "split": {
            "projection_sha256": projection["projection_sha256"],
            "canonical_split_sha256": projected["canonical_split_sha256"],
            "official_train_count": len(identifiers.official_train),
            "official_train_ids_sha256": projected["ordered_id_sha256"][
                "official_train_ids"
            ],
            "development_train_count": len(identifiers.development_train),
            "development_train_ids_sha256": projected["ordered_id_sha256"][
                "development_train_ids"
            ],
            "internal_validation_count": len(identifiers.internal_validation),
            "internal_validation_ids_sha256": projected["ordered_id_sha256"][
                "internal_validation_ids"
            ],
            "candidate_split_sha256": models.candidate_split_sha256,
        },
        "source": _source_payload(source),
        "checkpoints": checkpoints,
        "normalization": normalization,
        "model_attestations": dict(models.attestations),
        "inference_contract": {
            "passes_over_official_train": 1,
            "forwards_per_sample": {
                "v3_candidate": 1,
                "current": 1,
                "original": 1,
            },
            "original_raw_logits": "outc_forward_hook_verified_against_probability",
            "v3_base_bitwise_current_for_every_sample": True,
            "atlas_source": "development_cache_current_probability",
        },
        "artifacts": {
            "development_cache": {
                "relative_path": DEVELOPMENT_CACHE_NAME,
                "sample_count": len(development.samples),
                "manifest_sha256": development.manifest["manifest_sha256"],
                "identity_sha256": dev_identity["identity_sha256"],
            },
            "internal_validation_cache": {
                "relative_path": VALIDATION_CACHE_NAME,
                "sample_count": len(validation.samples),
                "manifest_sha256": validation.manifest["manifest_sha256"],
                "identity_sha256": val_identity["identity_sha256"],
            },
            "component_atlas": {
                "relative_path": ATLAS_NAME,
                "sample_count": len(atlas_manifest["samples"]),
                "manifest_sha256": atlas_manifest["manifest_sha256"],
            },
        },
        "official_test_accessed": False,
        "test_index_parsed": False,
    }
    manifest["manifest_sha256"] = cache_io.canonical_sha256(manifest)
    return manifest


def _write_manifest_exclusive(path: Path, manifest: Mapping[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(cache_io.canonical_json_bytes(manifest, trailing_newline=True))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_root(
    root: Path,
    *,
    dataset_name: str,
    role: str,
    projection: Mapping[str, Any],
    identifiers: ProjectionIds,
    models: StrictModelBundle,
    source: SourceContext,
) -> PreparedInternalArtifacts:
    candidate = Path(root)
    _require(
        candidate.is_dir() and not candidate.is_symlink(),
        "artifact root must be a regular non-symlink directory",
    )
    ready = candidate.resolve(strict=True)
    _require(
        {entry.name for entry in ready.iterdir()}
        == {
            DEVELOPMENT_CACHE_NAME,
            VALIDATION_CACHE_NAME,
            ATLAS_NAME,
            MANIFEST_NAME,
        },
        "artifact root entries differ",
    )
    development = cache_io.read_cache(
        ready / DEVELOPMENT_CACHE_NAME,
        split_projection=projection,
    )
    validation = cache_io.read_cache(
        ready / VALIDATION_CACHE_NAME,
        split_projection=projection,
    )
    atlas_manifest = atlas_builder.validate_component_atlas_artifact(
        ready / ATLAS_NAME
    )
    expected = _artifact_summary(
        dataset_name=dataset_name,
        role=role,
        projection=projection,
        identifiers=identifiers,
        models=models,
        source=source,
        development=development,
        validation=validation,
        atlas_manifest=atlas_manifest,
    )
    observed = _read_json_object(ready / MANIFEST_NAME, name="preparation manifest")
    _require(observed == expected, "preparation manifest differs from full replay")
    return PreparedInternalArtifacts(
        root=ready,
        development_cache=ready / DEVELOPMENT_CACHE_NAME,
        validation_cache=ready / VALIDATION_CACHE_NAME,
        atlas=ready / ATLAS_NAME,
        manifest_path=ready / MANIFEST_NAME,
        manifest_sha256=expected["manifest_sha256"],
        replayed=True,
    )


def _commit_directory_noreplace(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if os.name == "posix":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(destination),
                1,
            )
            if result == 0:
                return
            observed_errno = ctypes.get_errno()
            if observed_errno == errno.EEXIST:
                raise FileExistsError(destination)
            if observed_errno not in (errno.ENOSYS, errno.EINVAL):
                raise OSError(
                    observed_errno,
                    os.strerror(observed_errno),
                    str(destination),
                )

    # Portable fallback serializes cooperating writers and rechecks before a
    # same-directory rename.
    lock = destination.parent / f".{destination.name}.commit.lock"
    descriptor: int | None = None
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        os.rename(source, destination)
    finally:
        if descriptor is not None:
            os.close(descriptor)
            lock.unlink(missing_ok=True)


def prepare_internal_artifacts(
    *,
    dataset_name: str,
    role: str,
    output_root: Path,
    source_lock_path: Path,
    device_name: str = "cuda:0",
) -> PreparedInternalArtifacts:
    """Build both caches and the development atlas, then commit once."""

    dataset_name = _require_dataset(dataset_name)
    role = _require_role(role)
    _require(device_name in ("cpu", "cuda:0"), "unsupported device")
    if device_name == "cuda:0":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    configure_formal_inference()
    source = load_source_context(source_lock_path)
    projection, identifiers = projection_and_ids(dataset_name)
    models = load_strict_models(dataset_name, role)
    projected_split = projection["datasets"][dataset_name][
        "canonical_split_sha256"
    ]
    _require(
        models.candidate_split_sha256 == projected_split,
        "completed V3 split differs from split projection",
    )

    requested = Path(output_root)
    _require(requested.name not in ("", ".", ".."), "output_root must name a child")
    requested.parent.mkdir(parents=True, exist_ok=True)
    _require(not requested.parent.is_symlink(), "output parent is a symlink")
    parent = requested.parent.resolve(strict=True)
    destination = parent / requested.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage.", dir=parent))
    committed = False
    try:
        device = torch.device(device_name)
        for model in (
            models.v3_candidate,
            models.current,
            models.original,
        ):
            model.to(device)
        development_dataset = build_partition_dataset(
            dataset_name=dataset_name,
            data_root=models.data_root,
            projection=projection,
            identifiers=identifiers,
            partition="development_train",
        )
        validation_dataset = build_partition_dataset(
            dataset_name=dataset_name,
            data_root=models.data_root,
            projection=projection,
            identifiers=identifiers,
            partition="internal_validation",
        )
        _require(
            dict(development_dataset.normalization)
            == dict(validation_dataset.normalization),
            "partition normalization differs",
        )
        write_partition_cache(
            staging / DEVELOPMENT_CACHE_NAME,
            dataset_name=dataset_name,
            role=role,
            partition="development_train",
            dataset=development_dataset,
            split_projection=projection,
            models=models,
            device=device,
            source=source,
        )
        write_partition_cache(
            staging / VALIDATION_CACHE_NAME,
            dataset_name=dataset_name,
            role=role,
            partition="internal_validation",
            dataset=validation_dataset,
            split_projection=projection,
            models=models,
            device=device,
            source=source,
        )
        development = cache_io.read_cache(
            staging / DEVELOPMENT_CACHE_NAME,
            split_projection=projection,
        )
        _build_atlas_from_development_cache(
            staging / ATLAS_NAME,
            dataset_name=dataset_name,
            role=role,
            development_cache=development,
            projection=projection,
            identifiers=identifiers,
            models=models,
            source=source,
        )
        validation = cache_io.read_cache(
            staging / VALIDATION_CACHE_NAME,
            split_projection=projection,
        )
        atlas_manifest = atlas_builder.validate_component_atlas_artifact(
            staging / ATLAS_NAME
        )
        manifest = _artifact_summary(
            dataset_name=dataset_name,
            role=role,
            projection=projection,
            identifiers=identifiers,
            models=models,
            source=source,
            development=development,
            validation=validation,
            atlas_manifest=atlas_manifest,
        )
        _write_manifest_exclusive(staging / MANIFEST_NAME, manifest)
        _validate_root(
            staging,
            dataset_name=dataset_name,
            role=role,
            projection=projection,
            identifiers=identifiers,
            models=models,
            source=source,
        )
        _commit_directory_noreplace(staging, destination)
        committed = True
        validated = _validate_root(
            destination,
            dataset_name=dataset_name,
            role=role,
            projection=projection,
            identifiers=identifiers,
            models=models,
            source=source,
        )
        return PreparedInternalArtifacts(
            root=validated.root,
            development_cache=validated.development_cache,
            validation_cache=validated.validation_cache,
            atlas=validated.atlas,
            manifest_path=validated.manifest_path,
            manifest_sha256=validated.manifest_sha256,
            replayed=False,
        )
    finally:
        if not committed and staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)


def replay_internal_artifacts(
    *,
    dataset_name: str,
    role: str,
    output_root: Path,
    source_lock_path: Path,
) -> PreparedInternalArtifacts:
    """Strictly reload authorities and fully replay existing artifact bytes."""

    dataset_name = _require_dataset(dataset_name)
    role = _require_role(role)
    configure_formal_inference()
    source = load_source_context(source_lock_path)
    projection, identifiers = projection_and_ids(dataset_name)
    models = load_strict_models(dataset_name, role)
    _require(
        models.candidate_split_sha256
        == projection["datasets"][dataset_name]["canonical_split_sha256"],
        "completed V3 split differs from split projection",
    )
    return _validate_root(
        output_root,
        dataset_name=dataset_name,
        role=role,
        projection=projection,
        identifiers=identifiers,
        models=models,
        source=source,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--replay-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.replay_existing:
        result = replay_internal_artifacts(
            dataset_name=args.dataset,
            role=args.role,
            output_root=args.output_root,
            source_lock_path=args.source_lock,
        )
    else:
        result = prepare_internal_artifacts(
            dataset_name=args.dataset,
            role=args.role,
            output_root=args.output_root,
            source_lock_path=args.source_lock,
            device_name=args.device,
        )
    print(
        cache_io.canonical_json_bytes(
            {
                "status": "validated" if result.replayed else "complete",
                "dataset": args.dataset,
                "role": args.role,
                "root": str(result.root),
                "manifest_sha256": result.manifest_sha256,
                "official_test_accessed": False,
            }
        ).decode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATLAS_NAME",
    "DATASETS",
    "DEVELOPMENT_CACHE_NAME",
    "MANIFEST_NAME",
    "PBDRV4InternalPreparationError",
    "PreparedInternalArtifacts",
    "ProjectionIds",
    "ROLES",
    "SCHEMA",
    "SourceContext",
    "StrictModelBundle",
    "VALIDATION_CACHE_NAME",
    "V3_RUN_DIRECTORIES",
    "build_partition_dataset",
    "configure_formal_inference",
    "load_projection_ids",
    "load_source_context",
    "load_strict_models",
    "main",
    "parse_args",
    "prepare_internal_artifacts",
    "projection_and_ids",
    "replay_internal_artifacts",
    "write_partition_cache",
]
