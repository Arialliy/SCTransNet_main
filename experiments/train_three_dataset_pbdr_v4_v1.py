#!/usr/bin/env python3
"""Deterministic three-dataset PBDR-V4 Stage-1/Stage-2 trainer.

This entry point is deliberately limited to the frozen official-train
projection.  It never imports or opens an official-test index.  A formal run
must bind one immutable source lock, the existing V3 80/20 split projection,
one Current-derived component atlas, and (for Stage-2) one complete selected
Stage-1 checkpoint.

Epoch selection uses the complete fixed-0.5 role key.  There is no pass flag,
epsilon, relative improvement, or minimum effect-size prefix; an exact key tie
keeps the earlier epoch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.build_pbdr_v4_component_atlas import (
    validate_component_atlas_artifact,
)
from experiments.component_matching_v2 import match_components_v2
from experiments.pbdr_v4_atlas_dataset import PBDRV4AtlasTrainDataset
from experiments.pbdr_v4_component_loss import (
    compute_pbdr_v4_loss,
    role_loss_manifest,
)
from experiments.pbdr_v4_internal_dataset import PBDRV4InternalInferenceDataset
from experiments.pbdr_v4_metric_core import PBDRV4MetricAccumulator
from experiments import pbdr_v4_models_seed42_v1 as models
from experiments.pbdr_v4_run_artifacts import (
    RunIdentity,
    atomic_rolling_torch_save,
    checkpoint_payload,
    epoch_checkpoint_path,
    exclusive_json,
    exclusive_torch_save,
    file_sha256,
    load_torch_artifact,
    optimizer_group_signature,
    validate_checkpoint_payload,
)
from experiments.pbdr_v4_source_lock import load_source_lock
from experiments import pbdr_v4_split_authority as split_authority
from experiments.pbdr_v4_state_contract import (
    audit_candidate_against_current,
    audit_training_modes,
    configure_stage_training,
    l2sp_to_current,
    state_semantic_sha256,
)
from experiments.pbdr_v4_training_core import (
    EVAL_EVERY,
    STAGE2_L2SP_WEIGHT,
    STAGE_EPOCHS,
    TRAIN_BATCH_SIZE,
    TRAINING_SEED,
    build_optimizer,
    capture_rng_state,
    checkpoint_epoch_key,
    configure_determinism,
    restore_rng_state,
    training_recipe,
)


SCHEMA = "sctransnet_three_dataset_pbdr_v4_training_v1/v1"
DATASETS = models.DATASETS
ROLES = models.ROLES
STAGES = models.STAGES
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results/pbdr_v4_v1"
ROLLING_NAME = "rolling_state.pth.tar"
SELECTED_NAME = "selected_candidate.pth.tar"
SUMMARY_NAME = "summary.json"
RUN_PROTOCOL_NAME = "run_protocol.json"
FORMAL_WORKERS = 0
FORMAL_GPU_UUIDS = {
    "NUDT-SIRST": "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    "IRSTD-1K": "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
    "NUAA-SIRST": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}


class PBDRV4TrainerError(RuntimeError):
    """A run binding, tensor, metric, resume, or artifact is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV4TrainerError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    _require(
        candidate.is_file() and not candidate.is_symlink(),
        f"{label} must be a regular non-symlink file",
    )
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PBDRV4TrainerError(f"cannot read {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must contain one object")
    return value


def _sha256(value: object, *, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return value


def _canonical_json_sha(value: object) -> str:
    try:
        content = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PBDRV4TrainerError(f"value is not canonical JSON: {error}") from error
    return hashlib.sha256(content).hexdigest()


def _candidate_manifest_sha(payload: Mapping[str, Any]) -> str:
    """Hash every candidate field except tensor bytes and this self hash."""

    unsigned = dict(payload)
    unsigned.pop("state_dict", None)
    unsigned.pop("candidate_manifest_sha256", None)
    return _canonical_json_sha(unsigned)


def _commit_or_replay_selected(
    path: Path,
    expected: Mapping[str, Any],
) -> Path:
    destination = Path(path)
    expected_manifest_sha = _sha256(
        expected.get("candidate_manifest_sha256"),
        name="expected candidate manifest SHA",
    )
    if not destination.exists() and not destination.is_symlink():
        return exclusive_torch_save(destination, expected)
    _require(destination.is_file() and not destination.is_symlink(), "selected candidate path is unsafe")
    observed = load_torch_artifact(destination)
    observed_manifest_sha = _sha256(
        observed.get("candidate_manifest_sha256"),
        name="observed candidate manifest SHA",
    )
    _require(
        observed_manifest_sha == _candidate_manifest_sha(observed),
        "existing candidate manifest SHA does not replay",
    )
    _require(observed_manifest_sha == expected_manifest_sha, "existing candidate manifest differs")
    observed_state = observed.get("state_dict")
    expected_state = expected.get("state_dict")
    _require(
        isinstance(observed_state, Mapping)
        and isinstance(expected_state, Mapping)
        and set(observed_state) == set(expected_state),
        "existing candidate state keys differ",
    )
    for name in expected_state:
        _require(
            isinstance(observed_state[name], torch.Tensor)
            and isinstance(expected_state[name], torch.Tensor)
            and torch.equal(observed_state[name], expected_state[name]),
            f"existing candidate state differs: {name}",
        )
    _require(
        observed.get("state_sha256") == expected.get("state_sha256")
        == state_semantic_sha256(expected_state),
        "existing candidate state SHA differs",
    )
    return destination.resolve(strict=True)


def _commit_or_validate_json(path: Path, expected: Mapping[str, Any]) -> Path:
    destination = Path(path)
    if not destination.exists() and not destination.is_symlink():
        return exclusive_json(destination, expected)
    _require(destination.is_file() and not destination.is_symlink(), "bound JSON path is unsafe")
    observed = _read_json(destination, label=destination.name)
    _require(observed == dict(expected), f"existing {destination.name} differs")
    return destination.resolve(strict=True)


def load_live_split_projection(path: Path) -> dict[str, Any]:
    """Load the persisted projection and replay it against frozen V3 files."""

    observed = _read_json(path, label="split projection")
    expected = split_authority.build_projection()
    _require(observed == expected, "split projection differs from live authority")
    return observed


def load_official_train_source_ids(
    projection: Mapping[str, Any],
    dataset: str,
) -> tuple[list[str], list[str], list[str]]:
    """Read only the code-pinned V3 official-train split manifest."""

    _require(dataset in DATASETS, "unsupported dataset")
    datasets = projection.get("datasets")
    _require(isinstance(datasets, Mapping), "projection datasets are malformed")
    record = datasets.get(dataset)
    _require(isinstance(record, Mapping), "dataset is absent from projection")
    source = Path(str(record.get("source_path")))
    _require(source.is_file() and not source.is_symlink(), "source split is unsafe")
    _require(source.stat().st_size == record.get("source_bytes"), "source split bytes differ")
    _require(
        split_authority.file_sha256(source) == record.get("source_file_sha256"),
        "source split file SHA differs",
    )
    payload = _read_json(source, label=f"{dataset} source split")
    split_authority.validate_split_payload(dataset, payload)
    official = list(payload["official_train_ids"])
    development = list(payload["development_train_ids"])
    validation = list(payload["internal_validation_ids"])
    return official, development, validation


def _normalization(dataset: str) -> dict[str, float]:
    from experiments import three_dataset_v2_protocol as data_protocol

    value = data_protocol.get_legacy_normalization(dataset)
    return {"mean": float(value["mean"]), "std": float(value["std"])}


def _atlas_bindings(
    atlas_root: Path,
    *,
    dataset: str,
    role: str,
    source_lock_sha256: str,
    split_projection_sha256: str,
    parent_checkpoint_sha256: str,
    parent_state_sha256: str,
) -> tuple[Path, dict[str, Any], str]:
    root = Path(atlas_root)
    manifest = validate_component_atlas_artifact(root)
    expected = {
        "dataset": dataset,
        "role": role,
        "source_lock_sha256": source_lock_sha256,
        "split_projection_sha256": split_projection_sha256,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_state_sha256": parent_state_sha256,
        "official_test_accessed": False,
    }
    for name, value in expected.items():
        _require(manifest.get(name) == value, f"atlas {name} binding differs")
    atlas_sha = _sha256(manifest.get("manifest_sha256"), name="atlas manifest SHA")
    return root.resolve(strict=True) / "manifest.json", manifest, atlas_sha


def _crop_batch_tensor(
    value: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    _require(value.ndim == 4 and value.shape[0] == value.shape[1] == 1, "validation tensor must be 1x1xHxW")
    _require(0 < height <= value.shape[-2] and 0 < width <= value.shape[-1], "validation crop is invalid")
    return value[..., :height, :width]


def _collated_original_size(value: Any) -> tuple[int, int]:
    """Decode batch-size-one default-collate output for ``(height, width)``."""

    if isinstance(value, (tuple, list)) and len(value) == 2:
        decoded: list[int] = []
        for item in value:
            if isinstance(item, torch.Tensor):
                _require(item.numel() == 1, "collated original size differs")
                decoded.append(int(item.item()))
            elif isinstance(item, (tuple, list)) and len(item) == 1:
                decoded.append(int(item[0]))
            else:
                decoded.append(int(item))
        return decoded[0], decoded[1]
    raise PBDRV4TrainerError("collated original size is malformed")


def _collated_identifier(value: Any) -> str:
    if isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    _require(isinstance(value, str) and bool(value), "validation identifier differs")
    return value


def validate_cuda_uuid(device: torch.device, expected_uuid: str | None) -> str | None:
    """Bind a CUDA-visible logical device to its physical UUID."""

    if device.type != "cuda":
        _require(expected_uuid is None, "CPU training cannot bind a CUDA UUID")
        return None
    _require(expected_uuid is not None and expected_uuid.startswith("GPU-"), "expected CUDA UUID is required")
    index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    observed_raw = getattr(properties, "uuid", None)
    _require(observed_raw is not None, "CUDA runtime did not expose a device UUID")
    observed = str(observed_raw)
    if not observed.startswith("GPU-"):
        observed = f"GPU-{observed}"
    _require(observed == expected_uuid, f"CUDA device UUID differs: {observed}")
    return observed


@dataclass
class TrainingDiagnostics:
    batches: int = 0
    samples: int = 0
    loss_sum: float = 0.0
    l2sp_sum: float = 0.0
    positive_delta_count: int = 0
    negative_delta_count: int = 0
    delta_abs_sum: float = 0.0
    delta_square_sum: float = 0.0
    threshold_up_crossings: int = 0
    threshold_down_crossings: int = 0
    region_delta_sum: dict[str, float] | None = None
    region_delta_square_sum: dict[str, float] | None = None
    region_pixel_count: dict[str, int] | None = None
    loss_components_sum: dict[str, float] | None = None

    def __post_init__(self) -> None:
        names = ("rescue", "suppress", "preserve")
        if self.region_delta_sum is None:
            self.region_delta_sum = {name: 0.0 for name in names}
        if self.region_delta_square_sum is None:
            self.region_delta_square_sum = {name: 0.0 for name in names}
        if self.region_pixel_count is None:
            self.region_pixel_count = {name: 0 for name in names}
        if self.loss_components_sum is None:
            self.loss_components_sum = {}

    def update(
        self,
        *,
        total: torch.Tensor,
        l2sp: torch.Tensor,
        loss_components: Mapping[str, float],
        base: torch.Tensor,
        routed: torch.Tensor,
        delta: torch.Tensor,
        component_maps: Mapping[str, torch.Tensor],
    ) -> None:
        self.batches += 1
        self.samples += int(delta.shape[0])
        self.loss_sum += float(total.detach().cpu().item())
        self.l2sp_sum += float(l2sp.detach().cpu().item())
        detached = delta.detach()
        self.positive_delta_count += int((detached > 0).sum().item())
        self.negative_delta_count += int((detached < 0).sum().item())
        self.delta_abs_sum += float(detached.abs().sum().cpu().item())
        self.delta_square_sum += float(detached.square().sum().cpu().item())
        self.threshold_up_crossings += int(((base.detach() <= 0) & (routed.detach() > 0)).sum().item())
        self.threshold_down_crossings += int(((base.detach() > 0) & (routed.detach() <= 0)).sum().item())
        assert self.region_delta_sum is not None
        assert self.region_delta_square_sum is not None
        assert self.region_pixel_count is not None
        for name, ids in component_maps.items():
            mask = ids > 0
            count = int(mask.sum().item())
            self.region_pixel_count[name] += count
            if count:
                selected = detached[mask]
                self.region_delta_sum[name] += float(selected.sum().cpu().item())
                self.region_delta_square_sum[name] += float(selected.square().sum().cpu().item())
        assert self.loss_components_sum is not None
        for name, value in loss_components.items():
            self.loss_components_sum[name] = self.loss_components_sum.get(name, 0.0) + float(value)

    def compute(self) -> dict[str, object]:
        _require(self.batches > 0, "cannot compute empty training diagnostics")
        assert self.region_delta_sum is not None
        assert self.region_delta_square_sum is not None
        assert self.region_pixel_count is not None
        assert self.loss_components_sum is not None
        regions: dict[str, object] = {}
        for name in self.region_pixel_count:
            count = self.region_pixel_count[name]
            regions[name] = {
                "pixel_count": count,
                "delta_mean": 0.0 if not count else self.region_delta_sum[name] / count,
                "delta_rms": 0.0 if not count else math.sqrt(self.region_delta_square_sum[name] / count),
            }
        return {
            "batch_count": self.batches,
            "sample_count": self.samples,
            "mean_total_loss": self.loss_sum / self.batches,
            "mean_l2sp": self.l2sp_sum / self.batches,
            "positive_delta_count": self.positive_delta_count,
            "negative_delta_count": self.negative_delta_count,
            "delta_abs_sum": self.delta_abs_sum,
            "delta_rms_global_numerator": self.delta_square_sum,
            "threshold_up_crossings": self.threshold_up_crossings,
            "threshold_down_crossings": self.threshold_down_crossings,
            "atlas_regions": regions,
            "mean_loss_components": {
                name: value / self.batches
                for name, value in sorted(self.loss_components_sum.items())
            },
        }


def validate_candidate(
    model: torch.nn.Module,
    validation_loader: Iterable[Any],
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, object]]:
    """One ordered internal-validation pass at the immutable 0.5 rule."""

    model.eval()
    accumulator = PBDRV4MetricAccumulator()
    delta_arrays: list[np.ndarray] = []
    component_transitions: list[dict[str, object]] = []
    up_crossings = 0
    down_crossings = 0
    with torch.no_grad():
        for batch in validation_loader:
            _require(isinstance(batch, (tuple, list)) and len(batch) == 4, "validation batch differs")
            image, target, original_size, identifier = batch
            image = image.to(device=device, dtype=torch.float32, non_blocking=False)
            target = target.to(device=device, dtype=torch.float32, non_blocking=False)
            _, auxiliary = model.forward_for_pbdr_v4_training(image)
            height, width = _collated_original_size(original_size)
            sample_id = _collated_identifier(identifier)
            logits = _crop_batch_tensor(auxiliary.routed_logits, height, width)
            base = _crop_batch_tensor(auxiliary.candidate_base_logits, height, width)
            delta = _crop_batch_tensor(auxiliary.delta_logits, height, width)
            target_crop = _crop_batch_tensor(target, height, width)
            probability = torch.sigmoid(logits)
            loss = F.binary_cross_entropy_with_logits(logits.float(), target_crop.float())
            accumulator.update(
                probability=np.ascontiguousarray(probability[0, 0].cpu().numpy(), dtype=np.float32),
                target=np.ascontiguousarray(target_crop[0, 0].cpu().numpy(), dtype=np.float32),
                loss=float(loss.cpu().item()),
                identifier=sample_id,
            )
            detached = delta[0, 0].cpu().numpy().astype(np.float32, copy=False)
            delta_arrays.append(np.ascontiguousarray(detached).reshape(-1))
            up_crossings += int(((base <= 0) & (logits > 0)).sum().cpu().item())
            down_crossings += int(((base > 0) & (logits <= 0)).sum().cpu().item())
            base_map = np.ascontiguousarray(base[0, 0].cpu().numpy(), dtype=np.float32)
            routed_map = np.ascontiguousarray(logits[0, 0].cpu().numpy(), dtype=np.float32)
            target_map = np.ascontiguousarray(target_crop[0, 0].cpu().numpy(), dtype=np.float32)
            target_binary = np.ascontiguousarray(target_map > np.float32(0.5))
            base_match = match_components_v2(
                prediction_mask=np.ascontiguousarray(base_map > np.float32(0.0)),
                target_mask=target_binary,
            )
            routed_match = match_components_v2(
                prediction_mask=np.ascontiguousarray(routed_map > np.float32(0.0)),
                target_mask=target_binary,
            )
            base_matched = set(base_match.matched_target_ids)
            routed_matched = set(routed_match.matched_target_ids)
            peaks: list[dict[str, object]] = []
            for target_record in base_match.targets:
                component_id = target_record.component_id
                component_mask = base_match.target_id_map == component_id
                peaks.append(
                    {
                        "target_component_id": component_id,
                        "area": target_record.area,
                        "base_peak_logit": float(base_map[component_mask].max()),
                        "routed_peak_logit": float(routed_map[component_mask].max()),
                        "base_matched": component_id in base_matched,
                        "routed_matched": component_id in routed_matched,
                    }
                )
            component_transitions.append(
                {
                    "sample_id": sample_id,
                    "recovered_target_component_ids": sorted(routed_matched - base_matched),
                    "lost_target_component_ids": sorted(base_matched - routed_matched),
                    "base_unmatched_prediction_component_ids": list(base_match.unmatched_prediction_ids),
                    "routed_unmatched_prediction_component_ids": list(routed_match.unmatched_prediction_ids),
                    "base_unmatched_prediction_pixels": base_match.unmatched_prediction_pixels,
                    "routed_unmatched_prediction_pixels": routed_match.unmatched_prediction_pixels,
                    "target_component_peaks": peaks,
                }
            )
    metrics = accumulator.compute()
    all_delta = np.concatenate(delta_arrays)
    quantiles = np.quantile(all_delta, (0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0))
    diagnostics = {
        "delta_quantiles": {
            name: float(value)
            for name, value in zip(
                ("q0", "q01", "q10", "q50", "q90", "q99", "q100"),
                quantiles,
            )
        },
        "positive_delta_count": int((all_delta > 0).sum()),
        "negative_delta_count": int((all_delta < 0).sum()),
        "zero_delta_count": int((all_delta == 0).sum()),
        "threshold_up_crossings": up_crossings,
        "threshold_down_crossings": down_crossings,
        "component_transitions": component_transitions,
    }
    return metrics, diagnostics


def _make_train_loader(
    dataset: PBDRV4AtlasTrainDataset,
    *,
    epoch: int,
    batch_size: int,
    max_samples: int | None,
) -> DataLoader[Any]:
    dataset.set_epoch(epoch)
    selected: Any = dataset
    if max_samples is not None:
        _require(0 < max_samples <= len(dataset), "max train samples is invalid")
        selected = Subset(dataset, range(max_samples))
    generator = torch.Generator()
    generator.manual_seed(TRAINING_SEED * 1_000_003 + epoch)
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=True,
        num_workers=FORMAL_WORKERS,
        pin_memory=False,
        drop_last=False,
        generator=generator,
    )


def train_one_epoch(
    model: torch.nn.Module,
    current_reference: torch.nn.Module,
    loader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    *,
    role: str,
    stage: str,
    device: torch.device,
    current_state: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    configure_stage_training(model, stage)  # type: ignore[arg-type]
    current_reference.eval()
    diagnostics = TrainingDiagnostics()
    for batch in loader:
        _require(isinstance(batch, (tuple, list)) and len(batch) == 6, "training batch differs")
        image, target, rescue, suppress, preserve, _ = batch
        image = image.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        rescue = rescue.to(device=device)
        suppress = suppress.to(device=device)
        preserve = preserve.to(device=device)
        optimizer.zero_grad(set_to_none=True)
        _, auxiliary = model.forward_for_pbdr_v4_training(image)
        with torch.no_grad():
            _, current_auxiliary = current_reference.forward_for_pbdr_v4_training(image)
            current_logits = current_auxiliary.candidate_base_logits.detach()
        if stage == "stage1":
            _require(
                torch.equal(
                    auxiliary.candidate_base_logits.detach(),
                    current_logits,
                ),
                "Stage-1 candidate base is not bitwise Current",
            )
        loss_output = compute_pbdr_v4_loss(
            role=role,  # type: ignore[arg-type]
            routed_logits=auxiliary.routed_logits,
            candidate_base_logits=auxiliary.candidate_base_logits,
            reference_current_logits=current_logits,
            delta_logits=auxiliary.delta_logits,
            target=target,
            rescue_component_ids=rescue,
            suppress_component_ids=suppress,
            preserve_component_ids=preserve,
        )
        if stage == "stage2":
            l2sp = l2sp_to_current(model, current_state=current_state)
            total = loss_output.total + STAGE2_L2SP_WEIGHT * l2sp
        else:
            l2sp = loss_output.total.new_zeros(())
            total = loss_output.total
        _require(bool(torch.isfinite(total)), "training loss is non-finite")
        total.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                _require(parameter.grad is not None, f"trainable parameter lacks gradient: {name}")
                _require(bool(torch.isfinite(parameter.grad).all()), f"gradient is non-finite: {name}")
            else:
                _require(parameter.grad is None, f"frozen parameter received gradient: {name}")
        optimizer.step()
        diagnostics.update(
            total=total,
            l2sp=l2sp,
            loss_components=loss_output.detached_scalars(),
            base=auxiliary.candidate_base_logits,
            routed=auxiliary.routed_logits,
            delta=auxiliary.delta_logits,
            component_maps={"rescue": rescue, "suppress": suppress, "preserve": preserve},
        )
    audit_training_modes(model, stage)  # type: ignore[arg-type]
    audit_candidate_against_current(model, current_state=current_state, stage=stage)  # type: ignore[arg-type]
    return diagnostics.compute()


def _json_role_key(value: Sequence[object]) -> list[object]:
    result: list[object] = []
    for item in value:
        if isinstance(item, Fraction):
            result.append({"numerator": item.numerator, "denominator": item.denominator})
        elif isinstance(item, (float, int, str)) or item is None:
            result.append(item)
        else:
            result.append(str(item))
    return result


def _completed_summary(
    run_dir: Path,
    identity: RunIdentity,
    *,
    protocol_sha256: str,
) -> Path | None:
    summary_path = run_dir / SUMMARY_NAME
    if not summary_path.exists() and not summary_path.is_symlink():
        return None
    summary = _read_json(summary_path, label="completed training summary")
    _require(summary.get("schema") == SCHEMA and summary.get("status") == "complete", "completed summary status differs")
    _require(summary.get("run_identity") == identity.as_dict(), "completed summary identity differs")
    declared_summary_sha = _sha256(summary.get("summary_sha256"), name="completed summary SHA")
    unsigned_summary = dict(summary)
    del unsigned_summary["summary_sha256"]
    _require(_canonical_json_sha(unsigned_summary) == declared_summary_sha, "completed summary SHA differs")
    _require(summary.get("run_protocol_sha256") == protocol_sha256, "completed summary protocol binding differs")
    selected = run_dir / SELECTED_NAME
    _require(selected.is_file() and not selected.is_symlink(), "completed selected candidate is missing")
    _require(summary.get("selected_checkpoint_sha256") == file_sha256(selected), "completed selected candidate bytes differ")
    return summary_path.resolve(strict=True)


def _validate_rolling_selection(
    selected: Mapping[str, Any],
    *,
    run_dir: Path,
    identity: RunIdentity,
    epochs: int,
    optimizer_signature: list[dict[str, Any]],
) -> dict[str, Any]:
    required = {
        "epoch",
        "metrics",
        "diagnostics",
        "selection_key",
        "selection_key_raw",
        "state_dict",
        "state_sha256",
        "epoch_checkpoint_path",
        "epoch_checkpoint_sha256",
    }
    _require(set(selected) == required, "rolling selected fields differ")
    epoch = selected["epoch"]
    _require(type(epoch) is int and 1 <= epoch <= epochs, "rolling selected epoch differs")
    metrics = selected["metrics"]
    _require(isinstance(metrics, Mapping), "rolling selected metrics differ")
    replayed_key = checkpoint_epoch_key(identity.role, metrics, epoch)
    _require(tuple(selected["selection_key_raw"]) == replayed_key, "rolling selected key does not replay")
    _require(selected["selection_key"] == _json_role_key(replayed_key), "rolling JSON role key differs")
    state = selected["state_dict"]
    _require(
        isinstance(state, Mapping)
        and bool(state)
        and all(isinstance(name, str) and isinstance(value, torch.Tensor) for name, value in state.items()),
        "rolling selected state differs",
    )
    selected_state_sha = _sha256(selected["state_sha256"], name="rolling selected state SHA")
    _require(state_semantic_sha256(state) == selected_state_sha, "rolling selected state SHA differs")
    expected_path = epoch_checkpoint_path(run_dir, epoch).resolve(strict=False)
    observed_path = Path(str(selected["epoch_checkpoint_path"]))
    _require(observed_path == expected_path, "rolling selected epoch-checkpoint path differs")
    observed_file_sha = _sha256(
        selected["epoch_checkpoint_sha256"],
        name="rolling selected epoch-checkpoint SHA",
    )
    _require(
        expected_path.is_file()
        and not expected_path.is_symlink()
        and file_sha256(expected_path) == observed_file_sha,
        "rolling selected epoch-checkpoint bytes differ",
    )
    epoch_payload = validate_checkpoint_payload(
        load_torch_artifact(expected_path),
        identity=identity,
        epochs=epochs,
        expected_optimizer_group_signature=optimizer_signature,
    )
    _require(epoch_payload["epoch"] == epoch, "selected epoch-checkpoint epoch differs")
    epoch_selected = epoch_payload["selected"]
    _require(
        isinstance(epoch_selected, Mapping)
        and epoch_selected.get("epoch") == epoch
        and epoch_selected.get("state_sha256") == selected_state_sha
        and epoch_selected.get("selection_key") == selected["selection_key"],
        "selected epoch-checkpoint selection binding differs",
    )
    return dict(selected)


def _initial_model_and_metadata(
    *,
    dataset: str,
    role: str,
    stage: str,
    stage1_checkpoint_path: Path | None,
    source_lock_sha256: str,
    split_projection_sha256: str,
    atlas_manifest_sha256: str,
) -> tuple[torch.nn.Module, dict[str, Any], str | None]:
    if stage == "stage1":
        _require(stage1_checkpoint_path is None, "Stage-1 cannot accept a Stage-1 checkpoint")
        model, metadata = models.build_stage1_training_model(dataset, role, "stage1")
        return model, metadata, None
    _require(stage1_checkpoint_path is not None, "Stage-2 requires a selected Stage-1 checkpoint")
    stage1_path = Path(stage1_checkpoint_path)
    stage1 = load_torch_artifact(stage1_path)
    initialization_sha = _sha256(stage1.get("initialization_sha256"), name="Stage-1 initialization SHA")
    model, metadata = models.build_stage2_training_model(
        stage1,
        dataset_name=dataset,
        role=role,
        stage="stage2",
        expected_source_sha256=source_lock_sha256,
        expected_split_sha256=split_projection_sha256,
        expected_atlas_sha256=atlas_manifest_sha256,
        expected_initialization_sha256=initialization_sha,
        expected_stage1_state_sha256=_sha256(stage1.get("state_sha256"), name="Stage-1 state SHA"),
    )
    return model, metadata, file_sha256(stage1_path)


def run(args: argparse.Namespace) -> Path:
    _require(args.dataset in DATASETS and args.role in ROLES and args.stage in STAGES, "dataset/role/stage differs")
    configure_determinism()
    source_lock = load_source_lock(args.source_lock, check_environment=True)
    source_sha = _sha256(source_lock.get("source_lock_sha256"), name="source lock SHA")
    projection = load_live_split_projection(args.split_projection)
    split_sha = _sha256(projection.get("projection_sha256"), name="split projection SHA")
    official_ids, development_ids, validation_ids = load_official_train_source_ids(projection, args.dataset)
    _, current_state, parent = models.load_current_checkpoint(args.dataset, args.role)
    parent_checkpoint_sha = _sha256(parent.get("sha256"), name="parent checkpoint SHA")
    parent_state_sha = _sha256(parent.get("state_sha256"), name="parent state SHA")
    atlas_manifest_path, atlas_manifest, atlas_sha = _atlas_bindings(
        args.atlas_root,
        dataset=args.dataset,
        role=args.role,
        source_lock_sha256=source_sha,
        split_projection_sha256=split_sha,
        parent_checkpoint_sha256=parent_checkpoint_sha,
        parent_state_sha256=parent_state_sha,
    )
    model, model_metadata, initialization_checkpoint_sha = _initial_model_and_metadata(
        dataset=args.dataset,
        role=args.role,
        stage=args.stage,
        stage1_checkpoint_path=args.stage1_checkpoint,
        source_lock_sha256=source_sha,
        split_projection_sha256=split_sha,
        atlas_manifest_sha256=atlas_sha,
    )
    identity = RunIdentity(
        dataset=args.dataset,
        role=args.role,
        stage=args.stage,
        source_lock_sha256=source_sha,
        split_projection_sha256=split_sha,
        atlas_manifest_sha256=atlas_sha,
        parent_checkpoint_sha256=parent_checkpoint_sha,
        parent_state_sha256=parent_state_sha,
        initialization_checkpoint_sha256=initialization_checkpoint_sha,
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _require(not run_dir.is_symlink(), "run directory cannot be a symlink")
    formal_epochs = STAGE_EPOCHS[args.stage]
    epochs = args.epochs if args.smoke else formal_epochs
    eval_every = args.eval_every if args.smoke else EVAL_EVERY
    batch_size = args.batch_size if args.smoke else TRAIN_BATCH_SIZE
    _require(type(epochs) is int and epochs > 0, "epoch budget is invalid")
    _require(type(eval_every) is int and eval_every > 0, "evaluation interval is invalid")
    _require(type(batch_size) is int and batch_size > 0, "batch size is invalid")
    if not args.smoke:
        _require(
            (epochs, eval_every, batch_size) == (formal_epochs, EVAL_EVERY, TRAIN_BATCH_SIZE),
            "formal training recipe differs",
        )
        _require(args.max_train_samples is None and args.max_val_samples is None, "formal run cannot limit samples")

    data_root = Path(args.data_root)
    _require(data_root.is_dir() and not data_root.is_symlink(), "data root is missing or unsafe")
    if not args.smoke:
        _require(
            data_root.resolve(strict=True) == DEFAULT_DATA_ROOT.resolve(strict=True),
            "formal data root differs from the frozen repository path",
        )
    protocol: dict[str, Any] = {
        "schema": f"{SCHEMA}/run_protocol",
        "run_identity": identity.as_dict(),
        "epochs": epochs,
        "eval_every": eval_every,
        "batch_size": batch_size,
        "workers": FORMAL_WORKERS,
        "device": args.device,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "data_root": str(data_root.resolve(strict=True)),
        "normalization": _normalization(args.dataset),
        "source_lock": {
            "path": str(Path(args.source_lock).resolve(strict=True)),
            "file_sha256": file_sha256(Path(args.source_lock)),
            "semantic_sha256": source_sha,
        },
        "split_projection": {
            "path": str(Path(args.split_projection).resolve(strict=True)),
            "file_sha256": file_sha256(Path(args.split_projection)),
            "semantic_sha256": split_sha,
        },
        "atlas_manifest": {
            "path": str(atlas_manifest_path.resolve(strict=True)),
            "file_sha256": file_sha256(atlas_manifest_path),
            "semantic_sha256": atlas_sha,
        },
        "smoke": bool(args.smoke),
        "smoke_limits": {
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
        },
        "official_test_accessed": False,
        "performance_acceptance_margin": None,
    }
    protocol["protocol_sha256"] = _canonical_json_sha(protocol)
    protocol_path = _commit_or_validate_json(
        run_dir / RUN_PROTOCOL_NAME,
        protocol,
    )
    protocol_sha = _sha256(protocol["protocol_sha256"], name="run protocol SHA")
    completed = _completed_summary(
        run_dir,
        identity,
        protocol_sha256=protocol_sha,
    )
    if completed is not None:
        _require(args.resume != "never", "completed run exists but resume=never")
        return completed

    train_dataset = PBDRV4AtlasTrainDataset(
        development_ids,
        official_ids,
        dataset_name=args.dataset,
        data_root=args.data_root,
        atlas_root=args.atlas_root,
        atlas_manifest=atlas_manifest_path,
        parent_state_sha256=parent_state_sha,
        normalization=_normalization(args.dataset),
    )
    validation_dataset: Any = PBDRV4InternalInferenceDataset(
        validation_ids,
        official_ids,
        manifest_scope="internal_validation_ids",
        selected_ids_ordered_sha256=split_authority.ordered_ids_sha256(validation_ids),
        official_train_count=len(official_ids),
        official_train_ordered_ids_sha256=split_authority.ordered_ids_sha256(official_ids),
        dataset_name=args.dataset,
        data_root=args.data_root,
    )
    if args.max_val_samples is not None:
        _require(args.smoke and 0 < args.max_val_samples <= len(validation_dataset), "max validation samples is invalid")
        validation_dataset = Subset(validation_dataset, range(args.max_val_samples))
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    _require(device.type in ("cpu", "cuda"), "unsupported device")
    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA is unavailable")
    expected_gpu_uuid = args.expected_gpu_uuid
    if not args.smoke and device.type == "cuda":
        _require(
            expected_gpu_uuid == FORMAL_GPU_UUIDS[args.dataset],
            "formal dataset/GPU UUID binding differs",
        )
    observed_gpu_uuid = validate_cuda_uuid(device, expected_gpu_uuid)
    model.to(device)
    current_reference, current_reference_metadata = models.build_frozen_current_reference_model(
        args.dataset, args.role, args.stage
    )
    current_reference.to(device)
    current_reference.eval()
    optimizer = build_optimizer(model, args.stage)
    optimizer_signature = optimizer_group_signature(optimizer.state_dict())

    start_epoch = 1
    best: dict[str, Any] | None = None
    rolling_path = run_dir / ROLLING_NAME
    if rolling_path.exists() or rolling_path.is_symlink():
        _require(args.resume != "never", "rolling checkpoint exists but resume=never")
        rolling = validate_checkpoint_payload(
            load_torch_artifact(rolling_path),
            identity=identity,
            epochs=epochs,
            expected_optimizer_group_signature=optimizer_signature,
        )
        model.load_state_dict(rolling["state_dict"], strict=True)
        optimizer.load_state_dict(rolling["optimizer"])
        restore_rng_state(rolling["rng_state"])
        selected = rolling["selected"]
        best = (
            None
            if not selected
            else _validate_rolling_selection(
                selected,
                run_dir=run_dir,
                identity=identity,
                epochs=epochs,
                optimizer_signature=optimizer_signature,
            )
        )
        audit_candidate_against_current(
            model,
            current_state=current_state,
            stage=args.stage,
        )
        configure_stage_training(model, args.stage)
        start_epoch = int(rolling["epoch"]) + 1
    else:
        _require(args.resume != "required", "resume=required but rolling checkpoint is absent")

    for epoch in range(start_epoch, epochs + 1):
        train_loader = _make_train_loader(
            train_dataset,
            epoch=epoch,
            batch_size=batch_size,
            max_samples=args.max_train_samples,
        )
        training_diagnostics = train_one_epoch(
            model,
            current_reference,
            train_loader,
            optimizer,
            role=args.role,
            stage=args.stage,
            device=device,
            current_state=current_state,
        )
        evaluation: dict[str, Any] | None = None
        if epoch % eval_every == 0 or epoch == epochs:
            metrics, validation_diagnostics = validate_candidate(
                model,
                validation_loader,
                device=device,
            )
            key = checkpoint_epoch_key(args.role, metrics, epoch)
            evaluation = {
                "epoch": epoch,
                "metrics": metrics,
                "diagnostics": validation_diagnostics,
                "selection_key": _json_role_key(key),
            }
            if best is None or key > tuple(best["selection_key_raw"]):
                best = {
                    "epoch": epoch,
                    "metrics": metrics,
                    "diagnostics": validation_diagnostics,
                    "selection_key": _json_role_key(key),
                    "selection_key_raw": tuple(key),
                    "state_dict": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    },
                    "state_sha256": state_semantic_sha256(model.state_dict()),
                }
            configure_stage_training(model, args.stage)

        event = {
            "epoch": epoch,
            "training": training_diagnostics,
            "evaluation": evaluation,
            "selected_epoch": None if best is None else best["epoch"],
        }
        epoch_selected = (
            {}
            if best is None
            else {
                "epoch": best["epoch"],
                "state_sha256": best["state_sha256"],
                "selection_key": best["selection_key"],
            }
        )
        epoch_payload = checkpoint_payload(
            identity=identity,
            epoch=epoch,
            epochs=epochs,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            rng_state=capture_rng_state(),
            selected=epoch_selected,
            event=event,
        )
        if evaluation is not None:
            committed_epoch_path = exclusive_torch_save(
                epoch_checkpoint_path(run_dir, epoch), epoch_payload
            )
            if best is not None and best["epoch"] == epoch:
                best["epoch_checkpoint_path"] = str(
                    committed_epoch_path.resolve(strict=True)
                )
                best["epoch_checkpoint_sha256"] = file_sha256(
                    committed_epoch_path
                )
        rolling_payload = checkpoint_payload(
            identity=identity,
            epoch=epoch,
            epochs=epochs,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            rng_state=epoch_payload["rng_state"],
            selected={} if best is None else best,
            event=event,
        )
        atomic_rolling_torch_save(rolling_path, rolling_payload)
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "dataset": args.dataset,
                    "role": args.role,
                    "stage": args.stage,
                    "epoch": epoch,
                    "epochs": epochs,
                    "mean_total_loss": training_diagnostics["mean_total_loss"],
                    "evaluated": evaluation is not None,
                    "selected_epoch": None if best is None else best["epoch"],
                    "official_test_accessed": False,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )

    _require(best is not None, "training completed without an evaluated epoch")
    model.load_state_dict(best["state_dict"], strict=True)
    candidate = models.build_candidate_checkpoint_payload(
        model,
        dataset_name=args.dataset,
        role=args.role,
        stage=args.stage,
        source_sha256=source_sha,
        split_sha256=split_sha,
        atlas_sha256=atlas_sha,
        initialization_sha256=_sha256(model_metadata["initialization_sha256"], name="model initialization SHA"),
    )
    candidate.update(
        {
            "epoch": best["epoch"],
            "validation_metrics": best["metrics"],
            "validation_diagnostics": best["diagnostics"],
            "selection_key": best["selection_key"],
            "run_identity": identity.as_dict(),
            "run_protocol_sha256": protocol_sha,
            "training_recipe": training_recipe(args.stage),
            "loss_manifest": role_loss_manifest(args.role),
            "source_lock_sha256": source_sha,
            "split_projection_sha256": split_sha,
            "atlas_manifest_sha256": atlas_sha,
            "official_test_accessed": False,
            "performance_acceptance_margin": None,
            "smoke": bool(args.smoke),
        }
    )
    candidate["candidate_manifest_sha256"] = _candidate_manifest_sha(candidate)
    selected_path = _commit_or_replay_selected(
        run_dir / SELECTED_NAME,
        candidate,
    )
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": args.dataset,
        "role": args.role,
        "stage": args.stage,
        "run_identity": identity.as_dict(),
        "run_protocol": str(protocol_path.resolve(strict=True)),
        "run_protocol_sha256": protocol_sha,
        "selected_epoch": best["epoch"],
        "selected_metrics": best["metrics"],
        "selected_diagnostics": best["diagnostics"],
        "selected_checkpoint": str(selected_path.resolve(strict=True)),
        "selected_checkpoint_sha256": file_sha256(selected_path),
        "selected_state_sha256": candidate["state_sha256"],
        "source_lock_sha256": source_sha,
        "split_projection_sha256": split_sha,
        "atlas_manifest_sha256": atlas_sha,
        "parent_checkpoint": parent,
        "model_metadata": model_metadata,
        "current_reference_metadata": current_reference_metadata,
        "atlas_component_statistics": atlas_manifest.get("aggregate_component_statistics"),
        "training_recipe": training_recipe(args.stage),
        "normalization": _normalization(args.dataset),
        "expected_gpu_uuid": expected_gpu_uuid,
        "observed_gpu_uuid": observed_gpu_uuid,
        "official_test_accessed": False,
        "performance_acceptance_margin": None,
        "smoke": bool(args.smoke),
    }
    summary["summary_sha256"] = _canonical_json_sha(summary)
    return exclusive_json(run_dir / SUMMARY_NAME, summary).resolve(strict=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--split-projection", type=Path, required=True)
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(run(parse_args(argv)), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "DATASETS",
    "PBDRV4TrainerError",
    "ROLES",
    "SCHEMA",
    "STAGES",
    "TrainingDiagnostics",
    "validate_cuda_uuid",
    "load_live_split_projection",
    "load_official_train_source_ids",
    "parse_args",
    "run",
    "train_one_epoch",
    "validate_candidate",
]
