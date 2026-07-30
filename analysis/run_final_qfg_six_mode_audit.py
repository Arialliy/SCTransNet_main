#!/usr/bin/env python3
"""Execute or verify the F1 six-mode QFG functional audit.

The formal execution loads the frozen head-free inference artifact, evaluates
only the 133-image internal validation split, and runs:

``full``, ``qfg_off``, and ``level1_off`` through ``level4_off``.

The implementation reuses the frozen metric/sweep implementation together
with the F1 cache and knockout primitives.  It never writes a derived
checkpoint and never reads an official-test index.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import functools
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from skimage import measure
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import audit_final_qfg_functional_use as audit_core  # noqa: E402
from analysis import (  # noqa: E402
    collect_final_model_validation_statistics as cache_core,
)
from experiments import evaluate_pd_fa_sweep as sweep_core  # noqa: E402
from experiments import (  # noqa: E402
    evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa as qfg_evaluator,
)
from experiments import (  # noqa: E402
    export_tpd_ner_v4_qfg_v2_croa_to_inference as exporter,
)
from experiments import (  # noqa: E402
    freeze_final_model_certification_parent_lock as parent_freezer,
)
from experiments import (  # noqa: E402
    freeze_final_model_certification_source_lock as source_freezer,
)
from experiments.evaluate_tpd_clean_v6_pd_fa import (  # noqa: E402
    adaptive_thresholds_closed_interval,
)
from experiments.evaluate_tpd_clean_v8_mprs_dch_pd_fa import (  # noqa: E402
    configure_v8_inference,
)
from experiments.train_tpd_pilot import (  # noqa: E402
    ValidationSubset,
    resolve_sample_file,
)


REPORT_SCHEMA = "sctransnet_final_model_qfg_six_mode_audit_v1"
ACTION_SCHEMA = "sctransnet_final_model_qfg_six_mode_audit_action_v1"
PREFLIGHT_SCHEMA = "sctransnet_final_model_qfg_six_mode_preflight_v1"
REGION_SCHEMA = "sctransnet_final_model_qfg_region_statistics_v1"
BOOTSTRAP_SCHEMA = "sctransnet_final_model_paired_image_bootstrap_v1"
COMPONENT_SCHEMA = "sctransnet_final_model_component_difference_v1"

PUBLIC_TO_PRIMITIVE_MODE = {
    "full": "full",
    "qfg_off": "all_off",
    "level1_off": "level_1_off",
    "level2_off": "level_2_off",
    "level3_off": "level_3_off",
    "level4_off": "level_4_off",
}
PUBLIC_MODES = tuple(PUBLIC_TO_PRIMITIVE_MODE)
COUNTERFACTUAL_MODES = PUBLIC_MODES[1:]

DEFAULT_PARENT_LOCK = (
    REPO_ROOT / parent_freezer.DEFAULT_OUTPUT_RELATIVE_PATH
)
DEFAULT_SOURCE_LOCK = REPO_ROOT / source_freezer.DEFAULT_OUTPUT_RELATIVE
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "analysis/results/final_model_qfg_six_mode_audit_v1"
)
REPORT_FILENAME = "final_model_qfg_six_mode_audit_v1.json"

EXPECTED_INFERENCE_SHA256 = (
    "997027bb2cc59e0e16ef85beba2c78ab8b3e195de962acbe7c97adc8c007c63a"
)
EXPECTED_SOURCE_CHECKPOINT_SHA256 = (
    "890c8cf0e0f7c3a4c21e5772e69cd89e3038b308a1d77be58365f2254b89b678"
)
EXPECTED_VALIDATION_IDS_SHA256 = (
    "86247e5970f93224c64005e1ac7f3a933bafb37baf279ab71fce5670ae925e06"
)
EXPECTED_VALIDATION_COUNT = 133
FIXED_THRESHOLD = 0.5
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
MATCH_RADIUS = 3.0
TINY_AREA = 9
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260730
SIMULTANEOUS_FAMILY_CI = 0.95
PER_METRIC_TWO_SIDED_CI = 0.99
REPEAT_MAX_ABS_TOLERANCE = 1e-7
METRIC_KEYS = (
    "pd",
    "fa",
    "miou",
    "tiny_pd",
    "false_objects_per_image",
)
EXTRA_THRESHOLDS = (0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, "
            f"observed={observed!r}"
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_object(path: Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular file: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    return value


def _repo_file(repo_root: Path, relative: str, label: str) -> Path:
    root = Path(repo_root).resolve()
    pure = PurePosixPath(relative)
    _require(
        relative == pure.as_posix()
        and not pure.is_absolute()
        and ".." not in pure.parts,
        f"{label} path is not canonical repository-relative: {relative}",
    )
    path = root.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} lies outside repository") from exc
    _require_equal(f"{label} canonical path", resolved, path)
    return path


def _source_bindings(repo_root: Path) -> dict[str, dict[str, str]]:
    root = Path(repo_root).resolve()
    sources = {
        "runner": Path(__file__).resolve(),
        "knockout_core": Path(audit_core.__file__).resolve(),
        "cache_core": Path(cache_core.__file__).resolve(),
        "qfg_evaluator": Path(qfg_evaluator.__file__).resolve(),
        "metric_core": Path(sweep_core.__file__).resolve(),
        "exporter": Path(exporter.__file__).resolve(),
    }
    records: dict[str, dict[str, str]] = {}
    for role, path in sources.items():
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"{role} source lies outside repository") from exc
        records[role] = {
            "path": relative,
            "sha256": cache_core.sha256_file(path),
        }
    return records


@dataclass(frozen=True, slots=True)
class FrozenAuditContext:
    repo_root: Path
    dataset_root: Path
    validation_ids: tuple[str, ...]
    normalization: dict[str, float]
    checkpoint_sha256: str
    source_checkpoint_sha256: str
    dataset_sha256: str
    evaluator_sha256: str
    normalization_sha256: str
    source_lock_sha256: str | None
    validation_ids_sha256: str
    authority_binding: dict[str, Any]
    live_authority_required: bool = True

    def cache_identity(self, public_mode: str) -> dict[str, Any]:
        _require(
            public_mode in PUBLIC_TO_PRIMITIVE_MODE,
            f"unknown public audit mode: {public_mode}",
        )
        _require(
            _is_sha256(self.source_lock_sha256),
            "an actual verified certification source-lock SHA is required "
            "before prediction collection",
        )
        return cache_core.build_cache_identity(
            checkpoint_sha256=self.checkpoint_sha256,
            dataset_sha256=self.dataset_sha256,
            evaluator_sha256=self.evaluator_sha256,
            mode=PUBLIC_TO_PRIMITIVE_MODE[public_mode],
            normalization_sha256=self.normalization_sha256,
            source_lock_sha256=str(self.source_lock_sha256),
            validation_ids_sha256=self.validation_ids_sha256,
            validation_count=len(self.validation_ids),
            match_radius=MATCH_RADIUS,
            tiny_area=TINY_AREA,
        )


def load_frozen_audit_context(
    repo_root: Path = REPO_ROOT,
    parent_lock: Path | None = None,
    source_lock: Path | None = None,
    *,
    require_source_lock: bool = True,
) -> FrozenAuditContext:
    """Validate F0 and return the exact internal-validation audit context."""

    root = Path(repo_root).expanduser().resolve()
    raw_lock_path = (
        root / parent_freezer.DEFAULT_OUTPUT_RELATIVE_PATH
        if parent_lock is None
        else Path(parent_lock).expanduser()
    )
    if not raw_lock_path.is_absolute():
        raw_lock_path = root / raw_lock_path
    if raw_lock_path.is_symlink():
        raise ValueError(f"parent lock must not be a symlink: {raw_lock_path}")
    lock_path = raw_lock_path.resolve()
    try:
        lock_relative = lock_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("parent lock lies outside repository") from exc
    lock = parent_freezer.verify_parent_lock(
        lock_path,
        repo_root=root,
    )
    parent_lock_sha256 = cache_core.sha256_file(lock_path)

    raw_source_lock_path = (
        root / source_freezer.DEFAULT_OUTPUT_RELATIVE
        if source_lock is None
        else Path(source_lock).expanduser()
    )
    if not raw_source_lock_path.is_absolute():
        raw_source_lock_path = root / raw_source_lock_path
    if raw_source_lock_path.is_symlink():
        raise ValueError(
            f"certification source lock must not be a symlink: "
            f"{raw_source_lock_path}"
        )
    source_lock_path = raw_source_lock_path.resolve()
    try:
        source_lock_relative = source_lock_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            "certification source lock lies outside repository"
        ) from exc
    source_lock_sha256: str | None
    source_lock_record: dict[str, Any]
    if source_lock_path.is_file():
        verified_source_lock = source_freezer.verify_source_lock(
            source_lock_path,
            repo_root=root,
        )
        source_lock_sha256 = cache_core.sha256_file(source_lock_path)
        source_lock_record = {
            "path": source_lock_relative,
            "sha256": source_lock_sha256,
            "schema": verified_source_lock["schema"],
            "verified": True,
        }
    else:
        if source_lock_path.exists():
            raise ValueError(
                "certification source lock must be a regular file: "
                f"{source_lock_path}"
            )
        if require_source_lock:
            raise ValueError(
                "formal F1 execution requires the actual certification "
                f"source lock: {source_lock_path}"
            )
        source_lock_sha256 = None
        source_lock_record = {
            "path": source_lock_relative,
            "sha256": None,
            "schema": source_freezer.SCHEMA,
            "verified": False,
            "status": "deferred_missing",
        }
    selected = lock["selected_model"]
    inference = selected["final_inference_artifact"]
    checkpoint = selected["d_training_checkpoint"]
    _require_equal(
        "final inference artifact SHA",
        inference["sha256"],
        EXPECTED_INFERENCE_SHA256,
    )
    _require_equal(
        "D source checkpoint SHA",
        checkpoint["sha256"],
        EXPECTED_SOURCE_CHECKPOINT_SHA256,
    )
    inference_path = _repo_file(
        root,
        inference["path"],
        "final inference artifact",
    )
    source_checkpoint_path = _repo_file(
        root,
        checkpoint["path"],
        "D source checkpoint",
    )
    _require_equal(
        "live final inference artifact SHA",
        cache_core.sha256_file(inference_path),
        EXPECTED_INFERENCE_SHA256,
    )
    _require_equal(
        "live D source checkpoint SHA",
        cache_core.sha256_file(source_checkpoint_path),
        EXPECTED_SOURCE_CHECKPOINT_SHA256,
    )

    authorities = lock["upstream_authorities"]
    split_record = authorities["d_run_split"]
    protocol_record = authorities["d_run_protocol"]
    split_path = _repo_file(root, split_record["path"], "frozen split")
    protocol_path = _repo_file(
        root,
        protocol_record["path"],
        "frozen D protocol",
    )
    split = _json_object(split_path, "frozen split")
    protocol = _json_object(protocol_path, "frozen D protocol")
    sweep_core.validate_identifier_manifest(split)
    validation_ids = tuple(split.get("used_val_ids", ()))
    _require_equal(
        "frozen validation count",
        len(validation_ids),
        EXPECTED_VALIDATION_COUNT,
    )
    validation_ids_sha256 = cache_core.validation_identifier_sha256(
        validation_ids
    )
    _require_equal(
        "frozen validation identifier SHA",
        validation_ids_sha256,
        EXPECTED_VALIDATION_IDS_SHA256,
    )
    _require_equal(
        "split official-test boundary",
        split.get("official_test_accessed"),
        False,
    )
    _require_equal(
        "split source",
        split.get("source"),
        "img_idx/train_NUDT-SIRST.txt",
    )
    normalization = {
        "mean": float(protocol["normalization"]["mean"]),
        "std": float(protocol["normalization"]["std"]),
    }
    _require_equal(
        "normalization",
        normalization,
        lock["data_contract"]["normalization"],
    )
    normalization_sha256 = cache_core.sha256_bytes(
        cache_core.canonical_json_bytes(normalization)
    )
    evaluator_relative = (
        "experiments/evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa.py"
    )
    evaluator_sha256 = lock["frozen_model_source_paths"][
        evaluator_relative
    ]["sha256"]
    _require_equal(
        "live evaluator SHA",
        cache_core.sha256_file(root / evaluator_relative),
        evaluator_sha256,
    )

    dataset_root = root / "datasets" / "NUDT-SIRST"
    _require(dataset_root.is_dir(), "frozen NUDT-SIRST dataset is missing")
    for identifier in validation_ids:
        resolve_sample_file(dataset_root / "images", identifier)
        resolve_sample_file(dataset_root / "masks", identifier)

    authority_binding = {
        "schema": "sctransnet_final_model_qfg_audit_authority_v1",
        "parent_lock": {
            "path": lock_relative,
            "sha256": parent_lock_sha256,
            "schema": lock["schema"],
        },
        "certification_source_lock": source_lock_record,
        "final_inference_artifact": {
            "path": inference["path"],
            "sha256": inference["sha256"],
        },
        "source_training_checkpoint": {
            "path": checkpoint["path"],
            "sha256": checkpoint["sha256"],
            "epoch": checkpoint["epoch"],
            "role": checkpoint["role"],
        },
        "split": {
            "path": split_record["path"],
            "sha256": split_record["sha256"],
            "validation_count": len(validation_ids),
            "validation_ids_sha256": validation_ids_sha256,
        },
        "normalization": {
            **normalization,
            "sha256": normalization_sha256,
        },
        "evaluator": {
            "path": evaluator_relative,
            "sha256": evaluator_sha256,
        },
        "official_test_accessed": False,
    }
    return FrozenAuditContext(
        repo_root=root,
        dataset_root=dataset_root,
        validation_ids=validation_ids,
        normalization=normalization,
        checkpoint_sha256=inference["sha256"],
        source_checkpoint_sha256=checkpoint["sha256"],
        dataset_sha256=lock["data_contract"]["training_data_sha256"],
        evaluator_sha256=evaluator_sha256,
        normalization_sha256=normalization_sha256,
        source_lock_sha256=source_lock_sha256,
        validation_ids_sha256=validation_ids_sha256,
        authority_binding=authority_binding,
        live_authority_required=True,
    )


def build_frozen_validation_loader(
    context: FrozenAuditContext,
) -> DataLoader:
    dataset = ValidationSubset(
        context.dataset_root,
        context.validation_ids,
        dict(context.normalization),
    )
    _require_equal("validation loader size", len(dataset), 133)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def load_final_inference_model_strict(
    context: FrozenAuditContext,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    """Strictly validate and load the exact head-free final artifact."""

    artifact = (
        context.repo_root
        / context.authority_binding["final_inference_artifact"]["path"]
    )
    source_checkpoint = (
        context.repo_root
        / context.authority_binding["source_training_checkpoint"]["path"]
    )
    before = cache_core.sha256_file(artifact)
    _require_equal(
        "final inference artifact before strict load",
        before,
        context.checkpoint_sha256,
    )
    validated = exporter.validate_exported_qfg_checkpoint(
        artifact,
        expected_source_checkpoint=source_checkpoint,
    )
    payload = torch.load(
        artifact,
        map_location="cpu",
        weights_only=False,
    )
    _require(isinstance(payload, Mapping), "inference artifact is not a mapping")
    state_dict = payload.get("state_dict")
    _require(
        isinstance(state_dict, Mapping),
        "inference artifact state_dict is missing",
    )
    model, metadata = exporter.build_frozen_qfg_inference_model()
    incompatible = model.load_state_dict(state_dict, strict=True)
    _require_equal("strict-load missing keys", list(incompatible.missing_keys), [])
    _require_equal(
        "strict-load unexpected keys",
        list(incompatible.unexpected_keys),
        [],
    )
    model.eval()
    model.to(device)
    _require(not model.training, "final model is not in eval mode")
    _require(
        not hasattr(model, "target_survival"),
        "head-free final model unexpectedly retains TSS",
    )
    _require_equal(
        "final inference artifact after strict load",
        cache_core.sha256_file(artifact),
        before,
    )
    return model, {
        **validated,
        "state_dict_strict_load": True,
        "model_eval": True,
        "inference_mode_required": True,
        "device": str(device),
        "model_metadata": metadata,
    }


def _tensor_values(
    value: torch.Tensor | None,
    label: str,
) -> np.ndarray:
    _require(isinstance(value, torch.Tensor), f"prepared QFG {label} missing")
    tensor = value.detach().to(device="cpu", dtype=torch.float64)
    _require(
        tensor.ndim == 4
        and tensor.shape[0] == 1
        and tensor.shape[1] == 1,
        f"prepared QFG {label} must have shape 1x1xHxW",
    )
    array = tensor[0, 0].contiguous().numpy()
    _require(np.isfinite(array).all(), f"prepared QFG {label} is non-finite")
    return array


@dataclass(slots=True)
class SpatialPreparedCapture:
    """Capture forward-local gates/factors and aggregate fixed regions."""

    pending: list[tuple[tuple[np.ndarray, np.ndarray], ...]] = field(
        default_factory=list
    )
    values: dict[
        int,
        dict[str, dict[str, list[np.ndarray]]],
    ] = field(
        default_factory=lambda: {
            level: {
                region: {"gate": [], "factor": []}
                for region in (
                    "global",
                    "target",
                    "hard_negative",
                    "ordinary_background",
                )
            }
            for level in range(4)
        }
    )
    image_count: int = 0

    def append_prepared(self, prepared: Any) -> None:
        levels = getattr(prepared, "levels", None)
        _require(
            isinstance(levels, (tuple, list)) and len(levels) == 4,
            "prepared QFG object must expose four levels",
        )
        self.pending.append(
            tuple(
                (
                    _tensor_values(level.gate, f"level {index + 1} gate"),
                    _tensor_values(level.factor, f"level {index + 1} factor"),
                )
                for index, level in enumerate(levels)
            )
        )

    def consume(
        self,
        target: np.ndarray,
        full_probability: np.ndarray,
    ) -> None:
        _require(bool(self.pending), "no prepared QFG record for forward")
        levels = self.pending.pop(0)
        target_tensor = torch.from_numpy(
            np.asarray(target > 0.5, dtype=np.float32)
        )[None, None]
        false_positive = np.logical_and(
            np.asarray(full_probability) > FIXED_THRESHOLD,
            np.asarray(target) <= 0.5,
        )
        false_tensor = torch.from_numpy(
            false_positive.astype(np.float32)
        )[None, None]
        for index, (gate, factor) in enumerate(levels):
            size = gate.shape
            _require_equal(f"level {index + 1} factor shape", factor.shape, size)
            target_cells = (
                F.adaptive_max_pool2d(target_tensor, size)[0, 0].numpy() > 0
            )
            hard_cells = (
                F.adaptive_max_pool2d(false_tensor, size)[0, 0].numpy() > 0
            )
            hard_cells = np.logical_and(hard_cells, ~target_cells)
            background = np.logical_and(~target_cells, ~hard_cells)
            masks = {
                "global": np.ones(size, dtype=bool),
                "target": target_cells,
                "hard_negative": hard_cells,
                "ordinary_background": background,
            }
            for region, mask in masks.items():
                self.values[index][region]["gate"].append(gate[mask])
                self.values[index][region]["factor"].append(factor[mask])
        self.image_count += 1

    def summary(self) -> dict[str, Any]:
        _require(not self.pending, "unconsumed prepared QFG records remain")
        _require(self.image_count > 0, "no spatial QFG records captured")
        levels: list[dict[str, Any]] = []
        for index in range(4):
            regions: dict[str, Any] = {}
            for region in self.values[index]:
                regions[region] = {
                    name: _distribution_summary(
                        np.concatenate(values)
                        if values
                        else np.empty(0, dtype=np.float64),
                        factor=name == "factor",
                    )
                    for name, values in self.values[index][region].items()
                }
            target_factor = regions["target"]["factor"]["mean"]
            background_factor = regions["ordinary_background"]["factor"]["mean"]
            hard_factor = regions["hard_negative"]["factor"]["mean"]
            target_gate = regions["target"]["gate"]["mean"]
            background_gate = regions["ordinary_background"]["gate"]["mean"]
            hard_gate = regions["hard_negative"]["gate"]["mean"]
            levels.append(
                {
                    "level": index + 1,
                    "regions": regions,
                    "factor_contrasts": {
                        "target_minus_background": _optional_difference(
                            target_factor,
                            background_factor,
                        ),
                        "hard_negative_minus_background": (
                            _optional_difference(
                                hard_factor,
                                background_factor,
                            )
                        ),
                    },
                    "gate_contrasts": {
                        "target_minus_background": _optional_difference(
                            target_gate,
                            background_gate,
                        ),
                        "hard_negative_minus_background": (
                            _optional_difference(
                                hard_gate,
                                background_gate,
                            )
                        ),
                    },
                }
            )
        return {
            "schema": REGION_SCHEMA,
            "status": "complete",
            "image_count": self.image_count,
            "region_definition": {
                "target": (
                    "adaptive_max_pool(target>0.5, gate_grid)>0"
                ),
                "hard_negative": (
                    "non_target gate cell containing a full-mode "
                    "false-positive pixel at threshold 0.5"
                ),
                "ordinary_background": (
                    "neither target nor fixed full-mode hard-negative"
                ),
                "mapping": "adaptive_max_pool_to_each_qfg_level",
                "hard_negative_reference_mode": "full",
            },
            "levels": levels,
        }


def _optional_difference(
    left: float | None,
    right: float | None,
) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _distribution_summary(
    values: np.ndarray,
    *,
    factor: bool,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        result: dict[str, Any] = {
            "count": 0,
            "mean": None,
            "rms": None,
            "p5": None,
            "p50": None,
            "p95": None,
            "minimum": None,
            "maximum": None,
        }
        if factor:
            result["mean_abs_factor_minus_one"] = None
        return result
    _require(np.isfinite(array).all(), "QFG summary values are non-finite")
    p5, p50, p95 = np.quantile(array, (0.05, 0.5, 0.95))
    result = {
        "count": int(array.size),
        "mean": float(array.mean()),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "p5": float(p5),
        "p50": float(p50),
        "p95": float(p95),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }
    if factor:
        result["mean_abs_factor_minus_one"] = float(
            np.mean(np.abs(array - 1.0))
        )
    return result


@contextmanager
def capture_spatial_qfg(
    model: nn.Module,
) -> Iterator[SpatialPreparedCapture]:
    qfg = audit_core._resolve_qfg(model)
    _require(not qfg.training, "spatial QFG capture requires eval mode")
    original = qfg.prepare
    capture = SpatialPreparedCapture()
    had_instance = "prepare" in qfg.__dict__
    previous = qfg.__dict__.get("prepare")

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        prepared = original(*args, **kwargs)
        capture.append_prepared(prepared)
        return prepared

    object.__setattr__(qfg, "prepare", wrapped)
    try:
        yield capture
    finally:
        if had_instance:
            object.__setattr__(qfg, "prepare", previous)
        else:
            object.__delattr__(qfg, "prepare")


def _batch_identity(identifiers: Any) -> str:
    if isinstance(identifiers, str):
        return identifiers
    if isinstance(identifiers, (tuple, list)) and len(identifiers) == 1:
        value = identifiers[0]
        _require(isinstance(value, str), "batch image ID is not a string")
        return value
    raise ValueError("F1 audit requires batch_size=1 image identifiers")


def _batch_size(sizes: Any) -> tuple[int, int]:
    tensor = torch.as_tensor(sizes)
    if tensor.ndim == 2 and tensor.shape == (1, 2):
        return int(tensor[0, 0].item()), int(tensor[0, 1].item())
    if tensor.ndim == 1 and tensor.shape == (2,):
        return int(tensor[0].item()), int(tensor[1].item())
    raise ValueError("F1 audit requires one H/W pair per batch")


@torch.inference_mode()
def collect_mode_cache(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    context: FrozenAuditContext,
    public_mode: str,
    *,
    full_reference: cache_core.PredictionCache | None,
) -> tuple[
    cache_core.PredictionCache,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Collect one complete mode under eval + inference_mode."""

    _require(torch.is_inference_mode_enabled(), "inference_mode is not active")
    model.eval()
    _require(not model.training, "model must remain in eval mode")
    primitive_mode = PUBLIC_TO_PRIMITIVE_MODE[public_mode]
    identity = context.cache_identity(public_mode)
    collector = cache_core.PredictionCacheCollector(
        identity=identity,
        match_radius=MATCH_RADIUS,
        tiny_area=TINY_AREA,
    )
    criterion = nn.BCELoss(reduction="mean")
    reference_by_id = (
        {}
        if full_reference is None
        else {
            record.image_id: record.probability
            for record in full_reference.records
        }
    )
    with audit_core.temporary_qfg_alpha_knockout(
        model,
        primitive_mode,
    ) as knockout:
        with audit_core.capture_qfg_prepared_factors(model) as factors:
            with capture_spatial_qfg(model) as spatial:
                for images, masks, sizes, identifiers in loader:
                    image_id = _batch_identity(identifiers)
                    height, width = _batch_size(sizes)
                    images = images.to(device, non_blocking=False)
                    masks = masks.to(device, non_blocking=False)
                    outputs = model(images)
                    qfg_evaluator._require_legacy_eval_output(outputs)
                    prediction = sweep_core.final_prediction(outputs)[
                        :, :, :height, :width
                    ]
                    target = masks[:, :, :height, :width]
                    probability = (
                        prediction[0, 0]
                        .float()
                        .cpu()
                        .contiguous()
                        .numpy()
                        .astype(np.float32, copy=False)
                    )
                    target_array = (
                        target[0, 0]
                        .float()
                        .cpu()
                        .contiguous()
                        .numpy()
                    )
                    loss = float(
                        criterion(
                            prediction.float(),
                            target.float(),
                        ).item()
                    )
                    collector.append(
                        image_id=image_id,
                        probability=probability,
                        target=target_array,
                        loss=loss,
                    )
                    full_probability = (
                        probability
                        if full_reference is None
                        else reference_by_id[image_id]
                    )
                    spatial.consume(target_array, full_probability)
    cache = collector.seal()
    _require_equal(
        f"{public_mode} complete validation count",
        len(cache.records),
        len(context.validation_ids),
    )
    factor_summary = factors.summary()
    region_summary = spatial.summary()
    _require_equal(
        f"{public_mode} factor forward count",
        factor_summary["forward_count"],
        len(cache.records),
    )
    return cache, knockout, factor_summary, region_summary


@torch.inference_mode()
def collect_repeat_full_cache(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    context: FrozenAuditContext,
) -> cache_core.PredictionCache:
    _require(torch.is_inference_mode_enabled(), "inference_mode is not active")
    model.eval()
    collector = cache_core.PredictionCacheCollector(
        identity=context.cache_identity("full"),
        match_radius=MATCH_RADIUS,
        tiny_area=TINY_AREA,
    )
    criterion = nn.BCELoss(reduction="mean")
    for images, masks, sizes, identifiers in loader:
        image_id = _batch_identity(identifiers)
        height, width = _batch_size(sizes)
        images = images.to(device, non_blocking=False)
        masks = masks.to(device, non_blocking=False)
        outputs = model(images)
        qfg_evaluator._require_legacy_eval_output(outputs)
        prediction = sweep_core.final_prediction(outputs)[
            :, :, :height, :width
        ]
        target = masks[:, :, :height, :width]
        collector.append(
            image_id=image_id,
            probability=(
                prediction[0, 0]
                .float()
                .cpu()
                .contiguous()
                .numpy()
                .astype(np.float32, copy=False)
            ),
            target=target[0, 0].float().cpu().contiguous().numpy(),
            loss=float(
                criterion(prediction.float(), target.float()).item()
            ),
        )
    return collector.seal()


def repeat_inference_audit(
    first: cache_core.PredictionCache,
    second: cache_core.PredictionCache,
) -> dict[str, Any]:
    _require_equal(
        "repeat cache identity",
        first.identity,
        second.identity,
    )
    maximum = 0.0
    total = 0.0
    count = 0
    for left, right in zip(first.records, second.records):
        _require_equal("repeat image ID", left.image_id, right.image_id)
        difference = np.abs(
            left.probability.astype(np.float64)
            - right.probability.astype(np.float64)
        )
        maximum = max(maximum, float(difference.max()))
        total += float(difference.sum())
        count += int(difference.size)
    return {
        "status": "complete",
        "max_abs": maximum,
        "mean_abs": total / count,
        "max_abs_tolerance": REPEAT_MAX_ABS_TOLERANCE,
        "equivalent": maximum <= REPEAT_MAX_ABS_TOLERANCE,
        "same_cache_content_sha256": (
            first.content_sha256 == second.content_sha256
        ),
    }


def fa_budget_scan(
    cache: cache_core.PredictionCache,
    *,
    thresholds_override: Sequence[float] | None = None,
) -> dict[str, Any]:
    probabilities = [record.probability for record in cache.records]
    if thresholds_override is None:
        base = sweep_core.threshold_grid(
            0.01,
            0.99,
            0.01,
            EXTRA_THRESHOLDS,
        )
        thresholds, provenance = adaptive_thresholds_closed_interval(
            probabilities,
            base,
            0.1,
        )
        formal = True
    else:
        thresholds = sorted(
            {float(value) for value in thresholds_override}
            | {FIXED_THRESHOLD, 1.0}
        )
        provenance = {
            "test_override": True,
            "total_unique_threshold_count": len(thresholds),
        }
        formal = False
    points: list[dict[str, Any]] = []
    for threshold in thresholds:
        point = cache_core.recompute_metrics(
            cache,
            threshold=threshold,
        )
        point["threshold"] = float(threshold)
        points.append(point)
    budgets = {
        f"{budget:.10g}": sweep_core.best_point_under_fa(points, budget)
        for budget in FA_BUDGETS
    }
    return {
        "status": "complete",
        "formal_closed_interval_grid": formal,
        "prediction_comparison": "probability > threshold",
        "fa_budgets": list(FA_BUDGETS),
        "budget_points": budgets,
        "threshold_provenance": provenance,
        "threshold_count": len(points),
    }


def component_difference(
    full: cache_core.PredictionCache,
    counterfactual: cache_core.PredictionCache,
    *,
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, Any]:
    audit_core._validate_cache_pair(full, counterfactual)
    rows: list[dict[str, Any]] = []
    totals = {
        "changed_image_count": 0,
        "changed_pixel_count": 0,
        "full_only_component_count": 0,
        "counterfactual_only_component_count": 0,
        "overlapping_full_component_count": 0,
        "overlapping_counterfactual_component_count": 0,
    }
    for left, right in zip(full.records, counterfactual.records):
        full_binary = left.probability > threshold
        other_binary = right.probability > threshold
        changed_pixels = int(np.count_nonzero(full_binary != other_binary))
        full_labels = measure.label(full_binary, connectivity=2)
        other_labels = measure.label(other_binary, connectivity=2)
        full_ids = range(1, int(full_labels.max()) + 1)
        other_ids = range(1, int(other_labels.max()) + 1)
        overlapping_full = sum(
            bool(np.any(other_binary[full_labels == label]))
            for label in full_ids
        )
        overlapping_other = sum(
            bool(np.any(full_binary[other_labels == label]))
            for label in other_ids
        )
        row = {
            "image_id": left.image_id,
            "changed_pixel_count": changed_pixels,
            "full_component_count": int(full_labels.max()),
            "counterfactual_component_count": int(other_labels.max()),
            "full_only_component_count": (
                int(full_labels.max()) - overlapping_full
            ),
            "counterfactual_only_component_count": (
                int(other_labels.max()) - overlapping_other
            ),
            "overlapping_full_component_count": overlapping_full,
            "overlapping_counterfactual_component_count": overlapping_other,
        }
        rows.append(row)
        totals["changed_image_count"] += int(changed_pixels > 0)
        totals["changed_pixel_count"] += changed_pixels
        for field in tuple(totals)[2:]:
            totals[field] += int(row[field])
    return {
        "schema": COMPONENT_SCHEMA,
        "status": "complete",
        "threshold": float(threshold),
        "connectivity": 2,
        "overlap_definition": "at_least_one_shared_positive_pixel",
        **totals,
        "per_image": rows,
    }


def paired_image_bootstrap(
    full: cache_core.PredictionCache,
    counterfactual: cache_core.PredictionCache,
    *,
    threshold: float = FIXED_THRESHOLD,
    replicates: int = BOOTSTRAP_REPLICATES,
    rng_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    audit_core._validate_cache_pair(full, counterfactual)
    _require(
        isinstance(replicates, int)
        and not isinstance(replicates, bool)
        and replicates > 0,
        "bootstrap replicates must be positive",
    )
    full_rows = cache_core.image_sufficient_statistics(
        full,
        threshold=threshold,
    )
    other_rows = cache_core.image_sufficient_statistics(
        counterfactual,
        threshold=threshold,
    )
    point_full = cache_core.aggregate_sufficient_statistics(full_rows)
    point_other = cache_core.aggregate_sufficient_statistics(other_rows)
    point_delta = {
        key: float(point_full[key]) - float(point_other[key])
        for key in METRIC_KEYS
    }
    rng = np.random.default_rng(rng_seed)
    count = len(full_rows)
    samples: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    for _ in range(replicates):
        indices = rng.integers(0, count, size=count).tolist()
        left = cache_core.aggregate_sufficient_statistics(
            full_rows,
            sample_indices=indices,
        )
        right = cache_core.aggregate_sufficient_statistics(
            other_rows,
            sample_indices=indices,
        )
        for key in METRIC_KEYS:
            delta = float(left[key]) - float(right[key])
            if math.isfinite(delta):
                samples[key].append(delta)
    intervals: dict[str, Any] = {}
    for key, values in samples.items():
        _require(bool(values), f"bootstrap produced no finite {key} deltas")
        lower, upper = np.quantile(values, (0.005, 0.995))
        intervals[key] = {
            "delta_orientation": "full_minus_counterfactual",
            "point_delta": point_delta[key],
            "lower": float(lower),
            "upper": float(upper),
            "finite_replicates": len(values),
        }
    return {
        "schema": BOOTSTRAP_SCHEMA,
        "status": "complete",
        "unit": "paired_image",
        "threshold": float(threshold),
        "replicates": replicates,
        "rng_seed": rng_seed,
        "shared_resample_indices": True,
        "metric_family": list(METRIC_KEYS),
        "simultaneous_family_confidence": SIMULTANEOUS_FAMILY_CI,
        "per_metric_two_sided_confidence": PER_METRIC_TWO_SIDED_CI,
        "method": "Bonferroni percentile intervals",
        "intervals": intervals,
    }


def _atomic_create(path: Path, content: bytes) -> None:
    output = Path(path).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace F1 audit JSON: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)


def _cache_binding(
    output_dir: Path,
    metadata_path: Path,
) -> dict[str, str]:
    return {
        "path": metadata_path.relative_to(output_dir).as_posix(),
        "sha256": cache_core.sha256_file(metadata_path),
    }


def _execute_six_mode_audit_staged(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    context: FrozenAuditContext,
    output_dir: Path,
    *,
    model_load_audit: Mapping[str, Any] | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    thresholds_override: Sequence[float] | None = None,
) -> Path:
    """Build and verify all F1 artifacts inside one private staging directory."""

    root = Path(output_dir).expanduser().resolve()
    _require(root.is_dir(), f"F1 staging directory is missing: {root}")
    _require(
        not any(root.iterdir()),
        f"F1 staging directory is not empty: {root}",
    )
    _require(
        _is_sha256(context.source_lock_sha256),
        "F1 execution requires an actual verified certification source lock",
    )
    cache_dir = root / "caches"
    cache_dir.mkdir()
    alpha_before = audit_core.alpha_state_sha256(model)
    caches: dict[str, cache_core.PredictionCache] = {}
    modes: dict[str, dict[str, Any]] = {}

    full, knockout, factors, regions = collect_mode_cache(
        model,
        loader,
        device,
        context,
        "full",
        full_reference=None,
    )
    caches["full"] = full
    metadata = cache_core.write_prediction_cache(full, cache_dir)
    modes["full"] = {
        "public_mode": "full",
        "primitive_mode": "full",
        "cache": _cache_binding(root, metadata),
        "alpha_knockout": knockout,
        "fixed_threshold_metrics": cache_core.recompute_metrics(
            full,
            threshold=FIXED_THRESHOLD,
        ),
        "fa_budget_scan": fa_budget_scan(
            full,
            thresholds_override=thresholds_override,
        ),
        "factor_summary": factors,
        "factor_gate_region_statistics": regions,
        "comparison_to_full": None,
        "component_difference": None,
        "paired_image_bootstrap": None,
    }
    repeat = collect_repeat_full_cache(model, loader, device, context)
    repeat_audit = repeat_inference_audit(full, repeat)

    for public_mode in COUNTERFACTUAL_MODES:
        cache, knockout, factors, regions = collect_mode_cache(
            model,
            loader,
            device,
            context,
            public_mode,
            full_reference=full,
        )
        caches[public_mode] = cache
        metadata = cache_core.write_prediction_cache(cache, cache_dir)
        probability_audit = audit_core.audit_probability_caches(
            full,
            cache,
            threshold=FIXED_THRESHOLD,
        )
        modes[public_mode] = {
            "public_mode": public_mode,
            "primitive_mode": PUBLIC_TO_PRIMITIVE_MODE[public_mode],
            "cache": _cache_binding(root, metadata),
            "alpha_knockout": knockout,
            "fixed_threshold_metrics": cache_core.recompute_metrics(
                cache,
                threshold=FIXED_THRESHOLD,
            ),
            "fa_budget_scan": fa_budget_scan(
                cache,
                thresholds_override=thresholds_override,
            ),
            "factor_summary": factors,
            "factor_gate_region_statistics": regions,
            "comparison_to_full": probability_audit,
            "component_difference": component_difference(full, cache),
            "paired_image_bootstrap": paired_image_bootstrap(
                full,
                cache,
                replicates=bootstrap_replicates,
            ),
        }
    _require_equal(
        "QFG alpha restoration after all modes",
        audit_core.alpha_state_sha256(model),
        alpha_before,
    )
    qfg_off = modes["qfg_off"]["comparison_to_full"]
    full_factor = modes["full"]["factor_summary"]
    functional = {
        "status": "complete",
        "repeat_inference_equivalent": repeat_audit["equivalent"],
        "full_vs_qfg_off_functionally_different": (
            qfg_off["output_difference"]["functionally_different"]
        ),
        "nontrivial_factor_use": full_factor["nontrivial_factor_use"],
        "qfg_functionally_active": (
            repeat_audit["equivalent"]
            and qfg_off["output_difference"]["functionally_different"]
            and full_factor["nontrivial_factor_use"]
        ),
        "performance_causal_claim_established": False,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "scope": "internal_validation_same_checkpoint_counterfactual",
        "official_test_accessed": False,
        "authority": context.authority_binding,
        "live_authority_required": context.live_authority_required,
        "source_bindings": _source_bindings(context.repo_root),
        "execution_contract": {
            "model_eval": True,
            "torch_inference_mode": True,
            "batch_size": 1,
            "shuffle": False,
            "workers": 0,
            "validation_count": len(context.validation_ids),
            "validation_ids_sha256": context.validation_ids_sha256,
            "checkpoint_sha256": context.checkpoint_sha256,
            "source_checkpoint_sha256": context.source_checkpoint_sha256,
            "certification_source_lock_sha256": (
                context.source_lock_sha256
            ),
            "parent_lock_sha256": context.authority_binding.get(
                "parent_lock",
                {},
            ).get("sha256"),
            "prediction_comparison": "probability > threshold",
            "fixed_threshold": FIXED_THRESHOLD,
            "fa_budgets": list(FA_BUDGETS),
            "match_radius": MATCH_RADIUS,
            "tiny_area": TINY_AREA,
            "modes": list(PUBLIC_MODES),
            "derived_checkpoint_written": False,
        },
        "model_load_audit": (
            dict(model_load_audit)
            if model_load_audit is not None
            else {
                "state_dict_strict_load": True,
                "model_eval": True,
                "inference_mode_required": True,
                "test_fixture": True,
            }
        ),
        "repeat_inference": repeat_audit,
        "modes": modes,
        "functional_gate": functional,
        "claim_boundary": {
            "diagnostic_only": True,
            "qfg_training_causal_contribution_supported": False,
            "paper_core_established_changed": False,
            "stability_claim_supported_changed": False,
            "official_test_claim": False,
        },
        "write_once": True,
        "overwrite_forbidden": True,
    }
    report_path = root / REPORT_FILENAME
    _atomic_create(report_path, canonical_json_bytes(report))
    verify_audit_report(
        report_path,
        repo_root=context.repo_root,
        expected_context=context,
    )
    return report_path


def execute_six_mode_audit(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    context: FrozenAuditContext,
    output_dir: Path,
    *,
    model_load_audit: Mapping[str, Any] | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    thresholds_override: Sequence[float] | None = None,
) -> Path:
    """Run all six modes and publish only after staged self-verification."""

    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        raise ValueError(
            f"F1 output directory must not be a symlink: {requested}"
        )
    root = requested.resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"refusing existing F1 output directory: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=root.parent,
            prefix=f".{root.name}.staging-",
        )
    )
    root_claimed = False
    try:
        staged_report = _execute_six_mode_audit_staged(
            model,
            loader,
            device,
            context,
            staging,
            model_load_audit=model_load_audit,
            bootstrap_replicates=bootstrap_replicates,
            thresholds_override=thresholds_override,
        )
        # Claim the final path without replacing any path created concurrently.
        root.mkdir()
        root_claimed = True
        os.rename(staging / "caches", root / "caches")
        os.rename(staged_report, root / REPORT_FILENAME)
        staging.rmdir()
        final_report = root / REPORT_FILENAME
        verify_audit_report(
            final_report,
            repo_root=context.repo_root,
            expected_context=context,
        )
        return final_report
    except Exception:
        if root_claimed and root.is_dir() and not root.is_symlink():
            shutil.rmtree(root)
        if staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise


def _safe_report_child(report_path: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    _require(
        relative == pure.as_posix()
        and not pure.is_absolute()
        and ".." not in pure.parts,
        f"report child path is invalid: {relative}",
    )
    path = report_path.parent.joinpath(*pure.parts)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"report child is not a regular file: {path}")
    _require(
        path.resolve().is_relative_to(report_path.parent.resolve()),
        "report child escapes report directory",
    )
    return path


def verify_audit_report(
    report_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    parent_lock: Path | None = None,
    source_lock: Path | None = None,
    expected_context: FrozenAuditContext | None = None,
) -> dict[str, Any]:
    unresolved = Path(report_path).expanduser()
    if unresolved.is_symlink():
        raise ValueError(
            f"audit report must not be a symlink: {unresolved}"
        )
    path = unresolved.resolve()
    if not path.is_file():
        raise ValueError(f"audit report must be a regular file: {path}")
    raw = path.read_bytes()
    report = _json_object(path, "F1 audit report")
    _require_equal("F1 report canonical bytes", raw, canonical_json_bytes(report))
    _require_equal("F1 report schema", report.get("schema"), REPORT_SCHEMA)
    _require_equal("F1 report status", report.get("status"), "complete")
    _require_equal(
        "F1 report official-test boundary",
        report.get("official_test_accessed"),
        False,
    )
    _require_equal(
        "F1 report mode set",
        set(report.get("modes", {})),
        set(PUBLIC_MODES),
    )
    context = expected_context
    if report.get("live_authority_required") is True:
        expected_authority = (
            {}
            if expected_context is None
            else expected_context.authority_binding
        )
        effective_parent_lock = parent_lock
        if effective_parent_lock is None and expected_authority:
            effective_parent_lock = (
                Path(repo_root)
                / expected_authority["parent_lock"]["path"]
            )
        effective_source_lock = source_lock
        if effective_source_lock is None and expected_authority:
            effective_source_lock = (
                Path(repo_root)
                / expected_authority["certification_source_lock"]["path"]
            )
        live_context = load_frozen_audit_context(
            repo_root,
            effective_parent_lock,
            effective_source_lock,
            require_source_lock=True,
        )
        if expected_context is not None:
            _require_equal(
                "F1 supplied/live authority",
                expected_context.authority_binding,
                live_context.authority_binding,
            )
        context = live_context
    _require(
        context is not None,
        "non-live report verification requires an explicit expected context",
    )
    _require_equal(
        "F1 authority binding",
        report.get("authority"),
        context.authority_binding,
    )
    _require_equal(
        "F1 source bindings",
        report.get("source_bindings"),
        _source_bindings(context.repo_root),
    )
    contract = report.get("execution_contract", {})
    expected_count = len(context.validation_ids)
    _require_equal(
        "F1 validation count",
        contract.get("validation_count"),
        expected_count,
    )
    _require_equal(
        "F1 validation ID SHA",
        contract.get("validation_ids_sha256"),
        context.validation_ids_sha256,
    )
    _require_equal(
        "F1 inference artifact SHA",
        contract.get("checkpoint_sha256"),
        context.checkpoint_sha256,
    )
    _require_equal(
        "F1 source checkpoint SHA",
        contract.get("source_checkpoint_sha256"),
        context.source_checkpoint_sha256,
    )
    _require(
        _is_sha256(context.source_lock_sha256),
        "expected context lacks a verified certification source-lock SHA",
    )
    _require_equal(
        "F1 certification source-lock SHA",
        contract.get("certification_source_lock_sha256"),
        context.source_lock_sha256,
    )
    _require_equal(
        "F1 parent-lock SHA",
        contract.get("parent_lock_sha256"),
        context.authority_binding.get("parent_lock", {}).get("sha256"),
    )
    if context.live_authority_required:
        _require_equal(
            "F1 frozen validation count",
            expected_count,
            EXPECTED_VALIDATION_COUNT,
        )
        _require_equal(
            "F1 frozen validation ID SHA",
            context.validation_ids_sha256,
            EXPECTED_VALIDATION_IDS_SHA256,
        )
        _require_equal(
            "F1 frozen inference artifact SHA",
            context.checkpoint_sha256,
            EXPECTED_INFERENCE_SHA256,
        )
        _require_equal(
            "F1 frozen source checkpoint SHA",
            context.source_checkpoint_sha256,
            EXPECTED_SOURCE_CHECKPOINT_SHA256,
        )
    _require_equal(
        "F1 mode order",
        contract.get("modes"),
        list(PUBLIC_MODES),
    )
    _require_equal(
        "F1 Fa budgets",
        contract.get("fa_budgets"),
        list(FA_BUDGETS),
    )
    _require_equal(
        "F1 no derived checkpoint",
        contract.get("derived_checkpoint_written"),
        False,
    )
    for public_mode in PUBLIC_MODES:
        mode = report["modes"][public_mode]
        _require_equal("public mode name", mode.get("public_mode"), public_mode)
        _require_equal(
            "primitive mode name",
            mode.get("primitive_mode"),
            PUBLIC_TO_PRIMITIVE_MODE[public_mode],
        )
        binding = mode.get("cache", {})
        metadata_path = _safe_report_child(path, binding.get("path", ""))
        _require_equal(
            f"{public_mode} cache metadata SHA",
            cache_core.sha256_file(metadata_path),
            binding.get("sha256"),
        )
        expected_identity = (
            None if context is None else context.cache_identity(public_mode)
        )
        cache = cache_core.load_prediction_cache(
            metadata_path,
            expected_identity=expected_identity,
        )
        _require_equal(
            f"{public_mode} cache image count",
            len(cache.records),
            expected_count,
        )
        _require_equal(
            f"{public_mode} fixed metrics",
            mode.get("fixed_threshold_metrics"),
            cache_core.recompute_metrics(cache, threshold=FIXED_THRESHOLD),
        )
        scan = mode.get("fa_budget_scan", {})
        _require_equal(
            f"{public_mode} budget keys",
            set(scan.get("budget_points", {})),
            {f"{budget:.10g}" for budget in FA_BUDGETS},
        )
        region = mode.get("factor_gate_region_statistics", {})
        _require_equal(f"{public_mode} region status", region.get("status"), "complete")
        _require_equal(
            f"{public_mode} region image count",
            region.get("image_count"),
            expected_count,
        )
    repeat = report.get("repeat_inference", {})
    _require_equal("repeat inference status", repeat.get("status"), "complete")
    gate = report.get("functional_gate", {})
    _require_equal("functional gate status", gate.get("status"), "complete")
    _require_equal(
        "functional gate causal boundary",
        gate.get("performance_causal_claim_established"),
        False,
    )
    return report


def preflight(
    repo_root: Path = REPO_ROOT,
    parent_lock: Path | None = None,
    source_lock: Path | None = None,
) -> dict[str, Any]:
    context = load_frozen_audit_context(
        repo_root,
        parent_lock,
        source_lock,
        require_source_lock=False,
    )
    source_ready = _is_sha256(context.source_lock_sha256)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "ready" if source_ready else "source_lock_pending",
        "action": "preflight",
        "official_test_accessed": False,
        "validation_count": len(context.validation_ids),
        "validation_ids_sha256": context.validation_ids_sha256,
        "final_inference_artifact_sha256": context.checkpoint_sha256,
        "source_checkpoint_sha256": context.source_checkpoint_sha256,
        "cache_identity_fields": {
            "dataset_sha256": context.dataset_sha256,
            "evaluator_sha256": context.evaluator_sha256,
            "normalization_sha256": context.normalization_sha256,
            "source_lock_sha256": context.source_lock_sha256,
        },
        "parent_lock": context.authority_binding["parent_lock"],
        "certification_source_lock": context.authority_binding[
            "certification_source_lock"
        ],
        "cache_identity_ready": source_ready,
        "modes": list(PUBLIC_MODES),
        "strict_model_load_deferred_to_run": True,
        "gpu_used": False,
        "writes_performed": False,
    }


def run_formal(
    *,
    repo_root: Path,
    parent_lock: Path | None,
    source_lock: Path | None,
    output_dir: Path,
    device_name: str,
) -> Path:
    context = load_frozen_audit_context(
        repo_root,
        parent_lock,
        source_lock,
        require_source_lock=True,
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    determinism = configure_v8_inference(device_name)
    model, load_audit = load_final_inference_model_strict(context, device)
    load_audit["determinism"] = determinism
    loader = build_frozen_validation_loader(context)
    return execute_six_mode_audit(
        model,
        loader,
        device,
        context,
        output_dir,
        model_load_audit=load_audit,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--parent-lock", type=Path)
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(
        list(sys.argv[1:] if argv is None else argv)
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.preflight:
        result = preflight(
            args.repo_root,
            args.parent_lock,
            args.source_lock,
        )
    elif args.run:
        report = run_formal(
            repo_root=args.repo_root,
            parent_lock=args.parent_lock,
            source_lock=args.source_lock,
            output_dir=args.output_dir,
            device_name=args.device,
        )
        result = {
            "schema": ACTION_SCHEMA,
            "status": "complete",
            "action": "run",
            "report": str(report),
            "report_sha256": cache_core.sha256_file(report),
            "verified": True,
        }
    else:
        report = (
            args.report
            if args.report is not None
            else args.output_dir / REPORT_FILENAME
        )
        verify_audit_report(
            report,
            repo_root=args.repo_root,
            parent_lock=args.parent_lock,
            source_lock=args.source_lock,
        )
        result = {
            "schema": ACTION_SCHEMA,
            "status": "complete",
            "action": "verify",
            "report": str(Path(report).resolve()),
            "report_sha256": cache_core.sha256_file(report),
            "verified": True,
        }
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "COUNTERFACTUAL_MODES",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SOURCE_LOCK",
    "FA_BUDGETS",
    "FIXED_THRESHOLD",
    "FrozenAuditContext",
    "PUBLIC_MODES",
    "PUBLIC_TO_PRIMITIVE_MODE",
    "REPORT_FILENAME",
    "REPORT_SCHEMA",
    "SpatialPreparedCapture",
    "build_frozen_validation_loader",
    "capture_spatial_qfg",
    "collect_mode_cache",
    "component_difference",
    "execute_six_mode_audit",
    "fa_budget_scan",
    "load_final_inference_model_strict",
    "load_frozen_audit_context",
    "main",
    "paired_image_bootstrap",
    "parse_args",
    "preflight",
    "repeat_inference_audit",
    "run_formal",
    "verify_audit_report",
]


if __name__ == "__main__":
    main()
