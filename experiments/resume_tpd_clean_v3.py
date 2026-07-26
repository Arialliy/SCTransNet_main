#!/usr/bin/env python3
"""Resume an interrupted TPD-Clean-v3 run at an evaluated epoch boundary.

The original training sources remain untouched.  This engine validates the
existing run as a closed prefix, restores the model/Adam/scaler payload, and
replays the original ``workers=0`` training loader without optimization so the
shuffle/crop/flip data stream reaches the next absolute epoch.  CUDA
forward/backward randomness is not recoverable from the legacy checkpoint, so
the provenance explicitly does not claim bitwise process continuity.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import train_tpd_pilot as base  # noqa: E402
from experiments.train_tpd_clean_v3 import build_clean_v3_model  # noqa: E402
from model.tpd_clean_v3 import SUPPORTED_CLEAN_V3_VARIANTS  # noqa: E402


PROVENANCE_NAME = "resume_provenance.json"
SEGMENTS_NAME = "resume_segments.jsonl"
PROVENANCE_SCHEMA = "sctransnet_tpd_clean_v3_resume_provenance_v1"
SEGMENT_SCHEMA = "sctransnet_tpd_clean_v3_resume_segment_v1"
ENGINE_SCHEMA = "sctransnet_tpd_clean_v3_resume_engine_v1"

PROTOCOL_ARGUMENT_KEYS = frozenset(
    {
        "variant",
        "dataset",
        "dataset_dir",
        "output_root",
        "run_tag",
        "device",
        "epochs",
        "batch_size",
        "patch_size",
        "workers",
        "seed",
        "split_seed",
        "val_fraction",
        "eval_every",
        "base_lr",
        "min_lr",
        "warmup_epochs",
        "threshold",
        "match_radius",
        "tiny_area",
        "amp",
        "max_train_images",
        "max_val_images",
    }
)

VALIDATION_METRIC_KEYS = (
    "val_loss",
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pd",
    "tiny_pd",
    "fa",
    "false_objects_per_image",
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)
EVENT_NUMERIC_KEYS = (
    "train_loss",
    "learning_rate",
    "processed_train_samples",
    "epoch_seconds",
    *VALIDATION_METRIC_KEYS,
)
CHECKPOINT_REQUIRED_KEYS = frozenset(
    {
        "epoch",
        "variant",
        "dataset",
        "seed",
        "split_seed",
        "state_dict",
        "optimizer",
        "scaler",
        "validation_metrics",
        "model_metadata",
        "split_hashes",
        "selection_source",
        "official_test_accessed",
        "checkpoint_role",
    }
)


class ResumeValidationError(ValueError):
    """The interrupted run is not a valid resume prefix."""


def _fail(message: str) -> None:
    raise ResumeValidationError(message)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _reject_json_constant(value: str) -> None:
    _fail(f"JSON contains a non-finite constant: {value}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except ResumeValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"{label}: cannot parse JSON: {exc}")
    if not isinstance(payload, dict):
        _fail(f"{label}: top level must be an object")
    return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _json_ready(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        _fail(f"{label}: missing regular file: {path}")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.resume.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    content = _canonical_json_bytes(payload)
    _atomic_write_bytes(path, content)
    return _sha256_bytes(content)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.resume.{os.getpid()}.tmp"
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> str:
    content = _canonical_json_bytes(payload)
    with path.open("ab") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha256_bytes(content)


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _require_regular_file(path, label)
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        _fail(f"{label}: cannot read: {exc}")
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            _fail(f"{label}: blank line at {line_number}")
        try:
            payload = json.loads(line, parse_constant=_reject_json_constant)
        except ResumeValidationError:
            raise
        except json.JSONDecodeError as exc:
            _fail(f"{label}: invalid JSON at line {line_number}: {exc}")
        if not isinstance(payload, dict):
            _fail(f"{label}: line {line_number} must be an object")
        rows.append(payload)
    return rows


def _load_checkpoint(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # PyTorch exposes several serialization errors.
        _fail(f"{label}: cannot load checkpoint: {exc}")
    if not isinstance(payload, dict):
        _fail(f"{label}: checkpoint must be a mapping")
    return payload


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        _fail(f"{label}: expected numeric value, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{label}: non-finite value")
    return result


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float, np.number)) and isinstance(
        right, (int, float, np.number)
    ):
        return float(left) == float(right)
    return _json_ready(left) == _json_ready(right)


def _extract_validation_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in VALIDATION_METRIC_KEYS:
        if key not in row:
            _fail(f"metrics event is missing validation field {key}")
        metrics[key] = copy.deepcopy(row[key])
    return metrics


@dataclass
class SelectionState:
    best_pd_key: tuple[float, float, float, float, float]
    best_pd_epoch: int
    best_pd_metrics: dict[str, Any]
    best_miou_key: tuple[float, float, float, float, float]
    best_miou_epoch: int
    best_miou_metrics: dict[str, Any]


def rebuild_best_selection(rows: Sequence[Mapping[str, Any]]) -> SelectionState:
    best_pd_key = (-float("inf"),) * 5
    best_pd_epoch = 0
    best_pd_metrics: dict[str, Any] = {}
    best_miou_key = (-float("inf"),) * 5
    best_miou_epoch = 0
    best_miou_metrics: dict[str, Any] = {}
    for row in rows:
        metrics = _extract_validation_metrics(row)
        epoch = int(row["epoch"])
        pd_key = base.pd_selection_key(metrics)
        miou_key = base.miou_selection_key(metrics)
        if pd_key > best_pd_key:
            best_pd_key = pd_key
            best_pd_epoch = epoch
            best_pd_metrics = metrics
        if miou_key > best_miou_key:
            best_miou_key = miou_key
            best_miou_epoch = epoch
            best_miou_metrics = metrics
    if not best_pd_epoch or not best_miou_epoch:
        _fail("metrics history has no evaluated epoch")
    return SelectionState(
        best_pd_key=best_pd_key,
        best_pd_epoch=best_pd_epoch,
        best_pd_metrics=best_pd_metrics,
        best_miou_key=best_miou_key,
        best_miou_epoch=best_miou_epoch,
        best_miou_metrics=best_miou_metrics,
    )


def _validate_protocol(
    protocol: Mapping[str, Any], run_dir: Path
) -> dict[str, Any]:
    arguments = protocol.get("arguments")
    if not isinstance(arguments, dict):
        _fail("protocol.arguments must be an object")
    if set(arguments) != PROTOCOL_ARGUMENT_KEYS:
        _fail(
            "protocol argument keys changed: "
            f"missing={sorted(PROTOCOL_ARGUMENT_KEYS - set(arguments))} "
            f"extra={sorted(set(arguments) - PROTOCOL_ARGUMENT_KEYS)}"
        )
    variant = arguments["variant"]
    if variant not in SUPPORTED_CLEAN_V3_VARIANTS:
        _fail(f"unsupported protocol variant: {variant!r}")
    if arguments["dataset"] != "NUDT-SIRST":
        _fail(f"unexpected protocol dataset: {arguments['dataset']!r}")
    integer_positive = ("epochs", "batch_size", "patch_size", "eval_every")
    for key in integer_positive:
        if isinstance(arguments[key], bool) or not isinstance(arguments[key], int):
            _fail(f"protocol argument {key} must be an integer")
        if arguments[key] < 1:
            _fail(f"protocol argument {key} must be positive")
    if arguments["workers"] != 0:
        _fail("strict data-stream replay requires protocol workers=0")
    if arguments["eval_every"] != 1:
        _fail("resume engine requires eval_every=1 for epoch-boundary checkpoints")
    for key in (
        "base_lr",
        "min_lr",
        "val_fraction",
        "threshold",
        "match_radius",
    ):
        _finite_number(arguments[key], f"protocol.arguments.{key}")
    if not isinstance(arguments["seed"], int) or not isinstance(
        arguments["split_seed"], int
    ):
        _fail("protocol seed and split_seed must be integers")
    if not isinstance(arguments["amp"], bool):
        _fail("protocol amp must be boolean")
    expected_run_dir = (
        Path(str(arguments["output_root"])).resolve()
        / str(arguments["dataset"])
        / str(variant)
        / f"seed_{arguments['seed']}_{arguments['run_tag']}"
    )
    if expected_run_dir != run_dir:
        _fail(
            "protocol arguments do not derive the requested run directory: "
            f"{expected_run_dir} != {run_dir}"
        )
    if Path(str(protocol.get("run_directory", ""))).resolve() != run_dir:
        _fail("protocol.run_directory does not match --run-dir")
    normalization = protocol.get("normalization")
    if not isinstance(normalization, dict) or set(normalization) != {"mean", "std"}:
        _fail("protocol.normalization must contain exactly mean/std")
    _finite_number(normalization["mean"], "protocol.normalization.mean")
    if _finite_number(normalization["std"], "protocol.normalization.std") <= 0:
        _fail("protocol normalization std must be positive")
    if not isinstance(protocol.get("model"), dict):
        _fail("protocol.model must be an object")
    if protocol["model"].get("variant") != variant:
        _fail("protocol model variant differs from protocol arguments")
    return arguments


def _validate_split(
    split: Mapping[str, Any],
    arguments: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    expected_scalars = {
        "dataset": arguments["dataset"],
        "split_seed": arguments["split_seed"],
        "val_fraction": arguments["val_fraction"],
    }
    for key, expected in expected_scalars.items():
        if not _same_value(split.get(key), expected):
            _fail(f"split.{key} differs from protocol arguments")
    train_ids = split.get("used_train_ids")
    val_ids = split.get("used_val_ids")
    if not isinstance(train_ids, list) or not all(
        isinstance(item, str) for item in train_ids
    ):
        _fail("split.used_train_ids must be a string list")
    if not isinstance(val_ids, list) or not all(
        isinstance(item, str) for item in val_ids
    ):
        _fail("split.used_val_ids must be a string list")
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        _fail("split used identifiers contain duplicates")
    if set(train_ids) & set(val_ids):
        _fail("split train/validation identifiers overlap")
    if split.get("used_train_count") != len(train_ids):
        _fail("split used_train_count differs from identifier list")
    if split.get("used_val_count") != len(val_ids):
        _fail("split used_val_count differs from identifier list")
    hashes = split.get("hashes")
    if not isinstance(hashes, dict):
        _fail("split.hashes must be an object")
    recomputed = {
        "used_train_sha256": base.identifier_hash(train_ids),
        "used_val_sha256": base.identifier_hash(val_ids),
    }
    for key, expected in recomputed.items():
        if hashes.get(key) != expected:
            _fail(f"split hash mismatch for {key}")
    checkpoint_hashes = checkpoint.get("split_hashes")
    if checkpoint_hashes != hashes:
        _fail("checkpoint split_hashes differ from split.json")


def validate_metrics_history(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint: Mapping[str, Any],
    arguments: Mapping[str, Any],
    split: Mapping[str, Any],
    expected_resume_epoch: int,
) -> None:
    checkpoint_epoch = checkpoint.get("epoch")
    if checkpoint_epoch != expected_resume_epoch:
        _fail(
            f"last checkpoint epoch={checkpoint_epoch!r}, "
            f"expected={expected_resume_epoch}"
        )
    if len(rows) != expected_resume_epoch:
        _fail(
            f"metrics row count={len(rows)}, checkpoint epoch={expected_resume_epoch}"
        )
    variant = arguments["variant"]
    for expected_epoch, row in enumerate(rows, start=1):
        if row.get("epoch") != expected_epoch:
            _fail(
                f"metrics epochs are not contiguous at row {expected_epoch}: "
                f"{row.get('epoch')!r}"
            )
        if row.get("variant") != variant:
            _fail(
                f"metrics variant mismatch at epoch {expected_epoch}: "
                f"{row.get('variant')!r} != {variant!r}"
            )
        for key in EVENT_NUMERIC_KEYS:
            _finite_number(row.get(key), f"metrics[{expected_epoch}].{key}")
        expected_lr = base.learning_rate_for_epoch(
            expected_epoch,
            int(arguments["epochs"]),
            float(arguments["base_lr"]),
            float(arguments["min_lr"]),
            int(arguments["warmup_epochs"]),
        )
        observed_lr = float(row["learning_rate"])
        if not math.isclose(observed_lr, expected_lr, rel_tol=0.0, abs_tol=1e-15):
            _fail(
                f"metrics learning rate mismatch at epoch {expected_epoch}: "
                f"{observed_lr} != {expected_lr}"
            )
        if int(row["processed_train_samples"]) != int(
            split["used_train_count"]
        ):
            _fail(
                f"metrics processed_train_samples mismatch at epoch {expected_epoch}"
            )
    if not rows:
        _fail("metrics history is empty")
    validation_metrics = checkpoint.get("validation_metrics")
    if not isinstance(validation_metrics, dict):
        _fail("last checkpoint validation_metrics must be an object")
    latest = _extract_validation_metrics(rows[-1])
    if set(validation_metrics) != set(latest):
        _fail("last checkpoint validation metric keys differ from latest event")
    for key, expected in latest.items():
        if not _same_value(validation_metrics.get(key), expected):
            _fail(
                f"last checkpoint validation metric {key} differs from metrics"
            )


def _validate_checkpoint_identity(
    checkpoint: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    arguments: Mapping[str, Any],
    role: str,
) -> None:
    missing = CHECKPOINT_REQUIRED_KEYS - set(checkpoint)
    if missing:
        _fail(f"checkpoint is missing required keys: {sorted(missing)}")
    expected = {
        "variant": arguments["variant"],
        "dataset": arguments["dataset"],
        "seed": arguments["seed"],
        "split_seed": arguments["split_seed"],
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
        "checkpoint_role": role,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            _fail(
                f"checkpoint {key}={checkpoint.get(key)!r}, expected={value!r}"
            )
    if checkpoint.get("model_metadata") != protocol.get("model"):
        _fail("checkpoint model metadata differs from protocol")
    optimizer = checkpoint.get("optimizer")
    if not isinstance(optimizer, dict) or set(optimizer) != {
        "state",
        "param_groups",
    }:
        _fail("checkpoint optimizer is not an Adam-compatible state dict")
    if not isinstance(checkpoint.get("scaler"), dict):
        _fail("checkpoint scaler must be an object")
    if not isinstance(checkpoint.get("state_dict"), dict):
        _fail("checkpoint state_dict must be an object")


def _validate_best_checkpoint(
    path: Path,
    *,
    expected_epoch: int,
    expected_metrics: Mapping[str, Any],
    expected_role: str,
    protocol: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = _load_checkpoint(path, path.name)
    _validate_checkpoint_identity(
        checkpoint,
        protocol=protocol,
        arguments=arguments,
        role=expected_role,
    )
    if checkpoint.get("epoch") != expected_epoch:
        _fail(
            f"{path.name} epoch={checkpoint.get('epoch')!r}, "
            f"recomputed={expected_epoch}"
        )
    observed = checkpoint.get("validation_metrics")
    if not isinstance(observed, dict) or set(observed) != set(expected_metrics):
        _fail(f"{path.name} validation metric keys differ from history")
    for key, value in expected_metrics.items():
        if not _same_value(observed.get(key), value):
            _fail(f"{path.name} metric {key} differs from history")
    return checkpoint


def _validate_existing_resume_binding(
    run_dir: Path, checkpoint: Mapping[str, Any]
) -> None:
    digest = checkpoint.get("resume_provenance_sha256")
    provenance_path = run_dir / PROVENANCE_NAME
    segments_path = run_dir / SEGMENTS_NAME
    if digest is None:
        if provenance_path.exists() or segments_path.exists():
            _fail(
                "legacy checkpoint lacks provenance but resume provenance "
                "artifacts already exist"
            )
        return
    if not isinstance(digest, str) or len(digest) != 64:
        _fail("checkpoint resume provenance digest is malformed")
    _require_regular_file(provenance_path, "resume provenance")
    if _sha256_file(provenance_path) != digest:
        _fail("checkpoint resume provenance digest does not match current file")
    provenance = _load_json(provenance_path, "resume provenance")
    if provenance.get("schema") != PROVENANCE_SCHEMA:
        _fail("resume provenance schema mismatch")
    _require_regular_file(segments_path, "resume segments")
    if provenance.get("segments_sha256") != _sha256_file(segments_path):
        _fail("resume provenance does not bind current segments file")
    segment_index = checkpoint.get("resume_segment_index")
    segment_digest = checkpoint.get("resume_segment_sha256")
    segments = _load_jsonl(segments_path, "resume segments")
    if not isinstance(segment_index, int) or not 1 <= segment_index <= len(segments):
        _fail("checkpoint resume segment index is invalid")
    actual_digest = _sha256_bytes(
        _canonical_json_bytes(segments[segment_index - 1])
    )
    if segment_digest != actual_digest:
        _fail("checkpoint resume segment digest does not match JSONL record")


@dataclass
class ResumeState:
    run_dir: Path
    protocol: dict[str, Any]
    arguments: dict[str, Any]
    split: dict[str, Any]
    rows: list[dict[str, Any]]
    last_checkpoint: dict[str, Any]
    best_pd_checkpoint: dict[str, Any]
    best_miou_checkpoint: dict[str, Any]
    selection: SelectionState
    completed_epoch: int
    target_epoch: int
    source_checkpoint_sha256: str


def load_resume_state(
    run_dir: Path,
    *,
    expected_resume_epoch: int,
    target_epoch: int,
) -> ResumeState:
    run_dir = run_dir.resolve(strict=True)
    if not run_dir.is_dir() or run_dir.is_symlink():
        _fail(f"run directory is not a regular directory: {run_dir}")
    protocol = _load_json(run_dir / "protocol.json", "protocol")
    arguments = _validate_protocol(protocol, run_dir)
    if not isinstance(expected_resume_epoch, int) or expected_resume_epoch < 1:
        _fail("--expected-resume-epoch must be a positive integer")
    if not isinstance(target_epoch, int):
        _fail("--target-epoch must be an integer")
    if target_epoch <= expected_resume_epoch:
        _fail(
            f"target epoch {target_epoch} must exceed resume epoch "
            f"{expected_resume_epoch}"
        )
    if target_epoch > int(arguments["epochs"]):
        _fail(
            f"target epoch {target_epoch} exceeds protocol epochs "
            f"{arguments['epochs']}"
        )
    if (run_dir / "summary.json").exists():
        _fail("summary.json already exists; run is not an unfinished prefix")

    split = _load_json(run_dir / "split.json", "split")
    metrics_path = run_dir / "metrics.jsonl"
    rows = _load_jsonl(metrics_path, "metrics")
    last_path = run_dir / "last.pth.tar"
    source_checkpoint_sha256 = _sha256_file(last_path)
    last_checkpoint = _load_checkpoint(last_path, "last checkpoint")
    _validate_checkpoint_identity(
        last_checkpoint,
        protocol=protocol,
        arguments=arguments,
        role="last_evaluated_epoch",
    )
    _validate_split(split, arguments, last_checkpoint)
    validate_metrics_history(
        rows,
        checkpoint=last_checkpoint,
        arguments=arguments,
        split=split,
        expected_resume_epoch=expected_resume_epoch,
    )
    _validate_existing_resume_binding(run_dir, last_checkpoint)

    selection = rebuild_best_selection(rows)
    best_pd_checkpoint = _validate_best_checkpoint(
        run_dir / "best.pth.tar",
        expected_epoch=selection.best_pd_epoch,
        expected_metrics=selection.best_pd_metrics,
        expected_role="best_validation_pd_primary",
        protocol=protocol,
        arguments=arguments,
    )
    best_miou_checkpoint = _validate_best_checkpoint(
        run_dir / "best_miou.pth.tar",
        expected_epoch=selection.best_miou_epoch,
        expected_metrics=selection.best_miou_metrics,
        expected_role="best_validation_miou_secondary",
        protocol=protocol,
        arguments=arguments,
    )
    return ResumeState(
        run_dir=run_dir,
        protocol=protocol,
        arguments=arguments,
        split=split,
        rows=rows,
        last_checkpoint=last_checkpoint,
        best_pd_checkpoint=best_pd_checkpoint,
        best_miou_checkpoint=best_miou_checkpoint,
        selection=selection,
        completed_epoch=expected_resume_epoch,
        target_epoch=target_epoch,
        source_checkpoint_sha256=source_checkpoint_sha256,
    )


def _validate_dataset_reconstruction(state: ResumeState) -> None:
    arguments = state.arguments
    dataset_dir = Path(str(arguments["dataset_dir"])).resolve(strict=True)
    dataset_root = dataset_dir / str(arguments["dataset"])
    identifiers = base.read_training_ids(dataset_root, str(arguments["dataset"]))
    mask_stats = [
        base.inspect_mask(
            dataset_root / "masks", identifier, int(arguments["tiny_area"])
        )
        for identifier in identifiers
    ]
    train_ids, val_ids = base.stratified_split(
        mask_stats,
        float(arguments["val_fraction"]),
        int(arguments["split_seed"]),
    )
    train_ids = base.subset_for_smoke(
        train_ids, arguments["max_train_images"], int(arguments["seed"]) + 101
    )
    val_ids = base.subset_for_smoke(
        val_ids, arguments["max_val_images"], int(arguments["seed"]) + 202
    )
    if train_ids != state.split["used_train_ids"]:
        _fail("reconstructed training identifier order differs from split.json")
    if val_ids != state.split["used_val_ids"]:
        _fail("reconstructed validation identifier order differs from split.json")
    normalization = base.training_only_normalization(
        dataset_root, train_ids
    )
    if normalization != state.protocol["normalization"]:
        _fail("recomputed training-only normalization differs from protocol")


@dataclass
class Runtime:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    train_loader: DataLoader
    val_loader: DataLoader
    criterion: nn.Module
    device: torch.device


def _validate_device_binding(device: torch.device, resume_gpu_uuid: str) -> None:
    if not resume_gpu_uuid:
        _fail("--resume-gpu-uuid must be non-empty")
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        _fail("CUDA device requested but CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        _fail("resume process must expose exactly one CUDA device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible != resume_gpu_uuid:
        _fail(
            "CUDA_VISIBLE_DEVICES must be the exact --resume-gpu-uuid: "
            f"{visible!r} != {resume_gpu_uuid!r}"
        )
    if torch.cuda.get_device_name(device) != "NVIDIA GeForce RTX 5090":
        _fail(
            f"resume CUDA device is not RTX 5090: "
            f"{torch.cuda.get_device_name(device)!r}"
        )


def build_runtime(
    state: ResumeState,
    *,
    device_text: str,
    resume_gpu_uuid: str,
) -> Runtime:
    device = torch.device(device_text)
    _validate_device_binding(device, resume_gpu_uuid)
    _validate_dataset_reconstruction(state)
    arguments = state.arguments

    model, model_metadata = build_clean_v3_model(
        str(arguments["variant"]), int(arguments["seed"])
    )
    if model_metadata != state.protocol["model"]:
        _fail("rebuilt model metadata differs from protocol")
    model.to(device)

    dataset_dir = Path(str(arguments["dataset_dir"])).resolve()
    train_set = base.TrainingSubset(
        dataset_dir,
        str(arguments["dataset"]),
        int(arguments["patch_size"]),
        state.split["used_train_ids"],
        state.protocol["normalization"],
    )
    dataset_root = dataset_dir / str(arguments["dataset"])
    val_set = base.ValidationSubset(
        dataset_root,
        state.split["used_val_ids"],
        state.protocol["normalization"],
    )

    # Match the original reset point exactly.  Replaying this loader then
    # advances the shuffle/crop/flip stream without any optimization.
    base.seed_everything(int(arguments["seed"]))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(arguments["seed"]))
    train_loader = DataLoader(
        train_set,
        batch_size=int(arguments["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(arguments["base_lr"])
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(arguments["amp"]))
    try:
        model.load_state_dict(state.last_checkpoint["state_dict"], strict=True)
    except Exception as exc:
        _fail(f"strict model state load failed: {exc}")
    try:
        optimizer.load_state_dict(state.last_checkpoint["optimizer"])
    except Exception as exc:
        _fail(f"Adam state load failed: {exc}")
    try:
        scaler.load_state_dict(state.last_checkpoint["scaler"])
    except Exception as exc:
        _fail(f"GradScaler state load failed: {exc}")
    criterion = nn.BCELoss(reduction="mean")
    return Runtime(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
    )


@dataclass
class ReplayStats:
    epochs: int
    batches: int
    samples: int
    elapsed_seconds: float


def _batch_size_from_replay_batch(batch: Any) -> int:
    if isinstance(batch, torch.Tensor):
        return int(batch.shape[0]) if batch.ndim else 1
    if isinstance(batch, (tuple, list)) and batch:
        first = batch[0]
        if isinstance(first, torch.Tensor):
            return int(first.shape[0]) if first.ndim else 1
        try:
            return len(first)
        except TypeError:
            return 1
    return 1


def replay_loader_epochs(
    loader: Iterable[Any],
    completed_epoch: int,
    *,
    progress_every: int = 25,
) -> ReplayStats:
    if completed_epoch < 0:
        _fail("completed_epoch must be non-negative")
    started = time.monotonic()
    batches = 0
    samples = 0
    for epoch in range(1, completed_epoch + 1):
        for batch in loader:
            batches += 1
            samples += _batch_size_from_replay_batch(batch)
        if progress_every > 0 and (
            epoch == completed_epoch or epoch % progress_every == 0
        ):
            print(
                "TPDCLEANV3_RESUME_REPLAY"
                f" epoch={epoch}/{completed_epoch}"
                f" batches={batches} samples={samples}",
                flush=True,
            )
    return ReplayStats(
        epochs=completed_epoch,
        batches=batches,
        samples=samples,
        elapsed_seconds=time.monotonic() - started,
    )


def _load_existing_segments(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / SEGMENTS_NAME
    if not path.exists():
        return []
    segments = _load_jsonl(path, "resume segments")
    for index, segment in enumerate(segments, start=1):
        if segment.get("schema") != SEGMENT_SCHEMA:
            _fail(f"resume segment {index} schema mismatch")
        if segment.get("segment_index") != index:
            _fail(f"resume segment index is not contiguous at {index}")
    return segments


def _write_segment_and_provenance(
    state: ResumeState,
    *,
    replay: ReplayStats,
    device_text: str,
    resume_gpu_uuid: str,
) -> dict[str, Any]:
    segments_path = state.run_dir / SEGMENTS_NAME
    existing = _load_existing_segments(state.run_dir)
    segment_index = len(existing) + 1
    segment = {
        "schema": SEGMENT_SCHEMA,
        "segment_index": segment_index,
        "created_at_utc": _utc_now(),
        "resume_from_epoch": state.completed_epoch,
        "first_training_epoch": state.completed_epoch + 1,
        "target_epoch": state.target_epoch,
        "expected_resume_epoch": state.completed_epoch,
        "source_checkpoint": "last.pth.tar",
        "source_checkpoint_sha256": state.source_checkpoint_sha256,
        "device": device_text,
        "resume_gpu_uuid": resume_gpu_uuid,
        "process_restarted": True,
        "model_state_restored_strict": True,
        "adam_state_restored": True,
        "scaler_state_restored": True,
        "data_stream_replay": {
            "workers": 0,
            "seed": state.arguments["seed"],
            "replayed_epochs": replay.epochs,
            "replayed_batches": replay.batches,
            "replayed_samples": replay.samples,
            "elapsed_seconds": replay.elapsed_seconds,
            "shuffle_generator_replayed": True,
            "crop_flip_python_random_replayed": True,
            "optimization_performed": False,
        },
        "continuity_claims": {
            "model_optimizer_scaler_restored": True,
            "shuffle_crop_flip_stream_replayed": True,
            "same_process_continuity": False,
            "cuda_bitwise_continuity": False,
        },
    }
    segment_sha = _append_jsonl(segments_path, segment)
    segments_sha = _sha256_file(segments_path)
    engine_path = Path(__file__).resolve(strict=True)
    provenance_path = state.run_dir / PROVENANCE_NAME
    previous_created = None
    if provenance_path.exists():
        previous_created = _load_json(
            provenance_path, "previous resume provenance"
        ).get("created_at_utc")
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "engine_schema": ENGINE_SCHEMA,
        "created_at_utc": previous_created or _utc_now(),
        "updated_at_utc": _utc_now(),
        "run_directory": str(state.run_dir),
        "variant": state.arguments["variant"],
        "dataset": state.arguments["dataset"],
        "seed": state.arguments["seed"],
        "protocol_sha256": _sha256_file(state.run_dir / "protocol.json"),
        "split_sha256": _sha256_file(state.run_dir / "split.json"),
        "engine_relative_path": str(engine_path.relative_to(REPO_ROOT)),
        "engine_sha256": _sha256_file(engine_path),
        "segments_file": SEGMENTS_NAME,
        "segments_sha256": segments_sha,
        "segment_count": segment_index,
        "latest_segment_index": segment_index,
        "latest_segment_sha256": segment_sha,
        "disclosure": {
            "process_restarted": True,
            "model_optimizer_scaler_restored": True,
            "data_shuffle_crop_flip_stream_replayed": True,
            "cuda_bitwise_continuity_claimed": False,
            "legacy_checkpoint_had_full_rng_state": False,
        },
    }
    provenance_sha = _atomic_write_json(provenance_path, provenance)
    return {
        "resume_engine_schema": ENGINE_SCHEMA,
        "resume_provenance_file": PROVENANCE_NAME,
        "resume_provenance_sha256": provenance_sha,
        "resume_segments_file": SEGMENTS_NAME,
        "resume_segments_sha256": segments_sha,
        "resume_segment_index": segment_index,
        "resume_segment_sha256": segment_sha,
        "resume_disclosure": copy.deepcopy(provenance["disclosure"]),
    }


def attach_resume_binding(
    checkpoint: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    payload = dict(checkpoint)
    for key, value in binding.items():
        payload[key] = copy.deepcopy(value)
    return payload


def _bind_existing_checkpoints(
    state: ResumeState, binding: Mapping[str, Any]
) -> None:
    checkpoint_payloads = {
        "last.pth.tar": state.last_checkpoint,
        "best.pth.tar": state.best_pd_checkpoint,
        "best_miou.pth.tar": state.best_miou_checkpoint,
    }
    for filename, payload in checkpoint_payloads.items():
        _atomic_torch_save(
            state.run_dir / filename,
            attach_resume_binding(payload, binding),
        )


def _checkpoint_payload(
    runtime: Runtime,
    state: ResumeState,
    metrics: Mapping[str, Any],
    epoch: int,
    role: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    namespace = argparse.Namespace(**state.arguments)
    payload = base.checkpoint_payload(
        runtime.model,
        runtime.optimizer,
        runtime.scaler,
        epoch,
        str(state.arguments["variant"]),
        namespace,
        dict(metrics),
        dict(state.protocol["model"]),
        dict(state.split["hashes"]),
    )
    payload["checkpoint_role"] = role
    return attach_resume_binding(payload, binding)


def _append_metric_event(path: Path, event: Mapping[str, Any]) -> None:
    content = _canonical_json_bytes(event)
    with path.open("ab") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _final_summary(
    state: ResumeState,
    selection: SelectionState,
    *,
    binding: Mapping[str, Any],
    skipped_singleton_batches: int,
    process_elapsed_seconds: float,
) -> dict[str, Any]:
    historical_seconds = sum(
        float(row["epoch_seconds"]) for row in state.rows
    )
    return {
        "status": "complete",
        "variant": state.arguments["variant"],
        "dataset": state.arguments["dataset"],
        "seed": state.arguments["seed"],
        "best_epoch": selection.best_pd_epoch,
        "best_validation_metrics": selection.best_pd_metrics,
        "best_pd_epoch": selection.best_pd_epoch,
        "best_pd_validation_metrics": selection.best_pd_metrics,
        "best_miou_epoch": selection.best_miou_epoch,
        "best_miou_validation_metrics": selection.best_miou_metrics,
        "primary_selection_metric": "validation Pd, then lower Fa",
        "selection_source": "internal_validation_only",
        "official_test_accessed": False,
        "model": state.protocol["model"],
        "split_hashes": state.split["hashes"],
        "skipped_singleton_batches": skipped_singleton_batches,
        "elapsed_seconds": historical_seconds + process_elapsed_seconds,
        "resume_process_elapsed_seconds": process_elapsed_seconds,
        "best_checkpoint": state.run_dir / "best.pth.tar",
        "best_miou_checkpoint": state.run_dir / "best_miou.pth.tar",
        "last_checkpoint": state.run_dir / "last.pth.tar",
        **copy.deepcopy(dict(binding)),
    }


def resume_training(
    state: ResumeState,
    runtime: Runtime,
    *,
    resume_gpu_uuid: str,
    replay_progress_every: int = 25,
) -> dict[str, Any]:
    process_started = time.monotonic()
    replay = replay_loader_epochs(
        runtime.train_loader,
        state.completed_epoch,
        progress_every=replay_progress_every,
    )
    binding = _write_segment_and_provenance(
        state,
        replay=replay,
        device_text=str(runtime.device),
        resume_gpu_uuid=resume_gpu_uuid,
    )
    _bind_existing_checkpoints(state, binding)

    selection = copy.deepcopy(state.selection)
    metrics_path = state.run_dir / "metrics.jsonl"
    singleton_per_epoch = (
        1
        if len(runtime.train_loader.dataset)
        % int(state.arguments["batch_size"])
        == 1
        else 0
    )
    skipped_singleton_batches = state.completed_epoch * singleton_per_epoch

    for epoch in range(state.completed_epoch + 1, state.target_epoch + 1):
        epoch_started = time.time()
        learning_rate = base.learning_rate_for_epoch(
            epoch,
            int(state.arguments["epochs"]),
            float(state.arguments["base_lr"]),
            float(state.arguments["min_lr"]),
            int(state.arguments["warmup_epochs"]),
        )
        base.set_learning_rate(runtime.optimizer, learning_rate)
        runtime.model.train()
        loss_sum = 0.0
        sample_count = 0
        for images, masks in runtime.train_loader:
            if images.shape[0] == 1:
                skipped_singleton_batches += 1
                continue
            images = images.to(runtime.device, non_blocking=True)
            masks = masks.to(runtime.device, non_blocking=True)
            runtime.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=runtime.device.type,
                enabled=bool(state.arguments["amp"]),
            ):
                outputs = runtime.model(images)
            with torch.autocast(
                device_type=runtime.device.type,
                enabled=False,
            ):
                if isinstance(outputs, tuple):
                    float_outputs: Any = tuple(
                        output.float() for output in outputs
                    )
                elif isinstance(outputs, list):
                    float_outputs = [output.float() for output in outputs]
                else:
                    float_outputs = outputs.float()
                loss = base.deep_supervision_loss(
                    float_outputs, masks.float(), runtime.criterion
                )
            runtime.scaler.scale(loss).backward()
            runtime.scaler.step(runtime.optimizer)
            runtime.scaler.update()
            batch_size = int(images.shape[0])
            loss_sum += float(loss.detach().item()) * batch_size
            sample_count += batch_size
        if not sample_count:
            _fail(f"no training samples were processed in epoch {epoch}")

        metrics = base.validate(
            runtime.model,
            runtime.val_loader,
            runtime.device,
            runtime.criterion,
            float(state.arguments["threshold"]),
            float(state.arguments["match_radius"]),
            int(state.arguments["tiny_area"]),
            bool(state.arguments["amp"]),
        )
        pd_key = base.pd_selection_key(metrics)
        miou_key = base.miou_selection_key(metrics)
        new_best_pd = pd_key > selection.best_pd_key
        new_best_miou = miou_key > selection.best_miou_key
        if new_best_pd:
            selection.best_pd_key = pd_key
            selection.best_pd_epoch = epoch
            selection.best_pd_metrics = dict(metrics)
        if new_best_miou:
            selection.best_miou_key = miou_key
            selection.best_miou_epoch = epoch
            selection.best_miou_metrics = dict(metrics)

        event: dict[str, Any] = {
            "epoch": epoch,
            "variant": state.arguments["variant"],
            "train_loss": loss_sum / sample_count,
            "learning_rate": learning_rate,
            "processed_train_samples": sample_count,
            "epoch_seconds": time.time() - epoch_started,
            **metrics,
            "new_best_pd": new_best_pd,
            "new_best_miou": new_best_miou,
            "resumed": True,
            "resume_segment_index": binding["resume_segment_index"],
            "resume_provenance_sha256": binding[
                "resume_provenance_sha256"
            ],
        }
        last_payload = _checkpoint_payload(
            runtime,
            state,
            metrics,
            epoch,
            "last_evaluated_epoch",
            binding,
        )
        _atomic_torch_save(state.run_dir / "last.pth.tar", last_payload)
        if new_best_pd:
            best_payload = dict(last_payload)
            best_payload["checkpoint_role"] = "best_validation_pd_primary"
            _atomic_torch_save(state.run_dir / "best.pth.tar", best_payload)
        if new_best_miou:
            best_miou_payload = dict(last_payload)
            best_miou_payload[
                "checkpoint_role"
            ] = "best_validation_miou_secondary"
            _atomic_torch_save(
                state.run_dir / "best_miou.pth.tar", best_miou_payload
            )
        _append_metric_event(metrics_path, event)
        print(
            "TPDCLEANV3_RESUME_EPOCH"
            f" epoch={epoch}/{state.target_epoch}"
            f" loss={event['train_loss']:.6f}"
            f" mIoU={metrics['miou']:.6f}"
            f" Pd={metrics['pd']:.6f}"
            f" Fa={metrics['fa']:.8f}"
            f" bestPdEpoch={selection.best_pd_epoch}"
            f" bestMiouEpoch={selection.best_miou_epoch}",
            flush=True,
        )

    summary = _final_summary(
        state,
        selection,
        binding=binding,
        skipped_singleton_batches=skipped_singleton_batches,
        process_elapsed_seconds=time.monotonic() - process_started,
    )
    _atomic_write_json(state.run_dir / "summary.json", summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume a validated TPD-Clean-v3 epoch-boundary checkpoint"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-epoch", type=int, default=800)
    parser.add_argument("--expected-resume-epoch", type=int, required=True)
    parser.add_argument("--resume-gpu-uuid", required=True)
    parser.add_argument("--replay-progress-every", type=int, default=25)
    args = parser.parse_args(argv)
    if args.replay_progress_every < 0:
        parser.error("--replay-progress-every must be >= 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        state = load_resume_state(
            args.run_dir,
            expected_resume_epoch=args.expected_resume_epoch,
            target_epoch=args.target_epoch,
        )
        runtime = build_runtime(
            state,
            device_text=args.device,
            resume_gpu_uuid=args.resume_gpu_uuid,
        )
        summary = resume_training(
            state,
            runtime,
            resume_gpu_uuid=args.resume_gpu_uuid,
            replay_progress_every=args.replay_progress_every,
        )
    except ResumeValidationError as exc:
        print(f"TPDCLEANV3_RESUME_ABORT reason={exc}", file=sys.stderr)
        return 2
    print(
        "TPDCLEANV3_RESUME_COMPLETE"
        f" variant={summary['variant']}"
        f" seed={summary['seed']}"
        f" epoch={args.target_epoch}"
        f" provenance={summary['resume_provenance_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
