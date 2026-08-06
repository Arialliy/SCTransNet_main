#!/usr/bin/env python3
"""Evaluate the frozen DORF V1 readout modes on one formal checkpoint.

DORF is a zero-training, raw-logit readout audit.  Every test image is passed
through the model exactly once.  Forward hooks capture the already-trained
``outc`` (``z_out``) and ``outconv`` (``z_d0``) logits from that same forward,
after which the five preregistered alpha values are evaluated in memory.

The analyzer supports both the current TSS-off Final model and the paired
three-dataset Original baseline.  It never writes a checkpoint or probability
cache; the only permitted artifact is one write-once JSON evaluation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import audit_final_qfg_functional_use as qfg_audit  # noqa: E402
from analysis import analyze_ner_stage2_mask_knockout_v1 as checkpoint_compat  # noqa: E402
from experiments import evaluate_three_dataset_tss_off_seed42_v1 as tss_off_adapter  # noqa: E402
from experiments import evaluate_three_dataset_v2 as core  # noqa: E402
from experiments import four_dataset_models_seed42_v1 as model_builder  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402
from experiments import (  # noqa: E402
    train_four_dataset_original_final_seed42_exact_v1 as training_engine,
)


SCHEMA = "sctransnet_three_dataset_dorf_v1/v1"
SEED = 42
FIXED_THRESHOLD = 0.5
SWEEP_THRESHOLDS = (0.5, 1.0)
METHODS = ("final_tss_off", "original")
CHECKPOINT_ROLES = tuple(core.CHECKPOINT_ROLES)
ORIGINAL_TRAINING_RUN_SCHEMA = (
    "sctransnet_three_dataset_seed42_global_tss_v2/v1"
)
ORIGINAL_TSS_CANDIDATES = (0.0025, 0.005, 0.01)

MODE_ALPHA = {
    "current_out": 0.0,
    "dorf_a025": 0.25,
    "dorf_a050": 0.5,
    "dorf_a075": 0.75,
    "d0_only": 1.0,
}
MODE_ORDER = tuple(MODE_ALPHA)
CURRENT_MODE = MODE_ORDER[0]
PUBLIC_MODES = MODE_ORDER

OUTPUT_EQUIVALENCE_MAX_ABS = qfg_audit.OUTPUT_EQUIVALENCE_MAX_ABS
OUTPUT_EQUIVALENCE_MEAN_ABS = qfg_audit.OUTPUT_EQUIVALENCE_MEAN_ABS

DEFAULT_FINAL_ROOT = REPO_ROOT / "results" / "three_dataset_tss_off_seed42_v1"
DEFAULT_ORIGINAL_ROOT = (
    REPO_ROOT / "results" / "three_dataset_seed42_global_tss_v2"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "three_dataset_dorf_v1"
DEFAULT_INPUT_MANIFEST = (
    DEFAULT_OUTPUT_ROOT / "manifests" / "dorf_v1_input_manifest.json"
)
INPUT_MANIFEST_SCHEMA = "sctransnet_three_dataset_dorf_v1_input_manifest/v1"
FROZEN_INPUT_MANIFEST_SHA256 = (
    "38bb9a2e4ae5662ae32da6b346444e6d34f5aba57ca13c5ae1dc4516f4230359"
)
BACKGROUND_AUTHORITY_SCHEMA = "sctransnet_additive_joint_metrics/v1"
EXPECTED_TRAINING_STATE_KEYS = {"final_tss_off": 568, "original": 510}
EXPECTED_INFERENCE_STATE_KEYS = {"final_tss_off": 564, "original": 510}
EXPECTED_REMOVED_TSS_KEYS = {"final_tss_off": 4, "original": 0}
EXPECTED_BUILDERS = {
    "final_tss_off": "build_final_inference_model_from_training_state_dict",
    "original": "build_paper_model_original_then_strict_load",
}

_REFERENCE_COUNT_KEYS = {
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "unmatched_predicted_pixels",
    "valid_pixel_count",
}
_REQUIRED_FIXED_FIELDS = {
    "threshold",
    "target_count",
    "tiny_target_count",
    "matched_target_count",
    "matched_tiny_target_count",
    "miou",
    "niou",
    "unmatched_predicted_pixels",
    "false_positive_pixels",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "false_objects_per_image",
    "valid_pixel_count",
    "pd",
    "tiny_pd",
    "fa",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "test_loss",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    ready = float(value)
    _require(math.isfinite(ready), f"{label} must be finite")
    return ready


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    ready = int(value)
    _require(ready >= 0, f"{label} must be non-negative")
    return ready


def _source_sha256() -> dict[str, str]:
    sources = {
        "analysis/analyze_three_dataset_dorf_v1.py": Path(__file__),
        "analysis/analyze_ner_stage2_mask_knockout_v1.py": Path(
            checkpoint_compat.__file__
        ),
        "experiments/evaluate_three_dataset_v2.py": Path(core.__file__),
        "experiments/evaluate_three_dataset_tss_off_seed42_v1.py": Path(
            tss_off_adapter.__file__
        ),
        "experiments/three_dataset_v2_protocol.py": Path(data_protocol.__file__),
        "experiments/four_dataset_models_seed42_v1.py": Path(
            model_builder.__file__
        ),
    }
    return {
        relative: file_sha256(path)
        for relative, path in sorted(sources.items())
    }


def _configure_core_for_method(method: str) -> None:
    """Select only the historical run admission contract for ``method``."""

    _require(method in METHODS, f"method must be one of {METHODS}")
    if method == "final_tss_off":
        tss_off_adapter.configure_core()
        return
    core.TRAINING_RUN_SCHEMA = ORIGINAL_TRAINING_RUN_SCHEMA
    core.TSS_CANDIDATES = ORIGINAL_TSS_CANDIDATES


def _request_for(method: str, dataset: str, checkpoint_role: str) -> core.EvaluationRequest:
    _configure_core_for_method(method)
    request = core.EvaluationRequest(
        dataset=dataset,
        method="final" if method == "final_tss_off" else "original",
        checkpoint_role=checkpoint_role,
        requested_tss_weight=(
            tss_off_adapter.REQUESTED_TSS_WEIGHT
            if method == "final_tss_off"
            else None
        ),
    )
    request.validate()
    return request


def _default_run_dir(method: str, dataset: str) -> Path:
    _require(method in METHODS, f"method must be one of {METHODS}")
    if method == "final_tss_off":
        return (
            DEFAULT_FINAL_ROOT
            / "runs"
            / dataset
            / "final_tss_off"
            / "seed_42"
        )
    return (
        DEFAULT_ORIGINAL_ROOT / "runs" / dataset / "original" / "seed_42"
    )


def _default_reference(run_dir: Path, checkpoint_role: str) -> Path:
    return Path(run_dir) / "evaluations" / f"{checkpoint_role}.json"


def _default_output(method: str, dataset: str, checkpoint_role: str) -> Path:
    return (
        DEFAULT_OUTPUT_ROOT
        / "runs"
        / method
        / dataset
        / checkpoint_role
        / "evaluation.json"
    )


def _json_object(path: Path, label: str) -> dict[str, Any]:
    ready_path = Path(path)
    if not ready_path.is_file() or ready_path.is_symlink():
        raise FileNotFoundError(ready_path)
    value = json.loads(ready_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _resolve_repository_relative_path(value: Any, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} path is malformed")
    declared = Path(value)
    _require(not declared.is_absolute(), f"{label} path must be repository-relative")
    resolved = (REPO_ROOT / declared).resolve(strict=True)
    _require(resolved.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    _require(not resolved.is_symlink(), f"{label} is a symlink")
    return resolved


def _entry_key(method: str, dataset: str, checkpoint_role: str) -> str:
    return f"{method}::{dataset}::{checkpoint_role}"


def _validate_manifest_entry_shape(entry: Mapping[str, Any]) -> tuple[str, str, str]:
    method = entry.get("method")
    dataset = entry.get("dataset")
    role = entry.get("checkpoint_role")
    _require(method in METHODS, "input manifest entry method differs")
    _require(dataset in data_protocol.DATASETS, "input manifest entry dataset differs")
    _require(role in CHECKPOINT_ROLES, "input manifest entry role differs")
    epoch = entry.get("epoch")
    _require(
        isinstance(epoch, int)
        and not isinstance(epoch, bool)
        and 10 <= epoch <= 1000
        and epoch % 10 == 0,
        "input manifest entry epoch differs",
    )
    _require(
        isinstance(entry.get("run_dir"), str) and bool(entry["run_dir"]),
        "input manifest entry run_dir differs",
    )
    for field in (
        "summary_sha256",
        "protocol_sha256",
        "checkpoint_sha256",
        "evaluation_sha256",
    ):
        _require(_is_sha256(entry.get(field)), f"input manifest entry {field} differs")
    return str(method), str(dataset), str(role)


def _validate_loader_contract(manifest: Mapping[str, Any]) -> None:
    raw = manifest.get("loader_contract")
    _require(isinstance(raw, Mapping), "input manifest lacks loader_contract")
    final = raw.get("final_tss_off")
    original = raw.get("original")
    _require(isinstance(final, Mapping) and isinstance(original, Mapping), "loader contract differs")
    expected_final = {
        "training_state_key_count": 568,
        "removed_training_only_tss_state_keys": 4,
        "inference_state_key_count": 564,
        "builder": "build_final_inference_model_from_training_state_dict",
        "strict_load": True,
    }
    expected_original = {
        "training_state_key_count": 510,
        "inference_state_key_count": 510,
        "builder": "build_paper_model_original_then_strict_load",
        "strict_load": True,
    }
    _require(dict(final) == expected_final, "Final loader contract differs")
    _require(dict(original) == expected_original, "Original loader contract differs")


def _find_background_authority_record(
    payload: Mapping[str, Any],
    *,
    method: str,
    dataset: str,
    checkpoint_role: str,
    entry: Mapping[str, Any],
    evaluation_path: Path,
) -> dict[str, Any]:
    _require(payload.get("schema") == BACKGROUND_AUTHORITY_SCHEMA, "background authority schema differs")
    _require(payload.get("status") == "complete", "background authority is incomplete")
    _require(payload.get("seed") == SEED, "background authority seed differs")
    _require(float(payload.get("threshold")) == FIXED_THRESHOLD, "background authority threshold differs")
    records = payload.get("records")
    _require(isinstance(records, list), "background authority records differ")
    recipe = "tss_off" if method == "final_tss_off" else "original"
    candidates = [
        record
        for record in records
        if isinstance(record, Mapping)
        and record.get("recipe") == recipe
        and record.get("dataset") == dataset
        and record.get("checkpoint_role") == checkpoint_role
    ]
    _require(len(candidates) == 1, "background authority must have exactly one bound record")
    record = dict(candidates[0])
    source = record.get("source")
    confusion = record.get("additive_pixel_confusion")
    _require(isinstance(source, Mapping), "background authority source differs")
    _require(isinstance(confusion, Mapping), "background authority confusion differs")
    _require(source.get("checkpoint_epoch") == entry["epoch"], "background authority epoch differs")
    _require(source.get("checkpoint_sha256") == entry["checkpoint_sha256"], "background authority checkpoint SHA differs")
    _require(source.get("evaluation_file_sha256") == entry["evaluation_sha256"], "background authority evaluation SHA differs")
    _require(
        Path(str(source.get("evaluation_path"))).resolve(strict=True)
        == evaluation_path,
        "background authority evaluation path differs",
    )
    false_positive_pixels = _nonnegative_int(
        confusion.get("false_positive_pixels"),
        "background authority false_positive_pixels",
    )
    return {
        "recipe": recipe,
        "dataset": dataset,
        "checkpoint_role": checkpoint_role,
        "checkpoint_epoch": int(entry["epoch"]),
        "checkpoint_sha256": str(entry["checkpoint_sha256"]),
        "evaluation_sha256": str(entry["evaluation_sha256"]),
        "false_positive_pixels": false_positive_pixels,
        "valid_pixel_count": _nonnegative_int(
            confusion.get("valid_pixel_count"),
            "background authority valid_pixel_count",
        ),
    }


def load_input_manifest_binding(
    *,
    input_manifest: Path,
    method: str,
    dataset: str,
    checkpoint_role: str,
    run_dir: Path | None,
    reference_evaluation: Path | None,
    data_protocol_manifest: Path,
) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    """Load and verify the frozen 12-input authority before model loading."""

    manifest_path = Path(input_manifest).resolve(strict=True)
    _require(
        manifest_path == DEFAULT_INPUT_MANIFEST.resolve(strict=True),
        "DORF input manifest path differs from the frozen path",
    )
    manifest_sha = file_sha256(manifest_path)
    _require(
        manifest_sha == FROZEN_INPUT_MANIFEST_SHA256,
        "DORF input manifest SHA differs",
    )
    manifest = _json_object(manifest_path, "DORF input manifest")
    _require(manifest.get("schema") == INPUT_MANIFEST_SCHEMA, "input manifest schema differs")
    _require(manifest.get("status") == "frozen_before_dorf_outputs", "input manifest status differs")
    _require(manifest.get("seed") == SEED, "input manifest seed differs")
    _require(manifest.get("method_order") == list(METHODS), "input manifest method order differs")
    _require(manifest.get("dataset_order") == list(data_protocol.DATASETS), "input manifest dataset order differs")
    _require(manifest.get("checkpoint_role_order") == list(CHECKPOINT_ROLES), "input manifest role order differs")
    _validate_loader_contract(manifest)
    authority = manifest.get("authority")
    _require(isinstance(authority, Mapping), "input manifest authority differs")
    _require(
        authority.get("historical_fixed_threshold_source")
        == "bound_evaluation_json_only"
        and authority.get("checkpoint_embedded_metrics_fallback_allowed") is False
        and authority.get("input_count") == 12
        and authority.get("outputs_exist_when_frozen") is False,
        "input manifest authority policy differs",
    )
    entries = manifest.get("entries")
    _require(isinstance(entries, list) and len(entries) == 12, "input manifest must have 12 entries")
    indexed: dict[str, dict[str, Any]] = {}
    for raw_entry in entries:
        _require(isinstance(raw_entry, Mapping), "input manifest entry is malformed")
        entry_method, entry_dataset, entry_role = _validate_manifest_entry_shape(raw_entry)
        key = _entry_key(entry_method, entry_dataset, entry_role)
        _require(key not in indexed, f"duplicate input manifest entry: {key}")
        indexed[key] = dict(raw_entry)
    expected_keys = {
        _entry_key(expected_method, expected_dataset, expected_role)
        for expected_method in METHODS
        for expected_dataset in data_protocol.DATASETS
        for expected_role in CHECKPOINT_ROLES
    }
    _require(set(indexed) == expected_keys, "input manifest matrix differs")
    selected_key = _entry_key(method, dataset, checkpoint_role)
    _require(selected_key in indexed, "selected DORF input is absent from manifest")
    entry = indexed[selected_key]

    selected_run_dir = _resolve_repository_relative_path(entry["run_dir"], "run_dir")
    _require(selected_run_dir.is_dir(), "manifest run_dir is not a directory")
    selected_summary = selected_run_dir / "summary.json"
    selected_protocol = selected_run_dir / "protocol.json"
    selected_checkpoint = (
        selected_run_dir / "checkpoints" / core.CHECKPOINT_FILENAMES[checkpoint_role]
    )
    selected_evaluation = (
        selected_run_dir / "evaluations" / f"{checkpoint_role}.json"
    )
    for path, label in (
        (selected_summary, "summary"),
        (selected_protocol, "protocol"),
        (selected_checkpoint, "checkpoint"),
        (selected_evaluation, "evaluation"),
    ):
        _require(path.is_file() and not path.is_symlink(), f"bound {label} file differs")
    if run_dir is not None:
        _require(Path(run_dir).resolve(strict=True) == selected_run_dir, "CLI run_dir differs from manifest")
    if reference_evaluation is not None:
        _require(
            Path(reference_evaluation).resolve(strict=True) == selected_evaluation,
            "CLI reference evaluation differs from manifest",
        )

    data_raw = manifest.get("data_protocol_manifest")
    _require(isinstance(data_raw, Mapping), "input manifest data protocol differs")
    bound_data_manifest = _resolve_repository_relative_path(
        data_raw.get("path"), "data protocol manifest"
    )
    _require(_is_sha256(data_raw.get("sha256")), "data protocol manifest SHA differs")
    _require(
        Path(data_protocol_manifest).resolve(strict=True) == bound_data_manifest,
        "CLI data protocol manifest differs from input manifest",
    )

    background_raw = manifest.get("background_pixel_authority")
    _require(isinstance(background_raw, Mapping), "input manifest background authority differs")
    background_path = _resolve_repository_relative_path(
        background_raw.get("path"), "background pixel authority"
    )
    _require(_is_sha256(background_raw.get("sha256")), "background authority SHA differs")
    _require(
        background_raw.get("required_record_count_for_bound_inputs") == 12,
        "background authority bound-record count differs",
    )
    resolved = {
        "manifest": manifest_path,
        "run_dir": selected_run_dir,
        "summary": selected_summary,
        "protocol": selected_protocol,
        "checkpoint": selected_checkpoint,
        "evaluation": selected_evaluation,
        "data_protocol_manifest": bound_data_manifest,
        "background_pixel_authority": background_path,
    }
    verify_bound_input_artifacts(entry, resolved, data_raw, background_raw)
    background_payload = _json_object(background_path, "background pixel authority")
    background_record = _find_background_authority_record(
        background_payload,
        method=method,
        dataset=dataset,
        checkpoint_role=checkpoint_role,
        entry=entry,
        evaluation_path=selected_evaluation,
    )
    binding = {
        "path": str(manifest_path),
        "sha256": manifest_sha,
        "schema": manifest["schema"],
        "status": manifest["status"],
        "entry_key": selected_key,
        "entry": dict(entry),
        "data_protocol_manifest": {
            "path": str(bound_data_manifest),
            "sha256": str(data_raw["sha256"]),
        },
        "background_pixel_authority": {
            "path": str(background_path),
            "sha256": str(background_raw["sha256"]),
        },
        "historical_metric_authority": "bound_evaluation_json_only",
        "checkpoint_embedded_metrics_fallback_allowed": False,
        "verified_before_model_load": True,
        "verified_after_inference": False,
    }
    return binding, resolved, background_record


def verify_bound_input_artifacts(
    entry: Mapping[str, Any],
    resolved: Mapping[str, Path],
    data_manifest_binding: Mapping[str, Any] | None = None,
    background_binding: Mapping[str, Any] | None = None,
) -> None:
    """Rehash every selected artifact named by the frozen manifest."""

    expected = {
        "summary": entry["summary_sha256"],
        "protocol": entry["protocol_sha256"],
        "checkpoint": entry["checkpoint_sha256"],
        "evaluation": entry["evaluation_sha256"],
    }
    for name, expected_sha in expected.items():
        _require(file_sha256(resolved[name]) == expected_sha, f"bound {name} SHA differs")
    if data_manifest_binding is not None:
        _require(
            file_sha256(resolved["data_protocol_manifest"])
            == data_manifest_binding.get("sha256"),
            "bound data protocol manifest SHA differs",
        )
    if background_binding is not None:
        _require(
            file_sha256(resolved["background_pixel_authority"])
            == background_binding.get("sha256"),
            "bound background pixel authority SHA differs",
        )


def fuse_raw_logits(
    z_out: torch.Tensor,
    z_d0: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Apply the frozen pre-sigmoid DORF formula.

    Boundary modes return the captured tensors directly.  Besides avoiding a
    needless arithmetic round trip, this makes alpha=0 an exact replay of the
    model's current raw output and alpha=1 an exact use of ``d0``.
    """

    ready_alpha = _finite_float(alpha, "alpha")
    _require(ready_alpha in MODE_ALPHA.values(), "alpha is not preregistered")
    _require(
        isinstance(z_out, torch.Tensor) and isinstance(z_d0, torch.Tensor),
        "DORF logits must be tensors",
    )
    _require(z_out.shape == z_d0.shape, "d0/out logit shapes differ")
    _require(z_out.device == z_d0.device, "d0/out logit devices differ")
    _require(z_out.dtype == z_d0.dtype, "d0/out logit dtypes differ")
    _require(
        bool(torch.isfinite(z_out).all()) and bool(torch.isfinite(z_d0).all()),
        "DORF logits contain non-finite values",
    )
    if ready_alpha == 0.0:
        return z_out
    if ready_alpha == 1.0:
        return z_d0
    return z_out + ready_alpha * (z_d0 - z_out)


class RawD0OutHookCapture:
    """Capture exactly one raw ``out`` and ``d0`` tensor per model forward."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.out_module = getattr(model, "outc", None)
        self.d0_module = getattr(model, "outconv", None)
        _require(isinstance(self.out_module, nn.Module), "model lacks outc module")
        _require(isinstance(self.d0_module, nn.Module), "model lacks outconv module")
        self._handles: list[Any] = []
        self._active = False
        self._current: dict[str, torch.Tensor] = {}
        self.total_counts = {"out": 0, "d0": 0}
        self.batch_count = 0
        self._out_hook_ids_before = tuple(self.out_module._forward_hooks)
        self._d0_hook_ids_before = tuple(self.d0_module._forward_hooks)
        self.temporary_hooks_restored = False

    def _hook(self, name: str):
        def record(_module: nn.Module, _inputs: Any, output: Any) -> None:
            _require(self._active, f"{name} hook fired outside active batch")
            _require(name not in self._current, f"{name} executed more than once in batch")
            _require(isinstance(output, torch.Tensor), f"{name} output is not a tensor")
            _require(output.ndim == 4, f"{name} output must be BCHW")
            _require(bool(torch.isfinite(output).all()), f"{name} output is non-finite")
            self._current[name] = output
            self.total_counts[name] += 1

        return record

    def __enter__(self) -> "RawD0OutHookCapture":
        _require(not self._handles, "raw-logit hooks are already installed")
        self._handles = [
            self.out_module.register_forward_hook(self._hook("out")),
            self.d0_module.register_forward_hook(self._hook("d0")),
        ]
        return self

    def begin_batch(self) -> None:
        _require(not self._active, "previous hook batch is still active")
        self._active = True
        self._current = {}

    def finish_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        _require(self._active, "no active hook batch")
        self._active = False
        _require(set(self._current) == {"out", "d0"}, "d0/out did not each execute once")
        z_out = self._current["out"]
        z_d0 = self._current["d0"]
        _require(z_out.shape == z_d0.shape, "captured d0/out shapes differ")
        _require(z_out.device == z_d0.device, "captured d0/out devices differ")
        _require(z_out.dtype == z_d0.dtype, "captured d0/out dtypes differ")
        self.batch_count += 1
        self._current = {}
        return z_out, z_d0

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self._active = False
        self._current = {}
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.temporary_hooks_restored = (
            tuple(self.out_module._forward_hooks) == self._out_hook_ids_before
            and tuple(self.d0_module._forward_hooks) == self._d0_hook_ids_before
        )


def _sample_id(sample_ids: Any) -> str:
    _require(
        isinstance(sample_ids, (tuple, list)) and len(sample_ids) == 1,
        "test loader must collate exactly one sample ID",
    )
    return str(sample_ids[0])


def _update_tensor_digest(
    digest: "hashlib._Hash",
    label: str,
    value: torch.Tensor,
) -> None:
    ready = value.detach().cpu().contiguous()
    descriptor = json.dumps(
        [label, str(ready.dtype), list(ready.shape)],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    raw = ready.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest.update(len(descriptor).to_bytes(8, "big"))
    digest.update(descriptor)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def probability_difference(
    current_probabilities: Sequence[np.ndarray],
    candidate_probabilities: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Aggregate differences over all original, unpadded test pixels."""

    _require(
        len(current_probabilities) == len(candidate_probabilities)
        and bool(current_probabilities),
        "current/candidate probability collections differ",
    )
    maximum = 0.0
    absolute_sum = 0.0
    element_count = 0
    for current, candidate in zip(current_probabilities, candidate_probabilities):
        current_array = np.asarray(current, dtype=np.float32)
        candidate_array = np.asarray(candidate, dtype=np.float32)
        _require(current_array.shape == candidate_array.shape, "probability shapes differ")
        difference = np.abs(
            current_array.astype(np.float64)
            - candidate_array.astype(np.float64)
        )
        maximum = max(maximum, float(difference.max()))
        absolute_sum += float(difference.sum())
        element_count += int(difference.size)
    _require(element_count > 0, "probability collection is empty")
    mean_abs = absolute_sum / element_count
    equivalent = (
        maximum <= OUTPUT_EQUIVALENCE_MAX_ABS
        and mean_abs <= OUTPUT_EQUIVALENCE_MEAN_ABS
    )
    return {
        "scope": "all_original_unpadded_test_pixels",
        "element_count": element_count,
        "absolute_difference_sum": absolute_sum,
        "max_abs": maximum,
        "mean_abs": mean_abs,
        "equivalent": equivalent,
        "functionally_different": not equivalent,
        "equivalence_max_abs_threshold": OUTPUT_EQUIVALENCE_MAX_ABS,
        "equivalence_mean_abs_threshold": OUTPUT_EQUIVALENCE_MEAN_ABS,
    }


def _background_false_positive_pixels(
    probabilities: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> int:
    _require(
        len(probabilities) == len(targets) and bool(probabilities),
        "probability/target collections differ",
    )
    total = 0
    for probability, target in zip(probabilities, targets):
        probability_array = np.asarray(probability)
        target_array = np.asarray(target)
        _require(probability_array.shape == target_array.shape, "probability/target shapes differ")
        total += int(
            np.count_nonzero(
                np.logical_and(
                    probability_array > FIXED_THRESHOLD,
                    target_array <= FIXED_THRESHOLD,
                )
            )
        )
    return total


def _annotate_two_point_sweep(evaluated: Mapping[str, Any]) -> dict[str, Any]:
    ready = dict(evaluated)
    descriptive = dict(ready["descriptive_pd_fa"])
    raw_points = descriptive.get("points")
    _require(isinstance(raw_points, list) and len(raw_points) == 2, "DORF sweep must have two points")
    points: list[dict[str, Any]] = []
    for raw in raw_points:
        point = dict(raw)
        point["selected_point_is_empty"] = core._point_is_empty(point)
        points.append(point)
    _require(
        [float(point["threshold"]) for point in points] == list(SWEEP_THRESHOLDS),
        "DORF sweep thresholds differ",
    )
    _require(
        points[1]["selected_point_is_empty"] is True
        and float(points[1]["pd"]) == 0.0
        and float(points[1]["fa"]) == 0.0,
        "threshold=1.0 is not the legal empty endpoint",
    )
    descriptive["points"] = points
    ready["descriptive_pd_fa"] = descriptive
    return ready


@torch.inference_mode()
def analyze_loaded_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    expected_identifiers: Sequence[str],
) -> dict[str, Any]:
    """Run the shared-forward DORF audit for an already loaded model."""

    _require(bool(expected_identifiers), "expected identifier sequence is empty")
    training_before = bool(model.training)
    mode_before = getattr(model, "mode", None)
    _require(training_before is False, "DORF model must already be in eval mode")
    _require(mode_before == "test", "DORF model mode must remain test")
    state_before = checkpoint_compat.module_state_sha256(model)
    criterion = nn.BCELoss(reduction="mean")
    probabilities: dict[str, list[np.ndarray]] = {
        mode: [] for mode in MODE_ORDER
    }
    losses: dict[str, list[float]] = {mode: [] for mode in MODE_ORDER}
    targets: list[np.ndarray] = []
    identifiers: list[str] = []
    input_digest = hashlib.sha256()
    returned_out_equal = True
    batch_count = 0

    with RawD0OutHookCapture(model) as capture:
        for images, masks, sizes, sample_ids in loader:
            _require(
                int(images.shape[0]) == 1 and int(masks.shape[0]) == 1,
                "DORF formal evaluator requires batch_size=1",
            )
            identifier = _sample_id(sample_ids)
            height, width = core._extract_hw(sizes)
            identifier_raw = identifier.encode("utf-8")
            input_digest.update(len(identifier_raw).to_bytes(8, "big"))
            input_digest.update(identifier_raw)
            input_digest.update(height.to_bytes(8, "big"))
            input_digest.update(width.to_bytes(8, "big"))
            _update_tensor_digest(input_digest, "padded_image", images)
            _update_tensor_digest(input_digest, "padded_mask", masks)

            images_device = images.to(device, non_blocking=True)
            masks_device = masks.to(device, non_blocking=True)
            capture.begin_batch()
            returned = model(images_device)
            z_out, z_d0 = capture.finish_batch()
            returned_probability = core._final_prediction(returned)
            expected_probability = torch.sigmoid(z_out)
            _require(
                returned_probability.shape == expected_probability.shape,
                "returned prediction shape differs from sigmoid(raw out)",
            )
            this_equal = torch.equal(returned_probability, expected_probability)
            returned_out_equal = returned_out_equal and this_equal
            _require(this_equal, "returned prediction is not exact sigmoid(raw out)")

            target = masks_device[:, :, :height, :width]
            _require(target.shape == z_out[:, :, :height, :width].shape, "target/logit shapes differ")
            target_array = target[0, 0].float().cpu().numpy().copy()
            _require(np.isfinite(target_array).all(), "target contains non-finite values")
            targets.append(target_array)
            identifiers.append(identifier)
            for mode in MODE_ORDER:
                alpha = MODE_ALPHA[mode]
                if mode == CURRENT_MODE:
                    # Alpha zero is the evaluator's actual returned prediction,
                    # never a value regenerated through the DORF formula.
                    probability_tensor = returned_probability[
                        :, :, :height, :width
                    ]
                else:
                    fused = fuse_raw_logits(z_out, z_d0, alpha)
                    probability_tensor = torch.sigmoid(fused)[
                        :, :, :height, :width
                    ]
                _require(
                    probability_tensor.shape == target.shape,
                    f"{mode} probability/target shapes differ",
                )
                _require(
                    bool(torch.isfinite(probability_tensor).all()),
                    f"{mode} probability is non-finite",
                )
                loss = criterion(probability_tensor.float(), target.float())
                loss_value = float(loss.item())
                _require(math.isfinite(loss_value), f"{mode} test loss is non-finite")
                probability_array = (
                    probability_tensor[0, 0].float().cpu().numpy().copy()
                )
                _require(np.isfinite(probability_array).all(), f"{mode} array is non-finite")
                probabilities[mode].append(probability_array)
                losses[mode].append(loss_value)
            batch_count += 1

    _require(capture.temporary_hooks_restored, "temporary d0/out hooks were not restored")
    _require(
        identifiers == [str(value) for value in expected_identifiers],
        "inference order differs from frozen img_idx/test order",
    )
    _require(batch_count == len(expected_identifiers), "inference batch count differs")
    _require(
        capture.batch_count == batch_count
        and capture.total_counts == {"out": batch_count, "d0": batch_count},
        "d0/out hook execution counts differ",
    )
    state_after = checkpoint_compat.module_state_sha256(model)
    _require(state_after == state_before, "model state changed during DORF audit")
    training_after = bool(model.training)
    mode_after = getattr(model, "mode", None)
    _require(training_after == training_before, "model training flag changed during DORF audit")
    _require(mode_after == mode_before, "model mode changed during DORF audit")

    current_probabilities = probabilities[CURRENT_MODE]
    mode_outputs: dict[str, Any] = {}
    for mode in MODE_ORDER:
        evaluated = core.evaluate_probability_arrays(
            probabilities[mode],
            targets,
            losses[mode],
            sweep_thresholds=SWEEP_THRESHOLDS,
        )
        evaluated = _annotate_two_point_sweep(evaluated)
        fixed = dict(evaluated["fixed_threshold_0_5"])
        fixed["false_positive_pixels"] = _background_false_positive_pixels(
            probabilities[mode], targets
        )
        fixed["image_count"] = batch_count
        evaluated["fixed_threshold_0_5"] = fixed
        mode_outputs[mode] = {
            "mode": mode,
            "alpha": MODE_ALPHA[mode],
            **evaluated,
            "probability_difference_to_current": probability_difference(
                current_probabilities, probabilities[mode]
            ),
        }

    current_difference = mode_outputs[CURRENT_MODE][
        "probability_difference_to_current"
    ]
    _require(
        current_difference["max_abs"] == 0.0
        and current_difference["absolute_difference_sum"] == 0.0
        and current_difference["mean_abs"] == 0.0,
        "current_out self-difference is nonzero",
    )
    all_metrics_finite = True
    json.dumps(mode_outputs, allow_nan=False)
    output = {
        "modes": mode_outputs,
        "input_binding": {
            "scope": "ordered_padded_images_masks_sizes_and_sample_ids",
            "sha256": input_digest.hexdigest(),
            "sample_count": batch_count,
            "ordered_ids_newline_sha256": hashlib.sha256(
                ("\n".join(identifiers) + "\n").encode("utf-8")
            ).hexdigest(),
        },
        "engineering_audit": {
            "passed": True,
            "all_metrics_finite": all_metrics_finite,
            "same_d0_out_logits_reused_for_all_modes": True,
            "one_model_forward_per_batch": True,
            "raw_logit_fusion": True,
            "model_state_unchanged": state_after == state_before,
            "model_training_flag_unchanged": training_after == training_before,
            "model_mode_unchanged": mode_after == mode_before,
            "model_training_flag_before": training_before,
            "model_training_flag_after": training_after,
            "model_mode_before": mode_before,
            "model_mode_after": mode_after,
            "derived_checkpoint_written": False,
            "probability_cache_written": False,
            "batch_count": batch_count,
            "model_forward_count": batch_count,
            "outc_hook_count": capture.total_counts["out"],
            "outconv_hook_count": capture.total_counts["d0"],
            "each_hook_exactly_once_per_batch": True,
            "returned_probability_equals_sigmoid_raw_out_bitwise": returned_out_equal,
            "temporary_hooks_restored": capture.temporary_hooks_restored,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
        },
    }
    # Full arrays are intentionally only local temporaries.  They are never
    # returned and therefore cannot enter the write-once JSON artifact.
    del probabilities, targets, losses
    return output


def _validate_reference_identity(
    payload: Mapping[str, Any],
    *,
    method: str,
    dataset: str,
    checkpoint_role: str,
    checkpoint_sha256: str,
) -> None:
    _require(payload.get("schema") == core.SCHEMA, "reference schema differs")
    _require(payload.get("status") == "complete", "reference is incomplete")
    _require(payload.get("dataset") == dataset, "reference dataset differs")
    _require(payload.get("method") == method, "reference method differs")
    _require(payload.get("checkpoint_role") == checkpoint_role, "reference role differs")
    _require(payload.get("seed") == SEED, "reference seed differs")
    point = payload.get("fixed_threshold_0_5")
    _require(isinstance(point, Mapping), "reference lacks fixed 0.5 metrics")
    _require(float(point.get("threshold")) == FIXED_THRESHOLD, "reference threshold differs")
    binding = payload.get("checkpoint_binding")
    _require(isinstance(binding, Mapping), "reference lacks checkpoint binding")
    checkpoint = binding.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "reference checkpoint binding is malformed")
    _require(checkpoint.get("role") == checkpoint_role, "reference checkpoint role differs")
    _require(checkpoint.get("sha256") == checkpoint_sha256, "reference checkpoint SHA differs")


def _load_reference(
    *,
    reference_evaluation: Path,
    expected_evaluation_sha256: str,
    method: str,
    dataset: str,
    checkpoint_role: str,
    checkpoint_binding: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    checkpoint = checkpoint_binding["checkpoint"]
    checkpoint_sha = str(checkpoint["sha256"])
    reference_path = Path(reference_evaluation).resolve(strict=True)
    observed_sha = file_sha256(reference_path)
    _require(
        observed_sha == expected_evaluation_sha256,
        "bound historical evaluation SHA differs",
    )
    payload = _json_object(reference_path, "bound historical evaluation")
    _validate_reference_identity(
        payload,
        method=method,
        dataset=dataset,
        checkpoint_role=checkpoint_role,
        checkpoint_sha256=checkpoint_sha,
    )
    return payload["fixed_threshold_0_5"], {
        "checkpoint_role": checkpoint_role,
        "path": str(reference_path),
        "sha256": observed_sha,
        "source": "historical_evaluation_fixed_threshold_0_5",
        "checkpoint_embedded_metrics_fallback_allowed": False,
    }


def alpha0_historical_replay_audit(
    observed: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    checkpoint_role: str,
    reference_evaluation_sha256: str,
    expected_background_false_positive_pixels: int,
    background_pixel_authority_sha256: str,
) -> dict[str, Any]:
    """Replay counts exactly and floats under the frozen core tolerances."""

    required = _REQUIRED_FIXED_FIELDS - {"false_positive_pixels"}
    _require(required <= set(observed), "current_out replay metrics are incomplete")
    _require(required <= set(reference), "historical replay metrics are incomplete")
    compared: dict[str, Any] = {}
    exact = True
    counts_exact = True
    within_tolerances = True
    for key in sorted(reference):
        if key not in observed or key in {"false_positive_pixels", "image_count"}:
            continue
        actual = observed[key]
        expected = reference[key]
        if key in _REFERENCE_COUNT_KEYS:
            equal = actual == expected
            within_tolerance = equal
            tolerance = 0.0
            counts_exact = counts_exact and equal
            absolute_difference = 0 if equal else abs(int(actual) - int(expected))
        elif expected is None:
            equal = actual is None
            within_tolerance = equal
            tolerance = 0.0
            absolute_difference = None
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            equal = float(actual) == float(expected)
            absolute_difference = abs(float(actual) - float(expected))
            tolerance = (
                1e-4
                if key
                in {
                    "miou",
                    "niou",
                    "pixel_precision",
                    "pixel_recall",
                    "pixel_f1",
                }
                else 1e-7
                if key == "test_loss"
                else 1e-15
            )
            within_tolerance = math.isclose(
                float(actual),
                float(expected),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        else:
            equal = actual == expected
            within_tolerance = equal
            tolerance = 0.0
            absolute_difference = None
        exact = exact and equal
        within_tolerances = within_tolerances and within_tolerance
        compared[key] = {
            "exactly_equal": equal,
            "within_frozen_tolerance": within_tolerance,
            "absolute_difference": absolute_difference,
            "absolute_tolerance": tolerance,
        }
    observed_background = _nonnegative_int(
        observed.get("false_positive_pixels"),
        "current_out background false-positive pixels",
    )
    expected_background = _nonnegative_int(
        expected_background_false_positive_pixels,
        "bound background false-positive pixels",
    )
    background_exact = observed_background == expected_background
    _require(counts_exact, "alpha=0 count replay differs from historical evaluation")
    _require(within_tolerances, "alpha=0 float replay exceeds frozen tolerance")
    _require(background_exact, "alpha=0 background false-positive pixels differ")
    return {
        "passed": True,
        "exact": exact and background_exact,
        "counts_exact": counts_exact,
        "background_false_positive_pixels_exact": background_exact,
        "within_frozen_float_tolerances": within_tolerances,
        "mode": CURRENT_MODE,
        "checkpoint_role": checkpoint_role,
        "reference_evaluation_sha256": reference_evaluation_sha256,
        "background_pixel_authority_sha256": background_pixel_authority_sha256,
        "background_false_positive_pixels": {
            "observed": observed_background,
            "expected": expected_background,
        },
        "comparison": "current_out_fixed_threshold_0_5_vs_bound_historical_metrics",
        "compared": compared,
    }


def analyze_run(
    *,
    method: str,
    dataset: str,
    checkpoint_role: str,
    run_dir: Path | None,
    dataset_root: Path,
    data_protocol_manifest: Path,
    reference_evaluation: Path | None,
    device_name: str,
    workers: int,
    input_manifest: Path = DEFAULT_INPUT_MANIFEST,
) -> dict[str, Any]:
    _require(method in METHODS, f"method must be one of {METHODS}")
    _require(dataset in data_protocol.DATASETS, "dataset is outside formal scope")
    _require(checkpoint_role in CHECKPOINT_ROLES, "checkpoint role differs")
    _require(workers >= 0, "workers must be non-negative")
    input_manifest_binding, resolved_inputs, background_record = (
        load_input_manifest_binding(
            input_manifest=input_manifest,
            method=method,
            dataset=dataset,
            checkpoint_role=checkpoint_role,
            run_dir=run_dir,
            reference_evaluation=reference_evaluation,
            data_protocol_manifest=data_protocol_manifest,
        )
    )
    bound_entry = input_manifest_binding["entry"]
    source_before = _source_sha256()
    _configure_core_for_method(method)
    training_engine.configure_determinism()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    manifest_path = resolved_inputs["data_protocol_manifest"]
    manifest = data_protocol.load_protocol_manifest(
        manifest_path, dataset_root=dataset_root
    )
    request = _request_for(method, dataset, checkpoint_role)
    checkpoint_payload, raw_binding = (
        checkpoint_compat._load_checkpoint_allowing_added_sources(
            request,
            resolved_inputs["run_dir"],
            manifest_path,
            manifest,
        )
    )
    checkpoint_binding = dict(raw_binding)
    _require(
        Path(checkpoint_binding["run_dir"]).resolve(strict=True)
        == resolved_inputs["run_dir"],
        "loaded run directory differs from input manifest",
    )
    for name, binding_name, sha_field in (
        ("summary", "summary", "summary_sha256"),
        ("protocol", "protocol", "protocol_sha256"),
        ("checkpoint", "checkpoint", "checkpoint_sha256"),
    ):
        observed_binding = checkpoint_binding[binding_name]
        _require(
            Path(observed_binding["path"]).resolve(strict=True)
            == resolved_inputs[name],
            f"loaded {name} path differs from input manifest",
        )
        _require(
            observed_binding["sha256"] == bound_entry[sha_field],
            f"loaded {name} SHA differs from input manifest",
        )
    _require(
        checkpoint_binding["checkpoint"]["epoch"] == bound_entry["epoch"],
        "loaded checkpoint epoch differs from input manifest",
    )
    training_state = checkpoint_payload["state_dict"]
    _require(
        len(training_state) == EXPECTED_TRAINING_STATE_KEYS[method],
        "training state-key count differs from frozen loader contract",
    )
    training_state_sha = model_builder.state_dict_sha256(
        training_state
    )
    checkpoint_binding["training_state_dict_sha256"] = training_state_sha
    checkpoint_binding["checkpoint_payload_schema"] = checkpoint_payload.get("schema")
    checkpoint_binding["input_manifest_entry_key"] = input_manifest_binding[
        "entry_key"
    ]

    reference_fixed, reference_binding = _load_reference(
        reference_evaluation=resolved_inputs["evaluation"],
        expected_evaluation_sha256=bound_entry["evaluation_sha256"],
        method=method,
        dataset=dataset,
        checkpoint_role=checkpoint_role,
        checkpoint_binding=checkpoint_binding,
    )

    model, model_metadata = core.build_inference_model(
        request, training_state
    )
    _require(model_metadata.get("strict_load") is True, "model strict-load metadata differs")
    inference_state_key_count = len(model.state_dict())
    _require(
        inference_state_key_count == EXPECTED_INFERENCE_STATE_KEYS[method],
        "inference state-key count differs from frozen loader contract",
    )
    _require(model.training is False, "strict loader did not return eval model")
    _require(getattr(model, "mode", None) == "test", "strict loader did not preserve mode=test")
    model_metadata = dict(model_metadata)
    model_metadata["dorf_loader_audit"] = {
        "passed": True,
        "builder": EXPECTED_BUILDERS[method],
        "training_state_key_count": len(training_state),
        "expected_training_state_key_count": EXPECTED_TRAINING_STATE_KEYS[method],
        "removed_training_only_tss_state_key_count": (
            len(model_metadata.get("removed_tss_state_keys", ()))
            if method == "final_tss_off"
            else 0
        ),
        "expected_removed_training_only_tss_state_key_count": EXPECTED_REMOVED_TSS_KEYS[
            method
        ],
        "inference_state_key_count": inference_state_key_count,
        "expected_inference_state_key_count": EXPECTED_INFERENCE_STATE_KEYS[method],
        "strict_load": True,
        "training_flag": bool(model.training),
        "mode": getattr(model, "mode", None),
    }
    _require(
        model_metadata["dorf_loader_audit"][
            "removed_training_only_tss_state_key_count"
        ]
        == EXPECTED_REMOVED_TSS_KEYS[method],
        "removed TSS state-key count differs from frozen loader contract",
    )
    model.to(device)
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
    replay = alpha0_historical_replay_audit(
        analyzed["modes"][CURRENT_MODE]["fixed_threshold_0_5"],
        reference_fixed,
        checkpoint_role=checkpoint_role,
        reference_evaluation_sha256=reference_binding["sha256"],
        expected_background_false_positive_pixels=background_record[
            "false_positive_pixels"
        ],
        background_pixel_authority_sha256=input_manifest_binding[
            "background_pixel_authority"
        ]["sha256"],
    )
    _require(
        int(reference_fixed["valid_pixel_count"])
        == int(background_record["valid_pixel_count"])
        == int(
            analyzed["modes"][CURRENT_MODE]["fixed_threshold_0_5"][
                "valid_pixel_count"
            ]
        ),
        "background authority valid-pixel count differs",
    )
    engineering = dict(analyzed["engineering_audit"])
    source_after = _source_sha256()
    _require(source_after == source_before, "DORF source bytes changed during analysis")
    manifest_data_binding = {
        "sha256": input_manifest_binding["data_protocol_manifest"]["sha256"]
    }
    manifest_background_binding = {
        "sha256": input_manifest_binding["background_pixel_authority"]["sha256"]
    }
    verify_bound_input_artifacts(
        bound_entry,
        resolved_inputs,
        manifest_data_binding,
        manifest_background_binding,
    )
    _require(
        file_sha256(resolved_inputs["manifest"])
        == FROZEN_INPUT_MANIFEST_SHA256,
        "input manifest changed during inference",
    )
    input_manifest_binding["verified_after_inference"] = True
    engineering.update(
        {
            "passed": True,
            "source_sha256_reverified_after_inference": True,
            "input_manifest_reverified_after_inference": True,
            "alpha0_historical_replay_passed": replay["passed"],
        }
    )
    output = {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": dataset,
        "method": method,
        "training_model_method": request.method,
        "checkpoint_role": checkpoint_role,
        "seed": SEED,
        "test_selected": True,
        "selection_is_optimistic": True,
        "mode_order": list(MODE_ORDER),
        "modes": analyzed["modes"],
        "alpha0_historical_replay_audit": replay,
        "engineering_audit": engineering,
        "input_manifest_binding": input_manifest_binding,
        "checkpoint_binding": checkpoint_binding,
        "reference_evaluation_binding": reference_binding,
        "source_sha256": source_before,
        "model_metadata": model_metadata,
        "background_pixel_authority_record": background_record,
        "data": {
            "protocol_module": "experiments.three_dataset_v2_protocol",
            "dataset_root": str(Path(dataset_root).resolve()),
            "protocol_manifest": {
                "path": str(manifest_path),
                "sha256": file_sha256(manifest_path),
                "schema": manifest.get("schema"),
                "manifest_id": manifest.get("manifest_id"),
            },
            "split": "img_idx/test",
            "test_count": len(dataset_object.sample_ids),
            "input_binding": analyzed["input_binding"],
            "normalization": core.NORMALIZATION[dataset],
            "sirst3_in_formal_matrix": False,
        },
        "intervention_contract": {
            "family": "DORF_V1_existing_deep_supervision_readout_reuse",
            "formula": "z_out + alpha * (z_d0 - z_out)",
            "fusion_space": "raw_logits_before_sigmoid",
            "alphas": [MODE_ALPHA[mode] for mode in MODE_ORDER],
            "one_checkpoint_per_unit": True,
            "model_parameters_changed": False,
            "persistent_buffers_changed": False,
            "derived_checkpoint_written": False,
        },
        "derived_checkpoint_written": False,
        "probability_cache_written": False,
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
    _require(payload.get("schema") == SCHEMA, "DORF analyzer schema differs")
    _require(payload.get("status") == "complete", "DORF analyzer is incomplete")
    _require(payload.get("dataset") in data_protocol.DATASETS, "dataset differs")
    method = payload.get("method")
    _require(method in METHODS, "method differs")
    role = payload.get("checkpoint_role")
    _require(role in CHECKPOINT_ROLES, "checkpoint role differs")
    _require(payload.get("seed") == SEED, "seed differs")
    _require(payload.get("test_selected") is True, "test-selected scope differs")
    _require(payload.get("selection_is_optimistic") is True, "selection scope differs")
    _require(payload.get("mode_order") == list(MODE_ORDER), "mode order differs")

    manifest_binding = payload.get("input_manifest_binding")
    _require(isinstance(manifest_binding, Mapping), "input manifest binding missing")
    _require(
        manifest_binding.get("path") == str(DEFAULT_INPUT_MANIFEST.resolve(strict=True))
        and manifest_binding.get("sha256") == FROZEN_INPUT_MANIFEST_SHA256
        and manifest_binding.get("schema") == INPUT_MANIFEST_SCHEMA
        and manifest_binding.get("status") == "frozen_before_dorf_outputs"
        and manifest_binding.get("entry_key")
        == _entry_key(str(method), str(payload.get("dataset")), str(role))
        and manifest_binding.get("historical_metric_authority")
        == "bound_evaluation_json_only"
        and manifest_binding.get("checkpoint_embedded_metrics_fallback_allowed")
        is False
        and manifest_binding.get("verified_before_model_load") is True
        and manifest_binding.get("verified_after_inference") is True,
        "input manifest identity/policy differs",
    )
    manifest_entry = manifest_binding.get("entry")
    _require(isinstance(manifest_entry, Mapping), "input manifest entry binding missing")
    entry_method, entry_dataset, entry_role = _validate_manifest_entry_shape(
        manifest_entry
    )
    _require(
        (entry_method, entry_dataset, entry_role)
        == (method, payload.get("dataset"), role),
        "input manifest selected entry differs",
    )
    bound_data_manifest = manifest_binding.get("data_protocol_manifest")
    bound_background = manifest_binding.get("background_pixel_authority")
    _require(
        isinstance(bound_data_manifest, Mapping)
        and isinstance(bound_data_manifest.get("path"), str)
        and _is_sha256(bound_data_manifest.get("sha256")),
        "bound data protocol manifest differs",
    )
    _require(
        isinstance(bound_background, Mapping)
        and isinstance(bound_background.get("path"), str)
        and _is_sha256(bound_background.get("sha256")),
        "bound background authority differs",
    )

    checkpoint_binding = payload.get("checkpoint_binding")
    _require(isinstance(checkpoint_binding, Mapping), "checkpoint binding missing")
    checkpoint = checkpoint_binding.get("checkpoint")
    _require(
        isinstance(checkpoint, Mapping)
        and checkpoint.get("role") == role
        and _is_sha256(checkpoint.get("sha256")),
        "checkpoint file binding differs",
    )
    _require(
        checkpoint.get("sha256") == manifest_entry.get("checkpoint_sha256")
        and checkpoint.get("epoch") == manifest_entry.get("epoch")
        and checkpoint_binding.get("input_manifest_entry_key")
        == manifest_binding.get("entry_key"),
        "checkpoint/input-manifest binding differs",
    )
    _require(
        _is_sha256(checkpoint_binding.get("training_state_dict_sha256")),
        "training state SHA differs",
    )
    reference = payload.get("reference_evaluation_binding")
    _require(
        isinstance(reference, Mapping)
        and reference.get("checkpoint_role") == role
        and isinstance(reference.get("path"), str)
        and bool(reference.get("path"))
        and _is_sha256(reference.get("sha256")),
        "reference evaluation binding differs",
    )
    _require(
        reference.get("sha256") == manifest_entry.get("evaluation_sha256")
        and reference.get("source")
        == "historical_evaluation_fixed_threshold_0_5"
        and reference.get("checkpoint_embedded_metrics_fallback_allowed") is False,
        "reference/input-manifest authority differs",
    )
    sources = payload.get("source_sha256")
    _require(
        isinstance(sources, Mapping)
        and bool(sources)
        and all(_is_sha256(value) for value in sources.values()),
        "source SHA map differs",
    )
    model_metadata = payload.get("model_metadata")
    _require(isinstance(model_metadata, Mapping) and bool(model_metadata), "model metadata missing")
    loader_audit = model_metadata.get("dorf_loader_audit")
    _require(
        isinstance(loader_audit, Mapping)
        and loader_audit.get("passed") is True
        and loader_audit.get("builder") == EXPECTED_BUILDERS[method]
        and loader_audit.get("training_state_key_count")
        == EXPECTED_TRAINING_STATE_KEYS[method]
        and loader_audit.get("expected_training_state_key_count")
        == EXPECTED_TRAINING_STATE_KEYS[method]
        and loader_audit.get("removed_training_only_tss_state_key_count")
        == EXPECTED_REMOVED_TSS_KEYS[method]
        and loader_audit.get("inference_state_key_count")
        == EXPECTED_INFERENCE_STATE_KEYS[method]
        and loader_audit.get("strict_load") is True
        and loader_audit.get("training_flag") is False
        and loader_audit.get("mode") == "test",
        "strict loader audit differs",
    )

    background_record = payload.get("background_pixel_authority_record")
    _require(
        isinstance(background_record, Mapping)
        and background_record.get("dataset") == payload.get("dataset")
        and background_record.get("checkpoint_role") == role
        and background_record.get("checkpoint_epoch") == manifest_entry.get("epoch")
        and background_record.get("checkpoint_sha256")
        == manifest_entry.get("checkpoint_sha256")
        and background_record.get("evaluation_sha256")
        == manifest_entry.get("evaluation_sha256"),
        "background authority record differs",
    )

    replay = payload.get("alpha0_historical_replay_audit")
    _require(
        isinstance(replay, Mapping)
        and replay.get("passed") is True
        and replay.get("counts_exact") is True
        and replay.get("background_false_positive_pixels_exact") is True
        and replay.get("within_frozen_float_tolerances") is True
        and replay.get("mode") == CURRENT_MODE
        and replay.get("checkpoint_role") == role
        and replay.get("reference_evaluation_sha256") == reference.get("sha256")
        and replay.get("background_pixel_authority_sha256")
        == bound_background.get("sha256"),
        "alpha0 historical replay differs",
    )
    modes = payload.get("modes")
    _require(isinstance(modes, Mapping) and set(modes) == set(MODE_ORDER), "mode set differs")
    invariant: tuple[int, int, int] | None = None
    for mode in MODE_ORDER:
        record = modes[mode]
        _require(isinstance(record, Mapping), f"{mode} is malformed")
        _require(record.get("mode") == mode, f"{mode} identity differs")
        _require(float(record.get("alpha")) == MODE_ALPHA[mode], f"{mode} alpha differs")
        fixed = record.get("fixed_threshold_0_5")
        _require(
            isinstance(fixed, Mapping) and _REQUIRED_FIXED_FIELDS <= set(fixed),
            f"{mode} fixed metrics are incomplete",
        )
        _require(float(fixed.get("threshold")) == FIXED_THRESHOLD, f"{mode} threshold differs")
        current_invariant = (
            _nonnegative_int(fixed.get("target_count"), f"{mode}.target_count"),
            _nonnegative_int(fixed.get("tiny_target_count"), f"{mode}.tiny_target_count"),
            _nonnegative_int(fixed.get("valid_pixel_count"), f"{mode}.valid_pixel_count"),
        )
        invariant = invariant or current_invariant
        _require(current_invariant == invariant, f"{mode} target/pixel totals differ")
        for key, value in fixed.items():
            if value is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
                _require(math.isfinite(float(value)), f"{mode}.{key} is non-finite")
        descriptive = record.get("descriptive_pd_fa")
        _require(isinstance(descriptive, Mapping), f"{mode} descriptive sweep missing")
        points = descriptive.get("points")
        _require(isinstance(points, list) and len(points) == 2, f"{mode} requires two sweep points")
        _require([float(point["threshold"]) for point in points] == list(SWEEP_THRESHOLDS), f"{mode} sweep differs")
        _require(
            points[1].get("selected_point_is_empty") is True
            and float(points[1]["pd"]) == 0.0
            and float(points[1]["fa"]) == 0.0,
            f"{mode} empty endpoint differs",
        )
        difference = record.get("probability_difference_to_current")
        _require(isinstance(difference, Mapping), f"{mode} probability difference missing")
        element_count = _nonnegative_int(difference.get("element_count"), f"{mode}.element_count")
        _require(element_count == current_invariant[2] and element_count > 0, f"{mode} difference scope differs")
        absolute_sum = _finite_float(difference.get("absolute_difference_sum"), f"{mode}.absolute_sum")
        mean_abs = _finite_float(difference.get("mean_abs"), f"{mode}.mean_abs")
        maximum = _finite_float(difference.get("max_abs"), f"{mode}.max_abs")
        _require(mean_abs == absolute_sum / element_count, f"{mode} mean difference identity differs")
        _require(maximum >= 0.0 and absolute_sum >= 0.0, f"{mode} difference is negative")
        if mode == CURRENT_MODE:
            _require(maximum == 0.0 and absolute_sum == 0.0 and mean_abs == 0.0, "current_out self-difference is nonzero")

    current_fixed = modes[CURRENT_MODE]["fixed_threshold_0_5"]
    _require(
        int(current_fixed["false_positive_pixels"])
        == int(background_record["false_positive_pixels"]),
        "current_out background authority replay differs",
    )
    data = payload.get("data")
    _require(isinstance(data, Mapping) and data.get("split") == "img_idx/test", "data binding differs")
    data_protocol_binding = data.get("protocol_manifest")
    input_binding = data.get("input_binding")
    _require(
        isinstance(data_protocol_binding, Mapping)
        and data_protocol_binding.get("path") == bound_data_manifest.get("path")
        and data_protocol_binding.get("sha256") == bound_data_manifest.get("sha256"),
        "data protocol/input-manifest binding differs",
    )
    _require(
        isinstance(input_binding, Mapping)
        and _is_sha256(input_binding.get("sha256"))
        and _is_sha256(input_binding.get("ordered_ids_newline_sha256")),
        "ordered input binding differs",
    )
    intervention = payload.get("intervention_contract")
    _require(
        isinstance(intervention, Mapping)
        and intervention.get("family")
        == "DORF_V1_existing_deep_supervision_readout_reuse"
        and intervention.get("formula") == "z_out + alpha * (z_d0 - z_out)"
        and intervention.get("fusion_space") == "raw_logits_before_sigmoid"
        and intervention.get("alphas")
        == [MODE_ALPHA[mode] for mode in MODE_ORDER]
        and intervention.get("one_checkpoint_per_unit") is True
        and intervention.get("model_parameters_changed") is False
        and intervention.get("persistent_buffers_changed") is False
        and intervention.get("derived_checkpoint_written") is False,
        "DORF intervention contract differs",
    )

    engineering = payload.get("engineering_audit")
    _require(isinstance(engineering, Mapping), "engineering audit missing")
    required_true = (
        "passed",
        "all_metrics_finite",
        "same_d0_out_logits_reused_for_all_modes",
        "one_model_forward_per_batch",
        "raw_logit_fusion",
        "model_state_unchanged",
        "model_training_flag_unchanged",
        "model_mode_unchanged",
        "each_hook_exactly_once_per_batch",
        "returned_probability_equals_sigmoid_raw_out_bitwise",
        "temporary_hooks_restored",
        "source_sha256_reverified_after_inference",
        "input_manifest_reverified_after_inference",
        "alpha0_historical_replay_passed",
        "derived_checkpoint_written",
        "probability_cache_written",
    )
    for key in required_true[:-2]:
        _require(engineering.get(key) is True, f"engineering audit failed: {key}")
    _require(
        engineering.get("derived_checkpoint_written") is False
        and engineering.get("probability_cache_written") is False,
        "engineering audit reports forbidden artifacts",
    )
    batch_count = _nonnegative_int(engineering.get("batch_count"), "engineering.batch_count")
    _require(
        batch_count > 0
        and engineering.get("model_forward_count") == batch_count
        and engineering.get("outc_hook_count") == batch_count
        and engineering.get("outconv_hook_count") == batch_count,
        "shared-forward/hook count contract differs",
    )
    _require(
        engineering.get("model_training_flag_before") is False
        and engineering.get("model_training_flag_after") is False
        and engineering.get("model_mode_before") == "test"
        and engineering.get("model_mode_after") == "test"
        and engineering.get("model_state_sha256_before")
        == engineering.get("model_state_sha256_after")
        and _is_sha256(engineering.get("model_state_sha256_before")),
        "model state/mode restoration contract differs",
    )
    _require(
        payload.get("derived_checkpoint_written") is False
        and payload.get("probability_cache_written") is False,
        "forbidden artifact flag differs",
    )
    json.dumps(payload, allow_nan=False)


def atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one JSON output exactly once, including concurrent writers."""

    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=data_protocol.DATASETS, required=True)
    parser.add_argument("--checkpoint-role", choices=CHECKPOINT_ROLES, required=True)
    parser.add_argument(
        "--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST
    )
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
    args.output = args.output or _default_output(
        args.method, args.dataset, args.checkpoint_role
    )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = analyze_run(
        method=args.method,
        dataset=args.dataset,
        checkpoint_role=args.checkpoint_role,
        run_dir=args.run_dir,
        dataset_root=args.dataset_root,
        data_protocol_manifest=args.data_protocol_manifest,
        reference_evaluation=args.reference_evaluation,
        device_name=args.device,
        workers=args.workers,
        input_manifest=args.input_manifest,
    )
    atomic_create_json(args.output, output)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_ROLES",
    "CURRENT_MODE",
    "DEFAULT_INPUT_MANIFEST",
    "DEFAULT_OUTPUT_ROOT",
    "FROZEN_INPUT_MANIFEST_SHA256",
    "METHODS",
    "MODE_ALPHA",
    "MODE_ORDER",
    "PUBLIC_MODES",
    "SCHEMA",
    "SEED",
    "SWEEP_THRESHOLDS",
    "RawD0OutHookCapture",
    "alpha0_historical_replay_audit",
    "analyze_loaded_model",
    "analyze_run",
    "atomic_create_json",
    "file_sha256",
    "fuse_raw_logits",
    "load_input_manifest_binding",
    "parse_args",
    "probability_difference",
    "validate_output_payload",
]
