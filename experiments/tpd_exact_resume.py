"""Reusable exact-resume state for future TPD training entries.

This module is deliberately independent from the currently running v4
screening jobs.  Legacy checkpoints that do not use ``EXACT_RESUME_SCHEMA``
cannot be upgraded after the fact because their random-number-generator
states and explicit DataLoader generator state were never recorded.

Two initialization intents are kept separate:

* exact resume restores a same-architecture training process at a recorded
  metrics boundary, including optimizer/scaler/scheduler and every RNG stream;
* parent warm start restores only a strictly matching parent model state and
  intentionally does not restore training or RNG state.

Callers should save exact-resume checkpoints only after the metrics event for
``epoch`` has been durably appended.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn as nn


EXACT_RESUME_SCHEMA = "sctransnet_tpd_exact_resume_v1"
PARENT_WARM_START_SCHEMA = "sctransnet_tpd_parent_warm_start_v1"
RNG_STATE_SCHEMA = "sctransnet_tpd_rng_state_v1"

EXACT_RESUME_MODE = "exact_resume"
PARENT_WARM_START_MODE = "parent_warm_start"

RUN_IDENTITY_REQUIRED_KEYS = frozenset(
    {
        "run_id",
        "variant",
        "architecture_id",
        "dataset",
        "seed",
        "split_seed",
        "split_sha256",
    }
)
METRICS_BOUNDARY_REQUIRED_KEYS = frozenset(
    {
        "completed_epoch",
        "event_count",
        "last_event_epoch",
        "metrics_sha256",
        "last_event_sha256",
    }
)
BEST_SELECTION_REQUIRED_KEYS = frozenset({"primary", "secondary"})
SELECTION_RECORD_REQUIRED_KEYS = frozenset({"role", "epoch", "key", "metrics"})
EXACT_RESUME_REQUIRED_KEYS = frozenset(
    {
        "schema",
        "mode",
        "epoch",
        "run_identity",
        "model",
        "optimizer",
        "scaler",
        "scheduler",
        "best_selection",
        "metrics_boundary",
        "rng_state",
        "extra_state",
    }
)
PARENT_WARM_START_REQUIRED_KEYS = frozenset(
    {
        "schema",
        "mode",
        "parent_epoch",
        "parent_identity",
        "model",
        "extra_state",
    }
)

_SHA256_LENGTH = 64


class ExactResumeValidationError(ValueError):
    """A checkpoint cannot satisfy the requested initialization contract."""


class InitializationMode(str, Enum):
    """Mutually exclusive training initialization choices."""

    FRESH = "fresh"
    EXACT_RESUME = EXACT_RESUME_MODE
    PARENT_WARM_START = PARENT_WARM_START_MODE


@dataclass(frozen=True)
class ExactResumeResult:
    """Non-module state returned after a successful exact restore."""

    epoch: int
    run_identity: dict[str, Any]
    best_selection: dict[str, Any]
    metrics_boundary: dict[str, Any]
    extra_state: dict[str, Any]


@dataclass(frozen=True)
class ParentWarmStartResult:
    """Provenance returned after loading parent weights only."""

    parent_epoch: int
    parent_identity: dict[str, Any]
    extra_state: dict[str, Any]


def _fail(message: str) -> None:
    raise ExactResumeValidationError(message)


def _qualified_class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        _fail(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{label} must be finite")
    return number


def _plain_json_value(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if _is_integer(value):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            _fail(f"{label} contains a non-finite number")
        return number
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                _fail(f"{label} keys must be non-empty strings")
            normalized[key] = _plain_json_value(item, f"{label}.{key}")
        return normalized
    if isinstance(value, (tuple, list)):
        return [
            _plain_json_value(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    _fail(f"{label} contains unsupported value type {_qualified_class_name(value)}")


def _canonical_json_bytes(value: Any, label: str) -> bytes:
    normalized = _plain_json_value(value, label)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _mapping_copy(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label} must be a mapping")
    return copy.deepcopy(dict(value))


def _require_exact_keys(
    value: Mapping[str, Any],
    required: frozenset[str],
    label: str,
) -> None:
    actual = set(value)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing:
        _fail(f"{label} missing required keys: {missing}")
    if extra:
        _fail(f"{label} has unsupported keys: {extra}")


def _validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_run_identity(
    identity: Any,
    label: str = "run_identity",
) -> dict[str, Any]:
    normalized = _plain_json_value(identity, label)
    if not isinstance(normalized, dict):
        _fail(f"{label} must be a mapping")
    missing = sorted(RUN_IDENTITY_REQUIRED_KEYS - set(normalized))
    if missing:
        _fail(f"{label} missing required keys: {missing}")
    for key in ("run_id", "variant", "architecture_id", "dataset"):
        if not isinstance(normalized[key], str) or not normalized[key]:
            _fail(f"{label}.{key} must be a non-empty string")
    for key in ("seed", "split_seed"):
        if not _is_integer(normalized[key]):
            _fail(f"{label}.{key} must be an integer")
        normalized[key] = int(normalized[key])
    _validate_sha256(normalized["split_sha256"], f"{label}.split_sha256")
    return normalized


def _identities_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _canonical_json_bytes(left, "left identity") == _canonical_json_bytes(
        right, "right identity"
    )


def _state_layout_records(state_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, value in state_dict.items():
        if not isinstance(name, str):
            _fail("model state_dict keys must be strings")
        if isinstance(value, torch.Tensor):
            records.append(
                {
                    "name": name,
                    "kind": "tensor",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )
        else:
            records.append(
                {
                    "name": name,
                    "kind": "extra_state",
                    "type": _qualified_class_name(value),
                }
            )
    return records


def model_layout(model: nn.Module) -> dict[str, Any]:
    """Return a value-independent model layout used for strict architecture checks."""

    state_dict = model.state_dict()
    records = _state_layout_records(state_dict)
    digest = hashlib.sha256(
        _canonical_json_bytes(records, "model layout records")
    ).hexdigest()
    tensor_values = [
        value for value in state_dict.values() if isinstance(value, torch.Tensor)
    ]
    return {
        "class": _qualified_class_name(model),
        "layout_sha256": digest,
        "state_entry_count": len(records),
        "state_tensor_count": len(tensor_values),
        "state_tensor_numel": sum(int(value.numel()) for value in tensor_values),
    }


def _validate_model_component(value: Any, label: str = "model") -> dict[str, Any]:
    component = _mapping_copy(value, label)
    _require_exact_keys(
        component,
        frozenset({"layout", "state_dict"}),
        label,
    )
    layout = _plain_json_value(component["layout"], f"{label}.layout")
    if not isinstance(layout, dict):
        _fail(f"{label}.layout must be a mapping")
    expected_layout_keys = {
        "class",
        "layout_sha256",
        "state_entry_count",
        "state_tensor_count",
        "state_tensor_numel",
    }
    if set(layout) != expected_layout_keys:
        _fail(f"{label}.layout keys mismatch")
    if not isinstance(component["state_dict"], Mapping):
        _fail(f"{label}.state_dict must be a mapping")
    _validate_sha256(layout["layout_sha256"], f"{label}.layout.layout_sha256")
    for key in ("state_entry_count", "state_tensor_count", "state_tensor_numel"):
        if not _is_integer(layout[key]) or int(layout[key]) < 0:
            _fail(f"{label}.layout.{key} must be a non-negative integer")
        layout[key] = int(layout[key])
    if not isinstance(layout["class"], str) or not layout["class"]:
        _fail(f"{label}.layout.class must be a non-empty string")
    component["layout"] = layout
    return component


def _training_component(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    return {
        "class": _qualified_class_name(value),
        "state_dict": copy.deepcopy(value.state_dict()),
    }


def _optimizer_parameter_names(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    label: str,
) -> list[list[str]]:
    """Bind optimizer positions to canonical fully qualified model names."""

    name_by_parameter_id = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    if not name_by_parameter_id:
        _fail(f"{label}: model has no named parameters")
    ordered_groups: list[list[str]] = []
    seen_names: set[str] = set()
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = group.get("params")
        if not isinstance(parameters, (tuple, list)):
            _fail(f"{label}: param group {group_index} params must be a sequence")
        group_names: list[str] = []
        for parameter_index, parameter in enumerate(parameters):
            name = name_by_parameter_id.get(id(parameter))
            if name is None:
                _fail(
                    f"{label}: param group {group_index} position "
                    f"{parameter_index} is not a model parameter"
                )
            if name in seen_names:
                _fail(f"{label}: model parameter {name!r} appears more than once")
            seen_names.add(name)
            group_names.append(name)
        ordered_groups.append(group_names)
    model_names = set(name_by_parameter_id.values())
    missing = sorted(model_names - seen_names)
    if missing:
        _fail(f"{label}: optimizer is missing model parameters: {missing}")
    return ordered_groups


def _optimizer_component(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "class": _qualified_class_name(optimizer),
        "parameter_names": _optimizer_parameter_names(
            model,
            optimizer,
            label="optimizer",
        ),
        "state_dict": copy.deepcopy(optimizer.state_dict()),
    }


def _validate_training_component(value: Any, label: str) -> dict[str, Any]:
    component = _mapping_copy(value, label)
    _require_exact_keys(component, frozenset({"class", "state_dict"}), label)
    if not isinstance(component["class"], str) or not component["class"]:
        _fail(f"{label}.class must be a non-empty string")
    if not isinstance(component["state_dict"], Mapping):
        _fail(f"{label}.state_dict must be a mapping")
    return component


def _validate_optimizer_component(
    value: Any,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    component = _mapping_copy(value, "optimizer")
    _require_exact_keys(
        component,
        frozenset({"class", "parameter_names", "state_dict"}),
        "optimizer",
    )
    if not isinstance(component["class"], str) or not component["class"]:
        _fail("optimizer.class must be a non-empty string")
    if not isinstance(component["state_dict"], Mapping):
        _fail("optimizer.state_dict must be a mapping")
    saved_groups = component["parameter_names"]
    if not isinstance(saved_groups, (tuple, list)):
        _fail("optimizer.parameter_names must be a sequence")

    model_names = {name for name, _ in model.named_parameters()}
    normalized_groups: list[list[str]] = []
    seen_names: set[str] = set()
    for group_index, group in enumerate(saved_groups):
        if not isinstance(group, (tuple, list)):
            _fail(
                f"optimizer.parameter_names[{group_index}] must be a sequence"
            )
        normalized_group: list[str] = []
        for parameter_index, name in enumerate(group):
            if not isinstance(name, str) or not name:
                _fail(
                    "optimizer parameter names must be non-empty strings "
                    f"(group {group_index}, position {parameter_index})"
                )
            if name not in model_names:
                _fail(f"optimizer names non-model parameter {name!r}")
            if name in seen_names:
                _fail(f"optimizer names duplicate model parameter {name!r}")
            seen_names.add(name)
            normalized_group.append(name)
        normalized_groups.append(normalized_group)
    missing = sorted(model_names - seen_names)
    if missing:
        _fail(f"optimizer names omit model parameters: {missing}")

    current_groups = _optimizer_parameter_names(
        model,
        optimizer,
        label="current optimizer",
    )
    if normalized_groups != current_groups:
        _fail("optimizer parameter name/order binding mismatch")
    component["parameter_names"] = normalized_groups
    return component


def _validate_metric_mapping(value: Any, label: str) -> dict[str, int | float]:
    metrics = _mapping_copy(value, label)
    normalized: dict[str, int | float] = {}
    tiny_count = metrics.get("tiny_target_count")
    for key, item in metrics.items():
        if not isinstance(key, str) or not key:
            _fail(f"{label} keys must be non-empty strings")
        if isinstance(item, bool) or not isinstance(item, (int, float, np.number)):
            _fail(f"{label}.{key} must be numeric")
        number = float(item)
        if not math.isfinite(number):
            allow_empty_tiny_pd = (
                key == "tiny_pd"
                and math.isnan(number)
                and _is_integer(tiny_count)
                and int(tiny_count) == 0
            )
            if not allow_empty_tiny_pd:
                _fail(f"{label}.{key} must be finite")
        normalized[key] = int(item) if _is_integer(item) else number
    if not normalized:
        _fail(f"{label} must not be empty")
    return normalized


def _validate_best_selection(value: Any, completed_epoch: int) -> dict[str, Any]:
    selection = _mapping_copy(value, "best_selection")
    _require_exact_keys(
        selection,
        BEST_SELECTION_REQUIRED_KEYS,
        "best_selection",
    )
    normalized: dict[str, Any] = {}
    for name in ("primary", "secondary"):
        record = _mapping_copy(selection[name], f"best_selection.{name}")
        _require_exact_keys(
            record,
            SELECTION_RECORD_REQUIRED_KEYS,
            f"best_selection.{name}",
        )
        role = record["role"]
        if not isinstance(role, str) or not role:
            _fail(f"best_selection.{name}.role must be a non-empty string")
        record_epoch = record["epoch"]
        if (
            not _is_integer(record_epoch)
            or int(record_epoch) < 1
            or int(record_epoch) > completed_epoch
        ):
            _fail(
                f"best_selection.{name}.epoch must be within "
                f"[1, {completed_epoch}]"
            )
        key = record["key"]
        if not isinstance(key, (tuple, list)) or not key:
            _fail(f"best_selection.{name}.key must be a non-empty sequence")
        normalized_key = [
            _finite_number(item, f"best_selection.{name}.key[{index}]")
            for index, item in enumerate(key)
        ]
        normalized[name] = {
            "role": role,
            "epoch": int(record_epoch),
            "key": normalized_key,
            "metrics": _validate_metric_mapping(
                record["metrics"],
                f"best_selection.{name}.metrics",
            ),
        }
    return normalized


def _nested_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _nested_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _nested_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, (float, np.floating)) and isinstance(
        right, (float, np.floating)
    ):
        if math.isnan(float(left)) and math.isnan(float(right)):
            return True
    return bool(left == right)


def _validate_metrics_boundary(
    value: Any,
    completed_epoch: int,
) -> dict[str, Any]:
    boundary = _plain_json_value(value, "metrics_boundary")
    if not isinstance(boundary, dict):
        _fail("metrics_boundary must be a mapping")
    _require_exact_keys(
        boundary,
        METRICS_BOUNDARY_REQUIRED_KEYS,
        "metrics_boundary",
    )
    for key in ("completed_epoch", "event_count", "last_event_epoch"):
        if not _is_integer(boundary[key]):
            _fail(f"metrics_boundary.{key} must be an integer")
        boundary[key] = int(boundary[key])
    if boundary["completed_epoch"] != completed_epoch:
        _fail("metrics_boundary.completed_epoch differs from checkpoint epoch")
    if boundary["last_event_epoch"] != completed_epoch:
        _fail("metrics_boundary.last_event_epoch differs from checkpoint epoch")
    if boundary["event_count"] != completed_epoch:
        _fail("metrics_boundary.event_count must equal checkpoint epoch")
    _validate_sha256(
        boundary["metrics_sha256"],
        "metrics_boundary.metrics_sha256",
    )
    _validate_sha256(
        boundary["last_event_sha256"],
        "metrics_boundary.last_event_sha256",
    )
    return boundary


def _reject_json_constant(value: str) -> None:
    _fail(f"metrics JSONL contains a non-finite constant: {value}")


def metrics_boundary_from_jsonl(
    path: str | os.PathLike[str],
    *,
    expected_epoch: int,
) -> dict[str, Any]:
    """Hash and validate a contiguous one-event-per-epoch metrics prefix."""

    metrics_path = Path(path)
    if not metrics_path.is_file() or metrics_path.is_symlink():
        _fail(f"metrics JSONL is not a regular file: {metrics_path}")
    if not _is_integer(expected_epoch) or int(expected_epoch) < 1:
        _fail("expected_epoch must be a positive integer")
    content = metrics_path.read_bytes()
    if not content or not content.endswith(b"\n"):
        _fail("metrics JSONL must be non-empty and newline terminated")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        _fail(f"metrics JSONL is not UTF-8: {exc}")
    raw_lines = text.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.endswith("\n") or not raw_line.strip():
            _fail(f"metrics JSONL has an invalid line at {line_number}")
        try:
            row = json.loads(raw_line, parse_constant=_reject_json_constant)
        except ExactResumeValidationError:
            raise
        except json.JSONDecodeError as exc:
            _fail(f"metrics JSONL line {line_number} is invalid: {exc}")
        if not isinstance(row, dict):
            _fail(f"metrics JSONL line {line_number} must be an object")
        if row.get("epoch") != line_number:
            _fail(f"metrics JSONL epochs are not contiguous at line {line_number}")
        rows.append(row)
    if len(rows) != int(expected_epoch):
        _fail(
            f"metrics JSONL event count {len(rows)} differs from "
            f"expected epoch {expected_epoch}"
        )
    last_line = raw_lines[-1].encode("utf-8")
    return {
        "completed_epoch": int(expected_epoch),
        "event_count": len(rows),
        "last_event_epoch": int(rows[-1]["epoch"]),
        "metrics_sha256": hashlib.sha256(content).hexdigest(),
        "last_event_sha256": hashlib.sha256(last_line).hexdigest(),
    }


def _validate_cpu_rng_tensor(value: Any, label: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.uint8
        or value.ndim != 1
        or value.numel() == 0
    ):
        _fail(f"{label} must be a non-empty 1-D CPU uint8 tensor")
    return value


def capture_rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU/CUDA, and DataLoader RNG streams."""

    if not isinstance(loader_generator, torch.Generator):
        _fail("loader_generator must be an explicit torch.Generator")
    if loader_generator.device.type != "cpu":
        _fail("DataLoader generator must be a CPU torch.Generator")
    cuda_available = bool(torch.cuda.is_available())
    cuda_states = torch.cuda.get_rng_state_all() if cuda_available else []
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    if len(cuda_states) != cuda_device_count:
        _fail("captured CUDA RNG state count differs from visible CUDA device count")
    return {
        "schema": RNG_STATE_SCHEMA,
        "python_random": copy.deepcopy(random.getstate()),
        "numpy_random": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": cuda_device_count,
        "torch_cuda": [state.detach().cpu().clone() for state in cuda_states],
        "loader_generator_device": str(loader_generator.device),
        "loader_generator": loader_generator.get_state().clone(),
    }


def _validate_rng_state(
    value: Any,
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    state = _mapping_copy(value, "rng_state")
    required = frozenset(
        {
            "schema",
            "python_random",
            "numpy_random",
            "torch_cpu",
            "torch_cuda_available",
            "torch_cuda_device_count",
            "torch_cuda",
            "loader_generator_device",
            "loader_generator",
        }
    )
    _require_exact_keys(state, required, "rng_state")
    if state["schema"] != RNG_STATE_SCHEMA:
        _fail("rng_state schema mismatch")
    if not isinstance(loader_generator, torch.Generator):
        _fail("loader_generator must be an explicit torch.Generator")
    if loader_generator.device.type != "cpu":
        _fail("DataLoader generator must be a CPU torch.Generator")
    if state["loader_generator_device"] != str(loader_generator.device):
        _fail("DataLoader generator device mismatch")

    try:
        random.Random().setstate(state["python_random"])
    except (TypeError, ValueError) as exc:
        _fail(f"invalid Python random state: {exc}")
    try:
        np.random.RandomState().set_state(state["numpy_random"])
    except (TypeError, ValueError) as exc:
        _fail(f"invalid NumPy random state: {exc}")
    cpu_state = _validate_cpu_rng_tensor(state["torch_cpu"], "rng_state.torch_cpu")
    loader_state = _validate_cpu_rng_tensor(
        state["loader_generator"],
        "rng_state.loader_generator",
    )
    try:
        torch.Generator(device="cpu").set_state(cpu_state)
        torch.Generator(device="cpu").set_state(loader_state)
    except RuntimeError as exc:
        _fail(f"invalid Torch CPU RNG state: {exc}")

    captured_cuda = state["torch_cuda_available"]
    if not isinstance(captured_cuda, bool):
        _fail("rng_state.torch_cuda_available must be boolean")
    captured_count = state["torch_cuda_device_count"]
    if not _is_integer(captured_count) or int(captured_count) < 0:
        _fail("rng_state.torch_cuda_device_count must be non-negative")
    captured_count = int(captured_count)
    cuda_states = state["torch_cuda"]
    if not isinstance(cuda_states, (tuple, list)):
        _fail("rng_state.torch_cuda must be a sequence")
    cuda_states = [
        _validate_cpu_rng_tensor(item, f"rng_state.torch_cuda[{index}]")
        for index, item in enumerate(cuda_states)
    ]
    if captured_cuda:
        if not torch.cuda.is_available():
            _fail("checkpoint requires CUDA RNG restoration but CUDA is unavailable")
        current_count = torch.cuda.device_count()
        if captured_count != current_count:
            _fail(
                "visible CUDA device count differs from checkpoint: "
                f"{current_count} != {captured_count}"
            )
        if len(cuda_states) != captured_count:
            _fail("checkpoint CUDA RNG state count mismatch")
    elif captured_count != 0 or cuda_states:
        _fail("CPU checkpoint contains inconsistent CUDA RNG metadata")

    state["torch_cpu"] = cpu_state
    state["loader_generator"] = loader_state
    state["torch_cuda_device_count"] = captured_count
    state["torch_cuda"] = cuda_states
    return state


def _apply_rng_state(
    state: Mapping[str, Any],
    loader_generator: torch.Generator,
) -> None:
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda_available"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    loader_generator.set_state(state["loader_generator"])


def restore_rng_state(
    state: Mapping[str, Any],
    loader_generator: torch.Generator,
) -> None:
    """Transactionally restore every random stream captured by this module."""

    validated = _validate_rng_state(state, loader_generator)
    previous = capture_rng_state(loader_generator)
    try:
        _apply_rng_state(validated, loader_generator)
    except BaseException:
        _apply_rng_state(previous, loader_generator)
        raise


def _validate_optimizer_topology(
    saved: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> None:
    saved_state = saved.get("state")
    saved_groups = saved.get("param_groups")
    current_groups = optimizer.state_dict().get("param_groups")
    if not isinstance(saved_state, Mapping):
        _fail("optimizer state mapping is invalid")
    if not isinstance(saved_groups, list) or not isinstance(current_groups, list):
        _fail("optimizer param_groups are invalid")
    if len(saved_groups) != len(current_groups):
        _fail("optimizer param-group count mismatch")
    saved_parameter_ids: set[int] = set()
    for index, (saved_group, current_group) in enumerate(
        zip(saved_groups, current_groups)
    ):
        if not isinstance(saved_group, Mapping) or not isinstance(
            current_group, Mapping
        ):
            _fail(f"optimizer param group {index} is invalid")
        saved_params = saved_group.get("params")
        current_params = current_group.get("params")
        if not isinstance(saved_params, list) or not isinstance(current_params, list):
            _fail(f"optimizer param group {index} has invalid params")
        if len(saved_params) != len(current_params):
            _fail(f"optimizer param group {index} parameter count mismatch")
        for parameter_id in saved_params:
            if not _is_integer(parameter_id) or int(parameter_id) < 0:
                _fail(f"optimizer param group {index} has invalid parameter ids")
            parameter_id = int(parameter_id)
            if parameter_id in saved_parameter_ids:
                _fail("optimizer state contains a duplicate parameter id")
            saved_parameter_ids.add(parameter_id)
    for parameter_id in saved_state:
        if not _is_integer(parameter_id) or int(parameter_id) not in saved_parameter_ids:
            _fail("optimizer state contains a non-group parameter id")


def _component_matches_instance(
    component: Mapping[str, Any],
    instance: Any,
    label: str,
) -> None:
    current_class = _qualified_class_name(instance)
    if component["class"] != current_class:
        _fail(
            f"{label} class mismatch: {component['class']!r} != {current_class!r}"
        )


def build_exact_resume_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    run_identity: Mapping[str, Any],
    best_selection: Mapping[str, Any],
    metrics_boundary: Mapping[str, Any],
    loader_generator: torch.Generator,
    scheduler: Any | None = None,
    extra_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fully self-contained exact-resume payload at an epoch boundary."""

    if not _is_integer(epoch) or int(epoch) < 1:
        _fail("epoch must be a positive integer")
    completed_epoch = int(epoch)
    identity = _validate_run_identity(run_identity)
    selection = _validate_best_selection(best_selection, completed_epoch)
    boundary = _validate_metrics_boundary(metrics_boundary, completed_epoch)
    extra = _plain_json_value(extra_state or {}, "extra_state")
    if not isinstance(extra, dict):
        _fail("extra_state must be a mapping")
    return {
        "schema": EXACT_RESUME_SCHEMA,
        "mode": EXACT_RESUME_MODE,
        "epoch": completed_epoch,
        "run_identity": identity,
        "model": {
            "layout": model_layout(model),
            "state_dict": copy.deepcopy(model.state_dict()),
        },
        "optimizer": _optimizer_component(model, optimizer),
        "scaler": _training_component(scaler, label="scaler"),
        "scheduler": (
            _training_component(scheduler, label="scheduler")
            if scheduler is not None
            else None
        ),
        "best_selection": selection,
        "metrics_boundary": boundary,
        "rng_state": capture_rng_state(loader_generator),
        "extra_state": extra,
    }


def _load_payload(
    checkpoint: Mapping[str, Any] | str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    if isinstance(checkpoint, Mapping):
        return copy.deepcopy(dict(checkpoint))
    path = Path(checkpoint)
    if not path.is_file() or path.is_symlink():
        _fail(f"checkpoint is not a regular file: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:
        _fail(f"cannot load checkpoint {path}: {exc}")
    if not isinstance(payload, Mapping):
        _fail("checkpoint top level must be a mapping")
    return dict(payload)


def _validate_exact_resume_payload(
    payload: Mapping[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    loader_generator: torch.Generator,
    scheduler: Any | None,
    expected_run_identity: Mapping[str, Any],
    expected_epoch: int | None,
    expected_metrics_boundary: Mapping[str, Any],
    expected_best_selection: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = _mapping_copy(payload, "checkpoint")
    _require_exact_keys(checkpoint, EXACT_RESUME_REQUIRED_KEYS, "checkpoint")
    if checkpoint["schema"] != EXACT_RESUME_SCHEMA:
        _fail("exact-resume checkpoint schema mismatch")
    if checkpoint["mode"] != EXACT_RESUME_MODE:
        _fail("checkpoint is not an exact-resume payload")
    epoch = checkpoint["epoch"]
    if not _is_integer(epoch) or int(epoch) < 1:
        _fail("checkpoint epoch must be a positive integer")
    epoch = int(epoch)
    if expected_epoch is not None:
        if not _is_integer(expected_epoch) or int(expected_epoch) < 1:
            _fail("expected_epoch must be a positive integer")
        if epoch != int(expected_epoch):
            _fail(
                f"checkpoint epoch mismatch: {epoch} != {int(expected_epoch)}"
            )

    identity = _validate_run_identity(checkpoint["run_identity"])
    expected_identity = _validate_run_identity(
        expected_run_identity,
        "expected_run_identity",
    )
    if not _identities_equal(identity, expected_identity):
        _fail("run identity mismatch")

    model_component = _validate_model_component(checkpoint["model"])
    current_layout = model_layout(model)
    if model_component["layout"] != current_layout:
        _fail("model architecture/layout mismatch")
    optimizer_component = _validate_optimizer_component(
        checkpoint["optimizer"],
        model=model,
        optimizer=optimizer,
    )
    scaler_component = _validate_training_component(
        checkpoint["scaler"],
        "scaler",
    )
    _component_matches_instance(optimizer_component, optimizer, "optimizer")
    _component_matches_instance(scaler_component, scaler, "scaler")
    _validate_optimizer_topology(
        optimizer_component["state_dict"],
        optimizer,
    )

    scheduler_component: dict[str, Any] | None
    if checkpoint["scheduler"] is None:
        if scheduler is not None:
            _fail("checkpoint has no scheduler state but a scheduler was supplied")
        scheduler_component = None
    else:
        if scheduler is None:
            _fail("checkpoint requires a scheduler but none was supplied")
        scheduler_component = _validate_training_component(
            checkpoint["scheduler"],
            "scheduler",
        )
        _component_matches_instance(scheduler_component, scheduler, "scheduler")

    selection = _validate_best_selection(checkpoint["best_selection"], epoch)
    expected_selection = _validate_best_selection(
        expected_best_selection,
        epoch,
    )
    if not _nested_values_equal(selection, expected_selection):
        _fail("best selection mismatch")
    boundary = _validate_metrics_boundary(checkpoint["metrics_boundary"], epoch)
    expected_boundary = _validate_metrics_boundary(
        expected_metrics_boundary,
        epoch,
    )
    if boundary != expected_boundary:
        _fail("metrics boundary mismatch")
    rng_state = _validate_rng_state(
        checkpoint["rng_state"],
        loader_generator,
    )
    extra = _plain_json_value(checkpoint["extra_state"], "extra_state")
    if not isinstance(extra, dict):
        _fail("extra_state must be a mapping")
    checkpoint.update(
        {
            "epoch": epoch,
            "run_identity": identity,
            "model": model_component,
            "optimizer": optimizer_component,
            "scaler": scaler_component,
            "scheduler": scheduler_component,
            "best_selection": selection,
            "metrics_boundary": boundary,
            "rng_state": rng_state,
            "extra_state": extra,
        }
    )
    return checkpoint


def restore_exact_resume(
    checkpoint: Mapping[str, Any] | str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    loader_generator: torch.Generator,
    expected_run_identity: Mapping[str, Any],
    expected_metrics_boundary: Mapping[str, Any],
    expected_best_selection: Mapping[str, Any],
    expected_epoch: int | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> ExactResumeResult:
    """Validate external run boundaries, then restore a same-run checkpoint."""

    payload = _load_payload(checkpoint, map_location=map_location)
    validated = _validate_exact_resume_payload(
        payload,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        loader_generator=loader_generator,
        scheduler=scheduler,
        expected_run_identity=expected_run_identity,
        expected_epoch=expected_epoch,
        expected_metrics_boundary=expected_metrics_boundary,
        expected_best_selection=expected_best_selection,
    )

    model_previous = copy.deepcopy(model.state_dict())
    optimizer_previous = copy.deepcopy(optimizer.state_dict())
    scaler_previous = copy.deepcopy(scaler.state_dict())
    scheduler_previous = (
        copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
    )
    rng_previous = capture_rng_state(loader_generator)
    try:
        model.load_state_dict(validated["model"]["state_dict"], strict=True)
        optimizer.load_state_dict(validated["optimizer"]["state_dict"])
        scaler.load_state_dict(validated["scaler"]["state_dict"])
        if scheduler is not None:
            assert validated["scheduler"] is not None
            scheduler.load_state_dict(validated["scheduler"]["state_dict"])
        _apply_rng_state(validated["rng_state"], loader_generator)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for label, restore in (
            (
                "model",
                lambda: model.load_state_dict(model_previous, strict=True),
            ),
            (
                "optimizer",
                lambda: optimizer.load_state_dict(optimizer_previous),
            ),
            ("scaler", lambda: scaler.load_state_dict(scaler_previous)),
            (
                "scheduler",
                lambda: (
                    scheduler.load_state_dict(scheduler_previous)
                    if scheduler is not None
                    else None
                ),
            ),
            ("rng", lambda: _apply_rng_state(rng_previous, loader_generator)),
        ):
            try:
                restore()
            except BaseException as rollback_exc:
                rollback_errors.append(f"{label}: {rollback_exc}")
        suffix = (
            f"; rollback also failed ({'; '.join(rollback_errors)})"
            if rollback_errors
            else ""
        )
        raise ExactResumeValidationError(
            f"exact-resume state load failed: {exc}{suffix}"
        ) from exc

    return ExactResumeResult(
        epoch=validated["epoch"],
        run_identity=copy.deepcopy(validated["run_identity"]),
        best_selection=copy.deepcopy(validated["best_selection"]),
        metrics_boundary=copy.deepcopy(validated["metrics_boundary"]),
        extra_state=copy.deepcopy(validated["extra_state"]),
    )


def build_parent_warm_start_checkpoint(
    *,
    parent_model: nn.Module,
    parent_epoch: int,
    parent_identity: Mapping[str, Any],
    extra_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a weight-only artifact for an explicitly selected parent module."""

    if not _is_integer(parent_epoch) or int(parent_epoch) < 1:
        _fail("parent_epoch must be a positive integer")
    identity = _validate_run_identity(parent_identity, "parent_identity")
    extra = _plain_json_value(extra_state or {}, "extra_state")
    if not isinstance(extra, dict):
        _fail("extra_state must be a mapping")
    return {
        "schema": PARENT_WARM_START_SCHEMA,
        "mode": PARENT_WARM_START_MODE,
        "parent_epoch": int(parent_epoch),
        "parent_identity": identity,
        "model": {
            "layout": model_layout(parent_model),
            "state_dict": copy.deepcopy(parent_model.state_dict()),
        },
        "extra_state": extra,
    }


def restore_parent_warm_start(
    checkpoint: Mapping[str, Any] | str | os.PathLike[str],
    *,
    parent_model: nn.Module,
    expected_parent_identity: Mapping[str, Any],
    expected_parent_epoch: int | None = None,
    map_location: str | torch.device = "cpu",
) -> ParentWarmStartResult:
    """Restore parent weights only; optimizer, selection, and RNG stay untouched."""

    payload = _load_payload(checkpoint, map_location=map_location)
    _require_exact_keys(
        payload,
        PARENT_WARM_START_REQUIRED_KEYS,
        "parent warm-start checkpoint",
    )
    if payload["schema"] != PARENT_WARM_START_SCHEMA:
        _fail("parent warm-start checkpoint schema mismatch")
    if payload["mode"] != PARENT_WARM_START_MODE:
        _fail("checkpoint is not a parent warm-start payload")
    parent_epoch = payload["parent_epoch"]
    if not _is_integer(parent_epoch) or int(parent_epoch) < 1:
        _fail("parent_epoch must be a positive integer")
    parent_epoch = int(parent_epoch)
    if expected_parent_epoch is not None:
        if (
            not _is_integer(expected_parent_epoch)
            or int(expected_parent_epoch) < 1
        ):
            _fail("expected_parent_epoch must be a positive integer")
        if parent_epoch != int(expected_parent_epoch):
            _fail("parent warm-start epoch mismatch")
    identity = _validate_run_identity(
        payload["parent_identity"],
        "parent_identity",
    )
    expected_identity = _validate_run_identity(
        expected_parent_identity,
        "expected_parent_identity",
    )
    if not _identities_equal(identity, expected_identity):
        _fail("parent identity mismatch")
    component = _validate_model_component(payload["model"])
    if component["layout"] != model_layout(parent_model):
        _fail("parent model architecture/layout mismatch")
    extra = _plain_json_value(payload["extra_state"], "extra_state")
    if not isinstance(extra, dict):
        _fail("extra_state must be a mapping")

    previous = copy.deepcopy(parent_model.state_dict())
    try:
        parent_model.load_state_dict(component["state_dict"], strict=True)
    except BaseException as exc:
        try:
            parent_model.load_state_dict(previous, strict=True)
        except BaseException as rollback_exc:
            raise ExactResumeValidationError(
                "parent warm-start load failed and rollback failed: "
                f"{exc}; {rollback_exc}"
            ) from exc
        raise ExactResumeValidationError(
            f"parent warm-start model load failed: {exc}"
        ) from exc
    return ParentWarmStartResult(
        parent_epoch=parent_epoch,
        parent_identity=copy.deepcopy(identity),
        extra_state=copy.deepcopy(extra),
    )


def select_initialization_mode(
    *,
    exact_resume: Any | None = None,
    parent_warm_start: Any | None = None,
) -> InitializationMode:
    """Reject ambiguous CLI/config requests before any checkpoint is loaded."""

    has_exact = exact_resume is not None
    has_parent = parent_warm_start is not None
    if has_exact and has_parent:
        _fail("exact resume and parent warm start are mutually exclusive")
    if has_exact:
        return InitializationMode.EXACT_RESUME
    if has_parent:
        return InitializationMode.PARENT_WARM_START
    return InitializationMode.FRESH


def atomic_torch_save(
    payload: Any,
    destination: str | os.PathLike[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Durably save with same-directory atomic replacement and cleanup."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.is_symlink():
        raise RuntimeError(
            f"refusing to replace checkpoint symlink: {destination_path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.tmp-",
        dir=destination_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path, *args, **kwargs)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination_path)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(destination_path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


__all__ = [
    "EXACT_RESUME_MODE",
    "EXACT_RESUME_SCHEMA",
    "PARENT_WARM_START_MODE",
    "PARENT_WARM_START_SCHEMA",
    "RNG_STATE_SCHEMA",
    "ExactResumeResult",
    "ExactResumeValidationError",
    "InitializationMode",
    "ParentWarmStartResult",
    "atomic_torch_save",
    "build_exact_resume_checkpoint",
    "build_parent_warm_start_checkpoint",
    "capture_rng_state",
    "metrics_boundary_from_jsonl",
    "model_layout",
    "restore_exact_resume",
    "restore_parent_warm_start",
    "restore_rng_state",
    "select_initialization_mode",
]
