"""Reusable control plane for exact TPD experiment runners.

The module intentionally owns no dataset or model-specific training loop.
Thin command-line entries construct their model, optimizer, scaler, explicit
DataLoader generator and immutable :class:`ExactRunSpec`, then use
``next_epoch_control`` and ``commit_epoch`` around their existing epoch code.

The active A/B epoch journal is the authority for completed epochs.  The
familiar ``metrics.jsonl``, ``last.pth.tar``, ``best.pth.tar`` and
``best_miou.pth.tar`` files are atomic, repairable compatibility views.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
import stat
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from experiments import tpd_exact_resume as exact
from experiments import tpd_exact_epoch_journal as journal_module
from experiments import tpd_extension_warm_start as extension_warm_start
from experiments.tpd_exact_epoch_journal import ActiveEpochState, ExactEpochJournal
from experiments.tpd_exact_training_runtime import (
    ExactTrainingRuntime,
    ExactTrainingSnapshot,
    PreparedExactEpoch,
)


RUNNER_SCHEMA = "sctransnet_tpd_exact_runner_v1"
ARCHITECTURE_SCHEMA = "sctransnet_tpd_architecture_identity_v1"
RUN_IDENTITY_SCHEMA = "sctransnet_tpd_run_identity_v1"
ORDERED_FINGERPRINT_SCHEMA = "sctransnet_tpd_ordered_fingerprint_v1"
DERIVED_CHECKPOINT_SCHEMA = "sctransnet_tpd_derived_checkpoint_v1"
MANUAL_COSINE_SCHEMA = "sctransnet_tpd_manual_cosine_v1"
EXTENSION_PARENT_MODE = "extension_parent_warm_start"
OPTIMIZER_CONTRACT_SCHEMA = "sctransnet_tpd_optimizer_contract_v1"
SCALER_CONTRACT_SCHEMA = "sctransnet_tpd_scaler_contract_v1"
INITIALIZATION_CONTRACT_SCHEMA = "sctransnet_tpd_initialization_contract_v1"
INITIAL_RNG_CONTRACT_SCHEMA = "sctransnet_tpd_initial_rng_contract_v1"
SELECTION_POLICY_SCHEMA = "sctransnet_tpd_selection_policy_v1"
STATE_CONTENT_SCHEMA = "sctransnet_tpd_state_content_v1"

METRICS_FILENAME = "metrics.jsonl"
LAST_FILENAME = "last.pth.tar"
BEST_FILENAME = "best.pth.tar"
BEST_MIOU_FILENAME = "best_miou.pth.tar"
JOURNAL_DIRECTORY = "exact_journal"

_SHA256_LENGTH = 64
_DERIVED_SOURCE_FIELDS = frozenset(
    {
        "source_exact_checkpoint_sha256",
        "state_dict_sha256",
        "optimizer_state_sha256",
        "scaler_state_sha256",
    }
)
_PROTECTED_EVENT_FIELDS = frozenset(
    {
        "epoch",
        "learning_rate",
        "new_best_pd",
        "new_best_miou",
    }
)


class ExactRunnerError(RuntimeError):
    """The generic runner contract or its derived views are inconsistent."""


def _fail(message: str) -> None:
    raise ExactRunnerError(message)


def _qualified_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _plain_json(value: Any, label: str) -> Any:
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
            normalized[key] = _plain_json(item, f"{label}.{key}")
        return normalized
    if isinstance(value, (tuple, list)):
        return [
            _plain_json(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    _fail(f"{label} contains unsupported type {_qualified_name(value)}")


def _canonical_json(value: Any, label: str) -> bytes:
    return json.dumps(
        _plain_json(value, label),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any, label: str) -> str:
    return hashlib.sha256(_canonical_json(value, label)).hexdigest()


def _validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty string")
    return value


def _validate_positive_integer(value: Any, label: str) -> int:
    if not _is_integer(value) or int(value) < 1:
        _fail(f"{label} must be a positive integer")
    return int(value)


def _digest_blob(digest: "hashlib._Hash", tag: str, content: bytes) -> None:
    tag_bytes = tag.encode("utf-8")
    digest.update(struct.pack(">Q", len(tag_bytes)))
    digest.update(tag_bytes)
    digest.update(struct.pack(">Q", len(content)))
    digest.update(content)


def _state_digest_update(
    digest: "hashlib._Hash",
    value: Any,
    *,
    label: str,
) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().resolve_conj().resolve_neg().cpu()
        if tensor.layout != torch.strided:
            tensor = tensor.to_dense()
        tensor = tensor.contiguous()
        metadata = _canonical_json(
            {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            f"{label} tensor metadata",
        )
        _digest_blob(digest, "tensor-metadata", metadata)
        _digest_blob(
            digest,
            "tensor-bytes",
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
        )
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        metadata = _canonical_json(
            {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
            },
            f"{label} ndarray metadata",
        )
        _digest_blob(digest, "ndarray-metadata", metadata)
        _digest_blob(digest, "ndarray-bytes", array.tobytes())
        return
    if isinstance(value, Mapping):
        _digest_blob(digest, "mapping-length", str(len(value)).encode("ascii"))
        ordered = sorted(
            value.items(),
            key=lambda item: (
                type(item[0]).__module__,
                type(item[0]).__qualname__,
                repr(item[0]),
            ),
        )
        for index, (key, item) in enumerate(ordered):
            _state_digest_update(
                digest,
                key,
                label=f"{label}.key[{index}]",
            )
            _state_digest_update(
                digest,
                item,
                label=f"{label}.value[{index}]",
            )
        return
    if isinstance(value, tuple):
        _digest_blob(digest, "tuple-length", str(len(value)).encode("ascii"))
        for index, item in enumerate(value):
            _state_digest_update(
                digest,
                item,
                label=f"{label}[{index}]",
            )
        return
    if isinstance(value, list):
        _digest_blob(digest, "list-length", str(len(value)).encode("ascii"))
        for index, item in enumerate(value):
            _state_digest_update(
                digest,
                item,
                label=f"{label}[{index}]",
            )
        return
    if isinstance(value, np.generic):
        _state_digest_update(digest, value.item(), label=label)
        return
    if value is None:
        _digest_blob(digest, "none", b"")
        return
    if isinstance(value, bool):
        _digest_blob(digest, "bool", b"1" if value else b"0")
        return
    if isinstance(value, int):
        _digest_blob(digest, "int", str(value).encode("ascii"))
        return
    if isinstance(value, float):
        _digest_blob(digest, "float64", struct.pack(">d", value))
        return
    if isinstance(value, str):
        _digest_blob(digest, "string", value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        _digest_blob(digest, "bytes", value)
        return
    if isinstance(value, (torch.dtype, torch.device)):
        _digest_blob(
            digest,
            _qualified_name(value),
            str(value).encode("utf-8"),
        )
        return
    _fail(f"{label} contains unsupported state type {_qualified_name(value)}")


def _state_content_sha256(value: Any, label: str) -> str:
    digest = hashlib.sha256()
    _digest_blob(digest, "schema", STATE_CONTENT_SCHEMA.encode("utf-8"))
    _state_digest_update(digest, value, label=label)
    return digest.hexdigest()


def initial_model_state_sha256(model: nn.Module) -> str:
    """Return a value-sensitive digest of the complete ordered model state."""

    if not isinstance(model, nn.Module):
        _fail("initial model must be an nn.Module")
    return _state_content_sha256(model.state_dict(), "initial model state")


def initial_rng_contract() -> dict[str, Any]:
    """Capture value-sensitive initial global RNG stream fingerprints."""

    cuda_available = bool(torch.cuda.is_available())
    cuda_states = torch.cuda.get_rng_state_all() if cuda_available else []
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    if len(cuda_states) != cuda_device_count:
        _fail("initial CUDA RNG state count differs from visible device count")
    return {
        "schema": INITIAL_RNG_CONTRACT_SCHEMA,
        "python_random_sha256": _state_content_sha256(
            random.getstate(),
            "initial Python RNG",
        ),
        "numpy_random_sha256": _state_content_sha256(
            np.random.get_state(),
            "initial NumPy RNG",
        ),
        "torch_cpu_sha256": _state_content_sha256(
            torch.get_rng_state(),
            "initial Torch CPU RNG",
        ),
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": cuda_device_count,
        "torch_cuda_sha256": [
            _state_content_sha256(
                state,
                f"initial Torch CUDA RNG {index}",
            )
            for index, state in enumerate(cuda_states)
        ],
    }


def _normalize_initial_rng_contract(value: Any) -> dict[str, Any]:
    contract = _plain_json(value, "initial RNG contract")
    required = {
        "schema",
        "python_random_sha256",
        "numpy_random_sha256",
        "torch_cpu_sha256",
        "torch_cuda_available",
        "torch_cuda_device_count",
        "torch_cuda_sha256",
    }
    if not isinstance(contract, dict) or set(contract) != required:
        _fail("initial RNG contract has an invalid schema")
    if contract["schema"] != INITIAL_RNG_CONTRACT_SCHEMA:
        _fail("initial RNG contract schema mismatch")
    for key in (
        "python_random_sha256",
        "numpy_random_sha256",
        "torch_cpu_sha256",
    ):
        _validate_sha256(contract[key], f"initial RNG contract {key}")
    if not isinstance(contract["torch_cuda_available"], bool):
        _fail("initial RNG CUDA availability must be boolean")
    count = contract["torch_cuda_device_count"]
    if not _is_integer(count) or int(count) < 0:
        _fail("initial RNG CUDA device count must be non-negative")
    count = int(count)
    digests = contract["torch_cuda_sha256"]
    if not isinstance(digests, list):
        _fail("initial RNG CUDA digests must be a list")
    for index, digest in enumerate(digests):
        _validate_sha256(digest, f"initial RNG CUDA digest {index}")
    if contract["torch_cuda_available"]:
        if count < 1 or len(digests) != count:
            _fail("initial RNG CUDA metadata is inconsistent")
    elif count != 0 or digests:
        _fail("initial RNG CPU-only metadata is inconsistent")
    contract["torch_cuda_device_count"] = count
    return contract


def _atomic_write_bytes(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        _fail(f"refusing to replace symlink: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_regular(path: Path, label: str) -> bytes:
    if path.is_symlink():
        _fail(f"{label} must not be a symlink")
    try:
        metadata = path.stat()
    except FileNotFoundError:
        _fail(f"{label} is missing: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a regular file: {path}")
    return path.read_bytes()


@dataclass(frozen=True)
class OrderedFingerprint:
    """A named digest whose construction preserves sequence order."""

    name: str
    count: int
    sha256: str
    schema: str = ORDERED_FINGERPRINT_SCHEMA

    def normalized(self) -> dict[str, Any]:
        _validate_nonempty_string(self.name, "fingerprint.name")
        if not _is_integer(self.count) or int(self.count) < 0:
            _fail("fingerprint.count must be a non-negative integer")
        _validate_sha256(self.sha256, "fingerprint.sha256")
        if self.schema != ORDERED_FINGERPRINT_SCHEMA:
            _fail("ordered fingerprint schema mismatch")
        return {
            "schema": self.schema,
            "name": self.name,
            "count": int(self.count),
            "sha256": self.sha256,
        }

    @classmethod
    def from_values(
        cls,
        name: str,
        values: Iterable[str],
    ) -> "OrderedFingerprint":
        ordered = list(values)
        for index, value in enumerate(ordered):
            if not isinstance(value, str):
                _fail(f"ordered fingerprint value {index} must be a string")
        digest = hashlib.sha256(
            _canonical_json(ordered, f"ordered fingerprint {name}")
        ).hexdigest()
        return cls(name=name, count=len(ordered), sha256=digest)


@dataclass(frozen=True)
class ManualCosineSchedule:
    """Identity-bound manual warmup/cosine schedule; no scheduler object exists."""

    total_epochs: int
    base_lr: float
    min_lr: float
    warmup_epochs: int
    schema: str = MANUAL_COSINE_SCHEMA

    def normalized(self) -> dict[str, Any]:
        total = _validate_positive_integer(self.total_epochs, "total_epochs")
        if not _is_integer(self.warmup_epochs):
            _fail("warmup_epochs must be an integer")
        warmup = int(self.warmup_epochs)
        if warmup < 0 or warmup > total:
            _fail("warmup_epochs must be within [0, total_epochs]")
        base = float(self.base_lr)
        minimum = float(self.min_lr)
        if (
            not math.isfinite(base)
            or not math.isfinite(minimum)
            or base <= 0.0
            or minimum < 0.0
            or minimum > base
        ):
            _fail("manual cosine learning rates must satisfy 0 <= min <= base")
        if self.schema != MANUAL_COSINE_SCHEMA:
            _fail("manual cosine schema mismatch")
        return {
            "schema": self.schema,
            "total_epochs": total,
            "base_lr": base,
            "min_lr": minimum,
            "warmup_epochs": warmup,
            "scheduler": None,
        }

    def learning_rate(self, epoch: int) -> float:
        config = self.normalized()
        current = _validate_positive_integer(epoch, "epoch")
        if current > config["total_epochs"]:
            _fail("epoch exceeds the identity-bound total_epochs")
        warmup = config["warmup_epochs"]
        base = config["base_lr"]
        minimum = config["min_lr"]
        if warmup > 0 and current <= warmup:
            return base * current / warmup
        decay_epochs = config["total_epochs"] - warmup
        if decay_epochs <= 0:
            return base
        progress = (current - warmup) / decay_epochs
        return minimum + 0.5 * (base - minimum) * (
            1.0 + math.cos(math.pi * progress)
        )


def _normalize_optimizer_contract(value: Any) -> dict[str, Any]:
    contract = _plain_json(value, "optimizer contract")
    required = {"schema", "class", "defaults", "param_groups"}
    if not isinstance(contract, dict) or set(contract) != required:
        _fail("optimizer contract has an invalid schema")
    if contract["schema"] != OPTIMIZER_CONTRACT_SCHEMA:
        _fail("optimizer contract schema mismatch")
    _validate_nonempty_string(contract["class"], "optimizer contract class")
    if not isinstance(contract["defaults"], dict):
        _fail("optimizer contract defaults must be a mapping")
    groups = contract["param_groups"]
    if not isinstance(groups, list) or not groups:
        _fail("optimizer contract param_groups must be a non-empty list")
    seen_parameter_names: set[str] = set()
    for expected_index, group in enumerate(groups):
        if (
            not isinstance(group, dict)
            or set(group)
            != {
                "index",
                "parameter_count",
                "parameter_names",
                "options",
            }
            or group["index"] != expected_index
        ):
            _fail("optimizer contract param group schema/order mismatch")
        if (
            not _is_integer(group["parameter_count"])
            or int(group["parameter_count"]) < 1
        ):
            _fail("optimizer param-group parameter_count must be positive")
        group["parameter_count"] = int(group["parameter_count"])
        names = group["parameter_names"]
        if (
            not isinstance(names, list)
            or len(names) != group["parameter_count"]
        ):
            _fail("optimizer parameter names/count mismatch")
        for position, name in enumerate(names):
            _validate_nonempty_string(
                name,
                f"optimizer group {expected_index} parameter {position}",
            )
            if name in seen_parameter_names:
                _fail(f"optimizer parameter {name!r} appears more than once")
            seen_parameter_names.add(name)
        if not isinstance(group["options"], dict):
            _fail("optimizer param-group options must be a mapping")
    return contract


def optimizer_contract(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Return optimizer defaults plus exact ordered model-parameter binding."""

    if not isinstance(model, nn.Module):
        _fail("optimizer contract model must be an nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        _fail("optimizer must be a torch.optim.Optimizer")
    name_by_parameter_id = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    if not name_by_parameter_id:
        _fail("optimizer contract model has no named parameters")
    seen_names: set[str] = set()
    groups: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = group.get("params")
        if not isinstance(parameters, (tuple, list)) or not parameters:
            _fail(f"optimizer param group {index} must contain parameters")
        parameter_names: list[str] = []
        for position, parameter in enumerate(parameters):
            name = name_by_parameter_id.get(id(parameter))
            if name is None:
                _fail(
                    f"optimizer group {index} position {position} is not "
                    "a model parameter"
                )
            if name in seen_names:
                _fail(f"optimizer parameter {name!r} appears more than once")
            seen_names.add(name)
            parameter_names.append(name)
        options = {
            key: value for key, value in group.items() if key != "params"
        }
        groups.append(
            {
                "index": index,
                "parameter_count": len(parameters),
                "parameter_names": parameter_names,
                "options": _plain_json(
                    options,
                    f"optimizer param group {index} options",
                ),
            }
        )
    missing = sorted(set(name_by_parameter_id.values()) - seen_names)
    if missing:
        _fail(f"optimizer omits model parameters: {missing}")
    contract = _normalize_optimizer_contract(
        {
            "schema": OPTIMIZER_CONTRACT_SCHEMA,
            "class": _qualified_name(optimizer),
            "defaults": _plain_json(
                optimizer.defaults,
                "optimizer defaults",
            ),
            "param_groups": groups,
        }
    )
    return contract


def _normalize_scaler_contract(value: Any) -> dict[str, Any]:
    contract = _plain_json(value, "scaler contract")
    required = {
        "schema",
        "class",
        "amp",
        "enabled",
        "config",
        "initial_state",
    }
    if not isinstance(contract, dict) or set(contract) != required:
        _fail("scaler contract has an invalid schema")
    if contract["schema"] != SCALER_CONTRACT_SCHEMA:
        _fail("scaler contract schema mismatch")
    _validate_nonempty_string(contract["class"], "scaler contract class")
    if not isinstance(contract["amp"], bool):
        _fail("scaler contract amp must be boolean")
    if not isinstance(contract["enabled"], bool):
        _fail("scaler contract enabled must be boolean")
    if contract["enabled"] != contract["amp"]:
        _fail("scaler enabled state must equal the AMP contract")
    if not isinstance(contract["config"], dict) or not contract["config"]:
        _fail("scaler contract config must be a non-empty mapping")
    if not isinstance(contract["initial_state"], dict):
        _fail("scaler contract initial_state must be a mapping")
    return contract


def scaler_contract(
    scaler: Any,
    *,
    amp: bool,
) -> dict[str, Any]:
    """Return a scaler's class, enabled state, configuration and initial state."""

    if not isinstance(amp, bool):
        _fail("scaler amp must be boolean")
    if not hasattr(scaler, "state_dict") or not callable(scaler.state_dict):
        _fail("scaler must provide state_dict()")
    if hasattr(scaler, "is_enabled") and callable(scaler.is_enabled):
        enabled = scaler.is_enabled()
    elif hasattr(scaler, "_enabled"):
        enabled = getattr(scaler, "_enabled")
    elif hasattr(scaler, "enabled"):
        enabled = getattr(scaler, "enabled")
    else:
        enabled = amp
    if not isinstance(enabled, bool):
        _fail("scaler enabled state must be boolean")
    config: dict[str, Any] = {"amp_requested": amp}
    for name in (
        "_device",
        "_init_scale",
        "_growth_factor",
        "_backoff_factor",
        "_growth_interval",
    ):
        if not hasattr(scaler, name):
            continue
        item = getattr(scaler, name)
        if isinstance(item, (torch.device, torch.dtype)):
            item = str(item)
        config[name.removeprefix("_")] = item
    return _normalize_scaler_contract(
        {
            "schema": SCALER_CONTRACT_SCHEMA,
            "class": _qualified_name(scaler),
            "amp": amp,
            "enabled": enabled,
            "config": config,
            "initial_state": _plain_json(
                scaler.state_dict(),
                "scaler initial state",
            ),
        }
    )


def fresh_initialization_contract() -> dict[str, Any]:
    """Identity record for a model initialized only by its child builder."""

    return {
        "schema": INITIALIZATION_CONTRACT_SCHEMA,
        "mode": exact.InitializationMode.FRESH.value,
    }


def same_layout_parent_initialization_contract(
    *,
    parent_checkpoint_sha256: str,
    parent_identity: Mapping[str, Any],
    parent_epoch: int,
    loaded_child_model_state_sha256: str,
) -> dict[str, Any]:
    """Identity record for strict same-layout parent weight initialization."""

    identity = exact._validate_run_identity(
        parent_identity,
        "parent_identity",
    )
    return {
        "schema": INITIALIZATION_CONTRACT_SCHEMA,
        "mode": exact.PARENT_WARM_START_MODE,
        "parent_checkpoint_sha256": _validate_sha256(
            parent_checkpoint_sha256,
            "parent checkpoint SHA-256",
        ),
        "parent_identity": identity,
        "parent_epoch": _validate_positive_integer(
            parent_epoch,
            "parent_epoch",
        ),
        "loaded_child_model_state_sha256": _validate_sha256(
            loaded_child_model_state_sha256,
            "loaded child model state SHA-256",
        ),
    }


def extension_parent_initialization_contract(
    provenance: Mapping[str, Any],
    *,
    loaded_child_model_state_sha256: str,
) -> dict[str, Any]:
    """Identity record for an externally completed strict extension load."""

    return {
        "schema": INITIALIZATION_CONTRACT_SCHEMA,
        "mode": EXTENSION_PARENT_MODE,
        "provenance": _normalize_extension_provenance(provenance),
        "loaded_child_model_state_sha256": _validate_sha256(
            loaded_child_model_state_sha256,
            "loaded extension child model state SHA-256",
        ),
    }


def _normalize_initialization_contract(value: Any) -> dict[str, Any]:
    contract = _plain_json(value, "initialization contract")
    if not isinstance(contract, dict):
        _fail("initialization contract must be a mapping")
    mode = contract.get("mode")
    if mode == exact.InitializationMode.FRESH.value:
        expected = fresh_initialization_contract()
    elif mode == exact.PARENT_WARM_START_MODE:
        if set(contract) != {
            "schema",
            "mode",
            "parent_checkpoint_sha256",
            "parent_identity",
            "parent_epoch",
            "loaded_child_model_state_sha256",
        }:
            _fail("same-layout parent initialization contract schema mismatch")
        expected = same_layout_parent_initialization_contract(
            parent_checkpoint_sha256=contract["parent_checkpoint_sha256"],
            parent_identity=contract["parent_identity"],
            parent_epoch=contract["parent_epoch"],
            loaded_child_model_state_sha256=contract[
                "loaded_child_model_state_sha256"
            ],
        )
    elif mode == EXTENSION_PARENT_MODE:
        if set(contract) != {
            "schema",
            "mode",
            "provenance",
            "loaded_child_model_state_sha256",
        }:
            _fail("extension parent initialization contract schema mismatch")
        expected = extension_parent_initialization_contract(
            contract["provenance"],
            loaded_child_model_state_sha256=contract[
                "loaded_child_model_state_sha256"
            ],
        )
    else:
        _fail(f"unsupported initialization contract mode: {mode!r}")
    if contract != expected:
        _fail("initialization contract is not canonical")
    return expected


def _normalize_fingerprints(
    values: Mapping[str, OrderedFingerprint],
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, Mapping) or not values:
        _fail(f"{label} must be a non-empty mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for role, fingerprint in values.items():
        _validate_nonempty_string(role, f"{label} role")
        if not isinstance(fingerprint, OrderedFingerprint):
            _fail(f"{label}.{role} must be an OrderedFingerprint")
        record = fingerprint.normalized()
        if record["name"] != role:
            _fail(f"{label}.{role} name differs from its mapping role")
        normalized[role] = record
    return dict(sorted(normalized.items()))


@dataclass(frozen=True)
class ExactRunSpec:
    """Immutable experiment fields that define one exact training trajectory."""

    run_id: str
    variant: str
    dataset: str
    seed: int
    split_seed: int
    builder_metadata: Mapping[str, Any]
    builder_manifest_sha256: str
    source_locks: Mapping[str, str]
    split_fingerprints: Mapping[str, OrderedFingerprint]
    data_fingerprints: Mapping[str, OrderedFingerprint]
    optimizer: Mapping[str, Any]
    scaler: Mapping[str, Any]
    initialization_contract: Mapping[str, Any]
    lr_schedule: ManualCosineSchedule
    loss: Mapping[str, Any]
    deep_supervision: Mapping[str, Any]
    batch_size: int
    patch_size: int
    workers: int
    amp: bool
    total_epochs: int
    eval_interval: int
    metric_config: Mapping[str, Any]
    environment: Mapping[str, Any]
    determinism: Mapping[str, Any]
    initial_model_state_sha256: str | None = None
    initial_rng: Mapping[str, Any] | None = None
    selection_policy: Mapping[str, Any] | None = None

    def normalized(self) -> dict[str, Any]:
        run_id = _validate_nonempty_string(self.run_id, "run_id")
        variant = _validate_nonempty_string(self.variant, "variant")
        dataset = _validate_nonempty_string(self.dataset, "dataset")
        if not _is_integer(self.seed) or not _is_integer(self.split_seed):
            _fail("seed and split_seed must be integers")
        builder = _plain_json(self.builder_metadata, "builder_metadata")
        optimizer = _normalize_optimizer_contract(self.optimizer)
        scaler = _normalize_scaler_contract(self.scaler)
        initialization = _normalize_initialization_contract(
            self.initialization_contract
        )
        loss = _plain_json(self.loss, "loss")
        deep = _plain_json(self.deep_supervision, "deep_supervision")
        metric = _plain_json(self.metric_config, "metric_config")
        environment = _plain_json(self.environment, "environment")
        determinism = _plain_json(self.determinism, "determinism")
        if self.initial_model_state_sha256 is None:
            _fail("initial_model_state_sha256 is required")
        model_state_sha256 = _validate_sha256(
            self.initial_model_state_sha256,
            "initial_model_state_sha256",
        )
        if self.initial_rng is None:
            _fail("initial_rng is required")
        initial_rng = _normalize_initial_rng_contract(self.initial_rng)
        if self.selection_policy is None:
            _fail("selection_policy is required")
        selection_policy = _normalize_selection_policy_contract(
            self.selection_policy
        )
        for label, value in (
            ("builder_metadata", builder),
            ("optimizer", optimizer),
            ("scaler", scaler),
            ("initialization_contract", initialization),
            ("loss", loss),
            ("deep_supervision", deep),
            ("metric_config", metric),
            ("environment", environment),
            ("determinism", determinism),
        ):
            if not isinstance(value, dict) or not value:
                _fail(f"{label} must be a non-empty mapping")
        _validate_sha256(
            self.builder_manifest_sha256,
            "builder_manifest_sha256",
        )
        if not isinstance(self.source_locks, Mapping) or not self.source_locks:
            _fail("source_locks must be a non-empty mapping")
        source_locks: dict[str, str] = {}
        for name, digest in self.source_locks.items():
            _validate_nonempty_string(name, "source lock name")
            source_locks[name] = _validate_sha256(
                digest,
                f"source_locks.{name}",
            )
        schedule = self.lr_schedule.normalized()
        total_epochs = _validate_positive_integer(
            self.total_epochs,
            "total_epochs",
        )
        if total_epochs != schedule["total_epochs"]:
            _fail("run total_epochs differs from manual LR total_epochs")
        initialization_mode = initialization["mode"]
        if initialization_mode in (
            exact.PARENT_WARM_START_MODE,
            EXTENSION_PARENT_MODE,
        ) and initialization["loaded_child_model_state_sha256"] != (
            model_state_sha256
        ):
            _fail(
                "initial model state differs from the loaded-child "
                "initialization contract"
            )
        if not _is_integer(self.workers) or int(self.workers) != 0:
            _fail("exact runner requires workers=0")
        if not isinstance(self.amp, bool):
            _fail("amp must be boolean")
        return {
            "run_id": run_id,
            "variant": variant,
            "dataset": dataset,
            "seed": int(self.seed),
            "split_seed": int(self.split_seed),
            "builder_metadata": builder,
            "builder_manifest_sha256": self.builder_manifest_sha256,
            "source_locks": dict(sorted(source_locks.items())),
            "ordered_split_fingerprints": _normalize_fingerprints(
                self.split_fingerprints,
                "split_fingerprints",
            ),
            "ordered_data_fingerprints": _normalize_fingerprints(
                self.data_fingerprints,
                "data_fingerprints",
            ),
            "optimizer": optimizer,
            "scaler": scaler,
            "initialization_contract": initialization,
            "manual_lr_schedule": schedule,
            "loss": loss,
            "deep_supervision": deep,
            "batch_size": _validate_positive_integer(
                self.batch_size,
                "batch_size",
            ),
            "patch_size": _validate_positive_integer(
                self.patch_size,
                "patch_size",
            ),
            "workers": 0,
            "amp": self.amp,
            "total_epochs": total_epochs,
            "eval_interval": _validate_positive_integer(
                self.eval_interval,
                "eval_interval",
            ),
            "metric_config": metric,
            "environment": environment,
            "determinism": determinism,
            "initial_model_state_sha256": model_state_sha256,
            "initial_rng": initial_rng,
            "selection_policy": selection_policy,
        }


def compute_architecture_id(
    model: nn.Module,
    spec: ExactRunSpec,
) -> str:
    """Compute a value-independent architecture ID bound to builder sources."""

    normalized = spec.normalized()
    architecture = {
        "schema": ARCHITECTURE_SCHEMA,
        "model_layout": exact.model_layout(model),
        "builder_metadata": normalized["builder_metadata"],
        "builder_manifest_sha256": normalized["builder_manifest_sha256"],
        "source_locks": normalized["source_locks"],
    }
    return _sha256_json(architecture, "architecture identity")


def build_run_identity(
    model: nn.Module,
    spec: ExactRunSpec,
) -> dict[str, Any]:
    """Normalize every trajectory-defining field into one strict identity."""

    normalized = spec.normalized()
    architecture_id = compute_architecture_id(model, spec)
    split_sha256 = _sha256_json(
        normalized["ordered_split_fingerprints"],
        "ordered split fingerprints",
    )
    data_sha256 = _sha256_json(
        normalized["ordered_data_fingerprints"],
        "ordered data fingerprints",
    )
    training_contract = {
        key: normalized[key]
        for key in (
            "optimizer",
            "scaler",
            "initialization_contract",
            "manual_lr_schedule",
            "loss",
            "deep_supervision",
            "batch_size",
            "patch_size",
            "workers",
            "amp",
            "total_epochs",
            "eval_interval",
            "metric_config",
            "environment",
            "determinism",
            "initial_model_state_sha256",
            "initial_rng",
            "selection_policy",
        )
    }
    contract = {
        "schema": RUN_IDENTITY_SCHEMA,
        "architecture_id": architecture_id,
        "builder_manifest_sha256": normalized["builder_manifest_sha256"],
        "source_locks": normalized["source_locks"],
        "ordered_split_fingerprints": normalized[
            "ordered_split_fingerprints"
        ],
        "ordered_data_fingerprints": normalized[
            "ordered_data_fingerprints"
        ],
        "data_sha256": data_sha256,
        "training": training_contract,
    }
    return {
        "run_id": normalized["run_id"],
        "variant": normalized["variant"],
        "architecture_id": architecture_id,
        "dataset": normalized["dataset"],
        "seed": normalized["seed"],
        "split_seed": normalized["split_seed"],
        "split_sha256": split_sha256,
        "schema": RUN_IDENTITY_SCHEMA,
        "builder_manifest_sha256": normalized["builder_manifest_sha256"],
        "source_locks": normalized["source_locks"],
        "ordered_split_fingerprints": normalized[
            "ordered_split_fingerprints"
        ],
        "ordered_data_fingerprints": normalized[
            "ordered_data_fingerprints"
        ],
        "data_sha256": data_sha256,
        "training_contract": training_contract,
        "contract_sha256": _sha256_json(contract, "run identity contract"),
    }


@dataclass(frozen=True)
class MetricOrder:
    """One lexicographic best-checkpoint metric."""

    name: str
    maximize: bool = True

    def normalized(self) -> dict[str, Any]:
        if not isinstance(self.maximize, bool):
            _fail("metric order maximize must be boolean")
        return {
            "name": _validate_nonempty_string(self.name, "metric order name"),
            "maximize": self.maximize,
        }


@dataclass(frozen=True)
class SelectionRule:
    """Serializable lexicographic rule for one compatibility checkpoint."""

    role: str
    order: tuple[MetricOrder, ...]
    stored_metrics: tuple[str, ...]
    new_best_field: str

    def __post_init__(self) -> None:
        _validate_nonempty_string(self.role, "selection role")
        _validate_nonempty_string(self.new_best_field, "new-best field")
        if not self.order:
            _fail("selection order must not be empty")
        if not self.stored_metrics:
            _fail("stored_metrics must not be empty")
        names = [item.normalized()["name"] for item in self.order]
        if len(names) != len(set(names)):
            _fail("selection order contains duplicate metrics")
        for name in self.stored_metrics:
            _validate_nonempty_string(name, "stored metric")
        if len(self.stored_metrics) != len(set(self.stored_metrics)):
            _fail("stored_metrics contains duplicates")
        missing = sorted(set(names) - set(self.stored_metrics))
        if missing:
            _fail(f"selection order metrics are not stored: {missing}")

    def normalized(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "order": [item.normalized() for item in self.order],
            "stored_metrics": list(self.stored_metrics),
            "new_best_field": self.new_best_field,
        }

    def eligible(self, event: Mapping[str, Any]) -> bool:
        present = [name in event for name in self.stored_metrics]
        if any(present) and not all(present):
            _fail(f"event has a partial metric record for role {self.role}")
        return all(present)

    def key(self, event: Mapping[str, Any]) -> list[float]:
        result: list[float] = []
        for order in self.order:
            value = event[order.name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.number))
                or not math.isfinite(float(value))
            ):
                _fail(f"selection metric {order.name} must be finite numeric")
            number = float(value)
            result.append(number if order.maximize else -number)
        return result

    def metrics(self, event: Mapping[str, Any]) -> dict[str, int | float]:
        result: dict[str, int | float] = {}
        for name in self.stored_metrics:
            value = event[name]
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float, np.number),
            ):
                _fail(f"stored selection metric {name} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                _fail(f"stored selection metric {name} must be finite")
            result[name] = int(value) if _is_integer(value) else number
        return result


@dataclass(frozen=True)
class SelectionPolicy:
    """Primary and secondary rules used for all selection reconstruction."""

    primary: SelectionRule
    secondary: SelectionRule

    def __post_init__(self) -> None:
        if self.primary.new_best_field == self.secondary.new_best_field:
            _fail("primary and secondary new-best fields must differ")

    def normalized(self) -> dict[str, Any]:
        return {
            "schema": SELECTION_POLICY_SCHEMA,
            "primary": self.primary.normalized(),
            "secondary": self.secondary.normalized(),
        }

    def annotate_event(
        self,
        history: Sequence[Mapping[str, Any]],
        event: Mapping[str, Any],
    ) -> dict[str, Any]:
        annotated = copy.deepcopy(dict(event))
        previous = self.recompute(history, require_flags=True) if history else None
        for slot, rule in (
            ("primary", self.primary),
            ("secondary", self.secondary),
        ):
            eligible = rule.eligible(annotated)
            if not eligible:
                annotated[rule.new_best_field] = False
                continue
            key = rule.key(annotated)
            previous_key = None if previous is None else previous[slot]["key"]
            annotated[rule.new_best_field] = (
                previous_key is None or key > list(previous_key)
            )
        return annotated

    def recompute(
        self,
        events: Sequence[Mapping[str, Any]],
        *,
        require_flags: bool = True,
    ) -> dict[str, Any]:
        if not events:
            _fail("best selection requires at least one event")
        records: dict[str, dict[str, Any] | None] = {
            "primary": None,
            "secondary": None,
        }
        for expected_epoch, event in enumerate(events, start=1):
            if event.get("epoch") != expected_epoch:
                _fail("selection history epochs are not contiguous")
            for slot, rule in (
                ("primary", self.primary),
                ("secondary", self.secondary),
            ):
                eligible = rule.eligible(event)
                previous = records[slot]
                key = rule.key(event) if eligible else None
                new_best = bool(
                    eligible
                    and (
                        previous is None
                        or key is not None
                        and key > previous["key"]
                    )
                )
                if require_flags:
                    flag = event.get(rule.new_best_field)
                    if not isinstance(flag, bool) or flag is not new_best:
                        _fail(
                            f"incorrect {rule.new_best_field} at epoch "
                            f"{expected_epoch}"
                        )
                if new_best:
                    assert key is not None
                    records[slot] = {
                        "role": rule.role,
                        "epoch": expected_epoch,
                        "key": key,
                        "metrics": rule.metrics(event),
                    }
        if records["primary"] is None or records["secondary"] is None:
            _fail("selection history contains no eligible metric event")
        return {
            "primary": copy.deepcopy(records["primary"]),
            "secondary": copy.deepcopy(records["secondary"]),
        }


def _normalize_selection_policy_contract(value: Any) -> dict[str, Any]:
    contract = _plain_json(value, "selection policy contract")
    if (
        not isinstance(contract, dict)
        or set(contract) != {"schema", "primary", "secondary"}
        or contract["schema"] != SELECTION_POLICY_SCHEMA
    ):
        _fail("selection policy contract has an invalid schema")

    def normalize_rule(rule: Any, label: str) -> dict[str, Any]:
        if not isinstance(rule, dict) or set(rule) != {
            "role",
            "order",
            "stored_metrics",
            "new_best_field",
        }:
            _fail(f"{label} has an invalid schema")
        role = _validate_nonempty_string(rule["role"], f"{label}.role")
        new_best = _validate_nonempty_string(
            rule["new_best_field"],
            f"{label}.new_best_field",
        )
        order = rule["order"]
        stored = rule["stored_metrics"]
        if not isinstance(order, list) or not order:
            _fail(f"{label}.order must be a non-empty list")
        normalized_order: list[dict[str, Any]] = []
        seen_order: set[str] = set()
        for index, item in enumerate(order):
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "maximize"}
                or not isinstance(item["maximize"], bool)
            ):
                _fail(f"{label}.order[{index}] has an invalid schema")
            name = _validate_nonempty_string(
                item["name"],
                f"{label}.order[{index}].name",
            )
            if name in seen_order:
                _fail(f"{label}.order contains duplicate metrics")
            seen_order.add(name)
            normalized_order.append(
                {"name": name, "maximize": item["maximize"]}
            )
        if not isinstance(stored, list) or not stored:
            _fail(f"{label}.stored_metrics must be a non-empty list")
        normalized_stored = [
            _validate_nonempty_string(
                name,
                f"{label}.stored_metrics[{index}]",
            )
            for index, name in enumerate(stored)
        ]
        if len(set(normalized_stored)) != len(normalized_stored):
            _fail(f"{label}.stored_metrics contains duplicates")
        if not seen_order.issubset(normalized_stored):
            _fail(f"{label}.order contains unstored metrics")
        return {
            "role": role,
            "order": normalized_order,
            "stored_metrics": normalized_stored,
            "new_best_field": new_best,
        }

    normalized = {
        "schema": SELECTION_POLICY_SCHEMA,
        "primary": normalize_rule(contract["primary"], "selection primary"),
        "secondary": normalize_rule(
            contract["secondary"],
            "selection secondary",
        ),
    }
    if (
        normalized["primary"]["new_best_field"]
        == normalized["secondary"]["new_best_field"]
    ):
        _fail("selection new-best fields must differ")
    return normalized


def pd_miou_selection_policy(
    stored_metrics: Sequence[str] = (
        "pd",
        "fa",
        "tiny_pd",
        "miou",
        "val_loss",
    ),
) -> SelectionPolicy:
    """Return the fixed SCTransNet Pd-primary and mIoU-secondary policy."""

    metrics = tuple(stored_metrics)
    return SelectionPolicy(
        primary=SelectionRule(
            role="best_validation_pd_primary",
            order=(
                MetricOrder("pd", True),
                MetricOrder("fa", False),
                MetricOrder("tiny_pd", True),
                MetricOrder("miou", True),
                MetricOrder("val_loss", False),
            ),
            stored_metrics=metrics,
            new_best_field="new_best_pd",
        ),
        secondary=SelectionRule(
            role="best_validation_miou_secondary",
            order=(
                MetricOrder("miou", True),
                MetricOrder("pd", True),
                MetricOrder("fa", False),
                MetricOrder("tiny_pd", True),
                MetricOrder("val_loss", False),
            ),
            stored_metrics=metrics,
            new_best_field="new_best_miou",
        ),
    )


def _normalize_extension_provenance(value: Any) -> dict[str, Any]:
    provenance = _plain_json(value, "extension_parent_provenance")
    required = {
        "schema",
        "parent_checkpoint_path",
        "parent_checkpoint_sha256",
        "parent_state_dict_path",
        "parent_state_key_count",
        "preserved_new_state_key_count",
        "new_module_prefixes",
        "zero_init_prefixes",
    }
    if not isinstance(provenance, dict) or set(provenance) != required:
        _fail("extension parent provenance has an invalid schema")
    if provenance["schema"] != extension_warm_start.PROVENANCE_SCHEMA:
        _fail("extension parent provenance schema mismatch")
    _validate_nonempty_string(
        provenance["parent_checkpoint_path"],
        "extension parent checkpoint path",
    )
    _validate_sha256(
        provenance["parent_checkpoint_sha256"],
        "extension parent checkpoint digest",
    )
    for key in ("parent_state_key_count", "preserved_new_state_key_count"):
        if not _is_integer(provenance[key]) or int(provenance[key]) < 0:
            _fail(f"extension parent provenance {key} must be non-negative")
        provenance[key] = int(provenance[key])
    for key in (
        "parent_state_dict_path",
        "new_module_prefixes",
        "zero_init_prefixes",
    ):
        values = provenance[key]
        if not isinstance(values, list):
            _fail(f"extension parent provenance {key} must be a list")
        for index, item in enumerate(values):
            _validate_nonempty_string(
                item,
                f"extension parent provenance {key}[{index}]",
            )
    if not provenance["parent_state_dict_path"]:
        _fail("extension parent state-dict path must not be empty")
    if not provenance["new_module_prefixes"]:
        _fail("extension parent new-module prefixes must not be empty")
    return provenance


def _model_layout_sha256(model: nn.Module) -> str:
    if not isinstance(model, nn.Module):
        _fail("child model must be an nn.Module")
    return _state_content_sha256(
        exact.model_layout(model),
        "child model layout",
    )


def _load_checkpoint_bytes(
    content: bytes,
    *,
    label: str,
    map_location: str | torch.device,
) -> dict[str, Any]:
    if not isinstance(content, bytes) or not content:
        _fail(f"{label} bytes must be non-empty")
    try:
        payload = torch.load(
            io.BytesIO(content),
            map_location=map_location,
            weights_only=False,
        )
    except Exception as exc:
        _fail(f"cannot load {label}: {exc}")
    if not isinstance(payload, Mapping):
        _fail(f"{label} must contain a mapping")
    return dict(payload)


@dataclass(frozen=True)
class PreparedSameLayoutParent:
    """Immutable parent bytes and the exact child state they produce."""

    source_path: str
    checkpoint_bytes: bytes
    checkpoint_sha256: str
    parent_identity: Mapping[str, Any]
    parent_epoch: int
    loaded_child_model_state_sha256: str
    child_model_layout_sha256: str

    def __post_init__(self) -> None:
        source_path = _validate_nonempty_string(
            self.source_path,
            "prepared parent source path",
        )
        if not isinstance(self.checkpoint_bytes, bytes) or not (
            self.checkpoint_bytes
        ):
            _fail("prepared parent checkpoint bytes must be non-empty bytes")
        checkpoint_sha256 = _validate_sha256(
            self.checkpoint_sha256,
            "prepared parent checkpoint SHA-256",
        )
        if hashlib.sha256(self.checkpoint_bytes).hexdigest() != (
            checkpoint_sha256
        ):
            _fail("prepared parent bytes differ from their SHA-256")
        parent_identity = exact._validate_run_identity(
            self.parent_identity,
            "prepared parent identity",
        )
        parent_epoch = _validate_positive_integer(
            self.parent_epoch,
            "prepared parent epoch",
        )
        loaded_child_sha256 = _validate_sha256(
            self.loaded_child_model_state_sha256,
            "prepared loaded child model state SHA-256",
        )
        layout_sha256 = _validate_sha256(
            self.child_model_layout_sha256,
            "prepared child model layout SHA-256",
        )
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(
            self,
            "parent_identity",
            copy.deepcopy(parent_identity),
        )
        object.__setattr__(self, "parent_epoch", parent_epoch)
        object.__setattr__(
            self,
            "loaded_child_model_state_sha256",
            loaded_child_sha256,
        )
        object.__setattr__(
            self,
            "child_model_layout_sha256",
            layout_sha256,
        )

    def initialization_contract(self) -> dict[str, Any]:
        return same_layout_parent_initialization_contract(
            parent_checkpoint_sha256=self.checkpoint_sha256,
            parent_identity=self.parent_identity,
            parent_epoch=self.parent_epoch,
            loaded_child_model_state_sha256=(
                self.loaded_child_model_state_sha256
            ),
        )


def prepare_same_layout_parent(
    checkpoint: str | os.PathLike[str],
    *,
    child_model: nn.Module,
    expected_parent_checkpoint_sha256: str,
    expected_parent_identity: Mapping[str, Any],
    expected_parent_epoch: int,
    map_location: str | torch.device = "cpu",
) -> PreparedSameLayoutParent:
    """Read one parent file once and bind preview plus startup to those bytes."""

    if not isinstance(checkpoint, (str, os.PathLike)):
        _fail("same-layout parent checkpoint must be a file path")
    path = Path(checkpoint).absolute()
    expected_sha256 = _validate_sha256(
        expected_parent_checkpoint_sha256,
        "expected parent checkpoint SHA-256",
    )
    expected_identity = exact._validate_run_identity(
        expected_parent_identity,
        "expected parent identity",
    )
    expected_epoch = _validate_positive_integer(
        expected_parent_epoch,
        "expected parent epoch",
    )
    rng_before = initial_rng_contract()
    content = _read_regular(path, "same-layout parent checkpoint")
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        _fail("same-layout parent checkpoint SHA-256 mismatch")
    payload = _load_checkpoint_bytes(
        content,
        label="same-layout parent checkpoint",
        map_location=map_location,
    )
    try:
        preview_model = copy.deepcopy(child_model)
    except BaseException as exc:
        _fail(f"cannot copy child model for parent preview: {exc}")
    result = exact.restore_parent_warm_start(
        payload,
        parent_model=preview_model,
        expected_parent_identity=expected_identity,
        expected_parent_epoch=expected_epoch,
        map_location=map_location,
    )
    loaded_child_sha256 = initial_model_state_sha256(preview_model)
    del preview_model
    if initial_rng_contract() != rng_before:
        _fail("same-layout parent preparation changed global RNG state")
    return PreparedSameLayoutParent(
        source_path=str(path),
        checkpoint_bytes=content,
        checkpoint_sha256=actual_sha256,
        parent_identity=result.parent_identity,
        parent_epoch=result.parent_epoch,
        loaded_child_model_state_sha256=loaded_child_sha256,
        child_model_layout_sha256=_model_layout_sha256(child_model),
    )


@dataclass(frozen=True)
class InitializationRequest:
    """One and only one startup intent."""

    mode: exact.InitializationMode
    prepared_same_layout_parent: PreparedSameLayoutParent | None = None
    extension_parent_provenance: Mapping[str, Any] | None = None
    loaded_child_model_state_sha256: str | None = None

    @classmethod
    def fresh(cls) -> "InitializationRequest":
        return cls(mode=exact.InitializationMode.FRESH)

    @classmethod
    def exact(cls) -> "InitializationRequest":
        return cls(mode=exact.InitializationMode.EXACT_RESUME)

    @classmethod
    def parent(
        cls,
        prepared: PreparedSameLayoutParent,
    ) -> "InitializationRequest":
        if not isinstance(prepared, PreparedSameLayoutParent):
            _fail(
                "same-layout parent request requires "
                "PreparedSameLayoutParent"
            )
        return cls(
            mode=exact.InitializationMode.PARENT_WARM_START,
            prepared_same_layout_parent=prepared,
            loaded_child_model_state_sha256=(
                prepared.loaded_child_model_state_sha256
            ),
        )

    @classmethod
    def extension_parent(
        cls,
        provenance: Mapping[str, Any],
        *,
        loaded_child_model_state_sha256: str,
    ) -> "InitializationRequest":
        """Record a strict extension load already completed by its loader."""

        return cls(
            mode=exact.InitializationMode.PARENT_WARM_START,
            extension_parent_provenance=provenance,
            loaded_child_model_state_sha256=loaded_child_model_state_sha256,
        )

    def validate(self) -> None:
        try:
            mode = exact.InitializationMode(self.mode)
        except (TypeError, ValueError):
            _fail(f"unsupported initialization mode: {self.mode!r}")
        has_parent_fields = any(
            value is not None
            for value in (
                self.prepared_same_layout_parent,
            )
        )
        has_extension_parent = self.extension_parent_provenance is not None
        has_loaded_child_sha = self.loaded_child_model_state_sha256 is not None
        if mode is exact.InitializationMode.PARENT_WARM_START:
            if has_parent_fields and has_extension_parent:
                _fail(
                    "same-layout and extension parent modes are mutually "
                    "exclusive"
                )
            if not has_parent_fields and not has_extension_parent:
                _fail("parent warm start requires one parent source")
            if not has_loaded_child_sha:
                _fail(
                    "parent warm start requires "
                    "loaded_child_model_state_sha256"
                )
            _validate_sha256(
                self.loaded_child_model_state_sha256,
                "loaded child model state SHA-256",
            )
            if has_parent_fields:
                if not isinstance(
                    self.prepared_same_layout_parent,
                    PreparedSameLayoutParent,
                ):
                    _fail(
                        "same-layout parent request requires prepared bytes"
                    )
                if self.loaded_child_model_state_sha256 != (
                    self.prepared_same_layout_parent
                    .loaded_child_model_state_sha256
                ):
                    _fail(
                        "prepared parent loaded-child digest differs from "
                        "the request"
                    )
            else:
                _normalize_extension_provenance(
                    self.extension_parent_provenance
                )
        elif has_parent_fields or has_extension_parent or has_loaded_child_sha:
            _fail("fresh/exact initialization must not contain parent fields")

    def initialization_contract(self) -> dict[str, Any] | None:
        """Return this request's child-initialization identity, if applicable."""

        self.validate()
        mode = exact.InitializationMode(self.mode)
        if mode is exact.InitializationMode.EXACT_RESUME:
            return None
        if mode is exact.InitializationMode.FRESH:
            return fresh_initialization_contract()
        if self.extension_parent_provenance is not None:
            return extension_parent_initialization_contract(
                self.extension_parent_provenance,
                loaded_child_model_state_sha256=(
                    self.loaded_child_model_state_sha256
                ),
            )
        assert self.prepared_same_layout_parent is not None
        return self.prepared_same_layout_parent.initialization_contract()


@dataclass(frozen=True)
class EpochControl:
    """Deterministic control values for the next epoch."""

    epoch: int
    learning_rate: float
    should_evaluate: bool


@dataclass(frozen=True)
class RunnerSnapshot:
    """Public runner state after startup or a durable epoch commit."""

    initialization_mode: exact.InitializationMode
    completed_epoch: int
    next_epoch: int | None
    run_identity: dict[str, Any]
    metrics_boundary: dict[str, Any] | None
    best_selection: dict[str, Any] | None
    parent_provenance: dict[str, Any] | None
    active: ActiveEpochState | None
    derived_artifacts_dirty: bool


@dataclass(frozen=True)
class CompatibilityPayloadContext:
    """Post-commit material supplied to a caller-specific payload adapter."""

    role: str
    epoch: int
    metrics: dict[str, Any]
    event: dict[str, Any]
    exact_payload: dict[str, Any]
    run_identity: dict[str, Any]
    normalized_spec: dict[str, Any]


CompatibilityPayloadFactory = Callable[
    [CompatibilityPayloadContext],
    Mapping[str, Any],
]


@dataclass(frozen=True)
class _PendingCommit:
    prepared: PreparedExactEpoch
    event: dict[str, Any]
    events: list[dict[str, Any]]
    best_selection: dict[str, Any]


def _state_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _state_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _state_values_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


class ExactRunner:
    """Exact startup, epoch commit and compatibility-artifact coordinator."""

    def __init__(
        self,
        run_directory: str | os.PathLike[str],
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        loader_generator: torch.Generator,
        spec: ExactRunSpec,
        selection_policy: SelectionPolicy | None = None,
        compatibility_payload_factory: CompatibilityPayloadFactory | None = None,
        map_location: str | torch.device = "cpu",
    ) -> None:
        self.run_directory = Path(run_directory).absolute()
        self.run_directory.mkdir(parents=True, exist_ok=True)
        if self.run_directory.is_symlink():
            _fail("run_directory must be a real directory")
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.loader_generator = loader_generator
        self.spec = spec
        self.normalized_spec = spec.normalized()
        if (
            not isinstance(loader_generator, torch.Generator)
            or loader_generator.device.type != "cpu"
        ):
            _fail("loader_generator must be an explicit CPU torch.Generator")
        expected_loader_generator = torch.Generator(device="cpu")
        expected_loader_generator.manual_seed(self.normalized_spec["seed"])
        if not torch.equal(
            loader_generator.get_state(),
            expected_loader_generator.get_state(),
        ):
            _fail(
                "loader_generator initial state must equal manual_seed(spec.seed)"
            )
        optimizer_state = optimizer.state_dict().get("state")
        if not isinstance(optimizer_state, Mapping) or optimizer_state:
            _fail(
                "runner construction requires a newly built optimizer with "
                "empty step state"
            )
        actual_optimizer = optimizer_contract(model, optimizer)
        if self.normalized_spec["optimizer"] != actual_optimizer:
            _fail("optimizer contract differs from the actual optimizer")
        base_lr = self.normalized_spec["manual_lr_schedule"]["base_lr"]
        if any(
            group["options"].get("lr") != base_lr
            for group in actual_optimizer["param_groups"]
        ):
            _fail(
                "optimizer param-group LR must equal manual schedule base_lr "
                "at runner construction"
            )
        actual_scaler = scaler_contract(
            scaler,
            amp=self.normalized_spec["amp"],
        )
        if self.normalized_spec["scaler"] != actual_scaler:
            _fail("scaler contract differs from the actual scaler")
        self.selection_policy = selection_policy or pd_miou_selection_policy()
        if (
            self.normalized_spec["selection_policy"]
            != self.selection_policy.normalized()
        ):
            _fail(
                "selection policy differs from the run identity contract"
            )
        self.compatibility_payload_factory = compatibility_payload_factory
        self.run_identity = build_run_identity(model, spec)
        self.journal = ExactEpochJournal(
            self.run_directory / JOURNAL_DIRECTORY
        )
        self.runtime = ExactTrainingRuntime(
            self.journal,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loader_generator=loader_generator,
            scheduler=None,
            map_location=map_location,
        )
        self._initialization_mode: exact.InitializationMode | None = None
        self._parent_provenance: dict[str, Any] | None = None
        self._events: list[dict[str, Any]] = []
        self._best_selection: dict[str, Any] | None = None
        self._open_control: EpochControl | None = None
        self._pending: _PendingCommit | None = None
        self._derived_dirty = False

    @property
    def started(self) -> bool:
        return self._initialization_mode is not None

    @property
    def snapshot(self) -> RunnerSnapshot:
        if not self.started:
            _fail("runner has not been started")
        runtime_snapshot = self.runtime.snapshot
        next_epoch = runtime_snapshot.completed_epoch + 1
        if next_epoch > self.normalized_spec["total_epochs"]:
            next_epoch = None
        return RunnerSnapshot(
            initialization_mode=self._initialization_mode,
            completed_epoch=runtime_snapshot.completed_epoch,
            next_epoch=next_epoch,
            run_identity=copy.deepcopy(self.run_identity),
            metrics_boundary=copy.deepcopy(runtime_snapshot.metrics_boundary),
            best_selection=copy.deepcopy(self._best_selection),
            parent_provenance=copy.deepcopy(self._parent_provenance),
            active=runtime_snapshot.active,
            derived_artifacts_dirty=self._derived_dirty,
        )

    def _read_events(self, active: ActiveEpochState) -> list[dict[str, Any]]:
        content = _read_regular(active.metrics_path, "active journal metrics")
        if hashlib.sha256(content).hexdigest() != active.metrics_boundary[
            "metrics_sha256"
        ]:
            _fail("active journal metrics digest mismatch")
        events: list[dict[str, Any]] = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            try:
                event = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                _fail(f"cannot parse active event {line_number}: {exc}")
            if not isinstance(event, dict) or event.get("epoch") != line_number:
                _fail("active journal contains a non-contiguous event")
            events.append(event)
        if len(events) != active.epoch:
            _fail("active journal event count differs from its epoch")
        return events

    def _require_no_derived_trajectory(self) -> None:
        existing = [
            name
            for name in (
                METRICS_FILENAME,
                LAST_FILENAME,
                BEST_FILENAME,
                BEST_MIOU_FILENAME,
            )
            if (self.run_directory / name).exists()
            or (self.run_directory / name).is_symlink()
        ]
        if existing:
            _fail(
                "empty journal has existing derived trajectory artifacts: "
                f"{existing}"
            )

    def _require_initial_rng_contract(self) -> None:
        actual = initial_rng_contract()
        expected = self.normalized_spec["initial_rng"]
        if actual != expected:
            _fail("initial global RNG state differs from the run identity")

    def _require_initial_model_state(self) -> None:
        actual = initial_model_state_sha256(self.model)
        expected = self.normalized_spec["initial_model_state_sha256"]
        if actual != expected:
            _fail("initial model state differs from the run identity")

    def _restore_same_layout_parent_once(
        self,
        request: InitializationRequest,
    ) -> exact.ParentWarmStartResult:
        assert request.prepared_same_layout_parent is not None
        assert request.loaded_child_model_state_sha256 is not None
        prepared = request.prepared_same_layout_parent
        if _model_layout_sha256(self.model) != (
            prepared.child_model_layout_sha256
        ):
            _fail("prepared parent child model layout differs")
        content = prepared.checkpoint_bytes
        actual_digest = hashlib.sha256(content).hexdigest()
        if actual_digest != prepared.checkpoint_sha256:
            _fail("prepared parent checkpoint bytes changed")
        payload = _load_checkpoint_bytes(
            content,
            label="prepared same-layout parent checkpoint",
            map_location=self.runtime.map_location,
        )

        previous_model = copy.deepcopy(self.model.state_dict())
        previous_rng = exact.capture_rng_state(self.loader_generator)
        try:
            result = exact.restore_parent_warm_start(
                payload,
                parent_model=self.model,
                expected_parent_identity=prepared.parent_identity,
                expected_parent_epoch=prepared.parent_epoch,
                map_location=self.runtime.map_location,
            )
            loaded_digest = initial_model_state_sha256(self.model)
            if loaded_digest != request.loaded_child_model_state_sha256:
                _fail(
                    "same-layout parent loaded child model state SHA-256 "
                    "mismatch"
                )
            if loaded_digest != self.normalized_spec[
                "initial_model_state_sha256"
            ]:
                _fail(
                    "same-layout parent result differs from the run identity"
                )
            if initial_rng_contract() != self.normalized_spec["initial_rng"]:
                _fail("same-layout parent restore changed global RNG state")
        except BaseException:
            self.model.load_state_dict(previous_model, strict=True)
            exact.restore_rng_state(previous_rng, self.loader_generator)
            raise
        return result

    def startup(
        self,
        request: InitializationRequest,
    ) -> RunnerSnapshot:
        """Start fresh, restore exact state, or load parent weights then start fresh."""

        if self.started:
            _fail("runner startup may be called only once")
        if not isinstance(request, InitializationRequest):
            _fail("request must be an InitializationRequest")
        request.validate()
        mode = exact.InitializationMode(request.mode)
        requested_initialization = request.initialization_contract()
        if (
            requested_initialization is not None
            and requested_initialization
            != self.normalized_spec["initialization_contract"]
        ):
            _fail(
                "initialization request differs from the run identity contract"
            )
        active = self.journal.load_active()

        if mode is exact.InitializationMode.EXACT_RESUME:
            if active is None:
                _fail("exact resume requires an active journal")
            events = self._read_events(active)
            selection = self.selection_policy.recompute(events)
            runtime_snapshot = self.runtime.startup(
                exact.InitializationMode.EXACT_RESUME,
                run_identity=self.run_identity,
                expected_epoch=active.epoch,
                expected_metrics_boundary=active.metrics_boundary,
                expected_best_selection=selection,
            )
            provenance = runtime_snapshot.extra_state.get(
                "initialization_provenance"
            )
            self._parent_provenance = (
                copy.deepcopy(provenance.get("parent"))
                if isinstance(provenance, Mapping)
                and isinstance(provenance.get("parent"), Mapping)
                else None
            )
            self._events = events
            self._best_selection = selection
        elif mode is exact.InitializationMode.PARENT_WARM_START:
            if active is not None:
                _fail("parent warm start requires an empty journal")
            self._require_no_derived_trajectory()
            self._require_initial_rng_contract()
            if request.extension_parent_provenance is not None:
                self._require_initial_model_state()
                self._parent_provenance = {
                    "mode": EXTENSION_PARENT_MODE,
                    "extension_warm_start": _normalize_extension_provenance(
                        request.extension_parent_provenance
                    ),
                }
            else:
                parent_result = self._restore_same_layout_parent_once(request)
                self._parent_provenance = {
                    "mode": exact.PARENT_WARM_START_MODE,
                    "parent_checkpoint_sha256": (
                        request.prepared_same_layout_parent
                        .checkpoint_sha256
                    ),
                    "parent_checkpoint_source_path": (
                        request.prepared_same_layout_parent.source_path
                    ),
                    "parent_epoch": parent_result.parent_epoch,
                    "parent_identity": parent_result.parent_identity,
                    "extra_state": parent_result.extra_state,
                }
            self.runtime.startup(
                exact.InitializationMode.FRESH,
                run_identity=self.run_identity,
            )
        else:
            self._require_no_derived_trajectory()
            self._require_initial_rng_contract()
            self._require_initial_model_state()
            self.runtime.startup(
                exact.InitializationMode.FRESH,
                run_identity=self.run_identity,
            )

        self._initialization_mode = mode
        if active is not None:
            self._derived_dirty = True
            self.repair_derived_artifacts()
        return self.snapshot

    def next_epoch_control(self) -> EpochControl:
        """Set and return the identity-bound LR/evaluation decision."""

        if not self.started:
            _fail("runner must be started before requesting an epoch")
        if self._pending is not None:
            _fail("a failed pending commit must be retried before training")
        if self._derived_dirty:
            _fail(
                "derived artifacts must be repaired before the next epoch"
            )
        if self._open_control is not None:
            _fail("the current epoch control has not been committed")
        epoch = self.runtime.snapshot.completed_epoch + 1
        total = self.normalized_spec["total_epochs"]
        if epoch > total:
            _fail("all identity-bound epochs are already complete")
        learning_rate = self.spec.lr_schedule.learning_rate(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        control = EpochControl(
            epoch=epoch,
            learning_rate=learning_rate,
            should_evaluate=(
                epoch == 1
                or epoch % self.normalized_spec["eval_interval"] == 0
                or epoch == total
            ),
        )
        self._open_control = control
        return control

    def _extra_state(self, caller: Mapping[str, Any] | None) -> dict[str, Any]:
        caller_state = _plain_json(caller or {}, "extra_state")
        if not isinstance(caller_state, dict):
            _fail("extra_state must be a mapping")
        return {
            "runner_schema": RUNNER_SCHEMA,
            "initialization_provenance": {
                "initial_mode": (
                    self._parent_provenance["mode"]
                    if self._parent_provenance is not None
                    else exact.InitializationMode.FRESH.value
                ),
                "parent": copy.deepcopy(self._parent_provenance),
            },
            "caller": caller_state,
        }

    def commit_epoch(
        self,
        fields: Mapping[str, Any],
        *,
        extra_state: Mapping[str, Any] | None = None,
    ) -> RunnerSnapshot:
        """Prepare and immediately commit one event without exposing a gap."""

        if not self.started:
            _fail("runner must be started before committing an epoch")
        if self._pending is not None:
            _fail("a failed pending commit exists; use retry_pending_commit")
        if self._open_control is None:
            _fail("next_epoch_control must be called before commit_epoch")
        if not isinstance(fields, Mapping):
            _fail("epoch fields must be a mapping")
        for index, group in enumerate(self.optimizer.param_groups):
            actual_lr = group.get("lr")
            if (
                isinstance(actual_lr, bool)
                or not isinstance(actual_lr, (int, float, np.number))
                or not math.isfinite(float(actual_lr))
                or float(actual_lr) != self._open_control.learning_rate
            ):
                _fail(
                    f"optimizer param-group {index} LR differs from the "
                    "open epoch control"
                )
        unexpected = sorted(_PROTECTED_EVENT_FIELDS & set(fields))
        if unexpected:
            _fail(f"epoch fields contain runner-owned keys: {unexpected}")
        event = _plain_json(fields, "epoch fields")
        if not isinstance(event, dict):
            _fail("epoch fields must be a mapping")
        event["epoch"] = self._open_control.epoch
        event["learning_rate"] = self._open_control.learning_rate
        annotated = self.selection_policy.annotate_event(self._events, event)
        primary_eligible = self.selection_policy.primary.eligible(annotated)
        secondary_eligible = self.selection_policy.secondary.eligible(annotated)
        if self._open_control.should_evaluate and not (
            primary_eligible and secondary_eligible
        ):
            _fail("scheduled evaluation epoch lacks selection metrics")
        proposed_events = [*self._events, annotated]
        selection = self.selection_policy.recompute(proposed_events)
        prepared = self.runtime.prepare_epoch(
            annotated,
            best_selection=selection,
            extra_state=self._extra_state(extra_state),
        )
        pending = _PendingCommit(
            prepared=prepared,
            event=copy.deepcopy(annotated),
            events=copy.deepcopy(proposed_events),
            best_selection=copy.deepcopy(selection),
        )
        self._pending = pending
        return self._commit_pending(pending)

    def _validate_pending_guard(self, pending: _PendingCommit) -> None:
        payload = pending.prepared.exact_payload
        checks = (
            ("model", self.model.state_dict(), payload["model"]["state_dict"]),
            (
                "optimizer",
                self.optimizer.state_dict(),
                payload["optimizer"]["state_dict"],
            ),
            ("scaler", self.scaler.state_dict(), payload["scaler"]["state_dict"]),
            (
                "random streams",
                exact.capture_rng_state(self.loader_generator),
                payload["rng_state"],
            ),
        )
        for label, current, captured in checks:
            if not _state_values_equal(current, captured):
                _fail(
                    f"{label} changed after epoch preparation; restart from "
                    "the active journal instead of committing stale state"
                )

    def _commit_pending(self, pending: _PendingCommit) -> RunnerSnapshot:
        self._validate_pending_guard(pending)
        runtime_snapshot = self.runtime.commit_epoch(pending.prepared)
        self._events = copy.deepcopy(pending.events)
        self._best_selection = copy.deepcopy(pending.best_selection)
        self._pending = None
        self._open_control = None
        self._derived_dirty = True
        try:
            self.repair_derived_artifacts(runtime_snapshot)
        except BaseException as exc:
            raise ExactRunnerError(
                "epoch is committed in the active journal but derived artifact "
                f"publication failed: {exc}"
            ) from exc
        return self.snapshot

    def retry_pending_commit(self) -> RunnerSnapshot:
        """Retry the identical payload after a journal write failure."""

        if self._pending is None:
            _fail("there is no failed pending commit to retry")
        adopted = self._adopt_pending_if_marker_switched(self._pending)
        if adopted is not None:
            return adopted
        return self._commit_pending(self._pending)

    def _adopt_pending_if_marker_switched(
        self,
        pending: _PendingCommit,
    ) -> RunnerSnapshot | None:
        active = self.journal.load_active()
        previous_epoch = pending.prepared.epoch - 1
        if active is None or active.epoch == previous_epoch:
            return None
        if active.epoch != pending.prepared.epoch:
            _fail("active journal advanced beyond the pending epoch")
        if active.metrics_boundary != pending.prepared.metrics_boundary:
            _fail("active journal boundary differs from the pending commit")
        metrics_bytes = _read_regular(
            active.metrics_path,
            "adopted active metrics",
        )
        if metrics_bytes != pending.prepared.event.metrics_bytes:
            _fail("active journal metrics differ from the pending commit")
        active_payload, _ = self._load_exact_payload(active.checkpoint_path)
        if not _state_values_equal(
            active_payload,
            pending.prepared.exact_payload,
        ):
            _fail("active checkpoint differs from the pending exact payload")
        runtime_previous = self.runtime.snapshot
        if runtime_previous.completed_epoch != previous_epoch:
            _fail("runtime state cannot adopt the already active epoch")
        if self.runtime._pending is not pending.prepared:
            _fail("runtime pending state differs from runner pending state")
        payload = pending.prepared.exact_payload
        runtime_snapshot = ExactTrainingSnapshot(
            mode=runtime_previous.mode,
            completed_epoch=pending.prepared.epoch,
            run_identity=copy.deepcopy(payload["run_identity"]),
            metrics_boundary=copy.deepcopy(payload["metrics_boundary"]),
            best_selection=copy.deepcopy(payload["best_selection"]),
            extra_state=copy.deepcopy(payload["extra_state"]),
            active=active,
        )
        self.runtime._snapshot = runtime_snapshot
        self.runtime._pending = None
        self._events = copy.deepcopy(pending.events)
        self._best_selection = copy.deepcopy(pending.best_selection)
        self._pending = None
        self._open_control = None
        self._derived_dirty = True
        try:
            self.repair_derived_artifacts(runtime_snapshot)
        except BaseException as exc:
            raise ExactRunnerError(
                "epoch was already committed in the active journal but "
                f"derived artifact publication failed: {exc}"
            ) from exc
        return self.snapshot

    def _load_exact_payload(
        self,
        path: Path,
    ) -> tuple[dict[str, Any], str]:
        content = _read_regular(path, "exact checkpoint")
        try:
            payload = torch.load(
                io.BytesIO(content),
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            _fail(f"cannot load exact checkpoint {path}: {exc}")
        if not isinstance(payload, Mapping):
            _fail(f"exact checkpoint is not a mapping: {path}")
        return dict(payload), hashlib.sha256(content).hexdigest()

    def _candidate_exact_payload(
        self,
        epoch: int,
        active: ActiveEpochState,
    ) -> tuple[dict[str, Any], str] | None:
        candidates = [active.checkpoint_path]
        for _, checkpoint_name in journal_module.SLOT_FILES.values():
            path = self.journal.root / checkpoint_name
            if path not in candidates and path.is_file() and not path.is_symlink():
                candidates.append(path)
        for path in candidates:
            try:
                payload, checkpoint_sha256 = self._load_exact_payload(path)
            except ExactRunnerError:
                continue
            if (
                payload.get("schema") == exact.EXACT_RESUME_SCHEMA
                and payload.get("mode") == exact.EXACT_RESUME_MODE
                and payload.get("epoch") == epoch
                and payload.get("run_identity") == self.run_identity
            ):
                return payload, checkpoint_sha256
        return None

    def _legacy_payload(
        self,
        source: Mapping[str, Any],
        *,
        source_exact_checkpoint_sha256: str,
        role: str,
        metrics: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> dict[str, Any]:
        identity = source["run_identity"]
        default_payload = {
            "derived_schema": DERIVED_CHECKPOINT_SCHEMA,
            "checkpoint_role": role,
            "epoch": int(source["epoch"]),
            "variant": identity["variant"],
            "dataset": identity["dataset"],
            "seed": identity["seed"],
            "split_seed": identity["split_seed"],
            "state_dict": copy.deepcopy(source["model"]["state_dict"]),
            "optimizer": copy.deepcopy(source["optimizer"]["state_dict"]),
            "scaler": copy.deepcopy(source["scaler"]["state_dict"]),
            "scheduler": None,
            "validation_metrics": copy.deepcopy(dict(metrics)),
            "model_metadata": copy.deepcopy(
                self.normalized_spec["builder_metadata"]
            ),
            "split_hashes": {
                name: record["sha256"]
                for name, record in self.run_identity[
                    "ordered_split_fingerprints"
                ].items()
            },
            "run_identity": copy.deepcopy(self.run_identity),
            "selection_source": "active_exact_epoch_journal",
        }
        if self.compatibility_payload_factory is None:
            payload = default_payload
        else:
            context = CompatibilityPayloadContext(
                role=role,
                epoch=int(source["epoch"]),
                metrics=copy.deepcopy(dict(metrics)),
                event=copy.deepcopy(dict(event)),
                exact_payload=copy.deepcopy(dict(source)),
                run_identity=copy.deepcopy(self.run_identity),
                normalized_spec=copy.deepcopy(self.normalized_spec),
            )
            adapted = self.compatibility_payload_factory(context)
            if not isinstance(adapted, Mapping):
                _fail("compatibility payload factory must return a mapping")
            payload = copy.deepcopy(dict(adapted))
            reserved = sorted(_DERIVED_SOURCE_FIELDS & set(payload))
            if reserved:
                _fail(
                    "compatibility payload factory returned runner-owned "
                    f"source fields: {reserved}"
                )
            existing_schema = payload.get(
                "derived_schema",
                DERIVED_CHECKPOINT_SCHEMA,
            )
            if existing_schema != DERIVED_CHECKPOINT_SCHEMA:
                _fail(
                    "compatibility payload factory returned an inconsistent "
                    "derived_schema"
                )
            payload["derived_schema"] = DERIVED_CHECKPOINT_SCHEMA
            required = {
                "epoch": context.epoch,
                "checkpoint_role": role,
                "run_identity": self.run_identity,
            }
            for key, expected in required.items():
                if payload.get(key) != expected:
                    _fail(
                        "compatibility payload factory returned an inconsistent "
                        f"{key}"
                    )
            if payload.get("validation_metrics") != context.metrics:
                _fail(
                    "compatibility payload factory returned inconsistent "
                    "validation_metrics"
                )
        exact_components = {
            "state_dict": source["model"]["state_dict"],
            "optimizer": source["optimizer"]["state_dict"],
            "scaler": source["scaler"]["state_dict"],
        }
        for key, expected in exact_components.items():
            actual = payload.get(key)
            if not isinstance(actual, Mapping):
                _fail(
                    "compatibility payload factory must return top-level "
                    f"{key}"
                )
            if not _state_values_equal(actual, expected):
                _fail(
                    "compatibility payload factory changed exact source "
                    f"{key}"
                )
        payload["source_exact_checkpoint_sha256"] = _validate_sha256(
            source_exact_checkpoint_sha256,
            "source exact checkpoint SHA-256",
        )
        payload["state_dict_sha256"] = _state_content_sha256(
            payload["state_dict"],
            "derived state_dict",
        )
        payload["optimizer_state_sha256"] = _state_content_sha256(
            payload["optimizer"],
            "derived optimizer",
        )
        payload["scaler_state_sha256"] = _state_content_sha256(
            payload["scaler"],
            "derived scaler",
        )
        return payload

    def _derived_checkpoint_valid(
        self,
        path: Path,
        *,
        epoch: int,
        role: str,
        metrics: Mapping[str, Any],
        active: ActiveEpochState,
    ) -> bool:
        if not path.is_file() or path.is_symlink():
            return False
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return False
        if not (
            isinstance(payload, Mapping)
            and payload.get("derived_schema") == DERIVED_CHECKPOINT_SCHEMA
            and payload.get("checkpoint_role") == role
            and payload.get("epoch") == epoch
            and payload.get("run_identity") == self.run_identity
            and isinstance(payload.get("state_dict"), Mapping)
            and isinstance(payload.get("optimizer"), Mapping)
            and isinstance(payload.get("scaler"), Mapping)
            and payload.get("validation_metrics") == dict(metrics)
        ):
            return False
        try:
            source_sha256 = _validate_sha256(
                payload.get("source_exact_checkpoint_sha256"),
                "derived source exact checkpoint SHA-256",
            )
            expected_digests = {
                "state_dict_sha256": _state_content_sha256(
                    payload["state_dict"],
                    "derived state_dict",
                ),
                "optimizer_state_sha256": _state_content_sha256(
                    payload["optimizer"],
                    "derived optimizer",
                ),
                "scaler_state_sha256": _state_content_sha256(
                    payload["scaler"],
                    "derived scaler",
                ),
            }
            for field, expected in expected_digests.items():
                if payload.get(field) != expected:
                    return False
            source = self._candidate_exact_payload(epoch, active)
            if source is not None:
                source_payload, actual_source_sha256 = source
                if source_sha256 != actual_source_sha256:
                    return False
                comparisons = (
                    (
                        payload["state_dict"],
                        source_payload["model"]["state_dict"],
                    ),
                    (
                        payload["optimizer"],
                        source_payload["optimizer"]["state_dict"],
                    ),
                    (
                        payload["scaler"],
                        source_payload["scaler"]["state_dict"],
                    ),
                )
                if not all(
                    _state_values_equal(actual, expected)
                    for actual, expected in comparisons
                ):
                    return False
        except (ExactRunnerError, KeyError, TypeError, ValueError):
            return False
        return True

    def _publish_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        role: str,
        metrics: Mapping[str, Any],
        event: Mapping[str, Any],
        active: ActiveEpochState,
        always: bool,
    ) -> None:
        if not always and self._derived_checkpoint_valid(
            path,
            epoch=epoch,
            role=role,
            metrics=metrics,
            active=active,
        ):
            return
        source = self._candidate_exact_payload(epoch, active)
        if source is None:
            _fail(
                f"journal no longer retains epoch {epoch} needed to rebuild "
                f"{path.name}; retain the valid derived best checkpoint"
            )
        source_payload, source_checkpoint_sha256 = source
        exact.atomic_torch_save(
            self._legacy_payload(
                source_payload,
                source_exact_checkpoint_sha256=(
                    source_checkpoint_sha256
                ),
                role=role,
                metrics=metrics,
                event=event,
            ),
            path,
        )

    def repair_derived_artifacts(
        self,
        runtime_snapshot: ExactTrainingSnapshot | None = None,
    ) -> None:
        """Rebuild compatibility views without changing any random stream."""

        if not self.started and runtime_snapshot is None:
            _fail("runner must be started before repairing derived artifacts")
        self._derived_dirty = True
        rng_before = exact.capture_rng_state(self.loader_generator)
        try:
            self._repair_derived_artifacts(runtime_snapshot)
        except BaseException as exc:
            rng_after_failure = exact.capture_rng_state(self.loader_generator)
            if not _state_values_equal(rng_before, rng_after_failure):
                exact.restore_rng_state(rng_before, self.loader_generator)
                raise ExactRunnerError(
                    "derived artifact publication consumed random streams; "
                    "the streams were restored"
                ) from exc
            raise
        rng_after = exact.capture_rng_state(self.loader_generator)
        if not _state_values_equal(rng_before, rng_after):
            exact.restore_rng_state(rng_before, self.loader_generator)
            _fail(
                "derived artifact publication consumed random streams; "
                "the streams were restored"
            )
        self._derived_dirty = False

    def _repair_derived_artifacts(
        self,
        runtime_snapshot: ExactTrainingSnapshot | None,
    ) -> None:
        snapshot = runtime_snapshot or self.runtime.snapshot
        active = snapshot.active or self.journal.load_active()
        if active is None:
            return
        events = self._read_events(active)
        selection = self.selection_policy.recompute(events)
        if snapshot.best_selection is not None and (
            snapshot.best_selection != selection
        ):
            _fail("runtime best selection differs from active metrics")
        metrics_bytes = _read_regular(active.metrics_path, "active metrics")
        _atomic_write_bytes(
            self.run_directory / METRICS_FILENAME,
            metrics_bytes,
        )
        (
            active_payload,
            active_checkpoint_sha256,
        ) = self._load_exact_payload(active.checkpoint_path)
        if active_checkpoint_sha256 != active.checkpoint_sha256:
            _fail("active checkpoint digest changed during derived repair")
        last_event = events[-1]
        last_metrics = {
            name: last_event[name]
            for name in self.selection_policy.primary.stored_metrics
            if name in last_event
        }
        exact.atomic_torch_save(
            self._legacy_payload(
                active_payload,
                source_exact_checkpoint_sha256=active_checkpoint_sha256,
                role="last_completed_epoch",
                metrics=last_metrics,
                event=last_event,
            ),
            self.run_directory / LAST_FILENAME,
        )
        for slot, filename in (
            ("primary", BEST_FILENAME),
            ("secondary", BEST_MIOU_FILENAME),
        ):
            record = selection[slot]
            self._publish_checkpoint(
                self.run_directory / filename,
                epoch=int(record["epoch"]),
                role=str(record["role"]),
                metrics=record["metrics"],
                event=events[int(record["epoch"]) - 1],
                active=active,
                always=False,
            )
        self._events = events
        self._best_selection = selection


__all__ = [
    "ARCHITECTURE_SCHEMA",
    "BEST_FILENAME",
    "BEST_MIOU_FILENAME",
    "DERIVED_CHECKPOINT_SCHEMA",
    "EXTENSION_PARENT_MODE",
    "INITIALIZATION_CONTRACT_SCHEMA",
    "INITIAL_RNG_CONTRACT_SCHEMA",
    "OPTIMIZER_CONTRACT_SCHEMA",
    "SCALER_CONTRACT_SCHEMA",
    "CompatibilityPayloadContext",
    "CompatibilityPayloadFactory",
    "EpochControl",
    "ExactRunSpec",
    "ExactRunner",
    "ExactRunnerError",
    "InitializationRequest",
    "JOURNAL_DIRECTORY",
    "LAST_FILENAME",
    "MANUAL_COSINE_SCHEMA",
    "ManualCosineSchedule",
    "METRICS_FILENAME",
    "MetricOrder",
    "OrderedFingerprint",
    "PreparedSameLayoutParent",
    "RUNNER_SCHEMA",
    "RUN_IDENTITY_SCHEMA",
    "SELECTION_POLICY_SCHEMA",
    "RunnerSnapshot",
    "SelectionPolicy",
    "SelectionRule",
    "build_run_identity",
    "compute_architecture_id",
    "extension_parent_initialization_contract",
    "fresh_initialization_contract",
    "initial_model_state_sha256",
    "initial_rng_contract",
    "optimizer_contract",
    "pd_miou_selection_policy",
    "prepare_same_layout_parent",
    "same_layout_parent_initialization_contract",
    "scaler_contract",
]
