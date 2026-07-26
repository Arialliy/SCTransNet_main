"""Strict parent-to-extension weight transfer for future TPD modules.

This loader is intentionally architecture-agnostic.  The caller supplies a
fresh parent model as the authoritative parent layout and a separately built
extension model.  Every parent state key must exist in the checkpoint and the
extension with the same tensor shape and dtype.  The only extension-only keys
accepted are those under explicitly declared, genuinely new module prefixes.

Initialization remains the extension builder's responsibility.  This module
does not invoke model-specific initialization helpers.  It verifies declared
zero-initialized state prefixes and preserves every extension-only state value
while copying the parent state.
"""

from __future__ import annotations

import copy
import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


PROVENANCE_SCHEMA = "sctransnet_tpd_extension_warm_start_v1"
_SHA256_LENGTH = 64


class ExtensionWarmStartError(ValueError):
    """The requested parent-to-extension transfer violates its contract."""


@dataclass(frozen=True)
class ExtensionWarmStartResult:
    """Auditable provenance returned after a successful transfer."""

    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    parent_state_dict_path: tuple[str, ...]
    parent_state_key_count: int
    preserved_new_state_key_count: int
    new_module_prefixes: tuple[str, ...]
    zero_init_prefixes: tuple[str, ...]

    def provenance(self) -> dict[str, Any]:
        return {
            "schema": PROVENANCE_SCHEMA,
            "parent_checkpoint_path": self.parent_checkpoint_path,
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "parent_state_dict_path": list(self.parent_state_dict_path),
            "parent_state_key_count": self.parent_state_key_count,
            "preserved_new_state_key_count": (
                self.preserved_new_state_key_count
            ),
            "new_module_prefixes": list(self.new_module_prefixes),
            "zero_init_prefixes": list(self.zero_init_prefixes),
        }


def _fail(message: str) -> None:
    raise ExtensionWarmStartError(message)


def _is_state_key_under(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(f"{prefix}.")


def _validate_state_name(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.startswith(".")
        or value.endswith(".")
        or ".." in value
        or any(part.strip() != part for part in value.split("."))
    ):
        _fail(f"{label} must be a canonical non-empty state prefix")
    return value


def _normalize_prefixes(
    values: Sequence[str],
    *,
    label: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        _fail(f"{label} must be a sequence of prefixes")
    normalized = tuple(
        _validate_state_name(value, f"{label}[{index}]")
        for index, value in enumerate(values)
    )
    if not normalized and not allow_empty:
        _fail(f"{label} must not be empty")
    if len(set(normalized)) != len(normalized):
        _fail(f"{label} contains duplicate prefixes")
    for left_index, left in enumerate(normalized):
        for right in normalized[left_index + 1 :]:
            if _is_state_key_under(left, right) or _is_state_key_under(
                right,
                left,
            ):
                _fail(
                    f"{label} contains overlapping prefixes "
                    f"{left!r} and {right!r}"
                )
    return normalized


def _normalize_state_dict_path(
    value: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail("parent_state_dict_path must be a sequence of mapping keys")
    path = tuple(
        _validate_state_name(item, f"parent_state_dict_path[{index}]")
        for index, item in enumerate(value)
    )
    if not path:
        _fail("parent_state_dict_path must not be empty")
    return path


def _validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _load_checkpoint_bytes(
    checkpoint: str | os.PathLike[str],
    *,
    map_location: str | torch.device,
) -> tuple[Path, str, Any]:
    path = Path(checkpoint)
    if path.is_symlink() or not path.is_file():
        _fail(f"parent checkpoint is not a regular file: {path}")
    try:
        content = path.read_bytes()
    except OSError as exc:
        _fail(f"cannot read parent checkpoint {path}: {exc}")
    digest = hashlib.sha256(content).hexdigest()
    try:
        payload = torch.load(
            io.BytesIO(content),
            map_location=map_location,
            weights_only=False,
        )
    except Exception as exc:
        _fail(f"cannot load parent checkpoint {path}: {exc}")
    return path, digest, payload


def _extract_state_dict(
    payload: Any,
    path: tuple[str, ...],
) -> dict[str, Any]:
    current = payload
    traversed: list[str] = []
    for item in path:
        traversed.append(item)
        if not isinstance(current, Mapping) or item not in current:
            _fail(
                "parent checkpoint is missing state-dict path "
                f"{'.'.join(traversed)!r}"
            )
        current = current[item]
    if not isinstance(current, Mapping):
        _fail("parent checkpoint state_dict must be a mapping")
    state = dict(current)
    for key in state:
        if not isinstance(key, str) or not key:
            _fail("parent checkpoint state_dict keys must be non-empty strings")
    return state


def _require_tensor(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        _fail(f"{label} must be a tensor state entry")
    return value


def _require_same_layout(
    *,
    key: str,
    checkpoint_value: Any,
    parent_value: Any,
    extension_value: Any,
) -> None:
    checkpoint_tensor = _require_tensor(
        checkpoint_value,
        label=f"checkpoint state {key!r}",
    )
    parent_tensor = _require_tensor(
        parent_value,
        label=f"parent model state {key!r}",
    )
    extension_tensor = _require_tensor(
        extension_value,
        label=f"extension model state {key!r}",
    )
    expected_shape = tuple(parent_tensor.shape)
    if tuple(checkpoint_tensor.shape) != expected_shape:
        _fail(
            f"parent checkpoint shape mismatch for {key!r}: "
            f"{tuple(checkpoint_tensor.shape)} != {expected_shape}"
        )
    if tuple(extension_tensor.shape) != expected_shape:
        _fail(
            f"extension model shape mismatch for parent key {key!r}: "
            f"{tuple(extension_tensor.shape)} != {expected_shape}"
        )
    expected_dtype = parent_tensor.dtype
    if checkpoint_tensor.dtype != expected_dtype:
        _fail(
            f"parent checkpoint dtype mismatch for {key!r}: "
            f"{checkpoint_tensor.dtype} != {expected_dtype}"
        )
    if extension_tensor.dtype != expected_dtype:
        _fail(
            f"extension model dtype mismatch for parent key {key!r}: "
            f"{extension_tensor.dtype} != {expected_dtype}"
        )


def _validate_new_state_contract(
    *,
    parent_keys: set[str],
    extension_only_keys: set[str],
    new_module_prefixes: tuple[str, ...],
    zero_init_prefixes: tuple[str, ...],
    extension_state: Mapping[str, Any],
) -> None:
    for prefix in new_module_prefixes:
        overlapping_parent = sorted(
            key for key in parent_keys if _is_state_key_under(key, prefix)
        )
        if overlapping_parent:
            _fail(
                f"new module prefix {prefix!r} overlaps parent state "
                f"{overlapping_parent[0]!r}"
            )
        matching_new = sorted(
            key
            for key in extension_only_keys
            if _is_state_key_under(key, prefix)
        )
        if not matching_new:
            _fail(
                f"new module prefix {prefix!r} does not identify extension-only "
                "state"
            )

    undeclared = sorted(
        key
        for key in extension_only_keys
        if not any(
            _is_state_key_under(key, prefix)
            for prefix in new_module_prefixes
        )
    )
    if undeclared:
        _fail(f"extension state key is not explicitly declared new: {undeclared[0]!r}")

    for key in extension_only_keys:
        _require_tensor(
            extension_state[key],
            label=f"extension-only state {key!r}",
        )

    for prefix in zero_init_prefixes:
        owning_prefixes = [
            allowed
            for allowed in new_module_prefixes
            if _is_state_key_under(prefix, allowed)
        ]
        if len(owning_prefixes) != 1:
            _fail(
                f"zero-init prefix {prefix!r} is not contained in exactly one "
                "new module prefix"
            )
        matching_keys = sorted(
            key
            for key in extension_only_keys
            if _is_state_key_under(key, prefix)
        )
        if not matching_keys:
            _fail(
                f"zero-init prefix {prefix!r} does not identify extension-only "
                "tensor state"
            )
        for key in matching_keys:
            tensor = _require_tensor(
                extension_state[key],
                label=f"zero-init state {key!r}",
            )
            if torch.count_nonzero(tensor).item() != 0:
                _fail(
                    f"zero-initialization contract violated for extension "
                    f"state {key!r}"
                )


def _restore_previous_state(
    model: nn.Module,
    previous: Mapping[str, Any],
    *,
    original_error: BaseException,
) -> None:
    try:
        model.load_state_dict(previous, strict=True)
    except BaseException as rollback_error:
        raise ExtensionWarmStartError(
            "extension warm-start failed and rollback failed: "
            f"{original_error}; {rollback_error}"
        ) from original_error
    raise ExtensionWarmStartError(
        f"extension warm-start state load failed: {original_error}"
    ) from original_error


def load_parent_into_extension(
    parent_checkpoint: str | os.PathLike[str],
    *,
    parent_model: nn.Module,
    extension_model: nn.Module,
    new_module_prefixes: Sequence[str],
    zero_init_prefixes: Sequence[str] = (),
    parent_state_dict_path: Sequence[str] = ("state_dict",),
    expected_parent_checkpoint_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
) -> ExtensionWarmStartResult:
    """Strictly copy a complete parent state into a larger model.

    ``parent_model`` is a freshly built reference architecture.  It is never
    modified.  ``extension_model`` must retain every parent state key under the
    same fully qualified name and may add state only below
    ``new_module_prefixes``.  Values under ``zero_init_prefixes`` are verified
    to be exactly zero before any parent value is loaded.
    """

    if not isinstance(parent_model, nn.Module):
        _fail("parent_model must be an nn.Module")
    if not isinstance(extension_model, nn.Module):
        _fail("extension_model must be an nn.Module")
    if parent_model is extension_model:
        _fail("parent_model and extension_model must be distinct instances")

    allowed_prefixes = _normalize_prefixes(
        new_module_prefixes,
        label="new_module_prefixes",
        allow_empty=False,
    )
    required_zero_prefixes = _normalize_prefixes(
        zero_init_prefixes,
        label="zero_init_prefixes",
        allow_empty=True,
    )
    state_path = _normalize_state_dict_path(parent_state_dict_path)
    path, checkpoint_sha256, payload = _load_checkpoint_bytes(
        parent_checkpoint,
        map_location=map_location,
    )
    if expected_parent_checkpoint_sha256 is not None:
        expected_digest = _validate_sha256(
            expected_parent_checkpoint_sha256,
            "expected_parent_checkpoint_sha256",
        )
        if checkpoint_sha256 != expected_digest:
            _fail(
                "parent checkpoint SHA-256 mismatch: "
                f"{checkpoint_sha256} != {expected_digest}"
            )

    checkpoint_state = _extract_state_dict(payload, state_path)
    parent_state = parent_model.state_dict()
    extension_state = extension_model.state_dict()
    checkpoint_keys = set(checkpoint_state)
    parent_keys = set(parent_state)
    extension_keys = set(extension_state)

    missing_parent_checkpoint_keys = sorted(parent_keys - checkpoint_keys)
    if missing_parent_checkpoint_keys:
        _fail(
            "parent checkpoint omits required parent key "
            f"{missing_parent_checkpoint_keys[0]!r}"
        )
    unexpected_checkpoint_keys = sorted(checkpoint_keys - parent_keys)
    if unexpected_checkpoint_keys:
        _fail(
            "parent checkpoint contains unexpected key "
            f"{unexpected_checkpoint_keys[0]!r}"
        )
    missing_extension_parent_keys = sorted(parent_keys - extension_keys)
    if missing_extension_parent_keys:
        _fail(
            "extension model omits parent key "
            f"{missing_extension_parent_keys[0]!r}"
        )

    for key in sorted(parent_keys):
        _require_same_layout(
            key=key,
            checkpoint_value=checkpoint_state[key],
            parent_value=parent_state[key],
            extension_value=extension_state[key],
        )

    extension_only_keys = extension_keys - parent_keys
    _validate_new_state_contract(
        parent_keys=parent_keys,
        extension_only_keys=extension_only_keys,
        new_module_prefixes=allowed_prefixes,
        zero_init_prefixes=required_zero_prefixes,
        extension_state=extension_state,
    )

    previous_state = copy.deepcopy(extension_state)
    merged_state = copy.deepcopy(extension_state)
    for key in parent_keys:
        merged_state[key] = copy.deepcopy(checkpoint_state[key])
    try:
        incompatible = extension_model.load_state_dict(merged_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "strict load returned incompatible keys: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        loaded_state = extension_model.state_dict()
        for key in parent_keys:
            if not torch.equal(loaded_state[key], checkpoint_state[key]):
                raise RuntimeError(
                    f"loaded parent tensor differs for state {key!r}"
                )
        for key in extension_only_keys:
            if not torch.equal(loaded_state[key], previous_state[key]):
                raise RuntimeError(
                    f"extension-only state was overwritten for {key!r}"
                )
    except BaseException as exc:
        _restore_previous_state(
            extension_model,
            previous_state,
            original_error=exc,
        )

    return ExtensionWarmStartResult(
        parent_checkpoint_path=str(path.resolve()),
        parent_checkpoint_sha256=checkpoint_sha256,
        parent_state_dict_path=state_path,
        parent_state_key_count=len(parent_keys),
        preserved_new_state_key_count=len(extension_only_keys),
        new_module_prefixes=allowed_prefixes,
        zero_init_prefixes=required_zero_prefixes,
    )


__all__ = [
    "ExtensionWarmStartError",
    "ExtensionWarmStartResult",
    "PROVENANCE_SCHEMA",
    "load_parent_into_extension",
]
