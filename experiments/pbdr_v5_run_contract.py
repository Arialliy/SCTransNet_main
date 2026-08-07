"""Fail-closed identity and resume contract for the single-arm PBDR-V5 run.

This module is pure with respect to datasets and checkpoints.  It defines the
identity fields that make two V5 runs interchangeable and validates rolling
payloads before a caller installs any tensor state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

from experiments.pbdr_v4_run_artifacts import optimizer_group_signature
from experiments.pbdr_v4_state_contract import state_semantic_sha256
from experiments.pbdr_v4_training_core import checkpoint_epoch_key


SCHEMA = "sctransnet_pbdr_v5_run_contract/v1"
ROLLING_SCHEMA = f"{SCHEMA}/rolling"
ARM = "target_preserve_stage2"
FORMAL_EPOCHS = 30
FORMAL_EVAL_EVERY = 5
FORMAL_BATCH_SIZE = 16
FORMAL_SEED = 42
DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
ROLES = ("best_miou", "best_pd")


class PBDRV5RunContractError(RuntimeError):
    """A V5 identity, selection, or rolling artifact is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PBDRV5RunContractError(message)


def require_sha256(value: object, *, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return value


def canonical_json_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise PBDRV5RunContractError(
            f"value cannot be represented as canonical JSON: {error}"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def ordered_strings_sha256(values: Sequence[str]) -> str:
    _require(
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes))
        and bool(values)
        and all(isinstance(value, str) and value for value in values),
        "ordered string sequence must be non-empty",
    )
    return canonical_json_sha256(list(values))


@dataclass(frozen=True, slots=True)
class V5RunIdentity:
    dataset: str
    role: str
    arm: str
    v4_source_lock_sha256: str
    split_projection_sha256: str
    atlas_manifest_sha256: str
    parent_checkpoint_sha256: str
    parent_state_sha256: str
    stage1_checkpoint_sha256: str
    stage1_state_sha256: str
    v5_source_sha256: str
    loss_manifest_sha256: str
    trainable_parameter_names_sha256: str

    def __post_init__(self) -> None:
        _require(self.dataset in DATASETS, "unsupported dataset")
        _require(self.role in ROLES, "unsupported role")
        _require(self.arm == ARM, "unsupported V5 arm")
        for name in (
            "v4_source_lock_sha256",
            "split_projection_sha256",
            "atlas_manifest_sha256",
            "parent_checkpoint_sha256",
            "parent_state_sha256",
            "stage1_checkpoint_sha256",
            "stage1_state_sha256",
            "v5_source_sha256",
            "loss_manifest_sha256",
            "trainable_parameter_names_sha256",
        ):
            require_sha256(getattr(self, name), name=name)

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def semantic_sha256(self) -> str:
        return canonical_json_sha256(self.as_dict())


def epoch_selection_key(
    role: str,
    metrics: Mapping[str, object],
    epoch: int,
) -> tuple[object, ...]:
    """Return the complete V4 role key with epoch zero as the earliest tie.

    V4's helper accepts positive epochs only.  Epoch zero uses the same metric
    prefix and a final tie component of zero, which is greater than every
    trained epoch's ``-epoch`` component.
    """

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise PBDRV5RunContractError("epoch must be a non-negative integer")
    if epoch > 0:
        return checkpoint_epoch_key(role, metrics, epoch)
    positive = checkpoint_epoch_key(role, metrics, 1)
    return (*positive[:-1], 0)


def json_selection_key(key: Sequence[object]) -> list[object]:
    result: list[object] = []
    for item in key:
        if isinstance(item, Fraction):
            result.append(
                {"numerator": item.numerator, "denominator": item.denominator}
            )
        elif isinstance(item, (float, int, str)) or item is None:
            result.append(item)
        else:
            raise PBDRV5RunContractError(
                f"unsupported selection-key component: {type(item).__name__}"
            )
    return result


def _clone_state(value: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    _require(
        isinstance(value, Mapping)
        and bool(value)
        and all(
            isinstance(name, str) and isinstance(tensor, torch.Tensor)
            for name, tensor in value.items()
        ),
        "model state must be a non-empty string-to-tensor mapping",
    )
    return {
        name: tensor.detach().cpu().clone() for name, tensor in value.items()
    }


def build_rolling_payload(
    *,
    identity: V5RunIdentity,
    epoch: int,
    epochs: int,
    model_state: Mapping[str, torch.Tensor],
    optimizer_state: Mapping[str, Any],
    rng_state: Mapping[str, Any],
    selected: Mapping[str, Any],
    evaluation_history: Sequence[Mapping[str, Any]],
    event: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        isinstance(epoch, int)
        and not isinstance(epoch, bool)
        and isinstance(epochs, int)
        and not isinstance(epochs, bool)
        and 1 <= epoch <= epochs,
        "rolling epoch is outside its budget",
    )
    _require(
        set(rng_state) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "RNG state keys differ",
    )
    _require(isinstance(selected, Mapping) and bool(selected), "selection is empty")
    _require(
        isinstance(evaluation_history, Sequence) and bool(evaluation_history),
        "evaluation history is empty",
    )
    state = _clone_state(model_state)
    return {
        "schema": ROLLING_SCHEMA,
        "identity": identity.as_dict(),
        "identity_sha256": identity.semantic_sha256,
        "epoch": epoch,
        "epochs": epochs,
        "state_dict": state,
        "state_sha256": state_semantic_sha256(state),
        "optimizer": dict(optimizer_state),
        "optimizer_group_signature": optimizer_group_signature(optimizer_state),
        "rng_state": dict(rng_state),
        "selected": dict(selected),
        "evaluation_history": [dict(item) for item in evaluation_history],
        "event": dict(event),
        "official_test_accessed": False,
        "performance_acceptance_margin": None,
    }


def _validate_selected(
    value: object,
    *,
    identity: V5RunIdentity,
    epochs: int,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "rolling selection must be a mapping")
    required = {
        "epoch",
        "metrics",
        "diagnostics",
        "selection_key",
        "selection_key_raw",
        "state_dict",
        "state_sha256",
    }
    _require(set(value) == required, "rolling selection fields differ")
    epoch = value["epoch"]
    _require(
        isinstance(epoch, int)
        and not isinstance(epoch, bool)
        and 0 <= epoch <= epochs,
        "selected epoch differs",
    )
    metrics = value["metrics"]
    _require(isinstance(metrics, Mapping), "selected metrics differ")
    replayed = epoch_selection_key(identity.role, metrics, epoch)
    _require(tuple(value["selection_key_raw"]) == replayed, "raw key does not replay")
    _require(value["selection_key"] == json_selection_key(replayed), "JSON key differs")
    state = value["state_dict"]
    _require(isinstance(state, Mapping), "selected state differs")
    observed_sha = state_semantic_sha256(state)  # type: ignore[arg-type]
    _require(value["state_sha256"] == observed_sha, "selected state SHA differs")
    return dict(value)


def validate_rolling_payload(
    payload: Mapping[str, Any],
    *,
    identity: V5RunIdentity,
    epochs: int,
    expected_optimizer_group_signature: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate identity and every resume-critical field before state install."""

    _require(isinstance(payload, Mapping), "rolling payload must be a mapping")
    _require(payload.get("schema") == ROLLING_SCHEMA, "rolling schema differs")
    _require(payload.get("identity") == identity.as_dict(), "run identity differs")
    _require(
        payload.get("identity_sha256") == identity.semantic_sha256,
        "run identity SHA differs",
    )
    epoch = payload.get("epoch")
    _require(
        isinstance(epoch, int)
        and not isinstance(epoch, bool)
        and payload.get("epochs") == epochs
        and 1 <= epoch <= epochs,
        "rolling epoch/budget differs",
    )
    state = payload.get("state_dict")
    _require(isinstance(state, Mapping) and bool(state), "rolling state differs")
    _require(
        payload.get("state_sha256")
        == state_semantic_sha256(state),  # type: ignore[arg-type]
        "rolling state SHA differs",
    )
    optimizer = payload.get("optimizer")
    _require(isinstance(optimizer, Mapping), "rolling optimizer differs")
    observed_signature = optimizer_group_signature(optimizer)
    _require(
        payload.get("optimizer_group_signature") == observed_signature
        == expected_optimizer_group_signature,
        "rolling optimizer-group signature differs",
    )
    rng = payload.get("rng_state")
    _require(
        isinstance(rng, Mapping)
        and set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"},
        "rolling RNG state differs",
    )
    selected = _validate_selected(
        payload.get("selected"), identity=identity, epochs=epochs
    )
    history = payload.get("evaluation_history")
    _require(isinstance(history, list) and bool(history), "evaluation history differs")
    _require(history[0].get("epoch") == 0, "history lacks epoch-zero baseline")
    _require(
        all(
            isinstance(item, Mapping)
            and isinstance(item.get("epoch"), int)
            and 0 <= item["epoch"] <= epoch
            for item in history
        ),
        "evaluation history entries differ",
    )
    _require(isinstance(payload.get("event"), Mapping), "rolling event differs")
    _require(payload.get("official_test_accessed") is False, "official flag differs")
    _require(
        payload.get("performance_acceptance_margin") is None,
        "performance margin differs",
    )
    result = dict(payload)
    result["selected"] = selected
    return result


__all__ = [
    "ARM",
    "DATASETS",
    "FORMAL_BATCH_SIZE",
    "FORMAL_EPOCHS",
    "FORMAL_EVAL_EVERY",
    "FORMAL_SEED",
    "PBDRV5RunContractError",
    "ROLLING_SCHEMA",
    "ROLES",
    "SCHEMA",
    "V5RunIdentity",
    "build_rolling_payload",
    "canonical_json_sha256",
    "epoch_selection_key",
    "json_selection_key",
    "ordered_strings_sha256",
    "require_sha256",
    "validate_rolling_payload",
]
