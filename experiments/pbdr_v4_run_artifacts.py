"""Fail-closed append-only and rolling artifacts for PBDR-V4 training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import torch

from experiments.pbdr_v4_state_contract import state_semantic_sha256


RUN_ARTIFACT_SCHEMA = "sctransnet_pbdr_v4_run_artifacts/v1"


class PBDRV4ArtifactError(RuntimeError):
    """An artifact path, payload, or resume binding is unsafe or inconsistent."""


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PBDRV4ArtifactError(f"{name} must be lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class RunIdentity:
    dataset: str
    role: str
    stage: str
    source_lock_sha256: str
    split_projection_sha256: str
    atlas_manifest_sha256: str
    parent_checkpoint_sha256: str
    parent_state_sha256: str
    initialization_checkpoint_sha256: str | None

    def __post_init__(self) -> None:
        if self.dataset not in ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K"):
            raise PBDRV4ArtifactError("unsupported dataset")
        if self.role not in ("best_miou", "best_pd"):
            raise PBDRV4ArtifactError("unsupported role")
        if self.stage not in ("stage1", "stage2"):
            raise PBDRV4ArtifactError("unsupported stage")
        for field in (
            "source_lock_sha256",
            "split_projection_sha256",
            "atlas_manifest_sha256",
            "parent_checkpoint_sha256",
            "parent_state_sha256",
        ):
            _sha256(getattr(self, field), name=field)
        if self.stage == "stage1":
            if self.initialization_checkpoint_sha256 is not None:
                raise PBDRV4ArtifactError(
                    "Stage-1 cannot bind a Stage-1 initialization checkpoint"
                )
        else:
            _sha256(
                self.initialization_checkpoint_sha256,
                name="initialization_checkpoint_sha256",
            )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def file_sha256(path: Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PBDRV4ArtifactError(f"artifact is not a regular file: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_parent(path: Path) -> Path:
    candidate = Path(path)
    parent = candidate.parent
    if parent.is_symlink():
        raise PBDRV4ArtifactError(f"artifact parent is a symlink: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    resolved = parent.resolve(strict=True)
    if candidate.name in ("", ".", ".."):
        raise PBDRV4ArtifactError("artifact filename is unsafe")
    return resolved / candidate.name


def _reject_existing_or_symlink(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise PBDRV4ArtifactError(f"exclusive artifact already exists: {path}")


def exclusive_json(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = _safe_parent(path)
    _reject_existing_or_symlink(destination)
    try:
        content = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise PBDRV4ArtifactError(f"JSON payload is not canonicalizable: {error}") from error
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError as error:
        raise PBDRV4ArtifactError(f"exclusive artifact already exists: {destination}") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return destination


def exclusive_torch_save(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = _safe_parent(path)
    _reject_existing_or_symlink(destination)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError as error:
        raise PBDRV4ArtifactError(f"exclusive artifact already exists: {destination}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # A partial exclusive artifact is intentionally retained.  Reusing the
        # same formal path after an interrupted append-only commit is unsafe.
        raise
    return destination


def atomic_rolling_torch_save(path: Path, payload: Mapping[str, Any]) -> Path:
    destination = _safe_parent(path)
    if destination.is_symlink():
        raise PBDRV4ArtifactError("rolling artifact destination is a symlink")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_torch_artifact(path: Path) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise PBDRV4ArtifactError(f"artifact is missing or unsafe: {candidate}")
    try:
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
    except Exception as error:
        raise PBDRV4ArtifactError(f"cannot load artifact {candidate}: {error}") from error
    if not isinstance(payload, Mapping) or not all(isinstance(key, str) for key in payload):
        raise PBDRV4ArtifactError("torch artifact must contain a string-key mapping")
    return dict(payload)


def optimizer_group_signature(optimizer_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = optimizer_state.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise PBDRV4ArtifactError("optimizer state has no parameter groups")
    signature: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            raise PBDRV4ArtifactError(f"optimizer group {index} is not a mapping")
        parameters = group.get("params")
        if not isinstance(parameters, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in parameters
        ):
            raise PBDRV4ArtifactError(f"optimizer group {index} params differ")
        signature.append(
            {
                "index": index,
                "parameter_count": len(parameters),
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "betas": [float(value) for value in group["betas"]],
                "eps": float(group["eps"]),
                "amsgrad": bool(group["amsgrad"]),
                "maximize": bool(group["maximize"]),
            }
        )
    return signature


def checkpoint_payload(
    *,
    identity: RunIdentity,
    epoch: int,
    epochs: int,
    model_state: Mapping[str, torch.Tensor],
    optimizer_state: Mapping[str, Any],
    rng_state: Mapping[str, Any],
    selected: Mapping[str, Any],
    event: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or isinstance(epochs, bool)
        or not isinstance(epochs, int)
        or not 1 <= epoch <= epochs
    ):
        raise PBDRV4ArtifactError("checkpoint epoch is outside the run budget")
    state = {
        name: value.detach().cpu().clone() for name, value in model_state.items()
    }
    if not state or not all(isinstance(name, str) for name in state):
        raise PBDRV4ArtifactError("model state is empty or malformed")
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(rng_state) != required_rng:
        raise PBDRV4ArtifactError("RNG state keys differ")
    optimizer_signature = optimizer_group_signature(optimizer_state)
    return {
        "schema": RUN_ARTIFACT_SCHEMA,
        "identity": identity.as_dict(),
        "epoch": epoch,
        "epochs": epochs,
        "state_dict": state,
        "state_sha256": state_semantic_sha256(state),
        "optimizer": dict(optimizer_state),
        "optimizer_group_signature": optimizer_signature,
        "rng_state": dict(rng_state),
        "selected": dict(selected),
        "event": dict(event),
    }


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    identity: RunIdentity,
    epochs: int,
    expected_optimizer_group_signature: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if payload.get("schema") != RUN_ARTIFACT_SCHEMA:
        raise PBDRV4ArtifactError("checkpoint schema differs")
    if payload.get("identity") != identity.as_dict():
        raise PBDRV4ArtifactError("checkpoint run identity differs")
    epoch = payload.get("epoch")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or payload.get("epochs") != epochs
        or not 1 <= epoch <= epochs
    ):
        raise PBDRV4ArtifactError("checkpoint epoch/budget differs")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise PBDRV4ArtifactError("checkpoint model state differs")
    observed_state_sha = state_semantic_sha256(state)  # type: ignore[arg-type]
    if payload.get("state_sha256") != observed_state_sha:
        raise PBDRV4ArtifactError("checkpoint state SHA-256 differs")
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise PBDRV4ArtifactError("checkpoint optimizer state differs")
    observed_signature = optimizer_group_signature(optimizer)
    if payload.get("optimizer_group_signature") != observed_signature:
        raise PBDRV4ArtifactError("stored optimizer group signature differs")
    if (
        expected_optimizer_group_signature is not None
        and observed_signature != expected_optimizer_group_signature
    ):
        raise PBDRV4ArtifactError("optimizer parameter-group structure differs")
    rng = payload.get("rng_state")
    if not isinstance(rng, Mapping) or set(rng) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise PBDRV4ArtifactError("checkpoint RNG state differs")
    if not isinstance(payload.get("selected"), Mapping) or not isinstance(
        payload.get("event"), Mapping
    ):
        raise PBDRV4ArtifactError("checkpoint selection/event differs")
    return dict(payload)


def epoch_checkpoint_path(run_dir: Path, epoch: int) -> Path:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise PBDRV4ArtifactError("epoch must be positive")
    return Path(run_dir) / "checkpoints" / f"epoch_{epoch:05d}.pth.tar"


__all__ = [
    "PBDRV4ArtifactError",
    "RUN_ARTIFACT_SCHEMA",
    "RunIdentity",
    "atomic_rolling_torch_save",
    "checkpoint_payload",
    "epoch_checkpoint_path",
    "exclusive_json",
    "exclusive_torch_save",
    "file_sha256",
    "load_torch_artifact",
    "optimizer_group_signature",
    "validate_checkpoint_payload",
]
