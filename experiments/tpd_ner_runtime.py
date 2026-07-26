"""Runtime guards used only by the isolated TPD-NER training entry."""

from __future__ import annotations

import contextlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator

import torch
import torch.nn as nn

from experiments import train_tpd_pilot as base


_ORIGINAL_DEEP_SUPERVISION_LOSS = base.deep_supervision_loss
_ORIGINAL_VALIDATE = base.validate
_ORIGINAL_TORCH_SAVE = torch.save


def checked_deep_supervision_loss(
    outputs: Any,
    target: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Reject a non-finite training objective before backward/optimizer.step."""

    loss = _ORIGINAL_DEEP_SUPERVISION_LOSS(outputs, target, criterion)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError("TPD-NER training loss is non-finite")
    return loss


def checked_validate(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Reject non-finite validation metrics before checkpoint selection."""

    metrics = _ORIGINAL_VALIDATE(*args, **kwargs)
    for name, value in metrics.items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
        ):
            if (
                name == "tiny_pd"
                and math.isnan(float(value))
                and metrics.get("tiny_target_count") == 0
            ):
                continue
            raise FloatingPointError(
                f"TPD-NER validation metric {name!r} is non-finite: {value!r}"
            )
    return metrics


def atomic_torch_save(
    payload: Any,
    destination: Any,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Write path-based checkpoints by atomic same-directory replacement."""

    if not isinstance(destination, (str, os.PathLike)):
        _ORIGINAL_TORCH_SAVE(payload, destination, *args, **kwargs)
        return
    destination_path = Path(destination)
    if destination_path.is_symlink():
        raise RuntimeError(
            f"refusing to replace checkpoint symlink: {destination_path}"
        )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.tmp-",
        dir=destination_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        _ORIGINAL_TORCH_SAVE(payload, temporary_path, *args, **kwargs)
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


@contextlib.contextmanager
def guarded_training_runtime() -> Iterator[None]:
    """Install NER-only finite-value and atomic-checkpoint guards."""

    previous_loss = base.deep_supervision_loss
    previous_validate = base.validate
    previous_save = torch.save
    base.deep_supervision_loss = checked_deep_supervision_loss
    base.validate = checked_validate
    torch.save = atomic_torch_save
    try:
        yield
    finally:
        base.deep_supervision_loss = previous_loss
        base.validate = previous_validate
        torch.save = previous_save
