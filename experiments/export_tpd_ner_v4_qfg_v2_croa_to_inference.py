#!/usr/bin/env python3
"""Export a formal QFG checkpoint to the strict head-free inference graph."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (  # noqa: E402
    FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
    PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS,
    QFG_STATE_KEYS,
    QFG_STATE_PREFIX,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
    build_formal_v4_qfg_v2_croa_inference_model,
    validate_formal_qfg_v2_croa_inference_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (  # noqa: E402
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)


EXPORT_SCHEMA = "sctransnet_tpd_ner_v4_qfg_v2_croa_inference_export_v1"
TRAINING_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT
)
INFERENCE_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT
)
INFERENCE_PARAMETER_COUNT = (
    PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS
)


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _validate_inference_model(
    model: torch.nn.Module,
    metadata: Mapping[str, Any],
) -> None:
    if type(model) is not TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet:
        raise TypeError("QFG export requires the exact head-free class")
    if hasattr(model, "target_survival"):
        raise RuntimeError("QFG inference model retains target_survival")
    state = model.state_dict()
    if len(state) != INFERENCE_STATE_KEY_COUNT:
        raise RuntimeError("QFG inference state-key count differs")
    if _parameter_count(model) != INFERENCE_PARAMETER_COUNT:
        raise RuntimeError("QFG inference parameter count differs")
    if metadata.get("state_key_count") != INFERENCE_STATE_KEY_COUNT:
        raise RuntimeError("QFG inference metadata state-key count differs")
    if metadata.get("total_parameters") != INFERENCE_PARAMETER_COUNT:
        raise RuntimeError("QFG inference metadata parameter count differs")
    qfg_keys = {
        key for key in state if key.startswith(QFG_STATE_PREFIX)
    }
    if qfg_keys != set(QFG_STATE_KEYS):
        raise RuntimeError("QFG inference model does not retain exact QFG state")
    if any(key.startswith(SURVIVAL_STATE_PREFIX) for key in state):
        raise RuntimeError("QFG inference model contains Survival state")
    validate_formal_qfg_v2_croa_inference_model(model)


def strip_survival_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Delete exactly four TSS keys while preserving every QFG tensor."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("QFG training state_dict must be a mapping")
    state = dict(state_dict)
    if len(state) != TRAINING_STATE_KEY_COUNT:
        raise ValueError(
            "formal QFG export requires exactly "
            f"{TRAINING_STATE_KEY_COUNT} training state keys"
        )
    for key, value in state.items():
        if not isinstance(key, str) or not key:
            raise TypeError("QFG training state keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"QFG training state {key!r} must be a Tensor")

    survival_keys = {
        key for key in state if key.startswith(SURVIVAL_STATE_PREFIX)
    }
    if survival_keys != set(SURVIVAL_STATE_KEYS):
        raise ValueError("formal QFG export requires exactly four TSS keys")
    qfg_keys = {
        key for key in state if key.startswith(QFG_STATE_PREFIX)
    }
    if qfg_keys != set(QFG_STATE_KEYS):
        raise ValueError("formal QFG export requires exactly twenty QFG keys")

    inference_state = {
        key: value
        for key, value in state.items()
        if not key.startswith(SURVIVAL_STATE_PREFIX)
    }
    if len(inference_state) != INFERENCE_STATE_KEY_COUNT:
        raise ValueError("stripped QFG inference state-key count differs")
    if {
        key for key in inference_state if key.startswith(QFG_STATE_PREFIX)
    } != set(QFG_STATE_KEYS):
        raise ValueError("stripping Survival state changed QFG state keys")
    if any(
        key.startswith(SURVIVAL_STATE_PREFIX)
        for key in inference_state
    ):
        raise ValueError("stripped inference state retains Survival keys")
    return inference_state


def build_frozen_qfg_inference_model(
    seed: int = 42,
) -> tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
    dict[str, Any],
]:
    model, metadata = build_formal_v4_qfg_v2_croa_inference_model(seed=seed)
    _validate_inference_model(model, metadata)
    return model, metadata


def build_inference_model_from_training_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    seed: int = 42,
) -> tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
    dict[str, Any],
]:
    inference_state = strip_survival_state_dict(state_dict)
    model, metadata = build_frozen_qfg_inference_model(seed=seed)
    if set(inference_state) != set(model.state_dict()):
        raise ValueError(
            "stripped state does not exactly match the QFG inference graph"
        )
    incompatible = model.load_state_dict(inference_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict QFG inference load returned incompatible keys")
    model.eval()
    _validate_inference_model(model, metadata)
    return model, metadata


def _regular_file_bytes(path: Path) -> bytes:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"checkpoint is not a regular file: {path}")
    return path.read_bytes()


def require_formal_qfg_checkpoint_payload(payload: Any) -> dict[str, Any]:
    """Validate trainer-owned run/checkpoint identity before export."""

    try:
        from experiments import train_tpd_ner_v4_qfg_v2_croa_exact as trainer
    except ImportError as exc:
        raise RuntimeError("formal QFG trainer is unavailable") from exc

    validator = getattr(
        trainer,
        "require_evaluator_checkpoint_payload",
        None,
    )
    if callable(validator):
        validated = validator(payload)
    else:
        adapter_type = getattr(trainer, "EvaluatorCheckpointAdapter", None)
        if adapter_type is None:
            raise RuntimeError(
                "QFG trainer exposes no evaluator checkpoint validator"
            )
        adapter = adapter_type(payload)
        validate = getattr(adapter, "validate", None)
        if callable(validate):
            validated = validate()
        elif isinstance(adapter, Mapping):
            validated = dict(adapter)
        else:
            raise RuntimeError(
                "QFG trainer checkpoint adapter has no validate() method"
            )
    if not isinstance(validated, Mapping):
        raise ValueError("QFG trainer checkpoint validator returned non-mapping")
    ready = copy.deepcopy(dict(validated))
    if not isinstance(ready.get("state_dict"), Mapping):
        raise ValueError("formal QFG checkpoint lacks top-level state_dict")
    return ready


def export_qfg_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write one strict, write-once, head-free QFG deployment checkpoint."""

    content = _regular_file_bytes(checkpoint_path)
    payload = torch.load(
        io.BytesIO(content),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("QFG checkpoint payload must be a mapping")
    validated = require_formal_qfg_checkpoint_payload(payload)
    state_dict = validated.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("QFG checkpoint lacks top-level state_dict")
    inference_state = strip_survival_state_dict(state_dict)

    # Strict construction/load proves that filtering removed only TSS state.
    build_inference_model_from_training_state_dict(state_dict)
    output_path = Path(output_path).resolve()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite export: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.parent.is_symlink() or not output_path.parent.is_dir():
        raise NotADirectoryError(output_path.parent)
    exported = {
        "schema": EXPORT_SCHEMA,
        "source_checkpoint_path": str(Path(checkpoint_path).resolve()),
        "source_checkpoint_sha256": hashlib.sha256(content).hexdigest(),
        "source_checkpoint_role": validated.get("checkpoint_role"),
        "source_checkpoint_identity": copy.deepcopy(
            validated.get("checkpoint_identity")
        ),
        "source_run_identity": copy.deepcopy(validated.get("run_identity")),
        "state_dict": inference_state,
        "survival_state_removed": list(SURVIVAL_STATE_KEYS),
        "qfg_state_preserved": list(QFG_STATE_KEYS),
        "qfg_inference_required": True,
        "inference_heads_required": False,
        "inference_state_key_count": INFERENCE_STATE_KEY_COUNT,
        "inference_parameter_count": INFERENCE_PARAMETER_COUNT,
    }
    serialized = io.BytesIO()
    torch.save(exported, serialized)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output_file:
            output_file.write(serialized.getbuffer())
            output_file.flush()
            os.fsync(output_file.fileno())
        try:
            os.link(temporary, output_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite export: {output_path}"
            ) from error
        directory_descriptor = os.open(str(output_path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return exported


def validate_exported_qfg_checkpoint(
    output_path: Path,
    *,
    expected_source_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Strictly re-open and validate a head-free deployment checkpoint."""

    content = _regular_file_bytes(output_path)
    payload = torch.load(
        io.BytesIO(content),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("QFG inference export payload must be a mapping")
    ready = copy.deepcopy(dict(payload))
    if ready.get("schema") != EXPORT_SCHEMA:
        raise ValueError("QFG inference export schema differs")
    state_dict = ready.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("QFG inference export lacks state_dict")
    if len(state_dict) != INFERENCE_STATE_KEY_COUNT:
        raise ValueError("QFG inference export state-key count differs")
    if any(key.startswith(SURVIVAL_STATE_PREFIX) for key in state_dict):
        raise ValueError("QFG inference export retains Survival state")
    if {
        key for key in state_dict if key.startswith(QFG_STATE_PREFIX)
    } != set(QFG_STATE_KEYS):
        raise ValueError("QFG inference export QFG state differs")
    model, metadata = build_frozen_qfg_inference_model()
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("QFG inference export strict load is incompatible")
    _validate_inference_model(model, metadata)
    if ready.get("inference_state_key_count") != INFERENCE_STATE_KEY_COUNT:
        raise ValueError("QFG inference export metadata state count differs")
    if ready.get("inference_parameter_count") != INFERENCE_PARAMETER_COUNT:
        raise ValueError("QFG inference export parameter count differs")
    source_path = Path(str(ready.get("source_checkpoint_path"))).resolve()
    source_sha256 = ready.get("source_checkpoint_sha256")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("QFG inference export source SHA is invalid")
    if expected_source_checkpoint is not None:
        expected = Path(expected_source_checkpoint).resolve()
        if source_path != expected:
            raise ValueError("QFG inference export source path differs")
        if hashlib.sha256(_regular_file_bytes(expected)).hexdigest() != source_sha256:
            raise ValueError("QFG inference export source content differs")
    return {
        "schema": EXPORT_SCHEMA,
        "path": str(Path(output_path).resolve()),
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_checkpoint_path": str(source_path),
        "source_checkpoint_sha256": source_sha256,
        "source_checkpoint_role": ready.get("source_checkpoint_role"),
        "inference_state_key_count": INFERENCE_STATE_KEY_COUNT,
        "inference_parameter_count": INFERENCE_PARAMETER_COUNT,
        "strict_load": True,
        "survival_state_absent": True,
        "qfg_state_preserved": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    export_qfg_checkpoint(
        args.checkpoint.resolve(),
        args.output.resolve(),
    )


__all__ = [
    "EXPORT_SCHEMA",
    "INFERENCE_PARAMETER_COUNT",
    "INFERENCE_STATE_KEY_COUNT",
    "TRAINING_STATE_KEY_COUNT",
    "build_frozen_qfg_inference_model",
    "build_inference_model_from_training_state_dict",
    "export_qfg_checkpoint",
    "main",
    "parse_args",
    "require_formal_qfg_checkpoint_payload",
    "strip_survival_state_dict",
    "validate_exported_qfg_checkpoint",
]


if __name__ == "__main__":
    main()
