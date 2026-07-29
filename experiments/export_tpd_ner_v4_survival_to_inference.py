#!/usr/bin/env python3
"""Strip training-only Survival state into a strict V4 inference checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    PRODUCTION_V4_RELAY_ON_PARAMETERS,
    TPDNERV8MPRSDCHV4SCTransNet,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    FORMAL_V4_PARENT_STATE_KEY_COUNT,
    FORMAL_V4_SURVIVAL_STATE_KEY_COUNT,
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
    build_formal_v4_reference,
)


EXPORT_SCHEMA = "sctransnet_tpd_ner_v4_survival_inference_export_v1"
INFERENCE_PARAMETER_COUNT = PRODUCTION_V4_RELAY_ON_PARAMETERS
INFERENCE_STATE_KEY_COUNT = FORMAL_V4_PARENT_STATE_KEY_COUNT


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _validate_frozen_v4_model(
    model: torch.nn.Module,
    metadata: Mapping[str, Any],
) -> None:
    """Reject any deployment architecture other than the formal V4 graph."""

    if type(model) is not TPDNERV8MPRSDCHV4SCTransNet:
        raise TypeError("inference export requires the exact formal V4 class")
    state = model.state_dict()
    if len(state) != INFERENCE_STATE_KEY_COUNT:
        raise RuntimeError(
            "formal V4 inference state-key count differs: "
            f"{len(state)} != {INFERENCE_STATE_KEY_COUNT}"
        )
    parameters = _parameter_count(model)
    if parameters != INFERENCE_PARAMETER_COUNT:
        raise RuntimeError(
            "formal V4 inference parameter count differs: "
            f"{parameters} != {INFERENCE_PARAMETER_COUNT}"
        )
    if metadata.get("state_key_count") != INFERENCE_STATE_KEY_COUNT:
        raise RuntimeError("formal V4 builder state-key metadata differs")
    if metadata.get("total_parameters") != INFERENCE_PARAMETER_COUNT:
        raise RuntimeError("formal V4 builder parameter metadata differs")


def strip_survival_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return exactly the V4 parent state after validating four TSS keys."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("Survival state_dict must be a mapping")
    state = dict(state_dict)
    if len(state) != FORMAL_V4_SURVIVAL_STATE_KEY_COUNT:
        raise ValueError(
            "formal Survival export requires exactly "
            f"{FORMAL_V4_SURVIVAL_STATE_KEY_COUNT} state keys"
        )
    for key, value in state.items():
        if not isinstance(key, str) or not key:
            raise TypeError("Survival state keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Survival state {key!r} must be a Tensor")

    survival_keys = {
        key for key in state if key.startswith(SURVIVAL_STATE_PREFIX)
    }
    if survival_keys != set(SURVIVAL_STATE_KEYS):
        raise ValueError("formal Survival export requires exactly four head keys")
    inference_state = {
        key: value
        for key, value in state.items()
        if not key.startswith(SURVIVAL_STATE_PREFIX)
    }
    if len(inference_state) != FORMAL_V4_PARENT_STATE_KEY_COUNT:
        raise ValueError("stripped inference state does not match formal V4")
    return inference_state


def build_frozen_v4_model(
    seed: int = 42,
) -> tuple[TPDNERV8MPRSDCHV4SCTransNet, dict[str, Any]]:
    """Build the authoritative head-free V4 deployment architecture."""

    model, metadata = build_formal_v4_reference(seed=seed)
    _validate_frozen_v4_model(model, metadata)
    return model, metadata


def build_inference_model_from_survival_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    seed: int = 42,
) -> tuple[TPDNERV8MPRSDCHV4SCTransNet, dict[str, Any]]:
    inference_state = strip_survival_state_dict(state_dict)
    model, metadata = build_frozen_v4_model(seed=seed)
    if set(inference_state) != set(model.state_dict()):
        raise ValueError(
            "stripped inference state keys do not exactly match formal V4"
        )
    incompatible = model.load_state_dict(inference_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict V4 export load returned incompatible state keys"
        )
    model.eval()
    _validate_frozen_v4_model(model, metadata)
    return model, metadata


def _regular_file_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"checkpoint is not a regular file: {path}")
    return path.read_bytes()


def require_formal_survival_checkpoint_payload(
    payload: Any,
) -> dict[str, Any]:
    """Validate the complete formal TSS identity before deployment export."""

    # Lazy import keeps state-dict-only model helpers independent from the
    # training entry while making the file-export boundary strict.
    from experiments import (
        train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact as trainer,
    )

    return trainer.require_evaluator_checkpoint_payload(payload)


def export_survival_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write a state-only V4 deployment checkpoint from a TSS checkpoint."""

    content = _regular_file_bytes(checkpoint_path)
    payload = torch.load(
        io.BytesIO(content),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Survival checkpoint payload must be a mapping")
    validated = require_formal_survival_checkpoint_payload(payload)
    state_dict = validated.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Survival checkpoint lacks top-level state_dict")
    inference_state = strip_survival_state_dict(state_dict)

    # A strict load proves that filtering did not omit or retain foreign state.
    build_inference_model_from_survival_state_dict(state_dict)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite export: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = {
        "schema": EXPORT_SCHEMA,
        "source_checkpoint_path": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": hashlib.sha256(content).hexdigest(),
        "state_dict": inference_state,
        "survival_state_removed": list(SURVIVAL_STATE_KEYS),
        "inference_heads_required": False,
        "inference_state_key_count": INFERENCE_STATE_KEY_COUNT,
        "inference_parameter_count": INFERENCE_PARAMETER_COUNT,
    }
    serialized = io.BytesIO()
    torch.save(exported, serialized)
    with output_path.open("xb") as output_file:
        output_file.write(serialized.getbuffer())
    return exported


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    export_survival_checkpoint(
        args.checkpoint.resolve(),
        args.output.resolve(),
    )


__all__ = [
    "EXPORT_SCHEMA",
    "INFERENCE_PARAMETER_COUNT",
    "INFERENCE_STATE_KEY_COUNT",
    "build_frozen_v4_model",
    "build_inference_model_from_survival_state_dict",
    "export_survival_checkpoint",
    "main",
    "parse_args",
    "require_formal_survival_checkpoint_payload",
    "strip_survival_state_dict",
]


if __name__ == "__main__":
    main()
