"""Strict in-memory exporter for formal GCSF training state.

The exporter removes exactly the four training-only ``target_survival`` keys.
All four GCSF tensors and all twenty QFG tensors remain required and are
strictly loaded into the authoritative head-free graph.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch

from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_gcsf import (
    FORMAL_V4_QFG_V2_CROA_GCSF_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_GCSF_SURVIVAL_STATE_KEY_COUNT,
    GCSF_STATE_KEYS,
    GCSF_STATE_PREFIX,
    PRODUCTION_V4_QFG_V2_CROA_GCSF_INFERENCE_PARAMETERS,
    TPDNERV8MPRSDCHV4QFGV2CROAGCSFInferenceSCTransNet,
    build_formal_v4_qfg_v2_croa_gcsf_inference_model,
    validate_formal_v4_qfg_v2_croa_gcsf_inference_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    QFG_STATE_KEYS,
    QFG_STATE_PREFIX,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)


EXPORT_SCHEMA = "sctransnet_tpd8_ner4_qfg2_gcsf_inference_state_v1"
TRAINING_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_GCSF_SURVIVAL_STATE_KEY_COUNT
)
INFERENCE_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_GCSF_INFERENCE_STATE_KEY_COUNT
)
INFERENCE_PARAMETER_COUNT = (
    PRODUCTION_V4_QFG_V2_CROA_GCSF_INFERENCE_PARAMETERS
)


def strip_gcsf_survival_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remove exactly four TSS tensors and preserve exact QFG/GCSF state."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("GCSF training state_dict must be a mapping")
    state = dict(state_dict)
    if len(state) != TRAINING_STATE_KEY_COUNT:
        raise ValueError(
            "formal GCSF export requires exactly "
            f"{TRAINING_STATE_KEY_COUNT} training state keys"
        )
    for key, value in state.items():
        if not isinstance(key, str) or not key:
            raise TypeError("GCSF training state keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"GCSF training state {key!r} must be a Tensor")

    survival_keys = {
        key for key in state if key.startswith(SURVIVAL_STATE_PREFIX)
    }
    if survival_keys != set(SURVIVAL_STATE_KEYS):
        raise ValueError("formal GCSF export requires exactly four TSS keys")
    qfg_keys = {key for key in state if key.startswith(QFG_STATE_PREFIX)}
    if qfg_keys != set(QFG_STATE_KEYS):
        raise ValueError("formal GCSF export requires exactly twenty QFG keys")
    gcsf_keys = {key for key in state if key.startswith(GCSF_STATE_PREFIX)}
    if gcsf_keys != set(GCSF_STATE_KEYS):
        raise ValueError("formal GCSF export requires exactly four GCSF keys")

    inference_state = {
        key: value
        for key, value in state.items()
        if not key.startswith(SURVIVAL_STATE_PREFIX)
    }
    if len(inference_state) != INFERENCE_STATE_KEY_COUNT:
        raise ValueError("stripped GCSF inference state-key count differs")
    if any(
        key.startswith(SURVIVAL_STATE_PREFIX) for key in inference_state
    ):
        raise ValueError("stripped GCSF inference state retains TSS keys")
    if {
        key for key in inference_state if key.startswith(QFG_STATE_PREFIX)
    } != set(QFG_STATE_KEYS):
        raise ValueError("stripping TSS changed QFG state")
    if {
        key for key in inference_state if key.startswith(GCSF_STATE_PREFIX)
    } != set(GCSF_STATE_KEYS):
        raise ValueError("stripping TSS changed GCSF state")
    return inference_state


def build_gcsf_inference_model_from_training_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    seed: int = 42,
) -> Tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAGCSFInferenceSCTransNet,
    Dict[str, Any],
]:
    """Strictly export and load training state into the head-free graph."""

    inference_state = strip_gcsf_survival_state_dict(state_dict)
    model, metadata = build_formal_v4_qfg_v2_croa_gcsf_inference_model(
        seed=seed
    )
    if set(inference_state) != set(model.state_dict()):
        raise ValueError("stripped state does not match GCSF inference graph")
    incompatible = model.load_state_dict(inference_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict GCSF inference load returned incompatible keys")
    model.eval()
    validated = validate_formal_v4_qfg_v2_croa_gcsf_inference_model(model)
    if validated["state_key_count"] != INFERENCE_STATE_KEY_COUNT:
        raise RuntimeError("validated GCSF inference state count differs")
    return model, metadata


def export_gcsf_training_payload_to_inference(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a strict machine-readable head-free payload without file I/O."""

    if not isinstance(payload, Mapping):
        raise TypeError("GCSF training checkpoint payload must be a mapping")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("GCSF training checkpoint lacks top-level state_dict")
    inference_state = strip_gcsf_survival_state_dict(state_dict)
    # Construction plus strict load proves that filtering removed only heads.
    build_gcsf_inference_model_from_training_state_dict(state_dict)
    return {
        "schema": EXPORT_SCHEMA,
        "state_dict": inference_state,
        "source_checkpoint_role": payload.get("checkpoint_role"),
        "source_checkpoint_identity": payload.get("checkpoint_identity"),
        "source_run_identity": payload.get("run_identity"),
        "survival_state_removed": SURVIVAL_STATE_KEYS,
        "qfg_state_preserved": QFG_STATE_KEYS,
        "gcsf_state_preserved": GCSF_STATE_KEYS,
        "gcsf_inference_required": True,
        "inference_heads_required": False,
        "inference_state_key_count": INFERENCE_STATE_KEY_COUNT,
        "inference_parameter_count": INFERENCE_PARAMETER_COUNT,
    }


__all__ = [
    "EXPORT_SCHEMA",
    "INFERENCE_PARAMETER_COUNT",
    "INFERENCE_STATE_KEY_COUNT",
    "TRAINING_STATE_KEY_COUNT",
    "build_gcsf_inference_model_from_training_state_dict",
    "export_gcsf_training_payload_to_inference",
    "strip_gcsf_survival_state_dict",
]
