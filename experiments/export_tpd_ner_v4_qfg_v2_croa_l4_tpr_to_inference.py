"""Strict in-memory exporter for formal NER-L4-TPR training state.

The formal training graph contains 569 state entries.  Deployment removes
exactly the four training-only ``target_survival`` entries and strictly loads
the remaining 565 entries into the authoritative head-free graph.  In
particular, the learned NER-L4-TPR gate and all QFG state are required and
preserved unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch

from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_l4_tpr import (
    FORMAL_V4_QFG_V2_CROA_L4_TPR_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_L4_TPR_SURVIVAL_STATE_KEY_COUNT,
    L4_TPR_STATE_KEYS,
    L4_TPR_STATE_PREFIX,
    PRODUCTION_V4_QFG_V2_CROA_L4_TPR_INFERENCE_PARAMETERS,
    TPDNERV8MPRSDCHV4QFGV2CROAL4TPRInferenceSCTransNet,
    build_formal_v4_qfg_v2_croa_l4_tpr_inference_model,
    validate_formal_v4_qfg_v2_croa_l4_tpr_inference_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    QFG_STATE_KEYS,
    QFG_STATE_PREFIX,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)


EXPORT_SCHEMA = "sctransnet_tpd8_ner4_qfg2_l4_tpr_inference_state_v1"
TRAINING_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_L4_TPR_SURVIVAL_STATE_KEY_COUNT
)
INFERENCE_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_L4_TPR_INFERENCE_STATE_KEY_COUNT
)
INFERENCE_PARAMETER_COUNT = (
    PRODUCTION_V4_QFG_V2_CROA_L4_TPR_INFERENCE_PARAMETERS
)


def strip_l4_tpr_survival_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remove only the four TSS entries from an exact 569-key training state."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("NER-L4-TPR training state_dict must be a mapping")
    state = dict(state_dict)
    if len(state) != TRAINING_STATE_KEY_COUNT:
        raise ValueError(
            "formal NER-L4-TPR export requires exactly "
            f"{TRAINING_STATE_KEY_COUNT} training state keys"
        )
    for key, value in state.items():
        if not isinstance(key, str) or not key:
            raise TypeError(
                "NER-L4-TPR training state keys must be non-empty strings"
            )
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"NER-L4-TPR training state {key!r} must be a Tensor"
            )

    survival_keys = {
        key for key in state if key.startswith(SURVIVAL_STATE_PREFIX)
    }
    if survival_keys != set(SURVIVAL_STATE_KEYS):
        raise ValueError(
            "formal NER-L4-TPR export requires exactly four TSS keys"
        )
    qfg_keys = {key for key in state if key.startswith(QFG_STATE_PREFIX)}
    if qfg_keys != set(QFG_STATE_KEYS):
        raise ValueError(
            "formal NER-L4-TPR export requires exactly twenty QFG keys"
        )
    l4_tpr_keys = {
        key for key in state if key.startswith(L4_TPR_STATE_PREFIX)
    }
    if l4_tpr_keys != set(L4_TPR_STATE_KEYS):
        raise ValueError(
            "formal NER-L4-TPR export requires exactly one L4-TPR key"
        )

    inference_state = {
        key: value
        for key, value in state.items()
        if not key.startswith(SURVIVAL_STATE_PREFIX)
    }
    if len(inference_state) != INFERENCE_STATE_KEY_COUNT:
        raise ValueError("stripped NER-L4-TPR inference state count differs")
    if any(
        key.startswith(SURVIVAL_STATE_PREFIX) for key in inference_state
    ):
        raise ValueError("stripped NER-L4-TPR inference state retains TSS")
    if {
        key for key in inference_state if key.startswith(QFG_STATE_PREFIX)
    } != set(QFG_STATE_KEYS):
        raise ValueError("stripping TSS changed QFG state")
    if {
        key for key in inference_state if key.startswith(L4_TPR_STATE_PREFIX)
    } != set(L4_TPR_STATE_KEYS):
        raise ValueError("stripping TSS changed NER-L4-TPR state")
    return inference_state


def build_l4_tpr_inference_model_from_training_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    seed: int = 42,
) -> Tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAL4TPRInferenceSCTransNet,
    Dict[str, Any],
]:
    """Strictly load an exact training state into the head-free graph."""

    inference_state = strip_l4_tpr_survival_state_dict(state_dict)
    model, metadata = build_formal_v4_qfg_v2_croa_l4_tpr_inference_model(
        seed=seed
    )
    if set(inference_state) != set(model.state_dict()):
        raise ValueError(
            "stripped state does not match NER-L4-TPR inference graph"
        )
    incompatible = model.load_state_dict(inference_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "strict NER-L4-TPR inference load returned incompatible keys"
        )
    model.eval()
    validated = validate_formal_v4_qfg_v2_croa_l4_tpr_inference_model(model)
    if validated["state_key_count"] != INFERENCE_STATE_KEY_COUNT:
        raise RuntimeError(
            "validated NER-L4-TPR inference state count differs"
        )
    return model, metadata


def export_l4_tpr_training_payload_to_inference(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a machine-readable head-free payload without file I/O."""

    if not isinstance(payload, Mapping):
        raise TypeError(
            "NER-L4-TPR training checkpoint payload must be a mapping"
        )
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError(
            "NER-L4-TPR training checkpoint lacks top-level state_dict"
        )
    inference_state = strip_l4_tpr_survival_state_dict(state_dict)
    # Construction and strict loading prove that only TSS state was removed.
    build_l4_tpr_inference_model_from_training_state_dict(state_dict)
    return {
        "schema": EXPORT_SCHEMA,
        "state_dict": inference_state,
        "source_checkpoint_role": payload.get("checkpoint_role"),
        "source_checkpoint_identity": payload.get("checkpoint_identity"),
        "source_run_identity": payload.get("run_identity"),
        "survival_state_removed": SURVIVAL_STATE_KEYS,
        "qfg_state_preserved": QFG_STATE_KEYS,
        "l4_tpr_state_preserved": L4_TPR_STATE_KEYS,
        "l4_tpr_inference_required": True,
        "inference_heads_required": False,
        "inference_state_key_count": INFERENCE_STATE_KEY_COUNT,
        "inference_parameter_count": INFERENCE_PARAMETER_COUNT,
    }


__all__ = [
    "EXPORT_SCHEMA",
    "INFERENCE_PARAMETER_COUNT",
    "INFERENCE_STATE_KEY_COUNT",
    "TRAINING_STATE_KEY_COUNT",
    "build_l4_tpr_inference_model_from_training_state_dict",
    "export_l4_tpr_training_payload_to_inference",
    "strip_l4_tpr_survival_state_dict",
]
