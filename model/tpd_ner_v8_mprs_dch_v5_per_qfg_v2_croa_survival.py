"""Formal NER V5-PER integration with QFG2-CROA and training-only TSS heads.

The module inherits the frozen V4 QFG/Survival data flow and replaces exactly
``self.tpd_ner`` with the state-layout-compatible V5-PER relay.  Separate
training and head-free inference classes prevent a V5 checkpoint from being
silently interpreted with V4 forward semantics.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn

from model.SCTransNet import SCTransNet
from model.tpd_frequency_gate_v2_croa import (
    QueryOnlyFrequencyGateV2CROA,
    validate_formal_qfg_v2_croa,
)
from model.tpd_ner_v8_mprs_dch import (
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    DEFAULT_TAIL_Z_THRESHOLDS,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    FORMAL_QFG_ALPHA_EFFECTIVE_INIT,
    FORMAL_QFG_DETACH_FREQUENCY_SOURCE,
    FORMAL_QFG_FEATURE_CHANNELS,
    FORMAL_QFG_HIDDEN_CHANNELS,
    FORMAL_QFG_INITIALIZATION_SEED,
    FORMAL_QFG_MODE,
    FORMAL_QFG_VALIDATE_FINITE,
    PRODUCTION_QFG_V2_CROA_PARAMETERS,
    PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT,
    PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS,
    PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS,
    QFG_STATE_KEYS,
    QFG_STATE_PREFIX,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    FORMAL_SURVIVAL_INITIALIZATION_SEED,
    FORMAL_SURVIVAL_VARIANT,
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)
from model.tpd_ner_v8_mprs_dch_v5_per import (
    PersistentEvidencePositiveRoutingRelay,
    V5_PER_FORMAL_DC_SUPPORT_MODE,
    V5_PER_RELAY_VERSION,
    replace_v4_relay_with_v5,
    v5_per_manifest_fields,
)
from model.tpd_survival import survival_parameter_count


V5_PER_QFG_V2_CROA_INTEGRATION_VERSION = "v5_per_survival_qfg_v2_croa_v1"

# V5 changes computation only.  Named tensor layouts and parameter counts are
# deliberately identical to the corresponding V4 graphs.
FORMAL_V5_PER_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT = 568
FORMAL_V5_PER_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT = 564
PRODUCTION_V5_PER_QFG_V2_CROA_SURVIVAL_PARAMETERS = (
    PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS
)
PRODUCTION_V5_PER_QFG_V2_CROA_INFERENCE_PARAMETERS = (
    PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS
)
PRODUCTION_V5_PER_SURVIVAL_PARAMETERS = 98


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _require_formal_seed(seed: int) -> int:
    if type(seed) is not int or seed != FORMAL_SURVIVAL_INITIALIZATION_SEED:
        raise ValueError("formal NER V5-PER construction requires seed=42")
    return seed


def _install_v5_per_relay(model: nn.Module) -> None:
    old_relay = getattr(model, "tpd_ner", None)
    if old_relay is None:
        raise RuntimeError("V5-PER integration requires the inherited V4 relay")
    replacement = replace_v4_relay_with_v5(old_relay)
    # Assignment replaces the existing Module key in place, retaining the
    # complete model's parameter traversal position and state key prefix.
    model.tpd_ner = replacement


def _v5_integration_manifest_fields(*, training_graph: bool) -> Dict[str, Any]:
    fields: Dict[str, Any] = dict(v5_per_manifest_fields())
    fields.update(
        {
            "qfg_integration_version": V5_PER_QFG_V2_CROA_INTEGRATION_VERSION,
            "v5_per_formal_mode_only": True,
            "v5_per_state_layout_compatible_with_v4_qfg2": True,
            "v5_per_checkpoint_semantically_interchangeable_with_v4_qfg2": False,
            "segmentation_path_modified": True,
            "segmentation_path_modification": (
                "bounded_query_only_frequency_modulation_plus_"
                "stage2_persistent_evidence_positive_routing"
            ),
            "training_graph": (
                "v5_per_qfg_v2_croa_with_training_only_tss"
                if training_graph
                else None
            ),
            "deployment_graph": "v5_per_qfg_v2_croa_no_tss",
        }
    )
    return fields


class TPDNERV8MPRSDCHV5PERQFGV2CROASurvivalSCTransNet(
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet
):
    """Formal V5-PER training graph retaining training-only TSS heads."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        variant: str,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
        tail_z_thresholds: Mapping[int, float] = DEFAULT_TAIL_Z_THRESHOLDS,
    ) -> None:
        super().__init__(
            parent,
            variant=variant,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
            dc_support_mode=V5_PER_FORMAL_DC_SUPPORT_MODE,
            tail_z_thresholds=tail_z_thresholds,
        )
        _install_v5_per_relay(self)

    def architecture_manifest(self) -> Dict[str, Any]:
        manifest = dict(super().architecture_manifest())
        manifest.update(_v5_integration_manifest_fields(training_graph=True))
        return manifest


class TPDNERV8MPRSDCHV5PERQFGV2CROAInferenceSCTransNet(
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet
):
    """Formal head-free V5-PER deployment graph."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        variant: str,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
        tail_z_thresholds: Mapping[int, float] = DEFAULT_TAIL_Z_THRESHOLDS,
    ) -> None:
        super().__init__(
            parent,
            variant=variant,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
            dc_support_mode=V5_PER_FORMAL_DC_SUPPORT_MODE,
            tail_z_thresholds=tail_z_thresholds,
        )
        _install_v5_per_relay(self)

    def architecture_manifest(self) -> Dict[str, Any]:
        manifest = dict(super().architecture_manifest())
        manifest.update(_v5_integration_manifest_fields(training_graph=False))
        manifest.update(
            {
                "target_survival_registered": False,
                "target_survival_state_removed": True,
                "inference_heads_required": False,
            }
        )
        return manifest


def _validate_v5_common(
    model: nn.Module,
    *,
    training_graph: bool,
    require_zero_initialized_heads: bool,
    require_identity_initialized_qfg: bool,
) -> Dict[str, Any]:
    expected_type = (
        TPDNERV8MPRSDCHV5PERQFGV2CROASurvivalSCTransNet
        if training_graph
        else TPDNERV8MPRSDCHV5PERQFGV2CROAInferenceSCTransNet
    )
    if type(model) is not expected_type:
        raise TypeError("formal NER V5-PER model must use its exact graph class")
    if (
        model.mode != "train"
        or model.deepsuper is not True
        or model.relay_enabled is not True
    ):
        raise RuntimeError(
            "formal NER V5-PER requires mode=train, deepsuper, and relay"
        )
    if model.tokenizer_variant != FORMAL_SURVIVAL_VARIANT:
        raise RuntimeError("formal NER V5-PER requires Full V8-MPRS-DCH")
    if model.relay_width != DEFAULT_RELAY_WIDTH:
        raise RuntimeError("formal NER V5-PER relay width differs")
    if model.relay_initialization_seed != DEFAULT_RELAY_INITIALIZATION_SEED:
        raise RuntimeError("formal NER V5-PER relay initialization seed differs")
    if type(model.tpd_ner) is not PersistentEvidencePositiveRoutingRelay:
        raise RuntimeError("formal model does not register the exact V5-PER relay")
    if model.tpd_ner.dc_support_mode != V5_PER_FORMAL_DC_SUPPORT_MODE:
        raise RuntimeError("formal NER V5-PER requires complement-tail support")
    if dict(model.tpd_ner.tail_z_thresholds) != dict(
        DEFAULT_TAIL_Z_THRESHOLDS
    ):
        raise RuntimeError("formal NER V5-PER tail thresholds differ")

    if not isinstance(model.tpd_qfg, QueryOnlyFrequencyGateV2CROA):
        raise RuntimeError("formal NER V5-PER QFG module type differs")
    qfg_validation = validate_formal_qfg_v2_croa(
        model.tpd_qfg,
        require_identity_initialization=require_identity_initialized_qfg,
    )
    if tuple(model.tpd_qfg.feature_channels) != FORMAL_QFG_FEATURE_CHANNELS:
        raise RuntimeError("formal NER V5-PER QFG feature channels differ")
    if model.tpd_qfg.mode != FORMAL_QFG_MODE:
        raise RuntimeError("formal NER V5-PER QFG mode differs")
    if model.tpd_qfg.hidden_channels != FORMAL_QFG_HIDDEN_CHANNELS:
        raise RuntimeError("formal NER V5-PER QFG hidden width differs")
    if (
        model.tpd_qfg.detach_frequency_source
        is not FORMAL_QFG_DETACH_FREQUENCY_SOURCE
    ):
        raise RuntimeError("formal NER V5-PER QFG detach boundary differs")
    if model.tpd_qfg.validate_finite is not FORMAL_QFG_VALIDATE_FINITE:
        raise RuntimeError("formal NER V5-PER QFG finite validation differs")
    if _parameter_count(model.tpd_qfg) != PRODUCTION_QFG_V2_CROA_PARAMETERS:
        raise RuntimeError("formal NER V5-PER QFG parameter count differs")

    state = model.state_dict()
    qfg_keys = {key for key in state if key.startswith(QFG_STATE_PREFIX)}
    if qfg_keys != set(QFG_STATE_KEYS):
        raise RuntimeError("formal NER V5-PER QFG state keys differ")
    if len(qfg_keys) != PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT:
        raise RuntimeError("formal NER V5-PER QFG state-key count differs")

    if training_graph:
        if not hasattr(model, "target_survival"):
            raise RuntimeError("formal V5-PER training graph lacks TSS heads")
        survival_keys = {
            key for key in state if key.startswith(SURVIVAL_STATE_PREFIX)
        }
        if survival_keys != set(SURVIVAL_STATE_KEYS):
            raise RuntimeError("formal V5-PER training TSS state keys differ")
        if (
            survival_parameter_count(model.target_survival)
            != PRODUCTION_V5_PER_SURVIVAL_PARAMETERS
        ):
            raise RuntimeError("formal V5-PER TSS parameter count differs")
        if require_zero_initialized_heads:
            for name, parameter in model.target_survival.named_parameters():
                if torch.count_nonzero(parameter).item() != 0:
                    raise RuntimeError(
                        f"formal V5-PER TSS parameter {name} is not exactly zero"
                    )
        expected_state_count = (
            FORMAL_V5_PER_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT
        )
        expected_parameters = PRODUCTION_V5_PER_QFG_V2_CROA_SURVIVAL_PARAMETERS
    else:
        if hasattr(model, "target_survival") or any(
            key.startswith(SURVIVAL_STATE_PREFIX) for key in state
        ):
            raise RuntimeError("formal V5-PER inference graph retains TSS state")
        expected_state_count = (
            FORMAL_V5_PER_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT
        )
        expected_parameters = PRODUCTION_V5_PER_QFG_V2_CROA_INFERENCE_PARAMETERS

    if len(state) != expected_state_count:
        raise RuntimeError("formal NER V5-PER total state-key count differs")
    if _parameter_count(model) != expected_parameters:
        raise RuntimeError("formal NER V5-PER total parameter count differs")

    manifest = model.architecture_manifest()
    expected_manifest = {
        "relay_version": V5_PER_RELAY_VERSION,
        "ner_version": "v5_per",
        "stage4_formula": "v4_exact",
        "stage3_formula": "v4_exact",
        "stage2_persistent_support_gradient": "stopped",
        "ner_v5_per_dc_support_mode": V5_PER_FORMAL_DC_SUPPORT_MODE,
        "qfg_integration_version": V5_PER_QFG_V2_CROA_INTEGRATION_VERSION,
        "qfg_enabled": True,
        "qfg_frequency_mode": FORMAL_QFG_MODE,
        "qfg_initialization_seed": FORMAL_QFG_INITIALIZATION_SEED,
        "qfg_alpha_effective_initialization": FORMAL_QFG_ALPHA_EFFECTIVE_INIT,
        "parameters_added_vs_v4": 0,
        "buffers_added_vs_v4": 0,
        "state_semantics_identical_to_v4": False,
        "checkpoint_semantically_interchangeable_with_v4": False,
        "v5_per_checkpoint_semantically_interchangeable_with_v4_qfg2": False,
        "deployment_graph": "v5_per_qfg_v2_croa_no_tss",
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            raise RuntimeError(
                f"formal NER V5-PER manifest field {name!r} differs"
            )

    return {
        "model": f"{type(model).__module__}.{type(model).__qualname__}",
        "variant": FORMAL_SURVIVAL_VARIANT,
        "relay_version": V5_PER_RELAY_VERSION,
        "qfg_integration_version": V5_PER_QFG_V2_CROA_INTEGRATION_VERSION,
        "state_key_count": len(state),
        "qfg_state_key_count": len(qfg_keys),
        "total_parameters": _parameter_count(model),
        "target_survival_registered": training_graph,
        "state_layout_compatible_with_v4": True,
        "checkpoint_semantically_interchangeable_with_v4": False,
        "qfg_core_manifest": qfg_validation,
        "architecture_manifest": manifest,
    }


def validate_formal_v5_per_qfg_v2_croa_survival_model(
    model: nn.Module,
    *,
    require_zero_initialized_heads: bool = False,
    require_identity_initialized_qfg: bool = False,
) -> Dict[str, Any]:
    """Validate the exact formal V5-PER training graph."""

    return _validate_v5_common(
        model,
        training_graph=True,
        require_zero_initialized_heads=require_zero_initialized_heads,
        require_identity_initialized_qfg=require_identity_initialized_qfg,
    )


def validate_formal_v5_per_qfg_v2_croa_inference_model(
    model: nn.Module,
    *,
    require_identity_initialized_qfg: bool = False,
) -> Dict[str, Any]:
    """Validate the exact formal head-free V5-PER graph."""

    return _validate_v5_common(
        model,
        training_graph=False,
        require_zero_initialized_heads=False,
        require_identity_initialized_qfg=require_identity_initialized_qfg,
    )


def _build_clean_parent(seed: int) -> Tuple[SCTransNet, Dict[str, Any]]:
    from experiments.train_tpd_clean_v8_mprs_dch import (
        build_clean_v8_mprs_dch_model,
    )

    return build_clean_v8_mprs_dch_model(
        FORMAL_SURVIVAL_VARIANT,
        _require_formal_seed(seed),
    )


def build_formal_v5_per_qfg_v2_croa_survival_model(
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> Tuple[
    TPDNERV8MPRSDCHV5PERQFGV2CROASurvivalSCTransNet,
    Dict[str, Any],
]:
    """Build the formal V5-PER training graph without loading a checkpoint."""

    parent, parent_metadata = _build_clean_parent(seed)
    model = TPDNERV8MPRSDCHV5PERQFGV2CROASurvivalSCTransNet(
        parent,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    metadata = validate_formal_v5_per_qfg_v2_croa_survival_model(
        model,
        require_zero_initialized_heads=True,
        require_identity_initialized_qfg=True,
    )
    metadata["raw_parent_metadata"] = parent_metadata
    return model, metadata


def build_formal_v5_per_qfg_v2_croa_inference_model(
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> Tuple[
    TPDNERV8MPRSDCHV5PERQFGV2CROAInferenceSCTransNet,
    Dict[str, Any],
]:
    """Build the formal V5-PER head-free graph without loading a checkpoint."""

    parent, parent_metadata = _build_clean_parent(seed)
    model = TPDNERV8MPRSDCHV5PERQFGV2CROAInferenceSCTransNet(
        parent,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    metadata = validate_formal_v5_per_qfg_v2_croa_inference_model(
        model,
        require_identity_initialized_qfg=True,
    )
    metadata["raw_parent_metadata"] = parent_metadata
    return model, metadata


__all__ = [
    "FORMAL_V5_PER_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT",
    "FORMAL_V5_PER_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT",
    "PRODUCTION_V5_PER_QFG_V2_CROA_INFERENCE_PARAMETERS",
    "PRODUCTION_V5_PER_QFG_V2_CROA_SURVIVAL_PARAMETERS",
    "TPDNERV8MPRSDCHV5PERQFGV2CROAInferenceSCTransNet",
    "TPDNERV8MPRSDCHV5PERQFGV2CROASurvivalSCTransNet",
    "V5_PER_QFG_V2_CROA_INTEGRATION_VERSION",
    "build_formal_v5_per_qfg_v2_croa_inference_model",
    "build_formal_v5_per_qfg_v2_croa_survival_model",
    "validate_formal_v5_per_qfg_v2_croa_inference_model",
    "validate_formal_v5_per_qfg_v2_croa_survival_model",
]
