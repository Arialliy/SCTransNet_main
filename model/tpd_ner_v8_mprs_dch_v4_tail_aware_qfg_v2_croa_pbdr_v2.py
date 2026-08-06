"""PBDR-V2 integration for TPD8 + NER4 + QFG2-CROA.

The current production graph is inherited unchanged.  One forward-local
router is evaluated only after the existing raw ``q4``, ``out`` and ``d0``
tensors have all been produced.  Deep-supervision outputs one through five
remain unchanged; only the sixth/final probability map uses the routed logit.
Training keeps the TSS objective disabled, matching the current Final recipe.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SCTransNet import SCTransNet
from model.tpd_clean_v8_mprs_dch import clean_v8_mprs_dch_variant_spec
from model.tpd_frequency_gate_v2_croa import (
    QueryOnlyFrequencyGateV2CROA,
    validate_formal_qfg_v2_croa,
)
from model.tpd_ner_v8_mprs_dch import (
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    DEFAULT_DC_SUPPORT_MODE,
    DEFAULT_TAIL_Z_THRESHOLDS,
    V4_RELAY_VERSION,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    FORMAL_QFG_FEATURE_CHANNELS,
    FORMAL_QFG_HIDDEN_CHANNELS,
    FORMAL_QFG_MODE,
    FORMAL_QFG_VALIDATE_FINITE,
    FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
    PRODUCTION_QFG_V2_CROA_PARAMETERS,
    PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS,
    PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS,
    QFG_STATE_KEYS,
    QFG_STATE_PREFIX,
    QFG_V2_CROA_INTEGRATION_VERSION,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
    _formal_context_gate,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    FORMAL_SURVIVAL_INITIALIZATION_SEED,
    FORMAL_SURVIVAL_VARIANT,
    PRODUCTION_SURVIVAL_PARAMETERS,
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)
from model.tpd_persistent_evidence_residual_router_v2 import (
    FORMAL_CONFIDENCE_FLOOR,
    FORMAL_DIRECT_RESIDUAL_LIMIT,
    FORMAL_DISAGREEMENT_STRENGTH_LIMIT,
    PBDR_V2_LOCAL_STATE_KEYS,
    PBDR_V2_VERSION,
    PRODUCTION_PBDR_V2_BUFFER_COUNT,
    PRODUCTION_PBDR_V2_PARAMETERS,
    PRODUCTION_PBDR_V2_STATE_KEY_COUNT,
    PersistentEvidenceResidualRouterV2,
    validate_formal_pbdr_v2_router,
)
from model.tpd_query_frequency_bridge import frequency_encoder_forward
from model.tpd_survival import survival_parameter_count


PBDR_V2_INTEGRATION_VERSION = "v4_qfg_v2_croa_pbdr_v2_v1"
PBDR_V2_STATE_PREFIX = "pbdr_v2."
PBDR_V2_STATE_KEYS = tuple(
    f"{PBDR_V2_STATE_PREFIX}{key}" for key in PBDR_V2_LOCAL_STATE_KEYS
)

PRODUCTION_V4_QFG_V2_CROA_PBDR_V2_SURVIVAL_PARAMETERS = (
    PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS
    + PRODUCTION_PBDR_V2_PARAMETERS
)
FORMAL_V4_QFG_V2_CROA_PBDR_V2_SURVIVAL_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT
    + PRODUCTION_PBDR_V2_STATE_KEY_COUNT
)
PRODUCTION_V4_QFG_V2_CROA_PBDR_V2_INFERENCE_PARAMETERS = (
    PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS
    + PRODUCTION_PBDR_V2_PARAMETERS
)
FORMAL_V4_QFG_V2_CROA_PBDR_V2_INFERENCE_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT
    + PRODUCTION_PBDR_V2_STATE_KEY_COUNT
)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _require_formal_seed(seed: int) -> int:
    if type(seed) is not int or seed != FORMAL_SURVIVAL_INITIALIZATION_SEED:
        raise ValueError("formal PBDR-V2 construction requires seed=42")
    return seed


def _install_formal_pbdr_v2(model: nn.Module) -> None:
    if hasattr(model, "pbdr_v2"):
        raise RuntimeError("PBDR-V2 integration attempted twice")
    reference = next(model.parameters())
    model.pbdr_v2 = PersistentEvidenceResidualRouterV2(
        device=reference.device,
        dtype=reference.dtype,
    )
    model.pbdr_v2.train(model.training)
    validate_formal_pbdr_v2_router(
        model.pbdr_v2,
        require_zero_initialization=True,
    )


def _pbdr_v2_manifest_fields() -> Dict[str, Any]:
    return {
        "pbdr_v2_integration_version": PBDR_V2_INTEGRATION_VERSION,
        "pbdr_v2_version": PBDR_V2_VERSION,
        "pbdr_v2_enabled": True,
        "pbdr_v2_inputs": ("raw_q4", "raw_out", "raw_d0"),
        "pbdr_v2_location": "after_d0_before_final_sigmoid",
        "pbdr_v2_q4_gradient_boundary": "stop_gradient_before_router",
        "pbdr_v2_confidence": "0.05+0.90*sigmoid(conv1x1(q4_rms))",
        "pbdr_v2_confidence_floor": FORMAL_CONFIDENCE_FLOOR,
        "pbdr_v2_direct_residual": "C*tanh(conv1x1_no_bias(q4_rms))",
        "pbdr_v2_direct_residual_limit": FORMAL_DIRECT_RESIDUAL_LIMIT,
        "pbdr_v2_rescue_strength": "0.5*tanh(rescue_strength_raw)",
        "pbdr_v2_suppression_strength": (
            "0.5*tanh(suppression_strength_raw)"
        ),
        "pbdr_v2_disagreement_strength_limit": (
            FORMAL_DISAGREEMENT_STRENGTH_LIMIT
        ),
        "pbdr_v2_rescue_and_suppression_independent": True,
        "pbdr_v2_interpolation": "bilinear_align_corners_false",
        "pbdr_v2_zero_anchor": "current_final_exact",
        "pbdr_v2_parameters": PRODUCTION_PBDR_V2_PARAMETERS,
        "pbdr_v2_state_prefix": PBDR_V2_STATE_PREFIX,
        "pbdr_v2_state_key_count": PRODUCTION_PBDR_V2_STATE_KEY_COUNT,
        "pbdr_v2_persistent_buffers": PRODUCTION_PBDR_V2_BUFFER_COUNT,
        "pbdr_v2_inference_required": True,
        "pbdr_v2_formal_training_precision": "fp32",
        "tpd_formula_changed": False,
        "ner_relay_formula_changed": False,
        "qfg_formula_changed": False,
        "tss_objective_enabled": False,
        "segmentation_path_modified": True,
        "segmentation_path_modification": (
            "qfg_query_modulation_plus_pbdr_v2_final_readout_routing"
        ),
    }


class _PBDRV2ForwardMixin:
    """Shared training/deployment forward with only the final readout changed."""

    pbdr_v2: PersistentEvidenceResidualRouterV2

    def _forward_with_relay(self, x: torch.Tensor):
        x1 = self.inc(x)
        x2 = self.down_encoder1(self.pool(x1))
        x3 = self.down_encoder2(self.pool(x2))
        x4 = self.down_encoder3(self.pool(x3))
        d5 = self.down_encoder4(self.pool(x4))
        f1, f2, f3, f4 = x1, x2, x3, x4

        emb1, emb2, emb3, emb4, evidence1, evidence2 = (
            self.explicit_embeddings(x1, x2, x3, x4)
        )
        h11, h12, h13 = evidence1
        h21, h22 = evidence2
        prepared_qfg = self.tpd_qfg.prepare(
            (x1, x2, x3, x4),
            tuple(
                tuple(embedding.shape[-2:])
                for embedding in (emb1, emb2, emb3, emb4)
            ),
        )
        encoded1, encoded2, encoded3, encoded4, _ = (
            frequency_encoder_forward(
                self.mtc.encoder,
                emb1,
                emb2,
                emb3,
                emb4,
                self.tpd_qfg,
                prepared_qfg,
            )
        )
        x1 = self.mtc.reconstruct_1(encoded1) + f1
        x2 = self.mtc.reconstruct_2(encoded2) + f2
        x3 = self.mtc.reconstruct_3(encoded3) + f3
        x4 = self.mtc.reconstruct_4(encoded4) + f4
        x1, x2, x3, x4 = x1 + f1, x2 + f2, x3 + f3, x4 + f4

        up4, skip4 = self.up_decoder4.prepare(d5, x4)
        q4, mask4 = self.tpd_ner.forward_stage(
            4,
            (h13, h22, up4),
            tuple(up4.shape[-2:]),
        )
        d4 = self.up_decoder4.finish(up4, skip4, mask4)

        up3, skip3 = self.up_decoder3.prepare(d4, x3)
        q3, mask3 = self.tpd_ner.forward_stage(
            3,
            (h12, h21, q4, up3),
            tuple(up3.shape[-2:]),
        )
        d3 = self.up_decoder3.finish(up3, skip3, mask3)

        up2, skip2 = self.up_decoder2.prepare(d3, x2)
        _, mask2 = self.tpd_ner.forward_stage(
            2,
            (h11, q3, up2),
            tuple(up2.shape[-2:]),
        )
        d2 = self.up_decoder2.finish(up2, skip2, mask2)
        out = self.outc(self.up_decoder1(d2, x1))

        if not self.deepsuper:
            raise RuntimeError("PBDR-V2 requires deepsuper=True to produce d0")
        gt_5 = self.gt_conv5(d5)
        gt_4 = self.gt_conv4(d4)
        gt_3 = self.gt_conv3(d3)
        gt_2 = self.gt_conv2(d2)
        gt5 = F.interpolate(
            gt_5,
            scale_factor=16,
            mode="bilinear",
            align_corners=True,
        )
        gt4 = F.interpolate(
            gt_4,
            scale_factor=8,
            mode="bilinear",
            align_corners=True,
        )
        gt3 = F.interpolate(
            gt_3,
            scale_factor=4,
            mode="bilinear",
            align_corners=True,
        )
        gt2 = F.interpolate(
            gt_2,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), dim=1))
        routed_out = self.pbdr_v2(out, d0, q4)
        if self.mode != "train":
            return torch.sigmoid(routed_out)
        return (
            torch.sigmoid(gt5),
            torch.sigmoid(gt4),
            torch.sigmoid(gt3),
            torch.sigmoid(gt2),
            torch.sigmoid(d0),
            torch.sigmoid(routed_out),
        )

    def architecture_manifest(self) -> Dict[str, Any]:
        manifest = dict(super().architecture_manifest())
        manifest.update(_pbdr_v2_manifest_fields())
        manifest["deployment_graph"] = (
            "v4_qfg_v2_croa_pbdr_v2_with_training_only_tss_heads"
            if hasattr(self, "target_survival")
            else "v4_qfg_v2_croa_pbdr_v2_no_tss"
        )
        manifest["pbdr_v2_core_manifest"] = (
            self.pbdr_v2.architecture_manifest()
        )
        return manifest


class TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2SurvivalSCTransNet(
    _PBDRV2ForwardMixin,
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
):
    """Scratch-training graph retaining training-only Survival heads."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        variant: str,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode: str = DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds: Mapping[int, float] = DEFAULT_TAIL_Z_THRESHOLDS,
    ) -> None:
        super().__init__(
            parent,
            variant=variant,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
            dc_support_mode=dc_support_mode,
            tail_z_thresholds=tail_z_thresholds,
        )
        _install_formal_pbdr_v2(self)


class TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2InferenceSCTransNet(
    _PBDRV2ForwardMixin,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
):
    """Head-free deployment graph retaining trained PBDR-V2 state."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        variant: str,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode: str = DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds: Mapping[int, float] = DEFAULT_TAIL_Z_THRESHOLDS,
    ) -> None:
        super().__init__(
            parent,
            variant=variant,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
            dc_support_mode=dc_support_mode,
            tail_z_thresholds=tail_z_thresholds,
        )
        _install_formal_pbdr_v2(self)


TPD8NER4QFG2PBDRV2SurvivalSCTransNet = (
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2SurvivalSCTransNet
)


def _validate_formal_pbdr_v2_model(
    model: nn.Module,
    *,
    training_graph: bool,
    require_zero_initialized_heads: bool,
    require_identity_initialized_qfg: bool,
    require_zero_initialized_pbdr_v2: bool,
) -> Dict[str, Any]:
    expected_type = (
        TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2SurvivalSCTransNet
        if training_graph
        else TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2InferenceSCTransNet
    )
    if type(model) is not expected_type:
        raise TypeError("formal PBDR-V2 model must use its exact graph class")
    if model.mode != "train" or model.deepsuper is not True or not model.relay_enabled:
        raise RuntimeError(
            "formal PBDR-V2 requires mode=train, deepsuper, and relay"
        )
    if model.tokenizer_variant != FORMAL_SURVIVAL_VARIANT:
        raise RuntimeError("formal PBDR-V2 requires Full V8-MPRS-DCH")
    if model.relay_width != DEFAULT_RELAY_WIDTH:
        raise RuntimeError("formal PBDR-V2 relay width differs")
    if model.relay_initialization_seed != DEFAULT_RELAY_INITIALIZATION_SEED:
        raise RuntimeError("formal PBDR-V2 relay seed differs")
    if model.tpd_ner.dc_support_mode != DEFAULT_DC_SUPPORT_MODE:
        raise RuntimeError("formal PBDR-V2 requires complement-tail NER4")
    if dict(model.tpd_ner.tail_z_thresholds) != dict(DEFAULT_TAIL_Z_THRESHOLDS):
        raise RuntimeError("formal PBDR-V2 tail thresholds differ")
    context_gate = _formal_context_gate(model)

    if not isinstance(model.tpd_qfg, QueryOnlyFrequencyGateV2CROA):
        raise RuntimeError("formal PBDR-V2 QFG module type differs")
    qfg_manifest = validate_formal_qfg_v2_croa(
        model.tpd_qfg,
        require_identity_initialization=require_identity_initialized_qfg,
    )
    if tuple(model.tpd_qfg.feature_channels) != FORMAL_QFG_FEATURE_CHANNELS:
        raise RuntimeError("formal PBDR-V2 QFG channels differ")
    if model.tpd_qfg.mode != FORMAL_QFG_MODE:
        raise RuntimeError("formal PBDR-V2 QFG mode differs")
    if model.tpd_qfg.hidden_channels != FORMAL_QFG_HIDDEN_CHANNELS:
        raise RuntimeError("formal PBDR-V2 QFG width differs")
    if model.tpd_qfg.validate_finite is not FORMAL_QFG_VALIDATE_FINITE:
        raise RuntimeError("formal PBDR-V2 QFG finite setting differs")
    if _parameter_count(model.tpd_qfg) != PRODUCTION_QFG_V2_CROA_PARAMETERS:
        raise RuntimeError("formal PBDR-V2 QFG parameter count differs")

    pbdr_manifest = validate_formal_pbdr_v2_router(
        model.pbdr_v2,
        require_zero_initialization=require_zero_initialized_pbdr_v2,
    )
    state = model.state_dict()
    pbdr_keys = tuple(key for key in state if key.startswith(PBDR_V2_STATE_PREFIX))
    if pbdr_keys != PBDR_V2_STATE_KEYS:
        raise RuntimeError("formal integrated PBDR-V2 state keys differ")
    expected_state_count = (
        FORMAL_V4_QFG_V2_CROA_PBDR_V2_SURVIVAL_STATE_KEY_COUNT
        if training_graph
        else FORMAL_V4_QFG_V2_CROA_PBDR_V2_INFERENCE_STATE_KEY_COUNT
    )
    expected_parameter_count = (
        PRODUCTION_V4_QFG_V2_CROA_PBDR_V2_SURVIVAL_PARAMETERS
        if training_graph
        else PRODUCTION_V4_QFG_V2_CROA_PBDR_V2_INFERENCE_PARAMETERS
    )
    if len(state) != expected_state_count:
        raise RuntimeError("formal integrated PBDR-V2 state count differs")
    if _parameter_count(model) != expected_parameter_count:
        raise RuntimeError("formal integrated PBDR-V2 parameter count differs")
    qfg_keys = tuple(key for key in state if key.startswith(QFG_STATE_PREFIX))
    if set(qfg_keys) != set(QFG_STATE_KEYS):
        raise RuntimeError("formal integrated QFG state keys differ")

    survival_keys = tuple(
        key for key in state if key.startswith(SURVIVAL_STATE_PREFIX)
    )
    if training_graph:
        if set(survival_keys) != set(SURVIVAL_STATE_KEYS):
            raise RuntimeError("formal training graph Survival keys differ")
        if survival_parameter_count(model.target_survival) != PRODUCTION_SURVIVAL_PARAMETERS:
            raise RuntimeError("formal training graph Survival parameters differ")
        if require_zero_initialized_heads:
            for name, parameter in model.target_survival.named_parameters():
                if int(torch.count_nonzero(parameter)) != 0:
                    raise RuntimeError(f"Survival parameter {name} is not zero")
    elif survival_keys or hasattr(model, "target_survival"):
        raise RuntimeError("formal inference graph retains Survival state")

    reference = next(
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(PBDR_V2_STATE_PREFIX)
    )
    for name, parameter in model.pbdr_v2.named_parameters():
        if parameter.device != reference.device or parameter.dtype != reference.dtype:
            raise RuntimeError(f"PBDR-V2 parameter {name} placement differs")
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError(f"PBDR-V2 parameter {name} is non-finite")

    manifest = model.architecture_manifest()
    expected_manifest = {
        "relay_version": V4_RELAY_VERSION,
        "qfg_integration_version": QFG_V2_CROA_INTEGRATION_VERSION,
        "pbdr_v2_integration_version": PBDR_V2_INTEGRATION_VERSION,
        "pbdr_v2_version": PBDR_V2_VERSION,
        "pbdr_v2_enabled": True,
        "pbdr_v2_parameters": PRODUCTION_PBDR_V2_PARAMETERS,
        "pbdr_v2_state_key_count": PRODUCTION_PBDR_V2_STATE_KEY_COUNT,
        "pbdr_v2_zero_anchor": "current_final_exact",
        "tss_objective_enabled": False,
        "segmentation_path_modified": True,
        "deployment_graph": (
            "v4_qfg_v2_croa_pbdr_v2_with_training_only_tss_heads"
            if training_graph
            else "v4_qfg_v2_croa_pbdr_v2_no_tss"
        ),
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            raise RuntimeError(f"formal PBDR-V2 manifest field {name!r} differs")
    return {
        "model": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "variant": FORMAL_SURVIVAL_VARIANT,
        "relay_version": V4_RELAY_VERSION,
        "qfg_integration_version": QFG_V2_CROA_INTEGRATION_VERSION,
        "pbdr_v2_integration_version": PBDR_V2_INTEGRATION_VERSION,
        "state_key_count": expected_state_count,
        "pbdr_v2_state_keys": PBDR_V2_STATE_KEYS,
        "pbdr_v2_parameters": PRODUCTION_PBDR_V2_PARAMETERS,
        "total_parameters": expected_parameter_count,
        "context_gate": context_gate,
        "qfg_core_manifest": qfg_manifest,
        "pbdr_v2_core_manifest": pbdr_manifest,
        "target_survival_registered": training_graph,
        "architecture_manifest": manifest,
    }


def validate_formal_v4_qfg_v2_croa_pbdr_v2_survival_model(
    model: nn.Module,
    *,
    require_zero_initialized_heads: bool = False,
    require_identity_initialized_qfg: bool = False,
    require_zero_initialized_pbdr_v2: bool = False,
) -> Dict[str, Any]:
    return _validate_formal_pbdr_v2_model(
        model,
        training_graph=True,
        require_zero_initialized_heads=require_zero_initialized_heads,
        require_identity_initialized_qfg=require_identity_initialized_qfg,
        require_zero_initialized_pbdr_v2=require_zero_initialized_pbdr_v2,
    )


def validate_formal_v4_qfg_v2_croa_pbdr_v2_inference_model(
    model: nn.Module,
    *,
    require_identity_initialized_qfg: bool = False,
    require_zero_initialized_pbdr_v2: bool = False,
) -> Dict[str, Any]:
    return _validate_formal_pbdr_v2_model(
        model,
        training_graph=False,
        require_zero_initialized_heads=False,
        require_identity_initialized_qfg=require_identity_initialized_qfg,
        require_zero_initialized_pbdr_v2=require_zero_initialized_pbdr_v2,
    )


def _build_raw_parent(seed: int) -> tuple[SCTransNet, Dict[str, Any]]:
    from experiments.train_tpd_clean_v8_mprs_dch import (
        build_clean_v8_mprs_dch_model,
    )

    return build_clean_v8_mprs_dch_model(
        FORMAL_SURVIVAL_VARIANT,
        _require_formal_seed(seed),
    )


def build_formal_v4_qfg_v2_croa_pbdr_v2_survival_model(
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> Tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2SurvivalSCTransNet,
    Dict[str, Any],
]:
    parent, parent_metadata = _build_raw_parent(seed)
    model = TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2SurvivalSCTransNet(
        parent,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    metadata = validate_formal_v4_qfg_v2_croa_pbdr_v2_survival_model(
        model,
        require_zero_initialized_heads=True,
        require_identity_initialized_qfg=True,
        require_zero_initialized_pbdr_v2=True,
    )
    metadata.update(
        {
            "construction": "scratch_seed42_no_parent_checkpoint",
            "raw_parent_metadata": parent_metadata,
        }
    )
    return model, metadata


def build_formal_v4_qfg_v2_croa_pbdr_v2_inference_model(
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> Tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2InferenceSCTransNet,
    Dict[str, Any],
]:
    parent, parent_metadata = _build_raw_parent(seed)
    model = TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2InferenceSCTransNet(
        parent,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    metadata = validate_formal_v4_qfg_v2_croa_pbdr_v2_inference_model(
        model,
        require_identity_initialized_qfg=True,
        require_zero_initialized_pbdr_v2=True,
    )
    metadata.update(
        {
            "construction": "scratch_seed42_no_parent_checkpoint",
            "raw_parent_metadata": parent_metadata,
        }
    )
    return model, metadata


__all__ = [
    "FORMAL_V4_QFG_V2_CROA_PBDR_V2_INFERENCE_STATE_KEY_COUNT",
    "FORMAL_V4_QFG_V2_CROA_PBDR_V2_SURVIVAL_STATE_KEY_COUNT",
    "PBDR_V2_INTEGRATION_VERSION",
    "PBDR_V2_STATE_KEYS",
    "PBDR_V2_STATE_PREFIX",
    "PRODUCTION_V4_QFG_V2_CROA_PBDR_V2_INFERENCE_PARAMETERS",
    "PRODUCTION_V4_QFG_V2_CROA_PBDR_V2_SURVIVAL_PARAMETERS",
    "TPD8NER4QFG2PBDRV2SurvivalSCTransNet",
    "TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2InferenceSCTransNet",
    "TPDNERV8MPRSDCHV4QFGV2CROAPBDRV2SurvivalSCTransNet",
    "build_formal_v4_qfg_v2_croa_pbdr_v2_inference_model",
    "build_formal_v4_qfg_v2_croa_pbdr_v2_survival_model",
    "validate_formal_v4_qfg_v2_croa_pbdr_v2_inference_model",
    "validate_formal_v4_qfg_v2_croa_pbdr_v2_survival_model",
]
