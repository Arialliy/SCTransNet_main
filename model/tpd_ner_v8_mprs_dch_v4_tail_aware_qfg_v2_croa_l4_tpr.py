"""TPD8 + NER4 + QFG2-CROA with NER-protected L4 reallocation.

Only the level-four reconstructed/encoder skip fusion is extended.  The
existing q4 evidence is computed before level-four CCA, converted into a
detached binary protection map by :class:`NERL4TargetProtectedReallocation`,
and used to suppress reallocation in target-evidence regions.  TPD8, the
five-node NER4 relay, QFG2-CROA, the decoder, deep supervision, and the
training-only Survival-head interface remain inherited unchanged.  Formal
training keeps the TSS objective disabled, matching the current Final recipe.
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
from model.tpd_ner_l4_target_protected_reallocation import (
    FORMAL_L4_CHANNELS,
    FORMAL_L4_GATE_LIMIT,
    FORMAL_L4_PROTECTION_DILATION_KERNEL,
    FORMAL_L4_TAIL_Z_THRESHOLD,
    NER_L4_TPR_LOCAL_STATE_KEYS,
    NER_L4_TPR_VERSION,
    NERL4TargetProtectedReallocation,
    PRODUCTION_NER_L4_TPR_BUFFER_COUNT,
    PRODUCTION_NER_L4_TPR_PARAMETERS,
    PRODUCTION_NER_L4_TPR_STATE_KEY_COUNT,
    validate_formal_ner_l4_target_protected_reallocation,
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
    FORMAL_QFG_DETACH_FREQUENCY_SOURCE,
    FORMAL_QFG_FEATURE_CHANNELS,
    FORMAL_QFG_HIDDEN_CHANNELS,
    FORMAL_QFG_MODE,
    FORMAL_QFG_VALIDATE_FINITE,
    PRODUCTION_QFG_V2_CROA_PARAMETERS,
    PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT,
    QFG_STATE_KEYS,
    QFG_STATE_PREFIX,
    QFG_V2_CROA_INTEGRATION_VERSION,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    FORMAL_SURVIVAL_INITIALIZATION_SEED,
    FORMAL_SURVIVAL_VARIANT,
    PRODUCTION_SURVIVAL_PARAMETERS,
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)
from model.tpd_query_frequency_bridge import frequency_encoder_forward
from model.tpd_survival import survival_parameter_count


L4_TPR_INTEGRATION_VERSION = "v4_qfg_v2_croa_ner_l4_tpr_v1"
L4_TPR_STATE_PREFIX = "ner_l4_tpr."
L4_TPR_STATE_KEYS = tuple(
    f"{L4_TPR_STATE_PREFIX}{key}" for key in NER_L4_TPR_LOCAL_STATE_KEYS
)

PRODUCTION_V4_QFG_V2_CROA_L4_TPR_SURVIVAL_PARAMETERS = 10_870_484
FORMAL_V4_QFG_V2_CROA_L4_TPR_SURVIVAL_STATE_KEY_COUNT = 569
PRODUCTION_V4_QFG_V2_CROA_L4_TPR_INFERENCE_PARAMETERS = 10_870_386
FORMAL_V4_QFG_V2_CROA_L4_TPR_INFERENCE_STATE_KEY_COUNT = 565


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _require_formal_seed(seed: int) -> int:
    if type(seed) is not int or seed != FORMAL_SURVIVAL_INITIALIZATION_SEED:
        raise ValueError("formal NER-L4-TPR construction requires seed=42")
    return seed


def _formal_context_gate(model: nn.Module) -> float:
    expected = float(
        clean_v8_mprs_dch_variant_spec(FORMAL_SURVIVAL_VARIANT)[
            "context_gate"
        ]
    )
    for embedding_name in ("embeddings_1", "embeddings_2"):
        embedding = getattr(model.mtc, embedding_name)
        for block in embedding.blocks:
            if float(block.context_gate) != expected:
                raise RuntimeError(
                    f"formal NER-L4-TPR {embedding_name} context gate differs"
                )
    return expected


def _install_formal_l4_tpr(model: nn.Module) -> None:
    if hasattr(model, "ner_l4_tpr"):
        raise RuntimeError("NER-L4-TPR integration attempted twice")
    reference = next(model.parameters())
    model.ner_l4_tpr = NERL4TargetProtectedReallocation(
        device=reference.device,
        dtype=reference.dtype,
    )
    model.ner_l4_tpr.train(model.training)
    validate_formal_ner_l4_target_protected_reallocation(
        model.ner_l4_tpr,
        require_zero_initialization=True,
    )


def _l4_tpr_manifest_fields() -> Dict[str, Any]:
    return {
        "l4_tpr_integration_version": L4_TPR_INTEGRATION_VERSION,
        "ner_l4_tpr_version": NER_L4_TPR_VERSION,
        "ner_l4_tpr_enabled": True,
        "ner_l4_tpr_level": 4,
        "ner_l4_tpr_channels": FORMAL_L4_CHANNELS,
        "ner_l4_tpr_gate": "0.25*tanh(reallocation_logits)",
        "ner_l4_tpr_gate_limit": FORMAL_L4_GATE_LIMIT,
        "ner_l4_tpr_evidence": "existing_q4_tail_support",
        "ner_l4_tpr_tail_z_threshold": FORMAL_L4_TAIL_Z_THRESHOLD,
        "ner_l4_tpr_binary_protection": True,
        "ner_l4_tpr_protection_detached": True,
        "ner_l4_tpr_protection_dilation_kernel": (
            FORMAL_L4_PROTECTION_DILATION_KERNEL
        ),
        "ner_l4_tpr_protected_region_fusion": "(T4+E4)+E4",
        "ner_l4_tpr_eligible_region_formula": (
            "((T4+E4)+E4)+(1-P4)*G4*T4-(1-P4)*G4*E4"
        ),
        "ner_l4_tpr_coefficient_sum": 3.0,
        "ner_l4_tpr_coefficient_sum_is_constant": True,
        "ner_l4_tpr_zero_anchor": "current_final_exact",
        "ner_l4_tpr_parameters": PRODUCTION_NER_L4_TPR_PARAMETERS,
        "ner_l4_tpr_state_prefix": L4_TPR_STATE_PREFIX,
        "ner_l4_tpr_state_key_count": PRODUCTION_NER_L4_TPR_STATE_KEY_COUNT,
        "ner_l4_tpr_persistent_buffers": PRODUCTION_NER_L4_TPR_BUFFER_COUNT,
        "ner_l4_tpr_inference_required": True,
        "tpd_formula_changed": False,
        "ner_relay_formula_changed": False,
        "qfg_formula_changed": False,
        "tss_objective_enabled": False,
        "segmentation_path_modified": True,
        "segmentation_path_modification": (
            "qfg_query_modulation_plus_ner_conditioned_l4_"
            "target_protected_reallocation"
        ),
    }


class _NERL4TPRForwardMixin:
    """Shared training/deployment forward with only the L4 fusion changed."""

    ner_l4_tpr: NERL4TargetProtectedReallocation

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
        transformed1 = self.mtc.reconstruct_1(encoded1)
        transformed2 = self.mtc.reconstruct_2(encoded2)
        transformed3 = self.mtc.reconstruct_3(encoded3)
        transformed4 = self.mtc.reconstruct_4(encoded4)

        # Levels 1--3 retain the exact current-Final operation order.
        x1 = transformed1.add(f1).add(f1)
        x2 = transformed2.add(f2).add(f2)
        x3 = transformed3.add(f3).add(f3)

        # q4 depends on h13, h22, and up4 only.  Computing it before CCA makes
        # the existing evidence available to L4 fusion without a dependency
        # cycle or a second execution of the upsample/relay modules.
        up4 = self.up_decoder4.up(d5)
        if up4.shape[-2:] != f4.shape[-2:]:
            up4 = F.interpolate(up4, size=f4.shape[-2:], mode="nearest")
        q4, mask4 = self.tpd_ner.forward_stage(
            4,
            (h13, h22, up4),
            tuple(up4.shape[-2:]),
        )
        x4 = self.ner_l4_tpr(transformed4, f4, q4)
        skip4 = self.up_decoder4.coatt(g=up4, x=x4)
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
            return torch.sigmoid(out)
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
        if self.mode != "train":
            return torch.sigmoid(out)
        return (
            torch.sigmoid(gt5),
            torch.sigmoid(gt4),
            torch.sigmoid(gt3),
            torch.sigmoid(gt2),
            torch.sigmoid(d0),
            torch.sigmoid(out),
        )

    def architecture_manifest(self) -> Dict[str, Any]:
        manifest = dict(super().architecture_manifest())
        manifest.update(_l4_tpr_manifest_fields())
        manifest["ner_l4_tpr_core_manifest"] = (
            self.ner_l4_tpr.architecture_manifest()
        )
        return manifest


class TPDNERV8MPRSDCHV4QFGV2CROAL4TPRSurvivalSCTransNet(
    _NERL4TPRForwardMixin,
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
        _install_formal_l4_tpr(self)


class TPDNERV8MPRSDCHV4QFGV2CROAL4TPRInferenceSCTransNet(
    _NERL4TPRForwardMixin,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
):
    """Head-free deployment graph retaining trained NER-L4-TPR state."""

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
        _install_formal_l4_tpr(self)


TPD8NER4QFG2L4TPRSurvivalSCTransNet = (
    TPDNERV8MPRSDCHV4QFGV2CROAL4TPRSurvivalSCTransNet
)


def _validate_formal_l4_tpr_model(
    model: nn.Module,
    *,
    training_graph: bool,
    require_zero_initialized_heads: bool,
    require_identity_initialized_qfg: bool,
    require_zero_initialized_l4_tpr: bool,
) -> Dict[str, Any]:
    expected_type = (
        TPDNERV8MPRSDCHV4QFGV2CROAL4TPRSurvivalSCTransNet
        if training_graph
        else TPDNERV8MPRSDCHV4QFGV2CROAL4TPRInferenceSCTransNet
    )
    if type(model) is not expected_type:
        raise TypeError(
            "formal NER-L4-TPR model must use its exact graph class"
        )
    if (
        model.mode != "train"
        or model.deepsuper is not True
        or model.relay_enabled is not True
    ):
        raise RuntimeError(
            "formal NER-L4-TPR requires mode=train, deepsuper, and relay"
        )
    if model.tokenizer_variant != FORMAL_SURVIVAL_VARIANT:
        raise RuntimeError("formal NER-L4-TPR requires Full V8-MPRS-DCH")
    if model.relay_width != DEFAULT_RELAY_WIDTH:
        raise RuntimeError("formal NER-L4-TPR relay width differs")
    if model.relay_initialization_seed != DEFAULT_RELAY_INITIALIZATION_SEED:
        raise RuntimeError("formal NER-L4-TPR relay seed differs")
    if model.tpd_ner.dc_support_mode != DEFAULT_DC_SUPPORT_MODE:
        raise RuntimeError(
            "formal NER-L4-TPR requires NER4 complement-tail support"
        )
    if dict(model.tpd_ner.tail_z_thresholds) != dict(
        DEFAULT_TAIL_Z_THRESHOLDS
    ):
        raise RuntimeError("formal NER-L4-TPR tail thresholds differ")
    context_gate = _formal_context_gate(model)

    if not isinstance(model.tpd_qfg, QueryOnlyFrequencyGateV2CROA):
        raise RuntimeError("formal NER-L4-TPR QFG module type differs")
    qfg_manifest = validate_formal_qfg_v2_croa(
        model.tpd_qfg,
        require_identity_initialization=require_identity_initialized_qfg,
    )
    if tuple(model.tpd_qfg.feature_channels) != FORMAL_QFG_FEATURE_CHANNELS:
        raise RuntimeError("formal NER-L4-TPR QFG channels differ")
    if model.tpd_qfg.mode != FORMAL_QFG_MODE:
        raise RuntimeError("formal NER-L4-TPR QFG mode differs")
    if model.tpd_qfg.hidden_channels != FORMAL_QFG_HIDDEN_CHANNELS:
        raise RuntimeError("formal NER-L4-TPR QFG width differs")
    if model.tpd_qfg.detach_frequency_source is not (
        FORMAL_QFG_DETACH_FREQUENCY_SOURCE
    ):
        raise RuntimeError("formal NER-L4-TPR QFG detach differs")
    if model.tpd_qfg.validate_finite is not FORMAL_QFG_VALIDATE_FINITE:
        raise RuntimeError("formal NER-L4-TPR QFG finite check differs")
    if _parameter_count(model.tpd_qfg) != PRODUCTION_QFG_V2_CROA_PARAMETERS:
        raise RuntimeError("formal NER-L4-TPR QFG parameters differ")

    l4_tpr_manifest = validate_formal_ner_l4_target_protected_reallocation(
        model.ner_l4_tpr,
        require_zero_initialization=require_zero_initialized_l4_tpr,
    )
    reference = next(
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(L4_TPR_STATE_PREFIX)
    )
    for name, parameter in model.ner_l4_tpr.named_parameters():
        if parameter.device != reference.device:
            raise RuntimeError(
                f"formal NER-L4-TPR parameter {name} device differs"
            )
        if parameter.dtype != reference.dtype:
            raise RuntimeError(
                f"formal NER-L4-TPR parameter {name} dtype differs"
            )
    expected_parameters = (
        PRODUCTION_V4_QFG_V2_CROA_L4_TPR_SURVIVAL_PARAMETERS
        if training_graph
        else PRODUCTION_V4_QFG_V2_CROA_L4_TPR_INFERENCE_PARAMETERS
    )
    expected_state_keys = (
        FORMAL_V4_QFG_V2_CROA_L4_TPR_SURVIVAL_STATE_KEY_COUNT
        if training_graph
        else FORMAL_V4_QFG_V2_CROA_L4_TPR_INFERENCE_STATE_KEY_COUNT
    )
    if _parameter_count(model) != expected_parameters:
        raise RuntimeError("formal NER-L4-TPR total parameter count differs")
    state = model.state_dict()
    if len(state) != expected_state_keys:
        raise RuntimeError("formal NER-L4-TPR state-key count differs")
    l4_tpr_keys = {key for key in state if key.startswith(L4_TPR_STATE_PREFIX)}
    if l4_tpr_keys != set(L4_TPR_STATE_KEYS):
        raise RuntimeError("formal NER-L4-TPR extension keys differ")
    qfg_keys = {key for key in state if key.startswith(QFG_STATE_PREFIX)}
    if qfg_keys != set(QFG_STATE_KEYS) or len(qfg_keys) != (
        PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT
    ):
        raise RuntimeError("formal NER-L4-TPR inherited QFG keys differ")

    if training_graph:
        survival_keys = {
            key for key in state if key.startswith(SURVIVAL_STATE_PREFIX)
        }
        if survival_keys != set(SURVIVAL_STATE_KEYS):
            raise RuntimeError("formal NER-L4-TPR Survival keys differ")
        if survival_parameter_count(model.target_survival) != (
            PRODUCTION_SURVIVAL_PARAMETERS
        ):
            raise RuntimeError("formal NER-L4-TPR Survival parameters differ")
        for name, parameter in model.target_survival.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise RuntimeError(
                    f"formal NER-L4-TPR Survival parameter {name} is non-finite"
                )
            if (
                require_zero_initialized_heads
                and torch.count_nonzero(parameter).item() != 0
            ):
                raise RuntimeError(
                    f"formal NER-L4-TPR Survival parameter {name} is not zero"
                )
    else:
        if hasattr(model, "target_survival"):
            raise RuntimeError("NER-L4-TPR inference retains Survival heads")
        if any(key.startswith(SURVIVAL_STATE_PREFIX) for key in state):
            raise RuntimeError("NER-L4-TPR inference retains Survival state")

    manifest = model.architecture_manifest()
    expected_manifest = {
        "relay_version": V4_RELAY_VERSION,
        "qfg_integration_version": QFG_V2_CROA_INTEGRATION_VERSION,
        "l4_tpr_integration_version": L4_TPR_INTEGRATION_VERSION,
        "ner_l4_tpr_version": NER_L4_TPR_VERSION,
        "ner_l4_tpr_enabled": True,
        "ner_l4_tpr_level": 4,
        "ner_l4_tpr_channels": FORMAL_L4_CHANNELS,
        "ner_l4_tpr_gate_limit": FORMAL_L4_GATE_LIMIT,
        "ner_l4_tpr_tail_z_threshold": FORMAL_L4_TAIL_Z_THRESHOLD,
        "ner_l4_tpr_binary_protection": True,
        "ner_l4_tpr_protection_detached": True,
        "ner_l4_tpr_protection_dilation_kernel": (
            FORMAL_L4_PROTECTION_DILATION_KERNEL
        ),
        "ner_l4_tpr_coefficient_sum": 3.0,
        "ner_l4_tpr_coefficient_sum_is_constant": True,
        "ner_l4_tpr_parameters": PRODUCTION_NER_L4_TPR_PARAMETERS,
        "ner_l4_tpr_state_key_count": (
            PRODUCTION_NER_L4_TPR_STATE_KEY_COUNT
        ),
        "ner_l4_tpr_persistent_buffers": 0,
        "ner_l4_tpr_inference_required": True,
        "tpd_formula_changed": False,
        "ner_relay_formula_changed": False,
        "qfg_formula_changed": False,
        "tss_objective_enabled": False,
    }
    for name, value in expected_manifest.items():
        if manifest.get(name) != value:
            raise RuntimeError(
                f"formal NER-L4-TPR manifest field {name!r} differs"
            )

    return {
        "model": f"{type(model).__module__}.{type(model).__name__}",
        "variant": FORMAL_SURVIVAL_VARIANT,
        "relay_version": V4_RELAY_VERSION,
        "qfg_integration_version": QFG_V2_CROA_INTEGRATION_VERSION,
        "l4_tpr_integration_version": L4_TPR_INTEGRATION_VERSION,
        "state_key_count": expected_state_keys,
        "total_parameters": expected_parameters,
        "qfg_state_keys": QFG_STATE_KEYS,
        "qfg_parameters": PRODUCTION_QFG_V2_CROA_PARAMETERS,
        "l4_tpr_state_keys": L4_TPR_STATE_KEYS,
        "l4_tpr_parameters": PRODUCTION_NER_L4_TPR_PARAMETERS,
        "survival_state_keys": SURVIVAL_STATE_KEYS if training_graph else (),
        "survival_parameters": (
            PRODUCTION_SURVIVAL_PARAMETERS if training_graph else 0
        ),
        "target_survival_registered": training_graph,
        "tss_objective_enabled": False,
        "context_gate": context_gate,
        "qfg_core_manifest": qfg_manifest,
        "l4_tpr_core_manifest": l4_tpr_manifest,
        "architecture_manifest": manifest,
    }


def validate_formal_v4_qfg_v2_croa_l4_tpr_survival_model(
    model: nn.Module,
    *,
    require_zero_initialized_heads: bool = False,
    require_identity_initialized_qfg: bool = False,
    require_zero_initialized_l4_tpr: bool = False,
) -> Dict[str, Any]:
    return _validate_formal_l4_tpr_model(
        model,
        training_graph=True,
        require_zero_initialized_heads=require_zero_initialized_heads,
        require_identity_initialized_qfg=require_identity_initialized_qfg,
        require_zero_initialized_l4_tpr=require_zero_initialized_l4_tpr,
    )


def validate_formal_v4_qfg_v2_croa_l4_tpr_inference_model(
    model: nn.Module,
    *,
    require_identity_initialized_qfg: bool = False,
    require_zero_initialized_l4_tpr: bool = False,
) -> Dict[str, Any]:
    return _validate_formal_l4_tpr_model(
        model,
        training_graph=False,
        require_zero_initialized_heads=False,
        require_identity_initialized_qfg=require_identity_initialized_qfg,
        require_zero_initialized_l4_tpr=require_zero_initialized_l4_tpr,
    )


def _build_raw_parent(seed: int) -> Tuple[SCTransNet, Dict[str, Any]]:
    from experiments.train_tpd_clean_v8_mprs_dch import (
        build_clean_v8_mprs_dch_model,
    )

    return build_clean_v8_mprs_dch_model(
        FORMAL_SURVIVAL_VARIANT,
        _require_formal_seed(seed),
    )


def build_formal_v4_qfg_v2_croa_l4_tpr_survival_model(
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> Tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAL4TPRSurvivalSCTransNet,
    Dict[str, Any],
]:
    """Build the authoritative scratch NER-L4-TPR training graph."""

    parent, parent_metadata = _build_raw_parent(seed)
    model = TPDNERV8MPRSDCHV4QFGV2CROAL4TPRSurvivalSCTransNet(
        parent,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    metadata = validate_formal_v4_qfg_v2_croa_l4_tpr_survival_model(
        model,
        require_zero_initialized_heads=True,
        require_identity_initialized_qfg=True,
        require_zero_initialized_l4_tpr=True,
    )
    metadata["construction"] = "scratch_seed42_no_parent_checkpoint"
    metadata["raw_parent_metadata"] = parent_metadata
    return model, metadata


def build_formal_v4_qfg_v2_croa_l4_tpr_inference_model(
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> Tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAL4TPRInferenceSCTransNet,
    Dict[str, Any],
]:
    """Build the authoritative scratch head-free NER-L4-TPR graph."""

    parent, parent_metadata = _build_raw_parent(seed)
    model = TPDNERV8MPRSDCHV4QFGV2CROAL4TPRInferenceSCTransNet(
        parent,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    metadata = validate_formal_v4_qfg_v2_croa_l4_tpr_inference_model(
        model,
        require_identity_initialized_qfg=True,
        require_zero_initialized_l4_tpr=True,
    )
    metadata["construction"] = "scratch_seed42_no_parent_checkpoint"
    metadata["raw_parent_metadata"] = parent_metadata
    return model, metadata


def load_formal_qfg_v2_croa_state_as_zero_l4_tpr_extension(
    model: nn.Module,
    source_state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    """Strictly load current-Final state as the sole one-key zero extension.

    This utility supports model-only identity audits.  Scratch-training
    builders above do not call it and do not authorize checkpoint warm-start.
    """

    if type(model) not in (
        TPDNERV8MPRSDCHV4QFGV2CROAL4TPRSurvivalSCTransNet,
        TPDNERV8MPRSDCHV4QFGV2CROAL4TPRInferenceSCTransNet,
    ):
        raise TypeError(
            "NER-L4-TPR extension load requires an exact integration graph"
        )
    if not isinstance(source_state_dict, Mapping):
        raise TypeError("source_state_dict must be a mapping")
    validate_formal_ner_l4_target_protected_reallocation(
        model.ner_l4_tpr,
        require_zero_initialization=True,
    )
    target_state = model.state_dict()
    expected_shared_keys = set(target_state) - set(L4_TPR_STATE_KEYS)
    if set(source_state_dict) != expected_shared_keys:
        missing = sorted(expected_shared_keys - set(source_state_dict))
        unexpected = sorted(set(source_state_dict) - expected_shared_keys)
        raise ValueError(
            "source is not the exact current-Final state: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, value in source_state_dict.items():
        if not isinstance(name, str) or not name:
            raise TypeError("source state keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"source state {name!r} must be a Tensor")

    incompatible = model.load_state_dict(source_state_dict, strict=False)
    if set(incompatible.missing_keys) != set(L4_TPR_STATE_KEYS):
        raise RuntimeError("NER-L4-TPR extension missing-key set differs")
    if incompatible.unexpected_keys:
        raise RuntimeError("NER-L4-TPR extension returned unexpected keys")
    loaded_state = model.state_dict()
    for name, expected in source_state_dict.items():
        if not torch.equal(loaded_state[name], expected):
            raise RuntimeError(
                f"NER-L4-TPR extension changed shared state {name!r}"
            )
    for name in L4_TPR_STATE_KEYS:
        if torch.count_nonzero(loaded_state[name]).item() != 0:
            raise RuntimeError(
                f"NER-L4-TPR extension state {name!r} is not zero"
            )
    return {
        "load_mode": "strict_model_only_one_key_zero_extension",
        "shared_state_key_count": len(expected_shared_keys),
        "new_state_keys": L4_TPR_STATE_KEYS,
        "new_state_key_count": len(L4_TPR_STATE_KEYS),
        "new_parameters_zero": True,
        "formal_training_warm_start_authorized": False,
    }


__all__ = [
    "FORMAL_V4_QFG_V2_CROA_L4_TPR_INFERENCE_STATE_KEY_COUNT",
    "FORMAL_V4_QFG_V2_CROA_L4_TPR_SURVIVAL_STATE_KEY_COUNT",
    "L4_TPR_INTEGRATION_VERSION",
    "L4_TPR_STATE_KEYS",
    "L4_TPR_STATE_PREFIX",
    "PRODUCTION_V4_QFG_V2_CROA_L4_TPR_INFERENCE_PARAMETERS",
    "PRODUCTION_V4_QFG_V2_CROA_L4_TPR_SURVIVAL_PARAMETERS",
    "TPD8NER4QFG2L4TPRSurvivalSCTransNet",
    "TPDNERV8MPRSDCHV4QFGV2CROAL4TPRInferenceSCTransNet",
    "TPDNERV8MPRSDCHV4QFGV2CROAL4TPRSurvivalSCTransNet",
    "build_formal_v4_qfg_v2_croa_l4_tpr_inference_model",
    "build_formal_v4_qfg_v2_croa_l4_tpr_survival_model",
    "load_formal_qfg_v2_croa_state_as_zero_l4_tpr_extension",
    "validate_formal_v4_qfg_v2_croa_l4_tpr_inference_model",
    "validate_formal_v4_qfg_v2_croa_l4_tpr_survival_model",
]
