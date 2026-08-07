"""PBDR-V3 integration for TPD8 + NER4 + QFG2-CROA.

The inherited Current graph is preserved through the final decoder feature
``u1`` and raw readouts ``out``/``d0``.  A bounded twin-gate calibrator uses
those tensors plus detached q4 evidence as context.  Ordinary ``forward``
retains the evaluator contract: six full-resolution probabilities in training
mode and one routed probability in test mode.  The explicit
``forward_for_pbdr_v3_training`` method returns raw logits and routing
diagnostics without disguising them as extra segmentation maps.

This graph is intentionally constructed as an extension shell.  Formal use
must warm-start all non-``pbdr_v3`` state from a trained Current checkpoint;
scratch construction is not a valid PBDR-V3 experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SCTransNet import SCTransNet
from model.tpd_clean_v8_mprs_dch import clean_v8_mprs_dch_variant_spec
from model.tpd_conservative_residual_calibrator_v3 import (
    ConservativeResidualCalibratorV3,
    FORMAL_DETACH_LOCAL_FEATURE,
    FORMAL_EVIDENCE_FLOOR,
    FORMAL_GATE_BIAS_INIT,
    FORMAL_HIDDEN_CHANNELS,
    FORMAL_LOCAL_CHANNELS,
    FORMAL_Q4_CHANNELS,
    FORMAL_RESIDUAL_LIMIT,
    FORMAL_UNCERTAINTY_FLOOR,
    PBDR_V3_LOCAL_STATE_KEYS,
    PBDR_V3_VERSION,
    PBDRV3RoutingOutput,
    PRODUCTION_PBDR_V3_BUFFER_COUNT,
    PRODUCTION_PBDR_V3_PARAMETERS,
    PRODUCTION_PBDR_V3_STATE_KEY_COUNT,
    validate_formal_pbdr_v3_calibrator,
)
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
from model.tpd_query_frequency_bridge import frequency_encoder_forward
from model.tpd_survival import survival_parameter_count


PBDR_V3_INTEGRATION_VERSION = "v4_qfg_v2_croa_pbdr_v3_v1"
PBDR_V3_STATE_PREFIX = "pbdr_v3."
PBDR_V3_STATE_KEYS = tuple(
    f"{PBDR_V3_STATE_PREFIX}{key}" for key in PBDR_V3_LOCAL_STATE_KEYS
)
FORMAL_PBDR_V3_INITIALIZATION_SEED = 42

PRODUCTION_V4_QFG_V2_CROA_PBDR_V3_SURVIVAL_PARAMETERS = (
    PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS
    + PRODUCTION_PBDR_V3_PARAMETERS
)
FORMAL_V4_QFG_V2_CROA_PBDR_V3_SURVIVAL_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT
    + PRODUCTION_PBDR_V3_STATE_KEY_COUNT
)
PRODUCTION_V4_QFG_V2_CROA_PBDR_V3_INFERENCE_PARAMETERS = (
    PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS
    + PRODUCTION_PBDR_V3_PARAMETERS
)
FORMAL_V4_QFG_V2_CROA_PBDR_V3_INFERENCE_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT
    + PRODUCTION_PBDR_V3_STATE_KEY_COUNT
)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _require_formal_seed(seed: int) -> int:
    if type(seed) is not int or seed != FORMAL_SURVIVAL_INITIALIZATION_SEED:
        raise ValueError("formal PBDR-V3 construction requires seed=42")
    return seed


def _install_formal_pbdr_v3(model: nn.Module) -> None:
    if hasattr(model, "pbdr_v3"):
        raise RuntimeError("PBDR-V3 integration attempted twice")
    local_channels = int(model.outc.in_channels)
    if local_channels != FORMAL_LOCAL_CHANNELS:
        raise RuntimeError(
            "formal PBDR-V3 requires a 32-channel final decoder feature"
        )

    # Isolate initialization so extending a Current graph does not perturb the
    # caller's RNG stream.  Only the hidden context projections are random;
    # the terminal twin-gate projection is exactly identity initialized.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(
            FORMAL_PBDR_V3_INITIALIZATION_SEED
        )
        calibrator = ConservativeResidualCalibratorV3(
            q_channels=FORMAL_Q4_CHANNELS,
            local_channels=local_channels,
            hidden_channels=FORMAL_HIDDEN_CHANNELS,
            residual_limit=FORMAL_RESIDUAL_LIMIT,
            evidence_floor=FORMAL_EVIDENCE_FLOOR,
            uncertainty_floor=FORMAL_UNCERTAINTY_FLOOR,
            gate_bias_init=FORMAL_GATE_BIAS_INIT,
            detach_local_feature=FORMAL_DETACH_LOCAL_FEATURE,
        )
    reference = next(model.parameters())
    calibrator.to(device=reference.device, dtype=reference.dtype)
    calibrator.train(model.training)
    model.pbdr_v3 = calibrator
    validate_formal_pbdr_v3_calibrator(
        model.pbdr_v3,
        require_identity_initialization=True,
    )


def _pbdr_v3_manifest_fields() -> Dict[str, Any]:
    return {
        "pbdr_v3_integration_version": PBDR_V3_INTEGRATION_VERSION,
        "pbdr_v3_version": PBDR_V3_VERSION,
        "pbdr_v3_enabled": True,
        "pbdr_v3_inputs": (
            "final_decoder_feature_u1",
            "raw_q4",
            "raw_out",
            "raw_d0",
        ),
        "pbdr_v3_location": "after_d0_before_final_sigmoid",
        "pbdr_v3_current_checkpoint_warm_start_required": True,
        "pbdr_v3_scratch_training_supported": False,
        "pbdr_v3_stage1_base_frozen_required": True,
        "pbdr_v3_stage1_batchnorm_statistics_frozen_required": True,
        "pbdr_v3_q4_role": "detached_safely_normalized_context_only",
        "pbdr_v3_d0_role": "detached_probability_context_only",
        "pbdr_v3_local_feature_role": "detached_full_resolution_context",
        "pbdr_v3_direct_q4_residual": False,
        "pbdr_v3_direct_d0_residual": False,
        "pbdr_v3_gate_mapping": "two_independent_nonnegative_sigmoids",
        "pbdr_v3_delta_formula": "L*B*(G_rescue-G_suppression)",
        "pbdr_v3_residual_limit": FORMAL_RESIDUAL_LIMIT,
        "pbdr_v3_evidence_floor": FORMAL_EVIDENCE_FLOOR,
        "pbdr_v3_uncertainty_floor": FORMAL_UNCERTAINTY_FLOOR,
        "pbdr_v3_gate_bias_initialization": FORMAL_GATE_BIAS_INIT,
        "pbdr_v3_interpolation": "bilinear_align_corners_false",
        "pbdr_v3_zero_anchor": "current_final_exact",
        "pbdr_v3_parameters": PRODUCTION_PBDR_V3_PARAMETERS,
        "pbdr_v3_state_prefix": PBDR_V3_STATE_PREFIX,
        "pbdr_v3_state_key_count": PRODUCTION_PBDR_V3_STATE_KEY_COUNT,
        "pbdr_v3_persistent_buffers": PRODUCTION_PBDR_V3_BUFFER_COUNT,
        "pbdr_v3_inference_required": True,
        "pbdr_v3_training_interface": "forward_for_pbdr_v3_training",
        "tpd_formula_changed": False,
        "ner_relay_formula_changed": False,
        "qfg_formula_changed": False,
        "tss_objective_enabled": False,
        "segmentation_path_modified": True,
        "segmentation_path_modification": (
            "qfg_query_modulation_plus_pbdr_v3_bounded_final_calibration"
        ),
    }


@dataclass(frozen=True, slots=True)
class PBDRV3TrainingAux:
    """Raw single-forward tensors for the dedicated PBDR-V3 objective."""

    auxiliary_logits: tuple[torch.Tensor, ...]
    base_logits: torch.Tensor
    routed_logits: torch.Tensor
    routing: PBDRV3RoutingOutput

    def __post_init__(self) -> None:
        if len(self.auxiliary_logits) != 5:
            raise ValueError("PBDR-V3 requires exactly five auxiliary logits")
        reference_shape = tuple(self.base_logits.shape)
        values = self.auxiliary_logits + (self.routed_logits,)
        if any(tuple(value.shape) != reference_shape for value in values):
            raise ValueError("all PBDR-V3 training logits must share one shape")
        if self.routing.routed_logits is not self.routed_logits:
            raise ValueError("routing and training routed logits must coincide")


PBDRV3ProbabilityTuple = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
PBDRV3TrainingReturn = tuple[PBDRV3ProbabilityTuple, PBDRV3TrainingAux]


class _PBDRV3ForwardMixin:
    """Shared training/deployment forward with conservative final routing."""

    pbdr_v3: ConservativeResidualCalibratorV3

    def _pbdr_v3_forward_impl(
        self,
        x: torch.Tensor,
    ) -> PBDRV3TrainingReturn:
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
        u1 = self.up_decoder1(d2, x1)
        out = self.outc(u1)

        if not self.deepsuper:
            raise RuntimeError("PBDR-V3 requires deepsuper=True to produce d0")
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
        routing = self.pbdr_v3.forward_with_diagnostics(
            z_out=out,
            z_d0=d0,
            q4=q4,
            local_feature=u1,
        )
        routed_out = routing.routed_logits
        probabilities: PBDRV3ProbabilityTuple = (
            torch.sigmoid(gt5),
            torch.sigmoid(gt4),
            torch.sigmoid(gt3),
            torch.sigmoid(gt2),
            torch.sigmoid(d0),
            torch.sigmoid(routed_out),
        )
        auxiliary_logits = (gt5, gt4, gt3, gt2, d0)
        return probabilities, PBDRV3TrainingAux(
            auxiliary_logits=auxiliary_logits,
            base_logits=out,
            routed_logits=routed_out,
            routing=routing,
        )

    def forward_for_pbdr_v3_training(
        self,
        x: torch.Tensor,
    ) -> PBDRV3TrainingReturn:
        """Return probabilities and raw training-only data in one forward."""

        return self._pbdr_v3_forward_impl(x)

    def _forward_with_relay(self, x: torch.Tensor):
        probabilities, auxiliary = self._pbdr_v3_forward_impl(x)
        if self.mode != "train":
            return torch.sigmoid(auxiliary.routed_logits)
        return probabilities

    def architecture_manifest(self) -> Dict[str, Any]:
        manifest = dict(super().architecture_manifest())
        manifest.update(_pbdr_v3_manifest_fields())
        manifest["deployment_graph"] = (
            "v4_qfg_v2_croa_pbdr_v3_with_training_only_tss_heads"
            if hasattr(self, "target_survival")
            else "v4_qfg_v2_croa_pbdr_v3_no_tss"
        )
        manifest["pbdr_v3_core_manifest"] = (
            self.pbdr_v3.architecture_manifest()
        )
        return manifest


class TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3SurvivalSCTransNet(
    _PBDRV3ForwardMixin,
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
):
    """Warm-start training graph retaining Current's training-only heads."""

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
        _install_formal_pbdr_v3(self)


class TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3InferenceSCTransNet(
    _PBDRV3ForwardMixin,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
):
    """Head-free deployment graph retaining trained PBDR-V3 state."""

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
        _install_formal_pbdr_v3(self)


TPD8NER4QFG2PBDRV3SurvivalSCTransNet = (
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3SurvivalSCTransNet
)


def _validate_formal_pbdr_v3_model(
    model: nn.Module,
    *,
    training_graph: bool,
    require_zero_initialized_heads: bool,
    require_identity_initialized_qfg: bool,
    require_identity_initialized_pbdr_v3: bool,
) -> Dict[str, Any]:
    expected_type = (
        TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3SurvivalSCTransNet
        if training_graph
        else TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3InferenceSCTransNet
    )
    if type(model) is not expected_type:
        raise TypeError("formal PBDR-V3 model must use its exact graph class")
    if (
        model.mode != "train"
        or model.deepsuper is not True
        or model.relay_enabled is not True
    ):
        raise RuntimeError(
            "formal PBDR-V3 requires mode=train, deepsuper, and relay"
        )
    if model.tokenizer_variant != FORMAL_SURVIVAL_VARIANT:
        raise RuntimeError("formal PBDR-V3 requires Full V8-MPRS-DCH")
    if model.relay_width != DEFAULT_RELAY_WIDTH:
        raise RuntimeError("formal PBDR-V3 relay width differs")
    if model.relay_initialization_seed != DEFAULT_RELAY_INITIALIZATION_SEED:
        raise RuntimeError("formal PBDR-V3 relay seed differs")
    if model.tpd_ner.dc_support_mode != DEFAULT_DC_SUPPORT_MODE:
        raise RuntimeError("formal PBDR-V3 requires complement-tail NER4")
    if dict(model.tpd_ner.tail_z_thresholds) != dict(DEFAULT_TAIL_Z_THRESHOLDS):
        raise RuntimeError("formal PBDR-V3 tail thresholds differ")
    context_gate = _formal_context_gate(model)

    if not isinstance(model.tpd_qfg, QueryOnlyFrequencyGateV2CROA):
        raise RuntimeError("formal PBDR-V3 QFG module type differs")
    qfg_manifest = validate_formal_qfg_v2_croa(
        model.tpd_qfg,
        require_identity_initialization=require_identity_initialized_qfg,
    )
    if tuple(model.tpd_qfg.feature_channels) != FORMAL_QFG_FEATURE_CHANNELS:
        raise RuntimeError("formal PBDR-V3 QFG channels differ")
    if model.tpd_qfg.mode != FORMAL_QFG_MODE:
        raise RuntimeError("formal PBDR-V3 QFG mode differs")
    if model.tpd_qfg.hidden_channels != FORMAL_QFG_HIDDEN_CHANNELS:
        raise RuntimeError("formal PBDR-V3 QFG width differs")
    if model.tpd_qfg.validate_finite is not FORMAL_QFG_VALIDATE_FINITE:
        raise RuntimeError("formal PBDR-V3 QFG finite setting differs")
    if _parameter_count(model.tpd_qfg) != PRODUCTION_QFG_V2_CROA_PARAMETERS:
        raise RuntimeError("formal PBDR-V3 QFG parameter count differs")

    pbdr_manifest = validate_formal_pbdr_v3_calibrator(
        model.pbdr_v3,
        require_identity_initialization=(
            require_identity_initialized_pbdr_v3
        ),
    )
    state = model.state_dict()
    pbdr_keys = tuple(
        key for key in state if key.startswith(PBDR_V3_STATE_PREFIX)
    )
    if pbdr_keys != PBDR_V3_STATE_KEYS:
        raise RuntimeError("formal integrated PBDR-V3 state keys differ")
    expected_state_count = (
        FORMAL_V4_QFG_V2_CROA_PBDR_V3_SURVIVAL_STATE_KEY_COUNT
        if training_graph
        else FORMAL_V4_QFG_V2_CROA_PBDR_V3_INFERENCE_STATE_KEY_COUNT
    )
    expected_parameter_count = (
        PRODUCTION_V4_QFG_V2_CROA_PBDR_V3_SURVIVAL_PARAMETERS
        if training_graph
        else PRODUCTION_V4_QFG_V2_CROA_PBDR_V3_INFERENCE_PARAMETERS
    )
    if len(state) != expected_state_count:
        raise RuntimeError("formal integrated PBDR-V3 state count differs")
    if _parameter_count(model) != expected_parameter_count:
        raise RuntimeError("formal integrated PBDR-V3 parameter count differs")
    qfg_keys = tuple(key for key in state if key.startswith(QFG_STATE_PREFIX))
    if set(qfg_keys) != set(QFG_STATE_KEYS):
        raise RuntimeError("formal integrated QFG state keys differ")

    survival_keys = tuple(
        key for key in state if key.startswith(SURVIVAL_STATE_PREFIX)
    )
    if training_graph:
        if set(survival_keys) != set(SURVIVAL_STATE_KEYS):
            raise RuntimeError("formal training graph Survival keys differ")
        if (
            survival_parameter_count(model.target_survival)
            != PRODUCTION_SURVIVAL_PARAMETERS
        ):
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
        if not name.startswith(PBDR_V3_STATE_PREFIX)
    )
    for name, parameter in model.pbdr_v3.named_parameters():
        if parameter.device != reference.device:
            raise RuntimeError(f"PBDR-V3 parameter {name} device differs")
        if parameter.dtype != reference.dtype:
            raise RuntimeError(f"PBDR-V3 parameter {name} dtype differs")
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError(f"PBDR-V3 parameter {name} is non-finite")

    manifest = model.architecture_manifest()
    expected_manifest = {
        "relay_version": V4_RELAY_VERSION,
        "qfg_integration_version": QFG_V2_CROA_INTEGRATION_VERSION,
        "pbdr_v3_integration_version": PBDR_V3_INTEGRATION_VERSION,
        "pbdr_v3_version": PBDR_V3_VERSION,
        "pbdr_v3_enabled": True,
        "pbdr_v3_parameters": PRODUCTION_PBDR_V3_PARAMETERS,
        "pbdr_v3_state_key_count": PRODUCTION_PBDR_V3_STATE_KEY_COUNT,
        "pbdr_v3_zero_anchor": "current_final_exact",
        "pbdr_v3_current_checkpoint_warm_start_required": True,
        "tss_objective_enabled": False,
        "segmentation_path_modified": True,
        "deployment_graph": (
            "v4_qfg_v2_croa_pbdr_v3_with_training_only_tss_heads"
            if training_graph
            else "v4_qfg_v2_croa_pbdr_v3_no_tss"
        ),
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            raise RuntimeError(f"formal PBDR-V3 manifest field {name!r} differs")
    return {
        "model": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "variant": FORMAL_SURVIVAL_VARIANT,
        "relay_version": V4_RELAY_VERSION,
        "qfg_integration_version": QFG_V2_CROA_INTEGRATION_VERSION,
        "pbdr_v3_integration_version": PBDR_V3_INTEGRATION_VERSION,
        "state_key_count": expected_state_count,
        "pbdr_v3_state_keys": PBDR_V3_STATE_KEYS,
        "pbdr_v3_parameters": PRODUCTION_PBDR_V3_PARAMETERS,
        "total_parameters": expected_parameter_count,
        "current_warm_start_expected_missing_keys": PBDR_V3_STATE_KEYS,
        "context_gate": context_gate,
        "qfg_core_manifest": qfg_manifest,
        "pbdr_v3_core_manifest": pbdr_manifest,
        "target_survival_registered": training_graph,
        "architecture_manifest": manifest,
    }


def validate_formal_v4_qfg_v2_croa_pbdr_v3_survival_model(
    model: nn.Module,
    *,
    require_zero_initialized_heads: bool = False,
    require_identity_initialized_qfg: bool = False,
    require_identity_initialized_pbdr_v3: bool = False,
) -> Dict[str, Any]:
    return _validate_formal_pbdr_v3_model(
        model,
        training_graph=True,
        require_zero_initialized_heads=require_zero_initialized_heads,
        require_identity_initialized_qfg=require_identity_initialized_qfg,
        require_identity_initialized_pbdr_v3=(
            require_identity_initialized_pbdr_v3
        ),
    )


def validate_formal_v4_qfg_v2_croa_pbdr_v3_inference_model(
    model: nn.Module,
    *,
    require_identity_initialized_qfg: bool = False,
    require_identity_initialized_pbdr_v3: bool = False,
) -> Dict[str, Any]:
    return _validate_formal_pbdr_v3_model(
        model,
        training_graph=False,
        require_zero_initialized_heads=False,
        require_identity_initialized_qfg=require_identity_initialized_qfg,
        require_identity_initialized_pbdr_v3=(
            require_identity_initialized_pbdr_v3
        ),
    )


def _build_raw_parent(seed: int) -> tuple[SCTransNet, Dict[str, Any]]:
    from experiments.train_tpd_clean_v8_mprs_dch import (
        build_clean_v8_mprs_dch_model,
    )

    return build_clean_v8_mprs_dch_model(
        FORMAL_SURVIVAL_VARIANT,
        _require_formal_seed(seed),
    )


def build_formal_v4_qfg_v2_croa_pbdr_v3_survival_model(
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> Tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3SurvivalSCTransNet,
    Dict[str, Any],
]:
    parent, parent_metadata = _build_raw_parent(seed)
    model = TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3SurvivalSCTransNet(
        parent,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    metadata = validate_formal_v4_qfg_v2_croa_pbdr_v3_survival_model(
        model,
        require_zero_initialized_heads=True,
        require_identity_initialized_qfg=True,
        require_identity_initialized_pbdr_v3=True,
    )
    metadata.update(
        {
            "construction": "seed42_graph_requires_trained_current_warm_start",
            "warm_start_required": True,
            "raw_parent_metadata": parent_metadata,
        }
    )
    return model, metadata


def build_formal_v4_qfg_v2_croa_pbdr_v3_inference_model(
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> Tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3InferenceSCTransNet,
    Dict[str, Any],
]:
    parent, parent_metadata = _build_raw_parent(seed)
    model = TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3InferenceSCTransNet(
        parent,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    metadata = validate_formal_v4_qfg_v2_croa_pbdr_v3_inference_model(
        model,
        require_identity_initialized_qfg=True,
        require_identity_initialized_pbdr_v3=True,
    )
    metadata.update(
        {
            "construction": "seed42_graph_requires_trained_current_warm_start",
            "warm_start_required": True,
            "raw_parent_metadata": parent_metadata,
        }
    )
    return model, metadata


__all__ = [
    "FORMAL_PBDR_V3_INITIALIZATION_SEED",
    "FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT",
    "FORMAL_V4_QFG_V2_CROA_PBDR_V3_INFERENCE_STATE_KEY_COUNT",
    "FORMAL_V4_QFG_V2_CROA_PBDR_V3_SURVIVAL_STATE_KEY_COUNT",
    "FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT",
    "PBDR_V3_INTEGRATION_VERSION",
    "PBDR_V3_STATE_KEYS",
    "PBDR_V3_STATE_PREFIX",
    "PBDRV3ProbabilityTuple",
    "PBDRV3TrainingAux",
    "PBDRV3TrainingReturn",
    "PRODUCTION_PBDR_V3_PARAMETERS",
    "PRODUCTION_V4_QFG_V2_CROA_PBDR_V3_INFERENCE_PARAMETERS",
    "PRODUCTION_V4_QFG_V2_CROA_PBDR_V3_SURVIVAL_PARAMETERS",
    "SURVIVAL_STATE_KEYS",
    "TPD8NER4QFG2PBDRV3SurvivalSCTransNet",
    "TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3InferenceSCTransNet",
    "TPDNERV8MPRSDCHV4QFGV2CROAPBDRV3SurvivalSCTransNet",
    "build_formal_v4_qfg_v2_croa_pbdr_v3_inference_model",
    "build_formal_v4_qfg_v2_croa_pbdr_v3_survival_model",
    "validate_formal_v4_qfg_v2_croa_pbdr_v3_inference_model",
    "validate_formal_v4_qfg_v2_croa_pbdr_v3_survival_model",
]
