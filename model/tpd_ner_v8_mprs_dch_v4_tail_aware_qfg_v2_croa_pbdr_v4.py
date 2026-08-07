"""Role-explicit PBDR-V4 integration for TPD8 + NER4 + QFG2-CROA.

The V4 graph is an independent extension of the trained Current graph.  It
preserves the legacy segmentation boundary (six probabilities in training
mode, one probability in test mode) while exposing named raw logits through a
dedicated training method.  ``gt2`` through ``gt5`` are always passed to the
calibrator as explicit named arguments; their semantic order is never inferred
from a sequence.

Formal construction, validation, warm-start and inference export all require
an explicit role.  Candidate export accepts a complete checkpoint payload and
validates its schema, role, stage, architecture manifest and state contract
against the immutable Current parent before producing an inference graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.pbdr_v4_state_contract import (
    Stage,
    audit_candidate_against_current,
)
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
from model.tpd_role_aligned_residual_calibrator_v4 import (
    FORMAL_DEBUG_VALIDATE_FINITE,
    FORMAL_DETACH_CONTEXT,
    FORMAL_EVIDENCE_FLOOR,
    FORMAL_HIDDEN_CHANNELS,
    FORMAL_LOCAL_CHANNELS,
    FORMAL_NEGATIVE_LIMITS,
    FORMAL_POSITIVE_LIMITS,
    FORMAL_Q4_CHANNELS,
    PBDR_V4_LOCAL_STATE_KEYS,
    PBDR_V4_VERSION,
    PBDRV4RoutingOutput,
    PRODUCTION_PBDR_V4_BUFFER_COUNT,
    PRODUCTION_PBDR_V4_PARAMETERS,
    PRODUCTION_PBDR_V4_STATE_KEY_COUNT,
    ROLE_CODES,
    Role,
    RoleAlignedResidualCalibratorV4,
    validate_formal_pbdr_v4_calibrator,
)
from model.tpd_survival import survival_parameter_count


PBDR_V4_INTEGRATION_VERSION = "v4_qfg_v2_croa_pbdr_v4_v1"
PBDR_V4_STATE_PREFIX = "pbdr_v4."
PBDR_V4_STATE_KEYS = tuple(
    f"{PBDR_V4_STATE_PREFIX}{key}" for key in PBDR_V4_LOCAL_STATE_KEYS
)
FORMAL_PBDR_V4_INITIALIZATION_SEED = 42
PBDR_V4_CANDIDATE_CHECKPOINT_SCHEMA = (
    "sctransnet_pbdr_v4_candidate_checkpoint/v1"
)
SUPPORTED_STAGES: tuple[Stage, ...] = ("stage1", "stage2")

PRODUCTION_V4_QFG_V2_CROA_PBDR_V4_SURVIVAL_PARAMETERS = (
    PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS
    + PRODUCTION_PBDR_V4_PARAMETERS
)
FORMAL_V4_QFG_V2_CROA_PBDR_V4_SURVIVAL_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT
    + PRODUCTION_PBDR_V4_STATE_KEY_COUNT
)
PRODUCTION_V4_QFG_V2_CROA_PBDR_V4_INFERENCE_PARAMETERS = (
    PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS
    + PRODUCTION_PBDR_V4_PARAMETERS
)
FORMAL_V4_QFG_V2_CROA_PBDR_V4_INFERENCE_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT
    + PRODUCTION_PBDR_V4_STATE_KEY_COUNT
)


class PBDRV4IntegrationError(RuntimeError):
    """A formal graph, warm-start, or checkpoint violates the V4 contract."""


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _require_role(role: str) -> Role:
    if role not in ROLE_CODES:
        raise PBDRV4IntegrationError(f"unsupported PBDR-V4 role: {role!r}")
    return role  # type: ignore[return-value]


def _require_stage(stage: str) -> Stage:
    if stage not in SUPPORTED_STAGES:
        raise PBDRV4IntegrationError(f"unsupported PBDR-V4 stage: {stage!r}")
    return stage  # type: ignore[return-value]


def _require_formal_seed(seed: int) -> int:
    if type(seed) is not int or seed != FORMAL_SURVIVAL_INITIALIZATION_SEED:
        raise ValueError("formal PBDR-V4 construction requires seed=42")
    return seed


def _install_formal_pbdr_v4(model: nn.Module, *, role: Role) -> None:
    ready_role = _require_role(role)
    if hasattr(model, "pbdr_v4"):
        raise PBDRV4IntegrationError("PBDR-V4 integration attempted twice")
    local_channels = int(model.outc.in_channels)
    if local_channels != FORMAL_LOCAL_CHANNELS:
        raise PBDRV4IntegrationError(
            "formal PBDR-V4 requires a 32-channel final decoder feature"
        )

    # Extending a Current graph must not perturb the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(FORMAL_PBDR_V4_INITIALIZATION_SEED)
        calibrator = RoleAlignedResidualCalibratorV4(
            role=ready_role,
            q_channels=FORMAL_Q4_CHANNELS,
            local_channels=local_channels,
            hidden_channels=FORMAL_HIDDEN_CHANNELS,
            positive_limit=FORMAL_POSITIVE_LIMITS[ready_role],
            negative_limit=FORMAL_NEGATIVE_LIMITS[ready_role],
            evidence_floor=FORMAL_EVIDENCE_FLOOR,
            detach_context=FORMAL_DETACH_CONTEXT,
            debug_validate_finite=FORMAL_DEBUG_VALIDATE_FINITE,
        )
    reference = next(model.parameters())
    calibrator.to(device=reference.device, dtype=reference.dtype)
    calibrator.train(model.training)
    model.pbdr_v4 = calibrator
    model.pbdr_v4_role = ready_role
    validate_formal_pbdr_v4_calibrator(
        model.pbdr_v4,
        expected_role=ready_role,
        require_identity_initialization=True,
    )


def _pbdr_v4_manifest_fields(
    role: Role,
    calibrator: RoleAlignedResidualCalibratorV4,
) -> Dict[str, Any]:
    core = calibrator.architecture_manifest()
    return {
        "pbdr_v4_integration_version": PBDR_V4_INTEGRATION_VERSION,
        "pbdr_v4_version": PBDR_V4_VERSION,
        "pbdr_v4_enabled": True,
        "pbdr_v4_role": role,
        "pbdr_v4_role_code": core["role_code"],
        "pbdr_v4_positive_limit": core["positive_limit"],
        "pbdr_v4_negative_limit": core["negative_limit"],
        "pbdr_v4_inputs": (
            "raw_out",
            "raw_d0",
            "raw_gt2",
            "raw_gt3",
            "raw_gt4",
            "raw_gt5",
            "raw_q4",
            "final_decoder_feature_u1",
        ),
        "pbdr_v4_auxiliary_logit_order": ("gt2", "gt3", "gt4", "gt5"),
        "pbdr_v4_auxiliary_interface": "explicit_named_arguments",
        "pbdr_v4_location": "after_d0_before_final_sigmoid",
        "pbdr_v4_current_checkpoint_warm_start_required": True,
        "pbdr_v4_scratch_training_supported": False,
        "pbdr_v4_stage1_base_frozen_required": True,
        "pbdr_v4_stage2_mutable_base_parameter_prefixes": (
            "outc.",
            "up_decoder1.",
        ),
        "pbdr_v4_all_base_buffers_current_required": True,
        "pbdr_v4_q4_role": "detached_safely_normalized_context_only",
        "pbdr_v4_d0_role": "detached_probability_context_only",
        "pbdr_v4_deep_supervision_role": "detached_named_probability_context",
        "pbdr_v4_local_feature_role": "detached_full_resolution_context",
        "pbdr_v4_zero_anchor": "current_final_exact",
        "pbdr_v4_parameters": PRODUCTION_PBDR_V4_PARAMETERS,
        "pbdr_v4_state_prefix": PBDR_V4_STATE_PREFIX,
        "pbdr_v4_state_key_count": PRODUCTION_PBDR_V4_STATE_KEY_COUNT,
        "pbdr_v4_persistent_buffers": PRODUCTION_PBDR_V4_BUFFER_COUNT,
        "pbdr_v4_persistent_semantics": (
            "role_code",
            "positive_limit",
            "negative_limit",
        ),
        "pbdr_v4_inference_required": True,
        "pbdr_v4_training_interface": "forward_for_pbdr_v4_training",
        "tpd_formula_changed": False,
        "ner_relay_formula_changed": False,
        "qfg_formula_changed": False,
        "tss_objective_enabled": False,
        "segmentation_path_modified": True,
        "segmentation_path_modification": (
            "qfg_query_modulation_plus_pbdr_v4_role_aligned_calibration"
        ),
    }


@dataclass(frozen=True, slots=True)
class PBDRV4TrainingAux:
    """Named raw tensors from one V4 forward; none are probabilities."""

    gt2_logits: torch.Tensor
    gt3_logits: torch.Tensor
    gt4_logits: torch.Tensor
    gt5_logits: torch.Tensor
    d0_logits: torch.Tensor
    candidate_base_logits: torch.Tensor
    routed_logits: torch.Tensor
    delta_logits: torch.Tensor
    routing: PBDRV4RoutingOutput

    def __post_init__(self) -> None:
        reference = self.candidate_base_logits
        values = (
            self.gt2_logits,
            self.gt3_logits,
            self.gt4_logits,
            self.gt5_logits,
            self.d0_logits,
            self.routed_logits,
            self.delta_logits,
        )
        if any(tuple(value.shape) != tuple(reference.shape) for value in values):
            raise ValueError("all PBDR-V4 training logits must share one shape")
        if any(value.dtype != reference.dtype for value in values):
            raise ValueError("all PBDR-V4 training logits must share one dtype")
        if any(value.device != reference.device for value in values):
            raise ValueError("all PBDR-V4 training logits must share one device")
        if self.routing.routed_logits is not self.routed_logits:
            raise ValueError("routing and auxiliary routed logits must coincide")
        if self.routing.delta_logits is not self.delta_logits:
            raise ValueError("routing and auxiliary delta logits must coincide")

    def ordered_deep_supervision_logits(self) -> tuple[torch.Tensor, ...]:
        """Return the single canonical context order: gt2, gt3, gt4, gt5."""

        return (
            self.gt2_logits,
            self.gt3_logits,
            self.gt4_logits,
            self.gt5_logits,
        )


PBDRV4ProbabilityTuple = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
PBDRV4TrainingReturn = tuple[PBDRV4ProbabilityTuple, PBDRV4TrainingAux]


class _PBDRV4ForwardMixin:
    """Shared survival/inference forward with role-aligned final routing."""

    pbdr_v4: RoleAlignedResidualCalibratorV4
    pbdr_v4_role: Role

    def _pbdr_v4_forward_impl(self, x: torch.Tensor) -> PBDRV4TrainingReturn:
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
        encoded1, encoded2, encoded3, encoded4, _ = frequency_encoder_forward(
            self.mtc.encoder,
            emb1,
            emb2,
            emb3,
            emb4,
            self.tpd_qfg,
            prepared_qfg,
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
            raise RuntimeError("PBDR-V4 requires deepsuper=True to produce d0")
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
        routing = self.pbdr_v4.forward_with_diagnostics(
            z_out=out,
            z_d0=d0,
            z_gt2=gt2,
            z_gt3=gt3,
            z_gt4=gt4,
            z_gt5=gt5,
            q4=q4,
            local_feature=u1,
        )
        routed_out = routing.routed_logits
        probabilities: PBDRV4ProbabilityTuple = (
            torch.sigmoid(gt5),
            torch.sigmoid(gt4),
            torch.sigmoid(gt3),
            torch.sigmoid(gt2),
            torch.sigmoid(d0),
            torch.sigmoid(routed_out),
        )
        return probabilities, PBDRV4TrainingAux(
            gt2_logits=gt2,
            gt3_logits=gt3,
            gt4_logits=gt4,
            gt5_logits=gt5,
            d0_logits=d0,
            candidate_base_logits=out,
            routed_logits=routed_out,
            delta_logits=routing.delta_logits,
            routing=routing,
        )

    def forward_for_pbdr_v4_training(
        self,
        x: torch.Tensor,
    ) -> PBDRV4TrainingReturn:
        """Return the legacy probabilities and named raw V4 tensors."""

        return self._pbdr_v4_forward_impl(x)

    def _forward_with_relay(self, x: torch.Tensor):
        probabilities, auxiliary = self._pbdr_v4_forward_impl(x)
        if self.mode != "train":
            return torch.sigmoid(auxiliary.routed_logits)
        return probabilities

    def architecture_manifest(self) -> Dict[str, Any]:
        manifest = dict(super().architecture_manifest())
        manifest.update(_pbdr_v4_manifest_fields(self.pbdr_v4_role, self.pbdr_v4))
        manifest["deployment_graph"] = (
            "v4_qfg_v2_croa_pbdr_v4_with_training_only_tss_heads"
            if hasattr(self, "target_survival")
            else "v4_qfg_v2_croa_pbdr_v4_no_tss"
        )
        manifest["pbdr_v4_core_manifest"] = self.pbdr_v4.architecture_manifest()
        return manifest


class TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4SurvivalSCTransNet(
    _PBDRV4ForwardMixin,
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
):
    """Warm-start training graph retaining Current's training-only heads."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        role: Role,
        variant: str,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode: str = DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds: Mapping[int, float] = DEFAULT_TAIL_Z_THRESHOLDS,
    ) -> None:
        ready_role = _require_role(role)
        super().__init__(
            parent,
            variant=variant,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
            dc_support_mode=dc_support_mode,
            tail_z_thresholds=tail_z_thresholds,
        )
        _install_formal_pbdr_v4(self, role=ready_role)


class TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4InferenceSCTransNet(
    _PBDRV4ForwardMixin,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
):
    """Head-free deployment graph retaining trained PBDR-V4 state."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        role: Role,
        variant: str,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode: str = DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds: Mapping[int, float] = DEFAULT_TAIL_Z_THRESHOLDS,
    ) -> None:
        ready_role = _require_role(role)
        super().__init__(
            parent,
            variant=variant,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
            dc_support_mode=dc_support_mode,
            tail_z_thresholds=tail_z_thresholds,
        )
        _install_formal_pbdr_v4(self, role=ready_role)


TPD8NER4QFG2PBDRV4SurvivalSCTransNet = (
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4SurvivalSCTransNet
)


def _validate_formal_pbdr_v4_model(
    model: nn.Module,
    *,
    expected_role: Role,
    training_graph: bool,
    require_zero_initialized_heads: bool,
    require_identity_initialized_qfg: bool,
    require_identity_initialized_pbdr_v4: bool,
    current_state: Mapping[str, torch.Tensor] | None,
    stage: Stage | None,
) -> Dict[str, Any]:
    role = _require_role(expected_role)
    expected_type = (
        TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4SurvivalSCTransNet
        if training_graph
        else TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4InferenceSCTransNet
    )
    if type(model) is not expected_type:
        raise TypeError("formal PBDR-V4 model must use its exact graph class")
    if model.pbdr_v4_role != role or model.pbdr_v4.role != role:
        raise PBDRV4IntegrationError("formal PBDR-V4 graph role differs")
    if (
        model.mode != "train"
        or model.deepsuper is not True
        or model.relay_enabled is not True
    ):
        raise PBDRV4IntegrationError(
            "formal PBDR-V4 requires mode=train, deepsuper, and relay"
        )
    if model.tokenizer_variant != FORMAL_SURVIVAL_VARIANT:
        raise PBDRV4IntegrationError("formal PBDR-V4 requires Full V8-MPRS-DCH")
    if model.relay_width != DEFAULT_RELAY_WIDTH:
        raise PBDRV4IntegrationError("formal PBDR-V4 relay width differs")
    if model.relay_initialization_seed != DEFAULT_RELAY_INITIALIZATION_SEED:
        raise PBDRV4IntegrationError("formal PBDR-V4 relay seed differs")
    if model.tpd_ner.dc_support_mode != DEFAULT_DC_SUPPORT_MODE:
        raise PBDRV4IntegrationError("formal PBDR-V4 requires complement-tail NER4")
    if dict(model.tpd_ner.tail_z_thresholds) != dict(DEFAULT_TAIL_Z_THRESHOLDS):
        raise PBDRV4IntegrationError("formal PBDR-V4 tail thresholds differ")
    context_gate = _formal_context_gate(model)

    if not isinstance(model.tpd_qfg, QueryOnlyFrequencyGateV2CROA):
        raise PBDRV4IntegrationError("formal PBDR-V4 QFG module type differs")
    qfg_manifest = validate_formal_qfg_v2_croa(
        model.tpd_qfg,
        require_identity_initialization=require_identity_initialized_qfg,
    )
    if tuple(model.tpd_qfg.feature_channels) != FORMAL_QFG_FEATURE_CHANNELS:
        raise PBDRV4IntegrationError("formal PBDR-V4 QFG channels differ")
    if model.tpd_qfg.mode != FORMAL_QFG_MODE:
        raise PBDRV4IntegrationError("formal PBDR-V4 QFG mode differs")
    if model.tpd_qfg.hidden_channels != FORMAL_QFG_HIDDEN_CHANNELS:
        raise PBDRV4IntegrationError("formal PBDR-V4 QFG width differs")
    if model.tpd_qfg.validate_finite is not FORMAL_QFG_VALIDATE_FINITE:
        raise PBDRV4IntegrationError("formal PBDR-V4 QFG finite setting differs")
    if _parameter_count(model.tpd_qfg) != PRODUCTION_QFG_V2_CROA_PARAMETERS:
        raise PBDRV4IntegrationError("formal PBDR-V4 QFG parameter count differs")

    pbdr_manifest = validate_formal_pbdr_v4_calibrator(
        model.pbdr_v4,
        expected_role=role,
        require_identity_initialization=require_identity_initialized_pbdr_v4,
    )
    state = model.state_dict()
    pbdr_keys = tuple(key for key in state if key.startswith(PBDR_V4_STATE_PREFIX))
    if pbdr_keys != PBDR_V4_STATE_KEYS:
        raise PBDRV4IntegrationError("formal integrated PBDR-V4 state keys differ")
    expected_state_count = (
        FORMAL_V4_QFG_V2_CROA_PBDR_V4_SURVIVAL_STATE_KEY_COUNT
        if training_graph
        else FORMAL_V4_QFG_V2_CROA_PBDR_V4_INFERENCE_STATE_KEY_COUNT
    )
    expected_parameter_count = (
        PRODUCTION_V4_QFG_V2_CROA_PBDR_V4_SURVIVAL_PARAMETERS
        if training_graph
        else PRODUCTION_V4_QFG_V2_CROA_PBDR_V4_INFERENCE_PARAMETERS
    )
    if len(state) != expected_state_count:
        raise PBDRV4IntegrationError("formal integrated PBDR-V4 state count differs")
    if _parameter_count(model) != expected_parameter_count:
        raise PBDRV4IntegrationError("formal integrated PBDR-V4 parameter count differs")
    qfg_keys = tuple(key for key in state if key.startswith(QFG_STATE_PREFIX))
    if set(qfg_keys) != set(QFG_STATE_KEYS):
        raise PBDRV4IntegrationError("formal integrated QFG state keys differ")

    survival_keys = tuple(
        key for key in state if key.startswith(SURVIVAL_STATE_PREFIX)
    )
    if training_graph:
        if set(survival_keys) != set(SURVIVAL_STATE_KEYS):
            raise PBDRV4IntegrationError("formal training graph Survival keys differ")
        if (
            survival_parameter_count(model.target_survival)
            != PRODUCTION_SURVIVAL_PARAMETERS
        ):
            raise PBDRV4IntegrationError(
                "formal training graph Survival parameters differ"
            )
        if require_zero_initialized_heads:
            for name, parameter in model.target_survival.named_parameters():
                if int(torch.count_nonzero(parameter)) != 0:
                    raise PBDRV4IntegrationError(
                        f"Survival parameter {name} is not zero"
                    )
    elif survival_keys or hasattr(model, "target_survival"):
        raise PBDRV4IntegrationError("formal inference graph retains Survival state")

    reference = next(
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(PBDR_V4_STATE_PREFIX)
    )
    for name, parameter in model.pbdr_v4.named_parameters():
        if parameter.device != reference.device:
            raise PBDRV4IntegrationError(f"PBDR-V4 parameter {name} device differs")
        if parameter.dtype != reference.dtype:
            raise PBDRV4IntegrationError(f"PBDR-V4 parameter {name} dtype differs")
        if not bool(torch.isfinite(parameter).all()):
            raise PBDRV4IntegrationError(f"PBDR-V4 parameter {name} is non-finite")

    manifest = model.architecture_manifest()
    expected_manifest = {
        "relay_version": V4_RELAY_VERSION,
        "qfg_integration_version": QFG_V2_CROA_INTEGRATION_VERSION,
        "pbdr_v4_integration_version": PBDR_V4_INTEGRATION_VERSION,
        "pbdr_v4_version": PBDR_V4_VERSION,
        "pbdr_v4_enabled": True,
        "pbdr_v4_role": role,
        "pbdr_v4_positive_limit": float(
            model.pbdr_v4.positive_limit.detach().cpu()
        ),
        "pbdr_v4_negative_limit": float(
            model.pbdr_v4.negative_limit.detach().cpu()
        ),
        "pbdr_v4_state_key_count": PRODUCTION_PBDR_V4_STATE_KEY_COUNT,
        "pbdr_v4_persistent_buffers": PRODUCTION_PBDR_V4_BUFFER_COUNT,
        "pbdr_v4_zero_anchor": "current_final_exact",
        "pbdr_v4_current_checkpoint_warm_start_required": True,
        "tss_objective_enabled": False,
        "segmentation_path_modified": True,
        "deployment_graph": (
            "v4_qfg_v2_croa_pbdr_v4_with_training_only_tss_heads"
            if training_graph
            else "v4_qfg_v2_croa_pbdr_v4_no_tss"
        ),
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            raise PBDRV4IntegrationError(
                f"formal PBDR-V4 manifest field {name!r} differs"
            )

    if (current_state is None) != (stage is None):
        raise PBDRV4IntegrationError(
            "current_state and stage must be provided together"
        )
    state_contract: dict[str, object] | None = None
    if current_state is not None and stage is not None:
        state_contract = audit_candidate_against_current(
            model,
            current_state=current_state,
            stage=_require_stage(stage),
        )

    return {
        "model": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "variant": FORMAL_SURVIVAL_VARIANT,
        "role": role,
        "relay_version": V4_RELAY_VERSION,
        "qfg_integration_version": QFG_V2_CROA_INTEGRATION_VERSION,
        "pbdr_v4_integration_version": PBDR_V4_INTEGRATION_VERSION,
        "state_key_count": expected_state_count,
        "pbdr_v4_state_keys": PBDR_V4_STATE_KEYS,
        "pbdr_v4_parameters": PRODUCTION_PBDR_V4_PARAMETERS,
        "total_parameters": expected_parameter_count,
        "current_warm_start_expected_missing_keys": PBDR_V4_STATE_KEYS,
        "context_gate": context_gate,
        "qfg_core_manifest": qfg_manifest,
        "pbdr_v4_core_manifest": pbdr_manifest,
        "target_survival_registered": training_graph,
        "state_contract": state_contract,
        "architecture_manifest": manifest,
    }


def validate_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
    model: nn.Module,
    *,
    expected_role: Role,
    require_zero_initialized_heads: bool = False,
    require_identity_initialized_qfg: bool = False,
    require_identity_initialized_pbdr_v4: bool = False,
    current_state: Mapping[str, torch.Tensor] | None = None,
    stage: Stage | None = None,
) -> Dict[str, Any]:
    return _validate_formal_pbdr_v4_model(
        model,
        expected_role=expected_role,
        training_graph=True,
        require_zero_initialized_heads=require_zero_initialized_heads,
        require_identity_initialized_qfg=require_identity_initialized_qfg,
        require_identity_initialized_pbdr_v4=(
            require_identity_initialized_pbdr_v4
        ),
        current_state=current_state,
        stage=stage,
    )


def validate_formal_v4_qfg_v2_croa_pbdr_v4_inference_model(
    model: nn.Module,
    *,
    expected_role: Role,
    require_identity_initialized_qfg: bool = False,
    require_identity_initialized_pbdr_v4: bool = False,
    current_state: Mapping[str, torch.Tensor] | None = None,
    stage: Stage | None = None,
) -> Dict[str, Any]:
    return _validate_formal_pbdr_v4_model(
        model,
        expected_role=expected_role,
        training_graph=False,
        require_zero_initialized_heads=False,
        require_identity_initialized_qfg=require_identity_initialized_qfg,
        require_identity_initialized_pbdr_v4=(
            require_identity_initialized_pbdr_v4
        ),
        current_state=current_state,
        stage=stage,
    )


def _build_raw_parent(seed: int) -> tuple[SCTransNet, Dict[str, Any]]:
    from experiments.train_tpd_clean_v8_mprs_dch import (
        build_clean_v8_mprs_dch_model,
    )

    return build_clean_v8_mprs_dch_model(
        FORMAL_SURVIVAL_VARIANT,
        _require_formal_seed(seed),
    )


def build_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
    *,
    role: Role,
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> Tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4SurvivalSCTransNet,
    Dict[str, Any],
]:
    ready_role = _require_role(role)
    parent, parent_metadata = _build_raw_parent(seed)
    model = TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4SurvivalSCTransNet(
        parent,
        role=ready_role,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    metadata = validate_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
        model,
        expected_role=ready_role,
        require_zero_initialized_heads=True,
        require_identity_initialized_qfg=True,
        require_identity_initialized_pbdr_v4=True,
    )
    metadata.update(
        {
            "construction": "role_explicit_seed42_graph_requires_current_warm_start",
            "warm_start_required": True,
            "raw_parent_metadata": parent_metadata,
        }
    )
    return model, metadata


def build_formal_v4_qfg_v2_croa_pbdr_v4_inference_model(
    *,
    role: Role,
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> Tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4InferenceSCTransNet,
    Dict[str, Any],
]:
    ready_role = _require_role(role)
    parent, parent_metadata = _build_raw_parent(seed)
    model = TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4InferenceSCTransNet(
        parent,
        role=ready_role,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    metadata = validate_formal_v4_qfg_v2_croa_pbdr_v4_inference_model(
        model,
        expected_role=ready_role,
        require_identity_initialized_qfg=True,
        require_identity_initialized_pbdr_v4=True,
    )
    metadata.update(
        {
            "construction": "role_explicit_seed42_graph_requires_current_warm_start",
            "warm_start_required": True,
            "raw_parent_metadata": parent_metadata,
        }
    )
    return model, metadata


def warm_start_formal_pbdr_v4_from_current(
    model: nn.Module,
    current_state: Mapping[str, torch.Tensor],
) -> Dict[str, Any]:
    """Install Current only after proving the exact 27-key extension boundary.

    V4 semantic buffers deliberately reject missing-state loads, even in
    non-strict mode.  Therefore warm-start uses an exact key-partition proof and
    a strict merged-state load: all Current tensors come from ``current_state``
    and all 27 V4 tensors remain at their role-specific initialization.
    """

    if not isinstance(current_state, Mapping) or not current_state:
        raise PBDRV4IntegrationError("Current state must be a non-empty mapping")
    if not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in current_state.items()
    ):
        raise PBDRV4IntegrationError("Current state must map strings to tensors")
    if any(name.startswith(PBDR_V4_STATE_PREFIX) for name in current_state):
        raise PBDRV4IntegrationError("Current state unexpectedly contains PBDR-V4")

    candidate_state = model.state_dict()
    candidate_pbdr_keys = tuple(
        name for name in candidate_state if name.startswith(PBDR_V4_STATE_PREFIX)
    )
    if candidate_pbdr_keys != PBDR_V4_STATE_KEYS:
        raise PBDRV4IntegrationError("candidate PBDR-V4 extension keys differ")
    candidate_base_keys = set(candidate_state) - set(PBDR_V4_STATE_KEYS)
    if candidate_base_keys != set(current_state):
        missing = sorted(set(current_state) - candidate_base_keys)
        extra = sorted(candidate_base_keys - set(current_state))
        raise PBDRV4IntegrationError(
            f"Current/V4 inherited key sets differ; missing={missing}, extra={extra}"
        )

    merged: dict[str, torch.Tensor] = {}
    for name, initialized in candidate_state.items():
        source = initialized if name.startswith(PBDR_V4_STATE_PREFIX) else current_state[name]
        merged[name] = source.detach().clone()
    incompatible = model.load_state_dict(merged, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise PBDRV4IntegrationError("strict merged Current warm-start failed")
    installed = model.state_dict()
    changed = [
        name
        for name, expected in current_state.items()
        if not torch.equal(installed[name].detach().cpu(), expected.detach().cpu())
    ]
    if changed:
        raise PBDRV4IntegrationError(
            f"Current tensors changed during warm-start: {changed[:5]}"
        )
    return {
        "load_mode": "strict_merged_after_exact_extension_partition",
        "current_state_key_count": len(current_state),
        "candidate_state_key_count": len(installed),
        "expected_missing_extension_keys": PBDR_V4_STATE_KEYS,
        "expected_missing_extension_key_count": len(PBDR_V4_STATE_KEYS),
        "all_current_tensors_bitwise_equal_after_load": True,
    }


def strip_training_only_survival_state(
    training_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if len(training_state) != FORMAL_V4_QFG_V2_CROA_PBDR_V4_SURVIVAL_STATE_KEY_COUNT:
        raise PBDRV4IntegrationError("candidate training state count differs")
    if not set(SURVIVAL_STATE_KEYS) <= set(training_state):
        raise PBDRV4IntegrationError("candidate training state lacks Survival keys")
    if not set(PBDR_V4_STATE_KEYS) <= set(training_state):
        raise PBDRV4IntegrationError("candidate training state lacks PBDR-V4 keys")
    stripped = {
        name: value.detach().cpu().clone()
        for name, value in training_state.items()
        if name not in set(SURVIVAL_STATE_KEYS)
    }
    if len(stripped) != FORMAL_V4_QFG_V2_CROA_PBDR_V4_INFERENCE_STATE_KEY_COUNT:
        raise PBDRV4IntegrationError("stripped V4 inference state count differs")
    return stripped


def _strip_current_survival_state(
    current_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if len(current_state) != FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT:
        raise PBDRV4IntegrationError("Current survival state count differs")
    if not set(SURVIVAL_STATE_KEYS) <= set(current_state):
        raise PBDRV4IntegrationError("Current state lacks Survival keys")
    stripped = {
        name: value.detach().cpu().clone()
        for name, value in current_state.items()
        if name not in set(SURVIVAL_STATE_KEYS)
    }
    if len(stripped) != FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT:
        raise PBDRV4IntegrationError("Current inference state count differs")
    return stripped


def build_formal_v4_qfg_v2_croa_pbdr_v4_inference_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    expected_role: Role,
    expected_stage: Stage,
    current_state: Mapping[str, torch.Tensor],
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4InferenceSCTransNet,
    Dict[str, Any],
]:
    """Export a candidate only from a complete, role/stage-bound payload."""

    role = _require_role(expected_role)
    stage = _require_stage(expected_stage)
    if not isinstance(checkpoint, Mapping):
        raise PBDRV4IntegrationError("candidate checkpoint must be a mapping")
    required = {"schema", "role", "stage", "architecture_manifest", "state_dict"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise PBDRV4IntegrationError(
            f"candidate checkpoint is incomplete; missing={missing}"
        )
    if checkpoint.get("schema") != PBDR_V4_CANDIDATE_CHECKPOINT_SCHEMA:
        raise PBDRV4IntegrationError("candidate checkpoint schema differs")
    if checkpoint.get("role") != role:
        raise PBDRV4IntegrationError("candidate checkpoint role differs")
    if checkpoint.get("stage") != stage:
        raise PBDRV4IntegrationError("candidate checkpoint stage differs")
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise PBDRV4IntegrationError("candidate checkpoint state_dict is invalid")

    training_model, _ = build_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
        role=role,
        seed=seed,
    )
    warm_start = warm_start_formal_pbdr_v4_from_current(
        training_model,
        current_state,
    )
    expected_manifest = training_model.architecture_manifest()
    if checkpoint.get("architecture_manifest") != expected_manifest:
        raise PBDRV4IntegrationError("candidate checkpoint architecture differs")
    incompatible = training_model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise PBDRV4IntegrationError("candidate strict training-state load failed")
    training_validation = validate_formal_v4_qfg_v2_croa_pbdr_v4_survival_model(
        training_model,
        expected_role=role,
        current_state=current_state,
        stage=stage,
    )

    stripped = strip_training_only_survival_state(training_model.state_dict())
    current_inference = _strip_current_survival_state(current_state)
    inference_model, raw = build_formal_v4_qfg_v2_croa_pbdr_v4_inference_model(
        role=role,
        seed=seed,
    )
    incompatible = inference_model.load_state_dict(stripped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise PBDRV4IntegrationError("candidate strict inference-state load failed")
    inference_validation = validate_formal_v4_qfg_v2_croa_pbdr_v4_inference_model(
        inference_model,
        expected_role=role,
        current_state=current_inference,
        stage=stage,
    )
    inference_model.eval()
    inference_model.mode = "test"
    metadata = {
        "checkpoint_schema": PBDR_V4_CANDIDATE_CHECKPOINT_SCHEMA,
        "role": role,
        "stage": stage,
        "strict_checkpoint_payload": True,
        "strict_training_state_load": True,
        "strict_inference_state_load": True,
        "training_state_key_count": len(state),
        "inference_state_key_count": len(stripped),
        "stripped_training_only_state_keys": list(SURVIVAL_STATE_KEYS),
        "warm_start": warm_start,
        "training_validation": training_validation,
        "inference_validation": inference_validation,
        "raw_inference_builder_metadata": raw,
    }
    return inference_model, metadata


__all__ = [
    "FORMAL_PBDR_V4_INITIALIZATION_SEED",
    "FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT",
    "FORMAL_V4_QFG_V2_CROA_PBDR_V4_INFERENCE_STATE_KEY_COUNT",
    "FORMAL_V4_QFG_V2_CROA_PBDR_V4_SURVIVAL_STATE_KEY_COUNT",
    "FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT",
    "PBDR_V4_CANDIDATE_CHECKPOINT_SCHEMA",
    "PBDR_V4_INTEGRATION_VERSION",
    "PBDR_V4_STATE_KEYS",
    "PBDR_V4_STATE_PREFIX",
    "PBDRV4IntegrationError",
    "PBDRV4ProbabilityTuple",
    "PBDRV4TrainingAux",
    "PBDRV4TrainingReturn",
    "PRODUCTION_PBDR_V4_PARAMETERS",
    "PRODUCTION_V4_QFG_V2_CROA_PBDR_V4_INFERENCE_PARAMETERS",
    "PRODUCTION_V4_QFG_V2_CROA_PBDR_V4_SURVIVAL_PARAMETERS",
    "SUPPORTED_STAGES",
    "SURVIVAL_STATE_KEYS",
    "TPD8NER4QFG2PBDRV4SurvivalSCTransNet",
    "TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4InferenceSCTransNet",
    "TPDNERV8MPRSDCHV4QFGV2CROAPBDRV4SurvivalSCTransNet",
    "build_formal_v4_qfg_v2_croa_pbdr_v4_inference_from_checkpoint",
    "build_formal_v4_qfg_v2_croa_pbdr_v4_inference_model",
    "build_formal_v4_qfg_v2_croa_pbdr_v4_survival_model",
    "strip_training_only_survival_state",
    "validate_formal_v4_qfg_v2_croa_pbdr_v4_inference_model",
    "validate_formal_v4_qfg_v2_croa_pbdr_v4_survival_model",
    "warm_start_formal_pbdr_v4_from_current",
]
