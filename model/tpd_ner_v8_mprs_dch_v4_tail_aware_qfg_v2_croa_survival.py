"""QFG-V2-CROA integration for the frozen V4 Survival model.

This module changes one operation in the inherited segmentation graph: the
four encoder features prepare one forward-local QFG object, and the existing
SCTransNet encoder is then evaluated through the Query-only bridge.  The
frozen TPD embeddings, V4 NER relay, decoder, target-survival heads, structured
training return, and legacy evaluation return remain inherited unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.SCTransNet import SCTransNet
from model.tpd_clean_v8_mprs_dch import (
    clean_v8_mprs_dch_variant_spec,
)
from model.tpd_frequency_gate_v2_croa import (
    QueryOnlyFrequencyGateV2CROA,
    validate_formal_qfg_v2_croa,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    DEFAULT_DC_SUPPORT_MODE,
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
    DEFAULT_TAIL_Z_THRESHOLDS,
    PRODUCTION_V4_RELAY_ON_PARAMETERS,
    TailDCSupportMode,
    TPDNERV8MPRSDCHV4SCTransNet,
    V4_RELAY_VERSION,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    FORMAL_SURVIVAL_INITIALIZATION_SEED,
    FORMAL_SURVIVAL_VARIANT,
    FORMAL_V4_PARENT_STATE_KEY_COUNT,
    FORMAL_V4_SURVIVAL_STATE_KEY_COUNT,
    PRODUCTION_SURVIVAL_PARAMETERS,
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
    SURVIVAL_VERSION,
    TPDNERV8MPRSDCHV4SurvivalSCTransNet,
)
from model.tpd_query_frequency_bridge import frequency_encoder_forward
from model.tpd_survival import survival_parameter_count


QFG_V2_CROA_INTEGRATION_VERSION = "v4_survival_qfg_v2_croa_v1"
QFG_STATE_PREFIX = "tpd_qfg."
FORMAL_QFG_MODE = "high_low"
FORMAL_QFG_HIDDEN_CHANNELS = 8
FORMAL_QFG_DETACH_FREQUENCY_SOURCE = True
FORMAL_QFG_VALIDATE_FINITE = True
FORMAL_QFG_INITIALIZATION_SEED = 42
FORMAL_QFG_ALPHA_EFFECTIVE_INIT = 0.1
FORMAL_QFG_FEATURE_CHANNELS = (32, 64, 128, 256)
PRODUCTION_QFG_V2_CROA_PARAMETERS = 15_684
PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT = 20
PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS = 10_870_228
FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT = 568
PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS = 10_870_130
FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT = 564

_QFG_LEVEL_STATE_SUFFIXES = (
    "alpha",
    "haar.kernels",
    "prior_projection.weight",
    "spatial_projection.0.weight",
    "gate_out.weight",
)
QFG_STATE_KEYS = tuple(
    f"{QFG_STATE_PREFIX}levels.{level}.{suffix}"
    for level in range(4)
    for suffix in _QFG_LEVEL_STATE_SUFFIXES
)
QFG_TERMINAL_STATE_KEYS = tuple(
    f"{QFG_STATE_PREFIX}levels.{level}.gate_out.weight"
    for level in range(4)
)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _require_formal_seed(seed: int) -> int:
    if type(seed) is not int or seed != FORMAL_SURVIVAL_INITIALIZATION_SEED:
        raise ValueError(
            "formal V4 QFG-V2-CROA Survival construction requires seed=42"
        )
    return seed


def _formal_context_gate(model: nn.Module) -> float:
    spec = clean_v8_mprs_dch_variant_spec(FORMAL_SURVIVAL_VARIANT)
    expected = float(spec["context_gate"])
    for embedding_name in ("embeddings_1", "embeddings_2"):
        embedding = getattr(model.mtc, embedding_name)
        for block in embedding.blocks:
            if float(block.context_gate) != expected:
                raise RuntimeError(
                    f"formal QFG {embedding_name} context gate differs"
                )
    return expected


def _build_formal_qfg(
    reference: torch.Tensor,
    *,
    training: bool,
) -> QueryOnlyFrequencyGateV2CROA:
    # Construction of the hidden projections is isolated from the caller's
    # CPU RNG stream.  Seed only the CPU default generator: torch.manual_seed
    # would also rewrite every CUDA generator even though construction is CPU.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(
            FORMAL_QFG_INITIALIZATION_SEED
        )
        qfg = QueryOnlyFrequencyGateV2CROA(
            FORMAL_QFG_FEATURE_CHANNELS,
            mode=FORMAL_QFG_MODE,
            hidden_channels=FORMAL_QFG_HIDDEN_CHANNELS,
            detach_frequency_source=FORMAL_QFG_DETACH_FREQUENCY_SOURCE,
            validate_finite=FORMAL_QFG_VALIDATE_FINITE,
            alpha_effective_init=FORMAL_QFG_ALPHA_EFFECTIVE_INIT,
        )
    qfg.to(device=reference.device, dtype=reference.dtype)
    qfg.train(training)
    if _parameter_count(qfg) != PRODUCTION_QFG_V2_CROA_PARAMETERS:
        raise RuntimeError("unexpected QFG-V2-CROA parameter count")
    if len(qfg.state_dict()) != PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT:
        raise RuntimeError("unexpected QFG-V2-CROA state-key count")
    return qfg


def _qfg_manifest_fields(
    qfg: QueryOnlyFrequencyGateV2CROA,
) -> Dict[str, Any]:
    return {
        "qfg_integration_version": QFG_V2_CROA_INTEGRATION_VERSION,
        "qfg_enabled": True,
        "qfg_module": "QueryOnlyFrequencyGateV2CROA",
        "qfg_frequency_mode": FORMAL_QFG_MODE,
        "qfg_feature_channels": FORMAL_QFG_FEATURE_CHANNELS,
        "qfg_hidden_channels": FORMAL_QFG_HIDDEN_CHANNELS,
        "qfg_detach_frequency_source": (
            FORMAL_QFG_DETACH_FREQUENCY_SOURCE
        ),
        "qfg_validate_finite": FORMAL_QFG_VALIDATE_FINITE,
        "qfg_initialization_seed": FORMAL_QFG_INITIALIZATION_SEED,
        "qfg_alpha_effective_initialization": (
            FORMAL_QFG_ALPHA_EFFECTIVE_INIT
        ),
        "qfg_terminal_initialization": "exact_zero",
        "qfg_state_prefix": QFG_STATE_PREFIX,
        "qfg_state_key_count": PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT,
        "qfg_parameters": PRODUCTION_QFG_V2_CROA_PARAMETERS,
        "qfg_prepare_execution": (
            "complete_modulation_once_per_model_forward_"
            "reused_by_all_sctb"
        ),
        "qfg_prepared_level_payload": (
            "raw_gate_logits",
            "normalized_logits",
            "gate",
            "factor",
        ),
        "qfg_prepared_modulation_reused_across_sctb": True,
        "qfg_modulation_location": "post_q_convolution_pre_normalization",
        "qfg_modified_attention_tensors": ("Q",),
        "qfg_kv_modified": False,
        "qfg_cfn_modified": False,
        "qfg_decoder_injection": False,
        "qfg_inference_required": True,
        "qfg_core_manifest": qfg.architecture_manifest(),
        "segmentation_path_modified": True,
        "segmentation_path_modification": (
            "bounded_query_only_frequency_modulation"
        ),
    }


class TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet(
    TPDNERV8MPRSDCHV4SurvivalSCTransNet
):
    """Frozen V4 Survival graph with one formal Query-only frequency gate."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        variant: str,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode: Union[
            str,
            TailDCSupportMode,
        ] = DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds: Mapping[
            int,
            float,
        ] = DEFAULT_TAIL_Z_THRESHOLDS,
    ) -> None:
        super().__init__(
            parent,
            variant=variant,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
            dc_support_mode=dc_support_mode,
            tail_z_thresholds=tail_z_thresholds,
        )

        reference = next(self.parameters())
        self.tpd_qfg = _build_formal_qfg(
            reference,
            training=self.training,
        )

    def _forward_with_relay(self, x: torch.Tensor):
        # This is the frozen V4 relay flow.  Only the encoder call is replaced
        # by prepare-once plus the pure Query-only bridge.
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
        manifest.update(_qfg_manifest_fields(self.tpd_qfg))
        return manifest


class TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet(
    TPDNERV8MPRSDCHV4SCTransNet
):
    """Head-free deployment graph retaining the trained Query-only gate."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        variant: str,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode: Union[
            str,
            TailDCSupportMode,
        ] = DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds: Mapping[
            int,
            float,
        ] = DEFAULT_TAIL_Z_THRESHOLDS,
    ) -> None:
        super().__init__(
            parent,
            variant=variant,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
            dc_support_mode=dc_support_mode,
            tail_z_thresholds=tail_z_thresholds,
        )
        reference = next(self.parameters())
        self.tpd_qfg = _build_formal_qfg(
            reference,
            training=self.training,
        )

    # Use the exact integration flow above without inheriting or registering
    # the training-only target-survival heads.
    _forward_with_relay = (
        TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet._forward_with_relay
    )

    def architecture_manifest(self) -> Dict[str, Any]:
        manifest = dict(super().architecture_manifest())
        manifest.update(_qfg_manifest_fields(self.tpd_qfg))
        manifest.update(
            {
                "deployment_graph": "v4_qfg_v2_croa_no_tss",
                "target_survival_registered": False,
                "target_survival_state_removed": True,
                "inference_heads_required": False,
            }
        )
        return manifest


# Short paper-facing name retained as an exact alias, not a second class.
TPD8NER4QFGV2CROASurvivalSCTransNet = (
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet
)


def validate_formal_qfg_v2_croa_survival_model(
    model: nn.Module,
    *,
    require_zero_initialized_heads: bool = False,
    require_identity_initialized_qfg: bool = False,
) -> Dict[str, Any]:
    """Validate the sole formal V4 + QFG-V2-CROA + Survival model."""

    if type(model) is not TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet:
        raise TypeError(
            "formal QFG model must use the exact integration class"
        )
    if (
        model.mode != "train"
        or model.deepsuper is not True
        or model.relay_enabled is not True
    ):
        raise RuntimeError(
            "formal QFG model requires mode=train, deepsuper, and relay"
        )
    if model.tokenizer_variant != FORMAL_SURVIVAL_VARIANT:
        raise RuntimeError("formal QFG model requires Full V8-MPRS-DCH")
    if model.relay_width != DEFAULT_RELAY_WIDTH:
        raise RuntimeError("formal QFG relay width differs")
    if model.relay_initialization_seed != DEFAULT_RELAY_INITIALIZATION_SEED:
        raise RuntimeError("formal QFG relay initialization seed differs")
    if model.tpd_ner.dc_support_mode != DEFAULT_DC_SUPPORT_MODE:
        raise RuntimeError("formal QFG model requires complement-tail support")
    if dict(model.tpd_ner.tail_z_thresholds) != dict(
        DEFAULT_TAIL_Z_THRESHOLDS
    ):
        raise RuntimeError("formal QFG tail thresholds differ")
    context_gate = _formal_context_gate(model)

    if not isinstance(model.tpd_qfg, QueryOnlyFrequencyGateV2CROA):
        raise RuntimeError("formal QFG module type differs")
    core_qfg_manifest = validate_formal_qfg_v2_croa(
        model.tpd_qfg,
        require_identity_initialization=require_identity_initialized_qfg,
    )
    if tuple(model.tpd_qfg.feature_channels) != FORMAL_QFG_FEATURE_CHANNELS:
        raise RuntimeError("formal QFG feature channels differ")
    if model.tpd_qfg.mode != FORMAL_QFG_MODE:
        raise RuntimeError("formal QFG mode differs")
    if model.tpd_qfg.hidden_channels != FORMAL_QFG_HIDDEN_CHANNELS:
        raise RuntimeError("formal QFG hidden width differs")
    if (
        model.tpd_qfg.detach_frequency_source
        is not FORMAL_QFG_DETACH_FREQUENCY_SOURCE
    ):
        raise RuntimeError("formal QFG detach boundary differs")
    if model.tpd_qfg.validate_finite is not FORMAL_QFG_VALIDATE_FINITE:
        raise RuntimeError("formal QFG finite-validation setting differs")

    if (
        _parameter_count(model)
        != PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS
    ):
        raise RuntimeError("formal QFG total parameter count differs")
    if (
        _parameter_count(model.tpd_qfg)
        != PRODUCTION_QFG_V2_CROA_PARAMETERS
    ):
        raise RuntimeError("formal QFG parameter count differs")
    if (
        survival_parameter_count(model.target_survival)
        != PRODUCTION_SURVIVAL_PARAMETERS
    ):
        raise RuntimeError("formal Survival head parameter count differs")

    state = model.state_dict()
    qfg_keys = tuple(
        key for key in state if key.startswith(QFG_STATE_PREFIX)
    )
    survival_keys = tuple(
        key for key in state if key.startswith(SURVIVAL_STATE_PREFIX)
    )
    if set(qfg_keys) != set(QFG_STATE_KEYS):
        raise RuntimeError("formal QFG state keys differ")
    if len(qfg_keys) != PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT:
        raise RuntimeError("formal QFG state-key count differs")
    if set(survival_keys) != set(SURVIVAL_STATE_KEYS):
        raise RuntimeError("formal Survival state keys differ")
    if len(state) != FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT:
        raise RuntimeError("formal integrated state-key count differs")
    if len(state) - len(qfg_keys) != FORMAL_V4_SURVIVAL_STATE_KEY_COUNT:
        raise RuntimeError("formal inherited Survival state-key count differs")

    reference = next(
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(QFG_STATE_PREFIX)
    )
    for name, parameter in model.tpd_qfg.named_parameters():
        if parameter.device != reference.device:
            raise RuntimeError(f"formal QFG parameter {name} device differs")
        if parameter.dtype != reference.dtype:
            raise RuntimeError(f"formal QFG parameter {name} dtype differs")
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError(f"formal QFG parameter {name} is non-finite")
    for name, parameter in model.target_survival.named_parameters():
        if parameter.device != reference.device:
            raise RuntimeError(
                f"formal Survival parameter {name} device differs"
            )
        if parameter.dtype != reference.dtype:
            raise RuntimeError(
                f"formal Survival parameter {name} dtype differs"
            )
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError(
                f"formal Survival parameter {name} is non-finite"
            )
        if (
            require_zero_initialized_heads
            and torch.count_nonzero(parameter).item() != 0
        ):
            raise RuntimeError(
                f"formal Survival parameter {name} is not exactly zero"
            )

    if require_identity_initialized_qfg:
        for level_index, level in enumerate(model.tpd_qfg.levels):
            effective_alpha = float(
                torch.tanh(level.alpha.detach().float())
            )
            if (
                abs(
                    effective_alpha
                    - FORMAL_QFG_ALPHA_EFFECTIVE_INIT
                )
                > 1e-7
            ):
                raise RuntimeError(
                    f"formal QFG level {level_index} alpha init differs"
                )
            if torch.count_nonzero(level.gate_out.weight).item() != 0:
                raise RuntimeError(
                    f"formal QFG level {level_index} terminal is not zero"
                )

    manifest = model.architecture_manifest()
    expected_manifest = {
        "relay_version": V4_RELAY_VERSION,
        "ner_dc_offset_support_mode": DEFAULT_DC_SUPPORT_MODE,
        "tail_z_thresholds": dict(DEFAULT_TAIL_Z_THRESHOLDS),
        "target_protective_complement": True,
        "survival_version": SURVIVAL_VERSION,
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "qfg_integration_version": QFG_V2_CROA_INTEGRATION_VERSION,
        "qfg_enabled": True,
        "qfg_module": "QueryOnlyFrequencyGateV2CROA",
        "qfg_frequency_mode": FORMAL_QFG_MODE,
        "qfg_feature_channels": FORMAL_QFG_FEATURE_CHANNELS,
        "qfg_hidden_channels": FORMAL_QFG_HIDDEN_CHANNELS,
        "qfg_detach_frequency_source": True,
        "qfg_validate_finite": True,
        "qfg_initialization_seed": FORMAL_QFG_INITIALIZATION_SEED,
        "qfg_alpha_effective_initialization": (
            FORMAL_QFG_ALPHA_EFFECTIVE_INIT
        ),
        "qfg_terminal_initialization": "exact_zero",
        "qfg_state_prefix": QFG_STATE_PREFIX,
        "qfg_state_key_count": PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT,
        "qfg_parameters": PRODUCTION_QFG_V2_CROA_PARAMETERS,
        "qfg_modified_attention_tensors": ("Q",),
        "qfg_kv_modified": False,
        "qfg_cfn_modified": False,
        "qfg_decoder_injection": False,
        "qfg_inference_required": True,
        "segmentation_path_modified": True,
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            raise RuntimeError(
                f"formal QFG manifest field {name!r} differs"
            )

    return {
        "model": (
            "model.tpd_ner_v8_mprs_dch_v4_tail_aware_"
            "qfg_v2_croa_survival."
            "TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet"
        ),
        "parent_model": (
            "model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival."
            "TPDNERV8MPRSDCHV4SurvivalSCTransNet"
        ),
        "variant": FORMAL_SURVIVAL_VARIANT,
        "relay_version": V4_RELAY_VERSION,
        "qfg_integration_version": QFG_V2_CROA_INTEGRATION_VERSION,
        "parent_state_key_count": FORMAL_V4_SURVIVAL_STATE_KEY_COUNT,
        "state_key_count": FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
        "qfg_state_key_count": len(QFG_STATE_KEYS),
        "qfg_state_keys": QFG_STATE_KEYS,
        "qfg_parameters": PRODUCTION_QFG_V2_CROA_PARAMETERS,
        "survival_state_keys": SURVIVAL_STATE_KEYS,
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "total_parameters": (
            PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS
        ),
        "context_gate": context_gate,
        "dc_support_mode": DEFAULT_DC_SUPPORT_MODE,
        "tail_z_thresholds": dict(DEFAULT_TAIL_Z_THRESHOLDS),
        "qfg_core_manifest": core_qfg_manifest,
        "architecture_manifest": manifest,
    }


def validate_formal_qfg_v2_croa_inference_model(
    model: nn.Module,
    *,
    require_identity_initialized_qfg: bool = False,
) -> Dict[str, Any]:
    """Validate the strict head-free V4 + QFG deployment graph."""

    if type(model) is not TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet:
        raise TypeError(
            "formal QFG inference model must use the exact deployment class"
        )
    if hasattr(model, "target_survival"):
        raise RuntimeError("formal QFG inference model retains Survival heads")
    if (
        model.mode != "train"
        or model.deepsuper is not True
        or model.relay_enabled is not True
    ):
        raise RuntimeError(
            "formal QFG inference model requires the frozen six-output graph"
        )
    if model.tokenizer_variant != FORMAL_SURVIVAL_VARIANT:
        raise RuntimeError(
            "formal QFG inference model requires Full V8-MPRS-DCH"
        )
    if model.relay_width != DEFAULT_RELAY_WIDTH:
        raise RuntimeError("formal QFG inference relay width differs")
    if model.relay_initialization_seed != DEFAULT_RELAY_INITIALIZATION_SEED:
        raise RuntimeError(
            "formal QFG inference relay initialization seed differs"
        )
    if model.tpd_ner.dc_support_mode != DEFAULT_DC_SUPPORT_MODE:
        raise RuntimeError(
            "formal QFG inference requires complement-tail support"
        )
    if dict(model.tpd_ner.tail_z_thresholds) != dict(
        DEFAULT_TAIL_Z_THRESHOLDS
    ):
        raise RuntimeError("formal QFG inference tail thresholds differ")
    context_gate = _formal_context_gate(model)
    core_qfg_manifest = validate_formal_qfg_v2_croa(
        model.tpd_qfg,
        require_identity_initialization=require_identity_initialized_qfg,
    )

    if (
        _parameter_count(model)
        != PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS
    ):
        raise RuntimeError(
            "formal QFG inference total parameter count differs"
        )
    if (
        _parameter_count(model) - _parameter_count(model.tpd_qfg)
        != PRODUCTION_V4_RELAY_ON_PARAMETERS
    ):
        raise RuntimeError("formal QFG inference V4 parameter count differs")

    state = model.state_dict()
    qfg_keys = tuple(
        key for key in state if key.startswith(QFG_STATE_PREFIX)
    )
    if set(qfg_keys) != set(QFG_STATE_KEYS):
        raise RuntimeError("formal QFG inference state keys differ")
    if len(state) != FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT:
        raise RuntimeError(
            "formal QFG inference total state-key count differs"
        )
    if len(state) - len(qfg_keys) != FORMAL_V4_PARENT_STATE_KEY_COUNT:
        raise RuntimeError("formal QFG inference V4 state-key count differs")
    if any(key.startswith(SURVIVAL_STATE_PREFIX) for key in state):
        raise RuntimeError("formal QFG inference state retains Survival keys")

    manifest = model.architecture_manifest()
    expected_manifest = {
        "relay_version": V4_RELAY_VERSION,
        "ner_dc_offset_support_mode": DEFAULT_DC_SUPPORT_MODE,
        "tail_z_thresholds": dict(DEFAULT_TAIL_Z_THRESHOLDS),
        "target_protective_complement": True,
        "qfg_integration_version": QFG_V2_CROA_INTEGRATION_VERSION,
        "qfg_enabled": True,
        "qfg_parameters": PRODUCTION_QFG_V2_CROA_PARAMETERS,
        "qfg_state_key_count": PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT,
        "qfg_inference_required": True,
        "deployment_graph": "v4_qfg_v2_croa_no_tss",
        "target_survival_registered": False,
        "target_survival_state_removed": True,
        "inference_heads_required": False,
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            raise RuntimeError(
                f"formal QFG inference manifest field {name!r} differs"
            )

    return {
        "model": (
            "model.tpd_ner_v8_mprs_dch_v4_tail_aware_"
            "qfg_v2_croa_survival."
            "TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet"
        ),
        "variant": FORMAL_SURVIVAL_VARIANT,
        "relay_version": V4_RELAY_VERSION,
        "qfg_integration_version": QFG_V2_CROA_INTEGRATION_VERSION,
        "state_key_count": (
            FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT
        ),
        "v4_state_key_count": FORMAL_V4_PARENT_STATE_KEY_COUNT,
        "qfg_state_key_count": len(QFG_STATE_KEYS),
        "qfg_state_keys": QFG_STATE_KEYS,
        "qfg_parameters": PRODUCTION_QFG_V2_CROA_PARAMETERS,
        "total_parameters": (
            PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS
        ),
        "context_gate": context_gate,
        "dc_support_mode": DEFAULT_DC_SUPPORT_MODE,
        "tail_z_thresholds": dict(DEFAULT_TAIL_Z_THRESHOLDS),
        "target_survival_registered": False,
        "qfg_core_manifest": core_qfg_manifest,
        "architecture_manifest": manifest,
    }


def build_formal_v4_qfg_v2_croa_inference_model(
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> Tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
    Dict[str, Any],
]:
    """Build the authoritative head-free QFG deployment architecture."""

    from experiments.train_tpd_clean_v8_mprs_dch import (
        build_clean_v8_mprs_dch_model,
    )

    parent, parent_metadata = build_clean_v8_mprs_dch_model(
        FORMAL_SURVIVAL_VARIANT,
        _require_formal_seed(seed),
    )
    model = TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet(
        parent,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    metadata = validate_formal_qfg_v2_croa_inference_model(
        model,
        require_identity_initialized_qfg=True,
    )
    metadata["raw_parent_metadata"] = parent_metadata
    return model, metadata


def build_formal_v4_qfg_v2_croa_survival_model(
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> Tuple[
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
    Dict[str, Any],
]:
    """Build the sole formal Full/complement-tail/QFG-V2 candidate."""

    from experiments.train_tpd_clean_v8_mprs_dch import (
        build_clean_v8_mprs_dch_model,
    )

    parent, parent_metadata = build_clean_v8_mprs_dch_model(
        FORMAL_SURVIVAL_VARIANT,
        _require_formal_seed(seed),
    )
    model = TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet(
        parent,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    metadata = validate_formal_qfg_v2_croa_survival_model(
        model,
        require_zero_initialized_heads=True,
        require_identity_initialized_qfg=True,
    )
    metadata["raw_parent_metadata"] = parent_metadata
    return model, metadata


__all__ = [
    "FORMAL_QFG_ALPHA_EFFECTIVE_INIT",
    "FORMAL_QFG_DETACH_FREQUENCY_SOURCE",
    "FORMAL_QFG_FEATURE_CHANNELS",
    "FORMAL_QFG_HIDDEN_CHANNELS",
    "FORMAL_QFG_INITIALIZATION_SEED",
    "FORMAL_QFG_MODE",
    "FORMAL_QFG_VALIDATE_FINITE",
    "FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT",
    "FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT",
    "PRODUCTION_QFG_V2_CROA_PARAMETERS",
    "PRODUCTION_QFG_V2_CROA_STATE_KEY_COUNT",
    "PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS",
    "PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS",
    "QFG_STATE_KEYS",
    "QFG_STATE_PREFIX",
    "QFG_TERMINAL_STATE_KEYS",
    "QFG_V2_CROA_INTEGRATION_VERSION",
    "TPD8NER4QFGV2CROASurvivalSCTransNet",
    "TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet",
    "TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet",
    "build_formal_v4_qfg_v2_croa_inference_model",
    "build_formal_v4_qfg_v2_croa_survival_model",
    "validate_formal_qfg_v2_croa_inference_model",
    "validate_formal_qfg_v2_croa_survival_model",
]
