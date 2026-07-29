"""Training-only target-survival extension for the frozen V4 model.

The extension captures the final ``emb1`` and ``emb2`` tokenizer endpoints
during the existing V4 forward pass and attaches two independent 1x1
cell-presence classifiers.  Evaluation remains exactly V4-compatible: no
survival head is executed and the legacy segmentation output is returned.

The public model class deliberately keeps the V4 constructor signature.  The
formal builder below is narrower: it accepts only a raw Clean-V8 Full parent
and fixes the authoritative complement-tail V4 configuration.  This separation
keeps diagnostic V4 construction possible without allowing non-state
architecture choices to leak into a formal warm-start.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple, Union

import torch
import torch.nn as nn

from model.SCTransNet import SCTransNet
from model.tpd_clean_v8_mprs_dch import (
    PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
    clean_v8_mprs_dch_variant_spec,
)
from model.tpd_forward_contract import ForwardOutput, TPDForwardOutput
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
from model.tpd_survival import (
    PairedTargetSurvivalHeads,
    build_structured_survival_output,
    survival_parameter_count,
)


SURVIVAL_VERSION = "dual_post_tpd_endpoint_presence_v1"
SURVIVAL_STATE_PREFIX = "target_survival."
PRODUCTION_SURVIVAL_PARAMETERS = 98
PRODUCTION_V4_SURVIVAL_PARAMETERS = (
    PRODUCTION_V4_RELAY_ON_PARAMETERS + PRODUCTION_SURVIVAL_PARAMETERS
)
FORMAL_V4_PARENT_STATE_KEY_COUNT = 544
FORMAL_V4_SURVIVAL_STATE_KEY_COUNT = 548
FORMAL_SURVIVAL_INITIALIZATION_SEED = 42
FORMAL_SURVIVAL_VARIANT = PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT

SURVIVAL_STATE_KEYS = (
    "target_survival.heads.emb1.classifier.weight",
    "target_survival.heads.emb1.classifier.bias",
    "target_survival.heads.emb2.classifier.weight",
    "target_survival.heads.emb2.classifier.bias",
)

EmbeddingOutput = Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Tuple[torch.Tensor, ...],
    Tuple[torch.Tensor, ...],
]


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _zero_initialize_survival_heads(
    heads: PairedTargetSurvivalHeads,
) -> None:
    for module in heads.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _require_raw_clean_v8_parent(parent: SCTransNet) -> None:
    if not isinstance(parent, SCTransNet):
        raise TypeError("Survival extension parent must be an SCTransNet")
    if hasattr(parent, "tpd_ner"):
        raise ValueError(
            "Survival extension requires a raw Clean-V8 parent; "
            "an already completed V4/NER model is not accepted"
        )


class TPDNERV8MPRSDCHV4SurvivalSCTransNet(
    TPDNERV8MPRSDCHV4SCTransNet
):
    """Frozen V4 inference graph plus training-only endpoint supervision."""

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
        _require_raw_clean_v8_parent(parent)
        super().__init__(
            parent,
            variant=variant,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
            dc_support_mode=dc_support_mode,
            tail_z_thresholds=tail_z_thresholds,
        )

        if self.mode != "train" or self.deepsuper is not True:
            raise RuntimeError(
                "Survival model requires mode='train' and deepsuper=True"
            )
        if not self.relay_enabled:
            raise RuntimeError("Survival model requires the V4 NER relay")

        emb1_channels = int(self.mtc.embeddings_1.blocks[0].channels)
        emb2_channels = int(self.mtc.embeddings_2.blocks[0].channels)
        if (emb1_channels, emb2_channels) != (32, 64):
            raise RuntimeError(
                "Survival model requires endpoint channels 32/64"
            )

        # Conv construction consumes CPU RNG internally.  The fork makes adding
        # this exactly-zero module neutral to the caller's global RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.target_survival = PairedTargetSurvivalHeads(
                emb1_channels=emb1_channels,
                emb2_channels=emb2_channels,
            )
        _zero_initialize_survival_heads(self.target_survival)

        reference = next(self.parameters())
        self.target_survival.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        self.target_survival.train(self.training)

        if (
            survival_parameter_count(self.target_survival)
            != PRODUCTION_SURVIVAL_PARAMETERS
        ):
            raise RuntimeError("unexpected Survival parameter count")

        # These are ordinary transient attributes, not parameters or buffers.
        self._survival_capture_active = False
        self._captured_survival_endpoints: (
            Tuple[torch.Tensor, torch.Tensor] | None
        ) = None

    def explicit_embeddings(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> EmbeddingOutput:
        values = super().explicit_embeddings(x1, x2, x3, x4)

        if self._survival_capture_active:
            if self._captured_survival_endpoints is not None:
                raise RuntimeError("Survival endpoint capture occurred twice")
            self._captured_survival_endpoints = (values[0], values[1])

        return values

    def forward(self, x: torch.Tensor) -> ForwardOutput:  # type: ignore[override]
        # Evaluation and deployment retain the exact V4 legacy return type.
        if not self.training:
            return super().forward(x)

        if self._survival_capture_active:
            raise RuntimeError("re-entrant Survival forward is unsupported")

        self._survival_capture_active = True
        self._captured_survival_endpoints = None
        try:
            segmentation = super().forward(x)
            endpoints = self._captured_survival_endpoints
            if endpoints is None:
                raise RuntimeError(
                    "V4 forward did not expose emb1/emb2 endpoints"
                )
            emb1, emb2 = endpoints
            return build_structured_survival_output(
                segmentation,
                emb1,
                emb2,
                self.target_survival,
            )
        finally:
            self._captured_survival_endpoints = None
            self._survival_capture_active = False

    def architecture_manifest(self) -> Dict[str, Any]:
        manifest = dict(super().architecture_manifest())
        manifest.update(
            {
                "survival_version": SURVIVAL_VERSION,
                "survival_training_only": True,
                "survival_endpoints": ("emb1", "emb2"),
                "survival_endpoint_grid": "stride_16",
                "survival_target": "max_pool_16_binary_presence",
                "survival_head": "independent_conv1x1_raw_logits",
                "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
                "survival_state_prefix": SURVIVAL_STATE_PREFIX,
                "survival_head_initialization": "exact_zero",
                "segmentation_path_modified": False,
                "inference_heads_required": False,
            }
        )
        return manifest


def _require_formal_seed(seed: int) -> int:
    if type(seed) is not int or seed != FORMAL_SURVIVAL_INITIALIZATION_SEED:
        raise ValueError(
            "formal V4 Survival construction requires initialization seed=42"
        )
    return seed


def _formal_parent_context_gate(model: nn.Module) -> float:
    spec = clean_v8_mprs_dch_variant_spec(FORMAL_SURVIVAL_VARIANT)
    expected = float(spec["context_gate"])
    for embedding_name in ("embeddings_1", "embeddings_2"):
        embedding = getattr(model.mtc, embedding_name)
        for block in embedding.blocks:
            if float(block.context_gate) != expected:
                raise RuntimeError(
                    f"formal Survival {embedding_name} context gate differs"
                )
    return expected


def validate_formal_survival_model(
    model: nn.Module,
    *,
    require_zero_initialized_heads: bool = False,
) -> Dict[str, Any]:
    """Validate state and non-state architecture of a formal Survival model."""

    if type(model) is not TPDNERV8MPRSDCHV4SurvivalSCTransNet:
        raise TypeError("formal Survival model must use the exact extension class")
    if (
        model.mode != "train"
        or model.deepsuper is not True
        or model.relay_enabled is not True
    ):
        raise RuntimeError(
            "formal Survival model requires mode=train, deepsuper, and relay"
        )
    if model.tokenizer_variant != FORMAL_SURVIVAL_VARIANT:
        raise RuntimeError("formal Survival model requires Full V8-MPRS-DCH")
    if model.relay_width != DEFAULT_RELAY_WIDTH:
        raise RuntimeError("formal Survival relay width differs")
    if model.relay_initialization_seed != DEFAULT_RELAY_INITIALIZATION_SEED:
        raise RuntimeError("formal Survival relay initialization seed differs")
    if model.tpd_ner.dc_support_mode != DEFAULT_DC_SUPPORT_MODE:
        raise RuntimeError("formal Survival requires complement-tail support")
    if dict(model.tpd_ner.tail_z_thresholds) != dict(
        DEFAULT_TAIL_Z_THRESHOLDS
    ):
        raise RuntimeError("formal Survival tail thresholds differ")
    context_gate = _formal_parent_context_gate(model)

    if _parameter_count(model) != PRODUCTION_V4_SURVIVAL_PARAMETERS:
        raise RuntimeError("formal Survival total parameter count differs")
    if (
        survival_parameter_count(model.target_survival)
        != PRODUCTION_SURVIVAL_PARAMETERS
    ):
        raise RuntimeError("formal Survival head parameter count differs")

    state = model.state_dict()
    survival_keys = tuple(
        key for key in state if key.startswith(SURVIVAL_STATE_PREFIX)
    )
    if set(survival_keys) != set(SURVIVAL_STATE_KEYS):
        raise RuntimeError("formal Survival state keys differ")
    if len(state) != FORMAL_V4_SURVIVAL_STATE_KEY_COUNT:
        raise RuntimeError("formal Survival state-key count differs")
    if len(state) - len(survival_keys) != FORMAL_V4_PARENT_STATE_KEY_COUNT:
        raise RuntimeError("formal V4 parent state-key count differs")
    if tuple(model.target_survival.named_buffers()):
        raise RuntimeError("formal Survival heads must not register buffers")

    reference = next(
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(SURVIVAL_STATE_PREFIX)
    )
    for name, parameter in model.target_survival.named_parameters():
        if parameter.device != reference.device:
            raise RuntimeError(
                f"formal Survival parameter {name} device differs"
            )
        if parameter.dtype != reference.dtype:
            raise RuntimeError(
                f"formal Survival parameter {name} dtype differs"
            )
        if (
            require_zero_initialized_heads
            and torch.count_nonzero(parameter).item() != 0
        ):
            raise RuntimeError(
                f"formal Survival parameter {name} is not exactly zero"
            )

    manifest = model.architecture_manifest()
    expected_manifest = {
        "relay_version": V4_RELAY_VERSION,
        "ner_dc_offset_support_mode": DEFAULT_DC_SUPPORT_MODE,
        "tail_z_thresholds": dict(DEFAULT_TAIL_Z_THRESHOLDS),
        "target_protective_complement": True,
        "survival_version": SURVIVAL_VERSION,
        "survival_training_only": True,
        "survival_endpoints": ("emb1", "emb2"),
        "survival_endpoint_grid": "stride_16",
        "survival_target": "max_pool_16_binary_presence",
        "survival_head": "independent_conv1x1_raw_logits",
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "survival_state_prefix": SURVIVAL_STATE_PREFIX,
        "survival_head_initialization": "exact_zero",
        "segmentation_path_modified": False,
        "inference_heads_required": False,
    }
    for name, expected in expected_manifest.items():
        if manifest.get(name) != expected:
            raise RuntimeError(
                f"formal Survival manifest field {name!r} differs"
            )

    return {
        "model": (
            "model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival."
            "TPDNERV8MPRSDCHV4SurvivalSCTransNet"
        ),
        "parent_model": (
            "model.tpd_ner_v8_mprs_dch_v4_tail_aware."
            "TPDNERV8MPRSDCHV4SCTransNet"
        ),
        "variant": FORMAL_SURVIVAL_VARIANT,
        "parent_variant": FORMAL_SURVIVAL_VARIANT,
        "relay_version": V4_RELAY_VERSION,
        "parent_relay_version": V4_RELAY_VERSION,
        "parent_state_key_count": FORMAL_V4_PARENT_STATE_KEY_COUNT,
        "state_key_count": FORMAL_V4_SURVIVAL_STATE_KEY_COUNT,
        "survival_state_key_count": len(SURVIVAL_STATE_KEYS),
        "survival_state_keys": SURVIVAL_STATE_KEYS,
        "survival_classifier_parameter_count": PRODUCTION_SURVIVAL_PARAMETERS,
        "survival_parameters": PRODUCTION_SURVIVAL_PARAMETERS,
        "total_parameters": PRODUCTION_V4_SURVIVAL_PARAMETERS,
        "survival_head_initialization": "exact_zero",
        "context_gate": context_gate,
        "dc_support_mode": DEFAULT_DC_SUPPORT_MODE,
        "tail_z_thresholds": dict(DEFAULT_TAIL_Z_THRESHOLDS),
        "architecture_manifest": manifest,
    }


def _build_raw_formal_parent(seed: int) -> Tuple[SCTransNet, Dict[str, Any]]:
    # Lazy import avoids making the model module depend on training entry-point
    # initialization during ordinary class imports.
    from experiments.train_tpd_clean_v8_mprs_dch import (
        build_clean_v8_mprs_dch_model,
    )

    return build_clean_v8_mprs_dch_model(
        FORMAL_SURVIVAL_VARIANT,
        _require_formal_seed(seed),
    )


def build_formal_v4_reference(
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> Tuple[TPDNERV8MPRSDCHV4SCTransNet, Dict[str, Any]]:
    """Build the strict V4 reference used to validate a parent checkpoint."""

    parent, parent_metadata = _build_raw_formal_parent(seed)
    model = TPDNERV8MPRSDCHV4SCTransNet(
        parent,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    manifest = model.architecture_manifest()
    if (
        model.mode != "train"
        or model.deepsuper is not True
        or model.relay_enabled is not True
        or model.tokenizer_variant != FORMAL_SURVIVAL_VARIANT
        or model.tpd_ner.dc_support_mode != DEFAULT_DC_SUPPORT_MODE
        or dict(model.tpd_ner.tail_z_thresholds)
        != dict(DEFAULT_TAIL_Z_THRESHOLDS)
        or len(model.state_dict()) != FORMAL_V4_PARENT_STATE_KEY_COUNT
        or _parameter_count(model) != PRODUCTION_V4_RELAY_ON_PARAMETERS
    ):
        raise RuntimeError("formal V4 reference architecture differs")
    _formal_parent_context_gate(model)
    return model, {
        "model": (
            "model.tpd_ner_v8_mprs_dch_v4_tail_aware."
            "TPDNERV8MPRSDCHV4SCTransNet"
        ),
        "variant": FORMAL_SURVIVAL_VARIANT,
        "relay_version": V4_RELAY_VERSION,
        "state_key_count": FORMAL_V4_PARENT_STATE_KEY_COUNT,
        "total_parameters": PRODUCTION_V4_RELAY_ON_PARAMETERS,
        "architecture_manifest": manifest,
        "raw_parent_metadata": parent_metadata,
    }


def build_formal_v4_survival_model(
    seed: int = FORMAL_SURVIVAL_INITIALIZATION_SEED,
) -> Tuple[TPDNERV8MPRSDCHV4SurvivalSCTransNet, Dict[str, Any]]:
    """Build the sole formal Full/complement-tail Survival candidate."""

    parent, parent_metadata = _build_raw_formal_parent(seed)
    model = TPDNERV8MPRSDCHV4SurvivalSCTransNet(
        parent,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    metadata = validate_formal_survival_model(
        model,
        require_zero_initialized_heads=True,
    )
    metadata["raw_parent_metadata"] = parent_metadata
    return model, metadata


__all__ = [
    "FORMAL_SURVIVAL_INITIALIZATION_SEED",
    "FORMAL_SURVIVAL_VARIANT",
    "FORMAL_V4_PARENT_STATE_KEY_COUNT",
    "FORMAL_V4_SURVIVAL_STATE_KEY_COUNT",
    "PRODUCTION_SURVIVAL_PARAMETERS",
    "PRODUCTION_V4_SURVIVAL_PARAMETERS",
    "SURVIVAL_STATE_KEYS",
    "SURVIVAL_STATE_PREFIX",
    "SURVIVAL_VERSION",
    "TPDNERV8MPRSDCHV4SurvivalSCTransNet",
    "build_formal_v4_reference",
    "build_formal_v4_survival_model",
    "validate_formal_survival_model",
]
