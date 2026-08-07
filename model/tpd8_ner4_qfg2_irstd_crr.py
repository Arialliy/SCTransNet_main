"""Formal IRSTD-only BGCR extension of the frozen Current inference graph.

The authoritative Current checkpoint is a 568-key TSS-off training graph.  A
formal BGCR construction validates that complete state, removes exactly the
four training-only Survival tensors, and strictly installs the resulting
564-key inference state below a new ``irstd_repair`` module.  No Current
parameter or buffer is trainable.

The wrapper deliberately exposes two training paths:

* :meth:`forward_for_irstd_training` computes the frozen Current context once;
* :meth:`forward_repair_from_context` consumes an already-bound patch context
  without executing Current again.

This separation prevents a cache-backed runner from silently performing a
second parent forward.  A cached context must correspond to the exact input
patch supplied to the repair head; full-image features cropped afterwards are
not equivalent to a Current forward on that patch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.pbdr_v4_state_contract import state_semantic_sha256
from experiments.pbdr_v4_run_artifacts import file_sha256, load_torch_artifact
from experiments.irstd_bgcr_run_contract import (
    DATASET as RUN_CONTRACT_DATASET,
    FOLD_ASSIGNMENT_SHA256,
    OFFICIAL_FALSE_FLAGS,
    OOF_EVALUATION_EPOCHS,
    PROBABILITY_COMPARISON,
    PROBABILITY_THRESHOLD,
    ROLE as RUN_CONTRACT_ROLE,
    SELECTION_SCHEMA as OOF_INNER_SELECTION_SCHEMA,
    SOURCE_SCOPE as RUN_CONTRACT_SOURCE_SCOPE,
    SOURCE_SPLIT_MANIFEST_FILE_SHA256,
    canonical_json_sha256,
)
from experiments.train_tpd_clean_v8_mprs_dch import (
    build_clean_v8_mprs_dch_model,
)
from model.SCTransNet import SCTransNet
from model.irstd_core_ring_repair import (
    HIDDEN_CHANNELS,
    IRSTD_CRR_VERSION,
    IRSTDCoreRingRepairHead,
    IRSTDCoreRingRepairOutput,
    LOCAL_CHANNELS,
    NEGATIVE_LOGIT_LIMIT,
    POSITIVE_LOGIT_LIMIT,
    PRODUCTION_PARAMETER_COUNT,
    PRODUCTION_PERSISTENT_BUFFER_COUNT,
    PRODUCTION_STATE_KEY_COUNT,
    validate_formal_irstd_core_ring_repair_head,
)
from model.tpd_ner_v8_mprs_dch import (
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    DEFAULT_DC_SUPPORT_MODE,
    DEFAULT_TAIL_Z_THRESHOLDS,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    FORMAL_SURVIVAL_VARIANT,
    FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
    PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
    build_formal_v4_qfg_v2_croa_inference_model,
    validate_formal_qfg_v2_croa_inference_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_KEYS,
)
from model.tpd_query_frequency_bridge import frequency_encoder_forward


IRSTD_BGCR_INTEGRATION_VERSION = "tpd8_ner4_qfg2_irstd_crr_v1"
IRSTD_CRR_STATE_PREFIX = "irstd_repair."
INTEGRATED_CANDIDATE_SCHEMA = (
    "sctransnet_train_irstd_bgcr_v1/v1/integrated_candidate"
)
OOF_SELECTOR_SCHEMA = "sctransnet_irstd_bgcr_oof_selector/v1"
TRAINING_IDENTITY_SCHEMA = "sctransnet_train_irstd_bgcr_v1/v1/identity"
STATE_SEMANTIC_HASH_ALGORITHM = "state_semantic_sha256"
FORMAL_DATASET = "IRSTD-1K"
FORMAL_PARENT_ROLE = "best_miou"
FORMAL_SEED = 42
FORMAL_REPAIR_INITIALIZATION_SEED = 42
CURRENT_TRAINING_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT
)
CURRENT_INFERENCE_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT
)
INTEGRATED_STATE_KEY_COUNT = (
    CURRENT_INFERENCE_STATE_KEY_COUNT + PRODUCTION_STATE_KEY_COUNT
)
INTEGRATED_PARAMETER_COUNT = (
    PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS
    + PRODUCTION_PARAMETER_COUNT
)


class IRSTDBGCRIntegrationError(RuntimeError):
    """A BGCR model, Current projection, or frozen-base contract differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IRSTDBGCRIntegrationError(message)


def _require_sha256(value: object, *, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return value


def _require_official_false_flags(
    payload: Mapping[str, object],
    *,
    name: str,
) -> None:
    for field, expected in OFFICIAL_FALSE_FLAGS.items():
        _require(
            payload.get(field) is expected,
            f"{name} official flag differs: {field}",
        )


def _require_formal_seed(value: int, *, name: str) -> int:
    _require(type(value) is int and value == FORMAL_SEED, f"{name} must be 42")
    return value


def _tensor_mapping(
    value: Mapping[str, torch.Tensor],
    *,
    name: str,
) -> Mapping[str, torch.Tensor]:
    _require(isinstance(value, Mapping) and bool(value), f"{name} must be a mapping")
    _require(
        all(type(key) is str and isinstance(tensor, torch.Tensor) for key, tensor in value.items()),
        f"{name} must map string keys to tensors",
    )
    return value


def _require_finite_state(
    value: Mapping[str, torch.Tensor],
    *,
    name: str,
) -> None:
    for key, tensor in value.items():
        _require(bool(torch.isfinite(tensor).all()), f"{name} tensor is non-finite: {key}")


def strip_current_survival_state_strict(
    training_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Validate the complete TSS-off Current state and remove exactly 4 keys."""

    state = _tensor_mapping(training_state, name="Current training state")
    _require(
        len(SURVIVAL_STATE_KEYS) == 4,
        "formal Survival training-only key count differs",
    )
    _require(
        len(state) == CURRENT_TRAINING_STATE_KEY_COUNT,
        f"Current training state must have {CURRENT_TRAINING_STATE_KEY_COUNT} keys",
    )
    survival = set(SURVIVAL_STATE_KEYS)
    _require(
        survival <= set(state),
        "Current training state lacks exact Survival keys",
    )
    _require(
        not any(
            key.startswith("target_survival.") and key not in survival
            for key in state
        ),
        "Current training state has unexpected Survival keys",
    )
    _require(
        all(int(torch.count_nonzero(state[key])) == 0 for key in SURVIVAL_STATE_KEYS),
        "formal BGCR parent must be the TSS-off Current state",
    )
    _require_finite_state(state, name="Current training state")
    projected = {
        key: tensor.detach().clone()
        for key, tensor in state.items()
        if key not in survival
    }
    _require(
        len(projected) == CURRENT_INFERENCE_STATE_KEY_COUNT,
        f"Current inference projection must have {CURRENT_INFERENCE_STATE_KEY_COUNT} keys",
    )
    _require(
        not any(key.startswith("target_survival.") for key in projected),
        "Current inference projection retains Survival state",
    )
    return projected


@dataclass(frozen=True, slots=True)
class FrozenIRSTDContext:
    """Detached full-resolution Current tensors for one exact input patch."""

    local_feature: torch.Tensor
    out_logits: torch.Tensor
    d0_logits: torch.Tensor
    gt2_logits: torch.Tensor
    gt3_logits: torch.Tensor
    gt4_logits: torch.Tensor
    gt5_logits: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.local_feature,
            self.out_logits,
            self.d0_logits,
            self.gt2_logits,
            self.gt3_logits,
            self.gt4_logits,
            self.gt5_logits,
        )


def _semantic_buffer_equal(
    observed: object,
    expected: torch.Tensor,
) -> bool:
    return (
        isinstance(observed, torch.Tensor)
        and observed.shape == expected.shape
        and observed.dtype == expected.dtype
        and torch.equal(observed.detach().cpu(), expected.detach().cpu())
    )


class TPD8NER4QFG2IRSTDCRRInferenceSCTransNet(
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet
):
    """Frozen Current plus one identity-initialized IRSTD repair head."""

    def __init__(
        self,
        parent: SCTransNet,
        *,
        variant: str = FORMAL_SURVIVAL_VARIANT,
        relay_width: int = DEFAULT_RELAY_WIDTH,
        relay_initialization_seed: int = DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode: str = DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds: Mapping[int, float] = DEFAULT_TAIL_Z_THRESHOLDS,
        repair_initialization_seed: int = FORMAL_REPAIR_INITIALIZATION_SEED,
    ) -> None:
        _require_formal_seed(
            repair_initialization_seed,
            name="repair_initialization_seed",
        )
        super().__init__(
            parent,
            variant=variant,
            relay_width=relay_width,
            relay_initialization_seed=relay_initialization_seed,
            dc_support_mode=dc_support_mode,
            tail_z_thresholds=tail_z_thresholds,
        )
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(repair_initialization_seed)
            repair = IRSTDCoreRingRepairHead(
                local_channels=int(self.outc.in_channels),
                hidden_channels=HIDDEN_CHANNELS,
                positive_limit=POSITIVE_LOGIT_LIMIT,
                negative_limit=NEGATIVE_LOGIT_LIMIT,
                detach_context=True,
            )
        reference = next(self.parameters())
        repair.to(device=reference.device, dtype=reference.dtype)
        self.irstd_repair = repair
        self.repair_initialization_seed = repair_initialization_seed
        self.bgcr_dataset = FORMAL_DATASET
        self.bgcr_parent_role = FORMAL_PARENT_ROLE
        self.freeze_current()

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Reject different-capacity repair checkpoints before installation."""

        guarded = state_dict.copy()
        for local_name in ("positive_limit", "negative_limit"):
            key = f"{prefix}{IRSTD_CRR_STATE_PREFIX}{local_name}"
            expected = getattr(self.irstd_repair, local_name)
            if not _semantic_buffer_equal(state_dict.get(key), expected):
                error_msgs.append(
                    "IRSTD BGCR semantic checkpoint mismatch for "
                    f"{key!r}; different-limit loads are forbidden"
                )
                guarded[key] = expected.detach().clone()
        super()._load_from_state_dict(
            guarded,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def freeze_current(self) -> None:
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(IRSTD_CRR_STATE_PREFIX))
        super().train(False)
        self.irstd_repair.train(True)

    def train(self, mode: bool = True):
        if type(mode) is not bool:
            raise TypeError("training mode must be bool")
        super().train(False)
        if hasattr(self, "irstd_repair"):
            self.irstd_repair.train(mode)
        return self

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        expected = tuple(self.irstd_repair.parameters())
        _require(bool(expected), "IRSTD repair parameter set is empty")
        _require(
            all(parameter.requires_grad for parameter in expected),
            "one IRSTD repair parameter is frozen",
        )
        actual = tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )
        _require(
            {id(parameter) for parameter in actual}
            == {id(parameter) for parameter in expected},
            "trainable parameter set differs from irstd_repair",
        )
        return expected

    @torch.no_grad()
    def _frozen_current_context(self, x: torch.Tensor) -> FrozenIRSTDContext:
        """Execute the exact Current order once and return detached logits."""

        _require(self.deepsuper is True, "formal BGCR requires deep supervision heads")
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

        gt5 = F.interpolate(
            self.gt_conv5(d5),
            scale_factor=16,
            mode="bilinear",
            align_corners=True,
        )
        gt4 = F.interpolate(
            self.gt_conv4(d4),
            scale_factor=8,
            mode="bilinear",
            align_corners=True,
        )
        gt3 = F.interpolate(
            self.gt_conv3(d3),
            scale_factor=4,
            mode="bilinear",
            align_corners=True,
        )
        gt2 = F.interpolate(
            self.gt_conv2(d2),
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), dim=1))
        return FrozenIRSTDContext(
            local_feature=u1.detach(),
            out_logits=out.detach(),
            d0_logits=d0.detach(),
            gt2_logits=gt2.detach(),
            gt3_logits=gt3.detach(),
            gt4_logits=gt4.detach(),
            gt5_logits=gt5.detach(),
        )

    def forward_repair_from_context(
        self,
        image: torch.Tensor,
        context: FrozenIRSTDContext,
        *,
        base_logits_override: torch.Tensor | None = None,
    ) -> IRSTDCoreRingRepairOutput:
        """Run only BGCR from one exact patch-bound frozen context."""

        _require(
            isinstance(context, FrozenIRSTDContext),
            "context must be FrozenIRSTDContext",
        )
        base = (
            context.out_logits
            if base_logits_override is None
            else base_logits_override
        )
        return self.irstd_repair.forward_with_diagnostics(
            image=image.detach(),
            z_out=base.detach(),
            z_d0=context.d0_logits.detach(),
            z_gt2=context.gt2_logits.detach(),
            z_gt3=context.gt3_logits.detach(),
            z_gt4=context.gt4_logits.detach(),
            z_gt5=context.gt5_logits.detach(),
            local_feature=context.local_feature.detach(),
        )

    def forward_for_irstd_training(
        self,
        x: torch.Tensor,
        *,
        base_logits_override: torch.Tensor | None = None,
    ) -> tuple[IRSTDCoreRingRepairOutput, FrozenIRSTDContext]:
        """Compute Current exactly once, then execute the repair-only path."""

        context = self._frozen_current_context(x)
        routing = self.forward_repair_from_context(
            x,
            context,
            base_logits_override=base_logits_override,
        )
        return routing, context

    def _forward_with_relay(self, x: torch.Tensor) -> torch.Tensor:
        routing, _ = self.forward_for_irstd_training(x)
        return torch.sigmoid(routing.routed_logits)

    def architecture_manifest(self) -> dict[str, Any]:
        inherited = dict(super().architecture_manifest())
        inherited.update(
            {
                "irstd_bgcr_integration_version": (
                    IRSTD_BGCR_INTEGRATION_VERSION
                ),
                "irstd_bgcr_enabled": True,
                "irstd_crr_version": IRSTD_CRR_VERSION,
                "irstd_bgcr_dataset": FORMAL_DATASET,
                "irstd_bgcr_parent_role": FORMAL_PARENT_ROLE,
                "irstd_bgcr_state_prefix": IRSTD_CRR_STATE_PREFIX,
                "irstd_bgcr_state_key_count": PRODUCTION_STATE_KEY_COUNT,
                "irstd_bgcr_parameter_count": PRODUCTION_PARAMETER_COUNT,
                "irstd_bgcr_persistent_buffer_count": (
                    PRODUCTION_PERSISTENT_BUFFER_COUNT
                ),
                "irstd_bgcr_initialization_seed": (
                    self.repair_initialization_seed
                ),
                "irstd_bgcr_core_manifest": (
                    self.irstd_repair.architecture_manifest()
                ),
                "current_inference_state_key_count": (
                    CURRENT_INFERENCE_STATE_KEY_COUNT
                ),
                "integrated_state_key_count": INTEGRATED_STATE_KEY_COUNT,
                "integrated_parameter_count": INTEGRATED_PARAMETER_COUNT,
                "current_context_gradient": "detached",
                "standard_forward": "single_probability",
                "cached_context_interface": "forward_repair_from_context",
                "deployment_graph": "v4_qfg_v2_croa_plus_irstd_bgcr_no_tss",
                "performance_acceptance_margin": None,
                **OFFICIAL_FALSE_FLAGS,
            }
        )
        return inherited


def _base_state(
    model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
) -> dict[str, torch.Tensor]:
    return {
        key: tensor
        for key, tensor in model.state_dict().items()
        if not key.startswith(IRSTD_CRR_STATE_PREFIX)
    }


def audit_frozen_current_base(
    model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
    expected_current_inference_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Prove the 564-key base and all its gradients/modes remain frozen."""

    _require(
        type(model) is TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
        "formal BGCR model type differs",
    )
    expected = _tensor_mapping(
        expected_current_inference_state,
        name="expected Current inference state",
    )
    _require(
        len(expected) == CURRENT_INFERENCE_STATE_KEY_COUNT,
        "expected Current inference state count differs",
    )
    observed = _base_state(model)
    _require(
        tuple(observed) == tuple(expected),
        "frozen Current state key order differs",
    )
    for key, expected_tensor in expected.items():
        observed_tensor = observed[key]
        _require(
            observed_tensor.shape == expected_tensor.shape
            and observed_tensor.dtype == expected_tensor.dtype,
            f"frozen Current tensor metadata differs: {key}",
        )
        _require(
            torch.equal(
                observed_tensor.detach().cpu(),
                expected_tensor.detach().cpu(),
            ),
            f"frozen Current tensor changed: {key}",
        )
    base_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(IRSTD_CRR_STATE_PREFIX)
    }
    _require(
        all(not parameter.requires_grad for parameter in base_parameters.values()),
        "one Current parameter is trainable",
    )
    _require(
        all(parameter.grad is None for parameter in base_parameters.values()),
        "one Current parameter received a gradient",
    )
    unexpected_training = [
        name
        for name, module in model.named_modules()
        if name
        and not name.startswith(IRSTD_CRR_STATE_PREFIX.rstrip("."))
        and module.training
    ]
    _require(
        not model.training and not unexpected_training,
        f"Current modules entered training mode: {unexpected_training[:8]}",
    )
    model.trainable_parameters()
    return {
        "current_state_key_count": len(observed),
        "current_state_semantic_sha256": state_semantic_sha256(observed),
        "current_state_semantic_hash_algorithm": STATE_SEMANTIC_HASH_ALGORITHM,
        "all_current_tensors_bitwise_equal": True,
        "all_current_parameters_frozen": True,
        "all_current_gradients_none": True,
        "current_training_modules": [],
        "repair_training": bool(model.irstd_repair.training),
        **OFFICIAL_FALSE_FLAGS,
    }


def load_current_into_frozen_base_strictly(
    model: TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
    current_inference_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Install exactly 564 Current tensors while retaining identity BGCR."""

    _require(
        type(model) is TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
        "formal BGCR model type differs",
    )
    current = _tensor_mapping(
        current_inference_state,
        name="Current inference state",
    )
    _require(
        len(current) == CURRENT_INFERENCE_STATE_KEY_COUNT,
        f"Current inference state must have {CURRENT_INFERENCE_STATE_KEY_COUNT} keys",
    )
    _require_finite_state(current, name="Current inference state")
    integrated = model.state_dict()
    base_keys = tuple(
        key for key in integrated if not key.startswith(IRSTD_CRR_STATE_PREFIX)
    )
    repair_keys = tuple(
        key for key in integrated if key.startswith(IRSTD_CRR_STATE_PREFIX)
    )
    _require(
        len(base_keys) == CURRENT_INFERENCE_STATE_KEY_COUNT
        and len(repair_keys) == PRODUCTION_STATE_KEY_COUNT
        and len(integrated) == INTEGRATED_STATE_KEY_COUNT,
        "integrated Current/repair state partition differs",
    )
    _require(
        tuple(current) == base_keys,
        "Current inference key set or order differs",
    )
    merged = dict(integrated)
    for key in base_keys:
        value = current[key]
        expected = integrated[key]
        _require(
            value.shape == expected.shape and value.dtype == expected.dtype,
            f"Current tensor metadata differs: {key}",
        )
        merged[key] = value.detach().clone()
    incompatible = model.load_state_dict(merged, strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "strict integrated Current load returned incompatible keys",
    )
    model.freeze_current()
    audit = audit_frozen_current_base(model, current)
    return {
        "current_keys": len(base_keys),
        "repair_keys": len(repair_keys),
        "integrated_keys": len(model.state_dict()),
        "current_state_semantic_sha256": state_semantic_sha256(current),
        "current_state_semantic_hash_algorithm": STATE_SEMANTIC_HASH_ALGORITHM,
        "strict_load": True,
        "base_audit": audit,
        **OFFICIAL_FALSE_FLAGS,
    }


def validate_formal_irstd_bgcr_model(
    model: nn.Module,
    *,
    expected_current_inference_state: Mapping[str, torch.Tensor],
    require_identity_initialization: bool,
) -> dict[str, Any]:
    """Validate the sole formal IRSTD BGCR graph without dataset access."""

    _require(
        type(model) is TPD8NER4QFG2IRSTDCRRInferenceSCTransNet,
        "formal BGCR model must use its exact class",
    )
    assert isinstance(model, TPD8NER4QFG2IRSTDCRRInferenceSCTransNet)
    _require(
        model.bgcr_dataset == FORMAL_DATASET
        and model.bgcr_parent_role == FORMAL_PARENT_ROLE,
        "formal BGCR dataset/parent role differs",
    )
    _require(
        model.repair_initialization_seed == FORMAL_REPAIR_INITIALIZATION_SEED,
        "formal BGCR repair seed differs",
    )
    _require(
        model.tokenizer_variant == FORMAL_SURVIVAL_VARIANT
        and model.relay_width == DEFAULT_RELAY_WIDTH
        and model.relay_initialization_seed == DEFAULT_RELAY_INITIALIZATION_SEED,
        "formal BGCR inherited variant/relay contract differs",
    )
    _require(
        model.tpd_ner.dc_support_mode == DEFAULT_DC_SUPPORT_MODE
        and dict(model.tpd_ner.tail_z_thresholds)
        == dict(DEFAULT_TAIL_Z_THRESHOLDS),
        "formal BGCR inherited NER contract differs",
    )
    repair_manifest = validate_formal_irstd_core_ring_repair_head(
        model.irstd_repair,
        require_identity_initialization=require_identity_initialization,
    )
    _require(
        int(model.outc.in_channels) == LOCAL_CHANNELS,
        "formal BGCR Current local feature width differs",
    )
    _require(
        sum(parameter.numel() for parameter in model.parameters())
        == INTEGRATED_PARAMETER_COUNT,
        "formal BGCR integrated parameter count differs",
    )
    _require(
        len(model.state_dict()) == INTEGRATED_STATE_KEY_COUNT,
        "formal BGCR integrated state-key count differs",
    )
    base_audit = audit_frozen_current_base(
        model,
        expected_current_inference_state,
    )
    manifest = model.architecture_manifest()
    expected_manifest = {
        "irstd_bgcr_integration_version": IRSTD_BGCR_INTEGRATION_VERSION,
        "irstd_bgcr_enabled": True,
        "irstd_bgcr_dataset": FORMAL_DATASET,
        "irstd_bgcr_parent_role": FORMAL_PARENT_ROLE,
        "irstd_bgcr_state_key_count": PRODUCTION_STATE_KEY_COUNT,
        "irstd_bgcr_parameter_count": PRODUCTION_PARAMETER_COUNT,
        "current_inference_state_key_count": CURRENT_INFERENCE_STATE_KEY_COUNT,
        "integrated_state_key_count": INTEGRATED_STATE_KEY_COUNT,
        "integrated_parameter_count": INTEGRATED_PARAMETER_COUNT,
        "performance_acceptance_margin": None,
    }
    for key, expected in expected_manifest.items():
        _require(manifest.get(key) == expected, f"BGCR manifest field differs: {key}")
    _require_official_false_flags(manifest, name="BGCR architecture manifest")
    return {
        "schema": IRSTD_BGCR_INTEGRATION_VERSION,
        "dataset": FORMAL_DATASET,
        "parent_role": FORMAL_PARENT_ROLE,
        "current_state_key_count": CURRENT_INFERENCE_STATE_KEY_COUNT,
        "repair_state_key_count": PRODUCTION_STATE_KEY_COUNT,
        "integrated_state_key_count": INTEGRATED_STATE_KEY_COUNT,
        "repair_parameter_count": PRODUCTION_PARAMETER_COUNT,
        "integrated_parameter_count": INTEGRATED_PARAMETER_COUNT,
        "repair_manifest": repair_manifest,
        "base_audit": base_audit,
        "architecture_manifest": manifest,
        **OFFICIAL_FALSE_FLAGS,
    }


def build_formal_irstd_bgcr_model(
    current_training_state: Mapping[str, torch.Tensor],
    *,
    seed: int = FORMAL_SEED,
    repair_initialization_seed: int = FORMAL_REPAIR_INITIALIZATION_SEED,
) -> tuple[TPD8NER4QFG2IRSTDCRRInferenceSCTransNet, dict[str, Any]]:
    """Build from a raw V8 parent, then strictly project/load Current."""

    _require_formal_seed(seed, name="seed")
    _require_formal_seed(
        repair_initialization_seed,
        name="repair_initialization_seed",
    )
    current_inference = strip_current_survival_state_strict(
        current_training_state
    )

    # Validate the projected state against the authoritative exact Current
    # class.  The existing validator intentionally rejects the BGCR subclass.
    current_reference, current_builder_metadata = (
        build_formal_v4_qfg_v2_croa_inference_model(seed)
    )
    _require(
        tuple(current_reference.state_dict()) == tuple(current_inference),
        "projected Current key order differs from authoritative inference graph",
    )
    incompatible = current_reference.load_state_dict(
        current_inference,
        strict=True,
    )
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "strict Current reference load returned incompatible keys",
    )
    current_reference_validation = validate_formal_qfg_v2_croa_inference_model(
        current_reference
    )
    current_architecture_manifest = current_reference.architecture_manifest()
    del current_reference

    raw_parent, raw_parent_metadata = build_clean_v8_mprs_dch_model(
        FORMAL_SURVIVAL_VARIANT,
        seed,
    )
    model = TPD8NER4QFG2IRSTDCRRInferenceSCTransNet(
        raw_parent,
        variant=FORMAL_SURVIVAL_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
        repair_initialization_seed=repair_initialization_seed,
    )
    del raw_parent
    load_metadata = load_current_into_frozen_base_strictly(
        model,
        current_inference,
    )
    validation = validate_formal_irstd_bgcr_model(
        model,
        expected_current_inference_state=current_inference,
        require_identity_initialization=True,
    )
    current_training_semantic_sha = state_semantic_sha256(current_training_state)
    current_inference_semantic_sha = state_semantic_sha256(current_inference)
    metadata = {
        "schema": IRSTD_BGCR_INTEGRATION_VERSION,
        "dataset": FORMAL_DATASET,
        "parent_role": FORMAL_PARENT_ROLE,
        "seed": seed,
        "repair_initialization_seed": repair_initialization_seed,
        "initialization": "raw_v8_then_strict_568_to_564_current_projection",
        "current_training_state_key_count": len(current_training_state),
        "current_training_state_semantic_sha256": current_training_semantic_sha,
        "current_training_state_semantic_hash_algorithm": (
            STATE_SEMANTIC_HASH_ALGORITHM
        ),
        "current_inference_state_key_count": len(current_inference),
        "current_inference_state_semantic_sha256": current_inference_semantic_sha,
        "current_inference_state_semantic_hash_algorithm": (
            STATE_SEMANTIC_HASH_ALGORITHM
        ),
        "current_architecture_manifest": current_architecture_manifest,
        "current_builder_metadata": current_builder_metadata,
        "current_reference_validation": current_reference_validation,
        "raw_parent_metadata": raw_parent_metadata,
        "load_metadata": load_metadata,
        "validation": validation,
        "epoch_zero_exact_current_by_zero_terminal_residuals": True,
        "performance_acceptance_margin": None,
        **OFFICIAL_FALSE_FLAGS,
    }
    _require_official_false_flags(metadata, name="BGCR builder metadata")
    return model, metadata


def _validate_oof_selector_binding(
    payload: Mapping[str, object],
) -> int:
    _require(payload.get("schema") == OOF_SELECTOR_SCHEMA, "OOF selector schema differs")
    _require(
        payload.get("dataset") == RUN_CONTRACT_DATASET
        and payload.get("role") == RUN_CONTRACT_ROLE
        and payload.get("source_scope") == RUN_CONTRACT_SOURCE_SCOPE,
        "OOF selector scope differs",
    )
    _require(
        payload.get("performance_acceptance_margin") is None,
        "OOF selector performance margin must be null",
    )
    _require(
        payload.get("fold_assignment_sha256") == FOLD_ASSIGNMENT_SHA256
        and payload.get("source_split_manifest_file_sha256")
        == SOURCE_SPLIT_MANIFEST_FILE_SHA256
        and payload.get("probability_threshold") == PROBABILITY_THRESHOLD
        and payload.get("probability_comparison") == PROBABILITY_COMPARISON,
        "OOF selector frozen protocol binding differs",
    )
    _require_official_false_flags(payload, name="OOF selector")
    unsigned = dict(payload)
    declared = _require_sha256(
        unsigned.pop("selection_sha256", None),
        name="OOF selector semantic SHA",
    )
    _require(
        canonical_json_sha256(unsigned) == declared,
        "OOF selector semantic SHA does not replay",
    )
    inner = payload.get("selection")
    _require(isinstance(inner, Mapping), "inner OOF selection is absent")
    _require(
        inner.get("schema") == OOF_INNER_SELECTION_SCHEMA
        and inner.get("dataset") == RUN_CONTRACT_DATASET
        and inner.get("role") == RUN_CONTRACT_ROLE
        and inner.get("candidate_epochs") == list(OOF_EVALUATION_EPOCHS)
        and inner.get("fold_assignment_sha256") == FOLD_ASSIGNMENT_SHA256
        and inner.get("source_split_manifest_file_sha256")
        == SOURCE_SPLIT_MANIFEST_FILE_SHA256,
        "inner OOF selection scope differs",
    )
    _require(
        inner.get("performance_acceptance_margin") is None,
        "inner OOF selection performance margin must be null",
    )
    _require_official_false_flags(inner, name="inner OOF selection")
    selected_epoch = payload.get("selected_epoch")
    _require(
        type(selected_epoch) is int
        and selected_epoch in OOF_EVALUATION_EPOCHS
        and inner.get("selected_epoch") == selected_epoch,
        "outer/inner selected epoch differs",
    )
    return selected_epoch


def load_formal_irstd_bgcr_integrated_candidate(
    candidate: Path | str | Mapping[str, object],
    *,
    current_training_state: Mapping[str, torch.Tensor],
    expected_file_sha256: str | None = None,
) -> tuple[TPD8NER4QFG2IRSTDCRRInferenceSCTransNet, dict[str, Any]]:
    """Strictly validate and load one formal 595-key BGCR candidate.

    A filesystem candidate must be a regular non-symlink file and must be
    bound by an explicitly supplied file SHA.  A mapping is accepted for the
    trainer's pre-publication self-check, in which case a file SHA is forbidden.
    Both paths validate the embedded selector, five official-boundary flags,
    semantic hash algorithms and the exact 564-Current/31-repair partition
    before any candidate tensor is installed.
    """

    artifact_file_sha256: str | None
    if isinstance(candidate, Mapping):
        _require(
            expected_file_sha256 is None,
            "mapping candidates cannot claim an artifact file SHA",
        )
        payload = dict(candidate)
        artifact_file_sha256 = None
    else:
        expected_sha = _require_sha256(
            expected_file_sha256,
            name="integrated candidate file SHA",
        )
        path = Path(candidate)
        try:
            observed_sha = file_sha256(path)
            payload = load_torch_artifact(path)
        except Exception as error:
            raise IRSTDBGCRIntegrationError(
                f"cannot read integrated candidate: {error}"
            ) from error
        _require(observed_sha == expected_sha, "integrated candidate file SHA differs")
        artifact_file_sha256 = observed_sha

    _require(
        payload.get("schema") == INTEGRATED_CANDIDATE_SCHEMA,
        "integrated candidate schema differs",
    )
    _require(
        payload.get("dataset") == FORMAL_DATASET
        and payload.get("role") == FORMAL_PARENT_ROLE
        and payload.get("mode") == "full"
        and payload.get("seed") == FORMAL_SEED,
        "integrated candidate scope differs",
    )
    _require(
        payload.get("performance_acceptance_margin") is None,
        "integrated candidate performance margin must be null",
    )
    _require_official_false_flags(payload, name="integrated candidate")
    selector = payload.get("oof_selection")
    _require(isinstance(selector, Mapping), "integrated candidate lacks OOF selector")
    selected_epoch = _validate_oof_selector_binding(selector)
    _require(payload.get("epoch") == selected_epoch, "candidate/selector epoch differs")
    _require_sha256(
        payload.get("oof_selection_file_sha256"),
        name="OOF selector file SHA",
    )

    identity = payload.get("identity")
    _require(isinstance(identity, Mapping), "integrated candidate identity is absent")
    _require(
        identity.get("schema") == TRAINING_IDENTITY_SCHEMA
        and identity.get("mode") == "full"
        and identity.get("dataset") == FORMAL_DATASET
        and identity.get("role") == FORMAL_PARENT_ROLE
        and identity.get("selected_epoch") == selected_epoch,
        "integrated candidate identity scope differs",
    )
    _require_official_false_flags(identity, name="integrated candidate identity")
    unsigned_identity = dict(identity)
    declared_identity_sha = _require_sha256(
        unsigned_identity.pop("identity_sha256", None),
        name="integrated candidate identity SHA",
    )
    _require(
        canonical_json_sha256(unsigned_identity) == declared_identity_sha
        and payload.get("identity_sha256") == declared_identity_sha,
        "integrated candidate identity SHA does not replay",
    )

    training_state = _tensor_mapping(
        current_training_state,
        name="candidate Current training state",
    )
    current_inference = strip_current_survival_state_strict(training_state)
    current_training_semantic_sha = state_semantic_sha256(training_state)
    current_inference_semantic_sha = state_semantic_sha256(current_inference)
    _require(
        identity.get("current_training_state_tensor_mapping_hash_algorithm")
        == "tensor_mapping_sha256"
        and identity.get("current_inference_state_semantic_hash_algorithm")
        == STATE_SEMANTIC_HASH_ALGORITHM
        and identity.get("current_inference_state_semantic_sha256")
        == current_inference_semantic_sha,
        "integrated candidate Current identity hash binding differs",
    )
    _require_sha256(
        identity.get("current_training_state_tensor_mapping_sha256"),
        name="Current training tensor-mapping SHA",
    )

    state = _tensor_mapping(payload.get("state_dict"), name="integrated candidate state")
    _require(len(state) == INTEGRATED_STATE_KEY_COUNT, "candidate must have 595 keys")
    _require_finite_state(state, name="integrated candidate state")
    base_state = {
        name: tensor
        for name, tensor in state.items()
        if not name.startswith(IRSTD_CRR_STATE_PREFIX)
    }
    repair_state = {
        name.removeprefix(IRSTD_CRR_STATE_PREFIX): tensor
        for name, tensor in state.items()
        if name.startswith(IRSTD_CRR_STATE_PREFIX)
    }
    _require(
        len(base_state) == CURRENT_INFERENCE_STATE_KEY_COUNT
        and len(repair_state) == PRODUCTION_STATE_KEY_COUNT,
        "candidate state partition must be 564 Current plus 31 repair keys",
    )
    expected_hashes = (
        (
            "state",
            state,
            INTEGRATED_STATE_KEY_COUNT,
        ),
        (
            "current_base_state",
            base_state,
            CURRENT_INFERENCE_STATE_KEY_COUNT,
        ),
        (
            "repair_state",
            repair_state,
            PRODUCTION_STATE_KEY_COUNT,
        ),
    )
    for prefix, observed_state, expected_count in expected_hashes:
        _require(
            payload.get(f"{prefix}_key_count") == expected_count,
            f"candidate {prefix} key count differs",
        )
        _require(
            payload.get(f"{prefix}_hash_algorithm")
            == STATE_SEMANTIC_HASH_ALGORITHM,
            f"candidate {prefix} hash algorithm differs",
        )
        _require(
            payload.get(f"{prefix}_semantic_sha256")
            == state_semantic_sha256(observed_state),
            f"candidate {prefix} semantic SHA differs",
        )
    _require(
        tuple(base_state) == tuple(current_inference),
        "candidate Current key order differs",
    )
    for name, expected in current_inference.items():
        observed = base_state[name]
        _require(
            observed.shape == expected.shape
            and observed.dtype == expected.dtype
            and torch.equal(observed.detach().cpu(), expected.detach().cpu()),
            f"candidate Current tensor differs: {name}",
        )
    _require(
        payload.get("current_base_state_semantic_sha256")
        == current_inference_semantic_sha,
        "candidate Current semantic SHA differs",
    )

    builder_metadata = payload.get("model_builder_metadata")
    _require(isinstance(builder_metadata, Mapping), "candidate builder metadata is absent")
    _require_official_false_flags(builder_metadata, name="candidate builder metadata")
    _require(
        builder_metadata.get("current_training_state_semantic_hash_algorithm")
        == STATE_SEMANTIC_HASH_ALGORITHM
        and builder_metadata.get("current_training_state_semantic_sha256")
        == current_training_semantic_sha
        and builder_metadata.get("current_inference_state_semantic_hash_algorithm")
        == STATE_SEMANTIC_HASH_ALGORITHM
        and builder_metadata.get("current_inference_state_semantic_sha256")
        == current_inference_semantic_sha,
        "candidate builder semantic hash binding differs",
    )
    architecture_manifest = payload.get("architecture_manifest")
    _require(isinstance(architecture_manifest, Mapping), "candidate architecture is absent")
    _require_official_false_flags(architecture_manifest, name="candidate architecture")
    base_audit = payload.get("base_audit")
    _require(isinstance(base_audit, Mapping), "candidate base audit is absent")
    _require_official_false_flags(base_audit, name="candidate base audit")

    model, expected_builder_metadata = build_formal_irstd_bgcr_model(training_state)
    _require(
        dict(builder_metadata) == expected_builder_metadata,
        "candidate builder metadata differs from formal reconstruction",
    )
    expected_initial_state = model.state_dict()
    _require(
        tuple(state) == tuple(expected_initial_state),
        "candidate integrated key order differs",
    )
    for name, tensor in state.items():
        expected = expected_initial_state[name]
        _require(
            tensor.shape == expected.shape and tensor.dtype == expected.dtype,
            f"candidate tensor metadata differs: {name}",
        )
    _require(
        dict(architecture_manifest) == model.architecture_manifest(),
        "candidate architecture manifest differs",
    )
    incompatible = model.load_state_dict(state, strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "strict integrated candidate load returned incompatible keys",
    )
    model.eval()
    validation = validate_formal_irstd_bgcr_model(
        model,
        expected_current_inference_state=current_inference,
        require_identity_initialization=False,
    )
    audit = {
        "schema": f"{INTEGRATED_CANDIDATE_SCHEMA}/load_audit",
        "dataset": FORMAL_DATASET,
        "role": FORMAL_PARENT_ROLE,
        "epoch": selected_epoch,
        "artifact_file_sha256": artifact_file_sha256,
        "state_key_count": len(state),
        "state_semantic_sha256": state_semantic_sha256(state),
        "current_base_state_key_count": len(base_state),
        "current_base_state_semantic_sha256": current_inference_semantic_sha,
        "repair_state_key_count": len(repair_state),
        "repair_state_semantic_sha256": state_semantic_sha256(repair_state),
        "state_hash_algorithm": STATE_SEMANTIC_HASH_ALGORITHM,
        "strict_load": True,
        "validation": validation,
        "performance_acceptance_margin": None,
        **OFFICIAL_FALSE_FLAGS,
    }
    return model, audit


__all__ = [
    "CURRENT_INFERENCE_STATE_KEY_COUNT",
    "CURRENT_TRAINING_STATE_KEY_COUNT",
    "FORMAL_DATASET",
    "FORMAL_PARENT_ROLE",
    "FORMAL_REPAIR_INITIALIZATION_SEED",
    "FORMAL_SEED",
    "FrozenIRSTDContext",
    "INTEGRATED_PARAMETER_COUNT",
    "INTEGRATED_CANDIDATE_SCHEMA",
    "INTEGRATED_STATE_KEY_COUNT",
    "IRSTD_BGCR_INTEGRATION_VERSION",
    "IRSTD_CRR_STATE_PREFIX",
    "IRSTDBGCRIntegrationError",
    "OOF_SELECTOR_SCHEMA",
    "STATE_SEMANTIC_HASH_ALGORITHM",
    "TPD8NER4QFG2IRSTDCRRInferenceSCTransNet",
    "audit_frozen_current_base",
    "build_formal_irstd_bgcr_model",
    "load_current_into_frozen_base_strictly",
    "load_formal_irstd_bgcr_integrated_candidate",
    "strip_current_survival_state_strict",
    "validate_formal_irstd_bgcr_model",
]
