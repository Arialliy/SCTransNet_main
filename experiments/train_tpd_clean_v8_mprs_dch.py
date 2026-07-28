#!/usr/bin/env python3
"""Training entry for TPD-Clean V8-MPRS-DCH.

V8 changes only the Saliency representation inside the two shallow
SCTransNet embeddings.  The ordinary optimizer, data pipeline, six-output
objective, validation metrics, and checkpoint selection remain owned by
``experiments.train_tpd_pilot``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import train_tpd_pilot as base  # noqa: E402
from model.SCTransNet import SCTransNet  # noqa: E402
from model.tpd_clean_v8_mprs_dch import (  # noqa: E402
    PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
    SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS,
    TPDCleanV8MPRSDCHBlock,
    clean_v8_mprs_dch_variant_spec,
    parameter_count,
    replace_shallow_embeddings_clean_v8_mprs_dch,
)


TOTAL_PARAMETERS = 10_843_155
SHALLOW_EMBEDDING_PARAMETERS = 66_176
PHASE_TIED_PROJECTION_FORMULA = "Wt[o,c]=sum_p(Wk[o,4c+p]),p=0..3"
MPRS_SOURCE_FORMULA = "S_p=(max_q(Z_q)-C0)+(Z_p-C0)/3"
MPRS_MASS_FORMULA = "sum_p(S_p)=4*(max_p(Z_p)-C0)"
MPRS_REUSE_FORMULA = "Sa8=Sa7+((K-b)-Ca)/3"
CONTEXT_CODE_FORMULA = (
    "Q=tanh((Ca-mean_hw(Ca))/"
    "sqrt(mean_hw((Ca-mean_hw(Ca))^2)+eps));eps=1e-6;"
    "V=0.5*(Q-mean_hw(Q))"
)
FULL_HEADROOM_FORMULA = "H=1+abs(a)*(1-abs(a))*V"
CAPACITY_HEADROOM_FORMULA = "H=1"
FUSION_FORMULA = "K+Sa8*(a*H),a=tanh(saliency_scale)"


def _validate_replacement_topology(model: SCTransNet) -> None:
    """Require the registered four-plus-three MPRS block topology."""

    for name, expected_blocks in (("embeddings_1", 4), ("embeddings_2", 3)):
        embedding = getattr(model.mtc, name, None)
        blocks = getattr(embedding, "blocks", None)
        if blocks is None or len(blocks) != expected_blocks:
            raise RuntimeError(
                f"TPD-Clean V8-MPRS-DCH {name} topology mismatch"
            )
        if not all(
            isinstance(block, TPDCleanV8MPRSDCHBlock) for block in blocks
        ):
            raise RuntimeError(
                f"TPD-Clean V8-MPRS-DCH {name} contains a foreign block"
            )


def build_clean_v8_mprs_dch_model(
    variant: str,
    seed: int,
) -> Tuple[SCTransNet, Dict[str, Any]]:
    """Build SCTransNet with only ``mtc.embeddings_1/2`` replaced."""

    variant = variant.lower()
    spec = clean_v8_mprs_dch_variant_spec(variant)
    base.seed_everything(seed)
    model = SCTransNet(base.get_SCTrans_config(), mode="train", deepsuper=True)
    model.apply(base.weights_init_kaiming)
    replacements = replace_shallow_embeddings_clean_v8_mprs_dch(
        model,
        variant,
    )
    if set(replacements) != {"embeddings_1", "embeddings_2"}:
        raise RuntimeError(
            "TPD-Clean V8-MPRS-DCH must replace only "
            "embeddings_1 and embeddings_2"
        )
    for replacement in replacements.values():
        replacement.apply(base.weights_init_kaiming)
    _validate_replacement_topology(model)

    shallow_parameters = sum(
        parameter_count(module) for module in replacements.values()
    )
    total_parameters = parameter_count(model)
    if total_parameters != TOTAL_PARAMETERS:
        raise RuntimeError(
            "TPD-Clean V8-MPRS-DCH total parameter count mismatch: "
            f"{total_parameters} != {TOTAL_PARAMETERS}"
        )
    if shallow_parameters != SHALLOW_EMBEDDING_PARAMETERS:
        raise RuntimeError(
            "TPD-Clean V8-MPRS-DCH shallow parameter count mismatch: "
            f"{shallow_parameters} != {SHALLOW_EMBEDDING_PARAMETERS}"
        )

    primary = variant == PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT
    if bool(spec["primary_candidate"]) is not primary:
        raise RuntimeError(
            "TPD-Clean V8-MPRS-DCH primary-candidate metadata mismatch"
        )
    metadata = {
        "variant": variant,
        "variant_spec": dict(spec),
        "candidate_family": spec["candidate_family"],
        "primary_candidate": primary,
        "mainline_contract": spec["mainline_contract"],
        "semantic_sources": spec["semantic_sources"],
        "kcs_only": True,
        "fourth_parallel_branch_added": spec[
            "fourth_parallel_branch_added"
        ],
        "replaced_embeddings": ("mtc.embeddings_1", "mtc.embeddings_2"),
        "context_modulates_saliency_only": True,
        "context_gate": spec["context_gate"],
        "context_reference": spec["context_reference"],
        "context_headroom": spec["context_headroom"],
        "phase_order": spec["phase_order"],
        "pixel_unshuffle_channel_order": spec[
            "pixel_unshuffle_channel_order"
        ],
        "context_projection": spec["context_projection"],
        "phase_tied_projection": spec["context_projection"],
        "phase_tied_projection_formula": PHASE_TIED_PROJECTION_FORMULA,
        "saliency_representation": spec["saliency_representation"],
        "saliency_formula": spec["saliency_formula"],
        "saliency_source_equation": MPRS_SOURCE_FORMULA,
        "saliency_mass_invariant": spec["saliency_mass_invariant"],
        "saliency_mass_equation": MPRS_MASS_FORMULA,
        "saliency_nonnegative": spec["saliency_nonnegative"],
        "saliency_projection": spec["saliency_projection"],
        "saliency_forward_implementation": spec[
            "saliency_forward_implementation"
        ],
        "saliency_reuse_equation": MPRS_REUSE_FORMULA,
        "phase_contrast_parameters": 0,
        "phase_contrast_buffers": 0,
        "derived_projection_parameters": 0,
        "derived_projection_buffers": 0,
        "context_code_formula": CONTEXT_CODE_FORMULA,
        "context_modulation_spatial_mean": "zero_up_to_fp32_roundoff",
        "context_headroom_spatial_mean": "one_up_to_fp32_roundoff",
        "context_headroom_formula": (
            FULL_HEADROOM_FORMULA
            if primary
            else CAPACITY_HEADROOM_FORMULA
        ),
        "fusion_formula": spec["fusion_formula"],
        "fusion_equation": FUSION_FORMULA,
        "learned_scales_per_block": spec["learned_scales_per_block"],
        "scale_parameter": spec["scale_parameter"],
        "headroom_bound": "0.75<=H<=1.25",
        "coefficient_bound": "abs(a*H)<=1",
        "zero_saliency_reference": "R=0",
        "zero_scale_reference": spec["zero_scale_reference"],
        "zero_scale_first_order_reference": spec[
            "zero_scale_first_order_reference"
        ],
        "state_compatible_with": spec["state_compatible_with"],
        "cross_version_exact_resume_supported": False,
        "projection_precision": "float32_in_formal_amp_off_path",
        "context_precision": "float32_in_formal_amp_off_path",
        "coefficient_precision": "float32_in_formal_amp_off_path",
        "residual_output_dtype": "feature_dtype",
        "standard_forward_conv2d_calls_per_block": 3,
        "total_parameters": total_parameters,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "shallow_embedding_parameters": shallow_parameters,
        "shared_initialization_sha256": base.model_checksum(
            model,
            exclude_shallow=True,
        ),
        "full_initialization_sha256": base.model_checksum(model),
    }
    return model, metadata


def main() -> None:
    base.SUPPORTED_VARIANTS = SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS
    base.build_model = build_clean_v8_mprs_dch_model
    base.main()


__all__ = [
    "CAPACITY_HEADROOM_FORMULA",
    "CONTEXT_CODE_FORMULA",
    "FULL_HEADROOM_FORMULA",
    "FUSION_FORMULA",
    "MPRS_MASS_FORMULA",
    "MPRS_REUSE_FORMULA",
    "MPRS_SOURCE_FORMULA",
    "PHASE_TIED_PROJECTION_FORMULA",
    "SHALLOW_EMBEDDING_PARAMETERS",
    "TOTAL_PARAMETERS",
    "build_clean_v8_mprs_dch_model",
    "main",
]


if __name__ == "__main__":
    main()
