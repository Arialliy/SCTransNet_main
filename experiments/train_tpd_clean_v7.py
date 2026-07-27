#!/usr/bin/env python3
"""Thin training entry for phase-resolved TPD-Clean-v7 candidates.

V7 changes only the Saliency alignment inside the established
Keep--Context--Saliency tokenizer.  It replaces ``mtc.embeddings_1/2`` and
delegates the inherited optimization loop to ``train_tpd_pilot``.  Importing
or building this module never launches training.
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
from model.tpd_clean_v7 import (  # noqa: E402
    PRIMARY_CLEAN_V7_VARIANT,
    SUPPORTED_CLEAN_V7_VARIANTS,
    clean_v7_variant_spec,
    parameter_count,
    replace_shallow_embeddings_clean_v7,
)


TOTAL_PARAMETERS = 10_843_155
SHALLOW_EMBEDDING_PARAMETERS = 66_176
CONTEXT_PROJECTION_FORMULA = "Wt[o,c]=sum_p(Wk[o,c,p]),p=TL,TR,BL,BR"
PHASE_RESOLVED_SALIENCY_FORMULA = (
    "Z=PixelUnshuffle2(X);C0=AvgPool2(X);D_p=ReLU(Z_p-C0);"
    "Sa[o]=sum_c,p(Wk[o,c,p]*D[c,p])"
)
CONTEXT_CODE_FORMULA = (
    "Q=tanh((Ca-mean_hw(Ca))/"
    "sqrt(mean_hw((Ca-mean_hw(Ca))^2)+eps));eps=1e-6;"
    "V=g*0.5*(Q-mean_hw(Q))"
)
FULL_HEADROOM_FORMULA = "g=1;H=1+0.5*(1-abs(a))*V"
CAPACITY_HEADROOM_FORMULA = "g=0;H=1"
FUSION_FORMULA = "K+Sa*(a*H),a=tanh(saliency_scale)"


def build_clean_v7_model(
    variant: str,
    seed: int,
) -> Tuple[SCTransNet, Dict[str, Any]]:
    """Build SCTransNet with only shallow embeddings replaced by V7."""

    variant = variant.lower()
    spec = clean_v7_variant_spec(variant)
    base.seed_everything(seed)
    model = SCTransNet(base.get_SCTrans_config(), mode="train", deepsuper=True)
    model.apply(base.weights_init_kaiming)
    replacements = replace_shallow_embeddings_clean_v7(model, variant)
    if set(replacements) != {"embeddings_1", "embeddings_2"}:
        raise RuntimeError(
            "TPD-Clean-v7 must replace only embeddings_1 and embeddings_2"
        )
    for replacement in replacements.values():
        replacement.apply(base.weights_init_kaiming)

    shallow_parameters = sum(
        parameter_count(module) for module in replacements.values()
    )
    total_parameters = parameter_count(model)
    if total_parameters != TOTAL_PARAMETERS:
        raise RuntimeError(
            "TPD-Clean-v7 total parameter count mismatch: "
            f"{total_parameters} != {TOTAL_PARAMETERS}"
        )
    if shallow_parameters != SHALLOW_EMBEDDING_PARAMETERS:
        raise RuntimeError(
            "TPD-Clean-v7 shallow parameter count mismatch: "
            f"{shallow_parameters} != {SHALLOW_EMBEDDING_PARAMETERS}"
        )

    primary = variant == PRIMARY_CLEAN_V7_VARIANT
    if bool(spec["primary_candidate"]) is not primary:
        raise RuntimeError("TPD-Clean-v7 primary-candidate metadata mismatch")
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
        "only_model_change_from_v6": "phase_resolved_saliency_alignment",
        "context_and_keep_paths_match_v6": True,
        "context_modulates_saliency_only": True,
        "context_reference": spec["context_reference"],
        "context_code": spec["context_code"],
        "context_gate": spec["context_gate"],
        "context_modulation": spec["context_modulation"],
        "context_headroom": spec["context_headroom"],
        "fusion_support": spec["fusion_support"],
        "phase_order": spec["phase_order"],
        "pixel_unshuffle_channel_order": spec[
            "pixel_unshuffle_channel_order"
        ],
        "context_projection": spec["context_projection"],
        "context_projection_formula": CONTEXT_PROJECTION_FORMULA,
        "saliency_representation": spec["saliency_representation"],
        "saliency_formula": spec["saliency_formula"],
        "saliency_projection": spec["saliency_projection"],
        "phase_resolved_saliency_formula": (
            PHASE_RESOLVED_SALIENCY_FORMULA
        ),
        "derived_projection_parameters": 0,
        "derived_projection_buffers": 0,
        "context_code_formula": CONTEXT_CODE_FORMULA,
        "context_modulation_spatial_mean": "zero_up_to_fp32_roundoff",
        "context_headroom_spatial_mean": "one_up_to_fp32_roundoff",
        "residual_mean_preserving": False,
        "context_headroom_formula": (
            FULL_HEADROOM_FORMULA
            if primary
            else CAPACITY_HEADROOM_FORMULA
        ),
        "fusion_formula": spec["fusion_formula"],
        "fusion_equation": FUSION_FORMULA,
        "learned_scales_per_block": spec["learned_scales_per_block"],
        "scale_parameter": spec["scale_parameter"],
        "headroom_bound": "0.5<=H<=1.5",
        "coefficient_bound": "abs(a*H)<=1",
        "residual_bound": "abs(R)<=abs(Sa)",
        "zero_saliency_reference": "R=0",
        "zero_scale_reference": spec["zero_scale_reference"],
        "state_compatible_with": spec["state_compatible_with"],
        "projection_precision": "float32_in_formal_amp_off_path",
        "context_precision": "float32_in_formal_amp_off_path",
        "coefficient_precision": "float32_in_formal_amp_off_path",
        "residual_output_dtype": "feature_dtype",
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
    base.SUPPORTED_VARIANTS = SUPPORTED_CLEAN_V7_VARIANTS
    base.build_model = build_clean_v7_model
    base.main()


if __name__ == "__main__":
    main()
