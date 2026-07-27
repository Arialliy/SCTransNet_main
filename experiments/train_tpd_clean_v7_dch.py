#!/usr/bin/env python3
"""Thin training entry for the isolated TPD-Clean V7-DCH pair.

V7-DCH preserves the Keep--Context--Saliency tokenizer mainline and replaces
only ``mtc.embeddings_1/2``.  This module owns the DCH builder and metadata,
then delegates the ordinary training loop to :mod:`experiments.train_tpd_pilot`.
Importing it does not mutate the shared runner or start training.
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
from model.tpd_clean_v7_dch import (  # noqa: E402
    PRIMARY_CLEAN_V7_DCH_VARIANT,
    SUPPORTED_CLEAN_V7_DCH_VARIANTS,
    clean_v7_dch_variant_spec,
    parameter_count,
    replace_shallow_embeddings_clean_v7_dch,
)


TOTAL_PARAMETERS = 10_843_155
SHALLOW_EMBEDDING_PARAMETERS = 66_176
PHASE_TIED_PROJECTION_FORMULA = "Wt[o,c]=sum_p(Wk[o,4c+p]),p=0..3"
CONTEXT_CODE_FORMULA = (
    "Q=tanh((Ca-mean_hw(Ca))/"
    "sqrt(mean_hw((Ca-mean_hw(Ca))^2)+eps));eps=1e-6;"
    "V=0.5*(Q-mean_hw(Q))"
)
FULL_HEADROOM_FORMULA = "H=1+abs(a)*(1-abs(a))*V"
CAPACITY_HEADROOM_FORMULA = "H=1"
FUSION_FORMULA = "K+Sa*(a*H),a=tanh(saliency_scale)"


def build_clean_v7_dch_model(
    variant: str,
    seed: int,
) -> Tuple[SCTransNet, Dict[str, Any]]:
    """Build SCTransNet with only its two shallow embeddings replaced."""

    variant = variant.lower()
    spec = clean_v7_dch_variant_spec(variant)
    base.seed_everything(seed)
    model = SCTransNet(base.get_SCTrans_config(), mode="train", deepsuper=True)
    model.apply(base.weights_init_kaiming)
    replacements = replace_shallow_embeddings_clean_v7_dch(model, variant)
    if set(replacements) != {"embeddings_1", "embeddings_2"}:
        raise RuntimeError(
            "TPD-Clean V7-DCH must replace only embeddings_1 and embeddings_2"
        )
    for replacement in replacements.values():
        replacement.apply(base.weights_init_kaiming)

    shallow_parameters = sum(
        parameter_count(module) for module in replacements.values()
    )
    total_parameters = parameter_count(model)
    if total_parameters != TOTAL_PARAMETERS:
        raise RuntimeError(
            "TPD-Clean V7-DCH total parameter count mismatch: "
            f"{total_parameters} != {TOTAL_PARAMETERS}"
        )
    if shallow_parameters != SHALLOW_EMBEDDING_PARAMETERS:
        raise RuntimeError(
            "TPD-Clean V7-DCH shallow parameter count mismatch: "
            f"{shallow_parameters} != {SHALLOW_EMBEDDING_PARAMETERS}"
        )

    primary = variant == PRIMARY_CLEAN_V7_DCH_VARIANT
    if bool(spec["primary_candidate"]) is not primary:
        raise RuntimeError(
            "TPD-Clean V7-DCH primary-candidate metadata mismatch"
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
        "context_code": spec["context_code"],
        "context_modulation": spec["context_modulation"],
        "context_headroom": spec["context_headroom"],
        "fusion_support": spec["fusion_support"],
        "phase_tied_projection": spec["phase_tied_projection"],
        "phase_tied_projection_formula": PHASE_TIED_PROJECTION_FORMULA,
        "pixel_unshuffle_channel_order": spec[
            "pixel_unshuffle_channel_order"
        ],
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
        "headroom_bound": "0.75<=H<=1.25",
        "coefficient_bound": "abs(a*H)<=1",
        "residual_bound": "abs(R)<=abs(Sa)",
        "zero_saliency_reference": "R=0",
        "zero_scale_reference": spec["zero_scale_reference"],
        "zero_scale_first_order_reference": spec[
            "zero_scale_first_order_reference"
        ],
        "context_residual_near_zero_order": "O(abs(a)^2)",
        "second_order_differentiability_claimed": False,
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
    base.SUPPORTED_VARIANTS = SUPPORTED_CLEAN_V7_DCH_VARIANTS
    base.build_model = build_clean_v7_dch_model
    base.main()


if __name__ == "__main__":
    main()
