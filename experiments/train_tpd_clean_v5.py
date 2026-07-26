#!/usr/bin/env python3
"""Training entry for the isolated TPD-Clean-v5 standby candidates.

This module only supplies the model builder and the same inherited training
entry used by earlier clean candidates.  Creating this file does not launch a
run; v5 formal training remains conditional on the completed v4 Gate A--E
decision.
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
from model.tpd_clean_v5 import (  # noqa: E402
    CONTEXT_SELECTOR_CEILING,
    CONTEXT_SELECTOR_FLOOR,
    PRIMARY_CLEAN_V5_VARIANT,
    SUPPORTED_CLEAN_V5_VARIANTS,
    clean_v5_variant_spec,
    parameter_count,
    replace_shallow_embeddings_clean_v5,
)


def build_clean_v5_model(
    variant: str,
    seed: int,
) -> Tuple[SCTransNet, Dict[str, Any]]:
    """Build an SCTransNet whose only change is the v5 KCS embedding."""

    variant = variant.lower()
    spec = clean_v5_variant_spec(variant)
    base.seed_everything(seed)
    model = SCTransNet(base.get_SCTrans_config(), mode="train", deepsuper=True)
    model.apply(base.weights_init_kaiming)
    replacements = replace_shallow_embeddings_clean_v5(model, variant)
    for replacement in replacements.values():
        replacement.apply(base.weights_init_kaiming)
    shallow_parameters = sum(
        parameter_count(module) for module in replacements.values()
    )
    metadata = {
        "variant": variant,
        "candidate_family": (
            "spd_anchored_tpd_clean_v5_positive_context_selector"
        ),
        "primary_candidate": variant == PRIMARY_CLEAN_V5_VARIANT,
        "mainline_contract": "Keep-Context-Saliency",
        "fourth_parallel_branch_added": False,
        "context_reference": spec["context_reference"],
        "context_code": spec["context_code"],
        "context_selector": spec["context_selector"],
        "context_selector_floor": CONTEXT_SELECTOR_FLOOR,
        "context_selector_ceiling": CONTEXT_SELECTOR_CEILING,
        "fusion_support": spec["fusion_support"],
        "fusion_formula": (
            "K+S*tanh(saliency_scale"
            "*(1+0.5*context_code))"
        ),
        "learned_scales_per_block": 1,
        "residual_bound": (
            "absolute_residual_at_most_absolute_saliency"
        ),
        "zero_scale_reference": "dense_spd_exact",
        "total_parameters": parameter_count(model),
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
    base.SUPPORTED_VARIANTS = SUPPORTED_CLEAN_V5_VARIANTS
    base.build_model = build_clean_v5_model
    base.main()


if __name__ == "__main__":
    main()
