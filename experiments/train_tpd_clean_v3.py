#!/usr/bin/env python3
"""Train the isolated TPD-Clean-v3 KCS fusion candidates.

The split, augmentation, optimizer, schedule, losses, metrics, and checkpoint
selection remain inherited from ``train_tpd_pilot``.  Only the shallow
``embeddings_1/2`` builder is replaced.
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
from model.tpd_clean_v3 import (  # noqa: E402
    PRIMARY_CLEAN_V3_VARIANT,
    SUPPORTED_CLEAN_V3_VARIANTS,
    clean_v3_variant_spec,
    parameter_count,
    replace_shallow_embeddings_clean_v3,
)


def build_clean_v3_model(
    variant: str, seed: int
) -> Tuple[SCTransNet, Dict[str, Any]]:
    """Build a paired SCTransNet whose only change is KCS embeddings_1/2."""
    variant = variant.lower()
    spec = clean_v3_variant_spec(variant)
    base.seed_everything(seed)
    model = SCTransNet(base.get_SCTrans_config(), mode="train", deepsuper=True)
    model.apply(base.weights_init_kaiming)
    replacements = replace_shallow_embeddings_clean_v3(model, variant)
    for replacement in replacements.values():
        replacement.apply(base.weights_init_kaiming)
    shallow_parameters = sum(
        parameter_count(module) for module in replacements.values()
    )
    metadata = {
        "variant": variant,
        "candidate_family": "spd_anchored_tpd_clean_v3_kcs",
        "primary_candidate": variant == PRIMARY_CLEAN_V3_VARIANT,
        "mainline_contract": "Keep-Context-Saliency",
        "fourth_parallel_branch_added": False,
        "context_reference": spec["context_reference"],
        "fusion_support": spec["fusion_support"],
        "context_code": spec["context_code"],
        "zero_scale_reference": "dense_spd_exact",
        "total_parameters": parameter_count(model),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "shallow_embedding_parameters": shallow_parameters,
        "shared_initialization_sha256": base.model_checksum(
            model, exclude_shallow=True
        ),
        "full_initialization_sha256": base.model_checksum(model),
    }
    return model, metadata


def main() -> None:
    # The base parser and main resolve these globals at call time.  Existing
    # v1/v2 runners remain byte-for-byte unchanged.
    base.SUPPORTED_VARIANTS = SUPPORTED_CLEAN_V3_VARIANTS
    base.build_model = build_clean_v3_model
    base.main()


if __name__ == "__main__":
    main()
