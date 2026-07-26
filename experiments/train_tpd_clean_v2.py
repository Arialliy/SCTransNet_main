#!/usr/bin/env python3
"""Train the isolated SPD-anchored TPD-v2 candidates.

The data split, optimizer, checkpoint selection, metrics, and output contract
are inherited unchanged from ``train_tpd_pilot``.  Only the accepted variants
and shallow-embedding builder are replaced.
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
from model.tpd_clean import (  # noqa: E402
    SUPPORTED_CLEAN_VARIANTS,
    parameter_count,
    replace_shallow_embeddings_clean,
)


def build_clean_model(variant: str, seed: int) -> Tuple[SCTransNet, Dict[str, Any]]:
    """Build a paired SCTransNet whose only change is embeddings_1/2."""
    base.seed_everything(seed)
    model = SCTransNet(base.get_SCTrans_config(), mode="train", deepsuper=True)
    model.apply(base.weights_init_kaiming)
    replacements = replace_shallow_embeddings_clean(model, variant)
    for replacement in replacements.values():
        replacement.apply(base.weights_init_kaiming)
    shallow_parameters = sum(parameter_count(module) for module in replacements.values())
    metadata = {
        "variant": variant,
        "candidate_family": "spd_anchored_tpd_clean_v2",
        "total_parameters": parameter_count(model),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "shallow_embedding_parameters": shallow_parameters,
        "shared_initialization_sha256": base.model_checksum(
            model, exclude_shallow=True
        ),
        "full_initialization_sha256": base.model_checksum(model),
    }
    return model, metadata


def main() -> None:
    # ``parse_args`` and ``main`` resolve these globals from the base module at
    # call time.  This preserves the sealed v1 runner byte-for-byte.
    base.SUPPORTED_VARIANTS = SUPPORTED_CLEAN_VARIANTS
    base.build_model = build_clean_model
    base.main()


if __name__ == "__main__":
    main()
