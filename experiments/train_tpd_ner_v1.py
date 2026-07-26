#!/usr/bin/env python3
"""Train the isolated TPD-NER model without changing the active TPD screen.

The primary innovation remains the complete Keep-Context-Saliency
``tpd_clean_full`` embedding.  Nested Evidence Relay (NER) is installed only
as a second decoder-side module on that full embedding.  The data protocol,
optimizer, losses, validation metrics, and checkpoint policy are delegated to
``train_tpd_pilot`` unchanged.

This entry point is intended for one CUDA device per process.  It does not use
``nn.DataParallel``; four-GPU experiments should launch four independent
processes, matching the existing repository protocol.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import train_tpd_pilot as base  # noqa: E402
from model.tpd_clean import (  # noqa: E402
    CleanTPD2,
    parameter_count,
)
from experiments.tpd_ner_runtime import guarded_training_runtime  # noqa: E402
from model.tpd_relay import relay_parameter_count  # noqa: E402
from model.tpd_sctransnet import (  # noqa: E402
    EVIDENCE_NODE_NAMES,
    TPDSCTransNet,
)


SUPPORTED_NER_VARIANTS = ("tpd_clean_full_ner",)
PARENT_VARIANT = "tpd_clean_full"
RELAY_WIDTH = 8
CONSTRUCTION_SCHEMA = "sctransnet_tpd_clean_full_ner_explicit_v1"


def _assert_zero_residual_contract(
    replacements: Mapping[str, nn.Module],
    relay: nn.Module,
) -> None:
    """Verify the exact SPD anchor after all initialization has finished."""
    block_count = 0
    for name in ("embeddings_1", "embeddings_2"):
        embedding = replacements[name]
        blocks = getattr(embedding, "blocks", ())
        for index, block in enumerate(blocks):
            if not isinstance(block, CleanTPD2):
                raise TypeError(f"{name}.blocks[{index}] is not CleanTPD2")
            if not block.use_context or not block.use_saliency:
                raise ValueError(
                    f"{name}.blocks[{index}] is not full Keep-Context-Saliency TPD"
                )
            for scale_name in ("context_scale", "saliency_scale"):
                scale = getattr(block, scale_name)
                if scale is None or torch.count_nonzero(scale).item() != 0:
                    raise RuntimeError(
                        f"{name}.blocks[{index}].{scale_name} must initialize to zero"
                    )
            block_count += 1
    if block_count != 7:
        raise RuntimeError(f"expected seven full TPD blocks, got {block_count}")

    gates = getattr(relay, "gates", None)
    if not isinstance(gates, nn.ModuleDict) or set(gates) != {"2", "3", "4"}:
        raise TypeError("NER must expose exactly the q4/q3/q2 spatial gates")
    for stage, gate in gates.items():
        if torch.count_nonzero(gate.weight).item() != 0:
            raise RuntimeError(f"NER stage-{stage} gate weight must initialize to zero")
        if gate.bias is not None and torch.count_nonzero(gate.bias).item() != 0:
            raise RuntimeError(f"NER stage-{stage} gate bias must initialize to zero")


def build_tpd_ner_model(
    variant: str,
    seed: int,
) -> Tuple[TPDSCTransNet, Dict[str, Any]]:
    """Construct the single approved TPD-NER v1 candidate.

    Initialization order is part of the experiment contract.  The complete
    TPD embedding is paired with SPD first, then NER is initialized, and its
    residual gates are reset to exact zero last.
    """
    if variant not in SUPPORTED_NER_VARIANTS:
        raise ValueError(
            f"Unknown TPD-NER variant {variant!r}; choices={SUPPORTED_NER_VARIANTS}"
        )

    base.seed_everything(seed)
    model = TPDSCTransNet(
        base.get_SCTrans_config(),
        mode="train",
        deepsuper=True,
        relay_width=RELAY_WIDTH,
        install_tpd=False,
    )
    model.apply(base.weights_init_kaiming)

    replacements = model.install_tpd_tokenizer()
    for replacement in replacements.values():
        replacement.apply(base.weights_init_kaiming)

    shared_initialization_sha256 = base.model_checksum(
        model,
        exclude_shallow=True,
    )
    clean_full_anchor_initialization_sha256 = base.model_checksum(model)
    shallow_parameters = sum(
        parameter_count(module) for module in replacements.values()
    )

    parts = model.install_nested_relay()
    relay = parts["relay"]
    relay.apply(base.weights_init_kaiming)
    model.zero_init_target_residuals()
    _assert_zero_residual_contract(replacements, relay)

    relay_parameters = relay_parameter_count(relay)
    relay_gate_parameters = sum(
        parameter.numel() for gate in relay.gates.values()
        for parameter in gate.parameters()
    )
    metadata = {
        "variant": variant,
        "candidate_family": "tpd_clean_full_ner_v1",
        "parent_candidate_family": "spd_anchored_tpd_clean_v2",
        "parent_variant": PARENT_VARIANT,
        "construction_schema": CONSTRUCTION_SCHEMA,
        "model_class": type(model).__name__,
        "tensor_handoff": "forward_local_explicit",
        "innovation_contract": {
            "primary_module": "Keep-Context-Saliency TPD",
            "secondary_module": "Nested Evidence Relay",
            "primary_module_replaced": False,
        },
        "relay_width": RELAY_WIDTH,
        "relay_topology": "q4->q3->q2",
        "relay_taps": {
            "evidence_nodes": EVIDENCE_NODE_NAMES,
            "embeddings_1_intermediates": 3,
            "embeddings_2_intermediates": 2,
        },
        "execution_contract": "one_device_per_process",
        "relay_parameters": relay_parameters,
        "relay_gate_parameters": relay_gate_parameters,
        "shallow_embedding_parameters": shallow_parameters,
        "total_parameters": parameter_count(model),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "initialization_contract": (
            "full TPD Context/Saliency scales and NER q4/q3/q2 gates are "
            "exactly zero after all Kaiming initialization; initial outputs "
            "therefore equal the paired SPD model"
        ),
        "shared_initialization_sha256": shared_initialization_sha256,
        "clean_full_anchor_initialization_sha256": (
            clean_full_anchor_initialization_sha256
        ),
        "full_initialization_sha256": base.model_checksum(model),
    }
    if relay_parameters != 11_291:
        raise RuntimeError(
            f"unexpected width-{RELAY_WIDTH} NER parameter count: {relay_parameters}"
        )
    if relay_gate_parameters != 27:
        raise RuntimeError(
            f"unexpected NER gate parameter count: {relay_gate_parameters}"
        )
    if shallow_parameters != 66_496:
        raise RuntimeError(
            f"unexpected full TPD embedding parameter count: {shallow_parameters}"
        )
    return model, metadata


def main() -> None:
    """Delegate the unchanged training protocol to the approved NER builder."""
    base.SUPPORTED_VARIANTS = SUPPORTED_NER_VARIANTS
    base.build_model = build_tpd_ner_model
    with guarded_training_runtime():
        base.main()


if __name__ == "__main__":
    main()
