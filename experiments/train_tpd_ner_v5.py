#!/usr/bin/env python3
"""Builder and explicit CLI entry for the isolated V5-NER 2x2 matrix.

Importing this module only exposes model construction.  It creates no run,
process, output directory, or gate connection.  Calling :func:`main`
explicitly delegates the selected variant to the existing training CLI.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import train_tpd_pilot as base  # noqa: E402
from experiments.tpd_ner_runtime import guarded_training_runtime  # noqa: E402
from model.tpd_clean_v5 import (  # noqa: E402
    CONTEXT_SELECTOR_CEILING,
    CONTEXT_SELECTOR_FLOOR,
    PRIMARY_CLEAN_V5_VARIANT,
    TPDCleanV5Block,
)
from model.tpd_ner_v5 import (  # noqa: E402
    CapacityMatchedProgressiveBlock,
    EVIDENCE_NODE_NAMES,
    PROGRESSIVE_TOKENIZER,
    RELAY_STAGE_ORDER,
    TPDNERV5SCTransNet,
    relay_parameter_count,
)


CONSTRUCTION_SCHEMA = "sctransnet_tpd_ner_v5_explicit_five_node_v1"
RELAY_WIDTH = 8
UINT32_MODULUS = 2**32
RELAY_INITIALIZATION_SEED_OFFSET = 0x5EED5EED
SUPPORTED_TPD_NER_V5_VARIANTS = (
    "tpd_clean_v5_full_relay_off",
    "tpd_clean_v5_full_relay_on",
    "progressive_relay_off",
    "progressive_relay_on",
)
# Short alias retained for callers that follow the legacy NER naming pattern.
SUPPORTED_NER_V5_VARIANTS = SUPPORTED_TPD_NER_V5_VARIANTS

_VARIANT_SPECS: Mapping[str, Mapping[str, object]] = {
    "tpd_clean_v5_full_relay_off": {
        "tokenizer_variant": PRIMARY_CLEAN_V5_VARIANT,
        "relay_enabled": False,
        "relay_pair": "tpd_clean_v5_full",
        "relay_off_reference": "tpd_clean_v5_full_relay_off",
    },
    "tpd_clean_v5_full_relay_on": {
        "tokenizer_variant": PRIMARY_CLEAN_V5_VARIANT,
        "relay_enabled": True,
        "relay_pair": "tpd_clean_v5_full",
        "relay_off_reference": "tpd_clean_v5_full_relay_off",
    },
    "progressive_relay_off": {
        "tokenizer_variant": PROGRESSIVE_TOKENIZER,
        "relay_enabled": False,
        "relay_pair": "progressive",
        "relay_off_reference": "progressive_relay_off",
    },
    "progressive_relay_on": {
        "tokenizer_variant": PROGRESSIVE_TOKENIZER,
        "relay_enabled": True,
        "relay_pair": "progressive",
        "relay_off_reference": "progressive_relay_off",
    },
}


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _state_checksum(
    model: nn.Module,
    *,
    excluded_prefixes: Tuple[str, ...] = (),
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(excluded_prefixes):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _v5_scales(model: TPDNERV5SCTransNet) -> Tuple[nn.Parameter, ...]:
    if model.tokenizer_variant != PRIMARY_CLEAN_V5_VARIANT:
        return ()
    scales = []
    for name in ("embeddings_1", "embeddings_2"):
        embedding = getattr(model.mtc, name)
        for block in embedding.blocks:
            if not isinstance(block, TPDCleanV5Block):
                raise TypeError(f"{name} contains a non-V5 block")
            if hasattr(block, "context_scale"):
                raise RuntimeError("V5 must not add a separate Context scale")
            scales.append(block.saliency_scale)
    return tuple(scales)


def _progressive_gains(
    model: TPDNERV5SCTransNet,
) -> Tuple[nn.Parameter, ...]:
    if model.tokenizer_variant != PROGRESSIVE_TOKENIZER:
        return ()
    gains = []
    for name in ("embeddings_1", "embeddings_2"):
        embedding = getattr(model.mtc, name)
        for block in embedding.blocks:
            if not isinstance(block, CapacityMatchedProgressiveBlock):
                raise TypeError(f"{name} contains a non-matched Progressive block")
            gains.append(block.channel_gain)
    return tuple(gains)


def _validate_zero_initialization(model: TPDNERV5SCTransNet) -> None:
    scales = _v5_scales(model)
    if model.tokenizer_variant == PRIMARY_CLEAN_V5_VARIANT:
        if len(scales) != 7:
            raise RuntimeError(f"V5 requires seven single scales, got {len(scales)}")
        if any(int(torch.count_nonzero(scale)) != 0 for scale in scales):
            raise RuntimeError("V5 saliency scales must initialize to zero")
    gains = _progressive_gains(model)
    if model.tokenizer_variant == PROGRESSIVE_TOKENIZER:
        if len(gains) != 7:
            raise RuntimeError(
                f"Progressive requires seven channel gains, got {len(gains)}"
            )
        if any(int(torch.count_nonzero(gain)) != 0 for gain in gains):
            raise RuntimeError("Progressive channel gains must initialize to zero")
    if not model.relay_enabled:
        if hasattr(model, "tpd_ner"):
            raise RuntimeError("relay-off must not register relay parameters")
        return
    gates = getattr(model.tpd_ner, "gates", None)
    if not isinstance(gates, nn.ModuleDict) or tuple(gates) != ("4", "3", "2"):
        raise TypeError("relay must expose the q4/q3/q2 gates")
    for stage, gate in gates.items():
        if int(torch.count_nonzero(gate.weight)) != 0:
            raise RuntimeError(f"stage-{stage} gate weight must initialize to zero")
        if gate.bias is not None and int(torch.count_nonzero(gate.bias)) != 0:
            raise RuntimeError(f"stage-{stage} gate bias must initialize to zero")


def build_tpd_ner_v5_model(
    variant: str,
    seed: int,
    *,
    config: Any | None = None,
    img_size: int = 256,
    relay_width: int = RELAY_WIDTH,
) -> Tuple[TPDNERV5SCTransNet, Dict[str, Any]]:
    """Build one member of the isolated ``Tokenizer x Relay`` matrix.

    For a fixed seed, relay-off and relay-on execute identical construction
    and initialization steps until all common parameters are complete.  The
    on variant then adds only ``tpd_ner.*`` parameters and zeroes its gates.
    """

    variant = variant.lower()
    if variant not in _VARIANT_SPECS:
        raise ValueError(
            f"Unknown V5-NER variant {variant!r}; "
            f"choices={SUPPORTED_TPD_NER_V5_VARIANTS}"
        )
    if relay_width < 1:
        raise ValueError(f"relay_width must be positive, got {relay_width}")
    spec = _VARIANT_SPECS[variant]
    tokenizer_variant = str(spec["tokenizer_variant"])
    relay_enabled = bool(spec["relay_enabled"])
    relay_initialization_seed = (
        int(seed) + RELAY_INITIALIZATION_SEED_OFFSET
    ) % UINT32_MODULUS

    base.seed_everything(seed)
    selected_config = config if config is not None else base.get_SCTrans_config()
    model = TPDNERV5SCTransNet(
        selected_config,
        img_size=img_size,
        mode="train",
        deepsuper=True,
        tokenizer_variant=tokenizer_variant,
        relay_enabled=relay_enabled,
        relay_width=relay_width,
        install_extension=False,
    )
    model.apply(base.weights_init_kaiming)

    replacements = model.install_evidence_tokenizer()
    for replacement in replacements.values():
        replacement.apply(base.weights_init_kaiming)

    if relay_enabled:
        # A local, tokenizer-independent RNG stream makes the relay tensors
        # identical in P+N and T+N without perturbing any common parameters.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(relay_initialization_seed)
            relay = model.install_nested_relay()["relay"]
            relay.apply(base.weights_init_kaiming)
    model.zero_init_extension_residuals()
    _validate_zero_initialization(model)

    shallow_parameters = sum(
        _parameter_count(module) for module in replacements.values()
    )
    expected_matched_shallow_parameters = sum(
        4 * block.channels * block.channels + 2 * block.channels
        for module in replacements.values()
        for block in module.blocks
    )
    if shallow_parameters != expected_matched_shallow_parameters:
        raise RuntimeError(
            "shallow tokenizer violates the strict V5 capacity formula: "
            f"actual={shallow_parameters} "
            f"expected={expected_matched_shallow_parameters}"
        )
    relay_parameters = relay_parameter_count(model)
    relay_gate_parameters = (
        _parameter_count(model.tpd_ner.gates) if relay_enabled else 0
    )
    total_parameters = _parameter_count(model)
    common_prefix_exclusions = ("tpd_ner.",)
    backbone_prefix_exclusions = (
        "mtc.embeddings_1.",
        "mtc.embeddings_2.",
        "tpd_ner.",
    )
    manifest = model.architecture_manifest()
    is_v5 = tokenizer_variant == PRIMARY_CLEAN_V5_VARIANT
    metadata: Dict[str, Any] = {
        "variant": variant,
        "candidate_family": "tpd_clean_v5_explicit_five_node_ner_v1",
        "construction_schema": CONSTRUCTION_SCHEMA,
        "tokenizer_variant": tokenizer_variant,
        "relay_pair": spec["relay_pair"],
        "relay_off_reference": spec["relay_off_reference"],
        "relay_enabled": relay_enabled,
        "relay_width": relay_width,
        "relay_topology": "q4->q3->q2",
        "relay_parameters": relay_parameters,
        "relay_gate_parameters": relay_gate_parameters,
        "relay_initialization_seed": relay_initialization_seed,
        "relay_initialization_sha256": (
            _state_checksum(model.tpd_ner) if relay_enabled else None
        ),
        "relay_initialization_contract": (
            "tokenizer_independent_local_rng_then_zero_gates"
        ),
        "relay_taps": {
            "evidence_nodes": EVIDENCE_NODE_NAMES,
            "embedding_1_intermediates": 3,
            "embedding_2_intermediates": 2,
            "stage_order": RELAY_STAGE_ORDER,
        },
        "tensor_handoff": "forward_local_explicit",
        "model_class": model.__class__.__name__,
        "mainline_contract": (
            "Keep-Context-Saliency"
            if is_v5
            else "strict_capacity_matched_progressive_control"
        ),
        "semantic_sources": (
            ("Keep", "Context", "Saliency")
            if is_v5
            else ("CapacityMatchedProgressiveConv",)
        ),
        "semantic_source_count": 3 if is_v5 else 1,
        "capacity_contract": (
            "v5_full_capacity_reference"
            if is_v5
            else "same_depth_strict_shallow_parameter_match_to_v5"
        ),
        "shallow_parameter_reference": "tpd_clean_v5_full",
        "shallow_parameter_formula_per_block": "4*C^2+2*C",
        "shallow_parameter_match_verified": True,
        "progressive_topology": (
            None
            if is_v5
            else {
                "embedding_depths": (4, 3),
                "spatial_projection": "Conv2d(C,C,kernel=2,stride=2,bias=True)",
                "channel_gain": "1+tanh(g), g_shape=C, g_init=0",
                "block_formula": (
                    "activation(spatial_projection(X)*(1+tanh(g)))"
                ),
                "sampling_lattice": "aligned_nonoverlapping_2x2",
                "all_capacity_parameters_forward_active": True,
            }
        ),
        "fourth_parallel_branch_added": False,
        "learned_scales_per_block": 1,
        "learned_saliency_scales_per_block": 1 if is_v5 else 0,
        "learned_capacity_gains_per_block": 0 if is_v5 else 1,
        "context_selector": (
            "positive_centered_0p5_to_1p5" if is_v5 else None
        ),
        "context_selector_floor": CONTEXT_SELECTOR_FLOOR if is_v5 else None,
        "context_selector_ceiling": (
            CONTEXT_SELECTOR_CEILING if is_v5 else None
        ),
        "zero_scale_reference": (
            "dense_spd_exact" if is_v5 else "identity_channel_gain"
        ),
        "zero_gate_reference": "paired_relay_off_exact",
        "pair_initialization_contract": (
            "same_seed_common_state_and_step0_output_exact"
        ),
        "relay_state_prefix": "tpd_ner.",
        "relay_on_adds_only_relay_parameters": True,
        "initialization_mode": "fresh_shared_seed",
        "warm_start_applied": False,
        "total_parameters": total_parameters,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "common_parameters": total_parameters - relay_parameters,
        "shallow_embedding_parameters": shallow_parameters,
        "matched_reference_shallow_parameters": (
            expected_matched_shallow_parameters
        ),
        "backbone_initialization_sha256": _state_checksum(
            model,
            excluded_prefixes=backbone_prefix_exclusions,
        ),
        "common_initialization_sha256": _state_checksum(
            model,
            excluded_prefixes=common_prefix_exclusions,
        ),
        "full_initialization_sha256": _state_checksum(model),
        "architecture_manifest": manifest,
        "automatic_launch": False,
        "gate_connection": "none",
    }
    return model, metadata


def main() -> None:
    """Delegate only when this entry is invoked explicitly from a CLI."""

    base.SUPPORTED_VARIANTS = SUPPORTED_TPD_NER_V5_VARIANTS
    base.build_model = build_tpd_ner_v5_model
    with guarded_training_runtime():
        base.main()


__all__ = [
    "CONSTRUCTION_SCHEMA",
    "RELAY_WIDTH",
    "SUPPORTED_NER_V5_VARIANTS",
    "SUPPORTED_TPD_NER_V5_VARIANTS",
    "build_tpd_ner_v5_model",
    "main",
]


if __name__ == "__main__":
    main()
