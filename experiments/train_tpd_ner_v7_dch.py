#!/usr/bin/env python3
"""Pre-gate builder for the isolated V7-DCH ``Tokenizer x NER`` matrix.

Importing this module constructs nothing and starts no experiment.  The four
members are paired by seed:

* V7-DCH Full, relay off/on;
* capacity-matched Progressive, relay off/on.

For a fixed tokenizer and seed, relay-off/on have exactly the same common
state.  Relay-on is initialized from an independent local RNG stream and its
three spatial gates are then zeroed, making the paired step-zero outputs
exactly equal.

The CLI remains deliberately disabled until the completed V7-DCH report says
that Gates A--E authorize the NER stage.
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
from experiments import tpd_exact_runner as exact_runner  # noqa: E402
from experiments.tpd_extension_warm_start import (  # noqa: E402
    load_parent_into_extension,
)
from model.tpd_clean_v7_dch import (  # noqa: E402
    PRIMARY_CLEAN_V7_DCH_VARIANT,
)
from model.tpd_ner_v7_dch import (  # noqa: E402
    EVIDENCE_NODE_NAMES,
    PROGRESSIVE_TOKENIZER,
    RELAY_STAGE_ORDER,
    TPDNERV7DCHSCTransNet,
    relay_parameter_count,
)


CONSTRUCTION_SCHEMA = "sctransnet_tpd_ner_v7_dch_explicit_five_node_v1"
RELAY_WIDTH = 8
UINT32_MODULUS = 2**32
RELAY_INITIALIZATION_SEED_OFFSET = 0x5EED5EED
FORMAL_LAUNCH_AUTHORIZED = False
FORMAL_GATE_CONNECTION = "awaiting_v7_dch_gates_A_E"

V7_FULL_RELAY_OFF = "tpd_clean_v7_dch_full_relay_off"
V7_FULL_RELAY_ON = "tpd_clean_v7_dch_full_relay_on"
PROGRESSIVE_RELAY_OFF = "progressive_v7_capacity_relay_off"
PROGRESSIVE_RELAY_ON = "progressive_v7_capacity_relay_on"
SUPPORTED_TPD_NER_V7_DCH_VARIANTS = (
    V7_FULL_RELAY_OFF,
    V7_FULL_RELAY_ON,
    PROGRESSIVE_RELAY_OFF,
    PROGRESSIVE_RELAY_ON,
)

EXPECTED_PRODUCTION_COMMON_PARAMETERS = 10_843_155
EXPECTED_PRODUCTION_RELAY_PARAMETERS = 11_291
EXPECTED_PRODUCTION_RELAY_GATE_PARAMETERS = 27
EXPECTED_PRODUCTION_SHALLOW_PARAMETERS = 66_176
EXPECTED_PARENT_VARIANT = "tpd_clean_v7_dch_full"
EXPECTED_PARENT_MAINLINE = "Keep-Context-Saliency"
EXPECTED_PARENT_SEMANTIC_SOURCES = ("Keep", "Context", "Saliency")
EXPECTED_PARENT_FUSION_EQUATION = (
    "K+Sa*(a*H),a=tanh(saliency_scale)"
)

_VARIANT_SPECS: Mapping[str, Mapping[str, object]] = {
    V7_FULL_RELAY_OFF: {
        "tokenizer_variant": PRIMARY_CLEAN_V7_DCH_VARIANT,
        "relay_enabled": False,
        "relay_pair": "tpd_clean_v7_dch_full",
        "relay_off_reference": V7_FULL_RELAY_OFF,
    },
    V7_FULL_RELAY_ON: {
        "tokenizer_variant": PRIMARY_CLEAN_V7_DCH_VARIANT,
        "relay_enabled": True,
        "relay_pair": "tpd_clean_v7_dch_full",
        "relay_off_reference": V7_FULL_RELAY_OFF,
    },
    PROGRESSIVE_RELAY_OFF: {
        "tokenizer_variant": PROGRESSIVE_TOKENIZER,
        "relay_enabled": False,
        "relay_pair": "progressive_v7_capacity",
        "relay_off_reference": PROGRESSIVE_RELAY_OFF,
    },
    PROGRESSIVE_RELAY_ON: {
        "tokenizer_variant": PROGRESSIVE_TOKENIZER,
        "relay_enabled": True,
        "relay_pair": "progressive_v7_capacity",
        "relay_off_reference": PROGRESSIVE_RELAY_OFF,
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


def _tokenizer_controls(
    model: TPDNERV7DCHSCTransNet,
) -> Tuple[nn.Parameter, ...]:
    controls = []
    attribute = (
        "saliency_scale"
        if model.tokenizer_variant == PRIMARY_CLEAN_V7_DCH_VARIANT
        else "channel_gain"
    )
    for embedding_name in ("embeddings_1", "embeddings_2"):
        embedding = getattr(model.mtc, embedding_name)
        for block in embedding.blocks:
            control = getattr(block, attribute, None)
            if not isinstance(control, nn.Parameter):
                raise TypeError(
                    f"{embedding_name} block lacks {attribute} Parameter"
                )
            controls.append(control)
    return tuple(controls)


def _validate_zero_initialization(model: TPDNERV7DCHSCTransNet) -> None:
    controls = _tokenizer_controls(model)
    if len(controls) != 7:
        raise RuntimeError(f"tokenizer requires seven controls, got {len(controls)}")
    if any(int(torch.count_nonzero(control)) != 0 for control in controls):
        raise RuntimeError("all fresh tokenizer controls must initialize to zero")

    if not model.relay_enabled:
        if hasattr(model, "tpd_ner"):
            raise RuntimeError("relay-off must not register tpd_ner")
        return
    gates = getattr(model.tpd_ner, "gates", None)
    if not isinstance(gates, nn.ModuleDict) or tuple(gates) != ("4", "3", "2"):
        raise TypeError("relay must expose the q4/q3/q2 gates")
    for stage, gate in gates.items():
        if int(torch.count_nonzero(gate.weight)) != 0:
            raise RuntimeError(f"stage-{stage} gate weight is not zero")
        if gate.bias is not None and int(torch.count_nonzero(gate.bias)) != 0:
            raise RuntimeError(f"stage-{stage} gate bias is not zero")


def _validate_capacity(
    replacements: Mapping[str, nn.Module],
) -> Tuple[int, int]:
    actual = sum(_parameter_count(module) for module in replacements.values())
    expected = 0
    for module in replacements.values():
        for block in module.blocks:
            channels = int(block.channels)
            expected += 4 * channels * channels + 2 * channels
    if actual != expected:
        raise RuntimeError(
            "tokenizer violates the capacity formula 4*C^2+2*C per block: "
            f"actual={actual}, expected={expected}"
        )
    return actual, expected


def _read_verified_parent_payload(
    checkpoint: str | Path,
    *,
    expected_seed: int,
    expected_checkpoint_role: str,
    expected_checkpoint_sha256: str | None,
) -> Tuple[Path, str, Mapping[str, Any]]:
    path = Path(checkpoint)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"parent checkpoint is not a regular file: {path}")
    content = path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(content).hexdigest()
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise ValueError(
            "parent checkpoint SHA-256 mismatch: "
            f"{checkpoint_sha256} != {expected_checkpoint_sha256}"
        )
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise TypeError("parent checkpoint payload must be a mapping")

    run_identity = payload.get("run_identity")
    model_metadata = payload.get("model_metadata")
    if not isinstance(run_identity, Mapping):
        raise TypeError("parent checkpoint lacks run_identity mapping")
    if not isinstance(model_metadata, Mapping):
        raise TypeError("parent checkpoint lacks model_metadata mapping")
    manifest = model_metadata.get("architecture_manifest")
    if not isinstance(manifest, Mapping):
        raise TypeError("parent checkpoint lacks architecture_manifest")

    variant_values = (
        payload.get("variant"),
        run_identity.get("variant"),
        model_metadata.get("variant"),
        manifest.get("variant"),
    )
    if any(value != EXPECTED_PARENT_VARIANT for value in variant_values):
        raise ValueError(
            "parent identity is not V7-DCH Full: "
            f"variants={variant_values}"
        )
    if payload.get("seed") != expected_seed:
        raise ValueError("parent checkpoint seed mismatch")
    if run_identity.get("seed") != expected_seed:
        raise ValueError("parent run_identity seed mismatch")
    if payload.get("checkpoint_role") != expected_checkpoint_role:
        raise ValueError("parent checkpoint role mismatch")
    if payload.get("official_test_accessed") is not False:
        raise ValueError("parent checkpoint test-access identity mismatch")
    if model_metadata.get("context_gate") != 1.0:
        raise ValueError("parent checkpoint context_gate is not Full=1.0")
    if model_metadata.get("mainline_contract") != EXPECTED_PARENT_MAINLINE:
        raise ValueError("parent checkpoint mainline identity mismatch")
    if tuple(model_metadata.get("semantic_sources", ())) != (
        EXPECTED_PARENT_SEMANTIC_SOURCES
    ):
        raise ValueError("parent checkpoint semantic-source identity mismatch")
    if manifest.get("mainline_contract") != EXPECTED_PARENT_MAINLINE:
        raise ValueError("parent manifest mainline identity mismatch")
    if tuple(manifest.get("semantic_sources", ())) != (
        EXPECTED_PARENT_SEMANTIC_SOURCES
    ):
        raise ValueError("parent manifest semantic-source identity mismatch")
    if manifest.get("fusion_equation") != EXPECTED_PARENT_FUSION_EQUATION:
        raise ValueError("parent manifest fusion equation mismatch")
    if (
        manifest.get("shallow_embedding_parameters")
        != EXPECTED_PRODUCTION_SHALLOW_PARAMETERS
    ):
        raise ValueError("parent manifest shallow capacity mismatch")
    if (
        manifest.get("total_parameters")
        != EXPECTED_PRODUCTION_COMMON_PARAMETERS
    ):
        raise ValueError("parent manifest total capacity mismatch")

    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("parent checkpoint lacks state_dict mapping")
    recorded_state_sha256 = payload.get("state_dict_sha256")
    actual_state_sha256 = exact_runner._state_content_sha256(
        state,
        "V7-DCH Full parent state",
    )
    if recorded_state_sha256 != actual_state_sha256:
        raise ValueError("parent checkpoint state_dict SHA-256 mismatch")
    return path.resolve(), checkpoint_sha256, payload


def load_verified_v7_dch_full_parent(
    checkpoint: str | Path,
    *,
    parent_model: TPDNERV7DCHSCTransNet,
    extension_model: TPDNERV7DCHSCTransNet,
    expected_seed: int,
    expected_checkpoint_role: str,
    expected_checkpoint_sha256: str | None = None,
) -> Dict[str, Any]:
    """Load an identity-verified V7-DCH Full parent into relay off/on.

    ``parent_model`` must be the freshly constructed Full relay-off member for
    the same seed.  Requiring it explicitly prevents a state-compatible
    Capacity checkpoint from silently entering a Full model.
    """

    for label, model in (
        ("parent_model", parent_model),
        ("extension_model", extension_model),
    ):
        if not isinstance(model, TPDNERV7DCHSCTransNet):
            raise TypeError(f"{label} has the wrong model class")
        if model.tokenizer_variant != PRIMARY_CLEAN_V7_DCH_VARIANT:
            raise ValueError(f"{label} is not V7-DCH Full")
    if parent_model.relay_enabled or hasattr(parent_model, "tpd_ner"):
        raise ValueError("parent_model must be the Full relay-off member")

    path, checkpoint_sha256, payload = _read_verified_parent_payload(
        checkpoint,
        expected_seed=expected_seed,
        expected_checkpoint_role=expected_checkpoint_role,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
    )
    state = payload["state_dict"]

    if not extension_model.relay_enabled:
        incompatible = extension_model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("relay-off parent strict-load returned mismatches")
        loaded = extension_model.state_dict()
        if any(not torch.equal(loaded[key], value) for key, value in state.items()):
            raise RuntimeError("relay-off loaded state differs from parent")
        transfer: Dict[str, Any] = {
            "mode": "strict_parent_load",
            "parent_state_key_count": len(state),
            "preserved_new_state_key_count": 0,
        }
    else:
        result = load_parent_into_extension(
            path,
            parent_model=parent_model,
            extension_model=extension_model,
            new_module_prefixes=("tpd_ner",),
            zero_init_prefixes=("tpd_ner.gates",),
            expected_parent_checkpoint_sha256=checkpoint_sha256,
            map_location="cpu",
        )
        transfer = {
            "mode": "strict_parent_to_relay_extension",
            **result.provenance(),
        }

    return {
        "parent_checkpoint_path": str(path),
        "parent_checkpoint_sha256": checkpoint_sha256,
        "parent_variant": EXPECTED_PARENT_VARIANT,
        "parent_seed": expected_seed,
        "parent_checkpoint_role": expected_checkpoint_role,
        "transfer": transfer,
    }


def build_tpd_ner_v7_dch_model(
    variant: str,
    seed: int,
    *,
    config: Any | None = None,
    img_size: int = 256,
    relay_width: int = RELAY_WIDTH,
) -> Tuple[TPDNERV7DCHSCTransNet, Dict[str, Any]]:
    """Build one isolated V7-DCH/Progressive x relay-off/on member."""

    variant = variant.lower()
    if variant not in _VARIANT_SPECS:
        raise ValueError(
            f"unknown V7-DCH NER variant {variant!r}; "
            f"choices={SUPPORTED_TPD_NER_V7_DCH_VARIANTS}"
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
    model = TPDNERV7DCHSCTransNet(
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
    if set(replacements) != {"embeddings_1", "embeddings_2"}:
        raise RuntimeError("exactly embeddings_1/2 must be replaced")
    for replacement in replacements.values():
        replacement.apply(base.weights_init_kaiming)

    if relay_enabled:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(relay_initialization_seed)
            relay = model.install_nested_relay()["relay"]
            relay.apply(base.weights_init_kaiming)
        model.zero_init_relay_gates()
    model.zero_init_fresh_tokenizer_controls()
    _validate_zero_initialization(model)

    shallow_parameters, expected_shallow_parameters = _validate_capacity(
        replacements
    )
    relay_parameters = relay_parameter_count(model)
    relay_gate_parameters = (
        _parameter_count(model.tpd_ner.gates) if relay_enabled else 0
    )
    total_parameters = _parameter_count(model)
    common_parameters = total_parameters - relay_parameters

    production_shape = (
        config is None
        and img_size == 256
        and relay_width == RELAY_WIDTH
    )
    if production_shape:
        if shallow_parameters != EXPECTED_PRODUCTION_SHALLOW_PARAMETERS:
            raise RuntimeError("production shallow parameter count mismatch")
        if common_parameters != EXPECTED_PRODUCTION_COMMON_PARAMETERS:
            raise RuntimeError("production common parameter count mismatch")
        expected_relay = (
            EXPECTED_PRODUCTION_RELAY_PARAMETERS if relay_enabled else 0
        )
        if relay_parameters != expected_relay:
            raise RuntimeError("production relay parameter count mismatch")
        expected_gates = (
            EXPECTED_PRODUCTION_RELAY_GATE_PARAMETERS if relay_enabled else 0
        )
        if relay_gate_parameters != expected_gates:
            raise RuntimeError("production relay gate parameter count mismatch")

    is_v7_dch = tokenizer_variant == PRIMARY_CLEAN_V7_DCH_VARIANT
    manifest = model.architecture_manifest()
    common_prefix_exclusions = ("tpd_ner.",)
    backbone_prefix_exclusions = (
        "mtc.embeddings_1.",
        "mtc.embeddings_2.",
        "tpd_ner.",
    )
    metadata: Dict[str, Any] = {
        "variant": variant,
        "candidate_family": "tpd_clean_v7_dch_explicit_five_node_ner_v1",
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
        "tensor_handoff": "forward_local_explicit_capability_checked",
        "model_class": model.__class__.__name__,
        "mainline_contract": (
            "Keep-Context-Saliency"
            if is_v7_dch
            else "strict_capacity_matched_progressive_control"
        ),
        "semantic_sources": (
            ("Keep", "Context", "Saliency")
            if is_v7_dch
            else ("CapacityMatchedProgressiveConv",)
        ),
        "semantic_source_count": 3 if is_v7_dch else 1,
        "evidence_node_count": 5,
        "evidence_layout": (3, 2),
        "fourth_parallel_branch_added": False,
        "learned_controls_per_block": 1,
        "capacity_contract": "same_depth_exact_4C2_plus_2C",
        "shallow_parameter_formula_per_block": "4*C^2+2*C",
        "shallow_parameter_match_verified": True,
        "shallow_embedding_parameters": shallow_parameters,
        "matched_reference_shallow_parameters": expected_shallow_parameters,
        "zero_control_reference": (
            "v7_dch_dense_phase_projection"
            if is_v7_dch
            else "progressive_identity_channel_gain"
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
        "common_parameters": common_parameters,
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
        "formal_launch_authorized": FORMAL_LAUNCH_AUTHORIZED,
        "automatic_launch": False,
        "gate_connection": FORMAL_GATE_CONNECTION,
    }
    return model, metadata


def main() -> None:
    """Refuse a formal run until the completed V7-DCH gate artifact binds it."""

    raise SystemExit(
        "V7-DCH+NER is implemented for pre-gate verification only; "
        "formal launch requires completed V7-DCH Gates A--E."
    )


__all__ = [
    "CONSTRUCTION_SCHEMA",
    "RELAY_WIDTH",
    "FORMAL_LAUNCH_AUTHORIZED",
    "FORMAL_GATE_CONNECTION",
    "V7_FULL_RELAY_OFF",
    "V7_FULL_RELAY_ON",
    "PROGRESSIVE_RELAY_OFF",
    "PROGRESSIVE_RELAY_ON",
    "SUPPORTED_TPD_NER_V7_DCH_VARIANTS",
    "build_tpd_ner_v7_dch_model",
    "load_verified_v7_dch_full_parent",
    "main",
]


if __name__ == "__main__":
    main()
