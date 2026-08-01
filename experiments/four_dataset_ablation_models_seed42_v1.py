#!/usr/bin/env python3
"""Cumulative A0--A4 builders for the frozen SIRST3 seed-42 ablation.

The ablation changes exactly one cumulative factor at each transition:

* A0: Original SCTransNet;
* A1: A0 tokenizer replaced by TPD8-MPRS-DCH;
* A2: A1 plus the five-node NER4 Tail-Aware relay;
* A3: A2 plus QFG2-CROA;
* A4: A3 plus the training-only TSS objective heads.

All five graphs are true-scratch constructions.  No checkpoint is accepted or
read.  To keep initialization paired, every compatible state tensor is copied
from the immediately preceding graph after deterministic module-local
initialization.  Consequently A1--A4 share the same initialized TPD state,
A2--A4 share NER state, and A3--A4 share QFG state.

This is a cumulative architectural ablation.  TPD8 itself changes the
tokenizer parameterization and parameter count relative to Original; that
capacity difference is deliberately locked as part of the named TPD8 module,
not silently described as a capacity-matched causal intervention.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from experiments.four_dataset_models_seed42_v1 import (
    ORIGINAL_PARAMETER_COUNT,
    PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
    PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS,
    PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS,
    QFG_STATE_PREFIX,
    QFG_TERMINAL_STATE_KEYS,
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
    TRAINING_SEED,
    _construct_original,
    _initialize_ner_substream,
    _initialize_qfg_substream,
    _initialize_tpd_substream,
    _weights_init_kaiming,
    stable_sha256_uint64,
    state_dict_sha256,
)
from model.SCTransNet import SCTransNet
from model.tpd_clean_v8_mprs_dch import (
    TPDCleanV8MPRSDCHPatchEmbedding,
    replace_shallow_embeddings_clean_v8_mprs_dch,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    DEFAULT_DC_SUPPORT_MODE,
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
    DEFAULT_TAIL_Z_THRESHOLDS,
    PRODUCTION_V4_RELAY_ON_PARAMETERS,
    TPDNERV8MPRSDCHV4SCTransNet,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
)


SCHEMA = "sctransnet_sirst3_cumulative_ablation_seed42_v1"
DATASET = "SIRST3"
ABLATION_IDS = ("A0", "A1", "A2", "A3", "A4")
TSS_WEIGHT_BY_ABLATION = {
    "A0": 0.0,
    "A1": 0.0,
    "A2": 0.0,
    "A3": 0.0,
    "A4": 0.005,
}
GRAPH_BY_ABLATION = {
    "A0": "SCTransNet",
    "A1": "SCTransNet+TPD8-MPRS-DCH",
    "A2": "SCTransNet+TPD8-MPRS-DCH+five-node NER4 Tail-Aware",
    "A3": (
        "SCTransNet+TPD8-MPRS-DCH+five-node NER4 Tail-Aware+QFG2-CROA"
    ),
    "A4": (
        "SCTransNet+TPD8-MPRS-DCH+five-node NER4 Tail-Aware+QFG2-CROA"
        "+TSS(training-only)"
    ),
}
MODULE_FLAGS = {
    "A0": {"tpd": False, "ner": False, "qfg": False, "tss": False},
    "A1": {"tpd": True, "ner": False, "qfg": False, "tss": False},
    "A2": {"tpd": True, "ner": True, "qfg": False, "tss": False},
    "A3": {"tpd": True, "ner": True, "qfg": True, "tss": False},
    "A4": {"tpd": True, "ner": True, "qfg": True, "tss": True},
}
EXPECTED_PARAMETER_COUNTS = {
    "A0": ORIGINAL_PARAMETER_COUNT,
    "A1": 10_843_155,
    "A2": PRODUCTION_V4_RELAY_ON_PARAMETERS,
    "A3": PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS,
    "A4": PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS,
}


def normalize_ablation_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("ablation_id must be a string")
    normalized = value.strip().upper()
    if normalized not in ABLATION_IDS:
        raise ValueError(
            f"ablation_id must be one of {ABLATION_IDS}, got {value!r}"
        )
    return normalized


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _derived_seed(namespace: str) -> int:
    return stable_sha256_uint64(TRAINING_SEED, namespace)


def _construct_tpd_parent() -> SCTransNet:
    """Build raw A1 without consuming the caller's RNG stream."""

    with torch.random.fork_rng(devices=[]):
        parent = _construct_original(TRAINING_SEED)
        replacements = replace_shallow_embeddings_clean_v8_mprs_dch(
            parent,
            PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        )
        if set(replacements) != {"embeddings_1", "embeddings_2"}:
            raise RuntimeError("A1 replaced an unexpected tokenizer module")
        for replacement in replacements.values():
            replacement.apply(_weights_init_kaiming)
        _initialize_tpd_substream(parent, _derived_seed("tpd"))
    parent.train()
    return parent


def _construct_a2() -> TPDNERV8MPRSDCHV4SCTransNet:
    parent = _construct_tpd_parent()
    model = TPDNERV8MPRSDCHV4SCTransNet(
        parent,
        variant=PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    _initialize_ner_substream(model, _derived_seed("ner"))
    model.train()
    return model


def _construct_a3() -> TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet:
    parent = _construct_tpd_parent()
    model = TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet(
        parent,
        variant=PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    _initialize_ner_substream(model, _derived_seed("ner"))
    _initialize_qfg_substream(model, _derived_seed("qfg"))
    model.train()
    return model


def _construct_a4() -> TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet:
    parent = _construct_tpd_parent()
    model = TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet(
        parent,
        variant=PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
        dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
        tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
    )
    _initialize_ner_substream(model, _derived_seed("ner"))
    _initialize_qfg_substream(model, _derived_seed("qfg"))
    with torch.no_grad():
        for parameter in model.target_survival.parameters():
            parameter.zero_()
    model.train()
    return model


def _compatible_keys(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> list[str]:
    keys: list[str] = []
    for key, source_value in source.items():
        target_value = target.get(key)
        if (
            target_value is not None
            and tuple(source_value.shape) == tuple(target_value.shape)
            and source_value.dtype == target_value.dtype
        ):
            keys.append(key)
    return sorted(keys)


def _copy_compatible_state(
    source_model: nn.Module,
    target_model: nn.Module,
) -> dict[str, Any]:
    source = source_model.state_dict()
    target = target_model.state_dict()
    shared = _compatible_keys(source, target)
    with torch.no_grad():
        for key in shared:
            target[key].copy_(source[key])
    if any(not torch.equal(source[key], target[key]) for key in shared):
        raise RuntimeError("cumulative paired-state copy failed")
    source_only = sorted(set(source) - set(shared))
    target_only = sorted(set(target) - set(shared))
    source_hash = state_dict_sha256(source, shared)
    target_hash = state_dict_sha256(target, shared)
    if source_hash != target_hash:
        raise RuntimeError("cumulative shared-state hashes differ after copy")
    return {
        "shared_key_count": len(shared),
        "shared_keys": shared,
        "source_only_key_count": len(source_only),
        "source_only_keys": source_only,
        "target_only_key_count": len(target_only),
        "target_only_keys": target_only,
        "source_shared_state_sha256": source_hash,
        "target_shared_state_sha256": target_hash,
        "shared_state_bitwise_equal": True,
    }


def _all_zero(
    state: Mapping[str, torch.Tensor],
    keys: Sequence[str],
) -> bool:
    return all(int(torch.count_nonzero(state[key]).item()) == 0 for key in keys)


def _validate_graph(ablation_id: str, model: nn.Module) -> None:
    flags = MODULE_FLAGS[ablation_id]
    has_tpd = all(
        isinstance(
            getattr(model.mtc, name),
            TPDCleanV8MPRSDCHPatchEmbedding,
        )
        for name in ("embeddings_1", "embeddings_2")
    )
    has_ner = hasattr(model, "tpd_ner")
    has_qfg = hasattr(model, "tpd_qfg")
    has_tss = hasattr(model, "target_survival")
    actual = {
        "tpd": has_tpd,
        "ner": has_ner,
        "qfg": has_qfg,
        "tss": has_tss,
    }
    if actual != flags:
        raise RuntimeError(
            f"{ablation_id} graph flags differ: {actual!r} != {flags!r}"
        )
    count = _parameter_count(model)
    if count != EXPECTED_PARAMETER_COUNTS[ablation_id]:
        raise RuntimeError(
            f"{ablation_id} parameter count differs: {count} != "
            f"{EXPECTED_PARAMETER_COUNTS[ablation_id]}"
        )
    state = model.state_dict()
    if flags["qfg"] and not _all_zero(state, QFG_TERMINAL_STATE_KEYS):
        raise RuntimeError(f"{ablation_id} QFG terminal is not identity-zero")
    if flags["tss"] and not _all_zero(state, SURVIVAL_STATE_KEYS):
        raise RuntimeError(f"{ablation_id} TSS heads are not exactly zero")
    if not flags["qfg"] and any(
        key.startswith(QFG_STATE_PREFIX) for key in state
    ):
        raise RuntimeError(f"{ablation_id} unexpectedly contains QFG state")
    if not flags["tss"] and any(
        key.startswith(SURVIVAL_STATE_PREFIX) for key in state
    ):
        raise RuntimeError(f"{ablation_id} unexpectedly contains TSS state")


def build_all_ablation_models(
    seed: int = TRAINING_SEED,
) -> tuple[dict[str, nn.Module], dict[str, Any]]:
    """Construct and pair all five true-scratch cumulative graphs."""

    if type(seed) is not int or seed != TRAINING_SEED:
        raise ValueError("the ablation protocol has the sole seed 42")
    with torch.random.fork_rng(devices=[]):
        models: dict[str, nn.Module] = {
            "A0": _construct_original(seed),
            "A1": _construct_tpd_parent(),
            "A2": _construct_a2(),
            "A3": _construct_a3(),
            "A4": _construct_a4(),
        }

    transitions: dict[str, dict[str, Any]] = {}
    for source_id, target_id in zip(ABLATION_IDS[:-1], ABLATION_IDS[1:]):
        transition = _copy_compatible_state(
            models[source_id],
            models[target_id],
        )
        transition["source"] = source_id
        transition["target"] = target_id
        transitions[f"{source_id}_to_{target_id}"] = transition

    records: dict[str, Any] = {}
    for ablation_id in ABLATION_IDS:
        model = models[ablation_id]
        _validate_graph(ablation_id, model)
        state = model.state_dict()
        records[ablation_id] = {
            "ablation_id": ablation_id,
            "graph": GRAPH_BY_ABLATION[ablation_id],
            "module_flags": copy.deepcopy(MODULE_FLAGS[ablation_id]),
            "class": f"{type(model).__module__}.{type(model).__qualname__}",
            "parameter_count": _parameter_count(model),
            "state_key_count": len(state),
            "state_keys": sorted(state),
            "initial_state_sha256": state_dict_sha256(state),
            "tss_training_weight": TSS_WEIGHT_BY_ABLATION[ablation_id],
        }

    metadata = {
        "schema": SCHEMA,
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "allowed_training_seeds": [TRAINING_SEED],
        "scratch": True,
        "warm_start_used": False,
        "parent_checkpoint": None,
        "optimizer_state_inherited": False,
        "cumulative_order": list(ABLATION_IDS),
        "one_added_factor_per_transition": {
            "A0_to_A1": "TPD8-MPRS-DCH tokenizer",
            "A1_to_A2": "five-node NER4 Tail-Aware",
            "A2_to_A3": "QFG2-CROA",
            "A3_to_A4": "training-only TSS heads and loss(weight=0.005)",
        },
        "attribution_scope": (
            "cumulative named-module contribution under each module's frozen "
            "implementation and parameterization"
        ),
        "capacity_matching_claimed": False,
        "capacity_note": (
            "A0->A1 includes the frozen TPD8 tokenizer parameterization; "
            "this ablation does not isolate capacity from tokenizer design"
        ),
        "derived_initialization_seeds": {
            name: _derived_seed(name) for name in ("tpd", "ner", "qfg")
        },
        "paired_transitions": transitions,
        "models": records,
    }
    return models, metadata


def build_ablation_model(
    ablation_id: str,
    seed: int = TRAINING_SEED,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build all paired graphs, then return an independent selected graph."""

    selected_id = normalize_ablation_id(ablation_id)
    models, metadata = build_all_ablation_models(seed)
    selected = models[selected_id]
    selected_metadata = {
        "schema": SCHEMA,
        "dataset": DATASET,
        "ablation_id": selected_id,
        "training_seed": TRAINING_SEED,
        "scratch": True,
        "warm_start_used": False,
        "parent_checkpoint": None,
        "tss_training_weight": TSS_WEIGHT_BY_ABLATION[selected_id],
        "selected": copy.deepcopy(metadata["models"][selected_id]),
        "pairing": copy.deepcopy(metadata),
    }
    return selected, selected_metadata


__all__ = [
    "ABLATION_IDS",
    "DATASET",
    "EXPECTED_PARAMETER_COUNTS",
    "GRAPH_BY_ABLATION",
    "MODULE_FLAGS",
    "SCHEMA",
    "TRAINING_SEED",
    "TSS_WEIGHT_BY_ABLATION",
    "build_ablation_model",
    "build_all_ablation_models",
    "normalize_ablation_id",
]
