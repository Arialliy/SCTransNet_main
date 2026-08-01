#!/usr/bin/env python3
"""Paired scratch model construction for the four-dataset paper protocol.

This module is intentionally independent from the historical formal trainers.
It constructs the frozen Original and Final graphs without reading a
checkpoint, pairs every compatible state tensor, and initializes every random
Final-only subsystem from a stable SHA-256-derived substream of the sole
training seed (42).

The Final training graph is the frozen
SCTransNet + TPD8-MPRS-DCH + five-node NER4 Tail-Aware + QFG2-CROA graph with
training-only TSS heads.  Its deployment graph removes exactly the TSS state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from io import StringIO
from typing import Any

import torch
import torch.nn as nn
from torch.nn import init

from model.Config import get_SCTrans_config
from model.SCTransNet import SCTransNet
from model.tpd_clean_v8_mprs_dch import (
    PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
    replace_shallow_embeddings_clean_v8_mprs_dch,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import (
    DEFAULT_DC_SUPPORT_MODE,
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
    DEFAULT_TAIL_Z_THRESHOLDS,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
    PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS,
    PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS,
    QFG_STATE_KEYS,
    QFG_STATE_PREFIX,
    QFG_TERMINAL_STATE_KEYS,
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
    TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet,
    validate_formal_qfg_v2_croa_inference_model,
    validate_formal_qfg_v2_croa_survival_model,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)


BUILDER_SCHEMA = "sctransnet_four_dataset_paired_scratch_seed42_v1"
INFERENCE_EXPORT_SCHEMA = (
    "sctransnet_four_dataset_final_inference_state_seed42_v1"
)
TRAINING_SEED = 42
TSS_TRAINING_WEIGHT = 0.005
SUPPORTED_DATASETS = (
    "SIRST3",
    "NUAA-SIRST",
    "NUDT-SIRST",
    "IRSTD-1K",
)
SUPPORTED_METHODS = ("original_scratch", "final_scratch")
ORIGINAL_PARAMETER_COUNT = 11_325_939
ORIGINAL_STATE_KEY_COUNT = 510
FINAL_TRAINING_PARAMETER_COUNT = (
    PRODUCTION_V4_QFG_V2_CROA_SURVIVAL_PARAMETERS
)
FINAL_TRAINING_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT
)
FINAL_INFERENCE_PARAMETER_COUNT = (
    PRODUCTION_V4_QFG_V2_CROA_INFERENCE_PARAMETERS
)
FINAL_INFERENCE_STATE_KEY_COUNT = (
    FORMAL_V4_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT
)

_FINAL_RANDOM_NAMESPACES = ("tpd", "ner", "qfg")
_METHOD_ALIASES = {
    "original": "original_scratch",
    "original_scratch": "original_scratch",
    "final": "final_scratch",
    "final_scratch": "final_scratch",
}


def _require_seed(seed: int) -> int:
    if type(seed) is not int or seed != TRAINING_SEED:
        raise ValueError(
            "the four-dataset paper protocol has exactly one training "
            "seed: 42"
        )
    return seed


def _require_dataset_name(dataset_name: str | None) -> str | None:
    if dataset_name is None:
        return None
    if type(dataset_name) is not str or dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset_name must be one of {SUPPORTED_DATASETS}, "
            f"got {dataset_name!r}"
        )
    return dataset_name


def _normalize_method(method: str) -> str:
    if type(method) is not str or method not in _METHOD_ALIASES:
        raise ValueError(
            "method must be original/final or "
            "original_scratch/final_scratch"
        )
    return _METHOD_ALIASES[method]


def stable_sha256_uint64(seed: int, *namespace: str) -> int:
    """Derive one persistent uint64 without Python's process-random hash()."""

    _require_seed(seed)
    if not namespace or any(
        type(component) is not str or not component
        for component in namespace
    ):
        raise ValueError("stable seed namespaces must be non-empty strings")
    payload = json.dumps(
        [seed, *namespace],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _derived_initialization_seeds(seed: int) -> dict[str, int]:
    return {
        namespace: stable_sha256_uint64(seed, namespace)
        for namespace in _FINAL_RANDOM_NAMESPACES
    }


def _weights_init_kaiming(module: nn.Module) -> None:
    """Match the Original SCTransNet scratch initialization policy."""

    classname = module.__class__.__name__
    weight = getattr(module, "weight", None)
    if "Conv" in classname and weight is not None:
        init.kaiming_normal_(weight.data, a=0, mode="fan_in")
    elif "Linear" in classname and weight is not None:
        init.kaiming_normal_(weight.data, a=0, mode="fan_in")
    elif "BatchNorm" in classname and weight is not None:
        init.normal_(weight.data, 1.0, 0.02)
        bias = getattr(module, "bias", None)
        if bias is not None:
            init.constant_(bias.data, 0.0)


def _construct_original(seed: int) -> SCTransNet:
    _require_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        # The upstream class prints a construction banner.  The paper builder
        # is a library boundary, so keep model creation free of log side effects.
        with redirect_stdout(StringIO()):
            model = SCTransNet(
                get_SCTrans_config(),
                mode="train",
                deepsuper=True,
            )
        model.apply(_weights_init_kaiming)
    model.train()
    return model


def _construct_raw_final(
    seed: int,
    *,
    with_tss: bool,
) -> nn.Module:
    """Construct the frozen graph from modules only; never load a checkpoint."""

    _require_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        with redirect_stdout(StringIO()):
            parent = SCTransNet(
                get_SCTrans_config(),
                mode="train",
                deepsuper=True,
            )
        parent.apply(_weights_init_kaiming)
        replacements = replace_shallow_embeddings_clean_v8_mprs_dch(
            parent,
            PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        )
        if set(replacements) != {"embeddings_1", "embeddings_2"}:
            raise RuntimeError("Final construction replaced unexpected modules")
        for replacement in replacements.values():
            replacement.apply(_weights_init_kaiming)

        final_type = (
            TPDNERV8MPRSDCHV4QFGV2CROASurvivalSCTransNet
            if with_tss
            else TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet
        )
        model = final_type(
            parent,
            variant=PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            relay_width=DEFAULT_RELAY_WIDTH,
            relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
            dc_support_mode=DEFAULT_DC_SUPPORT_MODE,
            tail_z_thresholds=DEFAULT_TAIL_Z_THRESHOLDS,
        )
    model.train()
    return model


def _initialize_tpd_substream(model: nn.Module, seed: int) -> None:
    """Initialize only the seven TPD8 blocks from the derived TPD stream."""

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        for embedding_name in ("embeddings_1", "embeddings_2"):
            embedding = getattr(model.mtc, embedding_name)
            for block in embedding.blocks:
                # reset_parameters makes Conv biases part of this substream;
                # the project-wide Kaiming policy then replaces the weights.
                block.phase_compress.reset_parameters()
            embedding.apply(_weights_init_kaiming)
    with torch.no_grad():
        for embedding_name in ("embeddings_1", "embeddings_2"):
            embedding = getattr(model.mtc, embedding_name)
            for block in embedding.blocks:
                block.saliency_scale.zero_()


def _initialize_ner_substream(model: nn.Module, seed: int) -> None:
    """Initialize NER projections, then restore its exact identity terminals."""

    relay = getattr(model, "tpd_ner", None)
    if relay is None:
        raise RuntimeError("Final graph does not register tpd_ner")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        relay.apply(_weights_init_kaiming)
    relay.zero_init_gates()


def _initialize_qfg_substream(model: nn.Module, seed: int) -> None:
    """Initialize QFG hidden projections and preserve its exact-zero terminal."""

    qfg = getattr(model, "tpd_qfg", None)
    if qfg is None:
        raise RuntimeError("Final graph does not register tpd_qfg")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        qfg.apply(_weights_init_kaiming)
    for level in qfg.levels:
        level.reset_identity()


def _zero_initialize_tss(model: nn.Module) -> None:
    heads = getattr(model, "target_survival", None)
    if heads is None:
        return
    with torch.no_grad():
        for parameter in heads.parameters():
            parameter.zero_()


def _initialize_final_only_subsystems(
    model: nn.Module,
    seed: int,
) -> dict[str, int]:
    derived = _derived_initialization_seeds(seed)
    _initialize_tpd_substream(model, derived["tpd"])
    _initialize_ner_substream(model, derived["ner"])
    _initialize_qfg_substream(model, derived["qfg"])
    _zero_initialize_tss(model)
    return derived


def _state_partition(
    original_state: Mapping[str, torch.Tensor],
    final_state: Mapping[str, torch.Tensor],
) -> tuple[list[str], list[str], list[str]]:
    shared: list[str] = []
    original_only: list[str] = []
    for key, original_value in original_state.items():
        final_value = final_state.get(key)
        if (
            final_value is not None
            and tuple(final_value.shape) == tuple(original_value.shape)
        ):
            if final_value.dtype != original_value.dtype:
                raise TypeError(
                    f"same-name/same-shape state {key!r} has different dtype"
                )
            shared.append(key)
        else:
            original_only.append(key)
    shared_set = set(shared)
    final_only = [key for key in final_state if key not in shared_set]
    return sorted(shared), sorted(original_only), sorted(final_only)


def _copy_shared_state(
    original: nn.Module,
    final: nn.Module,
) -> tuple[list[str], list[str], list[str]]:
    original_state = original.state_dict()
    final_state = final.state_dict()
    shared, original_only, final_only = _state_partition(
        original_state,
        final_state,
    )
    with torch.no_grad():
        for key in shared:
            final_state[key].copy_(original_state[key])
    for key in shared:
        if not torch.equal(original_state[key], final_state[key]):
            raise RuntimeError(f"paired state copy failed for {key!r}")
    return shared, original_only, final_only


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    if tensor.numel() == 0:
        return b""
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def state_dict_sha256(
    state_dict: Mapping[str, torch.Tensor],
    keys: Sequence[str] | None = None,
) -> str:
    """Hash named tensor values with dtype and shape in stable key order."""

    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping")
    selected = sorted(state_dict) if keys is None else sorted(keys)
    if len(selected) != len(set(selected)):
        raise ValueError("state hash keys must be unique")
    digest = hashlib.sha256()
    for key in selected:
        if key not in state_dict:
            raise KeyError(f"state hash key is absent: {key!r}")
        value = state_dict[key]
        if type(key) is not str or not isinstance(value, torch.Tensor):
            raise TypeError("state_dict must map string keys to Tensors")
        descriptor = json.dumps(
            [key, str(value.dtype), list(value.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        raw = _tensor_bytes(value)
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _all_state_zero(
    state: Mapping[str, torch.Tensor],
    keys: Sequence[str],
) -> bool:
    return all(
        torch.count_nonzero(state[key]).item() == 0
        for key in keys
    )


def _validate_built_pair(
    original: SCTransNet,
    final: nn.Module,
    *,
    with_tss: bool,
    shared: Sequence[str],
    final_only: Sequence[str],
) -> None:
    if type(original) is not SCTransNet:
        raise TypeError("Original must use the exact SCTransNet class")
    if len(original.state_dict()) != ORIGINAL_STATE_KEY_COUNT:
        raise RuntimeError("Original state-key count differs")
    if _parameter_count(original) != ORIGINAL_PARAMETER_COUNT:
        raise RuntimeError("Original parameter count differs")

    final_state = final.state_dict()
    if with_tss:
        validate_formal_qfg_v2_croa_survival_model(
            final,
            require_zero_initialized_heads=True,
            require_identity_initialized_qfg=True,
        )
        if len(final_state) != FINAL_TRAINING_STATE_KEY_COUNT:
            raise RuntimeError("Final training state-key count differs")
        if _parameter_count(final) != FINAL_TRAINING_PARAMETER_COUNT:
            raise RuntimeError("Final training parameter count differs")
    else:
        validate_formal_qfg_v2_croa_inference_model(
            final,
            require_identity_initialized_qfg=True,
        )
        if len(final_state) != FINAL_INFERENCE_STATE_KEY_COUNT:
            raise RuntimeError("Final inference state-key count differs")
        if _parameter_count(final) != FINAL_INFERENCE_PARAMETER_COUNT:
            raise RuntimeError("Final inference parameter count differs")
    if set(QFG_STATE_KEYS) - set(final_state):
        raise RuntimeError("Final graph is missing QFG state")
    if not _all_state_zero(final_state, QFG_TERMINAL_STATE_KEYS):
        raise RuntimeError("QFG terminal projections are not exactly zero")
    if with_tss and not _all_state_zero(final_state, SURVIVAL_STATE_KEYS):
        raise RuntimeError("TSS classifiers are not exactly zero")
    if not with_tss and any(
        key.startswith(SURVIVAL_STATE_PREFIX) for key in final_state
    ):
        raise RuntimeError("Final inference graph retains TSS state")
    if not set(shared).isdisjoint(final_only):
        raise RuntimeError("shared and Final-only state partitions overlap")


def _pair_metadata(
    original: SCTransNet,
    final: nn.Module,
    *,
    dataset_name: str | None,
    with_tss: bool,
    shared: Sequence[str],
    original_only: Sequence[str],
    final_only: Sequence[str],
    derived: Mapping[str, int],
) -> dict[str, Any]:
    original_state = original.state_dict()
    final_state = final.state_dict()
    original_shared_hash = state_dict_sha256(original_state, shared)
    final_shared_hash = state_dict_sha256(final_state, shared)
    if original_shared_hash != final_shared_hash:
        raise RuntimeError("Original/Final shared-state hashes differ")
    extension_hash = state_dict_sha256(final_state, final_only)
    tss_zero = (
        _all_state_zero(final_state, SURVIVAL_STATE_KEYS)
        if with_tss
        else True
    )
    qfg_terminal_zero = _all_state_zero(
        final_state,
        QFG_TERMINAL_STATE_KEYS,
    )
    return {
        "schema": BUILDER_SCHEMA,
        "dataset_name": dataset_name,
        "supported_datasets": list(SUPPORTED_DATASETS),
        "training_seed": TRAINING_SEED,
        "allowed_training_seeds": [TRAINING_SEED],
        "initialization_mode": "true_scratch",
        "parent_checkpoint": None,
        "parent_checkpoint_load_count": 0,
        "warm_start_used": False,
        "optimizer_state_inherited": False,
        "scheduler_state_inherited": False,
        "paired_initialization": True,
        "model_construction_preserves_caller_rng_stream": True,
        "shared_state_match_rule": "same_name_same_shape_same_dtype",
        "shared_state_key_count": len(shared),
        "shared_state_keys": list(shared),
        "shared_state_sha256": original_shared_hash,
        "shared_state_tensor_sha256": original_shared_hash,
        "original_shared_state_sha256": original_shared_hash,
        "final_shared_state_sha256": final_shared_hash,
        "shared_state_bitwise_equal": True,
        "original_only_state_key_count": len(original_only),
        "original_only_state_keys": list(original_only),
        "final_only_state_key_count": len(final_only),
        "final_only_state_keys": list(final_only),
        "final_only_state_sha256": extension_hash,
        "extension_state_key_count": len(final_only),
        "extension_state_keys": list(final_only),
        "extension_state_sha256": extension_hash,
        "derived_initialization_seed_algorithm": (
            "sha256(canonical_json([42,namespace]))[:8]_uint64_be"
        ),
        "derived_initialization_seeds": dict(derived),
        "derived_seeds_are_additional_training_seeds": False,
        "tss_zero_initialized": tss_zero,
        "qfg_terminal_zero_initialized": qfg_terminal_zero,
        "original_parameter_count": _parameter_count(original),
        "final_parameter_count": _parameter_count(final),
        "original": {
            "method": "original_scratch",
            "class": (
                f"{type(original).__module__}.{type(original).__qualname__}"
            ),
            "parameter_count": _parameter_count(original),
            "state_key_count": len(original_state),
            "state_sha256": state_dict_sha256(original_state),
            "training_graph": "original_sctransnet",
            "inference_graph": "original_sctransnet",
        },
        "final": {
            "method": "final_scratch",
            "class": f"{type(final).__module__}.{type(final).__qualname__}",
            "parameter_count": _parameter_count(final),
            "state_key_count": len(final_state),
            "state_sha256": state_dict_sha256(final_state),
            "training_graph": (
                "sctransnet_tpd8_mprs_dch_ner4_tail_aware_"
                "qfg2_croa_tss"
                if with_tss
                else None
            ),
            "inference_graph": (
                "sctransnet_tpd8_mprs_dch_ner4_tail_aware_qfg2_croa"
            ),
            "tss_registered": with_tss,
            "tss_training_only": True,
            "tss_training_weight": TSS_TRAINING_WEIGHT,
            "tss_state_keys": (
                list(SURVIVAL_STATE_KEYS) if with_tss else []
            ),
            "tss_zero_initialized": tss_zero,
            "qfg_terminal_state_keys": list(QFG_TERMINAL_STATE_KEYS),
            "qfg_terminal_zero_initialized": qfg_terminal_zero,
        },
    }


def build_paired_models(
    seed: int = TRAINING_SEED,
    *,
    dataset_name: str | None = None,
    final_with_tss: bool = True,
) -> tuple[SCTransNet, nn.Module, dict[str, Any]]:
    """Build one strict Original/Final scratch pair and its audit metadata."""

    seed = _require_seed(seed)
    dataset_name = _require_dataset_name(dataset_name)
    if type(final_with_tss) is not bool:
        raise TypeError("final_with_tss must be bool")

    original = _construct_original(seed)
    final = _construct_raw_final(seed, with_tss=final_with_tss)
    derived = _initialize_final_only_subsystems(final, seed)
    shared, original_only, final_only = _copy_shared_state(original, final)
    _validate_built_pair(
        original,
        final,
        with_tss=final_with_tss,
        shared=shared,
        final_only=final_only,
    )
    metadata = _pair_metadata(
        original,
        final,
        dataset_name=dataset_name,
        with_tss=final_with_tss,
        shared=shared,
        original_only=original_only,
        final_only=final_only,
        derived=derived,
    )
    return original, final, metadata


def build_paired_paper_models(
    dataset_name: str,
    seed: int = TRAINING_SEED,
    final_with_tss: bool = True,
) -> tuple[SCTransNet, nn.Module, dict[str, Any]]:
    """Dataset-explicit spelling of :func:`build_paired_models`."""

    return build_paired_models(
        seed,
        dataset_name=dataset_name,
        final_with_tss=final_with_tss,
    )


def _state_from_model_or_mapping(
    source: nn.Module | Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    if isinstance(source, nn.Module):
        return source.state_dict()
    if not isinstance(source, Mapping):
        raise TypeError("source must be a Final model or state mapping")
    nested = source.get("state_dict")
    candidate = nested if isinstance(nested, Mapping) else source
    if not all(
        type(key) is str and isinstance(value, torch.Tensor)
        for key, value in candidate.items()
    ):
        raise TypeError("Final state must map string keys to Tensors")
    return candidate


def strip_tss_for_inference_state_dict(
    training_state_dict: Mapping[str, torch.Tensor],
    *,
    to_cpu: bool = False,
) -> dict[str, torch.Tensor]:
    """Clone Final state while removing exactly the four training-only TSS keys."""

    if not isinstance(training_state_dict, Mapping):
        raise TypeError("training_state_dict must be a mapping")
    if len(training_state_dict) != FINAL_TRAINING_STATE_KEY_COUNT:
        raise ValueError(
            "Final training state must contain exactly "
            f"{FINAL_TRAINING_STATE_KEY_COUNT} keys"
        )
    survival_keys = {
        key
        for key in training_state_dict
        if key.startswith(SURVIVAL_STATE_PREFIX)
    }
    if survival_keys != set(SURVIVAL_STATE_KEYS):
        raise ValueError("Final training state does not contain exact TSS keys")
    qfg_keys = {
        key for key in training_state_dict if key.startswith(QFG_STATE_PREFIX)
    }
    if qfg_keys != set(QFG_STATE_KEYS):
        raise ValueError("Final training state does not contain exact QFG keys")

    inference_state: dict[str, torch.Tensor] = {}
    for key, value in training_state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Final state {key!r} is not a Tensor")
        if key in survival_keys:
            continue
        clone = value.detach().clone()
        inference_state[key] = clone.cpu() if to_cpu else clone
    if len(inference_state) != FINAL_INFERENCE_STATE_KEY_COUNT:
        raise RuntimeError("stripped Final inference state-key count differs")
    if any(
        key.startswith(SURVIVAL_STATE_PREFIX) for key in inference_state
    ):
        raise RuntimeError("stripped Final state retains TSS")
    return inference_state


def export_final_inference_state(
    source: nn.Module | Mapping[str, Any],
    *,
    to_cpu: bool = True,
) -> dict[str, torch.Tensor]:
    """Return a detached, checkpoint-ready Final inference state mapping."""

    state = _state_from_model_or_mapping(source)
    return strip_tss_for_inference_state_dict(state, to_cpu=to_cpu)


def build_final_inference_model_from_training_state_dict(
    training_state_dict: Mapping[str, torch.Tensor],
    dataset_name: str | None = None,
    seed: int = TRAINING_SEED,
) -> tuple[
    TPDNERV8MPRSDCHV4QFGV2CROAInferenceSCTransNet,
    dict[str, Any],
]:
    """Strict-load a TSS-free deployment graph from one Final training state."""

    seed = _require_seed(seed)
    dataset_name = _require_dataset_name(dataset_name)
    inference_state = strip_tss_for_inference_state_dict(training_state_dict)
    model = _construct_raw_final(seed, with_tss=False)
    if set(inference_state) != set(model.state_dict()):
        raise ValueError(
            "stripped state does not exactly match the frozen inference graph"
        )
    incompatible = model.load_state_dict(inference_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict inference load returned incompatible keys")
    validation = validate_formal_qfg_v2_croa_inference_model(model)
    model.eval()
    model.mode = "test"
    metadata = {
        "schema": INFERENCE_EXPORT_SCHEMA,
        "dataset_name": dataset_name,
        "training_seed": seed,
        "source_training_state_sha256": state_dict_sha256(
            training_state_dict
        ),
        "inference_state_sha256": state_dict_sha256(model.state_dict()),
        "removed_tss_state_keys": list(SURVIVAL_STATE_KEYS),
        "qfg_state_preserved": list(QFG_STATE_KEYS),
        "state_key_count": len(model.state_dict()),
        "parameter_count": _parameter_count(model),
        "strict_load": True,
        "target_survival_registered": False,
        "warm_start_used": False,
        "validation": validation,
    }
    return model, metadata


def build_method_model(
    method: str,
    seed: int = TRAINING_SEED,
    *,
    dataset_name: str | None = None,
    training: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the complete pair first, then select one requested method."""

    canonical_method = _normalize_method(method)
    if type(training) is not bool:
        raise TypeError("training must be bool")
    original, final, pair_metadata = build_paired_models(
        seed,
        dataset_name=dataset_name,
        final_with_tss=True,
    )
    if canonical_method == "original_scratch":
        model: nn.Module = original
        if not training:
            model.eval()
            model.mode = "test"
    elif training:
        model = final
    else:
        model, inference_metadata = (
            build_final_inference_model_from_training_state_dict(
                final.state_dict(),
                dataset_name=dataset_name,
                seed=seed,
            )
        )
        pair_metadata = dict(pair_metadata)
        pair_metadata["inference_export"] = inference_metadata

    metadata = {
        "schema": BUILDER_SCHEMA,
        "method": canonical_method,
        "training_graph_requested": training,
        "dataset_name": dataset_name,
        "training_seed": seed,
        "pair": pair_metadata,
        "selected_model_state_sha256": state_dict_sha256(
            model.state_dict()
        ),
        "selected_model_parameter_count": _parameter_count(model),
        "selected_model_state_key_count": len(model.state_dict()),
        "warm_start_used": False,
        "parent_checkpoint": None,
    }
    return model, metadata


def build_paper_model(
    method: str,
    dataset_name: str,
    seed: int = TRAINING_SEED,
    *,
    training: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    """Dataset-explicit spelling used by trainers and evaluators."""

    return build_method_model(
        method,
        seed,
        dataset_name=dataset_name,
        training=training,
    )


__all__ = [
    "BUILDER_SCHEMA",
    "FINAL_INFERENCE_PARAMETER_COUNT",
    "FINAL_INFERENCE_STATE_KEY_COUNT",
    "FINAL_TRAINING_PARAMETER_COUNT",
    "FINAL_TRAINING_STATE_KEY_COUNT",
    "INFERENCE_EXPORT_SCHEMA",
    "ORIGINAL_PARAMETER_COUNT",
    "ORIGINAL_STATE_KEY_COUNT",
    "SUPPORTED_DATASETS",
    "SUPPORTED_METHODS",
    "TRAINING_SEED",
    "TSS_TRAINING_WEIGHT",
    "build_final_inference_model_from_training_state_dict",
    "build_method_model",
    "build_paired_models",
    "build_paired_paper_models",
    "build_paper_model",
    "export_final_inference_state",
    "stable_sha256_uint64",
    "state_dict_sha256",
    "strip_tss_for_inference_state_dict",
]
