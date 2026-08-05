#!/usr/bin/env python3
"""Fresh-seed42 V5-PER training/inference registry for three datasets."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (  # noqa: E402
    SURVIVAL_STATE_KEYS,
    SURVIVAL_STATE_PREFIX,
)
from model.tpd_ner_v8_mprs_dch_v5_per import (  # noqa: E402
    V5_PER_FORMAL_DC_SUPPORT_MODE,
    V5_PER_RELAY_VERSION,
)
from model.tpd_ner_v8_mprs_dch_v5_per_qfg_v2_croa_survival import (  # noqa: E402
    FORMAL_V5_PER_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT,
    FORMAL_V5_PER_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT,
    V5_PER_QFG_V2_CROA_INTEGRATION_VERSION,
    build_formal_v5_per_qfg_v2_croa_inference_model,
    build_formal_v5_per_qfg_v2_croa_survival_model,
    validate_formal_v5_per_qfg_v2_croa_inference_model,
    validate_formal_v5_per_qfg_v2_croa_survival_model,
)
from experiments import four_dataset_models_seed42_v1 as v4_registry  # noqa: E402


SCHEMA = "sctransnet_three_dataset_ner_v5_per_models_seed42_v1"
TRAINING_SEED = 42
DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
METHOD = "final"
RECIPE_ID = "ner_v5_per_tss_off"
TRAINING_STATE_KEY_COUNT = FORMAL_V5_PER_QFG_V2_CROA_SURVIVAL_STATE_KEY_COUNT
INFERENCE_STATE_KEY_COUNT = FORMAL_V5_PER_QFG_V2_CROA_INFERENCE_STATE_KEY_COUNT

# This is an explicit transitive closure for the V5 runtime, not a glob over
# model/*.py.  Adding an unrelated model file must not invalidate V5 artifacts.
RUNTIME_DEPENDENCY_RELATIVE_PATHS = (
    "experiments/three_dataset_ner_v5_per_models_seed42_v1.py",
    "experiments/four_dataset_models_seed42_v1.py",
    "experiments/train_four_dataset_original_final_seed42_exact_v1.py",
    "experiments/train_three_dataset_seed42_global_tss_v2.py",
    "experiments/train_three_dataset_tss_off_seed42_v1.py",
    "experiments/train_three_dataset_ner_v5_per_tss_off_seed42.py",
    "experiments/three_dataset_v2_protocol.py",
    "experiments/paper_three_dataset_v2.py",
    "experiments/tpd_training_loss.py",
    "experiments/train_tpd_clean_v8_mprs_dch.py",
    "experiments/train_tpd_pilot.py",
    "model/Config.py",
    "model/SCTransNet.py",
    "model/tpd.py",
    "model/tpd_clean.py",
    "model/tpd_clean_v8_mprs_dch.py",
    "model/tpd_forward_contract.py",
    "model/tpd_frequency_gate.py",
    "model/tpd_frequency_gate_v2_croa.py",
    "model/tpd_ner_v8_mprs_dch.py",
    "model/tpd_ner_v8_mprs_dch_v2.py",
    "model/tpd_ner_v8_mprs_dch_v3.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py",
    "model/tpd_ner_v8_mprs_dch_v5_per.py",
    "model/tpd_ner_v8_mprs_dch_v5_per_qfg_v2_croa_survival.py",
    "model/tpd_query_frequency_bridge.py",
    "model/tpd_relay.py",
    "model/tpd_sctransnet.py",
    "model/tpd_survival.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise FileNotFoundError(candidate)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_source_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for relative in RUNTIME_DEPENDENCY_RELATIVE_PATHS:
        path = (REPO_ROOT / relative).resolve(strict=True)
        key = (
            f"architecture::{relative}"
            if relative.startswith("model/")
            else f"runtime::{relative}"
        )
        paths[key] = path
    return dict(sorted(paths.items()))


def runtime_source_records() -> dict[str, dict[str, str]]:
    return {
        key: {"path": str(path), "sha256": file_sha256(path)}
        for key, path in runtime_source_paths().items()
    }


def _require_dataset(dataset_name: str) -> str:
    _require(dataset_name in DATASETS, f"dataset must be one of {DATASETS}")
    return dataset_name


def _require_seed(seed: int) -> int:
    _require(type(seed) is int and seed == TRAINING_SEED, "V5 requires seed=42")
    return seed


def _all_zero(state: Mapping[str, torch.Tensor], keys: tuple[str, ...]) -> bool:
    return all(torch.count_nonzero(state[key]).item() == 0 for key in keys)


def _validate_manifest(manifest: Mapping[str, Any], *, training: bool) -> None:
    expected = {
        "relay_version": V5_PER_RELAY_VERSION,
        "ner_v5_per_dc_support_mode": V5_PER_FORMAL_DC_SUPPORT_MODE,
        "qfg_integration_version": V5_PER_QFG_V2_CROA_INTEGRATION_VERSION,
        "checkpoint_semantically_interchangeable_with_v4": False,
        "deployment_graph": "v5_per_qfg_v2_croa_no_tss",
    }
    for key, value in expected.items():
        _require(manifest.get(key) == value, f"V5 manifest {key} differs")
    if training:
        _require(
            manifest.get("survival_training_only") is True
            and manifest.get("survival_state_prefix") == SURVIVAL_STATE_PREFIX,
            "V5 training manifest lacks its training-only TSS identity",
        )
    else:
        _require(
            manifest.get("target_survival_registered") is False
            and manifest.get("target_survival_state_removed") is True,
            "V5 inference manifest retains TSS identity",
        )


def _metadata(
    model: nn.Module,
    raw: Mapping[str, Any],
    *,
    dataset_name: str,
    training: bool,
) -> dict[str, Any]:
    manifest = model.architecture_manifest()
    _validate_manifest(manifest, training=training)
    state = model.state_dict()
    expected_count = TRAINING_STATE_KEY_COUNT if training else INFERENCE_STATE_KEY_COUNT
    _require(len(state) == expected_count, "V5 state-key count differs")
    qfg = getattr(model, "tpd_qfg", None)
    _require(isinstance(qfg, nn.Module), "V5 graph lacks QFG2")
    _require(
        all(parameter.requires_grad for parameter in qfg.parameters()),
        "QFG2 parameters must remain jointly trainable",
    )
    return {
        "schema": SCHEMA,
        "dataset_name": dataset_name,
        "method": METHOD,
        "recipe_id": RECIPE_ID,
        "training_seed": TRAINING_SEED,
        "initialization_mode": "fresh_seed42_scratch",
        "parent_checkpoint": None,
        "warm_start_used": False,
        "resume_scope": "same_v5_run_only",
        "training_graph": training,
        "state_key_count": len(state),
        "target_survival_registered": training,
        "requested_tss_weight": 0.0,
        "tss_enabled": False,
        "tss_heads_registered": training,
        "tss_training_forward_computes_logits": training,
        "tss_loss_consumes_logits": False,
        "tss_survival_target_constructed": False,
        "qfg_architecture_frozen": True,
        "qfg_parameters_jointly_trainable": True,
        "relay_version": V5_PER_RELAY_VERSION,
        "dc_support_mode": V5_PER_FORMAL_DC_SUPPORT_MODE,
        "architecture_manifest": dict(manifest),
        "architecture_id": canonical_sha256(manifest),
        "graph_constructor_metadata_before_state_installation": dict(raw),
        "runtime_dependency_manifest": runtime_source_records(),
    }


@contextmanager
def _preserve_process_rng() -> Any:
    """Isolate the convenience V5 constructor before loading paired state."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def build_v5_training_model(
    dataset_name: str,
    seed: int = TRAINING_SEED,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build V5 from fresh seed42 scratch; never read a checkpoint."""

    dataset_name = _require_dataset(dataset_name)
    seed = _require_seed(seed)
    # The paper-development runner must use the same paired-scratch state as
    # the historical V4 registry.  The model-level formal builder is used only
    # to obtain the exact V5 Python graph; every one of its 568 tensors is then
    # replaced by the paired V4 seed42 tensor before training begins.
    _original, v4_model, paired = v4_registry.build_paired_models(
        seed,
        dataset_name=dataset_name,
        final_with_tss=True,
    )
    with _preserve_process_rng():
        model, raw = build_formal_v5_per_qfg_v2_croa_survival_model(seed)
    v4_state = v4_model.state_dict()
    incompatible = model.load_state_dict(v4_state, strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "paired V4-to-V5 strict initialization returned incompatible keys",
    )
    v5_state = model.state_dict()
    _require(tuple(v4_state) == tuple(v5_state), "V4/V5 initial state layouts differ")
    _require(
        all(torch.equal(v4_state[key], v5_state[key]) for key in v4_state),
        "V4/V5 seed42 initial states are not bitwise equal",
    )
    validate_formal_v5_per_qfg_v2_croa_survival_model(
        model,
        require_zero_initialized_heads=True,
        require_identity_initialized_qfg=True,
    )
    state = model.state_dict()
    _require(set(SURVIVAL_STATE_KEYS) <= set(state), "V5 training graph lacks TSS state")
    _require(_all_zero(state, SURVIVAL_STATE_KEYS), "TSS heads are not exact-zero")
    model.train()
    metadata = _metadata(model, raw, dataset_name=dataset_name, training=True)
    metadata.update(
        {
            "paired_v4_registry_schema": paired.get("schema"),
            "paired_initialization": True,
            "initial_state_bitwise_equal_to_v4": True,
            "initial_state_sha256": v4_registry.state_dict_sha256(v5_state),
            "shared_state_bitwise_equal_to_original": paired.get(
                "shared_state_bitwise_equal"
            ),
            "shared_state_sha256": paired.get("shared_state_sha256"),
        }
    )
    return model, metadata


def strip_tss_for_inference_state_dict(
    training_state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    _require(isinstance(training_state_dict, Mapping), "training state must be a mapping")
    _require(len(training_state_dict) == TRAINING_STATE_KEY_COUNT, "training state must have 568 keys")
    _require(set(SURVIVAL_STATE_KEYS) <= set(training_state_dict), "training state lacks TSS keys")
    _require(
        _all_zero(training_state_dict, SURVIVAL_STATE_KEYS),
        "TSS-off checkpoint TSS state is not exact-zero",
    )
    stripped = {
        key: value.detach().cpu().clone()
        for key, value in training_state_dict.items()
        if key not in set(SURVIVAL_STATE_KEYS)
    }
    _require(len(stripped) == INFERENCE_STATE_KEY_COUNT, "stripped state must have 564 keys")
    _require(
        not any(key.startswith(SURVIVAL_STATE_PREFIX) for key in stripped),
        "stripped state retains TSS",
    )
    return stripped


def build_v5_inference_model_from_training_state_dict(
    training_state_dict: Mapping[str, torch.Tensor],
    *,
    dataset_name: str,
    seed: int = TRAINING_SEED,
) -> tuple[nn.Module, dict[str, Any]]:
    dataset_name = _require_dataset(dataset_name)
    stripped = strip_tss_for_inference_state_dict(training_state_dict)
    model, raw = build_formal_v5_per_qfg_v2_croa_inference_model(_require_seed(seed))
    incompatible = model.load_state_dict(stripped, strict=True)
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "V5 inference strict load returned incompatible keys",
    )
    validated = validate_formal_v5_per_qfg_v2_croa_inference_model(model)
    model.eval()
    model.mode = "test"
    metadata = _metadata(model, raw, dataset_name=dataset_name, training=False)
    metadata.update(
        {
            "strict_load": True,
            "training_state_key_count": TRAINING_STATE_KEY_COUNT,
            "inference_state_key_count": INFERENCE_STATE_KEY_COUNT,
            "stripped_state_key_count": len(SURVIVAL_STATE_KEYS),
            "stripped_state_keys": list(SURVIVAL_STATE_KEYS),
            "target_survival_registered": False,
            "validator": validated,
        }
    )
    return model, metadata


__all__ = [
    "DATASETS",
    "INFERENCE_STATE_KEY_COUNT",
    "RECIPE_ID",
    "RUNTIME_DEPENDENCY_RELATIVE_PATHS",
    "SCHEMA",
    "TRAINING_SEED",
    "TRAINING_STATE_KEY_COUNT",
    "build_v5_inference_model_from_training_state_dict",
    "build_v5_training_model",
    "canonical_sha256",
    "runtime_source_paths",
    "runtime_source_records",
    "strip_tss_for_inference_state_dict",
]
