#!/usr/bin/env python3
"""Exact epoch-boundary entry for V8-MPRS-DCH plus five-node NER.

Each explicit Full/Capacity × relay-off/on combination is a different exact
trajectory.  Resume is allowed only from the identical combination candidate.
A V8 tokenizer-only, V7, opposite-relay, or otherwise different journal is
rejected before model, optimizer, RNG, or DataLoader-generator state is
restored.

The verified V8 numerical loop and shared exact runner remain authoritative
for model/optimizer/scaler, manual scheduler position, epoch, RNG,
DataLoader-generator, metric history, and selection-history restoration.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import io
import json
import math
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import tpd_exact_resume as exact_resume  # noqa: E402
from experiments import tpd_exact_runner as exact_runner  # noqa: E402
from experiments import train_tpd_clean_v8_mprs_dch_exact as v8_kernel  # noqa: E402
from model.tpd_clean_v8_mprs_dch import (  # noqa: E402
    SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS,
)
from model.tpd_ner_v8_mprs_dch import (  # noqa: E402
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
    PRODUCTION_PARENT_PARAMETERS,
    PRODUCTION_RELAY_ON_PARAMETERS,
    PRODUCTION_RELAY_PARAMETERS,
    TPDNERV8MPRSDCHSCTransNet,
    relay_parameter_count,
)


ENTRY_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_exact_entry_v1"
EXACT_SOURCE_LOCK_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_exact_source_lock_v1"
)
ARCHITECTURE_MANIFEST_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_exact_architecture_manifest_v1"
)
CHECKPOINT_IDENTITY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_exact_checkpoint_identity_v1"
)
CHECKPOINT_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_exact_checkpoint_v1"
)
COMPLETION_SUMMARY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_completion_summary_v1"
)
SOURCE_LOCK_KEY = "tpd_ner_v8_mprs_dch_exact_source_lock"
RUN_ID_PREFIX = "tpd-ner-v8-mprs-dch-exact:"
EVALUATOR_CHECKPOINT_REQUIRED_FIELDS = (
    "schema",
    "epoch",
    "checkpoint_role",
    "variant",
    "parent_variant",
    "relay_enabled",
    "relay_width",
    "dataset",
    "seed",
    "split_seed",
    "state_dict",
    "optimizer",
    "scaler",
    "scheduler",
    "validation_metrics",
    "model_metadata",
    "split_hashes",
    "run_identity",
    "checkpoint_identity",
    "selection_source",
    "official_test_accessed",
)
DEFAULT_EXACT_SOURCE_LOCK_PATH = (
    REPO_ROOT / "experiments/tpd_ner_v8_mprs_dch_exact_source_lock.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "experiments/results/tpd_ner_v8_mprs_dch_exact_v1"
)

TRAINING_SEED = 42
SPLIT_SEED = 20260722
RELAY_WIDTH = 8
RELAY_INITIALIZATION_SEED = 42
FALLBACK_CANDIDATE_VARIANTS = (
    "tpd_ner_v8_mprs_dch_full_relay_off",
    "tpd_ner_v8_mprs_dch_full_relay_on",
    "tpd_ner_v8_mprs_dch_capacity_relay_off",
    "tpd_ner_v8_mprs_dch_capacity_relay_on",
)

FORMAL_EPOCHS = v8_kernel.FORMAL_EPOCHS
FORMAL_EVAL_EVERY = v8_kernel.FORMAL_EVAL_EVERY
FORMAL_WORKERS = v8_kernel.FORMAL_WORKERS
FORMAL_AMP = v8_kernel.FORMAL_AMP
FORMAL_EPS = v8_kernel.FORMAL_EPS
FORMAL_CUBLAS_WORKSPACE_CONFIG = (
    v8_kernel.FORMAL_CUBLAS_WORKSPACE_CONFIG
)
FORMAL_INITIALIZATION_MODES = v8_kernel.FORMAL_INITIALIZATION_MODES
PHYSICAL_GPU_UUIDS = dict(v8_kernel.PHYSICAL_GPU_UUIDS)
SELECTION_METRICS = v8_kernel.SELECTION_METRICS
STORED_VALIDATION_METRICS = v8_kernel.STORED_VALIDATION_METRICS

RUNTIME_SOURCE_PATHS = (
    REPO_ROOT / "experiments/train_tpd_ner_v8_mprs_dch_exact.py",
    REPO_ROOT / "experiments/train_tpd_ner_v8_mprs_dch.py",
    REPO_ROOT / "experiments/tpd_ner_runtime.py",
    REPO_ROOT / "model/tpd_ner_v8_mprs_dch.py",
    REPO_ROOT / "model/tpd_sctransnet.py",
    REPO_ROOT / "model/tpd_relay.py",
    REPO_ROOT / "model/tpd_clean.py",
    REPO_ROOT / "experiments/TPD_NER_V8_MPRS_DCH_PROTOCOL.md",
    *v8_kernel.RUNTIME_SOURCE_PATHS,
)

file_sha256 = v8_kernel.file_sha256
canonical_sha256 = v8_kernel.canonical_sha256
load_json_mapping = v8_kernel.load_json_mapping
PreparedData = v8_kernel.PreparedData
prepare_data = v8_kernel.prepare_data
split_fingerprints = v8_kernel.split_fingerprints
data_fingerprints = v8_kernel.data_fingerprints
configure_determinism = v8_kernel.configure_determinism
write_or_verify_json = v8_kernel.write_or_verify_json
base = v8_kernel.base
shared_exact = v8_kernel.shared_exact


def _ordinary_module():
    """Load the independently owned ordinary builder only when needed."""

    return importlib.import_module(
        "experiments.train_tpd_ner_v8_mprs_dch"
    )


def supported_candidate_variants() -> tuple[str, ...]:
    """Return the explicit combination identities owned by the builder."""

    ordinary = _ordinary_module()
    candidates = getattr(
        ordinary,
        "SUPPORTED_TPD_NER_V8_MPRS_DCH_VARIANTS",
        FALLBACK_CANDIDATE_VARIANTS,
    )
    if (
        not isinstance(candidates, (tuple, list))
        or not candidates
        or any(not isinstance(value, str) or not value for value in candidates)
    ):
        raise TypeError("ordinary V8+NER candidate matrix is invalid")
    normalized = tuple(candidates)
    if len(set(normalized)) != len(normalized):
        raise ValueError("ordinary V8+NER candidate matrix has duplicates")
    for candidate in normalized:
        candidate_contract(candidate)
    return normalized


def candidate_contract(candidate_variant: str) -> dict[str, Any]:
    """Decode one combination identity without creating an independent flag."""

    if not isinstance(candidate_variant, str):
        raise TypeError("V8+NER candidate variant must be a string")
    prefix = "tpd_ner_v8_mprs_dch_"
    if not candidate_variant.startswith(prefix):
        raise ValueError(f"invalid V8+NER candidate: {candidate_variant!r}")
    remainder = candidate_variant[len(prefix) :]
    if remainder.endswith("_relay_on"):
        relay_enabled = True
        parent_suffix = remainder[: -len("_relay_on")]
    elif remainder.endswith("_relay_off"):
        relay_enabled = False
        parent_suffix = remainder[: -len("_relay_off")]
    else:
        raise ValueError(
            "V8+NER candidate must end in _relay_off or _relay_on"
        )
    parent_variant = f"tpd_clean_v8_mprs_dch_{parent_suffix}"
    if parent_variant not in SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS:
        raise ValueError(
            f"V8+NER candidate maps to unsupported parent {parent_variant!r}"
        )
    return {
        "candidate_variant": candidate_variant,
        "parent_variant": parent_variant,
        "relay_enabled": relay_enabled,
    }


def formal_contract() -> dict[str, Any]:
    return {
        "epochs": FORMAL_EPOCHS,
        "eval_every": FORMAL_EVAL_EVERY,
        "workers": FORMAL_WORKERS,
        "amp": FORMAL_AMP,
        "eps": FORMAL_EPS,
        "cublas_workspace_config": FORMAL_CUBLAS_WORKSPACE_CONFIG,
        "initialization_modes": list(FORMAL_INITIALIZATION_MODES),
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "candidate_variants": list(supported_candidate_variants()),
        "relay_identity_source": "candidate_variant_suffix",
        "relay_width": RELAY_WIDTH,
        "relay_initialization_seed": RELAY_INITIALIZATION_SEED,
        "scheduler_restore": (
            "identity_bound_manual_schedule_from_completed_epoch"
        ),
    }


def _validate_formal_args(args: argparse.Namespace) -> None:
    contract = candidate_contract(getattr(args, "variant", None))
    if args.variant not in supported_candidate_variants():
        raise ValueError(
            f"unsupported formal V8+NER candidate {args.variant!r}"
        )
    expected = {
        "epochs": FORMAL_EPOCHS,
        "eval_every": FORMAL_EVAL_EVERY,
        "workers": FORMAL_WORKERS,
        "amp": FORMAL_AMP,
        "eps": FORMAL_EPS,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "parent_variant": contract["parent_variant"],
        "relay_enabled": contract["relay_enabled"],
        "relay_width": RELAY_WIDTH,
        "relay_initialization_seed": RELAY_INITIALIZATION_SEED,
    }
    for name, value in expected.items():
        if getattr(args, name, None) != value:
            raise ValueError(
                "V8-MPRS-DCH+NER exact training requires "
                f"{name}={value!r}"
            )


def _option_present(arguments: Sequence[str], option: str) -> bool:
    return any(
        value == option or value.startswith(f"{option}=")
        for value in arguments
    )


def _emit_ner_help() -> None:
    """Reuse the full exact option list while exposing only NER identities."""

    output = io.StringIO()
    parent_variant = candidate_contract(
        supported_candidate_variants()[0]
    )["parent_variant"]
    try:
        with contextlib.redirect_stdout(output):
            v8_kernel.parse_args(
                ["--variant", parent_variant, "--help"]
            )
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
    else:
        raise RuntimeError("underlying exact parser did not exit for --help")
    choices = (
        "{tpd_clean_v7_dch_full,tpd_clean_v7_dch_capacity}"
    )
    ner_choices = "{" + ",".join(supported_candidate_variants()) + "}"
    help_text = output.getvalue()
    help_text = help_text.replace(
        "train_tpd_clean_v8_mprs_dch_exact.py",
        Path(__file__).name,
    )
    help_text = help_text.replace(choices, ner_choices)
    help_text = help_text.replace(
        "Exact-resume TPD-Clean V7-DCH validation training",
        "Exact-resume V8-MPRS-DCH five-node NER training",
    )
    sys.stdout.write(help_text)
    raise SystemExit(0)


def _mapped_parent_arguments(
    arguments: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    """Map only the parser-facing candidate token to its V8 parent token."""

    forbidden = (
        "--relay-enabled",
        "--relay-off",
        "--no-relay-enabled",
        "--relay-width",
        "--relay-initialization-seed",
    )
    for value in arguments:
        if any(value == option or value.startswith(f"{option}=") for option in forbidden):
            raise ValueError(
                "relay identity, width, and initialization seed are "
                "fixed by the candidate variant"
            )

    mapped = list(arguments)
    selected: str | None = None
    for index, value in enumerate(mapped):
        if value == "--variant":
            if index + 1 >= len(mapped):
                break
            selected = mapped[index + 1]
            break
        if value.startswith("--variant="):
            selected = value.split("=", 1)[1]
            break
    if selected not in supported_candidate_variants():
        choices = ", ".join(supported_candidate_variants())
        raise ValueError(
            "V8+NER exact entry requires one explicit combination "
            f"--variant={{{choices}}}"
        )
    contract = candidate_contract(selected)
    for index, value in enumerate(mapped):
        if value == "--variant":
            mapped[index + 1] = contract["parent_variant"]
            break
        if value.startswith("--variant="):
            mapped[index] = f"--variant={contract['parent_variant']}"
            break
    return mapped, contract


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(value in ("-h", "--help") for value in arguments):
        _emit_ner_help()
    mapped, contract = _mapped_parent_arguments(arguments)
    args = v8_kernel.parse_args(mapped)
    args.variant = contract["candidate_variant"]
    args.parent_variant = contract["parent_variant"]
    if not _option_present(arguments, "--output-root"):
        args.output_root = DEFAULT_OUTPUT_ROOT
    if not _option_present(arguments, "--exact-source-lock"):
        args.exact_source_lock = DEFAULT_EXACT_SOURCE_LOCK_PATH
    args.relay_enabled = contract["relay_enabled"]
    args.relay_width = RELAY_WIDTH
    args.relay_initialization_seed = RELAY_INITIALIZATION_SEED
    _validate_formal_args(args)
    return args


def run_directory(args: argparse.Namespace) -> Path:
    _validate_formal_args(args)
    return (
        args.output_root.resolve()
        / args.dataset
        / args.variant
        / f"seed_{args.seed}_{args.run_tag}"
    )


InitializationPlan = v8_kernel.InitializationPlan


def _require_ner_run_identity(
    identity: Any,
    *,
    label: str,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} has no V8-MPRS-DCH+NER run identity")
    value = copy.deepcopy(dict(identity))
    variant = value.get("variant")
    run_id = value.get("run_id")
    source_locks = value.get("source_locks")
    training = value.get("training_contract")
    determinism = (
        training.get("determinism")
        if isinstance(training, Mapping)
        else None
    )
    entry_schema = (
        determinism.get("entry_schema")
        if isinstance(determinism, Mapping)
        else None
    )

    if (
        isinstance(variant, str)
        and variant.startswith("tpd_clean_v7_dch_")
    ) or (
        isinstance(run_id, str)
        and run_id.startswith("tpd-clean-v7-dch-exact:")
    ) or entry_schema == v8_kernel.exact_kernel.ENTRY_SCHEMA:
        raise ValueError("V7 exact resume into V8+NER is forbidden")
    if (
        entry_schema == v8_kernel.ENTRY_SCHEMA
        or (
            isinstance(run_id, str)
            and run_id.startswith(v8_kernel.RUN_ID_PREFIX)
        )
    ):
        raise ValueError("V8 tokenizer-only exact resume into NER is forbidden")
    if entry_schema != ENTRY_SCHEMA:
        raise ValueError(f"{label} entry schema is not V8+NER")
    if variant not in supported_candidate_variants():
        raise ValueError(f"{label} variant is not a V8+NER combination")
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(
            f"{label} variant {variant!r} differs from "
            f"requested {expected_variant!r}"
        )
    contract = candidate_contract(variant)
    if not isinstance(run_id, str) or not run_id.startswith(RUN_ID_PREFIX):
        raise ValueError(f"{label} run_id is not V8+NER")
    if (
        not isinstance(source_locks, Mapping)
        or SOURCE_LOCK_KEY not in source_locks
        or v8_kernel.SOURCE_LOCK_KEY in source_locks
        or "tpd_clean_v7_dch_exact_source_lock" in source_locks
    ):
        raise ValueError(f"{label} source-lock identity is not V8+NER")
    required_determinism = {
        "parent_variant": contract["parent_variant"],
        "relay_enabled": contract["relay_enabled"],
        "relay_width": RELAY_WIDTH,
        "relay_initialization_seed": RELAY_INITIALIZATION_SEED,
    }
    if not isinstance(determinism, Mapping):
        raise ValueError(f"{label} determinism identity is missing")
    for name, expected in required_determinism.items():
        if determinism.get(name) != expected:
            raise ValueError(
                f"{label} {name} differs from the candidate identity"
            )
    if value.get("seed") != TRAINING_SEED:
        raise ValueError(f"{label} training seed is not 42")
    if value.get("split_seed") != SPLIT_SEED:
        raise ValueError(f"{label} split seed is not 20260722")
    return value


def _existing_training_contract(
    args: argparse.Namespace,
    directory: Path,
) -> dict[str, Any]:
    protocol = load_json_mapping(
        directory / "protocol.json",
        "existing V8-MPRS-DCH+NER exact protocol",
    )
    schema = protocol.get("schema")
    if schema == v8_kernel.ENTRY_SCHEMA:
        raise ValueError("V8 tokenizer-only protocol cannot exact-resume NER")
    if schema == v8_kernel.exact_kernel.ENTRY_SCHEMA:
        raise ValueError("V7 protocol cannot exact-resume V8+NER")
    if schema != ENTRY_SCHEMA:
        raise ValueError("existing protocol is not a V8+NER exact run")
    identity = _require_ner_run_identity(
        protocol.get("run_identity"),
        label="existing protocol",
        expected_variant=args.variant,
    )
    try:
        training = identity["training_contract"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "existing V8+NER protocol has no training contract"
        ) from exc
    if not isinstance(training, Mapping):
        raise ValueError("existing training contract is not a mapping")
    required = (
        "initialization_contract",
        "initial_model_state_sha256",
        "initial_rng",
        "selection_policy",
    )
    missing = [name for name in required if name not in training]
    if missing:
        raise ValueError(
            f"existing V8+NER training contract lacks fields: {missing}"
        )
    return copy.deepcopy(dict(training))


def initialization_plan(
    args: argparse.Namespace,
    directory: Path,
    model: nn.Module,
) -> InitializationPlan:
    _validate_formal_args(args)
    if args.fresh:
        return InitializationPlan(
            request=exact_runner.InitializationRequest.fresh(),
            contract=exact_runner.fresh_initialization_contract(),
            initial_model_state_sha256=(
                exact_runner.initial_model_state_sha256(model)
            ),
        )
    if args.exact_resume:
        training = _existing_training_contract(args, directory)
        initial_rng = training["initial_rng"]
        selection_policy = training["selection_policy"]
        if not isinstance(initial_rng, Mapping):
            raise ValueError("existing initial_rng is not a mapping")
        if not isinstance(selection_policy, Mapping):
            raise ValueError("existing selection_policy is not a mapping")
        return InitializationPlan(
            request=exact_runner.InitializationRequest.exact(),
            contract=copy.deepcopy(training["initialization_contract"]),
            initial_model_state_sha256=str(
                training["initial_model_state_sha256"]
            ),
            initial_rng=copy.deepcopy(dict(initial_rng)),
            selection_policy=copy.deepcopy(dict(selection_policy)),
        )
    raise RuntimeError("V8+NER exact entry requires fresh or exact-resume")


def _architecture_manifest(
    variant: str,
    model: TPDNERV8MPRSDCHSCTransNet,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    relay_manifest = model.architecture_manifest()
    return {
        "schema": ARCHITECTURE_MANIFEST_SCHEMA,
        "variant": variant,
        "model": (
            "model.tpd_ner_v8_mprs_dch."
            "TPDNERV8MPRSDCHSCTransNet"
        ),
        "parent_model": "model.SCTransNet.SCTransNet",
        "candidate_family": metadata.get(
            "candidate_family",
            "tpd_ner_v8_mprs_dch",
        ),
        "mainline_contract": relay_manifest["mainline_contract"],
        "semantic_sources": relay_manifest["semantic_sources"],
        "embedding_replacements": (
            "mtc.embeddings_1",
            "mtc.embeddings_2",
        ),
        "evidence_nodes": relay_manifest["evidence_nodes"],
        "evidence_layout": relay_manifest["evidence_layout"],
        "relay_stage_order": relay_manifest["relay_stage_order"],
        "relay_enabled": relay_manifest["relay_enabled"],
        "relay_width": relay_manifest["relay_width"],
        "relay_initialization_seed": model.relay_initialization_seed,
        "relay_state_prefix": "tpd_ner.",
        "relay_parameters": relay_parameter_count(model),
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "ordinary_forward_uses_mprs_diagnostics": False,
        "exact_resume_scope": "same_combination_variant_only",
        "cross_version_exact_resume_supported": False,
        "eps": FORMAL_EPS,
        "formal_amp": FORMAL_AMP,
    }


def build_selected_model(
    variant: str,
    seed: int,
    *,
    eps: float = FORMAL_EPS,
) -> tuple[nn.Module, dict[str, Any]]:
    if variant not in supported_candidate_variants():
        raise ValueError(f"unsupported V8+NER exact variant: {variant!r}")
    contract = candidate_contract(variant)
    relay_enabled = contract["relay_enabled"]
    if seed != TRAINING_SEED:
        raise ValueError("V8+NER exact builder requires seed=42")
    if eps != FORMAL_EPS:
        raise ValueError(f"V8+NER exact builder requires eps={FORMAL_EPS}")
    ordinary = _ordinary_module()
    builder = getattr(
        ordinary,
        "build_tpd_ner_v8_mprs_dch_model",
        None,
    )
    if not callable(builder):
        raise TypeError(
            "ordinary V8+NER trainer must expose "
            "build_tpd_ner_v8_mprs_dch_model"
        )
    model, raw_metadata = builder(variant, seed)
    if not isinstance(model, TPDNERV8MPRSDCHSCTransNet):
        raise TypeError("ordinary builder did not return the V8+NER model")
    if model.relay_enabled is not relay_enabled:
        raise ValueError("exact model relay identity differs from candidate")
    if hasattr(model, "tpd_ner") is not relay_enabled:
        raise ValueError("exact model relay registration differs")
    if model.tokenizer_variant != contract["parent_variant"]:
        raise ValueError("exact model parent variant differs from candidate")
    if model.relay_width != RELAY_WIDTH:
        raise ValueError("exact model relay width differs from 8")
    if model.relay_initialization_seed != RELAY_INITIALIZATION_SEED:
        raise ValueError("exact model relay initialization seed differs")
    expected_relay_parameters = (
        PRODUCTION_RELAY_PARAMETERS if relay_enabled else 0
    )
    if relay_parameter_count(model) != expected_relay_parameters:
        raise ValueError("exact model relay parameter count differs")
    expected_total_parameters = (
        PRODUCTION_RELAY_ON_PARAMETERS
        if relay_enabled
        else PRODUCTION_PARENT_PARAMETERS
    )
    if sum(parameter.numel() for parameter in model.parameters()) != (
        expected_total_parameters
    ):
        raise ValueError("exact model total parameter count differs")
    if not isinstance(raw_metadata, Mapping):
        raise TypeError("ordinary builder metadata is not a mapping")
    metadata = copy.deepcopy(dict(raw_metadata))
    if metadata.get("variant", variant) != variant:
        raise ValueError("ordinary builder metadata variant mismatch")
    if metadata.get("relay_enabled") is not relay_enabled:
        raise ValueError("ordinary builder metadata relay identity differs")
    if metadata.get("parent_variant") != contract["parent_variant"]:
        raise ValueError("ordinary builder metadata parent variant differs")
    metadata.update(
        {
            "variant": variant,
            "parent_variant": contract["parent_variant"],
            "relay_enabled": relay_enabled,
            "relay_width": RELAY_WIDTH,
            "relay_initialization_seed": (
                RELAY_INITIALIZATION_SEED
            ),
            "training_seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "cross_version_exact_resume_supported": False,
        }
    )
    ordinary_architecture_id = metadata.get("architecture_id")
    if (
        not isinstance(ordinary_architecture_id, str)
        or len(ordinary_architecture_id) != 64
    ):
        raise ValueError("ordinary builder architecture_id is invalid")
    metadata["ordinary_builder_architecture_id"] = (
        ordinary_architecture_id
    )
    exact_manifest = _architecture_manifest(
        variant,
        model,
        metadata,
    )
    metadata["architecture_manifest"] = exact_manifest
    metadata["architecture_id"] = canonical_sha256(exact_manifest)
    return model, metadata


def resolve_device(args: argparse.Namespace) -> torch.device:
    _validate_formal_args(args)
    return v8_kernel.resolve_device(args)


def environment_contract(device: torch.device) -> dict[str, Any]:
    payload = shared_exact.environment_contract(device)
    if device.type != "cuda":
        payload.update(
            {
                "physical_gpu_index": None,
                "physical_gpu_uuid": None,
                "physical_gpu_assignment_source": None,
            }
        )
        return payload
    physical_index = os.environ.get(
        "TPD_NER_V8_MPRS_DCH_PHYSICAL_GPU_INDEX"
    )
    physical_uuid = os.environ.get(
        "TPD_NER_V8_MPRS_DCH_PHYSICAL_GPU_UUID"
    )
    if physical_index not in PHYSICAL_GPU_UUIDS:
        raise RuntimeError(
            "TPD_NER_V8_MPRS_DCH_PHYSICAL_GPU_INDEX must identify "
            "physical GPU 2 or 3"
        )
    expected_uuid = PHYSICAL_GPU_UUIDS[physical_index]
    if physical_uuid != expected_uuid:
        raise RuntimeError(
            "TPD_NER_V8_MPRS_DCH_PHYSICAL_GPU_UUID differs from "
            f"physical GPU {physical_index}"
        )
    if payload.get("device_uuid") != expected_uuid:
        raise RuntimeError("visible CUDA UUID differs from physical assignment")
    if payload.get("cuda_visible_devices") != expected_uuid:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must use the registered UUID")
    payload.update(
        {
            "physical_gpu_index": int(physical_index),
            "physical_gpu_uuid": expected_uuid,
            "physical_gpu_assignment_source": (
                "verified_v8_ner_worker_environment"
            ),
        }
    )
    return payload


def source_lock_contract(
    training_data_sha256: str,
    exact_source_lock_path: Path,
) -> dict[str, str]:
    path = Path(exact_source_lock_path).resolve()
    payload = load_json_mapping(path, "V8+NER exact source lock")
    if payload.get("schema") != EXACT_SOURCE_LOCK_SCHEMA:
        raise ValueError("V8+NER exact source-lock schema mismatch")
    if (
        tuple(payload.get("variants", ()))
        != supported_candidate_variants()
    ):
        raise ValueError("V8+NER exact source-lock variant matrix mismatch")
    if payload.get("formal_contract") != formal_contract():
        raise ValueError("V8+NER exact source-lock formal contract mismatch")
    if payload.get("training_data_sha256") != training_data_sha256:
        raise ValueError("training data differs from the V8+NER source lock")
    locked = payload.get("source_sha256")
    if not isinstance(locked, Mapping):
        raise ValueError("V8+NER exact source lock has no source mapping")
    required = {
        str(runtime.relative_to(REPO_ROOT))
        for runtime in RUNTIME_SOURCE_PATHS
    }
    missing = sorted(required - set(locked))
    if missing:
        raise ValueError(
            f"V8+NER exact source lock omits runtime sources: {missing}"
        )
    for relative, expected in locked.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("V8+NER source lock has an invalid path")
        runtime = (REPO_ROOT / relative).resolve()
        try:
            canonical = str(runtime.relative_to(REPO_ROOT))
        except ValueError as exc:
            raise ValueError("V8+NER source path escapes repository") from exc
        if canonical != relative:
            raise ValueError("V8+NER source path is not canonical")
        if file_sha256(runtime) != expected:
            raise ValueError(
                f"V8+NER source lock differs for {relative}"
            )
    return {
        SOURCE_LOCK_KEY: file_sha256(path),
        "training_data": training_data_sha256,
    }


def make_exact_run_spec(
    args: argparse.Namespace,
    *,
    model: nn.Module,
    model_metadata: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    initialization_contract: Mapping[str, Any],
    initial_model_state_sha256: str,
    initial_rng: Mapping[str, Any],
    selection_policy: Mapping[str, Any],
    source_locks: Mapping[str, str],
    split_records: Mapping[str, exact_runner.OrderedFingerprint],
    data_records: Mapping[str, exact_runner.OrderedFingerprint],
    environment: Mapping[str, Any],
) -> exact_runner.ExactRunSpec:
    _validate_formal_args(args)
    candidate = candidate_contract(args.variant)
    manifest = model_metadata.get("architecture_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("V8+NER metadata has no architecture manifest")
    if manifest.get("schema") != ARCHITECTURE_MANIFEST_SCHEMA:
        raise ValueError("V8+NER architecture manifest schema mismatch")
    if manifest.get("variant") != args.variant:
        raise ValueError("V8+NER architecture manifest variant mismatch")
    if manifest.get("relay_enabled") is not candidate["relay_enabled"]:
        raise ValueError("V8+NER architecture manifest relay mismatch")
    if manifest.get("relay_width") != RELAY_WIDTH:
        raise ValueError("V8+NER architecture manifest width differs")

    adapted = v8_kernel._clone_kernel_function(
        v8_kernel.exact_kernel.make_exact_run_spec,
        {
            "_validate_formal_args": _validate_formal_args,
            "ENTRY_SCHEMA": ENTRY_SCHEMA,
            "formal_contract": formal_contract,
            "FORMAL_EPOCHS": FORMAL_EPOCHS,
            "FORMAL_EVAL_EVERY": FORMAL_EVAL_EVERY,
            "FORMAL_WORKERS": FORMAL_WORKERS,
            "FORMAL_AMP": FORMAL_AMP,
            "FORMAL_EPS": FORMAL_EPS,
            "FORMAL_CUBLAS_WORKSPACE_CONFIG": (
                FORMAL_CUBLAS_WORKSPACE_CONFIG
            ),
        },
    )
    spec = adapted(
        args,
        model=model,
        model_metadata=model_metadata,
        optimizer=optimizer,
        scaler=scaler,
        initialization_contract=initialization_contract,
        initial_model_state_sha256=initial_model_state_sha256,
        initial_rng=initial_rng,
        selection_policy=selection_policy,
        source_locks=source_locks,
        split_records=split_records,
        data_records=data_records,
        environment=environment,
    )
    determinism = dict(spec.determinism)
    determinism.update(
        {
            "entry_schema": ENTRY_SCHEMA,
            "parent_variant": candidate["parent_variant"],
            "relay_enabled": candidate["relay_enabled"],
            "relay_width": RELAY_WIDTH,
            "relay_initialization_seed": (
                RELAY_INITIALIZATION_SEED
            ),
            "scheduler_restore_mode": (
                "identity_bound_manual_schedule_from_completed_epoch"
            ),
        }
    )
    return replace(
        spec,
        run_id=(
            f"{RUN_ID_PREFIX}{args.dataset}:{args.variant}:"
            f"seed-{args.seed}:{args.run_tag}"
        ),
        determinism=determinism,
    )


def _require_complete_validation_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [
        name for name in STORED_VALIDATION_METRICS if name not in metrics
    ]
    if missing:
        raise ValueError(
            f"V8+NER validation metrics lack fields: {missing}"
        )
    return {
        name: copy.deepcopy(metrics[name])
        for name in STORED_VALIDATION_METRICS
    }


def require_evaluator_checkpoint_payload(
    payload: Any,
    *,
    expected_variant: str | None = None,
) -> dict[str, Any]:
    """Validate the exact compatibility artifact consumed by evaluation."""

    if not isinstance(payload, Mapping):
        raise ValueError("V8+NER exact evaluator checkpoint is not a mapping")
    value = copy.deepcopy(dict(payload))
    missing = [
        name
        for name in EVALUATOR_CHECKPOINT_REQUIRED_FIELDS
        if name not in value
    ]
    if missing:
        raise ValueError(
            f"V8+NER exact evaluator checkpoint lacks fields: {missing}"
        )
    if value["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError("V8+NER exact evaluator checkpoint schema differs")
    identity = _require_ner_run_identity(
        value["run_identity"],
        label="evaluator checkpoint",
        expected_variant=expected_variant,
    )
    candidate = candidate_contract(identity["variant"])
    expected_top_level = {
        "variant": identity["variant"],
        "parent_variant": candidate["parent_variant"],
        "relay_enabled": candidate["relay_enabled"],
        "relay_width": RELAY_WIDTH,
        "dataset": identity["dataset"],
        "seed": identity["seed"],
        "split_seed": identity["split_seed"],
        "official_test_accessed": False,
    }
    for name, expected in expected_top_level.items():
        if value.get(name) != expected:
            raise ValueError(
                f"V8+NER evaluator checkpoint {name} differs"
            )
    epoch = value["epoch"]
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 1
        or epoch > FORMAL_EPOCHS
    ):
        raise ValueError("V8+NER evaluator checkpoint epoch is invalid")
    if (
        not isinstance(value["checkpoint_role"], str)
        or value["checkpoint_role"]
        not in {
            "last_evaluated_epoch",
            "best_validation_pd_primary",
            "best_validation_miou_secondary",
        }
    ):
        raise ValueError("V8+NER evaluator checkpoint role is invalid")
    for name in ("state_dict", "optimizer", "scaler"):
        if not isinstance(value[name], Mapping) or (
            name == "state_dict" and not value[name]
        ):
            raise ValueError(
                f"V8+NER evaluator checkpoint {name} is invalid"
            )
    expected_scheduler = {
        "kind": "identity_bound_manual_schedule",
        "completed_epoch": epoch,
    }
    if (
        not isinstance(value["scheduler"], Mapping)
        or dict(value["scheduler"]) != expected_scheduler
    ):
        raise ValueError("V8+NER evaluator checkpoint scheduler differs")
    validation_metrics = _require_complete_validation_metrics(
        value["validation_metrics"]
    )
    for name, metric in validation_metrics.items():
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float, np.number))
            or not math.isfinite(float(metric))
        ):
            raise ValueError(
                f"V8+NER evaluator checkpoint metric {name} is invalid"
            )
    metadata = value["model_metadata"]
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("variant") != identity["variant"]
        or metadata.get("parent_variant") != candidate["parent_variant"]
        or metadata.get("relay_enabled") is not candidate["relay_enabled"]
        or metadata.get("relay_width") != RELAY_WIDTH
        or metadata.get("relay_initialization_seed")
        != RELAY_INITIALIZATION_SEED
    ):
        raise ValueError("V8+NER evaluator model metadata differs")
    manifest = metadata.get("architecture_manifest")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != ARCHITECTURE_MANIFEST_SCHEMA
        or manifest.get("variant") != identity["variant"]
        or manifest.get("relay_enabled") is not candidate["relay_enabled"]
        or manifest.get("relay_width") != RELAY_WIDTH
    ):
        raise ValueError("V8+NER evaluator architecture manifest differs")
    manifest_sha256 = canonical_sha256(manifest)
    if (
        metadata.get("architecture_id") != manifest_sha256
        or identity["builder_manifest_sha256"] != manifest_sha256
    ):
        raise ValueError("V8+NER evaluator architecture digest differs")
    split_hashes = value["split_hashes"]
    if not isinstance(split_hashes, Mapping) or not split_hashes:
        raise ValueError("V8+NER evaluator split hashes are invalid")
    for name, digest in split_hashes.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("V8+NER evaluator split hash is invalid")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(
                "V8+NER evaluator split hash is not hexadecimal"
            ) from exc
    checkpoint_identity = value["checkpoint_identity"]
    if not isinstance(checkpoint_identity, Mapping):
        raise ValueError("V8+NER checkpoint identity is invalid")
    expected_checkpoint_identity = {
        "schema": CHECKPOINT_IDENTITY_SCHEMA,
        "variant": identity["variant"],
        "parent_variant": candidate["parent_variant"],
        "relay_enabled": candidate["relay_enabled"],
        "relay_width": RELAY_WIDTH,
        "run_id": identity["run_id"],
        "architecture_id": identity["architecture_id"],
        "builder_manifest_sha256": identity[
            "builder_manifest_sha256"
        ],
    }
    if dict(checkpoint_identity) != expected_checkpoint_identity:
        raise ValueError("V8+NER checkpoint identity differs")
    if value["selection_source"] != "internal_validation_only":
        raise ValueError("V8+NER evaluator selection source differs")
    return value


@dataclass(frozen=True)
class EvaluatorCheckpointAdapter:
    model_metadata: Mapping[str, Any]
    split_hashes: Mapping[str, str]

    def __call__(
        self,
        context: exact_runner.CompatibilityPayloadContext,
    ) -> Mapping[str, Any]:
        identity = _require_ner_run_identity(
            context.run_identity,
            label="checkpoint context",
        )
        candidate = candidate_contract(identity["variant"])
        exact_payload = context.exact_payload
        validation_metrics = _require_complete_validation_metrics(
            context.metrics
        )
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "epoch": context.epoch,
            "checkpoint_role": context.role,
            "variant": identity["variant"],
            "parent_variant": candidate["parent_variant"],
            "relay_enabled": candidate["relay_enabled"],
            "relay_width": RELAY_WIDTH,
            "dataset": identity["dataset"],
            "seed": identity["seed"],
            "split_seed": identity["split_seed"],
            "state_dict": copy.deepcopy(
                exact_payload["model"]["state_dict"]
            ),
            "optimizer": copy.deepcopy(
                exact_payload["optimizer"]["state_dict"]
            ),
            "scaler": copy.deepcopy(
                exact_payload["scaler"]["state_dict"]
            ),
            "scheduler": {
                "kind": "identity_bound_manual_schedule",
                "completed_epoch": context.epoch,
            },
            "validation_metrics": validation_metrics,
            "model_metadata": copy.deepcopy(dict(self.model_metadata)),
            "split_hashes": copy.deepcopy(dict(self.split_hashes)),
            "run_identity": identity,
            "checkpoint_identity": {
                "schema": CHECKPOINT_IDENTITY_SCHEMA,
                "variant": identity["variant"],
                "parent_variant": candidate["parent_variant"],
                "relay_enabled": candidate["relay_enabled"],
                "relay_width": RELAY_WIDTH,
                "run_id": identity["run_id"],
                "architecture_id": identity["architecture_id"],
                "builder_manifest_sha256": identity[
                    "builder_manifest_sha256"
                ],
            },
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
        }
        return require_evaluator_checkpoint_payload(
            payload,
            expected_variant=identity["variant"],
        )


class TPDNERV8ExactRunner(v8_kernel.MPRSDCHExactRunner):
    """Exact runner whose pre-restore guard requires the same NER identity."""

    def _require_v8_active_journal(self) -> None:
        active = self.journal.load_active()
        if active is None:
            return
        payload, _ = self._load_exact_payload(active.checkpoint_path)
        _require_ner_run_identity(
            payload.get("run_identity"),
            label="active exact journal",
            expected_variant=self.spec.variant,
        )
        if not isinstance(payload.get("optimizer"), Mapping):
            raise ValueError("active V8+NER journal has no optimizer state")


DCHExactRunner = TPDNERV8ExactRunner


def six_output_bce_loss(
    outputs: Any,
    target: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    return v8_kernel.six_output_bce_loss(outputs, target, criterion)


def training_arguments(args: argparse.Namespace) -> dict[str, Any]:
    _validate_formal_args(args)
    candidate = candidate_contract(args.variant)
    payload = v8_kernel.training_arguments(args)
    payload.update(
        {
            "parent_variant": candidate["parent_variant"],
            "relay_enabled": candidate["relay_enabled"],
            "relay_width": RELAY_WIDTH,
            "relay_initialization_seed": RELAY_INITIALIZATION_SEED,
        }
    )
    return payload


def protocol_payload(
    args: argparse.Namespace,
    *,
    directory: Path,
    model_metadata: Mapping[str, Any],
    normalization: Mapping[str, float],
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = candidate_contract(args.variant)
    identity = _require_ner_run_identity(
        run_identity,
        label="protocol",
        expected_variant=args.variant,
    )
    adapted = v8_kernel._clone_kernel_function(
        v8_kernel.exact_kernel.protocol_payload,
        {
            "ENTRY_SCHEMA": ENTRY_SCHEMA,
            "formal_contract": formal_contract,
            "training_arguments": training_arguments,
            "SELECTION_METRICS": SELECTION_METRICS,
            "STORED_VALIDATION_METRICS": (
                STORED_VALIDATION_METRICS
            ),
        },
    )
    payload = adapted(
        args,
        directory=directory,
        model_metadata=model_metadata,
        normalization=normalization,
        run_identity=identity,
    )
    payload["relay_identity"] = {
        "source": "candidate_variant_suffix",
        "parent_variant": candidate["parent_variant"],
        "enabled": candidate["relay_enabled"],
        "width": RELAY_WIDTH,
        "initialization_seed": RELAY_INITIALIZATION_SEED,
    }
    payload["exact_resume_policy"] = {
        "same_version": "same_combination_variant_epoch_boundary_only",
        "relay_off_to_on": "forbidden",
        "relay_on_to_off": "forbidden",
        "v8_tokenizer_only_optimizer_journal": "forbidden",
        "v7_optimizer_journal": "forbidden",
        "scheduler_restore": (
            "manual_schedule_reconstructed_from_identity_and_epoch"
        ),
    }
    return payload


def _load_complete_events(
    path: Path,
    epochs: int,
) -> list[dict[str, Any]]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(events) != epochs or [
        event.get("epoch") for event in events
    ] != list(range(1, epochs + 1)):
        raise RuntimeError(
            "V8+NER exact metrics are not a contiguous complete history"
        )
    for event in events:
        _require_complete_validation_metrics(event)
    return events


def completion_summary(
    args: argparse.Namespace,
    *,
    directory: Path,
    model_metadata: Mapping[str, Any],
    split_hashes: Mapping[str, str],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = candidate_contract(args.variant)
    adapted = v8_kernel._clone_kernel_function(
        v8_kernel.exact_kernel.completion_summary,
        {
            "FORMAL_EPOCHS": FORMAL_EPOCHS,
            "formal_contract": formal_contract,
            "SELECTION_METRICS": SELECTION_METRICS,
            "STORED_VALIDATION_METRICS": (
                STORED_VALIDATION_METRICS
            ),
            "_load_complete_events": _load_complete_events,
            "_require_complete_validation_metrics": (
                _require_complete_validation_metrics
            ),
        },
    )
    payload = adapted(
        args,
        directory=directory,
        model_metadata=model_metadata,
        split_hashes=split_hashes,
        selection=selection,
    )
    payload["schema"] = COMPLETION_SUMMARY_SCHEMA
    payload["candidate_family"] = "tpd_ner_v8_mprs_dch"
    protocol = load_json_mapping(
        directory / "protocol.json",
        "completed V8+NER exact protocol",
    )
    if protocol.get("schema") != ENTRY_SCHEMA:
        raise ValueError("completed V8+NER protocol schema differs")
    payload["run_identity"] = _require_ner_run_identity(
        protocol.get("run_identity"),
        label="completion protocol",
        expected_variant=args.variant,
    )
    payload["split_seed"] = args.split_seed
    payload["parent_variant"] = candidate["parent_variant"]
    payload["relay_enabled"] = candidate["relay_enabled"]
    payload["relay_width"] = RELAY_WIDTH
    return payload


def _check_metrics(metrics: Mapping[str, Any]) -> None:
    _require_complete_validation_metrics(metrics)
    for name, value in metrics.items():
        if (
            isinstance(value, (int, float, np.number))
            and not isinstance(value, bool)
            and not math.isfinite(float(value))
        ):
            raise FloatingPointError(
                f"non-finite validation metric {name!r}: {value!r}"
            )


def _reused_training_kernel() -> Any:
    base_loop = v8_kernel._reused_training_kernel()
    return v8_kernel._clone_kernel_function(
        base_loop,
        {
            "_validate_formal_args": _validate_formal_args,
            "resolve_device": resolve_device,
            "run_directory": run_directory,
            "source_lock_contract": source_lock_contract,
            "build_selected_model": build_selected_model,
            "initialization_plan": initialization_plan,
            "make_exact_run_spec": make_exact_run_spec,
            "environment_contract": environment_contract,
            "EvaluatorCheckpointAdapter": EvaluatorCheckpointAdapter,
            "DCHExactRunner": TPDNERV8ExactRunner,
            "six_output_bce_loss": six_output_bce_loss,
            "_check_metrics": _check_metrics,
            "protocol_payload": protocol_payload,
            "completion_summary": completion_summary,
            "FORMAL_EPOCHS": FORMAL_EPOCHS,
            "FORMAL_EVAL_EVERY": FORMAL_EVAL_EVERY,
            "FORMAL_WORKERS": FORMAL_WORKERS,
            "FORMAL_AMP": FORMAL_AMP,
            "FORMAL_EPS": FORMAL_EPS,
            "STORED_VALIDATION_METRICS": (
                STORED_VALIDATION_METRICS
            ),
        },
    )


def run_training(args: argparse.Namespace) -> Path:
    _validate_formal_args(args)
    return _reused_training_kernel()(args)


def main(argv: Sequence[str] | None = None) -> None:
    run_training(parse_args(argv))


__all__ = [
    "ARCHITECTURE_MANIFEST_SCHEMA",
    "CHECKPOINT_IDENTITY_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "COMPLETION_SUMMARY_SCHEMA",
    "DEFAULT_EXACT_SOURCE_LOCK_PATH",
    "DEFAULT_OUTPUT_ROOT",
    "DCHExactRunner",
    "ENTRY_SCHEMA",
    "EVALUATOR_CHECKPOINT_REQUIRED_FIELDS",
    "EXACT_SOURCE_LOCK_SCHEMA",
    "EvaluatorCheckpointAdapter",
    "FORMAL_AMP",
    "FORMAL_CUBLAS_WORKSPACE_CONFIG",
    "FORMAL_EPOCHS",
    "FORMAL_EPS",
    "FORMAL_EVAL_EVERY",
    "FORMAL_WORKERS",
    "FALLBACK_CANDIDATE_VARIANTS",
    "PHYSICAL_GPU_UUIDS",
    "RELAY_INITIALIZATION_SEED",
    "RELAY_WIDTH",
    "RUNTIME_SOURCE_PATHS",
    "RUN_ID_PREFIX",
    "SELECTION_METRICS",
    "SOURCE_LOCK_KEY",
    "SPLIT_SEED",
    "STORED_VALIDATION_METRICS",
    "TPDNERV8ExactRunner",
    "TRAINING_SEED",
    "build_selected_model",
    "candidate_contract",
    "completion_summary",
    "environment_contract",
    "formal_contract",
    "initialization_plan",
    "main",
    "make_exact_run_spec",
    "parse_args",
    "protocol_payload",
    "require_evaluator_checkpoint_payload",
    "run_directory",
    "run_training",
    "six_output_bce_loss",
    "source_lock_contract",
    "supported_candidate_variants",
]


if __name__ == "__main__":
    main()
