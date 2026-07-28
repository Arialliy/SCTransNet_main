#!/usr/bin/env python3
"""Exact epoch-boundary training entry for TPD-Clean V8-MPRS-DCH.

This module is an identity adapter around the already verified V7-DCH
``run_training`` numerical loop and the shared exact-resume engine.  It owns
every V8-persisted identity while deliberately reusing the established data,
optimizer, RNG, epoch-journal, checkpoint-selection, and training-step code.

Only V8-to-V8 exact resume is supported.  A V7 checkpoint may be strict-loaded
as model state by a separate read-only diagnostic, but this entry refuses a V7
protocol or active exact journal before any V7 optimizer/RNG state can load.
Importing the module creates no files and mutates no shared training globals.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import types
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
from experiments import train_tpd_clean_v7_dch_exact as exact_kernel  # noqa: E402
from experiments import train_tpd_clean_v8_mprs_dch as ordinary  # noqa: E402
from model.tpd_clean_v8_mprs_dch import (  # noqa: E402
    SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS,
    TPDCleanV8MPRSDCHBlock,
)


ENTRY_SCHEMA = "sctransnet_tpd_clean_v8_mprs_dch_exact_entry_v1"
EXACT_SOURCE_LOCK_SCHEMA = (
    "sctransnet_tpd_clean_v8_mprs_dch_exact_source_lock_v1"
)
ARCHITECTURE_MANIFEST_SCHEMA = (
    "sctransnet_tpd_clean_v8_mprs_dch_exact_architecture_manifest_v1"
)
CHECKPOINT_IDENTITY_SCHEMA = (
    "sctransnet_tpd_clean_v8_mprs_dch_checkpoint_identity_v1"
)
COMPLETION_SUMMARY_SCHEMA = (
    "sctransnet_tpd_clean_v8_mprs_dch_completion_summary_v1"
)
SOURCE_LOCK_KEY = "tpd_clean_v8_mprs_dch_exact_source_lock"
RUN_ID_PREFIX = "tpd-clean-v8-mprs-dch-exact:"
DEFAULT_EXACT_SOURCE_LOCK_PATH = (
    REPO_ROOT / "experiments/tpd_clean_v8_mprs_dch_exact_source_lock.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "experiments/results/tpd_clean_v8_mprs_dch_exact_v1"
)

FORMAL_EPOCHS = exact_kernel.FORMAL_EPOCHS
FORMAL_EVAL_EVERY = exact_kernel.FORMAL_EVAL_EVERY
FORMAL_WORKERS = exact_kernel.FORMAL_WORKERS
FORMAL_AMP = exact_kernel.FORMAL_AMP
FORMAL_EPS = exact_kernel.FORMAL_EPS
FORMAL_CUBLAS_WORKSPACE_CONFIG = (
    exact_kernel.FORMAL_CUBLAS_WORKSPACE_CONFIG
)
FORMAL_INITIALIZATION_MODES = exact_kernel.FORMAL_INITIALIZATION_MODES
PHYSICAL_GPU_UUIDS = dict(exact_kernel.PHYSICAL_GPU_UUIDS)

SELECTION_METRICS = exact_kernel.SELECTION_METRICS
STORED_VALIDATION_METRICS = exact_kernel.STORED_VALIDATION_METRICS

# The V7 exact entry is locked here only because its verified numerical loop is
# executed by ``run_training`` below.  No V7-owned identity is copied into a V8
# run.  The remaining paths are the actual eager/runtime dependencies of that
# shared loop plus the V8-owned model, entry, and protocol.
RUNTIME_SOURCE_PATHS = (
    REPO_ROOT / "experiments/train_tpd_clean_v8_mprs_dch_exact.py",
    REPO_ROOT / "experiments/train_tpd_clean_v8_mprs_dch.py",
    REPO_ROOT / "model/tpd_clean_v8_mprs_dch.py",
    REPO_ROOT / "experiments/TPD_CLEAN_V8_MPRS_DCH_PROTOCOL.md",
    REPO_ROOT
    / "experiments/TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md",
    REPO_ROOT / "experiments/train_tpd_clean_v7_dch_exact.py",
    REPO_ROOT / "experiments/train_tpd_clean_v7_dch.py",
    REPO_ROOT / "model/tpd_clean_v7_dch.py",
    REPO_ROOT / "experiments/train_tpd_clean_v6_exact.py",
    REPO_ROOT / "experiments/train_tpd_clean_v6.py",
    REPO_ROOT / "model/tpd_clean_v6.py",
    REPO_ROOT / "experiments/tpd_exact_runner.py",
    REPO_ROOT / "experiments/tpd_exact_resume.py",
    REPO_ROOT / "experiments/tpd_exact_epoch_journal.py",
    REPO_ROOT / "experiments/tpd_exact_training_runtime.py",
    REPO_ROOT / "experiments/tpd_extension_warm_start.py",
    REPO_ROOT / "experiments/train_tpd_pilot.py",
    REPO_ROOT / "experiments/fingerprint_tpd_training_data.py",
    REPO_ROOT / "model/SCTransNet.py",
    REPO_ROOT / "model/Config.py",
    REPO_ROOT / "model/tpd.py",
    REPO_ROOT / "dataset.py",
    REPO_ROOT / "utils.py",
    REPO_ROOT / "warmup_scheduler.py",
)

file_sha256 = exact_kernel.file_sha256
canonical_sha256 = exact_kernel.canonical_sha256
load_json_mapping = exact_kernel.load_json_mapping
PreparedData = exact_kernel.PreparedData
prepare_data = exact_kernel.prepare_data
split_fingerprints = exact_kernel.split_fingerprints
data_fingerprints = exact_kernel.data_fingerprints
normalized_gpu_uuid = exact_kernel.normalized_gpu_uuid
visible_gpu_identity = exact_kernel.visible_gpu_identity
configure_determinism = exact_kernel.configure_determinism
write_or_verify_json = exact_kernel.write_or_verify_json
base = exact_kernel.base
shared_exact = exact_kernel.shared_exact


def formal_contract() -> dict[str, Any]:
    """Return the immutable V8 formal-training axes."""

    return {
        "epochs": FORMAL_EPOCHS,
        "eval_every": FORMAL_EVAL_EVERY,
        "workers": FORMAL_WORKERS,
        "amp": FORMAL_AMP,
        "eps": FORMAL_EPS,
        "cublas_workspace_config": FORMAL_CUBLAS_WORKSPACE_CONFIG,
        "initialization_modes": list(FORMAL_INITIALIZATION_MODES),
    }


def _validate_formal_args(args: argparse.Namespace) -> None:
    expected = {
        "epochs": FORMAL_EPOCHS,
        "eval_every": FORMAL_EVAL_EVERY,
        "workers": FORMAL_WORKERS,
        "amp": FORMAL_AMP,
        "eps": FORMAL_EPS,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(
                "V8-MPRS-DCH formal exact training requires "
                f"{name}={value!r}"
            )


def _option_present(arguments: Sequence[str], option: str) -> bool:
    return any(
        value == option or value.startswith(f"{option}=")
        for value in arguments
    )


def _mapped_v7_arguments(
    arguments: Sequence[str],
) -> tuple[list[str], str]:
    mapped = list(arguments)
    selected: str | None = None
    for index, value in enumerate(mapped):
        if value == "--variant":
            if index + 1 >= len(mapped):
                break
            selected = mapped[index + 1]
            if selected in SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS:
                mapped[index + 1] = selected.replace(
                    "tpd_clean_v8_mprs_dch_",
                    "tpd_clean_v7_dch_",
                    1,
                )
            break
        if value.startswith("--variant="):
            selected = value.split("=", 1)[1]
            if selected in SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS:
                mapped[index] = "--variant=" + selected.replace(
                    "tpd_clean_v8_mprs_dch_",
                    "tpd_clean_v7_dch_",
                    1,
                )
            break
    if selected not in SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS:
        choices = ", ".join(SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS)
        raise ValueError(
            "V8-MPRS-DCH exact entry requires one of "
            f"--variant={{{choices}}}"
        )
    return mapped, selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Reuse the frozen formal CLI validation while rebinding V8 identities."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    mapped, selected = _mapped_v7_arguments(arguments)
    args = exact_kernel.parse_args(mapped)
    args.variant = selected
    if not _option_present(arguments, "--output-root"):
        args.output_root = DEFAULT_OUTPUT_ROOT
    if not _option_present(arguments, "--exact-source-lock"):
        args.exact_source_lock = DEFAULT_EXACT_SOURCE_LOCK_PATH
    _validate_formal_args(args)
    return args


def run_directory(args: argparse.Namespace) -> Path:
    return (
        args.output_root.resolve()
        / args.dataset
        / args.variant
        / f"seed_{args.seed}_{args.run_tag}"
    )


InitializationPlan = exact_kernel.InitializationPlan


def _require_v8_run_identity(
    identity: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} has no V8 run identity")
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
    ) or entry_schema == exact_kernel.ENTRY_SCHEMA:
        raise ValueError(
            "cross-version exact resume from a V7 optimizer/journal "
            "is forbidden"
        )
    if entry_schema != ENTRY_SCHEMA:
        raise ValueError(
            f"{label} entry schema is not V8-MPRS-DCH"
        )
    if variant not in SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS:
        raise ValueError(f"{label} variant is not V8-MPRS-DCH")
    if not isinstance(run_id, str) or not run_id.startswith(RUN_ID_PREFIX):
        raise ValueError(f"{label} run_id is not V8-MPRS-DCH")
    if (
        not isinstance(source_locks, Mapping)
        or SOURCE_LOCK_KEY not in source_locks
        or "tpd_clean_v7_dch_exact_source_lock" in source_locks
    ):
        raise ValueError(f"{label} source-lock identity is not V8-MPRS-DCH")
    return value


def _existing_training_contract(directory: Path) -> dict[str, Any]:
    protocol = load_json_mapping(
        directory / "protocol.json",
        "existing V8-MPRS-DCH exact protocol",
    )
    if protocol.get("schema") != ENTRY_SCHEMA:
        if protocol.get("schema") == exact_kernel.ENTRY_SCHEMA:
            raise ValueError(
                "cross-version exact resume from a V7 protocol is forbidden"
            )
        raise ValueError(
            "existing protocol is not a V8-MPRS-DCH exact run"
        )
    identity = _require_v8_run_identity(
        protocol.get("run_identity"),
        label="existing protocol",
    )
    try:
        training = identity["training_contract"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "existing V8-MPRS-DCH protocol has no exact training contract"
        ) from exc
    if not isinstance(training, dict):
        raise ValueError("existing exact training contract is not a mapping")
    required = (
        "initialization_contract",
        "initial_model_state_sha256",
        "initial_rng",
        "selection_policy",
    )
    missing = [name for name in required if name not in training]
    if missing:
        raise ValueError(
            f"existing exact training contract lacks fields: {missing}"
        )
    return copy.deepcopy(training)


def initialization_plan(
    args: argparse.Namespace,
    directory: Path,
    model: nn.Module,
) -> InitializationPlan:
    if args.fresh:
        return InitializationPlan(
            request=exact_runner.InitializationRequest.fresh(),
            contract=exact_runner.fresh_initialization_contract(),
            initial_model_state_sha256=(
                exact_runner.initial_model_state_sha256(model)
            ),
        )
    if args.exact_resume:
        training = _existing_training_contract(directory)
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
    raise RuntimeError(
        "V8-MPRS-DCH exact entry supports only fresh or exact-resume"
    )


def _architecture_manifest(
    variant: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": ARCHITECTURE_MANIFEST_SCHEMA,
        "variant": variant,
        "model": "model.SCTransNet.SCTransNet",
        "block": (
            "model.tpd_clean_v8_mprs_dch."
            "TPDCleanV8MPRSDCHBlock"
        ),
        "candidate_family": metadata["candidate_family"],
        "mainline_contract": metadata["mainline_contract"],
        "semantic_sources": metadata["semantic_sources"],
        "embedding_replacements": metadata["replaced_embeddings"],
        "embedding_topology": {
            "mtc.embeddings_1": {
                "channels": 32,
                "stride": 16,
                "blocks": 4,
                "evidence_nodes": 3,
            },
            "mtc.embeddings_2": {
                "channels": 64,
                "stride": 8,
                "blocks": 3,
                "evidence_nodes": 2,
            },
        },
        "phase_order": metadata["phase_order"],
        "pixel_unshuffle_channel_order": metadata[
            "pixel_unshuffle_channel_order"
        ],
        "saliency_representation": metadata["saliency_representation"],
        "saliency_source_equation": metadata[
            "saliency_source_equation"
        ],
        "saliency_mass_equation": metadata["saliency_mass_equation"],
        "saliency_nonnegative": metadata["saliency_nonnegative"],
        "saliency_projection": metadata["saliency_projection"],
        "saliency_reuse_equation": metadata["saliency_reuse_equation"],
        "phase_tied_projection_formula": metadata[
            "phase_tied_projection_formula"
        ],
        "context_code_formula": metadata["context_code_formula"],
        "context_headroom_formula": metadata["context_headroom_formula"],
        "fusion_equation": metadata["fusion_equation"],
        "zero_scale_first_order_reference": metadata[
            "zero_scale_first_order_reference"
        ],
        "cross_version_exact_resume_supported": False,
        "eps": FORMAL_EPS,
        "formal_amp": FORMAL_AMP,
        "phase_contrast_parameters": metadata[
            "phase_contrast_parameters"
        ],
        "phase_contrast_buffers": metadata["phase_contrast_buffers"],
        "derived_projection_parameters": metadata[
            "derived_projection_parameters"
        ],
        "derived_projection_buffers": metadata[
            "derived_projection_buffers"
        ],
        "shallow_embedding_parameters": metadata[
            "shallow_embedding_parameters"
        ],
        "total_parameters": metadata["total_parameters"],
    }


def _sanitize_exact_builder_metadata(
    raw_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove legacy-layout names from the V8 trajectory identity.

    State-layout compatibility remains an ordinary-entry diagnostic fact.  It
    is intentionally represented here as a policy rather than as a foreign
    candidate name, so it cannot be mistaken for cross-version resume
    authority.
    """

    metadata = copy.deepcopy(dict(raw_metadata))
    legacy = metadata.pop("state_compatible_with", None)
    if legacy not in (None, "tpd_clean_v7_dch"):
        raise ValueError(
            "V8-MPRS-DCH builder declared an unknown legacy layout"
        )
    variant_spec = metadata.get("variant_spec")
    if isinstance(variant_spec, Mapping):
        normalized_spec = copy.deepcopy(dict(variant_spec))
        nested_legacy = normalized_spec.pop(
            "state_compatible_with",
            None,
        )
        if nested_legacy not in (None, "tpd_clean_v7_dch"):
            raise ValueError(
                "V8-MPRS-DCH variant spec declared an unknown "
                "legacy layout"
            )
        normalized_spec["model_state_diagnostic_compatibility"] = (
            "strict_same_layout_model_state_only"
        )
        metadata["variant_spec"] = normalized_spec
    metadata["model_state_diagnostic_compatibility"] = (
        "strict_same_layout_model_state_only"
    )
    metadata["cross_version_exact_resume_supported"] = False
    return metadata


def build_selected_model(
    variant: str,
    seed: int,
    *,
    eps: float = FORMAL_EPS,
) -> tuple[nn.Module, dict[str, Any]]:
    if variant not in SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS:
        raise ValueError(
            f"unsupported V8-MPRS-DCH exact variant: {variant!r}"
        )
    if eps != FORMAL_EPS:
        raise ValueError(
            f"V8-MPRS-DCH exact builder requires eps={FORMAL_EPS}"
        )
    model, raw_metadata = ordinary.build_clean_v8_mprs_dch_model(
        variant,
        seed,
    )
    metadata = _sanitize_exact_builder_metadata(raw_metadata)
    if metadata.get("variant") != variant:
        raise ValueError(
            "V8-MPRS-DCH builder metadata variant mismatch"
        )
    for name, expected_blocks in (
        ("embeddings_1", 4),
        ("embeddings_2", 3),
    ):
        embedding = getattr(model.mtc, name, None)
        blocks = getattr(embedding, "blocks", None)
        if (
            not isinstance(blocks, nn.ModuleList)
            or len(blocks) != expected_blocks
        ):
            raise TypeError(
                f"V8-MPRS-DCH {name} has an invalid block topology"
            )
        for index, block in enumerate(blocks):
            if not isinstance(block, TPDCleanV8MPRSDCHBlock):
                raise TypeError(
                    "V8-MPRS-DCH "
                    f"{name}.blocks[{index}] has the wrong type"
                )
            if block.eps != FORMAL_EPS:
                raise ValueError(
                    "V8-MPRS-DCH "
                    f"{name}.blocks[{index}] eps differs "
                    "from the formal contract"
                )
    metadata["formal_eps"] = FORMAL_EPS
    metadata["formal_amp"] = FORMAL_AMP
    metadata["architecture_manifest"] = _architecture_manifest(
        variant,
        metadata,
    )
    return model, metadata


def resolve_device(args: argparse.Namespace) -> torch.device:
    _validate_formal_args(args)
    return exact_kernel.resolve_device(args)


def environment_contract(device: torch.device) -> dict[str, Any]:
    """Attach a V8-owned physical GPU assignment to the shared record."""

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
        "TPD_V8_MPRS_DCH_PHYSICAL_GPU_INDEX"
    )
    physical_uuid = os.environ.get(
        "TPD_V8_MPRS_DCH_PHYSICAL_GPU_UUID"
    )
    if physical_index not in PHYSICAL_GPU_UUIDS:
        raise RuntimeError(
            "TPD_V8_MPRS_DCH_PHYSICAL_GPU_INDEX must identify "
            "physical GPU 2 or 3"
        )
    expected_uuid = PHYSICAL_GPU_UUIDS[physical_index]
    if physical_uuid != expected_uuid:
        raise RuntimeError(
            "TPD_V8_MPRS_DCH_PHYSICAL_GPU_UUID differs from the "
            f"registered physical GPU {physical_index}"
        )
    if payload.get("device_uuid") != expected_uuid:
        raise RuntimeError(
            "visible cuda:0 UUID differs from the registered physical GPU"
        )
    if payload.get("cuda_visible_devices") != expected_uuid:
        raise RuntimeError(
            "formal V8 CUDA_VISIBLE_DEVICES must use the registered UUID"
        )
    payload.update(
        {
            "physical_gpu_index": int(physical_index),
            "physical_gpu_uuid": expected_uuid,
            "physical_gpu_assignment_source": (
                "verified_v8_worker_environment"
            ),
        }
    )
    return payload


def source_lock_contract(
    training_data_sha256: str,
    exact_source_lock_path: Path,
) -> dict[str, str]:
    exact_source_lock_path = Path(exact_source_lock_path).resolve()
    payload = load_json_mapping(
        exact_source_lock_path,
        "V8-MPRS-DCH exact source lock",
    )
    if payload.get("schema") != EXACT_SOURCE_LOCK_SCHEMA:
        raise ValueError(
            "V8-MPRS-DCH exact source-lock schema mismatch"
        )
    if (
        tuple(payload.get("variants", ()))
        != SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS
    ):
        raise ValueError(
            "V8-MPRS-DCH exact source-lock variant matrix mismatch"
        )
    if payload.get("formal_contract") != formal_contract():
        raise ValueError(
            "V8-MPRS-DCH exact source-lock formal contract mismatch"
        )
    if payload.get("training_data_sha256") != training_data_sha256:
        raise ValueError(
            "training data differs from the V8-MPRS-DCH exact "
            "source lock"
        )
    locked_sources = payload.get("source_sha256")
    if not isinstance(locked_sources, dict):
        raise ValueError(
            "V8-MPRS-DCH exact source lock has no source_sha256 mapping"
        )
    required_relative = {
        str(path.relative_to(REPO_ROOT)) for path in RUNTIME_SOURCE_PATHS
    }
    missing_sources = sorted(required_relative - set(locked_sources))
    if missing_sources:
        raise ValueError(
            "V8-MPRS-DCH exact source lock omits runtime sources: "
            f"{missing_sources}"
        )
    for relative, expected_digest in locked_sources.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError(
                "V8-MPRS-DCH exact source lock has an invalid source path"
            )
        path = (REPO_ROOT / relative).resolve()
        try:
            canonical_relative = str(path.relative_to(REPO_ROOT))
        except ValueError as exc:
            raise ValueError(
                "V8-MPRS-DCH exact source path escapes the repository: "
                f"{relative!r}"
            ) from exc
        if canonical_relative != relative:
            raise ValueError(
                "V8-MPRS-DCH exact source path is not canonical: "
                f"{relative!r}"
            )
        actual_digest = file_sha256(path)
        if expected_digest != actual_digest:
            raise ValueError(
                "V8-MPRS-DCH exact source lock differs for runtime "
                f"source {relative}"
            )
    # The V8-owned lock digest binds the complete per-source mapping above.
    # Do not duplicate dependency path names into the run identity.
    return {
        SOURCE_LOCK_KEY: file_sha256(exact_source_lock_path),
        "training_data": training_data_sha256,
    }


def _clone_kernel_function(
    function: Any,
    bindings: Mapping[str, Any],
) -> Any:
    namespace = dict(function.__globals__)
    namespace.update(bindings)
    cloned = types.FunctionType(
        function.__code__,
        namespace,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    cloned.__kwdefaults__ = copy.deepcopy(function.__kwdefaults__)
    return cloned


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
    manifest = model_metadata.get("architecture_manifest")
    if not isinstance(manifest, Mapping) or not manifest:
        raise ValueError(
            "V8-MPRS-DCH exact builder metadata has no "
            "architecture manifest"
        )
    if manifest.get("schema") != ARCHITECTURE_MANIFEST_SCHEMA:
        raise ValueError(
            "V8-MPRS-DCH architecture manifest schema mismatch"
        )
    if manifest.get("variant") != args.variant:
        raise ValueError(
            "V8-MPRS-DCH architecture manifest variant mismatch"
        )
    if manifest.get("block") != (
        "model.tpd_clean_v8_mprs_dch.TPDCleanV8MPRSDCHBlock"
    ):
        raise ValueError(
            "V8-MPRS-DCH architecture manifest block mismatch"
        )
    adapted = _clone_kernel_function(
        exact_kernel.make_exact_run_spec,
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
    return replace(
        spec,
        run_id=(
            f"{RUN_ID_PREFIX}{args.dataset}:{args.variant}:"
            f"seed-{args.seed}:{args.run_tag}"
        ),
    )


def _require_complete_validation_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [
        name for name in STORED_VALIDATION_METRICS if name not in metrics
    ]
    if missing:
        raise ValueError(
            "V8-MPRS-DCH validation metrics lack stored fields: "
            f"{missing}"
        )
    return {
        name: copy.deepcopy(metrics[name])
        for name in STORED_VALIDATION_METRICS
    }


@dataclass(frozen=True)
class EvaluatorCheckpointAdapter:
    model_metadata: Mapping[str, Any]
    split_hashes: Mapping[str, str]

    def __call__(
        self,
        context: exact_runner.CompatibilityPayloadContext,
    ) -> Mapping[str, Any]:
        identity = _require_v8_run_identity(
            context.run_identity,
            label="checkpoint context",
        )
        exact_payload = context.exact_payload
        validation_metrics = _require_complete_validation_metrics(
            context.metrics
        )
        return {
            "epoch": context.epoch,
            "checkpoint_role": context.role,
            "variant": identity["variant"],
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
            "scheduler": None,
            "validation_metrics": validation_metrics,
            "model_metadata": copy.deepcopy(dict(self.model_metadata)),
            "split_hashes": copy.deepcopy(dict(self.split_hashes)),
            "run_identity": identity,
            "checkpoint_identity": {
                "schema": CHECKPOINT_IDENTITY_SCHEMA,
                "variant": identity["variant"],
                "run_id": identity["run_id"],
                "architecture_id": identity["architecture_id"],
                "builder_manifest_sha256": identity[
                    "builder_manifest_sha256"
                ],
            },
            "selection_source": "internal_validation_only",
            "official_test_accessed": False,
        }


class MPRSDCHExactRunner(exact_kernel.DCHExactRunner):
    """V8 exact runner with a pre-restore cross-version journal guard."""

    def _require_v8_active_journal(self) -> None:
        active = self.journal.load_active()
        if active is None:
            return
        payload, _ = self._load_exact_payload(active.checkpoint_path)
        identity = payload.get("run_identity")
        _require_v8_run_identity(identity, label="active exact journal")
        optimizer = payload.get("optimizer")
        if not isinstance(optimizer, Mapping):
            raise ValueError(
                "active V8 exact journal has no optimizer state"
            )

    def startup(
        self,
        request: exact_runner.InitializationRequest,
    ) -> exact_runner.RunnerSnapshot:
        if not isinstance(request, exact_runner.InitializationRequest):
            return super().startup(request)
        request.validate()
        if (
            exact_resume.InitializationMode(request.mode)
            is exact_resume.InitializationMode.EXACT_RESUME
        ):
            self._require_v8_active_journal()
        return super().startup(request)


# Compatibility alias for callers that used the previous candidate's generic
# class name.  The object itself remains V8-owned.
DCHExactRunner = MPRSDCHExactRunner


def six_output_bce_loss(
    outputs: Any,
    target: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    return exact_kernel.six_output_bce_loss(outputs, target, criterion)


def training_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return exact_kernel.training_arguments(args)


def protocol_payload(
    args: argparse.Namespace,
    *,
    directory: Path,
    model_metadata: Mapping[str, Any],
    normalization: Mapping[str, float],
    run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _require_v8_run_identity(
        run_identity,
        label="protocol",
    )
    adapted = _clone_kernel_function(
        exact_kernel.protocol_payload,
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
    payload["exact_resume_policy"] = {
        "same_version": "v8_to_v8_epoch_boundary_only",
        "cross_version_optimizer_journal": "forbidden",
        "model_state_diagnostic": "strict_load_only",
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
            "V8-MPRS-DCH exact metrics are not a complete "
            "contiguous run"
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
    adapted = _clone_kernel_function(
        exact_kernel.completion_summary,
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
    payload["candidate_family"] = "tpd_clean_v8_mprs_dch"
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
    """Clone the verified loop with V8 bindings, without mutating its module."""

    return _clone_kernel_function(
        exact_kernel.run_training,
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
            "DCHExactRunner": MPRSDCHExactRunner,
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
    "COMPLETION_SUMMARY_SCHEMA",
    "DEFAULT_EXACT_SOURCE_LOCK_PATH",
    "DEFAULT_OUTPUT_ROOT",
    "DCHExactRunner",
    "ENTRY_SCHEMA",
    "EXACT_SOURCE_LOCK_SCHEMA",
    "EvaluatorCheckpointAdapter",
    "FORMAL_AMP",
    "FORMAL_CUBLAS_WORKSPACE_CONFIG",
    "FORMAL_EPOCHS",
    "FORMAL_EPS",
    "FORMAL_EVAL_EVERY",
    "FORMAL_WORKERS",
    "MPRSDCHExactRunner",
    "PHYSICAL_GPU_UUIDS",
    "RUNTIME_SOURCE_PATHS",
    "RUN_ID_PREFIX",
    "SELECTION_METRICS",
    "SOURCE_LOCK_KEY",
    "STORED_VALIDATION_METRICS",
    "build_selected_model",
    "completion_summary",
    "environment_contract",
    "formal_contract",
    "initialization_plan",
    "main",
    "make_exact_run_spec",
    "parse_args",
    "protocol_payload",
    "run_directory",
    "run_training",
    "six_output_bce_loss",
    "source_lock_contract",
]


if __name__ == "__main__":
    main()
