#!/usr/bin/env python3
"""Single-seed formal trainer for the V2 five-node NER candidate.

Only ``tpd_ner_v8_mprs_dch_v2_full_relay_on`` is a trainable V2 candidate.
The required relay-off control is the already existing V1
``tpd_ner_v8_mprs_dch_full_relay_off`` run; this entry never schedules or
re-trains a relay-off model.

The shared SCTransNet numerical loop remains authoritative for the frozen
530/133 split, Adam, the learning-rate schedule, validation metrics, and the
unweighted sum of BCE over exactly six post-sigmoid outputs.  This module owns
the V2 builder and all V2 run/artifact identities.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import train_tpd_ner_v8_mprs_dch as v1_entry  # noqa: E402
from experiments import train_tpd_pilot as base  # noqa: E402
from experiments.tpd_ner_runtime import guarded_training_runtime  # noqa: E402
from experiments.train_tpd_clean_v8_mprs_dch import (  # noqa: E402
    TOTAL_PARAMETERS as V8_PARENT_PARAMETERS,
    build_clean_v8_mprs_dch_model,
)
from model.tpd_clean_v8_mprs_dch import (  # noqa: E402
    PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
)
from model.tpd_ner_v8_mprs_dch import (  # noqa: E402
    DEFAULT_RELAY_INITIALIZATION_SEED,
    DEFAULT_RELAY_WIDTH,
    EVIDENCE_NODE_NAMES,
    RELAY_STAGE_ORDER,
    TPDNERV8MPRSDCHSCTransNet,
)
from model.tpd_ner_v8_mprs_dch_v2 import (  # noqa: E402
    PRODUCTION_V2_RELAY_ON_PARAMETERS,
    PRODUCTION_V2_RELAY_PARAMETERS,
    RELAY_RMS_EPS,
    TPDNERV8MPRSDCHV2SCTransNet,
    V2_MASK_LIMIT,
    V2_SKIP_FACTOR_BOUNDS,
    adapt_v8_mprs_dch_parent_v2,
    v2_relay_parameter_count,
)


ENTRY_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_v2_entry_v1"
SPLIT_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_v2_split_v1"
METRIC_EVENT_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_v2_metric_event_v1"
CHECKPOINT_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_v2_checkpoint_v1"
CHECKPOINT_IDENTITY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v2_checkpoint_identity_v1"
)
SUMMARY_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_v2_summary_v1"
ARCHITECTURE_MANIFEST_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v2_architecture_manifest_v1"
)
CONSTRUCTION_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v2_five_node_rms_centered_v1"
)
CANDIDATE_FAMILY = (
    "tpd_clean_v8_mprs_dch_explicit_five_node_ner_v2_rms_centered"
)

TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON = (
    "tpd_ner_v8_mprs_dch_v2_full_relay_on"
)
V1_RELAY_OFF_REFERENCE = v1_entry.TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF
SUPPORTED_TPD_NER_V8_MPRS_DCH_V2_VARIANTS = (
    TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
)
FORMAL_TPD_NER_V8_MPRS_DCH_V2_VARIANTS = (
    TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
)

DATASET = v1_entry.DATASET
TRAINING_SEED = 42
SPLIT_SEED = 20260722
FORMAL_EPOCHS = 800
FORMAL_BATCH_SIZE = 16
FORMAL_PATCH_SIZE = 256
FORMAL_WORKERS = 0
FORMAL_VAL_FRACTION = 0.20
FORMAL_EVAL_EVERY = 1
FORMAL_BASE_LR = 1e-3
FORMAL_MIN_LR = 1e-5
FORMAL_WARMUP_EPOCHS = 10
FORMAL_THRESHOLD = 0.5
FORMAL_MATCH_RADIUS = 3.0
FORMAL_TINY_AREA = 9
FORMAL_RUN_TAG = "formal800_fp32_seed42_v2"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "experiments/results/tpd_ner_v8_mprs_dch_v2_formal800_seed42_v1"
)
RUN_ID_PREFIX = "tpd-ner-v8-mprs-dch-v2:"
FA_BUDGETS = v1_entry.FA_BUDGETS
STORED_VALIDATION_METRICS = v1_entry.STORED_VALIDATION_METRICS
PHYSICAL_GPU_UUIDS = {
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}

_BASE_CHECKPOINT_PAYLOAD = base.checkpoint_payload
_BASE_WRITE_JSON = base.write_json
_BASE_APPEND_JSONL = base.append_jsonl


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _state_checksum(
    module: nn.Module,
    *,
    excluded_prefixes: Tuple[str, ...] = (),
) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if name.startswith(excluded_prefixes):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        base.json_ready(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ordered_identifier_sha256(identifiers: Sequence[str]) -> str:
    return v1_entry.ordered_identifier_sha256(identifiers)


def variant_spec(candidate_variant: str) -> Dict[str, object]:
    if not isinstance(candidate_variant, str):
        raise TypeError("V2 candidate variant must be a string")
    normalized = candidate_variant.lower()
    if normalized != TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON:
        raise ValueError(
            f"unknown V2 NER variant {candidate_variant!r}; "
            f"choices={SUPPORTED_TPD_NER_V8_MPRS_DCH_V2_VARIANTS}"
        )
    return {
        "candidate_variant": normalized,
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": True,
        "comparison_role": "tpd_plus_ner_v2_rms_centered",
        "required_control": V1_RELAY_OFF_REFERENCE,
        "relay_off_retrained": False,
    }


def build_v1_relay_off_reference(
    seed: int = TRAINING_SEED,
) -> tuple[TPDNERV8MPRSDCHSCTransNet, Dict[str, Any]]:
    """Build the existing V1 relay-off reference without changing its identity."""

    return v1_entry.build_tpd_ner_v8_mprs_dch_model(
        V1_RELAY_OFF_REFERENCE,
        seed,
    )


def build_v2_relay_off_identity_probe(
    seed: int = TRAINING_SEED,
) -> tuple[TPDNERV8MPRSDCHSCTransNet, Dict[str, Any]]:
    """Exercise V2's relay-off adapter for strict identity tests only.

    This helper is deliberately absent from the supported/formal variant
    matrix and therefore cannot be selected by either training entry.
    """

    if type(seed) is not int or seed != TRAINING_SEED:
        raise ValueError(f"V2 relay-off probe requires seed={TRAINING_SEED}")
    parent, parent_metadata = build_clean_v8_mprs_dch_model(
        PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        seed,
    )
    model = adapt_v8_mprs_dch_parent_v2(
        parent,
        variant=PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        relay_enabled=False,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
    )
    return model, {
        "variant": V1_RELAY_OFF_REFERENCE,
        "reference_identity": "v1_relay_off_exact",
        "formal_training_scheduled": False,
        "parent_model_metadata": parent_metadata,
    }


def _architecture_manifest(
    model: TPDNERV8MPRSDCHV2SCTransNet,
) -> Dict[str, Any]:
    manifest = dict(model.architecture_manifest())
    manifest.update(
        {
            "schema": ARCHITECTURE_MANIFEST_SCHEMA,
            "variant": TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
            "candidate_family": CANDIDATE_FAMILY,
            "comparison_role": "tpd_plus_ner_v2_rms_centered",
            "required_control": V1_RELAY_OFF_REFERENCE,
            "relay_off_retrained": False,
            "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            "parent_parameters": V8_PARENT_PARAMETERS,
            "relay_parameters": v2_relay_parameter_count(model),
            "total_parameters": _parameter_count(model),
            "deep_supervision_outputs": 6,
            "training_output_semantics": (
                "six post-sigmoid outputs; BCE per output; unweighted sum"
            ),
        }
    )
    return manifest


def build_tpd_ner_v8_mprs_dch_v2_model(
    candidate_variant: str,
    seed: int,
) -> tuple[TPDNERV8MPRSDCHV2SCTransNet, Dict[str, Any]]:
    """Build the sole formal V2 relay-on candidate."""

    if type(seed) is not int or seed != TRAINING_SEED:
        raise ValueError(
            f"V2 NER supports only model seed={TRAINING_SEED}, got {seed!r}"
        )
    spec = variant_spec(candidate_variant)
    parent, parent_metadata = build_clean_v8_mprs_dch_model(
        PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        seed,
    )
    model = adapt_v8_mprs_dch_parent_v2(
        parent,
        variant=PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        relay_enabled=True,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
    )
    if not isinstance(model, TPDNERV8MPRSDCHV2SCTransNet):
        raise TypeError("V2 builder did not return the V2 relay-on model")
    if model.mode != "train" or model.deepsuper is not True:
        raise RuntimeError("V2 formal model must retain six-output train mode")
    if model.tokenizer_variant != PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT:
        raise RuntimeError("V2 adapter changed the Full parent identity")
    if model.relay_enabled is not True or not hasattr(model, "tpd_ner"):
        raise RuntimeError("V2 formal candidate must register the relay")
    if v2_relay_parameter_count(model) != PRODUCTION_V2_RELAY_PARAMETERS:
        raise RuntimeError("V2 relay parameter count differs")
    if _parameter_count(model) != PRODUCTION_V2_RELAY_ON_PARAMETERS:
        raise RuntimeError("V2 total parameter count differs")

    manifest = _architecture_manifest(model)
    architecture_id = _canonical_sha256(manifest)
    metadata: Dict[str, Any] = {
        "variant": spec["candidate_variant"],
        "candidate_family": CANDIDATE_FAMILY,
        "construction_schema": CONSTRUCTION_SCHEMA,
        "comparison_role": spec["comparison_role"],
        "required_control": V1_RELAY_OFF_REFERENCE,
        "relay_off_retrained": False,
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "parent_candidate_family": parent_metadata["candidate_family"],
        "parent_model_metadata": parent_metadata,
        "relay_enabled": True,
        "relay_version": "v2_rms_centered_arctangent",
        "relay_width": DEFAULT_RELAY_WIDTH,
        "relay_topology": "q4->q3->q2",
        "relay_parameters": PRODUCTION_V2_RELAY_PARAMETERS,
        "relay_initialization_seed": DEFAULT_RELAY_INITIALIZATION_SEED,
        "relay_initialization_sha256": _state_checksum(model.tpd_ner),
        "relay_state_prefix": "tpd_ner.",
        "relay_rms_scope": "per_sample_full_tensor",
        "relay_rms_eps": RELAY_RMS_EPS,
        "source_projection_rms_normalized": True,
        "fusion_relu_output_rms_normalized": True,
        "gate_bias": False,
        "gate_spatial_centering": "per_sample_mean_hw",
        "mask_mapping": "atan(pi*z)/pi",
        "mask_bounds": (-V2_MASK_LIMIT, V2_MASK_LIMIT),
        "skip_factor_bounds": V2_SKIP_FACTOR_BOUNDS,
        "evidence_nodes": tuple(EVIDENCE_NODE_NAMES),
        "evidence_node_count": 5,
        "evidence_layout": (3, 2),
        "relay_stage_order": tuple(RELAY_STAGE_ORDER),
        "tensor_handoff": "forward_local_explicit",
        "mainline_contract": "Keep-Context-Saliency",
        "semantic_sources": ("Keep", "Context", "Saliency"),
        "semantic_source_count": 3,
        "fourth_parallel_branch_added": False,
        "training_seed": TRAINING_SEED,
        "split_seed_is_model_input": False,
        "initialization_mode": "paired_fresh_full_v8_parent",
        "warm_start_applied": False,
        "zero_gate_reference": "v1_relay_off_exact",
        "six_output_training_semantics": True,
        "loss": "sum of BCE over six post-sigmoid outputs",
        "total_parameters": PRODUCTION_V2_RELAY_ON_PARAMETERS,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "common_parameters": (
            PRODUCTION_V2_RELAY_ON_PARAMETERS - PRODUCTION_V2_RELAY_PARAMETERS
        ),
        "common_initialization_sha256": _state_checksum(
            model,
            excluded_prefixes=("tpd_ner.",),
        ),
        "full_initialization_sha256": _state_checksum(model),
        "architecture_manifest": manifest,
        "architecture_id": architecture_id,
    }
    return model, metadata


def formal_training_contract() -> Dict[str, Any]:
    return {
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "epochs": FORMAL_EPOCHS,
        "batch_size": FORMAL_BATCH_SIZE,
        "patch_size": FORMAL_PATCH_SIZE,
        "workers": FORMAL_WORKERS,
        "val_fraction": FORMAL_VAL_FRACTION,
        "eval_every": FORMAL_EVAL_EVERY,
        "base_lr": FORMAL_BASE_LR,
        "min_lr": FORMAL_MIN_LR,
        "warmup_epochs": FORMAL_WARMUP_EPOCHS,
        "amp": False,
        "precision": "FP32",
        "optimizer": "Adam",
        "loss": "sum of BCE over six post-sigmoid outputs",
        "threshold": FORMAL_THRESHOLD,
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "fa_budgets": list(FA_BUDGETS),
        "official_test_accessed": False,
        "formal_variants": list(FORMAL_TPD_NER_V8_MPRS_DCH_V2_VARIANTS),
        "required_control": V1_RELAY_OFF_REFERENCE,
        "relay_off_retrained": False,
        "multi_seed_scheduled": False,
        "logical_device": "cuda:0",
        "physical_gpu_choices": [2, 3],
        "physical_gpu_model": "NVIDIA GeForce RTX 5090",
        "physical_gpu_index_environment": (
            "TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_INDEX"
        ),
        "physical_gpu_uuid_environment": (
            "TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_UUID"
        ),
        "cuda_visible_devices_semantics": "single_registered_gpu_uuid",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formal single-seed V2 five-node NER trainer"
    )
    parser.add_argument(
        "--variant",
        choices=FORMAL_TPD_NER_V8_MPRS_DCH_V2_VARIANTS,
        required=True,
    )
    parser.add_argument("--dataset", choices=(DATASET,), default=DATASET)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-tag", choices=(FORMAL_RUN_TAG,), default=FORMAL_RUN_TAG
    )
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument(
        "--epochs", type=int, choices=(FORMAL_EPOCHS,), default=FORMAL_EPOCHS
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(FORMAL_BATCH_SIZE,),
        default=FORMAL_BATCH_SIZE,
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        choices=(FORMAL_PATCH_SIZE,),
        default=FORMAL_PATCH_SIZE,
    )
    parser.add_argument(
        "--workers", type=int, choices=(FORMAL_WORKERS,), default=FORMAL_WORKERS
    )
    parser.add_argument(
        "--seed", type=int, choices=(TRAINING_SEED,), default=TRAINING_SEED
    )
    parser.add_argument(
        "--split-seed", type=int, choices=(SPLIT_SEED,), default=SPLIT_SEED
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        choices=(FORMAL_VAL_FRACTION,),
        default=FORMAL_VAL_FRACTION,
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        choices=(FORMAL_EVAL_EVERY,),
        default=FORMAL_EVAL_EVERY,
    )
    parser.add_argument(
        "--base-lr", type=float, choices=(FORMAL_BASE_LR,), default=FORMAL_BASE_LR
    )
    parser.add_argument(
        "--min-lr", type=float, choices=(FORMAL_MIN_LR,), default=FORMAL_MIN_LR
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        choices=(FORMAL_WARMUP_EPOCHS,),
        default=FORMAL_WARMUP_EPOCHS,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        choices=(FORMAL_THRESHOLD,),
        default=FORMAL_THRESHOLD,
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        choices=(FORMAL_MATCH_RADIUS,),
        default=FORMAL_MATCH_RADIUS,
    )
    parser.add_argument(
        "--tiny-area",
        type=int,
        choices=(FORMAL_TINY_AREA,),
        default=FORMAL_TINY_AREA,
    )
    args = parser.parse_args(None if argv is None else list(argv))
    args.amp = False
    args.max_train_images = None
    args.max_val_images = None
    validate_formal_args(args)
    return args


def validate_formal_args(args: argparse.Namespace) -> None:
    expected = {
        "dataset": DATASET,
        "variant": TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "epochs": FORMAL_EPOCHS,
        "batch_size": FORMAL_BATCH_SIZE,
        "patch_size": FORMAL_PATCH_SIZE,
        "workers": FORMAL_WORKERS,
        "val_fraction": FORMAL_VAL_FRACTION,
        "eval_every": FORMAL_EVAL_EVERY,
        "base_lr": FORMAL_BASE_LR,
        "min_lr": FORMAL_MIN_LR,
        "warmup_epochs": FORMAL_WARMUP_EPOCHS,
        "threshold": FORMAL_THRESHOLD,
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "amp": False,
        "max_train_images": None,
        "max_val_images": None,
        "run_tag": FORMAL_RUN_TAG,
        "device": "cuda:0",
    }
    observed = {name: getattr(args, name, None) for name in expected}
    if observed != expected:
        raise ValueError(
            f"formal V2 arguments differ: expected={expected}, observed={observed}"
        )


def _normalized_gpu_uuid(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("GPU UUID is empty")
    return text if text.startswith("GPU-") else f"GPU-{text}"


def validate_physical_gpu_runtime(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Bind ordinary formal execution to one registered physical RTX 5090."""

    validate_formal_args(args)
    physical_index = os.environ.get(
        "TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_INDEX"
    )
    physical_uuid = os.environ.get(
        "TPD_NER_V8_MPRS_DCH_V2_PHYSICAL_GPU_UUID"
    )
    if physical_index not in PHYSICAL_GPU_UUIDS:
        raise RuntimeError("V2 physical GPU index must identify GPU 2 or 3")
    expected_uuid = PHYSICAL_GPU_UUIDS[physical_index]
    if physical_uuid != expected_uuid:
        raise RuntimeError(
            f"V2 physical GPU UUID differs for GPU {physical_index}"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain the assigned GPU UUID"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V2 ordinary training requires one visible CUDA GPU")
    device_name = torch.cuda.get_device_name(0)
    if device_name != "NVIDIA GeForce RTX 5090":
        raise RuntimeError(f"unexpected V2 CUDA device model: {device_name!r}")
    properties = torch.cuda.get_device_properties(0)
    runtime_uuid = getattr(properties, "uuid", None)
    if runtime_uuid is None:
        raise RuntimeError("CUDA device properties do not expose a UUID")
    normalized_uuid = _normalized_gpu_uuid(runtime_uuid)
    if normalized_uuid != expected_uuid:
        raise RuntimeError("visible cuda:0 UUID differs from V2 assignment")
    return {
        "logical_device": "cuda:0",
        "visible_cuda_device_count": 1,
        "physical_gpu_index": int(physical_index),
        "physical_gpu_uuid": expected_uuid,
        "device_name": device_name,
        "assignment_source": "verified_v2_ordinary_environment",
    }


def formal_run_id(args: argparse.Namespace) -> str:
    validate_formal_args(args)
    return (
        f"{RUN_ID_PREFIX}{args.dataset}:{args.variant}:"
        f"seed-{args.seed}:split-{args.split_seed}:{args.run_tag}"
    )


def run_identity(args: argparse.Namespace) -> Dict[str, Any]:
    validate_formal_args(args)
    physical_identity = getattr(args, "physical_gpu_identity", None)
    if physical_identity is not None and not isinstance(
        physical_identity,
        Mapping,
    ):
        raise TypeError("V2 physical GPU identity must be a mapping")
    return {
        "schema": ENTRY_SCHEMA,
        "run_id": formal_run_id(args),
        "candidate_family": CANDIDATE_FAMILY,
        "dataset": args.dataset,
        "variant": args.variant,
        "comparison_role": "tpd_plus_ner_v2_rms_centered",
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": True,
        "relay_version": "v2_rms_centered_arctangent",
        "required_control": V1_RELAY_OFF_REFERENCE,
        "relay_off_retrained": False,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "run_tag": args.run_tag,
        "logical_device": args.device,
        "physical_gpu_identity": (
            copy.deepcopy(dict(physical_identity))
            if physical_identity is not None
            else None
        ),
    }


def annotate_json_artifact(
    path: Path,
    payload: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    ready = dict(payload)
    identity = run_identity(args)
    name = Path(path).name
    if name == "split.json":
        if (
            ready.get("full_official_train_count") != 663
            or ready.get("used_train_count") != 530
            or ready.get("used_val_count") != 133
        ):
            raise ValueError("formal V2 split is not the frozen 663 -> 530/133 split")
        train_ids = ready.get("used_train_ids")
        val_ids = ready.get("used_val_ids")
        if not isinstance(train_ids, list) or not isinstance(val_ids, list):
            raise TypeError("formal V2 split lacks ordered ID lists")
        ready.update(
            {
                "schema": SPLIT_SCHEMA,
                "run_identity": identity,
                "ordered_used_train_sha256": ordered_identifier_sha256(train_ids),
                "ordered_used_val_sha256": ordered_identifier_sha256(val_ids),
            }
        )
    elif name == "protocol.json":
        model_metadata = ready.get("model")
        if (
            not isinstance(model_metadata, Mapping)
            or model_metadata.get("variant") != args.variant
        ):
            raise ValueError("formal V2 protocol model identity differs")
        ready.update(
            {
                "schema": ENTRY_SCHEMA,
                "run_identity": identity,
                "training_contract": formal_training_contract(),
                "stored_validation_metrics": list(STORED_VALIDATION_METRICS),
                "comparison_design": {
                    "primary": [
                        "baseline_sctransnet",
                        V1_RELAY_OFF_REFERENCE,
                        TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
                    ],
                    "required_control": V1_RELAY_OFF_REFERENCE,
                    "relay_off_source": "existing_v1_formal_run",
                    "relay_off_retrained": False,
                },
            }
        )
    elif name == "summary.json":
        ready.update(
            {
                "schema": SUMMARY_SCHEMA,
                "run_identity": identity,
                "training_contract": formal_training_contract(),
                "stored_validation_metrics": list(STORED_VALIDATION_METRICS),
            }
        )
    return ready


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    variant: str,
    args: argparse.Namespace,
    metrics: Dict[str, Any],
    model_metadata: Dict[str, Any],
    split_hashes: Dict[str, str],
) -> Dict[str, Any]:
    validate_formal_args(args)
    if variant != args.variant or model_metadata.get("variant") != variant:
        raise ValueError("V2 checkpoint candidate identity differs")
    payload = _BASE_CHECKPOINT_PAYLOAD(
        model,
        optimizer,
        scaler,
        epoch,
        variant,
        args,
        metrics,
        model_metadata,
        split_hashes,
    )
    architecture_id = model_metadata.get("architecture_id")
    if not isinstance(architecture_id, str) or len(architecture_id) != 64:
        raise ValueError("V2 checkpoint metadata lacks architecture_id")
    payload.update(
        {
            "schema": CHECKPOINT_SCHEMA,
            "run_identity": run_identity(args),
            "checkpoint_identity": {
                "schema": CHECKPOINT_IDENTITY_SCHEMA,
                "run_id": formal_run_id(args),
                "variant": variant,
                "relay_version": "v2_rms_centered_arctangent",
                "relay_enabled": True,
                "architecture_id": architecture_id,
            },
            "training_contract": formal_training_contract(),
            "stored_validation_metrics": list(STORED_VALIDATION_METRICS),
            "six_output_training_semantics": True,
        }
    )
    return payload


def require_six_output_loss_inputs(
    outputs: Any,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    return v1_entry.require_six_output_loss_inputs(outputs, target)


@contextlib.contextmanager
def _formal_runtime_bindings() -> Iterator[None]:
    previous = {
        "SUPPORTED_VARIANTS": base.SUPPORTED_VARIANTS,
        "build_model": base.build_model,
        "parse_args": base.parse_args,
        "checkpoint_payload": base.checkpoint_payload,
        "write_json": base.write_json,
        "append_jsonl": base.append_jsonl,
    }
    active: Dict[str, argparse.Namespace] = {}

    def bound_parse_args() -> argparse.Namespace:
        args = parse_args()
        args.physical_gpu_identity = validate_physical_gpu_runtime(args)
        active["args"] = args
        return args

    def bound_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        args = active.get("args")
        if args is None:
            raise RuntimeError("formal V2 arguments are not active")
        _BASE_WRITE_JSON(path, annotate_json_artifact(path, payload, args))

    def bound_append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
        args = active.get("args")
        if args is None:
            raise RuntimeError("formal V2 arguments are not active")
        if payload.get("variant") != args.variant:
            raise ValueError("V2 metric event candidate differs")
        ready = {
            **dict(payload),
            "schema": METRIC_EVENT_SCHEMA,
            "run_id": formal_run_id(args),
            "seed": args.seed,
            "split_seed": args.split_seed,
        }
        _BASE_APPEND_JSONL(path, ready)

    base.SUPPORTED_VARIANTS = SUPPORTED_TPD_NER_V8_MPRS_DCH_V2_VARIANTS
    base.build_model = build_tpd_ner_v8_mprs_dch_v2_model
    base.parse_args = bound_parse_args
    base.checkpoint_payload = checkpoint_payload
    base.write_json = bound_write_json
    base.append_jsonl = bound_append_jsonl
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


@contextlib.contextmanager
def _six_output_training_runtime() -> Iterator[None]:
    previous = base.deep_supervision_loss

    def checked_loss(
        outputs: Any,
        target: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        ready = require_six_output_loss_inputs(outputs, target)
        return previous(ready, target, criterion)

    base.deep_supervision_loss = checked_loss
    try:
        yield
    finally:
        base.deep_supervision_loss = previous


@contextlib.contextmanager
def _checkpoint_save_identity_runtime() -> Iterator[None]:
    previous = torch.save

    def identity_save(
        payload: Any,
        destination: Any,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if isinstance(payload, Mapping) and payload.get("schema") == CHECKPOINT_SCHEMA:
            ready = copy.copy(dict(payload))
            checkpoint_identity = dict(ready["checkpoint_identity"])
            checkpoint_identity["checkpoint_role"] = ready.get("checkpoint_role")
            checkpoint_identity["checkpoint_filename"] = (
                Path(destination).name
                if isinstance(destination, (str, Path))
                else None
            )
            ready["checkpoint_identity"] = checkpoint_identity
            payload = ready
        previous(payload, destination, *args, **kwargs)

    torch.save = identity_save
    try:
        yield
    finally:
        torch.save = previous


def main() -> None:
    """Run the shared numerical loop under V2-owned identities."""

    with _formal_runtime_bindings():
        with guarded_training_runtime():
            with _six_output_training_runtime():
                with _checkpoint_save_identity_runtime():
                    base.main()


__all__ = [
    "ARCHITECTURE_MANIFEST_SCHEMA",
    "CANDIDATE_FAMILY",
    "CHECKPOINT_IDENTITY_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CONSTRUCTION_SCHEMA",
    "DATASET",
    "DEFAULT_OUTPUT_ROOT",
    "ENTRY_SCHEMA",
    "FA_BUDGETS",
    "FORMAL_BATCH_SIZE",
    "FORMAL_EPOCHS",
    "FORMAL_RUN_TAG",
    "FORMAL_TPD_NER_V8_MPRS_DCH_V2_VARIANTS",
    "METRIC_EVENT_SCHEMA",
    "PHYSICAL_GPU_UUIDS",
    "RUN_ID_PREFIX",
    "SPLIT_SCHEMA",
    "SPLIT_SEED",
    "STORED_VALIDATION_METRICS",
    "SUMMARY_SCHEMA",
    "SUPPORTED_TPD_NER_V8_MPRS_DCH_V2_VARIANTS",
    "TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON",
    "TRAINING_SEED",
    "V1_RELAY_OFF_REFERENCE",
    "annotate_json_artifact",
    "build_tpd_ner_v8_mprs_dch_v2_model",
    "build_v1_relay_off_reference",
    "build_v2_relay_off_identity_probe",
    "checkpoint_payload",
    "formal_run_id",
    "formal_training_contract",
    "main",
    "ordered_identifier_sha256",
    "parse_args",
    "require_six_output_loss_inputs",
    "run_identity",
    "validate_formal_args",
    "validate_physical_gpu_runtime",
    "variant_spec",
]


if __name__ == "__main__":
    main()
