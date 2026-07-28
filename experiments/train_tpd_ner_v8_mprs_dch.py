#!/usr/bin/env python3
"""Single-seed formal trainer for V8-MPRS-DCH with five-node NER.

This is an import-safe identity and runtime adapter over
``experiments.train_tpd_pilot``.  The numerical training loop, 530/133 split,
Adam optimizer, learning-rate schedule, checkpoint selection, and six-output
post-sigmoid BCE objective remain owned by the shared runner.  This module
adds only the frozen V8 Full relay-off/on builders, a single-seed CLI, and
NER-owned artifact identities.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    PRODUCTION_RELAY_ON_PARAMETERS,
    PRODUCTION_RELAY_PARAMETERS,
    RELAY_STAGE_ORDER,
    TPDNERV8MPRSDCHSCTransNet,
    adapt_v8_mprs_dch_parent,
    relay_parameter_count,
)


ENTRY_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_entry_v1"
SPLIT_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_split_v1"
METRIC_EVENT_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_metric_event_v1"
CHECKPOINT_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_checkpoint_v1"
CHECKPOINT_IDENTITY_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_checkpoint_identity_v1"
)
SUMMARY_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_summary_v1"
ARCHITECTURE_MANIFEST_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_architecture_manifest_v1"
)
CONSTRUCTION_SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_five_node_v1"
CANDIDATE_FAMILY = "tpd_clean_v8_mprs_dch_explicit_five_node_ner"

TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF = (
    "tpd_ner_v8_mprs_dch_full_relay_off"
)
TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON = (
    "tpd_ner_v8_mprs_dch_full_relay_on"
)
SUPPORTED_TPD_NER_V8_MPRS_DCH_VARIANTS = (
    TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
    TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
)
FORMAL_TPD_NER_V8_MPRS_DCH_VARIANTS = (
    TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
    TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
)

DATASET = "NUDT-SIRST"
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
FORMAL_RUN_TAG = "formal800_fp32_seed42_v1"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "experiments/results/tpd_ner_v8_mprs_dch_formal800_seed42_v1"
)
RUN_ID_PREFIX = "tpd-ner-v8-mprs-dch:"
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)

STORED_VALIDATION_METRICS = (
    "val_loss",
    "miou",
    "niou",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pd",
    "tiny_pd",
    "fa",
    "false_objects_per_image",
    "target_count",
    "matched_target_count",
    "tiny_target_count",
    "matched_tiny_target_count",
    "predicted_object_count",
    "unmatched_predicted_object_count",
    "valid_pixel_count",
)

_VARIANT_SPECS: Mapping[str, Mapping[str, object]] = {
    TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF: {
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": False,
        "comparison_role": "tpd_only_v8_full_relay_off_control",
        "relay_pair": "tpd_ner_v8_mprs_dch_full",
    },
    TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON: {
        "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
        "relay_enabled": True,
        "comparison_role": "tpd_plus_ner_v8_full",
        "relay_pair": "tpd_ner_v8_mprs_dch_full",
    },
}

# Capture shared implementations once.  Runtime binding below restores every
# global even when argument parsing or training raises.
_BASE_SUPPORTED_VARIANTS = base.SUPPORTED_VARIANTS
_BASE_BUILD_MODEL = base.build_model
_BASE_PARSE_ARGS = base.parse_args
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
    return hashlib.sha256(
        "\n".join(str(identifier) for identifier in identifiers).encode("utf-8")
    ).hexdigest()


def variant_spec(candidate_variant: str) -> Dict[str, object]:
    candidate_variant = candidate_variant.lower()
    if candidate_variant not in _VARIANT_SPECS:
        raise ValueError(
            f"unknown V8-MPRS-DCH NER variant {candidate_variant!r}; "
            f"choices={SUPPORTED_TPD_NER_V8_MPRS_DCH_VARIANTS}"
        )
    return dict(_VARIANT_SPECS[candidate_variant])


def _architecture_manifest(
    model: TPDNERV8MPRSDCHSCTransNet,
    candidate_variant: str,
    comparison_role: str,
) -> Dict[str, Any]:
    manifest = dict(model.architecture_manifest())
    manifest.update(
        {
            "schema": ARCHITECTURE_MANIFEST_SCHEMA,
            "variant": candidate_variant,
            "candidate_family": CANDIDATE_FAMILY,
            "comparison_role": comparison_role,
            "parent_variant": PRIMARY_CLEAN_V8_MPRS_DCH_VARIANT,
            "parent_parameters": V8_PARENT_PARAMETERS,
            "relay_parameters": relay_parameter_count(model),
            "total_parameters": _parameter_count(model),
            "deep_supervision_outputs": 6,
            "training_output_semantics": (
                "six post-sigmoid outputs; BCE per output; unweighted sum"
            ),
        }
    )
    return manifest


def build_tpd_ner_v8_mprs_dch_model(
    candidate_variant: str,
    seed: int,
) -> Tuple[TPDNERV8MPRSDCHSCTransNet, Dict[str, Any]]:
    """Build one paired V8 Full relay-off/on candidate.

    The model seed and the relay-local initialization seed are both frozen to
    42.  ``split_seed`` is deliberately absent from construction and is
    enforced only by the CLI/data contract.
    """

    if type(seed) is not int or seed != TRAINING_SEED:
        raise ValueError(
            f"V8-MPRS-DCH NER supports only model seed={TRAINING_SEED}, got {seed!r}"
        )
    candidate_variant = candidate_variant.lower()
    spec = variant_spec(candidate_variant)
    parent_variant = str(spec["parent_variant"])
    relay_enabled = bool(spec["relay_enabled"])

    parent, parent_metadata = build_clean_v8_mprs_dch_model(
        parent_variant,
        seed,
    )
    model = adapt_v8_mprs_dch_parent(
        parent,
        variant=parent_variant,
        relay_enabled=relay_enabled,
        relay_width=DEFAULT_RELAY_WIDTH,
        relay_initialization_seed=DEFAULT_RELAY_INITIALIZATION_SEED,
    )
    if model.mode != "train" or model.deepsuper is not True:
        raise RuntimeError("NER formal model must retain six-output train mode")
    if model.tokenizer_variant != parent_variant:
        raise RuntimeError("NER adapter changed the V8 parent variant identity")
    if model.relay_enabled is not relay_enabled:
        raise RuntimeError("NER adapter relay identity differs from candidate")

    relay_parameters = relay_parameter_count(model)
    total_parameters = _parameter_count(model)
    expected_relay = PRODUCTION_RELAY_PARAMETERS if relay_enabled else 0
    expected_total = (
        PRODUCTION_RELAY_ON_PARAMETERS if relay_enabled else V8_PARENT_PARAMETERS
    )
    if relay_parameters != expected_relay:
        raise RuntimeError(
            f"NER relay parameter mismatch: {relay_parameters} != {expected_relay}"
        )
    if total_parameters != expected_total:
        raise RuntimeError(
            f"NER total parameter mismatch: {total_parameters} != {expected_total}"
        )
    if not relay_enabled and hasattr(model, "tpd_ner"):
        raise RuntimeError("relay-off candidate unexpectedly registers tpd_ner")

    manifest = _architecture_manifest(
        model,
        candidate_variant,
        str(spec["comparison_role"]),
    )
    architecture_id = _canonical_sha256(manifest)
    metadata: Dict[str, Any] = {
        "variant": candidate_variant,
        "candidate_family": CANDIDATE_FAMILY,
        "construction_schema": CONSTRUCTION_SCHEMA,
        "comparison_role": spec["comparison_role"],
        "parent_variant": parent_variant,
        "parent_candidate_family": parent_metadata["candidate_family"],
        "parent_model_metadata": parent_metadata,
        "relay_pair": spec["relay_pair"],
        "relay_enabled": relay_enabled,
        "relay_width": DEFAULT_RELAY_WIDTH,
        "relay_topology": "q4->q3->q2",
        "relay_parameters": relay_parameters,
        "relay_initialization_seed": DEFAULT_RELAY_INITIALIZATION_SEED,
        "relay_initialization_sha256": (
            _state_checksum(model.tpd_ner) if relay_enabled else None
        ),
        "relay_state_prefix": "tpd_ner.",
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
        "relay_seed": DEFAULT_RELAY_INITIALIZATION_SEED,
        "split_seed_is_model_input": False,
        "initialization_mode": "paired_fresh_full_v8_parent",
        "warm_start_applied": False,
        "zero_gate_reference": "paired_relay_off_exact",
        "six_output_training_semantics": True,
        "loss": "sum of BCE over six post-sigmoid outputs",
        "total_parameters": total_parameters,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "common_parameters": total_parameters - relay_parameters,
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
        "amp": False,
        "optimizer": "Adam",
        "loss": "sum of BCE over six post-sigmoid outputs",
        "threshold": FORMAL_THRESHOLD,
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "fa_budgets": list(FA_BUDGETS),
        "official_test_accessed": False,
        "formal_variants": list(FORMAL_TPD_NER_V8_MPRS_DCH_VARIANTS),
        "multi_seed_scheduled": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Formal single-seed V8-MPRS-DCH five-node NER trainer"
    )
    parser.add_argument(
        "--variant",
        choices=FORMAL_TPD_NER_V8_MPRS_DCH_VARIANTS,
        required=True,
    )
    parser.add_argument("--dataset", choices=(DATASET,), default=DATASET)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPO_ROOT / "datasets",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--run-tag",
        choices=(FORMAL_RUN_TAG,),
        default=FORMAL_RUN_TAG,
    )
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
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
        "--workers",
        type=int,
        choices=(FORMAL_WORKERS,),
        default=FORMAL_WORKERS,
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
        "--base-lr",
        type=float,
        choices=(FORMAL_BASE_LR,),
        default=FORMAL_BASE_LR,
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        choices=(FORMAL_MIN_LR,),
        default=FORMAL_MIN_LR,
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
    }
    observed = {name: getattr(args, name, None) for name in expected}
    if observed != expected:
        raise ValueError(
            f"formal V8-MPRS-DCH NER arguments differ: "
            f"expected={expected}, observed={observed}"
        )
    if args.variant not in FORMAL_TPD_NER_V8_MPRS_DCH_VARIANTS:
        raise ValueError("candidate is not in the two-run Full relay matrix")
    if args.run_tag != FORMAL_RUN_TAG:
        raise ValueError("formal NER run tag differs")


def formal_run_id(args: argparse.Namespace) -> str:
    validate_formal_args(args)
    return (
        f"{RUN_ID_PREFIX}{args.dataset}:{args.variant}:"
        f"seed-{args.seed}:split-{args.split_seed}:{args.run_tag}"
    )


def run_identity(args: argparse.Namespace) -> Dict[str, Any]:
    spec = variant_spec(args.variant)
    return {
        "schema": ENTRY_SCHEMA,
        "run_id": formal_run_id(args),
        "candidate_family": CANDIDATE_FAMILY,
        "dataset": args.dataset,
        "variant": args.variant,
        "comparison_role": spec["comparison_role"],
        "parent_variant": spec["parent_variant"],
        "relay_enabled": spec["relay_enabled"],
        "seed": args.seed,
        "split_seed": args.split_seed,
        "run_tag": args.run_tag,
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
            raise ValueError("formal NER split is not the frozen 663 -> 530/133 split")
        train_ids = ready.get("used_train_ids")
        val_ids = ready.get("used_val_ids")
        if not isinstance(train_ids, list) or not isinstance(val_ids, list):
            raise TypeError("formal NER split lacks ordered ID lists")
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
            raise ValueError("formal NER protocol model identity differs")
        ready.update(
            {
                "schema": ENTRY_SCHEMA,
                "run_identity": identity,
                "training_contract": formal_training_contract(),
                "stored_validation_metrics": list(STORED_VALIDATION_METRICS),
                "comparison_design": {
                    "primary": [
                        "baseline_sctransnet",
                        TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF,
                        TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON,
                    ],
                    "historical_reference_not_formal_column": (
                        "tpd_clean_v8_mprs_dch_full"
                    ),
                    "required_control": (
                        TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF
                    ),
                    "required_ablation": (
                        TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF
                    ),
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
        raise ValueError("checkpoint candidate identity differs from formal run")
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
        raise ValueError("checkpoint model metadata lacks architecture_id")
    payload.update(
        {
            "schema": CHECKPOINT_SCHEMA,
            "run_identity": run_identity(args),
            "checkpoint_identity": {
                "schema": CHECKPOINT_IDENTITY_SCHEMA,
                "run_id": formal_run_id(args),
                "variant": variant,
                "comparison_role": model_metadata["comparison_role"],
                "relay_enabled": model_metadata["relay_enabled"],
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
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        raise RuntimeError(
            "V8-MPRS-DCH NER training requires exactly six deep-supervision outputs"
        )
    tensors = tuple(outputs)
    for index, output in enumerate(tensors):
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"deep-supervision output {index} is not a Tensor")
        if output.shape != target.shape:
            raise ValueError(
                f"deep-supervision output {index} shape {tuple(output.shape)} "
                f"differs from target {tuple(target.shape)}"
            )
    return tensors


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
        active["args"] = args
        return args

    def bound_write_json(path: Path, payload: Mapping[str, Any]) -> None:
        args = active.get("args")
        if args is None:
            raise RuntimeError("formal NER arguments are not active")
        _BASE_WRITE_JSON(path, annotate_json_artifact(path, payload, args))

    def bound_append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
        args = active.get("args")
        if args is None:
            raise RuntimeError("formal NER arguments are not active")
        if payload.get("variant") != args.variant:
            raise ValueError("metric event candidate differs from active run")
        ready = {
            **dict(payload),
            "schema": METRIC_EVENT_SCHEMA,
            "run_id": formal_run_id(args),
            "seed": args.seed,
            "split_seed": args.split_seed,
        }
        _BASE_APPEND_JSONL(path, ready)

    base.SUPPORTED_VARIANTS = SUPPORTED_TPD_NER_V8_MPRS_DCH_VARIANTS
    base.build_model = build_tpd_ner_v8_mprs_dch_model
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
            ready = dict(payload)
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
    """Run the shared numerical loop under the frozen NER identity adapters."""

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
    "ENTRY_SCHEMA",
    "FA_BUDGETS",
    "FORMAL_EPOCHS",
    "FORMAL_TPD_NER_V8_MPRS_DCH_VARIANTS",
    "METRIC_EVENT_SCHEMA",
    "SPLIT_SCHEMA",
    "SPLIT_SEED",
    "STORED_VALIDATION_METRICS",
    "SUMMARY_SCHEMA",
    "SUPPORTED_TPD_NER_V8_MPRS_DCH_VARIANTS",
    "TPD_NER_V8_MPRS_DCH_FULL_RELAY_OFF",
    "TPD_NER_V8_MPRS_DCH_FULL_RELAY_ON",
    "TRAINING_SEED",
    "annotate_json_artifact",
    "build_tpd_ner_v8_mprs_dch_model",
    "checkpoint_payload",
    "formal_run_id",
    "formal_training_contract",
    "main",
    "ordered_identifier_sha256",
    "parse_args",
    "require_six_output_loss_inputs",
    "run_identity",
    "validate_formal_args",
    "variant_spec",
]


if __name__ == "__main__":
    main()
