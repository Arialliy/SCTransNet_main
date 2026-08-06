#!/usr/bin/env python3
"""Independent seed-42 scratch trainer for NER-L4-TPR on three datasets.

The completed three-dataset TSS-off protocol remains the only source for data
splits, optimization, metrics, evaluation cadence, and checkpoint selection.
This entry changes only the model graph and run identity:

* three independent 1,000-epoch seed-42 runs use ``img_idx`` protocol data;
* evaluation starts at epoch 10 and repeats every 10 epochs;
* only ``best_miou`` and ``best_pd`` are selected checkpoints;
* Adam starts fresh and TSS has exact weight zero;
* no Final/GCSF checkpoint may initialize the model;
* resume is restricted to this exact 569-key recipe and run.

Formal execution also requires an explicitly supplied decision JSON produced
by the later screening stage.  There is intentionally no default decision
path and no import-time dependency on a decision artifact that does not yet
exist.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import four_dataset_models_seed42_v1 as paired_registry  # noqa: E402
from experiments import train_four_dataset_original_final_seed42_exact_v1 as engine  # noqa: E402
from experiments import train_three_dataset_seed42_global_tss_v2 as positive  # noqa: E402
from experiments import train_three_dataset_tss_off_seed42_v1 as base  # noqa: E402
from model import (  # noqa: E402
    tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_l4_tpr as l4_tpr,
)


SCHEMA = "sctransnet_three_dataset_l4_tpr_tss_off_seed42_v1/v1"
RECIPE_ID = "final_tss_off_ner_l4_tpr_v1"
ARCHITECTURE = "tpd8_ner4_qfg2_croa_ner_l4_tpr_v1"
EXECUTION_DECISION_SCHEMA = "sctransnet_ner_l4_tpr_execution_decision/v1"
EXECUTION_DECISION_AUTHORIZE = "authorize_formal_training"
TRAINING_SEED = base.TRAINING_SEED
DATASETS = tuple(base.DATASETS)
METHOD = "final"
TSS_REQUESTED_WEIGHT = 0.0
FORMAL_EPOCHS = base.FORMAL_EPOCHS
FORMAL_BEGIN_TEST = base.FORMAL_BEGIN_TEST
FORMAL_EVAL_EVERY = base.FORMAL_EVAL_EVERY
FORMAL_BATCH_SIZE = base.FORMAL_BATCH_SIZE
FORMAL_PATCH_SIZE = base.FORMAL_PATCH_SIZE
FORMAL_WORKERS = base.FORMAL_WORKERS
FORMAL_BASE_LR = base.FORMAL_BASE_LR
FORMAL_MIN_LR = base.FORMAL_MIN_LR
FORMAL_WARMUP_EPOCHS = base.FORMAL_WARMUP_EPOCHS
FORMAL_THRESHOLD = base.FORMAL_THRESHOLD
FORMAL_MATCH_RADIUS = base.FORMAL_MATCH_RADIUS
FORMAL_TINY_AREA = base.FORMAL_TINY_AREA
CHECKPOINT_ROLES = tuple(base.CHECKPOINT_ROLES)
TRAINING_STATE_KEY_COUNT = (
    l4_tpr.FORMAL_V4_QFG_V2_CROA_L4_TPR_SURVIVAL_STATE_KEY_COUNT
)
INFERENCE_STATE_KEY_COUNT = (
    l4_tpr.FORMAL_V4_QFG_V2_CROA_L4_TPR_INFERENCE_STATE_KEY_COUNT
)
DEFAULT_DATA_ROOT = base.DEFAULT_DATA_ROOT
DEFAULT_PROTOCOL_MANIFEST = base.DEFAULT_PROTOCOL_MANIFEST
DEFAULT_TSS_STATISTICS = base.DEFAULT_TSS_STATISTICS
DEFAULT_RESULTS_ROOT = (
    REPO_ROOT / "results/three_dataset_l4_tpr_tss_off_seed42_v1"
)
PROTOCOL_DOCUMENT = REPO_ROOT / "SCTransNet_NER_L4_TPR性能优化与代码实现方案.md"
GPU_UUIDS = {
    "0": "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    "1": "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}

_BASE_PROTOCOL_PAYLOAD = base._protocol_payload
_BASE_MODEL_BUILDER = base._build_method_model
_ENGINE_WRITE_JSON = engine.write_json_atomic


RUNTIME_DEPENDENCY_RELATIVE_PATHS = (
    "experiments/train_three_dataset_l4_tpr_tss_off_seed42_v1.py",
    "experiments/export_tpd_ner_v4_qfg_v2_croa_l4_tpr_to_inference.py",
    "experiments/train_three_dataset_tss_off_seed42_v1.py",
    "experiments/train_three_dataset_seed42_global_tss_v2.py",
    "experiments/train_four_dataset_original_final_seed42_exact_v1.py",
    "experiments/four_dataset_models_seed42_v1.py",
    "experiments/three_dataset_v2_protocol.py",
    "experiments/paper_three_dataset_v2.py",
    "experiments/tpd_training_loss.py",
    "experiments/train_tpd_clean_v8_mprs_dch.py",
    "model/Config.py",
    "model/SCTransNet.py",
    "model/tpd.py",
    "model/tpd_clean.py",
    "model/tpd_clean_v8_mprs_dch.py",
    "model/tpd_forward_contract.py",
    "model/tpd_frequency_gate.py",
    "model/tpd_frequency_gate_v2_croa.py",
    "model/tpd_ner_l4_target_protected_reallocation.py",
    "model/tpd_ner_v8_mprs_dch.py",
    "model/tpd_ner_v8_mprs_dch_v2.py",
    "model/tpd_ner_v8_mprs_dch_v3.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_l4_tpr.py",
    "model/tpd_query_frequency_bridge.py",
    "model/tpd_relay.py",
    "model/tpd_sctransnet.py",
    "model/tpd_survival.py",
)


class L4TPRTrainingProtocolError(ValueError):
    """One command or persisted artifact violates the frozen recipe."""


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise L4TPRTrainingProtocolError(
            f"{name} differs: {actual!r} != {expected!r}"
        )


def runtime_source_paths() -> dict[str, Path]:
    ready: dict[str, Path] = {}
    for relative in RUNTIME_DEPENDENCY_RELATIVE_PATHS:
        path = (REPO_ROOT / relative).resolve(strict=True)
        prefix = "architecture" if relative.startswith("model/") else "runtime"
        ready[f"{prefix}::{relative}"] = path
    return dict(sorted(ready.items()))


def runtime_source_records() -> dict[str, dict[str, str]]:
    return {
        key: {"path": str(path), "sha256": engine.file_sha256(path)}
        for key, path in runtime_source_paths().items()
    }


def recipe_identity(args: argparse.Namespace) -> dict[str, Any]:
    _require_equal("method", args.method, METHOD)
    if args.tss_weight not in (None, 0, 0.0):
        raise L4TPRTrainingProtocolError(
            "NER-L4-TPR formal training requires TSS weight 0"
        )
    return {
        "method": METHOD,
        "recipe_id": RECIPE_ID,
        "architecture": ARCHITECTURE,
        "l4_tpr_integration_version": l4_tpr.L4_TPR_INTEGRATION_VERSION,
        "requested_tss_weight": TSS_REQUESTED_WEIGHT,
        "tss_enabled": False,
        "tss_heads_registered": True,
        "tss_training_forward_computes_logits": True,
        "tss_loss_consumes_logits": False,
        "tss_survival_target_constructed": False,
        "fresh_seed42_scratch": True,
        "optimizer": "Adam",
        "optimizer_state_initialized_fresh": True,
        "parent_checkpoint": None,
        "warm_start_used": False,
        "resume_scope": "same_l4_tpr_v1_run_only",
        "old_final_568_state_accepted": False,
        "l4_tpr_parameters_jointly_trainable": True,
        "qfg_parameters_jointly_trainable": True,
    }


def load_execution_decision(path: Path | None) -> dict[str, Any]:
    """Load the later screening decision supplied explicitly by the caller."""

    if path is None:
        raise L4TPRTrainingProtocolError(
            "formal NER-L4-TPR execution requires "
            "--execution-decision-json from the later screening stage"
        )
    supplied = Path(path)
    ready = supplied.resolve(strict=True)
    if not ready.is_file():
        raise FileNotFoundError(ready)
    payload = json.loads(ready.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise L4TPRTrainingProtocolError(
            "execution decision JSON must contain an object"
        )
    expected = (
        ("schema", EXECUTION_DECISION_SCHEMA),
        ("status", "complete"),
        ("decision", EXECUTION_DECISION_AUTHORIZE),
        ("architecture", ARCHITECTURE),
        ("recipe_id", RECIPE_ID),
        ("seed", TRAINING_SEED),
        ("datasets", list(DATASETS)),
        ("training_authorized", True),
        ("fresh_seed42_scratch", True),
        ("parent_checkpoint", None),
        ("warm_start_used", False),
        ("requested_tss_weight", TSS_REQUESTED_WEIGHT),
    )
    for field, value in expected:
        _require_equal(f"execution decision {field}", payload.get(field), value)
    return {
        "path": str(ready),
        "sha256": engine.file_sha256(ready),
        "schema": EXECUTION_DECISION_SCHEMA,
        "decision": EXECUTION_DECISION_AUTHORIZE,
        "training_authorized": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=(METHOD,), required=True)
    parser.add_argument("--tss-weight", type=float, default=0.0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL_MANIFEST
    )
    parser.add_argument(
        "--tss-statistics", type=Path, default=DEFAULT_TSS_STATISTICS
    )
    parser.add_argument("--execution-decision-json", type=Path)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--begin-test", type=int, default=FORMAL_BEGIN_TEST)
    parser.add_argument("--eval-every", type=int, default=FORMAL_EVAL_EVERY)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--patch-size", type=int, default=FORMAL_PATCH_SIZE)
    parser.add_argument("--workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--base-lr", type=float, default=FORMAL_BASE_LR)
    parser.add_argument("--min-lr", type=float, default=FORMAL_MIN_LR)
    parser.add_argument(
        "--warmup-epochs", type=int, default=FORMAL_WARMUP_EPOCHS
    )
    parser.add_argument("--threshold", type=float, default=FORMAL_THRESHOLD)
    parser.add_argument("--match-radius", type=float, default=FORMAL_MATCH_RADIUS)
    parser.add_argument("--tiny-area", type=int, default=FORMAL_TINY_AREA)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--physical-gpu-index", choices=tuple(GPU_UUIDS))
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument(
        "--resume", choices=("auto", "never", "required"), default="auto"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-test-images", type=int)
    args = parser.parse_args(argv)
    args.manifest_root = args.protocol_manifest.parent
    args.survival_pos_weight = None
    recipe_identity(args)
    return args


def validate_args(args: argparse.Namespace) -> dict[str, Any]:
    positive.data_protocol.require_dataset(args.dataset)
    positive.data_protocol.require_seed(args.seed)
    recipe_identity(args)
    if args.eval_every < 1 or args.epochs < 1 or args.begin_test < 1:
        raise L4TPRTrainingProtocolError("epoch controls must be positive")
    if args.batch_size < 1 or args.workers < 0:
        raise L4TPRTrainingProtocolError("loader controls differ")
    _require_equal("patch size", args.patch_size, FORMAL_PATCH_SIZE)
    _require_equal("fixed threshold", args.threshold, FORMAL_THRESHOLD)
    if args.smoke:
        if args.epochs > 2:
            raise L4TPRTrainingProtocolError(
                "smoke runs are limited to two epochs"
            )
        if args.max_train_images is None or args.max_test_images is None:
            raise L4TPRTrainingProtocolError(
                "smoke requires train/test image limits"
            )
        if args.device == "cuda:0":
            if args.physical_gpu_index not in GPU_UUIDS:
                raise L4TPRTrainingProtocolError(
                    "CUDA smoke requires an explicit physical GPU"
                )
            _require_equal(
                "GPU UUID",
                args.expected_gpu_uuid,
                GPU_UUIDS[args.physical_gpu_index],
            )
    else:
        if args.max_train_images is not None or args.max_test_images is not None:
            raise L4TPRTrainingProtocolError(
                "formal runs cannot limit train or test images"
            )
        expected = {
            "epochs": FORMAL_EPOCHS,
            "begin_test": FORMAL_BEGIN_TEST,
            "eval_every": FORMAL_EVAL_EVERY,
            "batch_size": FORMAL_BATCH_SIZE,
            "patch_size": FORMAL_PATCH_SIZE,
            "workers": FORMAL_WORKERS,
            "base_lr": FORMAL_BASE_LR,
            "min_lr": FORMAL_MIN_LR,
            "warmup_epochs": FORMAL_WARMUP_EPOCHS,
            "threshold": FORMAL_THRESHOLD,
            "match_radius": FORMAL_MATCH_RADIUS,
            "tiny_area": FORMAL_TINY_AREA,
            "device": "cuda:0",
        }
        for field, value in expected.items():
            _require_equal(f"formal {field}", getattr(args, field), value)
        if args.physical_gpu_index not in GPU_UUIDS:
            raise L4TPRTrainingProtocolError(
                "formal physical GPU index is required"
            )
        _require_equal(
            "GPU UUID",
            args.expected_gpu_uuid,
            GPU_UUIDS[args.physical_gpu_index],
        )

    decision = load_execution_decision(args.execution_decision_json)
    previous = getattr(args, "execution_decision_binding", None)
    if previous is not None:
        _require_equal("revalidated execution decision", previous, decision)
    args.execution_decision_binding = copy.deepcopy(decision)
    return decision


def _decision_binding_from_args(args: argparse.Namespace) -> Mapping[str, Any]:
    decision = getattr(args, "execution_decision_binding", None)
    if not isinstance(decision, Mapping):
        raise L4TPRTrainingProtocolError(
            "execution decision must be validated before artifact construction"
        )
    sha256 = decision.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise L4TPRTrainingProtocolError("execution decision SHA differs")
    return decision


def _run_directory(args: argparse.Namespace) -> Path:
    root = args.results_root.resolve()
    if args.smoke:
        root = root / "smoke"
    return root / "runs" / args.dataset / RECIPE_ID / "seed_42"


def _validate_existing_run_artifacts(args: argparse.Namespace) -> None:
    run_dir = _run_directory(args)
    if not run_dir.exists():
        return
    if not run_dir.is_dir():
        raise L4TPRTrainingProtocolError(
            f"NER-L4-TPR run path is not a directory: {run_dir}"
        )
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, Mapping) or summary.get("status") != "complete":
            raise L4TPRTrainingProtocolError(
                "existing NER-L4-TPR summary is incomplete"
            )
        return
    latest = run_dir / "resume/latest_training_state.pth.tar"
    if latest.is_file():
        if args.resume == "never":
            raise FileExistsError(
                f"rolling state exists but --resume never: {latest}"
            )
        return
    leftovers = [
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "run.lock"
    ]
    if leftovers:
        raise L4TPRTrainingProtocolError(
            "partial NER-L4-TPR artifacts exist without rolling state"
        )


def _architecture_binding(model: nn.Module) -> tuple[str, Mapping[str, Any]]:
    manifest = model.architecture_manifest()
    return engine.canonical_sha256(manifest), manifest


def _build_method_model(
    method: str,
    seed: int,
    *,
    dataset_name: str,
) -> tuple[nn.Module, dict[str, Any]]:
    _require_equal("builder method", method, METHOD)
    _require_equal("builder seed", seed, TRAINING_SEED)
    positive.data_protocol.require_dataset(dataset_name)

    # Both graphs are freshly constructed with seed 42.  The copy below pairs
    # every pre-extension tensor exactly; it never reads learned checkpoint
    # state and therefore is not a warm start.
    reference, reference_metadata = _BASE_MODEL_BUILDER(
        METHOD,
        seed,
        dataset_name=dataset_name,
    )
    model, raw = l4_tpr.build_formal_v4_qfg_v2_croa_l4_tpr_survival_model(seed)
    reference_state = reference.state_dict()
    candidate_state = model.state_dict()
    _require_equal(
        "paired NER-L4-TPR shared state keys",
        set(reference_state),
        set(candidate_state) - set(l4_tpr.L4_TPR_STATE_KEYS),
    )
    with torch.no_grad():
        for name, value in reference_state.items():
            candidate_state[name].copy_(value)
    model.load_state_dict(candidate_state, strict=True)
    for name, expected in reference_state.items():
        if not torch.equal(model.state_dict()[name], expected):
            raise L4TPRTrainingProtocolError(
                f"paired scratch initialization differs at {name!r}"
            )
    reference_state_sha256 = paired_registry.state_dict_sha256(reference_state)
    del reference

    validated = (
        l4_tpr.validate_formal_v4_qfg_v2_croa_l4_tpr_survival_model(
            model,
            require_zero_initialized_heads=True,
            require_identity_initialized_qfg=True,
            require_zero_initialized_l4_tpr=True,
        )
    )
    state = model.state_dict()
    _require_equal("training state-key count", len(state), TRAINING_STATE_KEY_COUNT)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise L4TPRTrainingProtocolError(
            "all NER-L4-TPR training parameters must be jointly trainable"
        )
    if any(
        int(torch.count_nonzero(state[name])) != 0
        for name in l4_tpr.L4_TPR_STATE_KEYS
    ):
        raise L4TPRTrainingProtocolError(
            "new NER-L4-TPR state must start at the zero anchor"
        )
    architecture_id, manifest = _architecture_binding(model)
    metadata = copy.deepcopy(dict(raw))
    metadata.update(
        {
            "schema": SCHEMA,
            "dataset_name": dataset_name,
            "method": METHOD,
            "recipe_id": RECIPE_ID,
            "training_seed": TRAINING_SEED,
            "initialization_mode": "fresh_seed42_paired_scratch_extension",
            "construction": "scratch_seed42_no_parent_checkpoint",
            "parent_checkpoint": None,
            "warm_start_used": False,
            "learned_state_loaded": False,
            "paired_reference_builder": (
                "three_dataset_tss_off_seed42_v1_exact_model_builder"
            ),
            "paired_reference_state_sha256": reference_state_sha256,
            "paired_reference_metadata": copy.deepcopy(reference_metadata),
            "all_pre_l4_tpr_state_bitwise_equal_to_reference": True,
            "l4_tpr_new_state_zero_initialized": True,
            "resume_scope": "same_l4_tpr_v1_run_only",
            "state_key_count": len(state),
            "architecture_id": architecture_id,
            "architecture_manifest": dict(manifest),
            "validated_model": validated,
            "requested_tss_weight": 0.0,
            "tss_enabled": False,
            "tss_heads_registered": True,
            "tss_training_forward_computes_logits": True,
            "tss_loss_consumes_logits": False,
            "tss_survival_target_constructed": False,
            "optimizer_initialization": "fresh_Adam",
            "l4_tpr_parameters_jointly_trainable": True,
            "qfg_parameters_jointly_trainable": True,
        }
    )
    return model, metadata


def _import_runtime_components() -> tuple[Any, Any, Any]:
    return (
        _build_method_model,
        positive._train_dataset_adapter,
        positive._test_dataset_adapter,
    )


def _protocol_payload(args: argparse.Namespace, **kwargs: Any) -> dict[str, Any]:
    payload = _BASE_PROTOCOL_PAYLOAD(args, **kwargs)
    metadata = copy.deepcopy(dict(kwargs["model_metadata"]))
    identity = recipe_identity(args)
    decision = copy.deepcopy(dict(_decision_binding_from_args(args)))
    payload.update(
        {
            "schema": SCHEMA,
            "method": METHOD,
            "recipe": identity,
            "requested_tss_weight": 0.0,
            "tss_enabled": False,
            "model": metadata,
            "planned_total_epochs": args.epochs,
            "execution_decision": decision,
            "runtime_sources": runtime_source_records(),
            "development_protocol": "seed42_img_idx_test_selected",
            "paper_unbiased_test_supported": False,
        }
    )
    payload["training"].update(identity)
    payload["training"].update(
        {
            "optimizer": "Adam",
            "optimizer_state_initialization": "fresh",
            "precision": "FP32",
            "amp": False,
            "initialization": "fresh_seed42_scratch",
            "parent_checkpoint": None,
            "warm_start_used": False,
            "resume_state_key_count": TRAINING_STATE_KEY_COUNT,
            "old_final_568_resume_allowed": False,
        }
    )
    payload["search_budget_disclosure"] = {
        "new_l4_tpr_training_runs": len(DATASETS),
        "one_run_per_dataset": True,
        "seed": TRAINING_SEED,
        "test_selected": True,
        "formal_execution_requires_explicit_later_decision": True,
    }
    return payload


def _selected_checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = base._ENGINE_SELECTED_PAYLOAD(*args, **kwargs)
    run_args = kwargs["args"]
    architecture_id, _ = _architecture_binding(kwargs["model"])
    payload.update(
        {
            "schema": SCHEMA,
            "recipe": recipe_identity(run_args),
            "requested_tss_weight": 0.0,
            "tss_enabled": False,
            "architecture_id": architecture_id,
            "l4_tpr_integration_version": l4_tpr.L4_TPR_INTEGRATION_VERSION,
            "training_state_key_count": TRAINING_STATE_KEY_COUNT,
            "execution_decision_sha256": _decision_binding_from_args(run_args)[
                "sha256"
            ],
        }
    )
    return payload


def _latest_checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    event = kwargs.get("event")
    if not isinstance(event, dict):
        raise TypeError("latest checkpoint event must be a mutable dictionary")
    run_args = kwargs["args"]
    identity = recipe_identity(run_args)
    event["recipe"] = identity
    event.update(base._AUDIT.payload())
    payload = base._ENGINE_LATEST_PAYLOAD(*args, **kwargs)
    architecture_id, _ = _architecture_binding(kwargs["model"])
    payload.update(
        {
            "schema": SCHEMA,
            "recipe": identity,
            "requested_tss_weight": 0.0,
            "tss_enabled": False,
            "architecture_id": architecture_id,
            "l4_tpr_integration_version": l4_tpr.L4_TPR_INTEGRATION_VERSION,
            "training_state_key_count": TRAINING_STATE_KEY_COUNT,
            "planned_total_epochs": run_args.epochs,
            "execution_decision_sha256": _decision_binding_from_args(run_args)[
                "sha256"
            ],
        }
    )
    base._AUDIT.reset()
    return payload


def _load_resume_l4_tpr(
    *,
    args: argparse.Namespace,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    protocol_sha256: str,
) -> tuple[int, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    del device
    if not path.exists():
        if args.resume == "required":
            raise FileNotFoundError(path)
        return 1, {}, {}, None
    if args.resume == "never":
        raise FileExistsError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    architecture_id, _ = _architecture_binding(model)
    decision_sha = _decision_binding_from_args(args)["sha256"]
    for field, expected in (
        ("schema", SCHEMA),
        ("dataset", args.dataset),
        ("method", METHOD),
        ("seed", TRAINING_SEED),
        ("protocol_sha256", protocol_sha256),
        ("recipe", recipe_identity(args)),
        ("architecture_id", architecture_id),
        ("l4_tpr_integration_version", l4_tpr.L4_TPR_INTEGRATION_VERSION),
        ("training_state_key_count", TRAINING_STATE_KEY_COUNT),
        ("planned_total_epochs", args.epochs),
        ("execution_decision_sha256", decision_sha),
    ):
        _require_equal(f"resume {field}", payload.get(field), expected)
    state = payload.get("state_dict")
    expected_state_keys = (
        len(model.state_dict()) if args.smoke else TRAINING_STATE_KEY_COUNT
    )
    if not isinstance(state, Mapping) or len(state) != expected_state_keys:
        raise L4TPRTrainingProtocolError(
            "resume state-key count differs from the exact 569-key "
            "NER-L4-TPR graph; old 568-key Final state is not accepted"
        )
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    engine.restore_rng_state(payload["rng_state"])
    completed = int(payload["epoch"])
    event = payload.get("event")
    if not isinstance(event, Mapping) or event.get("recipe") != recipe_identity(args):
        raise L4TPRTrainingProtocolError(
            "resume event is not from this NER-L4-TPR recipe"
        )
    _require_equal("resume event epoch", event.get("epoch"), completed)
    return (
        completed + 1,
        dict(payload.get("best_miou", {})),
        dict(payload.get("best_pd", {})),
        dict(event),
    )


_ACTIVE_ARGS: argparse.Namespace | None = None


def _enrich_json(path: Path, value: Any) -> Any:
    if not isinstance(value, Mapping) or path.name not in {
        "progress.json",
        "summary.json",
    }:
        return value
    enriched = copy.deepcopy(dict(value))
    enriched.update(
        {
            "schema": SCHEMA,
            "recipe": recipe_identity(
                argparse.Namespace(method=METHOD, tss_weight=0.0)
            ),
            "requested_tss_weight": 0.0,
            "tss_enabled": False,
            "planned_total_epochs": enriched.get(
                "total_epochs", enriched.get("epochs", FORMAL_EPOCHS)
            ),
            "checkpoint_roles": list(CHECKPOINT_ROLES),
            "l4_tpr_integration_version": l4_tpr.L4_TPR_INTEGRATION_VERSION,
            "training_state_key_count": TRAINING_STATE_KEY_COUNT,
        }
    )
    if _ACTIVE_ARGS is not None:
        enriched["execution_decision_sha256"] = _decision_binding_from_args(
            _ACTIVE_ARGS
        )["sha256"]
    return enriched


def _write_json_atomic(path: Path, value: Any) -> None:
    _ENGINE_WRITE_JSON(path, _enrich_json(path, value))


@contextmanager
def _patched_base_and_engine(args: argparse.Namespace) -> Iterator[None]:
    global _ACTIVE_ARGS
    replacements = {
        "SCHEMA": SCHEMA,
        "DATASETS": DATASETS,
        "PROTOCOL_DOCUMENT": PROTOCOL_DOCUMENT,
        "GPU_UUIDS": GPU_UUIDS,
        "recipe_identity": recipe_identity,
        "validate_args": validate_args,
        "_run_directory": _run_directory,
        "_build_method_model": _build_method_model,
        "_import_runtime_components": _import_runtime_components,
        "_protocol_payload": _protocol_payload,
        "_selected_checkpoint_payload": _selected_checkpoint_payload,
        "_latest_checkpoint_payload": _latest_checkpoint_payload,
        "_load_resume_off": _load_resume_l4_tpr,
        "_write_json_atomic": _write_json_atomic,
    }
    previous = {key: getattr(base, key) for key in replacements}
    if _ACTIVE_ARGS is not None:
        raise RuntimeError("NER-L4-TPR trainer patch is already active")
    for key, value in replacements.items():
        setattr(base, key, value)
    _ACTIVE_ARGS = args
    try:
        with base._patched_engine(args):
            yield
    finally:
        _ACTIVE_ARGS = None
        for key, value in previous.items():
            setattr(base, key, value)
        base._AUDIT.reset()


def run(args: argparse.Namespace) -> Path:
    validate_args(args)
    _validate_existing_run_artifacts(args)
    with _patched_base_and_engine(args):
        output = engine.run(args)
    summary = json.loads(output.read_text(encoding="utf-8"))
    for field, expected in (
        ("schema", SCHEMA),
        ("recipe", recipe_identity(args)),
        ("requested_tss_weight", 0.0),
        ("tss_enabled", False),
        ("planned_total_epochs", args.epochs),
    ):
        _require_equal(f"summary {field}", summary.get(field), expected)
    return output


def main(argv: list[str] | None = None) -> None:
    print(run(parse_args(argv)), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "ARCHITECTURE",
    "CHECKPOINT_ROLES",
    "DATASETS",
    "EXECUTION_DECISION_AUTHORIZE",
    "EXECUTION_DECISION_SCHEMA",
    "FORMAL_BEGIN_TEST",
    "FORMAL_EPOCHS",
    "FORMAL_EVAL_EVERY",
    "INFERENCE_STATE_KEY_COUNT",
    "L4TPRTrainingProtocolError",
    "RECIPE_ID",
    "SCHEMA",
    "TRAINING_SEED",
    "TRAINING_STATE_KEY_COUNT",
    "_build_method_model",
    "_load_resume_l4_tpr",
    "_run_directory",
    "load_execution_decision",
    "parse_args",
    "recipe_identity",
    "run",
    "validate_args",
]
