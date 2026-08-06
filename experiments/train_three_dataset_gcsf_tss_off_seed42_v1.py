#!/usr/bin/env python3
"""Seed-42 three-dataset scratch trainer for TPD8+NER4+QFG2+GCSF.

The frozen three-dataset TSS-off runner remains the sole optimization, data,
metric, and checkpoint-selection protocol source.  This entry changes only
the model graph/run identity and adds two controls:

* formal execution is fail-closed unless the write-once six-role GCSF branch
  comparator proves quantified Trigger A and authorizes the pilot;
* ``--pause-after-epoch 200`` stops after the rolling model/Adam/RNG state and
  epoch event are durable.  ``--resume required`` then continues the same
  1,000-epoch run at epoch 201 with the original scheduler denominator.

Formal construction is fresh seed-42 scratch.  No checkpoint is accepted as
initialization input, and resume is restricted to this exact GCSF run only.
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

from analysis import compare_three_dataset_gcsf_branch_audit_v1 as comparator  # noqa: E402
from experiments import four_dataset_models_seed42_v1 as paired_registry  # noqa: E402
from experiments import train_four_dataset_original_final_seed42_exact_v1 as engine  # noqa: E402
from experiments import train_three_dataset_seed42_global_tss_v2 as positive  # noqa: E402
from experiments import train_three_dataset_tss_off_seed42_v1 as base  # noqa: E402
from model import tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_gcsf as gcsf  # noqa: E402


SCHEMA = "sctransnet_three_dataset_gcsf_tss_off_seed42_v1/v1"
RECIPE_ID = "final_tss_off_gcsf_v1"
TRAINING_SEED = base.TRAINING_SEED
DATASETS = tuple(base.DATASETS)
METHOD = "final"
TSS_REQUESTED_WEIGHT = 0.0
FORMAL_EPOCHS = base.FORMAL_EPOCHS
FORMAL_PAUSE_EPOCH = 200
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
    gcsf.FORMAL_V4_QFG_V2_CROA_GCSF_SURVIVAL_STATE_KEY_COUNT
)
DEFAULT_DATA_ROOT = base.DEFAULT_DATA_ROOT
DEFAULT_PROTOCOL_MANIFEST = base.DEFAULT_PROTOCOL_MANIFEST
DEFAULT_TSS_STATISTICS = base.DEFAULT_TSS_STATISTICS
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results/three_dataset_gcsf_tss_off_seed42_v1"
DEFAULT_DECISION_JSON = comparator.DEFAULT_OUTPUT_DIR / "decision.json"
PROTOCOL_DOCUMENT = REPO_ROOT / "SCTransNet_TPD8诊断后完整模型级GCSF优化方案.md"
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
    "experiments/train_three_dataset_gcsf_tss_off_seed42_v1.py",
    "experiments/train_three_dataset_tss_off_seed42_v1.py",
    "experiments/train_three_dataset_seed42_global_tss_v2.py",
    "experiments/train_four_dataset_original_final_seed42_exact_v1.py",
    "experiments/four_dataset_models_seed42_v1.py",
    "experiments/three_dataset_v2_protocol.py",
    "experiments/paper_three_dataset_v2.py",
    "experiments/tpd_training_loss.py",
    "experiments/train_tpd_clean_v8_mprs_dch.py",
    "experiments/train_tpd_pilot.py",
    "analysis/analyze_three_dataset_gcsf_branch_audit_v1.py",
    "analysis/compare_three_dataset_gcsf_branch_audit_v1.py",
    "analysis/compare_three_dataset_qfg_level_knockout_v1.py",
    "model/Config.py",
    "model/SCTransNet.py",
    "model/tpd.py",
    "model/tpd_clean.py",
    "model/tpd_clean_v8_mprs_dch.py",
    "model/tpd_forward_contract.py",
    "model/tpd_frequency_gate.py",
    "model/tpd_frequency_gate_v2_croa.py",
    "model/tpd_global_constant_sum_skip_fusion.py",
    "model/tpd_ner_v8_mprs_dch.py",
    "model/tpd_ner_v8_mprs_dch_v2.py",
    "model/tpd_ner_v8_mprs_dch_v3.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_survival.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival.py",
    "model/tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_gcsf.py",
    "model/tpd_query_frequency_bridge.py",
    "model/tpd_relay.py",
    "model/tpd_sctransnet.py",
    "model/tpd_survival.py",
)


class GCSFTrainingProtocolError(ValueError):
    """A command or artifact violates the frozen GCSF training contract."""


class _PauseAfterEpoch(RuntimeError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"paused after durable epoch: {path}")
        self.path = path


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise GCSFTrainingProtocolError(
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
        raise GCSFTrainingProtocolError("GCSF formal training requires TSS weight 0")
    return {
        "method": METHOD,
        "recipe_id": RECIPE_ID,
        "architecture": "tpd8_ner4_qfg2_croa_gcsf_v1",
        "gcsf_integration_version": gcsf.GCSF_INTEGRATION_VERSION,
        "requested_tss_weight": TSS_REQUESTED_WEIGHT,
        "tss_enabled": False,
        "tss_heads_registered": True,
        "tss_training_forward_computes_logits": True,
        "tss_loss_consumes_logits": False,
        "tss_survival_target_constructed": False,
        "fresh_seed42_scratch": True,
        "parent_checkpoint": None,
        "warm_start_used": False,
        "resume_scope": "same_gcsf_v1_run_only",
        "gcsf_parameters_jointly_trainable": True,
        "qfg_parameters_jointly_trainable": True,
    }


def _expected_comparator_sources() -> dict[str, str]:
    return {
        "analysis/compare_three_dataset_gcsf_branch_audit_v1.py": engine.file_sha256(
            Path(comparator.__file__).resolve()
        ),
        "analysis/analyze_three_dataset_gcsf_branch_audit_v1.py": engine.file_sha256(
            Path(comparator.analyzer.__file__).resolve()
        ),
        "analysis/compare_three_dataset_qfg_level_knockout_v1.py": engine.file_sha256(
            Path(comparator.gate_core.__file__).resolve()
        ),
    }


def _validate_trigger_a_authorization(payload: Mapping[str, Any]) -> None:
    comparator.validate_comparison_payload(payload)
    _require_equal("decision status", payload.get("status"), "complete")
    _require_equal(
        "decision",
        payload.get("decision"),
        comparator.DECISION_AUTHORIZE,
    )
    trigger_a = payload.get("trigger_a")
    if not isinstance(trigger_a, Mapping):
        raise GCSFTrainingProtocolError("decision lacks Trigger A")
    for field, expected in (
        ("implemented", True),
        ("sole_training_authorization_trigger", True),
        ("passed", True),
    ):
        _require_equal(f"Trigger A {field}", trigger_a.get(field), expected)
    _require_equal(
        "GCSF pilot authorization",
        payload.get("gcsf_v1_implementation_and_pilot_authorized"),
        True,
    )
    for name in ("trigger_b", "trigger_c"):
        trigger = payload.get(name)
        if not isinstance(trigger, Mapping):
            raise GCSFTrainingProtocolError(f"decision lacks {name}")
        _require_equal(f"{name} training authorization", trigger.get("authorizes_training"), False)
    _require_equal(
        "comparator source closure",
        payload.get("source_sha256"),
        _expected_comparator_sources(),
    )


def load_trigger_a_decision(path: Path) -> dict[str, Any]:
    """Validate and replay the fixed six-role decision before any training."""

    supplied = Path(path)
    if supplied.is_symlink():
        raise GCSFTrainingProtocolError("GCSF decision JSON cannot be a symlink")
    ready = supplied.resolve(strict=True)
    if not ready.is_file():
        raise FileNotFoundError(ready)
    raw = ready.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise GCSFTrainingProtocolError("GCSF decision JSON must be an object")
    _validate_trigger_a_authorization(payload)

    bindings = payload.get("input_bindings")
    if not isinstance(bindings, Mapping):
        raise GCSFTrainingProtocolError("GCSF decision lacks six input bindings")
    analyzer_payloads: dict[str, Mapping[str, Any]] = {}
    for key, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise GCSFTrainingProtocolError(f"invalid decision binding: {key}")
        supplied_input = Path(str(binding.get("path", "")))
        if supplied_input.is_symlink():
            raise GCSFTrainingProtocolError(
                f"decision input cannot be a symlink: {key}"
            )
        input_path = supplied_input.resolve(strict=True)
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        _require_equal(
            f"decision input SHA {key}",
            engine.file_sha256(input_path),
            binding.get("sha256"),
        )
        one = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(one, Mapping):
            raise GCSFTrainingProtocolError(f"analyzer input is not an object: {key}")
        analyzer_payloads[str(key)] = one
    replay = comparator.compare_payloads(
        analyzer_payloads,
        input_bindings=bindings,
    )
    _require_equal(
        "decision replay",
        engine.canonical_sha256(replay),
        engine.canonical_sha256(payload),
    )
    return {
        "path": str(ready),
        "sha256": engine.file_sha256(ready),
        "schema": comparator.SCHEMA,
        "decision": comparator.DECISION_AUTHORIZE,
        "trigger_a_passed": True,
        "qualifying_modes": copy.deepcopy(payload["trigger_a"]["qualifying_modes"]),
        "replayed_from_six_bound_inputs": True,
        "source_sha256": copy.deepcopy(payload["source_sha256"]),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=(METHOD,), required=True)
    parser.add_argument("--tss-weight", type=float, default=0.0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL_MANIFEST)
    parser.add_argument("--tss-statistics", type=Path, default=DEFAULT_TSS_STATISTICS)
    parser.add_argument("--gcsf-decision-json", type=Path, default=DEFAULT_DECISION_JSON)
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    parser.add_argument("--begin-test", type=int, default=FORMAL_BEGIN_TEST)
    parser.add_argument("--eval-every", type=int, default=FORMAL_EVAL_EVERY)
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--patch-size", type=int, default=FORMAL_PATCH_SIZE)
    parser.add_argument("--workers", type=int, default=FORMAL_WORKERS)
    parser.add_argument("--base-lr", type=float, default=FORMAL_BASE_LR)
    parser.add_argument("--min-lr", type=float, default=FORMAL_MIN_LR)
    parser.add_argument("--warmup-epochs", type=int, default=FORMAL_WARMUP_EPOCHS)
    parser.add_argument("--threshold", type=float, default=FORMAL_THRESHOLD)
    parser.add_argument("--match-radius", type=float, default=FORMAL_MATCH_RADIUS)
    parser.add_argument("--tiny-area", type=int, default=FORMAL_TINY_AREA)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--physical-gpu-index", choices=tuple(GPU_UUIDS))
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--pause-after-epoch", type=int)
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
        raise GCSFTrainingProtocolError("epoch controls must be positive")
    _require_equal("patch size", args.patch_size, FORMAL_PATCH_SIZE)
    _require_equal("fixed threshold", args.threshold, FORMAL_THRESHOLD)
    if args.smoke:
        if args.epochs > 3 or args.max_train_images is None or args.max_test_images is None:
            raise GCSFTrainingProtocolError("smoke requires <=3 epochs and image limits")
        if args.pause_after_epoch is not None and not 1 <= args.pause_after_epoch < args.epochs:
            raise GCSFTrainingProtocolError("smoke pause must precede its final epoch")
        if args.device == "cuda:0":
            _require_equal("GPU UUID", args.expected_gpu_uuid, GPU_UUIDS.get(args.physical_gpu_index))
    else:
        if args.max_train_images is not None or args.max_test_images is not None:
            raise GCSFTrainingProtocolError("formal runs cannot limit images")
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
        if args.pause_after_epoch not in (None, FORMAL_PAUSE_EPOCH):
            raise GCSFTrainingProtocolError("formal pause must be epoch 200 or omitted")
        if args.physical_gpu_index not in GPU_UUIDS:
            raise GCSFTrainingProtocolError("formal physical GPU index is required")
        _require_equal("GPU UUID", args.expected_gpu_uuid, GPU_UUIDS[args.physical_gpu_index])
        _require_equal(
            "formal comparator path",
            args.gcsf_decision_json.resolve(),
            DEFAULT_DECISION_JSON.resolve(),
        )
    decision = load_trigger_a_decision(args.gcsf_decision_json)
    previous = getattr(args, "gcsf_trigger_a_decision_binding", None)
    if previous is not None:
        _require_equal("revalidated Trigger A decision", previous, decision)
    args.gcsf_trigger_a_decision_binding = copy.deepcopy(decision)
    return decision


def _decision_binding_from_args(args: argparse.Namespace) -> Mapping[str, Any]:
    decision = getattr(args, "gcsf_trigger_a_decision_binding", None)
    if not isinstance(decision, Mapping):
        raise GCSFTrainingProtocolError(
            "Trigger A decision must be validated before artifact construction"
        )
    sha256 = decision.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise GCSFTrainingProtocolError("validated Trigger A decision SHA differs")
    return decision


def _run_directory(args: argparse.Namespace) -> Path:
    root = args.results_root.resolve()
    if args.smoke:
        root = root / "smoke"
    return root / "runs" / args.dataset / RECIPE_ID / "seed_42"


def _validate_existing_run_artifacts(args: argparse.Namespace) -> None:
    """Reject orphaned partial artifacts instead of overwriting them."""

    run_dir = _run_directory(args)
    if not run_dir.exists():
        return
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise GCSFTrainingProtocolError(f"GCSF run path is not a directory: {run_dir}")
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, Mapping) or summary.get("status") != "complete":
            raise GCSFTrainingProtocolError("existing GCSF summary is incomplete")
        return
    latest = run_dir / "resume/latest_training_state.pth.tar"
    if latest.is_file():
        if args.resume == "never":
            raise FileExistsError(
                f"GCSF rolling state exists but --resume never: {latest}"
            )
        return
    leftovers = [
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.name != "run.lock"
    ]
    if leftovers:
        raise GCSFTrainingProtocolError(
            "orphaned GCSF artifacts exist without a rolling state; "
            "refusing implicit overwrite"
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
    positive.data_protocol.require_dataset(dataset_name)
    # Construct the exact frozen TSS-off/QFG2 scratch reference in memory.
    # It contains no learned checkpoint state; its sole purpose is paired
    # initialization so every pre-GCSF tensor starts identically to the
    # completed formal protocol.
    reference, reference_metadata = _BASE_MODEL_BUILDER(
        METHOD,
        seed,
        dataset_name=dataset_name,
    )
    model, raw = gcsf.build_formal_v4_qfg_v2_croa_gcsf_survival_model(seed)
    reference_state = reference.state_dict()
    candidate_state = model.state_dict()
    _require_equal(
        "paired GCSF shared state keys",
        set(reference_state),
        set(candidate_state) - set(gcsf.GCSF_STATE_KEYS),
    )
    with torch.no_grad():
        for name, value in reference_state.items():
            candidate_state[name].copy_(value)
    model.load_state_dict(candidate_state, strict=True)
    for name, expected in reference_state.items():
        if not torch.equal(model.state_dict()[name], expected):
            raise GCSFTrainingProtocolError(
                f"paired GCSF scratch initialization differs at {name!r}"
            )
    reference_state_sha256 = paired_registry.state_dict_sha256(reference_state)
    del reference
    validated = gcsf.validate_formal_v4_qfg_v2_croa_gcsf_survival_model(
        model,
        require_zero_initialized_heads=True,
        require_identity_initialized_qfg=True,
        require_zero_initialized_gcsf=True,
    )
    state = model.state_dict()
    _require_equal("GCSF training state-key count", len(state), TRAINING_STATE_KEY_COUNT)
    if not all(parameter.requires_grad for parameter in model.global_skip_fusion.parameters()):
        raise GCSFTrainingProtocolError("GCSF parameters must be jointly trainable")
    if not all(parameter.requires_grad for parameter in model.tpd_qfg.parameters()):
        raise GCSFTrainingProtocolError("QFG parameters must remain jointly trainable")
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
            "all_pre_gcsf_state_bitwise_equal_to_reference": True,
            "gcsf_new_state_zero_initialized": True,
            "resume_scope": "same_gcsf_v1_run_only",
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
            "gcsf_parameters_jointly_trainable": True,
            "qfg_parameters_jointly_trainable": True,
        }
    )
    return model, metadata


def _import_runtime_components() -> tuple[Any, Any, Any]:
    return _build_method_model, positive._train_dataset_adapter, positive._test_dataset_adapter


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
            "gcsf_trigger_a_authorization": decision,
            "runtime_sources": runtime_source_records(),
            "pause_resume_contract": {
                "pause_epoch": FORMAL_PAUSE_EPOCH,
                "planned_total_epochs": args.epochs,
                "pilot_is_prefix_of_same_run": True,
                "pilot_creates_additional_run": False,
                "same_run_identity": True,
                "same_optimizer_rng_and_scheduler": True,
                "continuation_resume_mode": "required",
            },
            "development_protocol": "seed42_img_idx_test_selected",
            "paper_unbiased_test_supported": False,
        }
    )
    payload["training"].update(identity)
    payload["training"]["optimizer"] = "Adam"
    payload["training"]["precision"] = "FP32"
    payload["training"]["amp"] = False
    payload["training"]["initialization"] = "fresh_seed42_scratch"
    payload["training"]["parent_checkpoint"] = None
    payload["search_budget_disclosure"] = {
        "new_gcsf_training_runs": len(DATASETS),
        "one_run_per_dataset": True,
        "pilot_is_prefix_of_same_run": True,
        "test_selected": True,
        "trigger_a_diagnostic_used_before_training": True,
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
            "gcsf_integration_version": gcsf.GCSF_INTEGRATION_VERSION,
            "training_state_key_count": TRAINING_STATE_KEY_COUNT,
            "gcsf_trigger_a_decision_sha256": _decision_binding_from_args(
                run_args
            )["sha256"],
        }
    )
    return payload


def _latest_checkpoint_payload(*args: Any, **kwargs: Any) -> dict[str, Any]:
    event = kwargs["event"]
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
            "gcsf_integration_version": gcsf.GCSF_INTEGRATION_VERSION,
            "training_state_key_count": TRAINING_STATE_KEY_COUNT,
            "planned_total_epochs": run_args.epochs,
            "gcsf_trigger_a_decision_sha256": _decision_binding_from_args(
                run_args
            )["sha256"],
        }
    )
    base._AUDIT.reset()
    return payload


def _load_resume_gcsf(
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
    expected_decision_sha = _decision_binding_from_args(args)["sha256"]
    for field, expected in (
        ("schema", SCHEMA),
        ("dataset", args.dataset),
        ("method", METHOD),
        ("seed", TRAINING_SEED),
        ("protocol_sha256", protocol_sha256),
        ("recipe", recipe_identity(args)),
        ("architecture_id", architecture_id),
        ("gcsf_integration_version", gcsf.GCSF_INTEGRATION_VERSION),
        ("training_state_key_count", TRAINING_STATE_KEY_COUNT),
        ("planned_total_epochs", args.epochs),
        ("gcsf_trigger_a_decision_sha256", expected_decision_sha),
    ):
        _require_equal(f"resume {field}", payload.get(field), expected)
    state = payload.get("state_dict")
    expected_state_keys = (
        len(model.state_dict()) if args.smoke else TRAINING_STATE_KEY_COUNT
    )
    if not isinstance(state, Mapping) or len(state) != expected_state_keys:
        raise GCSFTrainingProtocolError(
            "resume state-key count differs from the exact GCSF graph"
        )
    model.load_state_dict(state, strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    engine.restore_rng_state(payload["rng_state"])
    completed = int(payload["epoch"])
    event = payload.get("event")
    if not isinstance(event, Mapping) or event.get("recipe") != recipe_identity(args):
        raise GCSFTrainingProtocolError("resume event is not GCSF-v1-only")
    _require_equal("resume event epoch", event.get("epoch"), completed)
    return completed + 1, dict(payload.get("best_miou", {})), dict(payload.get("best_pd", {})), dict(event)


_ACTIVE_ARGS: argparse.Namespace | None = None


def _enrich_json(path: Path, value: Any) -> Any:
    if not isinstance(value, Mapping) or path.name not in {"progress.json", "summary.json"}:
        return value
    enriched = copy.deepcopy(dict(value))
    enriched.update(
        {
            "schema": SCHEMA,
            "recipe": recipe_identity(argparse.Namespace(method=METHOD, tss_weight=0.0)),
            "requested_tss_weight": 0.0,
            "tss_enabled": False,
            "planned_total_epochs": enriched.get(
                "total_epochs", enriched.get("epochs", FORMAL_EPOCHS)
            ),
            "checkpoint_roles": list(CHECKPOINT_ROLES),
            "gcsf_integration_version": gcsf.GCSF_INTEGRATION_VERSION,
        }
    )
    return enriched


def _write_json_atomic(path: Path, value: Any) -> None:
    enriched = _enrich_json(path, value)
    active = _ACTIVE_ARGS
    should_pause = (
        active is not None
        and active.pause_after_epoch is not None
        and path.name == "progress.json"
        and isinstance(enriched, Mapping)
        and enriched.get("status") in {"running", "finalizing"}
        and enriched.get("completed_epoch") == active.pause_after_epoch
        and active.pause_after_epoch < active.epochs
    )
    if not should_pause:
        _ENGINE_WRITE_JSON(path, enriched)
        return
    latest_path = path.parent / "resume/latest_training_state.pth.tar"
    if not latest_path.is_file():
        raise GCSFTrainingProtocolError("pause boundary lacks rolling state")
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    _require_equal("pause rolling epoch", latest.get("epoch"), active.pause_after_epoch)
    protocol_path = path.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha256 = protocol.get("protocol_sha256")
    if not isinstance(protocol_sha256, str) or not protocol_sha256:
        raise GCSFTrainingProtocolError("pause protocol lacks protocol_sha256")
    _require_equal("pause rolling protocol", latest.get("protocol_sha256"), protocol_sha256)
    paused = dict(enriched)
    paused.update(
        {
            "status": "paused",
            "pause_after_epoch": active.pause_after_epoch,
            "resume_required": True,
            "required_resume_mode": "required",
            "protocol_sha256": protocol_sha256,
            "rolling_resume_state": {
                "path": str(latest_path),
                "sha256": engine.file_sha256(latest_path),
                "epoch": active.pause_after_epoch,
            },
        }
    )
    _ENGINE_WRITE_JSON(path, paused)
    raise _PauseAfterEpoch(path)


def validate_paused_run(
    run_dir: Path,
    dataset: str,
    pause_epoch: int = FORMAL_PAUSE_EPOCH,
    planned_total_epochs: int = FORMAL_EPOCHS,
) -> dict[str, Any]:
    path = Path(run_dir) / "progress.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field, expected in (
        ("schema", SCHEMA),
        ("status", "paused"),
        ("dataset", dataset),
        ("completed_epoch", pause_epoch),
        ("pause_after_epoch", pause_epoch),
        ("planned_total_epochs", planned_total_epochs),
        ("resume_required", True),
    ):
        _require_equal(f"paused {field}", payload.get(field), expected)
    rolling = payload.get("rolling_resume_state")
    if not isinstance(rolling, Mapping):
        raise GCSFTrainingProtocolError("paused run lacks rolling binding")
    latest = Path(str(rolling["path"]))
    _require_equal("paused rolling SHA", engine.file_sha256(latest), rolling.get("sha256"))
    latest_payload = torch.load(latest, map_location="cpu", weights_only=False)
    _require_equal("paused rolling protocol", latest_payload.get("protocol_sha256"), payload.get("protocol_sha256"))
    return payload


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
        "_load_resume_off": _load_resume_gcsf,
        "_write_json_atomic": _write_json_atomic,
    }
    previous = {key: getattr(base, key) for key in replacements}
    if _ACTIVE_ARGS is not None:
        raise RuntimeError("GCSF trainer patch is already active")
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
    run_dir = _run_directory(args)
    progress_path = run_dir / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("status") == "paused" and args.resume != "required":
            raise GCSFTrainingProtocolError(
                "paused GCSF continuation requires --resume required"
            )
    try:
        with _patched_base_and_engine(args):
            output = engine.run(args)
    except _PauseAfterEpoch as signal:
        _require_equal("pause path", signal.path, progress_path)
        validate_paused_run(
            run_dir,
            args.dataset,
            int(args.pause_after_epoch),
            args.epochs,
        )
        return progress_path
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
