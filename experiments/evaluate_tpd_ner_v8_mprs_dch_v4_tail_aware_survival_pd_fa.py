#!/usr/bin/env python3
"""Formal checkpoint-local Pd/Fa evaluator for V4 Target Survival.

The executable is intentionally a thin adapter over the repository's frozen
``evaluate_pd_fa_sweep`` implementation and closed-probability-interval
threshold core.  It accepts the two validation-owned checkpoints from either
``tss_control`` or ``tss_on``.  ``--all-four`` merely runs four independent
evaluations; it never pools thresholds or selects a point across checkpoints.

Only the NUDT-SIRST internal validation identifiers recorded by each completed
run are consumed.  The official test split is never read.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_pd_fa_sweep as sweep_core  # noqa: E402
from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa as v4_evaluator,
)
from experiments import (  # noqa: E402
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact as exact,
)
from experiments.evaluate_tpd_clean_v8_mprs_dch_pd_fa import (  # noqa: E402
    CUBLAS_WORKSPACE_CONFIG,
    DETERMINISM_SETTINGS,
    LAST_FLOAT32_BELOW_ONE,
    UPPER_BOUNDARY_THRESHOLD,
    adaptive_thresholds_closed_interval,
    configure_v8_inference,
)


EVALUATION_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_survival_pd_fa_v1"
)
SOURCE_BINDING_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v4_survival_"
    "evaluation_source_binding_v1"
)
DATASET = "NUDT-SIRST"
TRAINING_SEED = exact.TRAINING_SEED
SPLIT_SEED = exact.SPLIT_SEED
EXPECTED_EPOCHS = exact.FORMAL_EPOCHS
EXPECTED_VALIDATION_COUNT = 133
EXPECTED_TARGET_COUNT = 189
EXPECTED_TINY_TARGET_COUNT = 39
SUPPORTED_VARIANTS = exact.SUPPORTED_CANDIDATE_VARIANTS
CHECKPOINT_ROLES = {
    "best.pth.tar": "best_validation_pd_primary",
    "best_miou.pth.tar": "best_validation_miou_secondary",
}
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
BUDGET_KEYS = tuple(f"{budget:.10g}" for budget in FA_BUDGETS)
EXTRA_THRESHOLDS = (0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999)
BASE_EVALUATOR_PATH = REPO_ROOT / "experiments/evaluate_pd_fa_sweep.py"
DEFAULT_RESULTS_ROOT = (
    REPO_ROOT
    / "experiments/results/tpd_ner_v4_survival_exact_v1"
)
DEFAULT_RUN_DIRS = {
    exact.TSS_CONTROL_VARIANT: (
        DEFAULT_RESULTS_ROOT
        / DATASET
        / exact.TSS_CONTROL_VARIANT
        / f"seed_{TRAINING_SEED}_{exact.FORMAL_CONTROL_RUN_TAG}"
    ),
    exact.TSS_ON_VARIANT: (
        DEFAULT_RESULTS_ROOT
        / DATASET
        / exact.TSS_ON_VARIANT
        / f"seed_{TRAINING_SEED}_{exact.FORMAL_TSS_RUN_TAG}"
    ),
}
PHYSICAL_GPU_INDEX_ENV = "TPD_NER_V4_SURVIVAL_PHYSICAL_GPU_INDEX"
PHYSICAL_GPU_UUID_ENV = "TPD_NER_V4_SURVIVAL_PHYSICAL_GPU_UUID"
PHYSICAL_GPU_UUIDS = {
    str(index): uuid for index, uuid in exact.PHYSICAL_GPU_UUIDS.items()
}

# Re-export the V4 checkpoint-local result validators.  The TSS adapter uses
# exactly the same object counts, point ordering, budget key, and closed
# interval semantics.
_normalize_budgets = v4_evaluator._normalize_budgets
_validate_closed_interval = v4_evaluator._validate_closed_interval
_validate_point_collection = v4_evaluator._validate_point_collection
_final_metric_coverage = v4_evaluator._final_metric_coverage


@dataclass(frozen=True)
class EvaluationRequest:
    variant: str
    run_dir: Path
    checkpoint: str


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, observed={observed!r}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    return json.loads(
        json.dumps(
            sweep_core.json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _canonical_equal(location: str, observed: Any, expected: Any) -> None:
    if _canonical(observed) != _canonical(expected):
        raise ValueError(f"{location} differs after JSON normalization")


def verify_frozen_training_sources(
    lock_path: Path = exact.DEFAULT_EXACT_SOURCE_LOCK_PATH,
) -> dict[str, Any]:
    """Verify the write-once training lock and bind reused evaluator cores."""

    resolved = Path(lock_path).resolve()
    payload = _load_json(resolved)
    training_data_sha256 = payload.get("training_data_sha256")
    if not isinstance(training_data_sha256, str):
        raise ValueError("training source lock has no training-data digest")
    locks = exact.source_lock_contract(training_data_sha256, resolved)
    return {
        "schema": SOURCE_BINDING_SCHEMA,
        "training_source_lock": {
            "path": str(resolved),
            "schema": exact.EXACT_SOURCE_LOCK_SCHEMA,
            "sha256": locks[exact.SOURCE_LOCK_KEY],
            "training_data_sha256": locks["training_data"],
            "survival_target_statistics_sha256": locks[
                "survival_target_statistics"
            ],
            "parent_checkpoint_sha256": locks["parent_checkpoint"],
        },
        "shared_metric_core": {
            "path": str(Path(sweep_core.__file__).resolve()),
            "sha256": _sha256_file(Path(sweep_core.__file__)),
        },
        "closed_interval_core": {
            "path": str(
                REPO_ROOT / "experiments/evaluate_tpd_clean_v6_pd_fa.py"
            ),
            "sha256": _sha256_file(
                REPO_ROOT / "experiments/evaluate_tpd_clean_v6_pd_fa.py"
            ),
        },
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__)),
        },
    }


def evaluator_contract() -> dict[str, Any]:
    binding = verify_frozen_training_sources()
    return {
        "schema": EVALUATION_SCHEMA,
        "dataset": DATASET,
        "variants": list(SUPPORTED_VARIANTS),
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "expected_epochs": EXPECTED_EPOCHS,
        "checkpoint_roles": dict(CHECKPOINT_ROLES),
        "fixed_threshold": 0.5,
        "fa_budgets": list(FA_BUDGETS),
        "threshold_selection_scope": "single_checkpoint_only",
        "cross_checkpoint_point_pooling": False,
        "metric_core": "experiments.evaluate_pd_fa_sweep",
        "metric_core_sha256": binding["shared_metric_core"]["sha256"],
        "closed_interval_core": (
            "experiments.evaluate_tpd_clean_v6_pd_fa."
            "adaptive_thresholds_closed_interval"
        ),
        "closed_interval_core_sha256": binding[
            "closed_interval_core"
        ]["sha256"],
        "prediction_comparison": "prediction > threshold",
        "score_dtype": "float32",
        "last_float32_below_one": LAST_FLOAT32_BELOW_ONE,
        "upper_boundary_threshold": UPPER_BOUNDARY_THRESHOLD,
        "official_test_accessed": False,
    }


def build_model(variant: str, seed: int):
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported TSS evaluation variant: {variant!r}")
    if seed != TRAINING_SEED:
        raise ValueError("formal TSS evaluator requires seed 42")
    return exact.build_selected_model(variant, seed)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint-local internal-validation Pd/Fa sweep for TSS"
        )
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", choices=tuple(CHECKPOINT_ROLES))
    parser.add_argument("--all-four", action="store_true")
    parser.add_argument(
        "--control-run-dir",
        type=Path,
        default=DEFAULT_RUN_DIRS[exact.TSS_CONTROL_VARIANT],
    )
    parser.add_argument(
        "--tss-run-dir",
        type=Path,
        default=DEFAULT_RUN_DIRS[exact.TSS_ON_VARIANT],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-epochs", type=int, default=EXPECTED_EPOCHS)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--extra-thresholds",
        type=float,
        nargs="+",
        default=list(EXTRA_THRESHOLDS),
    )
    parser.add_argument("--tail-logit-step", type=float, default=0.1)
    parser.add_argument(
        "--fa-budgets",
        type=float,
        nargs="+",
        default=list(FA_BUDGETS),
    )
    parser.add_argument("--match-radius", type=float, default=None)
    parser.add_argument("--tiny-area", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_formal_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    args = _argument_parser().parse_args(argv)
    if args.overwrite:
        raise ValueError("formal TSS evaluator forbids --overwrite")
    if args.device not in ("cpu", "cuda:0"):
        raise ValueError("formal TSS evaluator device must be cpu or cuda:0")
    if args.expected_epochs != EXPECTED_EPOCHS:
        raise ValueError("formal TSS evaluator requires expected_epochs=800")
    if args.all_four:
        if args.run_dir is not None or args.checkpoint is not None:
            raise ValueError("--all-four cannot be combined with single-run arguments")
    else:
        if args.run_dir is None:
            raise ValueError("--run-dir is required unless --all-four is used")
        if args.checkpoint is None:
            args.checkpoint = "best.pth.tar"
    for name, observed, expected in (
        ("threshold_min", args.threshold_min, 0.01),
        ("threshold_max", args.threshold_max, 0.99),
        ("threshold_step", args.threshold_step, 0.01),
        ("extra_thresholds", tuple(args.extra_thresholds), EXTRA_THRESHOLDS),
        ("tail_logit_step", args.tail_logit_step, 0.1),
        ("fa_budgets", tuple(args.fa_budgets), FA_BUDGETS),
    ):
        _require_equal(name, observed, expected)
    if args.match_radius not in (None, exact.FORMAL_MATCH_RADIUS):
        raise ValueError("formal TSS match_radius must be omitted or 3.0")
    if args.tiny_area not in (None, exact.FORMAL_TINY_AREA):
        raise ValueError("formal TSS tiny_area must be omitted or 9")
    return args


def evaluation_requests(args: argparse.Namespace) -> tuple[EvaluationRequest, ...]:
    if args.all_four:
        requests = tuple(
            EvaluationRequest(variant, Path(run_dir).resolve(), checkpoint)
            for variant, run_dir in (
                (
                    exact.TSS_CONTROL_VARIANT,
                    args.control_run_dir,
                ),
                (exact.TSS_ON_VARIANT, args.tss_run_dir),
            )
            for checkpoint in CHECKPOINT_ROLES
        )
    else:
        run_dir = Path(args.run_dir).resolve()
        variant = run_dir.parent.name
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                "single-run directory is not under tss_control or tss_on"
            )
        requests = (
            EvaluationRequest(variant, run_dir, str(args.checkpoint)),
        )
    identities = {
        (request.variant, str(request.run_dir), request.checkpoint)
        for request in requests
    }
    if len(identities) != len(requests):
        raise ValueError("evaluation request set contains duplicate checkpoints")
    return requests


def _device_assignment(device: str) -> dict[str, Any]:
    if device == "cpu":
        return {
            "device": "cpu",
            "physical_gpu_index": None,
            "physical_gpu_uuid": None,
            "cuda_visible_devices": None,
            "device_name": "cpu",
        }
    physical_index = os.environ.get(PHYSICAL_GPU_INDEX_ENV)
    physical_uuid = os.environ.get(PHYSICAL_GPU_UUID_ENV)
    if physical_index not in PHYSICAL_GPU_UUIDS:
        raise RuntimeError("TSS evaluation physical GPU must be 2 or 3")
    expected_uuid = PHYSICAL_GPU_UUIDS[physical_index]
    if (
        physical_uuid != expected_uuid
        or os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid
    ):
        raise RuntimeError("TSS evaluation GPU UUID assignment differs")
    environment = exact.shared_exact.environment_contract(
        torch.device("cuda:0")
    )
    _require_equal(
        "visible CUDA count",
        environment.get("visible_cuda_device_count"),
        1,
    )
    _require_equal(
        "visible CUDA UUID",
        environment.get("device_uuid"),
        expected_uuid,
    )
    return {
        "device": "cuda:0",
        "physical_gpu_index": int(physical_index),
        "physical_gpu_uuid": expected_uuid,
        "cuda_visible_devices": expected_uuid,
        "device_name": environment.get("device_name"),
    }


def _require_legacy_eval_output(output: Any) -> tuple[torch.Tensor, ...]:
    if not isinstance(output, tuple) or len(output) != 6:
        raise RuntimeError(
            "TSS evaluation must return the legacy six-tensor segmentation tuple"
        )
    if not all(isinstance(value, torch.Tensor) for value in output):
        raise RuntimeError("TSS legacy evaluation output contains a non-tensor")
    return output


def validate_run_artifacts(
    run_dir: Path,
    checkpoint_name: str = "best.pth.tar",
) -> dict[str, Any]:
    """Strictly verify one completed TSS run and one owned checkpoint."""

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise NotADirectoryError(run_dir)
    if checkpoint_name not in CHECKPOINT_ROLES:
        raise ValueError("TSS evaluator accepts only best or best_miou")
    variant = run_dir.parent.name
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError("TSS run directory has an unsupported variant")
    expected_name = f"seed_{TRAINING_SEED}_{exact.FORMAL_RUN_TAGS[variant]}"
    _require_equal("TSS run-directory name", run_dir.name, expected_name)
    _require_equal("TSS run dataset directory", run_dir.parent.parent.name, DATASET)

    binding = verify_frozen_training_sources()
    protocol = _load_json(run_dir / "protocol.json")
    split = _load_json(run_dir / "split.json")
    summary = _load_json(run_dir / "summary.json")
    _require_equal("TSS protocol schema", protocol.get("schema"), exact.ENTRY_SCHEMA)
    _require_equal(
        "TSS completion schema",
        summary.get("schema"),
        exact.COMPLETION_SUMMARY_SCHEMA,
    )
    _require_equal("TSS completion status", summary.get("status"), "complete")
    _require_equal("TSS split official-test access", split.get("official_test_accessed"), False)
    _require_equal("TSS split source", split.get("source"), "img_idx/train_NUDT-SIRST.txt")
    _canonical_equal(
        "TSS protocol formal contract",
        protocol.get("formal_contract"),
        exact.formal_contract(),
    )
    arguments = protocol.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("TSS protocol arguments are missing")
    for name, expected in {
        "dataset": DATASET,
        "variant": variant,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "epochs": EXPECTED_EPOCHS,
        "eval_every": 1,
        "threshold": 0.5,
        "match_radius": 3.0,
        "tiny_area": 9,
        "run_tag": exact.FORMAL_RUN_TAGS[variant],
    }.items():
        _require_equal(f"TSS protocol argument {name}", arguments.get(name), expected)
    run_identity = exact.require_tss_run_identity(
        protocol.get("run_identity"),
        label="TSS evaluation protocol",
        expected_variant=variant,
    )
    _canonical_equal("TSS summary run identity", summary.get("run_identity"), run_identity)
    source_locks = run_identity["source_locks"]
    _require_equal(
        "TSS frozen training source lock",
        source_locks.get(exact.SOURCE_LOCK_KEY),
        binding["training_source_lock"]["sha256"],
    )

    checkpoint_path = (run_dir / checkpoint_name).resolve()
    if checkpoint_path.parent != run_dir or not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    raw_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint = exact.require_evaluator_checkpoint_payload(
        raw_checkpoint,
        expected_variant=variant,
    )
    _require_equal("TSS checkpoint file stability", _sha256_file(checkpoint_path), checkpoint_sha256)
    _require_equal(
        "TSS checkpoint role",
        checkpoint["checkpoint_role"],
        CHECKPOINT_ROLES[checkpoint_name],
    )
    _canonical_equal("TSS checkpoint run identity", checkpoint["run_identity"], run_identity)
    _canonical_equal("TSS summary split hashes", summary.get("split_hashes"), checkpoint["split_hashes"])
    for artifact_name, artifact in {
        "protocol": protocol,
        "summary": summary,
        "checkpoint": checkpoint,
    }.items():
        _require_equal(
            f"{artifact_name} official-test access",
            artifact.get("official_test_accessed"),
            False,
        )
    model, metadata = build_model(variant, TRAINING_SEED)
    _canonical_equal("TSS rebuilt/checkpoint metadata", metadata, checkpoint["model_metadata"])
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    _require_equal("TSS strict-load missing keys", list(incompatible.missing_keys), [])
    _require_equal("TSS strict-load unexpected keys", list(incompatible.unexpected_keys), [])
    model.eval()
    if model.training:
        raise RuntimeError("TSS evaluator failed to switch the model to eval mode")

    audit = {
        "run_directory": str(run_dir),
        "variant": variant,
        "checkpoint_filename": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_role": checkpoint["checkpoint_role"],
        "checkpoint_validation_metrics": copy.deepcopy(
            checkpoint["validation_metrics"]
        ),
        "checkpoint_identity": copy.deepcopy(
            checkpoint["checkpoint_identity"]
        ),
        "run_identity": copy.deepcopy(run_identity),
        "source_binding": binding,
        "state_dict_strict_load": True,
        "legacy_eval_output_verified": False,
    }
    # ``require_evaluator_checkpoint_payload`` deliberately deep-copies the
    # checkpoint before validating it.  Release both tensor graphs and the
    # rebuilt model here so --all-four has bounded host-memory use.
    del model, checkpoint, raw_checkpoint
    gc.collect()
    return audit


def _require_checkpoint_unchanged(
    audit: Mapping[str, Any],
    *,
    stage: str,
) -> None:
    observed = _sha256_file(Path(str(audit["checkpoint_path"])))
    _require_equal(f"checkpoint SHA {stage}", observed, audit["checkpoint_sha256"])


def finalize_evaluation_output(
    payload: Mapping[str, Any],
    artifact_audit: Mapping[str, Any],
    *,
    device_assignment: Mapping[str, Any],
) -> dict[str, Any]:
    ready = copy.deepcopy(dict(payload))
    checkpoint_metrics = artifact_audit["checkpoint_validation_metrics"]
    fixed = _validate_point_collection(ready, checkpoint_metrics)
    budgets = _normalize_budgets(ready)
    _validate_closed_interval(ready)
    if not artifact_audit.get("legacy_eval_output_verified"):
        raise ValueError("legacy eval-output guard did not observe a forward pass")
    for name, expected in {
        "variant": artifact_audit["variant"],
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "checkpoint_role": artifact_audit["checkpoint_role"],
        "checkpoint_epoch": artifact_audit["checkpoint_epoch"],
        "checkpoint_sha256": artifact_audit["checkpoint_sha256"],
        "official_test_accessed": False,
    }.items():
        _require_equal(f"TSS evaluation {name}", ready.get(name), expected)
    ready.update(
        {
            "schema": EVALUATION_SCHEMA,
            "run_identity": copy.deepcopy(artifact_audit["run_identity"]),
            "source_checkpoint_identity": copy.deepcopy(
                artifact_audit["checkpoint_identity"]
            ),
            "evaluation_source_binding": copy.deepcopy(
                artifact_audit["source_binding"]
            ),
            "evaluator_contract": evaluator_contract(),
            "threshold_selection_scope": "single_checkpoint_only",
            "cross_checkpoint_point_pooling": False,
            "evaluated_checkpoint_count": 1,
            "final_metric_coverage": _final_metric_coverage(fixed, budgets),
        }
    )
    audit = dict(ready["audit"])
    audit["device_assignment"] = copy.deepcopy(dict(device_assignment))
    audit["tss_checkpoint_payload_strict"] = True
    audit["tss_state_dict_strict_load"] = True
    audit["legacy_six_tensor_eval_output"] = True
    ready["audit"] = audit
    validate_output_identity(ready, artifact_audit=artifact_audit)
    return ready


def validate_output_identity(
    payload: Mapping[str, Any],
    *,
    artifact_audit: Mapping[str, Any],
) -> None:
    for name, expected in {
        "schema": EVALUATION_SCHEMA,
        "variant": artifact_audit["variant"],
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "threshold_selection_scope": "single_checkpoint_only",
        "cross_checkpoint_point_pooling": False,
        "evaluated_checkpoint_count": 1,
        "official_test_accessed": False,
    }.items():
        _require_equal(f"TSS output {name}", payload.get(name), expected)
    _canonical_equal("TSS output run identity", payload.get("run_identity"), artifact_audit["run_identity"])
    _canonical_equal(
        "TSS output checkpoint identity",
        payload.get("source_checkpoint_identity"),
        artifact_audit["checkpoint_identity"],
    )
    fixed = _validate_point_collection(
        payload,
        artifact_audit["checkpoint_validation_metrics"],
    )
    budgets = _normalize_budgets(payload)
    _validate_closed_interval(payload)
    _canonical_equal(
        "TSS output metric coverage",
        payload.get("final_metric_coverage"),
        _final_metric_coverage(fixed, budgets),
    )


def _atomic_write_output(
    path: Path,
    payload: Mapping[str, Any],
    overwrite: bool,
    *,
    artifact_audit: Mapping[str, Any],
    device_assignment: Mapping[str, Any],
    json_ready,
) -> None:
    if overwrite:
        raise ValueError("formal TSS evaluator forbids overwrite")
    ready = json_ready(
        finalize_evaluation_output(
            payload,
            artifact_audit,
            device_assignment=device_assignment,
        )
    )
    content = (
        json.dumps(ready, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to replace existing TSS sweep: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _base_namespace(
    args: argparse.Namespace,
    request: EvaluationRequest,
) -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=request.run_dir,
        checkpoint=request.checkpoint,
        device=args.device,
        expected_epochs=EXPECTED_EPOCHS,
        threshold_min=0.01,
        threshold_max=0.99,
        threshold_step=0.01,
        extra_thresholds=list(EXTRA_THRESHOLDS),
        tail_logit_step=0.1,
        fa_budgets=list(FA_BUDGETS),
        match_radius=None,
        tiny_area=None,
        overwrite=False,
    )


def _load_isolated_base_evaluator(
    args: argparse.Namespace,
    request: EvaluationRequest,
    artifact_audit: dict[str, Any],
    device_assignment: Mapping[str, Any],
) -> ModuleType:
    module_name = (
        "_sctransnet_tss_pd_fa_"
        f"{request.variant}_{Path(request.checkpoint).stem}"
    )
    spec = importlib.util.spec_from_file_location(module_name, BASE_EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the shared Pd/Fa evaluator")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    namespace = _base_namespace(args, request)
    original_collect = evaluator.collect_predictions

    def bound_parse_args() -> argparse.Namespace:
        return argparse.Namespace(**vars(namespace))

    def bound_collect_predictions(model, loader, device):
        observed = {"legacy": False}

        def guard(_module, _inputs, output):
            _require_legacy_eval_output(output)
            observed["legacy"] = True

        hook = model.register_forward_hook(guard)
        try:
            result = original_collect(model, loader, device)
        finally:
            hook.remove()
        if not observed["legacy"]:
            raise RuntimeError("TSS evaluator observed no model forward pass")
        artifact_audit["legacy_eval_output_verified"] = True
        return result

    def bound_write(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
        _atomic_write_output(
            path,
            payload,
            overwrite,
            artifact_audit=artifact_audit,
            device_assignment=device_assignment,
            json_ready=evaluator.json_ready,
        )

    evaluator.adaptive_thresholds = adaptive_thresholds_closed_interval
    evaluator.build_model = build_model
    evaluator.collect_predictions = bound_collect_predictions
    evaluator.parse_args = bound_parse_args
    evaluator.write_output_json = bound_write
    evaluator.__file__ = __file__
    return evaluator


def evaluate_one(
    args: argparse.Namespace,
    request: EvaluationRequest,
    *,
    device_assignment: Mapping[str, Any],
    artifact_audit: dict[str, Any] | None = None,
) -> Path:
    audit = (
        validate_run_artifacts(request.run_dir, request.checkpoint)
        if artifact_audit is None
        else artifact_audit
    )
    _require_equal("request/preflight variant", request.variant, audit["variant"])
    output = request.run_dir / (
        f"pd_fa_sweep_{Path(request.checkpoint).stem}.json"
    )
    if output.exists():
        raise FileExistsError(f"refusing to replace existing TSS sweep: {output}")
    evaluator = _load_isolated_base_evaluator(
        args,
        request,
        audit,
        device_assignment,
    )
    per_checkpoint_argv = [
        str(Path(__file__).resolve()),
        "--run-dir",
        str(request.run_dir),
        "--checkpoint",
        request.checkpoint,
        "--device",
        args.device,
        "--expected-epochs",
        str(EXPECTED_EPOCHS),
    ]
    original_argv = sys.argv
    sys.argv = per_checkpoint_argv
    try:
        _require_checkpoint_unchanged(audit, stage="before shared evaluator")
        evaluator.main()
        _require_checkpoint_unchanged(audit, stage="after shared evaluator")
    finally:
        sys.argv = original_argv
    if not output.is_file():
        raise RuntimeError(f"TSS evaluator did not create {output}")
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = validate_formal_arguments(argv)
    requests = evaluation_requests(args)
    configure_v8_inference(args.device)
    assignment = _device_assignment(args.device)

    # Preflight all requested checkpoints before producing any output.
    audits = [
        validate_run_artifacts(request.run_dir, request.checkpoint)
        for request in requests
    ]
    for request in requests:
        output = request.run_dir / (
            f"pd_fa_sweep_{Path(request.checkpoint).stem}.json"
        )
        if output.exists():
            raise FileExistsError(
                f"refusing to replace existing TSS sweep: {output}"
            )
    for request, audit in zip(requests, audits):
        evaluate_one(
            args,
            request,
            device_assignment=assignment,
            artifact_audit=audit,
        )


__all__ = [
    "BUDGET_KEYS",
    "CHECKPOINT_ROLES",
    "DEFAULT_RUN_DIRS",
    "EVALUATION_SCHEMA",
    "EvaluationRequest",
    "FA_BUDGETS",
    "SUPPORTED_VARIANTS",
    "build_model",
    "evaluate_one",
    "evaluation_requests",
    "evaluator_contract",
    "finalize_evaluation_output",
    "main",
    "validate_formal_arguments",
    "validate_output_identity",
    "validate_run_artifacts",
    "verify_frozen_training_sources",
]


if __name__ == "__main__":
    main()
