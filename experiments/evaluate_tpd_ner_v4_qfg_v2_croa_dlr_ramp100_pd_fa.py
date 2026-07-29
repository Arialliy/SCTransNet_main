#!/usr/bin/env python3
"""Own-checkpoint closed-interval Pd/Fa evaluation for paired ramp100 runs.

This evaluator owns the two ``qfg_dlr``/``tss_qfg_dlr`` variants produced by
``train_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact.py``.  Each validation-owned
``best`` or ``best_miou`` checkpoint is swept independently.  Threshold
points and Fa-budget selections never cross checkpoint boundaries.

Existing output is accepted only after a full live revalidation.  New output
is installed atomically and write-once; there is no overwrite mode.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
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
    evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa as qfg_evaluator,
)
from experiments import (  # noqa: E402
    train_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_exact as exact,
)
from experiments.evaluate_tpd_clean_v8_mprs_dch_pd_fa import (  # noqa: E402
    LAST_FLOAT32_BELOW_ONE,
    UPPER_BOUNDARY_THRESHOLD,
    adaptive_thresholds_closed_interval,
    configure_v8_inference,
)


EVALUATION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "checkpoint_local_pd_fa_v1"
)
SOURCE_BINDING_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
    "evaluation_source_binding_v1"
)
DATASET = "NUDT-SIRST"
TRAINING_SEED = exact.TRAINING_SEED
SPLIT_SEED = exact.SPLIT_SEED
EXPECTED_EPOCHS = exact.FORMAL_EPOCHS
EXPECTED_VALIDATION_COUNT = 133
EXPECTED_TARGET_COUNT = 189
EXPECTED_TINY_TARGET_COUNT = 39
SUPPORTED_VARIANTS = tuple(exact.SUPPORTED_CANDIDATE_VARIANTS)
QFG_DLR_VARIANT = exact.QFG_DLR_VARIANT
TSS_QFG_DLR_VARIANT = exact.TSS_QFG_DLR_VARIANT
if set(SUPPORTED_VARIANTS) != {
    QFG_DLR_VARIANT,
    TSS_QFG_DLR_VARIANT,
}:
    raise RuntimeError("ramp100 evaluator requires exactly the paired arms")

CHECKPOINT_ROLES = {
    "best.pth.tar": "best_validation_pd_primary",
    "best_miou.pth.tar": "best_validation_miou_secondary",
}
CHECKPOINT_SLOTS = {
    "best.pth.tar": "primary",
    "best_miou.pth.tar": "secondary",
}
SUMMARY_CHECKPOINT_KEYS = {
    "best.pth.tar": "best",
    "best_miou.pth.tar": "best_miou",
}
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
BUDGET_KEYS = tuple(f"{budget:.10g}" for budget in FA_BUDGETS)
FIXED_THRESHOLD = exact.FORMAL_THRESHOLD
FORMAL_MATCH_RADIUS = exact.FORMAL_MATCH_RADIUS
FORMAL_TINY_AREA = exact.FORMAL_TINY_AREA
EXTRA_THRESHOLDS = (0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999)
BASE_EVALUATOR_PATH = REPO_ROOT / "experiments/evaluate_pd_fa_sweep.py"
CLOSED_INTERVAL_CORE_PATH = (
    REPO_ROOT / "experiments/evaluate_tpd_clean_v6_pd_fa.py"
)
DETERMINISM_CORE_PATH = (
    REPO_ROOT / "experiments/evaluate_tpd_clean_v8_mprs_dch_pd_fa.py"
)
DEFAULT_RESULTS_ROOT = exact.DEFAULT_OUTPUT_ROOT
DEFAULT_RUN_DIRS = {
    QFG_DLR_VARIANT: (
        DEFAULT_RESULTS_ROOT
        / "qfg_dlr_lane"
        / DATASET
        / QFG_DLR_VARIANT
        / (
            f"seed_{TRAINING_SEED}_"
            f"{exact.FORMAL_RUN_TAGS[QFG_DLR_VARIANT]}"
        )
    ),
    TSS_QFG_DLR_VARIANT: (
        DEFAULT_RESULTS_ROOT
        / "tss_qfg_dlr_lane"
        / DATASET
        / TSS_QFG_DLR_VARIANT
        / (
            f"seed_{TRAINING_SEED}_"
            f"{exact.FORMAL_RUN_TAGS[TSS_QFG_DLR_VARIANT]}"
        )
    ),
}
PHYSICAL_GPU_INDEX_ENV = "TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX"
PHYSICAL_GPU_UUID_ENV = "TPD_NER_V4_QFG_PHYSICAL_GPU_UUID"

_normalize_budgets = qfg_evaluator._normalize_budgets
_validate_closed_interval = qfg_evaluator._validate_closed_interval
_validate_point_collection = qfg_evaluator._validate_point_collection
_final_metric_coverage = qfg_evaluator._final_metric_coverage
_require_legacy_eval_output = qfg_evaluator._require_legacy_eval_output
_canonical = qfg_evaluator._canonical
_canonical_equal = qfg_evaluator._canonical_equal


@dataclass(frozen=True)
class EvaluationRequest:
    variant: str
    run_dir: Path
    checkpoint: str


def _require_equal(location: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(
            f"{location} differs: expected={expected!r}, "
            f"observed={observed!r}"
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


def _repo_relative(path: Path, label: str) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError as exc:
        raise ValueError(f"{label} lies outside the repository") from exc


def _candidate_contract(variant: str) -> dict[str, Any]:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported paired ramp100 candidate: {variant!r}")
    contract = exact.candidate_contract(variant)
    if not isinstance(contract, Mapping):
        raise RuntimeError("ramp100 candidate contract is not a mapping")
    return copy.deepcopy(dict(contract))


def _candidate_run_tag(variant: str) -> str:
    run_tag = _candidate_contract(variant).get("formal_run_tag")
    if not isinstance(run_tag, str) or not run_tag:
        raise RuntimeError(f"ramp100 candidate {variant!r} has no run tag")
    return run_tag


def verify_frozen_training_sources() -> dict[str, Any]:
    """Live-verify the frozen 51-source trainer lock without changing it."""

    lock_path = Path(exact.DEFAULT_EXACT_SOURCE_LOCK_PATH).resolve()
    lock = _load_json(lock_path)
    training_data_sha256 = lock.get("training_data_sha256")
    if not isinstance(training_data_sha256, str):
        raise ValueError("ramp100 source lock lacks training_data_sha256")
    live_locks = exact.source_lock_contract(
        training_data_sha256,
        lock_path,
        exact.DEFAULT_TARGET_STATISTICS_PATH,
    )
    _require_equal(
        "ramp100 live source-lock digest",
        live_locks.get(exact.SOURCE_LOCK_KEY),
        _sha256_file(lock_path),
    )
    return {
        "path": str(lock_path),
        "relative_path": _repo_relative(lock_path, "training source lock"),
        "sha256": _sha256_file(lock_path),
        "schema": lock.get("schema"),
        "source_count": lock.get("source_count"),
        "training_data_sha256": training_data_sha256,
        "live_source_locks": live_locks,
    }


def source_binding() -> dict[str, Any]:
    """Bind every executable evaluation source to the frozen trainer."""

    training_lock = verify_frozen_training_sources()
    source_paths = {
        "trainer": Path(exact.__file__).resolve(),
        "evaluator": Path(__file__).resolve(),
        "shared_metric_core": Path(sweep_core.__file__).resolve(),
        "closed_interval_core": CLOSED_INTERVAL_CORE_PATH.resolve(),
        "determinism_core": DETERMINISM_CORE_PATH.resolve(),
    }
    records: dict[str, Any] = {}
    for name, path in source_paths.items():
        records[name] = {
            "path": str(path),
            "relative_path": _repo_relative(path, name),
            "sha256": _sha256_file(path),
        }
    return {
        "schema": SOURCE_BINDING_SCHEMA,
        "training_source_lock": training_lock,
        **records,
    }


def evaluator_contract() -> dict[str, Any]:
    binding = source_binding()
    return {
        "schema": EVALUATION_SCHEMA,
        "dataset": DATASET,
        "variants": list(SUPPORTED_VARIANTS),
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "expected_epochs": EXPECTED_EPOCHS,
        "expected_validation_count": EXPECTED_VALIDATION_COUNT,
        "checkpoint_roles": dict(CHECKPOINT_ROLES),
        "fixed_threshold": FIXED_THRESHOLD,
        "fa_budgets": list(FA_BUDGETS),
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "threshold_selection_scope": "single_checkpoint_only",
        "cross_checkpoint_point_pooling": False,
        "cross_checkpoint_overwrite": False,
        "expected_sweep_count": 4,
        "prediction_comparison": "prediction > threshold",
        "score_dtype": "float32",
        "last_float32_below_one": LAST_FLOAT32_BELOW_ONE,
        "upper_boundary_threshold": UPPER_BOUNDARY_THRESHOLD,
        "trainer_formal_contract": exact.formal_contract(),
        "training_source_lock_sha256": binding[
            "training_source_lock"
        ]["sha256"],
        "metric_core_sha256": binding["shared_metric_core"]["sha256"],
        "closed_interval_core_sha256": binding[
            "closed_interval_core"
        ]["sha256"],
        "official_test_accessed": False,
    }


def build_model(variant: str, seed: int):
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported ramp100 variant: {variant!r}")
    if seed != TRAINING_SEED:
        raise ValueError("formal ramp100 evaluator requires seed 42")
    return exact.build_selected_model(variant, seed)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        choices=tuple(CHECKPOINT_ROLES),
        default="best.pth.tar",
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
    parser.add_argument(
        "--match-radius",
        type=float,
        default=FORMAL_MATCH_RADIUS,
    )
    parser.add_argument("--tiny-area", type=int, default=FORMAL_TINY_AREA)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate sources/run/checkpoint only; do not touch CUDA or output.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_formal_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    args = _argument_parser().parse_args(argv)
    if args.overwrite:
        raise ValueError("formal ramp100 evaluator forbids --overwrite")
    if args.device not in ("cpu", "cuda:0"):
        raise ValueError("formal ramp100 device must be cpu or cuda:0")
    for name, observed, expected in (
        ("expected_epochs", args.expected_epochs, EXPECTED_EPOCHS),
        ("threshold_min", args.threshold_min, 0.01),
        ("threshold_max", args.threshold_max, 0.99),
        ("threshold_step", args.threshold_step, 0.01),
        ("extra_thresholds", tuple(args.extra_thresholds), EXTRA_THRESHOLDS),
        ("tail_logit_step", args.tail_logit_step, 0.1),
        ("fa_budgets", tuple(args.fa_budgets), FA_BUDGETS),
        ("match_radius", args.match_radius, FORMAL_MATCH_RADIUS),
        ("tiny_area", args.tiny_area, FORMAL_TINY_AREA),
    ):
        _require_equal(name, observed, expected)
    return args


def evaluation_request(args: argparse.Namespace) -> EvaluationRequest:
    run_dir = Path(args.run_dir).resolve()
    variant = run_dir.parent.name
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(
            "run directory must belong to qfg_dlr or tss_qfg_dlr"
        )
    return EvaluationRequest(variant, run_dir, str(args.checkpoint))


def _physical_gpu_uuids() -> dict[str, str]:
    raw = getattr(exact.v2, "PHYSICAL_GPU_UUIDS", {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def expected_physical_gpu(variant: str) -> int:
    return 2 if variant == QFG_DLR_VARIANT else 3


def _device_assignment(
    device: str,
    *,
    variant: str,
) -> dict[str, Any]:
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
    expected_index = str(expected_physical_gpu(variant))
    uuids = _physical_gpu_uuids()
    expected_uuid = uuids.get(expected_index)
    if physical_index != expected_index or expected_uuid is None:
        raise RuntimeError(
            f"{variant} evaluation must use physical GPU {expected_index}"
        )
    if (
        physical_uuid != expected_uuid
        or os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid
    ):
        raise RuntimeError("ramp100 evaluation GPU UUID assignment differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("ramp100 evaluator requires one visible CUDA device")
    device_name = torch.cuda.get_device_name(0)
    if device_name != "NVIDIA GeForce RTX 5090":
        raise RuntimeError(f"unexpected CUDA device: {device_name}")
    return {
        "device": "cuda:0",
        "physical_gpu_index": int(expected_index),
        "physical_gpu_uuid": expected_uuid,
        "cuda_visible_devices": expected_uuid,
        "device_name": device_name,
    }


def _require_checkpoint_payload(
    payload: Any,
    *,
    expected_variant: str,
) -> dict[str, Any]:
    """Validate a derived checkpoint by replaying the trainer adapter."""

    if not isinstance(payload, Mapping):
        raise ValueError("ramp100 evaluator checkpoint is not a mapping")
    value = dict(payload)
    identity = exact.require_paired_run_identity(
        value.get("run_identity"),
        label="ramp100 evaluator checkpoint",
        expected_variant=expected_variant,
    )
    epoch = value.get("epoch")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or not 1 <= epoch <= EXPECTED_EPOCHS
    ):
        raise ValueError("ramp100 evaluator checkpoint epoch is invalid")
    role = value.get("checkpoint_role")
    if role not in {
        "best_validation_pd_primary",
        "best_validation_miou_secondary",
    }:
        raise ValueError("ramp100 evaluator checkpoint role is invalid")
    state = value.get("state_dict")
    optimizer = value.get("optimizer")
    scaler = value.get("scaler")
    for name, component in (
        ("state_dict", state),
        ("optimizer", optimizer),
        ("scaler", scaler),
    ):
        if not isinstance(component, Mapping):
            raise ValueError(f"ramp100 checkpoint {name} is invalid")
    metrics = exact.v2._require_complete_validation_metrics(
        value.get("validation_metrics")
    )
    metadata = exact._require_model_metadata(
        value.get("model_metadata"),
        variant=expected_variant,
    )
    split_hashes = value.get("split_hashes")
    if not isinstance(split_hashes, Mapping) or not split_hashes:
        raise ValueError("ramp100 checkpoint split hashes are invalid")
    for name, digest in split_hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("ramp100 checkpoint split-hash name is invalid")
        exact.v2._validate_sha256(digest, f"ramp100 split hash {name}")

    adapter = exact.EvaluatorCheckpointAdapter(
        model_metadata=metadata,
        split_hashes=split_hashes,
    )
    context = exact.exact_runner.CompatibilityPayloadContext(
        role=role,
        epoch=epoch,
        metrics=copy.deepcopy(metrics),
        event={},
        exact_payload={
            "model": {"state_dict": state},
            "optimizer": {"state_dict": optimizer},
            "scaler": {"state_dict": scaler},
        },
        run_identity=identity,
        normalized_spec={},
    )
    expected = dict(adapter(context))
    runner_fields = {
        "derived_schema",
        "source_exact_checkpoint_sha256",
        "state_dict_sha256",
        "optimizer_state_sha256",
        "scaler_state_sha256",
    }
    _require_equal(
        "ramp100 checkpoint strict field set",
        set(value),
        set(expected) | runner_fields,
    )
    for name, expected_value in expected.items():
        if not exact.exact_runner._state_values_equal(
            value.get(name),
            expected_value,
        ):
            raise ValueError(
                f"ramp100 checkpoint adapter field differs: {name}"
            )
    _require_equal(
        "ramp100 derived checkpoint schema",
        value.get("derived_schema"),
        exact.exact_runner.DERIVED_CHECKPOINT_SCHEMA,
    )
    exact.v2._validate_sha256(
        value.get("source_exact_checkpoint_sha256"),
        "ramp100 source exact checkpoint SHA-256",
    )
    for field, component, label in (
        ("state_dict_sha256", state, "state_dict"),
        ("optimizer_state_sha256", optimizer, "optimizer"),
        ("scaler_state_sha256", scaler, "scaler"),
    ):
        _require_equal(
            f"ramp100 checkpoint {field}",
            value.get(field),
            exact.exact_runner._state_content_sha256(
                component,
                f"ramp100 checkpoint {label}",
            ),
        )
    return value


def _summary_checkpoint_record(
    summary: Mapping[str, Any],
    checkpoint_name: str,
) -> Mapping[str, Any]:
    checkpoints = summary.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ValueError("ramp100 summary checkpoint records are missing")
    record = checkpoints.get(SUMMARY_CHECKPOINT_KEYS[checkpoint_name])
    if not isinstance(record, Mapping):
        raise ValueError("ramp100 selected checkpoint record is missing")
    return record


def validate_run_artifacts(
    run_dir: Path,
    checkpoint_name: str = "best.pth.tar",
) -> dict[str, Any]:
    """Bind a complete trajectory to one of its own selected checkpoints."""

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise NotADirectoryError(run_dir)
    if checkpoint_name not in CHECKPOINT_ROLES:
        raise ValueError("ramp100 evaluator accepts only best or best_miou")
    variant = run_dir.parent.name
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError("ramp100 run has an unsupported variant")
    expected_name = (
        f"seed_{TRAINING_SEED}_{_candidate_run_tag(variant)}"
    )
    _require_equal("ramp100 run-directory name", run_dir.name, expected_name)
    _require_equal(
        "ramp100 run dataset directory",
        run_dir.parent.parent.name,
        DATASET,
    )

    protocol = _load_json(run_dir / "protocol.json")
    split = _load_json(run_dir / "split.json")
    summary = _load_json(run_dir / "summary.json")
    metrics_path = run_dir / exact.exact_runner.METRICS_FILENAME
    formal_contract = exact.formal_contract()
    candidate = _candidate_contract(variant)
    for artifact_name, artifact, schema in (
        ("protocol", protocol, exact.ENTRY_SCHEMA),
        ("summary", summary, exact.COMPLETION_SUMMARY_SCHEMA),
    ):
        _require_equal(
            f"ramp100 {artifact_name} schema",
            artifact.get("schema"),
            schema,
        )
        _canonical_equal(
            f"ramp100 {artifact_name} formal contract",
            artifact.get("formal_contract"),
            formal_contract,
        )
        for name, expected in {
            "variant": variant,
            "candidate_variant": variant,
            "base_model_variant": candidate["base_model_variant"],
            "qfg_variant": candidate["qfg_variant"],
            "tss_variant": candidate["tss_variant"],
            "family_recipe": exact.FAMILY_RECIPE,
            "candidate_recipe": candidate["candidate_recipe"],
            "official_test_accessed": False,
        }.items():
            _require_equal(
                f"ramp100 {artifact_name} {name}",
                artifact.get(name),
                expected,
            )
    _require_equal("ramp100 summary status", summary.get("status"), "complete")
    _require_equal(
        "ramp100 split official-test access",
        split.get("official_test_accessed"),
        False,
    )
    _require_equal(
        "ramp100 split source",
        split.get("source"),
        "img_idx/train_NUDT-SIRST.txt",
    )

    arguments = protocol.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("ramp100 protocol arguments are missing")
    for name, expected in {
        "dataset": DATASET,
        "variant": variant,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "epochs": EXPECTED_EPOCHS,
        "eval_every": 1,
        "threshold": FIXED_THRESHOLD,
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "run_tag": _candidate_run_tag(variant),
        "qfg_variant": candidate["qfg_variant"],
        "tss_variant": candidate["tss_variant"],
        "family_recipe": exact.FAMILY_RECIPE,
        "candidate_recipe": candidate["candidate_recipe"],
    }.items():
        _require_equal(
            f"ramp100 protocol argument {name}",
            arguments.get(name),
            expected,
        )
    _canonical_equal(
        "ramp100 protocol survival schedule",
        protocol.get("survival_weight_schedule"),
        exact.survival_schedule_contract(variant),
    )
    _canonical_equal(
        "ramp100 summary survival schedule",
        summary.get("survival_weight_schedule"),
        exact.survival_schedule_contract(variant),
    )

    run_identity = exact.require_paired_run_identity(
        protocol.get("run_identity"),
        label="ramp100 evaluation protocol",
        expected_variant=variant,
    )
    _canonical_equal(
        "ramp100 summary run identity",
        summary.get("run_identity"),
        run_identity,
    )
    _canonical_equal(
        "ramp100 protocol source locks",
        protocol.get("source_locks"),
        run_identity["source_locks"],
    )
    _canonical_equal(
        "ramp100 summary source locks",
        summary.get("source_locks"),
        run_identity["source_locks"],
    )
    live_locks = exact.source_lock_contract(
        run_identity["source_locks"]["training_data"],
        exact.DEFAULT_EXACT_SOURCE_LOCK_PATH,
        exact.DEFAULT_TARGET_STATISTICS_PATH,
    )
    _canonical_equal(
        "ramp100 run/live source locks",
        run_identity["source_locks"],
        live_locks,
    )

    events = exact._load_complete_events(
        metrics_path,
        EXPECTED_EPOCHS,
        variant=variant,
    )
    policy = exact.exact_runner.pd_miou_selection_policy(
        stored_metrics=exact.STORED_VALIDATION_METRICS,
    )
    selection = policy.recompute(events, require_flags=True)
    slot = CHECKPOINT_SLOTS[checkpoint_name]
    selected = selection[slot]
    if slot == "primary":
        for name, expected in {
            "best_epoch": selected["epoch"],
            "best_pd_epoch": selected["epoch"],
        }.items():
            _require_equal(
                f"ramp100 summary {name}",
                summary.get(name),
                expected,
            )
        _canonical_equal(
            "ramp100 summary best metrics",
            summary.get("best_validation_metrics"),
            selected["metrics"],
        )
        _canonical_equal(
            "ramp100 summary best-Pd metrics",
            summary.get("best_pd_validation_metrics"),
            selected["metrics"],
        )
    else:
        _require_equal(
            "ramp100 summary best_miou_epoch",
            summary.get("best_miou_epoch"),
            selected["epoch"],
        )
        _canonical_equal(
            "ramp100 summary best-mIoU metrics",
            summary.get("best_miou_validation_metrics"),
            selected["metrics"],
        )

    checkpoint_path = (run_dir / checkpoint_name).resolve()
    if checkpoint_path.parent != run_dir or not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if checkpoint_path.is_symlink():
        raise ValueError("ramp100 checkpoint must be a regular file")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    raw_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint = _require_checkpoint_payload(
        raw_checkpoint,
        expected_variant=variant,
    )
    _require_equal(
        "ramp100 checkpoint stability after load",
        _sha256_file(checkpoint_path),
        checkpoint_sha256,
    )
    _require_equal(
        "ramp100 checkpoint role",
        checkpoint["checkpoint_role"],
        CHECKPOINT_ROLES[checkpoint_name],
    )
    _require_equal(
        "ramp100 checkpoint own-selected epoch",
        checkpoint["epoch"],
        selected["epoch"],
    )
    _canonical_equal(
        "ramp100 checkpoint own-selected metrics",
        checkpoint["validation_metrics"],
        selected["metrics"],
    )
    _canonical_equal(
        "ramp100 checkpoint run identity",
        checkpoint["run_identity"],
        run_identity,
    )
    _canonical_equal(
        "ramp100 summary/checkpoint split hashes",
        summary.get("split_hashes"),
        checkpoint["split_hashes"],
    )
    recomputed_split_hashes = sweep_core.validate_identifier_manifest(split)
    _canonical_equal(
        "ramp100 split/summary hashes",
        split.get("hashes"),
        summary.get("split_hashes"),
    )
    _canonical_equal(
        "ramp100 recomputed/checkpoint split hashes",
        recomputed_split_hashes,
        checkpoint["split_hashes"],
    )
    _require_equal(
        "ramp100 validation split count",
        split.get("used_val_count"),
        EXPECTED_VALIDATION_COUNT,
    )
    validation_ids = split.get("used_val_ids")
    if not isinstance(validation_ids, list):
        raise ValueError("ramp100 validation identifiers are missing")
    _require_equal(
        "ramp100 validation identifier count",
        len(validation_ids),
        EXPECTED_VALIDATION_COUNT,
    )
    record = _summary_checkpoint_record(summary, checkpoint_name)
    _require_equal(
        "ramp100 summary checkpoint path",
        Path(str(record.get("path"))).resolve(),
        checkpoint_path,
    )
    for name, observed, expected in (
        ("SHA", record.get("sha256"), checkpoint_sha256),
        ("epoch", record.get("epoch"), selected["epoch"]),
        (
            "role",
            record.get("role"),
            CHECKPOINT_ROLES[checkpoint_name],
        ),
    ):
        _require_equal(
            f"ramp100 summary checkpoint {name}",
            observed,
            expected,
        )

    model, metadata = build_model(variant, TRAINING_SEED)
    _canonical_equal(
        "ramp100 rebuilt/checkpoint metadata",
        metadata,
        checkpoint["model_metadata"],
    )
    incompatible = model.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )
    _require_equal(
        "ramp100 strict-load missing keys",
        list(incompatible.missing_keys),
        [],
    )
    _require_equal(
        "ramp100 strict-load unexpected keys",
        list(incompatible.unexpected_keys),
        [],
    )
    if variant == QFG_DLR_VARIANT:
        exact._require_zero_tss_state(
            checkpoint["state_dict"],
            label="evaluated qfg_dlr checkpoint",
        )
    model.eval()
    if model.training:
        raise RuntimeError("ramp100 evaluator failed to switch model to eval")

    binding = source_binding()
    audit = {
        "run_directory": str(run_dir),
        "variant": variant,
        "candidate_contract": candidate,
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
        "checkpoint_survival_weight_effective": checkpoint[
            exact.SURVIVAL_WEIGHT_FIELD
        ],
        "checkpoint_tss_ramp_fraction": checkpoint[
            exact.TSS_RAMP_FRACTION_FIELD
        ],
        "validation_split_sha256": checkpoint["split_hashes"][
            "used_val_sha256"
        ],
        "run_identity": copy.deepcopy(run_identity),
        "selection": copy.deepcopy(selection),
        "source_binding": binding,
        "state_dict_strict_load": True,
        "adapter_payload_strict": True,
        "legacy_eval_output_verified": False,
    }
    del model, checkpoint, raw_checkpoint, events
    gc.collect()
    return audit


def _require_checkpoint_unchanged(
    audit: Mapping[str, Any],
    *,
    stage: str,
) -> None:
    _require_equal(
        f"ramp100 checkpoint SHA {stage}",
        _sha256_file(Path(str(audit["checkpoint_path"]))),
        audit["checkpoint_sha256"],
    )


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
    qfg_evaluator.v4_evaluator._validate_standard_audit(ready["audit"])
    if not artifact_audit.get("legacy_eval_output_verified"):
        raise ValueError("ramp100 legacy eval-output guard saw no forward")
    candidate = artifact_audit["candidate_contract"]
    for name, expected in {
        "variant": artifact_audit["variant"],
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "checkpoint_role": artifact_audit["checkpoint_role"],
        "checkpoint_epoch": artifact_audit["checkpoint_epoch"],
        "checkpoint_sha256": artifact_audit["checkpoint_sha256"],
        "validation_count": EXPECTED_VALIDATION_COUNT,
        "validation_split_sha256": artifact_audit[
            "validation_split_sha256"
        ],
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "official_test_accessed": False,
    }.items():
        _require_equal(f"ramp100 evaluation {name}", ready.get(name), expected)
    live_binding = source_binding()
    _canonical_equal(
        "ramp100 preflight/final source binding",
        artifact_audit["source_binding"],
        live_binding,
    )
    ready.update(
        {
            "schema": EVALUATION_SCHEMA,
            "base_model_variant": candidate["base_model_variant"],
            "family_recipe": exact.FAMILY_RECIPE,
            "candidate_recipe": candidate["candidate_recipe"],
            "survival_weight_schedule": exact.survival_schedule_contract(
                artifact_audit["variant"]
            ),
            "checkpoint_survival_weight_effective": artifact_audit[
                "checkpoint_survival_weight_effective"
            ],
            "checkpoint_tss_ramp_fraction": artifact_audit[
                "checkpoint_tss_ramp_fraction"
            ],
            "optimizer_recipe": exact.optimizer_recipe_contract(),
            "batchnorm_recipe": exact.batchnorm_recipe_contract(),
            "run_identity": copy.deepcopy(artifact_audit["run_identity"]),
            "source_checkpoint_identity": copy.deepcopy(
                artifact_audit["checkpoint_identity"]
            ),
            "evaluation_source_binding": live_binding,
            "evaluator_contract": evaluator_contract(),
            "threshold_selection_scope": "single_checkpoint_only",
            "cross_checkpoint_point_pooling": False,
            "evaluated_checkpoint_count": 1,
            "own_checkpoint_selection_verified": True,
            "final_metric_coverage": _final_metric_coverage(fixed, budgets),
        }
    )
    audit = dict(ready["audit"])
    audit["device_assignment"] = copy.deepcopy(dict(device_assignment))
    audit["ramp100_checkpoint_adapter_strict"] = True
    audit["ramp100_state_dict_strict_load"] = True
    audit["legacy_six_tensor_eval_output"] = True
    audit["own_checkpoint_selection_verified"] = True
    ready["audit"] = audit
    validate_output_identity(ready, artifact_audit=artifact_audit)
    return ready


def validate_output_identity(
    payload: Mapping[str, Any],
    *,
    artifact_audit: Mapping[str, Any],
) -> None:
    candidate = artifact_audit["candidate_contract"]
    for name, expected in {
        "schema": EVALUATION_SCHEMA,
        "variant": artifact_audit["variant"],
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "checkpoint_role": artifact_audit["checkpoint_role"],
        "checkpoint_epoch": artifact_audit["checkpoint_epoch"],
        "checkpoint_sha256": artifact_audit["checkpoint_sha256"],
        "validation_count": EXPECTED_VALIDATION_COUNT,
        "validation_split_sha256": artifact_audit[
            "validation_split_sha256"
        ],
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "base_model_variant": candidate["base_model_variant"],
        "family_recipe": exact.FAMILY_RECIPE,
        "candidate_recipe": candidate["candidate_recipe"],
        "checkpoint_survival_weight_effective": artifact_audit[
            "checkpoint_survival_weight_effective"
        ],
        "checkpoint_tss_ramp_fraction": artifact_audit[
            "checkpoint_tss_ramp_fraction"
        ],
        "threshold_selection_scope": "single_checkpoint_only",
        "cross_checkpoint_point_pooling": False,
        "evaluated_checkpoint_count": 1,
        "own_checkpoint_selection_verified": True,
        "official_test_accessed": False,
    }.items():
        _require_equal(f"ramp100 output {name}", payload.get(name), expected)
    _canonical_equal(
        "ramp100 output run identity",
        payload.get("run_identity"),
        artifact_audit["run_identity"],
    )
    _canonical_equal(
        "ramp100 output checkpoint identity",
        payload.get("source_checkpoint_identity"),
        artifact_audit["checkpoint_identity"],
    )
    _require_equal(
        "ramp100 output run directory",
        Path(str(payload.get("run_directory"))).resolve(),
        Path(str(artifact_audit["run_directory"])).resolve(),
    )
    _require_equal(
        "ramp100 output checkpoint path",
        Path(str(payload.get("checkpoint"))).resolve(),
        Path(str(artifact_audit["checkpoint_path"])).resolve(),
    )
    _canonical_equal(
        "ramp100 output checkpoint validation metrics",
        payload.get("checkpoint_validation_metrics"),
        artifact_audit["checkpoint_validation_metrics"],
    )
    _canonical_equal(
        "ramp100 output source binding",
        payload.get("evaluation_source_binding"),
        artifact_audit["source_binding"],
    )
    _canonical_equal(
        "ramp100 output evaluator contract",
        payload.get("evaluator_contract"),
        evaluator_contract(),
    )
    _canonical_equal(
        "ramp100 output survival schedule",
        payload.get("survival_weight_schedule"),
        exact.survival_schedule_contract(artifact_audit["variant"]),
    )
    _canonical_equal(
        "ramp100 output optimizer recipe",
        payload.get("optimizer_recipe"),
        exact.optimizer_recipe_contract(),
    )
    _canonical_equal(
        "ramp100 output BatchNorm recipe",
        payload.get("batchnorm_recipe"),
        exact.batchnorm_recipe_contract(),
    )
    _canonical_equal(
        "ramp100 output threshold configuration",
        payload.get("threshold_configuration"),
        {
            "threshold_min": 0.01,
            "threshold_max": 0.99,
            "threshold_step": 0.01,
            "extra_thresholds": list(EXTRA_THRESHOLDS),
            "tail_logit_step": 0.1,
            "fa_budgets": list(FA_BUDGETS),
        },
    )
    audit = payload.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("ramp100 output audit is missing")
    qfg_evaluator.v4_evaluator._validate_standard_audit(audit)
    fixed = _validate_point_collection(
        payload,
        artifact_audit["checkpoint_validation_metrics"],
    )
    budgets = _normalize_budgets(payload)
    _validate_closed_interval(payload)
    _canonical_equal(
        "ramp100 output metric coverage",
        payload.get("final_metric_coverage"),
        _final_metric_coverage(fixed, budgets),
    )


def validate_existing_output(
    path: Path,
    *,
    artifact_audit: Mapping[str, Any],
    device_assignment: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    before = _sha256_file(path)
    payload = _load_json(path)
    validate_output_identity(payload, artifact_audit=artifact_audit)
    _canonical_equal(
        "ramp100 existing-output device assignment",
        payload.get("audit", {}).get("device_assignment"),
        device_assignment,
    )
    _require_equal(
        "ramp100 existing output stability",
        _sha256_file(path),
        before,
    )
    return payload


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
        raise ValueError("formal ramp100 evaluator forbids overwrite")
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
                f"refusing to replace existing ramp100 sweep: {path}"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
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
        match_radius=FORMAL_MATCH_RADIUS,
        tiny_area=FORMAL_TINY_AREA,
        overwrite=False,
    )


def _load_isolated_base_evaluator(
    args: argparse.Namespace,
    request: EvaluationRequest,
    artifact_audit: dict[str, Any],
    device_assignment: Mapping[str, Any],
) -> ModuleType:
    module_name = (
        "_sctransnet_ramp100_pd_fa_"
        f"{request.variant}_{Path(request.checkpoint).stem}"
    )
    spec = importlib.util.spec_from_file_location(
        module_name,
        BASE_EVALUATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load shared Pd/Fa evaluator")
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
            raise RuntimeError("ramp100 evaluator observed no forward pass")
        artifact_audit["legacy_eval_output_verified"] = True
        return result

    def bound_write(
        path: Path,
        payload: dict[str, Any],
        overwrite: bool,
    ) -> None:
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


def output_path(request: EvaluationRequest) -> Path:
    return request.run_dir / (
        f"pd_fa_sweep_{Path(request.checkpoint).stem}.json"
    )


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
    _require_equal(
        "ramp100 request/preflight variant",
        request.variant,
        audit["variant"],
    )
    output = output_path(request)
    if output.exists() or output.is_symlink():
        validate_existing_output(
            output,
            artifact_audit=audit,
            device_assignment=device_assignment,
        )
        return output
    evaluator = _load_isolated_base_evaluator(
        args,
        request,
        audit,
        device_assignment,
    )
    original_argv = sys.argv
    sys.argv = [
        str(Path(__file__).resolve()),
        "--run-dir",
        str(request.run_dir),
        "--checkpoint",
        request.checkpoint,
        "--device",
        args.device,
        "--expected-epochs",
        str(EXPECTED_EPOCHS),
        "--match-radius",
        str(FORMAL_MATCH_RADIUS),
        "--tiny-area",
        str(FORMAL_TINY_AREA),
    ]
    try:
        _require_checkpoint_unchanged(
            audit,
            stage="before shared evaluator",
        )
        evaluator.main()
        _require_checkpoint_unchanged(
            audit,
            stage="after shared evaluator",
        )
    finally:
        sys.argv = original_argv
    if not output.is_file() or output.is_symlink():
        raise RuntimeError(f"ramp100 evaluator did not create {output}")
    validate_existing_output(
        output,
        artifact_audit=audit,
        device_assignment=device_assignment,
    )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = validate_formal_arguments(argv)
    request = evaluation_request(args)
    audit = validate_run_artifacts(request.run_dir, request.checkpoint)
    if args.preflight:
        print(
            "TPDNER_DLR_RAMP100_EVAL_PREFLIGHT_OK"
            f" variant={request.variant}"
            f" checkpoint={request.checkpoint}"
            f" epoch={audit['checkpoint_epoch']}"
            " writes_performed=false",
            flush=True,
        )
        return
    configure_v8_inference(args.device)
    assignment = _device_assignment(args.device, variant=request.variant)
    existed = output_path(request).is_file()
    path = evaluate_one(
        args,
        request,
        device_assignment=assignment,
        artifact_audit=audit,
    )
    status = "IDEMPOTENT_COMPLETE" if existed else "COMPLETE"
    print(
        f"TPDNER_DLR_RAMP100_EVAL_{status}"
        f" variant={request.variant}"
        f" checkpoint={request.checkpoint}"
        f" physical_gpu={assignment['physical_gpu_index']}"
        f" output={path}",
        flush=True,
    )


__all__ = [
    "BUDGET_KEYS",
    "CHECKPOINT_ROLES",
    "DATASET",
    "DEFAULT_RUN_DIRS",
    "EVALUATION_SCHEMA",
    "EXPECTED_TARGET_COUNT",
    "EXPECTED_TINY_TARGET_COUNT",
    "EXPECTED_VALIDATION_COUNT",
    "EvaluationRequest",
    "FA_BUDGETS",
    "FORMAL_MATCH_RADIUS",
    "FORMAL_TINY_AREA",
    "QFG_DLR_VARIANT",
    "SOURCE_BINDING_SCHEMA",
    "SUPPORTED_VARIANTS",
    "TSS_QFG_DLR_VARIANT",
    "build_model",
    "evaluate_one",
    "evaluation_request",
    "evaluator_contract",
    "finalize_evaluation_output",
    "main",
    "output_path",
    "source_binding",
    "validate_existing_output",
    "validate_formal_arguments",
    "validate_output_identity",
    "validate_run_artifacts",
    "verify_frozen_training_sources",
]


if __name__ == "__main__":
    main()
