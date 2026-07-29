#!/usr/bin/env python3
"""Checkpoint-local closed-interval Pd/Fa evaluation for QFG arms C and D.

The two validation-selected checkpoints of ``qfg_only`` and ``tss_qfg`` are
evaluated independently.  ``--all-four`` is orchestration only: thresholds,
budget selection, points, and output artifacts remain owned by one checkpoint,
and existing sweep files are never replaced.
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
from experiments import train_tpd_ner_v4_qfg_v2_croa_exact as exact  # noqa: E402
from experiments.evaluate_tpd_clean_v8_mprs_dch_pd_fa import (  # noqa: E402
    LAST_FLOAT32_BELOW_ONE,
    UPPER_BOUNDARY_THRESHOLD,
    adaptive_thresholds_closed_interval,
    configure_v8_inference,
)


EVALUATION_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_checkpoint_local_pd_fa_v1"
)
SOURCE_BINDING_SCHEMA = (
    "sctransnet_tpd_ner_v4_qfg_v2_croa_evaluation_source_binding_v1"
)
DATASET = "NUDT-SIRST"
TRAINING_SEED = exact.TRAINING_SEED
SPLIT_SEED = exact.SPLIT_SEED
EXPECTED_EPOCHS = exact.FORMAL_EPOCHS
EXPECTED_VALIDATION_COUNT = 133
EXPECTED_TARGET_COUNT = 189
EXPECTED_TINY_TARGET_COUNT = 39
SUPPORTED_VARIANTS = tuple(exact.SUPPORTED_CANDIDATE_VARIANTS)
QFG_ONLY_VARIANT = exact.QFG_ONLY_VARIANT
TSS_QFG_VARIANT = exact.TSS_QFG_VARIANT
if set(SUPPORTED_VARIANTS) != {QFG_ONLY_VARIANT, TSS_QFG_VARIANT}:
    raise RuntimeError("formal QFG evaluator requires exactly arms C and D")
CHECKPOINT_ROLES = {
    "best.pth.tar": "best_validation_pd_primary",
    "best_miou.pth.tar": "best_validation_miou_secondary",
}
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
BUDGET_KEYS = tuple(f"{budget:.10g}" for budget in FA_BUDGETS)
FIXED_THRESHOLD = exact.FORMAL_THRESHOLD
FORMAL_MATCH_RADIUS = exact.FORMAL_MATCH_RADIUS
FORMAL_TINY_AREA = exact.FORMAL_TINY_AREA
EXTRA_THRESHOLDS = (0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999)
BASE_EVALUATOR_PATH = REPO_ROOT / "experiments/evaluate_pd_fa_sweep.py"
DEFAULT_RESULTS_ROOT = Path(
    getattr(
        exact,
        "DEFAULT_OUTPUT_ROOT",
        REPO_ROOT
        / "experiments/results/tpd_ner_v4_qfg_v2_croa_exact_v1",
    )
)
PHYSICAL_GPU_INDEX_ENV = "TPD_NER_V4_QFG_PHYSICAL_GPU_INDEX"
PHYSICAL_GPU_UUID_ENV = "TPD_NER_V4_QFG_PHYSICAL_GPU_UUID"

_normalize_budgets = v4_evaluator._normalize_budgets
_validate_closed_interval = v4_evaluator._validate_closed_interval
_validate_point_collection = v4_evaluator._validate_point_collection
_final_metric_coverage = v4_evaluator._final_metric_coverage


def _candidate_contract(variant: str) -> dict[str, Any]:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported QFG candidate: {variant!r}")
    contract = exact.candidate_contract(variant)
    if not isinstance(contract, Mapping):
        raise RuntimeError("QFG trainer candidate_contract returned non-mapping")
    return copy.deepcopy(dict(contract))


def _candidate_run_tag(variant: str) -> str:
    contract = _candidate_contract(variant)
    run_tag = contract.get("formal_run_tag")
    if not isinstance(run_tag, str) or not run_tag:
        raise RuntimeError(f"QFG candidate {variant!r} has no run_tag")
    return run_tag


DEFAULT_RUN_DIRS = {
    variant: (
        DEFAULT_RESULTS_ROOT
        / DATASET
        / variant
        / f"seed_{TRAINING_SEED}_{_candidate_run_tag(variant)}"
    )
    for variant in SUPPORTED_VARIANTS
}


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


def source_binding() -> dict[str, Any]:
    """Bind the trainer and the two reused evaluator implementations."""

    return {
        "schema": SOURCE_BINDING_SCHEMA,
        "trainer": {
            "path": str(Path(exact.__file__).resolve()),
            "sha256": _sha256_file(Path(exact.__file__)),
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
    binding = source_binding()
    contract = exact.formal_contract()
    if not isinstance(contract, Mapping):
        raise RuntimeError("QFG trainer formal_contract returned non-mapping")
    return {
        "schema": EVALUATION_SCHEMA,
        "dataset": DATASET,
        "variants": list(SUPPORTED_VARIANTS),
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "expected_epochs": EXPECTED_EPOCHS,
        "checkpoint_roles": dict(CHECKPOINT_ROLES),
        "fixed_threshold": FIXED_THRESHOLD,
        "fa_budgets": list(FA_BUDGETS),
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "threshold_selection_scope": "single_checkpoint_only",
        "cross_checkpoint_point_pooling": False,
        "cross_checkpoint_overwrite": False,
        "expected_sweep_count": 4,
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
        "trainer_formal_contract": copy.deepcopy(dict(contract)),
        "official_test_accessed": False,
    }


def build_model(variant: str, seed: int):
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"unsupported QFG evaluation variant: {variant!r}")
    if seed != TRAINING_SEED:
        raise ValueError("formal QFG evaluator requires seed 42")
    return exact.build_selected_model(variant, seed)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", choices=tuple(CHECKPOINT_ROLES))
    parser.add_argument("--all-four", action="store_true")
    parser.add_argument(
        "--qfg-only-run-dir",
        type=Path,
        default=DEFAULT_RUN_DIRS[QFG_ONLY_VARIANT],
    )
    parser.add_argument(
        "--tss-qfg-run-dir",
        type=Path,
        default=DEFAULT_RUN_DIRS[TSS_QFG_VARIANT],
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
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_formal_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    args = _argument_parser().parse_args(argv)
    if args.overwrite:
        raise ValueError("formal QFG evaluator forbids --overwrite")
    if args.device not in ("cpu", "cuda:0"):
        raise ValueError("formal QFG evaluator device must be cpu or cuda:0")
    if args.expected_epochs != EXPECTED_EPOCHS:
        raise ValueError("formal QFG evaluator requires expected_epochs=800")
    if args.all_four:
        if args.run_dir is not None or args.checkpoint is not None:
            raise ValueError(
                "--all-four cannot be combined with single-run arguments"
            )
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
        ("match_radius", args.match_radius, FORMAL_MATCH_RADIUS),
        ("tiny_area", args.tiny_area, FORMAL_TINY_AREA),
    ):
        _require_equal(name, observed, expected)
    return args


def evaluation_requests(
    args: argparse.Namespace,
) -> tuple[EvaluationRequest, ...]:
    if args.all_four:
        requests = tuple(
            EvaluationRequest(variant, Path(run_dir).resolve(), checkpoint)
            for variant, run_dir in (
                (QFG_ONLY_VARIANT, args.qfg_only_run_dir),
                (TSS_QFG_VARIANT, args.tss_qfg_run_dir),
            )
            for checkpoint in CHECKPOINT_ROLES
        )
    else:
        run_dir = Path(args.run_dir).resolve()
        variant = run_dir.parent.name
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                "single-run directory is not under qfg_only or tss_qfg"
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
    if args.all_four and len(requests) != 4:
        raise RuntimeError("formal QFG all-four did not produce four requests")
    return requests


def _physical_gpu_uuids() -> dict[str, str]:
    raw = getattr(exact, "PHYSICAL_GPU_UUIDS", {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


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
    uuids = _physical_gpu_uuids()
    if physical_index not in ("2", "3"):
        raise RuntimeError("QFG evaluation physical GPU must be 2 or 3")
    expected_uuid = uuids.get(physical_index)
    if expected_uuid is None:
        raise RuntimeError("QFG trainer has no registered GPU UUID")
    if (
        physical_uuid != expected_uuid
        or os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid
    ):
        raise RuntimeError("QFG evaluation GPU UUID assignment differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("QFG evaluator requires one visible CUDA device")
    return {
        "device": "cuda:0",
        "physical_gpu_index": int(physical_index),
        "physical_gpu_uuid": expected_uuid,
        "cuda_visible_devices": expected_uuid,
        "device_name": torch.cuda.get_device_name(0),
    }


def _require_legacy_eval_output(output: Any) -> tuple[torch.Tensor, ...]:
    if not isinstance(output, tuple) or len(output) != 6:
        raise RuntimeError(
            "QFG evaluation requires the legacy six-tensor output"
        )
    if not all(isinstance(value, torch.Tensor) for value in output):
        raise RuntimeError("QFG legacy evaluation output contains a non-tensor")
    return output


def _require_checkpoint_payload(
    payload: Any,
    *,
    expected_variant: str,
) -> dict[str, Any]:
    validator = getattr(
        exact,
        "require_evaluator_checkpoint_payload",
        None,
    )
    if callable(validator):
        try:
            validated = validator(
                payload,
                expected_variant=expected_variant,
            )
        except TypeError:
            validated = validator(payload)
    else:
        adapter_type = getattr(exact, "EvaluatorCheckpointAdapter", None)
        if adapter_type is None:
            raise RuntimeError(
                "QFG trainer exposes no evaluator checkpoint validator"
            )
        try:
            adapter = adapter_type(
                payload,
                expected_variant=expected_variant,
            )
        except TypeError:
            adapter = adapter_type(payload)
        validate = getattr(adapter, "validate", None)
        if callable(validate):
            validated = validate()
        elif isinstance(adapter, Mapping):
            validated = dict(adapter)
        else:
            raise RuntimeError(
                "QFG evaluator checkpoint adapter has no validate() method"
            )
    if not isinstance(validated, Mapping):
        raise ValueError("QFG checkpoint validator returned non-mapping")
    return copy.deepcopy(dict(validated))


def validate_run_artifacts(
    run_dir: Path,
    checkpoint_name: str = "best.pth.tar",
) -> dict[str, Any]:
    """Strictly bind one completed arm to one validation-owned checkpoint."""

    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise NotADirectoryError(run_dir)
    if checkpoint_name not in CHECKPOINT_ROLES:
        raise ValueError("QFG evaluator accepts only best or best_miou")
    variant = run_dir.parent.name
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError("QFG run directory has an unsupported variant")
    expected_name = (
        f"seed_{TRAINING_SEED}_{_candidate_run_tag(variant)}"
    )
    _require_equal("QFG run-directory name", run_dir.name, expected_name)
    _require_equal(
        "QFG run dataset directory",
        run_dir.parent.parent.name,
        DATASET,
    )

    protocol = _load_json(run_dir / "protocol.json")
    split = _load_json(run_dir / "split.json")
    summary = _load_json(run_dir / "summary.json")
    formal_contract = exact.formal_contract()
    candidate = _candidate_contract(variant)
    _require_equal(
        "QFG protocol schema",
        protocol.get("schema"),
        exact.ENTRY_SCHEMA,
    )
    _require_equal(
        "QFG completion schema",
        summary.get("schema"),
        exact.COMPLETION_SUMMARY_SCHEMA,
    )
    _canonical_equal(
        "QFG protocol formal contract",
        protocol.get("formal_contract"),
        formal_contract,
    )
    _canonical_equal(
        "QFG summary formal contract",
        summary.get("formal_contract"),
        formal_contract,
    )
    _require_equal("QFG completion status", summary.get("status"), "complete")
    _require_equal(
        "QFG split official-test access",
        split.get("official_test_accessed"),
        False,
    )
    _require_equal(
        "QFG split source",
        split.get("source"),
        "img_idx/train_NUDT-SIRST.txt",
    )
    arguments = protocol.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("QFG protocol arguments are missing")
    expected_arguments = {
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
    }
    for name, expected in expected_arguments.items():
        _require_equal(
            f"QFG protocol argument {name}",
            arguments.get(name),
            expected,
        )
    run_identity = exact.require_qfg_run_identity(
        protocol.get("run_identity"),
        label="QFG evaluation protocol",
        expected_variant=variant,
    )
    _canonical_equal(
        "QFG summary run identity",
        summary.get("run_identity"),
        run_identity,
    )
    for artifact_name, artifact in {
        "protocol": protocol,
        "summary": summary,
    }.items():
        for name, expected in {
            "candidate_variant": variant,
            "qfg_variant": candidate["qfg_variant"],
            "tss_variant": candidate["tss_variant"],
        }.items():
            _require_equal(
                f"QFG {artifact_name} {name}",
                artifact.get(name),
                expected,
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
    checkpoint = _require_checkpoint_payload(
        raw_checkpoint,
        expected_variant=variant,
    )
    _require_equal(
        "QFG checkpoint file stability",
        _sha256_file(checkpoint_path),
        checkpoint_sha256,
    )
    _require_equal(
        "QFG checkpoint role",
        checkpoint.get("checkpoint_role"),
        CHECKPOINT_ROLES[checkpoint_name],
    )
    _canonical_equal(
        "QFG checkpoint run identity",
        checkpoint.get("run_identity"),
        run_identity,
    )
    _canonical_equal(
        "QFG summary split hashes",
        summary.get("split_hashes"),
        checkpoint.get("split_hashes"),
    )
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
    _canonical_equal(
        "QFG rebuilt/checkpoint metadata",
        metadata,
        checkpoint.get("model_metadata"),
    )
    incompatible = model.load_state_dict(
        checkpoint.get("state_dict"),
        strict=True,
    )
    _require_equal(
        "QFG strict-load missing keys",
        list(incompatible.missing_keys),
        [],
    )
    _require_equal(
        "QFG strict-load unexpected keys",
        list(incompatible.unexpected_keys),
        [],
    )
    model.eval()
    if model.training:
        raise RuntimeError("QFG evaluator failed to switch model to eval")

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
        "source_binding": source_binding(),
        "state_dict_strict_load": True,
        "legacy_eval_output_verified": False,
    }
    del model, checkpoint, raw_checkpoint
    gc.collect()
    return audit


def _require_checkpoint_unchanged(
    audit: Mapping[str, Any],
    *,
    stage: str,
) -> None:
    observed = _sha256_file(Path(str(audit["checkpoint_path"])))
    _require_equal(
        f"checkpoint SHA {stage}",
        observed,
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
    if not artifact_audit.get("legacy_eval_output_verified"):
        raise ValueError("legacy eval-output guard observed no forward pass")
    for name, expected in {
        "variant": artifact_audit["variant"],
        "dataset": DATASET,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "checkpoint_role": artifact_audit["checkpoint_role"],
        "checkpoint_epoch": artifact_audit["checkpoint_epoch"],
        "checkpoint_sha256": artifact_audit["checkpoint_sha256"],
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "official_test_accessed": False,
    }.items():
        _require_equal(f"QFG evaluation {name}", ready.get(name), expected)
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
    audit["qfg_checkpoint_payload_strict"] = True
    audit["qfg_state_dict_strict_load"] = True
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
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "threshold_selection_scope": "single_checkpoint_only",
        "cross_checkpoint_point_pooling": False,
        "evaluated_checkpoint_count": 1,
        "official_test_accessed": False,
    }.items():
        _require_equal(f"QFG output {name}", payload.get(name), expected)
    _canonical_equal(
        "QFG output run identity",
        payload.get("run_identity"),
        artifact_audit["run_identity"],
    )
    _canonical_equal(
        "QFG output checkpoint identity",
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
        "QFG output metric coverage",
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
        raise ValueError("formal QFG evaluator forbids overwrite")
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
                f"refusing to replace existing QFG sweep: {path}"
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
        "_sctransnet_qfg_v2_croa_pd_fa_"
        f"{request.variant}_{Path(request.checkpoint).stem}"
    )
    spec = importlib.util.spec_from_file_location(
        module_name,
        BASE_EVALUATOR_PATH,
    )
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
            raise RuntimeError("QFG evaluator observed no model forward pass")
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
        "request/preflight variant",
        request.variant,
        audit["variant"],
    )
    output = request.run_dir / (
        f"pd_fa_sweep_{Path(request.checkpoint).stem}.json"
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"refusing to replace existing QFG sweep: {output}"
        )
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
        "--match-radius",
        str(FORMAL_MATCH_RADIUS),
        "--tiny-area",
        str(FORMAL_TINY_AREA),
    ]
    original_argv = sys.argv
    sys.argv = per_checkpoint_argv
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
    if not output.is_file():
        raise RuntimeError(f"QFG evaluator did not create {output}")
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = validate_formal_arguments(argv)
    requests = evaluation_requests(args)
    configure_v8_inference(args.device)
    assignment = _device_assignment(args.device)

    # Preflight all four before producing the first output.  This prevents a
    # partial matrix when one arm/checkpoint identity is invalid.
    audits = [
        validate_run_artifacts(request.run_dir, request.checkpoint)
        for request in requests
    ]
    output_paths = [
        request.run_dir
        / f"pd_fa_sweep_{Path(request.checkpoint).stem}.json"
        for request in requests
    ]
    if len(set(output_paths)) != len(output_paths):
        raise RuntimeError("QFG evaluation output paths are not unique")
    for output in output_paths:
        if output.exists() or output.is_symlink():
            raise FileExistsError(
                f"refusing to replace existing QFG sweep: {output}"
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
    "FORMAL_MATCH_RADIUS",
    "FORMAL_TINY_AREA",
    "QFG_ONLY_VARIANT",
    "SUPPORTED_VARIANTS",
    "TSS_QFG_VARIANT",
    "build_model",
    "evaluate_one",
    "evaluation_requests",
    "evaluator_contract",
    "finalize_evaluation_output",
    "main",
    "source_binding",
    "validate_formal_arguments",
    "validate_output_identity",
    "validate_run_artifacts",
]


if __name__ == "__main__":
    main()
