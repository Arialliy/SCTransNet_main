#!/usr/bin/env python3
"""Post-training evaluation and aggregate for V8-MPRS-DCH + five-node NER.

The program is deliberately outside the frozen training and acceptance
manifests.  Before training is complete it performs read-only progress
inspection.  Once both exact runs have 800 contiguous metric events and a
complete summary, it evaluates the relay-off/on pair checkpoint-role by
checkpoint-role on physical GPU 2/3, evaluates the existing same-split
SCTransNet reference through the same metric core and closed interval, and
emits one JSON plus one Markdown comparison.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    evaluate_sctransnet_baseline_reference_closed_interval as baseline_eval,
)
from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_pd_fa as ner_eval,
)


SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_posttraining_aggregate_v1"
DATASET = "NUDT-SIRST"
TRAINING_SEED = 42
SPLIT_SEED = 20260722
EXPECTED_EPOCHS = 800
TARGET_COUNT = 189
CHECKPOINTS = ("best.pth.tar", "best_miou.pth.tar")
CHECKPOINT_ROLES = {
    "best.pth.tar": "best_validation_pd_primary",
    "best_miou.pth.tar": "best_validation_miou_secondary",
}
ROLE_NAMES = {
    "best_validation_pd_primary": "pd_primary",
    "best_validation_miou_secondary": "miou_secondary",
}
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
BUDGET_KEYS = tuple(f"{budget:.10g}" for budget in FA_BUDGETS)
VARIANT_OFF = "tpd_ner_v8_mprs_dch_full_relay_off"
VARIANT_ON = "tpd_ner_v8_mprs_dch_full_relay_on"
VARIANTS = (VARIANT_OFF, VARIANT_ON)
RUN_TAG = "formal800_exact_v1"
RESULT_ROOT = (
    REPO_ROOT / "experiments/results/tpd_ner_v8_mprs_dch_exact_v1"
)
RUN_DIRS = {
    variant: RESULT_ROOT / DATASET / variant / f"seed_42_{RUN_TAG}"
    for variant in VARIANTS
}
GPU_UUIDS = {
    VARIANT_OFF: "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    VARIANT_ON: "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
GPU_INDICES = {VARIANT_OFF: 2, VARIANT_ON: 3}
BASELINE_SOURCE_RUN = (
    REPO_ROOT
    / "experiments/results/tpd_pe_formal800_4x5090_v1"
    / DATASET
    / "original"
    / "seed_42_formal800_pd_fp32_4x5090_v1"
)
BASELINE_VIEW_RUN = (
    RESULT_ROOT
    / "baseline_reference_closed_interval_v1"
    / DATASET
    / "original"
    / "seed_42_formal800_pd_fp32_4x5090_v1"
)
COMPARISON_DIR = RESULT_ROOT / DATASET / "comparison"
JSON_OUTPUT = COMPARISON_DIR / "tpd_ner_v8_mprs_dch_formal800_comparison.json"
MARKDOWN_OUTPUT = (
    COMPARISON_DIR / "tpd_ner_v8_mprs_dch_formal800_comparison.md"
)
COMPLETE_MARKER = COMPARISON_DIR / "POSTPROCESS_COMPLETE.json"
NER_EVALUATOR = (
    REPO_ROOT / "experiments/evaluate_tpd_ner_v8_mprs_dch_pd_fa.py"
)
BASELINE_EVALUATOR = (
    REPO_ROOT
    / "experiments/evaluate_sctransnet_baseline_reference_closed_interval.py"
)
TRAINING_LOCK = (
    REPO_ROOT / "experiments/tpd_ner_v8_mprs_dch_exact_source_lock.json"
)
ACCEPTANCE_LOCK = (
    REPO_ROOT / "experiments/tpd_ner_v8_mprs_dch_acceptance_source_lock.json"
)
EXPECTED_TRAINING_LOCK_SHA256 = (
    "56834090792d10d5d42808f960701634c0e2c6833e582634a042f523661219da"
)
EXPECTED_ACCEPTANCE_LOCK_SHA256 = (
    "e73eab04e3f7dc092da9f44c9989752ce30534587162701a843cd58a278cd1d2"
)
BASELINE_VIEW_FILES = (
    "protocol.json",
    "split.json",
    "summary.json",
    "metrics.jsonl",
    *CHECKPOINTS,
)


class IncompleteTraining(RuntimeError):
    """Raised when post-training work is requested before both runs finish."""


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"expected a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(location: str, value: Any) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{location} must be finite, got {value!r}")
    return float(value)


def verify_frozen_manifests() -> dict[str, Any]:
    """Verify, without updating, both frozen NER manifests and their sources."""

    observed_training = sha256_file(TRAINING_LOCK)
    observed_acceptance = sha256_file(ACCEPTANCE_LOCK)
    _require(
        observed_training == EXPECTED_TRAINING_LOCK_SHA256,
        "training source-lock file hash differs",
    )
    _require(
        observed_acceptance == EXPECTED_ACCEPTANCE_LOCK_SHA256,
        "acceptance source-lock file hash differs",
    )
    training = load_json(TRAINING_LOCK)
    acceptance = load_json(ACCEPTANCE_LOCK)
    _require(training.get("lock_kind") == "training", "training lock kind differs")
    _require(
        acceptance.get("lock_kind") == "acceptance",
        "acceptance lock kind differs",
    )
    _require(
        acceptance.get("training_source_lock_sha256") == observed_training,
        "acceptance lock does not bind the current training lock",
    )
    _require(
        training.get("training_data_sha256")
        == acceptance.get("training_data_sha256"),
        "training-data fingerprints differ between manifests",
    )
    _require(training.get("variants") == list(VARIANTS), "training variants differ")
    _require(
        acceptance.get("variants") == list(VARIANTS),
        "acceptance variants differ",
    )
    for lock_name, payload in (
        ("training", training),
        ("acceptance", acceptance),
    ):
        sources = payload.get("source_sha256")
        _require(isinstance(sources, Mapping), f"{lock_name} source map is missing")
        _require(
            payload.get("source_count") == len(sources),
            f"{lock_name} source count differs",
        )
        for relative, expected in sources.items():
            _require(isinstance(relative, str), f"{lock_name} source path is invalid")
            path = (REPO_ROOT / relative).resolve()
            _require(
                path.is_relative_to(REPO_ROOT),
                f"{lock_name} source escapes repository: {relative}",
            )
            _require(
                sha256_file(path) == expected,
                f"{lock_name} source hash differs: {relative}",
            )
    return {
        "training_source_lock": str(TRAINING_LOCK),
        "training_source_lock_sha256": observed_training,
        "acceptance_source_lock": str(ACCEPTANCE_LOCK),
        "acceptance_source_lock_sha256": observed_acceptance,
        "training_data_sha256": training["training_data_sha256"],
        "training_source_count": training["source_count"],
        "acceptance_source_count": acceptance["source_count"],
    }


def _metrics_progress(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        return {
            "exists": False,
            "event_count": 0,
            "last_epoch": 0,
            "contiguous_from_one": False,
        }
    events: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        events.append(value)
    epochs = [event.get("epoch") for event in events]
    return {
        "exists": True,
        "event_count": len(events),
        "last_epoch": epochs[-1] if epochs else 0,
        "contiguous_from_one": epochs == list(range(1, len(events) + 1)),
    }


def inspect_run_progress(variant: str, run_dir: Path | None = None) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unexpected variant: {variant}")
    directory = RUN_DIRS[variant] if run_dir is None else Path(run_dir)
    metrics = _metrics_progress(directory / "metrics.jsonl")
    summary_path = directory / "summary.json"
    summary: dict[str, Any] | None = None
    summary_status: Any = None
    if summary_path.is_file() and not summary_path.is_symlink():
        summary = load_json(summary_path)
        summary_status = summary.get("status")
    required = (
        "protocol.json",
        "split.json",
        "metrics.jsonl",
        "summary.json",
        "best.pth.tar",
        "best_miou.pth.tar",
        "last.pth.tar",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    complete = bool(
        not missing
        and summary_status == "complete"
        and metrics["event_count"] == EXPECTED_EPOCHS
        and metrics["last_epoch"] == EXPECTED_EPOCHS
        and metrics["contiguous_from_one"]
    )
    if summary is not None:
        _require(
            summary.get("variant") == variant,
            f"{variant} completion summary identity differs",
        )
        _require(
            summary.get("seed") == TRAINING_SEED,
            f"{variant} completion summary seed differs",
        )
        _require(
            summary.get("split_seed") == SPLIT_SEED,
            f"{variant} completion summary split seed differs",
        )
    return {
        "variant": variant,
        "run_dir": str(directory.resolve()),
        "physical_gpu_index": GPU_INDICES[variant],
        "physical_gpu_uuid": GPU_UUIDS[variant],
        "metrics": metrics,
        "summary_exists": summary is not None,
        "summary_status": summary_status,
        "missing_required_artifacts": missing,
        "complete": complete,
    }


def inspect_training_readiness() -> dict[str, Any]:
    runs = {variant: inspect_run_progress(variant) for variant in VARIANTS}
    return {
        "schema": "sctransnet_tpd_ner_v8_mprs_dch_posttraining_readiness_v1",
        "expected_epochs": EXPECTED_EPOCHS,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "runs": runs,
        "both_runs_complete": all(record["complete"] for record in runs.values()),
        "posttraining_action": (
            "evaluate"
            if all(record["complete"] for record in runs.values())
            else "wait"
        ),
    }


def _same_split_and_training_contract() -> dict[str, Any]:
    """Establish that the historical baseline is a same-protocol reference."""

    splits = {
        VARIANT_OFF: load_json(RUN_DIRS[VARIANT_OFF] / "split.json"),
        VARIANT_ON: load_json(RUN_DIRS[VARIANT_ON] / "split.json"),
        "baseline_sctransnet": load_json(BASELINE_SOURCE_RUN / "split.json"),
    }
    protocols = {
        VARIANT_OFF: load_json(RUN_DIRS[VARIANT_OFF] / "protocol.json"),
        VARIANT_ON: load_json(RUN_DIRS[VARIANT_ON] / "protocol.json"),
        "baseline_sctransnet": load_json(
            BASELINE_SOURCE_RUN / "protocol.json"
        ),
    }
    baseline_summary = load_json(BASELINE_SOURCE_RUN / "summary.json")
    arguments: dict[str, Mapping[str, Any]] = {}
    for name, protocol in protocols.items():
        value = protocol.get("arguments")
        _require(
            isinstance(value, Mapping),
            f"{name} protocol arguments missing",
        )
        arguments[name] = value

    fixed_axes = {
        "dataset": DATASET,
        "epochs": EXPECTED_EPOCHS,
        "batch_size": 16,
        "patch_size": 256,
        "workers": 0,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "val_fraction": 0.2,
        "eval_every": 1,
        "base_lr": 1e-3,
        "min_lr": 1e-5,
        "warmup_epochs": 10,
        "threshold": 0.5,
        "match_radius": 3.0,
        "tiny_area": 9,
        "amp": False,
        "max_train_images": None,
        "max_val_images": None,
        "device": "cuda:0",
    }
    for source, source_arguments in arguments.items():
        for name, expected in fixed_axes.items():
            _require(
                source_arguments.get(name) == expected,
                f"{source} protocol argument differs: {name}",
            )
    dataset_directories = {
        str(Path(str(value["dataset_dir"])).resolve())
        for value in arguments.values()
    }
    _require(
        len(dataset_directories) == 1,
        "baseline/off/on dataset directories differ",
    )
    _require(
        arguments[VARIANT_OFF].get("run_tag") == RUN_TAG
        and arguments[VARIANT_ON].get("run_tag") == RUN_TAG,
        "NER run tags differ",
    )
    _require(
        arguments["baseline_sctransnet"].get("run_tag")
        == "formal800_pd_fp32_4x5090_v1",
        "baseline run tag differs",
    )

    for name, expected in (
        ("dataset", DATASET),
        ("split_seed", SPLIT_SEED),
        ("used_train_count", 530),
        ("used_val_count", 133),
        ("full_official_train_count", 663),
        ("official_test_accessed", False),
    ):
        for source, split in splits.items():
            _require(
                split.get(name) == expected,
                f"{source} split differs: {name}",
            )
    split_reference = splits[VARIANT_OFF]
    for source, split in splits.items():
        _require(
            split.get("used_train_ids")
            == split_reference.get("used_train_ids"),
            f"{source} ordered training IDs differ",
        )
        _require(
            split.get("used_val_ids")
            == split_reference.get("used_val_ids"),
            f"{source} ordered validation IDs differ",
        )
        _require(
            split.get("hashes") == split_reference.get("hashes"),
            f"{source} split hashes differ",
        )

    normalization = protocols[VARIANT_OFF].get("normalization")
    _require(
        isinstance(normalization, Mapping),
        "NER normalization is missing",
    )
    for source, protocol in protocols.items():
        _require(
            protocol.get("normalization") == normalization,
            f"{source} normalization differs",
        )
    for field in ("optimizer", "loss"):
        expected = protocols[VARIANT_OFF].get(field)
        _require(expected is not None, f"NER {field} contract is missing")
        for source, protocol in protocols.items():
            _require(
                protocol.get(field) == expected,
                f"{source} {field} contract differs",
            )
    for field in ("primary_selection_rule", "secondary_selection_rule"):
        expected = protocols[VARIANT_OFF].get(field)
        _require(isinstance(expected, list), f"NER {field} is missing")
        for source, protocol in protocols.items():
            _require(
                protocol.get(field) == expected,
                f"{source} {field} differs",
            )

    ner_lr_text = "manual warmup then cosine; no scheduler object"
    baseline_lr_text = (
        "10-epoch linear warmup then cosine decay (or CLI overrides)"
    )
    _require(
        protocols[VARIANT_OFF].get("lr_schedule") == ner_lr_text
        and protocols[VARIANT_ON].get("lr_schedule") == ner_lr_text
        and protocols["baseline_sctransnet"].get("lr_schedule")
        == baseline_lr_text,
        "learning-rate schedule declarations differ from their expected forms",
    )
    ner_checkpoint_policy = (
        "best.pth.tar is Pd-primary; best_miou.pth.tar is mIoU-secondary; "
        "last.pth.tar is the last evaluated epoch; exact journal is authoritative"
    )
    baseline_checkpoint_policy = (
        "best.pth.tar is Pd-primary; best_miou.pth.tar is a secondary "
        "analysis checkpoint; all selection uses internal validation only"
    )
    _require(
        protocols[VARIANT_OFF].get("checkpoint_policy")
        == ner_checkpoint_policy
        and protocols[VARIANT_ON].get("checkpoint_policy")
        == ner_checkpoint_policy
        and protocols["baseline_sctransnet"].get("checkpoint_policy")
        == baseline_checkpoint_policy,
        "checkpoint policy declarations differ from their expected forms",
    )
    _require(
        protocols[VARIANT_OFF].get("selection_source")
        == "internal_validation_only"
        and protocols[VARIANT_ON].get("selection_source")
        == "internal_validation_only"
        and baseline_summary.get("selection_source")
        == "internal_validation_only",
        "checkpoint selection source differs",
    )
    _require(
        baseline_summary.get("status") == "complete",
        "baseline summary is not complete",
    )
    _require(
        _metrics_progress(BASELINE_SOURCE_RUN / "metrics.jsonl")
        == {
            "exists": True,
            "event_count": EXPECTED_EPOCHS,
            "last_epoch": EXPECTED_EPOCHS,
            "contiguous_from_one": True,
        },
        "baseline metrics history is not 800 contiguous epochs",
    )
    return {
        "same_fixed_training_axes": True,
        "same_learning_rate_axes": True,
        "same_normalization": True,
        "same_optimizer": True,
        "same_loss": True,
        "same_selection_rules": True,
        "checkpoint_policies_semantically_aligned": True,
        "same_off_on_fixed_protocol": True,
        "same_ordered_train_ids": True,
        "same_ordered_validation_ids": True,
        "same_split_hashes": True,
        "official_test_accessed": False,
        "baseline_source_run": str(BASELINE_SOURCE_RUN),
        "reference_semantics": (
            "historical checkpoint, currently re-evaluated with the shared "
            "closed-interval reference evaluator"
        ),
        "endpoint_protocol_preregistered_before_historical_training": False,
        "normalization": dict(normalization),
        "fixed_training_axes": fixed_axes,
        "baseline_source_artifact_sha256": {
            name: sha256_file(BASELINE_SOURCE_RUN / name)
            for name in BASELINE_VIEW_FILES
        },
    }


def prepare_baseline_reference_view() -> dict[str, Any]:
    """Create a no-overwrite hard-linked view for closed-interval re-evaluation."""

    contract = _same_split_and_training_contract()
    BASELINE_VIEW_RUN.mkdir(parents=True, exist_ok=True)
    if BASELINE_VIEW_RUN.is_symlink():
        raise ValueError("baseline reference view may not be a symlink")
    linked: dict[str, str] = {}
    for name in BASELINE_VIEW_FILES:
        source = BASELINE_SOURCE_RUN / name
        target = BASELINE_VIEW_RUN / name
        expected = sha256_file(source)
        if target.exists() or target.is_symlink():
            _require(
                target.is_file()
                and not target.is_symlink()
                and sha256_file(target) == expected,
                f"existing baseline reference artifact differs: {target}",
            )
        else:
            os.link(source, target)
            _require(
                sha256_file(target) == expected,
                f"linked baseline reference artifact differs: {target}",
            )
        linked[name] = expected
    return {
        **contract,
        "baseline_reference_view": str(BASELINE_VIEW_RUN),
        "reference_view_artifact_sha256": linked,
        "reference_view_uses_hard_links": True,
    }


def _run_artifact_sha256(
    run_dir: Path,
    checkpoint: str,
    evaluator_path: Path,
) -> dict[str, str]:
    return {
        "protocol.json": sha256_file(run_dir / "protocol.json"),
        "split.json": sha256_file(run_dir / "split.json"),
        "summary.json": sha256_file(run_dir / "summary.json"),
        "metrics.jsonl": sha256_file(run_dir / "metrics.jsonl"),
        "checkpoint": sha256_file(run_dir / checkpoint),
        "evaluator": sha256_file(evaluator_path),
    }


def current_sweep_binding(
    *,
    variant: str,
    checkpoint: str,
) -> dict[str, Any]:
    """Bind one sweep to the current completed run and evaluator bytes."""

    role = _checkpoint_role(checkpoint)
    if variant in VARIANTS:
        run_dir = RUN_DIRS[variant].resolve()
        evaluator_path = NER_EVALUATOR.resolve()
        artifact_identity = ner_eval.validate_run_artifacts(
            run_dir,
            checkpoint,
        )
        _require(
            artifact_identity.get("variant") == variant,
            "NER preflight variant differs",
        )
        _require(
            artifact_identity.get("checkpoint_filename") == checkpoint,
            "NER preflight checkpoint filename differs",
        )
        _require(
            artifact_identity.get("checkpoint_role") == role,
            "NER preflight checkpoint role differs",
        )
    elif variant == "baseline_sctransnet":
        prepare_baseline_reference_view()
        run_dir = BASELINE_VIEW_RUN.resolve()
        evaluator_path = BASELINE_EVALUATOR.resolve()
        artifact_identity = None
    else:
        raise ValueError(f"unexpected sweep variant: {variant}")
    split = load_json(run_dir / "split.json")
    split_hashes = split.get("hashes")
    _require(
        isinstance(split_hashes, Mapping),
        f"{variant} split hashes are missing",
    )
    validation_split_sha256 = split_hashes.get("used_val_sha256")
    _require(
        isinstance(validation_split_sha256, str),
        f"{variant} validation split hash is missing",
    )
    artifact_hashes = _run_artifact_sha256(
        run_dir,
        checkpoint,
        evaluator_path,
    )
    return {
        "variant": variant,
        "run_dir": run_dir,
        "checkpoint_path": (run_dir / checkpoint).resolve(),
        "checkpoint_name": checkpoint,
        "checkpoint_role": role,
        "checkpoint_sha256": artifact_hashes["checkpoint"],
        "validation_split_sha256": validation_split_sha256,
        "evaluator_path": evaluator_path,
        "evaluator_sha256": artifact_hashes["evaluator"],
        "artifact_sha256": artifact_hashes,
        "artifact_identity": artifact_identity,
    }


def sweep_path(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unexpected checkpoint: {checkpoint}")
    return Path(run_dir) / f"pd_fa_sweep_{Path(checkpoint).stem}.json"


def _checkpoint_role(checkpoint: str) -> str:
    try:
        return CHECKPOINT_ROLES[checkpoint]
    except KeyError as exc:
        raise ValueError(f"unexpected checkpoint: {checkpoint}") from exc


def _absolute_path_field(location: str, value: Any) -> Path:
    raw = Path(str(value))
    if not raw.is_absolute() or raw != raw.resolve():
        raise ValueError(f"{location} must be an absolute normalized path")
    return raw


def _validate_common_sweep_binding(
    payload: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    label: str,
) -> None:
    run_dir = _absolute_path_field(
        f"{label}.run_directory",
        payload.get("run_directory"),
    )
    checkpoint_path = _absolute_path_field(
        f"{label}.checkpoint",
        payload.get("checkpoint"),
    )
    _require(
        run_dir == binding["run_dir"],
        f"{label} run directory differs from current run",
    )
    _require(
        checkpoint_path == binding["checkpoint_path"],
        f"{label} checkpoint path differs from current checkpoint",
    )
    _require(
        checkpoint_path.parent == run_dir,
        f"{label} checkpoint is outside its run directory",
    )
    _require(
        payload.get("checkpoint_role") == binding["checkpoint_role"],
        f"{label} checkpoint role differs",
    )
    _require(
        payload.get("checkpoint_sha256") == binding["checkpoint_sha256"],
        f"{label} checkpoint SHA differs from current checkpoint",
    )
    _require(payload.get("seed") == TRAINING_SEED, f"{label} seed differs")
    _require(
        payload.get("split_seed") == SPLIT_SEED,
        f"{label} split seed differs",
    )
    _require(
        payload.get("validation_count") == 133,
        f"{label} validation count differs",
    )
    _require(
        payload.get("validation_split_sha256")
        == binding["validation_split_sha256"],
        f"{label} validation split SHA differs",
    )
    _require(
        payload.get("match_radius") == 3.0,
        f"{label} match radius differs",
    )
    _require(payload.get("tiny_area") == 9, f"{label} tiny area differs")
    threshold_configuration = payload.get("threshold_configuration")
    _require(
        isinstance(threshold_configuration, Mapping),
        f"{label} threshold configuration is missing",
    )
    expected_threshold_configuration = {
        "threshold_min": 0.01,
        "threshold_max": 0.99,
        "threshold_step": 0.01,
        "extra_thresholds": [0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999],
        "tail_logit_step": 0.1,
        "fa_budgets": list(FA_BUDGETS),
    }
    _require(
        dict(threshold_configuration) == expected_threshold_configuration,
        f"{label} threshold configuration differs",
    )
    audit = payload.get("audit")
    _require(isinstance(audit, Mapping), f"{label} audit is missing")
    _require(
        audit.get("expected_epochs") == EXPECTED_EPOCHS,
        f"{label} expected epochs differ",
    )
    _require(
        audit.get("metrics_event_count") == EXPECTED_EPOCHS,
        f"{label} metrics count differs",
    )
    _require(
        audit.get("metrics_epoch_range") == [1, EXPECTED_EPOCHS],
        f"{label} metrics epoch range differs",
    )
    _require(
        audit.get("summary_status") == "complete",
        f"{label} summary status differs",
    )
    invocation = audit.get("invocation_argv")
    invocation_path = (
        Path(str(invocation[1]))
        if isinstance(invocation, list) and len(invocation) >= 2
        else None
    )
    _require(
        invocation_path is not None
        and invocation_path.is_absolute()
        and invocation_path == invocation_path.resolve()
        and invocation_path == binding["evaluator_path"],
        f"{label} evaluator invocation differs",
    )
    parsed = audit.get("parsed_arguments")
    _require(
        isinstance(parsed, Mapping),
        f"{label} parsed arguments are missing",
    )
    parsed_run_dir = _absolute_path_field(
        f"{label}.audit.parsed_arguments.run_dir",
        parsed.get("run_dir"),
    )
    _require(
        parsed_run_dir == binding["run_dir"],
        f"{label} parsed run directory differs",
    )
    _require(
        parsed.get("checkpoint") == binding["checkpoint_name"],
        f"{label} parsed checkpoint differs",
    )
    for name, expected in {
        "expected_epochs": EXPECTED_EPOCHS,
        "threshold_min": 0.01,
        "threshold_max": 0.99,
        "threshold_step": 0.01,
        "extra_thresholds": [0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999],
        "tail_logit_step": 0.1,
        "fa_budgets": list(FA_BUDGETS),
        "match_radius": None,
        "tiny_area": None,
        "overwrite": False,
    }.items():
        _require(
            parsed.get(name) == expected,
            f"{label} parsed evaluator argument differs: {name}",
        )
    artifact_hashes = audit.get("artifact_sha256")
    _require(
        artifact_hashes == binding["artifact_sha256"],
        f"{label} artifact SHA binding differs",
    )
    checks = audit.get("integrity_checks_passed")
    _require(
        isinstance(checks, Mapping)
        and checks
        and all(value is True for value in checks.values()),
        f"{label} evaluator checks are incomplete",
    )


def _validate_ner_closed_interval(payload: Mapping[str, Any]) -> None:
    provenance = payload.get("threshold_provenance")
    _require(
        isinstance(provenance, Mapping),
        "NER threshold provenance is missing",
    )
    for name, expected in {
        "closed_probability_interval": True,
        "score_dtype": "float32",
        "posthoc_endpoint_completion": False,
        "preregistered_endpoint_completion": True,
        "endpoint_protocol_stage": "before_formal_training",
        "last_float32_below_one": ner_eval.LAST_FLOAT32_BELOW_ONE,
        "upper_boundary_threshold": ner_eval.UPPER_BOUNDARY_THRESHOLD,
        "upper_boundary_comparison": "prediction > threshold",
        "upper_boundary_semantics": "empty_prediction_pd0_fa0",
    }.items():
        _require(
            provenance.get(name) == expected,
            f"NER threshold provenance differs: {name}",
        )
    _require(
        list(map(float, provenance.get("added_thresholds", ())))
        == [
            ner_eval.LAST_FLOAT32_BELOW_ONE,
            ner_eval.UPPER_BOUNDARY_THRESHOLD,
        ],
        "NER closed-interval added thresholds differ",
    )
    points = payload.get("points")
    _require(isinstance(points, list), "NER sweep points are missing")
    by_threshold = {
        float(point["threshold"]): point
        for point in points
        if isinstance(point, Mapping) and "threshold" in point
    }
    _require(
        ner_eval.LAST_FLOAT32_BELOW_ONE in by_threshold
        and ner_eval.UPPER_BOUNDARY_THRESHOLD in by_threshold,
        "NER closed-interval endpoint points are missing",
    )
    upper = by_threshold[ner_eval.UPPER_BOUNDARY_THRESHOLD]
    for name, expected in {
        "pd": 0.0,
        "fa": 0.0,
        "matched_target_count": 0,
        "predicted_object_count": 0,
        "unmatched_predicted_object_count": 0,
    }.items():
        _require(
            upper.get(name) == expected,
            f"NER upper endpoint differs: {name}",
        )


def _normalize_point(location: str, point: Any) -> dict[str, Any]:
    _require(isinstance(point, Mapping), f"{location} is missing")
    target_count = point.get("target_count")
    matched = point.get("matched_target_count")
    _require(type(target_count) is int, f"{location}.target_count is invalid")
    _require(type(matched) is int, f"{location}.matched_target_count is invalid")
    _require(target_count == TARGET_COUNT, f"{location}.target_count differs")
    _require(0 <= matched <= target_count, f"{location}.matched count is invalid")
    pd = _finite_number(f"{location}.pd", point.get("pd"))
    fa = _finite_number(f"{location}.fa", point.get("fa"))
    miou = _finite_number(f"{location}.miou", point.get("miou"))
    false_per_image = _finite_number(
        f"{location}.false_objects_per_image",
        point.get("false_objects_per_image"),
    )
    threshold = _finite_number(f"{location}.threshold", point.get("threshold"))
    _require(abs(pd - matched / target_count) <= 1e-12, f"{location}.Pd differs")
    _require(0.0 <= fa <= 1.0, f"{location}.Fa is invalid")
    _require(0.0 <= miou <= 1.0, f"{location}.mIoU is invalid")
    _require(false_per_image >= 0.0, f"{location}.false objects is invalid")
    return {
        "matched_target_count": matched,
        "target_count": target_count,
        "pd": pd,
        "fa": fa,
        "miou": miou,
        "false_objects_per_image": false_per_image,
        "threshold": threshold,
    }


def _absolute_gate(role: str, fixed: Mapping[str, Any], budgets: Mapping[str, Any]) -> dict[str, Any]:
    contract = ner_eval.performance_gate_contract()
    gate_name = (
        "pd_primary_fixed_threshold_0_5"
        if role == "best_validation_pd_primary"
        else "miou_selected_fixed_threshold_0_5"
    )
    requirement = contract[gate_name]
    fixed_checks = {
        "matched_targets": (
            int(fixed["matched_target_count"])
            >= int(requirement["minimum_matched_targets"])
        ),
        "pd": float(fixed["pd"]) >= float(requirement["minimum_pd"]),
        "fa": float(fixed["fa"]) <= float(requirement["maximum_fa"]),
        "miou": float(fixed["miou"]) >= float(requirement["minimum_miou"]),
    }
    budget_checks: dict[str, Any] = {}
    for key in BUDGET_KEYS:
        point = budgets[key]
        needed = contract["pd_at_fa_budget"][key]
        checks = {
            "matched_targets": (
                int(point["matched_target_count"])
                >= int(needed["minimum_matched_targets"])
            ),
            "pd": float(point["pd"]) >= float(needed["minimum_pd"]),
        }
        budget_checks[key] = {
            "required_matched_target_count": needed["minimum_matched_targets"],
            "required_pd": needed["minimum_pd"],
            "checks": checks,
            "passed": all(checks.values()),
        }
    passed = all(fixed_checks.values()) and all(
        record["passed"] for record in budget_checks.values()
    )
    return {
        "fixed_threshold_gate": gate_name,
        "fixed_threshold_checks": fixed_checks,
        "budget_checks": budget_checks,
        "passed": passed,
    }


def normalize_ner_sweep(
    payload: Mapping[str, Any],
    *,
    variant: str,
    checkpoint: str,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = (
        current_sweep_binding(variant=variant, checkpoint=checkpoint)
        if binding is None
        else dict(binding)
    )
    role = _checkpoint_role(checkpoint)
    _validate_common_sweep_binding(payload, current, label="NER sweep")
    _require(payload.get("schema") == ner_eval.EVALUATION_SCHEMA, "NER sweep schema differs")
    _require(payload.get("variant") == variant, "NER sweep variant differs")
    _require(payload.get("seed") == TRAINING_SEED, "NER sweep seed differs")
    _require(payload.get("split_seed") == SPLIT_SEED, "NER sweep split seed differs")
    _require(
        payload.get("official_test_accessed") is False,
        "NER sweep official-test marker differs",
    )
    _require(
        Path(str(payload.get("checkpoint"))).name == checkpoint,
        "NER sweep checkpoint filename differs",
    )
    _require(payload.get("checkpoint_role") == role, "NER sweep role differs")
    artifact_identity = current.get("artifact_identity")
    _require(
        isinstance(artifact_identity, Mapping),
        "NER current artifact identity is missing",
    )
    _require(
        payload.get("artifact_identity_preflight_passed") is True,
        "NER artifact identity preflight marker differs",
    )
    _require(
        payload.get("run_identity") == artifact_identity.get("run_identity"),
        "NER sweep run identity differs from current run",
    )
    _require(
        payload.get("source_checkpoint_identity")
        == artifact_identity.get("checkpoint_identity"),
        "NER sweep source checkpoint identity differs",
    )
    _require(
        payload.get("training_artifact_mode")
        == artifact_identity.get("training_artifact_mode"),
        "NER sweep training artifact mode differs",
    )
    evaluated_identity = payload.get("evaluated_checkpoint_identity")
    _require(
        evaluated_identity
        == {
            "training_artifact_mode": artifact_identity.get(
                "training_artifact_mode"
            ),
            "filename": checkpoint,
            "role": role,
            "sha256": current["checkpoint_sha256"],
        },
        "NER evaluated checkpoint identity differs",
    )
    _require(
        payload.get("evaluator_contract") == ner_eval.evaluator_contract(),
        "NER evaluator contract differs",
    )
    _validate_ner_closed_interval(payload)
    coverage = payload.get("final_metric_coverage")
    _require(isinstance(coverage, Mapping), "NER final metric coverage missing")
    _require(
        coverage.get("schema") == ner_eval.FINAL_METRIC_COVERAGE_SCHEMA,
        "NER final metric coverage schema differs",
    )
    _require(
        coverage.get("fixed_threshold") == 0.5,
        "NER final metric coverage threshold differs",
    )
    _require(
        coverage.get("all_required_metrics_present") is True,
        "NER final metric coverage is incomplete",
    )
    fixed = _normalize_point("NER fixed_threshold_0_5", payload.get("fixed_threshold_0_5"))
    _require(abs(fixed["threshold"] - 0.5) <= 1e-12, "NER fixed threshold differs")
    raw_budgets = payload.get("best_points_under_fa_budget")
    _require(isinstance(raw_budgets, Mapping), "NER budget points missing")
    _require(set(raw_budgets) == set(BUDGET_KEYS), "NER budget keys differ")
    budgets: dict[str, Any] = {}
    for budget, key in zip(FA_BUDGETS, BUDGET_KEYS):
        point = _normalize_point(f"NER budget {key}", raw_budgets[key])
        _require(point["fa"] <= budget, f"NER budget {key} exceeds Fa")
        budgets[key] = point
    _require(
        coverage.get("fixed_threshold_0_5")
        == {
            name: payload["fixed_threshold_0_5"][name]
            for name in (
                "pd",
                "fa",
                "miou",
                "false_objects_per_image",
            )
        },
        "NER fixed metric coverage differs from raw sweep",
    )
    expected_budget_coverage = {
        key: {
            "budget": budget,
            "pd": raw_budgets[key]["pd"],
            "achieved_fa": raw_budgets[key]["fa"],
            "threshold": raw_budgets[key]["threshold"],
            "matched_target_count": raw_budgets[key][
                "matched_target_count"
            ],
            "target_count": raw_budgets[key]["target_count"],
        }
        for budget, key in zip(FA_BUDGETS, BUDGET_KEYS)
    }
    _require(
        coverage.get("pd_at_fa_budget") == expected_budget_coverage,
        "NER budget metric coverage differs from raw sweep",
    )
    gate = _absolute_gate(role, fixed, budgets)
    recorded = payload.get("performance_gate_assessment")
    _require(isinstance(recorded, Mapping), "NER recorded gate assessment missing")
    _require(
        recorded.get("contract") == ner_eval.performance_gate_contract(),
        "NER recorded gate contract differs",
    )
    _require(
        recorded.get("fixed_threshold_gate")
        == gate["fixed_threshold_gate"],
        "NER recorded fixed gate role differs",
    )
    expected_fixed_observed = {
        "matched_target_count": fixed["matched_target_count"],
        "target_count": fixed["target_count"],
        "pd": fixed["pd"],
        "fa": fixed["fa"],
        "miou": fixed["miou"],
    }
    _require(
        recorded.get("fixed_threshold_observed")
        == expected_fixed_observed,
        "NER recorded fixed observations differ",
    )
    _require(
        recorded.get("fixed_threshold_checks")
        == gate["fixed_threshold_checks"],
        "NER recorded fixed checks differ",
    )
    expected_recorded_budgets = {
        key: {
            "required_matched_target_count": gate["budget_checks"][key][
                "required_matched_target_count"
            ],
            "required_pd": gate["budget_checks"][key]["required_pd"],
            "observed_matched_target_count": budgets[key][
                "matched_target_count"
            ],
            "observed_target_count": budgets[key]["target_count"],
            "observed_pd": budgets[key]["pd"],
            "checks": gate["budget_checks"][key]["checks"],
            "passed": gate["budget_checks"][key]["passed"],
        }
        for key in BUDGET_KEYS
    }
    _require(
        recorded.get("budget_checks") == expected_recorded_budgets,
        "NER recorded budget checks differ",
    )
    _require(
        recorded.get("absolute_checkpoint_gate_passed") == gate["passed"],
        "NER recorded absolute gate differs from aggregate recomputation",
    )
    expected_paired_status = (
        "requires_relay_off_and_relay_on_aggregate"
        if variant == VARIANT_ON
        else "reference_control_not_applicable"
    )
    _require(
        recorded.get("paired_relay_on_gate_status")
        == expected_paired_status,
        "NER recorded paired-gate status differs",
    )
    _require(
        recorded.get("formal_success_claim_authorized") is False,
        "NER evaluator may not authorize aggregate success",
    )
    return {
        "source": "new_model",
        "variant": variant,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "checkpoint": checkpoint,
        "checkpoint_role": role,
        "checkpoint_epoch": payload.get("checkpoint_epoch"),
        "checkpoint_sha256": current["checkpoint_sha256"],
        "run_directory": str(current["run_dir"]),
        "run_identity": dict(artifact_identity["run_identity"]),
        "source_checkpoint_identity": dict(
            artifact_identity["checkpoint_identity"]
        ),
        "fixed_threshold_0_5": fixed,
        "pd_at_fa_budget": budgets,
        "absolute_gate": gate,
    }


def normalize_baseline_sweep(
    payload: Mapping[str, Any],
    *,
    checkpoint: str,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = (
        current_sweep_binding(
            variant="baseline_sctransnet",
            checkpoint=checkpoint,
        )
        if binding is None
        else dict(binding)
    )
    role = _checkpoint_role(checkpoint)
    baseline_eval.validate_output_identity(
        payload,
        expected_run_dir=current["run_dir"],
        expected_checkpoint=checkpoint,
    )
    _validate_common_sweep_binding(payload, current, label="baseline sweep")
    _require(
        Path(str(payload.get("checkpoint"))).name == checkpoint,
        "baseline checkpoint filename differs",
    )
    _require(payload.get("checkpoint_role") == role, "baseline checkpoint role differs")
    fixed = _normalize_point(
        "baseline fixed_threshold_0_5",
        payload.get("fixed_threshold_0_5"),
    )
    _require(abs(fixed["threshold"] - 0.5) <= 1e-12, "baseline fixed threshold differs")
    raw_budgets = payload.get("best_points_under_fa_budget")
    _require(isinstance(raw_budgets, Mapping), "baseline budget points missing")
    _require(set(raw_budgets) == set(BUDGET_KEYS), "baseline budget keys differ")
    budgets: dict[str, Any] = {}
    for budget, key in zip(FA_BUDGETS, BUDGET_KEYS):
        point = _normalize_point(f"baseline budget {key}", raw_budgets[key])
        _require(point["fa"] <= budget, f"baseline budget {key} exceeds Fa")
        budgets[key] = point
    return {
        "source": "same_protocol_external_reference",
        "variant": "baseline_sctransnet",
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "checkpoint": checkpoint,
        "checkpoint_role": role,
        "checkpoint_epoch": payload.get("checkpoint_epoch"),
        "checkpoint_sha256": current["checkpoint_sha256"],
        "run_directory": str(current["run_dir"]),
        "reference_provenance": dict(payload["reference_provenance"]),
        "fixed_threshold_0_5": fixed,
        "pd_at_fa_budget": budgets,
        "absolute_gate": None,
    }


def validate_existing_sweep(
    path: Path,
    *,
    variant: str,
    checkpoint: str,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = load_json(path)
    if variant == "baseline_sctransnet":
        return normalize_baseline_sweep(
            payload,
            checkpoint=checkpoint,
            binding=binding,
        )
    return normalize_ner_sweep(
        payload,
        variant=variant,
        checkpoint=checkpoint,
        binding=binding,
    )


def evaluation_command(
    *,
    variant: str,
    checkpoint: str,
    python: Path | str = sys.executable,
    device_mode: str = "gpu23",
) -> tuple[list[str], dict[str, str], Path]:
    if variant in VARIANTS:
        run_dir = RUN_DIRS[variant]
        evaluator = NER_EVALUATOR
        gpu_uuid = GPU_UUIDS[variant]
    elif variant == "baseline_sctransnet":
        run_dir = BASELINE_VIEW_RUN
        evaluator = BASELINE_EVALUATOR
        gpu_uuid = GPU_UUIDS[VARIANT_OFF]
    else:
        raise ValueError(f"unexpected evaluation variant: {variant}")
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unexpected checkpoint: {checkpoint}")
    if device_mode not in ("gpu23", "cpu"):
        raise ValueError(f"unexpected device mode: {device_mode}")
    device = "cuda:0" if device_mode == "gpu23" else "cpu"
    command = [
        str(python),
        str(evaluator),
        "--run-dir",
        str(run_dir),
        "--checkpoint",
        checkpoint,
        "--device",
        device,
        "--expected-epochs",
        str(EXPECTED_EPOCHS),
    ]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(TRAINING_SEED)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if device_mode == "gpu23":
        environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    else:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    return command, environment, sweep_path(run_dir, checkpoint)


def _new_rejected_directory(parent: Path) -> Path:
    root = parent / "rejected_postprocess"
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError(f"rejected-postprocess directory is a symlink: {root}")
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for index in range(1000):
        candidate = root / (
            timestamp if index == 0 else f"{timestamp}.{index:03d}"
        )
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("could not allocate a rejected-postprocess directory")


def quarantine_postprocess_artifacts(
    paths: Iterable[Path],
    *,
    parent: Path,
    reason: str,
) -> dict[str, Any]:
    """Move invalid/incomplete outputs aside without replacing or deleting."""

    existing = [
        Path(path)
        for path in paths
        if Path(path).exists() or Path(path).is_symlink()
    ]
    if not existing:
        return {"moved": {}, "reason": reason}
    destination = _new_rejected_directory(parent)
    moved: dict[str, str] = {}
    for source in existing:
        target = destination / source.name
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        os.replace(source, target)
        moved[str(source)] = str(target)
    reason_path = destination / "reason.json"
    content = (
        json.dumps(
            {
                "schema": "sctransnet_tpd_ner_rejected_postprocess_v1",
                "reason": reason,
                "moved": moved,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        reason_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "directory": str(destination),
        "moved": moved,
        "reason": reason,
    }


def _run_evaluation(
    *,
    variant: str,
    checkpoint: str,
    python: Path | str,
    device_mode: str,
) -> dict[str, Any]:
    command, environment, output = evaluation_command(
        variant=variant,
        checkpoint=checkpoint,
        python=python,
        device_mode=device_mode,
    )
    binding = current_sweep_binding(
        variant=variant,
        checkpoint=checkpoint,
    )
    rejected: list[dict[str, Any]] = []
    if output.exists() or output.is_symlink():
        try:
            row = validate_existing_sweep(
                output,
                variant=variant,
                checkpoint=checkpoint,
                binding=binding,
            )
        except Exception as exc:
            rejected.append(
                quarantine_postprocess_artifacts(
                    [output],
                    parent=output.parent,
                    reason=(
                        "existing sweep failed current artifact binding: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
        else:
            return {
                "variant": variant,
                "checkpoint": checkpoint,
                "status": "reused_valid_existing",
                "output": str(output),
                "row": row,
                "rejected_previous_outputs": rejected,
            }
    print(
        f"EVALUATE variant={variant} checkpoint={checkpoint} "
        f"device={command[command.index('--device') + 1]}",
        flush=True,
    )
    try:
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            check=True,
        )
    except BaseException as exc:
        if output.exists() or output.is_symlink():
            rejected.append(
                quarantine_postprocess_artifacts(
                    [output],
                    parent=output.parent,
                    reason=(
                        "evaluator exited before a valid result was accepted: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
        raise
    post_binding = current_sweep_binding(
        variant=variant,
        checkpoint=checkpoint,
    )
    try:
        row = validate_existing_sweep(
            output,
            variant=variant,
            checkpoint=checkpoint,
            binding=post_binding,
        )
    except BaseException as exc:
        if output.exists() or output.is_symlink():
            rejected.append(
                quarantine_postprocess_artifacts(
                    [output],
                    parent=output.parent,
                    reason=(
                        "new sweep failed current artifact binding: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
        raise
    return {
        "variant": variant,
        "checkpoint": checkpoint,
        "status": "completed",
        "output": str(output),
        "row": row,
        "rejected_previous_outputs": rejected,
    }


def run_role_synchronized_evaluations(
    *,
    python: Path | str = sys.executable,
    device_mode: str = "gpu23",
) -> list[dict[str, Any]]:
    """Run off/on in parallel for each role, best before best_miou."""

    readiness = inspect_training_readiness()
    if readiness["both_runs_complete"] is not True:
        raise IncompleteTraining(
            "both NER runs must have complete summaries and 800 contiguous "
            "metrics before evaluation"
        )
    prepare_baseline_reference_view()
    records: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    _run_evaluation,
                    variant=variant,
                    checkpoint=checkpoint,
                    python=python,
                    device_mode=device_mode,
                )
                for variant in VARIANTS
            ]
            pair = [future.result() for future in futures]
        records.extend(pair)
        # The external baseline is evaluated after the paired candidate role,
        # on the GPU2 lane (or CPU in explicit CPU mode).
        records.append(
            _run_evaluation(
                variant="baseline_sctransnet",
                checkpoint=checkpoint,
                python=python,
                device_mode=device_mode,
            )
        )
    return records


def _paired_gate(off: Mapping[str, Any], on: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        off.get("checkpoint_role") == on.get("checkpoint_role"),
        "paired rows have different checkpoint roles",
    )
    comparisons: dict[str, Any] = {}
    non_inferior = 0
    strictly_better = 0
    for key in BUDGET_KEYS:
        off_point = off["pd_at_fa_budget"][key]
        on_point = on["pd_at_fa_budget"][key]
        off_count = int(off_point["matched_target_count"])
        on_count = int(on_point["matched_target_count"])
        no_worse = on_count >= off_count
        better = on_count > off_count
        non_inferior += int(no_worse)
        strictly_better += int(better)
        comparisons[key] = {
            "relay_off_matched_target_count": off_count,
            "relay_on_matched_target_count": on_count,
            "relay_off_pd": off_point["pd"],
            "relay_on_pd": on_point["pd"],
            "relay_on_non_inferior": no_worse,
            "relay_on_strictly_better": better,
        }
    contract = ner_eval.performance_gate_contract()[
        "relay_on_paired_budget_gate"
    ]
    passed = bool(
        non_inferior >= contract["minimum_non_inferior_budget_count"]
        and strictly_better >= contract["minimum_strictly_better_budget_count"]
    )
    return {
        "checkpoint_role": off["checkpoint_role"],
        "comparisons": comparisons,
        "non_inferior_budget_count": non_inferior,
        "strictly_better_budget_count": strictly_better,
        "required_non_inferior_budget_count": contract[
            "minimum_non_inferior_budget_count"
        ],
        "required_strictly_better_budget_count": contract[
            "minimum_strictly_better_budget_count"
        ],
        "passed": passed,
    }


def _compare_row_to_baseline(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        candidate["checkpoint_role"] == baseline["checkpoint_role"],
        "candidate and baseline roles differ",
    )
    fixed_candidate = candidate["fixed_threshold_0_5"]
    fixed_baseline = baseline["fixed_threshold_0_5"]
    budgets: dict[str, Any] = {}
    for key in BUDGET_KEYS:
        c_point = candidate["pd_at_fa_budget"][key]
        b_point = baseline["pd_at_fa_budget"][key]
        budgets[key] = {
            "delta_matched_targets": (
                int(c_point["matched_target_count"])
                - int(b_point["matched_target_count"])
            ),
            "delta_pd": float(c_point["pd"]) - float(b_point["pd"]),
            "candidate_achieved_fa": c_point["fa"],
            "baseline_achieved_fa": b_point["fa"],
        }
    return {
        "variant": candidate["variant"],
        "checkpoint_role": candidate["checkpoint_role"],
        "fixed_threshold_0_5_delta_candidate_minus_baseline": {
            "matched_targets": (
                int(fixed_candidate["matched_target_count"])
                - int(fixed_baseline["matched_target_count"])
            ),
            "pd": float(fixed_candidate["pd"]) - float(fixed_baseline["pd"]),
            "fa": float(fixed_candidate["fa"]) - float(fixed_baseline["fa"]),
            "miou": (
                float(fixed_candidate["miou"]) - float(fixed_baseline["miou"])
            ),
            "false_objects_per_image": (
                float(fixed_candidate["false_objects_per_image"])
                - float(fixed_baseline["false_objects_per_image"])
            ),
        },
        "pd_at_fa_budget": budgets,
    }


def load_all_rows() -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for checkpoint in CHECKPOINTS:
        for variant, run_dir in (
            (VARIANT_OFF, RUN_DIRS[VARIANT_OFF]),
            (VARIANT_ON, RUN_DIRS[VARIANT_ON]),
            ("baseline_sctransnet", BASELINE_VIEW_RUN),
        ):
            path = sweep_path(run_dir, checkpoint)
            binding = current_sweep_binding(
                variant=variant,
                checkpoint=checkpoint,
            )
            rows[(variant, checkpoint)] = validate_existing_sweep(
                path,
                variant=variant,
                checkpoint=checkpoint,
                binding=binding,
            )
    return rows


def build_report(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    lock_bindings: Mapping[str, Any],
    baseline_contract: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        (variant, checkpoint)
        for variant in (*VARIANTS, "baseline_sctransnet")
        for checkpoint in CHECKPOINTS
    }
    _require(set(rows) == expected, "aggregate row matrix differs")
    paired: dict[str, Any] = {}
    absolute: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []
    ordered_rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        role = _checkpoint_role(checkpoint)
        role_name = ROLE_NAMES[role]
        off = rows[(VARIANT_OFF, checkpoint)]
        on = rows[(VARIANT_ON, checkpoint)]
        baseline = rows[("baseline_sctransnet", checkpoint)]
        paired[role_name] = _paired_gate(off, on)
        for candidate in (off, on):
            key = f"{candidate['variant']}:{role_name}"
            absolute[key] = candidate["absolute_gate"]
            comparisons.append(_compare_row_to_baseline(candidate, baseline))
        ordered_rows.extend([dict(baseline), dict(off), dict(on)])
    all_absolute = all(record["passed"] for record in absolute.values())
    all_paired = all(record["passed"] for record in paired.values())
    aggregate_passed = all_absolute and all_paired
    sweep_bindings = {
        f"{variant}:{checkpoint}": {
            "path": str(
                sweep_path(
                    BASELINE_VIEW_RUN
                    if variant == "baseline_sctransnet"
                    else RUN_DIRS[variant],
                    checkpoint,
                )
            ),
            "sha256": sha256_file(
                sweep_path(
                    BASELINE_VIEW_RUN
                    if variant == "baseline_sctransnet"
                    else RUN_DIRS[variant],
                    checkpoint,
                )
            ),
        }
        for variant, checkpoint in sorted(expected)
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "official_test_accessed": False,
        "scope": "single_seed_internal_validation",
        "rows": ordered_rows,
        "absolute_gate_assessments": absolute,
        "paired_relay_on_gate_by_role": paired,
        "all_four_absolute_checkpoint_gates_passed": all_absolute,
        "both_role_paired_relay_on_gates_passed": all_paired,
        "aggregate_full_model_gate_passed": aggregate_passed,
        "decision": (
            "FULL_MODEL_GATE_PASSED"
            if aggregate_passed
            else "RETURN_TO_MODEL_OPTIMIZATION"
        ),
        "comparisons_vs_baseline": comparisons,
        "baseline_reference_qualification": dict(baseline_contract),
        "bindings": {
            **dict(lock_bindings),
            "ner_evaluator": str(NER_EVALUATOR),
            "ner_evaluator_sha256": sha256_file(NER_EVALUATOR),
            "baseline_reference_evaluator": str(BASELINE_EVALUATOR),
            "baseline_reference_evaluator_sha256": sha256_file(
                BASELINE_EVALUATOR
            ),
            "postprocess": str(Path(__file__).resolve()),
            "postprocess_sha256": sha256_file(Path(__file__).resolve()),
            "sweeps": sweep_bindings,
        },
        "claim_boundary": {
            "single_seed_only": True,
            "cross_seed_stability_claim": False,
            "cross_dataset_claim": False,
            "official_test_claim": False,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V8-MPRS-DCH + five-node NER formal800 comparison",
        "",
        f"- Decision: `{report['decision']}`",
        (
            "- Aggregate full-model gate passed: "
            f"`{str(report['aggregate_full_model_gate_passed']).lower()}`"
        ),
        "- Scope: seed 42, NUDT-SIRST internal 530/133 validation",
        "- Official test accessed: `false`",
        "",
        "## Fixed threshold 0.5 and Pd@Fa",
        "",
        "| Source | Variant | Role | Pd@0.5 | Fa@0.5 | mIoU@0.5 | False objects/image@0.5 | Pd@1e-6 | Pd@5e-6 | Pd@1e-5 | Pd@5e-5 | Pd@1e-4 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        fixed = row["fixed_threshold_0_5"]
        budget_cells = []
        for key in BUDGET_KEYS:
            point = row["pd_at_fa_budget"][key]
            budget_cells.append(
                f"{point['matched_target_count']}/{point['target_count']} "
                f"({point['pd']:.9f}; Fa={point['fa']:.9g})"
            )
        lines.append(
            f"| {row['source']} | {row['variant']} | "
            f"{ROLE_NAMES[row['checkpoint_role']]} | "
            f"{fixed['matched_target_count']}/{fixed['target_count']} "
            f"({fixed['pd']:.9f}) | {fixed['fa']:.9g} | "
            f"{fixed['miou']:.9f} | "
            f"{fixed['false_objects_per_image']:.9f} | "
            + " | ".join(budget_cells)
            + " |"
        )
    lines.extend(["", "## Frozen gate outcome", ""])
    for name, gate in report["absolute_gate_assessments"].items():
        lines.append(f"- `{name}` absolute gate: `{str(gate['passed']).lower()}`")
    for name, gate in report["paired_relay_on_gate_by_role"].items():
        lines.append(
            f"- `{name}` paired relay-on gate: `{str(gate['passed']).lower()}` "
            f"({gate['non_inferior_budget_count']}/5 non-inferior, "
            f"{gate['strictly_better_budget_count']}/5 strictly better)"
        )
    lines.extend(
        [
            "",
            "## Conclusion boundary",
            "",
            "- This report covers the fixed seed-42 internal-validation model decision.",
            "- It does not claim cross-seed stability, cross-dataset transfer, or official-test performance.",
            "",
        ]
    )
    return "\n".join(lines)


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_publish_new(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
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
            raise FileExistsError(path) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _completion_marker_bytes(
    report: Mapping[str, Any],
    json_bytes: bytes,
    markdown_bytes: bytes,
) -> bytes:
    marker = {
        "schema": "sctransnet_tpd_ner_v8_mprs_dch_postprocess_complete_v1",
        "status": "complete",
        "decision": report["decision"],
        "aggregate_full_model_gate_passed": report[
            "aggregate_full_model_gate_passed"
        ],
        "outputs": {
            JSON_OUTPUT.name: hashlib.sha256(json_bytes).hexdigest(),
            MARKDOWN_OUTPUT.name: hashlib.sha256(markdown_bytes).hexdigest(),
        },
    }
    return _canonical_bytes(marker)


def write_report(report: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    if COMPARISON_DIR.is_symlink():
        raise ValueError("comparison directory may not be a symlink")
    json_bytes = _canonical_bytes(report)
    markdown_bytes = render_markdown(report).encode("utf-8")
    marker_bytes = _completion_marker_bytes(
        report,
        json_bytes,
        markdown_bytes,
    )
    expected = {
        JSON_OUTPUT: json_bytes,
        MARKDOWN_OUTPUT: markdown_bytes,
    }
    conflicting = [
        path
        for path, content in expected.items()
        if (path.exists() or path.is_symlink())
        and (
            not path.is_file()
            or path.is_symlink()
            or path.read_bytes() != content
        )
    ]
    if conflicting:
        quarantine_postprocess_artifacts(
            [JSON_OUTPUT, MARKDOWN_OUTPUT, COMPLETE_MARKER],
            parent=COMPARISON_DIR,
            reason=(
                "existing final report files differ from the fully rebound "
                "aggregate"
            ),
        )
    for path, content in expected.items():
        if not path.exists() and not path.is_symlink():
            _atomic_publish_new(path, content)
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.read_bytes() == content,
            f"final report publication differs: {path}",
        )
    if COMPLETE_MARKER.exists() or COMPLETE_MARKER.is_symlink():
        marker_valid = bool(
            COMPLETE_MARKER.is_file()
            and not COMPLETE_MARKER.is_symlink()
            and COMPLETE_MARKER.read_bytes() == marker_bytes
        )
        if not marker_valid:
            quarantine_postprocess_artifacts(
                [COMPLETE_MARKER],
                parent=COMPARISON_DIR,
                reason="existing completion marker differs",
            )
    if not COMPLETE_MARKER.exists() and not COMPLETE_MARKER.is_symlink():
        _atomic_publish_new(COMPLETE_MARKER, marker_bytes)
    _require(
        COMPLETE_MARKER.is_file()
        and not COMPLETE_MARKER.is_symlink()
        and COMPLETE_MARKER.read_bytes() == marker_bytes,
        "postprocess completion marker differs",
    )
    return JSON_OUTPUT, MARKDOWN_OUTPUT, COMPLETE_MARKER


def aggregate_and_write() -> tuple[dict[str, Any], tuple[Path, Path, Path]]:
    readiness = inspect_training_readiness()
    if readiness["both_runs_complete"] is not True:
        raise IncompleteTraining(
            "aggregate requires both summaries complete and two contiguous "
            "800-epoch metrics histories"
        )
    locks = verify_frozen_manifests()
    baseline_contract = _same_split_and_training_contract()
    rows = load_all_rows()
    report = build_report(
        rows,
        lock_bindings=locks,
        baseline_contract=baseline_contract,
    )
    report["readiness_binding"] = readiness
    return report, write_report(report)


def execution_plan(
    *,
    python: Path | str = sys.executable,
    device_mode: str = "gpu23",
) -> dict[str, Any]:
    phases = []
    for checkpoint in CHECKPOINTS:
        pair = []
        for variant in VARIANTS:
            command, _, output = evaluation_command(
                variant=variant,
                checkpoint=checkpoint,
                python=python,
                device_mode=device_mode,
            )
            pair.append(
                {
                    "variant": variant,
                    "physical_gpu_index": GPU_INDICES[variant],
                    "command": command,
                    "output": str(output),
                }
            )
        baseline_command, _, baseline_output = evaluation_command(
            variant="baseline_sctransnet",
            checkpoint=checkpoint,
            python=python,
            device_mode=device_mode,
        )
        phases.append(
            {
                "checkpoint": checkpoint,
                "parallel_candidate_pair": pair,
                "then_baseline_reference": {
                    "physical_gpu_index": 2,
                    "command": baseline_command,
                    "output": str(baseline_output),
                },
            }
        )
    return {
        "readiness": inspect_training_readiness(),
        "evaluation_order": phases,
        "aggregate_outputs": [
            str(JSON_OUTPUT),
            str(MARKDOWN_OUTPUT),
            str(COMPLETE_MARKER),
        ],
        "existing_frozen_manifests_modified": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess V8-MPRS-DCH+NER formal800"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run-now", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    mode.add_argument("--wait-and-run", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--device-mode",
        choices=("gpu23", "cpu"),
        default="gpu23",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or not 1.0 <= args.poll_seconds <= 60.0:
        parser.error("--poll-seconds must lie in [1, 60]")
    return args


def _print_json(value: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.status:
        _print_json(inspect_training_readiness())
        return
    if args.plan:
        verify_frozen_manifests()
        _print_json(
            execution_plan(
                python=args.python,
                device_mode=args.device_mode,
            )
        )
        return
    if args.aggregate_only:
        report, paths = aggregate_and_write()
        print(
            f"AGGREGATE decision={report['decision']} "
            f"json={paths[0]} markdown={paths[1]} marker={paths[2]}",
            flush=True,
        )
        return
    verify_frozen_manifests()
    if args.wait_and_run:
        last_counts: tuple[int, int] | None = None
        while True:
            readiness = inspect_training_readiness()
            counts = tuple(
                int(readiness["runs"][variant]["metrics"]["event_count"])
                for variant in VARIANTS
            )
            if counts != last_counts:
                print(
                    f"WAIT relay_off={counts[0]}/{EXPECTED_EPOCHS} "
                    f"relay_on={counts[1]}/{EXPECTED_EPOCHS}",
                    flush=True,
                )
                last_counts = counts
            if readiness["both_runs_complete"]:
                break
            time.sleep(args.poll_seconds)
    run_role_synchronized_evaluations(
        python=args.python,
        device_mode=args.device_mode,
    )
    report, paths = aggregate_and_write()
    print(
        f"COMPLETE decision={report['decision']} "
        f"json={paths[0]} markdown={paths[1]} marker={paths[2]}",
        flush=True,
    )


__all__ = [
    "ACCEPTANCE_LOCK",
    "BASELINE_SOURCE_RUN",
    "BASELINE_VIEW_RUN",
    "BUDGET_KEYS",
    "CHECKPOINTS",
    "COMPARISON_DIR",
    "EXPECTED_EPOCHS",
    "FA_BUDGETS",
    "GPU_INDICES",
    "GPU_UUIDS",
    "IncompleteTraining",
    "JSON_OUTPUT",
    "MARKDOWN_OUTPUT",
    "RESULT_ROOT",
    "RUN_DIRS",
    "SCHEMA",
    "TRAINING_LOCK",
    "VARIANT_OFF",
    "VARIANT_ON",
    "aggregate_and_write",
    "build_report",
    "evaluation_command",
    "execution_plan",
    "inspect_run_progress",
    "inspect_training_readiness",
    "load_all_rows",
    "main",
    "normalize_baseline_sweep",
    "normalize_ner_sweep",
    "parse_args",
    "prepare_baseline_reference_view",
    "render_markdown",
    "run_role_synchronized_evaluations",
    "sha256_file",
    "sweep_path",
    "verify_frozen_manifests",
    "write_report",
]


if __name__ == "__main__":
    main()
