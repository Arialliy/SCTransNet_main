#!/usr/bin/env python3
"""Independent post-training closure for the single-seed V2 NER candidate.

Only the two V2 relay-on checkpoint roles may be evaluated here.  The V1
relay-off control and the SCTransNet baseline are immutable reference sweeps:
they are rebound to their current artifacts and validated, never regenerated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_v2_pd_fa as v2_eval,
)
from experiments import (  # noqa: E402
    freeze_tpd_ner_v8_mprs_dch_v2_source_locks as v2_freeze,
)
from experiments import (  # noqa: E402
    postprocess_tpd_ner_v8_mprs_dch_formal800 as v1_post,
)
from experiments import train_tpd_ner_v8_mprs_dch_v2 as v2_train  # noqa: E402
from experiments import (  # noqa: E402
    train_tpd_ner_v8_mprs_dch_v2_exact as v2_exact,
)


SCHEMA = "sctransnet_tpd_ner_v8_mprs_dch_v2_posttraining_aggregate_v1"
READINESS_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v2_posttraining_readiness_v1"
)
COMPLETE_MARKER_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v2_postprocess_complete_v1"
)
REJECTED_SCHEMA = (
    "sctransnet_tpd_ner_v8_mprs_dch_v2_rejected_postprocess_v1"
)
DATASET = v2_train.DATASET
TRAINING_SEED = 42
SPLIT_SEED = 20260722
EXPECTED_EPOCHS = 800
TARGET_COUNT = 189
CHECKPOINTS = ("best.pth.tar", "best_miou.pth.tar")
CHECKPOINT_ROLES = dict(v2_eval.CHECKPOINT_ROLES)
ROLE_NAMES = {
    "best_validation_pd_primary": "pd_primary",
    "best_validation_miou_secondary": "miou_secondary",
}
FA_BUDGETS = tuple(v2_eval.FA_BUDGETS)
BUDGET_KEYS = tuple(v2_eval.BUDGET_KEYS)
VARIANT_V2_ON = v2_eval.VARIANT
VARIANT_V1_OFF = v2_eval.V1_CONTROL
BASELINE_VARIANT = "baseline_sctransnet"
V2_RESULT_ROOT = v2_exact.DEFAULT_OUTPUT_ROOT
V2_RUN_DIR = (
    V2_RESULT_ROOT
    / DATASET
    / VARIANT_V2_ON
    / f"seed_{TRAINING_SEED}_{v2_exact.FORMAL_RUN_TAG}"
)
V1_OFF_RUN_DIR = v1_post.RUN_DIRS[v1_post.VARIANT_OFF]
BASELINE_RUN_DIR = v1_post.BASELINE_VIEW_RUN
V2_EVALUATOR = (
    REPO_ROOT / "experiments/evaluate_tpd_ner_v8_mprs_dch_v2_pd_fa.py"
)
TRAINING_LOCK = v2_freeze.DEFAULT_TRAINING_LOCK
ACCEPTANCE_LOCK = v2_freeze.DEFAULT_ACCEPTANCE_LOCK
COMPARISON_DIR = V2_RESULT_ROOT / DATASET / "comparison"
JSON_OUTPUT = (
    COMPARISON_DIR / "tpd_ner_v8_mprs_dch_v2_formal800_comparison.json"
)
MARKDOWN_OUTPUT = (
    COMPARISON_DIR / "tpd_ner_v8_mprs_dch_v2_formal800_comparison.md"
)
COMPLETE_MARKER = COMPARISON_DIR / "POSTPROCESS_COMPLETE.json"
GPU_UUIDS = dict(v2_exact.PHYSICAL_GPU_UUIDS)


class IncompleteTraining(RuntimeError):
    """Raised when either required single-seed trajectory is incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _metrics_progress(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        return {
            "exists": False,
            "event_count": 0,
            "last_epoch": 0,
            "contiguous_from_one": False,
        }
    epochs: list[Any] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        epochs.append(value.get("epoch"))
    return {
        "exists": True,
        "event_count": len(epochs),
        "last_epoch": epochs[-1] if epochs else 0,
        "contiguous_from_one": epochs == list(range(1, len(epochs) + 1)),
    }


def inspect_v2_progress(run_dir: Path = V2_RUN_DIR) -> dict[str, Any]:
    directory = Path(run_dir)
    metrics = _metrics_progress(directory / "metrics.jsonl")
    summary_path = directory / "summary.json"
    summary: dict[str, Any] | None = None
    if summary_path.is_file() and not summary_path.is_symlink():
        summary = load_json(summary_path)
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
    if summary is not None:
        _require(
            summary.get("schema") == v2_exact.COMPLETION_SUMMARY_SCHEMA,
            "V2 completion summary schema differs",
        )
        for name, expected in {
            "variant": VARIANT_V2_ON,
            "seed": TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "status": "complete",
        }.items():
            if name == "status" and summary.get(name) != expected:
                continue
            _require(
                summary.get(name) == expected,
                f"V2 completion summary differs: {name}",
            )
    complete = bool(
        not missing
        and summary is not None
        and summary.get("status") == "complete"
        and metrics["event_count"] == EXPECTED_EPOCHS
        and metrics["last_epoch"] == EXPECTED_EPOCHS
        and metrics["contiguous_from_one"]
    )
    return {
        "variant": VARIANT_V2_ON,
        "run_dir": str(directory.resolve()),
        "metrics": metrics,
        "summary_exists": summary is not None,
        "summary_status": None if summary is None else summary.get("status"),
        "missing_required_artifacts": missing,
        "complete": complete,
    }


def inspect_training_readiness() -> dict[str, Any]:
    v2_on = inspect_v2_progress()
    v1_off = v1_post.inspect_run_progress(VARIANT_V1_OFF)
    ready = bool(v2_on["complete"] and v1_off["complete"])
    return {
        "schema": READINESS_SCHEMA,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "expected_epochs": EXPECTED_EPOCHS,
        "multi_seed_scheduled": False,
        "v2_on": v2_on,
        "v1_off_read_only_control": v1_off,
        "required_runs_complete": ready,
        "posttraining_action": "evaluate_v2_only" if ready else "wait",
    }


def verify_frozen_manifests() -> dict[str, Any]:
    training = v2_freeze.verify_training_lock(TRAINING_LOCK)
    acceptance = v2_freeze.verify_acceptance_lock(
        ACCEPTANCE_LOCK,
        TRAINING_LOCK,
    )
    v1 = v1_post.verify_frozen_manifests()
    return {
        "v2_training_source_lock": str(TRAINING_LOCK.resolve()),
        "v2_training_source_lock_sha256": sha256_file(TRAINING_LOCK),
        "v2_acceptance_source_lock": str(ACCEPTANCE_LOCK.resolve()),
        "v2_acceptance_source_lock_sha256": sha256_file(ACCEPTANCE_LOCK),
        "v2_training_data_sha256": training["training_data_sha256"],
        "v2_training_source_count": training["source_count"],
        "v2_acceptance_source_count": acceptance["source_count"],
        "v1_reference_manifests": dict(v1),
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


def _run_artifact_sha256(
    run_dir: Path,
    checkpoint: str,
    evaluator: Path,
) -> dict[str, str]:
    return {
        "protocol.json": sha256_file(run_dir / "protocol.json"),
        "split.json": sha256_file(run_dir / "split.json"),
        "summary.json": sha256_file(run_dir / "summary.json"),
        "metrics.jsonl": sha256_file(run_dir / "metrics.jsonl"),
        "checkpoint": sha256_file(run_dir / checkpoint),
        "evaluator": sha256_file(evaluator),
    }


def current_v2_binding(checkpoint: str) -> dict[str, Any]:
    role = _checkpoint_role(checkpoint)
    run_dir = V2_RUN_DIR.resolve()
    artifact_identity = v2_eval.validate_run_artifacts(run_dir, checkpoint)
    hashes = _run_artifact_sha256(run_dir, checkpoint, V2_EVALUATOR)
    _require(
        artifact_identity.get("variant") == VARIANT_V2_ON,
        "V2 current variant differs",
    )
    _require(
        artifact_identity.get("checkpoint_role") == role,
        "V2 current checkpoint role differs",
    )
    split = load_json(run_dir / "split.json")
    split_hashes = split.get("hashes")
    _require(isinstance(split_hashes, Mapping), "V2 split hashes are missing")
    validation_sha = split_hashes.get("used_val_sha256")
    _require(isinstance(validation_sha, str), "V2 validation SHA is missing")
    _require(
        artifact_identity.get("validation_split_sha256") == validation_sha,
        "V2 artifact validation split SHA differs",
    )
    return {
        "variant": VARIANT_V2_ON,
        "run_dir": run_dir,
        "checkpoint_path": (run_dir / checkpoint).resolve(),
        "checkpoint_name": checkpoint,
        "checkpoint_role": role,
        "checkpoint_sha256": hashes["checkpoint"],
        "validation_split_sha256": validation_sha,
        "evaluator_path": V2_EVALUATOR.resolve(),
        "evaluator_sha256": hashes["evaluator"],
        "artifact_sha256": hashes,
        "artifact_identity": artifact_identity,
    }


def current_reference_binding(
    variant: str,
    checkpoint: str,
) -> dict[str, Any]:
    """Build a current V1 binding without creating or replacing any file."""

    if variant == VARIANT_V1_OFF:
        return v1_post.current_sweep_binding(
            variant=VARIANT_V1_OFF,
            checkpoint=checkpoint,
        )
    if variant != BASELINE_VARIANT:
        raise ValueError(f"unexpected reference variant: {variant}")
    role = _checkpoint_role(checkpoint)
    run_dir = BASELINE_RUN_DIR.resolve()
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise FileNotFoundError(
            "existing baseline reference view is unavailable; V2 does not "
            f"create it: {run_dir}"
        )
    split = load_json(run_dir / "split.json")
    split_hashes = split.get("hashes")
    _require(
        isinstance(split_hashes, Mapping),
        "baseline split hashes are missing",
    )
    validation_sha = split_hashes.get("used_val_sha256")
    _require(
        isinstance(validation_sha, str),
        "baseline validation split SHA is missing",
    )
    evaluator = v1_post.BASELINE_EVALUATOR.resolve()
    hashes = _run_artifact_sha256(run_dir, checkpoint, evaluator)
    return {
        "variant": BASELINE_VARIANT,
        "run_dir": run_dir,
        "checkpoint_path": (run_dir / checkpoint).resolve(),
        "checkpoint_name": checkpoint,
        "checkpoint_role": role,
        "checkpoint_sha256": hashes["checkpoint"],
        "validation_split_sha256": validation_sha,
        "evaluator_path": evaluator,
        "evaluator_sha256": hashes["evaluator"],
        "artifact_sha256": hashes,
        "artifact_identity": None,
    }


def _reference_paths() -> tuple[Path, ...]:
    paths: set[Path] = set()
    for run_dir in (V1_OFF_RUN_DIR, BASELINE_RUN_DIR):
        paths.update(
            run_dir / name
            for name in (
                "protocol.json",
                "split.json",
                "summary.json",
                "metrics.jsonl",
                *CHECKPOINTS,
            )
        )
        paths.update(sweep_path(run_dir, name) for name in CHECKPOINTS)
    return tuple(sorted(path.resolve() for path in paths))


def reference_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for path in _reference_paths():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        stat = path.stat()
        snapshot[str(path)] = {
            "sha256": sha256_file(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return snapshot


def _normalize_v2_sweep(
    payload: Mapping[str, Any],
    *,
    checkpoint: str,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = binding.get("artifact_identity")
    _require(isinstance(artifact, Mapping), "V2 artifact identity is missing")
    _require(
        payload.get("validation_split_sha256")
        == binding.get("validation_split_sha256"),
        "V2 sweep validation split SHA differs",
    )
    v2_eval.validate_output_identity(payload, artifact_audit=artifact)
    fixed = v2_eval._normalize_fixed(payload.get("fixed_threshold_0_5"))
    raw_budgets = v2_eval._normalize_budgets(payload)
    budgets = {
        key: {
            "matched_target_count": point["matched_target_count"],
            "target_count": point["target_count"],
            "pd": point["pd"],
            "fa": point["achieved_fa"],
            "threshold": point["threshold"],
        }
        for key, point in raw_budgets.items()
    }
    gate = payload.get("performance_gate_assessment")
    _require(isinstance(gate, Mapping), "V2 absolute gate is missing")
    _require(
        gate.get("absolute_checkpoint_gate_passed")
        is v2_eval._absolute_gate(
            _checkpoint_role(checkpoint),
            fixed,
            raw_budgets,
        )["absolute_checkpoint_gate_passed"],
        "V2 absolute gate differs",
    )
    return {
        "source": "new_v2_candidate",
        "variant": VARIANT_V2_ON,
        "seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "checkpoint": checkpoint,
        "checkpoint_role": _checkpoint_role(checkpoint),
        "checkpoint_epoch": payload.get("checkpoint_epoch"),
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "run_directory": str(binding["run_dir"]),
        "run_identity": dict(artifact["run_identity"]),
        "source_checkpoint_identity": dict(
            artifact["checkpoint_identity"]
        ),
        "fixed_threshold_0_5": fixed,
        "pd_at_fa_budget": budgets,
        "absolute_gate": dict(gate),
    }


def validate_v2_sweep(
    path: Path,
    *,
    checkpoint: str,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    current = (
        current_v2_binding(checkpoint)
        if binding is None
        else dict(binding)
    )
    return _normalize_v2_sweep(
        load_json(path),
        checkpoint=checkpoint,
        binding=current,
    )


def load_reference_rows() -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for checkpoint in CHECKPOINTS:
        for variant, run_dir in (
            (VARIANT_V1_OFF, V1_OFF_RUN_DIR),
            (BASELINE_VARIANT, BASELINE_RUN_DIR),
        ):
            binding = current_reference_binding(variant, checkpoint)
            rows[(variant, checkpoint)] = v1_post.validate_existing_sweep(
                sweep_path(run_dir, checkpoint),
                variant=variant,
                checkpoint=checkpoint,
                binding=binding,
            )
    return rows


def load_all_rows() -> dict[tuple[str, str], dict[str, Any]]:
    rows = load_reference_rows()
    for checkpoint in CHECKPOINTS:
        binding = current_v2_binding(checkpoint)
        rows[(VARIANT_V2_ON, checkpoint)] = validate_v2_sweep(
            sweep_path(V2_RUN_DIR, checkpoint),
            checkpoint=checkpoint,
            binding=binding,
        )
    return rows


def evaluation_command(
    *,
    checkpoint: str,
    python: Path | str = sys.executable,
    device_mode: str = "gpu23",
    physical_gpu: int = 2,
) -> tuple[list[str], dict[str, str], Path]:
    _checkpoint_role(checkpoint)
    if device_mode not in ("gpu23", "cpu"):
        raise ValueError(f"unexpected device mode: {device_mode}")
    gpu_key = str(physical_gpu)
    if gpu_key not in GPU_UUIDS:
        raise ValueError("V2 evaluation physical GPU must be 2 or 3")
    command = [
        str(python),
        str(V2_EVALUATOR.resolve()),
        "--run-dir",
        str(V2_RUN_DIR.resolve()),
        "--checkpoint",
        checkpoint,
        "--device",
        "cuda:0" if device_mode == "gpu23" else "cpu",
        "--expected-epochs",
        str(EXPECTED_EPOCHS),
    ]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(TRAINING_SEED)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if device_mode == "gpu23":
        environment["CUDA_VISIBLE_DEVICES"] = GPU_UUIDS[gpu_key]
    else:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    return command, environment, sweep_path(V2_RUN_DIR, checkpoint)


def _new_rejected_directory(parent: Path) -> Path:
    root = parent / "rejected_postprocess"
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError(f"rejected-postprocess directory is a symlink: {root}")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for index in range(1000):
        candidate = root / (stamp if index == 0 else f"{stamp}.{index:03d}")
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
    descriptor = os.open(
        destination / "reason.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    content = _canonical_bytes(
        {
            "schema": REJECTED_SCHEMA,
            "reason": reason,
            "moved": moved,
        }
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


def _run_v2_evaluation(
    *,
    checkpoint: str,
    python: Path | str,
    device_mode: str,
    physical_gpu: int,
) -> dict[str, Any]:
    command, environment, output = evaluation_command(
        checkpoint=checkpoint,
        python=python,
        device_mode=device_mode,
        physical_gpu=physical_gpu,
    )
    binding = current_v2_binding(checkpoint)
    rejected: list[dict[str, Any]] = []
    if output.exists() or output.is_symlink():
        try:
            row = validate_v2_sweep(
                output,
                checkpoint=checkpoint,
                binding=binding,
            )
        except Exception as exc:
            rejected.append(
                quarantine_postprocess_artifacts(
                    [output],
                    parent=output.parent,
                    reason=(
                        "existing V2 sweep failed current binding: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
        else:
            return {
                "variant": VARIANT_V2_ON,
                "checkpoint": checkpoint,
                "status": "reused_valid_existing",
                "output": str(output),
                "row": row,
                "rejected_previous_outputs": rejected,
            }
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
                        "V2 evaluator exited before a valid result was "
                        f"accepted: {type(exc).__name__}: {exc}"
                    ),
                )
            )
        raise
    post_binding = current_v2_binding(checkpoint)
    try:
        row = validate_v2_sweep(
            output,
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
                        "new V2 sweep failed current binding: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
        raise
    return {
        "variant": VARIANT_V2_ON,
        "checkpoint": checkpoint,
        "status": "completed",
        "output": str(output),
        "row": row,
        "rejected_previous_outputs": rejected,
    }


def run_v2_evaluations(
    *,
    python: Path | str = sys.executable,
    device_mode: str = "gpu23",
    physical_gpu: int = 2,
) -> list[dict[str, Any]]:
    readiness = inspect_training_readiness()
    if readiness["required_runs_complete"] is not True:
        raise IncompleteTraining(
            "V2-on and the reused V1-off control must each have a complete "
            "summary and contiguous 1..800 metrics"
        )
    before = reference_snapshot()
    load_reference_rows()
    _require(
        reference_snapshot() == before,
        "reference files changed during read-only preflight",
    )
    return [
        _run_v2_evaluation(
            checkpoint=checkpoint,
            python=python,
            device_mode=device_mode,
            physical_gpu=physical_gpu,
        )
        for checkpoint in CHECKPOINTS
    ]


def relay_off_identity_contract() -> dict[str, Any]:
    """Execute the non-trainable V2-off probe against the V1-off builder."""

    v1_model, _ = v2_train.build_v1_relay_off_reference(TRAINING_SEED)
    probe_model, probe_metadata = (
        v2_train.build_v2_relay_off_identity_probe(TRAINING_SEED)
    )
    v2_on_model, _ = v2_train.build_tpd_ner_v8_mprs_dch_v2_model(
        VARIANT_V2_ON,
        TRAINING_SEED,
    )
    try:
        v1_state = v1_model.state_dict()
        probe_state = probe_model.state_dict()
        on_state = v2_on_model.state_dict()
        v1_keys = tuple(v1_state)
        probe_keys = tuple(probe_state)
        relay_keys = tuple(
            name for name in on_state if name.startswith("tpd_ner.")
        )
        common_keys = tuple(name for name in on_state if name not in relay_keys)
        type_identical = type(v1_model) is type(probe_model)
        keys_identical = v1_keys == probe_keys
        tensors_identical = bool(
            keys_identical
            and all(v1_state[name].equal(probe_state[name]) for name in v1_keys)
        )
        on_common_identical = bool(
            common_keys == v1_keys
            and all(on_state[name].equal(v1_state[name]) for name in v1_keys)
        )
        passed = bool(
            type_identical
            and keys_identical
            and tensors_identical
            and on_common_identical
            and len(relay_keys) == 16
            and probe_metadata.get("variant") == VARIANT_V1_OFF
            and probe_metadata.get("formal_training_scheduled") is False
            and VARIANT_V1_OFF
            not in v2_train.SUPPORTED_TPD_NER_V8_MPRS_DCH_V2_VARIANTS
        )
        _require(passed, "V2 relay-off identity probe differs from V1 off")
        return {
            "required_control": VARIANT_V1_OFF,
            "relay_off_retrained": False,
            "training_seed": TRAINING_SEED,
            "same_model_class": type_identical,
            "same_ordered_state_keys": keys_identical,
            "all_off_state_tensors_identical": tensors_identical,
            "v2_on_common_state_matches_v1_off_initial_state": (
                on_common_identical
            ),
            "v2_on_added_relay_state_key_count": len(relay_keys),
            "v2_off_present_in_formal_variant_matrix": False,
            "passed": passed,
        }
    finally:
        del v1_model, probe_model, v2_on_model
        gc.collect()


def same_split_and_training_contract() -> dict[str, Any]:
    """Prove the V2-on/V1-off/baseline comparison axes."""

    run_dirs = {
        VARIANT_V2_ON: V2_RUN_DIR,
        VARIANT_V1_OFF: V1_OFF_RUN_DIR,
        BASELINE_VARIANT: BASELINE_RUN_DIR,
    }
    protocols = {
        name: load_json(path / "protocol.json")
        for name, path in run_dirs.items()
    }
    splits = {
        name: load_json(path / "split.json")
        for name, path in run_dirs.items()
    }
    arguments: dict[str, Mapping[str, Any]] = {}
    for name, protocol in protocols.items():
        value = protocol.get("arguments")
        _require(isinstance(value, Mapping), f"{name} arguments are missing")
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
                f"{source} training axis differs: {name}",
            )
    reference_split = splits[VARIANT_V1_OFF]
    for source, split in splits.items():
        for name, expected in {
            "dataset": DATASET,
            "split_seed": SPLIT_SEED,
            "used_train_count": 530,
            "used_val_count": 133,
            "full_official_train_count": 663,
            "official_test_accessed": False,
        }.items():
            _require(
                split.get(name) == expected,
                f"{source} split differs: {name}",
            )
        for name in ("used_train_ids", "used_val_ids", "hashes"):
            _require(
                split.get(name) == reference_split.get(name),
                f"{source} ordered split differs: {name}",
            )
    for field in (
        "normalization",
        "optimizer",
        "loss",
        "primary_selection_rule",
        "secondary_selection_rule",
    ):
        expected = protocols[VARIANT_V1_OFF].get(field)
        _require(expected is not None, f"V1 off protocol lacks {field}")
        for source, protocol in protocols.items():
            _require(
                protocol.get(field) == expected,
                f"{source} protocol differs: {field}",
            )
    _require(
        arguments[VARIANT_V2_ON].get("parent_variant")
        == arguments[VARIANT_V1_OFF].get("parent_variant")
        == "tpd_clean_v8_mprs_dch_full",
        "V2-on and V1-off parent variants differ",
    )
    _require(
        arguments[VARIANT_V2_ON].get("relay_enabled") is True
        and arguments[VARIANT_V1_OFF].get("relay_enabled") is False,
        "relay role identities differ",
    )
    _require(
        protocols[VARIANT_V2_ON].get("selection_source")
        == protocols[VARIANT_V1_OFF].get("selection_source")
        == "internal_validation_only",
        "V2-on/V1-off selection source differs",
    )
    _require(
        protocols[VARIANT_V2_ON].get("checkpoint_policy")
        == protocols[VARIANT_V1_OFF].get("checkpoint_policy"),
        "V2-on/V1-off checkpoint policies differ",
    )
    return {
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "multi_seed_scheduled": False,
        "same_fixed_training_axes": True,
        "same_parent_variant": True,
        "same_normalization": True,
        "same_optimizer": True,
        "same_loss": True,
        "same_selection_rules": True,
        "same_checkpoint_roles": True,
        "same_ordered_train_ids": True,
        "same_ordered_validation_ids": True,
        "same_split_hashes": True,
        "official_test_accessed": False,
        "fixed_training_axes": fixed_axes,
        "relay_off_builder_identity": relay_off_identity_contract(),
    }


def _paired_gate(
    off: Mapping[str, Any],
    on: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        off.get("checkpoint_role") == on.get("checkpoint_role"),
        "paired rows have different checkpoint roles",
    )
    contract = v2_freeze.performance_gate_contract()[
        "paired_v2_on_vs_v1_off_each_checkpoint_role"
    ]
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
            "v1_off_matched_target_count": off_count,
            "v2_on_matched_target_count": on_count,
            "v1_off_pd": off_point["pd"],
            "v2_on_pd": on_point["pd"],
            "v2_on_non_inferior": no_worse,
            "v2_on_strictly_better": better,
        }
    passed = bool(
        non_inferior >= contract["minimum_non_inferior_budget_count"]
        and strictly_better
        >= contract["minimum_strictly_better_budget_count"]
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


def _compare_to_baseline(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        candidate["checkpoint_role"] == baseline["checkpoint_role"],
        "candidate and baseline roles differ",
    )
    fixed_candidate = candidate["fixed_threshold_0_5"]
    fixed_baseline = baseline["fixed_threshold_0_5"]
    budgets = {}
    for key in BUDGET_KEYS:
        candidate_point = candidate["pd_at_fa_budget"][key]
        baseline_point = baseline["pd_at_fa_budget"][key]
        budgets[key] = {
            "delta_matched_targets": (
                int(candidate_point["matched_target_count"])
                - int(baseline_point["matched_target_count"])
            ),
            "delta_pd": (
                float(candidate_point["pd"]) - float(baseline_point["pd"])
            ),
            "candidate_achieved_fa": candidate_point["fa"],
            "baseline_achieved_fa": baseline_point["fa"],
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
                float(fixed_candidate["miou"])
                - float(fixed_baseline["miou"])
            ),
            "false_objects_per_image": (
                float(fixed_candidate["false_objects_per_image"])
                - float(fixed_baseline["false_objects_per_image"])
            ),
        },
        "pd_at_fa_budget": budgets,
    }


def build_report(
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    lock_bindings: Mapping[str, Any],
    comparison_contract: Mapping[str, Any],
    reference_before: Mapping[str, Any],
    reference_after: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        (variant, checkpoint)
        for variant in (
            BASELINE_VARIANT,
            VARIANT_V1_OFF,
            VARIANT_V2_ON,
        )
        for checkpoint in CHECKPOINTS
    }
    _require(set(rows) == expected, "V2 aggregate row matrix differs")
    _require(
        dict(reference_before) == dict(reference_after),
        "V1 reference artifacts changed during aggregation",
    )
    candidate_absolute: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []
    ordered_rows: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINTS:
        role = _checkpoint_role(checkpoint)
        role_name = ROLE_NAMES[role]
        baseline = rows[(BASELINE_VARIANT, checkpoint)]
        off = rows[(VARIANT_V1_OFF, checkpoint)]
        on = rows[(VARIANT_V2_ON, checkpoint)]
        gate = on.get("absolute_gate")
        _require(isinstance(gate, Mapping), "V2 candidate gate is missing")
        candidate_absolute[role_name] = dict(gate)
        paired[role_name] = _paired_gate(off, on)
        comparisons.append(_compare_to_baseline(on, baseline))
        ordered_rows.extend([dict(baseline), dict(off), dict(on)])
    pd_absolute = bool(
        candidate_absolute["pd_primary"][
            "absolute_checkpoint_gate_passed"
        ]
    )
    miou_absolute = bool(
        candidate_absolute["miou_secondary"][
            "absolute_checkpoint_gate_passed"
        ]
    )
    pd_paired = bool(paired["pd_primary"]["passed"])
    miou_paired = bool(paired["miou_secondary"]["passed"])
    aggregate_passed = bool(
        pd_absolute and miou_absolute and pd_paired and miou_paired
    )
    sweep_bindings = {}
    for variant, checkpoint in sorted(expected):
        run_dir = {
            BASELINE_VARIANT: BASELINE_RUN_DIR,
            VARIANT_V1_OFF: V1_OFF_RUN_DIR,
            VARIANT_V2_ON: V2_RUN_DIR,
        }[variant]
        path = sweep_path(run_dir, checkpoint)
        sweep_bindings[f"{variant}:{checkpoint}"] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
        }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "dataset": DATASET,
        "training_seed": TRAINING_SEED,
        "split_seed": SPLIT_SEED,
        "multi_seed_scheduled": False,
        "official_test_accessed": False,
        "scope": "single_seed_internal_validation",
        "rows": ordered_rows,
        "v2_candidate_absolute_gate_by_role": candidate_absolute,
        "paired_v2_on_vs_v1_off_gate_by_role": paired,
        "success_components": {
            "v2_on_pd_primary_absolute": pd_absolute,
            "v2_on_miou_secondary_absolute": miou_absolute,
            "pd_primary_paired_v2_on_vs_v1_off": pd_paired,
            "miou_secondary_paired_v2_on_vs_v1_off": miou_paired,
            "v1_off_absolute_gate_required": False,
            "baseline_affects_decision": False,
        },
        "aggregate_full_model_gate_passed": aggregate_passed,
        "decision": (
            "FULL_MODEL_GATE_PASSED"
            if aggregate_passed
            else "RETURN_TO_MODEL_OPTIMIZATION"
        ),
        "comparisons_vs_baseline": comparisons,
        "comparison_contract": dict(comparison_contract),
        "v1_reference_read_only": {
            "before": dict(reference_before),
            "after": dict(reference_after),
            "unchanged": True,
        },
        "bindings": {
            **dict(lock_bindings),
            "v2_evaluator": str(V2_EVALUATOR.resolve()),
            "v2_evaluator_sha256": sha256_file(V2_EVALUATOR),
            "v1_off_evaluator": str(v1_post.NER_EVALUATOR.resolve()),
            "v1_off_evaluator_sha256": sha256_file(v1_post.NER_EVALUATOR),
            "baseline_evaluator": str(
                v1_post.BASELINE_EVALUATOR.resolve()
            ),
            "baseline_evaluator_sha256": sha256_file(
                v1_post.BASELINE_EVALUATOR
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
        "# V8-MPRS-DCH + five-node NER V2 formal800 comparison",
        "",
        f"- Decision: `{report['decision']}`",
        (
            "- Aggregate full-model gate passed: "
            f"`{str(report['aggregate_full_model_gate_passed']).lower()}`"
        ),
        "- Scope: fixed seed 42, NUDT-SIRST internal 530/133 validation",
        "- V1 relay-off and SCTransNet reference files changed: `false`",
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
    for role, gate in report[
        "v2_candidate_absolute_gate_by_role"
    ].items():
        lines.append(
            f"- V2-on `{role}` absolute gate: "
            f"`{str(gate['absolute_checkpoint_gate_passed']).lower()}`"
        )
    for role, gate in report[
        "paired_v2_on_vs_v1_off_gate_by_role"
    ].items():
        lines.append(
            f"- `{role}` paired gate: `{str(gate['passed']).lower()}` "
            f"({gate['non_inferior_budget_count']}/5 non-inferior, "
            f"{gate['strictly_better_budget_count']}/5 strictly better)"
        )
    lines.extend(
        [
            "",
            "V1-off is a paired control only; its own V2 absolute gate is not "
            "part of the decision. The baseline reports deltas only.",
            "",
            "## Conclusion boundary",
            "",
            "- This report covers only the fixed seed-42 internal-validation "
            "model decision.",
            "- It does not claim cross-seed stability, cross-dataset transfer, "
            "or official-test performance.",
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
    return _canonical_bytes(
        {
            "schema": COMPLETE_MARKER_SCHEMA,
            "status": "complete",
            "decision": report["decision"],
            "aggregate_full_model_gate_passed": report[
                "aggregate_full_model_gate_passed"
            ],
            "outputs": {
                JSON_OUTPUT.name: hashlib.sha256(json_bytes).hexdigest(),
                MARKDOWN_OUTPUT.name: hashlib.sha256(
                    markdown_bytes
                ).hexdigest(),
            },
        }
    )


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
    conflicts = [
        path
        for path, content in expected.items()
        if (path.exists() or path.is_symlink())
        and (
            not path.is_file()
            or path.is_symlink()
            or path.read_bytes() != content
        )
    ]
    if conflicts:
        quarantine_postprocess_artifacts(
            [JSON_OUTPUT, MARKDOWN_OUTPUT, COMPLETE_MARKER],
            parent=COMPARISON_DIR,
            reason="existing V2 report differs from the rebound aggregate",
        )
    for path, content in expected.items():
        if not path.exists() and not path.is_symlink():
            _atomic_publish_new(path, content)
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.read_bytes() == content,
            f"V2 report publication differs: {path}",
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
                reason="existing V2 completion marker differs",
            )
    if not COMPLETE_MARKER.exists() and not COMPLETE_MARKER.is_symlink():
        _atomic_publish_new(COMPLETE_MARKER, marker_bytes)
    _require(
        COMPLETE_MARKER.is_file()
        and not COMPLETE_MARKER.is_symlink()
        and COMPLETE_MARKER.read_bytes() == marker_bytes,
        "V2 completion marker differs",
    )
    return JSON_OUTPUT, MARKDOWN_OUTPUT, COMPLETE_MARKER


def aggregate_and_write() -> tuple[dict[str, Any], tuple[Path, Path, Path]]:
    readiness = inspect_training_readiness()
    if readiness["required_runs_complete"] is not True:
        raise IncompleteTraining(
            "aggregate requires complete V2-on and V1-off seed-42 runs with "
            "contiguous 1..800 metrics"
        )
    locks = verify_frozen_manifests()
    before = reference_snapshot()
    comparison_contract = same_split_and_training_contract()
    rows = load_all_rows()
    after = reference_snapshot()
    report = build_report(
        rows,
        lock_bindings=locks,
        comparison_contract=comparison_contract,
        reference_before=before,
        reference_after=after,
    )
    report["readiness_binding"] = readiness
    return report, write_report(report)


def execution_plan(
    *,
    python: Path | str = sys.executable,
    device_mode: str = "gpu23",
    physical_gpu: int = 2,
) -> dict[str, Any]:
    evaluations = []
    for checkpoint in CHECKPOINTS:
        command, _, output = evaluation_command(
            checkpoint=checkpoint,
            python=python,
            device_mode=device_mode,
            physical_gpu=physical_gpu,
        )
        evaluations.append(
            {
                "variant": VARIANT_V2_ON,
                "checkpoint": checkpoint,
                "physical_gpu_index": (
                    physical_gpu if device_mode == "gpu23" else None
                ),
                "command": command,
                "output": str(output),
            }
        )
    return {
        "readiness": inspect_training_readiness(),
        "new_evaluation_count": len(evaluations),
        "new_evaluations": evaluations,
        "v1_off_evaluations": 0,
        "baseline_evaluations": 0,
        "reference_mode": "read_only_reuse_and_revalidation",
        "aggregate_outputs": [
            str(JSON_OUTPUT),
            str(MARKDOWN_OUTPUT),
            str(COMPLETE_MARKER),
        ],
        "training_seed": TRAINING_SEED,
        "multi_seed_scheduled": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Postprocess the single-seed V2 five-node NER candidate"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run-now", action="store_true")
    mode.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--device-mode",
        choices=("gpu23", "cpu"),
        default="gpu23",
    )
    parser.add_argument(
        "--physical-gpu",
        type=int,
        choices=(2, 3),
        default=2,
    )
    return parser.parse_args(argv)


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
                physical_gpu=args.physical_gpu,
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
    run_v2_evaluations(
        python=args.python,
        device_mode=args.device_mode,
        physical_gpu=args.physical_gpu,
    )
    report, paths = aggregate_and_write()
    print(
        f"COMPLETE decision={report['decision']} "
        f"json={paths[0]} markdown={paths[1]} marker={paths[2]}",
        flush=True,
    )


__all__ = [
    "ACCEPTANCE_LOCK",
    "BASELINE_RUN_DIR",
    "BUDGET_KEYS",
    "CHECKPOINTS",
    "COMPARISON_DIR",
    "COMPLETE_MARKER",
    "JSON_OUTPUT",
    "MARKDOWN_OUTPUT",
    "TRAINING_LOCK",
    "TRAINING_SEED",
    "V1_OFF_RUN_DIR",
    "V2_RUN_DIR",
    "aggregate_and_write",
    "build_report",
    "current_reference_binding",
    "current_v2_binding",
    "evaluation_command",
    "execution_plan",
    "inspect_training_readiness",
    "load_all_rows",
    "relay_off_identity_contract",
    "run_v2_evaluations",
    "same_split_and_training_contract",
    "validate_v2_sweep",
    "write_report",
]


if __name__ == "__main__":
    main()
