#!/usr/bin/env python3
"""Complete the frozen three-dataset seed-42 post-training workflow.

This module is an additive orchestration layer.  It does not change the
frozen evaluator, selector, model, training protocol, threshold semantics, or
checkpoint-selection rules.  It verifies the completed 12-run launch plan,
evaluates exactly 24 selected checkpoints on physical GPUs 2 and 3, assembles
the selector input from the fixed-threshold points, executes the frozen global
TSS selector, and writes a machine-auditable result bundle.

The workflow is restartable.  An existing evaluator output is reused only
after its dataset/method/lambda/role, checkpoint SHA, protocol identity,
fixed-threshold contract, and evaluator-source hashes all pass validation.
Invalid or conflicting write-once artifacts are rejected rather than
overwritten.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_three_dataset_v2 as evaluator  # noqa: E402
from experiments import select_three_dataset_global_tss_recipe_v2 as selector  # noqa: E402
from experiments import three_dataset_v2_protocol as data_protocol  # noqa: E402


SCHEMA_PREFIX = "sctransnet_three_dataset_seed42_posttraining_v2"
LAUNCH_PLAN_SCHEMA = "sctransnet_three_dataset_seed42_global_tss_launcher_v2/v1"
RESULTS_ROOT = REPO_ROOT / "results" / "three_dataset_seed42_global_tss_v2"
DATASET_ROOT = REPO_ROOT / "datasets"
PROTOCOL_MANIFEST = (
    REPO_ROOT
    / "results"
    / "three_dataset_v2"
    / "manifests"
    / "three_dataset_v2_protocol.json"
)
LAUNCH_PLAN = RESULTS_ROOT / "launch" / "formal" / "launch_plan.json"
PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")

GPU_BINDINGS = {
    "best_miou": {
        "physical_index": 2,
        "uuid": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    },
    "best_pd": {
        "physical_index": 3,
        "uuid": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
    },
}


class PosttrainingError(RuntimeError):
    """Raised when a formal post-training contract is violated."""


@dataclass(frozen=True)
class EvaluationTask:
    dataset: str
    method: str
    requested_tss_weight: float | None
    checkpoint_role: str
    run_dir: Path
    output_path: Path

    @property
    def lambda_key(self) -> str | None:
        if self.requested_tss_weight is None:
            return None
        return format(self.requested_tss_weight, ".4g")

    @property
    def key(self) -> str:
        recipe = (
            "original"
            if self.method == "original"
            else f"final_lambda_{str(self.requested_tss_weight).replace('.', 'p')}"
        )
        return f"{self.dataset}__{recipe}__{self.checkpoint_role}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PosttrainingError(message)


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PosttrainingError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PosttrainingError(f"non-finite JSON constant: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise PosttrainingError(f"JSON artifact is missing or linked: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PosttrainingError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PosttrainingError(f"JSON artifact must contain one object: {path}")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise PosttrainingError(f"file is missing or linked: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(_canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _write_once_or_identical(path: Path, value: Mapping[str, Any]) -> str:
    expected = _canonical_bytes(value)
    path = Path(path)
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise PosttrainingError(f"write-once path is not a regular file: {path}")
        observed = path.read_bytes()
        if observed != expected:
            raise PosttrainingError(f"write-once artifact conflicts: {path}")
        return "reused_identical"
    _atomic_json(path, value)
    return "written"


def _artifact(path: Path) -> dict[str, Any]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"invalid artifact: {path}")
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _archive_existing_status(status_path: Path, history_root: Path) -> Path | None:
    status_path = Path(status_path)
    if not status_path.exists():
        return None
    _require(
        status_path.is_file() and not status_path.is_symlink(),
        f"existing status is not a regular file: {status_path}",
    )
    digest = _file_sha256(status_path)
    history_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    destination = history_root / f"status.{stamp}.{digest[:12]}.json"
    suffix = 2
    while destination.exists():
        destination = history_root / (
            f"status.{stamp}.{digest[:12]}.attempt_{suffix}.json"
        )
        suffix += 1
    status_path.replace(destination)
    return destination


def _next_log_path(log_root: Path, task_key: str) -> Path:
    base = Path(log_root) / f"{task_key}.log"
    if not base.exists():
        return base
    _require(base.is_file() and not base.is_symlink(), f"invalid prior log: {base}")
    attempt = 2
    while True:
        candidate = Path(log_root) / f"{task_key}.attempt_{attempt}.log"
        if not candidate.exists():
            return candidate
        _require(
            candidate.is_file() and not candidate.is_symlink(),
            f"invalid prior log: {candidate}",
        )
        attempt += 1


def _acquire_posttraining_lock(
    post_root: Path, status_path: Path
) -> Any:
    lock_handle = (Path(post_root) / "posttraining.lock").open(
        "a+", encoding="utf-8"
    )
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise PosttrainingError(
            "another post-training orchestrator is active"
        ) from exc
    try:
        _archive_existing_status(
            status_path, Path(post_root) / "status_history"
        )
    except Exception:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
        raise
    return lock_handle


def _weight_key(weight: float) -> str:
    for candidate in evaluator.TSS_CANDIDATES:
        if float(weight) == float(candidate):
            return format(candidate, ".4g")
    raise PosttrainingError(f"unexpected TSS weight: {weight!r}")


def _expected_run_dir(results_root: Path, task: EvaluationTask) -> Path:
    base = Path(results_root).resolve() / "runs" / task.dataset
    if task.method == "original":
        return base / "original" / "seed_42"
    token = _weight_key(float(task.requested_tss_weight)).replace(".", "p")
    return base / "final" / f"lambda_{token}" / "seed_42"


def _expected_worker_keys() -> set[tuple[str, str, float | None]]:
    expected: set[tuple[str, str, float | None]] = set()
    for dataset in data_protocol.DATASETS:
        expected.add((dataset, "original", None))
        for weight in evaluator.TSS_CANDIDATES:
            expected.add((dataset, "final", float(weight)))
    return expected


def _verify_static_source(
    plan: Mapping[str, Any], key: str, expected_path: Path
) -> None:
    static = plan.get("static_inputs")
    _require(isinstance(static, Mapping), "launch plan lacks static_inputs")
    sources = static.get("training_sources")
    _require(isinstance(sources, Mapping), "launch plan lacks training_sources")
    record = sources.get(key)
    _require(isinstance(record, Mapping), f"launch plan lacks source {key}")
    recorded_path = Path(str(record.get("path", ""))).resolve()
    expected_path = expected_path.resolve(strict=True)
    _require(recorded_path == expected_path, f"launch-plan {key} path differs")
    _require(
        record.get("sha256") == _file_sha256(expected_path),
        f"launch-plan {key} SHA differs from current frozen source",
    )


def load_completed_plan(plan_path: Path = LAUNCH_PLAN) -> dict[str, Any]:
    plan = _load_json(plan_path)
    _require(plan.get("schema") == LAUNCH_PLAN_SCHEMA, "launch-plan schema differs")
    _require(
        plan.get("status") == "all_workers_complete",
        "launch plan has not reached all_workers_complete",
    )
    _require(plan.get("worker_count") == 12, "launch-plan worker_count is not 12")
    _require(plan.get("wave_count") == 6, "launch-plan wave_count is not 6")
    _require(
        plan.get("dataset_order") == list(data_protocol.DATASETS),
        "launch-plan dataset order differs",
    )
    _require(plan.get("original_run_count") == 3, "Original run budget differs")
    _require(plan.get("final_run_count") == 9, "Final run budget differs")
    _require(plan.get("total_search_budget_equal") is False, "budget disclosure differs")
    _verify_static_source(plan, "posttraining_evaluator", Path(evaluator.__file__))
    _verify_static_source(plan, "global_recipe_selector", Path(selector.__file__))
    _verify_static_source(plan, "data_protocol", Path(data_protocol.__file__))
    workers = plan.get("workers")
    _require(isinstance(workers, list) and len(workers) == 12, "launch plan lacks 12 workers")
    observed: set[tuple[str, str, float | None]] = set()
    for worker in workers:
        _require(isinstance(worker, Mapping), "launch-plan worker is not an object")
        dataset = worker.get("dataset")
        method = worker.get("method")
        raw_weight = worker.get("requested_tss_weight")
        weight = None if method == "original" else float(raw_weight)
        identity = (str(dataset), str(method), weight)
        _require(identity not in observed, f"duplicate launch-plan worker: {identity}")
        observed.add(identity)
        _require(worker.get("seed") == 42, f"{identity} seed differs")
        _require(worker.get("epochs") == 1000, f"{identity} epochs differ")
        _require(worker.get("eval_every") == 10, f"{identity} eval_every differs")
        _require(worker.get("threshold") == 0.5, f"{identity} threshold differs")
        _require(
            worker.get("checkpoint_roles") == list(evaluator.CHECKPOINT_ROLES),
            f"{identity} checkpoint roles differ",
        )
        if method == "original":
            _require(float(raw_weight) == 0.0, f"{identity} Original weight differs")
        else:
            _require(weight in evaluator.TSS_CANDIDATES, f"{identity} Final weight differs")
        run_dir = Path(str(worker.get("run_directory", "")))
        _require(run_dir.is_dir() and not run_dir.is_symlink(), f"run dir missing: {run_dir}")
    _require(observed == _expected_worker_keys(), "launch-plan worker matrix differs")
    return plan


def build_tasks(plan: Mapping[str, Any]) -> tuple[EvaluationTask, ...]:
    worker_by_identity: dict[tuple[str, str, float | None], Mapping[str, Any]] = {}
    for worker in plan["workers"]:
        method = str(worker["method"])
        weight = None if method == "original" else float(worker["requested_tss_weight"])
        worker_by_identity[(str(worker["dataset"]), method, weight)] = worker
    tasks: list[EvaluationTask] = []
    for dataset in data_protocol.DATASETS:
        recipes: list[tuple[str, float | None]] = [("original", None)] + [
            ("final", float(weight)) for weight in evaluator.TSS_CANDIDATES
        ]
        for method, weight in recipes:
            worker = worker_by_identity[(dataset, method, weight)]
            run_dir = Path(str(worker["run_directory"])).resolve(strict=True)
            for role in evaluator.CHECKPOINT_ROLES:
                tasks.append(
                    EvaluationTask(
                        dataset=dataset,
                        method=method,
                        requested_tss_weight=weight,
                        checkpoint_role=role,
                        run_dir=run_dir,
                        output_path=run_dir / "evaluations" / f"{role}.json",
                    )
                )
    _require(len(tasks) == 24, "evaluation task count is not 24")
    _require(len({task.key for task in tasks}) == 24, "evaluation task keys collide")
    _require(
        {task.checkpoint_role for task in tasks} == set(evaluator.CHECKPOINT_ROLES),
        "evaluation roles differ",
    )
    return tuple(tasks)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PosttrainingError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PosttrainingError(f"{label} is not finite")
    return result


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PosttrainingError(f"{label} is not a non-negative JSON integer")
    return value


def validate_evaluation(task: EvaluationTask) -> dict[str, Any]:
    payload = _load_json(task.output_path)
    expected = {
        "schema": evaluator.SCHEMA,
        "status": "complete",
        "dataset": task.dataset,
        "method": task.method,
        "requested_tss_weight": task.requested_tss_weight,
        "checkpoint_role": task.checkpoint_role,
        "seed": 42,
        "test_selected": True,
        "selection_is_optimistic": True,
        "stability_claim_supported": False,
    }
    for field, value in expected.items():
        _require(payload.get(field) == value, f"{task.key} evaluation {field} differs")
    roles = payload.get("threshold_roles")
    _require(isinstance(roles, Mapping), f"{task.key} lacks threshold_roles")
    _require(
        roles.get("checkpoint_selection_threshold") == 0.5
        and roles.get("global_lambda_selection_threshold") == 0.5
        and roles.get("main_table_threshold") == 0.5
        and roles.get("descriptive_sweep_only") is True
        and roles.get("sweep_reselects_checkpoint") is False
        and roles.get("sweep_reselects_global_lambda") is False,
        f"{task.key} threshold-role contract differs",
    )
    fixed = payload.get("fixed_threshold_0_5")
    _require(isinstance(fixed, Mapping), f"{task.key} lacks fixed_threshold_0_5")
    _require(_finite_number(fixed.get("threshold"), "threshold") == 0.5, f"{task.key} fixed threshold differs")
    for field in ("miou", "niou", "pd", "fa"):
        _finite_number(fixed.get(field), f"{task.key}.{field}")
    for field in (
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "unmatched_predicted_pixels",
        "valid_pixel_count",
    ):
        _nonnegative_integer(fixed.get(field), f"{task.key}.{field}")
    _require(
        fixed["matched_target_count"] <= fixed["target_count"],
        f"{task.key} matched targets exceed target count",
    )
    _require(
        fixed["valid_pixel_count"] > 0
        and fixed["unmatched_predicted_pixels"] <= fixed["valid_pixel_count"],
        f"{task.key} unmatched/valid pixel counts are invalid",
    )
    tiny_count = fixed["tiny_target_count"]
    if tiny_count == 0:
        _require(
            fixed.get("matched_tiny_target_count") is None,
            f"{task.key} tiny matched count must be null",
        )
    else:
        matched_tiny = _nonnegative_integer(
            fixed.get("matched_tiny_target_count"),
            f"{task.key}.matched_tiny_target_count",
        )
        _require(
            matched_tiny <= tiny_count,
            f"{task.key} matched tiny targets exceed tiny target count",
        )
    audit = payload.get("checkpoint_metric_audit")
    _require(isinstance(audit, Mapping) and audit.get("passed") is True, f"{task.key} checkpoint metric audit failed")
    binding = payload.get("checkpoint_binding")
    _require(isinstance(binding, Mapping), f"{task.key} lacks checkpoint binding")
    _require(
        Path(str(binding.get("run_dir", ""))).resolve() == task.run_dir,
        f"{task.key} checkpoint binding run_dir differs",
    )
    expected_observed_weight = (
        0.0 if task.method == "original" else task.requested_tss_weight
    )
    _require(
        binding.get("requested_tss_weight") == expected_observed_weight,
        f"{task.key} checkpoint binding TSS weight differs",
    )
    for name in ("summary", "protocol"):
        record = binding.get(name)
        _require(isinstance(record, Mapping), f"{task.key} lacks {name} binding")
        expected_path = task.run_dir / f"{name}.json"
        _require(
            Path(str(record.get("path", ""))).resolve()
            == expected_path.resolve(strict=True),
            f"{task.key} {name} path differs",
        )
        _require(
            record.get("sha256") == _file_sha256(expected_path),
            f"{task.key} {name} SHA differs",
        )
    checkpoint = binding.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), f"{task.key} lacks checkpoint identity")
    expected_checkpoint = task.run_dir / "checkpoints" / evaluator.CHECKPOINT_FILENAMES[task.checkpoint_role]
    _require(
        Path(str(checkpoint.get("path", ""))).resolve() == expected_checkpoint.resolve(strict=True),
        f"{task.key} checkpoint path differs",
    )
    _require(
        checkpoint.get("sha256") == _file_sha256(expected_checkpoint),
        f"{task.key} checkpoint SHA differs",
    )
    _require(checkpoint.get("role") == task.checkpoint_role, f"{task.key} checkpoint role differs")
    checkpoint_payload = torch.load(
        expected_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    _require(
        isinstance(checkpoint_payload, Mapping),
        f"{task.key} checkpoint payload is not an object",
    )
    evaluator._checkpoint_metric_audit(checkpoint_payload, fixed)
    _require(
        fixed["unmatched_predicted_pixels"]
        == round(float(fixed["fa"]) * fixed["valid_pixel_count"]),
        f"{task.key} exact unmatched-pixel count disagrees with Fa",
    )
    runtime = binding.get("training_runtime_sources")
    _require(isinstance(runtime, Mapping) and runtime.get("validated") is True, f"{task.key} training source audit missing")
    data = payload.get("data")
    _require(isinstance(data, Mapping), f"{task.key} lacks data identity")
    expected_split = data_protocol.EXPECTED_SPLITS[task.dataset]["test"]
    _require(data.get("split") == "img_idx/test", f"{task.key} split differs")
    _require(data.get("sirst3_in_formal_matrix") is False, f"{task.key} includes SIRST3")
    _require(data.get("img_idx_test_sha256") == expected_split["file_sha256"], f"{task.key} img_idx SHA differs")
    _require(
        data.get("img_idx_test_ordered_ids_sha256") == expected_split["ordered_ids_sha256"],
        f"{task.key} ordered ID SHA differs",
    )
    manifest = data.get("protocol_manifest")
    _require(isinstance(manifest, Mapping), f"{task.key} lacks manifest binding")
    expected_manifest = data_protocol.DEFAULT_MANIFEST_PATH.resolve(strict=True)
    _require(
        Path(str(manifest.get("path", ""))).resolve() == expected_manifest,
        f"{task.key} manifest path differs",
    )
    _require(
        manifest.get("sha256") == _file_sha256(expected_manifest),
        f"{task.key} manifest SHA differs",
    )
    _require(
        payload.get("source_sha256") == evaluator.evaluator_source_sha256(),
        f"{task.key} evaluator source hashes differ",
    )
    sweep = payload.get("descriptive_pd_fa")
    _require(isinstance(sweep, Mapping), f"{task.key} lacks descriptive sweep")
    _require(sweep.get("selection_effect") == "none", f"{task.key} sweep affects selection")
    points = sweep.get("points")
    _require(isinstance(points, list) and points, f"{task.key} sweep points are missing")
    empty_points = [
        point
        for point in points
        if isinstance(point, Mapping) and point.get("threshold") == 1.0
    ]
    _require(len(empty_points) == 1, f"{task.key} lacks one threshold=1.0 point")
    empty = empty_points[0]
    _require(
        empty.get("predicted_object_count") == 0
        and empty.get("pd") == 0.0
        and empty.get("fa") == 0.0,
        f"{task.key} threshold=1.0 point is not the empty endpoint",
    )
    return payload


def _task_command(
    task: EvaluationTask,
    *,
    python: Path,
    dataset_root: Path,
    protocol_manifest: Path,
    workers: int,
) -> tuple[str, ...]:
    command = [
        str(python),
        str(Path(evaluator.__file__).resolve()),
        "--dataset",
        task.dataset,
        "--method",
        task.method,
        "--checkpoint-role",
        task.checkpoint_role,
        "--run-dir",
        str(task.run_dir),
        "--dataset-root",
        str(dataset_root),
        "--data-protocol-manifest",
        str(protocol_manifest),
        "--output",
        str(task.output_path),
        "--device",
        "cuda:0",
        "--workers",
        str(workers),
    ]
    if task.method == "final":
        command[6:6] = ["--requested-tss-weight", str(task.requested_tss_weight)]
    return tuple(command)


def _gpu_inventory() -> dict[int, str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise PosttrainingError(f"nvidia-smi failed: {completed.stderr.strip()}")
    inventory: dict[int, str] = {}
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 1)]
        _require(len(fields) == 2, f"invalid nvidia-smi row: {line!r}")
        inventory[int(fields[0])] = fields[1]
    return inventory


def verify_gpu_bindings() -> dict[str, Any]:
    inventory = _gpu_inventory()
    result: dict[str, Any] = {}
    for role, binding in GPU_BINDINGS.items():
        index = int(binding["physical_index"])
        uuid = str(binding["uuid"])
        _require(inventory.get(index) == uuid, f"physical GPU {index} UUID differs")
        result[role] = {"physical_index": index, "uuid": uuid}
    _require(
        GPU_BINDINGS["best_miou"]["uuid"] != GPU_BINDINGS["best_pd"]["uuid"],
        "post-training GPU UUIDs collide",
    )
    return result


def _run_task(
    task: EvaluationTask,
    *,
    python: Path,
    dataset_root: Path,
    protocol_manifest: Path,
    workers: int,
    log_root: Path,
) -> dict[str, Any]:
    started = time.time()
    if task.output_path.exists():
        validate_evaluation(task)
        return {
            "task": task.key,
            "action": "skipped_verified_complete",
            "elapsed_seconds": time.time() - started,
            "output": _artifact(task.output_path),
        }
    binding = GPU_BINDINGS[task.checkpoint_role]
    command = _task_command(
        task,
        python=python,
        dataset_root=dataset_root,
        protocol_manifest=protocol_manifest,
        workers=workers,
    )
    log_path = _next_log_path(log_root, task.key)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(binding["uuid"])
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "task": task.key,
                    "physical_gpu_index": binding["physical_index"],
                    "gpu_uuid": binding["uuid"],
                    "command": list(command),
                    "command_shell_display": shlex.join(command),
                    "started_at_unix": started,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode != 0:
        raise PosttrainingError(
            f"evaluation {task.key} failed with return code {completed.returncode}; log={log_path}"
        )
    validate_evaluation(task)
    return {
        "task": task.key,
        "action": "executed",
        "physical_gpu_index": binding["physical_index"],
        "gpu_uuid": binding["uuid"],
        "elapsed_seconds": time.time() - started,
        "output": _artifact(task.output_path),
        "log": _artifact(log_path),
    }


def _selector_input(tasks: Sequence[EvaluationTask]) -> dict[str, Any]:
    by_identity = {
        (
            task.dataset,
            task.method,
            task.requested_tss_weight,
            task.checkpoint_role,
        ): validate_evaluation(task)
        for task in tasks
    }
    datasets: dict[str, Any] = {}
    for dataset in data_protocol.DATASETS:
        expected_split = data_protocol.EXPECTED_SPLITS[dataset]["test"]
        original: dict[str, Any] = {}
        final_candidates: dict[str, Any] = {}
        for role in evaluator.CHECKPOINT_ROLES:
            original[role] = copy.deepcopy(
                by_identity[(dataset, "original", None, role)]["fixed_threshold_0_5"]
            )
        for weight in evaluator.TSS_CANDIDATES:
            weight_float = float(weight)
            final_candidates[_weight_key(weight_float)] = {
                role: copy.deepcopy(
                    by_identity[(dataset, "final", weight_float, role)][
                        "fixed_threshold_0_5"
                    ]
                )
                for role in evaluator.CHECKPOINT_ROLES
            }
        datasets[dataset] = {
            "selection_split": "img_idx/test",
            "img_idx_test_sha256": expected_split["file_sha256"],
            "img_idx_test_ordered_ids_sha256": expected_split[
                "ordered_ids_sha256"
            ],
            "original": original,
            "final_candidates": final_candidates,
        }
    payload = {
        "schema": selector.INPUT_SCHEMA,
        "selection_split": "img_idx/test",
        "test_selected": True,
        "training_seed": 42,
        "threshold": 0.5,
        "checkpoint_roles": list(evaluator.CHECKPOINT_ROLES),
        "candidate_lambdas": list(map(float, evaluator.TSS_CANDIDATES)),
        "datasets": datasets,
    }
    selector.validate_input(payload)
    return payload


def _all_results(
    tasks: Sequence[EvaluationTask],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in data_protocol.DATASETS:
        rows: list[dict[str, Any]] = []
        for task in tasks:
            if task.dataset != dataset:
                continue
            payload = validate_evaluation(task)
            rows.append(
                {
                    "method": task.method,
                    "requested_tss_weight": task.requested_tss_weight,
                    "checkpoint_role": task.checkpoint_role,
                    "fixed_threshold_0_5": payload["fixed_threshold_0_5"],
                    "descriptive_pd_fa": {
                        "best_points_under_fa_budget": payload[
                            "descriptive_pd_fa"
                        ]["best_points_under_fa_budget"],
                        "pareto_frontier": payload["descriptive_pd_fa"][
                            "pareto_frontier"
                        ],
                    },
                    "evaluation": _artifact(task.output_path),
                }
            )
        datasets[dataset] = rows
    return {
        "schema": f"{SCHEMA_PREFIX}_all_results_v1",
        "status": "complete",
        "seed": 42,
        "selection_split": "img_idx/test",
        "test_selected": True,
        "selection_is_optimistic": True,
        "fixed_threshold": 0.5,
        "evaluation_count": len(tasks),
        "checkpoint_roles": list(evaluator.CHECKPOINT_ROLES),
        "datasets": datasets,
        "global_tss_selection": {
            "decision": selection.get("decision"),
            "global_tss_recipe_established": selection.get(
                "global_tss_recipe_established"
            ),
            "global_tss_lambda": selection.get("global_tss_lambda"),
        },
        "paper_core_established": False,
        "stability_claim_supported": False,
        "no_fabricated_results": True,
    }


def _status_payload(
    *,
    status: str,
    started_at: float,
    task_records: Mapping[str, Mapping[str, Any]],
    tasks: Sequence[EvaluationTask],
    gpu_bindings: Mapping[str, Any],
    error: str | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    complete = sum(
        1
        for task in tasks
        if task.key in task_records
        and task_records[task.key].get("action")
        in {"executed", "skipped_verified_complete"}
    )
    payload: dict[str, Any] = {
        "schema": f"{SCHEMA_PREFIX}_status_v1",
        "status": status,
        "started_at_unix": started_at,
        "updated_at_unix": time.time(),
        "planned_evaluation_count": len(tasks),
        "completed_evaluation_count": complete,
        "remaining_evaluation_count": len(tasks) - complete,
        "gpu_bindings": dict(gpu_bindings),
        "task_records": dict(sorted(task_records.items())),
    }
    if error is not None:
        payload["error"] = error
    if artifacts is not None:
        payload["artifacts"] = dict(artifacts)
    return payload


def run_posttraining(
    *,
    launch_plan_path: Path = LAUNCH_PLAN,
    results_root: Path = RESULTS_ROOT,
    dataset_root: Path = DATASET_ROOT,
    protocol_manifest: Path = PROTOCOL_MANIFEST,
    python: Path = PYTHON,
    workers: int = 0,
) -> dict[str, Any]:
    if workers < 0:
        raise PosttrainingError("workers must be non-negative")
    python = Path(python)
    _require(python.exists(), f"Python executable is missing: {python}")
    dataset_root = Path(dataset_root).resolve(strict=True)
    protocol_manifest = Path(protocol_manifest).resolve(strict=True)
    results_root = Path(results_root).resolve(strict=True)
    post_root = results_root / "posttraining_v2"
    post_root.mkdir(parents=True, exist_ok=True)
    status_path = post_root / "status.json"
    lock_handle = _acquire_posttraining_lock(post_root, status_path)

    started = time.time()
    launch_plan_sha256 = _file_sha256(launch_plan_path)
    try:
        plan = load_completed_plan(launch_plan_path)
        tasks = build_tasks(plan)
        for task in tasks:
            _require(
                task.run_dir == _expected_run_dir(results_root, task),
                f"{task.key} run directory differs from the canonical results tree",
            )
        gpu_bindings = verify_gpu_bindings()
    except Exception as exc:
        try:
            _atomic_json(
                status_path,
                {
                    "schema": f"{SCHEMA_PREFIX}_status_v1",
                    "status": "failed_preflight",
                    "started_at_unix": started,
                    "updated_at_unix": time.time(),
                    "error": str(exc),
                },
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
        raise
    log_root = post_root / "logs"
    task_records: dict[str, dict[str, Any]] = {}
    status_lock = threading.Lock()
    stop_event = threading.Event()

    def record(task: EvaluationTask, value: Mapping[str, Any]) -> None:
        with status_lock:
            task_records[task.key] = dict(value)
            _atomic_json(
                status_path,
                _status_payload(
                    status="evaluating",
                    started_at=started,
                    task_records=task_records,
                    tasks=tasks,
                    gpu_bindings=gpu_bindings,
                ),
            )

    _atomic_json(
        status_path,
        _status_payload(
            status="evaluating",
            started_at=started,
            task_records=task_records,
            tasks=tasks,
            gpu_bindings=gpu_bindings,
        ),
    )

    def run_queue(role: str) -> None:
        for task in tasks:
            if task.checkpoint_role != role:
                continue
            if stop_event.is_set():
                return
            try:
                value = _run_task(
                    task,
                    python=python,
                    dataset_root=dataset_root,
                    protocol_manifest=protocol_manifest,
                    workers=workers,
                    log_root=log_root,
                )
                record(task, value)
            except Exception:
                stop_event.set()
                raise

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(run_queue, role): role
                for role in evaluator.CHECKPOINT_ROLES
            }
            for future in concurrent.futures.as_completed(futures):
                role = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    stop_event.set()
                    for other in futures:
                        if other is not future:
                            other.cancel()
                    raise PosttrainingError(f"{role} GPU queue failed: {exc}") from exc

        for task in tasks:
            validate_evaluation(task)
        _require(len(task_records) == 24, "not all evaluation tasks were recorded")
        _require(
            _file_sha256(launch_plan_path) == launch_plan_sha256,
            "launch plan changed during post-training execution",
        )
        selector_input = _selector_input(tasks)
        selection_root = results_root / "selection"
        selector_input_path = selection_root / "selector_input_v2.json"
        selector_input_action = _write_once_or_identical(
            selector_input_path, selector_input
        )
        selection = selector.select_global_recipe(selector_input)
        selection["launch_plan_binding"] = selector.validate_launch_plan_binding(
            launch_plan_path,
            current_sources=selection["source_sha256"],
        )
        selection_path = selection_root / "global_tss_recipe_selection_v2.json"
        selection_action = _write_once_or_identical(selection_path, selection)
        all_results = _all_results(tasks, selection)
        all_results_path = post_root / "all_results_v2.json"
        all_results_action = _write_once_or_identical(all_results_path, all_results)
        artifacts = {
            "launch_plan": _artifact(launch_plan_path),
            "selector_input": {
                **_artifact(selector_input_path),
                "action": selector_input_action,
            },
            "selection": {
                **_artifact(selection_path),
                "action": selection_action,
            },
            "all_results": {
                **_artifact(all_results_path),
                "action": all_results_action,
            },
            "evaluations": [_artifact(task.output_path) for task in tasks],
        }
        complete = _status_payload(
            status="complete",
            started_at=started,
            task_records=task_records,
            tasks=tasks,
            gpu_bindings=gpu_bindings,
            artifacts=artifacts,
        )
        complete["decision"] = selection.get("decision")
        complete["global_tss_recipe_established"] = selection.get(
            "global_tss_recipe_established"
        )
        complete["global_tss_lambda"] = selection.get("global_tss_lambda")
        _atomic_json(status_path, complete)
        return complete
    except Exception as exc:
        _atomic_json(
            status_path,
            _status_payload(
                status="failed",
                started_at=started,
                task_records=task_records,
                tasks=tasks,
                gpu_bindings=gpu_bindings,
                error=str(exc),
            ),
        )
        raise
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def dry_run(
    *,
    launch_plan_path: Path = LAUNCH_PLAN,
) -> dict[str, Any]:
    plan = load_completed_plan(launch_plan_path)
    tasks = build_tasks(plan)
    reusable: list[str] = []
    pending: list[str] = []
    for task in tasks:
        if task.output_path.exists():
            validate_evaluation(task)
            reusable.append(task.key)
        else:
            pending.append(task.key)
    return {
        "schema": f"{SCHEMA_PREFIX}_dry_run_v1",
        "status": "ready",
        "will_execute": False,
        "planned_evaluation_count": len(tasks),
        "verified_reusable_count": len(reusable),
        "pending_evaluation_count": len(pending),
        "verified_reusable_tasks": reusable,
        "pending_tasks": pending,
        "queue_counts": {
            role: sum(task.checkpoint_role == role for task in tasks)
            for role in evaluator.CHECKPOINT_ROLES
        },
        "gpu_bindings": verify_gpu_bindings(),
        "launch_plan": _artifact(launch_plan_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-plan", type=Path, default=LAUNCH_PLAN)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--data-protocol-manifest", type=Path, default=PROTOCOL_MANIFEST
    )
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        result = dry_run(launch_plan_path=args.launch_plan)
    else:
        result = run_posttraining(
            launch_plan_path=args.launch_plan,
            results_root=args.results_root,
            dataset_root=args.dataset_root,
            protocol_manifest=args.data_protocol_manifest,
            python=args.python,
            workers=args.workers,
        )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
