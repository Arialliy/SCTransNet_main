#!/usr/bin/env python3
"""Shared launch protocol for the four-dataset seed-42 experiment.

This module deliberately has no torch import.  Its responsibilities are
limited to constructing exact worker commands, binding each method to its
assigned GPU UUID, recording per-task process state, and supervising the four
ordered waves.  Training and rolling recovery remain the responsibility of the
formal runner.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "sctransnet_four_dataset_seed42_launcher/v1"
TRAINING_SEED = 42
DATASETS = ("SIRST3", "NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
METHODS = ("original", "final")

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
RUNNER = REPO_ROOT / "experiments" / (
    "train_four_dataset_original_final_seed42_exact_v1.py"
)
DATA_ROOT = REPO_ROOT / "datasets"
RESULTS_ROOT = REPO_ROOT / "results" / "four_dataset_seed42_v1"
MANIFEST_ROOT = RESULTS_ROOT / "manifests"
TSS_STATISTICS = MANIFEST_ROOT / "four_dataset_tss_seed42_v1.json"

GPU_ASSIGNMENT = {
    "original": {
        "physical_index": "2",
        "uuid": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    },
    "final": {
        "physical_index": "3",
        "uuid": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
    },
}
CPU_THREAD_ENV = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "OMP_WAIT_POLICY": "PASSIVE",
    "KMP_BLOCKTIME": "0",
}


class LaunchProtocolError(RuntimeError):
    """Raised when a frozen launch invariant or worker postcondition fails."""


@dataclass(frozen=True)
class WorkerSpec:
    dataset: str
    method: str
    mode: str
    command: tuple[str, ...]
    environment: Mapping[str, str]
    run_directory: Path
    launch_directory: Path

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.method}"


@dataclass
class ActiveWorker:
    spec: WorkerSpec
    process: subprocess.Popen[bytes]
    stdout_handle: Any
    stderr_handle: Any
    started_at: float


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_dataset(dataset: str) -> None:
    if dataset not in DATASETS:
        raise LaunchProtocolError(
            f"dataset must be one of {DATASETS}, got {dataset!r}"
        )


def _require_method(method: str) -> None:
    if method not in METHODS:
        raise LaunchProtocolError(
            f"method must be one of {METHODS}, got {method!r}"
        )


def validate_static_inputs() -> dict[str, Any]:
    missing = [
        str(path)
        for path in (PYTHON, RUNNER, DATA_ROOT, MANIFEST_ROOT)
        if not path.exists()
    ]
    if missing:
        raise LaunchProtocolError(f"missing static launch inputs: {missing}")
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise LaunchProtocolError(f"python is not executable: {PYTHON}")
    if not RUNNER.is_file():
        raise LaunchProtocolError(f"runner is not a file: {RUNNER}")
    return {
        "python": str(PYTHON),
        "python_sha256": _file_sha256(PYTHON),
        "runner": str(RUNNER),
        "runner_sha256": _file_sha256(RUNNER),
        "data_root": str(DATA_ROOT),
        "results_root": str(RESULTS_ROOT),
        "manifest_root": str(MANIFEST_ROOT),
    }


def validate_tss_statistics(
    path: Path = TSS_STATISTICS,
) -> dict[str, Any]:
    if not path.is_file():
        raise LaunchProtocolError(
            "the frozen four-dataset TSS statistics artifact is missing: "
            f"{path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected_top = {
        "schema": "sctransnet_four_dataset_exact_tss_statistics/v1",
        "training_seed": TRAINING_SEED,
        "epochs": 1000,
    }
    for field, expected in expected_top.items():
        if payload.get(field) != expected:
            raise LaunchProtocolError(
                f"TSS statistics {field} differs: "
                f"{payload.get(field)!r} != {expected!r}"
            )
    records = payload.get("datasets")
    if not isinstance(records, Mapping) or set(records) != set(DATASETS):
        raise LaunchProtocolError(
            "TSS statistics must contain exactly the four training regimes"
        )
    compact_records: dict[str, Any] = {}
    for dataset in DATASETS:
        record = records[dataset]
        if not isinstance(record, Mapping):
            raise LaunchProtocolError(
                f"TSS dataset record is invalid: {dataset}"
            )
        for field, expected in (
            ("dataset", dataset),
            ("training_seed", TRAINING_SEED),
            ("epochs", 1000),
            ("completed_through_epoch", 1000),
            ("complete", True),
        ):
            if record.get(field) != expected:
                raise LaunchProtocolError(
                    f"TSS {dataset} {field} differs: "
                    f"{record.get(field)!r} != {expected!r}"
                )
        weight = record.get("survival_pos_weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise LaunchProtocolError(
                f"TSS {dataset} survival_pos_weight is invalid"
            )
        compact_records[dataset] = {
            "survival_pos_weight": float(weight),
            "positive_cells": record.get("positive_cells"),
            "negative_cells": record.get("negative_cells"),
            "aggregate_plan_sha256": record.get(
                "aggregate_plan_sha256"
            ),
        }
    return {
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "schema": payload["schema"],
        "training_seed": payload["training_seed"],
        "epochs": payload["epochs"],
        "datasets": compact_records,
    }


def build_worker_spec(
    dataset: str,
    method: str,
    *,
    mode: str,
    max_train_images: int = 2,
    max_test_images: int = 2,
    base_environment: Mapping[str, str] | None = None,
) -> WorkerSpec:
    """Build one exact worker command without starting a process."""

    _require_dataset(dataset)
    _require_method(method)
    if mode not in {"smoke", "formal"}:
        raise LaunchProtocolError("mode must be 'smoke' or 'formal'")
    if max_train_images < 1 or max_test_images < 1:
        raise LaunchProtocolError("smoke sample limits must be positive")

    gpu = GPU_ASSIGNMENT[method]
    command = [
        str(PYTHON),
        str(RUNNER),
        "--dataset",
        dataset,
        "--method",
        method,
        "--data-root",
        str(DATA_ROOT),
        "--results-root",
        str(RESULTS_ROOT),
        "--manifest-root",
        str(MANIFEST_ROOT),
        "--seed",
        "42",
        "--patch-size",
        "256",
        "--workers",
        "0",
        "--base-lr",
        "0.001",
        "--min-lr",
        "0.00001",
        "--warmup-epochs",
        "10",
        "--threshold",
        "0.5",
        "--match-radius",
        "3.0",
        "--tiny-area",
        "9",
        "--device",
        "cuda:0",
        "--physical-gpu-index",
        gpu["physical_index"],
        "--expected-gpu-uuid",
        gpu["uuid"],
        "--resume",
        "auto",
    ]
    if method == "final":
        command.extend(["--tss-statistics", str(TSS_STATISTICS)])
    if mode == "smoke":
        command.extend(
            [
                "--smoke",
                "--epochs",
                "2",
                "--begin-test",
                "1",
                "--eval-every",
                "1",
                "--batch-size",
                "2",
                "--max-train-images",
                str(max_train_images),
                "--max-test-images",
                str(max_test_images),
            ]
        )
        run_directory = (
            RESULTS_ROOT
            / "smoke"
            / "runs"
            / dataset
            / method
            / "seed_42"
        )
    else:
        command.extend(
            [
                "--epochs",
                "1000",
                "--begin-test",
                "10",
                "--eval-every",
                "10",
                "--batch-size",
                "16",
            ]
        )
        run_directory = (
            RESULTS_ROOT / "runs" / dataset / method / "seed_42"
        )

    environment = dict(
        os.environ if base_environment is None else base_environment
    )
    environment["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment.update(CPU_THREAD_ENV)
    launch_directory = (
        RESULTS_ROOT / "launch" / mode / dataset / method / "seed_42"
    )
    return WorkerSpec(
        dataset=dataset,
        method=method,
        mode=mode,
        command=tuple(command),
        environment=environment,
        run_directory=run_directory,
        launch_directory=launch_directory,
    )


def build_all_worker_specs(
    *,
    mode: str,
    max_train_images: int = 2,
    max_test_images: int = 2,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[WorkerSpec, ...]:
    return tuple(
        build_worker_spec(
            dataset,
            method,
            mode=mode,
            max_train_images=max_train_images,
            max_test_images=max_test_images,
            base_environment=base_environment,
        )
        for dataset in DATASETS
        for method in METHODS
    )


def command_record(spec: WorkerSpec) -> dict[str, Any]:
    gpu = GPU_ASSIGNMENT[spec.method]
    return {
        "schema": SCHEMA,
        "dataset": spec.dataset,
        "method": spec.method,
        "mode": spec.mode,
        "seed": TRAINING_SEED,
        "command": list(spec.command),
        "cwd": str(REPO_ROOT),
        "gpu": dict(gpu),
        "cuda_visible_devices": spec.environment.get(
            "CUDA_VISIBLE_DEVICES"
        ),
        "cpu_thread_environment": {
            name: spec.environment.get(name) for name in CPU_THREAD_ENV
        },
        "run_directory": str(spec.run_directory),
        "launch_directory": str(spec.launch_directory),
        "checkpoint_roles": ["best_miou", "best_pd"],
        "rolling_resume_state_is_selected_checkpoint": False,
    }


def _checkpoint_postcondition(spec: WorkerSpec) -> dict[str, Any]:
    summary_path = spec.run_directory / "summary.json"
    if not summary_path.is_file():
        raise LaunchProtocolError(f"worker summary is missing: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    expected = {
        "status": "complete",
        "dataset": spec.dataset,
        "method": spec.method,
        "seed": TRAINING_SEED,
        "epochs": 2 if spec.mode == "smoke" else 1000,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise LaunchProtocolError(
                f"{spec.key} summary {field} differs: "
                f"{summary.get(field)!r} != {value!r}"
            )
    checkpoint_records = summary.get("checkpoints")
    if (
        not isinstance(checkpoint_records, Mapping)
        or set(checkpoint_records) != {"best_miou", "best_pd"}
    ):
        raise LaunchProtocolError(
            f"{spec.key} must report exactly best_miou and best_pd"
        )
    checkpoint_dir = spec.run_directory / "checkpoints"
    expected_names = {"best_miou.pth.tar", "best_pd.pth.tar"}
    actual_names = {
        path.name
        for path in checkpoint_dir.iterdir()
        if path.is_file() and (".pth" in path.name or ".tar" in path.name)
    }
    if actual_names != expected_names:
        raise LaunchProtocolError(
            f"{spec.key} selected checkpoint files differ: "
            f"{sorted(actual_names)} != {sorted(expected_names)}"
        )
    artifacts: dict[str, Any] = {}
    for role in ("best_miou", "best_pd"):
        path = checkpoint_dir / f"{role}.pth.tar"
        if not path.is_file() or path.stat().st_size <= 0:
            raise LaunchProtocolError(
                f"{spec.key} checkpoint is missing or empty: {path}"
            )
        artifacts[role] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    latest = (
        spec.run_directory / "resume" / "latest_training_state.pth.tar"
    )
    if latest.exists():
        raise LaunchProtocolError(
            f"successful worker retained rolling resume state: {latest}"
        )
    return {
        "summary": str(summary_path),
        "summary_sha256": _file_sha256(summary_path),
        "checkpoints": artifacts,
    }


class WaveSupervisor:
    """Run four ordered waves with one Original and one Final per wave."""

    def __init__(
        self,
        *,
        mode: str,
        max_train_images: int = 2,
        max_test_images: int = 2,
        poll_seconds: float = 1.0,
    ) -> None:
        if mode not in {"smoke", "formal"}:
            raise LaunchProtocolError("unsupported supervisor mode")
        if poll_seconds <= 0:
            raise LaunchProtocolError("poll_seconds must be positive")
        self.mode = mode
        self.max_train_images = max_train_images
        self.max_test_images = max_test_images
        self.poll_seconds = poll_seconds
        self.root = RESULTS_ROOT / "launch" / mode
        self.status_path = self.root / "supervisor_status.json"
        self._active: dict[str, ActiveWorker] = {}
        self._stop_signal: int | None = None

    def specs(self) -> tuple[WorkerSpec, ...]:
        return build_all_worker_specs(
            mode=self.mode,
            max_train_images=self.max_train_images,
            max_test_images=self.max_test_images,
        )

    def dry_run_payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "mode": self.mode,
            "wave_order": list(DATASETS),
            "parallel_methods_per_wave": list(METHODS),
            "tasks": [command_record(spec) for spec in self.specs()],
        }

    def _status(
        self,
        status: str,
        *,
        wave_index: int | None = None,
        dataset: str | None = None,
        detail: Any = None,
    ) -> None:
        payload = {
            "schema": SCHEMA,
            "mode": self.mode,
            "status": status,
            "wave_order": list(DATASETS),
            "parallel_methods_per_wave": list(METHODS),
            "wave_index": wave_index,
            "dataset": dataset,
            "active_workers": {
                key: {
                    "pid": active.process.pid,
                    "dataset": active.spec.dataset,
                    "method": active.spec.method,
                    "started_at_unix": active.started_at,
                }
                for key, active in self._active.items()
            },
            "detail": detail,
            "updated_at_unix": time.time(),
        }
        _write_json_atomic(self.status_path, payload)

    def _signal_handler(self, signum: int, _frame: Any) -> None:
        self._stop_signal = signum
        for active in self._active.values():
            if active.process.poll() is None:
                active.process.terminate()

    def _start_worker(self, spec: WorkerSpec) -> ActiveWorker:
        directory = spec.launch_directory
        directory.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(directory / "command.json", command_record(spec))
        stdout_handle = (directory / "stdout.log").open("ab", buffering=0)
        stderr_handle = (directory / "stderr.log").open("ab", buffering=0)
        started_at = time.time()
        _write_json_atomic(
            directory / "status.json",
            {
                "schema": SCHEMA,
                "status": "starting",
                "dataset": spec.dataset,
                "method": spec.method,
                "mode": spec.mode,
                "started_at_unix": started_at,
            },
        )
        try:
            process = subprocess.Popen(
                list(spec.command),
                cwd=REPO_ROOT,
                env=dict(spec.environment),
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )
        except BaseException:
            stdout_handle.close()
            stderr_handle.close()
            raise
        active = ActiveWorker(
            spec=spec,
            process=process,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            started_at=started_at,
        )
        _write_json_atomic(
            directory / "status.json",
            {
                "schema": SCHEMA,
                "status": "running",
                "dataset": spec.dataset,
                "method": spec.method,
                "mode": spec.mode,
                "pid": process.pid,
                "started_at_unix": started_at,
                "stdout": str(directory / "stdout.log"),
                "stderr": str(directory / "stderr.log"),
            },
        )
        return active

    def _finish_worker(
        self,
        active: ActiveWorker,
        exit_code: int,
    ) -> dict[str, Any]:
        active.stdout_handle.close()
        active.stderr_handle.close()
        completed_at = time.time()
        postcondition: dict[str, Any] | None = None
        postcondition_error: str | None = None
        if exit_code == 0:
            try:
                postcondition = _checkpoint_postcondition(active.spec)
            except BaseException as error:
                postcondition_error = f"{type(error).__name__}: {error}"
        successful = exit_code == 0 and postcondition_error is None
        record = {
            "schema": SCHEMA,
            "status": "complete" if successful else "failed",
            "dataset": active.spec.dataset,
            "method": active.spec.method,
            "mode": active.spec.mode,
            "pid": active.process.pid,
            "exit_code": exit_code,
            "postcondition": postcondition,
            "postcondition_error": postcondition_error,
            "started_at_unix": active.started_at,
            "completed_at_unix": completed_at,
            "elapsed_seconds": completed_at - active.started_at,
        }
        _write_json_atomic(active.spec.launch_directory / "exit.json", record)
        _write_json_atomic(
            active.spec.launch_directory / "status.json", record
        )
        return record

    def _stop_siblings(self, failed_key: str) -> None:
        for key, active in self._active.items():
            if key != failed_key and active.process.poll() is None:
                active.process.terminate()
        deadline = time.monotonic() + 20.0
        for key, active in self._active.items():
            if key == failed_key:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            try:
                active.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                active.process.kill()

    def _run_wave(
        self,
        wave_index: int,
        dataset: str,
    ) -> dict[str, Any]:
        specs = [
            build_worker_spec(
                dataset,
                method,
                mode=self.mode,
                max_train_images=self.max_train_images,
                max_test_images=self.max_test_images,
            )
            for method in METHODS
        ]
        self._active = {}
        try:
            for spec in specs:
                self._active[spec.key] = self._start_worker(spec)
        except BaseException:
            for active in self._active.values():
                if active.process.poll() is None:
                    active.process.terminate()
            for active in self._active.values():
                try:
                    exit_code = active.process.wait(timeout=20.0)
                except subprocess.TimeoutExpired:
                    active.process.kill()
                    exit_code = active.process.wait()
                self._finish_worker(active, exit_code)
            self._active = {}
            raise
        self._status(
            "running_wave", wave_index=wave_index, dataset=dataset
        )
        records: dict[str, Any] = {}
        failure_key: str | None = None
        while self._active:
            for key, active in list(self._active.items()):
                exit_code = active.process.poll()
                if exit_code is None:
                    continue
                record = self._finish_worker(active, exit_code)
                records[key] = record
                del self._active[key]
                if record["status"] != "complete":
                    failure_key = key
                    self._stop_siblings(failure_key)
                    break
            if failure_key is not None:
                for key, active in list(self._active.items()):
                    exit_code = active.process.wait()
                    records[key] = self._finish_worker(active, exit_code)
                    del self._active[key]
                break
            if self._stop_signal is not None:
                for active in self._active.values():
                    if active.process.poll() is None:
                        active.process.terminate()
                for key, active in list(self._active.items()):
                    exit_code = active.process.wait()
                    records[key] = self._finish_worker(active, exit_code)
                    del self._active[key]
                break
            if self._active:
                time.sleep(self.poll_seconds)
        return {
            "dataset": dataset,
            "wave_index": wave_index,
            "status": (
                "complete"
                if all(
                    record["status"] == "complete"
                    for record in records.values()
                )
                and len(records) == 2
                else "failed"
            ),
            "tasks": records,
        }

    def run(self) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        static_inputs = validate_static_inputs()
        tss = validate_tss_statistics()
        lock_path = self.root / "supervisor.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            try:
                fcntl.flock(
                    lock_handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError as error:
                raise LaunchProtocolError(
                    f"{self.mode} supervisor is already active"
                ) from error
            previous_handlers = {
                signum: signal.getsignal(signum)
                for signum in (signal.SIGINT, signal.SIGTERM)
            }
            for signum in previous_handlers:
                signal.signal(signum, self._signal_handler)
            waves: list[dict[str, Any]] = []
            try:
                self._status(
                    "starting",
                    detail={"static_inputs": static_inputs, "tss": tss},
                )
                for wave_index, dataset in enumerate(DATASETS, start=1):
                    if self._stop_signal is not None:
                        break
                    wave = self._run_wave(wave_index, dataset)
                    waves.append(wave)
                    if wave["status"] != "complete":
                        self._status(
                            "failed",
                            wave_index=wave_index,
                            dataset=dataset,
                            detail={"waves": waves},
                        )
                        return 1
                if self._stop_signal is not None:
                    self._status(
                        "interrupted",
                        detail={
                            "signal": self._stop_signal,
                            "waves": waves,
                        },
                    )
                    return 128 + self._stop_signal
                self._status(
                    "complete",
                    detail={
                        "static_inputs": static_inputs,
                        "tss": tss,
                        "waves": waves,
                    },
                )
                return 0
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)


def shell_quoted_command(command: Sequence[str]) -> str:
    """Return a copy/paste-safe display form without executing a shell."""

    import shlex

    return shlex.join(list(command))


__all__ = [
    "DATASETS",
    "DATA_ROOT",
    "CPU_THREAD_ENV",
    "GPU_ASSIGNMENT",
    "LaunchProtocolError",
    "MANIFEST_ROOT",
    "METHODS",
    "PYTHON",
    "RESULTS_ROOT",
    "RUNNER",
    "SCHEMA",
    "TRAINING_SEED",
    "TSS_STATISTICS",
    "WaveSupervisor",
    "WorkerSpec",
    "build_all_worker_specs",
    "build_worker_spec",
    "command_record",
    "shell_quoted_command",
    "validate_static_inputs",
    "validate_tss_statistics",
]
