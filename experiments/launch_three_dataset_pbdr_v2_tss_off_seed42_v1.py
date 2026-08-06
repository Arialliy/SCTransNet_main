#!/usr/bin/env python3
"""Plan or launch the three independent PBDR-V2 seed42 formal1000 runs.

Dry-run is the default.  The launcher fixes one scratch run per dataset and
binds the child CUDA process by the complete GPU UUID.  ``--resume never`` is
the fresh-run default; ``required`` and ``auto`` are available for an explicit
restart policy without changing the run identity.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sctransnet_three_dataset_pbdr_v2_launcher_v1/v1"
PYTHON = Path("/home/ly/SCTransNet/.venv/bin/python")
TRAINER = (
    REPO_ROOT
    / "experiments"
    / "train_three_dataset_pbdr_v2_tss_off_seed42_v1.py"
)
DEFAULT_RESULTS_ROOT = (
    REPO_ROOT / "results" / "three_dataset_pbdr_v2_tss_off_seed42_v1"
)
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_PROTOCOL_MANIFEST = (
    REPO_ROOT
    / "results"
    / "three_dataset_v2"
    / "manifests"
    / "three_dataset_v2_protocol.json"
)
DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
TRAINING_SEED = 42
PLANNED_TOTAL_EPOCHS = 1000
RECIPE_ID = "pbdr_v2_tss_off"
TRAINING_LAYOUT = (
    ("NUAA-SIRST", "0"),
    ("NUDT-SIRST", "1"),
    ("IRSTD-1K", "2"),
)
GPU_ASSIGNMENTS = {
    "0": {
        "physical_index": "0",
        "uuid": "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    },
    "1": {
        "physical_index": "1",
        "uuid": "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
    },
    "2": {
        "physical_index": "2",
        "uuid": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    },
    "3": {
        "physical_index": "3",
        "uuid": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
    },
}
CPU_ENVIRONMENT = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "OMP_WAIT_POLICY": "PASSIVE",
    "KMP_BLOCKTIME": "0",
}
RESUME_MODES = ("never", "required", "auto")


@dataclass(frozen=True)
class WorkerSpec:
    dataset: str
    gpu_index: str
    expected_gpu_uuid: str
    resume: str
    run_directory: Path
    log_directory: Path
    command: tuple[str, ...]
    environment: Mapping[str, str]

    @property
    def key(self) -> str:
        return f"{self.dataset}__{RECIPE_ID}__seed42"


def run_directory(results_root: Path, dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    return (
        Path(results_root).resolve()
        / "runs"
        / dataset
        / RECIPE_ID
        / "seed_42"
    )


def _environment(
    gpu_index: str,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if gpu_index not in GPU_ASSIGNMENTS:
        raise ValueError(f"unknown physical GPU: {gpu_index}")
    environment = dict(os.environ if base is None else base)
    environment.update(CPU_ENVIRONMENT)
    # The full UUID is intentional: a numeric value would weaken the
    # trainer's physical-index/UUID attestation after CUDA remapping.
    environment["CUDA_VISIBLE_DEVICES"] = GPU_ASSIGNMENTS[gpu_index]["uuid"]
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return environment


def _training_command(
    *,
    dataset: str,
    gpu_index: str,
    resume: str,
    results_root: Path,
    data_root: Path,
    protocol_manifest: Path,
    python: Path,
) -> tuple[str, ...]:
    gpu = GPU_ASSIGNMENTS[gpu_index]
    return (
        str(python),
        str(TRAINER),
        "--dataset",
        dataset,
        "--method",
        "final",
        "--tss-weight",
        "0",
        "--data-root",
        str(Path(data_root).resolve()),
        "--results-root",
        str(Path(results_root).resolve()),
        "--protocol-manifest",
        str(Path(protocol_manifest).resolve()),
        "--seed",
        str(TRAINING_SEED),
        "--epochs",
        str(PLANNED_TOTAL_EPOCHS),
        "--begin-test",
        "10",
        "--eval-every",
        "10",
        "--batch-size",
        "16",
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
        resume,
    )


def build_worker_specs(
    *,
    resume: str = "never",
    results_root: Path = DEFAULT_RESULTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_manifest: Path = DEFAULT_PROTOCOL_MANIFEST,
    python: Path = PYTHON,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[WorkerSpec, ...]:
    if resume not in RESUME_MODES:
        raise ValueError(f"resume must be one of {RESUME_MODES}")
    specs: list[WorkerSpec] = []
    for dataset, gpu_index in TRAINING_LAYOUT:
        gpu_uuid = GPU_ASSIGNMENTS[gpu_index]["uuid"]
        specs.append(
            WorkerSpec(
                dataset=dataset,
                gpu_index=gpu_index,
                expected_gpu_uuid=gpu_uuid,
                resume=resume,
                run_directory=run_directory(results_root, dataset),
                log_directory=(
                    Path(results_root).resolve()
                    / "launch"
                    / "formal"
                    / "logs"
                    / dataset
                    / RECIPE_ID
                    / "seed_42"
                ),
                command=_training_command(
                    dataset=dataset,
                    gpu_index=gpu_index,
                    resume=resume,
                    results_root=results_root,
                    data_root=data_root,
                    protocol_manifest=protocol_manifest,
                    python=python,
                ),
                environment=_environment(gpu_index, base_environment),
            )
        )
    if tuple((spec.dataset, spec.gpu_index) for spec in specs) != TRAINING_LAYOUT:
        raise RuntimeError("PBDR-V2 training layout differs")
    if len({spec.run_directory for spec in specs}) != len(DATASETS):
        raise RuntimeError("each dataset must own an independent run directory")
    return tuple(specs)


def dry_run_payload(specs: Sequence[WorkerSpec]) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "dry_run": True,
        "recipe_id": RECIPE_ID,
        "training_seed": TRAINING_SEED,
        "planned_total_epochs": PLANNED_TOTAL_EPOCHS,
        "fresh_scratch_default": True,
        "datasets": list(DATASETS),
        "independent_run_per_dataset": True,
        "single_gpu_per_run": True,
        "ddp": False,
        "workers": [
            asdict(spec)
            | {
                "run_directory": str(spec.run_directory),
                "log_directory": str(spec.log_directory),
                "command": list(spec.command),
                "environment": {
                    "CUDA_VISIBLE_DEVICES": spec.environment[
                        "CUDA_VISIBLE_DEVICES"
                    ],
                    "PYTHONUNBUFFERED": spec.environment["PYTHONUNBUFFERED"],
                    "CUBLAS_WORKSPACE_CONFIG": spec.environment[
                        "CUBLAS_WORKSPACE_CONFIG"
                    ],
                },
            }
            for spec in specs
        ],
    }


def execute_workers(specs: Sequence[WorkerSpec]) -> int:
    processes: list[tuple[WorkerSpec, subprocess.Popen[bytes], object]] = []
    try:
        for spec in specs:
            spec.log_directory.mkdir(parents=True, exist_ok=True)
            log_handle = (spec.log_directory / "train.log").open("ab")
            process = subprocess.Popen(
                spec.command,
                cwd=REPO_ROOT,
                env=dict(spec.environment),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((spec, process, log_handle))
        return max((process.wait() for _, process, _ in processes), default=0)
    finally:
        for _, _, log_handle in processes:
            log_handle.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", choices=RESUME_MODES, default="never")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=DEFAULT_PROTOCOL_MANIFEST,
    )
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    specs = build_worker_specs(
        resume=args.resume,
        results_root=args.results_root,
        data_root=args.data_root,
        protocol_manifest=args.protocol_manifest,
        python=args.python,
    )
    if not args.execute:
        print(
            json.dumps(
                dry_run_payload(specs),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return
    raise SystemExit(execute_workers(specs))


if __name__ == "__main__":
    main()
