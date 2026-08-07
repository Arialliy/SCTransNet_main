#!/usr/bin/env python3
"""Dry-run-first launcher for NUDT/IRSTD PBDR-V3 Stage-1 workers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import train_two_dataset_pbdr_v3_stage1_v1 as trainer


SCHEMA = "sctransnet_two_dataset_pbdr_v3_launcher_v1/v1"
PYTHON = Path("/home/ly/SCTransNet/.venv/bin/python")
TRAINER = REPO_ROOT / "experiments/train_two_dataset_pbdr_v3_stage1_v1.py"
DATASETS = trainer.DATASETS
PARENT_ROLES = trainer.PARENT_ROLES
DEFAULT_RESULTS_ROOT = trainer.DEFAULT_RESULTS_ROOT
DEFAULT_DATA_ROOT = trainer.DEFAULT_DATA_ROOT
DEFAULT_PROTOCOL_MANIFEST = trainer.DEFAULT_PROTOCOL_MANIFEST
GPU_INDICES = {"NUDT-SIRST": "0", "IRSTD-1K": "1"}
GPU_UUIDS = trainer.GPU_UUIDS
UNIT_NAMES = {
    "NUDT-SIRST": "sctransnet-pbdr-v3-xdata-v1-nudt",
    "IRSTD-1K": "sctransnet-pbdr-v3-xdata-v1-irstd",
}
CPU_ENVIRONMENT = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "OMP_WAIT_POLICY": "PASSIVE",
    "KMP_BLOCKTIME": "0",
}


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    dataset: str
    parent_role: str
    run_directory: Path
    command: tuple[str, ...]


def dataset_environment(
    dataset: str,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if dataset not in DATASETS:
        raise ValueError("unsupported dataset")
    environment = dict(os.environ if base is None else base)
    environment.update(CPU_ENVIRONMENT)
    environment.update(
        CUDA_VISIBLE_DEVICES=GPU_UUIDS[dataset],
        PYTHONUNBUFFERED="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    return environment


def run_directory(
    results_root: Path,
    dataset: str,
    role: str,
    *,
    smoke: bool,
) -> Path:
    if dataset not in DATASETS or role not in PARENT_ROLES:
        raise ValueError("unsupported dataset/role")
    return (
        trainer.dataset_results_root(results_root, dataset)
        / ("smoke" if smoke else "formal")
        / role
        / "core"
    )


def build_worker_specs(
    *,
    dataset: str = "all",
    results_root: Path = DEFAULT_RESULTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_manifest: Path = DEFAULT_PROTOCOL_MANIFEST,
    python: Path = PYTHON,
    resume: str = "auto",
    smoke: bool = False,
    max_train_images: int = 2,
    max_val_images: int = 8,
) -> tuple[WorkerSpec, ...]:
    if resume not in ("auto", "never", "required"):
        raise ValueError("invalid resume mode")
    selected = DATASETS if dataset == "all" else (dataset,)
    if any(value not in DATASETS for value in selected):
        raise ValueError("unsupported dataset")
    specs: list[WorkerSpec] = []
    for dataset_name in selected:
        for role in PARENT_ROLES:
            command = [
                str(Path(python).absolute()),
                str(TRAINER),
                "--dataset",
                dataset_name,
                "--parent-role",
                role,
                "--recipe",
                "core",
                "--data-root",
                str(Path(data_root).resolve()),
                "--protocol-manifest",
                str(Path(protocol_manifest).resolve()),
                "--results-root",
                str(Path(results_root).resolve()),
                "--device",
                "cuda:0",
                "--expected-gpu-uuid",
                GPU_UUIDS[dataset_name],
                "--resume",
                resume,
            ]
            if smoke:
                command.extend(
                    (
                        "--smoke",
                        "--epochs",
                        "1",
                        "--eval-every",
                        "1",
                        "--batch-size",
                        "2",
                        "--max-train-images",
                        str(max_train_images),
                        "--max-val-images",
                        str(max_val_images),
                    )
                )
            specs.append(
                WorkerSpec(
                    dataset_name,
                    role,
                    run_directory(
                        results_root, dataset_name, role, smoke=smoke
                    ),
                    tuple(command),
                )
            )
    return tuple(specs)


def verify_gpu_bindings() -> dict[str, str]:
    completed = subprocess.run(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    observed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if line.strip():
            index, uuid = (part.strip() for part in line.split(",", 1))
            observed[index] = uuid
    for dataset in DATASETS:
        if observed.get(GPU_INDICES[dataset]) != GPU_UUIDS[dataset]:
            raise RuntimeError(f"{dataset} GPU UUID binding differs")
    return observed


def execute_dataset_sequence(
    dataset: str,
    specs: Sequence[WorkerSpec],
) -> int:
    selected = [spec for spec in specs if spec.dataset == dataset]
    if [spec.parent_role for spec in selected] != list(PARENT_ROLES):
        raise ValueError("dataset sequence must contain both ordered roles")
    environment = dataset_environment(dataset)
    for spec in selected:
        completed = subprocess.run(
            spec.command,
            cwd=REPO_ROOT,
            env=environment,
            check=False,
        )
        if completed.returncode:
            return int(completed.returncode)
    return 0


def systemd_command(args: argparse.Namespace, dataset: str) -> tuple[str, ...]:
    environment = dataset_environment(dataset, {})
    command = [
        "systemd-run",
        "--user",
        "--collect",
        "--wait",
        f"--unit={UNIT_NAMES[dataset]}{'-smoke' if args.smoke else ''}",
        f"--working-directory={REPO_ROOT}",
        "--property=Restart=no",
        "--property=Type=exec",
    ]
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "PYTHONUNBUFFERED",
        "CUBLAS_WORKSPACE_CONFIG",
        *CPU_ENVIRONMENT,
    ):
        command.append(f"--setenv={name}={environment[name]}")
    command.extend(
        (
            str(args.python.absolute()),
            str(Path(__file__).resolve()),
            "--worker-sequence",
            "--dataset",
            dataset,
            "--results-root",
            str(args.results_root.resolve()),
            "--data-root",
            str(args.data_root.resolve()),
            "--protocol-manifest",
            str(args.protocol_manifest.resolve()),
            "--python",
            str(args.python.absolute()),
            "--resume",
            args.resume,
        )
    )
    if args.smoke:
        command.extend(
            (
                "--smoke",
                "--max-train-images",
                str(args.max_train_images),
                "--max-val-images",
                str(args.max_val_images),
            )
        )
    return tuple(command)


def dry_run_payload(
    args: argparse.Namespace,
    specs: Sequence[WorkerSpec],
) -> dict[str, Any]:
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    return {
        "schema": SCHEMA,
        "dry_run": True,
        "datasets": list(selected),
        "dataset_parallelism": len(selected),
        "roles_sequential_within_dataset": True,
        "official_test_evaluation_launched": False,
        "gpu_mapping": {
            dataset: {
                "physical_index": GPU_INDICES[dataset],
                "uuid": GPU_UUIDS[dataset],
            }
            for dataset in selected
        },
        "systemd_commands": {
            dataset: list(systemd_command(args, dataset)) for dataset in selected
        },
        "workers": [
            asdict(spec)
            | {
                "run_directory": str(spec.run_directory),
                "command": list(spec.command),
            }
            for spec in specs
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL_MANIFEST
    )
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--resume", choices=("auto", "never", "required"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-images", type=int, default=2)
    parser.add_argument("--max-val-images", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker-sequence", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    specs = build_worker_specs(
        dataset=args.dataset,
        results_root=args.results_root,
        data_root=args.data_root,
        protocol_manifest=args.protocol_manifest,
        python=args.python,
        resume=args.resume,
        smoke=args.smoke,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
    )
    if args.worker_sequence:
        if args.dataset == "all":
            raise ValueError("worker sequence requires one dataset")
        raise SystemExit(execute_dataset_sequence(args.dataset, specs))
    if not args.execute:
        print(json.dumps(dry_run_payload(args, specs), indent=2, sort_keys=True))
        return
    verify_gpu_bindings()
    selected = DATASETS if args.dataset == "all" else (args.dataset,)
    processes = [
        subprocess.Popen(systemd_command(args, dataset), cwd=REPO_ROOT)
        for dataset in selected
    ]
    return_codes = [process.wait() for process in processes]
    raise SystemExit(next((code for code in return_codes if code), 0))


if __name__ == "__main__":
    main()
