#!/usr/bin/env python3
"""Plan or launch the three independent NER-V5-PER seed-42 runs.

Dry-run is the default and has no side effects.  The three dataset jobs are
bound one-to-one to physical GPUs 0, 1, and 2.  Epoch 200 is a durable prefix
of the same 1000-epoch run; the resume phase continues that run at epoch 201.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
TRAINER = REPO_ROOT / "experiments/train_three_dataset_ner_v5_per_tss_off_seed42.py"
EVALUATOR = REPO_ROOT / "experiments/evaluate_three_dataset_ner_v5_per.py"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results/three_dataset_ner_v5_per_seed42_v1"
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "results"
    / "three_dataset_v2"
    / "manifests"
    / "three_dataset_v2_protocol.json"
)
DATASET_GPU = {
    "NUAA-SIRST": "0",
    "NUDT-SIRST": "1",
    "IRSTD-1K": "2",
}
GPU_UUIDS = {
    "0": "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    "1": "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
CHECKPOINT_ROLES = ("best_miou", "best_pd")
SCHEMA = "sctransnet_three_dataset_ner_v5_per_launch_v1"


@dataclass(frozen=True)
class Job:
    phase: str
    dataset: str
    physical_gpu: str
    expected_gpu_uuid: str
    argv: tuple[str, ...]


def _run_dir(results_root: Path, dataset: str) -> Path:
    return (
        results_root.resolve()
        / "runs"
        / dataset
        / "ner_v5_per_tss_off"
        / "seed_42"
    )


def build_jobs(
    phase: str,
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_manifest: Path = DEFAULT_MANIFEST,
    python: Path = PYTHON,
) -> list[Job]:
    if phase not in {"pilot", "resume", "evaluate"}:
        raise ValueError("phase must be pilot, resume, or evaluate")
    jobs: list[Job] = []
    for dataset, gpu in DATASET_GPU.items():
        common = (
            "--dataset",
            dataset,
            "--data-root",
            str(data_root.resolve()),
            "--results-root",
            str(results_root.resolve()),
            "--protocol-manifest",
            str(protocol_manifest.resolve()),
        )
        if phase in {"pilot", "resume"}:
            argv = (
                str(python),
                str(TRAINER),
                *common,
                "--method",
                "final",
                "--tss-weight",
                "0",
                "--seed",
                "42",
                "--epochs",
                "1000",
                "--device",
                "cuda:0",
                "--physical-gpu-index",
                gpu,
                "--expected-gpu-uuid",
                GPU_UUIDS[gpu],
                "--resume",
                "never" if phase == "pilot" else "required",
                *(() if phase == "resume" else ("--pause-after-epoch", "200")),
            )
            jobs.append(Job(phase, dataset, gpu, GPU_UUIDS[gpu], argv))
            continue
        for role in CHECKPOINT_ROLES:
            argv = (
                str(python),
                str(EVALUATOR),
                "--dataset",
                dataset,
                "--checkpoint-role",
                role,
                "--run-dir",
                str(_run_dir(results_root, dataset)),
                "--dataset-root",
                str(data_root.resolve()),
                "--data-protocol-manifest",
                str(protocol_manifest.resolve()),
                "--device",
                "cuda:0",
            )
            jobs.append(Job(f"evaluate:{role}", dataset, gpu, GPU_UUIDS[gpu], argv))
    return jobs


def dry_run_payload(jobs: Sequence[Job]) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "dry_run": True,
        "training_seed": 42,
        "planned_total_epochs": 1000,
        "durable_pause_epoch": 200,
        "pilot_is_prefix_of_same_run": True,
        "fresh_scratch": True,
        "dataset_gpu_mapping": dict(DATASET_GPU),
        "gpu3_role": "unassigned_evaluation_or_recovery_capacity",
        "jobs": [asdict(job) | {"argv": list(job.argv)} for job in jobs],
    }


def execute_jobs(jobs: Sequence[Job]) -> int:
    # Training jobs use three distinct GPUs and run in one wave.  Evaluation
    # has two checkpoint roles per GPU, so execute it in role waves to avoid
    # placing best_miou and best_pd inference on the same GPU concurrently.
    phases = list(dict.fromkeys(job.phase for job in jobs))
    waves = (
        [[job for job in jobs if job.phase == phase] for phase in phases]
        if any(job.phase.startswith("evaluate:") for job in jobs)
        else [list(jobs)]
    )
    exit_code = 0
    for wave in waves:
        processes: list[subprocess.Popen[bytes]] = []
        for job in wave:
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = job.physical_gpu
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            processes.append(subprocess.Popen(job.argv, cwd=REPO_ROOT, env=env))
        exit_code = max(exit_code, *(process.wait() for process in processes))
    return exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "resume", "evaluate"), required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--protocol-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    jobs = build_jobs(
        args.phase,
        results_root=args.results_root,
        data_root=args.data_root,
        protocol_manifest=args.protocol_manifest,
        python=args.python,
    )
    if not args.execute:
        print(json.dumps(dry_run_payload(jobs), ensure_ascii=False, sort_keys=True, indent=2))
        return
    raise SystemExit(execute_jobs(jobs))


if __name__ == "__main__":
    main()
