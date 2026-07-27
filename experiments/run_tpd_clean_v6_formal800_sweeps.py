#!/usr/bin/env python3
"""Run the eight preregistered V6 closed-interval sweeps once.

The candidate root and checkpoint-role mapping are fixed.  Existing sweep
files are never replaced.  If training is incomplete, only ``--preflight`` is
accepted and no subprocess or output file is created.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import summarize_tpd_clean_v6_formal800 as summary  # noqa: E402


EVALUATOR = REPO_ROOT / "experiments/evaluate_tpd_clean_v6_pd_fa.py"
POSTPROCESS_GPUS = {
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}


def sweep_jobs(device: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in summary.SEEDS:
        for variant in summary.VARIANTS:
            run_dir = (
                summary.DEFAULT_CANDIDATE_ROOT
                / summary.DATASET
                / variant
                / f"seed_{seed}_{summary.RUN_TAG}"
            )
            for role_name, spec in summary.ROLE_SPECS.items():
                output = run_dir / spec["sweep"]
                command = [
                    sys.executable,
                    str(EVALUATOR),
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    spec["checkpoint"],
                    "--device",
                    device,
                    "--expected-epochs",
                    str(summary.EXPECTED_EPOCHS),
                    "--fa-budgets",
                    *[str(float(key)) for key in summary.BUDGET_KEYS],
                ]
                jobs.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "role": role_name,
                        "run_directory": str(run_dir),
                        "output": str(output),
                        "output_exists": output.exists() or output.is_symlink(),
                        "command": command,
                    }
                )
    return jobs


def preflight(device: str, physical_gpu: str | None = None) -> dict[str, Any]:
    readiness = summary.inspect_training_readiness()
    jobs = sweep_jobs(device)
    return {
        "schema": "sctransnet_tpd_clean_v6_sweep_preflight_v1",
        "mode": "preflight",
        "formal_matrix_complete": readiness["formal_matrix_complete"],
        "gate_evaluated": False,
        "engineering_gate_passed": None,
        "device": device,
        "physical_gpu": physical_gpu,
        "gpu_uuid": POSTPROCESS_GPUS.get(str(physical_gpu)),
        "candidate_root": str(summary.DEFAULT_CANDIDATE_ROOT.resolve()),
        "training": readiness,
        "sweep_jobs": jobs,
        "subprocesses_started": 0,
        "outputs_written": 0,
    }


def _gpu_environment(device: str, physical_gpu: str | None) -> dict[str, str]:
    environment = dict(os.environ)
    if device == "cpu":
        if physical_gpu is not None:
            raise ValueError("--physical-gpu is only valid with --device cuda:0")
        return environment
    if device != "cuda:0" or physical_gpu not in POSTPROCESS_GPUS:
        raise ValueError(
            "CUDA sweep requires --device cuda:0 and --physical-gpu 2 or 3"
        )
    gpu_uuid = POSTPROCESS_GPUS[physical_gpu]
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu_uuid,
            "--query-gpu=index,name,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fields = [field.strip() for field in query.split(",")]
    if fields != [
        physical_gpu,
        "NVIDIA GeForce RTX 5090",
        gpu_uuid,
    ]:
        raise RuntimeError(
            f"physical GPU identity differs: expected={physical_gpu},{gpu_uuid} "
            f"actual={query!r}"
        )
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    return environment


def run_sweeps(
    device: str, physical_gpu: str | None = None
) -> dict[str, list[Path]]:
    readiness = summary.inspect_training_readiness()
    if readiness["formal_matrix_complete"] is not True:
        raise RuntimeError(
            "V6 formal800 matrix is incomplete; only --preflight is allowed"
        )
    training_lock, _ = summary._validate_current_training_contract()
    summary.validate_postprocess_source_lock()
    evaluator_sha = training_lock["source_sha256"][
        "experiments/evaluate_tpd_clean_v6_pd_fa.py"
    ]
    environment = _gpu_environment(device, physical_gpu)
    completed: list[Path] = []
    skipped: list[Path] = []
    for job in sweep_jobs(device):
        output = Path(job["output"])
        if output.is_symlink() or (output.exists() and not output.is_file()):
            raise FileExistsError(
                f"existing formal sweep is not a regular file: {output}"
            )
        if output.is_file():
            summary.validate_existing_sweep(
                Path(job["run_directory"]),
                variant=job["variant"],
                seed=job["seed"],
                role_name=job["role"],
                evaluator_sha256=evaluator_sha,
            )
            skipped.append(output)
            continue
        subprocess.run(
            job["command"],
            cwd=REPO_ROOT,
            check=True,
            env=environment,
        )
        if not output.is_file() or output.is_symlink():
            raise RuntimeError(f"evaluator did not create a regular sweep: {output}")
        summary.validate_existing_sweep(
            Path(job["run_directory"]),
            variant=job["variant"],
            seed=job["seed"],
            role_name=job["role"],
            evaluator_sha256=evaluator_sha,
        )
        completed.append(output)
    return {"created": completed, "validated_existing": skipped}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed eight-sweep V6 formal800 matrix"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--physical-gpu", choices=tuple(POSTPROCESS_GPUS))
    args = parser.parse_args(argv)
    if not args.device.strip():
        parser.error("--device must be non-empty")
    if args.run and args.device == "cuda:0" and args.physical_gpu is None:
        parser.error("CUDA run requires --physical-gpu 2 or 3")
    if args.device == "cpu" and args.physical_gpu is not None:
        parser.error("--physical-gpu is only valid for cuda:0")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.preflight:
        print(
            json.dumps(
                preflight(args.device, args.physical_gpu),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            flush=True,
        )
        return
    completed = run_sweeps(args.device, args.physical_gpu)
    print(
        f"COMPLETE created={len(completed['created'])} "
        f"validated_existing={len(completed['validated_existing'])} "
        f"candidate_root={summary.DEFAULT_CANDIDATE_ROOT}",
        flush=True,
    )


__all__ = [
    "EVALUATOR",
    "POSTPROCESS_GPUS",
    "main",
    "parse_args",
    "preflight",
    "run_sweeps",
    "sweep_jobs",
]


if __name__ == "__main__":
    main()
