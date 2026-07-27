#!/usr/bin/env python3
"""Run the fixed eight V6 sweeps through the post-freeze audit adapter."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import accept_tpd_clean_v6_formal800_results as old_acceptance  # noqa: E402
from experiments import evaluate_tpd_clean_v6_pd_fa_checkpoint_compat as compat  # noqa: E402
from experiments import run_tpd_clean_v6_formal800_sweeps as frozen_runner  # noqa: E402
from experiments import summarize_tpd_clean_v6_formal800 as summary  # noqa: E402
from experiments import validate_tpd_clean_v6_checkpoint_compatibility as validator  # noqa: E402
from experiments import validate_tpd_clean_v6_strict_sweeps as strict  # noqa: E402


EVALUATOR = Path(compat.__file__).resolve()
POSTPROCESS_GPUS = frozen_runner.POSTPROCESS_GPUS
POSTPROCESS_LOCK = summary.DEFAULT_CANDIDATE_ROOT / ".postprocess.lock"
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
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
                jobs.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "role": role_name,
                        "run_directory": str(run_dir),
                        "output": str(output),
                        "output_exists": output.exists() or output.is_symlink(),
                        "command": [
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
                            *[
                                str(float(key))
                                for key in summary.BUDGET_KEYS
                            ],
                        ],
                    }
                )
    return jobs


def preflight(
    device: str, physical_gpu: str | None = None
) -> dict[str, Any]:
    _, lock_sha = compat.validate_compatibility_source_lock()
    readiness = summary.inspect_training_readiness()
    jobs = sweep_jobs(device)
    for job in jobs:
        job["training_physical_gpu"] = _training_physical_gpu(job)
        job["training_gpu_uuid"] = POSTPROCESS_GPUS[
            job["training_physical_gpu"]
        ]
    return {
        "schema": "sctransnet_tpd_clean_v6_compat_sweep_preflight_v1",
        "mode": "preflight",
        "formal_matrix_complete": readiness["formal_matrix_complete"],
        "device": device,
        "physical_gpu": physical_gpu,
        "gpu_uuid": POSTPROCESS_GPUS.get(str(physical_gpu)),
        "candidate_root": str(summary.DEFAULT_CANDIDATE_ROOT.resolve()),
        "compatibility_source_lock_sha256": lock_sha,
        "training": readiness,
        "sweep_jobs": jobs,
        "subprocesses_started": 0,
        "outputs_written": 0,
    }


def _training_physical_gpu(job: Mapping[str, Any]) -> str:
    protocol_path = Path(str(job["run_directory"])) / "protocol.json"
    if not protocol_path.is_file() or protocol_path.is_symlink():
        raise FileNotFoundError(f"training protocol is not regular: {protocol_path}")
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    environment = (
        payload.get("run_identity", {})
        .get("training_contract", {})
        .get("environment", {})
    )
    gpu_uuid = environment.get("device_uuid")
    matches = [
        physical
        for physical, expected_uuid in POSTPROCESS_GPUS.items()
        if expected_uuid == gpu_uuid
    ]
    if len(matches) != 1:
        raise ValueError(f"training GPU is not physical GPU 2/3: {gpu_uuid!r}")
    if environment.get("cuda_visible_devices") != gpu_uuid:
        raise ValueError("training visible GPU differs from training device UUID")
    return matches[0]


def _job_environment(
    device: str,
    physical_gpu: str | None,
    job: Mapping[str, Any],
) -> dict[str, str]:
    training_gpu = _training_physical_gpu(job)
    selected_gpu = training_gpu if physical_gpu == "training" else physical_gpu
    if device == "cuda:0" and selected_gpu != training_gpu:
        raise RuntimeError(
            f"formal sweep must replay training GPU {training_gpu}, "
            f"not physical GPU {selected_gpu}"
        )
    environment = frozen_runner._gpu_environment(device, selected_gpu)
    if device == "cuda:0":
        environment["PYTHONHASHSEED"] = str(int(job["seed"]))
        environment["CUBLAS_WORKSPACE_CONFIG"] = (
            compat.FORMAL_CUBLAS_WORKSPACE_CONFIG
        )
        environment.update(THREAD_ENVIRONMENT)
    return environment


def _validate_one(
    job: dict[str, Any],
    evaluator_sha256: str,
) -> dict[str, Any]:
    run_dir = Path(job["run_directory"])
    summary.validate_existing_sweep(
        run_dir,
        variant=job["variant"],
        seed=job["seed"],
        role_name=job["role"],
        evaluator_sha256=evaluator_sha256,
    )
    with Path(job["output"]).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    strict.validate_sweep_payload(
        payload,
        f"{job['variant']}/seed={job['seed']}/{job['role']}",
    )
    return validator.validate_compatibility_sweep(
        Path(job["output"]),
        run_dir=run_dir,
        variant=job["variant"],
        seed=job["seed"],
        role_name=job["role"],
    )


@contextmanager
def _exclusive_postprocess_lock(
    path: Path = POSTPROCESS_LOCK,
) -> Iterator[None]:
    """Share the exact output lock used by the frozen finalizer."""

    path = Path(path).absolute()
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise FileExistsError(f"postprocess lock is not a regular file: {path}")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise NotADirectoryError(path.parent)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "formal postprocess lock is already held"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _run_sweeps_locked(
    device: str, physical_gpu: str | None = None
) -> dict[str, list[Path]]:
    training_lock, _ = summary._validate_current_training_contract()
    summary.validate_postprocess_source_lock()
    old_acceptance.validate_supplemental_source_lock()
    compat.validate_compatibility_source_lock()
    evaluator_sha = training_lock["source_sha256"][
        "experiments/evaluate_tpd_clean_v6_pd_fa.py"
    ]
    created: list[Path] = []
    validated_existing: list[Path] = []
    for job in sweep_jobs(device):
        output = Path(job["output"])
        if output.is_symlink() or (output.exists() and not output.is_file()):
            raise FileExistsError(
                f"existing formal sweep is not a regular file: {output}"
            )
        if output.is_file():
            _validate_one(job, evaluator_sha)
            validated_existing.append(output)
            continue
        subprocess.run(
            job["command"],
            cwd=REPO_ROOT,
            check=True,
            env=_job_environment(device, physical_gpu, job),
        )
        if not output.is_file() or output.is_symlink():
            raise RuntimeError(
                f"compatibility evaluator did not create a regular sweep: {output}"
            )
        _validate_one(job, evaluator_sha)
        created.append(output)
    return {
        "created": created,
        "validated_existing": validated_existing,
    }


def run_sweeps(
    device: str, physical_gpu: str | None = None
) -> dict[str, list[Path]]:
    readiness = summary.inspect_training_readiness()
    if readiness["formal_matrix_complete"] is not True:
        raise RuntimeError(
            "V6 formal800 matrix is incomplete; only --preflight is allowed"
        )
    with _exclusive_postprocess_lock():
        return _run_sweeps_locked(device, physical_gpu)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed eight V6 sweeps through the audit adapter"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument(
        "--physical-gpu",
        choices=("training", *tuple(POSTPROCESS_GPUS)),
    )
    args = parser.parse_args(argv)
    if args.run and args.device != "cuda:0":
        parser.error("formal compatibility sweep run requires --device cuda:0")
    if args.run and args.device == "cuda:0" and args.physical_gpu is None:
        parser.error("CUDA run requires --physical-gpu training, 2, or 3")
    if args.device == "cpu" and args.physical_gpu is not None:
        parser.error("--physical-gpu is only valid for cuda:0")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.preflight:
        result = preflight(args.device, args.physical_gpu)
        print(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2),
            flush=True,
        )
        return
    result = run_sweeps(args.device, args.physical_gpu)
    print(
        f"COMPLETE created={len(result['created'])} "
        f"validated_existing={len(result['validated_existing'])} "
        f"candidate_root={summary.DEFAULT_CANDIDATE_ROOT}",
        flush=True,
    )


__all__ = [
    "EVALUATOR",
    "POSTPROCESS_GPUS",
    "POSTPROCESS_LOCK",
    "THREAD_ENVIRONMENT",
    "_job_environment",
    "_training_physical_gpu",
    "main",
    "parse_args",
    "preflight",
    "run_sweeps",
    "sweep_jobs",
]


if __name__ == "__main__":
    main()
