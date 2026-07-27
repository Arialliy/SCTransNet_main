#!/usr/bin/env python3
"""Run the eight preregistered V7-DCH closed-interval sweeps once.

The candidate root, variants, seeds, checkpoint roles, Fa budgets, and
physical postprocess GPUs are fixed.  Existing sweep files are validated and
never replaced.  No subprocess is started until all four training runs are
complete with the native 17-field schema.  Each CUDA job replays its run's
training GPU and records the exact deterministic inference configuration.
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

from experiments import (  # noqa: E402
    validate_tpd_clean_v7_dch_formal800_completion as completion,
)


EVALUATOR = completion.DEFAULT_EVALUATOR
POSTPROCESS_GPUS = {
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXECUTION_PROVENANCE_KEY = "dch_formal_execution_provenance"
DETERMINISM_SETTINGS = {
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
    "deterministic_algorithms": True,
    "float32_matmul_precision": "highest",
}


def _execution_provenance(
    device: str,
    physical_gpu: str | None,
) -> dict[str, Any]:
    gpu_uuid = POSTPROCESS_GPUS.get(str(physical_gpu))
    return {
        "schema": "sctransnet_tpd_clean_v7_dch_sweep_execution_v1",
        "device": device,
        "logical_device": device,
        "physical_gpu_index": (
            int(physical_gpu) if physical_gpu is not None else None
        ),
        "physical_gpu_uuid": gpu_uuid,
        "cuda_visible_devices": gpu_uuid if device == "cuda:0" else None,
        "cublas_workspace_config": (
            CUBLAS_WORKSPACE_CONFIG if device == "cuda:0" else None
        ),
        "determinism": dict(DETERMINISM_SETTINGS),
        "determinism_applied_before_model_compute": True,
        "determinism_owner": (
            "experiments.evaluate_tpd_clean_v7_dch_pd_fa."
            "configure_dch_inference"
        ),
        "evaluator": str(EVALUATOR.resolve()),
        "evaluator_sha256": completion.sha256_file(EVALUATOR),
    }


def _load_sweep_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise completion.IncompleteArtifact(
            f"invalid DCH formal sweep JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise completion.IncompleteArtifact(
            f"DCH formal sweep is not a JSON object: {path}"
        )
    return payload


def _write_execution_provenance(
    path: Path,
    *,
    device: str,
    physical_gpu: str | None,
) -> None:
    payload = _load_sweep_payload(path)
    if EXECUTION_PROVENANCE_KEY in payload:
        raise completion.IncompleteArtifact(
            f"new DCH sweep already contains execution provenance: {path}"
        )
    audit = payload.get("audit")
    if device == "cuda:0" and (
        not isinstance(audit, dict)
        or audit.get("cuda_visible_devices")
        != POSTPROCESS_GPUS[str(physical_gpu)]
    ):
        raise completion.IncompleteArtifact(
            "evaluator did not record the requested physical GPU UUID"
        )
    payload[EXECUTION_PROVENANCE_KEY] = _execution_provenance(
        device, physical_gpu
    )
    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(path.name + ".provenance.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"provenance temporary already exists: {temporary}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_execution_provenance(
    path: Path,
    *,
    device: str,
    physical_gpu: str | None,
) -> dict[str, Any]:
    payload = _load_sweep_payload(path)
    observed = payload.get(EXECUTION_PROVENANCE_KEY)
    expected = _execution_provenance(device, physical_gpu)
    if observed != expected:
        raise completion.IncompleteArtifact(
            f"DCH sweep execution provenance differs: {path}"
        )
    audit = payload.get("audit")
    if device == "cuda:0" and (
        not isinstance(audit, dict)
        or audit.get("cuda_visible_devices")
        != POSTPROCESS_GPUS[str(physical_gpu)]
    ):
        raise completion.IncompleteArtifact(
            f"DCH sweep evaluator GPU provenance differs: {path}"
        )
    return dict(observed)


def sweep_jobs(device: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in completion.SEEDS:
        for variant in completion.VARIANTS:
            run_dir = (
                completion.DEFAULT_CANDIDATE_ROOT
                / completion.DATASET
                / variant
                / f"seed_{seed}_{completion.RUN_TAG}"
            )
            physical_gpu = None
            gpu_uuid = None
            if device == "cuda:0":
                assigned_index, assigned_uuid = completion.GPU_ASSIGNMENTS[
                    (variant, seed)
                ]
                physical_gpu = str(assigned_index)
                gpu_uuid = assigned_uuid
            for role_name, spec in completion.ROLE_SPECS.items():
                output = run_dir / str(spec["sweep"])
                command = [
                    sys.executable,
                    str(EVALUATOR),
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    str(spec["checkpoint"]),
                    "--device",
                    device,
                    "--expected-epochs",
                    str(completion.EXPECTED_EPOCHS),
                    "--fa-budgets",
                    *[str(float(key)) for key in completion.BUDGET_KEYS],
                ]
                jobs.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "role": role_name,
                        "run_directory": str(run_dir),
                        "output": str(output),
                        "output_exists": output.exists() or output.is_symlink(),
                        "physical_gpu": physical_gpu,
                        "physical_gpu_uuid": gpu_uuid,
                        "command": command,
                    }
                )
    return jobs


def preflight(
    device: str,
    physical_gpu: str | None = None,
) -> dict[str, Any]:
    readiness = completion.inspect_training_readiness()
    return {
        "schema": "sctransnet_tpd_clean_v7_dch_sweep_preflight_v1",
        "mode": "preflight",
        "candidate_family": "tpd_clean_v7_dch",
        "formal_matrix_complete": readiness["formal_matrix_complete"],
        "gate_evaluated": False,
        "engineering_gate_passed": None,
        "device": device,
        "physical_gpu_mode": (
            "auto_training_gpu_replay" if device == "cuda:0" else "cpu"
        ),
        "legacy_physical_gpu_argument": physical_gpu,
        "gpu_assignments": {
            f"{variant}/seed_{seed}": {
                "physical_gpu_index": completion.GPU_ASSIGNMENTS[
                    (variant, seed)
                ][0],
                "physical_gpu_uuid": completion.GPU_ASSIGNMENTS[
                    (variant, seed)
                ][1],
            }
            for variant in completion.VARIANTS
            for seed in completion.SEEDS
        }
        if device == "cuda:0"
        else {},
        "execution_provenance_contract": {
            "cublas_workspace_config": (
                CUBLAS_WORKSPACE_CONFIG if device == "cuda:0" else None
            ),
            "determinism": dict(DETERMINISM_SETTINGS),
            "per_job_training_gpu_replay": device == "cuda:0",
        },
        "candidate_root": str(
            completion.DEFAULT_CANDIDATE_ROOT.resolve()
        ),
        "training": readiness,
        "sweep_jobs": sweep_jobs(device),
        "expected_runs": 4,
        "expected_checkpoints": 12,
        "expected_sweeps": 8,
        "native_validation_field_count": len(completion.VALIDATION_FIELDS),
        "subprocesses_started": 0,
        "outputs_written": 0,
    }


def _gpu_environment(
    device: str,
    physical_gpu: str | None,
) -> dict[str, str]:
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
    if fields != [physical_gpu, "NVIDIA GeForce RTX 5090", gpu_uuid]:
        raise RuntimeError(
            f"physical GPU identity differs: expected={physical_gpu},{gpu_uuid} "
            f"actual={query!r}"
        )
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    environment["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    return environment


def run_sweeps(
    device: str,
    physical_gpu: str | None = None,
) -> dict[str, list[Path]]:
    if device == "cuda:0" and physical_gpu is not None:
        raise ValueError(
            "formal CUDA sweeps auto-replay each run's training GPU; "
            "--physical-gpu must be omitted"
        )
    readiness = completion.inspect_training_readiness()
    if readiness["formal_matrix_complete"] is not True:
        raise RuntimeError(
            "V7-DCH formal800 matrix is incomplete; only --preflight is allowed"
        )
    acceptance_lock, _ = completion.validate_acceptance_source_lock()
    evaluator_relative = str(EVALUATOR.relative_to(REPO_ROOT))
    evaluator_sha = acceptance_lock["source_sha256"].get(evaluator_relative)
    if evaluator_sha != completion.sha256_file(EVALUATOR):
        raise completion.IncompleteArtifact(
            "DCH evaluator differs from the acceptance source lock"
        )
    jobs = sweep_jobs(device)
    environments: dict[str | None, dict[str, str]] = {}
    selected_gpus = {
        str(job.get("physical_gpu"))
        for job in jobs
        if job.get("physical_gpu") is not None
    }
    if device == "cuda:0" and selected_gpus != set(POSTPROCESS_GPUS):
        raise RuntimeError("formal CUDA sweep GPU replay matrix differs")
    environment_gpus: list[str | None] = (
        sorted(selected_gpus) if selected_gpus else [None]
    )
    for selected_gpu in environment_gpus:
        environments[selected_gpu] = _gpu_environment(
            device, selected_gpu
        )
    completed: list[Path] = []
    skipped: list[Path] = []
    for job in jobs:
        job_physical_gpu = (
            str(job.get("physical_gpu"))
            if job.get("physical_gpu") is not None
            else None
        )
        expected_assignment = (
            completion.GPU_ASSIGNMENTS[(job["variant"], job["seed"])]
            if device == "cuda:0"
            else None
        )
        if device == "cuda:0" and (
            expected_assignment is None
            or job_physical_gpu != str(expected_assignment[0])
            or job.get("physical_gpu_uuid") != expected_assignment[1]
        ):
            raise RuntimeError("sweep job training-GPU replay identity differs")
        output = Path(job["output"])
        if output.is_symlink() or (output.exists() and not output.is_file()):
            raise FileExistsError(
                f"existing formal sweep is not a regular file: {output}"
            )
        if output.is_file():
            _validate_execution_provenance(
                output,
                device=device,
                physical_gpu=job_physical_gpu,
            )
            completion.validate_existing_sweep(
                Path(job["run_directory"]),
                variant=job["variant"],
                seed=job["seed"],
                role_name=job["role"],
                evaluator_path=EVALUATOR,
            )
            skipped.append(output)
            continue
        subprocess.run(
            job["command"],
            cwd=REPO_ROOT,
            check=True,
            env=environments[job_physical_gpu],
        )
        if not output.is_file() or output.is_symlink():
            raise RuntimeError(f"evaluator did not create a regular sweep: {output}")
        _write_execution_provenance(
            output,
            device=device,
            physical_gpu=job_physical_gpu,
        )
        _validate_execution_provenance(
            output,
            device=device,
            physical_gpu=job_physical_gpu,
        )
        completion.validate_existing_sweep(
            Path(job["run_directory"]),
            variant=job["variant"],
            seed=job["seed"],
            role_name=job["role"],
            evaluator_path=EVALUATOR,
        )
        completed.append(output)
    return {"created": completed, "validated_existing": skipped}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed eight-sweep V7-DCH formal800 matrix"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--physical-gpu", choices=tuple(POSTPROCESS_GPUS))
    args = parser.parse_args(argv)
    if args.run and args.device == "cuda:0" and args.physical_gpu is not None:
        parser.error(
            "formal CUDA run auto-replays GPU 2/3; omit --physical-gpu"
        )
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
    result = run_sweeps(args.device, args.physical_gpu)
    print(
        f"COMPLETE created={len(result['created'])} "
        f"validated_existing={len(result['validated_existing'])} "
        f"candidate_root={completion.DEFAULT_CANDIDATE_ROOT}",
        flush=True,
    )


__all__ = [
    "CUBLAS_WORKSPACE_CONFIG",
    "DETERMINISM_SETTINGS",
    "EVALUATOR",
    "EXECUTION_PROVENANCE_KEY",
    "POSTPROCESS_GPUS",
    "main",
    "parse_args",
    "preflight",
    "run_sweeps",
    "sweep_jobs",
]


if __name__ == "__main__":
    main()
