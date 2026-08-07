#!/usr/bin/env python3
"""Dry-run-first systemd user launcher for NUAA PBDR-V3 Stage 1.

All selected role/recipe workers execute sequentially inside one persistent
user service because GPU0 is the sole authorized device.  The default action
only prints the complete plan; ``--execute`` is the only mutating action.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sctransnet_nuaa_pbdr_v3_stage1_launcher_v1/v1"
TRAINER_SCHEMA = "sctransnet_nuaa_pbdr_v3_stage1_v1/v1"
# Use the same environment that produced the PBDR-V2 artifacts and currently
# runs the sibling SCTransNet services.  A separate pytest-capable environment
# may be used for unit tests, but it is not the formal training interpreter.
PYTHON = Path("/home/ly/SCTransNet/.venv/bin/python")
TRAINER = REPO_ROOT / "experiments/train_nuaa_pbdr_v3_stage1_v1.py"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results/nuaa_pbdr_v3_stage1_v1"
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_PROTOCOL_MANIFEST = (
    REPO_ROOT
    / "results/three_dataset_v2/manifests/three_dataset_v2_protocol.json"
)
GPU_INDEX = "0"
GPU_UUID = "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70"
PARENT_ROLES = ("best_miou", "best_pd")
RECIPES = ("core", "constrained")
RESUME_MODES = ("never", "required", "auto")
UNIT_PREFIX = "sctransnet-nuaa-pbdr-v3-stage1-v1"
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
    parent_role: str
    recipe: str
    resume: str
    run_directory: Path
    command: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.parent_role}__{self.recipe}"


def run_directory(
    results_root: Path,
    parent_role: str,
    recipe: str,
    *,
    smoke: bool = False,
) -> Path:
    if parent_role not in PARENT_ROLES or recipe not in RECIPES:
        raise ValueError("unsupported role/recipe")
    return (
        Path(results_root).resolve()
        / ("smoke" if smoke else "formal")
        / parent_role
        / recipe
    )


def worker_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    environment.update(CPU_ENVIRONMENT)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": GPU_UUID,
            "PYTHONUNBUFFERED": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    return environment


def _selected(value: str, choices: tuple[str, ...]) -> tuple[str, ...]:
    if value == "all":
        return choices
    if value not in choices:
        raise ValueError(f"unsupported selection {value!r}")
    return (value,)


def build_worker_specs(
    *,
    parent_role: str = "all",
    recipe: str = "all",
    resume: str = "auto",
    results_root: Path = DEFAULT_RESULTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    protocol_manifest: Path = DEFAULT_PROTOCOL_MANIFEST,
    python: Path = PYTHON,
    smoke: bool = False,
    max_train_images: int = 2,
    max_val_images: int = 1,
) -> tuple[WorkerSpec, ...]:
    if resume not in RESUME_MODES:
        raise ValueError(f"resume must be one of {RESUME_MODES}")
    specs: list[WorkerSpec] = []
    for selected_role in _selected(parent_role, PARENT_ROLES):
        for selected_recipe in _selected(recipe, RECIPES):
            command = [
                str(Path(python).absolute()),
                str(TRAINER),
                "--parent-role",
                selected_role,
                "--recipe",
                selected_recipe,
                "--data-root",
                str(Path(data_root).resolve()),
                "--protocol-manifest",
                str(Path(protocol_manifest).resolve()),
                "--results-root",
                str(Path(results_root).resolve()),
                "--device",
                "cuda:0",
                "--expected-gpu-uuid",
                GPU_UUID,
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
                    parent_role=selected_role,
                    recipe=selected_recipe,
                    resume=resume,
                    run_directory=run_directory(
                        results_root,
                        selected_role,
                        selected_recipe,
                        smoke=smoke,
                    ),
                    command=tuple(command),
                )
            )
    if len({spec.run_directory for spec in specs}) != len(specs):
        raise RuntimeError("Stage-1 workers do not own unique run directories")
    return tuple(specs)


def verify_gpu0_uuid() -> dict[str, str]:
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
        if not line.strip():
            continue
        index, uuid = (part.strip() for part in line.split(",", 1))
        observed[index] = uuid
    if observed.get(GPU_INDEX) != GPU_UUID:
        raise RuntimeError(
            f"physical GPU0 UUID differs: {observed.get(GPU_INDEX)!r}"
        )
    return observed


def _sequence_arguments(args: argparse.Namespace) -> tuple[str, ...]:
    command = [
        str(Path(args.python).absolute()),
        str(Path(__file__).resolve()),
        "--worker-sequence",
        "--parent-role",
        args.parent_role,
        "--recipe",
        args.recipe,
        "--resume",
        args.resume,
        "--results-root",
        str(args.results_root.resolve()),
        "--data-root",
        str(args.data_root.resolve()),
        "--protocol-manifest",
        str(args.protocol_manifest.resolve()),
        "--python",
        str(args.python.absolute()),
    ]
    if args.smoke:
        command.extend(("--smoke", "--max-train-images", str(args.max_train_images), "--max-val-images", str(args.max_val_images)))
    return tuple(command)


def systemd_command(args: argparse.Namespace) -> tuple[str, ...]:
    scope = f"{args.parent_role}-{args.recipe}".replace("_", "-")
    unit = f"{UNIT_PREFIX}-{scope}"
    environment = worker_environment({})
    command = [
        "systemd-run",
        "--user",
        "--collect",
        "--unit",
        unit,
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
    command.extend(_sequence_arguments(args))
    return tuple(command)


def dry_run_payload(
    args: argparse.Namespace,
    specs: Sequence[WorkerSpec],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "dry_run": True,
        "execution_backend": "systemd_user_service",
        "dataset": "NUAA-SIRST",
        "training_seed": 42,
        "physical_gpu_index": GPU_INDEX,
        "gpu_uuid": GPU_UUID,
        "single_gpu": True,
        "sequential_workers": True,
        "conditional_recipe_policy": (
            "run constrained only when the same-role core internal gate fails"
        ),
        "official_test_evaluation_launched": False,
        "systemd_command": list(systemd_command(args)),
        "workers": [
            asdict(spec)
            | {
                "run_directory": str(spec.run_directory),
                "command": list(spec.command),
            }
            for spec in specs
        ],
    }


def read_internal_gate_result(spec: WorkerSpec) -> bool:
    summary_path = spec.run_directory / "summary.json"
    if summary_path.is_symlink() or not summary_path.is_file():
        raise RuntimeError(f"worker summary is missing: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read worker summary: {summary_path}") from error
    if not isinstance(summary, Mapping):
        raise RuntimeError(f"worker summary is not a JSON object: {summary_path}")
    expected = {
        "schema": TRAINER_SCHEMA,
        "status": "complete",
        "parent_role": spec.parent_role,
        "recipe": spec.recipe,
        "official_test_accessed": False,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise RuntimeError(f"worker summary {field} differs: {summary_path}")
    passed = summary.get("internal_gate_passed")
    if not isinstance(passed, bool):
        raise RuntimeError(f"worker summary gate is not boolean: {summary_path}")
    return passed


def execute_sequence(
    specs: Sequence[WorkerSpec],
    *,
    environment: Mapping[str, str] | None = None,
) -> int:
    child_environment = worker_environment(environment)
    core_passed_roles: set[str] = set()
    for spec in specs:
        if spec.recipe == "constrained" and spec.parent_role in core_passed_roles:
            print(
                json.dumps(
                    {
                        "worker": spec.key,
                        "status": "skipped",
                        "reason": "same_role_core_internal_gate_passed",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        completed = subprocess.run(
            spec.command,
            cwd=REPO_ROOT,
            env=child_environment,
            check=False,
        )
        if completed.returncode:
            return int(completed.returncode)
        gate_passed = read_internal_gate_result(spec)
        print(
            json.dumps(
                {
                    "worker": spec.key,
                    "status": "complete",
                    "internal_gate_passed": gate_passed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if spec.recipe == "core" and gate_passed:
            core_passed_roles.add(spec.parent_role)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-role", choices=("all", *PARENT_ROLES), default="all"
    )
    parser.add_argument("--recipe", choices=("all", *RECIPES), default="all")
    parser.add_argument("--resume", choices=RESUME_MODES, default="auto")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--protocol-manifest", type=Path, default=DEFAULT_PROTOCOL_MANIFEST
    )
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-images", type=int, default=2)
    parser.add_argument("--max-val-images", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker-sequence", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    specs = build_worker_specs(
        parent_role=args.parent_role,
        recipe=args.recipe,
        resume=args.resume,
        results_root=args.results_root,
        data_root=args.data_root,
        protocol_manifest=args.protocol_manifest,
        python=args.python,
        smoke=args.smoke,
        max_train_images=args.max_train_images,
        max_val_images=args.max_val_images,
    )
    if args.worker_sequence:
        raise SystemExit(execute_sequence(specs))
    if not args.execute:
        print(
            json.dumps(
                dry_run_payload(args, specs),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    verify_gpu0_uuid()
    completed = subprocess.run(systemd_command(args), cwd=REPO_ROOT, check=False)
    raise SystemExit(int(completed.returncode))


if __name__ == "__main__":
    main()
