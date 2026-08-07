#!/usr/bin/env python3
"""Plan or execute the complete three-dataset PBDR-V4 pipeline.

One worker owns one dataset and one physical GPU for every phase.  Both role
branches are completed before the worker reaches the single dataset-level
official claim.  Dry-run is the default, and this module never imports or
constructs a dataset, index, sample resolver, or loader.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

from experiments import evaluate_three_dataset_pbdr_v4_v1 as evaluator
from experiments import pbdr_v4_training_core as training_core


SCHEMA = "sctransnet_three_dataset_pbdr_v4_launcher_v1/v1"
STATUS_SCHEMA = "sctransnet_three_dataset_pbdr_v4_phase_status_v1/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/ly/SCTransNet/.venv/bin/python")
EVALUATOR_PATH = REPO_ROOT / "experiments/evaluate_three_dataset_pbdr_v4_v1.py"
LAUNCHER_PATH = Path(__file__).resolve()
PREPARE_PATH = REPO_ROOT / "experiments/prepare_pbdr_v4_internal_artifacts.py"
SWEEP_PATH = REPO_ROOT / "experiments/sweep_pbdr_v3_residual_calibration.py"
TRAINER_PATH = REPO_ROOT / "experiments/train_three_dataset_pbdr_v4_v1.py"
FREEZER_PATH = REPO_ROOT / "experiments/freeze_pbdr_v4_protocol.py"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results/pbdr_v4_v1"
DEFAULT_DATA_ROOT = REPO_ROOT / "datasets"
DEFAULT_CANDIDATE_POOL_ROOT = DEFAULT_RESULTS_ROOT / "candidate_pools"
DATASETS = evaluator.DATASETS
ROLES = evaluator.ROLES
PHASE_ORDER = (
    "prepare",
    "sweep",
    "smoke",
    "stage1",
    "stage2",
    "freeze_pools",
    "joint_official",
)
ALLOWED_GPU_INDICES = ("0", "1", "3")
DATASET_GPU_LAYOUT = (
    ("NUDT-SIRST", "0"),
    ("IRSTD-1K", "1"),
    ("NUAA-SIRST", "3"),
)
GPU_ASSIGNMENTS: Mapping[str, Mapping[str, str]] = {
    "0": {
        "physical_index": "0",
        "uuid": evaluator.FORMAL_GPU_UUIDS["NUDT-SIRST"],
    },
    "1": {
        "physical_index": "1",
        "uuid": evaluator.FORMAL_GPU_UUIDS["IRSTD-1K"],
    },
    "3": {
        "physical_index": "3",
        "uuid": evaluator.FORMAL_GPU_UUIDS["NUAA-SIRST"],
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


@dataclass(frozen=True, slots=True)
class PhaseCommand:
    phase: str
    commands: tuple[tuple[str, ...], ...]
    requires: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.phase not in PHASE_ORDER:
            raise ValueError("unsupported phase")
        index = PHASE_ORDER.index(self.phase)
        expected = () if index == 0 else (PHASE_ORDER[index - 1],)
        if self.requires != expected:
            raise ValueError("phase dependency differs")
        if not self.commands or any(
            not command or any(not isinstance(item, str) or not item for item in command)
            for command in self.commands
        ):
            raise ValueError("phase command is empty")


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    dataset: str
    gpu_index: str
    expected_gpu_uuid: str
    run_directory: Path
    candidate_pool_paths: Mapping[str, Path]
    phases: tuple[PhaseCommand, ...]
    environment: Mapping[str, str]
    preflight_by_role: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        if (self.dataset, self.gpu_index) not in DATASET_GPU_LAYOUT:
            raise ValueError("dataset/GPU mapping differs from the frozen schedule")
        expected_uuid = evaluator.FORMAL_GPU_UUIDS[self.dataset]
        if self.expected_gpu_uuid != expected_uuid:
            raise ValueError("worker GPU UUID differs from the dataset authority")
        if GPU_ASSIGNMENTS[self.gpu_index]["uuid"] != expected_uuid:
            raise ValueError("physical GPU assignment differs")
        if self.environment.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
            raise ValueError("worker CUDA visibility differs")
        if tuple(self.candidate_pool_paths) != ROLES:
            raise ValueError("worker candidate-pool roles/order differ")
        if tuple(item.phase for item in self.phases) != PHASE_ORDER:
            raise ValueError("worker phase order differs")
        if self.preflight_by_role:
            if tuple(self.preflight_by_role) != ROLES:
                raise ValueError("worker preflight roles/order differ")
            for role in ROLES:
                record = self.preflight_by_role[role]
                if (
                    record.get("dataset") != self.dataset
                    or record.get("role") != role
                    or record.get("official_test_accessed") is not False
                ):
                    raise ValueError("worker preflight identity differs")

    @property
    def key(self) -> str:
        return self.dataset


def candidate_pool_path(root: Path, dataset: str, role: str) -> Path:
    if dataset not in DATASETS or role not in ROLES:
        raise ValueError("candidate-pool dataset/role differs")
    return Path(root).resolve() / dataset / role / "candidate_pool.json"


def run_directory(root: Path, dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError("run dataset differs")
    return Path(root).resolve() / "official" / dataset


def gpu_environment(
    gpu_index: str,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if gpu_index not in ALLOWED_GPU_INDICES or gpu_index not in GPU_ASSIGNMENTS:
        raise ValueError("GPU must be one of physical indices 0, 1, or 3")
    environment = dict(os.environ if base is None else base)
    environment.update(CPU_ENVIRONMENT)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": GPU_ASSIGNMENTS[gpu_index]["uuid"],
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "42",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    inherited_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + inherited_pythonpath if inherited_pythonpath else ""
    )
    return environment


def _role_root(root: Path, branch: str, dataset: str, role: str) -> Path:
    return Path(root).resolve() / branch / dataset / role


def _phase_plan(
    *,
    dataset: str,
    expected_gpu_uuid: str,
    results_root: Path,
    data_root: Path,
    source_lock_path: Path,
    split_projection_path: Path,
    candidate_pool_paths: Mapping[str, Path],
    python: Path,
    prepare_path: Path,
    sweep_path: Path,
    trainer_path: Path,
    freezer_path: Path,
    evaluator_path: Path,
    workers: int,
) -> tuple[PhaseCommand, ...]:
    executable = str(python)
    source = str(source_lock_path)
    projection = str(split_projection_path)
    data = str(Path(data_root).resolve())

    prepare_commands: list[tuple[str, ...]] = []
    sweep_commands: list[tuple[str, ...]] = []
    smoke_stage1_commands: list[tuple[str, ...]] = []
    smoke_stage2_commands: list[tuple[str, ...]] = []
    stage1_commands: list[tuple[str, ...]] = []
    stage2_commands: list[tuple[str, ...]] = []
    freeze_commands: list[tuple[str, ...]] = []
    for role in ROLES:
        internal = _role_root(results_root, "internal_artifacts", dataset, role)
        atlas = internal / "component_atlas"
        sweep_output = _role_root(results_root, "v3_calibration", dataset, role) / "sweep_result.json"
        smoke_root = _role_root(results_root, "smoke", dataset, role)
        smoke_stage1 = smoke_root / "stage1"
        smoke_stage2 = smoke_root / "stage2"
        stage1 = _role_root(results_root, "training", dataset, role) / "stage1"
        stage2 = _role_root(results_root, "training", dataset, role) / "stage2"
        stage1_selected = stage1 / "selected_candidate.pth.tar"
        prepare_commands.append(
            (
                executable,
                str(prepare_path),
                "--dataset",
                dataset,
                "--role",
                role,
                "--output-root",
                str(internal),
                "--source-lock",
                source,
                "--device",
                "cuda:0",
            )
        )
        sweep_commands.append(
            (
                executable,
                str(sweep_path),
                "--cache",
                str(internal / "internal_validation_cache"),
                "--split-projection",
                projection,
                "--output",
                str(sweep_output),
            )
        )
        common = (
            "--dataset",
            dataset,
            "--role",
            role,
            "--source-lock",
            source,
            "--split-projection",
            projection,
            "--atlas-root",
            str(atlas),
            "--data-root",
            data,
            "--device",
            "cuda:0",
            "--expected-gpu-uuid",
            expected_gpu_uuid,
        )
        smoke_stage1_commands.append(
            (
                executable,
                str(trainer_path),
                *common,
                "--stage",
                "stage1",
                "--run-dir",
                str(smoke_stage1),
                "--resume",
                "never",
                "--smoke",
                "--epochs",
                "1",
                "--eval-every",
                "1",
                "--max-train-samples",
                "1",
                "--max-val-samples",
                "1",
            )
        )
        smoke_stage2_commands.append(
            (
                executable,
                str(trainer_path),
                *common,
                "--stage",
                "stage2",
                "--stage1-checkpoint",
                str(smoke_stage1 / "selected_candidate.pth.tar"),
                "--run-dir",
                str(smoke_stage2),
                "--resume",
                "never",
                "--smoke",
                "--epochs",
                "1",
                "--eval-every",
                "1",
                "--max-train-samples",
                "1",
                "--max-val-samples",
                "1",
            )
        )
        stage1_commands.append(
            (
                executable,
                str(trainer_path),
                *common,
                "--stage",
                "stage1",
                "--run-dir",
                str(stage1),
                "--resume",
                "auto",
                "--epochs",
                "150",
                "--eval-every",
                "5",
            )
        )
        stage2_commands.append(
            (
                executable,
                str(trainer_path),
                *common,
                "--stage",
                "stage2",
                "--stage1-checkpoint",
                str(stage1_selected),
                "--run-dir",
                str(stage2),
                "--resume",
                "auto",
                "--epochs",
                "50",
                "--eval-every",
                "5",
            )
        )
        freeze_commands.append(
            (
                executable,
                str(freezer_path),
                "freeze-pool",
                "--dataset",
                dataset,
                "--role",
                role,
                "--source-lock",
                source,
                "--split-projection",
                projection,
                "--internal-cache",
                str(internal / "internal_validation_cache"),
                "--v3-sweep",
                str(sweep_output),
                "--v3-calibrated-artifact",
                str(candidate_pool_paths[role].parent / "v3_calibrated_candidate.pth.tar"),
                "--stage1-checkpoint",
                str(stage1_selected),
                "--stage2-checkpoint",
                str(stage2 / "selected_candidate.pth.tar"),
                "--output",
                str(candidate_pool_paths[role]),
            )
        )
    joint_command = (
        executable,
        str(evaluator_path),
        "--dataset",
        dataset,
        "--run-dir",
        str(run_directory(results_root, dataset)),
        "--source-lock",
        source,
        "--split-projection",
        projection,
        "--best-miou-candidate-pool",
        str(candidate_pool_paths["best_miou"]),
        "--best-pd-candidate-pool",
        str(candidate_pool_paths["best_pd"]),
        "--data-root",
        data,
        "--device",
        "cuda:0",
        "--expected-gpu-uuid",
        expected_gpu_uuid,
        "--workers",
        str(workers),
        "--operational-test-selected",
    )
    commands_by_phase = {
        "prepare": tuple(prepare_commands),
        "sweep": tuple(sweep_commands),
        "smoke": tuple(smoke_stage1_commands + smoke_stage2_commands),
        "stage1": tuple(stage1_commands),
        "stage2": tuple(stage2_commands),
        "freeze_pools": tuple(freeze_commands),
        "joint_official": (joint_command,),
    }
    return tuple(
        PhaseCommand(
            phase=phase,
            commands=commands_by_phase[phase],
            requires=() if index == 0 else (PHASE_ORDER[index - 1],),
        )
        for index, phase in enumerate(PHASE_ORDER)
    )


def _regular_file(path: Path, *, name: str) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{name} is missing or unsafe")
    return candidate.resolve(strict=True)


def _preflight_pools(
    *,
    dataset: str,
    source_lock_path: Path,
    split_projection_path: Path,
    candidate_pool_paths: Mapping[str, Path],
    check_environment: bool,
) -> dict[str, Mapping[str, object]]:
    return {
        role: dict(
            evaluator.preflight_artifacts_only(
                dataset=dataset,
                role=role,
                source_lock_path=source_lock_path,
                split_projection_path=split_projection_path,
                candidate_pool_path=_regular_file(
                    candidate_pool_paths[role],
                    name=f"{dataset}/{role} candidate pool",
                ),
                check_environment=check_environment,
            )
        )
        for role in ROLES
    }


def build_worker_specs(
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    source_lock_path: Path,
    split_projection_path: Path,
    candidate_pool_root: Path | None = None,
    python: Path = PYTHON,
    evaluator_path: Path = EVALUATOR_PATH,
    launcher_path: Path = LAUNCHER_PATH,
    prepare_path: Path = PREPARE_PATH,
    sweep_path: Path = SWEEP_PATH,
    trainer_path: Path = TRAINER_PATH,
    freezer_path: Path = FREEZER_PATH,
    workers: int = 0,
    base_environment: Mapping[str, str] | None = None,
    check_environment: bool = True,
    preflight_frozen_pools: bool = False,
) -> tuple[WorkerSpec, ...]:
    """Return three complete dataset workers without touching any data index."""

    training_core.configure_determinism(seed=training_core.TRAINING_SEED)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("workers must be a non-negative integer")
    source = _regular_file(source_lock_path, name="source lock")
    projection = _regular_file(split_projection_path, name="split projection")
    root = Path(results_root).resolve()
    pool_root = (
        root / "candidate_pools"
        if candidate_pool_root is None
        else Path(candidate_pool_root).resolve()
    )
    specs: list[WorkerSpec] = []
    for dataset, gpu_index in DATASET_GPU_LAYOUT:
        pools = {
            role: candidate_pool_path(pool_root, dataset, role) for role in ROLES
        }
        preflight = (
            _preflight_pools(
                dataset=dataset,
                source_lock_path=source,
                split_projection_path=projection,
                candidate_pool_paths=pools,
                check_environment=check_environment,
            )
            if preflight_frozen_pools
            else {}
        )
        expected_uuid = evaluator.FORMAL_GPU_UUIDS[dataset]
        specs.append(
            WorkerSpec(
                dataset=dataset,
                gpu_index=gpu_index,
                expected_gpu_uuid=expected_uuid,
                run_directory=run_directory(root, dataset),
                candidate_pool_paths=pools,
                phases=_phase_plan(
                    dataset=dataset,
                    expected_gpu_uuid=expected_uuid,
                    results_root=root,
                    data_root=data_root,
                    source_lock_path=source,
                    split_projection_path=projection,
                    candidate_pool_paths=pools,
                    python=python,
                    prepare_path=prepare_path,
                    sweep_path=sweep_path,
                    trainer_path=trainer_path,
                    freezer_path=freezer_path,
                    evaluator_path=evaluator_path,
                    workers=workers,
                ),
                environment=gpu_environment(gpu_index, base_environment),
                preflight_by_role=preflight,
            )
        )
    return tuple(specs)


def synthetic_worker_spec_for_tests(*, dataset: str, gpu_index: str) -> WorkerSpec:
    """Construct a path-only worker; no filesystem or artifact is inspected."""

    root = Path("/tmp/pbdr_v4_launcher_test").resolve()
    pools = {role: candidate_pool_path(root / "candidate_pools", dataset, role) for role in ROLES}
    expected_uuid = evaluator.FORMAL_GPU_UUIDS[dataset]
    return WorkerSpec(
        dataset=dataset,
        gpu_index=gpu_index,
        expected_gpu_uuid=expected_uuid,
        run_directory=run_directory(root, dataset),
        candidate_pool_paths=pools,
        phases=_phase_plan(
            dataset=dataset,
            expected_gpu_uuid=expected_uuid,
            results_root=root,
            data_root=Path("/tmp/data"),
            source_lock_path=Path("/tmp/source.json"),
            split_projection_path=Path("/tmp/split.json"),
            candidate_pool_paths=pools,
            python=Path("/venv/python"),
            prepare_path=PREPARE_PATH,
            sweep_path=SWEEP_PATH,
            trainer_path=TRAINER_PATH,
            freezer_path=FREEZER_PATH,
            evaluator_path=EVALUATOR_PATH,
            workers=0,
        ),
        environment=gpu_environment(gpu_index, {}),
        preflight_by_role={},
    )


def validate_phase_statuses(statuses: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(statuses, Mapping) or tuple(statuses) != PHASE_ORDER:
        raise ValueError("phase status order differs")
    ready = dict(statuses)
    if any(value not in ("pending", "complete") for value in ready.values()):
        raise ValueError("phase status must be pending or complete")
    pending_seen = False
    for phase in PHASE_ORDER:
        if ready[phase] == "pending":
            pending_seen = True
        elif pending_seen:
            raise ValueError(f"phase skip is forbidden: {phase}")
    return ready


def phase_status_manifest(
    spec: WorkerSpec,
    statuses: Mapping[str, str] | None = None,
) -> dict[str, object]:
    ready = validate_phase_statuses(
        {phase: "pending" for phase in PHASE_ORDER}
        if statuses is None
        else statuses
    )
    return {
        "schema": STATUS_SCHEMA,
        "dataset": spec.dataset,
        "gpu_index": spec.gpu_index,
        "expected_gpu_uuid": spec.expected_gpu_uuid,
        "phase_order": list(PHASE_ORDER),
        "phases": [
            {
                "phase": phase.phase,
                "status": ready[phase.phase],
                "requires": list(phase.requires),
                "commands": [list(command) for command in phase.commands],
            }
            for phase in spec.phases
        ],
        "official_test_accessed": False,
    }


def next_pending_phase(
    spec: WorkerSpec,
    statuses: Mapping[str, str],
) -> PhaseCommand:
    ready = validate_phase_statuses(statuses)
    for phase in spec.phases:
        if ready[phase.phase] == "pending":
            return phase
    raise ValueError("all phases are already complete")


def dry_run_payload(specs: Sequence[WorkerSpec]) -> dict[str, object]:
    ready = tuple(specs)
    return {
        "schema": SCHEMA,
        "dry_run": True,
        "datasets": [spec.dataset for spec in ready],
        "roles": list(ROLES),
        "phase_order": list(PHASE_ORDER),
        "allowed_gpu_indices": list(ALLOWED_GPU_INDICES),
        "gpu2_forbidden": True,
        "worker_count": len(ready),
        "one_dataset_per_worker": True,
        "one_joint_official_claim_per_dataset": True,
        "launcher_constructs_dataset_or_loader": False,
        "official_test_accessed": False,
        "operational_test_selected": True,
        "selection_is_optimistic": True,
        "workers": [phase_status_manifest(spec) for spec in ready],
    }


def execute_workers(specs: Sequence[WorkerSpec]) -> int:
    """Execute each phase on the three fixed GPUs before advancing the gate."""

    ready = tuple(specs)
    if tuple((spec.dataset, spec.gpu_index) for spec in ready) != DATASET_GPU_LAYOUT:
        raise ValueError("formal execution requires all three frozen dataset workers")
    statuses = {
        spec.dataset: {phase: "pending" for phase in PHASE_ORDER}
        for spec in ready
    }
    for phase_index, phase_name in enumerate(PHASE_ORDER):
        for spec in ready:
            validate_phase_statuses(statuses[spec.dataset])
            if next_pending_phase(spec, statuses[spec.dataset]).phase != phase_name:
                raise ValueError("phase skip is forbidden")
            if phase_name == "joint_official":
                command = spec.phases[phase_index].commands[0]
                _preflight_pools(
                    dataset=spec.dataset,
                    source_lock_path=Path(command[command.index("--source-lock") + 1]),
                    split_projection_path=Path(
                        command[command.index("--split-projection") + 1]
                    ),
                    candidate_pool_paths=spec.candidate_pool_paths,
                    check_environment=True,
                )
        phase_commands = [spec.phases[phase_index].commands for spec in ready]
        width = max(len(commands) for commands in phase_commands)
        for command_index in range(width):
            processes = [
                subprocess.Popen(
                    commands[command_index],
                    cwd=REPO_ROOT,
                    env=dict(spec.environment),
                )
                for spec, commands in zip(ready, phase_commands, strict=True)
                if command_index < len(commands)
            ]
            return_codes = [process.wait() for process in processes]
            failures = [code for code in return_codes if code != 0]
            if failures:
                return int(failures[0])
        for spec in ready:
            statuses[spec.dataset][phase_name] = "complete"
    return 0


def _validate_frozen_pools_action(arguments: argparse.Namespace) -> int:
    if arguments.dataset not in DATASETS:
        raise ValueError("--dataset is required for frozen-pool validation")
    source = _regular_file(arguments.source_lock, name="source lock")
    projection = _regular_file(arguments.split_projection, name="split projection")
    pools = {
        role: candidate_pool_path(arguments.candidate_pool_root, arguments.dataset, role)
        for role in ROLES
    }
    result = _preflight_pools(
        dataset=arguments.dataset,
        source_lock_path=source,
        split_projection_path=projection,
        candidate_pool_paths=pools,
        check_environment=True,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--split-projection", type=Path, required=True)
    parser.add_argument("--candidate-pool-root", type=Path, default=DEFAULT_CANDIDATE_POOL_ROOT)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--evaluator", type=Path, default=EVALUATOR_PATH)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--phase-action", choices=("validate-frozen-pools",))
    parser.add_argument("--dataset", choices=DATASETS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    training_core.configure_determinism(seed=training_core.TRAINING_SEED)
    if arguments.phase_action == "validate-frozen-pools":
        return _validate_frozen_pools_action(arguments)
    specs = build_worker_specs(
        results_root=arguments.results_root,
        data_root=arguments.data_root,
        source_lock_path=arguments.source_lock,
        split_projection_path=arguments.split_projection,
        candidate_pool_root=arguments.candidate_pool_root,
        python=arguments.python,
        evaluator_path=arguments.evaluator,
        workers=arguments.workers,
    )
    if not arguments.execute:
        print(
            json.dumps(
                dry_run_payload(specs),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    return execute_workers(specs)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_GPU_INDICES",
    "DATASETS",
    "DATASET_GPU_LAYOUT",
    "DEFAULT_RESULTS_ROOT",
    "GPU_ASSIGNMENTS",
    "PHASE_ORDER",
    "ROLES",
    "SCHEMA",
    "STATUS_SCHEMA",
    "PhaseCommand",
    "WorkerSpec",
    "build_worker_specs",
    "candidate_pool_path",
    "dry_run_payload",
    "execute_workers",
    "gpu_environment",
    "next_pending_phase",
    "parse_args",
    "phase_status_manifest",
    "run_directory",
    "synthetic_worker_spec_for_tests",
    "validate_phase_statuses",
]
