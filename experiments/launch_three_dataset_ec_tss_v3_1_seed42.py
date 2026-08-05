#!/usr/bin/env python3
"""Prepare or execute the EC-TSS V3.1 seed-42 three-dataset workflow.

The three formal runs remain independent single-GPU jobs.  Their first
invocation follows the 1000-epoch schedule but pauses after epoch 200.  Only
after all three paused prefixes pass the frozen runtime checks does the
supervisor resume the same run directories from epoch 201 to epoch 1000.

GPU assignment is fixed to one training lane per dataset:

* physical GPU 0: NUAA-SIRST;
* physical GPU 1: NUDT-SIRST;
* physical GPU 2: IRSTD-1K;
* physical GPU 3: smoke/scale checks and post-training evaluation only.

No DDP process and no duplicate dataset run is created by this launcher.
Comparison/finalization is intentionally left for the separately frozen
EC-TSS comparator revision.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import evaluate_three_dataset_ec_tss_v3_1 as evaluator  # noqa: E402
from experiments import tss_off_diagnostic_common_v1 as artifacts  # noqa: E402


SCHEMA = "sctransnet_three_dataset_ec_tss_v3_1_launcher_v1/v1"
TRAINER_SCHEMA = "sctransnet_three_dataset_ec_tss_v3_1_seed42/v1"
OBJECTIVE_ID = "ec_tss_v3_1"
RECIPE_ID = "final_ec_tss_v3_1"
METHOD = "final"
OUTPUT_METHOD = "final_ec_tss_v3_1"
TRAINING_SEED = 42
PLANNED_TOTAL_EPOCHS = 1000
PAUSE_AFTER_EPOCH = 200
REQUESTED_TSS_WEIGHT = 0.005
SURVIVAL_RATIO_CAP = 0.10
CONFIDENCE_THRESHOLD = 0.5
TARGET_DILATION_RADIUS = 3
CHECKPOINT_ROLES = ("best_miou", "best_pd")
DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
RUNNER = REPO_ROOT / "experiments" / "train_three_dataset_ec_tss_v3_1_seed42.py"
EVALUATOR = REPO_ROOT / "experiments" / "evaluate_three_dataset_ec_tss_v3_1.py"
PROTOCOL_DOCUMENT = REPO_ROOT / "SCTransNet_EC-TSS_V3性能提升与下一步方案.md"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "three_dataset_ec_tss_v3_1_seed42"
DEFAULT_LAUNCH_PLAN = DEFAULT_RESULTS_ROOT / "launch" / "formal" / "launch_plan.json"
DEFAULT_STATUS = DEFAULT_RESULTS_ROOT / "launch" / "formal" / "supervisor_status.json"
DEFAULT_LOCK = DEFAULT_RESULTS_ROOT / "launch" / "formal" / "supervisor.lock"
DEFAULT_PILOT_GATE = DEFAULT_RESULTS_ROOT / "pilot_gate" / "pilot200_runtime_gate.json"
DATASET_ROOT = REPO_ROOT / "datasets"
DATA_PROTOCOL_MANIFEST = (
    REPO_ROOT
    / "results"
    / "three_dataset_v2"
    / "manifests"
    / "three_dataset_v2_protocol.json"
)
CHECKPOINT_FILENAMES = {
    "best_miou": "best_miou.pth.tar",
    "best_pd": "best_pd.pth.tar",
}
GPU_ASSIGNMENTS = {
    "0": {
        "physical_index": "0",
        "uuid": "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
        "role": "NUAA-SIRST training",
    },
    "1": {
        "physical_index": "1",
        "uuid": "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
        "role": "NUDT-SIRST training",
    },
    "2": {
        "physical_index": "2",
        "uuid": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
        "role": "IRSTD-1K training",
    },
    "3": {
        "physical_index": "3",
        "uuid": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
        "role": "smoke, scale checks, and evaluation",
    },
}
TRAINING_LAYOUT = (
    ("NUAA-SIRST", "0"),
    ("NUDT-SIRST", "1"),
    ("IRSTD-1K", "2"),
)
CPU_ENVIRONMENT = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "OMP_WAIT_POLICY": "PASSIVE",
    "KMP_BLOCKTIME": "0",
}


@dataclass(frozen=True)
class WorkerSpec:
    dataset: str
    gpu_index: str
    run_directory: Path
    log_directory: Path
    pilot_command: tuple[str, ...]
    resume_command: tuple[str, ...]
    environment: Mapping[str, str]

    @property
    def key(self) -> str:
        return f"{self.dataset}__{RECIPE_ID}"


@dataclass(frozen=True)
class EvaluationSpec:
    dataset: str
    checkpoint_role: str
    gpu_index: str
    run_directory: Path
    output_path: Path
    log_path: Path
    command: tuple[str, ...]
    environment: Mapping[str, str]

    @property
    def key(self) -> str:
        return f"{self.dataset}__{RECIPE_ID}__{self.checkpoint_role}"


@dataclass(frozen=True)
class SmokeScaleSpec:
    dataset: str
    gpu_index: str
    results_root: Path
    run_directory: Path
    log_directory: Path
    pause_command: tuple[str, ...]
    resume_command: tuple[str, ...]
    environment: Mapping[str, str]

    @property
    def key(self) -> str:
        return f"gpu3_screen__{self.dataset}__{RECIPE_ID}"


def run_directory(results_root: Path, dataset: str) -> Path:
    artifacts.require(dataset in DATASETS, f"unsupported dataset: {dataset}")
    return (
        Path(results_root)
        / "runs"
        / dataset
        / RECIPE_ID
        / "seed_42"
    )


def _environment(
    gpu_index: str,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    artifacts.require(gpu_index in GPU_ASSIGNMENTS, "unknown physical GPU")
    environment = dict(os.environ if base is None else base)
    environment.update(CPU_ENVIRONMENT)
    environment["CUDA_VISIBLE_DEVICES"] = GPU_ASSIGNMENTS[gpu_index]["uuid"]
    environment["PYTHONUNBUFFERED"] = "1"
    # This must be present before the Python process initializes CUDA.
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return environment


def _base_training_command(
    *,
    python: Path,
    dataset: str,
    results_root: Path,
    gpu_index: str,
) -> tuple[str, ...]:
    gpu = GPU_ASSIGNMENTS[gpu_index]
    return (
        str(python),
        str(RUNNER),
        "--dataset",
        dataset,
        "--method",
        METHOD,
        "--tss-weight",
        str(REQUESTED_TSS_WEIGHT),
        "--data-root",
        str(DATASET_ROOT),
        "--results-root",
        str(Path(results_root)),
        "--protocol-manifest",
        str(DATA_PROTOCOL_MANIFEST),
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
    )


def build_worker_specs(
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    python: Path = PYTHON,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[WorkerSpec, ...]:
    specs: list[WorkerSpec] = []
    for dataset, gpu_index in TRAINING_LAYOUT:
        base = _base_training_command(
            python=python,
            dataset=dataset,
            results_root=results_root,
            gpu_index=gpu_index,
        )
        specs.append(
            WorkerSpec(
                dataset=dataset,
                gpu_index=gpu_index,
                run_directory=run_directory(results_root, dataset),
                log_directory=(
                    Path(results_root)
                    / "launch"
                    / "formal"
                    / "logs"
                    / dataset
                    / RECIPE_ID
                    / "seed_42"
                ),
                pilot_command=(
                    *base,
                    "--resume",
                    "auto",
                    "--pause-after-epoch",
                    str(PAUSE_AFTER_EPOCH),
                ),
                resume_command=(*base, "--resume", "required"),
                environment=_environment(gpu_index, base_environment),
            )
        )
    artifacts.require(len(specs) == 3, "training worker count differs")
    artifacts.require(
        {(spec.dataset, spec.gpu_index) for spec in specs}
        == set(TRAINING_LAYOUT),
        "training GPU layout differs",
    )
    artifacts.require(
        len({spec.dataset for spec in specs}) == len(DATASETS),
        "a dataset run is duplicated",
    )
    return tuple(specs)


def _worker_record(spec: WorkerSpec) -> dict[str, Any]:
    return {
        "key": spec.key,
        "dataset": spec.dataset,
        "method": OUTPUT_METHOD,
        "training_model_method": METHOD,
        "objective_id": OBJECTIVE_ID,
        "recipe_id": RECIPE_ID,
        "requested_tss_weight": REQUESTED_TSS_WEIGHT,
        "survival_ratio_cap": SURVIVAL_RATIO_CAP,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "target_dilation_radius": TARGET_DILATION_RADIUS,
        "physical_gpu_index": spec.gpu_index,
        "gpu_uuid": GPU_ASSIGNMENTS[spec.gpu_index]["uuid"],
        "run_directory": str(spec.run_directory),
        "log_directory": str(spec.log_directory),
        "pilot_command": list(spec.pilot_command),
        "resume_command": list(spec.resume_command),
        "environment": {
            key: spec.environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONUNBUFFERED",
                "CUBLAS_WORKSPACE_CONFIG",
                *CPU_ENVIRONMENT.keys(),
            )
        },
        "seed": TRAINING_SEED,
        "planned_total_epochs": PLANNED_TOTAL_EPOCHS,
        "pause_after_epoch": PAUSE_AFTER_EPOCH,
        "eval_every": 10,
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "threshold": 0.5,
        "pilot_resume": "auto",
        "formal_resume": "required",
        "single_gpu": True,
        "ddp": False,
    }


def build_evaluation_specs(
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    python: Path = PYTHON,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[EvaluationSpec, ...]:
    specs: list[EvaluationSpec] = []
    gpu_index = "3"
    for dataset in DATASETS:
        current_run = run_directory(results_root, dataset)
        for role in CHECKPOINT_ROLES:
            output = current_run / "evaluations" / f"{role}.json"
            specs.append(
                EvaluationSpec(
                    dataset=dataset,
                    checkpoint_role=role,
                    gpu_index=gpu_index,
                    run_directory=current_run,
                    output_path=output,
                    log_path=(
                        Path(results_root)
                        / "launch"
                        / "formal"
                        / "evaluation_logs"
                        / f"{dataset}__{role}.log"
                    ),
                    command=(
                        str(python),
                        str(EVALUATOR),
                        "--dataset",
                        dataset,
                        "--checkpoint-role",
                        role,
                        "--run-dir",
                        str(current_run),
                        "--dataset-root",
                        str(DATASET_ROOT),
                        "--data-protocol-manifest",
                        str(DATA_PROTOCOL_MANIFEST),
                        "--output",
                        str(output),
                        "--device",
                        "cuda:0",
                        "--workers",
                        "0",
                    ),
                    environment=_environment(gpu_index, base_environment),
                )
            )
    artifacts.require(len(specs) == 6, "evaluation matrix differs")
    artifacts.require(
        {spec.gpu_index for spec in specs} == {"3"},
        "post-training evaluation must use GPU3",
    )
    return tuple(specs)


def build_smoke_scale_spec(
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    python: Path = PYTHON,
    base_environment: Mapping[str, str] | None = None,
) -> SmokeScaleSpec:
    """Build the one non-formal GPU3 forward/backward/resume screen."""

    dataset = "NUAA-SIRST"
    gpu_index = "3"
    screen_root = Path(results_root) / "screen_gpu3"
    actual_run = (
        screen_root
        / "smoke"
        / "runs"
        / dataset
        / RECIPE_ID
        / "seed_42"
    )
    gpu = GPU_ASSIGNMENTS[gpu_index]
    base = (
        str(python),
        str(RUNNER),
        "--dataset",
        dataset,
        "--method",
        METHOD,
        "--tss-weight",
        str(REQUESTED_TSS_WEIGHT),
        "--data-root",
        str(DATASET_ROOT),
        "--results-root",
        str(screen_root),
        "--protocol-manifest",
        str(DATA_PROTOCOL_MANIFEST),
        "--seed",
        str(TRAINING_SEED),
        "--epochs",
        "2",
        "--begin-test",
        "1",
        "--eval-every",
        "1",
        "--batch-size",
        "2",
        "--patch-size",
        "256",
        "--workers",
        "0",
        "--base-lr",
        "0.001",
        "--min-lr",
        "0.00001",
        "--warmup-epochs",
        "1",
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
        "--smoke",
        "--max-train-images",
        "8",
        "--max-test-images",
        "2",
    )
    return SmokeScaleSpec(
        dataset=dataset,
        gpu_index=gpu_index,
        results_root=screen_root,
        run_directory=actual_run,
        log_directory=screen_root / "launch" / "logs",
        pause_command=(
            *base,
            "--resume",
            "auto",
            "--pause-after-epoch",
            "1",
        ),
        resume_command=(*base, "--resume", "required"),
        environment=_environment(gpu_index, base_environment),
    )


def _smoke_record(spec: SmokeScaleSpec) -> dict[str, Any]:
    return {
        "key": spec.key,
        "formal_run": False,
        "counts_toward_formal_run_budget": False,
        "dataset": spec.dataset,
        "physical_gpu_index": spec.gpu_index,
        "gpu_uuid": GPU_ASSIGNMENTS[spec.gpu_index]["uuid"],
        "results_root": str(spec.results_root),
        "run_directory": str(spec.run_directory),
        "pause_command": list(spec.pause_command),
        "resume_command": list(spec.resume_command),
        "environment": {
            key: spec.environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONUNBUFFERED",
                "CUBLAS_WORKSPACE_CONFIG",
                *CPU_ENVIRONMENT.keys(),
            )
        },
        "epochs": 2,
        "pause_after_epoch": 1,
        "max_train_images": 8,
        "max_test_images": 2,
        "coverage": [
            "forward",
            "loss",
            "backward",
            "optimizer_step",
            "checkpoint_save",
            "strict_model_and_optimizer_resume",
            "fresh_model_strict_checkpoint_reload",
            "risk_branch_scale_checks",
        ],
    }


def _required_sources() -> dict[str, Path]:
    paths = {
        "launcher": Path(__file__).resolve(),
        "runner": RUNNER,
        "ec_tss_loss": (
            REPO_ROOT / "experiments" / "tpd_training_loss_ec_tss_v3_1.py"
        ),
        "ec_tss_protocol": (
            REPO_ROOT / "experiments" / "EC_TSS_V3_1_PROTOCOL.md"
        ),
        "evaluator_adapter": EVALUATOR,
        "evaluator_core": REPO_ROOT / "experiments" / "evaluate_three_dataset_v2.py",
        "artifact_helpers": Path(artifacts.__file__).resolve(),
        "training_engine": (
            REPO_ROOT
            / "experiments"
            / "train_four_dataset_original_final_seed42_exact_v1.py"
        ),
        "positive_runner": (
            REPO_ROOT / "experiments" / "train_three_dataset_seed42_global_tss_v2.py"
        ),
        "data_protocol": REPO_ROOT / "experiments" / "three_dataset_v2_protocol.py",
        "torch_datasets": REPO_ROOT / "experiments" / "paper_three_dataset_v2.py",
        "model_builder": REPO_ROOT / "experiments" / "four_dataset_models_seed42_v1.py",
        "legacy_training_loss": REPO_ROOT / "experiments" / "tpd_training_loss.py",
        "metric_schedule": REPO_ROOT / "experiments" / "train_tpd_pilot.py",
        "evaluation_metrics": (
            REPO_ROOT / "experiments" / "four_dataset_evaluation_protocol_v1.py"
        ),
        "protocol_document": PROTOCOL_DOCUMENT,
    }
    for path in sorted((REPO_ROOT / "model").rglob("*.py")):
        paths[f"architecture::{path.relative_to(REPO_ROOT).as_posix()}"] = path
    return dict(sorted(paths.items()))


def static_inputs(
    *,
    results_root: Path,
    python: Path,
) -> dict[str, Any]:
    python_entrypoint = Path(python).absolute()
    artifacts.require(
        python_entrypoint.is_file() and os.access(python_entrypoint, os.X_OK),
        f"Python is not executable: {python_entrypoint}",
    )
    return {
        "python_entrypoint": str(python_entrypoint),
        "python": artifacts.artifact_record(python_entrypoint),
        "results_root": str(Path(results_root).resolve()),
        "dataset_root": str(DATASET_ROOT.resolve(strict=True)),
        "data_protocol_manifest": artifacts.artifact_record(DATA_PROTOCOL_MANIFEST),
        "sources": {
            name: artifacts.artifact_record(path)
            for name, path in _required_sources().items()
        },
    }


def prepare_launch_plan(
    *,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    python: Path = PYTHON,
    launch_plan_path: Path | None = None,
) -> dict[str, Any]:
    launch_plan_path = (
        Path(launch_plan_path)
        if launch_plan_path is not None
        else Path(results_root) / "launch" / "formal" / "launch_plan.json"
    )
    workers = build_worker_specs(results_root=results_root, python=python)
    evaluations = build_evaluation_specs(results_root=results_root, python=python)
    smoke = build_smoke_scale_spec(results_root=results_root, python=python)
    plan = {
        "schema": SCHEMA,
        "status": "prepared_not_started",
        "training_started": False,
        "formal_execution_requires_explicit_execute": True,
        "objective_id": OBJECTIVE_ID,
        "recipe_id": RECIPE_ID,
        "dataset_order": list(DATASETS),
        "execution_strategy": (
            "three_single_gpu_training_lanes_global_pause200_then_resume1000"
        ),
        "training_lane_count": 3,
        "training_worker_count": 3,
        "no_ddp": True,
        "no_duplicate_runs": True,
        "gpu_roles": {
            index: dict(record) for index, record in GPU_ASSIGNMENTS.items()
        },
        "workers": [_worker_record(spec) for spec in workers],
        "gpu3_smoke_scale_screen": _smoke_record(smoke),
        "pilot_gate": {
            "global": True,
            "all_three_prefixes_required": True,
            "planned_total_epochs": PLANNED_TOTAL_EPOCHS,
            "pause_after_epoch": PAUSE_AFTER_EPOCH,
            "pass_then_resume_same_run": True,
            "resume_mode": "required",
            "output": str(
                Path(results_root) / "pilot_gate" / "pilot200_runtime_gate.json"
            ),
        },
        "posttraining": {
            "evaluation_count": 6,
            "evaluation_gpu": 3,
            "fixed_threshold": 0.5,
            "evaluation_commands": [
                {
                    "key": spec.key,
                    "dataset": spec.dataset,
                    "method": OUTPUT_METHOD,
                    "training_model_method": METHOD,
                    "objective_id": OBJECTIVE_ID,
                    "requested_tss_weight": REQUESTED_TSS_WEIGHT,
                    "checkpoint_role": spec.checkpoint_role,
                    "physical_gpu_index": spec.gpu_index,
                    "gpu_uuid": GPU_ASSIGNMENTS[spec.gpu_index]["uuid"],
                    "run_directory": str(spec.run_directory),
                    "output": str(spec.output_path),
                    "command": list(spec.command),
                }
                for spec in evaluations
            ],
            "comparison": {
                "status": "deferred_to_compare_finalize_ec_tss_v3_1",
                "training_or_evaluation_blocked_by_deferral": False,
            },
        },
        "static_inputs": static_inputs(
            results_root=results_root,
            python=python,
        ),
    }
    action = artifacts.write_once_or_identical(launch_plan_path, plan)
    return {
        "status": "complete",
        "action": action,
        "launch_plan": artifacts.artifact_record(launch_plan_path),
        "plan": plan,
    }


def _verify_static_inputs(plan: Mapping[str, Any]) -> None:
    expected = plan.get("static_inputs")
    artifacts.require(isinstance(expected, dict), "launch plan lacks static inputs")
    for name, record in expected.get("sources", {}).items():
        artifacts.require(isinstance(record, dict), f"malformed source lock: {name}")
        artifacts.require(
            artifacts.artifact_record(Path(record["path"])) == record,
            f"source changed after prepare: {name}",
        )
    for label in ("python", "data_protocol_manifest"):
        record = expected.get(label)
        artifacts.require(isinstance(record, dict), f"launch plan lacks {label}")
        artifacts.require(
            artifacts.artifact_record(Path(record["path"])) == record,
            f"{label} changed after prepare",
        )


def _recipe_subset(container: Mapping[str, Any], label: str) -> None:
    recipe = container.get("recipe")
    artifacts.require(isinstance(recipe, Mapping), f"{label} lacks recipe")
    required = {
        "method": METHOD,
        "recipe_id": RECIPE_ID,
        "objective_id": OBJECTIVE_ID,
        "requested_tss_weight": REQUESTED_TSS_WEIGHT,
        "tss_ratio_cap": SURVIVAL_RATIO_CAP,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "target_dilation_radius": TARGET_DILATION_RADIUS,
    }
    for field, expected in required.items():
        artifacts.require(
            recipe.get(field) == expected,
            f"{label} recipe {field} differs",
        )


def _load_metric_rows(path: Path) -> list[dict[str, Any]]:
    return [row for _, row in artifacts.iter_jsonl(path)]


def _diagnostic_value(row: Mapping[str, Any], name: str) -> float:
    candidates = (
        name,
        f"train_{name}",
        f"train_ec_tss_{name}",
        f"train_ec_tss_v3_1_{name}",
    )
    matches = [key for key in candidates if key in row]
    if not matches:
        matches = [key for key in row if key.endswith(f"_{name}")]
    artifacts.require(len(matches) == 1, f"diagnostic {name} is missing or ambiguous")
    value = float(row[matches[0]])
    artifacts.require(math.isfinite(value), f"diagnostic {name} is non-finite")
    return value


def _protocol_sha(protocol: Mapping[str, Any]) -> str:
    declared = protocol.get("protocol_sha256")
    artifacts.require(isinstance(declared, str) and declared, "protocol lacks SHA")
    unsigned = dict(protocol)
    del unsigned["protocol_sha256"]
    artifacts.require(
        artifacts.compact_sha256(unsigned) == declared,
        "protocol payload SHA differs",
    )
    return declared


def _validate_paused_run(spec: WorkerSpec) -> dict[str, Any] | None:
    progress_path = spec.run_directory / "progress.json"
    resume_path = spec.run_directory / "resume" / "latest_training_state.pth.tar"
    protocol_path = spec.run_directory / "protocol.json"
    metrics_path = spec.run_directory / "metrics.jsonl"
    if not progress_path.is_file():
        return None
    progress = artifacts.load_json(progress_path)
    if progress.get("status") != "paused":
        return None
    from experiments import train_three_dataset_ec_tss_v3_1_seed42 as runner

    runner_progress = runner.validate_paused_run(
        spec.run_directory,
        spec.dataset,
        pause_epoch=PAUSE_AFTER_EPOCH,
    )
    artifacts.require(
        runner_progress == progress,
        f"paused {spec.key} differs from the runner validator",
    )
    for field, expected in (
        ("schema", TRAINER_SCHEMA),
        ("dataset", spec.dataset),
        ("method", METHOD),
        ("seed", TRAINING_SEED),
        ("completed_epoch", PAUSE_AFTER_EPOCH),
        ("total_epochs", PLANNED_TOTAL_EPOCHS),
        ("planned_total_epochs", PLANNED_TOTAL_EPOCHS),
        ("pause_after_epoch", PAUSE_AFTER_EPOCH),
        ("resume_required", True),
        ("required_resume_mode", "required"),
        ("objective_id", OBJECTIVE_ID),
    ):
        artifacts.require(progress.get(field) == expected, f"paused {spec.key} {field} differs")
    _recipe_subset(progress, f"paused {spec.key} progress")
    protocol = artifacts.load_json(protocol_path)
    for field, expected in (
        ("schema", TRAINER_SCHEMA),
        ("dataset", spec.dataset),
        ("method", METHOD),
        ("training_seed", TRAINING_SEED),
        ("epochs", PLANNED_TOTAL_EPOCHS),
        ("begin_test", 10),
        ("eval_every", 10),
    ):
        artifacts.require(protocol.get(field) == expected, f"paused {spec.key} protocol {field} differs")
    _recipe_subset(protocol, f"paused {spec.key} protocol")
    protocol_sha = _protocol_sha(protocol)
    artifacts.require(resume_path.is_file(), f"paused {spec.key} lacks rolling state")
    import torch

    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    artifacts.require(isinstance(resume, dict), f"paused {spec.key} rolling state is invalid")
    for field, expected in (
        ("schema", TRAINER_SCHEMA),
        ("dataset", spec.dataset),
        ("method", METHOD),
        ("seed", TRAINING_SEED),
        ("epoch", PAUSE_AFTER_EPOCH),
        ("protocol_sha256", protocol_sha),
        ("planned_total_epochs", PLANNED_TOTAL_EPOCHS),
        ("objective_id", OBJECTIVE_ID),
    ):
        artifacts.require(resume.get(field) == expected, f"paused {spec.key} rolling {field} differs")
    _recipe_subset(resume, f"paused {spec.key} rolling state")
    event = resume.get("event")
    artifacts.require(isinstance(event, Mapping), f"paused {spec.key} lacks event")
    artifacts.require(event.get("epoch") == PAUSE_AFTER_EPOCH, f"paused {spec.key} event epoch differs")
    artifacts.require(event.get("evaluated") is True, f"paused {spec.key} epoch 200 was not evaluated")
    rows = _load_metric_rows(metrics_path)
    artifacts.require(len(rows) == PAUSE_AFTER_EPOCH, f"paused {spec.key} metrics length differs")
    artifacts.require(
        rows[-1].get("epoch") == event.get("epoch")
        and rows[-1].get("dataset") == event.get("dataset")
        and rows[-1].get("train_ec_tss_objective_id")
        == event.get("train_ec_tss_objective_id"),
        f"paused {spec.key} last metric/event identity differs",
    )
    for row in rows:
        for field in ("train_total_loss", "train_segmentation_loss", "train_survival_loss"):
            value = float(row[field])
            artifacts.require(math.isfinite(value), f"paused {spec.key} {field} is non-finite")
    evaluated = rows[-1]
    for field in ("miou", "niou", "pd", "fa"):
        value = float(evaluated[field])
        artifacts.require(math.isfinite(value), f"paused {spec.key} {field} is non-finite")
    diagnostic_sums = {
        name: sum(_diagnostic_value(row, name) for row in rows)
        for name in (
            "positive_risk_mass_mean",
            "negative_risk_mass_mean",
            "positive_active_cells_mean",
            "negative_active_cells_mean",
        )
    }
    artifacts.require(
        diagnostic_sums["positive_risk_mass_mean"] > 0.0
        and diagnostic_sums["positive_active_cells_mean"] > 0.0,
        f"paused {spec.key} positive EC-TSS branch never activated",
    )
    artifacts.require(
        diagnostic_sums["negative_risk_mass_mean"] > 0.0
        and diagnostic_sums["negative_active_cells_mean"] > 0.0,
        f"paused {spec.key} negative EC-TSS branch never activated",
    )
    return {
        "dataset": spec.dataset,
        "completed_epoch": PAUSE_AFTER_EPOCH,
        "planned_total_epochs": PLANNED_TOTAL_EPOCHS,
        "epoch_200_fixed_threshold": {
            key: evaluated[key] for key in ("miou", "niou", "pd", "fa")
        },
        "diagnostic_sums": diagnostic_sums,
        "metrics_prefix_sha256": artifacts.compact_sha256(rows),
        "protocol_sha256": protocol_sha,
    }


def _validate_complete_run(spec: WorkerSpec) -> dict[str, Any] | None:
    summary_path = spec.run_directory / "summary.json"
    if not summary_path.is_file():
        return None
    summary = artifacts.load_json(summary_path)
    for field, expected in (
        ("schema", TRAINER_SCHEMA),
        ("status", "complete"),
        ("dataset", spec.dataset),
        ("method", METHOD),
        ("seed", TRAINING_SEED),
        ("epochs", PLANNED_TOTAL_EPOCHS),
        ("planned_total_epochs", PLANNED_TOTAL_EPOCHS),
        ("objective_id", OBJECTIVE_ID),
    ):
        artifacts.require(summary.get(field) == expected, f"completed {spec.key} {field} differs")
    _recipe_subset(summary, f"completed {spec.key} summary")
    protocol = artifacts.load_json(spec.run_directory / "protocol.json")
    for field, expected in (
        ("schema", TRAINER_SCHEMA),
        ("dataset", spec.dataset),
        ("method", METHOD),
        ("training_seed", TRAINING_SEED),
        ("epochs", PLANNED_TOTAL_EPOCHS),
        ("planned_total_epochs", PLANNED_TOTAL_EPOCHS),
        ("begin_test", 10),
        ("eval_every", 10),
        ("objective_id", OBJECTIVE_ID),
    ):
        artifacts.require(
            protocol.get(field) == expected,
            f"completed {spec.key} protocol {field} differs",
        )
    _recipe_subset(protocol, f"completed {spec.key} protocol")
    declared = _protocol_sha(protocol)
    artifacts.require(summary.get("protocol_sha256") == declared, f"completed {spec.key} protocol SHA differs")
    checkpoints = summary.get("checkpoints")
    artifacts.require(
        isinstance(checkpoints, dict) and set(checkpoints) == set(CHECKPOINT_ROLES),
        f"completed {spec.key} checkpoints differ",
    )
    for role, record in checkpoints.items():
        artifacts.require(isinstance(record, dict), f"completed {spec.key} {role} record malformed")
        path = Path(str(record.get("path", "")))
        artifacts.require(
            artifacts.file_sha256(path) == record.get("sha256"),
            f"completed {spec.key} {role} SHA differs",
        )
    artifacts.require(
        not (spec.run_directory / "resume" / "latest_training_state.pth.tar").exists(),
        f"completed {spec.key} retained rolling state",
    )
    return summary


def _run_worker_command(
    spec: WorkerSpec,
    *,
    phase: str,
    command: Sequence[str],
) -> str | None:
    spec.log_directory.mkdir(parents=True, exist_ok=True)
    log_path = spec.log_directory / f"{phase}.log"
    with log_path.open("ab") as handle:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=dict(spec.environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return None if completed.returncode == 0 else f"{spec.key}:{phase}:exit_{completed.returncode}"


def _run_parallel_phase(
    specs: Sequence[WorkerSpec],
    *,
    phase: str,
    command_getter: Any,
) -> list[str]:
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            spec.key: executor.submit(
                _run_worker_command,
                spec,
                phase=phase,
                command=command_getter(spec),
            )
            for spec in specs
        }
        for key, future in futures.items():
            try:
                failure = future.result()
            except Exception as exc:
                failure = f"{key}:{phase}:{type(exc).__name__}:{exc}"
            if failure is not None:
                failures.append(failure)
    return failures


def _run_smoke_command(
    spec: SmokeScaleSpec,
    *,
    phase: str,
    command: Sequence[str],
) -> None:
    spec.log_directory.mkdir(parents=True, exist_ok=True)
    log_path = spec.log_directory / f"{phase}.log"
    with log_path.open("ab") as handle:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            env=dict(spec.environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    artifacts.require(
        completed.returncode == 0,
        f"{spec.key} {phase} failed with exit {completed.returncode}",
    )


def _validate_smoke_complete(spec: SmokeScaleSpec) -> dict[str, Any] | None:
    summary_path = spec.run_directory / "summary.json"
    if not summary_path.is_file():
        return None
    summary = artifacts.load_json(summary_path)
    for field, expected in (
        ("schema", TRAINER_SCHEMA),
        ("status", "complete"),
        ("dataset", spec.dataset),
        ("method", METHOD),
        ("seed", TRAINING_SEED),
        ("epochs", 2),
        ("planned_total_epochs", 2),
    ):
        artifacts.require(
            summary.get(field) == expected,
            f"GPU3 screen summary {field} differs",
        )
    _recipe_subset(summary, "GPU3 screen summary")
    metrics = _load_metric_rows(spec.run_directory / "metrics.jsonl")
    artifacts.require(
        len(metrics) == 2 and [row.get("epoch") for row in metrics] == [1, 2],
        "GPU3 screen metrics do not contain the exact two epochs",
    )
    sums = {
        name: sum(_diagnostic_value(row, name) for row in metrics)
        for name in (
            "positive_risk_mass_mean",
            "negative_risk_mass_mean",
            "positive_active_cells_mean",
            "negative_active_cells_mean",
        )
    }
    artifacts.require(
        sums["positive_risk_mass_mean"] > 0.0
        and sums["positive_active_cells_mean"] > 0.0,
        "GPU3 screen positive risk branch did not activate",
    )
    artifacts.require(
        sums["negative_risk_mass_mean"] > 0.0
        and sums["negative_active_cells_mean"] > 0.0,
        "GPU3 screen negative risk branch did not activate",
    )
    for row in metrics:
        weighted = _diagnostic_value(row, "weighted_survival_mean")
        ratio = _diagnostic_value(
            row,
            "effective_weighted_to_segmentation_ratio_mean",
        )
        artifacts.require(
            math.isfinite(weighted) and weighted >= 0.0,
            "GPU3 screen weighted survival is invalid",
        )
        artifacts.require(
            0.0 <= ratio <= SURVIVAL_RATIO_CAP + 1e-6,
            "GPU3 screen weighted-to-segmentation ratio exceeds its cap",
        )
    checkpoints = summary.get("checkpoints")
    artifacts.require(
        isinstance(checkpoints, dict) and set(checkpoints) == set(CHECKPOINT_ROLES),
        "GPU3 screen selected checkpoints differ",
    )
    checkpoint = Path(str(checkpoints["best_miou"]["path"]))
    artifacts.require(
        artifacts.file_sha256(checkpoint) == checkpoints["best_miou"]["sha256"],
        "GPU3 screen checkpoint SHA differs",
    )
    import torch
    from experiments import four_dataset_models_seed42_v1 as models

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, _ = models.build_method_model(
        METHOD,
        seed=TRAINING_SEED,
        dataset_name=spec.dataset,
        training=True,
    )
    incompatible = model.load_state_dict(payload["state_dict"], strict=True)
    artifacts.require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "GPU3 screen fresh-model strict reload returned incompatible keys",
    )
    return {
        "schema": "sctransnet_ec_tss_v3_1_gpu3_smoke_scale_v1",
        "status": "passed",
        "formal_run": False,
        "dataset": spec.dataset,
        "physical_gpu_index": 3,
        "gpu_uuid": GPU_ASSIGNMENTS["3"]["uuid"],
        "epochs": 2,
        "pause_resume_exercised": True,
        "optimizer_state_present_before_strict_resume": True,
        "fresh_model_checkpoint_strict_reload": True,
        "coverage": _smoke_record(spec)["coverage"],
        "diagnostic_sums": sums,
        "metrics_sha256": artifacts.compact_sha256(metrics),
        "summary": artifacts.artifact_record(summary_path),
        "checkpoint": artifacts.artifact_record(checkpoint),
    }


def _run_smoke_scale(spec: SmokeScaleSpec) -> dict[str, Any]:
    attestation_path = spec.results_root / "gpu3_smoke_scale_attestation.json"
    existing = _validate_smoke_complete(spec)
    if existing is not None:
        artifacts.write_once_or_identical(attestation_path, existing)
        return existing
    progress_path = spec.run_directory / "progress.json"
    if not progress_path.is_file() or artifacts.load_json(progress_path).get("status") != "paused":
        _run_smoke_command(
            spec,
            phase="pause_after_epoch_1",
            command=spec.pause_command,
        )
    from experiments import train_three_dataset_ec_tss_v3_1_seed42 as runner

    paused = runner.validate_paused_run(
        spec.run_directory,
        spec.dataset,
        pause_epoch=1,
    )
    rolling = paused["rolling_resume_state"]
    import torch

    rolling_payload = torch.load(
        Path(rolling["path"]),
        map_location="cpu",
        weights_only=False,
    )
    optimizer = rolling_payload.get("optimizer")
    artifacts.require(
        isinstance(optimizer, dict)
        and isinstance(optimizer.get("state"), dict)
        and bool(optimizer["state"]),
        "GPU3 screen rolling state lacks initialized optimizer state",
    )
    _run_smoke_command(
        spec,
        phase="strict_resume_to_epoch_2",
        command=spec.resume_command,
    )
    completed = _validate_smoke_complete(spec)
    artifacts.require(completed is not None, "GPU3 smoke/scale screen did not complete")
    artifacts.write_once_or_identical(attestation_path, completed)
    return completed


def _run_pilot_commands_with_smoke(
    pending: Sequence[WorkerSpec],
    smoke: SmokeScaleSpec,
) -> tuple[list[str], dict[str, Any] | None]:
    failures: list[str] = []
    smoke_result: dict[str, Any] | None = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        worker_futures = {
            spec.key: executor.submit(
                _run_worker_command,
                spec,
                phase="pilot_1_200",
                command=spec.pilot_command,
            )
            for spec in pending
        }
        smoke_future = executor.submit(_run_smoke_scale, smoke)
        for key, future in worker_futures.items():
            try:
                failure = future.result()
            except Exception as exc:
                failure = f"{key}:pilot_1_200:{type(exc).__name__}:{exc}"
            if failure is not None:
                failures.append(failure)
        try:
            smoke_result = smoke_future.result()
        except Exception as exc:
            failures.append(
                f"{smoke.key}:{type(exc).__name__}:{exc}"
            )
    return failures, smoke_result


def _write_status(path: Path, payload: Mapping[str, Any]) -> None:
    artifacts.atomic_write_json(path, payload)


def _run_pilot_phase(
    specs: Sequence[WorkerSpec],
    *,
    smoke: SmokeScaleSpec,
    status_path: Path,
    plan_record: Mapping[str, Any],
    gate_path: Path,
) -> dict[str, Any]:
    if gate_path.is_file():
        existing = artifacts.load_json(gate_path)
        for field, expected in (
            (
                "schema",
                "sctransnet_three_dataset_ec_tss_v3_1_pilot200_gate_v1",
            ),
            ("status", "passed"),
            ("gate_passed", True),
            ("objective_id", OBJECTIVE_ID),
            ("planned_total_epochs", PLANNED_TOTAL_EPOCHS),
            ("pause_after_epoch", PAUSE_AFTER_EPOCH),
            ("all_three_runs_checked", True),
        ):
            artifacts.require(
                existing.get(field) == expected,
                f"existing pilot gate {field} differs",
            )
        artifacts.require(
            set(existing.get("datasets", {})) == set(DATASETS),
            "existing pilot gate dataset matrix differs",
        )
        screen = existing.get("gpu3_smoke_scale")
        artifacts.require(
            isinstance(screen, dict) and screen.get("status") == "passed",
            "existing pilot gate lacks its GPU3 screen",
        )
        return existing
    pending = [
        spec
        for spec in specs
        if _validate_complete_run(spec) is None
        and _validate_paused_run(spec) is None
    ]
    failures, smoke_result = _run_pilot_commands_with_smoke(pending, smoke)
    if failures:
        _write_status(
            status_path,
            {
                "schema": SCHEMA,
                "status": "pilot_failed",
                "launch_plan": dict(plan_record),
                "failures": failures,
            },
        )
        raise artifacts.TSSOffDiagnosticError(f"EC-TSS pilot failed: {failures}")
    records: dict[str, Any] = {}
    for spec in specs:
        complete = _validate_complete_run(spec)
        if complete is not None:
            records[spec.dataset] = {
                "dataset": spec.dataset,
                "already_complete": True,
                "completed_epoch": PLANNED_TOTAL_EPOCHS,
            }
            continue
        paused = _validate_paused_run(spec)
        artifacts.require(paused is not None, f"{spec.key} did not pause at epoch 200")
        records[spec.dataset] = paused
    gate = {
        "schema": "sctransnet_three_dataset_ec_tss_v3_1_pilot200_gate_v1",
        "status": "passed",
        "gate_passed": True,
        "objective_id": OBJECTIVE_ID,
        "planned_total_epochs": PLANNED_TOTAL_EPOCHS,
        "pause_after_epoch": PAUSE_AFTER_EPOCH,
        "all_three_runs_checked": set(records) == set(DATASETS),
        "test_informed_optimistic_development_decision": True,
        "paper_claim_supported_by_pilot": False,
        "gpu3_smoke_scale": smoke_result,
        "datasets": records,
    }
    artifacts.require(gate["all_three_runs_checked"], "pilot gate lacks a dataset")
    artifacts.require(
        isinstance(smoke_result, dict) and smoke_result.get("status") == "passed",
        "GPU3 smoke/scale screen did not pass",
    )
    artifacts.write_once_or_identical(gate_path, gate)
    return gate


def _run_resume_phase(
    specs: Sequence[WorkerSpec],
    *,
    status_path: Path,
    plan_record: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> None:
    artifacts.require(gate.get("gate_passed") is True, "pilot gate did not pass")
    pending = [spec for spec in specs if _validate_complete_run(spec) is None]
    for spec in pending:
        artifacts.require(
            _validate_paused_run(spec) is not None,
            f"{spec.key} cannot resume without its exact epoch-200 state",
        )
    failures = _run_parallel_phase(
        pending,
        phase="resume_201_1000",
        command_getter=lambda spec: spec.resume_command,
    )
    if failures:
        _write_status(
            status_path,
            {
                "schema": SCHEMA,
                "status": "resume_failed",
                "launch_plan": dict(plan_record),
                "pilot_gate": dict(gate),
                "failures": failures,
            },
        )
        raise artifacts.TSSOffDiagnosticError(f"EC-TSS resume failed: {failures}")
    for spec in specs:
        artifacts.require(_validate_complete_run(spec) is not None, f"{spec.key} did not complete")


def _validate_evaluation(spec: EvaluationSpec) -> dict[str, Any] | None:
    if not spec.output_path.is_file():
        return None
    payload = evaluator.validate_completed_output(
        spec.output_path,
        dataset=spec.dataset,
        checkpoint_role=spec.checkpoint_role,
    )
    binding = payload.get("checkpoint_binding")
    artifacts.require(isinstance(binding, dict), f"evaluation {spec.key} lacks binding")
    artifacts.require(
        binding.get("run_dir") == str(spec.run_directory.resolve()),
        f"evaluation {spec.key} run binding differs",
    )
    checkpoint = binding.get("checkpoint")
    expected_checkpoint = (
        spec.run_directory
        / "checkpoints"
        / CHECKPOINT_FILENAMES[spec.checkpoint_role]
    )
    artifacts.require(isinstance(checkpoint, dict), f"evaluation {spec.key} lacks checkpoint")
    artifacts.require(
        Path(str(checkpoint.get("path", ""))).resolve() == expected_checkpoint.resolve(),
        f"evaluation {spec.key} checkpoint path differs",
    )
    artifacts.require(
        checkpoint.get("sha256") == artifacts.file_sha256(expected_checkpoint),
        f"evaluation {spec.key} checkpoint SHA differs",
    )
    return payload


def _run_evaluations(specs: Sequence[EvaluationSpec]) -> None:
    # GPU3 is dedicated to this phase; run sequentially to keep every role an
    # independent, deterministic single-GPU evaluation.
    for spec in specs:
        if _validate_evaluation(spec) is not None:
            continue
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        with spec.log_path.open("ab") as handle:
            completed = subprocess.run(
                list(spec.command),
                cwd=REPO_ROOT,
                env=dict(spec.environment),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        artifacts.require(
            completed.returncode == 0,
            f"evaluation {spec.key} failed with exit {completed.returncode}",
        )
        artifacts.require(_validate_evaluation(spec) is not None, f"evaluation {spec.key} incomplete")


def execute_launch_plan(
    *,
    launch_plan_path: Path = DEFAULT_LAUNCH_PLAN,
    status_path: Path = DEFAULT_STATUS,
    lock_path: Path = DEFAULT_LOCK,
) -> None:
    plan = artifacts.load_json(launch_plan_path)
    artifacts.require(plan.get("schema") == SCHEMA, "launch-plan schema differs")
    artifacts.require(plan.get("status") == "prepared_not_started", "launch plan is not prepared")
    _verify_static_inputs(plan)
    results_root = Path(plan["static_inputs"]["results_root"])
    python = Path(plan["static_inputs"]["python_entrypoint"])
    specs = build_worker_specs(results_root=results_root, python=python)
    artifacts.require([_worker_record(spec) for spec in specs] == plan.get("workers"), "worker commands differ from plan")
    evaluations = build_evaluation_specs(results_root=results_root, python=python)
    smoke = build_smoke_scale_spec(results_root=results_root, python=python)
    artifacts.require(
        _smoke_record(smoke) == plan.get("gpu3_smoke_scale_screen"),
        "GPU3 smoke/scale command differs from plan",
    )
    plan_record = artifacts.artifact_record(launch_plan_path)
    gate_path = Path(plan["pilot_gate"]["output"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise artifacts.TSSOffDiagnosticError("another EC-TSS supervisor holds the lock") from exc
        _write_status(
            status_path,
            {
                "schema": SCHEMA,
                "status": "pilot_training",
                "launch_plan": plan_record,
                "training_started": True,
                "phase": "epochs_1_200",
            },
        )
        gate = _run_pilot_phase(
            specs,
            smoke=smoke,
            status_path=status_path,
            plan_record=plan_record,
            gate_path=gate_path,
        )
        _verify_static_inputs(plan)
        _write_status(
            status_path,
            {
                "schema": SCHEMA,
                "status": "resuming_formal",
                "launch_plan": plan_record,
                "training_started": True,
                "pilot_complete": True,
                "pilot_gate": artifacts.artifact_record(gate_path),
                "phase": "epochs_201_1000",
            },
        )
        _run_resume_phase(
            specs,
            status_path=status_path,
            plan_record=plan_record,
            gate=gate,
        )
        _verify_static_inputs(plan)
        _write_status(
            status_path,
            {
                "schema": SCHEMA,
                "status": "evaluating",
                "launch_plan": plan_record,
                "training_complete": True,
                "pilot_gate": artifacts.artifact_record(gate_path),
            },
        )
        _run_evaluations(evaluations)
        _write_status(
            status_path,
            {
                "schema": SCHEMA,
                "status": "evaluation_complete_comparison_pending",
                "launch_plan": plan_record,
                "training_started": True,
                "training_complete": True,
                "evaluation_complete": True,
                "pilot_gate": artifacts.artifact_record(gate_path),
                "comparison_status": "deferred_to_compare_finalize_ec_tss_v3_1",
            },
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--launch-plan", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--lock", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    launch_plan = (
        args.launch_plan
        if args.launch_plan is not None
        else args.results_root / "launch" / "formal" / "launch_plan.json"
    )
    if args.prepare:
        result = prepare_launch_plan(
            results_root=args.results_root,
            python=args.python,
            launch_plan_path=launch_plan,
        )
        print(
            json.dumps(
                {
                    "status": "prepared_not_started",
                    "action": result["action"],
                    "launch_plan": result["launch_plan"],
                    "training_worker_count": 3,
                    "training_lane_count": 3,
                    "pilot_pause_epoch": PAUSE_AFTER_EPOCH,
                    "planned_total_epochs": PLANNED_TOTAL_EPOCHS,
                    "evaluation_gpu": 3,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    execute_launch_plan(
        launch_plan_path=launch_plan,
        status_path=(
            args.status
            if args.status is not None
            else args.results_root / "launch" / "formal" / "supervisor_status.json"
        ),
        lock_path=(
            args.lock
            if args.lock is not None
            else args.results_root / "launch" / "formal" / "supervisor.lock"
        ),
    )


if __name__ == "__main__":
    main()
