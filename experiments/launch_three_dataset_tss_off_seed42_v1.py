#!/usr/bin/env python3
"""Prepare or execute the independent three-run TSS-off GPU2/3 workflow.

Preparation is non-training and write-once: it seals Gate O1, audits Original
reuse, freezes every source and command, and emits a two-lane launch plan.
Formal processes start only with the explicit ``--execute`` flag.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import preflight_three_dataset_tss_off_seed42_v1 as preflight  # noqa: E402
from experiments import finalize_tss_off_diagnostic_v1 as finalizer  # noqa: E402
from experiments import tss_off_diagnostic_common_v1 as common  # noqa: E402


SCHEMA = "sctransnet_three_dataset_tss_off_launcher_v1/v1"
TRAINER_SCHEMA = "sctransnet_three_dataset_tss_off_seed42_v1/v1"
PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
RUNNER = REPO_ROOT / "experiments" / "train_three_dataset_tss_off_seed42_v1.py"
EVALUATOR = (
    REPO_ROOT / "experiments" / "evaluate_three_dataset_tss_off_seed42_v1.py"
)
COMPARATOR = (
    REPO_ROOT / "experiments" / "compare_tss_off_positive_original_v1.py"
)
FINALIZER = REPO_ROOT / "experiments" / "finalize_tss_off_diagnostic_v1.py"
SHELL_ENTRYPOINT = (
    REPO_ROOT / "experiments" / "launch_three_dataset_tss_off_seed42_v1.sh"
)
PROTOCOL_DOCUMENT = (
    REPO_ROOT / "SCTransNet_正TSS全局配方失败后的TSS-Off因果诊断方案.md"
)
DEFAULT_LAUNCH_PLAN = (
    common.TSS_OFF_RESULTS_ROOT / "launch" / "formal" / "launch_plan.json"
)
DEFAULT_STATUS = (
    common.TSS_OFF_RESULTS_ROOT / "launch" / "formal" / "supervisor_status.json"
)
DEFAULT_LOCK = (
    common.TSS_OFF_RESULTS_ROOT / "launch" / "formal" / "supervisor.lock"
)
CHECKPOINT_FILENAMES = {
    "best_miou": "best_miou.pth.tar",
    "best_pd": "best_pd.pth.tar",
}
GPU_ASSIGNMENTS = {
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


@dataclass(frozen=True)
class WorkerSpec:
    dataset: str
    wave: int
    gpu_index: str
    run_directory: Path
    log_directory: Path
    command: tuple[str, ...]
    environment: Mapping[str, str]

    @property
    def key(self) -> str:
        return f"{self.dataset}__final_tss_off"


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
        return f"{self.dataset}__final_tss_off__{self.checkpoint_role}"


def _environment(gpu_index: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    common.require(gpu_index in GPU_ASSIGNMENTS, "only physical GPUs 2 and 3 are allowed")
    environment = dict(os.environ if base is None else base)
    environment.update(CPU_ENVIRONMENT)
    environment["CUDA_VISIBLE_DEVICES"] = GPU_ASSIGNMENTS[gpu_index]["uuid"]
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return environment


def build_worker_specs(
    *,
    results_root: Path = common.TSS_OFF_RESULTS_ROOT,
    python: Path = PYTHON,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[WorkerSpec, ...]:
    layout = (
        ("NUAA-SIRST", 0, "2"),
        ("NUDT-SIRST", 0, "3"),
        ("IRSTD-1K", 1, "2"),
    )
    specs: list[WorkerSpec] = []
    for dataset, wave, gpu_index in layout:
        gpu = GPU_ASSIGNMENTS[gpu_index]
        run_dir = common.tss_off_run_directory(results_root, dataset)
        command = (
            str(python),
            str(RUNNER),
            "--dataset",
            dataset,
            "--method",
            "final",
            "--data-root",
            str(common.DATASET_ROOT),
            "--results-root",
            str(Path(results_root)),
            "--protocol-manifest",
            str(common.DATA_PROTOCOL_MANIFEST),
            "--seed",
            "42",
            "--epochs",
            "1000",
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
            "auto",
            "--tss-weight",
            "0.0",
        )
        specs.append(
            WorkerSpec(
                dataset=dataset,
                wave=wave,
                gpu_index=gpu_index,
                run_directory=run_dir,
                log_directory=(
                    Path(results_root)
                    / "launch"
                    / "formal"
                    / "logs"
                    / dataset
                    / "final_tss_off"
                    / "seed_42"
                ),
                command=command,
                environment=_environment(gpu_index, base_environment),
            )
        )
    common.require(len(specs) == 3 and len({spec.key for spec in specs}) == 3, "worker matrix differs")
    common.require(
        {spec.gpu_index for spec in specs if spec.wave == 0} == {"2", "3"},
        "wave 0 must use GPUs 2 and 3",
    )
    common.require(
        [(spec.dataset, spec.gpu_index) for spec in specs if spec.wave == 1]
        == [("IRSTD-1K", "2")],
        "wave 1 must continue IRSTD-1K on GPU 2",
    )
    return tuple(specs)


def _worker_record(spec: WorkerSpec) -> dict[str, Any]:
    return {
        "key": spec.key,
        "dataset": spec.dataset,
        "method": "final_tss_off",
        "training_model_method": "final",
        "recipe_id": "final_tss_off",
        "requested_tss_weight": 0.0,
        "tss_ratio_cap": 0.10,
        "wave": spec.wave,
        "physical_gpu_index": spec.gpu_index,
        "gpu_uuid": GPU_ASSIGNMENTS[spec.gpu_index]["uuid"],
        "run_directory": str(spec.run_directory),
        "log_directory": str(spec.log_directory),
        "command": list(spec.command),
        "environment": {
            key: spec.environment[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "PYTHONUNBUFFERED",
                "CUBLAS_WORKSPACE_CONFIG",
                *CPU_ENVIRONMENT.keys(),
            )
        },
        "seed": 42,
        "epochs": 1000,
        "eval_every": 10,
        "checkpoint_roles": list(common.CHECKPOINT_ROLES),
        "threshold": 0.5,
        "resume": "auto",
    }


def build_evaluation_specs(
    *,
    results_root: Path = common.TSS_OFF_RESULTS_ROOT,
    python: Path = PYTHON,
    base_environment: Mapping[str, str] | None = None,
) -> tuple[EvaluationSpec, ...]:
    specs: list[EvaluationSpec] = []
    role_gpu = {"best_miou": "2", "best_pd": "3"}
    for dataset in common.DATASETS:
        run_dir = common.tss_off_run_directory(results_root, dataset)
        for role in common.CHECKPOINT_ROLES:
            gpu_index = role_gpu[role]
            output = run_dir / "evaluations" / f"{role}.json"
            specs.append(
                EvaluationSpec(
                    dataset=dataset,
                    checkpoint_role=role,
                    gpu_index=gpu_index,
                    run_directory=run_dir,
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
                        str(run_dir),
                        "--dataset-root",
                        str(common.DATASET_ROOT),
                        "--data-protocol-manifest",
                        str(common.DATA_PROTOCOL_MANIFEST),
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
    common.require(len(specs) == 6, "evaluation matrix differs")
    return tuple(specs)


def _required_sources() -> dict[str, Path]:
    paths = {
        "launcher": Path(__file__).resolve(),
        "shell_entrypoint": SHELL_ENTRYPOINT,
        "runner": RUNNER,
        "evaluator_adapter": EVALUATOR,
        "evaluator_core": REPO_ROOT / "experiments" / "evaluate_three_dataset_v2.py",
        "comparator": COMPARATOR,
        "finalizer": FINALIZER,
        "preflight": Path(preflight.__file__).resolve(),
        "effective_lambda_analyzer": (
            REPO_ROOT / "experiments" / "analyze_positive_tss_effective_weights_v1.py"
        ),
        "violation_summarizer": (
            REPO_ROOT / "experiments" / "summarize_tss_violation_types_v1.py"
        ),
        "diagnostic_common": Path(common.__file__).resolve(),
        "positive_runner": (
            REPO_ROOT / "experiments" / "train_three_dataset_seed42_global_tss_v2.py"
        ),
        "training_engine": (
            REPO_ROOT / "experiments" / "train_four_dataset_original_final_seed42_exact_v1.py"
        ),
        "data_protocol": REPO_ROOT / "experiments" / "three_dataset_v2_protocol.py",
        "torch_datasets": REPO_ROOT / "experiments" / "paper_three_dataset_v2.py",
        "model_builder": REPO_ROOT / "experiments" / "four_dataset_models_seed42_v1.py",
        "training_loss": REPO_ROOT / "experiments" / "tpd_training_loss.py",
        "training_metrics_and_schedule": REPO_ROOT / "experiments" / "train_tpd_pilot.py",
        "evaluation_metric_protocol": (
            REPO_ROOT / "experiments" / "four_dataset_evaluation_protocol_v1.py"
        ),
        "protocol_document": PROTOCOL_DOCUMENT,
    }
    for path in sorted((REPO_ROOT / "model").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        paths[f"architecture::{relative}"] = path
    return dict(sorted(paths.items()))


def static_inputs(
    *,
    positive_root: Path,
    results_root: Path,
    python: Path,
    gate_result: Mapping[str, Any],
) -> dict[str, Any]:
    python_entrypoint = Path(python).absolute()
    common.require(
        python_entrypoint.is_file() and os.access(python_entrypoint, os.X_OK),
        f"Python is not executable: {python_entrypoint}",
    )
    sources = _required_sources()
    source_records = {
        name: common.artifact_record(path) for name, path in sources.items()
    }
    return {
        "python_entrypoint": str(python_entrypoint),
        "python": common.artifact_record(python_entrypoint),
        "positive_root": str(Path(positive_root).resolve(strict=True)),
        "results_root": str(Path(results_root).resolve()),
        "dataset_root": str(common.DATASET_ROOT.resolve(strict=True)),
        "data_protocol_manifest": common.artifact_record(
            common.DATA_PROTOCOL_MANIFEST
        ),
        "gate_o1": dict(gate_result),
        "sources": source_records,
    }


def prepare_launch_plan(
    *,
    positive_root: Path = common.POSITIVE_RESULTS_ROOT,
    results_root: Path = common.TSS_OFF_RESULTS_ROOT,
    python: Path = PYTHON,
    launch_plan_path: Path | None = None,
) -> dict[str, Any]:
    launch_plan_path = (
        Path(launch_plan_path)
        if launch_plan_path is not None
        else Path(results_root) / "launch" / "formal" / "launch_plan.json"
    )
    gate = preflight.prepare_gate_o1(positive_root)
    gate_binding = {
        "status": gate["status"],
        "gate_passed": gate["gate_passed"],
        "seal": gate["seal"],
        "artifacts": gate["artifacts"],
    }
    specs = build_worker_specs(results_root=results_root, python=python)
    evaluations = build_evaluation_specs(results_root=results_root, python=python)
    comparison_dir = Path(results_root) / "comparison"
    selection_dir = Path(results_root) / "selection"
    plan = {
        "schema": SCHEMA,
        "status": "prepared_not_started",
        "training_started": False,
        "formal_execution_requires_explicit_execute": True,
        "dataset_order": list(common.DATASETS),
        "execution_strategy": "two_fixed_gpu_lanes_with_automatic_continuation",
        "execution_order": [
            {
                "lane": "gpu2",
                "physical_gpu": 2,
                "sequential": True,
                "datasets": ["NUAA-SIRST", "IRSTD-1K"],
                "continuation": "IRSTD-1K starts immediately after NUAA-SIRST completes",
            },
            {
                "lane": "gpu3",
                "physical_gpu": 3,
                "sequential": True,
                "datasets": ["NUDT-SIRST"],
            },
        ],
        "lane_count": 2,
        "worker_count": 3,
        "workers": [_worker_record(spec) for spec in specs],
        "posttraining": {
            "evaluation_count": 6,
            "fixed_threshold": 0.5,
            "evaluation_commands": [
                {
                    "key": spec.key,
                    "dataset": spec.dataset,
                    "method": "final_tss_off",
                    "training_model_method": "final",
                    "requested_tss_weight": 0.0,
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
                "command": [
                    str(python),
                    str(COMPARATOR),
                    "--positive-root",
                    str(Path(positive_root)),
                    "--tss-off-root",
                    str(Path(results_root)),
                    "--tss-off-launch-plan",
                    str(launch_plan_path),
                    "--output-dir",
                    str(comparison_dir),
                ],
                "output": str(comparison_dir / "tss_off_comparison_v1.json"),
            },
            "finalize": {
                "command": [
                    str(python),
                    str(FINALIZER),
                    "--positive-root",
                    str(Path(positive_root)),
                    "--tss-off-root",
                    str(Path(results_root)),
                    "--comparison-dir",
                    str(comparison_dir),
                    "--output-dir",
                    str(selection_dir),
                ],
                "output": str(selection_dir / "tss_off_diagnostic_v1.json"),
            },
        },
        "static_inputs": static_inputs(
            positive_root=positive_root,
            results_root=results_root,
            python=python,
            gate_result=gate_binding,
        ),
    }
    action = common.write_once_or_identical(launch_plan_path, plan)
    return {
        "status": "complete",
        "action": action,
        "launch_plan": common.artifact_record(launch_plan_path),
        "plan": plan,
    }


def _verify_static_inputs(plan: Mapping[str, Any]) -> None:
    expected = plan.get("static_inputs")
    common.require(isinstance(expected, dict), "launch plan lacks static inputs")
    for name, record in expected.get("sources", {}).items():
        common.require(isinstance(record, dict), f"malformed source lock: {name}")
        path = Path(str(record.get("path", "")))
        common.require(common.artifact_record(path) == record, f"source changed after prepare: {name}")
    for label in ("python", "data_protocol_manifest"):
        record = expected.get(label)
        common.require(isinstance(record, dict), f"launch plan lacks {label}")
        common.require(common.artifact_record(Path(record["path"])) == record, f"{label} changed after prepare")
    python_entrypoint = Path(str(expected.get("python_entrypoint", "")))
    common.require(
        python_entrypoint.is_file()
        and os.access(python_entrypoint, os.X_OK)
        and python_entrypoint.resolve(strict=True)
        == Path(expected["python"]["path"]).resolve(strict=True),
        "Python entrypoint changed after prepare",
    )
    gate = expected.get("gate_o1")
    common.require(isinstance(gate, dict) and gate.get("gate_passed") is True, "Gate O1 is not passed")
    for record in gate.get("artifacts", {}).values():
        common.require(common.artifact_record(Path(record["path"])) == record, "Gate O1 artifact changed")
    common.require(common.artifact_record(Path(gate["seal"]["path"])) == gate["seal"], "Gate O1 seal changed")


def _validate_complete_run(spec: WorkerSpec) -> dict[str, Any] | None:
    summary_path = spec.run_directory / "summary.json"
    if not summary_path.is_file():
        return None
    summary = common.load_json(summary_path)
    expected_recipe = {
        "method": "final",
        "recipe_id": "final_tss_off",
        "requested_tss_weight": 0.0,
        "tss_enabled": False,
        "tss_lambda_token": "off",
        "tss_ratio_cap": 0.10,
        "tss_ratio_cap_applied": False,
        "tss_heads_registered": True,
        "tss_training_forward_computes_logits": True,
        "tss_loss_consumes_logits": False,
        "tss_survival_target_constructed": False,
    }
    for field, expected in (
        ("schema", TRAINER_SCHEMA),
        ("status", "complete"),
        ("dataset", spec.dataset),
        ("method", "final"),
        ("seed", 42),
        ("epochs", 1000),
        ("requested_tss_weight", 0.0),
        ("recipe", expected_recipe),
    ):
        common.require(summary.get(field) == expected, f"completed {spec.key} {field} differs")
    protocol_path = spec.run_directory / "protocol.json"
    protocol = common.load_json(protocol_path)
    common.require(protocol.get("schema") == TRAINER_SCHEMA, f"{spec.key} protocol schema differs")
    common.require(protocol.get("recipe") == expected_recipe, f"{spec.key} protocol recipe differs")
    common.require(protocol.get("training_seed") == 42, f"{spec.key} protocol seed differs")
    common.require(protocol.get("epochs") == 1000, f"{spec.key} protocol epochs differ")
    common.require(protocol.get("eval_every") == 10, f"{spec.key} eval cadence differs")
    common.require(protocol.get("begin_test") == 10, f"{spec.key} eval start differs")
    declared = protocol.get("protocol_sha256")
    unsigned = dict(protocol)
    del unsigned["protocol_sha256"]
    common.require(common.compact_sha256(unsigned) == declared, f"{spec.key} protocol SHA differs")
    common.require(summary.get("protocol_sha256") == declared, f"{spec.key} summary protocol SHA differs")
    runtime_sources = protocol.get("runtime_sources")
    common.require(isinstance(runtime_sources, dict) and bool(runtime_sources), f"{spec.key} lacks source lock")
    for name, record in runtime_sources.items():
        common.require(isinstance(record, dict), f"{spec.key} source {name} is malformed")
        common.require(
            common.file_sha256(Path(str(record.get("path", ""))))
            == record.get("sha256"),
            f"{spec.key} source {name} changed",
        )
    checkpoints = summary.get("checkpoints")
    common.require(isinstance(checkpoints, dict) and set(checkpoints) == set(common.CHECKPOINT_ROLES), f"{spec.key} checkpoints differ")
    for role, record in checkpoints.items():
        common.require(isinstance(record, dict), f"{spec.key} {role} checkpoint record is malformed")
        path = Path(str(record.get("path", "")))
        common.require(common.file_sha256(path) == record.get("sha256"), f"{spec.key} {role} checkpoint SHA differs")
    common.require(not (spec.run_directory / "resume" / "latest_training_state.pth.tar").exists(), f"{spec.key} retained rolling resume state")
    return summary


def _write_status(status_path: Path, payload: Mapping[str, Any]) -> None:
    common.atomic_write_json(status_path, payload)


def _run_lane(specs: Sequence[WorkerSpec]) -> list[str]:
    """Run one fixed-GPU lane sequentially and return any worker failures."""

    failures: list[str] = []
    for spec in specs:
        if _validate_complete_run(spec) is not None:
            continue
        spec.log_directory.mkdir(parents=True, exist_ok=True)
        log_path = spec.log_directory / "train.log"
        handle = log_path.open("ab")
        process = subprocess.Popen(
            list(spec.command),
            cwd=REPO_ROOT,
            env=dict(spec.environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        try:
            returncode = process.wait()
        finally:
            handle.close()
        if returncode != 0:
            failures.append(f"{spec.key}:exit_{returncode}")
            break
        common.require(
            _validate_complete_run(spec) is not None,
            f"{spec.key} did not complete",
        )
    return failures


def _run_training_lanes(
    specs: Sequence[WorkerSpec],
    *,
    status_path: Path,
    plan_record: Mapping[str, Any],
) -> None:
    """Keep GPU2 and GPU3 independent so IRSTD need not wait for NUDT."""

    lanes = {
        "2": tuple(sorted(
            (spec for spec in specs if spec.gpu_index == "2"),
            key=lambda spec: spec.wave,
        )),
        "3": tuple(sorted(
            (spec for spec in specs if spec.gpu_index == "3"),
            key=lambda spec: spec.wave,
        )),
    }
    common.require(
        [[spec.dataset for spec in lanes[gpu]] for gpu in ("2", "3")]
        == [["NUAA-SIRST", "IRSTD-1K"], ["NUDT-SIRST"]],
        "training lanes differ from the frozen GPU2/3 schedule",
    )
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            gpu: executor.submit(_run_lane, lane)
            for gpu, lane in lanes.items()
        }
        for gpu, future in futures.items():
            try:
                failures.extend(future.result())
            except Exception as exc:
                failures.append(f"gpu{gpu}_lane:{type(exc).__name__}:{exc}")
    _write_status(
        status_path,
        {
            "schema": SCHEMA,
            "status": "training_failed" if failures else "training_complete",
            "launch_plan": dict(plan_record),
            "execution_strategy": "two_fixed_gpu_lanes_with_automatic_continuation",
            "lanes": {
                f"gpu{gpu}": [spec.key for spec in lane]
                for gpu, lane in lanes.items()
            },
            "failures": failures,
        },
    )
    common.require(not failures, f"training lanes failed: {failures}")
    for spec in specs:
        common.require(
            _validate_complete_run(spec) is not None,
            f"{spec.key} did not complete",
        )


def _validate_evaluation(spec: EvaluationSpec) -> dict[str, Any] | None:
    if not spec.output_path.is_file():
        return None
    payload = common.load_json(spec.output_path)
    for field, expected in (
        ("schema", "sctransnet_three_dataset_v2_evaluation_v1"),
        ("status", "complete"),
        ("dataset", spec.dataset),
        ("method", "final_tss_off"),
        ("training_model_method", "final"),
        ("checkpoint_role", spec.checkpoint_role),
        ("requested_tss_weight", 0.0),
    ):
        common.require(payload.get(field) == expected, f"evaluation {spec.key} {field} differs")
    fixed = payload.get("fixed_threshold_0_5")
    common.require(isinstance(fixed, dict) and fixed.get("threshold") == 0.5, f"evaluation {spec.key} fixed point differs")
    adapter = payload.get("tss_off_evaluator_adapter")
    common.require(isinstance(adapter, dict), f"evaluation {spec.key} lacks adapter lock")
    common.require(
        adapter.get("core_evaluator_sha256")
        == common.file_sha256(REPO_ROOT / "experiments" / "evaluate_three_dataset_v2.py"),
        f"evaluation {spec.key} evaluator core SHA differs",
    )
    common.require(
        adapter.get("adapter_sha256") == common.file_sha256(EVALUATOR),
        f"evaluation {spec.key} adapter SHA differs",
    )
    determinism_source = (
        REPO_ROOT
        / "experiments"
        / "train_four_dataset_original_final_seed42_exact_v1.py"
    )
    common.require(
        adapter.get("training_determinism_contract_reapplied") is True,
        f"evaluation {spec.key} omitted the training determinism contract",
    )
    common.require(
        adapter.get("determinism_source_sha256")
        == common.file_sha256(determinism_source),
        f"evaluation {spec.key} determinism-source SHA differs",
    )
    binding = payload.get("checkpoint_binding")
    common.require(isinstance(binding, dict), f"evaluation {spec.key} lacks checkpoint binding")
    common.require(
        binding.get("run_dir") == str(spec.run_directory.resolve()),
        f"evaluation {spec.key} run binding differs",
    )
    bound_files = {
        "summary": spec.run_directory / "summary.json",
        "protocol": spec.run_directory / "protocol.json",
        "checkpoint": (
            spec.run_directory
            / "checkpoints"
            / CHECKPOINT_FILENAMES[spec.checkpoint_role]
        ),
    }
    for label, expected_path in bound_files.items():
        record = binding.get(label)
        common.require(
            isinstance(record, dict),
            f"evaluation {spec.key} lacks bound {label}",
        )
        common.require(
            Path(str(record.get("path", ""))).resolve() == expected_path.resolve(),
            f"evaluation {spec.key} bound {label} path differs",
        )
        common.require(
            record.get("sha256") == common.file_sha256(expected_path),
            f"evaluation {spec.key} bound {label} SHA differs",
        )
    checkpoint_record = binding["checkpoint"]
    common.require(
        checkpoint_record.get("role") == spec.checkpoint_role,
        f"evaluation {spec.key} bound checkpoint role differs",
    )
    return payload


def _run_evaluations(specs: Sequence[EvaluationSpec]) -> None:
    for dataset in common.DATASETS:
        dataset_specs = [spec for spec in specs if spec.dataset == dataset]
        active: list[tuple[EvaluationSpec, subprocess.Popen[bytes], Any]] = []
        for spec in dataset_specs:
            if _validate_evaluation(spec) is not None:
                continue
            spec.log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = spec.log_path.open("ab")
            process = subprocess.Popen(
                list(spec.command),
                cwd=REPO_ROOT,
                env=dict(spec.environment),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active.append((spec, process, handle))
        failures: list[str] = []
        for spec, process, handle in active:
            returncode = process.wait()
            handle.close()
            if returncode != 0:
                failures.append(f"{spec.key}:exit_{returncode}")
        common.require(not failures, f"evaluation failed: {failures}")
        for spec in dataset_specs:
            common.require(_validate_evaluation(spec) is not None, f"evaluation {spec.key} incomplete")


def _run_postprocessor(
    command: Sequence[str],
    output: Path,
    log_path: Path,
    *,
    validator: Any,
) -> None:
    if output.is_file():
        validator(output)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    common.require(completed.returncode == 0, f"postprocessor failed with exit {completed.returncode}")
    common.require(output.is_file(), f"postprocessor did not create {output}")
    validator(output)


def _validate_comparison_output(
    path: Path,
    *,
    positive_root: Path,
    results_root: Path,
) -> dict[str, Any]:
    payload = common.load_json(path)
    finalizer.validate_comparison(payload)
    finalizer.validate_comparison_bindings(
        payload,
        positive_root=positive_root,
        tss_off_root=results_root,
    )
    return payload


def _validate_final_output(path: Path, *, comparison_path: Path) -> dict[str, Any]:
    payload = common.load_json(path)
    comparison = common.load_json(comparison_path)
    expected = finalizer.build_final(comparison, comparison_path=comparison_path)
    common.require(
        payload == expected,
        "existing final diagnostic differs from the strictly validated comparison",
    )
    return payload


def execute_launch_plan(
    *,
    launch_plan_path: Path = DEFAULT_LAUNCH_PLAN,
    status_path: Path = DEFAULT_STATUS,
    lock_path: Path = DEFAULT_LOCK,
) -> None:
    plan = common.load_json(launch_plan_path)
    common.require(plan.get("schema") == SCHEMA, "launch-plan schema differs")
    common.require(plan.get("status") == "prepared_not_started", "launch plan is not prepared")
    _verify_static_inputs(plan)
    results_root = Path(plan["static_inputs"]["results_root"])
    positive_root = Path(plan["static_inputs"]["positive_root"])
    python = Path(plan["static_inputs"]["python_entrypoint"])
    specs = build_worker_specs(results_root=results_root, python=python)
    common.require([_worker_record(spec) for spec in specs] == plan.get("workers"), "worker commands differ from plan")
    evaluation_specs = build_evaluation_specs(results_root=results_root, python=python)
    plan_record = common.artifact_record(launch_plan_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise common.TSSOffDiagnosticError("another TSS-off supervisor holds the lock") from exc
        _write_status(
            status_path,
            {
                "schema": SCHEMA,
                "status": "training",
                "launch_plan": plan_record,
                "training_started": True,
            },
        )
        _verify_static_inputs(plan)
        _run_training_lanes(
            specs,
            status_path=status_path,
            plan_record=plan_record,
        )
        _verify_static_inputs(plan)
        _write_status(
            status_path,
            {
                "schema": SCHEMA,
                "status": "evaluating",
                "launch_plan": plan_record,
                "training_started": True,
                "training_complete": True,
            },
        )
        _run_evaluations(evaluation_specs)
        comparison = plan["posttraining"]["comparison"]
        finalize = plan["posttraining"]["finalize"]
        _run_postprocessor(
            comparison["command"],
            Path(comparison["output"]),
            results_root / "launch" / "formal" / "comparison.log",
            validator=lambda path: _validate_comparison_output(
                path,
                positive_root=positive_root,
                results_root=results_root,
            ),
        )
        _run_postprocessor(
            finalize["command"],
            Path(finalize["output"]),
            results_root / "launch" / "formal" / "finalize.log",
            validator=lambda path: _validate_final_output(
                path,
                comparison_path=Path(comparison["output"]),
            ),
        )
        _write_status(
            status_path,
            {
                "schema": SCHEMA,
                "status": "complete",
                "launch_plan": plan_record,
                "training_started": True,
                "training_complete": True,
                "evaluation_complete": True,
                "comparison": common.artifact_record(Path(comparison["output"])),
                "final_diagnostic": common.artifact_record(Path(finalize["output"])),
            },
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--positive-root", type=Path, default=common.POSITIVE_RESULTS_ROOT)
    parser.add_argument("--results-root", type=Path, default=common.TSS_OFF_RESULTS_ROOT)
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
            positive_root=args.positive_root,
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
                    "worker_count": 3,
                    "lane_count": 2,
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
