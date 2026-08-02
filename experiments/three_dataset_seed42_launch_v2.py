#!/usr/bin/env python3
"""Prepare and optionally execute the exact 12-run V2 training matrix.

The default action is a dry run: prepare immutable launch artifacts, perform
one complete indexed-pair audit, and write the 12 commands without starting a
training process.  ``--execute`` is required to run the six two-GPU waves.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import three_dataset_v2_protocol as data_protocol


SCHEMA = "sctransnet_three_dataset_seed42_global_tss_launcher_v2/v1"
PAIR_AUDIT_SCHEMA = "sctransnet_three_dataset_v2_pair_audit/v1"
TSS_STATISTICS_SCHEMA = "sctransnet_three_dataset_v2_tss_statistics/v1"
TRAINING_SEED = 42
DATASETS = data_protocol.DATASETS
METHODS = ("original", "final")
TSS_LAMBDAS = (0.0025, 0.005, 0.01)
CHECKPOINT_ROLES = ("best_miou", "best_pd")

PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
RUNNER = REPO_ROOT / "experiments" / (
    "train_three_dataset_seed42_global_tss_v2.py"
)
CORE_TRAINING_SOURCES = {
    "runner": RUNNER,
    "training_engine": REPO_ROOT
    / "experiments"
    / "train_four_dataset_original_final_seed42_exact_v1.py",
    "data_protocol": REPO_ROOT / "experiments" / "three_dataset_v2_protocol.py",
    "torch_datasets": REPO_ROOT / "experiments" / "paper_three_dataset_v2.py",
    "model_builder": REPO_ROOT
    / "experiments"
    / "four_dataset_models_seed42_v1.py",
    "training_loss": REPO_ROOT / "experiments" / "tpd_training_loss.py",
    "training_metrics_and_schedule": REPO_ROOT
    / "experiments"
    / "train_tpd_pilot.py",
    "protocol_document": REPO_ROOT
    / "SCTransNet_V2全数据集混合结果复盘与全局TSS配方定型方案.md",
}
TRAINER_SCHEMA = "sctransnet_three_dataset_seed42_global_tss_v2/v1"
DATA_ROOT = REPO_ROOT / "datasets"
RESULTS_ROOT = REPO_ROOT / "results" / (
    "three_dataset_seed42_global_tss_v2"
)
ARTIFACT_ROOT = REPO_ROOT / "results" / "three_dataset_v2" / "manifests"
PROTOCOL_MANIFEST = ARTIFACT_ROOT / "three_dataset_v2_protocol.json"
TSS_STATISTICS = ARTIFACT_ROOT / "three_dataset_v2_tss_statistics.json"
PAIR_AUDIT = ARTIFACT_ROOT / "three_dataset_v2_pair_audit.json"
LAUNCH_PLAN = RESULTS_ROOT / "launch" / "formal" / "launch_plan.json"
SUPERVISOR_LOCK = RESULTS_ROOT / "launch" / "formal" / "supervisor.lock"
SUPERVISOR_STATUS = RESULTS_ROOT / "launch" / "formal" / "supervisor_status.json"
TSS_PROVENANCE_SOURCE = (
    REPO_ROOT
    / "results"
    / "four_dataset_seed42_v1"
    / "manifests"
    / "four_dataset_tss_seed42_v1.json"
)

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
CPU_THREAD_ENV = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "OMP_WAIT_POLICY": "PASSIVE",
    "KMP_BLOCKTIME": "0",
}

# Exact three-source records extracted after confirming that the historical
# train IDs, order hashes, seed, epoch count, and stateless crop derivation are
# identical to the new V2 protocol.  The legacy artifact is provenance only;
# it is not a runtime input to the 12-run launcher or trainer.
TSS_PROVENANCE_SOURCE_SHA256 = (
    "885403ce46dff8f8b74a67237c225d2f11786c1b7e3dff9115bfaad081e753ae"
)
TSS_RECORDS: dict[str, dict[str, Any]] = {
    "NUAA-SIRST": {
        "dataset": "NUAA-SIRST",
        "training_seed": 42,
        "epochs": 1000,
        "completed_through_epoch": 1000,
        "complete": True,
        "survival_pos_weight": 132.78674792798364,
        "positive_cells": 407574,
        "negative_cells": 54120426,
        "aggregate_plan_sha256": (
            "5887f035b38f0dc36e4afb9d73b7ee936109ebe87f5213a1d2e48efde8cf0594"
        ),
        "train_ids_sha256": data_protocol.EXPECTED_SPLITS["NUAA-SIRST"][
            "train"
        ]["ordered_ids_sha256"],
        "survival_pos_weight_formula": "negative_cells / positive_cells",
    },
    "NUDT-SIRST": {
        "dataset": "NUDT-SIRST",
        "training_seed": 42,
        "epochs": 1000,
        "completed_through_epoch": 1000,
        "complete": True,
        "survival_pos_weight": 100.08874329958309,
        "positive_cells": 1679000,
        "negative_cells": 168049000,
        "aggregate_plan_sha256": (
            "97c53b5dd76b291470dae8d0f7fbe4020cbdc67218adcf57564cb925bee0bd95"
        ),
        "train_ids_sha256": data_protocol.EXPECTED_SPLITS["NUDT-SIRST"][
            "train"
        ]["ordered_ids_sha256"],
        "survival_pos_weight_formula": "negative_cells / positive_cells",
    },
    "IRSTD-1K": {
        "dataset": "IRSTD-1K",
        "training_seed": 42,
        "epochs": 1000,
        "completed_through_epoch": 1000,
        "complete": True,
        "survival_pos_weight": 169.36752967913839,
        "positive_cells": 1202107,
        "negative_cells": 203597893,
        "aggregate_plan_sha256": (
            "256202eb0b38aa9d761364d5bb2b91aac4bb94695b53f467e7487913899b946d"
        ),
        "train_ids_sha256": data_protocol.EXPECTED_SPLITS["IRSTD-1K"][
            "train"
        ]["ordered_ids_sha256"],
        "survival_pos_weight_formula": "negative_cells / positive_cells",
    },
}


class LaunchProtocolError(RuntimeError):
    """A launch input, preflight record, or worker result is invalid."""


@dataclass(frozen=True)
class WorkerSpec:
    dataset: str
    method: str
    requested_tss_weight: float
    global_wave: int
    dataset_wave: int
    gpu_index: str
    command: tuple[str, ...]
    environment: Mapping[str, str]
    run_directory: Path
    log_directory: Path

    @property
    def key(self) -> str:
        if self.method == "original":
            recipe = "original"
        else:
            recipe = f"final_lambda_{lambda_token(self.requested_tss_weight)}"
        return f"{self.dataset}__{recipe}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_sources() -> dict[str, Path]:
    sources = dict(CORE_TRAINING_SOURCES)
    for path in sorted((REPO_ROOT / "model").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        sources[f"architecture::{relative}"] = path
    return sources


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            dict(payload),
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=str,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def lambda_token(value: float) -> str:
    mapping = {0.0025: "0p0025", 0.005: "0p005", 0.01: "0p01"}
    if value not in mapping:
        raise LaunchProtocolError(f"unsupported TSS weight: {value!r}")
    return mapping[value]


def build_tss_statistics_payload() -> dict[str, Any]:
    records = json.loads(json.dumps(TSS_RECORDS))
    for dataset in DATASETS:
        record = records[dataset]
        expected_order = data_protocol.EXPECTED_SPLITS[dataset]["train"][
            "ordered_ids_sha256"
        ]
        if record["train_ids_sha256"] != expected_order:
            raise LaunchProtocolError(
                f"TSS train-ID identity differs for {dataset}"
            )
        expected_weight = record["negative_cells"] / record["positive_cells"]
        if record["survival_pos_weight"] != expected_weight:
            raise LaunchProtocolError(
                f"TSS pos_weight arithmetic differs for {dataset}"
            )
    payload = {
        "schema": TSS_STATISTICS_SCHEMA,
        "training_seed": TRAINING_SEED,
        "epochs": 1000,
        "datasets": list(DATASETS),
        "records": records,
        "provenance": {
            "source_artifact_sha256": TSS_PROVENANCE_SOURCE_SHA256,
            "extraction": "exact_three_dataset_records_only",
            "reused_after_exact_three_train_identity_validation": True,
            "runtime_depends_on_source_artifact": False,
        },
    }
    if "SIRST3" in json.dumps(payload, ensure_ascii=False):
        raise LaunchProtocolError("compact TSS artifact is outside V2 scope")
    return payload


def prepare_tss_statistics(path: Path = TSS_STATISTICS) -> dict[str, Any]:
    if not TSS_PROVENANCE_SOURCE.is_file():
        raise LaunchProtocolError(
            f"TSS provenance source is missing: {TSS_PROVENANCE_SOURCE}"
        )
    observed_source_sha = file_sha256(TSS_PROVENANCE_SOURCE)
    if observed_source_sha != TSS_PROVENANCE_SOURCE_SHA256:
        raise LaunchProtocolError(
            "TSS provenance source SHA-256 differs: "
            f"{observed_source_sha} != {TSS_PROVENANCE_SOURCE_SHA256}"
        )
    source = json.loads(TSS_PROVENANCE_SOURCE.read_text(encoding="utf-8"))
    if source.get("training_seed") != 42 or source.get("epochs") != 1000:
        raise LaunchProtocolError("TSS provenance seed/epochs differ")
    source_records = source.get("datasets")
    if not isinstance(source_records, Mapping):
        raise LaunchProtocolError("TSS provenance has no dataset records")
    for dataset, expected_record in TSS_RECORDS.items():
        observed_record = source_records.get(dataset)
        if not isinstance(observed_record, Mapping):
            raise LaunchProtocolError(
                f"TSS provenance lacks record for {dataset}"
            )
        for field, expected in expected_record.items():
            if observed_record.get(field) != expected:
                raise LaunchProtocolError(
                    f"TSS provenance {dataset} {field} differs"
                )
    expected = build_tss_statistics_payload()
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise LaunchProtocolError(
                f"existing compact TSS artifact differs: {path}"
            )
    else:
        write_json_atomic(path, expected)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "schema": TSS_STATISTICS_SCHEMA,
        "datasets": list(DATASETS),
    }


def _tiny_component_count(mask_path: Path, area_limit: int = 9) -> int:
    """Count 8-connected positive components without a numeric dependency."""

    with Image.open(mask_path) as image:
        mask = image.convert("L")
        width, height = mask.size
        positive = {
            index
            for index, value in enumerate(mask.getdata())
            if int(value) > 127
        }
    tiny = 0
    while positive:
        first = positive.pop()
        stack = [first]
        area = 0
        while stack:
            current = stack.pop()
            area += 1
            row, column = divmod(current, width)
            for delta_row in (-1, 0, 1):
                next_row = row + delta_row
                if next_row < 0 or next_row >= height:
                    continue
                for delta_column in (-1, 0, 1):
                    if delta_row == 0 and delta_column == 0:
                        continue
                    next_column = column + delta_column
                    if next_column < 0 or next_column >= width:
                        continue
                    neighbor = next_row * width + next_column
                    if neighbor in positive:
                        positive.remove(neighbor)
                        stack.append(neighbor)
        if area <= area_limit:
            tiny += 1
    return tiny


def audit_all_indexed_pairs(
    *,
    dataset_root: Path = DATA_ROOT,
    manifest_path: Path = PROTOCOL_MANIFEST,
    output_path: Path | None = PAIR_AUDIT,
) -> dict[str, Any]:
    """Validate all 2,755 indexed pairs once before any worker is started."""

    manifest = data_protocol.load_protocol_manifest(
        manifest_path, dataset_root=dataset_root
    )
    manifest_sha = file_sha256(manifest_path)
    records: dict[str, Any] = {}
    pair_total = 0
    tiny_test_total = 0
    for dataset in DATASETS:
        split_records: dict[str, Any] = {}
        for split in data_protocol.SPLITS:
            identifiers = data_protocol.load_frozen_index(
                dataset_root, dataset, split, manifest
            )
            known = frozenset(identifiers)
            correction_count = 0
            tiny_count = 0
            for sample_id in identifiers:
                sample = data_protocol.resolve_sample(
                    dataset_root,
                    dataset,
                    sample_id,
                    split=split,
                    known_ids=known,
                )
                data_protocol.validate_sample_pair(sample)
                correction_count += int(sample.correction_applied)
                if split == "test":
                    tiny_count += _tiny_component_count(sample.mask_path)
            count = len(identifiers)
            pair_total += count
            if split == "test":
                tiny_test_total += tiny_count
            split_records[split] = {
                "pair_count": count,
                "missing_image_count": 0,
                "missing_effective_mask_count": 0,
                "size_mismatch_count": 0,
                "correction_applied_count": correction_count,
                "tiny_gt_component_count_area_le_9": (
                    tiny_count if split == "test" else None
                ),
                "tiny_metric_defined": (
                    tiny_count > 0 if split == "test" else None
                ),
            }
        records[dataset] = split_records
    expected_total = sum(
        int(data_protocol.EXPECTED_SPLITS[dataset][split]["count"])
        for dataset in DATASETS
        for split in data_protocol.SPLITS
    )
    if pair_total != expected_total:
        raise LaunchProtocolError(
            f"pair audit count differs: {pair_total} != {expected_total}"
        )
    payload = {
        "schema": PAIR_AUDIT_SCHEMA,
        "status": "complete",
        "dataset_root": str(dataset_root.resolve()),
        "protocol_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha,
            "schema": data_protocol.SCHEMA,
            "manifest_id": data_protocol.MANIFEST_ID,
        },
        "datasets": records,
        "pair_count": pair_total,
        "error_count": 0,
        "tiny_gt_component_count_area_le_9_test_total": tiny_test_total,
        "connectivity": 8,
        "tiny_area_limit": 9,
    }
    if output_path is not None:
        write_json_atomic(output_path, payload)
    return payload


def run_directory(
    dataset: str, method: str, requested_tss_weight: float
) -> Path:
    data_protocol.require_dataset(dataset)
    if method == "original":
        if requested_tss_weight != 0.0:
            raise LaunchProtocolError("Original must use zero TSS weight")
        branch = Path("original")
    elif method == "final":
        branch = Path("final") / f"lambda_{lambda_token(requested_tss_weight)}"
    else:
        raise LaunchProtocolError(f"unsupported method: {method!r}")
    return RESULTS_ROOT / "runs" / dataset / branch / "seed_42"


def build_worker_spec(
    dataset: str,
    method: str,
    requested_tss_weight: float,
    *,
    global_wave: int,
    dataset_wave: int,
    gpu_index: str,
    base_environment: Mapping[str, str] | None = None,
) -> WorkerSpec:
    data_protocol.require_dataset(dataset)
    if gpu_index not in GPU_ASSIGNMENTS:
        raise LaunchProtocolError("workers may use only physical GPU 2 or 3")
    if method == "original":
        if requested_tss_weight != 0.0:
            raise LaunchProtocolError("Original must use zero TSS weight")
    elif method == "final":
        lambda_token(requested_tss_weight)
    else:
        raise LaunchProtocolError(f"unsupported method: {method!r}")
    gpu = GPU_ASSIGNMENTS[gpu_index]
    command = [
        str(PYTHON),
        str(RUNNER),
        "--dataset",
        dataset,
        "--method",
        method,
        "--data-root",
        str(DATA_ROOT),
        "--results-root",
        str(RESULTS_ROOT),
        "--protocol-manifest",
        str(PROTOCOL_MANIFEST),
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
    ]
    if method == "final":
        command.extend(
            [
                "--tss-weight",
                str(requested_tss_weight),
                "--tss-statistics",
                str(TSS_STATISTICS),
            ]
        )
    environment = dict(
        os.environ if base_environment is None else base_environment
    )
    environment["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment.update(CPU_THREAD_ENV)
    run_dir = run_directory(dataset, method, requested_tss_weight)
    return WorkerSpec(
        dataset=dataset,
        method=method,
        requested_tss_weight=requested_tss_weight,
        global_wave=global_wave,
        dataset_wave=dataset_wave,
        gpu_index=gpu_index,
        command=tuple(command),
        environment=environment,
        run_directory=run_dir,
        log_directory=(
            RESULTS_ROOT / "launch" / "formal" / "logs" / run_dir.relative_to(
                RESULTS_ROOT / "runs"
            )
        ),
    )


def build_all_worker_specs(
    *, base_environment: Mapping[str, str] | None = None
) -> tuple[WorkerSpec, ...]:
    """Return three dataset-local two-wave groups, exactly 12 workers."""

    specs: list[WorkerSpec] = []
    for dataset_index, dataset in enumerate(DATASETS):
        first_global_wave = dataset_index * 2
        specs.extend(
            [
                build_worker_spec(
                    dataset,
                    "original",
                    0.0,
                    global_wave=first_global_wave,
                    dataset_wave=0,
                    gpu_index="2",
                    base_environment=base_environment,
                ),
                build_worker_spec(
                    dataset,
                    "final",
                    0.0025,
                    global_wave=first_global_wave,
                    dataset_wave=0,
                    gpu_index="3",
                    base_environment=base_environment,
                ),
                build_worker_spec(
                    dataset,
                    "final",
                    0.005,
                    global_wave=first_global_wave + 1,
                    dataset_wave=1,
                    gpu_index="2",
                    base_environment=base_environment,
                ),
                build_worker_spec(
                    dataset,
                    "final",
                    0.01,
                    global_wave=first_global_wave + 1,
                    dataset_wave=1,
                    gpu_index="3",
                    base_environment=base_environment,
                ),
            ]
        )
    if len(specs) != 12 or len({spec.key for spec in specs}) != 12:
        raise LaunchProtocolError("launcher did not build exactly 12 unique runs")
    for wave in range(6):
        members = [spec for spec in specs if spec.global_wave == wave]
        if len(members) != 2 or {spec.gpu_index for spec in members} != {"2", "3"}:
            raise LaunchProtocolError(f"wave {wave} is not one worker per GPU")
    return tuple(specs)


def command_record(spec: WorkerSpec) -> dict[str, Any]:
    return {
        "key": spec.key,
        "dataset": spec.dataset,
        "method": spec.method,
        "requested_tss_weight": spec.requested_tss_weight,
        "global_wave": spec.global_wave,
        "dataset_wave": spec.dataset_wave,
        "physical_gpu_index": spec.gpu_index,
        "gpu_uuid": GPU_ASSIGNMENTS[spec.gpu_index]["uuid"],
        "command": list(spec.command),
        "run_directory": str(spec.run_directory),
        "log_directory": str(spec.log_directory),
        "checkpoint_roles": list(CHECKPOINT_ROLES),
        "threshold": 0.5,
        "seed": 42,
        "epochs": 1000,
        "eval_every": 10,
    }


def validate_static_inputs() -> dict[str, Any]:
    current_sources = training_sources()
    missing = [
        str(path)
        for path in (
            PYTHON,
            DATA_ROOT,
            PROTOCOL_MANIFEST,
            *current_sources.values(),
        )
        if not path.exists()
    ]
    if missing:
        raise LaunchProtocolError(f"missing launch inputs: {missing}")
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK):
        raise LaunchProtocolError(f"Python is not executable: {PYTHON}")
    manifest = data_protocol.load_protocol_manifest(
        PROTOCOL_MANIFEST, dataset_root=DATA_ROOT
    )
    return {
        "python": str(PYTHON),
        "python_sha256": file_sha256(PYTHON),
        "training_sources": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in current_sources.items()
        },
        "data_root": str(DATA_ROOT),
        "protocol_manifest": str(PROTOCOL_MANIFEST),
        "protocol_manifest_sha256": file_sha256(PROTOCOL_MANIFEST),
        "protocol_manifest_id": manifest["manifest_id"],
    }


def verify_static_inputs(expected: Mapping[str, Any]) -> None:
    observed = validate_static_inputs()
    if observed != dict(expected):
        raise LaunchProtocolError(
            "training inputs changed after the 12-run plan was prepared"
        )


def verify_prepared_artifacts(plan: Mapping[str, Any]) -> None:
    tss = plan.get("tss_statistics")
    pair = plan.get("pair_audit")
    if not isinstance(tss, Mapping) or not isinstance(pair, Mapping):
        raise LaunchProtocolError("prepared plan lacks artifact locks")
    for label, path, record in (
        ("TSS statistics", TSS_STATISTICS, tss),
        ("pair audit", PAIR_AUDIT, pair),
    ):
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise LaunchProtocolError(f"prepared {label} changed")


def prepare_launch_plan() -> dict[str, Any]:
    static = validate_static_inputs()
    tss = prepare_tss_statistics()
    pair_audit = audit_all_indexed_pairs()
    specs = build_all_worker_specs()
    plan = {
        "schema": SCHEMA,
        "status": "prepared_not_started",
        "training_started": False,
        "dataset_order": list(DATASETS),
        "execution_order": (
            "finish both waves for one dataset before advancing to the next"
        ),
        "wave_count": 6,
        "worker_count": len(specs),
        "original_run_count": 3,
        "final_run_count": 9,
        "per_run_schedule_data_evaluation_matched": True,
        "total_search_budget_equal": False,
        "final_to_original_run_budget_ratio": 3.0,
        "static_inputs": static,
        "tss_statistics": tss,
        "pair_audit": {
            "path": str(PAIR_AUDIT.resolve()),
            "sha256": file_sha256(PAIR_AUDIT),
            "pair_count": pair_audit["pair_count"],
            "error_count": pair_audit["error_count"],
            "tiny_gt_component_count_area_le_9_test_total": pair_audit[
                "tiny_gt_component_count_area_le_9_test_total"
            ],
        },
        "workers": [command_record(spec) for spec in specs],
    }
    write_json_atomic(LAUNCH_PLAN, plan)
    return plan


def _expected_recipe(spec: WorkerSpec) -> dict[str, Any]:
    if spec.method == "original":
        return {
            "method": "original",
            "recipe_id": "original_no_tss",
            "requested_tss_weight": 0.0,
            "tss_lambda_token": None,
            "tss_ratio_cap": None,
            "tss_enabled": False,
        }
    token = lambda_token(spec.requested_tss_weight)
    return {
        "method": "final",
        "recipe_id": f"final_lambda_{token}",
        "requested_tss_weight": spec.requested_tss_weight,
        "tss_lambda_token": token,
        "tss_ratio_cap": 0.10,
        "tss_enabled": True,
    }


def _validate_complete_spec(
    spec: WorkerSpec,
    summary: Mapping[str, Any],
    *,
    static_inputs: Mapping[str, Any],
    tss_sha256: str,
) -> None:
    expected_recipe = _expected_recipe(spec)
    for field, expected in (
        ("schema", TRAINER_SCHEMA),
        ("status", "complete"),
        ("dataset", spec.dataset),
        ("method", spec.method),
        ("seed", TRAINING_SEED),
        ("epochs", 1000),
        ("recipe", expected_recipe),
        ("requested_tss_weight", spec.requested_tss_weight),
        ("checkpoint_roles", list(CHECKPOINT_ROLES)),
    ):
        if summary.get(field) != expected:
            raise LaunchProtocolError(
                f"completed run {spec.key} has invalid {field}"
            )

    checkpoints = summary.get("checkpoints")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(
        CHECKPOINT_ROLES
    ):
        raise LaunchProtocolError(
            f"completed run {spec.key} has invalid checkpoint roles"
        )
    for role in CHECKPOINT_ROLES:
        expected_path = spec.run_directory / "checkpoints" / f"{role}.pth.tar"
        record = checkpoints[role]
        if not isinstance(record, Mapping):
            raise LaunchProtocolError(f"invalid {role} record for {spec.key}")
        if Path(str(record.get("path"))).resolve() != expected_path.resolve():
            raise LaunchProtocolError(f"invalid {role} path for {spec.key}")
        if not expected_path.is_file() or expected_path.stat().st_size <= 0:
            raise LaunchProtocolError(f"missing {role} checkpoint for {spec.key}")
        if record.get("sha256") != file_sha256(expected_path):
            raise LaunchProtocolError(f"{role} SHA differs for {spec.key}")
        if record.get("bytes") != expected_path.stat().st_size:
            raise LaunchProtocolError(f"{role} byte count differs for {spec.key}")

    protocol_path = spec.run_directory / "protocol.json"
    if Path(str(summary.get("protocol"))).resolve() != protocol_path.resolve():
        raise LaunchProtocolError(f"invalid protocol path for {spec.key}")
    if not protocol_path.is_file():
        raise LaunchProtocolError(f"missing protocol for {spec.key}")
    protocol_payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    for field, expected in (
        ("schema", TRAINER_SCHEMA),
        ("dataset", spec.dataset),
        ("method", spec.method),
        ("recipe", expected_recipe),
        ("training_seed", 42),
        ("epochs", 1000),
        ("begin_test", 10),
        ("eval_every", 10),
        ("smoke", False),
        ("checkpoint_roles", list(CHECKPOINT_ROLES)),
    ):
        if protocol_payload.get(field) != expected:
            raise LaunchProtocolError(
                f"completed protocol {spec.key} has invalid {field}"
            )
    if protocol_payload.get("protocol_sha256") != summary.get(
        "protocol_sha256"
    ):
        raise LaunchProtocolError(f"protocol identity differs for {spec.key}")
    binding = protocol_payload.get("three_dataset_v2_data_protocol")
    if not isinstance(binding, Mapping) or binding.get(
        "manifest_sha256"
    ) != static_inputs.get("protocol_manifest_sha256"):
        raise LaunchProtocolError(f"data manifest binding differs for {spec.key}")
    runtime = protocol_payload.get("runtime_sources")
    planned_sources = static_inputs.get("training_sources")
    if not isinstance(runtime, Mapping) or not isinstance(planned_sources, Mapping):
        raise LaunchProtocolError(f"runtime source lock missing for {spec.key}")
    for name, planned_record in planned_sources.items():
        if not isinstance(planned_record, Mapping) or runtime.get(name, {}).get(
            "sha256"
        ) != planned_record.get("sha256"):
            raise LaunchProtocolError(
                f"runtime source {name} differs for {spec.key}"
            )
    if spec.method == "final" and protocol_payload.get("tss", {}).get(
        "sha256"
    ) != tss_sha256:
        raise LaunchProtocolError(f"TSS source differs for {spec.key}")
    latest = spec.run_directory / "resume" / "latest_training_state.pth.tar"
    if latest.exists():
        raise LaunchProtocolError(
            f"completed run retained rolling resume state: {spec.key}"
        )


def _spec_is_complete(
    spec: WorkerSpec,
    *,
    static_inputs: Mapping[str, Any],
    tss_sha256: str,
) -> bool:
    summary_path = spec.run_directory / "summary.json"
    if not summary_path.is_file():
        return False
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        return False
    _validate_complete_spec(
        spec,
        payload,
        static_inputs=static_inputs,
        tss_sha256=tss_sha256,
    )
    return True


def _execute_wave(
    members: Sequence[WorkerSpec],
    *,
    wave: int,
    completed_waves: Sequence[int],
    static_inputs: Mapping[str, Any],
    tss_sha256: str,
) -> None:
    pending = [
        spec
        for spec in members
        if not _spec_is_complete(
            spec,
            static_inputs=static_inputs,
            tss_sha256=tss_sha256,
        )
    ]
    if not pending:
        return
    active: list[tuple[WorkerSpec, subprocess.Popen[bytes], Any, Any]] = []
    for spec in pending:
        spec.log_directory.mkdir(parents=True, exist_ok=True)
        stdout_handle = (spec.log_directory / "stdout.log").open("ab")
        stderr_handle = (spec.log_directory / "stderr.log").open("ab")
        process = subprocess.Popen(
            spec.command,
            cwd=REPO_ROOT,
            env=dict(spec.environment),
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        active.append((spec, process, stdout_handle, stderr_handle))
    write_json_atomic(
        SUPERVISOR_STATUS,
        {
            "schema": SCHEMA,
            "status": "running",
            "current_wave": wave,
            "completed_waves": list(completed_waves),
            "active_workers": [
                {
                    "key": spec.key,
                    "pid": process.pid,
                    "gpu_index": spec.gpu_index,
                    "run_directory": str(spec.run_directory),
                }
                for spec, process, _, _ in active
            ],
            "updated_at_unix": time.time(),
        },
    )
    failures: list[str] = []
    for spec, process, stdout_handle, stderr_handle in active:
        return_code = process.wait()
        stdout_handle.close()
        stderr_handle.close()
        if return_code != 0:
            failures.append(f"{spec.key}: exit {return_code}")
    if failures:
        write_json_atomic(
            SUPERVISOR_STATUS,
            {
                "schema": SCHEMA,
                "status": "worker_failed",
                "current_wave": wave,
                "completed_waves": list(completed_waves),
                "failures": failures,
                "updated_at_unix": time.time(),
            },
        )
        raise LaunchProtocolError("worker wave failed: " + "; ".join(failures))
    for spec, _, _, _ in active:
        if not _spec_is_complete(
            spec,
            static_inputs=static_inputs,
            tss_sha256=tss_sha256,
        ):
            raise LaunchProtocolError(
                f"worker exited successfully without a complete run: {spec.key}"
            )


def execute_prepared_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("training_started") is not False:
        raise LaunchProtocolError("plan is not in prepared state")
    SUPERVISOR_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = SUPERVISOR_LOCK.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise LaunchProtocolError("another launcher supervisor is active") from error
    try:
        specs = build_all_worker_specs()
        completed_waves: list[int] = []
        running = dict(plan)
        running.update(
            {
                "status": "running",
                "training_started": True,
                "started_at_unix": time.time(),
            }
        )
        write_json_atomic(LAUNCH_PLAN, running)
        for wave in range(6):
            verify_static_inputs(plan["static_inputs"])
            verify_prepared_artifacts(plan)
            members = [spec for spec in specs if spec.global_wave == wave]
            running["current_wave"] = wave
            running["updated_at_unix"] = time.time()
            write_json_atomic(LAUNCH_PLAN, running)
            _execute_wave(
                members,
                wave=wave,
                completed_waves=completed_waves,
                static_inputs=plan["static_inputs"],
                tss_sha256=plan["tss_statistics"]["sha256"],
            )
            completed_waves.append(wave)
            write_json_atomic(
                SUPERVISOR_STATUS,
                {
                    "schema": SCHEMA,
                    "status": "between_waves",
                    "completed_waves": list(completed_waves),
                    "active_workers": [],
                    "updated_at_unix": time.time(),
                },
            )
        completed = dict(running)
        completed.update(
            {
                "status": "all_workers_complete",
                "completed_at_unix": time.time(),
            }
        )
        write_json_atomic(LAUNCH_PLAN, completed)
        write_json_atomic(
            SUPERVISOR_STATUS,
            {
                "schema": SCHEMA,
                "status": "all_workers_complete",
                "completed_waves": list(range(6)),
                "active_workers": [],
                "updated_at_unix": time.time(),
            },
        )
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="start the six formal waves; default only prepares the plan",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = prepare_launch_plan()
    print(
        f"PREPARED workers={plan['worker_count']} waves={plan['wave_count']} "
        f"plan={LAUNCH_PLAN}",
        flush=True,
    )
    if args.execute:
        execute_prepared_plan(plan)
    else:
        print("DRY_RUN training_not_started=true", flush=True)


if __name__ == "__main__":
    main()
