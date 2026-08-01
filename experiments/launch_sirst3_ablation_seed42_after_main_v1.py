#!/usr/bin/env python3
"""Schedule the SIRST3 A0--A4 ablation after the eight main runs.

A0 and A4 are protocol-identical to the already required SIRST3 Original and
Final scratch runs, so they are referenced by immutable path and SHA-256 rather
than trained a second time.  Once all eight main summaries are complete, this
launcher runs only A1--A3:

* wave 1: A1 on physical GPU 2 and A2 on physical GPU 3;
* wave 2: A3 on physical GPU 2.

The launcher never waits for a GPU utilization condition.  Its only wait gate
is completion of the scientifically prior eight-run matrix.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
RUNNER = (
    REPO_ROOT / "experiments" / "train_sirst3_ablation_seed42_exact_v1.py"
)
RESULTS_ROOT = REPO_ROOT / "results" / "four_dataset_seed42_v1"
MAIN_RUNS_ROOT = RESULTS_ROOT / "runs"
ABLATION_ROOT = RESULTS_ROOT / "ablations"
MANIFEST_ROOT = RESULTS_ROOT / "manifests"
TSS_STATISTICS = MANIFEST_ROOT / "four_dataset_tss_seed42_v1.json"
DATASETS = ("SIRST3", "NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
MAIN_METHODS = ("original", "final")
GPU = {
    "2": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "3": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
WAVES = ((("A1", "2"), ("A2", "3")), (("A3", "2"),))
SCHEMA = "sctransnet_sirst3_ablation_after_main_launcher_v1"
CPU_ENV = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "OMP_WAIT_POLICY": "PASSIVE",
    "KMP_BLOCKTIME": "0",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    return parser.parse_args(argv)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _main_summary_path(root: Path, dataset: str, method: str) -> Path:
    return root / "runs" / dataset / method / "seed_42" / "summary.json"


def _read_complete_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("status") == "complete" else None


def main_gate(root: Path) -> tuple[bool, dict[str, Any]]:
    records: dict[str, Any] = {}
    complete = True
    for dataset in DATASETS:
        for method in MAIN_METHODS:
            key = f"{dataset}__{method}"
            path = _main_summary_path(root, dataset, method)
            summary = _read_complete_summary(path)
            if summary is None:
                complete = False
                records[key] = {"complete": False, "path": str(path)}
                continue
            expected = {
                "dataset": dataset,
                "method": method,
                "seed": 42,
                "epochs": 1000,
                "test_selected": True,
                "selection_is_optimistic": True,
            }
            for field, value in expected.items():
                if summary.get(field) != value:
                    raise RuntimeError(
                        f"main summary {key} field {field} differs"
                    )
            checkpoints = summary.get("checkpoints")
            if not isinstance(checkpoints, dict) or set(checkpoints) != {
                "best_miou",
                "best_pd",
            }:
                raise RuntimeError(
                    f"main summary {key} selected roles differ"
                )
            for role, record in checkpoints.items():
                checkpoint = Path(record["path"])
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                if _sha256(checkpoint) != record["sha256"]:
                    raise RuntimeError(f"main {key} {role} SHA-256 differs")
            protocol_path = Path(summary["protocol"])
            if not protocol_path.is_file():
                raise FileNotFoundError(protocol_path)
            with protocol_path.open("r", encoding="utf-8") as handle:
                protocol = json.load(handle)
            expected_candidates = list(range(10, 1001, 10))
            for field, value in (
                ("begin_test", 10),
                ("eval_every", 10),
                ("candidate_epochs", expected_candidates),
            ):
                if protocol.get(field) != value:
                    raise RuntimeError(
                        f"main protocol {key} field {field} differs"
                    )
            records[key] = {
                "complete": True,
                "path": str(path),
                "sha256": _sha256(path),
                "protocol": {
                    "path": str(protocol_path),
                    "sha256": _sha256(protocol_path),
                    "candidate_epochs": expected_candidates,
                },
                "checkpoints": checkpoints,
            }
    return complete, records


def _wait_for_main(root: Path, poll_seconds: int) -> dict[str, Any]:
    if poll_seconds < 5:
        raise ValueError("--poll-seconds must be at least 5")
    while True:
        complete, records = main_gate(root)
        if complete:
            return records
        remaining = sum(
            1 for record in records.values() if not record["complete"]
        )
        print(
            f"WAIT_MAIN incomplete_runs={remaining}/8 "
            f"poll_seconds={poll_seconds}",
            flush=True,
        )
        time.sleep(poll_seconds)


def _reuse_manifest(
    root: Path,
    main_records: dict[str, Any],
) -> dict[str, Any]:
    mapping = {
        "A0": ("SIRST3__original", "Original SCTransNet"),
        "A4": ("SIRST3__final", "Full Final"),
    }
    records: dict[str, Any] = {}
    for ablation_id, (main_key, description) in mapping.items():
        source = main_records[main_key]
        records[ablation_id] = {
            "ablation_id": ablation_id,
            "description": description,
            "reuse_reason": (
                "protocol-identical SIRST3 scratch seed42 1000-epoch run"
            ),
            "retrained": False,
            "source_main_run": main_key,
            "source_summary": source,
            "selected_checkpoint_roles": ["best_miou", "best_pd"],
        }
    return {
        "schema": SCHEMA,
        "training_seed": 42,
        "dataset": "SIRST3",
        "duplicate_training_avoided": True,
        "A0_A4_protocol_identity_required": True,
        "records": records,
        "results_root": str(root),
    }


def _command(root: Path, ablation_id: str, physical_gpu: str) -> list[str]:
    uuid = GPU[physical_gpu]
    return [
        str(PYTHON),
        str(RUNNER),
        "--ablation",
        ablation_id,
        "--data-root",
        str(REPO_ROOT / "datasets"),
        "--results-root",
        str(root),
        "--manifest-root",
        str(root / "manifests"),
        "--tss-statistics",
        str(root / "manifests" / "four_dataset_tss_seed42_v1.json"),
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
        "3",
        "--tiny-area",
        "9",
        "--device",
        "cuda:0",
        "--physical-gpu-index",
        physical_gpu,
        "--expected-gpu-uuid",
        uuid,
        "--resume",
        "auto",
    ]


def dry_run_manifest(root: Path) -> dict[str, Any]:
    complete, records = main_gate(root)
    return {
        "schema": SCHEMA,
        "mode": "dry_run",
        "main_gate_complete_now": complete,
        "main_gate": records,
        "will_wait_for_main_matrix": True,
        "waits_for_gpu_utilization": False,
        "candidate_epochs": list(range(10, 1001, 10)),
        "candidate_epoch_count": 100,
        "waves": [
            [
                {
                    "ablation_id": ablation_id,
                    "physical_gpu": gpu,
                    "gpu_uuid": GPU[gpu],
                    "command": _command(root, ablation_id, gpu),
                }
                for ablation_id, gpu in wave
            ]
            for wave in WAVES
        ],
        "A0_A4_reused_from_main": True,
        "persistent_checkpoint_roles_per_run": ["best_miou", "best_pd"],
    }


def _validate_completed_ablation(root: Path, ablation_id: str) -> Path:
    run_dir = (
        root
        / "ablations"
        / "runs"
        / "SIRST3"
        / ablation_id
        / "seed_42"
    )
    summary_path = run_dir / "summary.json"
    summary = _read_complete_summary(summary_path)
    if summary is None:
        raise RuntimeError(f"ablation did not complete: {ablation_id}")
    checkpoints = run_dir / "checkpoints"
    persistent = sorted(path.name for path in checkpoints.glob("*.pth*"))
    if persistent != ["best_miou.pth.tar", "best_pd.pth.tar"]:
        raise RuntimeError(
            f"{ablation_id} persistent checkpoints differ: {persistent}"
        )
    if (run_dir / "resume" / "latest_training_state.pth.tar").exists():
        raise RuntimeError(f"{ablation_id} retained rolling resume state")
    return summary_path


def _run_wave(root: Path, wave_index: int, wave: tuple[tuple[str, str], ...]) -> None:
    processes: list[tuple[str, subprocess.Popen[bytes], Any, Any]] = []
    launch_root = root / "ablations" / "launch"
    launch_root.mkdir(parents=True, exist_ok=True)
    for ablation_id, gpu in wave:
        run_summary = (
            root
            / "ablations"
            / "runs"
            / "SIRST3"
            / ablation_id
            / "seed_42"
            / "summary.json"
        )
        if _read_complete_summary(run_summary) is not None:
            _validate_completed_ablation(root, ablation_id)
            print(f"SKIP_COMPLETE ablation={ablation_id}", flush=True)
            continue
        stdout = (launch_root / f"{ablation_id}.stdout.log").open("ab")
        stderr = (launch_root / f"{ablation_id}.stderr.log").open("ab")
        environment = os.environ.copy()
        environment.update(CPU_ENV)
        environment.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": GPU[gpu],
                "PYTHONUNBUFFERED": "1",
            }
        )
        process = subprocess.Popen(
            _command(root, ablation_id, gpu),
            cwd=REPO_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        processes.append((ablation_id, process, stdout, stderr))
        print(
            f"LAUNCHED wave={wave_index} ablation={ablation_id} "
            f"physical_gpu={gpu} pid={process.pid}",
            flush=True,
        )
    failures: list[str] = []
    for ablation_id, process, stdout, stderr in processes:
        returncode = process.wait()
        stdout.close()
        stderr.close()
        if returncode != 0:
            failures.append(f"{ablation_id}:exit={returncode}")
        else:
            _validate_completed_ablation(root, ablation_id)
    if failures:
        raise RuntimeError(f"ablation wave {wave_index} failed: {failures}")


def run_launcher(args: argparse.Namespace) -> Path:
    root = args.results_root.resolve()
    for path in (PYTHON, RUNNER, REPO_ROOT / "datasets"):
        if not path.exists():
            raise FileNotFoundError(path)
    launch_root = root / "ablations" / "launch"
    launch_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (launch_root / "launcher.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("ablation launcher is already active") from error

    if args.dry_run:
        path = launch_root / "dry_run.json"
        _write_json(path, dry_run_manifest(root))
        print(path)
        return path

    main_records = _wait_for_main(root, args.poll_seconds)
    reuse = _reuse_manifest(root, main_records)
    reuse_path = root / "ablations" / "A0_A4_reuse_manifest.json"
    _write_json(reuse_path, reuse)
    _write_json(
        launch_root / "status.json",
        {
            "schema": SCHEMA,
            "status": "running",
            "main_gate_complete": True,
            "started_at_unix": time.time(),
        },
    )
    for wave_index, wave in enumerate(WAVES, start=1):
        _run_wave(root, wave_index, wave)
    summaries = {
        ablation_id: str(_validate_completed_ablation(root, ablation_id))
        for ablation_id in ("A1", "A2", "A3")
    }
    final = {
        "schema": SCHEMA,
        "status": "complete",
        "training_seed": 42,
        "dataset": "SIRST3",
        "main_gate_complete": True,
        "A0_A4_reuse_manifest": str(reuse_path),
        "trained_ablation_summaries": summaries,
        "persistent_checkpoint_roles_per_run": ["best_miou", "best_pd"],
        "completed_at_unix": time.time(),
    }
    path = launch_root / "summary.json"
    _write_json(path, final)
    _write_json(launch_root / "status.json", final)
    print(path)
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_launcher(args)


if __name__ == "__main__":
    main()
