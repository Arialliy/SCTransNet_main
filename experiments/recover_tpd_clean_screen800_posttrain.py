#!/usr/bin/env python3
"""Publish auditable completion evidence for recovered Clean-v2 sweeps.

The four training processes reached epoch 800 and wrote complete checkpoints,
but their newly spawned CUDA evaluators could not start during a temporary
driver/library mismatch.  This utility is intentionally post-training only:

* it verifies that all training artifacts remain bound to the recovered sweeps;
* it preserves the original worker logs and failed finalizer state;
* it writes a recovery manifest and checksum marker; and
* only after every check passes, it atomically publishes one canonical
  ``TPDCLEAN_COMPLETE`` record per variant.

It never changes a checkpoint, metrics log, summary, split, or protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = (
    REPO_ROOT / "experiments/results/tpd_clean_screen800_4x5090_v1"
)
RUN_NAME = "seed_42_screen800_pd_fp32_shared4x5090_v1"
DATASET = "NUDT-SIRST"
EXPECTED_EPOCHS = 800
EXPECTED_DATA_FINGERPRINT = (
    "39ce329032b7d6e70dcf16e7cd6a0624f52ecfe8c1b9d7c2a04e2bf0180b9b0e"
)
SOURCE_LOCK = REPO_ROOT / "experiments/tpd_clean_screen800_source_lock.json"
BASE_EVALUATOR = REPO_ROOT / "experiments/evaluate_pd_fa_sweep.py"
DATA_FINGERPRINT_SCRIPT = (
    REPO_ROOT / "experiments/fingerprint_tpd_training_data.py"
)
VARIANT_GPU_UUIDS = {
    "grouped_keep": "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    "tpd_clean_ctx": "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
    "tpd_clean_sal": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "tpd_clean_full": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}
TRAINING_FILES = (
    "protocol.json",
    "split.json",
    "metrics.jsonl",
    "summary.json",
    "best.pth.tar",
    "best_miou.pth.tar",
    "last.pth.tar",
)
SWEEP_CONTRACTS = {
    "pd_fa_sweep_best.pth.json": (
        "best.pth.tar",
        "best_validation_pd_primary",
    ),
    "pd_fa_sweep_best_miou.pth.json": (
        "best_miou.pth.tar",
        "best_validation_miou_secondary",
    ),
}


class RecoveryError(RuntimeError):
    """Raised when the post-training recovery evidence is inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise RecoveryError(f"{label} is missing, linked, or not regular: {path}")


def load_json(path: Path) -> Dict[str, Any]:
    require_regular(path, "JSON artifact")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryError(f"Expected JSON object: {path}")
    return value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def ensure_archive_copy(source: Path, destination: Path, apply: bool) -> str:
    require_regular(source, "archive source")
    source_hash = sha256(source)
    if destination.exists():
        require_regular(destination, "archive destination")
        if sha256(destination) != source_hash:
            raise RecoveryError(f"Existing archive differs from source: {destination}")
    elif apply:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != source_hash:
            raise RecoveryError(f"Archive copy verification failed: {destination}")
    return source_hash


def verify_source_lock() -> Dict[str, Any]:
    payload = load_json(SOURCE_LOCK)
    sources = payload.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise RecoveryError("Clean-v2 source lock has no source entries")
    for relative, expected in sources.items():
        path = REPO_ROOT / relative
        require_regular(path, f"locked source {relative}")
        actual = sha256(path)
        if actual != expected:
            raise RecoveryError(
                f"Source mismatch: {relative} expected={expected} actual={actual}"
            )
    return {
        "path": str(SOURCE_LOCK),
        "sha256": sha256(SOURCE_LOCK),
        "matched_files": len(sources),
    }


def verify_data_fingerprint() -> str:
    completed = subprocess.run(
        [
            str(Path("/home/ly/BasicIRSTD/infrarenet/bin/python")),
            str(DATA_FINGERPRINT_SCRIPT),
            "--dataset",
            DATASET,
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
        timeout=300,
    )
    actual = completed.stdout.strip()
    if actual != EXPECTED_DATA_FINGERPRINT:
        raise RecoveryError(
            f"Training data changed: expected={EXPECTED_DATA_FINGERPRINT} "
            f"actual={actual}"
        )
    return actual


def load_metrics(path: Path, variant: str) -> list[Dict[str, Any]]:
    require_regular(path, f"{variant} metrics")
    events = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            raise RecoveryError(f"{variant}: blank metrics line {line_number}")
        event = json.loads(line)
        if not isinstance(event, dict):
            raise RecoveryError(f"{variant}: invalid metrics line {line_number}")
        events.append(event)
    epochs = [event.get("epoch") for event in events]
    if epochs != list(range(1, EXPECTED_EPOCHS + 1)):
        raise RecoveryError(f"{variant}: metrics are not contiguous 1..800")
    if any(event.get("variant") != variant for event in events):
        raise RecoveryError(f"{variant}: metrics variant mismatch")
    return events


def verify_fixed_audit(payload: Mapping[str, Any], variant: str) -> None:
    audit = payload.get("fixed_threshold_0_5_checkpoint_audit")
    if not isinstance(audit, dict):
        raise RecoveryError(f"{variant}: missing fixed-threshold checkpoint audit")
    exact = audit.get("exact_matches")
    if not isinstance(exact, dict) or not exact:
        raise RecoveryError(f"{variant}: empty exact fixed-threshold audit")
    for key, values in exact.items():
        if (
            not isinstance(values, dict)
            or values.get("checkpoint") != values.get("sweep_0_5")
        ):
            raise RecoveryError(f"{variant}: fixed-threshold mismatch for {key}")
    if audit.get("max_abs_non_strict_numeric_delta") != 0.0:
        raise RecoveryError(f"{variant}: non-zero fixed-threshold numeric delta")


def verify_sweep(
    path: Path,
    *,
    variant: str,
    run_dir: Path,
    checkpoint_name: str,
    checkpoint_role: str,
) -> Dict[str, Any]:
    payload = load_json(path)
    if payload.get("variant") != variant:
        raise RecoveryError(f"{variant}: sweep variant mismatch")
    if payload.get("dataset") != DATASET or payload.get("seed") != 42:
        raise RecoveryError(f"{variant}: sweep dataset/seed mismatch")
    if payload.get("official_test_accessed") is not False:
        raise RecoveryError(f"{variant}: sweep accessed an unapproved split")
    if payload.get("checkpoint_role") != checkpoint_role:
        raise RecoveryError(f"{variant}: sweep checkpoint role mismatch")
    checkpoint_path = run_dir / checkpoint_name
    if Path(payload.get("checkpoint", "")).resolve() != checkpoint_path.resolve():
        raise RecoveryError(f"{variant}: sweep checkpoint path mismatch")
    if payload.get("checkpoint_sha256") != sha256(checkpoint_path):
        raise RecoveryError(f"{variant}: sweep checkpoint hash mismatch")
    audit = payload.get("audit")
    if not isinstance(audit, dict):
        raise RecoveryError(f"{variant}: sweep audit is missing")
    checks = audit.get("integrity_checks_passed")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise RecoveryError(f"{variant}: sweep integrity checks did not all pass")
    if audit.get("expected_epochs") != EXPECTED_EPOCHS:
        raise RecoveryError(f"{variant}: sweep expected-epochs mismatch")
    if audit.get("metrics_event_count") != EXPECTED_EPOCHS:
        raise RecoveryError(f"{variant}: sweep metrics count mismatch")
    artifacts = audit.get("artifact_sha256")
    if not isinstance(artifacts, dict):
        raise RecoveryError(f"{variant}: sweep artifact hashes are missing")
    expected_artifacts = {
        "protocol.json": sha256(run_dir / "protocol.json"),
        "split.json": sha256(run_dir / "split.json"),
        "summary.json": sha256(run_dir / "summary.json"),
        "metrics.jsonl": sha256(run_dir / "metrics.jsonl"),
        "checkpoint": sha256(checkpoint_path),
        "evaluator": sha256(BASE_EVALUATOR),
    }
    if artifacts != expected_artifacts:
        raise RecoveryError(f"{variant}: sweep artifact binding mismatch")
    verify_fixed_audit(payload, variant)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "checkpoint": checkpoint_name,
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": payload.get("checkpoint_epoch"),
        "checkpoint_role": checkpoint_role,
        "invocation_argv": audit.get("invocation_argv"),
        "integrity_checks_passed": checks,
    }


def gpu_snapshot() -> list[Dict[str, str]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=30,
    )
    keys = ("index", "uuid", "name", "driver_version", "memory_used_mib", "memory_free_mib")
    return [
        dict(zip(keys, (part.strip() for part in line.split(","))))
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def build_recovery(
    recovery_dir: Path,
    *,
    apply: bool,
) -> Dict[str, Any]:
    recovery_dir = recovery_dir.resolve()
    if CANDIDATE_ROOT.resolve() not in recovery_dir.parents:
        raise RecoveryError("Recovery directory must be inside the candidate root")
    original_logs = recovery_dir / "original_logs"
    archived_state = recovery_dir / "before" / "finalizer_state.json"
    current_state = CANDIDATE_ROOT / "launch/finalizer_state.json"
    state_hash = ensure_archive_copy(current_state, archived_state, apply)
    source_record = verify_source_lock()
    data_fingerprint = verify_data_fingerprint()

    records: Dict[str, Any] = {}
    for variant, gpu_uuid in VARIANT_GPU_UUIDS.items():
        run_dir = CANDIDATE_ROOT / DATASET / variant / RUN_NAME
        if not run_dir.is_dir() or run_dir.is_symlink():
            raise RecoveryError(f"{variant}: invalid run directory")
        summary = load_json(run_dir / "summary.json")
        if (
            summary.get("status") != "complete"
            or summary.get("variant") != variant
            or summary.get("dataset") != DATASET
            or summary.get("seed") != 42
            or summary.get("official_test_accessed") is not False
        ):
            raise RecoveryError(f"{variant}: summary contract mismatch")
        metrics = load_metrics(run_dir / "metrics.jsonl", variant)
        training_hashes = {}
        for name in TRAINING_FILES:
            path = run_dir / name
            require_regular(path, f"{variant} {name}")
            training_hashes[name] = sha256(path)

        current_log = CANDIDATE_ROOT / "logs" / f"{variant}.log"
        archived_log = original_logs / f"{variant}.log"
        original_log_hash = ensure_archive_copy(current_log, archived_log, apply)
        launch = CANDIDATE_ROOT / "launch" / f"{variant}.json"
        archived_launch = recovery_dir / "before" / "launch" / f"{variant}.json"
        launch_hash = ensure_archive_copy(launch, archived_launch, apply)
        launch_payload = load_json(launch)
        if launch_payload.get("gpu_uuid") != gpu_uuid:
            raise RecoveryError(f"{variant}: launch GPU UUID mismatch")

        sweeps = {}
        for sweep_name, (checkpoint_name, role) in SWEEP_CONTRACTS.items():
            sweeps[sweep_name] = verify_sweep(
                run_dir / sweep_name,
                variant=variant,
                run_dir=run_dir,
                checkpoint_name=checkpoint_name,
                checkpoint_role=role,
            )
        records[variant] = {
            "variant": variant,
            "run_directory": str(run_dir),
            "epochs": [metrics[0]["epoch"], metrics[-1]["epoch"]],
            "metric_rows": len(metrics),
            "summary_status": summary["status"],
            "original_gpu_uuid": gpu_uuid,
            "launch_manifest": {
                "path": str(launch),
                "sha256": launch_hash,
                "archive": str(archived_launch),
            },
            "original_worker_log": {
                "path": str(current_log),
                "sha256": original_log_hash,
                "archive": str(archived_log),
            },
            "training_artifact_sha256": training_hashes,
            "sweeps": sweeps,
        }

    manifest = {
        "schema": "sctransnet_tpd_clean_screen800_posttrain_recovery_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "All four trainers completed epoch 800; only their first new CUDA "
            "evaluator process failed during a temporary driver/library mismatch."
        ),
        "training_reused_without_rerun": True,
        "checkpoints_or_metrics_modified": False,
        "candidate_root": str(CANDIDATE_ROOT),
        "source_lock": source_record,
        "training_data_sha256": data_fingerprint,
        "original_finalizer_state": {
            "path": str(current_state),
            "sha256": state_hash,
            "archive": str(archived_state),
        },
        "gpu_snapshot": gpu_snapshot(),
        "variants": records,
    }
    if not apply:
        return manifest

    manifest_path = recovery_dir / "recovery_manifest.json"
    atomic_json(manifest_path, manifest)
    manifest_hash = sha256(manifest_path)

    canonical_log_hashes = {}
    for variant, record in records.items():
        sweep_lines = []
        for name, sweep in record["sweeps"].items():
            sweep_lines.append(
                f"RECOVERED_SWEEP name={name} sha256={sweep['sha256']} "
                f"checkpoint={sweep['checkpoint']} epoch={sweep['checkpoint_epoch']}"
            )
        canonical_log = (
            "TPDCLEAN_POSTTRAIN_RECOVERY "
            "schema=sctransnet_tpd_clean_screen800_posttrain_recovery_v1\n"
            f"variant={variant}\n"
            f"original_worker_log_archive={record['original_worker_log']['archive']}\n"
            f"original_worker_log_sha256={record['original_worker_log']['sha256']}\n"
            "training_status=complete metrics_epochs=1..800 "
            "training_reused_without_rerun=true\n"
            + "\n".join(sweep_lines)
            + "\n"
            f"recovery_manifest={manifest_path}\n"
            f"recovery_manifest_sha256={manifest_hash}\n"
            f"TPDCLEAN_COMPLETE variant={variant} "
            f"gpu_uuid={record['original_gpu_uuid']} epochs=800\n"
        )
        log_path = CANDIDATE_ROOT / "logs" / f"{variant}.log"
        atomic_text(log_path, canonical_log)
        canonical_log_hashes[variant] = sha256(log_path)

    marker_entries = [
        (manifest_path, manifest_hash),
        *[
            (
                Path(record["original_worker_log"]["archive"]),
                record["original_worker_log"]["sha256"],
            )
            for record in records.values()
        ],
        *[
            (Path(sweep["path"]), sweep["sha256"])
            for record in records.values()
            for sweep in record["sweeps"].values()
        ],
        *[
            (
                CANDIDATE_ROOT / "logs" / f"{variant}.log",
                log_hash,
            )
            for variant, log_hash in canonical_log_hashes.items()
        ],
    ]
    marker_text = "".join(
        f"{file_hash}  {path}\n" for path, file_hash in marker_entries
    )
    marker_path = recovery_dir / "RECOVERY_COMPLETE.sha256"
    atomic_text(marker_path, marker_text)
    return {
        "status": "complete",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "marker": str(marker_path),
        "canonical_log_sha256": canonical_log_hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Publish the manifest and canonical completion logs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_recovery(args.recovery_dir, apply=args.apply)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
