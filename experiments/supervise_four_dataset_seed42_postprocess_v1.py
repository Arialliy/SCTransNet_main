#!/usr/bin/env python3
"""Fail-fast post-processing supervisor for the eight formal seed-42 runs.

The supervisor never trains a model and never waits for training.  It first
requires all eight formal summaries and exactly two checkpoint files per run,
then executes the frozen post-processing stages in this order:

1. freeze the 16 selected checkpoints and their manifest;
2. evaluate 16 dataset-specific fixed-threshold points;
3. evaluate 12 SIRST3-checkpoint source-subset fixed-threshold points;
4. evaluate 16 dataset-specific Pd--Fa sweeps;
5. evaluate 12 SIRST3-checkpoint source-subset Pd--Fa sweeps;
6. finalize the paper tables and result summary.

Each stage has an input fingerprint, an atomic status record, a combined log,
and SHA-256 records for all expected outputs.  A repeated invocation skips an
already complete stage only when its fingerprint and every recorded output
hash still match.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import four_dataset_evaluation_protocol_v1 as protocol  # noqa: E402


SCHEMA_PREFIX = "sctransnet_four_dataset_seed42_postprocess"
RESULTS_ROOT = REPO_ROOT / "results" / "four_dataset_seed42_v1"
DATA_ROOT = REPO_ROOT / "datasets"
PYTHON = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
GPU2_UUID = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
POSTPROCESS_ROOT = RESULTS_ROOT / "postprocess"
STATUS_ROOT = POSTPROCESS_ROOT / "status"
LOG_ROOT = POSTPROCESS_ROOT / "logs"
OVERALL_STATUS = POSTPROCESS_ROOT / "postprocess_status.json"
GATE_SNAPSHOT = POSTPROCESS_ROOT / "training_gate_snapshot.json"
ARTIFACT_MANIFEST = POSTPROCESS_ROOT / "postprocess_artifact_manifest.json"

DATA_ARTIFACT_NAMES = (
    "four_dataset_imgidx_v1.json",
    "four_dataset_legacy_norm_v1.json",
    "nuaa_misc111_correction_v1.json",
    "four_dataset_data_gate_v1.json",
)

SOURCE_FILES = (
    REPO_ROOT / "experiments" / "four_dataset_evaluation_protocol_v1.py",
    REPO_ROOT / "experiments" / "select_four_dataset_test_checkpoints_v1.py",
    REPO_ROOT / "experiments" / "evaluate_four_dataset_seed42_v1.py",
    REPO_ROOT / "experiments" / "evaluate_sirst3_three_official_tests_v1.py",
    REPO_ROOT / "experiments" / "evaluate_four_dataset_pd_fa_sweep_v1.py",
    REPO_ROOT / "experiments" / "finalize_four_dataset_seed42_paper_results_v1.py",
    REPO_ROOT / "experiments" / "four_dataset_models_seed42_v1.py",
    REPO_ROOT / "experiments" / "paper_four_dataset_v1.py",
    Path(__file__).resolve(),
)


class PostprocessError(RuntimeError):
    """Raised when a frozen post-processing precondition is not satisfied."""


@dataclass(frozen=True)
class Stage:
    index: int
    name: str
    module: str
    arguments: tuple[str, ...]
    outputs: tuple[Path, ...]

    @property
    def key(self) -> str:
        return f"{self.index:02d}_{self.name}"

    @property
    def command(self) -> tuple[str, ...]:
        return (str(PYTHON), "-m", self.module, *self.arguments)

    @property
    def status_path(self) -> Path:
        return STATUS_ROOT / f"{self.key}.json"

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / f"{self.key}.log"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PostprocessError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    protocol.atomic_write_json(path, dict(payload), overwrite=True)


def _artifact(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _artifacts(paths: Iterable[Path]) -> list[dict[str, Any]]:
    resolved = sorted({Path(path).resolve() for path in paths}, key=str)
    return [_artifact(path) for path in resolved]


def _checkpoint_paths(root: Path = RESULTS_ROOT) -> tuple[Path, ...]:
    return tuple(
        root
        / "selected_checkpoints"
        / dataset
        / method
        / protocol.CHECKPOINT_FILENAMES[role]
        for dataset in protocol.DATASETS
        for method in protocol.METHODS
        for role in protocol.CHECKPOINT_ROLES
    )


def _fixed_paths(root: Path = RESULTS_ROOT) -> tuple[Path, ...]:
    return tuple(
        root
        / "evaluations"
        / "fixed_0_5"
        / dataset
        / method
        / f"{role}.json"
        for dataset in protocol.DATASETS
        for method in protocol.METHODS
        for role in protocol.CHECKPOINT_ROLES
    )


def _source_fixed_paths(root: Path = RESULTS_ROOT) -> tuple[Path, ...]:
    return tuple(
        root
        / "evaluations"
        / "sirst3_three_sources"
        / source
        / method
        / f"{role}.json"
        for source in protocol.SOURCE_DATASETS
        for method in protocol.METHODS
        for role in protocol.CHECKPOINT_ROLES
    )


def _sweep_paths(root: Path = RESULTS_ROOT) -> tuple[Path, ...]:
    return tuple(
        root
        / "evaluations"
        / "pd_fa_sweeps"
        / dataset
        / method
        / f"{role}.json"
        for dataset in protocol.DATASETS
        for method in protocol.METHODS
        for role in protocol.CHECKPOINT_ROLES
    )


def _source_sweep_paths(root: Path = RESULTS_ROOT) -> tuple[Path, ...]:
    return tuple(
        root
        / "evaluations"
        / "pd_fa_sweeps"
        / "sirst3_three_sources"
        / source
        / method
        / f"{role}.json"
        for source in protocol.SOURCE_DATASETS
        for method in protocol.METHODS
        for role in protocol.CHECKPOINT_ROLES
    )


def _final_paths(root: Path = RESULTS_ROOT) -> tuple[Path, ...]:
    table_stems = (
        "table2_best_miou",
        "table4a_best_pd",
        "table4b_last_epoch1000",
        "table3a_best_miou_sirst3_three_sources",
        "table3b_best_pd_sirst3_three_sources",
    )
    table_pairs = tuple(
        root / "tables" / f"{stem}.{suffix}"
        for stem in table_stems
        for suffix in ("csv", "md")
    )
    return (
        root / "paper_results_summary.json",
        root / "paired_deltas_final_minus_original.json",
        *table_pairs,
        root / "tables" / "table7_pd_at_fa_budgets.md",
    )


def build_stages(
    *,
    device: str = "cuda:0",
    workers: int = 0,
) -> tuple[Stage, ...]:
    common_eval = (
        "--results-root",
        str(RESULTS_ROOT),
        "--data-root",
        str(DATA_ROOT),
        "--device",
        device,
        "--workers",
        str(workers),
        "--overwrite",
    )
    selected_root = RESULTS_ROOT / "selected_checkpoints"
    return (
        Stage(
            1,
            "freeze_checkpoints",
            "experiments.select_four_dataset_test_checkpoints_v1",
            (
                "--runs-root",
                str(RESULTS_ROOT / "runs"),
                "--selected-root",
                str(selected_root),
                "--manifest",
                str(selected_root / "checkpoint_manifest.json"),
                "--all-runs",
                "--overwrite",
            ),
            (
                selected_root / "checkpoint_manifest.json",
                *_checkpoint_paths(),
            ),
        ),
        Stage(
            2,
            "fixed_0_5_dataset_specific",
            "experiments.evaluate_four_dataset_seed42_v1",
            (*common_eval, "--all-dataset-specific"),
            _fixed_paths(),
        ),
        Stage(
            3,
            "fixed_0_5_sirst3_sources",
            "experiments.evaluate_sirst3_three_official_tests_v1",
            (*common_eval, "--all"),
            _source_fixed_paths(),
        ),
        Stage(
            4,
            "pd_fa_sweep_dataset_specific",
            "experiments.evaluate_four_dataset_pd_fa_sweep_v1",
            (*common_eval, "--all-dataset-specific"),
            _sweep_paths(),
        ),
        Stage(
            5,
            "pd_fa_sweep_sirst3_sources",
            "experiments.evaluate_four_dataset_pd_fa_sweep_v1",
            (*common_eval, "--all-sirst3-sources"),
            _source_sweep_paths(),
        ),
        Stage(
            6,
            "finalize_tables",
            "experiments.finalize_four_dataset_seed42_paper_results_v1",
            (
                "--results-root",
                str(RESULTS_ROOT),
                "--overwrite",
            ),
            _final_paths(),
        ),
    )


def _summary_run_artifacts(
    dataset: str,
    method: str,
    *,
    root: Path,
) -> tuple[dict[str, Any], list[Path]]:
    run_dir = root / "runs" / dataset / method / "seed_42"
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file() or summary_path.is_symlink():
        raise PostprocessError(f"formal summary is missing: {summary_path}")
    summary = _load_json(summary_path)
    expected = {
        "status": "complete",
        "dataset": dataset,
        "method": method,
        "seed": protocol.TRAINING_SEED,
        "epochs": protocol.EXPECTED_EPOCHS,
        "test_selected": True,
        "selection_is_optimistic": True,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise PostprocessError(
                f"{dataset}/{method} summary {field} differs: "
                f"{summary.get(field)!r} != {value!r}"
            )
    records = summary.get("checkpoints")
    if not isinstance(records, Mapping) or set(records) != set(
        protocol.CHECKPOINT_ROLES
    ):
        raise PostprocessError(
            f"{dataset}/{method} summary must report exactly "
            "best_miou and best_pd"
        )
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        raise PostprocessError(
            f"checkpoint directory is missing: {checkpoint_dir}"
        )
    entries = sorted(path.name for path in checkpoint_dir.iterdir())
    expected_entries = sorted(protocol.CHECKPOINT_FILENAMES.values())
    if entries != expected_entries:
        raise PostprocessError(
            f"{dataset}/{method} checkpoint directory entries differ: "
            f"{entries!r} != {expected_entries!r}"
        )

    paths = [summary_path]
    checkpoint_audit: dict[str, Any] = {}
    for role in protocol.CHECKPOINT_ROLES:
        checkpoint = checkpoint_dir / protocol.CHECKPOINT_FILENAMES[role]
        if (
            not checkpoint.is_file()
            or checkpoint.is_symlink()
            or checkpoint.stat().st_size <= 0
        ):
            raise PostprocessError(
                f"checkpoint is missing, linked, or empty: {checkpoint}"
            )
        record = records[role]
        if not isinstance(record, Mapping):
            raise PostprocessError(
                f"{dataset}/{method}/{role} summary record is invalid"
            )
        digest = _file_sha256(checkpoint)
        if record.get("sha256") != digest:
            raise PostprocessError(
                f"{dataset}/{method}/{role} summary SHA differs"
            )
        if record.get("bytes") != checkpoint.stat().st_size:
            raise PostprocessError(
                f"{dataset}/{method}/{role} summary byte count differs"
            )
        recorded_path = record.get("path")
        if not isinstance(recorded_path, str) or Path(recorded_path).resolve() != (
            checkpoint.resolve()
        ):
            raise PostprocessError(
                f"{dataset}/{method}/{role} summary path differs"
            )
        paths.append(checkpoint)
        checkpoint_audit[role] = _artifact(checkpoint)

    for name in ("protocol.json", "metrics.jsonl"):
        path = run_dir / name
        if not path.is_file() or path.is_symlink():
            raise PostprocessError(f"formal run artifact is missing: {path}")
        paths.append(path)
    rolling = run_dir / "resume" / "latest_training_state.pth.tar"
    if rolling.exists():
        raise PostprocessError(
            f"completed run retained rolling resume state: {rolling}"
        )
    return (
        {
            "dataset": dataset,
            "method": method,
            "summary": _artifact(summary_path),
            "checkpoints": checkpoint_audit,
        },
        paths,
    )


def audit_training_gate(
    *,
    root: Path = RESULTS_ROOT,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    paths: list[Path] = []
    errors: list[str] = []
    for dataset in protocol.DATASETS:
        for method in protocol.METHODS:
            try:
                record, run_paths = _summary_run_artifacts(
                    dataset,
                    method,
                    root=root,
                )
            except (FileNotFoundError, OSError, ValueError, PostprocessError) as exc:
                errors.append(f"{dataset}/{method}: {exc}")
            else:
                records.append(record)
                paths.extend(run_paths)
    ready = len(records) == 8 and not errors
    payload = {
        "schema": f"{SCHEMA_PREFIX}_training_gate_v1",
        "status": "complete" if ready else "not_ready",
        "ready": ready,
        "expected_run_count": 8,
        "complete_run_count": len(records),
        "checkpoint_roles": list(protocol.CHECKPOINT_ROLES),
        "records": records,
        "input_artifacts": _artifacts(paths) if ready else [],
        "errors": errors,
    }
    payload["gate_sha256"] = _canonical_sha256(payload)
    if not ready and raise_on_error:
        details = "; ".join(errors[:3])
        raise PostprocessError(
            f"formal training gate is not ready ({len(records)}/8): {details}"
        )
    return payload


def _data_artifact_paths() -> tuple[Path, ...]:
    return tuple(
        RESULTS_ROOT / "manifests" / name for name in DATA_ARTIFACT_NAMES
    )


def _stage_inputs(
    stage: Stage,
    gate: Mapping[str, Any],
) -> tuple[Path, ...]:
    training_paths = tuple(
        Path(record["path"]) for record in gate["input_artifacts"]
    )
    selected = (
        RESULTS_ROOT
        / "selected_checkpoints"
        / "checkpoint_manifest.json",
        *_checkpoint_paths(),
    )
    data_and_sources = (*_data_artifact_paths(), *SOURCE_FILES)
    if stage.index == 1:
        return (*training_paths, *SOURCE_FILES)
    if stage.index in (2, 3, 4, 5):
        return (*selected, *data_and_sources)
    if stage.index == 6:
        return (
            *selected,
            *_fixed_paths(),
            *_source_fixed_paths(),
            *_sweep_paths(),
            *_source_sweep_paths(),
            *SOURCE_FILES,
        )
    raise AssertionError(stage.index)


def _validate_json_outputs(stage: Stage) -> None:
    for path in stage.outputs:
        if not path.is_file() or path.is_symlink():
            raise PostprocessError(f"stage output is missing: {path}")
        if path.suffix == ".json":
            payload = _load_json(path)
            if path.name == "checkpoint_manifest.json":
                if (
                    payload.get("status") != "complete"
                    or payload.get("record_count") != 8
                ):
                    raise PostprocessError(
                        f"checkpoint manifest is incomplete: {path}"
                    )
            elif path.name == "paper_results_summary.json":
                if payload.get("status") != "complete":
                    raise PostprocessError(
                        f"paper result summary is incomplete: {path}"
                    )
            elif "evaluations" in path.parts:
                if payload.get("status") != "complete":
                    raise PostprocessError(
                        f"evaluation output is incomplete: {path}"
                    )


def _validate_selected_checkpoint_tree() -> None:
    selected_root = RESULTS_ROOT / "selected_checkpoints"
    for dataset in protocol.DATASETS:
        for method in protocol.METHODS:
            directory = selected_root / dataset / method
            if not directory.is_dir() or directory.is_symlink():
                raise PostprocessError(
                    f"frozen checkpoint directory is missing: {directory}"
                )
            entries = sorted(path.name for path in directory.iterdir())
            expected = sorted(protocol.CHECKPOINT_FILENAMES.values())
            if entries != expected:
                raise PostprocessError(
                    f"frozen checkpoint entries differ in {directory}: "
                    f"{entries!r} != {expected!r}"
                )
            for path in directory.iterdir():
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or path.stat().st_size <= 0
                ):
                    raise PostprocessError(
                        f"invalid frozen checkpoint artifact: {path}"
                    )


def _stage_fingerprint(
    stage: Stage,
    input_artifacts: Sequence[Mapping[str, Any]],
    *,
    device: str,
    workers: int,
) -> str:
    return _canonical_sha256(
        {
            "stage": stage.key,
            "command": list(stage.command),
            "device": device,
            "workers": workers,
            "cuda_visible_devices": GPU2_UUID if device == "cuda:0" else None,
            "inputs": list(input_artifacts),
            "expected_outputs": [str(path.resolve()) for path in stage.outputs],
        }
    )


def _existing_stage_is_reusable(
    stage: Stage,
    *,
    fingerprint: str,
) -> bool:
    if not stage.status_path.is_file() or stage.status_path.is_symlink():
        return False
    try:
        status = _load_json(stage.status_path)
    except (FileNotFoundError, OSError, ValueError, PostprocessError):
        return False
    if (
        status.get("status") != "complete"
        or status.get("fingerprint_sha256") != fingerprint
    ):
        return False
    expected = status.get("outputs")
    if not isinstance(expected, list) or len(expected) != len(stage.outputs):
        return False
    expected_index = {
        str(Path(record["path"]).resolve()): record
        for record in expected
        if isinstance(record, Mapping) and isinstance(record.get("path"), str)
    }
    if set(expected_index) != {
        str(path.resolve()) for path in stage.outputs
    }:
        return False
    try:
        _validate_json_outputs(stage)
        if stage.index == 1:
            _validate_selected_checkpoint_tree()
        for path in stage.outputs:
            record = expected_index[str(path.resolve())]
            if (
                record.get("sha256") != _file_sha256(path)
                or record.get("bytes") != path.stat().st_size
            ):
                return False
    except (FileNotFoundError, OSError, ValueError, PostprocessError):
        return False
    return True


def _previous_attempt(stage: Stage) -> int:
    if not stage.status_path.is_file():
        return 0
    try:
        value = _load_json(stage.status_path).get("attempt", 0)
        return int(value) if int(value) >= 0 else 0
    except (OSError, ValueError, TypeError, PostprocessError):
        return 0


def _run_stage(
    stage: Stage,
    *,
    gate: Mapping[str, Any],
    device: str,
    workers: int,
    force_rerun: bool,
) -> dict[str, Any]:
    input_paths = _stage_inputs(stage, gate)
    input_artifacts = _artifacts(input_paths)
    fingerprint = _stage_fingerprint(
        stage,
        input_artifacts,
        device=device,
        workers=workers,
    )
    if not force_rerun and _existing_stage_is_reusable(
        stage,
        fingerprint=fingerprint,
    ):
        status = _load_json(stage.status_path)
        return {
            "stage": stage.key,
            "action": "skipped_verified_complete",
            "status_path": str(stage.status_path.resolve()),
            "status_sha256": _file_sha256(stage.status_path),
            "fingerprint_sha256": fingerprint,
            "outputs": status["outputs"],
        }

    attempt = _previous_attempt(stage) + 1
    stage.status_path.parent.mkdir(parents=True, exist_ok=True)
    stage.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    running = {
        "schema": f"{SCHEMA_PREFIX}_stage_status_v1",
        "status": "running",
        "stage": stage.key,
        "attempt": attempt,
        "command": list(stage.command),
        "command_sha256": _canonical_sha256(list(stage.command)),
        "cwd": str(REPO_ROOT),
        "device": device,
        "workers": workers,
        "cuda_visible_devices": GPU2_UUID if device == "cuda:0" else None,
        "fingerprint_sha256": fingerprint,
        "inputs": input_artifacts,
        "expected_outputs": [str(path.resolve()) for path in stage.outputs],
        "started_at_unix": started,
    }
    _atomic_json(stage.status_path, running)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    if device == "cuda:0":
        environment["CUDA_VISIBLE_DEVICES"] = GPU2_UUID
    else:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    with stage.log_path.open("w", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "stage": stage.key,
                    "attempt": attempt,
                    "command": list(stage.command),
                    "command_shell_display": shlex.join(stage.command),
                    "cwd": str(REPO_ROOT),
                    "started_at_unix": started,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        completed = subprocess.run(
            list(stage.command),
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    finished = time.time()
    base = {
        **running,
        "returncode": completed.returncode,
        "finished_at_unix": finished,
        "elapsed_seconds": finished - started,
        "log": _artifact(stage.log_path),
    }
    if completed.returncode != 0:
        failed = {
            **base,
            "status": "failed",
            "outputs_present_after_failure": [
                _artifact(path) for path in stage.outputs if path.is_file()
            ],
        }
        _atomic_json(stage.status_path, failed)
        raise PostprocessError(
            f"stage {stage.key} failed with return code "
            f"{completed.returncode}; log={stage.log_path}"
        )
    try:
        _validate_json_outputs(stage)
        if stage.index == 1:
            _validate_selected_checkpoint_tree()
        output_artifacts = _artifacts(stage.outputs)
    except (FileNotFoundError, OSError, ValueError, PostprocessError) as exc:
        failed = {
            **base,
            "status": "failed_postcondition",
            "postcondition_error": str(exc),
            "outputs_present_after_failure": [
                _artifact(path) for path in stage.outputs if path.is_file()
            ],
        }
        _atomic_json(stage.status_path, failed)
        raise PostprocessError(
            f"stage {stage.key} output audit failed: {exc}"
        ) from exc
    complete = {
        **base,
        "status": "complete",
        "outputs": output_artifacts,
    }
    _atomic_json(stage.status_path, complete)
    return {
        "stage": stage.key,
        "action": "executed",
        "status_path": str(stage.status_path.resolve()),
        "status_sha256": _file_sha256(stage.status_path),
        "fingerprint_sha256": fingerprint,
        "outputs": output_artifacts,
    }


def dry_run_payload(*, device: str, workers: int) -> dict[str, Any]:
    gate = audit_training_gate(raise_on_error=False)
    stages = build_stages(device=device, workers=workers)
    return {
        "schema": f"{SCHEMA_PREFIX}_dry_run_v1",
        "status": "dry_run",
        "results_root": str(RESULTS_ROOT),
        "data_root": str(DATA_ROOT),
        "python": str(PYTHON),
        "training_gate": gate,
        "will_execute": False,
        "stage_count": len(stages),
        "stages": [
            {
                "stage": stage.key,
                "module": stage.module,
                "command": list(stage.command),
                "expected_output_count": len(stage.outputs),
                "expected_outputs": [
                    str(path.resolve()) for path in stage.outputs
                ],
            }
            for stage in stages
        ],
    }


def _write_overall(
    *,
    status: str,
    gate: Mapping[str, Any],
    stage_records: Sequence[Mapping[str, Any]],
    started: float,
    error: str | None = None,
    artifact_manifest: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema": f"{SCHEMA_PREFIX}_overall_status_v1",
        "status": status,
        "results_root": str(RESULTS_ROOT),
        "training_gate": {
            "path": str(GATE_SNAPSHOT.resolve()),
            "sha256": _file_sha256(GATE_SNAPSHOT),
            "gate_sha256": gate["gate_sha256"],
        },
        "stages": list(stage_records),
        "started_at_unix": started,
        "updated_at_unix": time.time(),
    }
    if error is not None:
        payload["error"] = error
    if artifact_manifest is not None:
        payload["artifact_manifest"] = dict(artifact_manifest)
    _atomic_json(OVERALL_STATUS, payload)


def run_supervisor(
    *,
    device: str,
    workers: int,
    force_rerun: bool = False,
) -> dict[str, Any]:
    # A virtual environment normally exposes ``bin/python`` as a symlink.
    # Artifact inputs remain symlink-rejected elsewhere; the executable is
    # valid as long as its resolved target is a regular file.
    if not PYTHON.is_file():
        raise PostprocessError(f"required Python is missing: {PYTHON}")
    started = time.time()
    POSTPROCESS_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = POSTPROCESS_ROOT / "postprocess.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise PostprocessError(
            f"another post-process supervisor is active: {lock_path}"
        ) from exc

    gate = audit_training_gate()
    _atomic_json(GATE_SNAPSHOT, gate)
    stages = build_stages(device=device, workers=workers)
    stage_records: list[dict[str, Any]] = []
    _write_overall(
        status="running",
        gate=gate,
        stage_records=stage_records,
        started=started,
    )
    try:
        for stage in stages:
            record = _run_stage(
                stage,
                gate=gate,
                device=device,
                workers=workers,
                force_rerun=force_rerun,
            )
            stage_records.append(record)
            _write_overall(
                status="running",
                gate=gate,
                stage_records=stage_records,
                started=started,
            )
    except Exception as exc:
        _write_overall(
            status="failed",
            gate=gate,
            stage_records=stage_records,
            started=started,
            error=str(exc),
        )
        raise

    manifest = {
        "schema": f"{SCHEMA_PREFIX}_artifact_manifest_v1",
        "status": "complete",
        "seed": protocol.TRAINING_SEED,
        "training_gate": _artifact(GATE_SNAPSHOT),
        "stage_statuses": [_artifact(stage.status_path) for stage in stages],
        "stage_logs": [_artifact(stage.log_path) for stage in stages],
        "final_outputs": _artifacts(_final_paths()),
        "checkpoint_roles": list(protocol.CHECKPOINT_ROLES),
        "dataset_specific_fixed_count": len(_fixed_paths()),
        "sirst3_source_fixed_count": len(_source_fixed_paths()),
        "dataset_specific_sweep_count": len(_sweep_paths()),
        "sirst3_source_sweep_count": len(_source_sweep_paths()),
    }
    _atomic_json(ARTIFACT_MANIFEST, manifest)
    manifest_record = _artifact(ARTIFACT_MANIFEST)
    _write_overall(
        status="complete",
        gate=gate,
        stage_records=stage_records,
        started=started,
        artifact_manifest=manifest_record,
    )
    return _load_json(OVERALL_STATUS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda:0"),
        default="cuda:0",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print readiness and exact commands without writing or executing",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="recompute all six stages even when prior hashes still match",
    )
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if args.dry_run and args.force_rerun:
        parser.error("--dry-run cannot be combined with --force-rerun")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print(
            json.dumps(
                dry_run_payload(device=args.device, workers=args.workers),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    try:
        result = run_supervisor(
            device=args.device,
            workers=args.workers,
            force_rerun=args.force_rerun,
        )
    except (FileNotFoundError, OSError, ValueError, PostprocessError) as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                    "results_root": str(RESULTS_ROOT),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
