#!/usr/bin/env python3
"""Publish and verify the resumed TPD-Clean-v3 completion bundle.

The original scientific summarizer remains the sole producer of the seven-gate
comparison.  This module adds a separate, marker-last completion layer for the
four in-place resumed runs.  It binds the interrupted-run evidence, immutable
resume boundaries, resume manifests/logs, replay provenance, final artifacts,
frozen references, and all relevant source locks without overwriting the
original comparison bundle.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import validate_tpd_clean_v3_completion as report_validation  # noqa: E402


DEFAULT_SUMMARIZER = (
    REPO_ROOT / "experiments/summarize_tpd_clean_v3_screen800.py"
)
DEFAULT_POSTPROCESS_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v3_resume_postprocess_source_lock.json"
)

DATASET = "NUDT-SIRST"
RUN_TAG = "screen800_pd_fp32_shared4x5090_v1"
RESUME_SUBDIR = "resume_2x5090_v1"
TARGET_EPOCH = 800
TRAIN_COUNT = 530
TRAIN_BATCHES = 34

PUBLISHED_JSON_NAME = "resume_tpd_clean_v3_screen800_comparison.json"
PUBLISHED_MARKDOWN_NAME = "resume_tpd_clean_v3_screen800_comparison.md"
MANIFEST_NAME = "resume_completion_inputs.json"
MARKER_NAME = "RESUME_COMPLETE.sha256"
MANIFEST_SCHEMA = "sctransnet_tpd_clean_v3_resume_completion_inputs_v1"
POSTPROCESS_LOCK_SCHEMA = (
    "sctransnet_tpd_clean_v3_resume_postprocess_source_lock_v1"
)
TRAINING_LOCK_SCHEMA = "sctransnet_tpd_clean_v3_screen800_source_lock_v1"
RESUME_LOCK_SCHEMA = "sctransnet_tpd_clean_v3_resume_2x_source_lock_v1"
ORIGINAL_LAUNCH_SCHEMA = "sctransnet_tpd_clean_v3_screen800_launch_v1"
RESUME_LAUNCH_SCHEMA = "sctransnet_tpd_clean_v3_resume_2x5090_launch_v1"
BOUNDARY_SCHEMA = "sctransnet_tpd_clean_v3_resume_boundary_v1"
PROVENANCE_SCHEMA = "sctransnet_tpd_clean_v3_resume_provenance_v1"
SEGMENT_SCHEMA = "sctransnet_tpd_clean_v3_resume_segment_v1"
ENGINE_SCHEMA = "sctransnet_tpd_clean_v3_resume_engine_v1"

GPU_UUIDS = {
    0: "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70",
    1: "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640",
    2: "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    3: "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}

BOUNDARY_SNAPSHOT_NAMES = (
    "metrics.jsonl",
    "last.pth.tar",
    "best.pth.tar",
    "best_miou.pth.tar",
    "protocol.json",
    "split.json",
    "original_launch_manifest.json",
    "original_worker.log",
)


@dataclass(frozen=True)
class JobSpec:
    variant: str
    seed: int
    boundary_epoch: int
    original_gpu_index: int
    resume_gpu_index: int
    tag: str

    @property
    def original_gpu_uuid(self) -> str:
        return GPU_UUIDS[self.original_gpu_index]

    @property
    def resume_gpu_uuid(self) -> str:
        return GPU_UUIDS[self.resume_gpu_index]


JOBS = (
    JobSpec("tpd_clean_v3_full", 42, 279, 0, 3, "full-s42"),
    JobSpec("tpd_clean_v3_sal_capacity", 42, 331, 1, 2, "cap-s42"),
    JobSpec("tpd_clean_v3_full", 3407, 323, 2, 2, "full-s3407"),
    JobSpec(
        "tpd_clean_v3_sal_capacity",
        3407,
        372,
        3,
        3,
        "cap-s3407",
    ),
)

EXPECTED_INPUT_COUNTS = {
    "candidate_run_files": 36,
    "original_launch_manifests": 4,
    "original_worker_logs": 4,
    "resume_launch_manifests": 4,
    "resume_worker_logs": 4,
    "resume_provenance_files": 4,
    "resume_segment_files": 4,
    "boundary_manifests": 4,
    "boundary_snapshot_files": 32,
    "frozen_reference_checkpoints": 8,
    "frozen_reference_sweeps": 8,
    "canonical_summarizers": 1,
    "source_locks": 3,
    "total_files": 116,
}


class ResumeCompletionValidationError(ValueError):
    """A resumed-run artifact or completion binding is invalid."""


def _fail(message: str) -> None:
    raise ResumeCompletionValidationError(message)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _require_utc_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str):
        _fail(f"{label}: timestamp must be a string")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        _fail(f"{label}: timestamp is not ISO-8601")
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        _fail(f"{label}: timestamp must include a UTC offset")


def _reject_json_constant(value: str) -> None:
    _fail(f"JSON contains non-finite constant {value}")


def _finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"{label}: non-finite numeric value")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _finite_tree(nested, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _finite_tree(nested, f"{label}[{index}]")


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label}: missing, linked, or non-regular file: {path}")
    return path


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        _fail(f"{label}: missing, linked, or non-directory path: {path}")
    return path.resolve(strict=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _regular(path, label)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except ResumeCompletionValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"{label}: invalid JSON: {exc}")
    if not isinstance(payload, dict):
        _fail(f"{label}: JSON root must be an object")
    _finite_tree(payload, label)
    return payload


def _load_jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], list[str]]:
    _regular(path, label)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        _fail(f"{label}: cannot read: {exc}")
    if not lines:
        _fail(f"{label}: empty JSONL")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        try:
            row = json.loads(line, parse_constant=_reject_json_constant)
        except ResumeCompletionValidationError:
            raise
        except json.JSONDecodeError as exc:
            _fail(f"{label}: invalid JSON row {index}: {exc}")
        if not isinstance(row, dict):
            _fail(f"{label}: row {index} is not an object")
        _finite_tree(row, f"{label}[{index}]")
        rows.append(row)
    return rows, lines


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _engine_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Match the resume engine's compact, canonical JSONL serialization."""

    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _relative(path: Path, root: Path, label: str) -> str:
    resolved = path.resolve(strict=True)
    if not _within(resolved, root):
        _fail(f"{label}: path escaped root: {resolved}")
    return resolved.relative_to(root).as_posix()


def _run_dir(candidate_root: Path, job: JobSpec) -> Path:
    return (
        candidate_root
        / DATASET
        / job.variant
        / f"seed_{job.seed}_{RUN_TAG}"
    ).resolve()


def _resume_root(candidate_root: Path) -> Path:
    return (candidate_root / RESUME_SUBDIR).resolve()


def _boundary_dir(candidate_root: Path, job: JobSpec) -> Path:
    return (
        _resume_root(candidate_root)
        / "boundaries"
        / (
            f"{job.variant}_seed{job.seed}_"
            f"epoch{job.boundary_epoch:03d}"
        )
    ).resolve()


def _source_lock(
    path: Path,
    *,
    repo: Path,
    schema: str,
    label: str,
) -> dict[str, Any]:
    payload = _load_json(path, label)
    if payload.get("schema") != schema:
        _fail(
            f"{label}: schema={payload.get('schema')!r}, expected={schema!r}"
        )
    entries = payload.get("source_sha256")
    if not isinstance(entries, dict) or not entries:
        _fail(f"{label}: source_sha256 must be a non-empty object")
    for relative, expected in entries.items():
        if not isinstance(relative, str) or not _valid_sha256(expected):
            _fail(f"{label}: invalid source entry {relative!r}")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
        ):
            _fail(f"{label}: non-canonical source path {relative!r}")
        source = _regular(
            repo / relative_path, f"{label}.{relative}"
        ).resolve()
        if not _within(source, repo):
            _fail(f"{label}: source path escaped repository: {relative}")
        actual = _sha256_file(source)
        if actual != expected:
            _fail(
                f"{label}: source digest mismatch for {relative}: "
                f"{actual} != {expected}"
            )
    return payload


def _validate_source_locks(
    *,
    repo: Path,
    postprocess_lock_path: Path,
) -> dict[str, Path]:
    training = _regular(
        repo / "experiments/tpd_clean_v3_screen800_source_lock.json",
        "original training source lock",
    ).resolve()
    resume = _regular(
        repo / "experiments/tpd_clean_v3_resume_2x_source_lock.json",
        "resume source lock",
    ).resolve()
    postprocess = _regular(
        postprocess_lock_path, "resume postprocess source lock"
    ).resolve()
    if _relative(
        postprocess, repo, "resume postprocess source lock"
    ) != "experiments/tpd_clean_v3_resume_postprocess_source_lock.json":
        _fail("resume postprocess source lock path is non-canonical")
    _source_lock(
        training,
        repo=repo,
        schema=TRAINING_LOCK_SCHEMA,
        label="original training source lock",
    )
    _source_lock(
        resume,
        repo=repo,
        schema=RESUME_LOCK_SCHEMA,
        label="resume source lock",
    )
    post_payload = _source_lock(
        postprocess,
        repo=repo,
        schema=POSTPROCESS_LOCK_SCHEMA,
        label="resume postprocess source lock",
    )
    if post_payload.get("training_source_lock_sha256") != _sha256_file(
        training
    ):
        _fail("resume postprocess lock does not bind original training lock")
    if post_payload.get("resume_source_lock_sha256") != _sha256_file(resume):
        _fail("resume postprocess lock does not bind resume source lock")
    required_runtime = {
        "experiments/summarize_tpd_clean_v3_screen800.py",
        "experiments/validate_tpd_clean_v3_completion.py",
        "experiments/validate_tpd_clean_v3_resume_completion.py",
        "experiments/run_tpd_clean_v3_resume_finalizer.sh",
        "experiments/launch_tpd_clean_v3_resume_finalizer.sh",
    }
    entries = post_payload["source_sha256"]
    if not required_runtime.issubset(entries):
        _fail(
            "resume postprocess source lock misses runtime entries: "
            f"{sorted(required_runtime - set(entries))}"
        )
    return {
        "training": training,
        "resume": resume,
        "postprocess": postprocess,
    }


def _load_summarizer(
    *,
    repo: Path,
    summarizer_path: Path,
    postprocess_lock_path: Path,
) -> ModuleType:
    expected = repo / "experiments/summarize_tpd_clean_v3_screen800.py"
    summarizer = _regular(summarizer_path, "canonical summarizer").resolve()
    if summarizer != expected.resolve(strict=True):
        _fail(f"canonical summarizer path mismatch: {summarizer}")
    postprocess = _load_json(
        postprocess_lock_path, "resume postprocess source lock"
    )
    relative = _relative(summarizer, repo, "canonical summarizer")
    if postprocess.get("source_sha256", {}).get(relative) != _sha256_file(
        summarizer
    ):
        _fail("resume postprocess lock does not bind canonical summarizer")
    module_name = (
        "_tpd_clean_v3_resume_summary_"
        + _sha256_file(summarizer)[:16]
        + f"_{os.getpid()}"
    )
    spec = importlib.util.spec_from_file_location(module_name, summarizer)
    if spec is None or spec.loader is None:
        _fail("cannot import canonical summarizer")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        _fail(f"canonical summarizer import failed: {exc}")
    required = (
        "SCHEMA",
        "JSON_OUTPUT_NAME",
        "MARKDOWN_OUTPUT_NAME",
        "VARIANTS",
        "SEEDS",
        "ROLE_SPECS",
        "BUDGET_KEYS",
        "render_markdown",
        "build_report",
        "evaluate_engineering_gate",
        "_candidate_run_dir",
        "_reference_paths",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        _fail(f"canonical summarizer missing attributes: {missing}")
    if (
        tuple(module.VARIANTS)
        != ("tpd_clean_v3_full", "tpd_clean_v3_sal_capacity")
        or tuple(module.SEEDS) != (42, 3407)
        or module.RUN_TAG != RUN_TAG
    ):
        _fail("canonical summarizer candidate scope changed")
    return module


def _validate_original_evidence(
    *,
    candidate_root: Path,
    run_dir: Path,
    job: JobSpec,
    training_lock: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    launch_path = (
        candidate_root
        / "launch"
        / f"{job.variant}_seed{job.seed}.json"
    )
    log_path = (
        candidate_root / "logs" / f"{job.variant}_seed{job.seed}.log"
    )
    launch = _load_json(launch_path, f"{job.tag} original launch")
    expected = {
        "schema": ORIGINAL_LAUNCH_SCHEMA,
        "variant": job.variant,
        "seed": job.seed,
        "candidate_family": "spd_anchored_tpd_clean_v3_kcs",
        "gpu_uuid": job.original_gpu_uuid,
        "gpu_name": "NVIDIA GeForce RTX 5090",
    }
    for key, value in expected.items():
        if launch.get(key) != value:
            _fail(f"{job.tag}: original launch {key} mismatch")
    if Path(str(launch.get("run_directory", ""))).resolve() != run_dir:
        _fail(f"{job.tag}: original launch run directory mismatch")
    if Path(str(launch.get("source_lock", ""))).resolve() != training_lock:
        _fail(f"{job.tag}: original launch source lock path mismatch")
    if launch.get("source_lock_sha256") != _sha256_file(training_lock):
        _fail(f"{job.tag}: original launch source lock digest mismatch")
    if not _valid_sha256(launch.get("training_data_sha256")):
        _fail(f"{job.tag}: invalid training-data digest")
    _regular(log_path, f"{job.tag} original worker log")
    lines = log_path.read_text(encoding="utf-8").splitlines()
    expected_start = (
        f"TPDCLEANV3_START variant={job.variant} seed={job.seed} "
        f"gpu_uuid={job.original_gpu_uuid} run_dir={run_dir}"
    )
    if expected_start not in lines:
        _fail(f"{job.tag}: original worker start record missing")
    return launch_path.resolve(), log_path.resolve(), launch


def _validate_boundary(
    *,
    candidate_root: Path,
    run_dir: Path,
    job: JobSpec,
    resume_manifest: Mapping[str, Any],
    original_launch_path: Path,
    original_log_path: Path,
) -> tuple[Path, dict[str, Any], list[str]]:
    boundary_dir = _boundary_dir(candidate_root, job)
    _directory(boundary_dir, f"{job.tag} boundary directory")
    boundary_path = boundary_dir / "boundary.json"
    boundary = _load_json(boundary_path, f"{job.tag} boundary manifest")
    if resume_manifest.get("boundary_manifest_sha256") != _sha256_file(
        boundary_path
    ):
        _fail(f"{job.tag}: resume manifest boundary digest mismatch")
    expected = {
        "schema": BOUNDARY_SCHEMA,
        "variant": job.variant,
        "seed": job.seed,
        "boundary_epoch": job.boundary_epoch,
        "run_directory": str(run_dir),
        "immutable_no_overwrite": True,
    }
    for key, value in expected.items():
        if boundary.get(key) != value:
            _fail(f"{job.tag}: boundary {key} mismatch")
    artifacts = boundary.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(
        BOUNDARY_SNAPSHOT_NAMES
    ):
        _fail(f"{job.tag}: boundary snapshot set mismatch")
    expected_sources = {
        "metrics.jsonl": run_dir / "metrics.jsonl",
        "last.pth.tar": run_dir / "last.pth.tar",
        "best.pth.tar": run_dir / "best.pth.tar",
        "best_miou.pth.tar": run_dir / "best_miou.pth.tar",
        "protocol.json": run_dir / "protocol.json",
        "split.json": run_dir / "split.json",
        "original_launch_manifest.json": original_launch_path,
        "original_worker.log": original_log_path,
    }
    for name, expected_source in expected_sources.items():
        snapshot = _regular(
            boundary_dir / name, f"{job.tag} boundary snapshot {name}"
        )
        record = artifacts[name]
        if not isinstance(record, Mapping) or set(record) != {
            "source",
            "source_sha256",
            "snapshot_sha256",
            "size_bytes",
        }:
            _fail(f"{job.tag}: malformed boundary record {name}")
        actual = _sha256_file(snapshot)
        if (
            Path(str(record["source"])).resolve()
            != expected_source.resolve()
            or record["source_sha256"] != actual
            or record["snapshot_sha256"] != actual
            or record["size_bytes"] != snapshot.stat().st_size
        ):
            _fail(f"{job.tag}: boundary snapshot binding mismatch for {name}")
    for unchanged_name in (
        "protocol.json",
        "split.json",
        "original_launch_manifest.json",
        "original_worker.log",
    ):
        source = expected_sources[unchanged_name]
        if _sha256_file(source) != artifacts[unchanged_name]["source_sha256"]:
            _fail(f"{job.tag}: immutable original source changed: {unchanged_name}")
    boundary_rows, boundary_lines = _load_jsonl(
        boundary_dir / "metrics.jsonl", f"{job.tag} boundary metrics"
    )
    if len(boundary_rows) != job.boundary_epoch:
        _fail(f"{job.tag}: boundary metrics row count mismatch")
    for epoch, row in enumerate(boundary_rows, start=1):
        if row.get("epoch") != epoch or row.get("variant") != job.variant:
            _fail(f"{job.tag}: boundary metrics discontinuity at {epoch}")
    try:
        checkpoint = torch.load(
            boundary_dir / "last.pth.tar",
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        _fail(f"{job.tag}: cannot load boundary last checkpoint: {exc}")
    if not isinstance(checkpoint, Mapping) or (
        checkpoint.get("epoch") != job.boundary_epoch
        or checkpoint.get("variant") != job.variant
        or checkpoint.get("seed") != job.seed
    ):
        _fail(f"{job.tag}: boundary last checkpoint metadata mismatch")
    return boundary_path.resolve(), boundary, boundary_lines


def _validate_resume_manifest_and_log(
    *,
    candidate_root: Path,
    run_dir: Path,
    job: JobSpec,
    original_launch_path: Path,
    original_launch: Mapping[str, Any],
    training_lock: Path,
    resume_lock: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    resume_root = _resume_root(candidate_root)
    manifest_path = (
        resume_root / "manifests" / f"{job.variant}_seed{job.seed}.json"
    )
    log_path = resume_root / "logs" / f"{job.variant}_seed{job.seed}.log"
    manifest = _load_json(manifest_path, f"{job.tag} resume manifest")
    expected = {
        "schema": RESUME_LAUNCH_SCHEMA,
        "variant": job.variant,
        "seed": job.seed,
        "candidate_family": "spd_anchored_tpd_clean_v3_kcs",
        "run_directory": str(run_dir),
        "run_tag": RUN_TAG,
        "boundary_epoch": job.boundary_epoch,
        "target_epoch": TARGET_EPOCH,
        "original_launch_manifest": str(original_launch_path),
        "original_launch_manifest_sha256": _sha256_file(original_launch_path),
        "original_gpu_uuid": job.original_gpu_uuid,
        "resume_gpu_uuid": job.resume_gpu_uuid,
        "resume_gpu_index": job.resume_gpu_index,
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "training_data_sha256": original_launch["training_data_sha256"],
        "source_lock": str(resume_lock),
        "source_lock_sha256": _sha256_file(resume_lock),
        "original_source_lock": str(training_lock),
        "original_source_lock_sha256": _sha256_file(training_lock),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            _fail(f"{job.tag}: resume manifest {key} mismatch")
    if Path(str(manifest.get("boundary_directory", ""))).resolve() != (
        _boundary_dir(candidate_root, job)
    ):
        _fail(f"{job.tag}: resume boundary directory mismatch")
    if not _valid_sha256(manifest.get("boundary_manifest_sha256")):
        _fail(f"{job.tag}: invalid boundary manifest digest")
    policy = manifest.get("policy")
    required_policy = {
        "in_place_resume": True,
        "fresh_run": False,
        "original_results_preserved_by_boundary": True,
        "immutable_resume_boundary": True,
        "paired_variants": True,
        "pre_registered_seeds": [42, 3407],
        "allowed_gpu_indices": [2, 3],
        "concurrent_jobs_per_gpu": 2,
        "counterbalanced_mapping": True,
        "efficiency_comparison_allowed": False,
        "official_test_accessed": False,
        "amp": False,
        "cpu_replay_thread_cap": 1,
    }
    if not isinstance(policy, Mapping) or any(
        policy.get(key) != value for key, value in required_policy.items()
    ):
        _fail(f"{job.tag}: resume policy mismatch")
    resource_snapshot = manifest.get("resource_snapshot")
    required_cpu_environment = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    if not isinstance(resource_snapshot, Mapping) or any(
        resource_snapshot.get(key) != value
        for key, value in required_cpu_environment.items()
    ):
        _fail(f"{job.tag}: resume CPU thread snapshot mismatch")
    _regular(log_path, f"{job.tag} resume worker log")
    lines = [
        line
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_start = (
        f"TPDCLEANV3_RESUME_2X_START variant={job.variant} seed={job.seed} "
        f"gpu_uuid={job.resume_gpu_uuid} "
        f"boundary_epoch={job.boundary_epoch} target_epoch=800 "
        "cpu_threads=1 "
        f"run_dir={run_dir}"
    )
    expected_complete = (
        f"TPDCLEANV3_RESUME_2X_COMPLETE variant={job.variant} "
        f"seed={job.seed} gpu_uuid={job.resume_gpu_uuid} "
        f"boundary_epoch={job.boundary_epoch} epochs=800"
    )
    if expected_start not in lines or not lines or lines[-1] != expected_complete:
        _fail(f"{job.tag}: resume worker start/completion evidence mismatch")
    return manifest_path.resolve(), log_path.resolve(), manifest


def _validate_resume_provenance(
    *,
    repo: Path,
    run_dir: Path,
    boundary: Mapping[str, Any],
    boundary_lines: Sequence[str],
    job: JobSpec,
) -> tuple[Path, Path, dict[str, Any]]:
    provenance_path = run_dir / "resume_provenance.json"
    segments_path = run_dir / "resume_segments.jsonl"
    provenance = _load_json(provenance_path, f"{job.tag} resume provenance")
    segments, _segment_lines = _load_jsonl(
        segments_path, f"{job.tag} resume segments"
    )
    if not segments:
        _fail(f"{job.tag}: expected at least one resume segment")
    canonical_segments = b"".join(
        _engine_json_bytes(segment) for segment in segments
    )
    if segments_path.read_bytes() != canonical_segments:
        _fail(f"{job.tag}: resume segment JSONL is not engine-canonical")
    segment_shas = [
        _sha256_bytes(_engine_json_bytes(segment)) for segment in segments
    ]
    latest_segment_index = len(segments)
    latest_segment_sha = segment_shas[-1]
    provenance_sha = _sha256_file(provenance_path)
    expected_provenance = {
        "schema": PROVENANCE_SCHEMA,
        "engine_schema": ENGINE_SCHEMA,
        "run_directory": str(run_dir),
        "variant": job.variant,
        "dataset": DATASET,
        "seed": job.seed,
        "engine_relative_path": "experiments/resume_tpd_clean_v3.py",
        "segments_file": "resume_segments.jsonl",
        "segments_sha256": _sha256_file(segments_path),
        "segment_count": latest_segment_index,
        "latest_segment_index": latest_segment_index,
        "latest_segment_sha256": latest_segment_sha,
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            _fail(f"{job.tag}: resume provenance {key} mismatch")
    engine_path = repo / "experiments/resume_tpd_clean_v3.py"
    if provenance.get("engine_sha256") != _sha256_file(engine_path):
        _fail(f"{job.tag}: resume engine digest mismatch")
    if (
        provenance.get("protocol_sha256")
        != _sha256_file(run_dir / "protocol.json")
        or provenance.get("split_sha256")
        != _sha256_file(run_dir / "split.json")
    ):
        _fail(f"{job.tag}: provenance protocol/split digest mismatch")
    disclosure = provenance.get("disclosure")
    required_disclosure = {
        "process_restarted": True,
        "model_optimizer_scaler_restored": True,
        "data_shuffle_crop_flip_stream_replayed": True,
        "cuda_bitwise_continuity_claimed": False,
        "legacy_checkpoint_had_full_rng_state": False,
    }
    if not isinstance(disclosure, Mapping) or any(
        disclosure.get(key) is not value
        for key, value in required_disclosure.items()
    ):
        _fail(f"{job.tag}: provenance disclosure mismatch")

    previous_resume_epoch = job.boundary_epoch
    for index, segment in enumerate(segments, start=1):
        resume_epoch = segment.get("resume_from_epoch")
        if (
            not isinstance(resume_epoch, int)
            or isinstance(resume_epoch, bool)
            or not job.boundary_epoch <= resume_epoch < TARGET_EPOCH
        ):
            _fail(f"{job.tag}: resume segment {index} epoch is invalid")
        if index == 1 and resume_epoch != job.boundary_epoch:
            _fail(f"{job.tag}: first resume segment misses fixed boundary")
        # A process may stop after recording a segment but before completing
        # its first epoch.  The next engine-created segment then resumes from
        # the same checkpoint epoch and represents a valid empty interval.
        if index > 1 and resume_epoch < previous_resume_epoch:
            _fail(f"{job.tag}: resume segment epochs are not continuous")
        expected_segment = {
            "schema": SEGMENT_SCHEMA,
            "segment_index": index,
            "resume_from_epoch": resume_epoch,
            "first_training_epoch": resume_epoch + 1,
            "target_epoch": TARGET_EPOCH,
            "expected_resume_epoch": resume_epoch,
            "source_checkpoint": "last.pth.tar",
            "resume_gpu_uuid": job.resume_gpu_uuid,
            "process_restarted": True,
            "model_state_restored_strict": True,
            "adam_state_restored": True,
            "scaler_state_restored": True,
        }
        for key, value in expected_segment.items():
            if segment.get(key) != value:
                _fail(f"{job.tag}: resume segment {index} {key} mismatch")
        source_sha = segment.get("source_checkpoint_sha256")
        if not _valid_sha256(source_sha):
            _fail(f"{job.tag}: resume segment {index} source digest invalid")
        if index == 1 and source_sha != boundary["artifacts"][
            "last.pth.tar"
        ]["snapshot_sha256"]:
            _fail(f"{job.tag}: first resume segment source digest mismatch")
        replay = segment.get("data_stream_replay")
        required_replay = {
            "workers": 0,
            "seed": job.seed,
            "replayed_epochs": resume_epoch,
            "replayed_batches": resume_epoch * TRAIN_BATCHES,
            "replayed_samples": resume_epoch * TRAIN_COUNT,
            "shuffle_generator_replayed": True,
            "crop_flip_python_random_replayed": True,
            "optimization_performed": False,
        }
        if not isinstance(replay, Mapping) or any(
            replay.get(key) != value for key, value in required_replay.items()
        ):
            _fail(
                f"{job.tag}: segment {index} data-stream replay mismatch"
            )
        continuity = segment.get("continuity_claims")
        required_continuity = {
            "model_optimizer_scaler_restored": True,
            "shuffle_crop_flip_stream_replayed": True,
            "same_process_continuity": False,
            "cuda_bitwise_continuity": False,
        }
        if not isinstance(continuity, Mapping) or any(
            continuity.get(key) is not value
            for key, value in required_continuity.items()
        ):
            _fail(f"{job.tag}: segment {index} continuity mismatch")
        previous_resume_epoch = resume_epoch

    final_rows, final_lines = _load_jsonl(
        run_dir / "metrics.jsonl", f"{job.tag} final metrics"
    )
    if len(final_rows) != TARGET_EPOCH:
        _fail(f"{job.tag}: final metrics must contain exactly 800 rows")
    if list(final_lines[: job.boundary_epoch]) != list(boundary_lines):
        _fail(f"{job.tag}: final metrics prefix differs from boundary snapshot")
    for epoch, row in enumerate(final_rows, start=1):
        if row.get("epoch") != epoch or row.get("variant") != job.variant:
            _fail(f"{job.tag}: final metrics discontinuity at epoch {epoch}")
    for index, segment in enumerate(segments, start=1):
        start_epoch = int(segment["first_training_epoch"])
        end_epoch = (
            int(segments[index]["resume_from_epoch"])
            if index < latest_segment_index
            else TARGET_EPOCH
        )
        if end_epoch < start_epoch - 1:
            _fail(f"{job.tag}: resume segment {index} range is invalid")
        observed_provenance_sha: str | None = None
        for epoch in range(start_epoch, end_epoch + 1):
            row = final_rows[epoch - 1]
            row_sha = row.get("resume_provenance_sha256")
            if (
                row.get("resumed") is not True
                or row.get("resume_segment_index") != index
                or not _valid_sha256(row_sha)
            ):
                _fail(
                    f"{job.tag}: resumed metric binding mismatch at "
                    f"epoch {epoch}"
                )
            if observed_provenance_sha is None:
                observed_provenance_sha = str(row_sha)
            elif row_sha != observed_provenance_sha:
                _fail(
                    f"{job.tag}: provenance digest changed within segment "
                    f"{index}"
                )
        if (
            index == latest_segment_index
            and observed_provenance_sha != provenance_sha
        ):
            _fail(f"{job.tag}: latest metrics do not bind final provenance")

    binding = {
        "resume_engine_schema": ENGINE_SCHEMA,
        "resume_provenance_file": "resume_provenance.json",
        "resume_provenance_sha256": provenance_sha,
        "resume_segments_file": "resume_segments.jsonl",
        "resume_segments_sha256": _sha256_file(segments_path),
        "resume_segment_index": latest_segment_index,
        "resume_segment_sha256": latest_segment_sha,
    }
    summary = _load_json(run_dir / "summary.json", f"{job.tag} summary")
    if (
        summary.get("status") != "complete"
        or summary.get("variant") != job.variant
        or summary.get("seed") != job.seed
        or summary.get("official_test_accessed") is not False
    ):
        _fail(f"{job.tag}: final summary metadata mismatch")
    for key, value in binding.items():
        if summary.get(key) != value:
            _fail(f"{job.tag}: summary resume binding mismatch for {key}")
    if summary.get("resume_disclosure") != dict(disclosure):
        _fail(f"{job.tag}: summary resume disclosure mismatch")
    for checkpoint_name in (
        "last.pth.tar",
        "best.pth.tar",
        "best_miou.pth.tar",
    ):
        checkpoint_path = run_dir / checkpoint_name
        _regular(checkpoint_path, f"{job.tag} {checkpoint_name}")
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except Exception as exc:
            _fail(f"{job.tag}: cannot load {checkpoint_name}: {exc}")
        if not isinstance(checkpoint, Mapping):
            _fail(f"{job.tag}: {checkpoint_name} is not a mapping")
        for key, value in binding.items():
            if checkpoint.get(key) != value:
                _fail(
                    f"{job.tag}: {checkpoint_name} resume binding mismatch "
                    f"for {key}"
                )
        if checkpoint.get("resume_disclosure") != dict(disclosure):
            _fail(f"{job.tag}: {checkpoint_name} disclosure mismatch")
    return provenance_path.resolve(), segments_path.resolve(), provenance


def validate_resume_job(
    *,
    repo: Path,
    candidate_root: Path,
    job: JobSpec,
    training_lock: Path,
    resume_lock: Path,
) -> dict[str, Any]:
    run_dir = _directory(_run_dir(candidate_root, job), f"{job.tag} run")
    original_launch_path, original_log_path, original_launch = (
        _validate_original_evidence(
            candidate_root=candidate_root,
            run_dir=run_dir,
            job=job,
            training_lock=training_lock,
        )
    )
    resume_manifest_path, resume_log_path, resume_manifest = (
        _validate_resume_manifest_and_log(
            candidate_root=candidate_root,
            run_dir=run_dir,
            job=job,
            original_launch_path=original_launch_path,
            original_launch=original_launch,
            training_lock=training_lock,
            resume_lock=resume_lock,
        )
    )
    boundary_path, boundary, boundary_lines = _validate_boundary(
        candidate_root=candidate_root,
        run_dir=run_dir,
        job=job,
        resume_manifest=resume_manifest,
        original_launch_path=original_launch_path,
        original_log_path=original_log_path,
    )
    provenance_path, segments_path, provenance = _validate_resume_provenance(
        repo=repo,
        run_dir=run_dir,
        boundary=boundary,
        boundary_lines=boundary_lines,
        job=job,
    )
    return {
        "tag": job.tag,
        "variant": job.variant,
        "seed": job.seed,
        "boundary_epoch": job.boundary_epoch,
        "original_gpu_uuid": job.original_gpu_uuid,
        "resume_gpu_uuid": job.resume_gpu_uuid,
        "resume_gpu_index": job.resume_gpu_index,
        "run_directory": str(run_dir),
        "original_launch_manifest": str(original_launch_path),
        "original_worker_log": str(original_log_path),
        "resume_launch_manifest": str(resume_manifest_path),
        "resume_worker_log": str(resume_log_path),
        "boundary_manifest": str(boundary_path),
        "boundary_manifest_sha256": _sha256_file(boundary_path),
        "resume_provenance": str(provenance_path),
        "resume_provenance_sha256": _sha256_file(provenance_path),
        "resume_segments": str(segments_path),
        "resume_segments_sha256": _sha256_file(segments_path),
        "resume_segment_count": provenance["segment_count"],
        "latest_resume_segment_index": provenance["latest_segment_index"],
        "data_stream_replayed": provenance["disclosure"][
            "data_shuffle_crop_flip_stream_replayed"
        ],
        "same_process_continuity": False,
        "cuda_bitwise_continuity": False,
    }


def audit_resume_evidence(
    *,
    repo: Path,
    candidate_root: Path,
    training_lock: Path,
    resume_lock: Path,
) -> dict[str, Any]:
    repo = _directory(repo, "repository")
    candidate = _directory(candidate_root, "candidate root")
    training = _regular(training_lock, "original training source lock").resolve()
    resume = _regular(resume_lock, "resume source lock").resolve()
    runs = [
        validate_resume_job(
            repo=repo,
            candidate_root=candidate,
            job=job,
            training_lock=training,
            resume_lock=resume,
        )
        for job in JOBS
    ]
    if {record["resume_gpu_index"] for record in runs} != {2, 3}:
        _fail("resume jobs are not restricted to GPU indices 2 and 3")
    multiplicity = {
        index: sum(record["resume_gpu_index"] == index for record in runs)
        for index in (2, 3)
    }
    if multiplicity != {2: 2, 3: 2}:
        _fail(f"resume GPU multiplicity mismatch: {multiplicity}")
    return {
        "status": "complete",
        "candidate_root": str(candidate),
        "run_tag": RUN_TAG,
        "target_epoch": TARGET_EPOCH,
        "run_count": 4,
        "boundary_snapshot_count": 32,
        "resume_gpu_indices": [2, 3],
        "jobs_per_resume_gpu": 2,
        "resume_segment_count": sum(
            record["resume_segment_count"] for record in runs
        ),
        "runs": runs,
    }


def _descriptor(
    identifier: str, category: str, root: str, relative_path: str
) -> dict[str, str]:
    return {
        "id": identifier,
        "category": category,
        "root": root,
        "relative_path": relative_path,
    }


def _expected_descriptors(
    *,
    module: ModuleType,
    roots: Mapping[str, Path],
    summarizer: Path,
    locks: Mapping[str, Path],
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    candidate = roots["candidate"]
    resume_root = _resume_root(candidate)
    common = (
        "protocol.json",
        "split.json",
        "summary.json",
        "metrics.jsonl",
        "last.pth.tar",
    )
    for job in JOBS:
        prefix = f"candidate/{job.variant}/seed_{job.seed}"
        run_dir = _run_dir(candidate, job)
        for name in common:
            entries.append(
                _descriptor(
                    f"{prefix}/{name}",
                    "candidate_run_file",
                    "candidate",
                    _relative(run_dir / name, candidate, name),
                )
            )
        for role_name, role in module.ROLE_SPECS.items():
            for kind, name, category in (
                ("checkpoint", role["checkpoint"], "candidate_run_file"),
                ("sweep", role["sweep"], "candidate_run_file"),
            ):
                entries.append(
                    _descriptor(
                        f"{prefix}/{role_name}/{kind}",
                        category,
                        "candidate",
                        _relative(run_dir / name, candidate, name),
                    )
                )
        original_launch = (
            candidate / "launch" / f"{job.variant}_seed{job.seed}.json"
        )
        original_log = (
            candidate / "logs" / f"{job.variant}_seed{job.seed}.log"
        )
        resume_manifest = (
            resume_root
            / "manifests"
            / f"{job.variant}_seed{job.seed}.json"
        )
        resume_log = (
            resume_root / "logs" / f"{job.variant}_seed{job.seed}.log"
        )
        for identifier, path, category in (
            ("original_launch", original_launch, "original_launch_manifest"),
            ("original_log", original_log, "original_worker_log"),
            ("resume_launch", resume_manifest, "resume_launch_manifest"),
            ("resume_log", resume_log, "resume_worker_log"),
            (
                "resume_provenance",
                run_dir / "resume_provenance.json",
                "resume_provenance_file",
            ),
            (
                "resume_segments",
                run_dir / "resume_segments.jsonl",
                "resume_segment_file",
            ),
        ):
            entries.append(
                _descriptor(
                    f"{prefix}/{identifier}",
                    category,
                    "candidate",
                    _relative(path, candidate, identifier),
                )
            )
        boundary_dir = _boundary_dir(candidate, job)
        entries.append(
            _descriptor(
                f"{prefix}/boundary/manifest",
                "boundary_manifest",
                "candidate",
                _relative(
                    boundary_dir / "boundary.json",
                    candidate,
                    "boundary manifest",
                ),
            )
        )
        for name in BOUNDARY_SNAPSHOT_NAMES:
            entries.append(
                _descriptor(
                    f"{prefix}/boundary/{name}",
                    "boundary_snapshot_file",
                    "candidate",
                    _relative(boundary_dir / name, candidate, name),
                )
            )
    references = module._reference_paths(
        roots["formal"], roots["v2"], roots["reference_miou"]
    )
    for method, roles in references.items():
        for role_name, (run_dir, _variant) in roles.items():
            role = module.ROLE_SPECS[role_name]
            for kind, name, category in (
                (
                    "checkpoint",
                    role["checkpoint"],
                    "frozen_reference_checkpoint",
                ),
                ("sweep", role["sweep"], "frozen_reference_sweep"),
            ):
                root_name = (
                    "formal"
                    if _within(run_dir.resolve(), roots["formal"])
                    else "v2"
                    if _within(run_dir.resolve(), roots["v2"])
                    else "reference_miou"
                )
                entries.append(
                    _descriptor(
                        f"reference/{method}/{role_name}/{kind}",
                        category,
                        root_name,
                        _relative(
                            run_dir / name,
                            roots[root_name],
                            f"reference {method}",
                        ),
                    )
                )
    entries.append(
        _descriptor(
            "source/canonical_summarizer",
            "canonical_summarizer",
            "repo",
            _relative(summarizer, roots["repo"], "canonical summarizer"),
        )
    )
    for name in ("training", "resume", "postprocess"):
        entries.append(
            _descriptor(
                f"lock/{name}",
                "source_lock",
                "repo",
                _relative(locks[name], roots["repo"], f"{name} source lock"),
            )
        )
    identifiers = [entry["id"] for entry in entries]
    if len(identifiers) != len(set(identifiers)):
        _fail("completion descriptors contain duplicate identifiers")
    entries.sort(key=lambda item: item["id"])
    return entries


def _materialize(
    descriptors: Sequence[Mapping[str, str]],
    roots: Mapping[str, Path],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    paths: set[Path] = set()
    for descriptor in descriptors:
        root_name = descriptor["root"]
        if root_name not in roots:
            _fail(f"unknown descriptor root {root_name}")
        path = _regular(
            roots[root_name] / descriptor["relative_path"],
            f"completion input {descriptor['id']}",
        ).resolve()
        if path in paths:
            _fail(f"duplicate completion input path: {path}")
        paths.add(path)
        entries.append(
            {
                **dict(descriptor),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return entries


def _count_inputs(inputs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    categories: dict[str, int] = {}
    for entry in inputs:
        category = str(entry["category"])
        categories[category] = categories.get(category, 0) + 1
    return {
        "candidate_run_files": categories.get("candidate_run_file", 0),
        "original_launch_manifests": categories.get(
            "original_launch_manifest", 0
        ),
        "original_worker_logs": categories.get("original_worker_log", 0),
        "resume_launch_manifests": categories.get(
            "resume_launch_manifest", 0
        ),
        "resume_worker_logs": categories.get("resume_worker_log", 0),
        "resume_provenance_files": categories.get(
            "resume_provenance_file", 0
        ),
        "resume_segment_files": categories.get("resume_segment_file", 0),
        "boundary_manifests": categories.get("boundary_manifest", 0),
        "boundary_snapshot_files": categories.get(
            "boundary_snapshot_file", 0
        ),
        "frozen_reference_checkpoints": categories.get(
            "frozen_reference_checkpoint", 0
        ),
        "frozen_reference_sweeps": categories.get(
            "frozen_reference_sweep", 0
        ),
        "canonical_summarizers": categories.get("canonical_summarizer", 0),
        "source_locks": categories.get("source_lock", 0),
        "total_files": len(inputs),
    }


def _roots(
    *,
    repo: Path,
    candidate: Path,
    formal: Path,
    v2: Path,
    reference_miou: Path,
) -> dict[str, Path]:
    return {
        "repo": repo,
        "candidate": candidate,
        "formal": formal,
        "v2": v2,
        "reference_miou": reference_miou,
    }


def _configuration(
    *,
    repo: Path,
    candidate_root: Path,
    formal_root: Path,
    v2_root: Path,
    reference_miou_root: Path,
    summarizer_path: Path,
    postprocess_lock_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    config = {
        "repo": _directory(repo, "repository"),
        "candidate": _directory(candidate_root, "candidate root"),
        "formal": _directory(formal_root, "formal reference root"),
        "v2": _directory(v2_root, "Clean-v2 reference root"),
        "reference_miou": _directory(
            reference_miou_root, "reference-mIoU root"
        ),
        "summarizer": _regular(
            summarizer_path, "canonical summarizer"
        ).resolve(),
        "postprocess_lock": _regular(
            postprocess_lock_path, "resume postprocess source lock"
        ).resolve(),
    }
    output = output_dir.resolve()
    if not _within(output, config["candidate"]):
        _fail("output directory escaped candidate root")
    if output != (config["candidate"] / DATASET / "comparison").resolve():
        _fail("resume comparison output directory is non-canonical")
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        _fail("resume comparison output path is linked or non-directory")
    config["output"] = output
    return config


def _validate_report(
    *,
    json_path: Path,
    markdown_path: Path,
    module: ModuleType,
    config: Mapping[str, Path],
) -> tuple[dict[str, Any], bytes, bytes]:
    try:
        return report_validation._validate_canonical_report(
            staging_json=json_path,
            staging_markdown=markdown_path,
            module=module,
            repo=config["repo"],
            candidate_root=config["candidate"],
            formal_root=config["formal"],
            v2_root=config["v2"],
            reference_miou_root=config["reference_miou"],
        )
    except report_validation.CompletionValidationError as exc:
        _fail(f"canonical scientific report failed validation: {exc}")


def _manifest_payload(
    *,
    report: Mapping[str, Any],
    json_bytes: bytes,
    markdown_bytes: bytes,
    module: ModuleType,
    roots: Mapping[str, Path],
    config: Mapping[str, Path],
    audit: Mapping[str, Any],
    inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = _count_inputs(inputs)
    if counts != EXPECTED_INPUT_COUNTS:
        _fail(f"completion input counts differ: {counts}")
    return {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "roots": {name: str(path) for name, path in roots.items()},
        "canonical_outputs": {
            "comparison_json": {
                "filename": PUBLISHED_JSON_NAME,
                "size_bytes": len(json_bytes),
                "sha256": _sha256_bytes(json_bytes),
            },
            "comparison_markdown": {
                "filename": PUBLISHED_MARKDOWN_NAME,
                "size_bytes": len(markdown_bytes),
                "sha256": _sha256_bytes(markdown_bytes),
            },
        },
        "source_binding": {
            "summarizer_relative_path": _relative(
                config["summarizer"], roots["repo"], "canonical summarizer"
            ),
            "summarizer_sha256": _sha256_file(config["summarizer"]),
            "resume_validator_relative_path": _relative(
                Path(__file__).resolve(strict=True),
                roots["repo"],
                "resume completion validator",
            ),
            "resume_validator_sha256": _sha256_file(
                Path(__file__).resolve(strict=True)
            ),
            "postprocess_lock_relative_path": _relative(
                config["postprocess_lock"],
                roots["repo"],
                "resume postprocess lock",
            ),
            "postprocess_lock_sha256": _sha256_file(
                config["postprocess_lock"]
            ),
            "postprocess_lock_schema": POSTPROCESS_LOCK_SCHEMA,
        },
        "report_binding": {
            "schema": module.SCHEMA,
            "generated_at_utc": report["generated_at_utc"],
            "status": "complete",
            "decision": report["decision"],
            "engineering_gate_passed": report["engineering_gate_passed"],
        },
        "resume_audit": audit,
        "validated_counts": {
            "candidate_runs": 4,
            "candidate_checkpoints": 8,
            "candidate_sweeps": 8,
            "candidate_metrics_events": 3200,
            "engineering_gates": 7,
            "resume_boundaries": 4,
            "boundary_snapshots": 32,
            "resume_segments": audit["resume_segment_count"],
            "resume_gpus": 2,
        },
        "input_counts": counts,
        "inputs": inputs,
    }


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        _fail(f"refusing to overwrite existing resume output: {path}")
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        _fail(f"resume output appeared concurrently: {path}")


def _marker_bytes(hashes: Mapping[str, str]) -> bytes:
    names = (
        PUBLISHED_JSON_NAME,
        PUBLISHED_MARKDOWN_NAME,
        MANIFEST_NAME,
    )
    return "".join(f"{hashes[name]}  {name}\n" for name in names).encode()


def _validate_marker(output: Path) -> dict[str, str]:
    marker = _regular(output / MARKER_NAME, "resume completion marker")
    lines = marker.read_text(encoding="utf-8").splitlines()
    names = (
        PUBLISHED_JSON_NAME,
        PUBLISHED_MARKDOWN_NAME,
        MANIFEST_NAME,
    )
    if len(lines) != len(names):
        _fail("resume completion marker row count mismatch")
    hashes: dict[str, str] = {}
    for line, name in zip(lines, names):
        match = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line)
        if match is None or match.group(2) != name:
            _fail("resume completion marker format/order mismatch")
        path = _regular(output / name, f"published resume output {name}")
        actual = _sha256_file(path)
        if actual != match.group(1):
            _fail(f"resume completion marker digest mismatch for {name}")
        hashes[name] = actual
    return hashes


def _prepare(
    *,
    config: Mapping[str, Path],
) -> tuple[ModuleType, dict[str, Path], dict[str, Any], list[dict[str, str]]]:
    locks = _validate_source_locks(
        repo=config["repo"],
        postprocess_lock_path=config["postprocess_lock"],
    )
    module = _load_summarizer(
        repo=config["repo"],
        summarizer_path=config["summarizer"],
        postprocess_lock_path=config["postprocess_lock"],
    )
    audit = audit_resume_evidence(
        repo=config["repo"],
        candidate_root=config["candidate"],
        training_lock=locks["training"],
        resume_lock=locks["resume"],
    )
    roots = _roots(
        repo=config["repo"],
        candidate=config["candidate"],
        formal=config["formal"],
        v2=config["v2"],
        reference_miou=config["reference_miou"],
    )
    descriptors = _expected_descriptors(
        module=module,
        roots=roots,
        summarizer=config["summarizer"],
        locks=locks,
    )
    return module, roots, audit, descriptors


def _common_config(
    *,
    output_dir: Path,
    candidate_root: Path,
    formal_root: Path,
    v2_root: Path,
    reference_miou_root: Path,
    repo: Path,
    summarizer_path: Path,
    postprocess_lock_path: Path,
) -> dict[str, Path]:
    return _configuration(
        repo=repo,
        candidate_root=candidate_root,
        formal_root=formal_root,
        v2_root=v2_root,
        reference_miou_root=reference_miou_root,
        summarizer_path=summarizer_path,
        postprocess_lock_path=postprocess_lock_path,
        output_dir=output_dir,
    )


def audit_completion(
    *,
    output_dir: Path,
    candidate_root: Path,
    formal_root: Path,
    v2_root: Path,
    reference_miou_root: Path,
    repo: Path = REPO_ROOT,
    summarizer_path: Path = DEFAULT_SUMMARIZER,
    postprocess_lock_path: Path = DEFAULT_POSTPROCESS_LOCK,
) -> dict[str, Any]:
    config = _common_config(
        output_dir=output_dir,
        candidate_root=candidate_root,
        formal_root=formal_root,
        v2_root=v2_root,
        reference_miou_root=reference_miou_root,
        repo=repo,
        summarizer_path=summarizer_path,
        postprocess_lock_path=postprocess_lock_path,
    )
    _module, roots, audit, descriptors = _prepare(config=config)
    inputs = _materialize(descriptors, roots)
    if _count_inputs(inputs) != EXPECTED_INPUT_COUNTS:
        _fail("resume audit input counts differ")
    return {
        "status": "audited",
        "runs": audit["run_count"],
        "metrics": 3200,
        "boundaries": 4,
        "boundary_snapshots": 32,
        "resume_segments": audit["resume_segment_count"],
        "input_files": len(inputs),
    }


def verify_completion(
    *,
    output_dir: Path,
    candidate_root: Path,
    formal_root: Path,
    v2_root: Path,
    reference_miou_root: Path,
    repo: Path = REPO_ROOT,
    summarizer_path: Path = DEFAULT_SUMMARIZER,
    postprocess_lock_path: Path = DEFAULT_POSTPROCESS_LOCK,
) -> dict[str, Any]:
    config = _common_config(
        output_dir=output_dir,
        candidate_root=candidate_root,
        formal_root=formal_root,
        v2_root=v2_root,
        reference_miou_root=reference_miou_root,
        repo=repo,
        summarizer_path=summarizer_path,
        postprocess_lock_path=postprocess_lock_path,
    )
    module, roots, audit, descriptors = _prepare(config=config)
    marker_hashes = _validate_marker(config["output"])
    json_path = config["output"] / PUBLISHED_JSON_NAME
    markdown_path = config["output"] / PUBLISHED_MARKDOWN_NAME
    report, _json_bytes, _markdown_bytes = _validate_report(
        json_path=json_path,
        markdown_path=markdown_path,
        module=module,
        config=config,
    )
    manifest_path = config["output"] / MANIFEST_NAME
    manifest = _load_json(manifest_path, "resume completion input manifest")
    _require_utc_timestamp(
        manifest.get("created_at_utc"),
        "resume completion input manifest.created_at_utc",
    )
    inputs = _materialize(descriptors, roots)
    expected_manifest = _manifest_payload(
        report=report,
        json_bytes=json_path.read_bytes(),
        markdown_bytes=markdown_path.read_bytes(),
        module=module,
        roots=roots,
        config=config,
        audit=audit,
        inputs=inputs,
    )
    expected_manifest["created_at_utc"] = manifest.get("created_at_utc")
    if manifest != expected_manifest:
        _fail("resume completion manifest differs from current evidence")
    if marker_hashes[MANIFEST_NAME] != _sha256_file(manifest_path):
        _fail("resume completion marker does not bind manifest")
    return {
        "status": "verified",
        "marker": str(config["output"] / MARKER_NAME),
        "decision": report["decision"],
        "engineering_gate_passed": report["engineering_gate_passed"],
        "input_files": len(inputs),
    }


def publish_completion(
    *,
    staging_dir: Path,
    output_dir: Path,
    candidate_root: Path,
    formal_root: Path,
    v2_root: Path,
    reference_miou_root: Path,
    repo: Path = REPO_ROOT,
    summarizer_path: Path = DEFAULT_SUMMARIZER,
    postprocess_lock_path: Path = DEFAULT_POSTPROCESS_LOCK,
) -> dict[str, Any]:
    config = _common_config(
        output_dir=output_dir,
        candidate_root=candidate_root,
        formal_root=formal_root,
        v2_root=v2_root,
        reference_miou_root=reference_miou_root,
        repo=repo,
        summarizer_path=summarizer_path,
        postprocess_lock_path=postprocess_lock_path,
    )
    marker = config["output"] / MARKER_NAME
    if marker.exists() or marker.is_symlink():
        verified = verify_completion(
            output_dir=output_dir,
            candidate_root=candidate_root,
            formal_root=formal_root,
            v2_root=v2_root,
            reference_miou_root=reference_miou_root,
            repo=repo,
            summarizer_path=summarizer_path,
            postprocess_lock_path=postprocess_lock_path,
        )
        return {**verified, "status": "published", "reused": True}
    module, roots, audit, descriptors = _prepare(config=config)
    staging = _directory(staging_dir, "resume staging directory")
    report, json_bytes, markdown_bytes = _validate_report(
        json_path=staging / module.JSON_OUTPUT_NAME,
        markdown_path=staging / module.MARKDOWN_OUTPUT_NAME,
        module=module,
        config=config,
    )
    inputs = _materialize(descriptors, roots)
    manifest = _manifest_payload(
        report=report,
        json_bytes=json_bytes,
        markdown_bytes=markdown_bytes,
        module=module,
        roots=roots,
        config=config,
        audit=audit,
        inputs=inputs,
    )
    manifest_bytes = _canonical_json_bytes(manifest)
    if _materialize(descriptors, roots) != inputs:
        _fail("resume inputs changed while preparing publication")
    config["output"].mkdir(parents=True, exist_ok=True)
    json_path = config["output"] / PUBLISHED_JSON_NAME
    markdown_path = config["output"] / PUBLISHED_MARKDOWN_NAME
    manifest_path = config["output"] / MANIFEST_NAME
    for path in (json_path, markdown_path, manifest_path, marker):
        if path.exists() or path.is_symlink():
            _fail(f"refusing to overwrite existing resume output: {path}")
    _write_new(json_path, json_bytes)
    _write_new(markdown_path, markdown_bytes)
    _write_new(manifest_path, manifest_bytes)
    if _materialize(descriptors, roots) != inputs:
        _fail("resume inputs changed before marker publication")
    output_hashes = {
        PUBLISHED_JSON_NAME: _sha256_file(json_path),
        PUBLISHED_MARKDOWN_NAME: _sha256_file(markdown_path),
        MANIFEST_NAME: _sha256_file(manifest_path),
    }
    expected_hashes = {
        PUBLISHED_JSON_NAME: _sha256_bytes(json_bytes),
        PUBLISHED_MARKDOWN_NAME: _sha256_bytes(markdown_bytes),
        MANIFEST_NAME: _sha256_bytes(manifest_bytes),
    }
    if output_hashes != expected_hashes:
        _fail("published resume outputs changed before marker publication")
    _write_new(marker, _marker_bytes(output_hashes))
    try:
        verified = verify_completion(
            output_dir=output_dir,
            candidate_root=candidate_root,
            formal_root=formal_root,
            v2_root=v2_root,
            reference_miou_root=reference_miou_root,
            repo=repo,
            summarizer_path=summarizer_path,
            postprocess_lock_path=postprocess_lock_path,
        )
    except Exception:
        if marker.is_file() and not marker.is_symlink():
            marker.unlink()
        raise
    return {**verified, "status": "published", "reused": False}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--formal-reference-root", type=Path, required=True)
    parser.add_argument("--v2-reference-root", type=Path, required=True)
    parser.add_argument("--reference-miou-root", type=Path, required=True)
    parser.add_argument("--summarizer", type=Path, default=DEFAULT_SUMMARIZER)
    parser.add_argument(
        "--postprocess-lock", type=Path, default=DEFAULT_POSTPROCESS_LOCK
    )
    parser.add_argument("--output-dir", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and publish the resumed TPD-Clean-v3 bundle"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "verify"):
        command = commands.add_parser(name)
        _add_common(command)
    publish = commands.add_parser("publish")
    _add_common(publish)
    publish.add_argument("--staging-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    common = {
        "output_dir": args.output_dir,
        "candidate_root": args.candidate_root,
        "formal_root": args.formal_reference_root,
        "v2_root": args.v2_reference_root,
        "reference_miou_root": args.reference_miou_root,
        "repo": args.repo,
        "summarizer_path": args.summarizer,
        "postprocess_lock_path": args.postprocess_lock,
    }
    try:
        if args.command == "publish":
            result = publish_completion(
                staging_dir=args.staging_dir, **common
            )
        elif args.command == "verify":
            result = verify_completion(**common)
        else:
            result = audit_completion(**common)
    except (
        ResumeCompletionValidationError,
        report_validation.CompletionValidationError,
        OSError,
        UnicodeError,
    ) as exc:
        print(
            f"TPDCLEANV3_RESUME_COMPLETION_INVALID reason={exc}",
            file=sys.stderr,
        )
        return 1
    print(
        "TPDCLEANV3_RESUME_COMPLETION_OK"
        f" command={args.command}"
        f" status={result['status']}"
        f" inputs={result['input_files']}"
        + (
            f" decision={result['decision']}"
            f" gate={result['engineering_gate_passed']}"
            f" marker={result['marker']}"
            if "decision" in result
            else ""
        )
        + (
            f" reused={str(result['reused']).lower()}"
            if "reused" in result
            else ""
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
