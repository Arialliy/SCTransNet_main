#!/usr/bin/env python3
"""Publish and verify the hash-bound TPD-Clean-v5 screen800 bundle.

The v5 canonical summarizer is the sole producer of the scientific report. This
module only supplies the completion transaction around that report:

* re-run the canonical audit before publication;
* bind the exact 63-file evidence set in ``completion_inputs.json``;
* atomically publish JSON, Markdown, and the input manifest;
* write ``COMPLETE.sha256`` last, with exactly three rows; and
* re-derive and re-hash every input when a completed bundle is verified.

Training outputs, frozen references, smoke reports, and source locks are
strictly read-only here.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_SUMMARIZER = (
    REPO_ROOT / "experiments/summarize_tpd_clean_v5_screen800.py"
)
DEFAULT_POSTPROCESS_LOCK = (
    REPO_ROOT / "experiments/tpd_clean_v5_2x_postprocess_source_lock.json"
)
DEFAULT_REFERENCE_MIOU_ROOT = (
    REPO_ROOT
    / "experiments/results/tpd_clean_screen800_4x5090_v1"
    / "frozen_reference_miou_runs"
)

MANIFEST_NAME = "completion_inputs.json"
MARKER_NAME = "COMPLETE.sha256"
MANIFEST_SCHEMA = "sctransnet_tpd_clean_v5_completion_inputs_v1"

DATASET = "NUDT-SIRST"
VARIANTS = ("tpd_clean_v5_full", "tpd_clean_v5_sal_capacity")
SEEDS = (42, 3407)
RUN_TAG = "screen800_pd_fp32_shared2x5090_v1"
REFERENCE_VARIANTS = ("spd", "tpd")
REFERENCE_RUN_TAG = "formal800_pd_fp32_4x5090_v1"

GPU2_UUID = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
GPU3_UUID = "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
EXPECTED_GPU_UUIDS = frozenset((GPU2_UUID, GPU3_UUID))
EXPECTED_GPU_ASSIGNMENTS = {
    ("tpd_clean_v5_full", 42): GPU2_UUID,
    ("tpd_clean_v5_sal_capacity", 42): GPU3_UUID,
    ("tpd_clean_v5_full", 3407): GPU3_UUID,
    ("tpd_clean_v5_sal_capacity", 3407): GPU2_UUID,
}

EXPECTED_INPUT_COUNTS = {
    "candidate_run_files": 36,
    "candidate_launch_manifests": 4,
    "candidate_worker_logs": 4,
    "frozen_reference_checkpoints": 4,
    "frozen_reference_sweeps": 4,
    "canonical_summarizers": 1,
    "source_locks": 6,
    "smoke_files": 4,
    "total_files": 63,
}

LOCK_SPECS = {
    "training": (
        "experiments/tpd_clean_v5_screen800_2x_source_lock.json",
        "sctransnet_tpd_clean_v5_screen800_2x_source_lock_v1",
    ),
    "postprocess": (
        "experiments/tpd_clean_v5_2x_postprocess_source_lock.json",
        "sctransnet_tpd_clean_v5_2x_postprocess_source_lock_v1",
    ),
    "v4": (
        "experiments/tpd_clean_v4_screen800_2x_source_lock.json",
        "sctransnet_tpd_clean_v4_screen800_2x_source_lock_v1",
    ),
    "v3": (
        "experiments/tpd_clean_v3_screen800_source_lock.json",
        "sctransnet_tpd_clean_v3_screen800_source_lock_v1",
    ),
    "v2": (
        "experiments/tpd_clean_screen800_source_lock.json",
        "sctransnet_tpd_clean_screen800_source_lock_v1",
    ),
    "ner": (
        "experiments/tpd_ner_v1_source_lock.json",
        "sctransnet_tpd_ner_v1_source_lock_v1",
    ),
}

POSTPROCESS_SOURCE_SET = frozenset(
    {
        "experiments/summarize_tpd_clean_v5_screen800.py",
        "experiments/validate_tpd_clean_v5_2x_completion.py",
        "experiments/run_tpd_clean_v5_screen800_2x5090_finalizer.sh",
        "experiments/launch_tpd_clean_v5_screen800_2x5090_finalizer.sh",
        "tests/test_summarize_tpd_clean_v5_screen800.py",
        "tests/test_validate_tpd_clean_v5_2x_completion.py",
        "tests/test_tpd_clean_v5_2x_finalizer.py",
        "experiments/TPD_CLEAN_V5_PROTOCOL.md",
        "experiments/TPD_CLEAN_V5_2GPU_PROTOCOL.md",
        "experiments/tpd_clean_v5_screen800_2x_source_lock.json",
    }
)

REQUIRED_POSTPROCESS_POLICY = {
    "separate_from_training_source_lock": True,
    "does_not_modify_frozen_training_results": True,
    "candidate_null_budget_points_forbidden": True,
    "unused_frozen_reference_null_points_disclosed": True,
    "required_gate_reference_null_points_forbidden": True,
    "automatic_mainline_replacement": False,
}


class CompletionValidationError(ValueError):
    """A publication, manifest, marker, or bound input is invalid."""


def _fail(message: str) -> None:
    raise CompletionValidationError(message)


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


def _reject_json_constant(value: str) -> None:
    _fail(f"JSON contains a non-finite constant: {value}")


def _require_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"{label}: non-finite numeric value")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _require_finite_tree(nested, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _require_finite_tree(nested, f"{label}[{index}]")


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label}: missing, linked, or non-regular file: {path}")


def _resolve_directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        _fail(f"{label}: missing, linked, or non-directory path: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        _fail(f"{label}: cannot resolve {path}: {exc}")
    cursor = path
    while True:
        if cursor.is_symlink():
            _fail(f"{label}: linked directory component: {cursor}")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    return resolved


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _relative_regular_file(root: Path, relative: str, label: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        _fail(f"{label}: unsafe relative path: {relative!r}")
    lexical = root.joinpath(*pure.parts)
    _require_regular_file(lexical, label)
    resolved = lexical.resolve(strict=True)
    if not _within(resolved, root):
        _fail(f"{label}: file escaped expected root {root}: {lexical}")
    return resolved


def _relative_to_root(path: Path, root: Path, label: str) -> str:
    _require_regular_file(path, label)
    resolved = path.resolve(strict=True)
    if not _within(resolved, root):
        _fail(f"{label}: file escaped expected root {root}: {path}")
    return resolved.relative_to(root).as_posix()


def _safe_output_directory(path: Path, candidate_root: Path) -> Path:
    resolved = path.resolve(strict=False)
    if not _within(resolved, candidate_root):
        _fail(
            f"output directory must be inside candidate root "
            f"{candidate_root}: {path}"
        )
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        _fail(f"output directory is linked or not a directory: {path}")
    cursor = resolved
    while cursor != candidate_root:
        if cursor.exists() and cursor.is_symlink():
            _fail(f"output directory has a linked component: {cursor}")
        if candidate_root not in cursor.parents:
            _fail(f"output directory escaped candidate root: {path}")
        cursor = cursor.parent
    return resolved


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except CompletionValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail(f"{label}: invalid JSON at {path}: {exc}")
    if not isinstance(payload, dict):
        _fail(f"{label}: JSON root must be an object")
    _require_finite_tree(payload, label)
    return payload


def _parse_utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str):
        _fail(f"{label}: expected an ISO timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        _fail(f"{label}: invalid ISO timestamp: {exc}")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail(f"{label}: timestamp must include a timezone")
    return value


def _load_source_lock(
    path: Path,
    *,
    repo: Path,
    expected_schema: str,
    label: str,
) -> dict[str, Any]:
    payload = _load_json(path, label)
    if payload.get("schema") != expected_schema:
        _fail(
            f"{label}: schema={payload.get('schema')!r}, "
            f"expected {expected_schema!r}"
        )
    entries = payload.get("source_sha256")
    if not isinstance(entries, dict) or not entries:
        _fail(f"{label}: source_sha256 must be a non-empty object")
    if any(not isinstance(relative, str) for relative in entries):
        _fail(f"{label}: every source path must be a string")
    for relative, expected_sha in entries.items():
        if not _valid_sha256(expected_sha):
            _fail(f"{label}: invalid SHA-256 for {relative!r}")
        source = _relative_regular_file(
            repo, relative, f"{label}.source[{relative}]"
        )
        actual_sha = _sha256_file(source)
        if actual_sha != expected_sha:
            _fail(
                f"{label}: source digest mismatch for {relative}; "
                f"expected={expected_sha} actual={actual_sha}"
            )
    return payload


def _load_all_locks(
    *,
    repo: Path,
    postprocess_lock_path: Path,
    summarizer_path: Path,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    expected_postprocess = _relative_regular_file(
        repo, LOCK_SPECS["postprocess"][0], "canonical postprocess source lock"
    )
    if postprocess_lock_path.resolve(strict=True) != expected_postprocess:
        _fail(
            "postprocess source-lock path is not canonical: "
            f"{postprocess_lock_path}"
        )
    locks: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name, (relative, schema) in LOCK_SPECS.items():
        path = (
            expected_postprocess
            if name == "postprocess"
            else _relative_regular_file(repo, relative, f"{name} source lock")
        )
        payload = _load_source_lock(
            path,
            repo=repo,
            expected_schema=schema,
            label=f"{name} source lock",
        )
        locks[name] = (path, payload)

    training_path, training = locks["training"]
    postprocess = locks["postprocess"][1]
    v4_path = locks["v4"][0]
    if (
        training.get("frozen_v4_source_lock_sha256")
        != _sha256_file(v4_path)
    ):
        _fail("v5 training source lock does not bind the current v4 lock")
    if (
        postprocess.get("training_source_lock_sha256")
        != _sha256_file(training_path)
    ):
        _fail(
            "postprocess source lock does not bind the current "
            "v5 training source lock"
        )
    policy = postprocess.get("policy")
    if not isinstance(policy, Mapping) or any(
        policy.get(key) != expected
        for key, expected in REQUIRED_POSTPROCESS_POLICY.items()
    ):
        _fail("postprocess source-lock policy is incomplete or changed")
    postprocess_entries = postprocess["source_sha256"]
    if set(postprocess_entries) != set(POSTPROCESS_SOURCE_SET):
        _fail(
            "postprocess source-lock exact source set differs; "
            f"missing={sorted(POSTPROCESS_SOURCE_SET - set(postprocess_entries))} "
            f"extra={sorted(set(postprocess_entries) - POSTPROCESS_SOURCE_SET)}"
        )
    expected_summarizer_relative = (
        "experiments/summarize_tpd_clean_v5_screen800.py"
    )
    observed_summarizer_relative = _relative_to_root(
        summarizer_path, repo, "canonical summarizer"
    )
    if observed_summarizer_relative != expected_summarizer_relative:
        _fail(
            "canonical summarizer path differs: "
            f"{observed_summarizer_relative!r}"
        )
    if (
        postprocess_entries.get(expected_summarizer_relative)
        != _sha256_file(summarizer_path)
    ):
        _fail("postprocess source lock does not bind the canonical summarizer")
    validator_relative = (
        "experiments/validate_tpd_clean_v5_2x_completion.py"
    )
    canonical_validator = _relative_regular_file(
        repo, validator_relative, "canonical completion validator"
    )
    canonical_validator_sha = _sha256_file(canonical_validator)
    runtime_validator_sha = _sha256_file(Path(__file__).resolve(strict=True))
    if (
        postprocess_entries.get(validator_relative) != canonical_validator_sha
        or runtime_validator_sha != canonical_validator_sha
    ):
        _fail(
            "postprocess source lock does not bind this exact completion "
            "validator"
        )
    return locks


def _load_canonical_summarizer(
    *,
    repo: Path,
    summarizer_path: Path,
    postprocess_lock_path: Path,
) -> tuple[ModuleType, dict[str, tuple[Path, dict[str, Any]]]]:
    locks = _load_all_locks(
        repo=repo,
        postprocess_lock_path=postprocess_lock_path,
        summarizer_path=summarizer_path,
    )
    summarizer_sha = _sha256_file(summarizer_path)
    module_name = (
        "_tpd_clean_v5_canonical_summary_"
        + summarizer_sha[:16]
        + "_"
        + str(os.getpid())
    )
    spec = importlib.util.spec_from_file_location(module_name, summarizer_path)
    if spec is None or spec.loader is None:
        _fail(f"cannot import canonical summarizer: {summarizer_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        _fail(f"canonical summarizer import failed: {exc}")
    required_attributes = (
        "SCHEMA",
        "JSON_OUTPUT_NAME",
        "MARKDOWN_OUTPUT_NAME",
        "VARIANTS",
        "SEEDS",
        "RUN_TAG",
        "build_report",
        "render_markdown",
    )
    missing = [name for name in required_attributes if not hasattr(module, name)]
    if missing:
        _fail(f"canonical summarizer lacks required attributes: {missing}")
    if Path(module.REPO_ROOT).resolve(strict=True) != repo:
        _fail("canonical summarizer repository root differs")
    if tuple(module.VARIANTS) != VARIANTS or tuple(module.SEEDS) != SEEDS:
        _fail("canonical summarizer candidate scope changed")
    if module.RUN_TAG != RUN_TAG:
        _fail("canonical summarizer run tag changed")
    if module.JSON_OUTPUT_NAME != "tpd_clean_v5_screen800_comparison.json":
        _fail("canonical summarizer JSON output name changed")
    if module.MARKDOWN_OUTPUT_NAME != "tpd_clean_v5_screen800_comparison.md":
        _fail("canonical summarizer Markdown output name changed")
    if module.SCHEMA != "sctransnet_tpd_clean_v5_screen800_comparison_v1":
        _fail("canonical summarizer report schema changed")
    return module, locks


def _validate_report_boundary(
    report: Mapping[str, Any],
    *,
    module: ModuleType,
    candidate_root: Path,
    formal_root: Path,
    smoke_root: Path,
    training_lock: Path,
) -> None:
    if report.get("schema") != module.SCHEMA:
        _fail("report schema is not canonical")
    _parse_utc_timestamp(report.get("generated_at_utc"), "report timestamp")
    if report.get("status") != "complete":
        _fail("incomplete canonical report cannot be published")
    if report.get("gate_evaluated") is not True:
        _fail("engineering gate was not evaluated")
    gate = report.get("engineering_gate_passed")
    if not isinstance(gate, bool):
        _fail("engineering_gate_passed must be boolean")
    expected_decision = (
        "ENGINEERING_GATE_PASS" if gate else "ENGINEERING_GATE_FAIL"
    )
    if report.get("decision") != expected_decision:
        _fail("report decision differs from engineering gate result")
    if report.get("ner_stage_authorized") is not gate:
        _fail("NER-stage authorization differs from engineering gate result")
    for key in (
        "mainline_changed",
        "paper_core_established",
        "stability_claim_supported",
    ):
        if report.get(key) is not False:
            _fail(f"report boundary flag must remain false: {key}")
    boundary = report.get("decision_boundary")
    required_boundary = {
        "gate_only_controls_next_engineering_stage": True,
        "automatic_mainline_replacement": False,
        "mainline_changed": False,
        "paper_core_established": False,
        "stability_claim_supported": False,
    }
    if not isinstance(boundary, Mapping) or any(
        boundary.get(key) is not expected
        for key, expected in required_boundary.items()
    ):
        _fail("report decision boundary changed")
    scope = report.get("scope")
    expected_scope = {
        "candidate_root": candidate_root,
        "formal_reference_root": formal_root,
        "smoke_root": smoke_root,
        "training_source_lock": training_lock,
    }
    if not isinstance(scope, Mapping):
        _fail("report scope is missing")
    if scope.get("dataset") != DATASET:
        _fail("report dataset scope changed")
    if scope.get("candidate_variants") != list(VARIANTS):
        _fail("report variant scope changed")
    if scope.get("model_seeds") != list(SEEDS):
        _fail("report seed scope changed")
    if scope.get("official_test_accessed") is not False:
        _fail("report official-test isolation changed")
    for key, expected_path in expected_scope.items():
        observed = scope.get(key)
        if not isinstance(observed, str):
            _fail(f"report scope path is missing: {key}")
        try:
            observed_path = Path(observed).resolve(strict=True)
        except OSError as exc:
            _fail(f"report scope path cannot resolve: {key}: {exc}")
        if observed_path != expected_path:
            _fail(f"report scope path differs: {key}")
    runs = report.get("candidate_runs")
    expected_run_keys = {
        f"{variant}/seed_{seed}" for variant in VARIANTS for seed in SEEDS
    }
    if not isinstance(runs, Mapping) or set(runs) != expected_run_keys:
        _fail("report candidate run set is not exactly the four registered runs")


def _canonical_comparable(report: Mapping[str, Any]) -> dict[str, Any]:
    comparable = copy.deepcopy(dict(report))
    comparable.pop("generated_at_utc", None)
    return comparable


def _validate_staged_and_rebuilt_report(
    *,
    staging_json: Path,
    staging_markdown: Path,
    module: ModuleType,
    candidate_root: Path,
    formal_root: Path,
    smoke_root: Path,
    training_lock: Path,
) -> tuple[dict[str, Any], bytes, bytes]:
    staged = _load_json(staging_json, "staged canonical report JSON")
    _validate_report_boundary(
        staged,
        module=module,
        candidate_root=candidate_root,
        formal_root=formal_root,
        smoke_root=smoke_root,
        training_lock=training_lock,
    )
    _require_regular_file(staging_markdown, "staged canonical report Markdown")
    try:
        markdown_text = staging_markdown.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        _fail(f"cannot read staged canonical Markdown: {exc}")
    try:
        expected_markdown = module.render_markdown(staged)
    except Exception as exc:
        _fail(f"canonical Markdown rendering failed: {exc}")
    if markdown_text != expected_markdown:
        _fail("staged Markdown is not the canonical rendering of staged JSON")
    try:
        rebuilt = module.build_report(
            candidate_root,
            formal_root,
            smoke_root,
            training_lock,
        )
    except Exception as exc:
        _fail(f"canonical current-artifact audit failed: {exc}")
    if not isinstance(rebuilt, dict):
        _fail("canonical current-artifact audit did not return an object")
    _validate_report_boundary(
        rebuilt,
        module=module,
        candidate_root=candidate_root,
        formal_root=formal_root,
        smoke_root=smoke_root,
        training_lock=training_lock,
    )
    if _canonical_comparable(staged) != _canonical_comparable(rebuilt):
        _fail("staged report differs semantically from a current canonical rebuild")
    return staged, staging_json.read_bytes(), staging_markdown.read_bytes()


def _descriptor(
    identifier: str,
    category: str,
    root_name: str,
    relative_path: str,
) -> dict[str, str]:
    return {
        "id": identifier,
        "category": category,
        "root": root_name,
        "relative_path": relative_path,
    }


def _candidate_run_dir(
    candidate_root: Path, variant: str, seed: int
) -> Path:
    return (
        candidate_root
        / DATASET
        / variant
        / f"seed_{seed}_{RUN_TAG}"
    )


def _validate_launch_and_log(
    *,
    launch_path: Path,
    log_path: Path,
    variant: str,
    seed: int,
    run_dir: Path,
    training_lock: Path,
) -> None:
    launch = _load_json(launch_path, f"launch {variant}/seed={seed}")
    expected = {
        "schema": "sctransnet_tpd_clean_v5_screen800_2x5090_launch_v1",
        "variant": variant,
        "seed": seed,
        "candidate_family": "spd_anchored_tpd_clean_v5_positive_context_selector",
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_uuid": EXPECTED_GPU_ASSIGNMENTS[(variant, seed)],
    }
    for key, value in expected.items():
        if launch.get(key) != value:
            _fail(
                f"launch {variant}/seed={seed}: {key} differs; "
                f"observed={launch.get(key)!r} expected={value!r}"
            )
    try:
        launch_run = Path(str(launch.get("run_directory", ""))).resolve(
            strict=True
        )
        launch_lock = Path(str(launch.get("source_lock", ""))).resolve(
            strict=True
        )
    except OSError as exc:
        _fail(f"launch {variant}/seed={seed}: unresolved path: {exc}")
    if launch_run != run_dir:
        _fail(f"launch {variant}/seed={seed}: run directory differs")
    if launch_lock != training_lock:
        _fail(f"launch {variant}/seed={seed}: training lock differs")
    if launch.get("source_lock_sha256") != _sha256_file(training_lock):
        _fail(f"launch {variant}/seed={seed}: training lock hash differs")
    if not _valid_sha256(launch.get("training_data_sha256")):
        _fail(f"launch {variant}/seed={seed}: invalid training-data hash")
    required_policy = {
        "paired_variants": True,
        "pre_registered_seeds": [42, 3407],
        "fresh_run": True,
        "old_results_preserved": True,
        "shared_resource_screening": True,
        "efficiency_comparison_allowed": False,
        "official_test_accessed": False,
        "amp": False,
    }
    policy = launch.get("policy")
    if not isinstance(policy, Mapping) or any(
        policy.get(key) != value for key, value in required_policy.items()
    ):
        _fail(f"launch {variant}/seed={seed}: policy differs")
    _require_regular_file(log_path, f"worker log {variant}/seed={seed}")
    try:
        lines = [
            line
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as exc:
        _fail(f"worker log {variant}/seed={seed}: cannot read: {exc}")
    completion = (
        f"TPDCLEANV5_2X_COMPLETE variant={variant} seed={seed} "
        f"gpu_uuid={EXPECTED_GPU_ASSIGNMENTS[(variant, seed)]} epochs=800"
    )
    if not lines or lines[-1] != completion:
        _fail(
            f"worker log {variant}/seed={seed}: final non-empty line is not "
            "the exact completion record"
        )


def _expected_input_descriptors(
    *,
    repo: Path,
    candidate_root: Path,
    formal_root: Path,
    reference_miou_root: Path,
    smoke_root: Path,
    summarizer_path: Path,
    postprocess_lock_path: Path,
) -> list[dict[str, str]]:
    descriptors: list[dict[str, str]] = []
    training_lock = _relative_regular_file(
        repo, LOCK_SPECS["training"][0], "v5 training source lock"
    )
    data_hashes: set[str] = set()
    gpu_uuids: set[str] = set()
    candidate_files = (
        ("protocol.json", "candidate_run_file"),
        ("split.json", "candidate_run_file"),
        ("summary.json", "candidate_run_file"),
        ("metrics.jsonl", "candidate_run_file"),
        ("last.pth.tar", "candidate_run_file"),
        ("best.pth.tar", "candidate_checkpoint"),
        ("best_miou.pth.tar", "candidate_checkpoint"),
        ("pd_fa_sweep_best.pth.json", "candidate_sweep"),
        ("pd_fa_sweep_best_miou.pth.json", "candidate_sweep"),
    )
    for variant in VARIANTS:
        for seed in SEEDS:
            run_dir = _candidate_run_dir(candidate_root, variant, seed).resolve()
            if not _within(run_dir, candidate_root):
                _fail(f"candidate run escaped candidate root: {run_dir}")
            prefix = f"candidate/{variant}/seed_{seed}"
            for filename, category in candidate_files:
                descriptors.append(
                    _descriptor(
                        f"{prefix}/{filename}",
                        category,
                        "candidate",
                        _relative_to_root(
                            run_dir / filename,
                            candidate_root,
                            f"{prefix}/{filename}",
                        ),
                    )
                )
            launch_path = (
                candidate_root / "launch" / f"{variant}_seed{seed}.json"
            )
            log_path = (
                candidate_root / "logs" / f"{variant}_seed{seed}.log"
            )
            _validate_launch_and_log(
                launch_path=launch_path,
                log_path=log_path,
                variant=variant,
                seed=seed,
                run_dir=run_dir,
                training_lock=training_lock,
            )
            launch = _load_json(
                launch_path, f"launch hash binding {variant}/seed={seed}"
            )
            data_hashes.add(str(launch["training_data_sha256"]))
            gpu_uuids.add(str(launch["gpu_uuid"]))
            descriptors.extend(
                [
                    _descriptor(
                        f"candidate/launch/{variant}/seed_{seed}",
                        "candidate_launch_manifest",
                        "candidate",
                        _relative_to_root(
                            launch_path,
                            candidate_root,
                            f"launch {variant}/seed={seed}",
                        ),
                    ),
                    _descriptor(
                        f"candidate/log/{variant}/seed_{seed}",
                        "candidate_worker_log",
                        "candidate",
                        _relative_to_root(
                            log_path,
                            candidate_root,
                            f"worker log {variant}/seed={seed}",
                        ),
                    ),
                ]
            )
    if gpu_uuids != EXPECTED_GPU_UUIDS:
        _fail("launch manifests do not bind exactly physical GPU2 and GPU3")
    if len(data_hashes) != 1:
        _fail("launch manifests do not share one training-data hash")

    for variant in REFERENCE_VARIANTS:
        formal_run = (
            formal_root
            / DATASET
            / variant
            / f"seed_42_{REFERENCE_RUN_TAG}"
        )
        miou_run = (
            reference_miou_root
            / DATASET
            / variant
            / f"seed_42_{REFERENCE_RUN_TAG}"
        )
        for role, root_name, root, run_dir, checkpoint, sweep in (
            (
                "pd_primary",
                "formal",
                formal_root,
                formal_run,
                "best.pth.tar",
                "pd_fa_sweep_best.pth.json",
            ),
            (
                "miou_primary",
                "reference_miou",
                reference_miou_root,
                miou_run,
                "best_miou.pth.tar",
                "pd_fa_sweep_best_miou.pth.json",
            ),
        ):
            for kind, filename, category in (
                ("checkpoint", checkpoint, "frozen_reference_checkpoint"),
                ("sweep", sweep, "frozen_reference_sweep"),
            ):
                descriptors.append(
                    _descriptor(
                        f"reference/{variant}/{role}/{kind}",
                        category,
                        root_name,
                        _relative_to_root(
                            run_dir / filename,
                            root,
                            f"frozen {variant}/{role}/{kind}",
                        ),
                    )
                )

    descriptors.append(
        _descriptor(
            "source/canonical_summarizer",
            "canonical_summarizer",
            "repo",
            _relative_to_root(
                summarizer_path, repo, "canonical summarizer"
            ),
        )
    )
    for name, (relative, _schema) in LOCK_SPECS.items():
        path = (
            postprocess_lock_path
            if name == "postprocess"
            else repo / relative
        )
        descriptors.append(
            _descriptor(
                f"lock/{name}",
                "source_lock",
                "repo",
                _relative_to_root(path, repo, f"{name} source lock"),
            )
        )
    for filename in (
        "SMOKE_REPORTS.sha256",
        "cpu_all.json",
        "gpu2_full.json",
        "gpu3_capacity.json",
    ):
        descriptors.append(
            _descriptor(
                f"smoke/{filename}",
                "smoke_file",
                "smoke",
                _relative_to_root(
                    smoke_root / filename, smoke_root, f"smoke/{filename}"
                ),
            )
        )

    identifiers = [descriptor["id"] for descriptor in descriptors]
    if len(identifiers) != len(set(identifiers)):
        _fail("completion input descriptors contain duplicate identifiers")
    descriptors.sort(key=lambda descriptor: descriptor["id"])
    return descriptors


def _root_map(
    *,
    repo: Path,
    candidate_root: Path,
    formal_root: Path,
    reference_miou_root: Path,
    smoke_root: Path,
) -> dict[str, Path]:
    return {
        "repo": repo,
        "candidate": candidate_root,
        "formal": formal_root,
        "reference_miou": reference_miou_root,
        "smoke": smoke_root,
    }


def _materialize_inputs(
    descriptors: Sequence[Mapping[str, str]],
    roots: Mapping[str, Path],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    observed_paths: set[Path] = set()
    for descriptor in descriptors:
        root_name = descriptor["root"]
        if root_name not in roots:
            _fail(f"unknown completion input root: {root_name}")
        path = _relative_regular_file(
            roots[root_name],
            descriptor["relative_path"],
            f"completion input {descriptor['id']}",
        )
        if path in observed_paths:
            _fail(f"completion input path is duplicated: {path}")
        observed_paths.add(path)
        entries.append(
            {
                **dict(descriptor),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return entries


def _count_inputs(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    categories: dict[str, int] = {}
    for entry in entries:
        category = str(entry["category"])
        categories[category] = categories.get(category, 0) + 1
    return {
        "candidate_run_files": (
            categories.get("candidate_run_file", 0)
            + categories.get("candidate_checkpoint", 0)
            + categories.get("candidate_sweep", 0)
        ),
        "candidate_launch_manifests": categories.get(
            "candidate_launch_manifest", 0
        ),
        "candidate_worker_logs": categories.get("candidate_worker_log", 0),
        "frozen_reference_checkpoints": categories.get(
            "frozen_reference_checkpoint", 0
        ),
        "frozen_reference_sweeps": categories.get(
            "frozen_reference_sweep", 0
        ),
        "canonical_summarizers": categories.get("canonical_summarizer", 0),
        "source_locks": categories.get("source_lock", 0),
        "smoke_files": categories.get("smoke_file", 0),
        "total_files": len(entries),
    }


def _validate_exact_counts(entries: Sequence[Mapping[str, Any]]) -> None:
    counts = _count_inputs(entries)
    if counts != EXPECTED_INPUT_COUNTS:
        _fail(
            "completion input counts differ; "
            f"observed={counts} expected={EXPECTED_INPUT_COUNTS}"
        )


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


def _manifest_payload(
    *,
    report: Mapping[str, Any],
    module: ModuleType,
    roots: Mapping[str, Path],
    summarizer_path: Path,
    postprocess_lock_path: Path,
    inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    _validate_exact_counts(inputs)
    return {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "report_schema": report["schema"],
        "decision": report["decision"],
        "engineering_gate_passed": report["engineering_gate_passed"],
        "outputs": {
            "json": module.JSON_OUTPUT_NAME,
            "markdown": module.MARKDOWN_OUTPUT_NAME,
            "manifest": MANIFEST_NAME,
            "marker": MARKER_NAME,
        },
        "roots": {name: str(path) for name, path in sorted(roots.items())},
        "canonical_summarizer": str(summarizer_path),
        "postprocess_source_lock": str(postprocess_lock_path),
        "input_counts": dict(EXPECTED_INPUT_COUNTS),
        "inputs": inputs,
        "policy": {
            "exact_input_set": True,
            "regular_files_only": True,
            "symlinks_forbidden": True,
            "canonical_rebuild_before_publish": True,
            "marker_written_last": True,
            "marker_rows": 3,
            "automatic_mainline_replacement": False,
        },
    }


def _marker_bytes(
    hashes: Mapping[str, str], ordered_names: Sequence[str]
) -> bytes:
    if len(ordered_names) != 3 or set(ordered_names) != set(hashes):
        _fail("completion marker must bind exactly three named outputs")
    for digest in hashes.values():
        if not _valid_sha256(digest):
            _fail("completion marker received an invalid SHA-256")
    return (
        "\n".join(f"{hashes[name]}  {name}" for name in ordered_names) + "\n"
    ).encode("utf-8")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _resolve_configuration(
    *,
    output_dir: Path,
    candidate_root: Path,
    formal_root: Path,
    reference_miou_root: Path,
    smoke_root: Path,
    repo: Path,
    summarizer_path: Path,
    postprocess_lock_path: Path,
    staging_dir: Path | None = None,
) -> dict[str, Path]:
    resolved = {
        "repo": _resolve_directory(repo, "repository root"),
        "candidate": _resolve_directory(candidate_root, "candidate root"),
        "formal": _resolve_directory(formal_root, "formal-reference root"),
        "reference_miou": _resolve_directory(
            reference_miou_root, "reference-mIoU root"
        ),
        "smoke": _resolve_directory(smoke_root, "smoke root"),
    }
    for name in ("formal", "reference_miou", "smoke"):
        other = resolved[name]
        candidate = resolved["candidate"]
        if (
            other == candidate
            or _within(other, candidate)
                or _within(candidate, other)
        ):
            _fail(f"candidate and {name} roots overlap")
    output = _safe_output_directory(output_dir, resolved["candidate"])
    _require_regular_file(summarizer_path, "canonical summarizer")
    _require_regular_file(
        postprocess_lock_path, "postprocess source lock"
    )
    summarizer = summarizer_path.resolve(strict=True)
    postprocess = postprocess_lock_path.resolve(strict=True)
    _relative_to_root(summarizer, resolved["repo"], "canonical summarizer")
    _relative_to_root(
        postprocess, resolved["repo"], "postprocess source lock"
    )
    result = {
        **resolved,
        "output": output,
        "summarizer": summarizer,
        "postprocess_lock": postprocess,
    }
    if staging_dir is not None:
        staging = _resolve_directory(staging_dir, "staging directory")
        if not _within(staging, resolved["candidate"]):
            _fail("staging directory must be inside candidate root")
        if staging == output:
            _fail("staging and publication directories must differ")
        result["staging"] = staging
    return result


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    report: Mapping[str, Any],
    module: ModuleType,
    roots: Mapping[str, Path],
    summarizer_path: Path,
    postprocess_lock_path: Path,
    expected_inputs: list[dict[str, Any]],
) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        _fail("completion input manifest schema differs")
    _parse_utc_timestamp(manifest.get("created_at_utc"), "manifest timestamp")
    expected_scalars = {
        "report_schema": report["schema"],
        "decision": report["decision"],
        "engineering_gate_passed": report["engineering_gate_passed"],
        "canonical_summarizer": str(summarizer_path),
        "postprocess_source_lock": str(postprocess_lock_path),
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            _fail(f"completion input manifest differs: {key}")
    expected_outputs = {
        "json": module.JSON_OUTPUT_NAME,
        "markdown": module.MARKDOWN_OUTPUT_NAME,
        "manifest": MANIFEST_NAME,
        "marker": MARKER_NAME,
    }
    if manifest.get("outputs") != expected_outputs:
        _fail("completion input manifest output set differs")
    expected_roots = {
        name: str(path) for name, path in sorted(roots.items())
    }
    if manifest.get("roots") != expected_roots:
        _fail("completion input manifest root binding differs")
    if manifest.get("input_counts") != EXPECTED_INPUT_COUNTS:
        _fail("completion input manifest exact counts differ")
    policy = manifest.get("policy")
    expected_policy = {
        "exact_input_set": True,
        "regular_files_only": True,
        "symlinks_forbidden": True,
        "canonical_rebuild_before_publish": True,
        "marker_written_last": True,
        "marker_rows": 3,
        "automatic_mainline_replacement": False,
    }
    if policy != expected_policy:
        _fail("completion input manifest policy differs")
    if manifest.get("inputs") != expected_inputs:
        _fail("completion input manifest differs from current exact input set")
    _validate_exact_counts(expected_inputs)


def _read_marker(
    marker_path: Path, *, expected_names: Sequence[str]
) -> dict[str, str]:
    _require_regular_file(marker_path, "completion marker")
    try:
        lines = marker_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        _fail(f"cannot read completion marker: {exc}")
    if len(lines) != 3:
        _fail("completion marker must contain exactly three rows")
    observed: dict[str, str] = {}
    for index, line in enumerate(lines):
        digest, separator, name = line.partition("  ")
        if (
            separator != "  "
            or not _valid_sha256(digest)
            or name != expected_names[index]
            or name in observed
        ):
            _fail(f"invalid completion marker row {index + 1}: {line!r}")
        observed[name] = digest
    return observed


def verify_completion(
    *,
    output_dir: Path,
    candidate_root: Path,
    formal_root: Path,
    reference_miou_root: Path,
    smoke_root: Path,
    repo: Path = REPO_ROOT,
    summarizer_path: Path = DEFAULT_SUMMARIZER,
    postprocess_lock_path: Path = DEFAULT_POSTPROCESS_LOCK,
) -> dict[str, Any]:
    config = _resolve_configuration(
        output_dir=output_dir,
        candidate_root=candidate_root,
        formal_root=formal_root,
        reference_miou_root=reference_miou_root,
        smoke_root=smoke_root,
        repo=repo,
        summarizer_path=summarizer_path,
        postprocess_lock_path=postprocess_lock_path,
    )
    output = _resolve_directory(config["output"], "publication directory")
    module, _locks = _load_canonical_summarizer(
        repo=config["repo"],
        summarizer_path=config["summarizer"],
        postprocess_lock_path=config["postprocess_lock"],
    )
    json_path = output / module.JSON_OUTPUT_NAME
    markdown_path = output / module.MARKDOWN_OUTPUT_NAME
    manifest_path = output / MANIFEST_NAME
    marker_path = output / MARKER_NAME
    names = (
        module.JSON_OUTPUT_NAME,
        module.MARKDOWN_OUTPUT_NAME,
        MANIFEST_NAME,
    )
    marker_hashes = _read_marker(marker_path, expected_names=names)
    for name, path in zip(
        names, (json_path, markdown_path, manifest_path)
    ):
        _require_regular_file(path, f"published output {name}")
        actual = _sha256_file(path)
        if marker_hashes[name] != actual:
            _fail(f"completion marker digest differs for {name}")

    report = _load_json(json_path, "published canonical report")
    _validate_report_boundary(
        report,
        module=module,
        candidate_root=config["candidate"],
        formal_root=config["formal"],
        smoke_root=config["smoke"],
        training_lock=_relative_regular_file(
            config["repo"],
            LOCK_SPECS["training"][0],
            "v5 training source lock",
        ),
    )
    _require_regular_file(markdown_path, "published canonical Markdown")
    if markdown_path.read_text(encoding="utf-8") != module.render_markdown(
        report
    ):
        _fail("published Markdown is not canonical for published JSON")
    descriptors = _expected_input_descriptors(
        repo=config["repo"],
        candidate_root=config["candidate"],
        formal_root=config["formal"],
        reference_miou_root=config["reference_miou"],
        smoke_root=config["smoke"],
        summarizer_path=config["summarizer"],
        postprocess_lock_path=config["postprocess_lock"],
    )
    roots = _root_map(
        repo=config["repo"],
        candidate_root=config["candidate"],
        formal_root=config["formal"],
        reference_miou_root=config["reference_miou"],
        smoke_root=config["smoke"],
    )
    current_inputs = _materialize_inputs(descriptors, roots)
    manifest = _load_json(manifest_path, "completion input manifest")
    _validate_manifest(
        manifest,
        report=report,
        module=module,
        roots=roots,
        summarizer_path=config["summarizer"],
        postprocess_lock_path=config["postprocess_lock"],
        expected_inputs=current_inputs,
    )
    return {
        "status": "verified",
        "decision": report["decision"],
        "engineering_gate_passed": report["engineering_gate_passed"],
        "input_files": len(current_inputs),
        "marker": str(marker_path),
    }


def publish_completion(
    *,
    staging_dir: Path,
    output_dir: Path,
    candidate_root: Path,
    formal_root: Path,
    reference_miou_root: Path,
    smoke_root: Path,
    repo: Path = REPO_ROOT,
    summarizer_path: Path = DEFAULT_SUMMARIZER,
    postprocess_lock_path: Path = DEFAULT_POSTPROCESS_LOCK,
) -> dict[str, Any]:
    config = _resolve_configuration(
        output_dir=output_dir,
        candidate_root=candidate_root,
        formal_root=formal_root,
        reference_miou_root=reference_miou_root,
        smoke_root=smoke_root,
        repo=repo,
        summarizer_path=summarizer_path,
        postprocess_lock_path=postprocess_lock_path,
        staging_dir=staging_dir,
    )
    marker_path = config["output"] / MARKER_NAME
    if marker_path.exists() or marker_path.is_symlink():
        verified = verify_completion(
            output_dir=config["output"],
            candidate_root=config["candidate"],
            formal_root=config["formal"],
            reference_miou_root=config["reference_miou"],
            smoke_root=config["smoke"],
            repo=config["repo"],
            summarizer_path=config["summarizer"],
            postprocess_lock_path=config["postprocess_lock"],
        )
        return {**verified, "status": "reused", "reused": True}

    module, _locks = _load_canonical_summarizer(
        repo=config["repo"],
        summarizer_path=config["summarizer"],
        postprocess_lock_path=config["postprocess_lock"],
    )
    training_lock = _relative_regular_file(
        config["repo"], LOCK_SPECS["training"][0], "v5 training source lock"
    )
    report, json_bytes, markdown_bytes = _validate_staged_and_rebuilt_report(
        staging_json=config["staging"] / module.JSON_OUTPUT_NAME,
        staging_markdown=config["staging"] / module.MARKDOWN_OUTPUT_NAME,
        module=module,
        candidate_root=config["candidate"],
        formal_root=config["formal"],
        smoke_root=config["smoke"],
        training_lock=training_lock,
    )
    descriptors = _expected_input_descriptors(
        repo=config["repo"],
        candidate_root=config["candidate"],
        formal_root=config["formal"],
        reference_miou_root=config["reference_miou"],
        smoke_root=config["smoke"],
        summarizer_path=config["summarizer"],
        postprocess_lock_path=config["postprocess_lock"],
    )
    roots = _root_map(
        repo=config["repo"],
        candidate_root=config["candidate"],
        formal_root=config["formal"],
        reference_miou_root=config["reference_miou"],
        smoke_root=config["smoke"],
    )
    inputs = _materialize_inputs(descriptors, roots)
    manifest = _manifest_payload(
        report=report,
        module=module,
        roots=roots,
        summarizer_path=config["summarizer"],
        postprocess_lock_path=config["postprocess_lock"],
        inputs=inputs,
    )
    manifest_bytes = _canonical_json_bytes(manifest)
    if _materialize_inputs(descriptors, roots) != inputs:
        _fail("completion inputs changed while preparing publication")

    config["output"].mkdir(parents=True, exist_ok=True)
    json_path = config["output"] / module.JSON_OUTPUT_NAME
    markdown_path = config["output"] / module.MARKDOWN_OUTPUT_NAME
    manifest_path = config["output"] / MANIFEST_NAME
    _atomic_write_bytes(json_path, json_bytes)
    _atomic_write_bytes(markdown_path, markdown_bytes)
    _atomic_write_bytes(manifest_path, manifest_bytes)

    if _materialize_inputs(descriptors, roots) != inputs:
        _fail("completion inputs changed before marker publication")
    output_hashes = {
        module.JSON_OUTPUT_NAME: _sha256_file(json_path),
        module.MARKDOWN_OUTPUT_NAME: _sha256_file(markdown_path),
        MANIFEST_NAME: _sha256_file(manifest_path),
    }
    expected_hashes = {
        module.JSON_OUTPUT_NAME: _sha256_bytes(json_bytes),
        module.MARKDOWN_OUTPUT_NAME: _sha256_bytes(markdown_bytes),
        MANIFEST_NAME: _sha256_bytes(manifest_bytes),
    }
    if output_hashes != expected_hashes:
        _fail("published outputs changed before completion marker")
    ordered_names = (
        module.JSON_OUTPUT_NAME,
        module.MARKDOWN_OUTPUT_NAME,
        MANIFEST_NAME,
    )
    _atomic_write_bytes(
        marker_path, _marker_bytes(output_hashes, ordered_names)
    )
    try:
        verified = verify_completion(
            output_dir=config["output"],
            candidate_root=config["candidate"],
            formal_root=config["formal"],
            reference_miou_root=config["reference_miou"],
            smoke_root=config["smoke"],
            repo=config["repo"],
            summarizer_path=config["summarizer"],
            postprocess_lock_path=config["postprocess_lock"],
        )
    except Exception:
        if marker_path.is_file() and not marker_path.is_symlink():
            marker_path.unlink()
        raise
    return {**verified, "status": "published", "reused": False}


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--formal-reference-root", type=Path, required=True)
    parser.add_argument(
        "--reference-miou-root",
        type=Path,
        default=DEFAULT_REFERENCE_MIOU_ROOT,
    )
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--summarizer", type=Path, default=DEFAULT_SUMMARIZER)
    parser.add_argument(
        "--postprocess-lock", type=Path, default=DEFAULT_POSTPROCESS_LOCK
    )
    parser.add_argument("--output-dir", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish or verify the exact 63-input TPD-Clean-v5 "
            "screen800 completion bundle."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish_parser = subparsers.add_parser(
        "publish", help="publish canonical outputs with marker written last"
    )
    _add_common_arguments(publish_parser)
    publish_parser.add_argument("--staging-dir", type=Path, required=True)
    verify_parser = subparsers.add_parser(
        "verify", help="re-derive and verify all current completion inputs"
    )
    _add_common_arguments(verify_parser)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    common = {
        "output_dir": args.output_dir,
        "candidate_root": args.candidate_root,
        "formal_root": args.formal_reference_root,
        "reference_miou_root": args.reference_miou_root,
        "smoke_root": args.smoke_root,
        "repo": args.repo,
        "summarizer_path": args.summarizer,
        "postprocess_lock_path": args.postprocess_lock,
    }
    try:
        if args.command == "publish":
            result = publish_completion(
                staging_dir=args.staging_dir, **common
            )
        else:
            result = verify_completion(**common)
    except (CompletionValidationError, OSError, UnicodeError) as exc:
        print(
            f"TPDCLEANV5_2X_COMPLETION_INVALID reason={exc}",
            file=sys.stderr,
        )
        return 1
    print(
        "TPDCLEANV5_2X_COMPLETION_OK"
        f" command={args.command}"
        f" status={result['status']}"
        f" decision={result['decision']}"
        f" gate={str(result['engineering_gate_passed']).lower()}"
        f" inputs={result['input_files']}"
        f" marker={result['marker']}"
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
