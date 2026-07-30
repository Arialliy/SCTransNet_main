#!/usr/bin/env python3
"""Post-training closure for the *new* fixed-seed-42 B/D replay.

This adapter is intentionally separate from the historical seed-42 module
experiments and from the 3407/426780603 engineering replication.  It accepts
exactly four checkpoint-local evaluations:

    B / each arm's own best_miou.pth.tar
    B / each arm's own best.pth.tar
    D / each arm's own best_miou.pth.tar
    D / each arm's own best.pth.tar

The lossless prediction collector, closed-interval threshold sweep, paired
image bootstrap, and Gate comparison primitives are reused from the existing
certification implementation.  This file only adapts their run identity from
the two-engineering-seed matrix to the new seed-42 replay matrix.

Plan/dry-run modes are read-only and never initialize CUDA.  Result, manifest,
paired-screen, Gate, and closure files are canonical write-once artifacts.
No result produced here establishes a paper-core or stability claim.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import (  # noqa: E402
    collect_final_model_validation_statistics as statistics_cache,
)
from experiments import (  # noqa: E402
    adjudicate_final_model_engineering_gate as gate_core,
)
from experiments import (  # noqa: E402
    analyze_final_model_engineering_paired_screen as paired_core,
)
from experiments import (  # noqa: E402
    evaluate_final_model_engineering_replication_pd_fa as evaluator,
)
from experiments import (  # noqa: E402
    final_model_replication_exact_core as frozen_replication_core,
)
from experiments import (  # noqa: E402
    final_model_seed42_certification_replay_contract as replay_contract,
)
from experiments import (  # noqa: E402
    final_model_seed42_certification_replay_exact_core as replay_core,
)
from experiments import (  # noqa: E402
    freeze_final_model_certification_parent_lock as parent_lock,
)
from experiments import (  # noqa: E402
    freeze_final_model_certification_source_lock as certification_source_lock,
)
from experiments import (  # noqa: E402
    freeze_final_model_seed42_certification_replay_source_lock
    as replay_source_lock,
)
from experiments import (  # noqa: E402
    summarize_final_model_engineering_replication as summary_core,
)


SCHEMA = "sctransnet_final_model_seed42_replay_posttraining_adapter_v1"
SUMMARY_SCHEMA = "sctransnet_final_model_seed42_replay_summary_v1"
MANIFEST_SCHEMA = (
    "sctransnet_final_model_seed42_replay_checkpoint_local_pd_fa_manifest_v1"
)
PAIRED_SCHEMA = "sctransnet_final_model_seed42_replay_paired_screen_v1"
GATE_SCHEMA = "sctransnet_final_model_seed42_replay_gate_v1"
CLOSURE_SCHEMA = "sctransnet_final_model_seed42_replay_closure_v1"
ACTION_SCHEMA = "sctransnet_final_model_seed42_replay_posttraining_action_v1"

SCOPE = "new_fixed_seed42_b_d_certification_replay_only"
TRAJECTORY_SEED = 42
EXPECTED_RUN_COUNT = 2
EXPECTED_CHECKPOINT_COUNT = 4
EXPECTED_SWEEP_COUNT = 4
FIXED_THRESHOLD = 0.5
CHECKPOINT_SPECS = evaluator.CHECKPOINT_SPECS
SELECTION_ROLES = tuple(spec[1] for spec in CHECKPOINT_SPECS)
PRIMARY_SELECTION_ROLE = "primary_best_miou"
SECONDARY_SELECTION_ROLE = "secondary_best_pd"

DEFAULT_OUTPUT_ROOT = replay_contract.DEFAULT_OUTPUT_ROOT
DEFAULT_REPLAY_CONTRACT = replay_contract.DEFAULT_CONTRACT
DEFAULT_MANIFEST_DIRECTORY = replay_contract.DEFAULT_MANIFEST_DIRECTORY
DEFAULT_REPLAY_SOURCE_LOCK = replay_source_lock.DEFAULT_OUTPUT
DEFAULT_CERTIFICATION_SOURCE_LOCK = certification_source_lock.DEFAULT_OUTPUT
DEFAULT_PARENT_LOCK = parent_lock.DEFAULT_OUTPUT
DEFAULT_SUMMARY = DEFAULT_OUTPUT_ROOT / "seed42_replay_summary_v1.json"
DEFAULT_MANIFEST = (
    DEFAULT_OUTPUT_ROOT
    / "seed42_replay_checkpoint_local_pd_fa_manifest_v1.json"
)
DEFAULT_PAIRED = (
    REPO_ROOT
    / "analysis/results/final_model_seed42_replay_paired_screen_v1.json"
)
DEFAULT_GATE = DEFAULT_OUTPUT_ROOT / "seed42_replay_gate_v1.json"
DEFAULT_CLOSURE = DEFAULT_OUTPUT_ROOT / "seed42_replay_closure_v1.json"

FORBIDDEN_SEEDS = (3407, 426780603)
LEGACY_RUN_DIRECTORY_MARKERS = (
    "tpd_ner_v4_survival_exact_v1",
    "tpd_ner_v4_qfg_v2_croa_exact_v2_optimized",
    "final_model_engineering_replication_v1",
)


class Seed42ReplayPosttrainingError(ValueError):
    """The new seed-42 replay post-training evidence violates its contract."""


def _fail(message: str) -> None:
    raise Seed42ReplayPosttrainingError(message)


def _equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        _fail(
            f"{label} differs: observed={observed!r}, expected={expected!r}"
        )


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(f"{label} must be finite numeric")
    return float(value)


def _sha256_file(path: str | os.PathLike[str], label: str) -> str:
    source = Path(path)
    if source.is_symlink():
        _fail(f"{label} must not be a symlink: {source}")
    try:
        metadata = source.stat()
    except FileNotFoundError:
        _fail(f"{label} is missing: {source}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail(f"value is not finite canonical JSON: {exc}")


def _canonical_equal(label: str, observed: Any, expected: Any) -> None:
    if _canonical_json_bytes(observed) != _canonical_json_bytes(expected):
        _fail(f"{label} differs")


def _load_canonical_object(path: Path, label: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        _fail(f"{label} must be a regular non-symlink file: {source}")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot parse {label}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must contain one object")
    if raw != _canonical_json_bytes(value):
        _fail(f"{label} is not canonical JSON")
    return value


def _write_once(path: Path, payload: Mapping[str, Any], label: str) -> Path:
    destination = Path(path).expanduser()
    content = _canonical_json_bytes(copy.deepcopy(dict(payload)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        _fail(f"{label} must not be a symlink: {destination}")
    if destination.exists():
        if not destination.is_file():
            _fail(f"{label} path is not a regular file: {destination}")
        if destination.read_bytes() != content:
            raise FileExistsError(
                f"refusing to replace differing {label}: {destination}"
            )
        return destination.resolve()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    return destination.resolve()


def _assert_new_replay_path(path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    root = DEFAULT_OUTPUT_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"{label} lies outside the new replay output root: {resolved}")
    path_text = resolved.as_posix()
    for marker in LEGACY_RUN_DIRECTORY_MARKERS:
        if marker in path_text:
            _fail(f"{label} references a legacy result directory: {marker}")
    for forbidden_seed in FORBIDDEN_SEEDS:
        if (
            f"seed_{forbidden_seed}_" in path_text
            or f"seed-{forbidden_seed}:" in path_text
        ):
            _fail(f"{label} references forbidden seed {forbidden_seed}")
    return resolved


def _manifest_path(manifest_directory: Path, arm: str) -> Path:
    return replay_contract.manifest_path(Path(manifest_directory), arm)


def _inputs(
    arm: str,
    *,
    replay_contract_path: Path,
    manifest_directory: Path,
    replay_source_lock_path: Path,
    certification_source_lock_path: Path,
    parent_lock_path: Path,
) -> replay_core.ReplayInputs:
    inputs = replay_core.validate_inputs(
        arm=arm,
        contract_path=Path(replay_contract_path),
        initialization_manifest_path=_manifest_path(
            manifest_directory,
            arm,
        ),
        certification_source_lock_path=Path(
            certification_source_lock_path
        ),
        certification_parent_lock_path=Path(parent_lock_path),
        replay_source_lock_path=Path(replay_source_lock_path),
    )
    _equal("replay trajectory seed", inputs.trajectory_seed, TRAJECTORY_SEED)
    _equal(
        "replay output root",
        inputs.output_root.resolve(),
        DEFAULT_OUTPUT_ROOT.resolve(),
    )
    _assert_new_replay_path(
        replay_core.run_directory(inputs),
        f"arm {arm} run directory",
    )
    for name, observed in (
        (
            "legacy checkpoint imported",
            inputs.metadata()["legacy_checkpoint_imported"],
        ),
        (
            "legacy exact journal imported",
            inputs.metadata()["legacy_exact_journal_imported"],
        ),
    ):
        _equal(name, observed, False)
    return inputs


def _read_metrics_epochs(path: Path) -> tuple[int, ...]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        _fail(f"metrics.jsonl must be a regular non-symlink file: {source}")
    epochs: list[int] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"metrics line {line_number} is invalid JSON: {exc}")
        if not isinstance(record, Mapping):
            _fail(f"metrics line {line_number} is not an object")
        epoch = record.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            _fail(f"metrics line {line_number} epoch is not an integer")
        for name in (
            "pd",
            "fa",
            "miou",
            "tiny_pd",
            "false_objects_per_image",
        ):
            _finite(record.get(name), f"metrics line {line_number}.{name}")
        epochs.append(epoch)
    return tuple(epochs)


def _checkpoint_records(
    inputs: replay_core.ReplayInputs,
    run_directory: Path,
) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    with replay_core.replay_trainer_overlay(inputs):
        for filename, selection_role, checkpoint_role in CHECKPOINT_SPECS:
            record = summary_core._checkpoint_record(
                inputs,
                run_directory,
                filename,
                selection_role,
            )
            _equal(
                f"{filename} checkpoint role",
                record.get("checkpoint_role"),
                checkpoint_role,
            )
            records.append(record)
    return tuple(records)


def _validate_request_shape(
    request: evaluator.CheckpointEvaluationRequest,
) -> None:
    _equal("request trajectory seed", request.trajectory_seed, TRAJECTORY_SEED)
    if request.arm not in replay_core.SUPPORTED_ARMS:
        _fail(f"unsupported replay arm: {request.arm!r}")
    definition = frozen_replication_core.arm_definition(request.arm)
    _equal("request variant", request.variant, definition.variant)
    expected_specs = {
        filename: (selection_role, checkpoint_role)
        for filename, selection_role, checkpoint_role in CHECKPOINT_SPECS
    }
    if request.checkpoint_filename not in expected_specs:
        _fail("request checkpoint is not one of the two registered roles")
    selection_role, checkpoint_role = expected_specs[
        request.checkpoint_filename
    ]
    _equal(
        "request selection role",
        request.selection_role,
        selection_role,
    )
    _equal(
        "request checkpoint role",
        request.checkpoint_role,
        checkpoint_role,
    )
    expected_inputs = _load_replay_inputs_for_request(request)
    expected_directory = replay_core.run_directory(expected_inputs).resolve()
    _equal(
        "request run directory",
        request.run_directory.resolve(),
        expected_directory,
    )
    _assert_new_replay_path(request.run_directory, "request run directory")
    _equal(
        "request checkpoint path",
        request.checkpoint_path.resolve(),
        (request.run_directory / request.checkpoint_filename).resolve(),
    )
    if (
        isinstance(request.checkpoint_epoch, bool)
        or not isinstance(request.checkpoint_epoch, int)
        or not 1 <= request.checkpoint_epoch <= replay_contract.FORMAL_EPOCHS
    ):
        _fail("request checkpoint epoch lies outside 1..800")
    identity = request.run_identity
    if not isinstance(identity, Mapping):
        _fail("request run identity is missing")
    for name, expected in (
        ("schema", frozen_replication_core.exact_runner.RUN_IDENTITY_SCHEMA),
        ("run_id", replay_core.expected_run_id(expected_inputs)),
        ("variant", request.variant),
        ("dataset", replay_contract.DATASET),
        ("seed", TRAJECTORY_SEED),
        ("split_seed", replay_contract.SPLIT_SEED),
    ):
        _equal(f"request run identity {name}", identity.get(name), expected)
    source_locks = identity.get("source_locks")
    if not isinstance(source_locks, Mapping):
        _fail("request run identity source locks are missing")
    _equal(
        "request replay source lock",
        source_locks.get(replay_core.SOURCE_LOCK_KEY),
        request.source_lock_sha256,
    )
    _equal(
        "request validation count",
        len(request.validation_ids),
        evaluator.EXPECTED_VALIDATION_COUNT,
    )
    if len(set(request.validation_ids)) != len(request.validation_ids):
        _fail("request validation identifiers are not unique")
    _equal(
        "request validation identifier hash",
        statistics_cache.validation_identifier_sha256(
            request.validation_ids
        ),
        request.validation_split_sha256,
    )
    evaluator._validate_checkpoint_metrics(
        request.checkpoint_validation_metrics,
        label="request checkpoint validation metrics",
    )


def _load_replay_inputs_for_request(
    request: evaluator.CheckpointEvaluationRequest,
) -> replay_core.ReplayInputs:
    inputs = replay_core.validate_inputs(
        arm=request.arm,
        contract_path=request.seed_contract_path,
        initialization_manifest_path=request.child_manifest_path,
        certification_source_lock_path=DEFAULT_CERTIFICATION_SOURCE_LOCK,
        certification_parent_lock_path=DEFAULT_PARENT_LOCK,
        replay_source_lock_path=request.source_lock_path,
    )
    for label, observed, expected in (
        ("arm", inputs.definition.arm, request.arm),
        ("variant", inputs.definition.variant, request.variant),
        ("seed", inputs.trajectory_seed, request.trajectory_seed),
        (
            "replay contract path",
            inputs.contract_path.resolve(),
            request.seed_contract_path.resolve(),
        ),
        (
            "replay contract SHA-256",
            inputs.contract_sha256,
            request.seed_contract_sha256,
        ),
        (
            "child manifest path",
            inputs.initialization_path.resolve(),
            request.child_manifest_path.resolve(),
        ),
        (
            "child manifest SHA-256",
            inputs.initialization_sha256,
            request.child_manifest_sha256,
        ),
        (
            "replay source-lock path",
            inputs.source_lock_path.resolve(),
            request.source_lock_path.resolve(),
        ),
        (
            "replay source-lock SHA-256",
            inputs.source_lock_sha256,
            request.source_lock_sha256,
        ),
    ):
        _equal(f"request/input {label}", observed, expected)
    return inputs


@contextlib.contextmanager
def _trajectory_model_builder(
    request: evaluator.CheckpointEvaluationRequest,
    inputs: replay_core.ReplayInputs,
) -> Iterator[Callable[[str, int], Any]]:
    _validate_request_shape(request)
    with replay_core.replay_trainer_overlay(inputs) as trainer:

        def build_model(variant: str, seed: int) -> Any:
            _equal("shared evaluator variant", variant, request.variant)
            _equal("shared evaluator seed", seed, TRAJECTORY_SEED)
            return trainer.build_selected_model(
                variant,
                seed,
                eps=trainer.FORMAL_EPS,
            )

        yield build_model


def _evaluation_source_binding() -> dict[str, dict[str, str]]:
    paths = {
        name: Path(path).resolve()
        for name, path in evaluator.FROZEN_CORE_PATHS.items()
    }
    paths["checkpoint_local_adapter"] = Path(__file__).resolve()
    return {
        role: {
            "path": str(path),
            "sha256": _sha256_file(path, f"{role} source"),
        }
        for role, path in paths.items()
    }


def _evaluator_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "scope": SCOPE,
        "trajectory_seeds": [TRAJECTORY_SEED],
        "excluded_seeds": list(FORBIDDEN_SEEDS),
        "arms": list(replay_core.SUPPORTED_ARMS),
        "checkpoint_policy": [
            {
                "filename": filename,
                "selection_role": selection_role,
                "checkpoint_role": checkpoint_role,
            }
            for filename, selection_role, checkpoint_role in CHECKPOINT_SPECS
        ],
        "expected_run_count": EXPECTED_RUN_COUNT,
        "expected_sweep_count": EXPECTED_SWEEP_COUNT,
        "fixed_threshold": FIXED_THRESHOLD,
        "reported_metrics": list(evaluator.METRIC_OUTPUTS),
        "fa_budgets": list(evaluator.FA_BUDGETS),
        "each_arm_uses_own_registered_checkpoint": True,
        "cross_arm_shared_epoch_required": False,
        "cross_checkpoint_point_pooling": False,
        "old_seed42_results_accepted": False,
        "official_test_accessed": False,
        "paper_core_established": False,
        "stability_claim_supported": False,
        "evaluation_source_binding": _evaluation_source_binding(),
    }


def _canonical_request_keys() -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (TRAJECTORY_SEED, arm, filename)
        for arm in replay_core.SUPPORTED_ARMS
        for filename, _, _ in CHECKPOINT_SPECS
    )


@contextlib.contextmanager
def _temporary_attributes(
    module: ModuleType,
    values: Mapping[str, Any],
) -> Iterator[None]:
    previous: dict[str, Any] = {}
    for name, value in values.items():
        if not hasattr(module, name):
            _fail(f"module {module.__name__} lacks attribute {name!r}")
        previous[name] = getattr(module, name)
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(module, name, value)


@contextlib.contextmanager
def _evaluator_overlay() -> Iterator[None]:
    with _temporary_attributes(
        evaluator,
        {
            "EXPECTED_SWEEP_COUNT": EXPECTED_SWEEP_COUNT,
            "_validate_request_shape": _validate_request_shape,
            "_load_replication_inputs_for_request": (
                _load_replay_inputs_for_request
            ),
            "trajectory_model_builder": _trajectory_model_builder,
            "frozen_evaluation_core_binding": _evaluation_source_binding,
            "evaluator_contract": _evaluator_contract,
            "_canonical_request_keys": _canonical_request_keys,
        },
    ), _temporary_attributes(
        evaluator.seeds,
        {
            "ENGINEERING_TRAJECTORY_SEEDS": (TRAJECTORY_SEED,),
            "BUILDER_COMPATIBILITY_SEED": TRAJECTORY_SEED,
        },
    ):
        yield


def preflight_completed_run(
    arm: str,
    *,
    replay_contract_path: Path = DEFAULT_REPLAY_CONTRACT,
    manifest_directory: Path = DEFAULT_MANIFEST_DIRECTORY,
    replay_source_lock_path: Path = DEFAULT_REPLAY_SOURCE_LOCK,
    certification_source_lock_path: Path = DEFAULT_CERTIFICATION_SOURCE_LOCK,
    parent_lock_path: Path = DEFAULT_PARENT_LOCK,
) -> tuple[evaluator.CheckpointEvaluationRequest, ...]:
    """Validate one completed new replay run and return its two requests."""

    inputs = _inputs(
        arm,
        replay_contract_path=replay_contract_path,
        manifest_directory=manifest_directory,
        replay_source_lock_path=replay_source_lock_path,
        certification_source_lock_path=certification_source_lock_path,
        parent_lock_path=parent_lock_path,
    )
    _equal(
        f"arm {arm} completion mode",
        replay_core.resolve_initialization_mode(inputs),
        "--complete",
    )
    run_directory = replay_core.run_directory(inputs).resolve()
    _assert_new_replay_path(run_directory, f"arm {arm} run directory")
    protocol_path = run_directory / "protocol.json"
    split_path = run_directory / "split.json"
    summary_path = run_directory / "summary.json"
    metrics_path = run_directory / "metrics.jsonl"
    protocol = replay_core._load_pretty_object(
        protocol_path,
        "new replay protocol",
    )
    split = replay_core._load_pretty_object(
        split_path,
        "new replay split",
    )
    completion = replay_core._load_pretty_object(
        summary_path,
        "new replay completion summary",
    )
    run_identity = replay_core._validate_existing_protocol(
        inputs,
        run_directory,
    )
    _equal("completion status", completion.get("status"), "complete")
    _canonical_equal(
        "completion/protocol run identity",
        completion.get("run_identity"),
        run_identity,
    )
    _equal(
        "metrics epoch sequence",
        _read_metrics_epochs(metrics_path),
        tuple(range(1, replay_contract.FORMAL_EPOCHS + 1)),
    )
    split_hashes = split.get("hashes")
    if not isinstance(split_hashes, Mapping):
        _fail("replay split hashes are missing")
    validation_ids_raw = split.get("used_val_ids")
    if not isinstance(validation_ids_raw, list):
        _fail("replay validation identifiers are missing")
    validation_ids = tuple(validation_ids_raw)
    _equal(
        "replay validation count",
        len(validation_ids),
        evaluator.EXPECTED_VALIDATION_COUNT,
    )
    validation_split_sha256 = str(split_hashes.get("used_val_sha256"))
    _equal(
        "replay validation identifier SHA-256",
        statistics_cache.validation_identifier_sha256(validation_ids),
        validation_split_sha256,
    )
    source_locks = run_identity.get("source_locks")
    if not isinstance(source_locks, Mapping):
        _fail("replay source locks are missing")
    normalization = protocol.get("normalization")
    if not isinstance(normalization, Mapping):
        _fail("replay normalization contract is missing")
    checkpoint_records = _checkpoint_records(inputs, run_directory)
    requests: list[evaluator.CheckpointEvaluationRequest] = []
    for spec, checkpoint in zip(CHECKPOINT_SPECS, checkpoint_records):
        filename, selection_role, checkpoint_role = spec
        checkpoint_path = (run_directory / filename).resolve()
        request = evaluator.CheckpointEvaluationRequest(
            arm=arm,
            variant=inputs.definition.variant,
            trajectory_seed=TRAJECTORY_SEED,
            run_directory=run_directory,
            run_identity=copy.deepcopy(run_identity),
            seed_contract_path=inputs.contract_path,
            seed_contract_sha256=inputs.contract_sha256,
            child_manifest_path=inputs.initialization_path,
            child_manifest_sha256=inputs.initialization_sha256,
            source_lock_path=inputs.source_lock_path,
            source_lock_sha256=inputs.source_lock_sha256,
            protocol_sha256=_sha256_file(protocol_path, "replay protocol"),
            split_sha256=_sha256_file(split_path, "replay split"),
            summary_sha256=_sha256_file(summary_path, "replay summary"),
            metrics_sha256=_sha256_file(metrics_path, "replay metrics"),
            training_data_sha256=str(source_locks["training_data"]),
            normalization_sha256=evaluator._canonical_digest(normalization),
            validation_split_sha256=validation_split_sha256,
            validation_ids=validation_ids,
            checkpoint_filename=filename,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=str(checkpoint["sha256"]),
            checkpoint_epoch=int(checkpoint["epoch"]),
            checkpoint_role=checkpoint_role,
            selection_role=selection_role,
            checkpoint_validation_metrics=copy.deepcopy(
                checkpoint["metrics"]
            ),
        )
        _validate_request_shape(request)
        requests.append(request)
    return tuple(requests)


def collect_requests(
    *,
    replay_contract_path: Path = DEFAULT_REPLAY_CONTRACT,
    manifest_directory: Path = DEFAULT_MANIFEST_DIRECTORY,
    replay_source_lock_path: Path = DEFAULT_REPLAY_SOURCE_LOCK,
    certification_source_lock_path: Path = DEFAULT_CERTIFICATION_SOURCE_LOCK,
    parent_lock_path: Path = DEFAULT_PARENT_LOCK,
) -> tuple[evaluator.CheckpointEvaluationRequest, ...]:
    requests = tuple(
        request
        for arm in replay_core.SUPPORTED_ARMS
        for request in preflight_completed_run(
            arm,
            replay_contract_path=replay_contract_path,
            manifest_directory=manifest_directory,
            replay_source_lock_path=replay_source_lock_path,
            certification_source_lock_path=certification_source_lock_path,
            parent_lock_path=parent_lock_path,
        )
    )
    with _evaluator_overlay():
        evaluator.assemble_evaluation_plan(requests)
    return requests


def build_summary(
    requests: Sequence[evaluator.CheckpointEvaluationRequest],
) -> dict[str, Any]:
    ready = tuple(requests)
    with _evaluator_overlay():
        evaluator.assemble_evaluation_plan(ready)
    runs: list[dict[str, Any]] = []
    for arm in replay_core.SUPPORTED_ARMS:
        arm_requests = [
            request for request in ready if request.arm == arm
        ]
        runs.append(
            {
                "trajectory_seed": TRAJECTORY_SEED,
                "arm": arm,
                "variant": arm_requests[0].variant,
                "run_directory": str(arm_requests[0].run_directory),
                "run_id": arm_requests[0].run_identity["run_id"],
                "replay_contract_sha256": (
                    arm_requests[0].seed_contract_sha256
                ),
                "replay_source_lock_sha256": (
                    arm_requests[0].source_lock_sha256
                ),
                "child_initialization_manifest_sha256": (
                    arm_requests[0].child_manifest_sha256
                ),
                "checkpoints": [
                    {
                        "selection_role": request.selection_role,
                        "filename": request.checkpoint_filename,
                        "path": str(request.checkpoint_path),
                        "sha256": request.checkpoint_sha256,
                        "epoch": request.checkpoint_epoch,
                        "checkpoint_role": request.checkpoint_role,
                        "metrics": copy.deepcopy(
                            dict(request.checkpoint_validation_metrics)
                        ),
                    }
                    for request in arm_requests
                ],
            }
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "complete",
        "scope": SCOPE,
        "trajectory_seeds": [TRAJECTORY_SEED],
        "excluded_seeds": list(FORBIDDEN_SEEDS),
        "run_count": EXPECTED_RUN_COUNT,
        "checkpoint_count": EXPECTED_CHECKPOINT_COUNT,
        "checkpoint_selection": {
            "primary": "each_arm_own_best_miou",
            "secondary": "each_arm_own_best_pd",
            "cross_arm_shared_epoch_required": False,
        },
        "fixed_threshold": FIXED_THRESHOLD,
        "official_test_accessed": False,
        "runs": runs,
        "claim_boundary": {
            "single_seed_engineering_replay_only": True,
            "paper_core_established": False,
            "stability_claim_supported": False,
        },
    }


def build_plan(
    requests: Sequence[evaluator.CheckpointEvaluationRequest],
) -> dict[str, Any]:
    ready = tuple(requests)
    with _evaluator_overlay():
        reused_plan = evaluator.assemble_evaluation_plan(ready)
    return {
        "schema": SCHEMA,
        "status": "ready_for_four_checkpoint_local_sweeps",
        "scope": SCOPE,
        "request_count": len(ready),
        "expected_sweep_count": EXPECTED_SWEEP_COUNT,
        "fixed_threshold": FIXED_THRESHOLD,
        "reported_metrics": list(evaluator.METRIC_OUTPUTS),
        "checkpoint_selection": {
            "primary": "each_arm_own_best_miou",
            "secondary": "each_arm_own_best_pd",
            "shared_epoch_required": False,
        },
        "old_seed42_results_accepted": False,
        "excluded_seeds": list(FORBIDDEN_SEEDS),
        "gpu_work_started": False,
        "persistent_artifact_written": False,
        "reused_evaluator_plan_schema": reused_plan["schema"],
        "requests": [request.as_dict() for request in ready],
        "claim_boundary": {
            "paper_core_established": False,
            "stability_claim_supported": False,
        },
    }


def execute_arm(
    requests: Sequence[evaluator.CheckpointEvaluationRequest],
    *,
    arm: str,
    device: str,
    physical_gpu_index: int,
    physical_gpu_uuid: str,
) -> list[dict[str, Any]]:
    if arm not in replay_core.SUPPORTED_ARMS:
        _fail(f"unsupported replay arm: {arm!r}")
    selected = tuple(request for request in requests if request.arm == arm)
    _equal("selected arm sweep count", len(selected), 2)
    with _evaluator_overlay():
        assignment = evaluator.device_assignment(
            device,
            arm=arm,
            physical_gpu_index=physical_gpu_index,
            physical_gpu_uuid=physical_gpu_uuid,
        )
        return [
            evaluator.evaluate_or_skip_checkpoint(
                request,
                assignment=assignment,
            )
            for request in selected
        ]


def build_manifest(
    requests: Sequence[evaluator.CheckpointEvaluationRequest],
) -> dict[str, Any]:
    ready = tuple(requests)
    with _evaluator_overlay():
        payload = evaluator.build_results_manifest(ready)
    payload = copy.deepcopy(payload)
    payload["schema"] = MANIFEST_SCHEMA
    payload["scope"] = SCOPE
    _equal("manifest result count", payload["result_count"], 4)
    _equal("manifest expected result count", payload["expected_result_count"], 4)
    _equal(
        "manifest paired group count",
        payload["paired_checkpoint_group_count"],
        2,
    )
    for result in payload["results"]:
        _equal(
            "manifest result trajectory seed",
            result["trajectory_seed"],
            TRAJECTORY_SEED,
        )
        _assert_new_replay_path(
            Path(result["result_path"]),
            "manifest checkpoint-local result",
        )
    # This inherited field means only that paired-image sufficient statistics
    # exist.  The paired, Gate, and closure artifacts explicitly keep
    # ``establishes_gate_m_train`` false for this single-seed replay.
    _equal(
        "manifest paired-image inputs ready",
        payload["gate_m_train_image_level_inputs_ready"],
        True,
    )
    return payload


def _paired_base_payload(
    *,
    status: str,
    decision: str,
    manifest_path: Path,
    missing: Sequence[Mapping[str, str]] = (),
    errors: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema": PAIRED_SCHEMA,
        "status": status,
        "decision": decision,
        "scope": SCOPE,
        "manifest_path": str(Path(manifest_path).expanduser().resolve()),
        "trajectory_seeds": [TRAJECTORY_SEED],
        "excluded_seeds": list(FORBIDDEN_SEEDS),
        "checkpoint_policy": {
            "primary": "each_arm_own_best_miou",
            "secondary": "each_arm_own_best",
            "top_level_route_uses": "primary_only",
            "cross_arm_shared_epoch_required": False,
        },
        "fixed_threshold": FIXED_THRESHOLD,
        "metric_family": list(paired_core.METRIC_KEYS),
        "fa_budgets": list(evaluator.FA_BUDGETS),
        "missing_artifacts": [dict(item) for item in missing],
        "errors": list(errors),
        "seed42_replay_paired_route_met": None,
        "establishes_gate_m_train": False,
        "gates": {
            "M-train": {
                "status": "single_seed_descriptive_only",
                "passed": None,
                "establishes_gate_m_train": False,
            }
        },
        "claim_boundary": {
            "single_seed_engineering_replay_only": True,
            "paper_core_established": False,
            "stability_claim_supported": False,
            "multiseed_replication_supported": False,
            "official_test_accessed": False,
        },
    }


_ORIGINAL_PAIRED_MANIFEST_VALIDATOR = paired_core._validate_manifest


def _validate_replay_manifest_for_paired(
    manifest: Mapping[str, Any],
) -> tuple[
    dict[tuple[int, str, str], dict[str, Any]],
    dict[tuple[int, str], dict[str, Any]],
]:
    for label, observed, expected in (
        ("manifest schema", manifest.get("schema"), MANIFEST_SCHEMA),
        ("manifest scope", manifest.get("scope"), SCOPE),
        ("manifest result count", manifest.get("result_count"), 4),
        ("manifest expected result count", manifest.get("expected_result_count"), 4),
        (
            "manifest paired group count",
            manifest.get("paired_checkpoint_group_count"),
            2,
        ),
        ("manifest fixed threshold", manifest.get("fixed_threshold"), 0.5),
    ):
        _equal(label, observed, expected)
    compatibility = copy.deepcopy(dict(manifest))
    compatibility["schema"] = evaluator.MANIFEST_SCHEMA
    compatibility["scope"] = "fixed_parent_engineering_b_d_only"
    # This is an in-memory shape adapter for one hard-coded historical count;
    # the actual group list remains exactly the two seed-42 checkpoint roles.
    compatibility["paired_checkpoint_group_count"] = 4
    return _ORIGINAL_PAIRED_MANIFEST_VALIDATOR(compatibility)


def _paired_source_binding() -> dict[str, dict[str, str]]:
    paths = {
        "seed42_replay_adapter": Path(__file__).resolve(),
        "reused_paired_screen": Path(paired_core.__file__).resolve(),
        "lossless_prediction_cache_core": Path(
            paired_core.cache_core.__file__
        ).resolve(),
        "bootstrap_contract": Path(
            paired_core.bootstrap_contract.__file__
        ).resolve(),
        "checkpoint_local_evaluator": Path(evaluator.__file__).resolve(),
    }
    return {
        role: {
            "path": str(path),
            "sha256": _sha256_file(path, f"{role} source"),
        }
        for role, path in paths.items()
    }


@contextlib.contextmanager
def _paired_overlay() -> Iterator[None]:
    with _evaluator_overlay(), _temporary_attributes(
        paired_core,
        {
            "_base_payload": _paired_base_payload,
            "_validate_manifest": _validate_replay_manifest_for_paired,
            "_source_binding": _paired_source_binding,
        },
    ):
        yield


def analyze_paired(manifest_path: Path) -> dict[str, Any]:
    with _paired_overlay():
        payload = paired_core.analyze(manifest_path=Path(manifest_path))
    ready = copy.deepcopy(payload)
    ready["schema"] = PAIRED_SCHEMA
    ready["scope"] = SCOPE
    status = ready.get("status")
    if status == "complete":
        route_met = bool(ready.pop("engineering_paired_route_met"))
        ready["seed42_replay_paired_route_met"] = route_met
        ready["decision"] = (
            "SEED42_REPLAY_PAIRED_MIOU_ROUTE_MET"
            if route_met
            else "SEED42_REPLAY_PAIRED_MIOU_ROUTE_NOT_MET"
        )
        for record in ready.get("per_seed_checkpoint_policy_results", []):
            record["seed_role"] = "fixed_seed42_certification_replay"
        compatibility = ready.get("cache_compatibility")
        if isinstance(compatibility, dict):
            for old, new in (
                ("all_eight_cache_targets_identical", "all_four_cache_targets_identical"),
                ("all_eight_cache_image_ids_identical", "all_four_cache_image_ids_identical"),
                ("all_eight_cache_shapes_identical", "all_four_cache_shapes_identical"),
            ):
                if old in compatibility:
                    compatibility[new] = compatibility.pop(old)
        ready["interpretation"] = {
            "primary_route_source": (
                "single-seed paired-image bootstrap over each arm's own "
                "best_miou checkpoint"
            ),
            "secondary_policy_is_sensitivity_only": True,
            "fa_budgets_are_reported_not_gate_inputs": True,
            "single_seed_does_not_establish_stability": True,
            "establishes_gate_m_train": False,
        }
    ready["establishes_gate_m_train"] = False
    ready["claim_boundary"] = {
        "single_seed_engineering_replay_only": True,
        "paper_core_established": False,
        "stability_claim_supported": False,
        "multiseed_replication_supported": False,
        "official_test_accessed": False,
    }
    return ready


def _load_and_validate_paired(path: Path, manifest_path: Path) -> dict[str, Any]:
    stored = _load_canonical_object(path, "seed42 paired result")
    _equal("paired schema", stored.get("schema"), PAIRED_SCHEMA)
    _equal("paired scope", stored.get("scope"), SCOPE)
    _equal(
        "paired manifest path",
        stored.get("manifest_path"),
        str(Path(manifest_path).resolve()),
    )
    boundary = stored.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        _fail("paired claim boundary is missing")
    _equal(
        "paired paper-core claim",
        boundary.get("paper_core_established"),
        False,
    )
    _equal(
        "paired stability claim",
        boundary.get("stability_claim_supported"),
        False,
    )
    return stored


def adjudicate_gate(
    requests: Sequence[evaluator.CheckpointEvaluationRequest],
    *,
    manifest_path: Path,
    paired_path: Path,
) -> dict[str, Any]:
    ready_requests = tuple(requests)
    with _evaluator_overlay():
        evaluator.assemble_evaluation_plan(ready_requests)
        results = {
            (
                request.arm,
                request.checkpoint_filename,
            ): evaluator.load_completed_result(request)[0]
            for request in ready_requests
        }
    manifest = _load_canonical_object(
        manifest_path,
        "seed42 four-result manifest",
    )
    _validate_replay_manifest_for_paired(manifest)
    paired = _load_and_validate_paired(paired_path, manifest_path)
    if paired.get("status") != "complete":
        _fail("paired screen is not complete")
    comparisons = [
        gate_core._comparison_record(
            trajectory_seed=TRAJECTORY_SEED,
            checkpoint_filename=filename,
            selection_role=selection_role,
            b_result=results[(replay_core.ARM_B, filename)],
            d_result=results[(replay_core.ARM_D, filename)],
        )
        for filename, selection_role, _ in CHECKPOINT_SPECS
    ]
    primary = next(
        record
        for record in comparisons
        if record["selection_role"] == PRIMARY_SELECTION_ROLE
    )
    primary_paired = next(
        record
        for record in paired["per_seed_checkpoint_policy_results"]
        if record["selection_role"] == PRIMARY_SELECTION_ROLE
    )
    simultaneous_route = primary_paired["paired_image_bootstrap"][
        "miou_route_delta_0"
    ]
    route_met = bool(simultaneous_route["met"])
    return {
        "schema": GATE_SCHEMA,
        "status": "complete",
        "decision": (
            "SEED42_REPLAY_ENGINEERING_COMPLETE_MIOU_ROUTE_MET"
            if route_met
            else "SEED42_REPLAY_ENGINEERING_COMPLETE_MIOU_ROUTE_NOT_MET"
        ),
        "scope": SCOPE,
        "trajectory_seeds": [TRAJECTORY_SEED],
        "excluded_seeds": list(FORBIDDEN_SEEDS),
        "fixed_threshold": FIXED_THRESHOLD,
        "checkpoint_policy": {
            "primary": "each_arm_own_best_miou",
            "secondary": "each_arm_own_best_pd",
            "cross_arm_shared_epoch_required": False,
        },
        "gates": {
            "S-E": {
                "status": "complete",
                "passed": True,
                "engineering_replication_complete": True,
            },
            "seed42_paired_MIOU_ROUTE": {
                "status": "met" if route_met else "not_met",
                "passed": route_met,
                "point_estimate": primary[
                    "descriptive_miou_route_point_estimate"
                ],
                "paired_simultaneous_95_ci": simultaneous_route,
                "single_seed_only": True,
            },
            "M-train": {
                "status": "single_seed_descriptive_only",
                "passed": None,
                "establishes_gate_m_train": False,
            },
        },
        "fixed_threshold_and_budget_comparisons": comparisons,
        "evidence": {
            "run_count": EXPECTED_RUN_COUNT,
            "selected_checkpoint_count": EXPECTED_CHECKPOINT_COUNT,
            "validated_sweep_count": EXPECTED_SWEEP_COUNT,
            "manifest": {
                "path": str(Path(manifest_path).resolve()),
                "sha256": _sha256_file(manifest_path, "four-result manifest"),
            },
            "paired_screen": {
                "path": str(Path(paired_path).resolve()),
                "sha256": _sha256_file(paired_path, "paired screen"),
            },
            "each_sweep_checkpoint_local": True,
            "cross_checkpoint_point_pooling": False,
            "old_seed42_results_used": False,
        },
        "claim_boundary": {
            "single_seed_engineering_replay_only": True,
            "paper_core_established": False,
            "stability_claim_supported": False,
            "multiseed_replication_supported": False,
            "official_test_accessed": False,
        },
    }


def build_closure(
    *,
    summary_path: Path,
    manifest_path: Path,
    paired_path: Path,
    gate_path: Path,
) -> dict[str, Any]:
    bindings = {}
    for role, path in (
        ("summary", summary_path),
        ("four_sweep_manifest", manifest_path),
        ("paired_screen", paired_path),
        ("gate", gate_path),
    ):
        bindings[role] = {
            "path": str(Path(path).resolve()),
            "sha256": _sha256_file(path, role),
        }
    gate = _load_canonical_object(gate_path, "seed42 replay Gate")
    _equal("Gate schema", gate.get("schema"), GATE_SCHEMA)
    _equal("Gate status", gate.get("status"), "complete")
    return {
        "schema": CLOSURE_SCHEMA,
        "status": "complete",
        "decision": gate["decision"],
        "scope": SCOPE,
        "trajectory_seeds": [TRAJECTORY_SEED],
        "excluded_seeds": list(FORBIDDEN_SEEDS),
        "run_count": EXPECTED_RUN_COUNT,
        "sweep_count": EXPECTED_SWEEP_COUNT,
        "fixed_threshold": FIXED_THRESHOLD,
        "artifacts": bindings,
        "old_seed42_results_used": False,
        "paper_core_established": False,
        "stability_claim_supported": False,
        "official_test_accessed": False,
    }


def verify_complete_closure(
    *,
    replay_contract_path: Path = DEFAULT_REPLAY_CONTRACT,
    manifest_directory: Path = DEFAULT_MANIFEST_DIRECTORY,
    replay_source_lock_path: Path = DEFAULT_REPLAY_SOURCE_LOCK,
    certification_source_lock_path: Path = DEFAULT_CERTIFICATION_SOURCE_LOCK,
    parent_lock_path: Path = DEFAULT_PARENT_LOCK,
    summary_path: Path = DEFAULT_SUMMARY,
    manifest_path: Path = DEFAULT_MANIFEST,
    paired_path: Path = DEFAULT_PAIRED,
    gate_path: Path = DEFAULT_GATE,
    closure_path: Path = DEFAULT_CLOSURE,
) -> dict[str, Any]:
    """Rebuild and compare every layer of the completed closure.

    This is deliberately stronger than checking hashes recorded by the final
    attestation.  It revalidates both completed replay runs and checkpoints,
    all four checkpoint-local results/caches, the paired analysis, the Gate,
    and finally the closure bytes.
    """

    requests = collect_requests(
        replay_contract_path=replay_contract_path,
        manifest_directory=manifest_directory,
        replay_source_lock_path=replay_source_lock_path,
        certification_source_lock_path=certification_source_lock_path,
        parent_lock_path=parent_lock_path,
    )
    stored_summary = _load_canonical_object(
        summary_path,
        "seed42 replay summary",
    )
    _canonical_equal(
        "stored/live seed42 replay summary",
        stored_summary,
        build_summary(requests),
    )
    stored_manifest = _load_canonical_object(
        manifest_path,
        "seed42 four-result manifest",
    )
    _canonical_equal(
        "stored/live seed42 four-result manifest",
        stored_manifest,
        build_manifest(requests),
    )
    stored_paired = _load_canonical_object(
        paired_path,
        "seed42 paired screen",
    )
    _canonical_equal(
        "stored/live seed42 paired screen",
        stored_paired,
        analyze_paired(manifest_path),
    )
    stored_gate = _load_canonical_object(
        gate_path,
        "seed42 replay Gate",
    )
    _canonical_equal(
        "stored/live seed42 replay Gate",
        stored_gate,
        adjudicate_gate(
            requests,
            manifest_path=manifest_path,
            paired_path=paired_path,
        ),
    )
    stored_closure = _load_canonical_object(
        closure_path,
        "seed42 replay closure",
    )
    _canonical_equal(
        "stored/live seed42 replay closure",
        stored_closure,
        build_closure(
            summary_path=summary_path,
            manifest_path=manifest_path,
            paired_path=paired_path,
            gate_path=gate_path,
        ),
    )
    for label, payload in (
        ("paired", stored_paired),
        ("Gate", stored_gate),
        ("closure", stored_closure),
    ):
        boundary = payload.get("claim_boundary")
        if label == "closure":
            paper_core = payload.get("paper_core_established")
            stability = payload.get("stability_claim_supported")
        elif isinstance(boundary, Mapping):
            paper_core = boundary.get("paper_core_established")
            stability = boundary.get("stability_claim_supported")
        else:
            _fail(f"{label} claim boundary is missing")
        _equal(f"{label} paper-core claim", paper_core, False)
        _equal(f"{label} stability claim", stability, False)
    return {
        "schema": ACTION_SCHEMA,
        "status": "verified_complete",
        "scope": SCOPE,
        "run_count": EXPECTED_RUN_COUNT,
        "sweep_count": EXPECTED_SWEEP_COUNT,
        "trajectory_seeds": [TRAJECTORY_SEED],
        "excluded_seeds": list(FORBIDDEN_SEEDS),
        "fixed_threshold": FIXED_THRESHOLD,
        "closure": {
            "path": str(Path(closure_path).resolve()),
            "sha256": _sha256_file(closure_path, "closure"),
        },
        "paper_core_established": False,
        "stability_claim_supported": False,
    }


def dry_run_payload(
    *,
    replay_contract_path: Path = DEFAULT_REPLAY_CONTRACT,
    manifest_directory: Path = DEFAULT_MANIFEST_DIRECTORY,
    replay_source_lock_path: Path = DEFAULT_REPLAY_SOURCE_LOCK,
    certification_source_lock_path: Path = DEFAULT_CERTIFICATION_SOURCE_LOCK,
    parent_lock_path: Path = DEFAULT_PARENT_LOCK,
) -> dict[str, Any]:
    runs = []
    for arm in replay_core.SUPPORTED_ARMS:
        inputs = _inputs(
            arm,
            replay_contract_path=replay_contract_path,
            manifest_directory=manifest_directory,
            replay_source_lock_path=replay_source_lock_path,
            certification_source_lock_path=certification_source_lock_path,
            parent_lock_path=parent_lock_path,
        )
        mode = replay_core.resolve_initialization_mode(inputs)
        metrics_path = replay_core.run_directory(inputs) / "metrics.jsonl"
        completed_epochs = (
            len(_read_metrics_epochs(metrics_path))
            if metrics_path.is_file() and not metrics_path.is_symlink()
            else 0
        )
        runs.append(
            {
                "arm": arm,
                "variant": inputs.definition.variant,
                "trajectory_seed": inputs.trajectory_seed,
                "run_directory": str(replay_core.run_directory(inputs)),
                "resolved_mode": mode,
                "completed_epochs": completed_epochs,
                "registered_checkpoints": [
                    spec[0] for spec in CHECKPOINT_SPECS
                ],
            }
        )
    ready = all(record["resolved_mode"] == "--complete" for record in runs)
    return {
        "schema": ACTION_SCHEMA,
        "status": (
            "ready_for_posttraining_execution"
            if ready
            else "waiting_for_seed42_replay_training"
        ),
        "scope": SCOPE,
        "gpu_used": False,
        "gpu_command_launched": False,
        "persistent_artifact_written": False,
        "trajectory_seeds": [TRAJECTORY_SEED],
        "excluded_seeds": list(FORBIDDEN_SEEDS),
        "old_seed42_results_accepted": False,
        "run_count": EXPECTED_RUN_COUNT,
        "planned_sweep_count": EXPECTED_SWEEP_COUNT,
        "fixed_threshold": FIXED_THRESHOLD,
        "reported_metrics": list(evaluator.METRIC_OUTPUTS),
        "runs": runs,
        "claim_boundary": {
            "paper_core_established": False,
            "stability_claim_supported": False,
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contract-only", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--write-summary", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--finalize-manifest", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    mode.add_argument("--gate", action="store_true")
    mode.add_argument("--finalize-closure", action="store_true")
    mode.add_argument("--verify-closure", action="store_true")
    parser.add_argument("--arm", choices=replay_core.SUPPORTED_ARMS)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--physical-gpu-uuid")
    parser.add_argument(
        "--replay-contract",
        type=Path,
        default=DEFAULT_REPLAY_CONTRACT,
    )
    parser.add_argument(
        "--manifest-directory",
        type=Path,
        default=DEFAULT_MANIFEST_DIRECTORY,
    )
    parser.add_argument(
        "--replay-source-lock",
        type=Path,
        default=DEFAULT_REPLAY_SOURCE_LOCK,
    )
    parser.add_argument(
        "--certification-source-lock",
        type=Path,
        default=DEFAULT_CERTIFICATION_SOURCE_LOCK,
    )
    parser.add_argument(
        "--parent-lock",
        type=Path,
        default=DEFAULT_PARENT_LOCK,
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--paired", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--gate-output", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    return parser.parse_args(argv)


def _common_inputs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "replay_contract_path": args.replay_contract,
        "manifest_directory": args.manifest_directory,
        "replay_source_lock_path": args.replay_source_lock,
        "certification_source_lock_path": args.certification_source_lock,
        "parent_lock_path": args.parent_lock,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.contract_only:
        payload: Mapping[str, Any] = _evaluator_contract()
    elif args.dry_run:
        payload = dry_run_payload(**_common_inputs(args))
    elif args.analyze:
        analyzed = analyze_paired(args.manifest)
        if analyzed.get("status") != "complete":
            _fail(
                "paired analysis did not complete: "
                f"{analyzed.get('decision')}: {analyzed.get('errors')}"
            )
        output = _write_once(args.paired, analyzed, "paired screen")
        payload = {
            "schema": ACTION_SCHEMA,
            "status": "complete",
            "output": str(output),
            "sha256": _sha256_file(output, "paired screen"),
        }
    elif args.verify_closure:
        payload = verify_complete_closure(
            replay_contract_path=args.replay_contract,
            manifest_directory=args.manifest_directory,
            replay_source_lock_path=args.replay_source_lock,
            certification_source_lock_path=args.certification_source_lock,
            parent_lock_path=args.parent_lock,
            summary_path=args.summary,
            manifest_path=args.manifest,
            paired_path=args.paired,
            gate_path=args.gate_output,
            closure_path=args.closure,
        )
    else:
        requests = collect_requests(**_common_inputs(args))
        if args.plan:
            payload = build_plan(requests)
        elif args.write_summary:
            output = _write_once(
                args.summary,
                build_summary(requests),
                "seed42 replay summary",
            )
            payload = {
                "schema": ACTION_SCHEMA,
                "status": "complete",
                "output": str(output),
                "sha256": _sha256_file(output, "summary"),
            }
        elif args.execute:
            if (
                args.arm is None
                or args.physical_gpu_index is None
                or args.physical_gpu_uuid is None
            ):
                _fail(
                    "--execute requires --arm, --physical-gpu-index, "
                    "and --physical-gpu-uuid"
                )
            payload = {
                "schema": ACTION_SCHEMA,
                "status": "complete",
                "arm": args.arm,
                "results": execute_arm(
                    requests,
                    arm=args.arm,
                    device=args.device,
                    physical_gpu_index=args.physical_gpu_index,
                    physical_gpu_uuid=args.physical_gpu_uuid,
                ),
            }
        elif args.finalize_manifest:
            output = _write_once(
                args.manifest,
                build_manifest(requests),
                "four-result manifest",
            )
            payload = {
                "schema": ACTION_SCHEMA,
                "status": "complete",
                "output": str(output),
                "sha256": _sha256_file(output, "manifest"),
                "result_count": EXPECTED_SWEEP_COUNT,
            }
        elif args.gate:
            output = _write_once(
                args.gate_output,
                adjudicate_gate(
                    requests,
                    manifest_path=args.manifest,
                    paired_path=args.paired,
                ),
                "seed42 replay Gate",
            )
            payload = {
                "schema": ACTION_SCHEMA,
                "status": "complete",
                "output": str(output),
                "sha256": _sha256_file(output, "Gate"),
            }
        elif args.finalize_closure:
            output = _write_once(
                args.closure,
                build_closure(
                    summary_path=args.summary,
                    manifest_path=args.manifest,
                    paired_path=args.paired,
                    gate_path=args.gate_output,
                ),
                "seed42 replay closure",
            )
            payload = {
                "schema": ACTION_SCHEMA,
                "status": "complete",
                "output": str(output),
                "sha256": _sha256_file(output, "closure"),
            }
        else:  # pragma: no cover - argparse makes this unreachable.
            _fail("no execution mode selected")
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
