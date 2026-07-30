#!/usr/bin/env python3
"""Checkpoint-local Pd/Fa adapter for completed engineering B/D replications.

The plan/contract modes are read-only.  Execution reuses the frozen sweep,
closed-interval threshold, metric, model-overlay, and prediction-cache cores.
Each checkpoint owns a write-once result JSON plus a lossless 133-image
probability/target cache.  Existing outputs are skipped only after complete
live validation; the final manifest is produced only when all eight outputs
and their paired-image inputs validate.

The matrix is:

* two registered engineering trajectory seeds;
* arms B and D;
* each run's own ``best_miou.pth.tar`` primary checkpoint;
* each run's own ``best.pth.tar`` secondary checkpoint.

Threshold candidates and Fa-budget selections belong to exactly one
checkpoint.  The frozen result validators recompute every budget winner from
that checkpoint's local point collection, so cross-checkpoint point pooling is
rejected.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import ctypes
import errno
import gc
import hashlib
import importlib.util
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import collect_final_model_validation_statistics as statistics_cache  # noqa: E402
from experiments import evaluate_pd_fa_sweep as sweep_core  # noqa: E402
from experiments import (  # noqa: E402
    evaluate_tpd_clean_v6_pd_fa as closed_interval_core,
)
from experiments import (  # noqa: E402
    evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa as d_frozen_evaluator,
)
from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_pd_fa as point_validator,
)
from experiments import (  # noqa: E402
    evaluate_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_pd_fa
    as b_frozen_evaluator,
)
from experiments import final_model_replication_exact_core as core  # noqa: E402
from experiments import final_model_replication_seed_contract as seeds  # noqa: E402
from experiments import prepare_final_model_engineering_replication as prepare  # noqa: E402
from experiments import summarize_final_model_engineering_replication as summary_core  # noqa: E402
from experiments import watch_final_model_engineering_replication as watcher  # noqa: E402


SCHEMA = (
    "sctransnet_final_model_engineering_checkpoint_local_pd_fa_plan_v1"
)
RESULT_SCHEMA = (
    "sctransnet_final_model_engineering_checkpoint_local_pd_fa_result_v1"
)
MANIFEST_SCHEMA = (
    "sctransnet_final_model_engineering_checkpoint_local_pd_fa_manifest_v1"
)
DATASET = "NUDT-SIRST"
EXPECTED_EPOCHS = 800
EXPECTED_VALIDATION_COUNT = 133
EXPECTED_TARGET_COUNT = 189
EXPECTED_TINY_TARGET_COUNT = 39
FIXED_THRESHOLD = 0.5
FORMAL_MATCH_RADIUS = 3.0
FORMAL_TINY_AREA = 9
FA_BUDGETS = (1e-6, 5e-6, 1e-5, 5e-5, 1e-4)
BUDGET_KEYS = tuple(f"{budget:.10g}" for budget in FA_BUDGETS)
EXTRA_THRESHOLDS = (0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999)
EXPECTED_SWEEP_COUNT = 8
DEFAULT_MANIFEST_FILENAME = (
    "engineering_checkpoint_local_pd_fa_manifest_v1.json"
)
CACHE_REQUEST_IDENTITY_SCHEMA = (
    "sctransnet_final_model_engineering_prediction_cache_request_identity_v1"
)
CACHE_EVALUATOR_DERIVATION_SCHEMA = (
    "sctransnet_final_model_engineering_cache_evaluator_derivation_v1"
)
CACHE_STAGING_PREFIX = ".engineering-prediction-cache-staging."
CACHE_QUARANTINE_PREFIX = ".engineering-prediction-cache-incomplete."
EVALUATION_PHYSICAL_GPU_INDEX_ENV = (
    "FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_INDEX"
)
EVALUATION_PHYSICAL_GPU_UUID_ENV = (
    "FINAL_MODEL_ENGINEERING_EVAL_PHYSICAL_GPU_UUID"
)
CHECKPOINT_SPECS = (
    (
        "best_miou.pth.tar",
        "primary_best_miou",
        "best_validation_miou_secondary",
    ),
    (
        "best.pth.tar",
        "secondary_best_pd",
        "best_validation_pd_primary",
    ),
)
METRIC_OUTPUTS = (
    "pd",
    "fa",
    "miou",
    "tiny_pd",
    "false_objects_per_image",
    "unmatched_predicted_object_count",
)
ARM_PHYSICAL_GPU_INDICES = {
    core.ARM_B: 2,
    core.ARM_D: 3,
}
FROZEN_CORE_PATHS = {
    "shared_metric_core": Path(sweep_core.__file__),
    "closed_interval_core": Path(closed_interval_core.__file__),
    "arm_b_frozen_evaluator": Path(b_frozen_evaluator.__file__),
    "arm_d_frozen_evaluator": Path(d_frozen_evaluator.__file__),
    "replication_overlay_core": Path(core.__file__),
    "paired_statistics_cache_core": Path(statistics_cache.__file__),
}


class EngineeringEvaluationError(ValueError):
    """An engineering evaluation plan or result violates its identity."""


def _fail(message: str) -> None:
    raise EngineeringEvaluationError(message)


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        _fail(
            f"{label} differs: expected={expected!r}, "
            f"observed={observed!r}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _regular_file(path: str | os.PathLike[str], label: str) -> Path:
    value = Path(path)
    if value.is_symlink():
        _fail(f"{label} must not be a symlink: {value}")
    try:
        metadata = value.stat()
    except FileNotFoundError:
        _fail(f"{label} is missing: {value}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"{label} must be a regular file: {value}")
    return value.resolve()


def _sha256_file(path: str | os.PathLike[str], label: str) -> str:
    value = _regular_file(path, label)
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                sweep_core.json_ready(value),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        _fail(f"value is not canonical JSON: {exc}")


def _canonical_equal(label: str, observed: Any, expected: Any) -> None:
    if _canonical(observed) != _canonical(expected):
        _fail(f"{label} differs after canonical JSON normalization")


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_engineering_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("trajectory seed must be an integer")
    if value not in seeds.ENGINEERING_TRAJECTORY_SEEDS:
        _fail(
            "trajectory seed is not registered by the engineering seed "
            "contract"
        )
    if value == seeds.BUILDER_COMPATIBILITY_SEED:
        _fail("trajectory seed must not be the builder compatibility seed")
    return value


def _validate_checkpoint_metrics(
    metrics: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    try:
        ready = summary_core.metric_projection(metrics)
    except (TypeError, ValueError) as exc:
        _fail(f"{label} is incomplete: {exc}")
    for name, value in ready.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            _fail(f"{label}.{name} must be finite numeric")
    count_names = (
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    )
    for name in count_names:
        value = ready[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail(f"{label}.{name} must be a non-negative integer")
    for name, expected in (
        ("target_count", EXPECTED_TARGET_COUNT),
        ("tiny_target_count", EXPECTED_TINY_TARGET_COUNT),
    ):
        _require_equal(f"{label}.{name}", ready[name], expected)
    if ready["matched_target_count"] > ready["target_count"]:
        _fail(f"{label}.matched_target_count exceeds target_count")
    if ready["matched_tiny_target_count"] > ready["tiny_target_count"]:
        _fail(f"{label}.matched_tiny_target_count exceeds tiny_target_count")
    if ready["valid_pixel_count"] < 1:
        _fail(f"{label}.valid_pixel_count must be positive")
    _require_equal(
        f"{label}.pd/counts",
        float(ready["pd"]),
        ready["matched_target_count"] / EXPECTED_TARGET_COUNT,
    )
    _require_equal(
        f"{label}.tiny_pd/counts",
        float(ready["tiny_pd"]),
        ready["matched_tiny_target_count"] / EXPECTED_TINY_TARGET_COUNT,
    )
    _require_equal(
        f"{label}.false_objects_per_image/counts",
        float(ready["false_objects_per_image"]),
        ready["unmatched_predicted_object_count"]
        / EXPECTED_VALIDATION_COUNT,
    )
    for name in ("pd", "fa", "miou", "tiny_pd"):
        if not 0.0 <= float(ready[name]) <= 1.0:
            _fail(f"{label}.{name} lies outside [0, 1]")
    return copy.deepcopy(ready)


@dataclass(frozen=True)
class CheckpointEvaluationRequest:
    """One immutable threshold domain owned by one selected checkpoint."""

    arm: str
    variant: str
    trajectory_seed: int
    run_directory: Path
    run_identity: Mapping[str, Any]
    seed_contract_path: Path
    seed_contract_sha256: str
    child_manifest_path: Path
    child_manifest_sha256: str
    source_lock_path: Path
    source_lock_sha256: str
    protocol_sha256: str
    split_sha256: str
    summary_sha256: str
    metrics_sha256: str
    training_data_sha256: str
    normalization_sha256: str
    validation_split_sha256: str
    validation_ids: tuple[str, ...]
    checkpoint_filename: str
    checkpoint_path: Path
    checkpoint_sha256: str
    checkpoint_epoch: int
    checkpoint_role: str
    selection_role: str
    checkpoint_validation_metrics: Mapping[str, Any]

    @property
    def threshold_domain_id(self) -> str:
        return _canonical_digest(
            {
                "run_id": self.run_identity.get("run_id"),
                "arm": self.arm,
                "trajectory_seed": self.trajectory_seed,
                "checkpoint_filename": self.checkpoint_filename,
                "checkpoint_sha256": self.checkpoint_sha256,
            }
        )

    @property
    def planned_output_path(self) -> Path:
        return self.run_directory / (
            f"pd_fa_sweep_{Path(self.checkpoint_filename).stem}.json"
        )

    @property
    def prediction_cache_directory(self) -> Path:
        return self.run_directory / (
            "engineering_prediction_cache_"
            f"{Path(self.checkpoint_filename).stem}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "variant": self.variant,
            "trajectory_seed": self.trajectory_seed,
            "run_directory": str(self.run_directory),
            "run_identity": copy.deepcopy(dict(self.run_identity)),
            "seed_contract_path": str(self.seed_contract_path),
            "seed_contract_sha256": self.seed_contract_sha256,
            "child_manifest_path": str(self.child_manifest_path),
            "child_manifest_sha256": self.child_manifest_sha256,
            "source_lock_path": str(self.source_lock_path),
            "source_lock_sha256": self.source_lock_sha256,
            "protocol_sha256": self.protocol_sha256,
            "split_sha256": self.split_sha256,
            "summary_sha256": self.summary_sha256,
            "metrics_sha256": self.metrics_sha256,
            "training_data_sha256": self.training_data_sha256,
            "normalization_sha256": self.normalization_sha256,
            "validation_split_sha256": self.validation_split_sha256,
            "validation_count": len(self.validation_ids),
            "checkpoint_filename": self.checkpoint_filename,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_role": self.checkpoint_role,
            "selection_role": self.selection_role,
            "checkpoint_validation_metrics": copy.deepcopy(
                dict(self.checkpoint_validation_metrics)
            ),
            "threshold_domain_id": self.threshold_domain_id,
            "threshold_selection_scope": "single_checkpoint_only",
            "cross_checkpoint_point_pooling": False,
            "planned_output_path": str(self.planned_output_path),
            "prediction_cache_directory": str(
                self.prediction_cache_directory
            ),
        }


def _validate_request_shape(request: CheckpointEvaluationRequest) -> None:
    _require_engineering_seed(request.trajectory_seed)
    if request.arm not in core.SUPPORTED_ARMS:
        _fail(f"unsupported engineering arm: {request.arm!r}")
    definition = core.arm_definition(request.arm)
    _require_equal("request variant", request.variant, definition.variant)
    expected_spec = {
        name: (selection_role, checkpoint_role)
        for name, selection_role, checkpoint_role in CHECKPOINT_SPECS
    }
    if request.checkpoint_filename not in expected_spec:
        _fail("request checkpoint filename is not selected by the contract")
    expected_selection, expected_role = expected_spec[
        request.checkpoint_filename
    ]
    _require_equal(
        "request checkpoint selection role",
        request.selection_role,
        expected_selection,
    )
    _require_equal(
        "request checkpoint payload role",
        request.checkpoint_role,
        expected_role,
    )
    expected_directory = watcher.run_directory(
        request.run_directory.parents[2],
        request.trajectory_seed,
        request.arm,
    ).resolve()
    _require_equal(
        "request run directory",
        request.run_directory.resolve(),
        expected_directory,
    )
    expected_checkpoint = (
        request.run_directory / request.checkpoint_filename
    ).resolve()
    _require_equal(
        "request checkpoint path",
        request.checkpoint_path.resolve(),
        expected_checkpoint,
    )
    if (
        isinstance(request.checkpoint_epoch, bool)
        or not isinstance(request.checkpoint_epoch, int)
        or not 1 <= request.checkpoint_epoch <= EXPECTED_EPOCHS
    ):
        _fail("request checkpoint epoch lies outside 1..800")
    for name, value in (
        ("seed-contract", request.seed_contract_sha256),
        ("child-manifest", request.child_manifest_sha256),
        ("source-lock", request.source_lock_sha256),
        ("protocol", request.protocol_sha256),
        ("split", request.split_sha256),
        ("summary", request.summary_sha256),
        ("metrics", request.metrics_sha256),
        ("training data", request.training_data_sha256),
        ("normalization", request.normalization_sha256),
        ("validation split", request.validation_split_sha256),
        ("checkpoint", request.checkpoint_sha256),
    ):
        _require_sha256(value, f"request {name} SHA-256")
    identity = request.run_identity
    if not isinstance(identity, Mapping):
        _fail("request run identity is not an object")
    expected_run_id = (
        f"{definition.trainer.RUN_ID_PREFIX}NUDT-SIRST:"
        f"{request.variant}:seed-{request.trajectory_seed}:"
        f"split-{seeds.SPLIT_SEED}:"
        f"{core.ENGINEERING_RUN_TAGS[request.arm]}"
    )
    for name, expected in (
        ("schema", core.exact_runner.RUN_IDENTITY_SCHEMA),
        ("run_id", expected_run_id),
        ("variant", request.variant),
        ("dataset", DATASET),
        ("seed", request.trajectory_seed),
        ("split_seed", seeds.SPLIT_SEED),
    ):
        _require_equal(f"request run identity {name}", identity.get(name), expected)
    source_locks = identity.get("source_locks")
    if not isinstance(source_locks, Mapping):
        _fail("request run identity source locks are missing")
    _require_equal(
        "request run identity certification source lock",
        source_locks.get(core.SOURCE_LOCK_KEY),
        request.source_lock_sha256,
    )
    _require_equal(
        "request run identity training data",
        source_locks.get("training_data"),
        request.training_data_sha256,
    )
    if (
        not isinstance(request.validation_ids, tuple)
        or len(request.validation_ids) != EXPECTED_VALIDATION_COUNT
        or any(
            not isinstance(identifier, str) or not identifier
            for identifier in request.validation_ids
        )
        or len(set(request.validation_ids)) != EXPECTED_VALIDATION_COUNT
    ):
        _fail("request validation IDs must contain 133 unique strings")
    _require_equal(
        "request validation ID SHA-256",
        statistics_cache.validation_identifier_sha256(
            request.validation_ids
        ),
        request.validation_split_sha256,
    )
    _validate_checkpoint_metrics(
        request.checkpoint_validation_metrics,
        label="request checkpoint validation metrics",
    )


def _validate_protocol_replication_binding(
    protocol: Mapping[str, Any],
    *,
    inputs: core.ReplicationInputs,
    run_identity: Mapping[str, Any],
) -> None:
    model = protocol.get("model")
    if not isinstance(model, Mapping):
        _fail("replication protocol model metadata is missing")
    replication = model.get("replication_contract")
    if not isinstance(replication, Mapping):
        _fail("replication protocol child contract is missing")
    _canonical_equal(
        "protocol/validated replication input contract",
        replication,
        inputs.metadata(),
    )
    source_locks = run_identity.get("source_locks")
    if not isinstance(source_locks, Mapping):
        _fail("replication run identity source locks are missing")
    _require_equal(
        "run identity/validated certification source lock",
        source_locks.get(core.SOURCE_LOCK_KEY),
        inputs.source_lock_sha256,
    )
    _require_equal(
        "protocol child manifest SHA-256",
        replication.get("child_initialization_manifest_sha256"),
        inputs.initialization_sha256,
    )
    _require_equal(
        "protocol seed-contract SHA-256",
        replication.get("seed_contract_sha256"),
        inputs.schedule_sha256,
    )


def _cross_check_run_summary(
    record: Mapping[str, Any],
    *,
    inputs: core.ReplicationInputs,
    run_directory: Path,
) -> tuple[Mapping[str, Any], ...]:
    for name, expected in (
        ("arm", inputs.definition.arm),
        ("variant", inputs.definition.variant),
        ("trajectory_seed", inputs.trajectory_seed),
        ("run_directory", str(run_directory)),
        ("seed_contract_sha256", inputs.schedule_sha256),
        ("source_lock_sha256", inputs.source_lock_sha256),
        (
            "child_initialization_manifest_sha256",
            inputs.initialization_sha256,
        ),
    ):
        _require_equal(f"checkpoint summary {name}", record.get(name), expected)
    checkpoints = record.get("checkpoints")
    if not isinstance(checkpoints, list):
        _fail("checkpoint summary has no checkpoint list")
    by_name: dict[str, Mapping[str, Any]] = {}
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping):
            _fail(f"checkpoint summary item {index} is not an object")
        filename = checkpoint.get("filename")
        if not isinstance(filename, str) or filename in by_name:
            _fail("checkpoint summary contains an invalid/duplicate filename")
        by_name[filename] = checkpoint
    expected_names = tuple(spec[0] for spec in CHECKPOINT_SPECS)
    if set(by_name) != set(expected_names):
        _fail("checkpoint summary selection set differs")
    return tuple(by_name[name] for name in expected_names)


def preflight_completed_run(
    *,
    arm: str,
    trajectory_seed: int,
    output_root: Path,
    source_lock_path: Path,
    seed_contract_path: Path,
    manifest_directory: Path,
) -> tuple[CheckpointEvaluationRequest, ...]:
    """Strictly bind one completed run and return its two local requests."""

    _require_engineering_seed(trajectory_seed)
    if arm not in core.SUPPORTED_ARMS:
        _fail(f"unsupported engineering arm: {arm!r}")
    manifest_path = prepare.manifest_path(
        manifest_directory,
        seed=trajectory_seed,
        arm=arm,
    )
    inputs = core.validate_replication_inputs(
        arm=arm,
        trajectory_seed=trajectory_seed,
        schedule_path=seed_contract_path,
        initialization_manifest_path=manifest_path,
        certification_source_lock_path=source_lock_path,
    )
    mode = watcher.resolve_initialization_mode(
        output_root,
        trajectory_seed,
        arm,
    )
    _require_equal("engineering run completion mode", mode, "--complete")
    run_directory = watcher.run_directory(
        output_root,
        trajectory_seed,
        arm,
    ).resolve()
    protocol_path = run_directory / "protocol.json"
    split_path = run_directory / "split.json"
    summary_path = run_directory / "summary.json"
    metrics_path = run_directory / "metrics.jsonl"
    protocol = watcher._load_canonical_object(
        protocol_path,
        "replication protocol",
    )
    split = watcher._load_canonical_object(
        split_path,
        "replication split",
    )
    completion = watcher._load_canonical_object(
        summary_path,
        "replication completion summary",
    )
    run_identity = watcher._validate_protocol_identity(
        protocol,
        trajectory_seed=trajectory_seed,
        arm=arm,
        directory=run_directory,
    )
    watcher._validate_split(split)
    _canonical_equal(
        "completion/protocol run identity",
        completion.get("run_identity"),
        run_identity,
    )
    _validate_protocol_replication_binding(
        protocol,
        inputs=inputs,
        run_identity=run_identity,
    )
    checkpoint_summary = summary_core.summarize_run(
        arm=arm,
        trajectory_seed=trajectory_seed,
        output_root=output_root,
        source_lock_path=source_lock_path,
        seed_contract_path=seed_contract_path,
        manifest_directory=manifest_directory,
    )
    checkpoint_records = _cross_check_run_summary(
        checkpoint_summary,
        inputs=inputs,
        run_directory=run_directory,
    )
    protocol_sha256 = _sha256_file(protocol_path, "replication protocol")
    split_sha256 = _sha256_file(split_path, "replication split")
    summary_sha256 = _sha256_file(summary_path, "replication summary")
    metrics_sha256 = _sha256_file(metrics_path, "replication metrics")
    _require_equal(
        "summary helper/file SHA-256",
        checkpoint_summary.get("summary_sha256"),
        summary_sha256,
    )
    split_hashes = split.get("hashes")
    if not isinstance(split_hashes, Mapping):
        _fail("replication split hashes are missing")
    validation_split_sha256 = _require_sha256(
        split_hashes.get("used_val_sha256"),
        "validation split SHA-256",
    )
    validation_ids_raw = split.get("used_val_ids")
    if not isinstance(validation_ids_raw, list):
        _fail("replication validation IDs are missing")
    validation_ids = tuple(validation_ids_raw)
    source_locks = run_identity.get("source_locks")
    if not isinstance(source_locks, Mapping):
        _fail("replication run identity source locks are missing")
    training_data_sha256 = _require_sha256(
        source_locks.get("training_data"),
        "training-data SHA-256",
    )
    normalization = protocol.get("normalization")
    if not isinstance(normalization, Mapping):
        _fail("replication protocol normalization is missing")
    normalization_sha256 = _canonical_digest(normalization)
    requests: list[CheckpointEvaluationRequest] = []
    for spec, checkpoint in zip(CHECKPOINT_SPECS, checkpoint_records):
        filename, selection_role, checkpoint_role = spec
        checkpoint_path = (
            run_directory / filename
        ).resolve()
        _require_equal(
            f"{filename} summary path",
            checkpoint.get("path"),
            str(checkpoint_path),
        )
        _require_equal(
            f"{filename} selection role",
            checkpoint.get("selection_role"),
            selection_role,
        )
        _require_equal(
            f"{filename} checkpoint role",
            checkpoint.get("checkpoint_role"),
            checkpoint_role,
        )
        checkpoint_sha256 = _sha256_file(
            checkpoint_path,
            f"{filename} checkpoint",
        )
        _require_equal(
            f"{filename} summary SHA-256",
            checkpoint.get("sha256"),
            checkpoint_sha256,
        )
        metrics = checkpoint.get("metrics")
        if not isinstance(metrics, Mapping):
            _fail(f"{filename} checkpoint metrics are missing")
        request = CheckpointEvaluationRequest(
            arm=arm,
            variant=inputs.definition.variant,
            trajectory_seed=trajectory_seed,
            run_directory=run_directory,
            run_identity=copy.deepcopy(run_identity),
            seed_contract_path=inputs.schedule_path,
            seed_contract_sha256=inputs.schedule_sha256,
            child_manifest_path=inputs.initialization_path,
            child_manifest_sha256=inputs.initialization_sha256,
            source_lock_path=inputs.source_lock_path,
            source_lock_sha256=inputs.source_lock_sha256,
            protocol_sha256=protocol_sha256,
            split_sha256=split_sha256,
            summary_sha256=summary_sha256,
            metrics_sha256=metrics_sha256,
            training_data_sha256=training_data_sha256,
            normalization_sha256=normalization_sha256,
            validation_split_sha256=validation_split_sha256,
            validation_ids=validation_ids,
            checkpoint_filename=filename,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_epoch=checkpoint.get("epoch"),
            checkpoint_role=checkpoint_role,
            selection_role=selection_role,
            checkpoint_validation_metrics=_validate_checkpoint_metrics(
                metrics,
                label=f"{filename} checkpoint metrics",
            ),
        )
        _validate_request_shape(request)
        requests.append(request)
    return tuple(requests)


def frozen_evaluation_core_binding() -> dict[str, Any]:
    """Return live hashes for the already frozen metric/threshold adapters."""

    paths: dict[str, Path] = {
        name: _regular_file(path, name)
        for name, path in FROZEN_CORE_PATHS.items()
    }
    paths["checkpoint_local_adapter"] = _regular_file(
        Path(__file__),
        "checkpoint-local adapter",
    )
    return {
        name: {
            "path": str(path),
            "sha256": _sha256_file(path, name),
        }
        for name, path in paths.items()
    }


def evaluator_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "delivery_stage": "executable_checkpoint_local_gpu_runner",
        "scope": "fixed_parent_engineering_b_d_only",
        "dataset": DATASET,
        "trajectory_seeds": list(seeds.ENGINEERING_TRAJECTORY_SEEDS),
        "builder_compatibility_seed": seeds.BUILDER_COMPATIBILITY_SEED,
        "split_seed": seeds.SPLIT_SEED,
        "arms": list(core.SUPPORTED_ARMS),
        "checkpoint_policy": [
            {
                "filename": filename,
                "selection_role": selection_role,
                "checkpoint_role": checkpoint_role,
            }
            for filename, selection_role, checkpoint_role in CHECKPOINT_SPECS
        ],
        "expected_sweep_count": EXPECTED_SWEEP_COUNT,
        "fixed_threshold": FIXED_THRESHOLD,
        "reported_metrics": list(METRIC_OUTPUTS),
        "fa_budgets": list(FA_BUDGETS),
        "threshold_selection_scope": "single_checkpoint_only",
        "cross_checkpoint_point_pooling": False,
        "per_checkpoint_threshold_candidates": True,
        "per_checkpoint_budget_selection": True,
        "closed_probability_interval": True,
        "prediction_comparison": "prediction > threshold",
        "score_dtype": "float32",
        "official_test_accessed": False,
        "inference_launched_by_plan": False,
        "persistent_result_writes_enabled": True,
        "result_write_policy": "atomic_write_once_no_overwrite",
        "completed_result_policy": "validate_then_skip",
        "subset_execution_supported": True,
        "parallel_arm_execution_supported": True,
        "final_manifest_requires_all_eight_results": True,
        "formal_execution_device": "cuda:0",
        "formal_arm_physical_gpu_indices": copy.deepcopy(
            ARM_PHYSICAL_GPU_INDICES
        ),
        "formal_cpu_result_accepted": False,
        "canonical_matrix_order_required": True,
        "frozen_evaluation_cores": frozen_evaluation_core_binding(),
    }


def _canonical_request_keys() -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (trajectory_seed, arm, checkpoint_filename)
        for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS
        for arm in core.SUPPORTED_ARMS
        for checkpoint_filename, _, _ in CHECKPOINT_SPECS
    )


def assemble_evaluation_plan(
    requests: Sequence[CheckpointEvaluationRequest],
) -> dict[str, Any]:
    """Validate the exact 2-seed x 2-arm x 2-checkpoint matrix."""

    ready = tuple(requests)
    if len(ready) != EXPECTED_SWEEP_COUNT:
        _fail(
            f"evaluation matrix requires {EXPECTED_SWEEP_COUNT} requests, "
            f"found {len(ready)}"
        )
    for request in ready:
        _validate_request_shape(request)
    expected_keys = _canonical_request_keys()
    observed_keys = tuple(
        (
            request.trajectory_seed,
            request.arm,
            request.checkpoint_filename,
        )
        for request in ready
    )
    _require_equal("evaluation matrix identities", observed_keys, expected_keys)
    threshold_domains = [request.threshold_domain_id for request in ready]
    if len(set(threshold_domains)) != len(threshold_domains):
        _fail("evaluation matrix contains duplicate threshold domains")
    checkpoint_paths = [request.checkpoint_path for request in ready]
    if len(set(checkpoint_paths)) != len(checkpoint_paths):
        _fail("evaluation matrix contains duplicate checkpoint paths")
    output_paths = [request.planned_output_path for request in ready]
    if len(set(output_paths)) != len(output_paths):
        _fail("evaluation matrix contains duplicate planned outputs")
    contract = evaluator_contract()
    return {
        "schema": SCHEMA,
        "status": "ready_for_checkpoint_local_execution",
        "scope": contract["scope"],
        "request_count": len(ready),
        "expected_sweep_count": EXPECTED_SWEEP_COUNT,
        "fixed_threshold": FIXED_THRESHOLD,
        "reported_metrics": list(METRIC_OUTPUTS),
        "fa_budgets": list(FA_BUDGETS),
        "threshold_selection_scope": "single_checkpoint_only",
        "cross_checkpoint_point_pooling": False,
        "threshold_domain_count": len(set(threshold_domains)),
        "checkpoint_selection": {
            "primary": "each_run_own_best_miou",
            "secondary": "each_run_own_best_pd",
            "shared_epoch_required": False,
        },
        "official_test_accessed": False,
        "gpu_work_started": False,
        "persistent_artifact_written": False,
        "execution_available_in_this_stage": True,
        "evaluator_contract": contract,
        "requests": [request.as_dict() for request in ready],
    }


def collect_evaluation_requests(
    *,
    output_root: Path = watcher.DEFAULT_OUTPUT_ROOT,
    source_lock_path: Path,
    seed_contract_path: Path = prepare.DEFAULT_SEED_CONTRACT,
    manifest_directory: Path = prepare.DEFAULT_MANIFEST_DIRECTORY,
    run_preflight: Callable[..., tuple[CheckpointEvaluationRequest, ...]] = (
        preflight_completed_run
    ),
) -> tuple[CheckpointEvaluationRequest, ...]:
    """Read all four completed runs and return the exact request matrix."""

    requests: list[CheckpointEvaluationRequest] = []
    for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS:
        for arm in core.SUPPORTED_ARMS:
            requests.extend(
                run_preflight(
                    arm=arm,
                    trajectory_seed=trajectory_seed,
                    output_root=Path(output_root),
                    source_lock_path=Path(source_lock_path),
                    seed_contract_path=Path(seed_contract_path),
                    manifest_directory=Path(manifest_directory),
                )
            )
    # Validate the full matrix before returning execution-capable objects.
    assemble_evaluation_plan(requests)
    return tuple(requests)


def build_evaluation_plan(
    *,
    output_root: Path = watcher.DEFAULT_OUTPUT_ROOT,
    source_lock_path: Path,
    seed_contract_path: Path = prepare.DEFAULT_SEED_CONTRACT,
    manifest_directory: Path = prepare.DEFAULT_MANIFEST_DIRECTORY,
    run_preflight: Callable[..., tuple[CheckpointEvaluationRequest, ...]] = (
        preflight_completed_run
    ),
) -> dict[str, Any]:
    """Read all four completed runs and assemble the eight-request plan."""

    requests = collect_evaluation_requests(
        output_root=output_root,
        source_lock_path=source_lock_path,
        seed_contract_path=seed_contract_path,
        manifest_directory=manifest_directory,
        run_preflight=run_preflight,
    )
    return assemble_evaluation_plan(requests)


def _validate_request_against_inputs(
    request: CheckpointEvaluationRequest,
    inputs: core.ReplicationInputs,
) -> None:
    """Bind a future non-42 model build to the same validated input files."""

    _validate_request_shape(request)
    for name, observed, expected in (
        ("arm", request.arm, inputs.definition.arm),
        ("variant", request.variant, inputs.definition.variant),
        ("trajectory seed", request.trajectory_seed, inputs.trajectory_seed),
        (
            "seed-contract path",
            request.seed_contract_path.resolve(),
            inputs.schedule_path.resolve(),
        ),
        (
            "seed-contract SHA-256",
            request.seed_contract_sha256,
            inputs.schedule_sha256,
        ),
        (
            "child-manifest path",
            request.child_manifest_path.resolve(),
            inputs.initialization_path.resolve(),
        ),
        (
            "child-manifest SHA-256",
            request.child_manifest_sha256,
            inputs.initialization_sha256,
        ),
        (
            "source-lock path",
            request.source_lock_path.resolve(),
            inputs.source_lock_path.resolve(),
        ),
        (
            "source-lock SHA-256",
            request.source_lock_sha256,
            inputs.source_lock_sha256,
        ),
    ):
        _require_equal(f"request/input {name}", observed, expected)


@contextlib.contextmanager
def trajectory_model_builder(
    request: CheckpointEvaluationRequest,
    inputs: core.ReplicationInputs,
) -> Iterator[Callable[[str, int], Any]]:
    """Yield the future shared-core builder for a registered non-42 seed.

    No model is instantiated merely by entering this context.  A caller must
    explicitly invoke the yielded function during a later execution stage.
    """

    _validate_request_against_inputs(request, inputs)
    with core.replication_trainer_overlay(inputs) as trainer:

        def build_model(variant: str, seed: int) -> Any:
            _require_equal("shared evaluator variant", variant, request.variant)
            _require_equal(
                "shared evaluator trajectory seed",
                seed,
                request.trajectory_seed,
            )
            return trainer.build_selected_model(
                variant,
                seed,
                eps=trainer.FORMAL_EPS,
            )

        yield build_model


def _engineering_cache_request_identity(
    request: CheckpointEvaluationRequest,
    *,
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one cache to its full engineering request and live adapter."""

    adapter = source_binding.get("checkpoint_local_adapter")
    if not isinstance(adapter, Mapping):
        _fail("evaluation source binding omits the checkpoint-local adapter")
    adapter_sha256 = _require_sha256(
        adapter.get("sha256"),
        "checkpoint-local adapter SHA-256",
    )
    core_identity = {
        "schema": CACHE_REQUEST_IDENTITY_SCHEMA,
        "arm": request.arm,
        "variant": request.variant,
        "trajectory_seed": request.trajectory_seed,
        "run_id": request.run_identity.get("run_id"),
        "run_directory": str(request.run_directory),
        "seed_contract_sha256": request.seed_contract_sha256,
        "child_manifest_sha256": request.child_manifest_sha256,
        "source_lock_sha256": request.source_lock_sha256,
        "checkpoint_filename": request.checkpoint_filename,
        "checkpoint_epoch": request.checkpoint_epoch,
        "checkpoint_sha256": request.checkpoint_sha256,
        "threshold_domain_id": request.threshold_domain_id,
        "adapter_source_sha256": adapter_sha256,
    }
    request_identity_sha256 = _canonical_digest(core_identity)
    derivation = {
        "schema": CACHE_EVALUATOR_DERIVATION_SCHEMA,
        "algorithm": "sha256_of_canonical_json",
        "adapter_source_sha256": adapter_sha256,
        "engineering_request_identity_sha256": request_identity_sha256,
    }
    return {
        **core_identity,
        "engineering_request_identity_sha256": request_identity_sha256,
        "collector_evaluator_sha256": _canonical_digest(derivation),
        "collector_evaluator_sha256_derivation": derivation,
    }


def _prediction_cache_identity(
    request: CheckpointEvaluationRequest,
    *,
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    engineering_identity = _engineering_cache_request_identity(
        request,
        source_binding=source_binding,
    )
    return statistics_cache.build_cache_identity(
        checkpoint_sha256=request.checkpoint_sha256,
        dataset_sha256=request.training_data_sha256,
        evaluator_sha256=_require_sha256(
            engineering_identity["collector_evaluator_sha256"],
            "request-bound collector evaluator SHA-256",
        ),
        mode="full",
        normalization_sha256=request.normalization_sha256,
        source_lock_sha256=request.source_lock_sha256,
        validation_ids_sha256=request.validation_split_sha256,
        validation_count=EXPECTED_VALIDATION_COUNT,
        match_radius=FORMAL_MATCH_RADIUS,
        tiny_area=FORMAL_TINY_AREA,
    )


def _cache_fixed_metric_projection(
    cache: statistics_cache.PredictionCache,
) -> dict[str, Any]:
    recomputed = statistics_cache.recompute_metrics(
        cache,
        threshold=FIXED_THRESHOLD,
    )
    names = (
        "pd",
        "fa",
        "miou",
        "tiny_pd",
        "false_objects_per_image",
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    )
    return {name: copy.deepcopy(recomputed[name]) for name in names}


def _prediction_cache_binding(
    request: CheckpointEvaluationRequest,
    *,
    metadata_path: Path,
    expected_identity: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], statistics_cache.PredictionCache]:
    engineering_identity = _engineering_cache_request_identity(
        request,
        source_binding=source_binding,
    )
    metadata_file = _regular_file(
        metadata_path,
        "prediction cache metadata",
    )
    cache = statistics_cache.load_prediction_cache(
        metadata_file,
        expected_identity=expected_identity,
    )
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    arrays = metadata.get("arrays")
    if not isinstance(arrays, Mapping):
        _fail("prediction cache arrays binding is missing")
    arrays_path = _regular_file(
        metadata_file.parent / str(arrays.get("filename")),
        "prediction cache arrays",
    )
    _require_equal(
        "prediction cache image count",
        len(cache.records),
        EXPECTED_VALIDATION_COUNT,
    )
    _require_equal(
        "prediction cache image ID order",
        tuple(record.image_id for record in cache.records),
        request.validation_ids,
    )
    return (
        {
            "schema": statistics_cache.CACHE_SCHEMA,
            "status": "complete",
            "metadata_path": str(metadata_file),
            "metadata_sha256": _sha256_file(
                metadata_file,
                "prediction cache metadata",
            ),
            "arrays_path": str(arrays_path),
            "arrays_sha256": _sha256_file(
                arrays_path,
                "prediction cache arrays",
            ),
            "identity": copy.deepcopy(cache.identity),
            "engineering_request_identity": copy.deepcopy(
                engineering_identity
            ),
            "prediction_content_sha256": cache.content_sha256,
            "image_count": len(cache.records),
            "image_ids_sha256": request.validation_split_sha256,
            "fixed_threshold_0_5_recomputed": (
                _cache_fixed_metric_projection(cache)
            ),
            "paired_image_statistics_available": True,
            "official_test_accessed": False,
        },
        cache,
    )


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename one directory without replacing any destination."""

    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.parent.resolve() != destination_path.parent.resolve():
        _fail("cache directory rename must remain on one filesystem parent")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        _fail("atomic no-replace directory rename is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source_path),
        at_fdcwd,
        os.fsencode(destination_path),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(destination_path)
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination_path),
    )


def _cache_pair_paths(
    directory: Path,
    identity: Mapping[str, Any],
) -> tuple[Path, Path]:
    return statistics_cache.cache_paths(Path(directory), identity)


def _cache_directory_pair_state(
    directory: Path,
    identity: Mapping[str, Any],
) -> tuple[str, Path, Path]:
    root = Path(directory)
    metadata_path, arrays_path = _cache_pair_paths(root, identity)
    if root.is_symlink():
        _fail(f"prediction cache directory must not be a symlink: {root}")
    if not root.exists():
        return "missing", metadata_path, arrays_path
    if not root.is_dir():
        _fail(f"prediction cache path must be a directory: {root}")
    entries = tuple(root.iterdir())
    expected = {metadata_path.name, arrays_path.name}
    observed = {entry.name for entry in entries}
    if not observed <= expected:
        _fail("prediction cache directory contains unexpected entries")
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            _fail("prediction cache directory contains a non-regular entry")
    metadata_exists = metadata_path.is_file() and not metadata_path.is_symlink()
    arrays_exists = arrays_path.is_file() and not arrays_path.is_symlink()
    if metadata_exists and arrays_exists and observed == expected:
        return "complete", metadata_path, arrays_path
    if metadata_exists != arrays_exists and len(entries) == 1:
        return "one_file_partial", metadata_path, arrays_path
    _fail(
        "prediction cache directory is neither a complete pair nor an "
        "isolatable one-file partial"
    )


def _validate_one_file_partial_cache(
    directory: Path,
    identity: Mapping[str, Any],
) -> None:
    state, metadata_path, arrays_path = _cache_directory_pair_state(
        directory,
        identity,
    )
    _require_equal("partial cache state", state, "one_file_partial")
    if arrays_path.is_file() and not arrays_path.is_symlink():
        return
    metadata_file = _regular_file(
        metadata_path,
        "partial prediction cache metadata",
    )
    raw = metadata_file.read_bytes()
    try:
        metadata = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"partial prediction cache metadata is invalid: {exc}")
    if not isinstance(metadata, Mapping):
        _fail("partial prediction cache metadata must be an object")
    _require_equal(
        "partial cache canonical metadata",
        raw,
        statistics_cache.canonical_json_bytes(dict(metadata)),
    )
    _canonical_equal(
        "partial cache identity",
        metadata.get("identity"),
        statistics_cache.validate_cache_identity(identity),
    )
    arrays = metadata.get("arrays")
    if not isinstance(arrays, Mapping):
        _fail("partial prediction cache arrays binding is missing")
    _require_equal(
        "partial prediction cache arrays filename",
        arrays.get("filename"),
        arrays_path.name,
    )


def _unique_absent_directory_path(parent: Path, prefix: str) -> Path:
    reserved = Path(tempfile.mkdtemp(dir=parent, prefix=prefix))
    reserved.rmdir()
    return reserved


def _quarantine_one_file_partial_cache(
    directory: Path,
    identity: Mapping[str, Any],
) -> Path:
    """Preserve a recognized one-file partial at a unique sibling path."""

    root = Path(directory)
    _validate_one_file_partial_cache(root, identity)
    quarantine = _unique_absent_directory_path(
        root.parent,
        f"{CACHE_QUARANTINE_PREFIX}{root.name}.",
    )
    _rename_directory_no_replace(root, quarantine)
    return quarantine


def _stage_prediction_cache(
    request: CheckpointEvaluationRequest,
    candidate: statistics_cache.PredictionCache,
    identity: Mapping[str, Any],
) -> Path:
    parent = request.run_directory
    if parent.is_symlink() or not parent.is_dir():
        _fail("prediction cache parent run directory is invalid")
    staging = Path(
        tempfile.mkdtemp(
            dir=parent,
            prefix=CACHE_STAGING_PREFIX,
        )
    )
    try:
        metadata_path = statistics_cache.write_prediction_cache(
            candidate,
            staging,
        )
        loaded = statistics_cache.load_prediction_cache(
            metadata_path,
            expected_identity=identity,
        )
        _require_equal(
            "staged prediction content SHA-256",
            loaded.content_sha256,
            candidate.content_sha256,
        )
        state, expected_metadata, _ = _cache_directory_pair_state(
            staging,
            identity,
        )
        _require_equal("staged prediction cache state", state, "complete")
        _require_equal(
            "staged prediction cache metadata path",
            metadata_path.resolve(),
            expected_metadata.resolve(),
        )
        return staging
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def seal_or_validate_prediction_cache(
    request: CheckpointEvaluationRequest,
    *,
    probabilities: Sequence[Any],
    targets: Sequence[Any],
    losses: Sequence[float],
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Write once, or validate an identical cache left by an interrupted run."""

    counts = {
        len(probabilities),
        len(targets),
        len(losses),
        len(request.validation_ids),
    }
    if counts != {EXPECTED_VALIDATION_COUNT}:
        _fail(
            "prediction cache requires exactly 133 aligned "
            "probability/target/loss/ID records"
        )
    identity = _prediction_cache_identity(
        request,
        source_binding=source_binding,
    )
    collector = statistics_cache.PredictionCacheCollector(
        identity=identity,
        match_radius=FORMAL_MATCH_RADIUS,
        tiny_area=FORMAL_TINY_AREA,
    )
    for image_id, probability, target, loss in zip(
        request.validation_ids,
        probabilities,
        targets,
        losses,
    ):
        collector.append(
            image_id=image_id,
            probability=probability,
            target=target,
            loss=loss,
        )
    candidate = collector.seal()
    cache_directory = request.prediction_cache_directory
    state, metadata_path, _ = _cache_directory_pair_state(
        cache_directory,
        identity,
    )
    if state == "one_file_partial":
        _quarantine_one_file_partial_cache(cache_directory, identity)
        state = "missing"
    if state == "missing":
        staging = _stage_prediction_cache(request, candidate, identity)
        try:
            _rename_directory_no_replace(staging, cache_directory)
        except FileExistsError:
            winner_state, _, _ = _cache_directory_pair_state(
                cache_directory,
                identity,
            )
            if winner_state != "complete":
                raise
            shutil.rmtree(staging, ignore_errors=True)
        state, metadata_path, _ = _cache_directory_pair_state(
            cache_directory,
            identity,
        )
        _require_equal("published prediction cache state", state, "complete")
    binding, loaded = _prediction_cache_binding(
        request,
        metadata_path=metadata_path,
        expected_identity=identity,
        source_binding=source_binding,
    )
    _require_equal(
        "collected/persisted prediction content SHA-256",
        loaded.content_sha256,
        candidate.content_sha256,
    )
    return binding


def _validate_prediction_cache_binding(
    binding: Mapping[str, Any],
    request: CheckpointEvaluationRequest,
    *,
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        _fail("completed result prediction cache binding is missing")
    expected_identity = _prediction_cache_identity(
        request,
        source_binding=source_binding,
    )
    metadata_path = Path(str(binding.get("metadata_path")))
    expected_metadata_path, expected_arrays_path = statistics_cache.cache_paths(
        request.prediction_cache_directory,
        expected_identity,
    )
    _require_equal(
        "prediction cache metadata path",
        metadata_path.resolve(),
        expected_metadata_path.resolve(),
    )
    observed, cache = _prediction_cache_binding(
        request,
        metadata_path=metadata_path,
        expected_identity=expected_identity,
        source_binding=source_binding,
    )
    _require_equal(
        "prediction cache arrays path",
        Path(str(binding.get("arrays_path"))).resolve(),
        expected_arrays_path.resolve(),
    )
    _canonical_equal(
        "recorded/live prediction cache binding",
        binding,
        observed,
    )
    return {
        **observed,
        "_cache": cache,
    }


def _expected_arm_gpu_binding(arm: str) -> tuple[int, str]:
    if arm not in ARM_PHYSICAL_GPU_INDICES:
        _fail(f"unsupported engineering arm: {arm!r}")
    index = ARM_PHYSICAL_GPU_INDICES[arm]
    registered = {
        core.arm_definition(candidate).trainer.PHYSICAL_GPU_UUIDS[str(index)]
        for candidate in core.SUPPORTED_ARMS
    }
    if len(registered) != 1:
        _fail(f"B/D physical GPU{index} UUID registries differ")
    return index, next(iter(registered))


def device_assignment(
    device: str,
    *,
    arm: str,
    physical_gpu_index: int | None = None,
    physical_gpu_uuid: str | None = None,
) -> dict[str, Any]:
    """Validate the one formal physical GPU assigned to an engineering arm."""

    if device == "cpu":
        _fail("formal engineering checkpoint results cannot use CPU")
    if device != "cuda:0":
        _fail("formal engineering evaluation device must be cuda:0")
    expected_index, expected_uuid = _expected_arm_gpu_binding(arm)
    index_value: Any = (
        physical_gpu_index
        if physical_gpu_index is not None
        else os.environ.get(EVALUATION_PHYSICAL_GPU_INDEX_ENV)
    )
    if isinstance(index_value, str) and index_value.isdecimal():
        index_value = int(index_value)
    _require_equal(
        f"formal arm {arm} physical GPU index",
        index_value,
        expected_index,
    )
    uuid_value = (
        physical_gpu_uuid
        if physical_gpu_uuid is not None
        else os.environ.get(EVALUATION_PHYSICAL_GPU_UUID_ENV)
    )
    _require_equal("declared physical GPU UUID", uuid_value, expected_uuid)
    _require_equal(
        "CUDA_VISIBLE_DEVICES",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        expected_uuid,
    )
    torch = sweep_core.torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        _fail("GPU evaluation requires exactly one visible CUDA device")
    return {
        "device": "cuda:0",
        "physical_gpu_index": index_value,
        "physical_gpu_uuid": expected_uuid,
        "cuda_visible_devices": expected_uuid,
        "visible_cuda_device_count": 1,
        "device_name": torch.cuda.get_device_name(0),
    }


def _validate_recorded_device_assignment(
    value: Mapping[str, Any],
    *,
    request: CheckpointEvaluationRequest,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("execution device assignment is missing")
    ready = copy.deepcopy(dict(value))
    expected_fields = {
        "device",
        "physical_gpu_index",
        "physical_gpu_uuid",
        "cuda_visible_devices",
        "visible_cuda_device_count",
        "device_name",
    }
    _require_equal(
        "recorded execution device fields",
        set(ready),
        expected_fields,
    )
    _require_equal("recorded execution device", ready.get("device"), "cuda:0")
    expected_index, expected_uuid = _expected_arm_gpu_binding(request.arm)
    for name, expected in (
        ("physical_gpu_index", expected_index),
        ("physical_gpu_uuid", expected_uuid),
        ("cuda_visible_devices", expected_uuid),
        ("visible_cuda_device_count", 1),
    ):
        _require_equal(f"recorded device {name}", ready.get(name), expected)
    if (
        isinstance(ready.get("visible_cuda_device_count"), bool)
        or not isinstance(ready.get("visible_cuda_device_count"), int)
    ):
        _fail("recorded visible CUDA device count must be an integer")
    if not isinstance(ready.get("device_name"), str) or not ready["device_name"]:
        _fail("recorded CUDA device name is missing")
    return ready


def validate_checkpoint_local_result(
    payload: Mapping[str, Any],
    request: CheckpointEvaluationRequest,
    *,
    execution_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and identity-bind one in-memory frozen-core sweep result."""

    _validate_request_shape(request)
    if not isinstance(payload, Mapping):
        _fail("shared evaluator result must be an object")
    # Preserve the frozen evaluator's registered Fa-budget insertion order.
    # ``_canonical`` sorts object keys for identity comparisons and therefore
    # must not be used as the working representation passed to the frozen
    # budget validator.
    try:
        ready = json.loads(
            json.dumps(
                sweep_core.json_ready(copy.deepcopy(dict(payload))),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        _fail(f"shared evaluator result is not finite JSON: {exc}")
    expected_identity = {
        "run_directory": str(request.run_directory),
        "checkpoint": str(request.checkpoint_path),
        "checkpoint_sha256": request.checkpoint_sha256,
        "checkpoint_epoch": request.checkpoint_epoch,
        "checkpoint_role": request.checkpoint_role,
        "variant": request.variant,
        "dataset": DATASET,
        "seed": request.trajectory_seed,
        "split_seed": seeds.SPLIT_SEED,
        "validation_count": EXPECTED_VALIDATION_COUNT,
        "validation_split_sha256": request.validation_split_sha256,
        "match_radius": FORMAL_MATCH_RADIUS,
        "tiny_area": FORMAL_TINY_AREA,
        "official_test_accessed": False,
    }
    for name, expected in expected_identity.items():
        _require_equal(f"checkpoint-local result {name}", ready.get(name), expected)
    _canonical_equal(
        "result/request checkpoint validation metrics",
        ready.get("checkpoint_validation_metrics"),
        request.checkpoint_validation_metrics,
    )
    configuration = ready.get("threshold_configuration")
    if not isinstance(configuration, Mapping):
        _fail("checkpoint-local threshold configuration is missing")
    for name, expected in (
        ("threshold_min", 0.01),
        ("threshold_max", 0.99),
        ("threshold_step", 0.01),
        ("extra_thresholds", list(EXTRA_THRESHOLDS)),
        ("tail_logit_step", 0.1),
        ("fa_budgets", list(FA_BUDGETS)),
    ):
        _require_equal(
            f"checkpoint-local threshold configuration {name}",
            configuration.get(name),
            expected,
        )
    try:
        fixed = point_validator._validate_point_collection(
            ready,
            request.checkpoint_validation_metrics,
        )
        budgets = point_validator._normalize_budgets(ready)
        point_validator._validate_closed_interval(ready)
    except (TypeError, ValueError) as exc:
        _fail(f"frozen checkpoint-local result validation failed: {exc}")
    _require_equal(
        "fixed threshold",
        float(fixed["threshold"]),
        FIXED_THRESHOLD,
    )
    _require_equal("Fa budget keys", tuple(budgets), BUDGET_KEYS)
    audit = ready.get("audit")
    if not isinstance(audit, Mapping):
        _fail("checkpoint-local result audit is missing")
    try:
        point_validator._validate_standard_audit(audit)
    except (TypeError, ValueError) as exc:
        _fail(f"shared evaluator audit validation failed: {exc}")
    artifact_sha256 = audit.get("artifact_sha256")
    if not isinstance(artifact_sha256, Mapping):
        _fail("checkpoint-local result artifact hashes are missing")
    source_binding = frozen_evaluation_core_binding()
    expected_artifact_hashes = {
        "protocol.json": request.protocol_sha256,
        "split.json": request.split_sha256,
        "summary.json": request.summary_sha256,
        "metrics.jsonl": request.metrics_sha256,
        "checkpoint": request.checkpoint_sha256,
        "evaluator": source_binding["checkpoint_local_adapter"]["sha256"],
    }
    for name, expected in expected_artifact_hashes.items():
        _require_equal(
            f"result audit {name} SHA-256",
            artifact_sha256.get(name),
            expected,
        )
    execution_complete = execution_context is not None
    prediction_binding: dict[str, Any] | None = None
    recorded_device: dict[str, Any] | None = None
    if execution_context is not None:
        if not isinstance(execution_context, Mapping):
            _fail("execution context must be an object")
        _require_equal(
            "shared evaluator completion",
            execution_context.get("shared_evaluator_completed"),
            True,
        )
        _require_equal(
            "legacy six-tensor output",
            execution_context.get("legacy_six_tensor_eval_output"),
            True,
        )
        _canonical_equal(
            "execution/current evaluation source binding",
            execution_context.get("evaluation_source_binding"),
            source_binding,
        )
        recorded_device = _validate_recorded_device_assignment(
            execution_context.get("device_assignment"),
            request=request,
        )
        cache_validation = _validate_prediction_cache_binding(
            execution_context.get("prediction_cache"),
            request,
            source_binding=source_binding,
        )
        cache = cache_validation.pop("_cache")
        prediction_binding = cache_validation
        _canonical_equal(
            "fixed point/prediction cache metrics",
            {
                name: fixed[name]
                for name in (
                    "pd",
                    "fa",
                    "miou",
                    "tiny_pd",
                    "false_objects_per_image",
                    "target_count",
                    "matched_target_count",
                    "tiny_target_count",
                    "matched_tiny_target_count",
                    "predicted_object_count",
                    "unmatched_predicted_object_count",
                    "valid_pixel_count",
                )
            },
            _cache_fixed_metric_projection(cache),
        )
    ready.update(
        {
            "schema": RESULT_SCHEMA,
            "source_run_identity": copy.deepcopy(dict(request.run_identity)),
            "source_checkpoint_identity": {
                "arm": request.arm,
                "variant": request.variant,
                "trajectory_seed": request.trajectory_seed,
                "selection_role": request.selection_role,
                "checkpoint_filename": request.checkpoint_filename,
                "checkpoint_role": request.checkpoint_role,
                "checkpoint_epoch": request.checkpoint_epoch,
                "checkpoint_sha256": request.checkpoint_sha256,
                "threshold_domain_id": request.threshold_domain_id,
            },
            "replication_input_binding": {
                "seed_contract_path": str(request.seed_contract_path),
                "seed_contract_sha256": request.seed_contract_sha256,
                "child_manifest_path": str(request.child_manifest_path),
                "child_manifest_sha256": request.child_manifest_sha256,
                "source_lock_path": str(request.source_lock_path),
                "source_lock_sha256": request.source_lock_sha256,
                "protocol_sha256": request.protocol_sha256,
                "split_sha256": request.split_sha256,
                "summary_sha256": request.summary_sha256,
                "metrics_sha256": request.metrics_sha256,
                "training_data_sha256": request.training_data_sha256,
                "normalization_sha256": request.normalization_sha256,
            },
            "evaluation_source_binding": copy.deepcopy(source_binding),
            "execution_complete": execution_complete,
            "execution_device_assignment": copy.deepcopy(recorded_device),
            "prediction_cache": copy.deepcopy(prediction_binding),
            "paired_image_statistics_available": execution_complete,
            "threshold_selection_scope": "single_checkpoint_only",
            "cross_checkpoint_point_pooling": False,
            "evaluated_checkpoint_count": 1,
            "reported_metrics": list(METRIC_OUTPUTS),
            "final_metric_coverage": point_validator._final_metric_coverage(
                fixed,
                budgets,
            ),
            "official_test_accessed": False,
        }
    )
    ready_audit = copy.deepcopy(dict(ready["audit"]))
    ready_audit.update(
        {
            "trajectory_seed_model_overlay": execution_complete,
            "legacy_six_tensor_eval_output": execution_complete,
            "prediction_cache_complete": execution_complete,
            "checkpoint_local_atomic_write": execution_complete,
        }
    )
    ready["audit"] = ready_audit
    validate_finalized_result(ready, request)
    return ready


def validate_finalized_result(
    payload: Mapping[str, Any],
    request: CheckpointEvaluationRequest,
    *,
    require_execution_complete: bool = False,
) -> None:
    """Recheck the identity fields added by this adapter."""

    for name, expected in (
        ("schema", RESULT_SCHEMA),
        ("threshold_selection_scope", "single_checkpoint_only"),
        ("cross_checkpoint_point_pooling", False),
        ("evaluated_checkpoint_count", 1),
        ("reported_metrics", list(METRIC_OUTPUTS)),
        ("official_test_accessed", False),
    ):
        _require_equal(f"finalized result {name}", payload.get(name), expected)
    if require_execution_complete:
        _require_equal(
            "finalized result execution_complete",
            payload.get("execution_complete"),
            True,
        )
        _require_equal(
            "finalized result paired image statistics",
            payload.get("paired_image_statistics_available"),
            True,
        )
    _canonical_equal(
        "finalized result run identity",
        payload.get("source_run_identity"),
        request.run_identity,
    )
    checkpoint_identity = payload.get("source_checkpoint_identity")
    if not isinstance(checkpoint_identity, Mapping):
        _fail("finalized source checkpoint identity is missing")
    for name, expected in (
        ("arm", request.arm),
        ("variant", request.variant),
        ("trajectory_seed", request.trajectory_seed),
        ("selection_role", request.selection_role),
        ("checkpoint_filename", request.checkpoint_filename),
        ("checkpoint_role", request.checkpoint_role),
        ("checkpoint_epoch", request.checkpoint_epoch),
        ("checkpoint_sha256", request.checkpoint_sha256),
        ("threshold_domain_id", request.threshold_domain_id),
    ):
        _require_equal(
            f"finalized checkpoint identity {name}",
            checkpoint_identity.get(name),
            expected,
        )
    source_binding = frozen_evaluation_core_binding()
    _canonical_equal(
        "finalized evaluation source binding",
        payload.get("evaluation_source_binding"),
        source_binding,
    )
    replication_binding = payload.get("replication_input_binding")
    if not isinstance(replication_binding, Mapping):
        _fail("finalized replication input binding is missing")
    for name, expected in (
        ("seed_contract_path", str(request.seed_contract_path)),
        ("seed_contract_sha256", request.seed_contract_sha256),
        ("child_manifest_path", str(request.child_manifest_path)),
        ("child_manifest_sha256", request.child_manifest_sha256),
        ("source_lock_path", str(request.source_lock_path)),
        ("source_lock_sha256", request.source_lock_sha256),
        ("protocol_sha256", request.protocol_sha256),
        ("split_sha256", request.split_sha256),
        ("summary_sha256", request.summary_sha256),
        ("metrics_sha256", request.metrics_sha256),
        ("training_data_sha256", request.training_data_sha256),
        ("normalization_sha256", request.normalization_sha256),
    ):
        _require_equal(
            f"finalized replication binding {name}",
            replication_binding.get(name),
            expected,
        )
    execution_complete = payload.get("execution_complete")
    if execution_complete is True:
        _validate_recorded_device_assignment(
            payload.get("execution_device_assignment"),
            request=request,
        )
        cache_validation = _validate_prediction_cache_binding(
            payload.get("prediction_cache"),
            request,
            source_binding=source_binding,
        )
        cache = cache_validation.pop("_cache")
        _require_equal(
            "finalized paired image statistics",
            payload.get("paired_image_statistics_available"),
            True,
        )
    elif execution_complete is False and not require_execution_complete:
        _require_equal(
            "non-executed result device assignment",
            payload.get("execution_device_assignment"),
            None,
        )
        _require_equal(
            "non-executed result prediction cache",
            payload.get("prediction_cache"),
            None,
        )
        _require_equal(
            "non-executed result paired statistics",
            payload.get("paired_image_statistics_available"),
            False,
        )
        cache = None
    else:
        _fail("finalized result execution state is invalid")
    fixed = point_validator._validate_point_collection(
        payload,
        request.checkpoint_validation_metrics,
    )
    budgets = point_validator._normalize_budgets(payload)
    point_validator._validate_closed_interval(payload)
    _canonical_equal(
        "finalized metric coverage",
        payload.get("final_metric_coverage"),
        point_validator._final_metric_coverage(fixed, budgets),
    )
    if cache is not None:
        _canonical_equal(
            "finalized fixed/cache metrics",
            {
                name: fixed[name]
                for name in (
                    "pd",
                    "fa",
                    "miou",
                    "tiny_pd",
                    "false_objects_per_image",
                    "target_count",
                    "matched_target_count",
                    "tiny_target_count",
                    "matched_tiny_target_count",
                    "predicted_object_count",
                    "unmatched_predicted_object_count",
                    "valid_pixel_count",
                )
            },
            _cache_fixed_metric_projection(cache),
        )


def _verify_request_files_unchanged(
    request: CheckpointEvaluationRequest,
) -> None:
    """Re-hash every request-owned file before a run is built or skipped."""

    for label, path, expected in (
        (
            "seed contract",
            request.seed_contract_path,
            request.seed_contract_sha256,
        ),
        (
            "child initialization manifest",
            request.child_manifest_path,
            request.child_manifest_sha256,
        ),
        (
            "certification source lock",
            request.source_lock_path,
            request.source_lock_sha256,
        ),
        (
            "replication protocol",
            request.run_directory / "protocol.json",
            request.protocol_sha256,
        ),
        (
            "replication split",
            request.run_directory / "split.json",
            request.split_sha256,
        ),
        (
            "replication summary",
            request.run_directory / "summary.json",
            request.summary_sha256,
        ),
        (
            "replication metrics",
            request.run_directory / "metrics.jsonl",
            request.metrics_sha256,
        ),
        (
            "selected checkpoint",
            request.checkpoint_path,
            request.checkpoint_sha256,
        ),
    ):
        _require_equal(
            f"{label} live SHA-256",
            _sha256_file(path, label),
            expected,
        )


def _load_replication_inputs_for_request(
    request: CheckpointEvaluationRequest,
) -> core.ReplicationInputs:
    inputs = core.validate_replication_inputs(
        arm=request.arm,
        trajectory_seed=request.trajectory_seed,
        schedule_path=request.seed_contract_path,
        initialization_manifest_path=request.child_manifest_path,
        certification_source_lock_path=request.source_lock_path,
    )
    _validate_request_against_inputs(request, inputs)
    return inputs


def _result_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                sweep_core.json_ready(copy.deepcopy(dict(payload))),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail(f"result is not finite JSON: {exc}")


def _load_result_object(path: Path) -> dict[str, Any]:
    source = _regular_file(path, "checkpoint-local result")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot parse checkpoint-local result: {exc}")
    if not isinstance(value, dict):
        _fail("checkpoint-local result must contain one JSON object")
    if raw != _result_json_bytes(value):
        _fail("checkpoint-local result is not canonical pretty JSON")
    return value


def _atomic_link_bytes(path: Path, content: bytes, *, label: str) -> None:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to replace existing {label}: {destination}")
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        _fail(f"{label} parent must be a regular directory: {parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to replace existing {label}: {destination}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def write_result_once(
    path: Path,
    payload: Mapping[str, Any],
    request: CheckpointEvaluationRequest,
) -> Path:
    destination = Path(path)
    _require_equal(
        "checkpoint-local result output path",
        destination.resolve(),
        request.planned_output_path.resolve(),
    )
    validate_finalized_result(
        payload,
        request,
        require_execution_complete=True,
    )
    _atomic_link_bytes(
        destination,
        _result_json_bytes(payload),
        label="checkpoint-local result",
    )
    loaded = _load_result_object(destination)
    validate_finalized_result(
        loaded,
        request,
        require_execution_complete=True,
    )
    return destination.resolve()


def load_completed_result(
    request: CheckpointEvaluationRequest,
) -> tuple[dict[str, Any], str]:
    _verify_request_files_unchanged(request)
    result_path = request.planned_output_path
    before = _sha256_file(result_path, "checkpoint-local result")
    payload = _load_result_object(result_path)
    validate_finalized_result(
        payload,
        request,
        require_execution_complete=True,
    )
    after = _sha256_file(result_path, "checkpoint-local result")
    _require_equal("checkpoint-local result stability", after, before)
    return payload, before


def _base_evaluator_namespace(
    request: CheckpointEvaluationRequest,
    *,
    device: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=request.run_directory,
        checkpoint=request.checkpoint_filename,
        device=device,
        expected_epochs=EXPECTED_EPOCHS,
        threshold_min=0.01,
        threshold_max=0.99,
        threshold_step=0.01,
        extra_thresholds=list(EXTRA_THRESHOLDS),
        tail_logit_step=0.1,
        fa_budgets=list(FA_BUDGETS),
        match_radius=FORMAL_MATCH_RADIUS,
        tiny_area=FORMAL_TINY_AREA,
        overwrite=False,
    )


def _legacy_output_validator(
    request: CheckpointEvaluationRequest,
) -> Callable[[Any], Any]:
    if request.arm == core.ARM_B:
        return b_frozen_evaluator._require_legacy_eval_output
    if request.arm == core.ARM_D:
        return d_frozen_evaluator._require_legacy_eval_output
    _fail(f"unsupported engineering arm: {request.arm!r}")


def _load_bound_shared_evaluator(
    request: CheckpointEvaluationRequest,
    *,
    build_model: Callable[[str, int], Any],
    assignment: Mapping[str, Any],
    source_binding: Mapping[str, Any],
) -> tuple[ModuleType, dict[str, Any]]:
    """Load one isolated shared evaluator with a checkpoint-local writer."""

    module_name = (
        "_sctransnet_engineering_pd_fa_"
        f"{request.arm}_{request.trajectory_seed}_"
        f"{request.threshold_domain_id}"
    )
    base_path = Path(sweep_core.__file__).resolve()
    spec = importlib.util.spec_from_file_location(module_name, base_path)
    if spec is None or spec.loader is None:
        _fail("cannot load the shared Pd/Fa evaluator")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    namespace = _base_evaluator_namespace(
        request,
        device=str(assignment["device"]),
    )
    state: dict[str, Any] = {
        "collect_called": False,
        "legacy_output_verified": False,
        "prediction_cache": None,
        "write_called": False,
        "output_path": None,
    }
    original_collect = evaluator.collect_predictions
    validate_legacy = _legacy_output_validator(request)

    def bound_parse_args() -> argparse.Namespace:
        return argparse.Namespace(**vars(namespace))

    def bound_build_model(variant: str, seed: int) -> Any:
        return build_model(variant, seed)

    def bound_collect_predictions(model, loader, device):
        dataset = getattr(loader, "dataset", None)
        _require_equal(
            "frozen validation loader identifiers",
            tuple(getattr(dataset, "identifiers", ())),
            request.validation_ids,
        )
        _require_equal(
            "frozen validation loader length",
            len(dataset) if dataset is not None else None,
            EXPECTED_VALIDATION_COUNT,
        )
        _require_equal(
            "frozen validation loader batch size",
            getattr(loader, "batch_size", None),
            1,
        )
        _require_equal(
            "frozen validation loader workers",
            getattr(loader, "num_workers", None),
            0,
        )
        _require_equal(
            "frozen validation loader sampler",
            type(getattr(loader, "sampler", None)).__name__,
            "SequentialSampler",
        )
        observed = {"legacy": False}

        def guard(_module, _inputs, output):
            validate_legacy(output)
            observed["legacy"] = True

        hook = model.register_forward_hook(guard)
        try:
            probabilities, targets, losses = original_collect(
                model,
                loader,
                device,
            )
        finally:
            hook.remove()
        if not observed["legacy"]:
            _fail("shared evaluator observed no legacy six-tensor forward")
        state["collect_called"] = True
        state["legacy_output_verified"] = True
        state["prediction_cache"] = seal_or_validate_prediction_cache(
            request,
            probabilities=probabilities,
            targets=targets,
            losses=losses,
            source_binding=source_binding,
        )
        return probabilities, targets, losses

    def bound_write(
        path: Path,
        payload: Mapping[str, Any],
        overwrite: bool,
    ) -> None:
        if overwrite:
            _fail("engineering checkpoint-local evaluation forbids overwrite")
        if state["write_called"]:
            _fail("shared evaluator attempted more than one result write")
        _require_equal(
            "shared evaluator output path",
            Path(path).resolve(),
            request.planned_output_path.resolve(),
        )
        if not state["collect_called"] or not state["legacy_output_verified"]:
            _fail("shared evaluator write preceded prediction collection")
        if not isinstance(state["prediction_cache"], Mapping):
            _fail("shared evaluator produced no complete prediction cache")
        finalized = validate_checkpoint_local_result(
            payload,
            request,
            execution_context={
                "shared_evaluator_completed": True,
                "legacy_six_tensor_eval_output": True,
                "device_assignment": copy.deepcopy(dict(assignment)),
                "evaluation_source_binding": copy.deepcopy(
                    dict(source_binding)
                ),
                "prediction_cache": copy.deepcopy(
                    dict(state["prediction_cache"])
                ),
            },
        )
        state["output_path"] = write_result_once(
            Path(path),
            finalized,
            request,
        )
        state["write_called"] = True

    evaluator.adaptive_thresholds = (
        closed_interval_core.adaptive_thresholds_closed_interval
    )
    evaluator.build_model = bound_build_model
    evaluator.collect_predictions = bound_collect_predictions
    evaluator.parse_args = bound_parse_args
    evaluator.write_output_json = bound_write
    # The shared evaluator records the operational adapter as the evaluator;
    # the frozen shared and closed-interval cores are bound separately.
    evaluator.__file__ = str(Path(__file__).resolve())
    return evaluator, state


def execute_checkpoint(
    request: CheckpointEvaluationRequest,
    *,
    assignment: Mapping[str, Any],
) -> Path:
    """Execute one missing checkpoint sweep; never replace an existing result."""

    _validate_request_shape(request)
    _verify_request_files_unchanged(request)
    if request.planned_output_path.exists() or request.planned_output_path.is_symlink():
        _fail("execute_checkpoint accepts only a missing output")
    inputs = _load_replication_inputs_for_request(request)
    source_binding = frozen_evaluation_core_binding()
    configured = b_frozen_evaluator.configure_v8_inference(
        str(assignment["device"])
    )
    _require_equal(
        "configured inference device",
        configured.get("device"),
        assignment.get("device"),
    )
    with trajectory_model_builder(request, inputs) as build_model:
        evaluator, state = _load_bound_shared_evaluator(
            request,
            build_model=build_model,
            assignment=assignment,
            source_binding=source_binding,
        )
        _verify_request_files_unchanged(request)
        evaluator.main()
        _verify_request_files_unchanged(request)
    if not state["write_called"] or state["output_path"] is None:
        _fail("shared evaluator completed without a checkpoint-local result")
    output = Path(state["output_path"])
    load_completed_result(request)
    del evaluator
    gc.collect()
    if assignment.get("device") == "cuda:0":
        sweep_core.torch.cuda.empty_cache()
    return output


def evaluate_or_skip_checkpoint(
    request: CheckpointEvaluationRequest,
    *,
    assignment: Mapping[str, Any],
    executor: Callable[..., Path] = execute_checkpoint,
) -> dict[str, Any]:
    """Skip one valid result or execute exactly one missing result."""

    output = request.planned_output_path
    if output.exists() or output.is_symlink():
        payload, digest = load_completed_result(request)
        status = "skipped_valid_complete"
    else:
        executor(request, assignment=assignment)
        payload, digest = load_completed_result(request)
        status = "created"
    return {
        "threshold_domain_id": request.threshold_domain_id,
        "arm": request.arm,
        "trajectory_seed": request.trajectory_seed,
        "checkpoint_filename": request.checkpoint_filename,
        "status": status,
        "result_path": str(output.resolve()),
        "result_sha256": digest,
        "fixed_threshold_0_5": copy.deepcopy(
            payload["fixed_threshold_0_5"]
        ),
        "best_points_under_fa_budget": copy.deepcopy(
            payload["best_points_under_fa_budget"]
        ),
        "prediction_cache": copy.deepcopy(payload["prediction_cache"]),
    }


def select_requests(
    requests: Sequence[CheckpointEvaluationRequest],
    *,
    arms: Sequence[str] | None = None,
    trajectory_seeds: Sequence[int] | None = None,
    checkpoints: Sequence[str] | None = None,
) -> tuple[CheckpointEvaluationRequest, ...]:
    """Select a safe independently executable subset from the full matrix."""

    full = tuple(requests)
    assemble_evaluation_plan(full)
    selected_arms = (
        set(core.SUPPORTED_ARMS)
        if arms is None
        else set(arms)
    )
    selected_seeds = (
        set(seeds.ENGINEERING_TRAJECTORY_SEEDS)
        if trajectory_seeds is None
        else set(trajectory_seeds)
    )
    selected_checkpoints = (
        {spec[0] for spec in CHECKPOINT_SPECS}
        if checkpoints is None
        else set(checkpoints)
    )
    if not selected_arms or not selected_arms <= set(core.SUPPORTED_ARMS):
        _fail("execution arm subset is empty or invalid")
    if (
        not selected_seeds
        or not selected_seeds
        <= set(seeds.ENGINEERING_TRAJECTORY_SEEDS)
    ):
        _fail("execution trajectory-seed subset is empty or invalid")
    if (
        not selected_checkpoints
        or not selected_checkpoints
        <= {spec[0] for spec in CHECKPOINT_SPECS}
    ):
        _fail("execution checkpoint subset is empty or invalid")
    selected = tuple(
        request
        for request in full
        if request.arm in selected_arms
        and request.trajectory_seed in selected_seeds
        and request.checkpoint_filename in selected_checkpoints
    )
    if not selected:
        _fail("execution subset selected no checkpoint")
    return selected


def _manifest_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                sweep_core.json_ready(copy.deepcopy(dict(payload))),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail(f"manifest is not canonical JSON: {exc}")


def _load_manifest_object(path: Path) -> dict[str, Any]:
    source = _regular_file(path, "eight-result manifest")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot parse eight-result manifest: {exc}")
    if not isinstance(value, dict):
        _fail("eight-result manifest must contain one JSON object")
    if raw != _manifest_json_bytes(value):
        _fail("eight-result manifest is not canonical JSON")
    return value


def default_manifest_path(output_root: Path) -> Path:
    return Path(output_root).resolve() / DEFAULT_MANIFEST_FILENAME


def build_results_manifest(
    requests: Sequence[CheckpointEvaluationRequest],
) -> dict[str, Any]:
    """Read and validate all eight outputs, then build the paired manifest."""

    full = tuple(requests)
    assemble_evaluation_plan(full)
    results: list[dict[str, Any]] = []
    payload_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    for request in full:
        payload, result_sha256 = load_completed_result(request)
        key = (
            request.trajectory_seed,
            request.arm,
            request.selection_role,
        )
        payload_by_key[key] = payload
        cache = payload["prediction_cache"]
        recorded_assignment = _validate_recorded_device_assignment(
            payload.get("execution_device_assignment"),
            request=request,
        )
        results.append(
            {
                "threshold_domain_id": request.threshold_domain_id,
                "arm": request.arm,
                "variant": request.variant,
                "trajectory_seed": request.trajectory_seed,
                "selection_role": request.selection_role,
                "checkpoint_filename": request.checkpoint_filename,
                "checkpoint_role": request.checkpoint_role,
                "checkpoint_epoch": request.checkpoint_epoch,
                "checkpoint_sha256": request.checkpoint_sha256,
                "result_path": str(request.planned_output_path.resolve()),
                "result_sha256": result_sha256,
                "execution_device_assignment": copy.deepcopy(
                    recorded_assignment
                ),
                "fixed_threshold_0_5": copy.deepcopy(
                    payload["fixed_threshold_0_5"]
                ),
                "best_points_under_fa_budget": copy.deepcopy(
                    payload["best_points_under_fa_budget"]
                ),
                "prediction_cache": {
                    name: copy.deepcopy(cache[name])
                    for name in (
                        "metadata_path",
                        "metadata_sha256",
                        "arrays_path",
                        "arrays_sha256",
                        "identity",
                        "engineering_request_identity",
                        "prediction_content_sha256",
                        "image_count",
                        "image_ids_sha256",
                        "paired_image_statistics_available",
                    )
                },
            }
        )
    paired_groups: list[dict[str, Any]] = []
    selection_roles = tuple(spec[1] for spec in CHECKPOINT_SPECS)
    for trajectory_seed in seeds.ENGINEERING_TRAJECTORY_SEEDS:
        for selection_role in selection_roles:
            b_payload = payload_by_key[
                (trajectory_seed, core.ARM_B, selection_role)
            ]
            d_payload = payload_by_key[
                (trajectory_seed, core.ARM_D, selection_role)
            ]
            b_cache = b_payload["prediction_cache"]
            d_cache = d_payload["prediction_cache"]
            for name, expected in (
                ("image_count", EXPECTED_VALIDATION_COUNT),
                ("image_ids_sha256", full[0].validation_split_sha256),
                ("paired_image_statistics_available", True),
            ):
                _require_equal(
                    f"paired B cache {name}",
                    b_cache.get(name),
                    expected,
                )
                _require_equal(
                    f"paired D cache {name}",
                    d_cache.get(name),
                    expected,
                )
            paired_groups.append(
                {
                    "trajectory_seed": trajectory_seed,
                    "selection_role": selection_role,
                    "validation_count": EXPECTED_VALIDATION_COUNT,
                    "validation_ids_sha256": (
                        full[0].validation_split_sha256
                    ),
                    "arm_b_cache_metadata_path": b_cache["metadata_path"],
                    "arm_b_cache_metadata_sha256": (
                        b_cache["metadata_sha256"]
                    ),
                    "arm_d_cache_metadata_path": d_cache["metadata_path"],
                    "arm_d_cache_metadata_sha256": (
                        d_cache["metadata_sha256"]
                    ),
                    "image_level_pairing_ready": True,
                }
            )
    source_locks = {request.source_lock_sha256 for request in full}
    seed_contracts = {request.seed_contract_sha256 for request in full}
    validation_hashes = {request.validation_split_sha256 for request in full}
    if (
        len(source_locks) != 1
        or len(seed_contracts) != 1
        or len(validation_hashes) != 1
    ):
        _fail("eight-result matrix has inconsistent global bindings")
    expected_gpu_bindings = {
        arm: {
            "physical_gpu_index": _expected_arm_gpu_binding(arm)[0],
            "physical_gpu_uuid": _expected_arm_gpu_binding(arm)[1],
            "logical_device": "cuda:0",
        }
        for arm in core.SUPPORTED_ARMS
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "scope": "fixed_parent_engineering_b_d_only",
        "result_count": len(results),
        "expected_result_count": EXPECTED_SWEEP_COUNT,
        "all_checkpoint_local_results_valid": True,
        "threshold_selection_scope": "single_checkpoint_only",
        "cross_checkpoint_point_pooling": False,
        "fixed_threshold": FIXED_THRESHOLD,
        "fa_budgets": list(FA_BUDGETS),
        "source_lock_sha256": next(iter(source_locks)),
        "seed_contract_sha256": next(iter(seed_contracts)),
        "validation_count": EXPECTED_VALIDATION_COUNT,
        "validation_ids_sha256": next(iter(validation_hashes)),
        "formal_gpu_binding_policy": {
            "cpu_results_accepted": False,
            "arm_assignments": expected_gpu_bindings,
        },
        "all_results_expected_physical_gpu_bound": True,
        "paired_checkpoint_group_count": len(paired_groups),
        "paired_checkpoint_groups": paired_groups,
        "gate_m_train_image_level_inputs_ready": True,
        "paired_confidence_intervals_computed": False,
        "paired_confidence_intervals_claimed": False,
        "official_test_accessed": False,
        "evaluation_source_binding": frozen_evaluation_core_binding(),
        "results": results,
    }


def write_or_validate_manifest(
    path: Path,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    destination = Path(path)
    content = _manifest_json_bytes(payload)
    if destination.exists() or destination.is_symlink():
        observed = _load_manifest_object(destination)
        _canonical_equal("existing/expected eight-result manifest", observed, payload)
        return destination.resolve(), "skipped_identical_complete"
    try:
        _atomic_link_bytes(
            destination,
            content,
            label="eight-result manifest",
        )
    except FileExistsError:
        observed = _load_manifest_object(destination)
        _canonical_equal(
            "concurrent/expected eight-result manifest",
            observed,
            payload,
        )
        return destination.resolve(), "skipped_identical_complete"
    observed = _load_manifest_object(destination)
    _canonical_equal("written/expected eight-result manifest", observed, payload)
    return destination.resolve(), "created"


def finalize_results_manifest(
    requests: Sequence[CheckpointEvaluationRequest],
    *,
    output_root: Path,
) -> dict[str, Any]:
    payload = build_results_manifest(requests)
    path, action = write_or_validate_manifest(
        default_manifest_path(output_root),
        payload,
    )
    return {
        "status": "complete",
        "manifest_action": action,
        "manifest_path": str(path),
        "manifest_sha256": _sha256_file(path, "eight-result manifest"),
        "result_count": EXPECTED_SWEEP_COUNT,
        "gate_m_train_image_level_inputs_ready": True,
        "paired_confidence_intervals_computed": False,
    }


def try_finalize_results_manifest(
    requests: Sequence[CheckpointEvaluationRequest],
    *,
    output_root: Path,
) -> dict[str, Any]:
    missing = [
        str(request.planned_output_path)
        for request in requests
        if not request.planned_output_path.is_file()
        or request.planned_output_path.is_symlink()
    ]
    if missing:
        return {
            "status": "partial",
            "manifest_written": False,
            "completed_result_count": EXPECTED_SWEEP_COUNT - len(missing),
            "missing_result_count": len(missing),
            "missing_result_paths": missing,
            "gate_m_train_image_level_inputs_ready": False,
            "paired_confidence_intervals_computed": False,
        }
    return finalize_results_manifest(requests, output_root=output_root)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Checkpoint-local runner for eight engineering B/D sweeps"
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contract-only", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--finalize-manifest", action="store_true")
    mode.add_argument("--verify-results", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=watcher.DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--source-lock", type=Path)
    parser.add_argument(
        "--seed-contract",
        type=Path,
        default=prepare.DEFAULT_SEED_CONTRACT,
    )
    parser.add_argument(
        "--manifest-directory",
        type=Path,
        default=prepare.DEFAULT_MANIFEST_DIRECTORY,
    )
    parser.add_argument(
        "--arm",
        action="append",
        choices=core.SUPPORTED_ARMS,
        dest="arms",
    )
    parser.add_argument(
        "--trajectory-seed",
        action="append",
        type=int,
        choices=seeds.ENGINEERING_TRAJECTORY_SEEDS,
        dest="trajectory_seeds",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        choices=tuple(spec[0] for spec in CHECKPOINT_SPECS),
        dest="checkpoints",
    )
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument(
        "--physical-gpu-index",
        type=int,
        choices=(2, 3),
    )
    parser.add_argument("--physical-gpu-uuid")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _argument_parser().parse_args(argv)
    if args.contract_only:
        payload = evaluator_contract()
    else:
        if args.source_lock is None:
            _fail("--source-lock is required outside --contract-only")
        full_requests = collect_evaluation_requests(
            output_root=args.output_root,
            source_lock_path=args.source_lock,
            seed_contract_path=args.seed_contract,
            manifest_directory=args.manifest_directory,
        )
        if args.plan:
            if (
                args.arms is not None
                or args.trajectory_seeds is not None
                or args.checkpoints is not None
                or args.physical_gpu_index is not None
                or args.physical_gpu_uuid is not None
            ):
                _fail("--plan does not accept execution subset/device options")
            payload = assemble_evaluation_plan(full_requests)
        elif args.execute:
            selected = select_requests(
                full_requests,
                arms=args.arms,
                trajectory_seeds=args.trajectory_seeds,
                checkpoints=args.checkpoints,
            )
            selected_arms = {request.arm for request in selected}
            if len(selected_arms) != 1:
                _fail(
                    "--execute requires exactly one arm so its formal "
                    "physical GPU assignment is unambiguous"
                )
            selected_arm = next(iter(selected_arms))
            assignment = device_assignment(
                args.device,
                arm=selected_arm,
                physical_gpu_index=args.physical_gpu_index,
                physical_gpu_uuid=args.physical_gpu_uuid,
            )
            records = [
                evaluate_or_skip_checkpoint(
                    request,
                    assignment=assignment,
                )
                for request in selected
            ]
            payload = {
                "schema": SCHEMA,
                "status": "execution_subset_complete",
                "selected_request_count": len(selected),
                "device_assignment": assignment,
                "created_count": sum(
                    record["status"] == "created" for record in records
                ),
                "skipped_valid_complete_count": sum(
                    record["status"] == "skipped_valid_complete"
                    for record in records
                ),
                "checkpoint_results": records,
                "matrix_finalization": try_finalize_results_manifest(
                    full_requests,
                    output_root=args.output_root,
                ),
                "official_test_accessed": False,
            }
        elif args.finalize_manifest:
            if (
                args.arms is not None
                or args.trajectory_seeds is not None
                or args.checkpoints is not None
                or args.physical_gpu_index is not None
                or args.physical_gpu_uuid is not None
            ):
                _fail(
                    "--finalize-manifest requires the full matrix and no "
                    "device/subset options"
                )
            payload = finalize_results_manifest(
                full_requests,
                output_root=args.output_root,
            )
        elif args.verify_results:
            if (
                args.arms is not None
                or args.trajectory_seeds is not None
                or args.checkpoints is not None
                or args.physical_gpu_index is not None
                or args.physical_gpu_uuid is not None
            ):
                _fail(
                    "--verify-results requires the full matrix and no "
                    "device/subset options"
                )
            expected = build_results_manifest(full_requests)
            manifest_path = default_manifest_path(args.output_root)
            observed = _load_manifest_object(manifest_path)
            _canonical_equal(
                "stored/recomputed eight-result manifest",
                observed,
                expected,
            )
            payload = {
                "schema": MANIFEST_SCHEMA,
                "status": "verified_complete",
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": _sha256_file(
                    manifest_path,
                    "eight-result manifest",
                ),
                "result_count": EXPECTED_SWEEP_COUNT,
                "gate_m_train_image_level_inputs_ready": True,
                "paired_confidence_intervals_computed": False,
                "official_test_accessed": False,
            }
        else:
            _fail("unreachable execution mode")
    print(
        json.dumps(
            _canonical(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


__all__ = [
    "BUDGET_KEYS",
    "CHECKPOINT_SPECS",
    "CheckpointEvaluationRequest",
    "EngineeringEvaluationError",
    "EXPECTED_SWEEP_COUNT",
    "FA_BUDGETS",
    "FIXED_THRESHOLD",
    "MANIFEST_SCHEMA",
    "RESULT_SCHEMA",
    "SCHEMA",
    "assemble_evaluation_plan",
    "build_results_manifest",
    "build_evaluation_plan",
    "collect_evaluation_requests",
    "default_manifest_path",
    "device_assignment",
    "evaluator_contract",
    "evaluate_or_skip_checkpoint",
    "execute_checkpoint",
    "finalize_results_manifest",
    "frozen_evaluation_core_binding",
    "load_completed_result",
    "main",
    "preflight_completed_run",
    "seal_or_validate_prediction_cache",
    "select_requests",
    "trajectory_model_builder",
    "try_finalize_results_manifest",
    "validate_checkpoint_local_result",
    "validate_finalized_result",
    "write_or_validate_manifest",
    "write_result_once",
]


if __name__ == "__main__":
    main()
