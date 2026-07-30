#!/usr/bin/env python3
"""Read-only progress report for the four engineering B/D replications."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import final_model_replication_exact_core as core  # noqa: E402
from experiments import final_model_replication_seed_contract as seeds  # noqa: E402
from experiments import freeze_final_model_certification_parent_lock as parent_lock  # noqa: E402
from experiments import freeze_final_model_certification_source_lock as source_lock  # noqa: E402
from experiments.tpd_exact_epoch_journal import ExactEpochJournal  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "experiments/results/final_model_engineering_replication_v1"
)
FORMAL_EPOCHS = 800


class ReplicationRunStateError(ValueError):
    """An existing engineering run cannot be classified without ambiguity."""


def _fail(message: str) -> None:
    raise ReplicationRunStateError(message)


def _canonical_pretty_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _fail(f"value is not canonical JSON: {exc}")


def _load_canonical_object(path: Path, label: str) -> dict[str, Any]:
    value_path = Path(path)
    if value_path.is_symlink() or not value_path.is_file():
        _fail(f"{label} must be a regular non-symlink file: {value_path}")
    raw = value_path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot parse {label}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must contain one JSON object")
    if raw != _canonical_pretty_json_bytes(value):
        _fail(f"{label} is not canonical pretty JSON")
    return value


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        _fail(
            f"{label} differs: expected={expected!r}, "
            f"observed={observed!r}"
        )


def _validate_split(split: Mapping[str, Any]) -> None:
    expected = {
        "dataset": "NUDT-SIRST",
        "split_seed": seeds.SPLIT_SEED,
        "full_internal_train_count": parent_lock.TRAIN_COUNT,
        "full_internal_val_count": parent_lock.VALIDATION_COUNT,
        "used_train_count": parent_lock.TRAIN_COUNT,
        "used_val_count": parent_lock.VALIDATION_COUNT,
        "official_test_accessed": False,
    }
    for name, required in expected.items():
        _require_equal(f"split {name}", split.get(name), required)
    hashes = split.get("hashes")
    if not isinstance(hashes, Mapping):
        _fail("split hashes are missing")
    expected_hashes = {
        "full_internal_train_sha256": parent_lock.TRAIN_IDS_SHA256,
        "full_internal_val_sha256": parent_lock.VALIDATION_IDS_SHA256,
        "used_train_sha256": parent_lock.TRAIN_IDS_SHA256,
        "used_val_sha256": parent_lock.VALIDATION_IDS_SHA256,
    }
    if set(hashes) != set(expected_hashes):
        _fail("split hash keys differ")
    for name, required in expected_hashes.items():
        _require_equal(f"split hash {name}", hashes.get(name), required)
    identifier_contract = (
        (
            "full_internal_train_ids",
            parent_lock.TRAIN_COUNT,
            "full_internal_train_sha256",
        ),
        (
            "full_internal_val_ids",
            parent_lock.VALIDATION_COUNT,
            "full_internal_val_sha256",
        ),
        (
            "used_train_ids",
            parent_lock.TRAIN_COUNT,
            "used_train_sha256",
        ),
        (
            "used_val_ids",
            parent_lock.VALIDATION_COUNT,
            "used_val_sha256",
        ),
    )
    identifier_sets: dict[str, set[str]] = {}
    for name, count, hash_name in identifier_contract:
        identifiers = split.get(name)
        if not isinstance(identifiers, list) or len(identifiers) != count:
            _fail(f"split {name} count differs")
        if any(
            not isinstance(identifier, str) or not identifier
            for identifier in identifiers
        ):
            _fail(f"split {name} must contain non-empty string identifiers")
        identifier_set = set(identifiers)
        if len(identifier_set) != len(identifiers):
            _fail(f"split {name} contains duplicate identifiers")
        observed_digest = hashlib.sha256(
            "\n".join(sorted(identifiers)).encode("utf-8")
        ).hexdigest()
        _require_equal(
            f"split recomputed hash {hash_name}",
            observed_digest,
            expected_hashes[hash_name],
        )
        identifier_sets[name] = identifier_set
    if (
        identifier_sets["full_internal_train_ids"]
        & identifier_sets["full_internal_val_ids"]
    ):
        _fail("split full internal train/validation identifiers overlap")
    if identifier_sets["used_train_ids"] & identifier_sets["used_val_ids"]:
        _fail("split used train/validation identifiers overlap")
    if (
        identifier_sets["used_train_ids"]
        != identifier_sets["full_internal_train_ids"]
    ):
        _fail("split used train identifiers differ from the full train split")
    if (
        identifier_sets["used_val_ids"]
        != identifier_sets["full_internal_val_ids"]
    ):
        _fail(
            "split used validation identifiers differ from the full "
            "validation split"
        )


def _validate_protocol_identity(
    protocol: Mapping[str, Any],
    *,
    trajectory_seed: int,
    arm: str,
    directory: Path,
) -> dict[str, Any]:
    definition = core.arm_definition(arm)
    trainer = definition.trainer
    _require_equal("protocol schema", protocol.get("schema"), trainer.ENTRY_SCHEMA)
    _require_equal(
        "protocol run directory",
        protocol.get("run_directory"),
        str(directory.resolve()),
    )
    identity = protocol.get("run_identity")
    if not isinstance(identity, Mapping):
        _fail("protocol run identity is missing")
    expected_run_id = (
        f"{trainer.RUN_ID_PREFIX}NUDT-SIRST:{definition.variant}:"
        f"seed-{trajectory_seed}:split-{seeds.SPLIT_SEED}:"
        f"{core.ENGINEERING_RUN_TAGS[arm]}"
    )
    expected_identity = {
        "schema": core.exact_runner.RUN_IDENTITY_SCHEMA,
        "run_id": expected_run_id,
        "variant": definition.variant,
        "dataset": "NUDT-SIRST",
        "seed": trajectory_seed,
        "split_seed": seeds.SPLIT_SEED,
    }
    for name, required in expected_identity.items():
        _require_equal(
            f"protocol run identity {name}",
            identity.get(name),
            required,
        )
    source_locks = identity.get("source_locks")
    expected_source_lock_keys = {
        core.SOURCE_LOCK_KEY,
        "training_data",
        "survival_target_statistics",
        "parent_checkpoint",
    }
    if (
        not isinstance(source_locks, Mapping)
        or set(source_locks) != expected_source_lock_keys
    ):
        _fail("protocol run identity source-lock keys differ")
    _require_equal(
        "protocol parent checkpoint source lock",
        source_locks.get("parent_checkpoint"),
        trainer.PARENT_CHECKPOINT_SHA256,
    )
    training = identity.get("training_contract")
    if not isinstance(training, Mapping):
        _fail("protocol training contract is missing")
    initialization = training.get("initialization_contract")
    if not isinstance(initialization, Mapping):
        _fail("protocol initialization contract is missing")
    _require_equal(
        "protocol initialization mode",
        initialization.get("mode"),
        core.exact_runner.EXTENSION_PARENT_MODE,
    )
    provenance = initialization.get("provenance")
    if not isinstance(provenance, Mapping):
        _fail("protocol extension-parent provenance is missing")
    _require_equal(
        "protocol initialization parent SHA-256",
        provenance.get("parent_checkpoint_sha256"),
        trainer.PARENT_CHECKPOINT_SHA256,
    )
    _require_equal(
        "protocol total epochs",
        training.get("manual_lr_schedule", {}).get("total_epochs")
        if isinstance(training.get("manual_lr_schedule"), Mapping)
        else None,
        FORMAL_EPOCHS,
    )
    determinism = training.get("determinism")
    if not isinstance(determinism, Mapping):
        _fail("protocol determinism contract is missing")
    _require_equal(
        "protocol loader-generator seed",
        determinism.get("loader_generator_seed"),
        trajectory_seed,
    )
    environment = training.get("environment")
    if not isinstance(environment, Mapping):
        _fail("protocol environment contract is missing")
    expected_gpu_index = 2 if arm == core.ARM_B else 3
    expected_gpu_uuid = trainer.PHYSICAL_GPU_UUIDS[str(expected_gpu_index)]
    _require_equal(
        "protocol physical GPU index",
        environment.get("physical_gpu_index"),
        expected_gpu_index,
    )
    _require_equal(
        "protocol physical GPU UUID",
        environment.get("physical_gpu_uuid"),
        expected_gpu_uuid,
    )
    model = protocol.get("model")
    if not isinstance(model, Mapping):
        _fail("protocol model metadata is missing")
    _require_equal("protocol model variant", model.get("variant"), definition.variant)
    _require_equal(
        "protocol model training seed",
        model.get("training_seed"),
        trajectory_seed,
    )
    replication = model.get("replication_contract")
    if not isinstance(replication, Mapping):
        _fail("protocol replication contract is missing")
    expected_replication = {
        "arm": arm,
        "variant": definition.variant,
        "trajectory_seed": trajectory_seed,
        "split_seed": seeds.SPLIT_SEED,
        "parent_training_seed": seeds.BUILDER_COMPATIBILITY_SEED,
        "parent_checkpoint_sha256": trainer.PARENT_CHECKPOINT_SHA256,
        "parent_state_dict_sha256": trainer.PARENT_STATE_DICT_SHA256,
        "parent_load_count": 1,
        "optimizer_inherited": False,
        "scheduler_inherited": False,
        "all_child_parameters_trainable": True,
    }
    for name, required in expected_replication.items():
        _require_equal(
            f"protocol replication contract {name}",
            replication.get(name),
            required,
        )
    return dict(identity)


def run_directory(output_root: Path, seed: int, arm: str) -> Path:
    definition = core.arm_definition(arm)
    return (
        Path(output_root)
        / "NUDT-SIRST"
        / definition.variant
        / f"seed_{seed}_{core.ENGINEERING_RUN_TAGS[arm]}"
    )


def _metrics_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"metrics path is not a regular file: {path}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid metrics line {line_number} in {path}: {exc}"
            )
        if not isinstance(value, Mapping):
            raise ValueError(f"metrics line {line_number} is not an object")
        event = dict(value)
        epoch = event.get("epoch")
        if epoch != len(events) + 1:
            raise ValueError(f"non-contiguous metrics epoch in {path}")
        events.append(event)
    return events


def _validate_complete_summary(
    summary: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    trajectory_seed: int,
    arm: str,
    directory: Path,
    active_epoch: int,
) -> None:
    definition = core.arm_definition(arm)
    trainer = definition.trainer
    expected = {
        "schema": trainer.COMPLETION_SUMMARY_SCHEMA,
        "status": "complete",
        "variant": definition.variant,
        "dataset": "NUDT-SIRST",
        "seed": trajectory_seed,
        "split_seed": seeds.SPLIT_SEED,
    }
    for name, required in expected.items():
        _require_equal(f"completion summary {name}", summary.get(name), required)
    _require_equal(
        "completion summary run identity",
        summary.get("run_identity"),
        dict(identity),
    )
    if active_epoch != FORMAL_EPOCHS:
        _fail(
            "completion summary exists before the exact journal reached "
            f"epoch {FORMAL_EPOCHS}"
        )
    events = _metrics_events(directory / "metrics.jsonl")
    if len(events) != FORMAL_EPOCHS:
        _fail("completed run metrics do not contain exactly 800 epochs")
    for name in ("best_epoch", "best_pd_epoch", "best_miou_epoch"):
        epoch = summary.get(name)
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 1
            or epoch > FORMAL_EPOCHS
        ):
            _fail(f"completion summary {name} is invalid")
    for name in (
        "best_validation_metrics",
        "best_pd_validation_metrics",
        "best_miou_validation_metrics",
    ):
        if not isinstance(summary.get(name), Mapping):
            _fail(f"completion summary {name} is missing")
    model = summary.get("model")
    if not isinstance(model, Mapping):
        _fail("completion summary model metadata is missing")
    protocol_model = _load_canonical_object(
        directory / "protocol.json",
        "replication protocol",
    ).get("model")
    _require_equal(
        "completion summary model metadata",
        model,
        protocol_model,
    )
    for name in ("last.pth.tar", "best.pth.tar", "best_miou.pth.tar"):
        checkpoint = directory / name
        if checkpoint.is_symlink() or not checkpoint.is_file():
            _fail(f"completed run checkpoint is missing: {checkpoint}")


def resolve_initialization_mode(
    output_root: Path,
    trajectory_seed: int,
    arm: str,
) -> str:
    """Return one unambiguous trainer flag or ``--complete``.

    Existing artifacts are validated before they can cause a resume or skip.
    The function is read-only: in particular, it does not instantiate a
    journal until its existing directory has been checked.
    """

    if trajectory_seed not in seeds.ENGINEERING_TRAJECTORY_SEEDS:
        _fail("trajectory seed is not in the engineering schedule")
    if arm not in core.SUPPORTED_ARMS:
        _fail(f"unsupported engineering arm: {arm!r}")
    directory = run_directory(output_root, trajectory_seed, arm)
    if directory.is_symlink():
        _fail(f"replication run directory must not be a symlink: {directory}")
    if not directory.exists():
        return "--parent-warm-start"
    if not directory.is_dir():
        _fail(f"replication run path is not a directory: {directory}")

    protocol = _load_canonical_object(
        directory / "protocol.json",
        "replication protocol",
    )
    split = _load_canonical_object(
        directory / "split.json",
        "replication split",
    )
    identity = _validate_protocol_identity(
        protocol,
        trajectory_seed=trajectory_seed,
        arm=arm,
        directory=directory,
    )
    _validate_split(split)

    journal_directory = directory / "exact_journal"
    if journal_directory.is_symlink() or not journal_directory.is_dir():
        _fail(
            "existing replication run has no regular exact journal "
            f"directory: {journal_directory}"
        )
    active = ExactEpochJournal(journal_directory).load_active()
    if active is None:
        _fail("existing replication run has no committed active epoch")
    if active.epoch > FORMAL_EPOCHS:
        _fail("active journal epoch exceeds the formal 800-epoch contract")

    summary_path = directory / "summary.json"
    if summary_path.is_symlink():
        _fail(f"completion summary must not be a symlink: {summary_path}")
    if summary_path.exists():
        summary = _load_canonical_object(
            summary_path,
            "completion summary",
        )
        _validate_complete_summary(
            summary,
            identity=identity,
            trajectory_seed=trajectory_seed,
            arm=arm,
            directory=directory,
            active_epoch=active.epoch,
        )
        return "--complete"
    return "--exact-resume"


def inspect_run(output_root: Path, seed: int, arm: str) -> dict[str, Any]:
    directory = run_directory(output_root, seed, arm)
    events = _metrics_events(directory / "metrics.jsonl")
    last = events[-1] if events else None
    summary_path = directory / "summary.json"
    checkpoints = {
        name: (directory / name).is_file()
        and not (directory / name).is_symlink()
        for name in ("last.pth.tar", "best.pth.tar", "best_miou.pth.tar")
    }
    if summary_path.is_file() and not summary_path.is_symlink():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, Mapping):
            raise ValueError(f"summary is not an object: {summary_path}")
        complete = summary.get("status") == "complete"
    else:
        complete = False
    progress: dict[str, Any] = {
        "arm": arm,
        "variant": core.arm_definition(arm).variant,
        "trajectory_seed": seed,
        "run_directory": str(directory.resolve()),
        "started": directory.is_dir(),
        "completed_epochs": len(events),
        "total_epochs": 800,
        "progress_fraction": len(events) / 800.0,
        "complete": complete,
        "checkpoints": checkpoints,
    }
    if last is not None:
        progress["latest"] = {
            key: last.get(key)
            for key in (
                "epoch",
                "miou",
                "pd",
                "fa",
                "tiny_pd",
                "false_objects_per_image",
                "epoch_seconds",
            )
        }
    return progress


def build_report(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    runs = [
        inspect_run(output_root, seed, arm)
        for seed in seeds.ENGINEERING_TRAJECTORY_SEEDS
        for arm in core.SUPPORTED_ARMS
    ]
    return {
        "schema": "sctransnet_final_model_engineering_progress_v1",
        "status": (
            "complete" if all(run["complete"] for run in runs) else "running"
        ),
        "run_count": len(runs),
        "complete_run_count": sum(
            int(bool(run["complete"])) for run in runs
        ),
        "official_test_accessed": False,
        "runs": runs,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=source_lock.DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--resolve-mode",
        action="store_true",
        help="validate one run and print its initialization mode",
    )
    parser.add_argument(
        "--trajectory-seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--arm",
        choices=core.SUPPORTED_ARMS,
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    source_lock.verify_source_lock(args.source_lock, repo_root=REPO_ROOT)
    if args.resolve_mode:
        if args.trajectory_seed is None or args.arm is None:
            raise ValueError(
                "--resolve-mode requires --trajectory-seed and --arm"
            )
        print(
            resolve_initialization_mode(
                args.output_root,
                args.trajectory_seed,
                args.arm,
            )
        )
        return
    if args.trajectory_seed is not None or args.arm is not None:
        raise ValueError(
            "--trajectory-seed/--arm are valid only with --resolve-mode"
        )
    print(
        json.dumps(
            build_report(args.output_root),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
