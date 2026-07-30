"""Exact adapter for the new fixed-seed-42 B/D certification replay.

The frozen B/D trainers and the 31-file certification lock are imported but
never modified.  This adapter gives the replay a new run tag, output root,
contract, child manifest, and exact-resume identity.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from experiments import final_model_replication_exact_core as frozen_core
from experiments import (
    final_model_seed42_certification_replay_contract as replay_contract,
)
from experiments import (
    freeze_final_model_certification_parent_lock as parent_lock,
)
from experiments import (
    freeze_final_model_certification_source_lock as source_lock,
)
from experiments import (
    freeze_final_model_seed42_certification_replay_source_lock
    as replay_source_lock,
)
from experiments.tpd_exact_epoch_journal import ExactEpochJournal


SCHEMA = "sctransnet_final_model_seed42_certification_replay_exact_core_v2"
ACTION_SCHEMA = (
    "sctransnet_final_model_seed42_certification_replay_exact_action_v2"
)
SOURCE_LOCK_KEY = "final_model_seed42_certification_replay_source_lock_v4"
ARM_B = frozen_core.ARM_B
ARM_D = frozen_core.ARM_D
SUPPORTED_ARMS = frozen_core.SUPPORTED_ARMS
_CONTROLLED_OPTIONS = (
    "--variant",
    "--dataset",
    "--dataset-dir",
    "--output-root",
    "--run-tag",
    "--device",
    "--epochs",
    "--batch-size",
    "--patch-size",
    "--workers",
    "--seed",
    "--split-seed",
    "--val-fraction",
    "--eval-every",
    "--base-lr",
    "--min-lr",
    "--warmup-epochs",
    "--threshold",
    "--match-radius",
    "--tiny-area",
    "--eps",
    "--survival-weight",
    "--survival-pos-weight",
    "--survival-target-statistics",
    "--parent-checkpoint",
    "--exact-source-lock",
    "--allow-cpu-smoke",
    "--max-train-images",
    "--max-val-images",
    "--fresh",
)


class Seed42ReplayExactError(ValueError):
    """A replay launch or existing replay state violates its identity."""


def _fail(message: str) -> None:
    raise Seed42ReplayExactError(message)


def _require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        _fail(
            f"{label} differs: expected={expected!r}, observed={observed!r}"
        )


def require_all_parameters_trainable(model: nn.Module) -> None:
    frozen = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    if frozen:
        _fail(
            "replay child contains non-trainable parameters: "
            f"{frozen[:8]}"
        )


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
        _fail(f"value is not canonical pretty JSON: {exc}")


def _load_pretty_object(path: Path, label: str) -> dict[str, Any]:
    value_path = Path(path)
    if value_path.is_symlink() or not value_path.is_file():
        _fail(f"{label} must be a regular non-symlink file: {value_path}")
    raw = value_path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot parse {label}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must contain one object")
    if raw != _canonical_pretty_json_bytes(value):
        _fail(f"{label} is not canonical pretty JSON")
    return value


@dataclass(frozen=True)
class ReplayInputs:
    definition: frozen_core.ArmDefinition
    trajectory_seed: int
    contract: dict[str, Any]
    contract_path: Path
    contract_sha256: str
    initialization: dict[str, Any]
    initialization_path: Path
    initialization_sha256: str
    source_lock_path: Path
    source_lock_sha256: str
    source_lock_schema: str
    upstream_certification_source_lock_path: Path
    upstream_certification_source_lock_sha256: str
    parent_lock_path: Path
    parent_lock_sha256: str
    parent_lock_schema: str
    parent_checkpoint_path: Path
    parent_checkpoint_sha256: str
    parent_state_dict_sha256: str
    parent_seed: int
    parent_epoch: int
    initialization_scope: str
    output_root: Path
    run_tag: str
    physical_gpu_index: int
    physical_gpu_uuid: str

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "scope": "new_fixed_seed42_b_d_certification_replay",
            "arm": self.definition.arm,
            "variant": self.definition.variant,
            "trajectory_seed": self.trajectory_seed,
            "builder_compatibility_seed": replay_contract.BUILDER_COMPATIBILITY_SEED,
            "extension_initialization_seed": (
                replay_contract.EXTENSION_INITIALIZATION_SEED
            ),
            "split_seed": replay_contract.SPLIT_SEED,
            "replay_contract_path": str(self.contract_path),
            "replay_contract_sha256": self.contract_sha256,
            "replay_source_lock_path": str(self.source_lock_path),
            "replay_source_lock_sha256": self.source_lock_sha256,
            "replay_source_lock_schema": self.source_lock_schema,
            "upstream_certification_source_lock_path": str(
                self.upstream_certification_source_lock_path
            ),
            "upstream_certification_source_lock_sha256": (
                self.upstream_certification_source_lock_sha256
            ),
            "certification_parent_lock_path": str(self.parent_lock_path),
            "certification_parent_lock_sha256": self.parent_lock_sha256,
            "certification_parent_lock_schema": self.parent_lock_schema,
            "child_initialization_manifest_path": str(
                self.initialization_path
            ),
            "child_initialization_manifest_sha256": (
                self.initialization_sha256
            ),
            "parent_training_seed": self.parent_seed,
            "parent_checkpoint_path": str(self.parent_checkpoint_path),
            "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
            "parent_state_dict_sha256": self.parent_state_dict_sha256,
            "parent_checkpoint_epoch": self.parent_epoch,
            "initialization_scope": self.initialization_scope,
            "parent_load_count": 1,
            "optimizer_inherited": False,
            "scheduler_inherited": False,
            "all_child_parameters_trainable": True,
            "fresh_adam": True,
            "run_tag": self.run_tag,
            "output_root": str(self.output_root),
            "legacy_checkpoint_imported": False,
            "legacy_exact_journal_imported": False,
        }


def _repo_path(relative: str) -> Path:
    path = Path(relative)
    return path.resolve() if path.is_absolute() else (
        replay_contract.REPO_ROOT / path
    ).resolve()


def validate_inputs(
    *,
    arm: str,
    contract_path: Path = replay_contract.DEFAULT_CONTRACT,
    initialization_manifest_path: Path,
    certification_source_lock_path: Path = source_lock.DEFAULT_OUTPUT,
    certification_parent_lock_path: Path = parent_lock.DEFAULT_OUTPUT,
    replay_source_lock_path: Path = replay_source_lock.DEFAULT_OUTPUT,
) -> ReplayInputs:
    if arm not in SUPPORTED_ARMS:
        _fail(f"unsupported replay arm: {arm!r}")
    contract = replay_contract.load_contract(
        contract_path,
        source_lock_path=certification_source_lock_path,
        parent_lock_path=certification_parent_lock_path,
    )
    initialization = replay_contract.load_child_manifest(
        initialization_manifest_path,
        arm=arm,
        contract_path=contract_path,
        source_lock_path=certification_source_lock_path,
        parent_lock_path=certification_parent_lock_path,
    )
    definition = frozen_core.arm_definition(arm)
    arm_record = next(
        record for record in contract["arms"] if record["arm"] == arm
    )
    verified_replay_source_lock = replay_source_lock.verify_source_lock(
        replay_source_lock_path,
        contract_path=contract_path,
        manifest_directory=Path(initialization_manifest_path).parent,
        upstream_source_lock_path=certification_source_lock_path,
        parent_lock_path=certification_parent_lock_path,
    )
    replay_source_lock_file = Path(replay_source_lock_path).resolve()
    replay_source_lock_sha256 = replay_contract.file_sha256(
        replay_source_lock_file
    )
    _require_equal(
        "verified replay source-lock schema",
        verified_replay_source_lock.get("schema"),
        replay_source_lock.SCHEMA,
    )
    parent = initialization["parent"]
    inputs = ReplayInputs(
        definition=definition,
        trajectory_seed=replay_contract.TRAJECTORY_SEED,
        contract=contract,
        contract_path=Path(contract_path).resolve(),
        contract_sha256=replay_contract.file_sha256(contract_path),
        initialization=initialization,
        initialization_path=Path(initialization_manifest_path).resolve(),
        initialization_sha256=replay_contract.file_sha256(
            initialization_manifest_path
        ),
        source_lock_path=replay_source_lock_file,
        source_lock_sha256=replay_source_lock_sha256,
        source_lock_schema=str(verified_replay_source_lock["schema"]),
        upstream_certification_source_lock_path=Path(
            certification_source_lock_path
        ).resolve(),
        upstream_certification_source_lock_sha256=(
            replay_contract.file_sha256(certification_source_lock_path)
        ),
        parent_lock_path=Path(certification_parent_lock_path).resolve(),
        parent_lock_sha256=replay_contract.file_sha256(
            certification_parent_lock_path
        ),
        parent_lock_schema=parent_lock.LOCK_SCHEMA,
        parent_checkpoint_path=_repo_path(str(parent["path"])),
        parent_checkpoint_sha256=str(parent["sha256"]),
        parent_state_dict_sha256=str(parent["state_dict_sha256"]),
        parent_seed=int(parent["seed"]),
        parent_epoch=int(parent["epoch"]),
        initialization_scope="fixed_seed42_certification_replay",
        output_root=_repo_path(contract["replay_identity"]["output_root"]),
        run_tag=str(arm_record["run_tag"]),
        physical_gpu_index=int(arm_record["physical_gpu_index"]),
        physical_gpu_uuid=str(arm_record["physical_gpu_uuid"]),
    )
    _require_equal(
        "trajectory seed",
        initialization["trajectory_seed"],
        replay_contract.TRAJECTORY_SEED,
    )
    _require_equal(
        "manifest run directory",
        _repo_path(initialization["run_directory"]),
        run_directory(inputs),
    )
    _require_equal(
        "certification source-lock SHA",
        inputs.upstream_certification_source_lock_sha256,
        initialization["certification_source_lock_sha256"],
    )
    _require_equal(
        "certification parent-lock SHA",
        inputs.parent_lock_sha256,
        initialization["certification_parent_lock_sha256"],
    )
    frozen_core.verify_parent_checkpoint_payload(inputs)
    return inputs


def run_directory(inputs: ReplayInputs) -> Path:
    return (
        inputs.output_root
        / replay_contract.DATASET
        / inputs.definition.variant
        / f"seed_{inputs.trajectory_seed}_{inputs.run_tag}"
    ).resolve()


@contextlib.contextmanager
def _temporary_attributes(
    module: ModuleType,
    values: Mapping[str, Any],
) -> Iterator[None]:
    previous: dict[str, Any] = {}
    for name, value in values.items():
        if not hasattr(module, name):
            _fail(f"trainer lacks required runtime attribute {name!r}")
        previous[name] = getattr(module, name)
        setattr(module, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(module, name, value)


@contextlib.contextmanager
def replay_trainer_overlay(inputs: ReplayInputs) -> Iterator[ModuleType]:
    """Expose the frozen trainer under the new replay-only identity."""

    trainer = inputs.definition.trainer
    original_builder = trainer.build_selected_model
    original_initialization_plan = trainer.initialization_plan

    def builder(
        variant: str,
        seed: int,
        *,
        eps: float,
    ) -> tuple[nn.Module, dict[str, Any]]:
        if variant != inputs.definition.variant:
            _fail("replay builder arm/variant mismatch")
        if seed != replay_contract.TRAJECTORY_SEED:
            _fail("replay builder seed must remain 42")
        model, raw_metadata = original_builder(variant, seed, eps=eps)
        require_all_parameters_trainable(model)
        if not isinstance(raw_metadata, Mapping):
            _fail("frozen builder returned invalid metadata")
        metadata = copy.deepcopy(dict(raw_metadata))
        metadata.update(
            {
                "training_seed": replay_contract.TRAJECTORY_SEED,
                "parent_checkpoint_path": str(
                    inputs.parent_checkpoint_path
                ),
                "parent_checkpoint_sha256": (
                    inputs.parent_checkpoint_sha256
                ),
                "parent_checkpoint_epoch": inputs.parent_epoch,
                "parent_checkpoint_state_dict_sha256": (
                    inputs.parent_state_dict_sha256
                ),
                "replication_contract": inputs.metadata(),
            }
        )
        return model, metadata

    def initialization_plan(
        args: argparse.Namespace,
        directory: Path,
        model: nn.Module,
    ) -> Any:
        if args.parent_warm_start:
            plan = frozen_core.prepare_extension_parent_once(inputs, model)
            require_all_parameters_trainable(model)
            return plan
        if args.exact_resume:
            require_all_parameters_trainable(model)
            return original_initialization_plan(args, directory, model)
        _fail("replay initialization mode is missing")

    def source_lock_contract(
        training_data_sha256: str,
        exact_source_lock_path: str | os.PathLike[str],
        statistics_path: str | os.PathLike[str],
    ) -> dict[str, str]:
        requested_path = Path(exact_source_lock_path)
        if requested_path.is_symlink() or not requested_path.is_file():
            _fail(
                "requested replay source lock must be a regular "
                f"non-symlink file: {requested_path}"
            )
        requested = requested_path.resolve()
        if requested != inputs.source_lock_path:
            _fail("trainer received a different certification source lock")
        _require_equal(
            "cached replay source-lock schema",
            inputs.source_lock_schema,
            replay_source_lock.SCHEMA,
        )
        _require_equal(
            "live replay source-lock SHA",
            replay_contract.file_sha256(requested),
            inputs.source_lock_sha256,
        )
        return {
            SOURCE_LOCK_KEY: inputs.source_lock_sha256,
            "training_data": str(training_data_sha256),
            "survival_target_statistics": replay_contract.file_sha256(
                statistics_path
            ),
            "parent_checkpoint": inputs.parent_checkpoint_sha256,
        }

    overlay = {
        "TRAINING_SEED": replay_contract.TRAJECTORY_SEED,
        "SPLIT_SEED": replay_contract.SPLIT_SEED,
        "SOURCE_LOCK_KEY": SOURCE_LOCK_KEY,
        "PARENT_CHECKPOINT_PATH": inputs.parent_checkpoint_path,
        "PARENT_CHECKPOINT_SHA256": inputs.parent_checkpoint_sha256,
        "PARENT_STATE_DICT_SHA256": inputs.parent_state_dict_sha256,
        "PARENT_CHECKPOINT_EPOCH": inputs.parent_epoch,
        "FORMAL_RUN_TAGS": {
            **dict(trainer.FORMAL_RUN_TAGS),
            inputs.definition.variant: inputs.run_tag,
        },
        "build_selected_model": builder,
        "initialization_plan": initialization_plan,
        "source_lock_contract": source_lock_contract,
    }
    with _temporary_attributes(trainer, overlay):
        yield trainer


def expected_run_id(inputs: ReplayInputs) -> str:
    return (
        f"{inputs.definition.trainer.RUN_ID_PREFIX}"
        f"{replay_contract.DATASET}:{inputs.definition.variant}:"
        f"seed-{inputs.trajectory_seed}:split-{replay_contract.SPLIT_SEED}:"
        f"{inputs.run_tag}"
    )


def _validate_existing_protocol(
    inputs: ReplayInputs,
    directory: Path,
) -> dict[str, Any]:
    protocol = _load_pretty_object(
        directory / "protocol.json",
        "seed-42 replay protocol",
    )
    trainer = inputs.definition.trainer
    _require_equal("protocol schema", protocol.get("schema"), trainer.ENTRY_SCHEMA)
    _require_equal(
        "protocol run directory",
        protocol.get("run_directory"),
        str(directory),
    )
    identity = protocol.get("run_identity")
    if not isinstance(identity, Mapping):
        _fail("protocol run identity is missing")
    expected = {
        "run_id": expected_run_id(inputs),
        "variant": inputs.definition.variant,
        "dataset": replay_contract.DATASET,
        "seed": replay_contract.TRAJECTORY_SEED,
        "split_seed": replay_contract.SPLIT_SEED,
    }
    for name, value in expected.items():
        _require_equal(f"protocol identity {name}", identity.get(name), value)
    locks = identity.get("source_locks")
    if not isinstance(locks, Mapping):
        _fail("protocol source-lock contract is missing")
    _require_equal(
        "protocol certification source lock",
        locks.get(SOURCE_LOCK_KEY),
        inputs.source_lock_sha256,
    )
    model = protocol.get("model")
    if not isinstance(model, Mapping):
        _fail("protocol model metadata is missing")
    replay = model.get("replication_contract")
    if not isinstance(replay, Mapping):
        _fail("protocol replay metadata is missing")
    _require_equal(
        "protocol replay contract SHA",
        replay.get("replay_contract_sha256"),
        inputs.contract_sha256,
    )
    _require_equal(
        "protocol child manifest SHA",
        replay.get("child_initialization_manifest_sha256"),
        inputs.initialization_sha256,
    )
    return dict(identity)


def resolve_initialization_mode(inputs: ReplayInputs) -> str:
    """Return parent-warm-start, exact-resume, or complete, read-only."""

    directory = run_directory(inputs)
    if directory.is_symlink():
        _fail(f"replay run directory must not be a symlink: {directory}")
    if not directory.exists():
        return "--parent-warm-start"
    if not directory.is_dir():
        _fail(f"replay run path is not a directory: {directory}")
    identity = _validate_existing_protocol(inputs, directory)
    split = _load_pretty_object(directory / "split.json", "replay split")
    for name, expected in (
        ("dataset", replay_contract.DATASET),
        ("split_seed", replay_contract.SPLIT_SEED),
        ("used_train_count", parent_lock.TRAIN_COUNT),
        ("used_val_count", parent_lock.VALIDATION_COUNT),
        ("official_test_accessed", False),
    ):
        _require_equal(f"split {name}", split.get(name), expected)
    journal_directory = directory / "exact_journal"
    if journal_directory.is_symlink() or not journal_directory.is_dir():
        _fail("existing replay run has no regular exact journal")
    active = ExactEpochJournal(journal_directory).load_active()
    if active is None or active.epoch < 1:
        _fail("existing replay run has no committed epoch")
    if active.epoch > replay_contract.FORMAL_EPOCHS:
        _fail("replay journal exceeds 800 epochs")
    summary_path = directory / "summary.json"
    if summary_path.is_symlink():
        _fail("replay completion summary must not be a symlink")
    if not summary_path.exists():
        return "--exact-resume"
    summary = _load_pretty_object(summary_path, "replay completion summary")
    for name, expected in (
        ("status", "complete"),
        ("variant", inputs.definition.variant),
        ("dataset", replay_contract.DATASET),
        ("seed", replay_contract.TRAJECTORY_SEED),
        ("split_seed", replay_contract.SPLIT_SEED),
        ("run_identity", identity),
    ):
        _require_equal(f"completion summary {name}", summary.get(name), expected)
    if active.epoch != replay_contract.FORMAL_EPOCHS:
        _fail("completion summary exists before epoch 800")
    metrics_path = directory / "metrics.jsonl"
    if metrics_path.is_symlink() or not metrics_path.is_file():
        _fail("completed replay has no regular metrics.jsonl")
    epochs = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, Mapping):
            _fail("replay metrics contains a non-object")
        epochs.append(value.get("epoch"))
    if epochs != list(range(1, replay_contract.FORMAL_EPOCHS + 1)):
        _fail("completed replay metrics are not exactly epochs 1..800")
    for filename in ("last.pth.tar", "best.pth.tar", "best_miou.pth.tar"):
        checkpoint = directory / filename
        if checkpoint.is_symlink() or not checkpoint.is_file():
            _fail(f"completed replay checkpoint is missing: {filename}")
    return "--complete"


def dry_run_payload(inputs: ReplayInputs) -> dict[str, Any]:
    return {
        **inputs.metadata(),
        "status": "DRY_RUN_CONTRACT_VALID",
        "formal_training_started": False,
        "resolved_mode": resolve_initialization_mode(inputs),
        "run_directory": str(run_directory(inputs)),
        "expected_run_id": expected_run_id(inputs),
        "frozen_trainer": str(
            Path(inputs.definition.trainer.__file__).resolve()
        ),
        "default_threshold": replay_contract.DEFAULT_THRESHOLD,
        "official_test_accessed": False,
        "gpu_used": False,
    }


def _option_supplied(arguments: Sequence[str], option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def _parse_args(
    argv: Sequence[str] | None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--replay-contract",
        type=Path,
        default=replay_contract.DEFAULT_CONTRACT,
    )
    parser.add_argument(
        "--child-initialization-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--certification-source-lock",
        type=Path,
        default=source_lock.DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--certification-parent-lock",
        type=Path,
        default=parent_lock.DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--replay-source-lock",
        type=Path,
        default=replay_source_lock.DEFAULT_OUTPUT,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run-contract", action="store_true")
    mode.add_argument("--resolve-mode", action="store_true")
    mode.add_argument("--parent-warm-start", action="store_true")
    mode.add_argument("--exact-resume", action="store_true")
    parsed, remaining = parser.parse_known_args(argv)
    for option in _CONTROLLED_OPTIONS:
        if _option_supplied(remaining, option):
            _fail(f"{option} is controlled by the replay contract")
    if remaining:
        _fail(f"unsupported replay arguments: {remaining}")
    return parsed, remaining


def _frozen_argv(inputs: ReplayInputs, mode: str) -> list[str]:
    return [
        "--variant",
        inputs.definition.variant,
        "--dataset",
        replay_contract.DATASET,
        "--output-root",
        str(inputs.output_root),
        "--run-tag",
        inputs.run_tag,
        "--device",
        "cuda:0",
        "--epochs",
        str(replay_contract.FORMAL_EPOCHS),
        "--seed",
        str(replay_contract.TRAJECTORY_SEED),
        "--split-seed",
        str(replay_contract.SPLIT_SEED),
        "--threshold",
        str(replay_contract.DEFAULT_THRESHOLD),
        "--parent-checkpoint",
        str(inputs.parent_checkpoint_path),
        "--exact-source-lock",
        str(inputs.source_lock_path),
        mode,
    ]


def _require_environment(inputs: ReplayInputs) -> None:
    _require_equal(
        "PYTHONHASHSEED",
        os.environ.get("PYTHONHASHSEED"),
        str(replay_contract.TRAJECTORY_SEED),
    )
    _require_equal(
        "CUDA_VISIBLE_DEVICES",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        inputs.physical_gpu_uuid,
    )
    prefix = (
        "TPD_NER_V4_SURVIVAL"
        if inputs.definition.arm == ARM_B
        else "TPD_NER_V4_QFG"
    )
    _require_equal(
        "physical GPU index environment",
        os.environ.get(f"{prefix}_PHYSICAL_GPU_INDEX"),
        str(inputs.physical_gpu_index),
    )
    _require_equal(
        "physical GPU UUID environment",
        os.environ.get(f"{prefix}_PHYSICAL_GPU_UUID"),
        inputs.physical_gpu_uuid,
    )


def run_arm(
    arm: str,
    argv: Sequence[str] | None = None,
) -> Path | dict[str, Any] | str:
    args, _ = _parse_args(argv)
    inputs = validate_inputs(
        arm=arm,
        contract_path=args.replay_contract,
        initialization_manifest_path=args.child_initialization_manifest,
        certification_source_lock_path=args.certification_source_lock,
        certification_parent_lock_path=args.certification_parent_lock,
        replay_source_lock_path=args.replay_source_lock,
    )
    if args.dry_run_contract:
        return dry_run_payload(inputs)
    if args.resolve_mode:
        return resolve_initialization_mode(inputs)
    requested_mode = (
        "--parent-warm-start" if args.parent_warm_start else "--exact-resume"
    )
    resolved_mode = resolve_initialization_mode(inputs)
    if resolved_mode == "--complete":
        _fail("replay run is already complete")
    _require_equal("requested/resolved initialization mode", requested_mode, resolved_mode)
    _require_environment(inputs)
    with replay_trainer_overlay(inputs) as trainer:
        frozen_args = trainer.parse_args(
            _frozen_argv(inputs, requested_mode)
        )
        return trainer.run_training(frozen_args)


def main_for_arm(
    arm: str,
    argv: Sequence[str] | None = None,
) -> None:
    result = run_arm(arm, argv)
    if isinstance(result, Mapping):
        print(
            json.dumps(
                dict(result),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )
    elif isinstance(result, Path):
        print(f"OUTPUT {result}")
    else:
        print(result)


__all__ = [
    "ACTION_SCHEMA",
    "ARM_B",
    "ARM_D",
    "ReplayInputs",
    "SCHEMA",
    "SOURCE_LOCK_KEY",
    "SUPPORTED_ARMS",
    "Seed42ReplayExactError",
    "dry_run_payload",
    "expected_run_id",
    "main_for_arm",
    "replay_trainer_overlay",
    "require_all_parameters_trainable",
    "resolve_initialization_mode",
    "run_arm",
    "run_directory",
    "validate_inputs",
]
