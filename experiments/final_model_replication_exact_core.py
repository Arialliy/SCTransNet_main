"""Shared replication adapter for the frozen final-model B and D trainers.

The formal seed-42 entry points remain byte-for-byte untouched.  This module
adds an explicit compatibility layer that:

1. builds the frozen child/reference layouts with builder seed 42;
2. performs exactly one strict extension-parent warm start;
3. resets every training RNG to the registered trajectory seed;
4. lets the already tested exact runner own checkpointing and exact resume.

The adapter currently authorizes the engineering fixed-parent B/D matrix.
Confirmatory full-pipeline runs require a separately locked parameterized
extension initializer and are rejected until that implementation exists.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import random
import stat
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from experiments import final_model_child_initialization_manifest as child_manifest
from experiments import final_model_replication_seed_contract as seed_contract
from experiments import (
    freeze_final_model_certification_parent_lock as parent_lock,
)
from experiments import (
    freeze_final_model_certification_source_lock as certification_source_lock,
)
from experiments import tpd_exact_runner as exact_runner
from experiments import (
    train_tpd_ner_v4_qfg_v2_croa_exact as d_trainer,
)
from experiments import (
    train_tpd_ner_v8_mprs_dch_v4_tail_aware_survival_exact as b_trainer,
)


SCHEMA = "sctransnet_final_model_replication_exact_core_v1"
SOURCE_LOCK_KEY = "final_model_certification_source_lock_v1"
ARM_B = "b"
ARM_D = "d"
SUPPORTED_ARMS = (ARM_B, ARM_D)
ENGINEERING_RUN_TAGS = {
    ARM_B: "final_model_replication_b_formal800",
    ARM_D: "final_model_replication_d_formal800",
}
_CONTROLLED_FROZEN_OPTIONS = (
    "--variant",
    "--dataset",
    "--seed",
    "--split-seed",
    "--run-tag",
    "--parent-checkpoint",
    "--exact-source-lock",
    "--survival-weight",
    "--survival-pos-weight",
)


class ReplicationExactError(ValueError):
    """The requested replication run violates its immutable identity."""


def _fail(message: str) -> None:
    raise ReplicationExactError(message)


def _validate_sha256(value: Any, label: str) -> str:
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


def _json_object(path: Path, label: str) -> dict[str, Any]:
    raw = _regular_file(path, label).read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot parse {label}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must contain one JSON object")
    return value


@dataclass(frozen=True)
class ArmDefinition:
    arm: str
    variant: str
    trainer: ModuleType
    new_module_prefixes: tuple[str, ...]
    zero_init_prefixes: tuple[str, ...]
    validate_loaded_child: Callable[[nn.Module], Any] | None


def arm_definition(arm: str) -> ArmDefinition:
    """Return the exact frozen implementation selected by one paper arm."""

    if arm == ARM_B:
        return ArmDefinition(
            arm=arm,
            variant=b_trainer.TSS_ON_VARIANT,
            trainer=b_trainer,
            new_module_prefixes=tuple(
                b_trainer.SURVIVAL_NEW_MODULE_PREFIXES
            ),
            zero_init_prefixes=tuple(
                b_trainer.SURVIVAL_ZERO_INIT_PREFIXES
            ),
            validate_loaded_child=lambda model: (
                b_trainer.survival_model.validate_formal_survival_model(
                    model,
                    require_zero_initialized_heads=True,
                )
            ),
        )
    if arm == ARM_D:
        return ArmDefinition(
            arm=arm,
            variant=d_trainer.TSS_QFG_VARIANT,
            trainer=d_trainer,
            new_module_prefixes=tuple(
                d_trainer.QFG_NEW_MODULE_PREFIXES
            ),
            zero_init_prefixes=tuple(
                d_trainer.QFG_ZERO_INIT_PREFIXES
            ),
            validate_loaded_child=lambda model: (
                d_trainer.qfg_model
                .validate_formal_qfg_v2_croa_survival_model(
                    model,
                    require_zero_initialized_heads=True,
                    require_identity_initialized_qfg=True,
                )
            ),
        )
    _fail(f"unsupported replication arm {arm!r}")


@dataclass(frozen=True)
class ReplicationInputs:
    """Validated immutable files and identities for one B or D run."""

    definition: ArmDefinition
    trajectory_seed: int
    schedule: seed_contract.ReplicationSeedScheduleContract
    schedule_path: Path
    schedule_sha256: str
    initialization: child_manifest.ChildInitializationManifest
    initialization_path: Path
    initialization_sha256: str
    source_lock_path: Path
    source_lock_sha256: str
    parent_lock_path: Path
    parent_lock_sha256: str
    parent_lock_schema: str
    parent_checkpoint_path: Path
    parent_checkpoint_sha256: str
    parent_state_dict_sha256: str
    parent_seed: int
    parent_epoch: int
    initialization_scope: str

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "arm": self.definition.arm,
            "variant": self.definition.variant,
            "trajectory_seed": self.trajectory_seed,
            "builder_compatibility_seed": (
                self.schedule.builder_compatibility_seed
            ),
            "extension_initialization_seed": (
                self.initialization.extension_initialization_seed
            ),
            "split_seed": self.schedule.split_seed,
            "seed_contract_path": str(self.schedule_path),
            "seed_contract_sha256": self.schedule_sha256,
            "certification_source_lock_path": str(self.source_lock_path),
            "certification_source_lock_sha256": self.source_lock_sha256,
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
        }


def validate_replication_inputs(
    *,
    arm: str,
    trajectory_seed: int,
    schedule_path: str | os.PathLike[str],
    initialization_manifest_path: str | os.PathLike[str],
    certification_source_lock_path: str | os.PathLike[str],
    certification_parent_lock_path: str | os.PathLike[str] = (
        parent_lock.DEFAULT_OUTPUT
    ),
    verify_parent_lock_live: bool = True,
    verify_source_lock_live: bool = True,
) -> ReplicationInputs:
    """Cross-check the three write-once inputs before any model construction."""

    definition = arm_definition(arm)
    schedule_file = _regular_file(schedule_path, "replication seed contract")
    initialization_file = _regular_file(
        initialization_manifest_path,
        "child initialization manifest",
    )
    source_lock_file = _regular_file(
        certification_source_lock_path,
        "certification source lock",
    )
    parent_lock_file = _regular_file(
        certification_parent_lock_path,
        "certification parent lock",
    )
    schedule = seed_contract.load_contract(schedule_file)
    initialization = child_manifest.load_manifest(initialization_file)
    schedule_sha256 = seed_contract.file_sha256(schedule_file)
    initialization_sha256 = seed_contract.file_sha256(initialization_file)
    source_lock_sha256 = seed_contract.file_sha256(source_lock_file)
    parent_lock_sha256 = seed_contract.file_sha256(parent_lock_file)
    if verify_source_lock_live:
        certification_source_lock.verify_source_lock(
            source_lock_file,
            repo_root=certification_source_lock.REPO_ROOT,
        )
    else:
        # Unit fixtures still require a real JSON object; they cannot use an
        # arbitrary digest-bearing byte file.
        _json_object(source_lock_file, "certification source lock")
    if verify_parent_lock_live:
        parent_lock_payload = parent_lock.verify_parent_lock(
            parent_lock_file,
            repo_root=parent_lock.REPO_ROOT,
        )
    else:
        parent_lock_payload = _json_object(
            parent_lock_file,
            "certification parent lock",
        )
    if parent_lock_payload.get("schema") != parent_lock.LOCK_SCHEMA:
        _fail("certification parent-lock schema mismatch")
    selected_model = parent_lock_payload.get("selected_model")
    if not isinstance(selected_model, Mapping):
        _fail("certification parent lock has no selected_model")
    locked_parent = selected_model.get("initialization_parent")
    if not isinstance(locked_parent, Mapping):
        _fail("certification parent lock has no initialization parent")

    if (
        isinstance(trajectory_seed, bool)
        or not isinstance(trajectory_seed, int)
        or trajectory_seed < 1
        or trajectory_seed > 0x7FFFFFFF
    ):
        _fail("trajectory seed must be within [1, 2147483647]")
    if initialization.arm != arm:
        _fail("child initialization manifest arm mismatch")
    if initialization.trajectory_seed != trajectory_seed:
        _fail("child initialization manifest trajectory seed mismatch")
    if initialization.seed_contract_sha256 != schedule_sha256:
        _fail("child initialization manifest seed-contract SHA mismatch")
    if (
        schedule.certification_source_lock_sha256
        != source_lock_sha256
    ):
        _fail("seed contract source-lock SHA mismatch")
    if (
        initialization.certification_source_lock_sha256
        != source_lock_sha256
    ):
        _fail("child initialization manifest source-lock SHA mismatch")
    allowed = set(seed_contract.ENGINEERING_TRAJECTORY_SEEDS) | set(
        schedule.trajectory_seeds
    )
    if trajectory_seed not in allowed:
        _fail("trajectory seed is absent from the frozen schedule")
    manifest_payload = initialization.normalized()
    scope = str(manifest_payload["initialization_scope"])
    if (
        trajectory_seed in seed_contract.ENGINEERING_TRAJECTORY_SEEDS
        and scope != "fixed_parent_child_trajectory"
    ):
        _fail("engineering B/D screening requires the frozen seed-42 parent")
    if initialization.builder_compatibility_seed != (
        seed_contract.BUILDER_COMPATIBILITY_SEED
    ):
        _fail("builder compatibility seed differs from 42")
    if initialization.extension_initialization_seed != (
        seed_contract.BUILDER_COMPATIBILITY_SEED
    ):
        _fail(
            "this v1 adapter supports fixed extension initialization seed 42 "
            "only; parameterized confirmatory initialization is not locked"
        )
    if trajectory_seed in schedule.trajectory_seeds:
        _fail(
            "confirmatory seeds require the separately locked parameterized "
            "extension initializer; this engineering runner cannot consume "
            "them"
        )
    parent = manifest_payload["parent"]
    locked_parent_path = (
        parent_lock.REPO_ROOT / str(locked_parent.get("path", ""))
    ).resolve()
    expected_locked_parent = {
        "path": Path(parent["checkpoint_path"]).resolve(),
        "sha256": parent["checkpoint_sha256"],
        "state_dict_sha256": parent["state_dict_sha256"],
        "epoch": parent["checkpoint_epoch"],
        "role": "best_validation_miou_secondary",
        "copied_shared_state_key_count": 544,
        "child_trainable_scope": "all_model_parameters",
        "parent_optimizer_inherited": False,
        "usage": "initialization_only",
    }
    observed_locked_parent = {
        "path": locked_parent_path,
        "sha256": locked_parent.get("sha256"),
        "state_dict_sha256": locked_parent.get("state_dict_sha256"),
        "epoch": locked_parent.get("epoch"),
        "role": locked_parent.get("role"),
        "copied_shared_state_key_count": locked_parent.get(
            "copied_shared_state_key_count"
        ),
        "child_trainable_scope": locked_parent.get(
            "child_trainable_scope"
        ),
        "parent_optimizer_inherited": locked_parent.get(
            "parent_optimizer_inherited"
        ),
        "usage": locked_parent.get("usage"),
    }
    if observed_locked_parent != expected_locked_parent:
        _fail(
            "child initialization parent differs from the certification "
            "parent lock"
        )
    return ReplicationInputs(
        definition=definition,
        trajectory_seed=trajectory_seed,
        schedule=schedule,
        schedule_path=schedule_file,
        schedule_sha256=schedule_sha256,
        initialization=initialization,
        initialization_path=initialization_file,
        initialization_sha256=initialization_sha256,
        source_lock_path=source_lock_file,
        source_lock_sha256=source_lock_sha256,
        parent_lock_path=parent_lock_file,
        parent_lock_sha256=parent_lock_sha256,
        parent_lock_schema=parent_lock.LOCK_SCHEMA,
        parent_checkpoint_path=Path(parent["checkpoint_path"]),
        parent_checkpoint_sha256=str(parent["checkpoint_sha256"]),
        parent_state_dict_sha256=str(parent["state_dict_sha256"]),
        parent_seed=int(parent["seed"]),
        parent_epoch=int(parent["checkpoint_epoch"]),
        initialization_scope=scope,
    )


def verify_parent_checkpoint_payload(inputs: ReplicationInputs) -> None:
    """Verify file, state content and identity fields without old seed-42 gates."""

    path = _regular_file(
        inputs.parent_checkpoint_path,
        "parent checkpoint",
    )
    actual_file_sha256 = seed_contract.file_sha256(path)
    if actual_file_sha256 != inputs.parent_checkpoint_sha256:
        _fail("parent checkpoint bytes differ from initialization manifest")
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        _fail(f"cannot load parent checkpoint {path}: {exc}")
    if not isinstance(payload, Mapping):
        _fail("parent checkpoint must contain one mapping")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        _fail("parent checkpoint has no state_dict mapping")
    state_sha256 = exact_runner._state_content_sha256(
        state,
        "replication parent state_dict",
    )
    if state_sha256 != inputs.parent_state_dict_sha256:
        _fail("parent state-dict content differs from initialization manifest")
    expected_fields = {
        "epoch": inputs.parent_epoch,
        "checkpoint_role": "best_validation_miou_secondary",
        "variant": child_manifest.PARENT_VARIANT,
        "dataset": "NUDT-SIRST",
        "seed": inputs.parent_seed,
        "split_seed": seed_contract.SPLIT_SEED,
    }
    for name, expected in expected_fields.items():
        if payload.get(name) != expected:
            _fail(
                f"parent checkpoint {name} differs: "
                f"{payload.get(name)!r} != {expected!r}"
            )


def reset_trajectory_rng(seed: int) -> torch.Generator:
    """Reset Python/NumPy/Torch streams and return the loader generator."""

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 1
        or seed > 0x7FFFFFFF
    ):
        _fail("trajectory seed must be within [1, 2147483647]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _build_child_with_compatibility_seed(
    inputs: ReplicationInputs,
    *,
    original_builder: Callable[..., tuple[nn.Module, Mapping[str, Any]]],
    eps: float,
) -> tuple[nn.Module, dict[str, Any]]:
    trainer = inputs.definition.trainer
    trajectory_seed = inputs.trajectory_seed
    previous_seed = trainer.TRAINING_SEED
    trainer.TRAINING_SEED = seed_contract.BUILDER_COMPATIBILITY_SEED
    try:
        model, raw_metadata = original_builder(
            inputs.definition.variant,
            seed_contract.BUILDER_COMPATIBILITY_SEED,
            eps=eps,
        )
    finally:
        trainer.TRAINING_SEED = previous_seed
    if not isinstance(raw_metadata, Mapping):
        _fail("frozen builder returned invalid metadata")
    metadata = copy.deepcopy(dict(raw_metadata))
    metadata.update(
        {
            "training_seed": trajectory_seed,
            "parent_checkpoint_path": str(
                inputs.parent_checkpoint_path.resolve()
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


def prepare_extension_parent_once(
    inputs: ReplicationInputs,
    model: nn.Module,
) -> Any:
    """Perform the only parent load permitted for a new child trajectory."""

    verify_parent_checkpoint_payload(inputs)
    definition = inputs.definition
    trainer = definition.trainer
    parent_model, _ = trainer.survival_model.build_formal_v4_reference(
        seed=seed_contract.BUILDER_COMPATIBILITY_SEED
    )
    result = trainer.load_parent_into_extension(
        parent_checkpoint=inputs.parent_checkpoint_path,
        parent_model=parent_model,
        extension_model=model,
        new_module_prefixes=definition.new_module_prefixes,
        zero_init_prefixes=definition.zero_init_prefixes,
        parent_state_dict_path=child_manifest.PARENT_STATE_DICT_PATH,
        expected_parent_checkpoint_sha256=(
            inputs.parent_checkpoint_sha256
        ),
        map_location="cpu",
    )
    if definition.validate_loaded_child is not None:
        definition.validate_loaded_child(model)
    loaded_sha256 = exact_runner.initial_model_state_sha256(model)
    request = exact_runner.InitializationRequest.extension_parent(
        result.provenance(),
        loaded_child_model_state_sha256=loaded_sha256,
    )
    contract = request.initialization_contract()
    if contract is None:
        _fail("extension parent request returned no initialization contract")
    return trainer.InitializationPlan(
        request=request,
        contract=contract,
        initial_model_state_sha256=loaded_sha256,
    )


@contextlib.contextmanager
def _temporary_module_attributes(
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
def replication_trainer_overlay(
    inputs: ReplicationInputs,
) -> Iterator[ModuleType]:
    """Parameterize one frozen trainer for one registered trajectory."""

    trainer = inputs.definition.trainer
    original_builder = trainer.build_selected_model
    original_initialization_plan = trainer.initialization_plan
    original_statistics_loader = trainer.load_survival_target_statistics

    def builder(
        variant: str,
        seed: int,
        *,
        eps: float,
    ) -> tuple[nn.Module, dict[str, Any]]:
        if variant != inputs.definition.variant:
            _fail("replication builder arm/variant mismatch")
        if seed != inputs.trajectory_seed:
            _fail("replication builder received the wrong trajectory seed")
        return _build_child_with_compatibility_seed(
            inputs,
            original_builder=original_builder,
            eps=eps,
        )

    def initialization_plan(
        args: argparse.Namespace,
        directory: Path,
        model: nn.Module,
    ) -> Any:
        if args.parent_warm_start:
            return prepare_extension_parent_once(inputs, model)
        if args.exact_resume:
            # No parent load occurs here.  The exact runner restores the same
            # child trajectory after identity validation.
            return original_initialization_plan(args, directory, model)
        _fail("replication initialization mode is missing")

    def load_survival_target_statistics(
        path: Path = trainer.DEFAULT_TARGET_STATISTICS_PATH,
    ) -> dict[str, Any]:
        # The class-balance file is a frozen split-derived artifact produced
        # during seed-42 development.  It is reused unchanged across child
        # trajectory seeds and must not be relabelled as freshly estimated.
        previous_seed = trainer.TRAINING_SEED
        trainer.TRAINING_SEED = seed_contract.BUILDER_COMPATIBILITY_SEED
        try:
            return original_statistics_loader(path)
        finally:
            trainer.TRAINING_SEED = previous_seed

    def source_lock_contract(
        training_data_sha256: str,
        exact_source_lock_path: str | os.PathLike[str],
        statistics_path: str | os.PathLike[str],
    ) -> dict[str, str]:
        requested = _regular_file(
            exact_source_lock_path,
            "requested certification source lock",
        )
        if requested != inputs.source_lock_path:
            _fail("trainer received a different certification source lock")
        lock_sha256 = seed_contract.file_sha256(requested)
        if lock_sha256 != inputs.source_lock_sha256:
            _fail("certification source lock changed before training")
        statistics_sha256 = seed_contract.file_sha256(statistics_path)
        return {
            SOURCE_LOCK_KEY: lock_sha256,
            "training_data": _validate_sha256(
                training_data_sha256,
                "training-data SHA-256",
            ),
            "survival_target_statistics": statistics_sha256,
            "parent_checkpoint": inputs.parent_checkpoint_sha256,
        }

    overlay = {
        "TRAINING_SEED": inputs.trajectory_seed,
        "SPLIT_SEED": seed_contract.SPLIT_SEED,
        "SOURCE_LOCK_KEY": SOURCE_LOCK_KEY,
        "PARENT_CHECKPOINT_PATH": inputs.parent_checkpoint_path,
        "PARENT_CHECKPOINT_SHA256": inputs.parent_checkpoint_sha256,
        "PARENT_STATE_DICT_SHA256": inputs.parent_state_dict_sha256,
        "PARENT_CHECKPOINT_EPOCH": inputs.parent_epoch,
        "FORMAL_RUN_TAGS": {
            **dict(trainer.FORMAL_RUN_TAGS),
            inputs.definition.variant: ENGINEERING_RUN_TAGS[
                inputs.definition.arm
            ],
        },
        "build_selected_model": builder,
        "initialization_plan": initialization_plan,
        "load_survival_target_statistics": load_survival_target_statistics,
        "source_lock_contract": source_lock_contract,
    }
    with _temporary_module_attributes(trainer, overlay):
        yield trainer


def _option_was_supplied(arguments: Sequence[str], option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def _parse_replication_args(
    argv: Sequence[str] | None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--trajectory-seed", type=int, required=True)
    parser.add_argument("--seed-contract", type=Path, required=True)
    parser.add_argument(
        "--child-initialization-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--certification-source-lock",
        type=Path,
        required=True,
    )
    parser.add_argument("--dry-run-contract", action="store_true")
    parsed, remaining = parser.parse_known_args(argv)
    for option in _CONTROLLED_FROZEN_OPTIONS:
        if _option_was_supplied(remaining, option):
            _fail(f"{option} is controlled by the replication contract")
    return parsed, remaining


def _frozen_argv(
    inputs: ReplicationInputs,
    remaining: Sequence[str],
) -> list[str]:
    return [
        "--variant",
        inputs.definition.variant,
        "--dataset",
        "NUDT-SIRST",
        "--seed",
        str(inputs.trajectory_seed),
        "--split-seed",
        str(seed_contract.SPLIT_SEED),
        "--run-tag",
        ENGINEERING_RUN_TAGS[inputs.definition.arm],
        "--parent-checkpoint",
        str(inputs.parent_checkpoint_path),
        "--exact-source-lock",
        str(inputs.source_lock_path),
        *remaining,
    ]


def _require_process_seed_environment(
    inputs: ReplicationInputs,
    *,
    require_cuda: bool,
) -> None:
    trajectory_seed = inputs.trajectory_seed
    if os.environ.get("PYTHONHASHSEED") != str(trajectory_seed):
        _fail(
            "PYTHONHASHSEED must equal the trajectory seed before Python "
            "starts"
        )
    if require_cuda:
        expected_index = {
            ARM_B: "2",
            ARM_D: "3",
        }[inputs.definition.arm]
        expected_uuid = inputs.definition.trainer.PHYSICAL_GPU_UUIDS[
            expected_index
        ]
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible != expected_uuid:
            _fail(
                "formal replication CUDA_VISIBLE_DEVICES differs from the "
                "registered physical GPU UUID"
            )
        environment_prefix = (
            "TPD_NER_V4_SURVIVAL"
            if inputs.definition.arm == ARM_B
            else "TPD_NER_V4_QFG"
        )
        if os.environ.get(
            f"{environment_prefix}_PHYSICAL_GPU_INDEX"
        ) != expected_index:
            _fail("formal replication physical GPU index differs")
        if os.environ.get(
            f"{environment_prefix}_PHYSICAL_GPU_UUID"
        ) != expected_uuid:
            _fail("formal replication physical GPU UUID differs")


def dry_run_payload(inputs: ReplicationInputs) -> dict[str, Any]:
    """Return a read-only launch plan without model construction or CUDA."""

    verify_parent_checkpoint_payload(inputs)
    return {
        **inputs.metadata(),
        "status": "DRY_RUN_CONTRACT_VALID",
        "formal_training_started": False,
        "frozen_trainer": str(
            Path(inputs.definition.trainer.__file__).resolve()
        ),
        "run_tag": ENGINEERING_RUN_TAGS[inputs.definition.arm],
        "initialization_order": [
            "build_child_layout_with_builder_compatibility_seed_42",
            "strict_extension_parent_warm_start_exactly_once",
            "reset_python_numpy_torch_rng_to_trajectory_seed",
            "create_loader_generator_with_trajectory_seed",
            "create_fresh_adam",
            "create_exact_run_spec_with_trajectory_seed",
        ],
    }


def run_arm(
    arm: str,
    argv: Sequence[str] | None = None,
) -> Path | dict[str, Any]:
    """Validate, overlay and execute one engineering B or D trajectory."""

    replication_args, remaining = _parse_replication_args(argv)
    inputs = validate_replication_inputs(
        arm=arm,
        trajectory_seed=replication_args.trajectory_seed,
        schedule_path=replication_args.seed_contract,
        initialization_manifest_path=(
            replication_args.child_initialization_manifest
        ),
        certification_source_lock_path=(
            replication_args.certification_source_lock
        ),
    )
    if replication_args.dry_run_contract:
        return dry_run_payload(inputs)
    require_cuda = "--allow-cpu-smoke" not in remaining
    _require_process_seed_environment(
        inputs,
        require_cuda=require_cuda,
    )
    with replication_trainer_overlay(inputs) as trainer:
        args = trainer.parse_args(_frozen_argv(inputs, remaining))
        # run_training performs the post-initialization trajectory RNG reset,
        # creates a fresh Adam and delegates exact resume to tpd_exact_runner.
        return trainer.run_training(args)


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
            )
        )
    else:
        print(f"OUTPUT {result}")


__all__ = [
    "ARM_B",
    "ARM_D",
    "ENGINEERING_RUN_TAGS",
    "ReplicationExactError",
    "ReplicationInputs",
    "arm_definition",
    "dry_run_payload",
    "main_for_arm",
    "prepare_extension_parent_once",
    "replication_trainer_overlay",
    "reset_trajectory_rng",
    "run_arm",
    "validate_replication_inputs",
    "verify_parent_checkpoint_payload",
]
