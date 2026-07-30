#!/usr/bin/env python3
"""Write-once contract and child manifests for a new seed-42 B/D replay.

This successor is deliberately separate from the frozen engineering-v1
replication schedule.  It does not alter or import legacy seed-42 results.
Instead, it authorizes two new full-parameter child trajectories with:

* builder, extension-initialization, parent, and trajectory seed all 42;
* the frozen NUDT-SIRST 530/133 split and default threshold 0.5;
* the same immutable V4 parent for arms B and D;
* independent output directories and run identities;
* seed 3407 retained only as external supplementary evidence;
* seed 426780603 forbidden from this replay.

Importing this module is read-only.  Publication is explicit and write-once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

from experiments import final_model_replication_exact_core as frozen_core
from experiments import final_model_replication_seed_contract as frozen_seeds
from experiments import (
    freeze_final_model_certification_parent_lock as parent_lock,
)
from experiments import (
    freeze_final_model_certification_source_lock as source_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "sctransnet_final_model_seed42_certification_replay_contract_v2"
MANIFEST_SCHEMA = (
    "sctransnet_final_model_seed42_certification_replay_child_manifest_v2"
)
ACTION_SCHEMA = (
    "sctransnet_final_model_seed42_certification_replay_prepare_action_v2"
)

TRAJECTORY_SEED = 42
BUILDER_COMPATIBILITY_SEED = 42
EXTENSION_INITIALIZATION_SEED = 42
PARENT_TRAINING_SEED = 42
SPLIT_SEED = 20260722
DEFAULT_THRESHOLD = 0.5
FORMAL_EPOCHS = 800
DATASET = "NUDT-SIRST"
SUPPLEMENTARY_SEED = 3407
FORBIDDEN_SEED = 426780603

DEFAULT_CONTRACT = (
    REPO_ROOT
    / "experiments/final_model_seed42_certification_replay_contract_v2.json"
)
DEFAULT_MANIFEST_DIRECTORY = (
    REPO_ROOT
    / "experiments/final_model_seed42_certification_replay_manifests_v2"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "experiments/results/final_model_seed42_certification_replay_v1"
)
RUN_TAGS = {
    frozen_core.ARM_B: "seed42_certification_replay_b_formal800",
    frozen_core.ARM_D: "seed42_certification_replay_d_formal800",
}
PHYSICAL_GPU_INDICES = {
    frozen_core.ARM_B: 2,
    frozen_core.ARM_D: 3,
}


class Seed42ReplayContractError(ValueError):
    """A seed-42 replay contract or child manifest differs from its lock."""


def _fail(message: str) -> None:
    raise Seed42ReplayContractError(message)


def canonical_json_bytes(value: Any) -> bytes:
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
        _fail(f"value is not canonical JSON: {exc}")


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


def file_sha256(path: str | os.PathLike[str]) -> str:
    value = _regular_file(path, "file")
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        _fail(f"path is outside the repository: {resolved}")


def _portable_path(path: Path) -> str:
    """Use a repository-relative path formally and an absolute path in tests."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def run_directory(arm: str) -> Path:
    definition = frozen_core.arm_definition(arm)
    return (
        DEFAULT_OUTPUT_ROOT
        / DATASET
        / definition.variant
        / f"seed_{TRAJECTORY_SEED}_{RUN_TAGS[arm]}"
    )


def _arm_record(
    arm: str,
    certification_source_lock: Mapping[str, Any],
) -> dict[str, Any]:
    definition = frozen_core.arm_definition(arm)
    trainer = definition.trainer
    gpu_index = PHYSICAL_GPU_INDICES[arm]
    upstream_locks = certification_source_lock.get("upstream_locks")
    if not isinstance(upstream_locks, Mapping):
        _fail("certification source lock has no upstream_locks")
    upstream_key = {
        frozen_core.ARM_B: "b_formal800",
        frozen_core.ARM_D: "d_formal800",
    }[arm]
    authoritative_lock = upstream_locks.get(upstream_key)
    if not isinstance(authoritative_lock, Mapping):
        _fail(
            "certification source lock has no authoritative "
            f"{upstream_key} binding"
        )
    source_lock_path = (
        REPO_ROOT / str(authoritative_lock.get("path", ""))
    ).resolve()
    locked_sha256 = authoritative_lock.get("sha256")
    if file_sha256(source_lock_path) != locked_sha256:
        _fail(f"authoritative {upstream_key} source lock bytes changed")
    return {
        "arm": arm,
        "variant": definition.variant,
        "trainer": {
            "path": _repo_relative(Path(trainer.__file__)),
            "sha256": file_sha256(Path(trainer.__file__)),
        },
        "upstream_training_source_lock": {
            "path": _repo_relative(source_lock_path),
            "sha256": locked_sha256,
            "schema": authoritative_lock.get("schema"),
        },
        "run_tag": RUN_TAGS[arm],
        "run_directory": _repo_relative(run_directory(arm)),
        "physical_gpu_index": gpu_index,
        "physical_gpu_uuid": trainer.PHYSICAL_GPU_UUIDS[str(gpu_index)],
        "all_child_parameters_trainable": True,
    }


def build_contract(
    *,
    source_lock_path: Path = source_lock.DEFAULT_OUTPUT,
    parent_lock_path: Path = parent_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Build the complete live replay contract without writing it."""

    certification_source_lock = source_lock.verify_source_lock(
        source_lock_path,
        repo_root=REPO_ROOT,
    )
    parent_payload = parent_lock.verify_parent_lock(
        parent_lock_path,
        repo_root=REPO_ROOT,
    )
    selected = parent_payload.get("selected_model")
    if not isinstance(selected, Mapping):
        _fail("certification parent lock has no selected_model")
    initialization_parent = selected.get("initialization_parent")
    if not isinstance(initialization_parent, Mapping):
        _fail("certification parent lock has no initialization parent")

    b_trainer = frozen_core.arm_definition(frozen_core.ARM_B).trainer
    expected_parent_path = Path(b_trainer.PARENT_CHECKPOINT_PATH).resolve()
    locked_parent_path = (
        REPO_ROOT / str(initialization_parent.get("path", ""))
    ).resolve()
    if locked_parent_path != expected_parent_path:
        _fail("certification parent path differs from frozen B/D trainer")
    expected_parent = {
        "path": _repo_relative(expected_parent_path),
        "sha256": b_trainer.PARENT_CHECKPOINT_SHA256,
        "state_dict_sha256": b_trainer.PARENT_STATE_DICT_SHA256,
        "epoch": b_trainer.PARENT_CHECKPOINT_EPOCH,
        "role": "best_validation_miou_secondary",
        "seed": PARENT_TRAINING_SEED,
    }
    observed_parent = {
        "path": str(initialization_parent.get("path")),
        "sha256": initialization_parent.get("sha256"),
        "state_dict_sha256": initialization_parent.get(
            "state_dict_sha256"
        ),
        "epoch": initialization_parent.get("epoch"),
        "role": initialization_parent.get("role"),
    }
    if observed_parent != {
        key: value for key, value in expected_parent.items() if key != "seed"
    }:
        _fail("certification parent identity differs from replay contract")
    if file_sha256(expected_parent_path) != expected_parent["sha256"]:
        _fail("certification parent checkpoint bytes changed")

    return {
        "schema": SCHEMA,
        "status": "locked",
        "scope": "new_fixed_seed42_b_d_certification_replay",
        "frozen_model": {
            "mainline": "SCTransNet+TPD8+five-node-NER4+QFG2-CROA",
            "mainline_changed": False,
            "innovation_changed": False,
            "default_threshold": DEFAULT_THRESHOLD,
            "deployment_weights_changed": False,
        },
        "seed_roles": {
            "trajectory_seed": TRAJECTORY_SEED,
            "builder_compatibility_seed": BUILDER_COMPATIBILITY_SEED,
            "extension_initialization_seed": EXTENSION_INITIALIZATION_SEED,
            "parent_training_seed": PARENT_TRAINING_SEED,
            "split_seed": SPLIT_SEED,
            "supplementary_seed_not_in_primary_gate": SUPPLEMENTARY_SEED,
            "forbidden_not_scheduled_seed": FORBIDDEN_SEED,
        },
        "data_and_metric_contract": {
            "dataset": DATASET,
            "train_count": parent_lock.TRAIN_COUNT,
            "validation_count": parent_lock.VALIDATION_COUNT,
            "train_ids_sha256": parent_lock.TRAIN_IDS_SHA256,
            "validation_ids_sha256": parent_lock.VALIDATION_IDS_SHA256,
            "epochs": FORMAL_EPOCHS,
            "threshold": DEFAULT_THRESHOLD,
            "prediction_comparison": "prediction > threshold",
            "match_radius": 3.0,
            "tiny_area": 9,
            "fa_budgets": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
            "primary_metrics": [
                "pd",
                "fa",
                "miou",
                "tiny_pd",
                "false_objects",
            ],
            "false_objects_definition": (
                "unmatched_predicted_object_count; "
                "false_objects_per_image=false_objects/133"
            ),
            "official_test_accessed": False,
        },
        "initialization": {
            "parent": expected_parent,
            "parent_load_count": 1,
            "child_epoch_starts_at": 1,
            "optimizer_inherited": False,
            "scheduler_inherited": False,
            "fresh_adam": True,
            "all_child_parameters_trainable": True,
        },
        "replay_identity": {
            "output_root": _repo_relative(DEFAULT_OUTPUT_ROOT),
            "independent_from_legacy_seed42_outputs": True,
            "legacy_checkpoints_imported": False,
            "legacy_exact_journal_imported": False,
            "cross_run_exact_resume_forbidden": True,
            "same_run_exact_resume_required": True,
        },
        "arms": [
            _arm_record(frozen_core.ARM_B, certification_source_lock),
            _arm_record(frozen_core.ARM_D, certification_source_lock),
        ],
        "source_bindings": {
            "certification_source_lock": {
                "path": _repo_relative(Path(source_lock_path)),
                "sha256": file_sha256(source_lock_path),
                "source_count": 31,
            },
            "certification_parent_lock": {
                "path": _repo_relative(Path(parent_lock_path)),
                "sha256": file_sha256(parent_lock_path),
            },
        },
        "checkpoint_selection_roles": {
            "trainer_primary": "each_arm_own_best_pd",
            "trainer_secondary": "each_arm_own_best_miou",
        },
        "replay_adjudication_policy": {
            "primary_comparison": "each_arm_own_best_miou",
            "secondary_comparison": "each_arm_own_best_pd",
            "shared_epoch_required": False,
            "fixed_threshold_primary_report": DEFAULT_THRESHOLD,
        },
        "claim_boundary": {
            "new_seed42_replay": True,
            "single_fixed_seed_only": True,
            "paper_stability_supported": False,
            "cross_seed_stability_supported": False,
            "official_test_claim": False,
        },
        "overwrite_forbidden": True,
    }


def build_child_manifest(
    arm: str,
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    source_lock_path: Path = source_lock.DEFAULT_OUTPUT,
    parent_lock_path: Path = parent_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    contract_file = _regular_file(contract_path, "seed-42 replay contract")
    contract = load_contract(
        contract_file,
        source_lock_path=source_lock_path,
        parent_lock_path=parent_lock_path,
    )
    if arm not in frozen_core.SUPPORTED_ARMS:
        _fail(f"unsupported replay arm: {arm!r}")
    arm_record = next(
        record for record in contract["arms"] if record["arm"] == arm
    )
    parent = contract["initialization"]["parent"]
    return {
        "schema": MANIFEST_SCHEMA,
        "arm": arm,
        "variant": arm_record["variant"],
        "trajectory_seed": TRAJECTORY_SEED,
        "builder_compatibility_seed": BUILDER_COMPATIBILITY_SEED,
        "extension_initialization_seed": EXTENSION_INITIALIZATION_SEED,
        "split_seed": SPLIT_SEED,
        "replay_contract": {
            "path": _portable_path(contract_file),
            "sha256": file_sha256(contract_file),
        },
        "certification_source_lock_sha256": contract["source_bindings"][
            "certification_source_lock"
        ]["sha256"],
        "certification_parent_lock_sha256": contract["source_bindings"][
            "certification_parent_lock"
        ]["sha256"],
        "parent": dict(parent),
        "initialization_mode": "extension_parent_warm_start",
        "parent_load_count": 1,
        "optimizer_inherited": False,
        "scheduler_inherited": False,
        "child_epoch_starts_at": 1,
        "all_child_parameters_trainable": True,
        "run_tag": arm_record["run_tag"],
        "run_directory": arm_record["run_directory"],
        "independent_from_legacy_seed42_outputs": True,
        "legacy_checkpoint_imported": False,
        "legacy_exact_journal_imported": False,
    }


def manifest_path(directory: Path, arm: str) -> Path:
    return Path(directory) / f"seed_42_{arm}_certification_replay_init.json"


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    value_path = _regular_file(path, label)
    raw = value_path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot parse {label}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{label} must contain one JSON object")
    if raw != canonical_json_bytes(value):
        _fail(f"{label} is not canonical JSON")
    return value


def load_contract(
    path: Path = DEFAULT_CONTRACT,
    *,
    source_lock_path: Path = source_lock.DEFAULT_OUTPUT,
    parent_lock_path: Path = parent_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    observed = _load_canonical(path, "seed-42 replay contract")
    expected = build_contract(
        source_lock_path=source_lock_path,
        parent_lock_path=parent_lock_path,
    )
    if observed != expected:
        _fail("seed-42 replay contract differs from live frozen inputs")
    return observed


def load_child_manifest(
    path: Path,
    *,
    arm: str,
    contract_path: Path = DEFAULT_CONTRACT,
    source_lock_path: Path = source_lock.DEFAULT_OUTPUT,
    parent_lock_path: Path = parent_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    observed = _load_canonical(path, "seed-42 replay child manifest")
    expected = build_child_manifest(
        arm,
        contract_path=contract_path,
        source_lock_path=source_lock_path,
        parent_lock_path=parent_lock_path,
    )
    if observed != expected:
        _fail("seed-42 replay child manifest differs from its contract")
    return observed


def write_once(path: Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    content = canonical_json_bytes(dict(value))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        _fail(f"refusing to write through symlink: {destination}")
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != content:
            raise FileExistsError(
                f"write-once artifact already differs: {destination}"
            )
        return destination
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if (
                not destination.is_file()
                or destination.is_symlink()
                or destination.read_bytes() != content
            ):
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def prepare(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    manifest_directory: Path = DEFAULT_MANIFEST_DIRECTORY,
    source_lock_path: Path = source_lock.DEFAULT_OUTPUT,
    parent_lock_path: Path = parent_lock.DEFAULT_OUTPUT,
) -> dict[str, Any]:
    contract = build_contract(
        source_lock_path=source_lock_path,
        parent_lock_path=parent_lock_path,
    )
    contract_file = write_once(contract_path, contract)
    load_contract(
        contract_file,
        source_lock_path=source_lock_path,
        parent_lock_path=parent_lock_path,
    )
    manifests = []
    for arm in frozen_core.SUPPORTED_ARMS:
        value = build_child_manifest(
            arm,
            contract_path=contract_file,
            source_lock_path=source_lock_path,
            parent_lock_path=parent_lock_path,
        )
        destination = write_once(
            manifest_path(manifest_directory, arm),
            value,
        )
        load_child_manifest(
            destination,
            arm=arm,
            contract_path=contract_file,
            source_lock_path=source_lock_path,
            parent_lock_path=parent_lock_path,
        )
        manifests.append(
            {
                "arm": arm,
                "path": str(destination.resolve()),
                "sha256": file_sha256(destination),
            }
        )
    return {
        "schema": ACTION_SCHEMA,
        "status": "complete",
        "contract_path": str(contract_file.resolve()),
        "contract_sha256": file_sha256(contract_file),
        "manifests": manifests,
        "trajectory_seed": TRAJECTORY_SEED,
        "run_count": 2,
        "gpu_used": False,
        "training_started": False,
        "official_test_accessed": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--manifest-directory",
        type=Path,
        default=DEFAULT_MANIFEST_DIRECTORY,
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=source_lock.DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--parent-lock",
        type=Path,
        default=parent_lock.DEFAULT_OUTPUT,
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.verify_only:
        contract = load_contract(
            args.contract,
            source_lock_path=args.source_lock,
            parent_lock_path=args.parent_lock,
        )
        manifests = []
        for arm in frozen_core.SUPPORTED_ARMS:
            path = manifest_path(args.manifest_directory, arm)
            load_child_manifest(
                path,
                arm=arm,
                contract_path=args.contract,
                source_lock_path=args.source_lock,
                parent_lock_path=args.parent_lock,
            )
            manifests.append(
                {
                    "arm": arm,
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                }
            )
        result = {
            "schema": ACTION_SCHEMA,
            "status": "verified",
            "contract_path": str(args.contract.resolve()),
            "contract_sha256": file_sha256(args.contract),
            "trajectory_seed": contract["seed_roles"]["trajectory_seed"],
            "manifests": manifests,
            "gpu_used": False,
            "training_started": False,
        }
    else:
        result = prepare(
            contract_path=args.contract,
            manifest_directory=args.manifest_directory,
            source_lock_path=args.source_lock,
            parent_lock_path=args.parent_lock,
        )
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )


__all__ = [
    "ACTION_SCHEMA",
    "BUILDER_COMPATIBILITY_SEED",
    "DATASET",
    "DEFAULT_CONTRACT",
    "DEFAULT_MANIFEST_DIRECTORY",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_THRESHOLD",
    "EXTENSION_INITIALIZATION_SEED",
    "FORBIDDEN_SEED",
    "FORMAL_EPOCHS",
    "MANIFEST_SCHEMA",
    "PHYSICAL_GPU_INDICES",
    "PARENT_TRAINING_SEED",
    "REPO_ROOT",
    "RUN_TAGS",
    "SCHEMA",
    "SPLIT_SEED",
    "SUPPLEMENTARY_SEED",
    "Seed42ReplayContractError",
    "TRAJECTORY_SEED",
    "build_child_manifest",
    "build_contract",
    "canonical_json_bytes",
    "file_sha256",
    "load_child_manifest",
    "load_contract",
    "manifest_path",
    "prepare",
    "run_directory",
    "write_once",
]


if __name__ == "__main__":
    main()
