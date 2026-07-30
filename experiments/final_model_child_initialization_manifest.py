"""Write-once parent-to-child initialization manifests for replication runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from experiments import final_model_replication_seed_contract as seed_contract
except ImportError:  # pragma: no cover - direct script execution
    import final_model_replication_seed_contract as seed_contract


SCHEMA = "sctransnet_final_model_child_initialization_manifest_v1"
SUPPORTED_ARMS = ("b", "d")
PARENT_STATE_DICT_PATH = ("state_dict",)
PARENT_CHECKPOINT_ROLE = "best_miou"
PARENT_VARIANT = (
    "tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on"
)


class ChildInitializationManifestError(ValueError):
    """A child initialization manifest violates the replication protocol."""


def _fail(message: str) -> None:
    raise ChildInitializationManifestError(message)


def _validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_seed(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > 0x7FFFFFFF
    ):
        _fail(f"{label} must be an integer within [1, 2147483647]")
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


def file_sha256(path: str | os.PathLike[str]) -> str:
    value = _regular_file(path, "file")
    digest = hashlib.sha256()
    with value.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


@dataclass(frozen=True)
class ChildInitializationManifest:
    """Bind one B or D child trajectory to one immutable V4 parent."""

    arm: str
    trajectory_seed: int
    seed_contract_sha256: str
    certification_source_lock_sha256: str
    parent_seed: int
    parent_checkpoint_path: str
    parent_checkpoint_sha256: str
    parent_state_dict_sha256: str
    parent_checkpoint_epoch: int
    extension_initialization_seed: int = (
        seed_contract.BUILDER_COMPATIBILITY_SEED
    )
    builder_compatibility_seed: int = (
        seed_contract.BUILDER_COMPATIBILITY_SEED
    )
    split_seed: int = seed_contract.SPLIT_SEED

    def normalized(self, *, verify_parent_bytes: bool = True) -> dict[str, Any]:
        if self.arm not in SUPPORTED_ARMS:
            _fail(f"arm must be one of {SUPPORTED_ARMS}")
        trajectory_seed = _validate_seed(
            self.trajectory_seed,
            "trajectory seed",
        )
        parent_seed = _validate_seed(self.parent_seed, "parent seed")
        builder_seed = _validate_seed(
            self.builder_compatibility_seed,
            "builder compatibility seed",
        )
        extension_seed = _validate_seed(
            self.extension_initialization_seed,
            "extension initialization seed",
        )
        split_seed = _validate_seed(self.split_seed, "split seed")
        if builder_seed != seed_contract.BUILDER_COMPATIBILITY_SEED:
            _fail("builder compatibility seed must remain 42")
        if split_seed != seed_contract.SPLIT_SEED:
            _fail(f"split seed must remain {seed_contract.SPLIT_SEED}")
        seed_contract_sha256 = _validate_sha256(
            self.seed_contract_sha256,
            "seed contract SHA-256",
        )
        source_lock_sha256 = _validate_sha256(
            self.certification_source_lock_sha256,
            "certification source-lock SHA-256",
        )
        parent_sha256 = _validate_sha256(
            self.parent_checkpoint_sha256,
            "parent checkpoint SHA-256",
        )
        parent_state_sha256 = _validate_sha256(
            self.parent_state_dict_sha256,
            "parent state-dict SHA-256",
        )
        if (
            isinstance(self.parent_checkpoint_epoch, bool)
            or not isinstance(self.parent_checkpoint_epoch, int)
            or self.parent_checkpoint_epoch < 1
        ):
            _fail("parent checkpoint epoch must be a positive integer")
        parent_path = Path(self.parent_checkpoint_path)
        if not parent_path.is_absolute():
            _fail("parent checkpoint path must be absolute")
        if verify_parent_bytes:
            parent_path = _regular_file(parent_path, "parent checkpoint")
            actual = file_sha256(parent_path)
            if actual != parent_sha256:
                _fail(
                    "parent checkpoint SHA-256 mismatch: "
                    f"{actual} != {parent_sha256}"
                )
        else:
            parent_path = parent_path.resolve()
        if parent_seed == seed_contract.BUILDER_COMPATIBILITY_SEED:
            initialization_scope = "fixed_parent_child_trajectory"
        elif parent_seed == trajectory_seed:
            initialization_scope = "seed_matched_full_pipeline"
        else:
            _fail(
                "parent seed must be either the frozen seed-42 parent or "
                "equal to the child trajectory seed"
            )
        return {
            "schema": SCHEMA,
            "arm": self.arm,
            "trajectory_seed": trajectory_seed,
            "builder_compatibility_seed": builder_seed,
            "extension_initialization_seed": extension_seed,
            "split_seed": split_seed,
            "seed_contract_sha256": seed_contract_sha256,
            "certification_source_lock_sha256": source_lock_sha256,
            "parent": {
                "variant": PARENT_VARIANT,
                "seed": parent_seed,
                "checkpoint_path": str(parent_path),
                "checkpoint_sha256": parent_sha256,
                "state_dict_path": list(PARENT_STATE_DICT_PATH),
                "state_dict_sha256": parent_state_sha256,
                "checkpoint_role": PARENT_CHECKPOINT_ROLE,
                "checkpoint_epoch": self.parent_checkpoint_epoch,
            },
            "initialization_mode": "extension_parent_warm_start",
            "parent_load_count": 1,
            "optimizer_inherited": False,
            "scheduler_inherited": False,
            "child_epoch_starts_at": 1,
            "all_child_parameters_trainable": True,
            "initialization_scope": initialization_scope,
        }


def parse_manifest(
    value: Any,
    *,
    verify_parent_bytes: bool = True,
) -> ChildInitializationManifest:
    if not isinstance(value, Mapping):
        _fail("child initialization manifest must contain one object")
    parent = value.get("parent")
    if not isinstance(parent, Mapping):
        _fail("child initialization manifest has no parent object")
    manifest = ChildInitializationManifest(
        arm=value.get("arm"),
        trajectory_seed=value.get("trajectory_seed"),
        seed_contract_sha256=value.get("seed_contract_sha256"),
        certification_source_lock_sha256=value.get(
            "certification_source_lock_sha256"
        ),
        parent_seed=parent.get("seed"),
        parent_checkpoint_path=parent.get("checkpoint_path"),
        parent_checkpoint_sha256=parent.get("checkpoint_sha256"),
        parent_state_dict_sha256=parent.get("state_dict_sha256"),
        parent_checkpoint_epoch=parent.get("checkpoint_epoch"),
        extension_initialization_seed=value.get(
            "extension_initialization_seed"
        ),
        builder_compatibility_seed=value.get(
            "builder_compatibility_seed"
        ),
        split_seed=value.get("split_seed"),
    )
    normalized = manifest.normalized(
        verify_parent_bytes=verify_parent_bytes
    )
    if dict(value) != normalized:
        _fail("child initialization manifest fields or values changed")
    return manifest


def load_manifest(
    path: str | os.PathLike[str],
    *,
    verify_parent_bytes: bool = True,
) -> ChildInitializationManifest:
    manifest_path = _regular_file(path, "child initialization manifest")
    raw = manifest_path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot parse child initialization manifest: {exc}")
    manifest = parse_manifest(
        value,
        verify_parent_bytes=verify_parent_bytes,
    )
    if raw != canonical_json_bytes(
        manifest.normalized(verify_parent_bytes=verify_parent_bytes)
    ):
        _fail("child initialization manifest is not canonical JSON")
    return manifest


def write_manifest_once(
    path: str | os.PathLike[str],
    manifest: ChildInitializationManifest,
) -> Path:
    destination = Path(path)
    content = canonical_json_bytes(
        manifest.normalized(verify_parent_bytes=True)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        _fail(f"refusing to write through symlink: {destination}")
    if destination.exists():
        if destination.read_bytes() != content:
            raise FileExistsError(
                "write-once child initialization manifest already differs: "
                f"{destination}"
            )
        load_manifest(destination)
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
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    load_manifest(destination)
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one final-model child initialization manifest"
    )
    parser.add_argument("--arm", choices=SUPPORTED_ARMS, required=True)
    parser.add_argument("--trajectory-seed", type=int, required=True)
    parser.add_argument("--seed-contract", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--parent-seed", type=int, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-state-dict-sha256", required=True)
    parser.add_argument("--parent-checkpoint-epoch", type=int, required=True)
    parser.add_argument(
        "--extension-initialization-seed",
        type=int,
        default=seed_contract.BUILDER_COMPATIBILITY_SEED,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    schedule = seed_contract.load_contract(args.seed_contract)
    schedule_payload = schedule.normalized()
    allowed = set(schedule_payload["engineering_trajectory_seeds"]) | set(
        schedule.trajectory_seeds
    )
    if args.trajectory_seed not in allowed:
        _fail("trajectory seed is absent from the frozen seed contract")
    source_lock_sha256 = seed_contract.file_sha256(args.source_lock)
    if (
        source_lock_sha256
        != schedule.certification_source_lock_sha256
    ):
        _fail("source lock differs from the seed contract")
    manifest = ChildInitializationManifest(
        arm=args.arm,
        trajectory_seed=args.trajectory_seed,
        seed_contract_sha256=seed_contract.file_sha256(args.seed_contract),
        certification_source_lock_sha256=source_lock_sha256,
        parent_seed=args.parent_seed,
        parent_checkpoint_path=str(args.parent_checkpoint.resolve()),
        parent_checkpoint_sha256=file_sha256(args.parent_checkpoint),
        parent_state_dict_sha256=args.parent_state_dict_sha256,
        parent_checkpoint_epoch=args.parent_checkpoint_epoch,
        extension_initialization_seed=args.extension_initialization_seed,
    )
    destination = write_manifest_once(args.output, manifest)
    print(f"LOCKED {destination} sha256={file_sha256(destination)}")


if __name__ == "__main__":
    main()
