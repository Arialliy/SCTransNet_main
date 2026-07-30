"""Write-once seed schedule for final-model replication.

The schedule deliberately separates four concepts:

* the frozen builder compatibility seed (42);
* the immutable train/validation split seed (20260722);
* the two engineering replication seeds, which are not confirmatory evidence;
* fresh confirmatory trajectory seeds derived before training from the
  certification source-lock digest.

This module never imports or mutates a frozen formal800 trainer.
"""

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


SCHEMA = "sctransnet_final_model_replication_seed_contract_v1"
BUILDER_COMPATIBILITY_SEED = 42
SPLIT_SEED = 20260722
HISTORICAL_PRESSURE_SEED = 3407
DEPLOYMENT_INFERENCE_ARTIFACT_SHA256 = (
    "997027bb2cc59e0e16ef85beba2c78ab8b3e195de962acbe7c97adc8c007c63a"
)
DEPLOYMENT_HASH_SEED = 426780603
ENGINEERING_TRAJECTORY_SEEDS = (
    HISTORICAL_PRESSURE_SEED,
    DEPLOYMENT_HASH_SEED,
)
CONFIRMATORY_EXCLUDED_SEEDS = (
    0,
    BUILDER_COMPATIBILITY_SEED,
    HISTORICAL_PRESSURE_SEED,
    DEPLOYMENT_HASH_SEED,
)
DEFAULT_CONFIRMATORY_SEED_COUNT = 5
_SHA256_HEX_LENGTH = 64
_MAX_SEED = 0x7FFFFFFF


class ReplicationSeedContractError(ValueError):
    """A replication seed schedule violates its frozen contract."""


def _fail(message: str) -> None:
    raise ReplicationSeedContractError(message)


def _validate_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_HEX_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_seed(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{label} must be an integer")
    lower = 0 if allow_zero else 1
    if value < lower or value > _MAX_SEED:
        _fail(f"{label} must be within [{lower}, {_MAX_SEED}]")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's deterministic JSON representation."""

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


def file_sha256(path: str | os.PathLike[str]) -> str:
    """Hash one regular, non-symlink file."""

    resolved = Path(path)
    if resolved.is_symlink():
        _fail(f"file must not be a symlink: {resolved}")
    try:
        metadata = resolved.stat()
    except FileNotFoundError:
        _fail(f"file is missing: {resolved}")
    if not stat.S_ISREG(metadata.st_mode):
        _fail(f"path is not a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_seed_from_deployment_artifact(artifact_sha256: str) -> int:
    """Derive the pre-registered engineering hash seed from deployment bytes."""

    digest = _validate_sha256(
        artifact_sha256,
        "deployment inference artifact SHA-256",
    )
    seed = int(digest[:8], 16) & _MAX_SEED
    if seed == 0:
        _fail("deployment artifact produced the forbidden zero seed")
    return seed


def _candidate_blocks(digest: str) -> tuple[int, ...]:
    return tuple(
        int(digest[offset : offset + 8], 16) & _MAX_SEED
        for offset in range(0, _SHA256_HEX_LENGTH, 8)
    )


def derive_confirmatory_seeds(
    certification_source_lock_sha256: str,
    *,
    count: int = DEFAULT_CONFIRMATORY_SEED_COUNT,
    excluded: Sequence[int] = CONFIRMATORY_EXCLUDED_SEEDS,
) -> tuple[int, ...]:
    """Derive unique, result-unknown seeds solely from a source-lock digest.

    The original digest is consumed in ordered eight-hex-character blocks.
    If it does not yield enough permitted unique seeds, SHA-256 digests of
    ``source_lock_sha256 + decimal_counter`` are consumed in the same way.
    """

    source_digest = _validate_sha256(
        certification_source_lock_sha256,
        "certification source-lock SHA-256",
    )
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        _fail("confirmatory seed count must be a positive integer")
    if isinstance(excluded, (str, bytes)) or not isinstance(excluded, Sequence):
        _fail("excluded seeds must be a sequence")
    excluded_set = {
        _validate_seed(value, f"excluded seed {index}", allow_zero=True)
        for index, value in enumerate(excluded)
    }
    selected: list[int] = []

    def consume(digest: str) -> None:
        for candidate in _candidate_blocks(digest):
            if (
                candidate in excluded_set
                or candidate in selected
                or candidate == 0
            ):
                continue
            selected.append(candidate)
            if len(selected) == count:
                return

    consume(source_digest)
    counter = 0
    while len(selected) < count:
        continuation = hashlib.sha256(
            f"{source_digest}{counter}".encode("ascii")
        ).hexdigest()
        consume(continuation)
        counter += 1
        if counter > count + 1024:
            _fail("unable to derive enough unique confirmatory seeds")
    return tuple(selected)


@dataclass(frozen=True)
class ReplicationSeedScheduleContract:
    """Immutable schedule consumed by all final-model replication runners."""

    certification_source_lock_sha256: str
    trajectory_seeds: tuple[int, ...]
    builder_compatibility_seed: int = BUILDER_COMPATIBILITY_SEED
    split_seed: int = SPLIT_SEED

    @classmethod
    def derive(
        cls,
        certification_source_lock_sha256: str,
        *,
        confirmatory_seed_count: int = DEFAULT_CONFIRMATORY_SEED_COUNT,
    ) -> "ReplicationSeedScheduleContract":
        digest = _validate_sha256(
            certification_source_lock_sha256,
            "certification source-lock SHA-256",
        )
        return cls(
            certification_source_lock_sha256=digest,
            trajectory_seeds=derive_confirmatory_seeds(
                digest,
                count=confirmatory_seed_count,
            ),
        )

    def normalized(self) -> dict[str, Any]:
        source_digest = _validate_sha256(
            self.certification_source_lock_sha256,
            "certification source-lock SHA-256",
        )
        builder_seed = _validate_seed(
            self.builder_compatibility_seed,
            "builder compatibility seed",
        )
        split_seed = _validate_seed(self.split_seed, "split seed")
        if builder_seed != BUILDER_COMPATIBILITY_SEED:
            _fail(
                "builder compatibility seed must remain "
                f"{BUILDER_COMPATIBILITY_SEED}"
            )
        if split_seed != SPLIT_SEED:
            _fail(f"split seed must remain {SPLIT_SEED}")
        if (
            isinstance(self.trajectory_seeds, (str, bytes))
            or not isinstance(self.trajectory_seeds, tuple)
            or not self.trajectory_seeds
        ):
            _fail("trajectory_seeds must be a non-empty tuple")
        seeds = tuple(
            _validate_seed(seed, f"trajectory seed {index}")
            for index, seed in enumerate(self.trajectory_seeds)
        )
        if len(set(seeds)) != len(seeds):
            _fail("trajectory_seeds contains duplicates")
        forbidden = set(CONFIRMATORY_EXCLUDED_SEEDS)
        overlap = sorted(set(seeds) & forbidden)
        if overlap:
            _fail(f"confirmatory trajectory seeds include exclusions: {overlap}")
        expected = derive_confirmatory_seeds(
            source_digest,
            count=len(seeds),
        )
        if seeds != expected:
            _fail(
                "trajectory_seeds are not the deterministic sequence derived "
                "from the certification source lock"
            )
        if (
            hash_seed_from_deployment_artifact(
                DEPLOYMENT_INFERENCE_ARTIFACT_SHA256
            )
            != DEPLOYMENT_HASH_SEED
        ):
            _fail("deployment hash-seed constant is inconsistent")
        return {
            "schema": SCHEMA,
            "certification_source_lock_sha256": source_digest,
            "builder_compatibility_seed": builder_seed,
            "split_seed": split_seed,
            "engineering_trajectory_seeds": list(
                ENGINEERING_TRAJECTORY_SEEDS
            ),
            "engineering_seed_roles": {
                str(HISTORICAL_PRESSURE_SEED): "historical_pressure_only",
                str(DEPLOYMENT_HASH_SEED): (
                    "deployment_artifact_hash_replication"
                ),
            },
            "confirmatory_trajectory_seeds": list(seeds),
            "confirmatory_seed_count": len(seeds),
            "confirmatory_excluded_seeds": list(
                CONFIRMATORY_EXCLUDED_SEEDS
            ),
            "confirmatory_derivation": {
                "input": "certification_source_lock_sha256",
                "block_width_hex_characters": 8,
                "mask": "0x7fffffff",
                "continuation": (
                    "sha256(source_lock_sha256 + decimal_counter)"
                ),
            },
            "seed_42_counts_toward_stability": False,
            "seed_3407_counts_toward_stability": False,
            "deployment_hash_seed_source_sha256": (
                DEPLOYMENT_INFERENCE_ARTIFACT_SHA256
            ),
            "deployment_hash_seed": DEPLOYMENT_HASH_SEED,
        }


def parse_contract(value: Any) -> ReplicationSeedScheduleContract:
    if not isinstance(value, Mapping):
        _fail("seed contract must contain one JSON object")
    required = {
        "schema",
        "certification_source_lock_sha256",
        "builder_compatibility_seed",
        "split_seed",
        "engineering_trajectory_seeds",
        "engineering_seed_roles",
        "confirmatory_trajectory_seeds",
        "confirmatory_seed_count",
        "confirmatory_excluded_seeds",
        "confirmatory_derivation",
        "seed_42_counts_toward_stability",
        "seed_3407_counts_toward_stability",
        "deployment_hash_seed_source_sha256",
        "deployment_hash_seed",
    }
    if set(value) != required:
        _fail("seed contract fields differ from the v1 schema")
    if value.get("schema") != SCHEMA:
        _fail("seed contract schema mismatch")
    seeds_value = value.get("confirmatory_trajectory_seeds")
    if not isinstance(seeds_value, list):
        _fail("confirmatory_trajectory_seeds must be a list")
    contract = ReplicationSeedScheduleContract(
        certification_source_lock_sha256=value.get(
            "certification_source_lock_sha256"
        ),
        trajectory_seeds=tuple(seeds_value),
        builder_compatibility_seed=value.get("builder_compatibility_seed"),
        split_seed=value.get("split_seed"),
    )
    normalized = contract.normalized()
    if dict(value) != normalized:
        _fail("seed contract contains non-canonical or altered values")
    return contract


def load_contract(
    path: str | os.PathLike[str],
) -> ReplicationSeedScheduleContract:
    contract_path = Path(path)
    raw = contract_path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"cannot parse seed contract {contract_path}: {exc}")
    contract = parse_contract(value)
    if raw != canonical_json_bytes(contract.normalized()):
        _fail(f"seed contract is not canonical JSON: {contract_path}")
    return contract


def write_contract_once(
    path: str | os.PathLike[str],
    contract: ReplicationSeedScheduleContract,
) -> Path:
    destination = Path(path)
    content = canonical_json_bytes(contract.normalized())
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        _fail(f"refusing to write through symlink: {destination}")
    if destination.exists():
        if destination.read_bytes() != content:
            raise FileExistsError(
                f"write-once seed contract already differs: {destination}"
            )
        load_contract(destination)
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
    load_contract(destination)
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or verify the final-model replication seed contract"
    )
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirmatory-seed-count",
        type=int,
        default=DEFAULT_CONFIRMATORY_SEED_COUNT,
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    source_digest = file_sha256(args.source_lock)
    expected = ReplicationSeedScheduleContract.derive(
        source_digest,
        confirmatory_seed_count=args.confirmatory_seed_count,
    )
    if args.verify_only:
        observed = load_contract(args.output)
        if observed.normalized() != expected.normalized():
            _fail("existing seed contract differs from the source lock")
        print(f"VERIFIED {args.output} sha256={file_sha256(args.output)}")
        return
    written = write_contract_once(args.output, expected)
    print(f"LOCKED {written} sha256={file_sha256(written)}")


if __name__ == "__main__":
    main()

