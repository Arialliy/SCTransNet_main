"""Resume a frozen PBDR-V5 run on a user-selected idle RTX 5090.

This launcher changes only the formal dataset/GPU allowlist in memory.  It
copies a validated rolling checkpoint into a separate run directory, records
the hardware migration in a hash-bound ledger, and delegates all model,
optimizer, loss, data, selection, and artifact work to the frozen V5 trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import train_three_dataset_pbdr_v5_v1 as v5_train
from experiments.pbdr_v4_run_artifacts import exclusive_json, file_sha256, load_torch_artifact
from experiments.pbdr_v5_run_contract import canonical_json_sha256


SCHEMA = "sctransnet_pbdr_v5_idle_gpu_migration/v1"
LEDGER_NAME = "idle_gpu_migration.json"
ROLLING_NAME = "rolling_state.pth.tar"
RUN_PROTOCOL_NAME = "run_protocol.json"

IDLE_GPU_BINDINGS = {
    "NUDT-SIRST": "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
    "IRSTD-1K": "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
}


class IdleGPUMigrationError(RuntimeError):
    """A requested V5 hardware-only migration is not audit-safe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IdleGPUMigrationError(message)


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing or unsafe JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON payload must be an object: {path}")
    return value


def _commit_or_validate_ledger(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists() or path.is_symlink():
        _require(path.is_file() and not path.is_symlink(), "migration ledger is unsafe")
        observed = _read_json(path)
        _require(observed == dict(payload), "migration ledger differs")
        return path.resolve(strict=True)
    return exclusive_json(path, payload).resolve(strict=True)


def prepare_migration(
    *,
    source_run_dir: Path,
    destination_run_dir: Path,
    dataset: str,
    role: str,
    expected_gpu_uuid: str,
) -> Path:
    source_run_dir = source_run_dir.resolve(strict=True)
    destination_run_dir = destination_run_dir.resolve()
    _require(source_run_dir != destination_run_dir, "migration requires a separate run directory")

    source_protocol_path = source_run_dir / RUN_PROTOCOL_NAME
    source_rolling_path = source_run_dir / ROLLING_NAME
    source_protocol = _read_json(source_protocol_path)
    _require(
        source_rolling_path.is_file() and not source_rolling_path.is_symlink(),
        "source rolling checkpoint is missing or unsafe",
    )
    source_rolling = load_torch_artifact(source_rolling_path)
    identity = source_protocol.get("run_identity")
    _require(isinstance(identity, Mapping), "source run identity differs")
    _require(identity.get("dataset") == dataset and identity.get("role") == role, "source dataset/role differs")
    _require(source_protocol.get("official_test_accessed") is False, "source official-test flag differs")
    _require(source_protocol.get("performance_acceptance_margin") is None, "source performance margin differs")
    _require(source_rolling.get("official_test_accessed") is False, "rolling official-test flag differs")
    _require(source_rolling.get("performance_acceptance_margin") is None, "rolling performance margin differs")
    _require(
        source_rolling.get("identity") == identity
        and source_rolling.get("identity_sha256") == source_protocol.get("run_identity_sha256"),
        "rolling identity does not match source protocol",
    )
    _require(
        isinstance(source_rolling.get("epoch"), int) and 1 <= source_rolling["epoch"] < source_rolling.get("epochs", 0),
        "source rolling epoch is not resumable",
    )
    _require(IDLE_GPU_BINDINGS.get(dataset) == expected_gpu_uuid, "idle-GPU binding differs")

    destination_run_dir.mkdir(parents=True, exist_ok=True)
    _require(destination_run_dir.is_dir() and not destination_run_dir.is_symlink(), "destination run directory is unsafe")
    destination_rolling_path = destination_run_dir / ROLLING_NAME
    destination_protocol_path = destination_run_dir / RUN_PROTOCOL_NAME
    source_rolling_file_sha = file_sha256(source_rolling_path)
    if not destination_rolling_path.exists():
        shutil.copy2(source_rolling_path, destination_rolling_path)
    elif not destination_protocol_path.exists():
        _require(
            destination_rolling_path.is_file()
            and not destination_rolling_path.is_symlink()
            and file_sha256(destination_rolling_path) == source_rolling_file_sha,
            "unstarted destination rolling checkpoint differs",
        )

    ledger: dict[str, Any] = {
        "schema": SCHEMA,
        "dataset": dataset,
        "role": role,
        "source_run_dir": str(source_run_dir),
        "destination_run_dir": str(destination_run_dir),
        "source_run_protocol": str(source_protocol_path.resolve(strict=True)),
        "source_run_protocol_file_sha256": file_sha256(source_protocol_path),
        "source_run_protocol_sha256": source_protocol.get("protocol_sha256"),
        "run_identity_sha256": source_protocol.get("run_identity_sha256"),
        "source_rolling_checkpoint": str(source_rolling_path.resolve(strict=True)),
        "source_rolling_checkpoint_file_sha256": source_rolling_file_sha,
        "source_rolling_epoch": source_rolling["epoch"],
        "source_rolling_state_sha256": source_rolling.get("state_sha256"),
        "source_expected_gpu_uuid": source_protocol.get("expected_gpu_uuid"),
        "destination_expected_gpu_uuid": expected_gpu_uuid,
        "migration_reason": "user_directed_move_to_idle_gpu",
        "model_loss_optimizer_data_or_selection_changed": False,
        "official_test_accessed": False,
        "performance_acceptance_margin": None,
        "launcher": str(Path(__file__).resolve(strict=True)),
        "launcher_file_sha256": file_sha256(Path(__file__).resolve(strict=True)),
    }
    ledger["ledger_sha256"] = canonical_json_sha256(ledger)
    return _commit_or_validate_ledger(destination_run_dir / LEDGER_NAME, ledger)


def parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, argparse.Namespace]:
    migration_parser = argparse.ArgumentParser(add_help=False)
    migration_parser.add_argument("--migration-source-run-dir", type=Path, required=True)
    migration_args, remaining = migration_parser.parse_known_args(argv)
    return migration_args, v5_train.parse_args(remaining)


def main(argv: Sequence[str] | None = None) -> None:
    migration_args, args = parse_args(argv)
    _require(not args.smoke, "idle-GPU migration is formal-run only")
    _require(args.resume == "required", "migration must require an existing rolling state")
    _require(isinstance(args.expected_gpu_uuid, str) and bool(args.expected_gpu_uuid), "expected GPU UUID is required")
    ledger = prepare_migration(
        source_run_dir=migration_args.migration_source_run_dir,
        destination_run_dir=args.run_dir,
        dataset=args.dataset,
        role=args.role,
        expected_gpu_uuid=args.expected_gpu_uuid,
    )
    original_binding = v5_train.v4_train.FORMAL_GPU_UUIDS[args.dataset]
    v5_train.v4_train.FORMAL_GPU_UUIDS[args.dataset] = args.expected_gpu_uuid
    try:
        print(json.dumps({"event": "idle_gpu_migration_validated", "ledger": str(ledger)}), flush=True)
        print(v5_train.run(args), flush=True)
    finally:
        v5_train.v4_train.FORMAL_GPU_UUIDS[args.dataset] = original_binding


if __name__ == "__main__":
    main()


__all__ = ["IDLE_GPU_BINDINGS", "IdleGPUMigrationError", "prepare_migration"]
