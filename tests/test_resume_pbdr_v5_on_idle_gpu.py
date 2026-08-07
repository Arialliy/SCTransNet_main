from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.pbdr_v4_run_artifacts import file_sha256
from experiments.pbdr_v5_run_contract import canonical_json_sha256
from experiments.resume_pbdr_v5_on_idle_gpu import (
    IDLE_GPU_BINDINGS,
    IdleGPUMigrationError,
    prepare_migration,
)


def _source_run(tmp_path: Path, *, official_test_accessed: bool = False) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    identity = {
        "dataset": "NUDT-SIRST",
        "role": "best_pd",
        "arm": "target_preserve_stage2",
    }
    protocol = {
        "run_identity": identity,
        "run_identity_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "expected_gpu_uuid": "GPU-source",
        "official_test_accessed": official_test_accessed,
        "performance_acceptance_margin": None,
    }
    (source / "run_protocol.json").write_text(
        json.dumps(protocol), encoding="utf-8"
    )
    torch.save(
        {
            "identity": identity,
            "identity_sha256": "a" * 64,
            "epoch": 15,
            "epochs": 30,
            "state_sha256": "c" * 64,
            "official_test_accessed": False,
            "performance_acceptance_margin": None,
        },
        source / "rolling_state.pth.tar",
    )
    return source


def test_prepare_migration_copies_exact_rolling_and_is_idempotent(
    tmp_path: Path,
) -> None:
    source = _source_run(tmp_path)
    destination = tmp_path / "destination"
    gpu_uuid = IDLE_GPU_BINDINGS["NUDT-SIRST"]

    ledger_path = prepare_migration(
        source_run_dir=source,
        destination_run_dir=destination,
        dataset="NUDT-SIRST",
        role="best_pd",
        expected_gpu_uuid=gpu_uuid,
    )
    assert prepare_migration(
        source_run_dir=source,
        destination_run_dir=destination,
        dataset="NUDT-SIRST",
        role="best_pd",
        expected_gpu_uuid=gpu_uuid,
    ) == ledger_path
    assert file_sha256(destination / "rolling_state.pth.tar") == file_sha256(
        source / "rolling_state.pth.tar"
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger_sha = ledger.pop("ledger_sha256")
    assert ledger_sha == canonical_json_sha256(ledger)
    assert ledger["source_rolling_epoch"] == 15
    assert ledger["destination_expected_gpu_uuid"] == gpu_uuid
    assert ledger["model_loss_optimizer_data_or_selection_changed"] is False
    assert ledger["official_test_accessed"] is False
    assert ledger["performance_acceptance_margin"] is None


def test_prepare_migration_rejects_unregistered_gpu(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    with pytest.raises(IdleGPUMigrationError, match="idle-GPU binding differs"):
        prepare_migration(
            source_run_dir=source,
            destination_run_dir=tmp_path / "destination",
            dataset="NUDT-SIRST",
            role="best_pd",
            expected_gpu_uuid="GPU-unregistered",
        )


def test_prepare_migration_rejects_official_access_flag(tmp_path: Path) -> None:
    source = _source_run(tmp_path, official_test_accessed=True)
    with pytest.raises(IdleGPUMigrationError, match="official-test flag differs"):
        prepare_migration(
            source_run_dir=source,
            destination_run_dir=tmp_path / "destination",
            dataset="NUDT-SIRST",
            role="best_pd",
            expected_gpu_uuid=IDLE_GPU_BINDINGS["NUDT-SIRST"],
        )
