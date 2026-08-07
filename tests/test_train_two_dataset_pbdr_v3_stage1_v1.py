from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from experiments import train_two_dataset_pbdr_v3_stage1_v1 as trainer


def _metrics() -> dict[str, float | int]:
    return {
        "matched_target_count": 8,
        "target_count": 10,
        "pd": 0.8,
        "fa": 0.01,
        "miou": 0.7,
        "niou": 0.65,
        "matched_tiny_target_count": 2,
        "tiny_target_count": 4,
        "tiny_pd": 0.5,
        "test_loss": 0.1,
    }


def test_selection_key_has_no_positive_margin_and_uses_earlier_epoch_last() -> None:
    original = _metrics()
    candidate = copy.deepcopy(original)
    candidate["miou"] = math.nextafter(float(original["miou"]), math.inf)
    assert trainer.selection_key("best_miou", candidate, 150) > trainer.selection_key(
        "best_miou", original, 1
    )
    assert trainer.selection_key("best_miou", original, 1) > trainer.selection_key(
        "best_miou", original, 2
    )


def test_validation_threshold_is_fixed_to_point_five() -> None:
    current = _metrics()
    candidate = copy.deepcopy(current)
    candidate["miou"] = math.nextafter(float(current["miou"]), math.inf)
    threshold, payload = trainer.fixed_validation_threshold(
        "best_miou",
        {
            "fixed_0_5": {"current": current, "candidate": candidate},
            "candidate_threshold_sweep": {"0.50": candidate},
        },
    )
    assert threshold == 0.5
    assert payload["threshold_optimization_performed"] is False
    assert payload["certification"]["passed"] is True


def test_real_engine_metrics_are_canonicalized_for_zero_margin_gate() -> None:
    probability = np.full((4, 4), 0.25, dtype=np.float32)
    target = np.zeros((4, 4), dtype=np.float32)
    target[1:3, 1:3] = 1.0
    metrics = trainer.cross_dataset_metrics([probability], [target], 0.5)
    assert "test_loss" in metrics
    assert "val_loss" not in metrics
    ready = trainer.zero_gate.CertificationMetrics.from_mapping(metrics)
    assert ready.test_loss == metrics["test_loss"]


def test_configure_engine_installs_canonical_metric_adapter() -> None:
    trainer.configure_engine("NUDT-SIRST", "best_miou")
    probability = np.full((4, 4), 0.25, dtype=np.float32)
    target = np.zeros((4, 4), dtype=np.float32)
    target[1:3, 1:3] = 1.0
    metrics = trainer.engine._metrics([probability], [target], 0.5)
    assert "test_loss" in metrics
    assert "val_loss" not in metrics
    trainer.selection_key("best_miou", metrics, 1)


def test_runtime_source_locks_cover_loss_and_imported_legacy_gate() -> None:
    records = trainer.DatasetModelsAdapter("NUDT-SIRST").runtime_source_records()
    assert "experiments/pbdr_v3_loss.py" in records
    assert "experiments/pbdr_v3_non_regression_gate.py" in records


def test_frozen_data_binding_requires_exact_root_manifest_and_sha(tmp_path: Path) -> None:
    binding = trainer.validate_frozen_data_binding(
        trainer.DEFAULT_DATA_ROOT,
        trainer.DEFAULT_PROTOCOL_MANIFEST,
    )
    assert (
        binding["protocol_manifest_sha256"]
        == trainer.FORMAL_PROTOCOL_MANIFEST_SHA256
    )
    copied = tmp_path / "protocol.json"
    copied.write_bytes(trainer.DEFAULT_PROTOCOL_MANIFEST.read_bytes())
    with pytest.raises(ValueError, match="formal path"):
        trainer.validate_frozen_data_binding(trainer.DEFAULT_DATA_ROOT, copied)


def test_completed_auto_resume_returns_without_rewriting_candidate(
    tmp_path: Path,
) -> None:
    args = trainer.parse_args(
        [
            "--dataset",
            "NUDT-SIRST",
            "--parent-role",
            "best_miou",
            "--results-root",
            str(tmp_path),
            "--resume",
            "auto",
        ]
    )
    run_dir = (
        tmp_path
        / "runs"
        / "NUDT-SIRST"
        / "formal"
        / "best_miou"
        / "core"
    )
    run_dir.mkdir(parents=True)
    candidate = run_dir / "selected_candidate.pth.tar"
    candidate.write_bytes(b"immutable-candidate-bytes")
    summary = run_dir / "summary.json"
    summary.write_text("{}\n", encoding="utf-8")
    before = hashlib.sha256(candidate.read_bytes()).hexdigest()
    with (
        mock.patch.object(
            trainer,
            "_completed_run_summary",
            return_value=summary.resolve(),
        ),
        mock.patch.object(
            trainer.engine,
            "run",
            side_effect=AssertionError("completed run was executed again"),
        ),
    ):
        observed = trainer.run(args)
    after = hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert observed == summary.resolve()
    assert before == after


def test_completed_certification_replay_rejects_a_tampered_artifact() -> None:
    metrics = _metrics()
    decision = trainer.zero_gate.certify(
        "best_miou",
        trainer.zero_gate.CertificationMetrics.from_mapping(metrics),
        trainer.zero_gate.CertificationMetrics.from_mapping(metrics),
    )
    artifact = trainer.zero_gate._json_payload(decision)
    core = {
        name: artifact[name]
        for name in ("passed", "selected", "checks", "current", "candidate")
    }
    core["scope"] = "frozen_internal_validation_split"
    trainer._validate_completed_certification(
        "best_miou", core, core, artifact
    )
    forged = dict(artifact)
    forged["decisive_term"] = "higher_miou"
    with pytest.raises(ValueError, match="artifact differs"):
        trainer._validate_completed_certification(
            "best_miou", core, core, forged
        )


def test_completed_torch_artifact_loader_rejects_arbitrary_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rolling_state.pth.tar"
    path.write_bytes(b"not-a-checkpoint")
    with pytest.raises(ValueError, match="cannot load completed rolling state"):
        trainer._load_torch_mapping(path, "completed rolling state")


def test_results_paths_are_dataset_isolated() -> None:
    root = Path("/tmp/pbdr-v3-results")
    nudt = trainer.dataset_results_root(root, "NUDT-SIRST")
    irstd = trainer.dataset_results_root(root, "IRSTD-1K")
    assert nudt != irstd
    assert nudt.parts[-2:] == ("runs", "NUDT-SIRST")
    assert irstd.parts[-2:] == ("runs", "IRSTD-1K")


def test_each_dataset_has_a_distinct_frozen_gpu_uuid() -> None:
    assert set(trainer.GPU_UUIDS) == set(trainer.DATASETS)
    assert len(set(trainer.GPU_UUIDS.values())) == len(trainer.DATASETS)


def test_smoke_rejects_a_validation_prefix_without_tiny_targets() -> None:
    args = trainer.parse_args(
        [
            "--dataset",
            "NUDT-SIRST",
            "--parent-role",
            "best_miou",
            "--smoke",
            "--epochs",
            "1",
            "--eval-every",
            "1",
            "--batch-size",
            "2",
            "--max-train-images",
            "2",
            "--max-val-images",
            "1",
        ]
    )
    with pytest.raises(ValueError, match="tiny-target"):
        trainer.run(args)
