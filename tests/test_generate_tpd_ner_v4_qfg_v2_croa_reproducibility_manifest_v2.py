from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments import (
    generate_tpd_ner_v4_qfg_v2_croa_reproducibility_manifest_v2 as subject,
)
from experiments import (
    publish_tpd_ner_v4_qfg_v2_croa_default_operating_point_v2 as publisher,
)


def _fixture() -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    artifact = {
        "path": "/tmp/inference.pth.tar",
        "sha256": "a" * 64,
    }
    checkpoint = {
        "checkpoint": "best_miou.pth.tar",
        "checkpoint_role": "best_validation_miou_secondary",
        "role_name": "miou_secondary",
        "checkpoint_epoch": 3,
        "checkpoint_path": "/tmp/best_miou.pth.tar",
        "checkpoint_sha256": subject.EXPECTED_CHECKPOINT_SHA256,
    }
    old_point = {
        "method_id": subject.EXPECTED_METHOD,
        "variant": subject.EXPECTED_VARIANT,
        **checkpoint,
        "threshold": 0.0001589996954134993,
        "operating_point_source": "fa_budget:0.0001",
    }
    selected = {
        "method_id": subject.EXPECTED_METHOD,
        "variant": subject.EXPECTED_VARIANT,
        **checkpoint,
        "candidate_id": publisher.EXPECTED_CANDIDATE_ID,
        "source": "fixed_threshold_0_5",
        "operating_point_source": "fixed_threshold_0_5",
        "threshold": 0.5,
        "metrics": copy.deepcopy(subject.EXPECTED_METRICS),
        "checkpoint_local_atomic_point": True,
    }
    reused_artifact = {
        **artifact,
        "source_checkpoint_sha256": subject.EXPECTED_CHECKPOINT_SHA256,
        "reused_from_legacy_v1": True,
        "bytes_unchanged": True,
        "new_artifact_created": False,
    }
    profile_binding = {
        "path": "/tmp/profile.json",
        "sha256": "b" * 64,
        "schema": publisher.PROFILE_SCHEMA,
    }
    deployment_binding = {
        "path": "/tmp/deployment-v2.json",
        "sha256": "c" * 64,
        "schema": publisher.MANIFEST_SCHEMA,
    }
    profile = {
        "schema": publisher.PROFILE_SCHEMA,
        "status": "complete",
        "dataset": "NUDT-SIRST",
        "training_seed": 42,
        "split_seed": 20260722,
        "official_test_accessed": False,
        "selected_method_id": subject.EXPECTED_METHOD,
        "selected_variant": subject.EXPECTED_VARIANT,
        "selected_checkpoint": checkpoint,
        "default_operating_point": selected,
        "legacy_v1": {
            "legacy_only": True,
            "authoritative_default": False,
            "deployment_operating_point": old_point,
        },
        "artifact": reused_artifact,
        "method_unchanged": True,
        "checkpoint_unchanged": True,
        "weights_unchanged": True,
        "artifact_reused": True,
    }
    deployment = {
        "schema": publisher.MANIFEST_SCHEMA,
        "status": "complete",
        "dataset": "NUDT-SIRST",
        "training_seed": 42,
        "split_seed": 20260722,
        "official_test_accessed": False,
        "selected_method_id": subject.EXPECTED_METHOD,
        "selected_variant": subject.EXPECTED_VARIANT,
        "selected_checkpoint": checkpoint,
        "deployment_operating_point": {"selected": selected},
        "default_operating_point_profile": profile_binding,
        "artifact": reused_artifact,
        "method_unchanged": True,
        "checkpoint_unchanged": True,
        "weights_unchanged": True,
        "artifact_reused": True,
    }
    base = {
        "terminal_family": "current",
        "terminal_authority": {
            "selected_method_id": subject.EXPECTED_METHOD,
            "selected_checkpoint": {
                "path": checkpoint["checkpoint_path"],
                "sha256": checkpoint["checkpoint_sha256"],
                "filename": checkpoint["checkpoint"],
                "role": checkpoint["checkpoint_role"],
                "epoch": checkpoint["checkpoint_epoch"],
            },
            "deployment_export": artifact,
        },
    }
    lock = {
        "schema": subject.freezer.LOCK_SCHEMA,
        "path": "/tmp/lock.json",
        "sha256": "d" * 64,
        "verified_live": True,
    }
    return (
        base,
        profile,
        deployment,
        profile_binding,
        deployment_binding,
        artifact,
        lock,
    )


def _validate_fixture(parts: tuple[dict, ...]) -> dict:
    (
        base,
        profile,
        deployment,
        profile_binding,
        deployment_binding,
        artifact,
        lock,
    ) = parts
    return subject._validate_overlay(
        legacy_payload=base,
        profile=profile,
        deployment=deployment,
        profile_binding=profile_binding,
        deployment_binding=deployment_binding,
        artifact_binding=artifact,
        operational_lock=lock,
    )


def test_overlay_accepts_only_corrected_fixed_half_point() -> None:
    result = _validate_fixture(_fixture())
    assert result["selected_method_id"] == "d_tss_qfg"
    assert result["selected_operating_point"]["threshold"] == 0.5
    assert result["only_default_operating_point_changed"] is True
    assert result["weights_changed"] is False


def test_overlay_rejects_legacy_extreme_point_as_default() -> None:
    parts = list(_fixture())
    parts[2]["deployment_operating_point"]["selected"]["threshold"] = (
        0.0001589996954134993
    )
    with pytest.raises(subject.EvidenceConflict, match="threshold"):
        _validate_fixture(tuple(parts))


def test_overlay_rejects_artifact_change() -> None:
    parts = list(_fixture())
    parts[2]["artifact"]["sha256"] = "e" * 64
    with pytest.raises(subject.EvidenceConflict, match="artifact binding"):
        _validate_fixture(tuple(parts))


def test_overlay_rejects_metric_drift() -> None:
    parts = list(_fixture())
    parts[2]["deployment_operating_point"]["selected"]["metrics"]["miou"] = 0.7
    with pytest.raises(subject.EvidenceConflict, match="miou"):
        _validate_fixture(tuple(parts))


def test_write_once_bundle_is_idempotent_and_conflict_detecting(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    expected = {
        "manifest.json": b'{"ok":true}\n',
        "manifest.md": b"# ok\n",
    }
    bindings, written = subject._publish_bundle(output, expected)
    assert written is True
    assert set(bindings) == {"manifest.json", "manifest.md"}
    bindings_again, written_again = subject._publish_bundle(output, expected)
    assert written_again is False
    assert bindings_again == bindings
    conflict = dict(expected)
    conflict["manifest.md"] = b"# changed\n"
    with pytest.raises(subject.EvidenceConflict, match="manifest.md"):
        subject._publish_bundle(output, conflict)


def test_allow_unfrozen_cannot_publish(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        subject,
        "build_manifest",
        lambda **_: {
            "schema": subject.SCHEMA,
            "terminal_authority": {
                "selected_checkpoint": {
                    "checkpoint": "best_miou.pth.tar",
                    "checkpoint_epoch": 3,
                    "checkpoint_sha256": subject.EXPECTED_CHECKPOINT_SHA256,
                },
                "selected_threshold": 0.5,
                "selected_metrics": subject.EXPECTED_METRICS,
            },
            "legacy_terminal_authority_v1": {"selected_threshold": 0.1},
            "operational_default_v2": {
                "operational_source_lock": {"sha256": None}
            },
        },
    )
    with pytest.raises(subject.EvidenceConflict, match="allow-unfrozen"):
        subject.execute(output_dir=tmp_path / "bundle", allow_unfrozen=True)


def test_markdown_identifies_v2_as_default() -> None:
    payload = {
        "schema": subject.SCHEMA,
        "terminal_authority": {
            "selected_checkpoint": {
                "checkpoint": "best_miou.pth.tar",
                "checkpoint_epoch": 3,
                "checkpoint_sha256": subject.EXPECTED_CHECKPOINT_SHA256,
            },
            "selected_threshold": 0.5,
            "selected_metrics": subject.EXPECTED_METRICS,
        },
        "legacy_terminal_authority_v1": {
            "selected_threshold": 0.0001589996954134993
        },
        "operational_default_v2": {
            "operational_source_lock": {"sha256": "d" * 64}
        },
    }
    rendered = subject.render_markdown(payload)
    assert "Authoritative default operating point" in rendered
    assert "Threshold: `0.5`" in rendered
    assert "legacy" in rendered.lower()
