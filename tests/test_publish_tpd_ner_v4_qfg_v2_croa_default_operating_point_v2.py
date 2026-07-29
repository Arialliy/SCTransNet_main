from __future__ import annotations
import copy
import json
from pathlib import Path

import pytest

from experiments import (
    publish_tpd_ner_v4_qfg_v2_croa_default_operating_point_v2 as subject,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_SELECTION = (
    REPO_ROOT
    / "experiments/results/"
    "tpd_ner_v4_qfg_v2_croa_exact_v2_optimized/"
    "NUDT-SIRST/final_selection/"
    "tpd_ner_v4_qfg_v2_croa_formal800_final_selection.json"
)


def _real_method_d() -> dict:
    report = json.loads(REAL_SELECTION.read_text(encoding="utf-8"))
    return report["methods"]["d_tss_qfg"]


def _context(tmp_path: Path) -> dict:
    selected, policy = subject.select_default_operating_point(_real_method_d())
    artifact_path = tmp_path / "legacy_inference.pth.tar"
    artifact_path.write_bytes(b"unchanged-v1-inference-bytes")
    legacy_selected = {
        "candidate_id": "miou_secondary:fa_budget:0.0001",
        "method_id": "d_tss_qfg",
        "variant": "tss_qfg",
        "checkpoint": "best_miou.pth.tar",
        "checkpoint_role": "best_validation_miou_secondary",
        "role_name": "miou_secondary",
        "checkpoint_epoch": 3,
        "checkpoint_path": selected["checkpoint_path"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "operating_point_source": "fa_budget:0.0001",
        "threshold": 0.0001589996954134993,
        "metrics": {
            "pd": 1.0,
            "tiny_pd": 1.0,
            "miou": 0.7018605182056843,
            "fa": 7.032810297227443e-05,
            "false_objects_per_image": 2.4586466165413534,
        },
    }
    return {
        "selection": {},
        "legacy_manifest": {},
        "legacy_selected": legacy_selected,
        "selection_binding": {
            "path": str(tmp_path / "selection.json"),
            "sha256": "1" * 64,
            "schema": "sctransnet_tpd_ner_v4_qfg_v2_croa_final_selection_v2",
        },
        "legacy_manifest_binding": {
            "path": str(tmp_path / "legacy_manifest.json"),
            "sha256": "2" * 64,
            "schema": (
                "sctransnet_tpd_ner_v4_qfg_v2_croa_deployment_manifest_v1"
            ),
        },
        "closure_binding": {
            "path": str(tmp_path / "closure.json"),
            "sha256": "3" * 64,
            "schema": (
                "sctransnet_tpd_ner_v4_qfg_v2_croa_"
                "posttraining_closure_source_lock_v1"
            ),
            "source_count": 15,
            "verified_live": True,
        },
        "sweep_binding": {
            "path": str(tmp_path / "pd_fa_sweep_best_miou.pth.json"),
            "sha256": "4" * 64,
            "schema": "sctransnet_tpd_ner_v4_qfg_v2_croa_pd_fa_sweep_v1",
            "checkpoint_identity_validated": True,
            "evaluator_output_identity_validated": True,
            "checkpoint_state_dict_strict_load": True,
        },
        "artifact": {
            "path": str(artifact_path),
            "sha256": subject.closure_policy.sha256_file(artifact_path),
            "schema": (
                "sctransnet_tpd_ner_v4_qfg_v2_croa_inference_export_v1"
            ),
            "source_checkpoint_path": selected["checkpoint_path"],
            "source_checkpoint_sha256": selected["checkpoint_sha256"],
            "source_checkpoint_role": selected["checkpoint_role"],
        },
        "selected": selected,
        "policy": policy,
    }


def _patch_context(monkeypatch: pytest.MonkeyPatch, context: dict) -> None:
    monkeypatch.setattr(
        subject,
        "_prepare_context",
        lambda **_kwargs: copy.deepcopy(context),
    )


def test_registered_low_fa_policy_selects_exact_d_best_miou_fixed_point() -> None:
    candidates = subject.preregistered_candidates(_real_method_d())
    selected, policy = subject.select_default_operating_point(_real_method_d())

    assert len(candidates) == 12
    assert policy["candidate_count"] == 12
    assert policy["eligible_candidate_count"] == 11
    assert policy["eligibility"]["maximum"] == pytest.approx(5e-6)
    assert selected["candidate_id"] == "miou_secondary:fixed_threshold_0_5"
    assert selected["method_id"] == "d_tss_qfg"
    assert selected["variant"] == "tss_qfg"
    assert selected["checkpoint"] == "best_miou.pth.tar"
    assert selected["checkpoint_epoch"] == 3
    assert selected["checkpoint_sha256"] == subject.EXPECTED_CHECKPOINT_SHA256
    assert selected["threshold"] == pytest.approx(0.5)
    assert selected["metrics"]["pd"] == pytest.approx(188 / 189)
    assert selected["metrics"]["tiny_pd"] == pytest.approx(1.0)
    assert selected["metrics"]["miou"] == pytest.approx(0.9370177924736262)
    assert selected["metrics"]["fa"] == pytest.approx(
        4.1301985432330825e-6
    )
    assert selected["metrics"]["unmatched_predicted_object_count"] == 5


def test_profile_and_manifest_make_v2_authoritative_and_v1_legacy(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    profile = subject.build_profile(context)
    profile_path = tmp_path / "profile.json"
    profile_path.write_bytes(subject.closure_policy.canonical_json_bytes(profile))
    profile_binding = subject._profile_binding(profile_path)
    manifest = subject.build_manifest(
        context,
        profile_binding=profile_binding,
    )

    assert profile["schema"] == subject.PROFILE_SCHEMA
    assert manifest["schema"] == subject.MANIFEST_SCHEMA
    assert profile["policy"]["authority"] == "authoritative_default"
    assert profile["policy"]["default_for_inference"] is True
    assert profile["legacy_v1"]["legacy_only"] is True
    assert profile["legacy_v1"]["authoritative_default"] is False
    assert manifest["deployment_operating_point"]["authority"] == (
        "authoritative_default"
    )
    assert manifest["deployment_operating_point"]["default_for_inference"] is True
    assert manifest["legacy_v1"]["legacy_only"] is True
    assert manifest["legacy_v1"]["authoritative_default"] is False
    assert profile["default_operating_point"]["source"] == (
        "fixed_threshold_0_5"
    )
    assert manifest["deployment_operating_point"]["selected"]["threshold"] == 0.5
    assert manifest["default_operating_point_profile"] == profile_binding
    assert profile["artifact"]["path"] == context["artifact"]["path"]
    assert manifest["artifact"]["sha256"] == context["artifact"]["sha256"]
    for payload in (profile, manifest):
        assert payload["method_unchanged"] is True
        assert payload["checkpoint_unchanged"] is True
        assert payload["weights_unchanged"] is True
        assert payload["artifact_reused"] is True


def test_preflight_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    _patch_context(monkeypatch, context)
    profile = tmp_path / "out" / "profile.json"
    manifest = tmp_path / "out" / "manifest.json"

    result = subject.publish_default_operating_point(
        profile_path=profile,
        manifest_path=manifest,
        preflight=True,
    )

    assert result["action"] == "preflight"
    assert result["writes_performed"] is False
    assert result["selected_threshold"] == 0.5
    assert not profile.exists()
    assert not manifest.exists()


def test_temp_publication_is_write_once_idempotent_and_reuses_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    _patch_context(monkeypatch, context)
    profile = tmp_path / "out" / "profile.json"
    manifest = tmp_path / "out" / "manifest.json"
    artifact = Path(context["artifact"]["path"])
    original_artifact = artifact.read_bytes()

    first = subject.publish_default_operating_point(
        profile_path=profile,
        manifest_path=manifest,
    )
    profile_bytes = profile.read_bytes()
    manifest_bytes = manifest.read_bytes()
    second = subject.publish_default_operating_point(
        profile_path=profile,
        manifest_path=manifest,
    )
    verified = subject.validate_publication(
        profile_path=profile,
        manifest_path=manifest,
    )

    assert first["action"] == "publish"
    assert second["action"] == "verify"
    assert verified["verified"] is True
    assert profile.read_bytes() == profile_bytes
    assert manifest.read_bytes() == manifest_bytes
    assert artifact.read_bytes() == original_artifact
    assert first["artifact_path"] == str(artifact)
    assert first["artifact_reused"] is True


def test_partial_exact_profile_recovers_by_only_creating_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    _patch_context(monkeypatch, context)
    profile = tmp_path / "out" / "profile.json"
    manifest = tmp_path / "out" / "manifest.json"
    subject._atomic_create_json(profile, subject.build_profile(context))
    profile_bytes = profile.read_bytes()

    result = subject.publish_default_operating_point(
        profile_path=profile,
        manifest_path=manifest,
    )

    assert result["action"] == "publish"
    assert profile.read_bytes() == profile_bytes
    assert manifest.is_file()


def test_conflicting_existing_profile_is_rejected_without_manifest_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    _patch_context(monkeypatch, context)
    profile = tmp_path / "out" / "profile.json"
    manifest = tmp_path / "out" / "manifest.json"
    wrong = subject.build_profile(context)
    wrong["default_operating_point"]["threshold"] = 0.6
    subject._atomic_create_json(profile, wrong)

    with pytest.raises(ValueError, match="partial v2 profile conflicts"):
        subject.publish_default_operating_point(
            profile_path=profile,
            manifest_path=manifest,
        )

    assert not manifest.exists()


def test_manifest_without_profile_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    _patch_context(monkeypatch, context)
    profile = tmp_path / "out" / "profile.json"
    manifest = tmp_path / "out" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="without its bound profile"):
        subject.publish_default_operating_point(
            profile_path=profile,
            manifest_path=manifest,
        )
