from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest import mock

import pytest

from experiments import (
    deploy_tpd_ner_v4_qfg_v2_croa_formal800 as deploy,
)
from experiments import (
    freeze_tpd_ner_v4_qfg_v2_croa_posttraining_closure as freezer,
)
from experiments import (
    tpd_ner_v4_qfg_v2_croa_posttraining_policy as policy,
)


def _point(
    *,
    threshold: float,
    pd: float,
    fa: float,
    miou: float,
    tiny_pd: float = 1.0,
    false_objects: float = 0.02,
) -> dict[str, float]:
    return {
        "threshold": threshold,
        "pd": pd,
        "fa": fa,
        "miou": miou,
        "tiny_pd": tiny_pd,
        "false_objects_per_image": false_objects,
    }


def _role(
    *,
    role_name: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    fixed: dict[str, float],
    budgets: dict[str, dict[str, float]],
) -> dict[str, object]:
    return {
        "checkpoint": policy.CHECKPOINT_FILENAMES[role_name],
        "checkpoint_role": (
            "best_validation_pd_primary"
            if role_name == "pd_primary"
            else "best_validation_miou_secondary"
        ),
        "role_name": role_name,
        "checkpoint_epoch": 11 if role_name == "pd_primary" else 17,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "fixed_threshold_0_5": fixed,
        "fa_budget_points": budgets,
    }


def _method(
    tmp_path: Path,
    *,
    method_id: str = "c_qfg_only",
    variant: str = "qfg_only",
) -> dict[str, object]:
    primary = tmp_path / "best.pth.tar"
    secondary = tmp_path / "best_miou.pth.tar"
    primary.write_bytes(b"primary")
    secondary.write_bytes(b"secondary")
    primary_fixed = _point(
        threshold=0.5,
        pd=188 / 189,
        fa=4e-6,
        miou=0.92,
    )
    secondary_fixed = _point(
        threshold=0.5,
        pd=187 / 189,
        fa=1e-6,
        miou=0.95,
    )
    primary_budgets = {
        key: copy.deepcopy(primary_fixed) for key in policy.BUDGET_KEYS
    }
    secondary_budgets = {
        key: copy.deepcopy(secondary_fixed) for key in policy.BUDGET_KEYS
    }
    return {
        "method_id": method_id,
        "variant": variant,
        "roles": {
            "pd_primary": _role(
                role_name="pd_primary",
                checkpoint_path=primary,
                checkpoint_sha256=policy.sha256_file(primary),
                fixed=primary_fixed,
                budgets=primary_budgets,
            ),
            "miou_secondary": _role(
                role_name="miou_secondary",
                checkpoint_path=secondary,
                checkpoint_sha256=policy.sha256_file(secondary),
                fixed=secondary_fixed,
                budgets=secondary_budgets,
            ),
        },
    }


def _closure_binding(tmp_path: Path) -> dict[str, object]:
    return {
        "schema": policy.LOCK_SCHEMA,
        "path": str(tmp_path / "closure.json"),
        "sha256": "c" * 64,
        "source_count": len(policy.POSTTRAINING_SOURCE_PATHS),
        "policy_summary_sha256": policy.policy_summary_sha256(),
        "training_source_lock_sha256": policy.TRAINING_LOCK_SHA256,
        "verified_live": True,
    }


def _selection_report(
    tmp_path: Path,
    *,
    method_id: str = "c_qfg_only",
    variant: str = "qfg_only",
) -> tuple[dict[str, object], Path, dict[str, object]]:
    method = _method(tmp_path, method_id=method_id, variant=variant)
    deployment = policy.select_deployment_operating_point(method)
    closure_binding = _closure_binding(tmp_path)
    report = {
        "schema": "synthetic_selection_v2",
        "status": "complete",
        "selection": {"selected_method_id": method_id},
        "deployment_selection": deployment,
        "posttraining_closure_source_lock": closure_binding,
    }
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report, path, closure_binding


def test_posttraining_source_set_is_exact_and_disjoint_from_training_lock() -> None:
    training = json.loads(policy.TRAINING_LOCK_PATH.read_text(encoding="utf-8"))
    training_sources = set(training["source_sha256"])
    closure_sources = set(policy.POSTTRAINING_SOURCE_PATHS)
    assert len(closure_sources) == 15
    assert len(closure_sources) == len(policy.POSTTRAINING_SOURCE_PATHS)
    assert not closure_sources.intersection(training_sources)
    assert {
        "experiments/evaluate_tpd_ner_v4_qfg_v2_croa_pd_fa.py",
        "experiments/run_tpd_ner_v4_qfg_v2_croa_formal800_sweeps_2x5090.sh",
        "experiments/compare_tss_qfg_v2_croa_factorial.py",
        "experiments/postprocess_tpd_ner_v4_qfg_v2_croa_formal800.py",
        "experiments/export_tpd_ner_v4_qfg_v2_croa_to_inference.py",
        "experiments/finalize_tpd_ner_v4_qfg_v2_croa_formal800_2x5090.sh",
    }.issubset(closure_sources)


def test_deployment_selection_is_one_atomic_checkpoint_local_point(
    tmp_path: Path,
) -> None:
    method = _method(tmp_path)
    result = policy.select_deployment_operating_point(method)
    assert result["candidate_count"] == 12
    assert result["cross_checkpoint_metric_stitching"] is False
    selected = result["selected"]
    assert selected["method_id"] == "c_qfg_only"
    assert selected["role_name"] == "pd_primary"
    assert selected["checkpoint"] == "best.pth.tar"
    assert selected["metrics"]["pd"] == pytest.approx(188 / 189)
    assert selected["metrics"]["miou"] == pytest.approx(0.92)
    assert selected["checkpoint_local_atomic_point"] is True
    source_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in policy.deployment_candidates(method)
    }
    assert selected == {
        key: value
        for key, value in source_candidates[selected["candidate_id"]].items()
        if not key.startswith("_")
    }


def test_closure_lock_is_write_once_and_live_verifiable(tmp_path: Path) -> None:
    output = tmp_path / "closure.json"
    action = freezer.write_once(output)
    assert action["verified"] is True
    assert action["source_count"] == len(policy.POSTTRAINING_SOURCE_PATHS)
    verified = freezer.verify(output)
    assert verified["output_sha256"] == action["output_sha256"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["policy_summary"] == policy.policy_summary()
    assert payload["policy_summary_sha256"] == policy.policy_summary_sha256()
    with pytest.raises(FileExistsError):
        freezer.write_once(output)


def test_qfg_deployment_invokes_exporter_once_and_is_idempotent(
    tmp_path: Path,
) -> None:
    report, selection_path, closure_binding = _selection_report(tmp_path)
    selected = report["deployment_selection"]["selected"]
    artifact = tmp_path / "deployment/inference.pth.tar"
    manifest = tmp_path / "deployment/manifest.json"

    def fake_export(source: Path, output: Path) -> dict[str, object]:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"head-free-export")
        return {"schema": "fake"}

    def fake_validate(
        output: Path,
        *,
        expected_source_checkpoint: Path | None = None,
    ) -> dict[str, object]:
        assert expected_source_checkpoint is not None
        return {
            "schema": "fake_qfg_export_v1",
            "path": str(Path(output).resolve()),
            "sha256": policy.sha256_file(output),
            "source_checkpoint_path": str(expected_source_checkpoint.resolve()),
            "source_checkpoint_sha256": selected["checkpoint_sha256"],
            "source_checkpoint_role": selected["checkpoint_role"],
            "inference_state_key_count": 564,
            "inference_parameter_count": 10870130,
            "strict_load": True,
            "survival_state_absent": True,
            "qfg_state_preserved": True,
        }

    with (
        mock.patch.object(
            deploy.policy,
            "load_closure_lock",
            return_value=({}, closure_binding),
        ),
        mock.patch.object(
            deploy.selector,
            "build_formal_report",
            return_value=report,
        ),
        mock.patch.object(
            deploy.qfg_exporter,
            "export_qfg_checkpoint",
            side_effect=fake_export,
        ) as exporter,
        mock.patch.object(
            deploy.qfg_exporter,
            "validate_exported_qfg_checkpoint",
            side_effect=fake_validate,
        ),
    ):
        first = deploy.publish_deployment(
            selection_path=selection_path,
            artifact_path=artifact,
            manifest_path=manifest,
        )
        second = deploy.publish_deployment(
            selection_path=selection_path,
            artifact_path=artifact,
            manifest_path=manifest,
        )
    assert first["action"] == "publish"
    assert second["action"] == "verify"
    exporter.assert_called_once()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["selected_checkpoint"]["checkpoint_sha256"] == (
        selected["checkpoint_sha256"]
    )
    assert payload["deployment_operating_point"]["selected"] == selected
    assert payload["posttraining_closure_source_lock"] == closure_binding
    assert payload["cross_checkpoint_metric_stitching"] is False


def test_v4_fallback_creates_byte_identical_native_artifact(
    tmp_path: Path,
) -> None:
    report, selection_path, closure_binding = _selection_report(
        tmp_path,
        method_id="v4",
        variant="tpd_ner_v8_mprs_dch_v4_tail_aware_full_relay_on",
    )
    source = Path(
        report["deployment_selection"]["selected"]["checkpoint_path"]
    )
    artifact = tmp_path / "deployment/v4.pth.tar"
    manifest = tmp_path / "deployment/v4.json"
    with (
        mock.patch.object(
            deploy.policy,
            "load_closure_lock",
            return_value=({}, closure_binding),
        ),
        mock.patch.object(
            deploy.selector,
            "build_formal_report",
            return_value=report,
        ),
        mock.patch.object(
            deploy.qfg_exporter,
            "export_qfg_checkpoint",
            side_effect=AssertionError("QFG exporter must not handle V4"),
        ),
        mock.patch.object(
            deploy.v4_evaluator,
            "validate_run_artifacts",
            return_value={
                "run_directory": str(source.parent),
                "checkpoint_filename": source.name,
                "checkpoint_sha256": policy.sha256_file(source),
                "checkpoint_role": report["deployment_selection"]["selected"][
                    "checkpoint_role"
                ],
                "checkpoint_epoch": report["deployment_selection"]["selected"][
                    "checkpoint_epoch"
                ],
            },
        ),
    ):
        result = deploy.publish_deployment(
            selection_path=selection_path,
            artifact_path=artifact,
            manifest_path=manifest,
        )
    assert result["action"] == "publish"
    assert artifact.read_bytes() == source.read_bytes()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["export_mode"] == "write_once_native_v4_checkpoint_copy"
    assert payload["artifact"]["byte_identical_copy"] is True
    assert payload["artifact"]["source_evaluator_validation"] is True
    assert payload["exporter"]["invoked"] is False
