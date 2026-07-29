from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from experiments import (
    compare_tpd_ner_v4_qfg_v2_croa_dlr_ramp100 as compare,
)
from experiments import (
    deploy_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800 as deploy,
)
from experiments import (
    freeze_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_posttraining_closure as freezer,
)
from experiments import (
    export_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_to_inference as dlr_export,
)
from experiments import (
    evaluate_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_pd_fa as dlr_evaluator,
)
from experiments import (
    tpd_ner_v4_qfg_v2_croa_dlr_ramp100_posttraining_policy as policy,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_survival import (
    build_formal_v4_qfg_v2_croa_survival_model,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _point(
    *,
    matched: int,
    unmatched: int,
    miou: float,
    threshold: float,
    fa: float,
    tiny_matched: int = 39,
) -> dict[str, object]:
    return {
        "threshold": threshold,
        "pd": matched / 189,
        "fa": fa,
        "miou": miou,
        "tiny_pd": tiny_matched / 39,
        "false_objects_per_image": unmatched / 133,
        "target_count": 189,
        "matched_target_count": matched,
        "tiny_target_count": 39,
        "matched_tiny_target_count": tiny_matched,
        "unmatched_predicted_object_count": unmatched,
    }


def _method(
    root: Path,
    method_id: str,
    *,
    matched: int,
    unmatched: int,
    miou: float,
) -> dict[str, object]:
    roles: dict[str, object] = {}
    for role_index, role_name in enumerate(policy.CHECKPOINT_ROLE_ORDER):
        checkpoint = root / method_id / policy.CHECKPOINT_FILENAMES[role_name]
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"{method_id}:{role_name}".encode())
        fixed = _point(
            matched=matched,
            unmatched=unmatched,
            miou=miou + role_index * 0.001,
            threshold=0.5,
            fa=unmatched * 1e-6,
        )
        budgets = {
            key: _point(
                matched=matched,
                unmatched=unmatched,
                miou=miou + role_index * 0.001 + index * 0.0001,
                threshold=0.6 + index * 0.01,
                fa=min(budget, unmatched * 1e-6),
            )
            for index, (key, budget) in enumerate(
                zip(policy.BUDGET_KEYS, policy.FA_BUDGETS)
            )
        }
        roles[role_name] = {
            "checkpoint": checkpoint.name,
            "checkpoint_role": policy.CHECKPOINT_ROLES[checkpoint.name],
            "role_name": role_name,
            "checkpoint_epoch": 10 + role_index,
            "checkpoint_sha256": _sha(checkpoint),
            "checkpoint_path": str(checkpoint),
            "run_directory": str(checkpoint.parent),
            "fixed_threshold_0_5": fixed,
            "fa_budget_points": budgets,
            "raw_point_count": 6,
            "sweep_binding": {
                "path": str(checkpoint),
                "sha256": _sha(checkpoint),
            },
        }
    return {
        "method_id": method_id,
        "display_name": method_id,
        "variant": f"fixture_{method_id}",
        "roles": roles,
    }


def _methods(root: Path) -> dict[str, dict[str, object]]:
    quality = {
        "baseline": (180, 8, 0.88),
        "v4": (183, 7, 0.89),
        "a_control": (184, 6, 0.90),
        "b_tss": (185, 5, 0.91),
        "c_qfg_only": (186, 4, 0.92),
        "d_tss_qfg": (187, 3, 0.93),
        "e_qfg_dlr": (188, 2, 0.94),
        "f_tss_qfg_dlr": (189, 1, 0.95),
    }
    return {
        method_id: _method(
            root,
            method_id,
            matched=quality[method_id][0],
            unmatched=quality[method_id][1],
            miou=quality[method_id][2],
        )
        for method_id in policy.METHOD_ORDER
    }


def _closure_binding(tmp_path: Path) -> dict[str, object]:
    return {
        "schema": policy.LOCK_SCHEMA,
        "path": str(tmp_path / "closure.json"),
        "sha256": "c" * 64,
        "source_count": len(policy.POSTTRAINING_SOURCE_PATHS),
        "policy_summary_sha256": policy.policy_summary_sha256(),
        "training_source_lock_sha256": policy.TRAINING_LOCK_SHA256,
        "reference_closure_lock_sha256": (
            policy.REFERENCE_CLOSURE_LOCK_SHA256
        ),
        "verified_live": True,
    }


def _report(tmp_path: Path) -> dict[str, object]:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    methods = _methods(tmp_path / "methods")
    return compare.build_report(
        methods,
        reference_binding={"fixture": True},
        input_bindings={
            "fixture": {"path": str(evidence), "sha256": _sha(evidence)}
        },
        closure_binding=_closure_binding(tmp_path),
    )


def test_policy_uses_all_twelve_aligned_locations_and_selects_one_atomic_point(
    tmp_path: Path,
) -> None:
    methods = _methods(tmp_path)
    selection = policy.select_method(methods)
    assert selection["aligned_location_count"] == 12
    assert selection["selected_method_id"] == "f_tss_qfg_dlr"
    assert selection["ranked_method_ids"][0] == "f_tss_qfg_dlr"
    assert selection["selected_outperforms_baseline_under_frozen_policy"] is True
    for summary in selection["method_summaries"].values():
        assert len(summary["location_ranks"]) == 12

    deployment = policy.select_deployment_operating_point(
        methods[selection["selected_method_id"]]
    )
    assert deployment["candidate_count"] == 12
    assert deployment["cross_checkpoint_metric_stitching"] is False
    selected = deployment["selected"]
    role = methods["f_tss_qfg_dlr"]["roles"][selected["role_name"]]
    assert selected["checkpoint_path"] == role["checkpoint_path"]
    assert selected["checkpoint_sha256"] == role["checkpoint_sha256"]
    source = policy.point_for_location(
        methods["f_tss_qfg_dlr"],
        selected["role_name"],
        selected["operating_point_source"],
    )
    assert selected["threshold"] == source["threshold"]
    assert selected["metrics"] == {
        field: float(source[field]) for field in policy.OBJECTIVE_FIELDS
    }


def test_report_compares_e_and_f_to_all_required_references(tmp_path: Path) -> None:
    report = _report(tmp_path)
    assert report["selected_method_id"] == "f_tss_qfg_dlr"
    assert report["dlr_recipe_selected"] is True
    assert report["meaningful_overall_improvement_under_frozen_policy"] is True
    assert report["cross_checkpoint_metric_stitching"] is False
    expected = {
        f"{candidate}_vs_{reference}"
        for candidate in ("e_qfg_dlr", "f_tss_qfg_dlr")
        for reference in compare.REFERENCE_METHOD_IDS
    } | {"f_tss_qfg_dlr_vs_e_qfg_dlr"}
    assert set(report["pairwise_comparisons"]) == expected
    for comparison in report["pairwise_comparisons"].values():
        assert comparison["location_count"] == 12
        assert comparison["cross_checkpoint_metric_stitching"] is False
        for location in comparison["locations"].values():
            assert location["checkpoint_local"] is True
            assert set(location["left_minus_right"]) == set(
                compare.OUTCOME_FIELDS
            )


def test_report_publication_is_write_once_and_recovers_missing_peer(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)
    json_output = tmp_path / "output" / "selection.json"
    markdown_output = tmp_path / "output" / "selection.md"
    first = compare.publish_report(
        report,
        json_output=json_output,
        markdown_output=markdown_output,
    )
    assert first["action"] == "publish"
    assert first["writes_performed"] is True
    second = compare.publish_report(
        report,
        json_output=json_output,
        markdown_output=markdown_output,
    )
    assert second["action"] == "verify"
    assert second["writes_performed"] is False

    markdown_output.unlink()
    recovered = compare.publish_report(
        report,
        json_output=json_output,
        markdown_output=markdown_output,
    )
    assert recovered["action"] == "publish"
    assert markdown_output.is_file()

    json_output.write_text('{"conflict":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="conflicts"):
        compare.publish_report(
            report,
            json_output=json_output,
            markdown_output=markdown_output,
        )


def test_closure_lock_write_once_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    training_lock = tmp_path / "training.json"
    reference_lock = tmp_path / "reference.json"
    legacy_lock = tmp_path / "legacy-v1.json"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    training_lock.write_text("{}\n", encoding="utf-8")
    reference_lock.write_text("{}\n", encoding="utf-8")
    legacy_lock.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "closure.json"
    with (
        mock.patch.object(policy, "REPO_ROOT", tmp_path),
        mock.patch.object(policy, "POSTTRAINING_SOURCE_PATHS", ("source.py",)),
        mock.patch.object(policy, "TRAINING_LOCK_PATH", training_lock),
        mock.patch.object(policy, "TRAINING_LOCK_SHA256", _sha(training_lock)),
        mock.patch.object(
            policy,
            "REFERENCE_CLOSURE_LOCK_PATH",
            reference_lock,
        ),
        mock.patch.object(
            policy,
            "REFERENCE_CLOSURE_LOCK_SHA256",
            _sha(reference_lock),
        ),
        mock.patch.object(policy, "LEGACY_V1_LOCK_PATH", legacy_lock),
        mock.patch.object(
            policy,
            "LEGACY_V1_LOCK_SHA256",
            _sha(legacy_lock),
        ),
        mock.patch.object(policy, "FINAL_EVALUATION_SOURCE_SHA256", {}),
    ):
        first = freezer.write_once(output)
        second = freezer.write_once(output)
        assert first["action"] == "write_once"
        assert first["writes_performed"] is True
        assert second["action"] == "verify"
        assert second["writes_performed"] is False
        assert freezer.verify(output)["verified"] is True


def test_dlr_deployment_calls_adapted_export_once_and_resumes_idempotently(
    tmp_path: Path,
) -> None:
    report = _report(tmp_path)
    selected = report["deployment_selection"]["selected"]
    assert selected["method_id"] == "f_tss_qfg_dlr"
    selection = tmp_path / "selection.json"
    selection.write_bytes(policy.canonical_json_bytes(report))
    artifact = tmp_path / "deployment" / "inference.pth.tar"
    manifest = tmp_path / "deployment" / "manifest.json"
    closure = _closure_binding(tmp_path)

    def fake_export(
        source: Path,
        output: Path,
        *,
        expected_variant: str | None = None,
    ) -> dict[str, object]:
        assert expected_variant == selected["variant"]
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"adapted-dlr-export")
        return {"schema": "fake"}

    def fake_validate(
        output: Path,
        *,
        expected_source_checkpoint: Path | None = None,
        expected_variant: str | None = None,
    ) -> dict[str, object]:
        assert expected_source_checkpoint is not None
        assert expected_variant == selected["variant"]
        return {
            "schema": "fake_dlr_export_v1",
            "path": str(Path(output).resolve()),
            "sha256": policy.sha256_file(output),
            "source_checkpoint_path": str(expected_source_checkpoint.resolve()),
            "source_checkpoint_sha256": selected["checkpoint_sha256"],
            "source_checkpoint_role": selected["checkpoint_role"],
            "source_checkpoint_epoch": selected["checkpoint_epoch"],
            "source_variant": selected["variant"],
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
            return_value=({}, closure),
        ),
        mock.patch.object(
            deploy.selector,
            "build_formal_report",
            return_value=report,
        ),
        mock.patch.object(
            deploy.dlr_exporter,
            "export_ramp100_qfg_checkpoint",
            side_effect=fake_export,
        ) as exporter,
        mock.patch.object(
            deploy.dlr_exporter,
            "validate_exported_ramp100_qfg_checkpoint",
            side_effect=fake_validate,
        ),
    ):
        first = deploy.publish_deployment(
            selection_path=selection,
            artifact_path=artifact,
            manifest_path=manifest,
        )
        second = deploy.publish_deployment(
            selection_path=selection,
            artifact_path=artifact,
            manifest_path=manifest,
        )
    assert first["action"] == "publish"
    assert first["artifact_written"] is True
    assert first["manifest_written"] is True
    assert second["action"] == "verify"
    exporter.assert_called_once()
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["selected_method_id"] == "f_tss_qfg_dlr"
    assert manifest_payload["selected_threshold"] == selected["threshold"]
    assert manifest_payload["export_mode"] == (
        "strict_head_free_dlr_qfg_export"
    )
    assert manifest_payload["cross_checkpoint_metric_stitching"] is False


def test_default_run_dirs_follow_the_lane_launcher_layout() -> None:
    assert "qfg_dlr_lane" in compare.DEFAULT_RUN_DIRS["e_qfg_dlr"].parts
    assert (
        "tss_qfg_dlr_lane"
        in compare.DEFAULT_RUN_DIRS["f_tss_qfg_dlr"].parts
    )
    assert compare.DEFAULT_RUN_DIRS["e_qfg_dlr"].name == (
        "seed_42_formal800_qfg_dlr_control"
    )
    assert compare.DEFAULT_RUN_DIRS["f_tss_qfg_dlr"].name == (
        "seed_42_formal800_tss_qfg_dlr_ramp100"
    )


def test_public_sweep_interface_is_explicit_and_checkpoint_local() -> None:
    interface = policy.interface_summary()
    assert interface["accepted_evaluation_schema"] == (
        policy.DLR_EVALUATION_SCHEMA
    )
    assert interface["threshold_selection_scope"] == "single_checkpoint_only"
    assert interface["cross_checkpoint_point_pooling"] is False
    assert interface["evaluated_checkpoint_count"] == 1
    assert set(policy.DLR_SWEEP_REQUIRED_FIELDS).issubset(
        interface["required_fields"]
    )
    assert dlr_evaluator.EVALUATION_SCHEMA == policy.DLR_EVALUATION_SCHEMA
    assert dlr_evaluator.DEFAULT_RUN_DIRS["qfg_dlr"] == (
        compare.DEFAULT_RUN_DIRS["e_qfg_dlr"]
    )
    assert dlr_evaluator.DEFAULT_RUN_DIRS["tss_qfg_dlr"] == (
        compare.DEFAULT_RUN_DIRS["f_tss_qfg_dlr"]
    )


def test_independent_closure_source_set_is_exact_and_not_training_sources() -> None:
    training = json.loads(policy.TRAINING_LOCK_PATH.read_text(encoding="utf-8"))
    closure_sources = set(policy.POSTTRAINING_SOURCE_PATHS)
    assert len(policy.POSTTRAINING_SOURCE_PATHS) == 12
    assert len(closure_sources) == len(policy.POSTTRAINING_SOURCE_PATHS)
    assert not closure_sources.intersection(training["source_sha256"])
    assert {
        "experiments/evaluate_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_pd_fa.py",
        "experiments/compare_tpd_ner_v4_qfg_v2_croa_dlr_ramp100.py",
        (
            "experiments/"
            "export_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_to_inference.py"
        ),
        (
            "experiments/"
            "deploy_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800.py"
        ),
        (
            "experiments/"
            "run_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_eval_lane.sh"
        ),
        (
            "experiments/"
            "run_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
            "formal800_sweeps_2x5090.sh"
        ),
    }.issubset(closure_sources)


def test_frozen_independent_closure_lock_live_verifies() -> None:
    payload, binding = policy.load_closure_lock(verify_sources=True)
    assert payload["source_count"] == 12
    assert binding["schema"] == policy.LOCK_SCHEMA
    assert binding["lock_generation"] == 2
    assert binding["sha256"] == (
        "289e9dc3c03097dd96c2c5dbe6c637b0ed5b30bfcd3a3ad3c710b8f55497e2ab"
    )
    assert "source_lock_v2" in Path(binding["path"]).name
    assert payload["supersession"] == policy.supersession_summary()
    assert payload["final_evaluation_source_sha256"] == (
        policy.FINAL_EVALUATION_SOURCE_SHA256
    )
    assert binding["training_source_lock_sha256"] == (
        policy.TRAINING_LOCK_SHA256
    )
    assert binding["reference_closure_lock_sha256"] == (
        policy.REFERENCE_CLOSURE_LOCK_SHA256
    )
    assert binding["superseded_lock_sha256"] == (
        policy.LEGACY_V1_LOCK_SHA256
    )


def test_v1_closure_lock_is_retained_but_explicitly_rejected() -> None:
    before = policy.sha256_file(policy.LEGACY_V1_LOCK_PATH)
    assert before == policy.LEGACY_V1_LOCK_SHA256
    with pytest.raises(ValueError, match="superseded"):
        policy.load_closure_lock(
            policy.LEGACY_V1_LOCK_PATH,
            verify_sources=False,
        )
    with pytest.raises(ValueError, match="superseded"):
        freezer.write_once(policy.LEGACY_V1_LOCK_PATH)
    assert policy.sha256_file(policy.LEGACY_V1_LOCK_PATH) == before


def test_v2_defaults_and_final_evaluation_sources_are_exact() -> None:
    assert policy.DEFAULT_LOCK_PATH.name.endswith("_source_lock_v2.json")
    assert compare.DEFAULT_CLOSURE_SOURCE_LOCK == policy.DEFAULT_LOCK_PATH
    assert deploy.DEFAULT_CLOSURE_SOURCE_LOCK == policy.DEFAULT_LOCK_PATH
    assert policy.FINAL_EVALUATION_SOURCE_SHA256 == {
        (
            "experiments/"
            "evaluate_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_pd_fa.py"
        ): "ae0ffd138e161db1aa20e91c8da5f884b832598454eac9577386331ddd7df90a",
        (
            "experiments/"
            "run_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_formal800_eval_lane.sh"
        ): "ec81eca1425e256a546c1f68463a968a5cdfadbec4012f75d846148f3588a588",
        (
            "experiments/"
            "run_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_"
            "formal800_sweeps_2x5090.sh"
        ): "56c79b29baceb4599f1a7002c3780b5cee8c770bae84e626c98dc4f08cb9850d",
    }
    policy.verify_final_evaluation_sources()


def test_adapted_export_is_head_free_write_once_and_source_bound(
    tmp_path: Path,
) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model, _ = build_formal_v4_qfg_v2_croa_survival_model()
        checkpoint = tmp_path / "best.pth.tar"
        output = tmp_path / "inference.pth.tar"
        payload = {
            "state_dict": model.state_dict(),
            "checkpoint_role": "best_validation_pd_primary",
            "epoch": 17,
            "variant": "tss_qfg_dlr",
            "checkpoint_identity": {"schema": "fixture"},
            "run_identity": {"schema": "fixture"},
        }
        torch.save(payload, checkpoint)
        with mock.patch.object(
            dlr_export,
            "require_ramp100_checkpoint_payload",
            side_effect=lambda value, expected_variant=None: dict(value),
        ):
            exported = dlr_export.export_ramp100_qfg_checkpoint(
                checkpoint,
                output,
                expected_variant="tss_qfg_dlr",
            )
        assert exported["schema"] == dlr_export.EXPORT_SCHEMA
        assert exported["source_variant"] == "tss_qfg_dlr"
        assert exported["source_checkpoint_sha256"] == policy.sha256_file(
            checkpoint
        )
        binding = dlr_export.validate_exported_ramp100_qfg_checkpoint(
            output,
            expected_source_checkpoint=checkpoint,
            expected_variant="tss_qfg_dlr",
        )
        assert binding["strict_load"] is True
        assert binding["survival_state_absent"] is True
        assert binding["qfg_state_preserved"] is True
        with (
            mock.patch.object(
                dlr_export,
                "require_ramp100_checkpoint_payload",
                side_effect=lambda value, expected_variant=None: dict(value),
            ),
            pytest.raises(FileExistsError),
        ):
            dlr_export.export_ramp100_qfg_checkpoint(
                checkpoint,
                output,
                expected_variant="tss_qfg_dlr",
            )
    finally:
        torch.set_num_threads(previous_threads)


def test_adapted_export_rejects_state_only_unowned_checkpoint() -> None:
    with pytest.raises(ValueError, match="variant"):
        dlr_export.require_ramp100_checkpoint_payload(
            {"state_dict": {}},
            expected_variant="qfg_dlr",
        )
