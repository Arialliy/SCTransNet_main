from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from experiments import evaluate_three_dataset_ec_tss_v3_1 as evaluator
from experiments import launch_three_dataset_ec_tss_v3_1_seed42 as launcher
from experiments import tss_off_diagnostic_common_v1 as artifacts


def test_frozen_ec_tss_recipe_is_complete() -> None:
    assert evaluator.expected_recipe() == {
        "method": "final",
        "recipe_id": "final_ec_tss_v3_1",
        "objective_id": "ec_tss_v3_1",
        "requested_tss_weight": 0.005,
        "tss_lambda_token": "0p005",
        "tss_ratio_cap": 0.10,
        "confidence_threshold": 0.5,
        "target_dilation_radius": 3,
        "positive_normalization": "risk_mass_clamp_min_1",
        "negative_normalization": "risk_mass_clamp_min_1",
        "tss_enabled": True,
        "survival_pos_weight_used": False,
    }


def test_evaluator_validates_identity_then_reapplies_determinism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Restore the shared core globals after this adapter-specific test.
    monkeypatch.setattr(
        evaluator.core,
        "TRAINING_RUN_SCHEMA",
        evaluator.core.TRAINING_RUN_SCHEMA,
    )
    monkeypatch.setattr(
        evaluator.core,
        "TSS_CANDIDATES",
        evaluator.core.TSS_CANDIDATES,
    )
    monkeypatch.setattr(
        evaluator.core,
        "_EVALUATOR_NON_MODEL_SOURCES",
        evaluator.core._EVALUATOR_NON_MODEL_SOURCES,
    )
    order: list[str] = []
    binding = {"run_dir": "/unused", "recipe": evaluator.expected_recipe()}
    captured: dict[str, object] = {}

    def validate_identity(*args: object, **kwargs: object) -> dict[str, object]:
        order.append("identity")
        return binding

    def configure_determinism() -> None:
        order.append("determinism")

    def core_evaluate(request: object, **kwargs: object) -> dict[str, object]:
        order.append("core")
        captured["request"] = request
        return {
            "schema": evaluator.core.SCHEMA,
            "status": "complete",
            "dataset": "NUAA-SIRST",
            "method": "final",
            "checkpoint_role": "best_miou",
            "requested_tss_weight": 0.005,
        }

    monkeypatch.setattr(
        evaluator,
        "_validate_training_identity",
        validate_identity,
    )
    monkeypatch.setattr(
        evaluator.training_engine,
        "configure_determinism",
        configure_determinism,
    )
    monkeypatch.setattr(evaluator.core, "evaluate_run", core_evaluate)

    output = evaluator.evaluate_run(
        dataset="NUAA-SIRST",
        checkpoint_role="best_miou",
        run_dir=Path("/not-read-by-mocks"),
        device_name="cpu",
    )

    assert order == ["identity", "determinism", "core"]
    request = captured["request"]
    assert request.method == "final"
    assert request.requested_tss_weight == 0.005
    assert evaluator.core.TRAINING_RUN_SCHEMA == evaluator.TRAINING_RUN_SCHEMA
    assert evaluator.core.TSS_CANDIDATES == (0.005,)
    assert output["method"] == "final_ec_tss_v3_1"
    assert output["training_model_method"] == "final"
    assert output["ec_tss_v3_1_training_identity_binding"] == binding
    adapter = output["ec_tss_v3_1_evaluator_adapter"]
    assert adapter["training_determinism_contract_reapplied"] is True
    assert adapter["semantic_change_to_metric_core"] is False


def test_pre_core_identity_check_rejects_objective_parameter_drift(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    recipe = evaluator.expected_recipe()
    protocol = {
        "schema": evaluator.TRAINING_RUN_SCHEMA,
        "dataset": "NUAA-SIRST",
        "method": "final",
        "objective_id": evaluator.OBJECTIVE_ID,
        "recipe": recipe,
        "training_seed": 42,
        "epochs": 1000,
        "planned_total_epochs": 1000,
        "begin_test": 10,
        "eval_every": 10,
        "training": {
            "objective_id": evaluator.OBJECTIVE_ID,
            "tss_requested_weight": 0.005,
            "tss_ratio_cap": 0.10,
            "confidence_threshold": 0.5,
            "target_dilation_radius": 3,
            "positive_normalization": "risk_mass_clamp_min_1",
            "negative_normalization": "risk_mass_clamp_min_1",
            "survival_pos_weight_used": False,
        },
        "metrics": {"threshold": 0.5},
    }
    protocol["protocol_sha256"] = artifacts.compact_sha256(protocol)
    summary = {
        "schema": evaluator.TRAINING_RUN_SCHEMA,
        "status": "complete",
        "dataset": "NUAA-SIRST",
        "method": "final",
        "objective_id": evaluator.OBJECTIVE_ID,
        "recipe": recipe,
        "seed": 42,
        "epochs": 1000,
        "planned_total_epochs": 1000,
        "protocol_sha256": protocol["protocol_sha256"],
    }
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    binding = evaluator._validate_training_identity(
        run_dir,
        dataset="NUAA-SIRST",
    )
    assert binding["recipe"] == recipe

    protocol["training"]["target_dilation_radius"] = 4
    protocol.pop("protocol_sha256")
    protocol["protocol_sha256"] = artifacts.compact_sha256(protocol)
    summary["protocol_sha256"] = protocol["protocol_sha256"]
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    with pytest.raises(
        artifacts.TSSOffDiagnosticError,
        match="training target_dilation_radius differs",
    ):
        evaluator._validate_training_identity(
            run_dir,
            dataset="NUAA-SIRST",
        )


def test_completed_evaluation_revalidates_bound_training_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = run_dir / "summary.json"
    protocol = run_dir / "protocol.json"
    checkpoint = run_dir / "checkpoints" / "best_pd.pth.tar"
    checkpoint.parent.mkdir()
    summary.write_text("summary", encoding="utf-8")
    protocol.write_text("protocol", encoding="utf-8")
    checkpoint.write_text("checkpoint", encoding="utf-8")
    identity = {
        "run_dir": str(run_dir),
        "summary": artifacts.artifact_record(summary),
        "protocol": artifacts.artifact_record(protocol),
        "protocol_payload_sha256": "protocol-payload",
        "recipe": evaluator.expected_recipe(),
    }
    monkeypatch.setattr(
        evaluator,
        "_validate_training_identity",
        lambda *args, **kwargs: identity,
    )
    payload = {
        "schema": evaluator.core.SCHEMA,
        "status": "complete",
        "dataset": "NUAA-SIRST",
        "method": evaluator.OUTPUT_METHOD,
        "training_model_method": "final",
        "checkpoint_role": "best_pd",
        "requested_tss_weight": 0.005,
        "fixed_threshold_0_5": {"threshold": 0.5},
        "ec_tss_v3_1_evaluator_adapter": {
            "schema": evaluator.ADAPTER_SCHEMA,
            "training_run_schema": evaluator.TRAINING_RUN_SCHEMA,
            "objective_id": evaluator.OBJECTIVE_ID,
            "recipe_id": evaluator.RECIPE_ID,
            "requested_tss_weight": 0.005,
            "survival_ratio_cap": 0.10,
            "confidence_threshold": 0.5,
            "target_dilation_radius": 3,
            "training_determinism_contract_reapplied": True,
            "determinism_source_sha256": artifacts.file_sha256(
                Path(evaluator.training_engine.__file__)
            ),
            "core_evaluator_sha256": artifacts.file_sha256(
                evaluator.REPO_ROOT
                / "experiments"
                / "evaluate_three_dataset_v2.py"
            ),
            "adapter_sha256": artifacts.file_sha256(Path(evaluator.__file__)),
        },
        "source_sha256": {
            evaluator.RELATIVE_SOURCE: artifacts.file_sha256(
                Path(evaluator.__file__)
            )
        },
        "ec_tss_v3_1_training_identity_binding": identity,
        "checkpoint_binding": {
            "run_dir": str(run_dir),
            "summary": identity["summary"],
            "protocol": identity["protocol"],
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": artifacts.file_sha256(checkpoint),
                "role": "best_pd",
            },
        },
    }
    output = tmp_path / "evaluation.json"
    output.write_text(json.dumps(payload), encoding="utf-8")
    assert evaluator.validate_completed_output(
        output,
        dataset="NUAA-SIRST",
        checkpoint_role="best_pd",
    ) == payload

    payload["ec_tss_v3_1_training_identity_binding"] = {
        **identity,
        "protocol_payload_sha256": "changed",
    }
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        artifacts.TSSOffDiagnosticError,
        match="training identity binding differs",
    ):
        evaluator.validate_completed_output(
            output,
            dataset="NUAA-SIRST",
            checkpoint_role="best_pd",
        )


def test_launcher_uses_three_distinct_single_gpu_training_lanes() -> None:
    specs = launcher.build_worker_specs(base_environment={"PATH": "/usr/bin"})
    assert [(spec.dataset, spec.gpu_index) for spec in specs] == [
        ("NUAA-SIRST", "0"),
        ("NUDT-SIRST", "1"),
        ("IRSTD-1K", "2"),
    ]
    assert len({spec.dataset for spec in specs}) == 3
    for spec in specs:
        record = launcher._worker_record(spec)
        assert record["single_gpu"] is True
        assert record["ddp"] is False
        assert record["planned_total_epochs"] == 1000
        assert record["pause_after_epoch"] == 200
        assert record["pilot_resume"] == "auto"
        assert record["formal_resume"] == "required"
        assert spec.pilot_command[spec.pilot_command.index("--epochs") + 1] == "1000"
        assert (
            spec.pilot_command[
                spec.pilot_command.index("--pause-after-epoch") + 1
            ]
            == "200"
        )
        assert "--pause-after-epoch" not in spec.resume_command
        assert (
            spec.resume_command[spec.resume_command.index("--resume") + 1]
            == "required"
        )
        assert spec.environment["CUDA_VISIBLE_DEVICES"] == (
            launcher.GPU_ASSIGNMENTS[spec.gpu_index]["uuid"]
        )


def test_gpu3_is_reserved_for_screen_and_posttraining_evaluation() -> None:
    environment = {"PATH": "/usr/bin"}
    smoke = launcher.build_smoke_scale_spec(base_environment=environment)
    assert smoke.gpu_index == "3"
    assert "screen_gpu3/smoke/runs/NUAA-SIRST" in str(smoke.run_directory)
    assert "--smoke" in smoke.pause_command
    assert smoke.pause_command[smoke.pause_command.index("--epochs") + 1] == "2"
    assert (
        smoke.pause_command[smoke.pause_command.index("--pause-after-epoch") + 1]
        == "1"
    )
    smoke_record = launcher._smoke_record(smoke)
    assert smoke_record["formal_run"] is False
    assert smoke_record["counts_toward_formal_run_budget"] is False
    assert "strict_model_and_optimizer_resume" in smoke_record["coverage"]

    evaluations = launcher.build_evaluation_specs(base_environment=environment)
    assert len(evaluations) == 6
    assert {spec.gpu_index for spec in evaluations} == {"3"}
    assert {(spec.dataset, spec.checkpoint_role) for spec in evaluations} == {
        (dataset, role)
        for dataset in launcher.DATASETS
        for role in launcher.CHECKPOINT_ROLES
    }


def test_launch_plan_freezes_global_pause_gate_and_defers_comparator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "launch_plan.json"
    monkeypatch.setattr(
        launcher,
        "static_inputs",
        lambda **kwargs: {
            "python_entrypoint": str(launcher.PYTHON),
            "results_root": str(tmp_path / "results"),
            "sources": {},
        },
    )
    result = launcher.prepare_launch_plan(
        results_root=tmp_path / "results",
        launch_plan_path=plan_path,
    )
    plan = result["plan"]
    assert plan["execution_strategy"] == (
        "three_single_gpu_training_lanes_global_pause200_then_resume1000"
    )
    assert plan["training_lane_count"] == 3
    assert plan["no_ddp"] is True
    assert plan["no_duplicate_runs"] is True
    assert plan["pilot_gate"] == {
        "global": True,
        "all_three_prefixes_required": True,
        "planned_total_epochs": 1000,
        "pause_after_epoch": 200,
        "pass_then_resume_same_run": True,
        "resume_mode": "required",
        "output": str(
            tmp_path / "results" / "pilot_gate" / "pilot200_runtime_gate.json"
        ),
    }
    assert plan["gpu3_smoke_scale_screen"]["physical_gpu_index"] == "3"
    assert plan["posttraining"]["evaluation_gpu"] == 3
    assert plan["posttraining"]["comparison"]["status"] == (
        "deferred_to_compare_finalize_ec_tss_v3_1"
    )


def test_diagnostic_lookup_accepts_runner_mean_field_names() -> None:
    row = {
        "train_ec_tss_positive_risk_mass_mean": 1.25,
        "train_ec_tss_effective_weighted_to_segmentation_ratio_mean": 0.10,
    }
    assert launcher._diagnostic_value(row, "positive_risk_mass_mean") == 1.25
    assert launcher._diagnostic_value(
        row,
        "effective_weighted_to_segmentation_ratio_mean",
    ) == 0.10


def test_existing_passed_pilot_gate_is_reused_after_supervisor_restart(
    tmp_path: Path,
) -> None:
    gate = {
        "schema": "sctransnet_three_dataset_ec_tss_v3_1_pilot200_gate_v1",
        "status": "passed",
        "gate_passed": True,
        "objective_id": "ec_tss_v3_1",
        "planned_total_epochs": 1000,
        "pause_after_epoch": 200,
        "all_three_runs_checked": True,
        "gpu3_smoke_scale": {"status": "passed"},
        "datasets": {dataset: {} for dataset in launcher.DATASETS},
    }
    path = tmp_path / "pilot_gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    assert launcher._run_pilot_phase(
        (),
        smoke=mock.sentinel.smoke,
        status_path=tmp_path / "unused_status.json",
        plan_record={},
        gate_path=path,
    ) == gate
