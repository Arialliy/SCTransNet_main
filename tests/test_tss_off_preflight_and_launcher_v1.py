from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from experiments import analyze_positive_tss_effective_weights_v1 as effective
from experiments import evaluate_three_dataset_tss_off_seed42_v1 as evaluator
from experiments import launch_three_dataset_tss_off_seed42_v1 as launcher
from experiments import preflight_three_dataset_tss_off_seed42_v1 as preflight
from experiments import summarize_tss_violation_types_v1 as violations
from experiments import tss_off_diagnostic_common_v1 as common


def test_real_positive_effective_lambda_seal_is_complete() -> None:
    artifact = effective.build_artifact()
    assert artifact["run_count"] == 9
    assert artifact["gate_o1"] == {
        "three_positive_lambda_results_complete": True,
        "effective_lambda_logs_complete": True,
        "source_lock_complete": True,
    }
    assert artifact["identifiability"][
        "lambda_005_vs_010_not_fully_identifiable"
    ] is True
    per_dataset = artifact["pairwise_pooled_by_dataset"]
    assert per_dataset["IRSTD-1K"]["0p005_vs_0p01"][
        "equal_batch_fraction"
    ] == pytest.approx(0.9607933333333334)
    assert per_dataset["NUDT-SIRST"]["0p005_vs_0p01"][
        "equal_batch_fraction"
    ] == pytest.approx(0.3418174603174603)
    ir_001 = artifact["runs"]["IRSTD-1K__lambda_0p01"]
    assert ir_001["effective_lambda"]["mean"] == pytest.approx(
        0.002182485927987873
    )
    assert ir_001["cap_active_batch_fraction"] == pytest.approx(0.99186)


def test_real_violation_matrices_keep_miou_and_niou_separate() -> None:
    artifact = violations.build_artifact()
    matrix = artifact["violation_type_matrix"]
    assert matrix["0.0025"]["counts"] == {
        "pd": 4,
        "tiny": 1,
        "fa": 1,
        "miou": 1,
        "niou": 1,
        "strict_domination": 0,
    }
    assert matrix["0.005"]["counts"] == {
        "pd": 2,
        "tiny": 2,
        "fa": 1,
        "miou": 0,
        "niou": 0,
        "strict_domination": 0,
    }
    assert matrix["0.01"]["counts"] == {
        "pd": 4,
        "tiny": 1,
        "fa": 2,
        "miou": 0,
        "niou": 1,
        "strict_domination": 0,
    }
    assert artifact["monotonicity"]["total"][
        "monotonic_either_direction"
    ] is False
    role_matrix = artifact["dataset_checkpoint_role_matrix"]
    assert role_matrix["NUAA-SIRST"]["best_miou"]["0.0025"]["count"] == 3
    assert role_matrix["IRSTD-1K"]["best_miou"]["0.01"]["count"] == 3


def test_real_original_reuse_audit_checks_all_data_sha() -> None:
    artifact = preflight.build_original_reuse_audit()
    assert artifact["reuse_decision"] == "REUSE_EXISTING_ORIGINALS"
    assert artifact["retrain_originals_before_tss_off"] is False
    assert set(artifact["datasets"]) == set(common.DATASETS)
    data = artifact["full_data_sha_audit"]
    assert data["record_counts"] == {
        "NUAA-SIRST": 427,
        "NUDT-SIRST": 1327,
        "IRSTD-1K": 1001,
    }
    assert data["hash_reference_count"] == 8265
    assert data["unique_file_count"] == 5511
    assert data["sha_mismatch_count"] == 0
    assert "stable_uint63" in artifact["augmentation_rng_contract"][
        "epoch_shuffle_seed"
    ]
    for record in artifact["datasets"].values():
        assert record["metric_rows"]["epochs"] == 1000
        assert record["metric_rows"]["evaluated_epochs"] == 100


def test_launcher_has_exact_gpu23_lanes_and_run_matrix() -> None:
    environment = {"PATH": "/usr/bin"}
    specs = launcher.build_worker_specs(base_environment=environment)
    assert [(spec.dataset, spec.wave, spec.gpu_index) for spec in specs] == [
        ("NUAA-SIRST", 0, "2"),
        ("NUDT-SIRST", 0, "3"),
        ("IRSTD-1K", 1, "2"),
    ]
    for spec in specs:
        record = launcher._worker_record(spec)
        assert record["method"] == "final_tss_off"
        assert record["training_model_method"] == "final"
        assert record["requested_tss_weight"] == 0.0
        assert record["resume"] == "auto"
        assert record["checkpoint_roles"] == ["best_miou", "best_pd"]
        assert str(spec.run_directory).endswith(
            f"runs/{spec.dataset}/final_tss_off/seed_42"
        )
        assert "--resume" in spec.command
        assert spec.command[spec.command.index("--resume") + 1] == "auto"
        assert spec.command[spec.command.index("--tss-weight") + 1] == "0.0"
        assert spec.environment["CUDA_VISIBLE_DEVICES"] == launcher.GPU_ASSIGNMENTS[
            spec.gpu_index
        ]["uuid"]
    lanes = {
        gpu: [
            spec.dataset
            for spec in sorted(
                (item for item in specs if item.gpu_index == gpu),
                key=lambda item: item.wave,
            )
        ]
        for gpu in ("2", "3")
    }
    assert lanes == {
        "2": ["NUAA-SIRST", "IRSTD-1K"],
        "3": ["NUDT-SIRST"],
    }


def test_launcher_posttraining_matrix_reuses_adapter_and_both_roles() -> None:
    specs = launcher.build_evaluation_specs(base_environment={"PATH": "/usr/bin"})
    assert len(specs) == 6
    assert {(spec.dataset, spec.checkpoint_role) for spec in specs} == {
        (dataset, role)
        for dataset in common.DATASETS
        for role in common.CHECKPOINT_ROLES
    }
    for spec in specs:
        assert spec.command[1] == str(launcher.EVALUATOR)
        assert spec.output_path == (
            common.tss_off_run_directory(common.TSS_OFF_RESULTS_ROOT, spec.dataset)
            / "evaluations"
            / f"{spec.checkpoint_role}.json"
        )


def test_custom_launch_plan_path_is_frozen_into_comparator_command(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "custom-launch-plan.json"
    gate = {
        "status": "complete",
        "gate_passed": True,
        "seal": {"path": "seal", "sha256": "sha"},
        "artifacts": {},
    }
    with (
        mock.patch.object(
            launcher.preflight,
            "prepare_gate_o1",
            return_value=gate,
        ),
        mock.patch.object(
            launcher,
            "static_inputs",
            return_value={"test": "static"},
        ),
    ):
        result = launcher.prepare_launch_plan(
            positive_root=tmp_path / "positive",
            results_root=tmp_path / "off",
            launch_plan_path=plan_path,
        )
    command = result["plan"]["posttraining"]["comparison"]["command"]
    assert command[command.index("--tss-off-launch-plan") + 1] == str(plan_path)
    assert result["plan"]["execution_strategy"] == (
        "two_fixed_gpu_lanes_with_automatic_continuation"
    )


def test_python_entrypoint_is_not_replaced_by_resolved_venv_target(
    tmp_path: Path,
) -> None:
    positive_root = tmp_path / "positive"
    positive_root.mkdir()
    with mock.patch.object(
        launcher,
        "_required_sources",
        return_value={"launcher": Path(launcher.__file__)},
    ):
        payload = launcher.static_inputs(
            positive_root=positive_root,
            results_root=tmp_path / "off",
            python=launcher.PYTHON,
            gate_result={},
        )
    assert payload["python_entrypoint"] == str(launcher.PYTHON.absolute())
    assert Path(payload["python"]["path"]) == launcher.PYTHON.resolve(strict=True)
    assert payload["python_entrypoint"] != payload["python"]["path"]


def test_existing_postprocessor_output_must_pass_strict_validator(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("{}", encoding="utf-8")
    validator = mock.Mock()
    launcher._run_postprocessor(
        ("command-must-not-run",),
        output,
        tmp_path / "unused.log",
        validator=validator,
    )
    validator.assert_called_once_with(output)


def test_launcher_evaluation_reuse_is_bound_to_current_checkpoint(
    tmp_path: Path,
) -> None:
    spec = launcher.build_evaluation_specs(
        results_root=tmp_path,
        base_environment={"PATH": "/usr/bin"},
    )[0]
    files = {
        "summary": spec.run_directory / "summary.json",
        "protocol": spec.run_directory / "protocol.json",
        "checkpoint": (
            spec.run_directory
            / "checkpoints"
            / launcher.CHECKPOINT_FILENAMES[spec.checkpoint_role]
        ),
    }
    for label, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(label, encoding="utf-8")
    payload = {
        "schema": "sctransnet_three_dataset_v2_evaluation_v1",
        "status": "complete",
        "dataset": spec.dataset,
        "method": "final_tss_off",
        "training_model_method": "final",
        "checkpoint_role": spec.checkpoint_role,
        "requested_tss_weight": 0.0,
        "fixed_threshold_0_5": {"threshold": 0.5},
        "tss_off_evaluator_adapter": {
            "core_evaluator_sha256": common.file_sha256(
                common.REPO_ROOT / "experiments" / "evaluate_three_dataset_v2.py"
            ),
            "adapter_sha256": common.file_sha256(launcher.EVALUATOR),
            "training_determinism_contract_reapplied": True,
            "determinism_source_sha256": common.file_sha256(
                common.REPO_ROOT
                / "experiments"
                / "train_four_dataset_original_final_seed42_exact_v1.py"
            ),
        },
        "checkpoint_binding": {
            "run_dir": str(spec.run_directory.resolve()),
            "summary": {
                "path": str(files["summary"].resolve()),
                "sha256": common.file_sha256(files["summary"]),
            },
            "protocol": {
                "path": str(files["protocol"].resolve()),
                "sha256": common.file_sha256(files["protocol"]),
            },
            "checkpoint": {
                "path": str(files["checkpoint"].resolve()),
                "sha256": common.file_sha256(files["checkpoint"]),
                "role": spec.checkpoint_role,
            },
        },
    }
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    spec.output_path.write_text(json.dumps(payload), encoding="utf-8")
    assert launcher._validate_evaluation(spec) is not None
    files["checkpoint"].write_text("changed", encoding="utf-8")
    with pytest.raises(common.TSSOffDiagnosticError, match="checkpoint SHA"):
        launcher._validate_evaluation(spec)


def test_evaluator_adapter_changes_only_admission_and_public_method() -> None:
    sentinel = {
        "schema": evaluator.core.SCHEMA,
        "status": "complete",
        "dataset": "NUAA-SIRST",
        "method": "final",
        "checkpoint_role": "best_miou",
        "requested_tss_weight": 0.0,
    }
    with (
        mock.patch.object(
            evaluator.core,
            "evaluate_run",
            return_value=dict(sentinel),
        ) as mocked,
        mock.patch.object(
            evaluator.training_engine,
            "configure_determinism",
        ) as configure_determinism,
    ):
        output = evaluator.evaluate_run(
            dataset="NUAA-SIRST",
            checkpoint_role="best_miou",
            run_dir=Path("/not/read/by/mock"),
            device_name="cpu",
        )
    request = mocked.call_args.args[0]
    assert request.method == "final"
    assert request.requested_tss_weight == 0.0
    assert evaluator.core.TRAINING_RUN_SCHEMA == evaluator.TRAINING_RUN_SCHEMA
    assert evaluator.core.TSS_CANDIDATES == (0.0,)
    assert output["method"] == "final_tss_off"
    assert output["training_model_method"] == "final"
    assert output["tss_off_evaluator_adapter"]["core_method"] == "final"
    assert output["tss_off_evaluator_adapter"]["semantic_change_to_metric_core"] is False
    assert output["tss_off_evaluator_adapter"][
        "training_determinism_contract_reapplied"
    ] is True
    configure_determinism.assert_called_once_with()


def test_evaluator_completed_output_validator_requires_adapter_lock(
    tmp_path: Path,
) -> None:
    evaluator.configure_core()
    path = tmp_path / "evaluation.json"
    payload = {
        "schema": evaluator.core.SCHEMA,
        "status": "complete",
        "dataset": "NUAA-SIRST",
        "method": "final_tss_off",
        "training_model_method": "final",
        "checkpoint_role": "best_miou",
        "requested_tss_weight": 0.0,
        "fixed_threshold_0_5": {"threshold": 0.5},
        "tss_off_evaluator_adapter": {
            "schema": evaluator.ADAPTER_SCHEMA,
            "training_determinism_contract_reapplied": True,
            "determinism_source_sha256": common.file_sha256(
                Path(evaluator.training_engine.__file__)
            ),
            "core_evaluator_sha256": common.file_sha256(
                common.REPO_ROOT / "experiments" / "evaluate_three_dataset_v2.py"
            ),
        },
        "source_sha256": {
            evaluator.RELATIVE_SOURCE: common.file_sha256(Path(evaluator.__file__))
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert evaluator.validate_completed_output(
        path,
        dataset="NUAA-SIRST",
        checkpoint_role="best_miou",
    )["method"] == "final_tss_off"
