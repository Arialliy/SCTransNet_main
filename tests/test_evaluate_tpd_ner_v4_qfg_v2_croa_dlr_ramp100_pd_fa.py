from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from experiments import tpd_exact_runner as exact_runner
from experiments import (
    evaluate_tpd_ner_v4_qfg_v2_croa_dlr_ramp100_pd_fa as evaluation,
)


def synthetic_point(
    threshold: float,
    *,
    matched: int,
    matched_tiny: int,
    unmatched: int,
    fa: float,
    miou: float,
) -> dict[str, object]:
    return {
        "val_loss": 0.001,
        "miou": miou,
        "niou": miou,
        "pixel_precision": 0.9,
        "pixel_recall": 0.9,
        "pixel_f1": 0.9,
        "pd": matched / evaluation.EXPECTED_TARGET_COUNT,
        "tiny_pd": matched_tiny / evaluation.EXPECTED_TINY_TARGET_COUNT,
        "fa": fa,
        "false_objects_per_image": (
            unmatched / evaluation.EXPECTED_VALIDATION_COUNT
        ),
        "target_count": evaluation.EXPECTED_TARGET_COUNT,
        "matched_target_count": matched,
        "tiny_target_count": evaluation.EXPECTED_TINY_TARGET_COUNT,
        "matched_tiny_target_count": matched_tiny,
        "predicted_object_count": matched + unmatched,
        "unmatched_predicted_object_count": unmatched,
        "valid_pixel_count": 8_716_288,
        "threshold": threshold,
    }


def synthetic_sweep() -> tuple[dict[str, object], dict[str, object]]:
    fixed = synthetic_point(
        0.5,
        matched=188,
        matched_tiny=39,
        unmatched=6,
        fa=4.0e-6,
        miou=0.91,
    )
    strict = synthetic_point(
        0.9,
        matched=187,
        matched_tiny=39,
        unmatched=0,
        fa=0.0,
        miou=0.90,
    )
    last = synthetic_point(
        evaluation.LAST_FLOAT32_BELOW_ONE,
        matched=0,
        matched_tiny=0,
        unmatched=0,
        fa=0.0,
        miou=0.0,
    )
    upper = synthetic_point(
        evaluation.UPPER_BOUNDARY_THRESHOLD,
        matched=0,
        matched_tiny=0,
        unmatched=0,
        fa=0.0,
        miou=0.0,
    )
    points = [fixed, strict, last, upper]
    checkpoint_metrics = {
        name: copy.deepcopy(fixed[name])
        for name in evaluation.exact.STORED_VALIDATION_METRICS
    }
    payload: dict[str, object] = {
        "run_directory": "/tmp/ramp100",
        "checkpoint": "/tmp/ramp100/best.pth.tar",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_epoch": 100,
        "checkpoint_role": "best_validation_pd_primary",
        "checkpoint_validation_metrics": checkpoint_metrics,
        "variant": evaluation.QFG_DLR_VARIANT,
        "dataset": evaluation.DATASET,
        "seed": evaluation.TRAINING_SEED,
        "split_seed": evaluation.SPLIT_SEED,
        "validation_count": evaluation.EXPECTED_VALIDATION_COUNT,
        "validation_split_sha256": "b" * 64,
        "official_test_accessed": False,
        "match_radius": evaluation.FORMAL_MATCH_RADIUS,
        "tiny_area": evaluation.FORMAL_TINY_AREA,
        "threshold_configuration": {
            "threshold_min": 0.01,
            "threshold_max": 0.99,
            "threshold_step": 0.01,
            "extra_thresholds": list(evaluation.EXTRA_THRESHOLDS),
            "tail_logit_step": 0.1,
            "fa_budgets": list(evaluation.FA_BUDGETS),
        },
        "threshold_provenance": {
            "posthoc_endpoint_completion": False,
            "preregistered_endpoint_completion": True,
            "endpoint_protocol_stage": "before_formal_training",
            "closed_probability_interval": True,
            "score_dtype": "float32",
            "score_count": 8_716_288,
            "added_thresholds": [
                evaluation.LAST_FLOAT32_BELOW_ONE,
                evaluation.UPPER_BOUNDARY_THRESHOLD,
            ],
            "last_float32_below_one": (
                evaluation.LAST_FLOAT32_BELOW_ONE
            ),
            "upper_boundary_threshold": (
                evaluation.UPPER_BOUNDARY_THRESHOLD
            ),
            "upper_boundary_comparison": "prediction > threshold",
            "upper_boundary_semantics": "empty_prediction_pd0_fa0",
            "total_unique_threshold_count": len(points),
        },
        "fixed_threshold_0_5": copy.deepcopy(fixed),
        "fixed_threshold_0_5_checkpoint_audit": (
            evaluation.qfg_evaluator.v4_evaluator
            ._fixed_threshold_checkpoint_audit(
                fixed,
                checkpoint_metrics,
            )
        ),
        "best_points_under_fa_budget": {
            "1e-06": copy.deepcopy(strict),
            "5e-06": copy.deepcopy(fixed),
            "1e-05": copy.deepcopy(fixed),
            "5e-05": copy.deepcopy(fixed),
            "0.0001": copy.deepcopy(fixed),
        },
        "points": points,
        "audit": {
            "expected_epochs": 800,
            "metrics_event_count": 800,
            "metrics_epoch_range": [1, 800],
            "summary_status": "complete",
            "selection_source": "internal_validation_only",
            "integrity_checks_passed": {
                name: True
                for name in (
                    evaluation.qfg_evaluator.v4_evaluator
                    .REQUIRED_INTEGRITY_CHECKS
                )
            }
        },
    }
    return payload, checkpoint_metrics


def synthetic_audit(
    checkpoint_metrics: dict[str, object],
) -> dict[str, object]:
    candidate = evaluation.exact.candidate_contract(
        evaluation.QFG_DLR_VARIANT
    )
    return {
        "run_directory": "/tmp/ramp100",
        "variant": evaluation.QFG_DLR_VARIANT,
        "candidate_contract": candidate,
        "checkpoint_filename": "best.pth.tar",
        "checkpoint_path": "/tmp/ramp100/best.pth.tar",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_epoch": 100,
        "checkpoint_role": "best_validation_pd_primary",
        "checkpoint_validation_metrics": checkpoint_metrics,
        "checkpoint_identity": {"schema": "checkpoint"},
        "checkpoint_survival_weight_effective": 0.0,
        "checkpoint_tss_ramp_fraction": 0.0,
        "validation_split_sha256": "b" * 64,
        "run_identity": {"schema": "run"},
        "source_binding": {"schema": "binding"},
        "state_dict_strict_load": True,
        "adapter_payload_strict": True,
        "legacy_eval_output_verified": True,
    }


def test_constants_and_lane_root_defaults_match_training_launcher() -> None:
    assert evaluation.SUPPORTED_VARIANTS == (
        "qfg_dlr",
        "tss_qfg_dlr",
    )
    assert evaluation.FA_BUDGETS == (
        1e-6,
        5e-6,
        1e-5,
        5e-5,
        1e-4,
    )
    assert evaluation.FIXED_THRESHOLD == 0.5
    assert evaluation.FORMAL_MATCH_RADIUS == 3.0
    assert evaluation.FORMAL_TINY_AREA == 9
    assert "qfg_dlr_lane" in str(
        evaluation.DEFAULT_RUN_DIRS[evaluation.QFG_DLR_VARIANT]
    )
    assert "tss_qfg_dlr_lane" in str(
        evaluation.DEFAULT_RUN_DIRS[evaluation.TSS_QFG_DLR_VARIANT]
    )
    assert str(
        evaluation.DEFAULT_RUN_DIRS[evaluation.QFG_DLR_VARIANT]
    ).endswith(
        "NUDT-SIRST/qfg_dlr/"
        "seed_42_formal800_qfg_dlr_control"
    )
    assert str(
        evaluation.DEFAULT_RUN_DIRS[evaluation.TSS_QFG_DLR_VARIANT]
    ).endswith(
        "NUDT-SIRST/tss_qfg_dlr/"
        "seed_42_formal800_tss_qfg_dlr_ramp100"
    )


def test_cli_freezes_checkpoint_local_formal_policy() -> None:
    run_dir = evaluation.DEFAULT_RUN_DIRS[evaluation.QFG_DLR_VARIANT]
    args = evaluation.validate_formal_arguments(
        [
            "--run-dir",
            str(run_dir),
            "--checkpoint",
            "best_miou.pth.tar",
            "--device",
            "cpu",
            "--preflight",
        ]
    )
    request = evaluation.evaluation_request(args)
    assert request.variant == evaluation.QFG_DLR_VARIANT
    assert request.checkpoint == "best_miou.pth.tar"
    for override, pattern in (
        (["--fa-budgets", "2e-6"], "fa_budgets"),
        (["--match-radius", "2"], "match_radius"),
        (["--tiny-area", "16"], "tiny_area"),
        (["--overwrite"], "forbids --overwrite"),
    ):
        with pytest.raises(ValueError, match=pattern):
            evaluation.validate_formal_arguments(
                ["--run-dir", str(run_dir), *override]
            )


def test_core_reuse_and_contract_forbid_cross_checkpoint_pooling() -> None:
    assert (
        evaluation._normalize_budgets
        is evaluation.qfg_evaluator._normalize_budgets
    )
    assert (
        evaluation._validate_closed_interval
        is evaluation.qfg_evaluator._validate_closed_interval
    )
    with mock.patch.object(
        evaluation,
        "source_binding",
        return_value={
            "training_source_lock": {"sha256": "a" * 64},
            "shared_metric_core": {"sha256": "b" * 64},
            "closed_interval_core": {"sha256": "c" * 64},
        },
    ):
        contract = evaluation.evaluator_contract()
    assert contract["expected_sweep_count"] == 4
    assert contract["threshold_selection_scope"] == "single_checkpoint_only"
    assert contract["cross_checkpoint_point_pooling"] is False
    assert contract["cross_checkpoint_overwrite"] is False


def test_live_source_binding_verifies_frozen_51_source_lock() -> None:
    binding = evaluation.source_binding()
    assert binding["schema"] == evaluation.SOURCE_BINDING_SCHEMA
    assert binding["training_source_lock"]["source_count"] == 51
    assert (
        binding["training_source_lock"]["sha256"]
        == "88b4839b40484c881544614e60675c4d2805a4fd6de1cc2f0aad28bdcb1395e8"
    )
    assert (
        binding["trainer"]["sha256"]
        == "b2c6842aaa7674ff8f23ff09034d22e5491f36825232e33ec2559c1e0b76e3f6"
    )
    for name in (
        "trainer",
        "evaluator",
        "shared_metric_core",
        "closed_interval_core",
        "determinism_core",
    ):
        assert len(binding[name]["sha256"]) == 64


def test_finalize_preserves_public_metrics_and_own_checkpoint_identity() -> None:
    payload, checkpoint_metrics = synthetic_sweep()
    audit = synthetic_audit(checkpoint_metrics)
    with (
        mock.patch.object(
            evaluation,
            "source_binding",
            return_value={"schema": "binding"},
        ),
        mock.patch.object(
            evaluation,
            "evaluator_contract",
            return_value={"schema": "contract"},
        ),
    ):
        ready = evaluation.finalize_evaluation_output(
            payload,
            audit,
            device_assignment={
                "device": "cpu",
                "physical_gpu_index": None,
            },
        )
    assert ready["schema"] == evaluation.EVALUATION_SCHEMA
    assert ready["validation_split_sha256"] == "b" * 64
    assert ready["evaluated_checkpoint_count"] == 1
    assert ready["own_checkpoint_selection_verified"] is True
    assert ready["threshold_selection_scope"] == "single_checkpoint_only"
    assert ready["cross_checkpoint_point_pooling"] is False
    assert tuple(ready["best_points_under_fa_budget"]) == (
        evaluation.BUDGET_KEYS
    )
    assert ready["run_identity"] == {"schema": "run"}
    assert ready["source_checkpoint_identity"] == {"schema": "checkpoint"}
    coverage = ready["final_metric_coverage"]
    assert coverage["required_metrics"] == [
        "pd",
        "fa",
        "miou",
        "false_objects_per_image",
        "tiny_pd",
    ]


def test_budget_point_cannot_be_borrowed_from_another_checkpoint() -> None:
    payload, checkpoint_metrics = synthetic_sweep()
    foreign = copy.deepcopy(
        payload["best_points_under_fa_budget"]["1e-06"]
    )
    foreign["miou"] = 0.123
    payload["best_points_under_fa_budget"]["1e-06"] = foreign
    with pytest.raises(ValueError, match="best point"):
        evaluation.finalize_evaluation_output(
            payload,
            synthetic_audit(checkpoint_metrics),
            device_assignment={"device": "cpu"},
        )


def test_atomic_output_is_write_once_and_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "pd_fa_sweep_best.pth.json"
    with mock.patch.object(
        evaluation,
        "finalize_evaluation_output",
        return_value={"status": "complete"},
    ):
        evaluation._atomic_write_output(
            path,
            {},
            False,
            artifact_audit={},
            device_assignment={},
            json_ready=lambda value: value,
        )
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "status": "complete"
        }
        with pytest.raises(FileExistsError, match="refusing to replace"):
            evaluation._atomic_write_output(
                path,
                {},
                False,
                artifact_audit={},
                device_assignment={},
                json_ready=lambda value: value,
            )


def test_existing_output_is_revalidated_without_rewrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pd_fa_sweep_best.pth.json"
    path.write_text('{"schema":"fixture","audit":{"device_assignment":{}}}\n')
    before = evaluation._sha256_file(path)
    with mock.patch.object(
        evaluation,
        "validate_output_identity",
    ) as validate:
        payload = evaluation.validate_existing_output(
            path,
            artifact_audit={},
            device_assignment={},
        )
    validate.assert_called_once()
    assert payload["schema"] == "fixture"
    assert evaluation._sha256_file(path) == before


class StateScaler:
    def state_dict(self) -> dict[str, int]:
        return {"updates": 0}


def _source_locks() -> dict[str, str]:
    exact = evaluation.exact
    statistics = exact.load_survival_target_statistics()
    return {
        exact.SOURCE_LOCK_KEY: "1" * 64,
        exact.UPSTREAM_SOURCE_LOCK_KEY: exact.UPSTREAM_SOURCE_LOCK_SHA256,
        "training_data": "2" * 64,
        "survival_target_statistics": statistics["sha256"],
        "parent_checkpoint": exact.PARENT_CHECKPOINT_SHA256,
    }


def _extension_provenance() -> dict[str, object]:
    exact = evaluation.exact
    return {
        "schema": exact.v2.EXTENSION_WARM_START_SCHEMA,
        "parent_checkpoint_path": str(exact.PARENT_CHECKPOINT_PATH.resolve()),
        "parent_checkpoint_sha256": exact.PARENT_CHECKPOINT_SHA256,
        "parent_state_dict_path": list(exact.v2.PARENT_STATE_DICT_PATH),
        "parent_state_key_count": exact.v2.FORMAL_PARENT_STATE_KEY_COUNT,
        "preserved_new_state_key_count": (
            len(exact.v2.SURVIVAL_STATE_KEYS)
            + len(exact.v2.QFG_STATE_KEYS)
        ),
        "new_module_prefixes": list(exact.v2.QFG_NEW_MODULE_PREFIXES),
        "zero_init_prefixes": list(exact.v2.QFG_ZERO_INIT_PREFIXES),
    }


def _checkpoint_fixture() -> dict[str, object]:
    exact = evaluation.exact
    variant = exact.TSS_QFG_DLR_VARIANT
    args = exact.parse_args(
        [
            "--variant",
            variant,
            "--device",
            "cpu",
            "--allow-cpu-smoke",
            "--fresh",
        ]
    )
    model, metadata = exact.build_selected_model(variant, 42)
    optimizer = exact.build_optimizer(model)
    scaler = StateScaler()
    initial_sha = exact_runner.initial_model_state_sha256(model)
    split_records = {
        "train": exact_runner.OrderedFingerprint.from_values(
            "train", ("a", "b")
        ),
        "validation": exact_runner.OrderedFingerprint.from_values(
            "validation", ("c",)
        ),
    }
    data_records = {
        "train_samples": exact_runner.OrderedFingerprint.from_values(
            "train_samples", ("a:image", "b:image")
        ),
        "validation_samples": exact_runner.OrderedFingerprint.from_values(
            "validation_samples", ("c:image",)
        ),
    }
    spec = exact.make_exact_run_spec(
        args,
        model=model,
        model_metadata=metadata,
        optimizer=optimizer,
        scaler=scaler,
        initialization_contract=(
            exact_runner.extension_parent_initialization_contract(
                _extension_provenance(),
                loaded_child_model_state_sha256=initial_sha,
            )
        ),
        initial_model_state_sha256=initial_sha,
        initial_rng=exact_runner.initial_rng_contract(),
        selection_policy=exact_runner.pd_miou_selection_policy(
            stored_metrics=exact.STORED_VALIDATION_METRICS
        ).normalized(),
        source_locks=_source_locks(),
        split_records=split_records,
        data_records=data_records,
        environment={"name": "ramp100-evaluator-cpu-fixture"},
    )
    identity = exact_runner.build_run_identity(model, spec)
    count_fields = {
        "target_count",
        "matched_target_count",
        "tiny_target_count",
        "matched_tiny_target_count",
        "predicted_object_count",
        "unmatched_predicted_object_count",
        "valid_pixel_count",
    }
    metrics = {
        name: 1 if name in count_fields else 0.5
        for name in exact.STORED_VALIDATION_METRICS
    }
    context = exact_runner.CompatibilityPayloadContext(
        role="best_validation_pd_primary",
        epoch=100,
        metrics=metrics,
        event={},
        exact_payload={
            "model": {"state_dict": model.state_dict()},
            "optimizer": {"state_dict": optimizer.state_dict()},
            "scaler": {"state_dict": scaler.state_dict()},
        },
        run_identity=identity,
        normalized_spec=spec.normalized(),
    )
    payload = dict(
        exact.EvaluatorCheckpointAdapter(
            model_metadata=metadata,
            split_hashes={"used_val_sha256": "a" * 64},
        )(context)
    )
    payload["derived_schema"] = exact_runner.DERIVED_CHECKPOINT_SCHEMA
    payload["source_exact_checkpoint_sha256"] = "b" * 64
    for field, component in (
        ("state_dict_sha256", payload["state_dict"]),
        ("optimizer_state_sha256", payload["optimizer"]),
        ("scaler_state_sha256", payload["scaler"]),
    ):
        payload[field] = exact_runner._state_content_sha256(
            component,
            f"fixture {field}",
        )
    return payload


def test_strict_checkpoint_validator_replays_ramp100_adapter() -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        payload = _checkpoint_fixture()
        validated = evaluation._require_checkpoint_payload(
            payload,
            expected_variant=evaluation.TSS_QFG_DLR_VARIANT,
        )
        assert validated["scheduler"]["kind"] == (
            "identity_bound_manual_group_scaled_schedule"
        )
        assert validated[evaluation.exact.SURVIVAL_WEIGHT_FIELD] == 0.005
        tampered = copy.deepcopy(payload)
        tampered["scheduler"]["tss_weight_state"] = "serialized"
        with pytest.raises(ValueError, match="adapter field differs"):
            evaluation._require_checkpoint_payload(
                tampered,
                expected_variant=evaluation.TSS_QFG_DLR_VARIANT,
            )
    finally:
        torch.set_num_threads(previous_threads)


def test_main_preflight_never_configures_cuda_or_evaluates() -> None:
    args = argparse.Namespace(preflight=True)
    request = evaluation.EvaluationRequest(
        evaluation.QFG_DLR_VARIANT,
        Path("/tmp/qfg_dlr"),
        "best.pth.tar",
    )
    with (
        mock.patch.object(
            evaluation,
            "validate_formal_arguments",
            return_value=args,
        ),
        mock.patch.object(
            evaluation,
            "evaluation_request",
            return_value=request,
        ),
        mock.patch.object(
            evaluation,
            "validate_run_artifacts",
            return_value={"checkpoint_epoch": 7},
        ),
        mock.patch.object(
            evaluation,
            "configure_v8_inference",
        ) as configure,
        mock.patch.object(evaluation, "evaluate_one") as evaluate,
    ):
        evaluation.main([])
    configure.assert_not_called()
    evaluate.assert_not_called()
