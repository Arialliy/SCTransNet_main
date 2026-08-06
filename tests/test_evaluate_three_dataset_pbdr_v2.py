from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from experiments import evaluate_three_dataset_pbdr_v2 as evaluator
from experiments import three_dataset_pbdr_v2_models_seed42_v1 as models
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_qfg_v2_croa_pbdr_v2 import (
    PBDR_V2_INTEGRATION_VERSION,
    PBDR_V2_STATE_KEYS,
)
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import (
    SURVIVAL_STATE_KEYS,
)


torch.set_num_threads(1)


@pytest.fixture(scope="module")
def training_model():
    model, metadata = models.build_pbdr_v2_training_model("NUAA-SIRST", 42)
    return model, metadata


def _final_request() -> evaluator.core.EvaluationRequest:
    return evaluator.core.EvaluationRequest(
        dataset="NUAA-SIRST",
        method="final",
        checkpoint_role="best_miou",
        requested_tss_weight=0.0,
    )


def test_evaluator_routes_through_exact_registry_conversion(training_model) -> None:
    training, _metadata = training_model
    training_state = training.state_dict()
    inference, metadata = evaluator._build_inference_model(
        _final_request(),
        training_state,
    )
    inference_state = inference.state_dict()
    assert len(training_state) == 573
    assert len(inference_state) == 569
    assert set(training_state) - set(inference_state) == set(SURVIVAL_STATE_KEYS)
    assert set(PBDR_V2_STATE_KEYS) <= set(inference_state)
    assert all(
        torch.equal(training_state[key], inference_state[key])
        for key in PBDR_V2_STATE_KEYS
    )
    assert metadata["strict_load"] is True
    assert metadata["training_state_key_count"] == 573
    assert metadata["inference_state_key_count"] == 569
    assert metadata["pbdr_v2_state_preserved"] == list(PBDR_V2_STATE_KEYS)


def test_builder_call_and_metadata_are_strict(training_model) -> None:
    training, _metadata = training_model
    state = training.state_dict()
    sentinel = object()
    valid = {
        "strict_load": True,
        "training_state_key_count": 573,
        "inference_state_key_count": 569,
        "stripped_state_key_count": 4,
        "target_survival_registered": False,
        "pbdr_v2_state_preserved": list(PBDR_V2_STATE_KEYS),
        "architecture_manifest": {
            "pbdr_v2_integration_version": PBDR_V2_INTEGRATION_VERSION,
            "target_survival_registered": False,
        },
    }
    with mock.patch.object(
        evaluator.models,
        "build_pbdr_v2_inference_model_from_training_state_dict",
        return_value=(sentinel, valid),
    ) as builder:
        built, observed = evaluator._build_inference_model(
            _final_request(), state
        )
    assert built is sentinel and observed == valid
    builder.assert_called_once_with(
        state,
        dataset_name="NUAA-SIRST",
        seed=42,
    )

    invalid = dict(valid)
    invalid["inference_state_key_count"] = 568
    with mock.patch.object(
        evaluator.models,
        "build_pbdr_v2_inference_model_from_training_state_dict",
        return_value=(sentinel, invalid),
    ):
        with pytest.raises(ValueError, match="inference_state_key_count"):
            evaluator._build_inference_model(_final_request(), state)


def test_evaluator_rejects_non_final_method(training_model) -> None:
    training, _metadata = training_model
    request = evaluator.core.EvaluationRequest(
        dataset="NUAA-SIRST",
        method="original",
        checkpoint_role="best_pd",
        requested_tss_weight=None,
    )
    with pytest.raises(ValueError, match="only the Final method"):
        evaluator._build_inference_model(request, training.state_dict())


def test_runtime_closure_is_explicit_and_core_binding_is_restored() -> None:
    architecture = evaluator._explicit_architecture_paths()
    runtime = evaluator.trainer.runtime_source_paths()
    assert architecture
    assert all(path.is_file() for path in architecture.values())
    assert all(path.is_file() for path in runtime.values())
    assert set(architecture) == {
        key.split("::", 1)[1]
        for key in runtime
        if key.startswith("architecture::")
    }
    assert len(architecture) < len(list((models.REPO_ROOT / "model").glob("*.py")))
    source_hashes = evaluator.evaluator_source_sha256()
    assert evaluator.RELATIVE_SOURCE in source_hashes
    assert set(architecture) <= set(source_hashes)

    original_schema = evaluator.core.TRAINING_RUN_SCHEMA
    original_builder = evaluator.core.build_inference_model
    with evaluator._configured_core():
        assert evaluator.core.TRAINING_RUN_SCHEMA == evaluator.TRAINING_RUN_SCHEMA
        assert evaluator.core.FIXED_THRESHOLD == 0.5
        assert evaluator.core.TSS_CANDIDATES == (0.0,)
        assert evaluator.core.build_inference_model is evaluator._build_inference_model
        assert evaluator.core._training_runtime_source_paths() == runtime
        assert evaluator.core._model_source_paths() == architecture
    assert evaluator.core.TRAINING_RUN_SCHEMA == original_schema
    assert evaluator.core.build_inference_model is original_builder


def test_training_identity_accepts_only_pbdr_v2_recipe(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary = {
        "schema": evaluator.TRAINING_RUN_SCHEMA,
        "dataset": "NUAA-SIRST",
        "method": "final",
        "requested_tss_weight": 0.0,
        "tss_enabled": False,
        "status": "complete",
        "seed": 42,
        "epochs": 1000,
    }
    recipe = {
        "recipe_id": models.RECIPE_ID,
        "architecture": "tpd8_ner4_qfg2_croa_pbdr_v2",
        "pbdr_v2_integration_version": PBDR_V2_INTEGRATION_VERSION,
        "pbdr_v2_parameter_count": 19,
        "pbdr_v2_state_key_count": len(PBDR_V2_STATE_KEYS),
        "fresh_seed42_scratch": True,
        "warm_start_used": False,
        "parent_checkpoint": None,
        "resume_scope": "same_pbdr_v2_run_only",
        "current_shared_initial_state_bitwise_equal": True,
        "pbdr_v2_new_state_exact_zero": True,
    }
    protocol = {
        "schema": evaluator.TRAINING_RUN_SCHEMA,
        "dataset": "NUAA-SIRST",
        "method": "final",
        "requested_tss_weight": 0.0,
        "tss_enabled": False,
        "recipe": recipe,
        "pbdr_v2_architecture_binding": {
            "architecture_id": "a" * 64,
            "integration_version": PBDR_V2_INTEGRATION_VERSION,
            "training_state_key_count": 573,
            "inference_state_key_count": 569,
            "new_state_keys": list(PBDR_V2_STATE_KEYS),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    identity = evaluator._validate_training_identity(run_dir, "NUAA-SIRST")
    assert identity["recipe"] == recipe
    assert identity["architecture_binding"]["training_state_key_count"] == 573

    protocol["pbdr_v2_architecture_binding"]["inference_state_key_count"] = 568
    (run_dir / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(
        evaluator.artifacts.TSSOffDiagnosticError,
        match="inference_state_key_count differs",
    ):
        evaluator._validate_training_identity(run_dir, "NUAA-SIRST")


def test_completed_output_contract(tmp_path: Path) -> None:
    path = tmp_path / "evaluation.json"
    payload = {
        "schema": evaluator.core.SCHEMA,
        "status": "complete",
        "dataset": "NUDT-SIRST",
        "method": evaluator.OUTPUT_METHOD,
        "checkpoint_role": "best_pd",
        "requested_tss_weight": 0.0,
        "pbdr_v2_evaluator_adapter": {
            "training_run_schema": evaluator.TRAINING_RUN_SCHEMA,
            "fixed_checkpoint_threshold": 0.5,
            "integration_version": PBDR_V2_INTEGRATION_VERSION,
            "training_state_key_count": 573,
            "inference_state_key_count": 569,
            "runtime_dependency_policy": "explicit_closure_no_model_tree_glob",
            "tss_state_stripping": (
                "573_to_569_exact_four_tss_keys_preserve_five_pbdr_v2_keys"
            ),
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert evaluator.validate_completed_output(
        path,
        dataset="NUDT-SIRST",
        checkpoint_role="best_pd",
    ) == payload


def test_cli_keeps_core_threshold_and_rejects_negative_workers(tmp_path: Path) -> None:
    args = evaluator.parse_args(
        [
            "--dataset",
            "IRSTD-1K",
            "--checkpoint-role",
            "best_miou",
            "--run-dir",
            str(tmp_path),
            "--device",
            "cpu",
        ]
    )
    assert args.workers == 0
    assert evaluator.FIXED_THRESHOLD == evaluator.core.FIXED_THRESHOLD == 0.5
    with pytest.raises(SystemExit):
        evaluator.parse_args(
            [
                "--dataset",
                "IRSTD-1K",
                "--checkpoint-role",
                "best_miou",
                "--run-dir",
                str(tmp_path),
                "--workers",
                "-1",
            ]
        )
