from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock

import pytest
import torch

from experiments import evaluate_three_dataset_ner_v5_per as evaluator
from experiments import four_dataset_models_seed42_v1 as v4_registry
from experiments import launch_three_dataset_ner_v5_per_seed42 as launcher
from experiments import three_dataset_ner_v5_per_models_seed42_v1 as models
from experiments import train_three_dataset_ner_v5_per_tss_off_seed42 as trainer
from model.tpd_ner_v8_mprs_dch_v4_tail_aware_survival import SURVIVAL_STATE_KEYS
from model.tpd_ner_v8_mprs_dch_v5_per import (
    V5_PER_FORMAL_DC_SUPPORT_MODE,
    V5_PER_RELAY_VERSION,
)


@pytest.fixture(scope="module")
def paired_models():
    v5, metadata = models.build_v5_training_model("NUAA-SIRST", 42)
    _original, v4, _metadata = v4_registry.build_paired_models(
        42,
        dataset_name="NUAA-SIRST",
        final_with_tss=True,
    )
    return v5, metadata, v4


def test_registry_uses_bitwise_identical_v4_seed42_state(paired_models) -> None:
    v5, metadata, v4 = paired_models
    v4_state = v4.state_dict()
    v5_state = v5.state_dict()
    assert len(v4_state) == len(v5_state) == 568
    assert tuple(v4_state) == tuple(v5_state)
    assert all(torch.equal(v4_state[key], v5_state[key]) for key in v4_state)
    assert metadata["initial_state_bitwise_equal_to_v4"] is True
    assert metadata["shared_state_bitwise_equal_to_original"] is True
    assert metadata["initialization_mode"] == "fresh_seed42_scratch"
    assert metadata["parent_checkpoint"] is None


def test_tss_off_and_qfg_joint_training_contract(paired_models) -> None:
    model, metadata, _v4 = paired_models
    state = model.state_dict()
    assert metadata["requested_tss_weight"] == 0.0
    assert metadata["tss_enabled"] is False
    assert metadata["tss_heads_registered"] is True
    assert metadata["tss_training_forward_computes_logits"] is True
    assert metadata["tss_loss_consumes_logits"] is False
    assert metadata["tss_survival_target_constructed"] is False
    assert all(torch.count_nonzero(state[key]).item() == 0 for key in SURVIVAL_STATE_KEYS)
    assert all(parameter.requires_grad for parameter in model.tpd_qfg.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert {id(parameter) for parameter in model.tpd_qfg.parameters()} <= optimized


def test_inference_export_strips_exactly_568_to_564(paired_models) -> None:
    model, _metadata, _v4 = paired_models
    training_state = model.state_dict()
    stripped = models.strip_tss_for_inference_state_dict(training_state)
    assert len(training_state) == 568
    assert len(stripped) == 564
    assert set(training_state) - set(stripped) == set(SURVIVAL_STATE_KEYS)
    inference, metadata = models.build_v5_inference_model_from_training_state_dict(
        training_state,
        dataset_name="NUAA-SIRST",
    )
    assert len(inference.state_dict()) == 564
    assert metadata["strict_load"] is True
    assert metadata["relay_version"] == V5_PER_RELAY_VERSION
    assert metadata["dc_support_mode"] == V5_PER_FORMAL_DC_SUPPORT_MODE


def test_runtime_closure_is_explicit_complete_and_not_model_tree_glob() -> None:
    paths = trainer.runtime_source_paths()
    expected_runtime = {
        "experiments/train_three_dataset_ner_v5_per_tss_off_seed42.py",
        "experiments/three_dataset_ner_v5_per_models_seed42_v1.py",
        "experiments/train_three_dataset_tss_off_seed42_v1.py",
        "experiments/train_four_dataset_original_final_seed42_exact_v1.py",
        "experiments/train_three_dataset_seed42_global_tss_v2.py",
        "experiments/three_dataset_v2_protocol.py",
        "experiments/paper_three_dataset_v2.py",
        "experiments/tpd_training_loss.py",
        "model/tpd_ner_v8_mprs_dch_v5_per.py",
        "model/tpd_ner_v8_mprs_dch_v5_per_qfg_v2_croa_survival.py",
    }
    observed = {key.split("::", 1)[1] for key in paths}
    assert expected_runtime <= observed
    assert all(path.is_file() for path in paths.values())
    assert "model/tpd_ner_v5.py" not in observed
    assert len({item for item in observed if item.startswith("model/")}) < len(
        list((models.REPO_ROOT / "model").glob("*.py"))
    )
    with evaluator._configured_core():
        assert evaluator.core._training_runtime_source_paths() == paths
        assert evaluator.core._model_source_paths() == evaluator._explicit_architecture_paths()


def test_launcher_default_manifest_exists_and_maps_three_gpus() -> None:
    assert launcher.DEFAULT_MANIFEST.is_file()
    jobs = launcher.build_jobs("pilot")
    assert len(jobs) == 3
    assert {job.dataset: job.physical_gpu for job in jobs} == launcher.DATASET_GPU
    for job in jobs:
        assert "--epochs" in job.argv and job.argv[job.argv.index("--epochs") + 1] == "1000"
        assert "--pause-after-epoch" in job.argv
        assert job.argv[job.argv.index("--pause-after-epoch") + 1] == "200"
        assert job.argv[job.argv.index("--resume") + 1] == "never"
    payload = launcher.dry_run_payload(jobs)
    assert payload["dry_run"] is True
    assert payload["pilot_is_prefix_of_same_run"] is True


def test_launcher_resume_reuses_same_run_without_pause_flag() -> None:
    jobs = launcher.build_jobs("resume")
    assert len(jobs) == 3
    for job in jobs:
        assert "--pause-after-epoch" not in job.argv
        assert job.argv[job.argv.index("--resume") + 1] == "required"
        assert "--epochs" in job.argv and job.argv[job.argv.index("--epochs") + 1] == "1000"


def test_recipe_and_formal_pause_contract() -> None:
    args = trainer.parse_args(
        [
            "--dataset", "NUAA-SIRST",
            "--method", "final",
            "--physical-gpu-index", "0",
            "--expected-gpu-uuid", trainer.GPU_UUIDS["0"],
            "--pause-after-epoch", "200",
        ]
    )
    trainer.validate_args(args)
    recipe = trainer.recipe_identity(args)
    assert recipe["fresh_seed42_scratch"] is True
    assert recipe["warm_start_from_v4"] is False
    assert recipe["resume_scope"] == "same_v5_run_only"
    assert args.epochs == 1000 and args.pause_after_epoch == 200


def test_protocol_payload_uses_explicit_full_runtime_closure(paired_models) -> None:
    _model, metadata, _v4 = paired_models
    args = argparse.Namespace(
        dataset="NUAA-SIRST",
        method="final",
        tss_weight=0.0,
        epochs=1000,
    )
    seed_payload = {
        "training": {},
        "rolling_resume_state": {},
        "runtime_sources": {"forbidden_glob": {}},
    }
    with mock.patch.object(trainer, "_BASE_PROTOCOL_PAYLOAD", return_value=seed_payload):
        payload = trainer._protocol_payload(
            args,
            model_metadata=metadata,
            tss_metadata={},
            data_manifests={},
            train_count=1,
            test_count=1,
            device=torch.device("cpu"),
        )
    assert payload["runtime_sources"] == trainer.runtime_source_records()
    assert "forbidden_glob" not in payload["runtime_sources"]
    assert payload["requested_tss_weight"] == 0.0
    assert payload["tss_enabled"] is False
    assert payload["planned_total_epochs"] == 1000
    assert payload["pause_resume_contract"]["pilot_creates_additional_run"] is False


def test_v5_only_resume_rejects_foreign_schema(tmp_path: Path) -> None:
    class Dummy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def architecture_manifest(self):
            return {"relay_version": V5_PER_RELAY_VERSION}

    args = argparse.Namespace(
        dataset="NUAA-SIRST",
        method="final",
        tss_weight=0.0,
        resume="required",
        epochs=1000,
    )
    model = Dummy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "latest.pth.tar"
    torch.save({"schema": "historical_v4"}, path)
    with pytest.raises(trainer.V5PERTrainingProtocolError, match="resume schema"):
        trainer._load_resume_v5(
            args=args,
            path=path,
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            protocol_sha256="sha",
        )


def test_evaluator_routes_to_v5_builder(paired_models) -> None:
    model, _metadata, _v4 = paired_models
    request = evaluator.core.EvaluationRequest(
        dataset="NUAA-SIRST",
        method="final",
        checkpoint_role="best_miou",
        requested_tss_weight=0.0,
    )
    sentinel = object()
    metadata = {
        "relay_version": V5_PER_RELAY_VERSION,
        "dc_support_mode": V5_PER_FORMAL_DC_SUPPORT_MODE,
    }
    state = model.state_dict()
    with mock.patch.object(
        evaluator.models,
        "build_v5_inference_model_from_training_state_dict",
        return_value=(sentinel, metadata),
    ) as builder:
        built, observed = evaluator._build_inference_model(request, state)
    assert built is sentinel and observed == metadata
    builder.assert_called_once_with(state, dataset_name="NUAA-SIRST", seed=42)


def test_pause_artifact_binds_latest_to_protocol(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "resume").mkdir(parents=True)
    protocol = {"protocol_sha256": "frozen-protocol"}
    (run_dir / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    latest_path = run_dir / "resume/latest_training_state.pth.tar"
    torch.save({"epoch": 200, "protocol_sha256": "wrong"}, latest_path)
    args = argparse.Namespace(pause_after_epoch=200, epochs=1000)
    value = {"status": "running", "completed_epoch": 200, "total_epochs": 1000}
    with mock.patch.object(trainer, "_ACTIVE_ARGS", args):
        with pytest.raises(trainer.V5PERTrainingProtocolError, match="pause rolling protocol"):
            trainer._write_json_atomic(run_dir / "progress.json", value)
