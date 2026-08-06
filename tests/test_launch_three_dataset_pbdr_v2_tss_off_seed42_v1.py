from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from experiments import launch_three_dataset_pbdr_v2_tss_off_seed42_v1 as launcher
from experiments import train_three_dataset_pbdr_v2_tss_off_seed42_v1 as trainer


def _argument(command: tuple[str, ...], name: str) -> str:
    return command[command.index(name) + 1]


def test_three_formal1000_scratch_runs_are_independent_and_fixed() -> None:
    assert launcher.RECIPE_ID == trainer.RECIPE_ID
    specs = launcher.build_worker_specs(base_environment={"PATH": "/usr/bin"})
    assert [(spec.dataset, spec.gpu_index) for spec in specs] == list(
        launcher.TRAINING_LAYOUT
    )
    assert len({spec.run_directory for spec in specs}) == 3
    for spec in specs:
        assert spec.resume == "never"
        assert _argument(spec.command, "--resume") == "never"
        assert _argument(spec.command, "--seed") == "42"
        assert _argument(spec.command, "--epochs") == "1000"
        assert _argument(spec.command, "--eval-every") == "10"
        assert _argument(spec.command, "--threshold") == "0.5"
        assert _argument(spec.command, "--tss-weight") == "0"
        assert spec.run_directory.as_posix().endswith(
            f"runs/{spec.dataset}/{launcher.RECIPE_ID}/seed_42"
        )
        args = trainer.parse_args(list(spec.command[2:]))
        assert spec.run_directory == trainer._run_directory(args)


@pytest.mark.parametrize("resume", launcher.RESUME_MODES)
def test_all_frozen_resume_policies_are_forwarded(resume: str) -> None:
    specs = launcher.build_worker_specs(
        resume=resume,
        base_environment={"PATH": "/usr/bin"},
    )
    assert {spec.resume for spec in specs} == {resume}
    assert {_argument(spec.command, "--resume") for spec in specs} == {resume}


def test_cuda_visibility_uses_complete_uuid_never_numeric_index() -> None:
    specs = launcher.build_worker_specs(base_environment={"PATH": "/usr/bin"})
    for spec in specs:
        expected = launcher.GPU_ASSIGNMENTS[spec.gpu_index]["uuid"]
        visible = spec.environment["CUDA_VISIBLE_DEVICES"]
        assert visible == expected == spec.expected_gpu_uuid
        assert visible.startswith("GPU-")
        assert visible != spec.gpu_index
        assert _argument(spec.command, "--physical-gpu-index") == spec.gpu_index
        assert _argument(spec.command, "--expected-gpu-uuid") == visible


@pytest.mark.parametrize("resume", launcher.RESUME_MODES)
def test_every_generated_command_is_accepted_by_the_pbdr_v2_trainer(
    resume: str,
) -> None:
    specs = launcher.build_worker_specs(
        resume=resume,
        base_environment={"PATH": "/usr/bin"},
    )
    for spec in specs:
        args = trainer.parse_args(list(spec.command[2:]))
        trainer.validate_args(args)
        assert args.dataset == spec.dataset
        assert args.resume == resume
        assert args.expected_gpu_uuid == spec.environment["CUDA_VISIBLE_DEVICES"]


def test_default_main_is_dry_run_and_does_not_start_process(capsys) -> None:
    with mock.patch.object(launcher.subprocess, "Popen") as popen:
        launcher.main([])
    popen.assert_not_called()
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["datasets"] == list(launcher.DATASETS)
    assert payload["planned_total_epochs"] == 1000
    assert payload["independent_run_per_dataset"] is True
    assert len(payload["workers"]) == 3
    for worker in payload["workers"]:
        visible = worker["environment"]["CUDA_VISIBLE_DEVICES"]
        assert visible.startswith("GPU-")
        assert visible != worker["gpu_index"]


def test_invalid_resume_and_dataset_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resume must be one of"):
        launcher.build_worker_specs(resume="sometimes")
    with pytest.raises(ValueError, match="unsupported dataset"):
        launcher.run_directory(tmp_path, "SIRST3")
