from __future__ import annotations

from pathlib import Path
from unittest import mock

from experiments import launch_two_dataset_pbdr_v3_stage1_v1 as launcher


def test_all_datasets_have_unique_gpu_and_run_directories() -> None:
    specs = launcher.build_worker_specs(
        dataset="all",
        results_root=Path("/tmp/pbdr-v3-launch-test"),
        smoke=True,
    )
    assert len(specs) == 4
    assert len({spec.run_directory for spec in specs}) == 4
    assert len(set(launcher.GPU_UUIDS.values())) == 2
    for spec in specs:
        command = list(spec.command)
        assert command[command.index("--dataset") + 1] == spec.dataset
        assert command[command.index("--parent-role") + 1] == spec.parent_role
        assert "--smoke" in command
        assert int(command[command.index("--max-val-images") + 1]) >= 8


def test_launcher_never_includes_official_evaluator() -> None:
    specs = launcher.build_worker_specs(dataset="NUDT-SIRST")
    assert all("evaluate_two_dataset" not in " ".join(spec.command) for spec in specs)


def test_dataset_environment_binds_exact_uuid() -> None:
    for dataset in launcher.DATASETS:
        environment = launcher.dataset_environment(dataset, {})
        assert environment["CUDA_VISIBLE_DEVICES"] == launcher.GPU_UUIDS[dataset]
        assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_systemd_commands_wait_for_final_worker_status() -> None:
    args = launcher.parse_args(["--dataset", "all", "--smoke", "--execute"])
    for dataset in launcher.DATASETS:
        command = launcher.systemd_command(args, dataset)
        assert "--wait" in command


def test_execute_starts_both_dataset_services_before_waiting() -> None:
    events: list[str] = []

    class Process:
        def __init__(self, dataset: str) -> None:
            self.dataset = dataset

        def wait(self) -> int:
            events.append(f"wait:{self.dataset}")
            assert events[:2] == [
                "start:NUDT-SIRST",
                "start:IRSTD-1K",
            ]
            return 0

    def popen(command, **kwargs):
        del kwargs
        dataset = command[command.index("--dataset") + 1]
        events.append(f"start:{dataset}")
        return Process(dataset)

    with (
        mock.patch.object(launcher, "verify_gpu_bindings", return_value={}),
        mock.patch.object(launcher.subprocess, "Popen", side_effect=popen),
    ):
        try:
            launcher.main(["--dataset", "all", "--smoke", "--execute"])
        except SystemExit as error:
            assert error.code == 0
    assert events == [
        "start:NUDT-SIRST",
        "start:IRSTD-1K",
        "wait:NUDT-SIRST",
        "wait:IRSTD-1K",
    ]
