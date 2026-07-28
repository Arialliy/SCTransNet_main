from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import (
    finalize_tpd_ner_v8_mprs_dch_v3_dc_knockout_gpu23 as subject,
)
from experiments import (
    tpd_ner_v8_mprs_dch_v3_dc_knockout_spec as spec,
)


class _SuccessfulPopen:
    def __init__(self, *_: object, **__: object) -> None:
        self.returncode = 0

    def communicate(self) -> tuple[str, str]:
        return "{}\n", ""

    def poll(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        raise AssertionError("successful evaluator must not be terminated")


class V3DCKnockoutGPU23FinalizerTests(unittest.TestCase):
    def test_fixed_lanes_commands_and_environment(self) -> None:
        expected = {
            "best.pth.tar": (
                2,
                "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
            ),
            "best_miou.pth.tar": (
                3,
                "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
            ),
        }
        self.assertEqual(tuple(subject.CHECKPOINT_LANES), spec.CHECKPOINTS)
        for checkpoint, (index, uuid) in expected.items():
            with self.subTest(checkpoint=checkpoint):
                self.assertEqual(
                    subject.evaluator_command(checkpoint),
                    [
                        str(subject.PYTHON),
                        str(subject.EVALUATOR),
                        "--run",
                        "--checkpoint",
                        checkpoint,
                        "--device",
                        "cuda:0",
                    ],
                )
                environment = subject.evaluator_environment(checkpoint)
                self.assertEqual(
                    environment[subject.CUDA_VISIBLE_DEVICES_ENV], uuid
                )
                self.assertEqual(
                    environment[subject.CUDA_DEVICE_ORDER_ENV], "PCI_BUS_ID"
                )
                self.assertEqual(
                    environment[subject.CUBLAS_WORKSPACE_CONFIG_ENV],
                    ":4096:8",
                )
                self.assertEqual(
                    environment[subject.PYTHONHASHSEED_ENV],
                    "42",
                )
                self.assertEqual(
                    environment[subject.KNOCKOUT_PHYSICAL_GPU_INDEX_ENV],
                    str(index),
                )
                self.assertEqual(
                    environment[subject.KNOCKOUT_PHYSICAL_GPU_UUID_ENV], uuid
                )

    def test_execution_plan_is_read_only_and_has_parallel_checkpoint_lanes(self) -> None:
        with mock.patch.object(subject, "FORMAL_COMPLETE_MARKER", Path("/absent/formal.json")):
            plan = subject.execution_plan()
        self.assertFalse(plan["formal_postprocess_complete"])
        self.assertTrue(plan["checkpoint_processes_parallel"])
        self.assertEqual(plan["checkpoint_count"], 2)
        self.assertEqual(plan["knockout_modes_per_checkpoint"], 4)
        self.assertFalse(plan["invokes_gpu"])
        self.assertEqual(
            plan["formal_aggregate_authority"],
            "versioned_selection_contract_repair_v1_only",
        )
        self.assertEqual(
            plan["formal_selection_contract_repair_id"],
            subject.FORMAL_REPAIR_ID,
        )
        self.assertTrue(
            plan["each_variant_uses_own_selected_checkpoints"]
        )
        for checkpoint in spec.CHECKPOINTS:
            detail = plan["sweeps"][checkpoint]
            self.assertEqual(detail["internal_knockout_modes"], list(spec.KNOCKOUT_MODES))
            self.assertTrue(detail["modes_evaluated_sequentially"])
            self.assertEqual(
                detail["environment"][
                    subject.CUBLAS_WORKSPACE_CONFIG_ENV
                ],
                ":4096:8",
            )
            self.assertEqual(
                detail["environment"][subject.PYTHONHASHSEED_ENV],
                "42",
            )
        self.assertTrue(
            str(plan["diagnostic_source_lock"]["path"]).endswith(
                "tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock_v2.json"
            )
        )

    def test_launches_missing_checkpoint_sweeps_in_parallel_with_mock_subprocess(self) -> None:
        with mock.patch.object(subject.subprocess, "Popen", side_effect=_SuccessfulPopen) as popen:
            subject._launch_evaluators(spec.CHECKPOINTS)
        self.assertEqual(popen.call_count, 2)
        calls = {call.args[0][4]: call for call in popen.call_args_list}
        self.assertEqual(set(calls), set(spec.CHECKPOINTS))
        for checkpoint, call in calls.items():
            self.assertEqual(call.kwargs["cwd"], subject.REPO_ROOT)
            self.assertEqual(call.kwargs["env"], subject.evaluator_environment(checkpoint))
            self.assertEqual(call.kwargs["stdout"], subprocess.PIPE)
            self.assertEqual(call.kwargs["stderr"], subprocess.PIPE)
            self.assertTrue(call.kwargs["text"])

    def test_run_now_reuses_valid_sweeps_and_aggregates_once(self) -> None:
        marker = {"schema": "diagnostic-marker"}
        with (
            mock.patch.object(subject.diagnostic_post, "inspect_complete", side_effect=[None, marker]) as complete,
            mock.patch.object(subject, "freeze_or_verify_diagnostic_source_lock") as freeze,
            mock.patch.object(subject, "_source_binding", return_value={"binding": 1}),
            mock.patch.object(subject, "_validate_existing_sweep", return_value=True) as validate,
            mock.patch.object(subject, "_launch_evaluators") as launch,
            mock.patch.object(subject, "_aggregate_and_verify_marker", return_value=marker) as aggregate,
        ):
            result = subject.run_now()
        freeze.assert_called_once_with()
        launch.assert_not_called()
        aggregate.assert_called_once_with()
        self.assertEqual(validate.call_count, 4)
        self.assertEqual(result["evaluator_invocations"], 0)
        self.assertEqual(result["aggregate_invocations"], 1)
        self.assertEqual(result["reused_checkpoints"], list(spec.CHECKPOINTS))
        self.assertEqual(complete.call_count, 1)

    def test_run_now_launches_only_missing_sweeps_then_revalidates(self) -> None:
        marker = {"schema": "diagnostic-marker"}
        # First matrix pass: best missing, best_miou reusable.  The second
        # validation pass proves both artifacts after the mocked evaluator.
        validations = iter([False, True, True, True])
        with (
            mock.patch.object(subject.diagnostic_post, "inspect_complete", return_value=None),
            mock.patch.object(subject, "freeze_or_verify_diagnostic_source_lock"),
            mock.patch.object(subject, "_source_binding", return_value={"binding": 1}),
            mock.patch.object(subject, "_validate_existing_sweep", side_effect=lambda *_args, **_kwargs: next(validations)),
            mock.patch.object(subject, "_launch_evaluators") as launch,
            mock.patch.object(subject, "_aggregate_and_verify_marker", return_value=marker),
        ):
            result = subject.run_now()
        launch.assert_called_once_with(["best.pth.tar"])
        self.assertEqual(result["launched_checkpoints"], ["best.pth.tar"])
        self.assertEqual(result["reused_checkpoints"], ["best_miou.pth.tar"])

    def test_existing_sweep_conflict_fails_before_gpu_launch(self) -> None:
        with (
            mock.patch.object(subject.diagnostic_post, "inspect_complete", return_value=None),
            mock.patch.object(subject, "freeze_or_verify_diagnostic_source_lock"),
            mock.patch.object(subject, "_source_binding", return_value={"binding": 1}),
            mock.patch.object(
                subject,
                "_validate_existing_sweep",
                side_effect=subject.KnockoutFinalizerError("existing diagnostic sweep conflicts"),
            ),
            mock.patch.object(subject, "_launch_evaluators") as launch,
            self.assertRaisesRegex(subject.KnockoutFinalizerError, "conflicts"),
        ):
            subject.run_now()
        launch.assert_not_called()

    def test_existing_complete_marker_is_idempotent_without_freezing_or_launching(self) -> None:
        marker = {"schema": "complete"}
        with (
            mock.patch.object(subject.diagnostic_post, "inspect_complete", return_value=marker),
            mock.patch.object(subject, "freeze_or_verify_diagnostic_source_lock") as freeze,
            mock.patch.object(subject, "_launch_evaluators") as launch,
        ):
            result = subject.run_now()
        freeze.assert_not_called()
        launch.assert_not_called()
        self.assertEqual(result["status"], "already_complete")
        self.assertEqual(result["aggregate_invocations"], 0)

    def test_freeze_or_verify_explicitly_freezes_only_after_formal_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "diagnostic-lock.json"
            payload = {"source": "lock"}
            with (
                mock.patch.object(subject, "SOURCE_LOCK", lock),
                mock.patch.object(subject, "inspect_formal_postprocess_complete", return_value={"formal": 1}) as formal,
                mock.patch.object(subject.freezer, "build_source_lock", return_value=payload) as build,
                mock.patch.object(subject.freezer, "publish_new_lock") as publish,
                mock.patch.object(subject.freezer, "verify_source_lock", return_value=payload) as verify,
            ):
                observed = subject.freeze_or_verify_diagnostic_source_lock()
        formal.assert_called_once_with()
        build.assert_called_once_with()
        publish.assert_called_once_with(lock, payload)
        verify.assert_called_once_with(lock)
        self.assertEqual(observed, payload)

    def test_formal_preflight_requires_repaired_selection_contract(
        self,
    ) -> None:
        binding = {
            "formal_completion_marker": {
                "path": str(subject.FORMAL_COMPLETE_MARKER.resolve()),
            },
            "formal_selection_contract_repair": {
                "repair_id": subject.FORMAL_REPAIR_ID,
                "authority": (
                    "versioned_selection_contract_repair_v1_only"
                ),
                "each_variant_uses_own_selected_checkpoints": True,
                "formal_aggregate_decision": (
                    subject.freezer.EXPECTED_FORMAL_DECISION
                ),
            },
        }
        with (
            mock.patch.object(
                subject,
                "_regular_file",
                return_value=subject.FORMAL_COMPLETE_MARKER,
            ),
            mock.patch.object(
                subject.freezer,
                "current_formal_artifact_binding",
                return_value=binding,
            ),
        ):
            self.assertEqual(
                subject.inspect_formal_postprocess_complete(),
                binding,
            )
        binding["formal_selection_contract_repair"][
            "each_variant_uses_own_selected_checkpoints"
        ] = False
        with (
            mock.patch.object(
                subject,
                "_regular_file",
                return_value=subject.FORMAL_COMPLETE_MARKER,
            ),
            mock.patch.object(
                subject.freezer,
                "current_formal_artifact_binding",
                return_value=binding,
            ),
            self.assertRaisesRegex(
                subject.KnockoutFinalizerError,
                "selection-contract authority differs",
            ),
        ):
            subject.inspect_formal_postprocess_complete()

    def test_watch_waits_for_absent_formal_marker_then_runs(self) -> None:
        absent = Path("/unavailable/formal-marker.json")
        with (
            mock.patch.object(subject, "FORMAL_COMPLETE_MARKER", absent),
            mock.patch.object(subject.diagnostic_post, "inspect_complete", side_effect=[None, None]),
            mock.patch.object(Path, "exists", side_effect=[False, True]),
            mock.patch.object(subject.time, "sleep") as sleep,
            mock.patch.object(subject, "inspect_formal_postprocess_complete") as formal,
            mock.patch.object(subject, "run_now", return_value={"status": "complete"}) as run_now,
        ):
            result = subject.watch_and_run(poll_seconds=0.25)
        sleep.assert_called_once_with(0.25)
        formal.assert_called_once_with()
        run_now.assert_called_once_with()
        self.assertEqual(result["status"], "complete")

    def test_aggregate_command_is_diagnostic_only_and_marker_is_required(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="{}\n", stderr="")
        marker = {"schema": "complete"}
        with (
            mock.patch.object(subject.subprocess, "run", return_value=completed) as run,
            mock.patch.object(subject.diagnostic_post, "inspect_complete", return_value=marker),
        ):
            self.assertEqual(subject._aggregate_and_verify_marker(), marker)
        self.assertEqual(
            run.call_args.args[0],
            [str(subject.PYTHON), str(subject.AGGREGATOR), "--aggregate"],
        )
        self.assertEqual(run.call_args.kwargs["cwd"], subject.REPO_ROOT)
        self.assertTrue(run.call_args.kwargs["capture_output"])

    def test_poll_interval_above_thirty_fails_before_inspecting_artifacts(self) -> None:
        with (
            mock.patch.object(subject.diagnostic_post, "inspect_complete") as complete,
            self.assertRaisesRegex(subject.KnockoutFinalizerError, "no more than 30"),
        ):
            subject.watch_and_run(poll_seconds=30.01)
        complete.assert_not_called()

    def test_cli_requires_one_action_and_enforces_poll_bound(self) -> None:
        with self.assertRaises(SystemExit):
            subject.parse_args([])
        with self.assertRaises(SystemExit):
            subject.parse_args(["--watch", "--poll-seconds", "31"])


if __name__ == "__main__":
    unittest.main()
