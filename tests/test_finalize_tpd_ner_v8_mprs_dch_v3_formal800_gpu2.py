from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments import (
    finalize_tpd_ner_v8_mprs_dch_v3_formal800_gpu2 as subject,
)


def _complete_marker_fixture(root: Path) -> dict:
    comparison = root / "comparison"
    comparison.mkdir(parents=True)
    json_output = comparison / subject.JSON_OUTPUT.name
    markdown_output = comparison / subject.MARKDOWN_OUTPUT.name
    report = {
        "schema": subject.POSTPROCESS_REPORT_SCHEMA,
        "status": "complete",
        "decision": "RETURN_TO_MODEL_OPTIMIZATION",
        "aggregate_full_model_gate_passed": False,
        "aggregate_tiny_pd_regressed": True,
        "tiny_pd_regression_affects_decision": False,
    }
    json_output.write_text(
        json.dumps(report, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_output.write_text("# fixture\n", encoding="utf-8")
    marker = {
        "schema": subject.POSTPROCESS_MARKER_SCHEMA,
        "status": "complete",
        "decision": report["decision"],
        "aggregate_full_model_gate_passed": report[
            "aggregate_full_model_gate_passed"
        ],
        "aggregate_tiny_pd_regressed": report[
            "aggregate_tiny_pd_regressed"
        ],
        "tiny_pd_regression_affects_decision": False,
        "outputs": {
            json_output.name: hashlib.sha256(
                json_output.read_bytes()
            ).hexdigest(),
            markdown_output.name: hashlib.sha256(
                markdown_output.read_bytes()
            ).hexdigest(),
        },
    }
    marker_path = comparison / subject.COMPLETE_MARKER.name
    marker_path.write_text(
        json.dumps(marker, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "comparison": comparison,
        "json": json_output,
        "markdown": markdown_output,
        "marker_path": marker_path,
        "marker": marker,
    }


def _valid_plan() -> dict:
    return {
        "readiness": {
            "schema": subject.POSTPROCESS_READINESS_SCHEMA,
            "required_runs_complete": True,
            "v3_on": {
                "complete": True,
                "run_dir": str(subject.CANONICAL_RUN_DIR.resolve()),
            },
        },
        "new_evaluation_count": 2,
        "new_evaluations": [
            {
                "variant": subject.VARIANT,
                "checkpoint": checkpoint,
                "physical_gpu_index": 2,
                "command": [
                    str(subject.PYTHON),
                    str(subject.EVALUATOR),
                    "--run-dir",
                    str(subject.CANONICAL_RUN_DIR),
                    "--checkpoint",
                    checkpoint,
                    "--device",
                    "cuda:0",
                    "--expected-epochs",
                    "800",
                ],
            }
            for checkpoint in ("best.pth.tar", "best_miou.pth.tar")
        ],
        "v2_on_evaluations": 0,
        "v1_off_evaluations": 0,
        "baseline_evaluations": 0,
        "aggregate_outputs": [
            str(subject.JSON_OUTPUT),
            str(subject.MARKDOWN_OUTPUT),
            str(subject.COMPLETE_MARKER),
        ],
        "training_seed": 42,
        "split_seed": 20260722,
        "multi_seed_scheduled": False,
    }


class V3GPU2FinalizerTests(unittest.TestCase):
    def test_scope_and_run_now_command_are_fixed(self) -> None:
        self.assertEqual(
            subject.SERVICE,
            "sctransnet-tpd-ner-v8-v3-relay-on-gpu2.service",
        )
        self.assertEqual(
            subject.CANONICAL_RUN_DIR,
            (
                subject.exact.DEFAULT_OUTPUT_ROOT
                / "NUDT-SIRST"
                / "tpd_ner_v8_mprs_dch_v3_full_relay_on"
                / "seed_42_formal800_exact_v3_seed42"
            ),
        )
        self.assertEqual(
            subject._postprocess_command("--run-now"),
            [
                "/home/ly/BasicIRSTD/infrarenet/bin/python",
                str(subject.POSTPROCESS),
                "--run-now",
                "--device-mode",
                "gpu23",
                "--physical-gpu",
                "2",
                "--python",
                "/home/ly/BasicIRSTD/infrarenet/bin/python",
            ],
        )

    def test_valid_marker_is_idempotent_without_inspecting_service(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _complete_marker_fixture(Path(temporary))
            with (
                mock.patch.object(
                    subject, "JSON_OUTPUT", fixture["json"]
                ),
                mock.patch.object(
                    subject, "MARKDOWN_OUTPUT", fixture["markdown"]
                ),
                mock.patch.object(
                    subject, "COMPLETE_MARKER", fixture["marker_path"]
                ),
                mock.patch.object(
                    subject, "inspect_training_service"
                ) as service,
                mock.patch.object(
                    subject, "inspect_strict_training_completion"
                ) as strict,
                mock.patch.object(subject, "locked_postprocess_plan") as plan,
                mock.patch.object(
                    subject, "run_locked_postprocess_once"
                ) as run_now,
            ):
                result = subject.watch_and_finalize(poll_seconds=30)
            self.assertEqual(result["status"], "already_complete")
            self.assertEqual(result["postprocess_invocations"], 0)
            service.assert_not_called()
            strict.assert_not_called()
            plan.assert_not_called()
            run_now.assert_not_called()

    def test_invalid_marker_hash_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _complete_marker_fixture(Path(temporary))
            fixture["json"].write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.object(
                    subject, "JSON_OUTPUT", fixture["json"]
                ),
                mock.patch.object(
                    subject, "MARKDOWN_OUTPUT", fixture["markdown"]
                ),
                mock.patch.object(
                    subject, "COMPLETE_MARKER", fixture["marker_path"]
                ),
                mock.patch.object(
                    subject, "inspect_training_service"
                ) as service,
                self.assertRaisesRegex(
                    subject.FinalizerError,
                    "digest differs",
                ),
            ):
                subject.watch_and_finalize()
            service.assert_not_called()

    def test_active_service_polls_then_finalizes_once(self) -> None:
        marker = {
            "decision": "FULL_MODEL_GATE_PASSED",
        }
        incomplete = {"complete": False, "reason": "epoch 20"}
        complete = {"complete": True, "reason": "strict complete"}
        with (
            mock.patch.object(
                subject,
                "inspect_postprocess_complete",
                side_effect=[None, None, marker],
            ),
            mock.patch.object(
                subject,
                "inspect_training_service",
                side_effect=[
                    {
                        "LoadState": "loaded",
                        "ActiveState": "active",
                        "SubState": "running",
                    },
                    {
                        "LoadState": "loaded",
                        "ActiveState": "inactive",
                        "SubState": "dead",
                        "Result": "success",
                        "ExecMainStatus": "0",
                    },
                ],
            ),
            mock.patch.object(
                subject,
                "inspect_strict_training_completion",
                side_effect=[incomplete, complete],
            ),
            mock.patch.object(subject.time, "sleep") as sleep,
            mock.patch.object(
                subject, "locked_postprocess_plan", return_value=_valid_plan()
            ) as plan,
            mock.patch.object(
                subject, "run_locked_postprocess_once"
            ) as run_now,
        ):
            result = subject.watch_and_finalize(poll_seconds=0.25)
        sleep.assert_called_once_with(0.25)
        plan.assert_called_once_with()
        run_now.assert_called_once_with()
        self.assertEqual(result["status"], "finalized")
        self.assertEqual(result["postprocess_invocations"], 1)

    def test_active_service_waits_even_if_artifacts_are_complete(self) -> None:
        marker = {"decision": "RETURN_TO_MODEL_OPTIMIZATION"}
        with (
            mock.patch.object(
                subject,
                "inspect_postprocess_complete",
                side_effect=[None, None, marker],
            ),
            mock.patch.object(
                subject,
                "inspect_training_service",
                side_effect=[
                    {"LoadState": "loaded", "ActiveState": "deactivating"},
                    {"LoadState": "loaded", "ActiveState": "inactive"},
                ],
            ),
            mock.patch.object(
                subject,
                "inspect_strict_training_completion",
                return_value={"complete": True, "reason": "complete"},
            ),
            mock.patch.object(subject.time, "sleep") as sleep,
            mock.patch.object(subject, "locked_postprocess_plan") as plan,
            mock.patch.object(
                subject, "run_locked_postprocess_once"
            ) as run_now,
        ):
            result = subject.watch_and_finalize(poll_seconds=30)
        sleep.assert_called_once_with(30.0)
        plan.assert_called_once_with()
        run_now.assert_called_once_with()
        self.assertEqual(result["postprocess_invocations"], 1)

    def test_failed_or_stopped_incomplete_service_never_evaluates(self) -> None:
        with (
            mock.patch.object(
                subject, "inspect_postprocess_complete", return_value=None
            ),
            mock.patch.object(
                subject,
                "inspect_training_service",
                return_value={
                    "LoadState": "loaded",
                    "ActiveState": "failed",
                    "SubState": "failed",
                    "Result": "exit-code",
                    "ExecMainStatus": "7",
                },
            ),
            mock.patch.object(
                subject,
                "inspect_strict_training_completion",
                return_value={
                    "complete": False,
                    "reason": "summary is absent",
                },
            ),
            mock.patch.object(subject, "locked_postprocess_plan") as plan,
            mock.patch.object(
                subject, "run_locked_postprocess_once"
            ) as run_now,
            self.assertRaisesRegex(
                subject.FinalizerError,
                "stopped before strict completion",
            ),
        ):
            subject.watch_and_finalize()
        plan.assert_not_called()
        run_now.assert_not_called()

    def test_unready_locked_plan_prevents_run_now(self) -> None:
        plan = _valid_plan()
        plan["readiness"]["required_runs_complete"] = False
        with self.assertRaisesRegex(
            subject.FinalizerError,
            "not ready",
        ):
            subject.validate_locked_postprocess_plan(plan)

    def test_plan_rejects_a_noncanonical_inner_evaluator_command(self) -> None:
        plan = _valid_plan()
        command = plan["new_evaluations"][0]["command"]
        command[command.index("--device") + 1] = "cpu"
        with self.assertRaisesRegex(
            subject.FinalizerError,
            "canonical GPU2 evaluator command",
        ):
            subject.validate_locked_postprocess_plan(plan)

    def test_poll_interval_above_thirty_fails_before_any_read(self) -> None:
        with (
            mock.patch.object(
                subject, "inspect_postprocess_complete"
            ) as marker,
            self.assertRaisesRegex(
                subject.FinalizerError,
                "no more than 30",
            ),
        ):
            subject.watch_and_finalize(poll_seconds=30.01)
        marker.assert_not_called()

    def test_systemd_query_names_only_the_fixed_service(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "LoadState=loaded\n"
                "ActiveState=active\n"
                "SubState=running\n"
                "Result=success\n"
                "ExecMainStatus=0\n"
            ),
            stderr="",
        )
        with mock.patch.object(
            subject.subprocess,
            "run",
            return_value=completed,
        ) as run:
            state = subject.inspect_training_service()
        command = run.call_args.args[0]
        self.assertEqual(command.count(subject.SERVICE), 1)
        self.assertNotIn("gpu3", " ".join(command))
        self.assertEqual(state["ActiveState"], "active")

    def test_plan_and_run_now_use_distinct_exact_commands(self) -> None:
        plan_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(_valid_plan()),
            stderr="",
        )
        run_result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        )
        with mock.patch.object(
            subject.subprocess,
            "run",
            side_effect=[plan_result, run_result],
        ) as run:
            subject.locked_postprocess_plan()
            subject.run_locked_postprocess_once()
        self.assertEqual(
            run.call_args_list[0].args[0],
            subject._postprocess_command("--plan"),
        )
        self.assertEqual(
            run.call_args_list[1].args[0],
            subject._postprocess_command("--run-now"),
        )
        self.assertEqual(
            sum(
                "--run-now" in call.args[0]
                for call in run.call_args_list
            ),
            1,
        )

    def test_strict_completion_checks_800_metrics_checkpoints_and_journal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "canonical-run"
            run_dir.mkdir()
            digest = "a" * 64
            identity = {
                "dataset": subject.DATASET,
                "variant": subject.VARIANT,
                "seed": subject.TRAINING_SEED,
                "split_seed": subject.SPLIT_SEED,
                "run_id": subject._expected_run_id(),
                "source_locks": {
                    subject.exact.SOURCE_LOCK_KEY: digest,
                },
            }
            (run_dir / "protocol.json").write_text(
                json.dumps(
                    {
                        "schema": subject.exact.ENTRY_SCHEMA,
                        "run_identity": identity,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "schema": subject.exact.COMPLETION_SUMMARY_SCHEMA,
                        "status": "complete",
                        "variant": subject.VARIANT,
                        "seed": 42,
                        "split_seed": 20260722,
                        "run_identity": identity,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            metric_rows = []
            for epoch in range(1, 801):
                event = {
                    name: 0.0
                    for name in subject.exact.STORED_VALIDATION_METRICS
                }
                event.update(
                    {
                        "epoch": epoch,
                        "variant": subject.VARIANT,
                    }
                )
                metric_rows.append(
                    json.dumps(event, sort_keys=True, separators=(",", ":"))
                )
            metrics_path = run_dir / "metrics.jsonl"
            metrics_path.write_text(
                "\n".join(metric_rows) + "\n",
                encoding="utf-8",
            )
            checkpoint_payloads = {}
            for checkpoint_name, role in subject.CHECKPOINT_CONTRACTS:
                path = run_dir / checkpoint_name
                path.write_bytes(b"fixture checkpoint")
                checkpoint_payloads[checkpoint_name] = {
                    "run_identity": identity,
                    "checkpoint_role": role,
                    "epoch": 800 if checkpoint_name == "last.pth.tar" else 700,
                }
            journal_root = run_dir / "exact_journal"
            journal_root.mkdir()
            (journal_root / "active.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            active_metrics = journal_root / "slot_a.metrics.jsonl"
            active_metrics.write_bytes(metrics_path.read_bytes())
            active_checkpoint = journal_root / "slot_a.exact.pth"
            active_checkpoint.write_bytes(b"fixture active checkpoint")
            active = SimpleNamespace(
                epoch=800,
                metrics_path=active_metrics,
                checkpoint_path=active_checkpoint,
            )

            def fake_torch_load(path: Path, **_: object) -> dict:
                name = Path(path).name
                if name == active_checkpoint.name:
                    return {"run_identity": identity}
                return checkpoint_payloads[name]

            with (
                mock.patch.object(
                    subject.exact,
                    "file_sha256",
                    return_value=digest,
                ),
                mock.patch.object(
                    subject.exact,
                    "require_v3_run_identity",
                    side_effect=lambda value, **_: dict(value),
                ),
                mock.patch.object(
                    subject.exact,
                    "require_evaluator_checkpoint_payload",
                    side_effect=lambda value, **_: dict(value),
                ),
                mock.patch.object(
                    subject.epoch_journal,
                    "ExactEpochJournal",
                    return_value=SimpleNamespace(
                        load_active=lambda: active
                    ),
                ),
                mock.patch.object(
                    subject.torch,
                    "load",
                    side_effect=fake_torch_load,
                ),
            ):
                result = subject.inspect_strict_training_completion(run_dir)
            self.assertTrue(result["complete"])
            self.assertEqual(result["event_count"], 800)
            self.assertEqual(result["last_epoch"], 800)
            self.assertEqual(result["active_journal_epoch"], 800)

    def test_strict_completion_rejects_truncated_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            identity = {
                "dataset": subject.DATASET,
                "variant": subject.VARIANT,
                "seed": 42,
                "split_seed": 20260722,
                "run_id": subject._expected_run_id(),
                "source_locks": {
                    subject.exact.SOURCE_LOCK_KEY: "a" * 64,
                },
            }
            for name, value in {
                "protocol.json": {
                    "schema": subject.exact.ENTRY_SCHEMA,
                    "run_identity": identity,
                },
                "summary.json": {
                    "schema": subject.exact.COMPLETION_SUMMARY_SCHEMA,
                    "status": "complete",
                    "variant": subject.VARIANT,
                    "seed": 42,
                    "split_seed": 20260722,
                    "run_identity": identity,
                },
            }.items():
                (run_dir / name).write_text(
                    json.dumps(value) + "\n",
                    encoding="utf-8",
                )
            (run_dir / "metrics.jsonl").write_text(
                json.dumps(
                    {
                        "epoch": 1,
                        "variant": subject.VARIANT,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    subject.exact,
                    "file_sha256",
                    return_value="a" * 64,
                ),
                mock.patch.object(
                    subject.exact,
                    "require_v3_run_identity",
                    side_effect=lambda value, **_: dict(value),
                ),
                self.assertRaisesRegex(
                    subject.FinalizerError,
                    "exactly 800",
                ),
            ):
                subject.inspect_strict_training_completion(run_dir)


if __name__ == "__main__":
    unittest.main()
