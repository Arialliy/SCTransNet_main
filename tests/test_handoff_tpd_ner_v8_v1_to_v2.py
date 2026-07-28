from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import handoff_tpd_ner_v8_v1_to_v2 as subject
from experiments import postprocess_tpd_ner_v8_mprs_dch_formal800 as v1_post


def _point(count: int = 188) -> dict[str, object]:
    return {
        "matched_target_count": count,
        "target_count": 189,
        "pd": count / 189,
        "fa": 1.0e-6,
    }


def _row(index: int) -> dict[str, object]:
    checkpoint_role = (
        "best_validation_pd_primary"
        if index < 3
        else "best_validation_miou_secondary"
    )
    variants = (
        "baseline_sctransnet",
        v1_post.VARIANT_OFF,
        v1_post.VARIANT_ON,
    )
    return {
        "source": "fixture",
        "variant": variants[index % 3],
        "checkpoint_role": checkpoint_role,
        "seed": 42,
        "split_seed": 20260722,
        "fixed_threshold_0_5": {
            "matched_target_count": 188,
            "target_count": 189,
            "pd": 188 / 189,
            "fa": 1.0e-6,
            "miou": 0.9,
            "false_objects_per_image": 0.01,
        },
        "pd_at_fa_budget": {
            key: _point()
            for key in v1_post.BUDGET_KEYS
        },
    }


def _report(decision: str) -> dict[str, object]:
    passed = decision == subject.FULL_MODEL_GATE_PASSED
    return {
        "schema": v1_post.SCHEMA,
        "status": "complete",
        "dataset": v1_post.DATASET,
        "training_seed": 42,
        "split_seed": 20260722,
        "official_test_accessed": False,
        "scope": "single_seed_internal_validation",
        "rows": [_row(index) for index in range(6)],
        "absolute_gate_assessments": {
            "fixture": {"passed": passed},
        },
        "paired_relay_on_gate_by_role": {
            "pd_primary": {
                "passed": passed,
                "non_inferior_budget_count": 5 if passed else 0,
                "strictly_better_budget_count": 1 if passed else 0,
            },
        },
        "all_four_absolute_checkpoint_gates_passed": passed,
        "both_role_paired_relay_on_gates_passed": passed,
        "aggregate_full_model_gate_passed": passed,
        "decision": decision,
        "readiness_binding": {
            "schema": (
                "sctransnet_tpd_ner_v8_mprs_dch_"
                "posttraining_readiness_v1"
            ),
            "training_seed": 42,
            "split_seed": 20260722,
            "both_runs_complete": True,
        },
    }


class Triplet:
    def __init__(self, root: Path, decision: str) -> None:
        self.root = root / "comparison"
        self.root.mkdir()
        self.json = self.root / v1_post.JSON_OUTPUT.name
        self.markdown = self.root / v1_post.MARKDOWN_OUTPUT.name
        self.marker = self.root / v1_post.COMPLETE_MARKER.name
        self.report = _report(decision)
        self.write()

    def write(self) -> None:
        json_bytes = v1_post._canonical_bytes(self.report)
        markdown_bytes = v1_post.render_markdown(self.report).encode("utf-8")
        marker_bytes = v1_post._completion_marker_bytes(
            self.report,
            json_bytes,
            markdown_bytes,
        )
        self.json.write_bytes(json_bytes)
        self.markdown.write_bytes(markdown_bytes)
        self.marker.write_bytes(marker_bytes)


def _v2_report(
    decision: str = subject.RETURN_TO_MODEL_OPTIMIZATION,
) -> dict[str, object]:
    passed = decision == subject.FULL_MODEL_GATE_PASSED
    return {
        "schema": subject.v2_post.SCHEMA,
        "status": "complete",
        "dataset": subject.v2_post.DATASET,
        "training_seed": 42,
        "split_seed": 20260722,
        "multi_seed_scheduled": False,
        "official_test_accessed": False,
        "scope": "single_seed_internal_validation",
        "rows": [_row(index) for index in range(6)],
        "v2_candidate_absolute_gate_by_role": {
            "pd_primary": {
                "absolute_checkpoint_gate_passed": passed,
            },
            "miou_secondary": {
                "absolute_checkpoint_gate_passed": passed,
            },
        },
        "paired_v2_on_vs_v1_off_gate_by_role": {
            role: {
                "passed": passed,
                "non_inferior_budget_count": 5 if passed else 0,
                "strictly_better_budget_count": 1 if passed else 0,
            }
            for role in ("pd_primary", "miou_secondary")
        },
        "success_components": {
            "v2_on_pd_primary_absolute": passed,
            "v2_on_miou_secondary_absolute": passed,
            "pd_primary_paired_v2_on_vs_v1_off": passed,
            "miou_secondary_paired_v2_on_vs_v1_off": passed,
            "v1_off_absolute_gate_required": False,
            "baseline_affects_decision": False,
        },
        "aggregate_full_model_gate_passed": passed,
        "decision": decision,
        "comparisons_vs_baseline": [],
        "comparison_contract": {"fixture": True},
        "v1_reference_read_only": {
            "before": {"fixture": True},
            "after": {"fixture": True},
            "unchanged": True,
        },
        "bindings": {
            "v2_training_source_lock": "/fixture/training.lock.json",
            "v2_training_source_lock_sha256": "1" * 64,
            "v2_acceptance_source_lock": "/fixture/acceptance.lock.json",
            "v2_acceptance_source_lock_sha256": "2" * 64,
            "v2_training_data_sha256": "3" * 64,
            "v2_evaluator": "/fixture/evaluator.py",
            "v2_evaluator_sha256": "4" * 64,
            "postprocess": "/fixture/postprocess.py",
            "postprocess_sha256": "5" * 64,
            "sweeps": {
                f"{variant}:{checkpoint}": {
                    "path": f"/fixture/{variant}/{checkpoint}.json",
                    "sha256": "6" * 64,
                }
                for variant in (
                    subject.v2_post.BASELINE_VARIANT,
                    subject.v2_post.VARIANT_V1_OFF,
                    subject.v2_post.VARIANT_V2_ON,
                )
                for checkpoint in subject.v2_post.CHECKPOINTS
            },
        },
        "claim_boundary": {
            "single_seed_only": True,
            "cross_seed_stability_claim": False,
            "cross_dataset_claim": False,
            "official_test_claim": False,
        },
        "readiness_binding": {
            "schema": subject.v2_post.READINESS_SCHEMA,
            "training_seed": 42,
            "split_seed": 20260722,
            "required_runs_complete": True,
        },
    }


class V2Triplet:
    def __init__(
        self,
        root: Path,
        decision: str = subject.RETURN_TO_MODEL_OPTIMIZATION,
    ) -> None:
        self.root = root / "v2_comparison"
        self.root.mkdir()
        self.json = self.root / subject.v2_post.JSON_OUTPUT.name
        self.markdown = self.root / subject.v2_post.MARKDOWN_OUTPUT.name
        self.marker = self.root / subject.v2_post.COMPLETE_MARKER.name
        self.report = _v2_report(decision)
        self.write()

    def write(self) -> None:
        json_bytes = subject.v2_post._canonical_bytes(self.report)
        markdown_bytes = subject.v2_post.render_markdown(
            self.report
        ).encode("utf-8")
        marker_bytes = subject.v2_post._completion_marker_bytes(
            self.report,
            json_bytes,
            markdown_bytes,
        )
        self.json.write_bytes(json_bytes)
        self.markdown.write_bytes(markdown_bytes)
        self.marker.write_bytes(marker_bytes)


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.states = {
            unit: "inactive"
            for unit in (
                *subject.V2_TRAINING_UNITS.values(),
                subject.V2_POSTPROCESS_UNIT,
            )
        }
        self.exec_starts = {unit: "" for unit in self.states}

    def __call__(
        self,
        command: list[str],
        *,
        check: bool,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        command = list(command)
        self.calls.append(command)
        if command[:3] == ["systemctl", "--user", "show"]:
            unit = command[3].removesuffix(".service")
            if "--property=ExecStart" in command:
                output = self.exec_starts.get(unit, "")
            else:
                output = self.states.get(unit, "inactive")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=output + "\n",
                stderr="",
            )
        if command[0] == str(subject.V2_LAUNCHER.resolve()):
            physical_gpu = int(command[command.index("--physical-gpu") + 1])
            unit = subject.V2_TRAINING_UNITS[physical_gpu]
            self.states[unit] = "active"
            self.exec_starts[unit] = (
                f"{{ path={subject.V2_LANE.resolve()} ; "
                f"argv[]={subject.V2_LANE.resolve()} {physical_gpu} "
                f"{subject.V2_GPU_UUIDS[physical_gpu]} ; "
                "ignore_errors=no ; }"
            )
        if command[0] == "systemd-run":
            self.states[subject.V2_POSTPROCESS_UNIT] = "active"
            executable_index = command.index("--property=RestartSec=10") + 1
            argv = " ".join(command[executable_index:])
            self.exec_starts[subject.V2_POSTPROCESS_UNIT] = (
                f"{{ path={command[executable_index]} ; "
                f"argv[]={argv} ; ignore_errors=no ; }}"
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


class HandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        # This suite exercises handoff behavior independently of the frozen
        # acceptance manifest. The production manifest is intentionally not
        # rewritten merely because this handoff implementation is under test.
        contract_patch = mock.patch.object(
            subject,
            "_fixed_v2_lane_contract",
            return_value={
                "variant": subject.v2_post.VARIANT_V2_ON,
                "training_seed": 42,
                "split_seed": 20260722,
                "multi_seed_scheduled": False,
                "fixture": True,
            },
        )
        contract_patch.start()
        self.addCleanup(contract_patch.stop)

    def _triplet(
        self,
        directory: str,
        decision: str = subject.RETURN_TO_MODEL_OPTIMIZATION,
    ) -> Triplet:
        return Triplet(Path(directory), decision)

    def _validate(self, triplet: Triplet) -> dict[str, object]:
        with mock.patch.object(
            subject,
            "_rebuild_v1_report",
            return_value=copy.deepcopy(triplet.report),
        ):
            return subject.validate_v1_triplet(
                triplet.json,
                triplet.markdown,
                triplet.marker,
            )

    @contextlib.contextmanager
    def _v2_paths(self, triplet: V2Triplet):
        with (
            mock.patch.object(
                subject.v2_post,
                "JSON_OUTPUT",
                triplet.json,
            ),
            mock.patch.object(
                subject.v2_post,
                "MARKDOWN_OUTPUT",
                triplet.markdown,
            ),
            mock.patch.object(
                subject.v2_post,
                "COMPLETE_MARKER",
                triplet.marker,
            ),
        ):
            yield

    def test_report_rebuild_uses_only_read_only_baseline_binding(self) -> None:
        def validated(
            path: Path,
            *,
            variant: str,
            checkpoint: str,
            binding: object,
        ) -> dict[str, str]:
            return {
                "variant": variant,
                "checkpoint": checkpoint,
            }

        def built(
            rows: dict[tuple[str, str], dict[str, str]],
            *,
            lock_bindings: object,
            baseline_contract: object,
        ) -> dict[str, object]:
            self.assertEqual(len(rows), 6)
            return {"rows": rows}

        with (
            mock.patch.object(
                subject.v1_post,
                "verify_frozen_manifests",
                return_value={"locks": True},
            ),
            mock.patch.object(
                subject.v1_post,
                "_same_split_and_training_contract",
                return_value={"baseline": True},
            ),
            mock.patch.object(
                subject.v1_post,
                "current_sweep_binding",
                return_value={"candidate": True},
            ) as candidate_binding,
            mock.patch.object(
                subject.v2_post,
                "current_reference_binding",
                return_value={"baseline": True},
            ) as baseline_binding,
            mock.patch.object(
                subject.v1_post,
                "validate_existing_sweep",
                side_effect=validated,
            ),
            mock.patch.object(
                subject.v1_post,
                "build_report",
                side_effect=built,
            ),
            mock.patch.object(
                subject.v1_post,
                "inspect_training_readiness",
                return_value={"both_runs_complete": True},
            ),
            mock.patch.object(
                subject.v1_post,
                "prepare_baseline_reference_view",
                side_effect=AssertionError("must remain read-only"),
            ) as prepare,
        ):
            rebuilt = subject._rebuild_v1_report()
        self.assertEqual(candidate_binding.call_count, 4)
        self.assertEqual(baseline_binding.call_count, 2)
        prepare.assert_not_called()
        self.assertEqual(
            rebuilt["readiness_binding"],
            {"both_runs_complete": True},
        )

    def test_valid_triplet_binds_fixed_single_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            marker_sha = hashlib.sha256(triplet.marker.read_bytes()).hexdigest()
            result = self._validate(triplet)
        self.assertEqual(
            result["decision"],
            subject.RETURN_TO_MODEL_OPTIMIZATION,
        )
        self.assertEqual(result["training_seed"], 42)
        self.assertEqual(result["split_seed"], 20260722)
        self.assertFalse(result["multi_seed_scheduled"])
        self.assertEqual(result["sha256"]["marker"], marker_sha)

    def test_pass_decision_calls_no_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(
                directory,
                subject.FULL_MODEL_GATE_PASSED,
            )

            def forbidden(*args: object, **kwargs: object) -> object:
                raise AssertionError("pass decision may not call subprocess")

            with mock.patch.object(
                subject,
                "_rebuild_v1_report",
                return_value=copy.deepcopy(triplet.report),
            ):
                result = subject.execute_handoff(
                    json_path=triplet.json,
                    markdown_path=triplet.markdown,
                    marker_path=triplet.marker,
                    lock_path=Path(directory) / "handoff.lock",
                    runner=forbidden,
                )
        self.assertEqual(result["action"], "v2_not_started_v1_gate_passed")
        self.assertFalse(result["v2_launcher_called"])
        self.assertFalse(result["v2_postprocess_service_started"])

    def test_return_decision_starts_gpu2_and_wait_service(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            runner = FakeRunner()
            with (
                mock.patch.object(
                    subject,
                    "_rebuild_v1_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
                mock.patch.object(
                    subject,
                    "v2_training_complete",
                    return_value=False,
                ),
                mock.patch.object(
                    subject,
                    "v2_postprocess_complete",
                    return_value=False,
                ),
            ):
                result = subject.execute_handoff(
                    json_path=triplet.json,
                    markdown_path=triplet.markdown,
                    marker_path=triplet.marker,
                    lock_path=Path(directory) / "handoff.lock",
                    runner=runner,
                )
        launcher = [
            call
            for call in runner.calls
            if call[0] == str(subject.V2_LAUNCHER.resolve())
        ]
        services = [call for call in runner.calls if call[0] == "systemd-run"]
        self.assertEqual(len(launcher), 1)
        self.assertEqual(launcher[0][-2:], ["--physical-gpu", "2"])
        self.assertEqual(len(services), 1)
        self.assertIn("--v2-postprocess-worker", services[0])
        self.assertNotIn("--wait-and-run", services[0])
        self.assertNotIn(str(subject.V2_POSTPROCESS.resolve()), services[0])
        self.assertTrue(result["v2_launcher_called"])
        self.assertTrue(result["v2_postprocess_service_started"])

    def test_virtualenv_python_symlink_is_preserved_in_worker_commands(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            python_link = Path(directory) / "venv-python"
            python_link.symlink_to(Path(sys.executable))
            runner = FakeRunner()
            wait_command = subject._start_v2_postprocess_wait_service(
                2,
                30.0,
                python_link,
                runner=runner,
            )
            run_command = subject._run_v2_postprocess_once(
                2,
                python_link,
                runner=runner,
            )
            identity = subject._postprocess_unit_identity(
                physical_gpu=2,
                python=python_link,
                runner=runner,
            )
            lexical_path = str(python_link.absolute())
            self.assertEqual(
                wait_command[
                    wait_command.index("--property=RestartSec=10") + 1
                ],
                lexical_path,
            )
            self.assertEqual(
                wait_command[wait_command.index("--python") + 1],
                lexical_path,
            )
            self.assertEqual(run_command[0], lexical_path)
            self.assertEqual(
                run_command[run_command.index("--python") + 1],
                lexical_path,
            )
            self.assertNotEqual(lexical_path, str(python_link.resolve()))
            self.assertTrue(identity["identity_verified"])
            self.assertIn(lexical_path, identity["exec_start"])

    def test_gpu3_is_propagated_to_both_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            runner = FakeRunner()
            with (
                mock.patch.object(
                    subject,
                    "_rebuild_v1_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
                mock.patch.object(
                    subject,
                    "v2_training_complete",
                    return_value=False,
                ),
                mock.patch.object(
                    subject,
                    "v2_postprocess_complete",
                    return_value=False,
                ),
            ):
                subject.execute_handoff(
                    json_path=triplet.json,
                    markdown_path=triplet.markdown,
                    marker_path=triplet.marker,
                    physical_gpu=3,
                    lock_path=Path(directory) / "handoff.lock",
                    runner=runner,
                )
        launcher = next(
            call
            for call in runner.calls
            if call[0] == str(subject.V2_LAUNCHER.resolve())
        )
        service = next(call for call in runner.calls if call[0] == "systemd-run")
        self.assertEqual(launcher[-1], "3")
        self.assertEqual(service[service.index("--physical-gpu") + 1], "3")

    def test_second_call_reuses_both_active_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            runner = FakeRunner()
            with (
                mock.patch.object(
                    subject,
                    "_rebuild_v1_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
                mock.patch.object(
                    subject,
                    "v2_training_complete",
                    return_value=False,
                ),
                mock.patch.object(
                    subject,
                    "v2_postprocess_complete",
                    return_value=False,
                ),
            ):
                results = [
                    subject.execute_handoff(
                        json_path=triplet.json,
                        markdown_path=triplet.markdown,
                        marker_path=triplet.marker,
                        lock_path=Path(directory) / "handoff.lock",
                        runner=runner,
                    )
                    for _ in range(2)
                ]
        self.assertEqual(
            sum(
                call[0] == str(subject.V2_LAUNCHER.resolve())
                for call in runner.calls
            ),
            1,
        )
        self.assertEqual(sum(call[0] == "systemd-run" for call in runner.calls), 1)
        self.assertTrue(
            results[1]["v2_training"]["active_unit_identity"][
                "identity_verified"
            ]
        )
        self.assertTrue(
            results[1]["v2_postprocess"]["active_unit_identity"][
                "identity_verified"
            ]
        )

    def test_completed_training_skips_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            runner = FakeRunner()
            with (
                mock.patch.object(
                    subject,
                    "_rebuild_v1_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
                mock.patch.object(
                    subject,
                    "v2_training_complete",
                    return_value=True,
                ),
                mock.patch.object(
                    subject,
                    "v2_postprocess_complete",
                    return_value=False,
                ),
            ):
                result = subject.execute_handoff(
                    json_path=triplet.json,
                    markdown_path=triplet.markdown,
                    marker_path=triplet.marker,
                    lock_path=Path(directory) / "handoff.lock",
                    runner=runner,
                )
        self.assertFalse(result["v2_launcher_called"])
        self.assertEqual(
            result["v2_training"]["action"],
            "v2_training_already_complete",
        )
        self.assertEqual(sum(call[0] == "systemd-run" for call in runner.calls), 1)

    def test_completed_postprocess_is_not_started_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            runner = FakeRunner()
            with (
                mock.patch.object(
                    subject,
                    "_rebuild_v1_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
                mock.patch.object(
                    subject,
                    "v2_training_complete",
                    return_value=True,
                ),
                mock.patch.object(
                    subject,
                    "v2_postprocess_complete",
                    return_value=True,
                ),
            ):
                result = subject.execute_handoff(
                    json_path=triplet.json,
                    markdown_path=triplet.markdown,
                    marker_path=triplet.marker,
                    lock_path=Path(directory) / "handoff.lock",
                    runner=runner,
                )
        self.assertFalse(result["v2_postprocess_service_started"])
        self.assertFalse(any(call[0] == "systemd-run" for call in runner.calls))

    def test_two_active_training_units_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            runner = FakeRunner()
            for unit in subject.V2_TRAINING_UNITS.values():
                runner.states[unit] = "active"
            with (
                mock.patch.object(
                    subject,
                    "_rebuild_v1_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
                self.assertRaisesRegex(ValueError, "more than one"),
            ):
                subject.execute_handoff(
                    json_path=triplet.json,
                    markdown_path=triplet.markdown,
                    marker_path=triplet.marker,
                    lock_path=Path(directory) / "handoff.lock",
                    runner=runner,
                )
        self.assertFalse(
            any(
                call[0] in {
                    "systemd-run",
                    str(subject.V2_LAUNCHER.resolve()),
                }
                for call in runner.calls
            )
        )

    def test_active_training_unit_with_wrong_command_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            runner = FakeRunner()
            unit = subject.V2_TRAINING_UNITS[2]
            runner.states[unit] = "active"
            runner.exec_starts[unit] = (
                "{ path=/tmp/other.sh ; argv[]=/tmp/other.sh 2 wrong ; }"
            )
            with (
                mock.patch.object(
                    subject,
                    "_rebuild_v1_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
                mock.patch.object(
                    subject,
                    "v2_training_complete",
                    return_value=False,
                ),
                mock.patch.object(
                    subject,
                    "v2_postprocess_complete",
                    return_value=False,
                ),
                self.assertRaisesRegex(ValueError, "training unit ExecStart"),
            ):
                subject.execute_handoff(
                    json_path=triplet.json,
                    markdown_path=triplet.markdown,
                    marker_path=triplet.marker,
                    lock_path=Path(directory) / "handoff.lock",
                    runner=runner,
                )
        self.assertFalse(
            any(
                call[0] in {
                    "systemd-run",
                    str(subject.V2_LAUNCHER.resolve()),
                }
                for call in runner.calls
            )
        )

    def test_active_postprocess_unit_with_wrong_command_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            runner = FakeRunner()
            runner.states[subject.V2_POSTPROCESS_UNIT] = "active"
            runner.exec_starts[subject.V2_POSTPROCESS_UNIT] = (
                "{ path=/tmp/other.py ; argv[]=/tmp/other.py ; }"
            )
            with (
                mock.patch.object(
                    subject,
                    "_rebuild_v1_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
                mock.patch.object(
                    subject,
                    "v2_training_complete",
                    return_value=True,
                ),
                mock.patch.object(
                    subject,
                    "v2_postprocess_complete",
                    return_value=False,
                ),
                self.assertRaisesRegex(ValueError, "postprocess wait unit"),
            ):
                subject.execute_handoff(
                    json_path=triplet.json,
                    markdown_path=triplet.markdown,
                    marker_path=triplet.marker,
                    lock_path=Path(directory) / "handoff.lock",
                    runner=runner,
                )
        self.assertFalse(any(call[0] == "systemd-run" for call in runner.calls))

    def test_worker_waits_then_calls_existing_run_now_cli(self) -> None:
        readiness = [
            {
                "training_seed": 42,
                "split_seed": 20260722,
                "required_runs_complete": False,
                "v1_off_read_only_control": {
                    "metrics": {"event_count": 799},
                },
                "v2_on": {"metrics": {"event_count": 400}},
            },
            {
                "training_seed": 42,
                "split_seed": 20260722,
                "required_runs_complete": True,
                "v1_off_read_only_control": {
                    "metrics": {"event_count": 800},
                },
                "v2_on": {"metrics": {"event_count": 800}},
            },
        ]
        runner = FakeRunner()
        sleep = mock.Mock()
        with (
            mock.patch.object(
                subject,
                "v2_postprocess_complete",
                side_effect=(False, False),
            ),
            mock.patch.object(
                subject.v2_post,
                "inspect_training_readiness",
                side_effect=readiness,
            ),
        ):
            result = subject.wait_for_v2_and_postprocess(
                physical_gpu=3,
                poll_seconds=1,
                python=Path("/usr/bin/python3"),
                runner=runner,
                sleep_fn=sleep,
            )
        sleep.assert_called_once_with(1)
        self.assertEqual(len(runner.calls), 1)
        command = runner.calls[0]
        self.assertEqual(command[1], str(subject.V2_POSTPROCESS.resolve()))
        self.assertIn("--run-now", command)
        self.assertEqual(command[command.index("--physical-gpu") + 1], "3")
        self.assertEqual(result["action"], "v2_postprocess_called")

    def test_worker_does_nothing_if_postprocess_is_complete(self) -> None:
        runner = FakeRunner()
        with (
            mock.patch.object(
                subject,
                "v2_postprocess_complete",
                return_value=True,
            ),
            mock.patch.object(
                subject.v2_post,
                "inspect_training_readiness",
            ) as readiness,
        ):
            result = subject.wait_for_v2_and_postprocess(runner=runner)
        readiness.assert_not_called()
        self.assertEqual(runner.calls, [])
        self.assertEqual(
            result["action"],
            "v2_postprocess_already_complete",
        )

    def test_v2_complete_rebuilds_and_validates_full_triplet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = V2Triplet(Path(directory))
            with (
                self._v2_paths(triplet),
                mock.patch.object(
                    subject,
                    "_rebuild_v2_report",
                    return_value=copy.deepcopy(triplet.report),
                ) as rebuild,
            ):
                evidence = subject.validate_v2_triplet()
                complete = subject.v2_postprocess_complete()
        self.assertEqual(rebuild.call_count, 2)
        self.assertEqual(
            evidence["decision"],
            subject.RETURN_TO_MODEL_OPTIMIZATION,
        )
        self.assertFalse(evidence["aggregate_full_model_gate_passed"])
        self.assertTrue(complete)

    def test_v2_hash_consistent_stale_binding_is_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = V2Triplet(Path(directory))
            rebuilt = copy.deepcopy(triplet.report)
            triplet.report["bindings"]["postprocess_sha256"] = "f" * 64
            triplet.write()
            with (
                self._v2_paths(triplet),
                mock.patch.object(
                    subject,
                    "_rebuild_v2_report",
                    return_value=rebuilt,
                ),
            ):
                complete = subject.v2_postprocess_complete()
        self.assertFalse(complete)

    def test_v2_marker_decision_mismatch_is_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = V2Triplet(Path(directory))
            marker = json.loads(triplet.marker.read_text(encoding="utf-8"))
            marker["decision"] = subject.FULL_MODEL_GATE_PASSED
            triplet.marker.write_bytes(
                subject.v2_post._canonical_bytes(marker)
            )
            with (
                self._v2_paths(triplet),
                mock.patch.object(
                    subject,
                    "_rebuild_v2_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
            ):
                complete = subject.v2_postprocess_complete()
        self.assertFalse(complete)

    def test_v2_hash_consistent_wrong_markdown_is_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = V2Triplet(Path(directory))
            changed = triplet.markdown.read_bytes() + b"\n"
            triplet.markdown.write_bytes(changed)
            marker = json.loads(triplet.marker.read_text(encoding="utf-8"))
            marker["outputs"][triplet.markdown.name] = hashlib.sha256(
                changed
            ).hexdigest()
            triplet.marker.write_bytes(
                subject.v2_post._canonical_bytes(marker)
            )
            with (
                self._v2_paths(triplet),
                mock.patch.object(
                    subject,
                    "_rebuild_v2_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
            ):
                complete = subject.v2_postprocess_complete()
        self.assertFalse(complete)

    def test_v2_marker_with_missing_report_sibling_is_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = V2Triplet(Path(directory))
            triplet.json.unlink()
            with self._v2_paths(triplet):
                complete = subject.v2_postprocess_complete()
        self.assertFalse(complete)

    def test_v2_unexpected_rebuild_failure_is_not_silenced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = V2Triplet(Path(directory))
            with (
                self._v2_paths(triplet),
                mock.patch.object(
                    subject,
                    "_rebuild_v2_report",
                    side_effect=RuntimeError("unexpected rebuild failure"),
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "unexpected rebuild failure",
                ),
            ):
                subject.v2_postprocess_complete()

    def test_invalid_v2_triplet_calls_existing_run_now_path(self) -> None:
        readiness = {
            "training_seed": 42,
            "split_seed": 20260722,
            "required_runs_complete": True,
            "v1_off_read_only_control": {
                "metrics": {"event_count": 800},
            },
            "v2_on": {"metrics": {"event_count": 800}},
        }
        with tempfile.TemporaryDirectory() as directory:
            triplet = V2Triplet(Path(directory))
            rebuilt = copy.deepcopy(triplet.report)
            triplet.report["bindings"]["sweeps"].pop(
                (
                    f"{subject.v2_post.VARIANT_V2_ON}:"
                    f"{subject.v2_post.CHECKPOINTS[0]}"
                )
            )
            triplet.write()
            runner = FakeRunner()
            with (
                self._v2_paths(triplet),
                mock.patch.object(
                    subject,
                    "_rebuild_v2_report",
                    return_value=rebuilt,
                ),
                mock.patch.object(
                    subject.v2_post,
                    "inspect_training_readiness",
                    return_value=readiness,
                ),
            ):
                result = subject.wait_for_v2_and_postprocess(
                    python=Path("/usr/bin/python3"),
                    runner=runner,
                )
        self.assertEqual(result["action"], "v2_postprocess_called")
        self.assertEqual(len(runner.calls), 1)
        self.assertIn("--run-now", runner.calls[0])

    def test_wait_and_run_polls_only_until_marker_is_ready(self) -> None:
        waiting = {
            "status": "waiting_for_v1_commit",
            "exists": {"json": True, "markdown": True, "marker": False},
        }
        ready = {"status": "ready"}
        sleep = mock.Mock()
        with (
            mock.patch.object(
                subject,
                "inspect_v1_triplet",
                side_effect=(waiting, ready),
            ),
            mock.patch.object(
                subject,
                "execute_handoff",
                return_value={"status": "complete"},
            ) as execute,
        ):
            result = subject.wait_and_run(
                poll_seconds=1,
                sleep_fn=sleep,
            )
        sleep.assert_called_once_with(1)
        execute.assert_called_once()
        self.assertEqual(result, {"status": "complete"})

    def test_missing_marker_is_waiting_and_does_not_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(subject, "_rebuild_v1_report") as rebuild:
                result = subject.inspect_v1_triplet(
                    root / v1_post.JSON_OUTPUT.name,
                    root / v1_post.MARKDOWN_OUTPUT.name,
                    root / v1_post.COMPLETE_MARKER.name,
                )
        rebuild.assert_not_called()
        self.assertEqual(result["status"], "waiting_for_v1_commit")

    def test_status_is_read_only_and_uses_only_show_queries(self) -> None:
        runner = FakeRunner()
        with mock.patch.object(
            subject,
            "inspect_v1_triplet",
            return_value={"status": "waiting_for_v1_commit"},
        ):
            result = subject.status_payload(runner=runner)
        self.assertFalse(result["mutations_performed"])
        self.assertTrue(runner.calls)
        self.assertTrue(
            all(call[:3] == ["systemctl", "--user", "show"] for call in runner.calls)
        )

    def test_parser_defaults_to_fixed_seed_gpu2(self) -> None:
        args = subject.parse_args(["--status"])
        self.assertTrue(args.status)
        self.assertEqual(args.physical_gpu, 2)
        self.assertFalse(hasattr(args, "seed"))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                subject.parse_args(["--status", "--seed", "7"])
            with self.assertRaises(SystemExit):
                subject.parse_args(["--status", "--poll-seconds", "0"])

    def test_json_seed_and_split_mismatch_are_rejected(self) -> None:
        for field, value in (("training_seed", 7), ("split_seed", 17)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                triplet = self._triplet(directory)
                triplet.report[field] = value
                triplet.write()
                with (
                    mock.patch.object(
                        subject,
                        "_rebuild_v1_report",
                        return_value=copy.deepcopy(triplet.report),
                    ),
                    self.assertRaisesRegex(ValueError, field),
                ):
                    subject.validate_v1_triplet(
                        triplet.json,
                        triplet.markdown,
                        triplet.marker,
                    )

    def test_gate_and_decision_inconsistency_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            triplet.report["aggregate_full_model_gate_passed"] = True
            triplet.write()
            with (
                mock.patch.object(
                    subject,
                    "_rebuild_v1_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
                self.assertRaisesRegex(ValueError, "aggregate result differs"),
            ):
                subject.validate_v1_triplet(
                    triplet.json,
                    triplet.markdown,
                    triplet.marker,
                )

    def test_marker_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            marker = json.loads(triplet.marker.read_text(encoding="utf-8"))
            marker["outputs"][triplet.json.name] = "0" * 64
            triplet.marker.write_text(
                json.dumps(marker, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    subject,
                    "_rebuild_v1_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
                self.assertRaisesRegex(ValueError, "output hashes"),
            ):
                subject.validate_v1_triplet(
                    triplet.json,
                    triplet.markdown,
                    triplet.marker,
                )

    def test_markdown_must_equal_full_render(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            changed = triplet.markdown.read_bytes() + b"\n"
            triplet.markdown.write_bytes(changed)
            marker = json.loads(triplet.marker.read_text(encoding="utf-8"))
            marker["outputs"][triplet.markdown.name] = hashlib.sha256(
                changed
            ).hexdigest()
            triplet.marker.write_text(
                json.dumps(marker, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    subject,
                    "_rebuild_v1_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
                self.assertRaisesRegex(ValueError, "Markdown differs"),
            ):
                subject.validate_v1_triplet(
                    triplet.json,
                    triplet.markdown,
                    triplet.marker,
                )

    def test_json_must_be_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            compact = json.dumps(triplet.report, sort_keys=True).encode("utf-8")
            triplet.json.write_bytes(compact)
            marker = json.loads(triplet.marker.read_text(encoding="utf-8"))
            marker["outputs"][triplet.json.name] = hashlib.sha256(
                compact
            ).hexdigest()
            triplet.marker.write_text(
                json.dumps(marker, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    subject,
                    "_rebuild_v1_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
                self.assertRaisesRegex(ValueError, "not the canonical"),
            ):
                subject.validate_v1_triplet(
                    triplet.json,
                    triplet.markdown,
                    triplet.marker,
                )

    def test_symlink_and_missing_sibling_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            target = triplet.root / "real.json"
            triplet.json.replace(target)
            triplet.json.symlink_to(target)
            with (
                mock.patch.object(
                    subject,
                    "_rebuild_v1_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
                self.assertRaisesRegex(ValueError, "regular file"),
            ):
                subject.validate_v1_triplet(
                    triplet.json,
                    triplet.markdown,
                    triplet.marker,
                )
        with tempfile.TemporaryDirectory() as directory:
            triplet = self._triplet(directory)
            triplet.markdown.unlink()
            with (
                mock.patch.object(
                    subject,
                    "_rebuild_v1_report",
                    return_value=copy.deepcopy(triplet.report),
                ),
                self.assertRaisesRegex(ValueError, "regular file"),
            ):
                subject.validate_v1_triplet(
                    triplet.json,
                    triplet.markdown,
                    triplet.marker,
                )

    def test_source_has_no_v1_stop_or_v1_launcher_action(self) -> None:
        source = Path(subject.__file__).read_text(encoding="utf-8")
        self.assertNotIn('"systemctl", "--user", "stop"', source)
        self.assertNotIn("launch_tpd_ner_v8_mprs_dch_formal800_2x5090", source)


if __name__ == "__main__":
    unittest.main()
