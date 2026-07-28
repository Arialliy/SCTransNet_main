from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from experiments import handoff_tpd_ner_v8_v2_to_v3 as subject


def _v2(decision: str) -> dict[str, object]:
    return {
        "status": "ready",
        "decision": decision,
        "readiness": {"required_runs_complete": True},
        "triplet": {
            "status": "ready",
            "decision": decision,
            "aggregate_full_model_gate_passed": (
                decision == subject.NOT_NEEDED_DECISION
            ),
        },
    }


class V2ToV3HandoffTests(unittest.TestCase):
    def test_waiting_and_passed_v2_never_verify_or_launch_v3(self) -> None:
        closure = mock.Mock(side_effect=AssertionError("closure called"))
        waiting = {
            "status": "waiting",
            "reason": "incomplete",
            "readiness": {"required_runs_complete": False},
        }
        with mock.patch.object(
            subject,
            "inspect_v2_result",
            return_value=waiting,
        ):
            plan = subject.build_handoff_plan(
                physical_gpu=2,
                closure_verifier=closure,
            )
        self.assertEqual(plan["status"], "waiting")
        self.assertFalse(plan["mutating"])
        self.assertEqual(plan["commands"], [])

        with mock.patch.object(
            subject,
            "inspect_v2_result",
            return_value=_v2(subject.NOT_NEEDED_DECISION),
        ):
            plan = subject.build_handoff_plan(
                physical_gpu=3,
                closure_verifier=closure,
            )
        self.assertEqual(plan["status"], "v3_not_needed")
        self.assertEqual(plan["commands"], [])
        closure.assert_not_called()

    def test_failed_v2_builds_exactly_one_frozen_launcher_command(
        self,
    ) -> None:
        closure = {"v3_acceptance_source_lock_sha256": "a" * 64}
        with mock.patch.object(
            subject,
            "inspect_v2_result",
            return_value=_v2(subject.READY_DECISION),
        ):
            plan = subject.build_handoff_plan(
                physical_gpu=2,
                closure_verifier=lambda: closure,
            )
        self.assertEqual(plan["status"], "ready_to_launch_v3")
        self.assertFalse(plan["mutating"])
        self.assertEqual(plan["physical_gpu"], 2)
        self.assertEqual(
            plan["commands"],
            [
                [
                    str(subject.V3_LAUNCHER.resolve()),
                    "--physical-gpu",
                    "2",
                ]
            ],
        )
        self.assertFalse(plan["v1_v2_services_modified"])
        self.assertFalse(plan["v1_v2_artifacts_modified"])

    def test_execute_invokes_one_launcher_and_no_other_command(self) -> None:
        with mock.patch.object(
            subject,
            "inspect_v2_result",
            return_value=_v2(subject.READY_DECISION),
        ):
            plan = subject.build_handoff_plan(
                physical_gpu=3,
                closure_verifier=lambda: {"verified": True},
            )
        calls: list[tuple[object, bool, bool]] = []

        def runner(
            command: object,
            *,
            check: bool,
            capture_output: bool,
        ) -> subprocess.CompletedProcess[object]:
            calls.append((command, check, capture_output))
            return subprocess.CompletedProcess(command, 0)

        result = subject.execute_handoff(plan, runner=runner)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], plan["commands"][0])
        self.assertTrue(calls[0][1])
        self.assertFalse(calls[0][2])
        self.assertEqual(result["status"], "v3_launcher_invoked")
        self.assertEqual(result["executed_command_count"], 1)
        self.assertFalse(result["v1_v2_services_modified"])
        self.assertFalse(result["v1_v2_artifacts_modified"])

    def test_invalid_gpu_and_nonlaunchable_plan_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "GPU must be 2 or 3"):
            subject.build_handoff_plan(physical_gpu=1)
        with self.assertRaisesRegex(ValueError, "not launchable"):
            subject.execute_handoff(
                {
                    "status": "waiting",
                    "commands": [],
                }
            )

    def test_cli_modes_are_read_only_unless_execute_is_explicit(self) -> None:
        default = subject.parse_args([])
        self.assertFalse(default.execute)
        self.assertFalse(default.dry_run)
        self.assertFalse(default.status)
        self.assertEqual(default.physical_gpu, 2)
        dry = subject.parse_args(["--dry-run", "--physical-gpu", "3"])
        self.assertTrue(dry.dry_run)
        self.assertFalse(dry.execute)
        with self.assertRaises(SystemExit):
            subject.parse_args(["--dry-run", "--execute"])

    def test_source_contains_no_v1_or_v2_service_mutation(self) -> None:
        text = subject.Path(subject.__file__).read_text(encoding="utf-8")
        for fragment in (
            "systemctl stop",
            "systemctl restart",
            "systemctl reset-failed",
            "systemd-run",
        ):
            self.assertNotIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
