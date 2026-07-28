from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import gpu23_memory_reservation_guard as subject


class GPU23MemoryReservationGuardTests(unittest.TestCase):
    def snapshot(
        self,
        *,
        physical_gpu_index: int = 2,
        free_mib: int = 22851,
    ) -> subject.GpuSnapshot:
        spec = subject.GPU_SPECS[physical_gpu_index]
        return subject.GpuSnapshot(
            physical_gpu_index=physical_gpu_index,
            uuid=spec["uuid"],
            name=subject.EXPECTED_GPU_NAME,
            total_mib=32607,
            used_mib=32607 - free_mib,
            free_mib=free_mib,
        )

    def test_fixed_mapping_excludes_gpu_zero_and_one(self) -> None:
        self.assertEqual(set(subject.GPU_SPECS), {2, 3})
        self.assertEqual(
            subject.GPU_SPECS[2]["uuid"],
            "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562",
        )
        self.assertEqual(
            subject.GPU_SPECS[3]["uuid"],
            "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3",
        )

    def test_defaults_leave_conservative_free_memory(self) -> None:
        gpu2 = subject.GPU_SPECS[2]
        plan2 = subject.build_reservation_plan(
            self.snapshot(physical_gpu_index=2, free_mib=22851),
            reserve_mib=gpu2["default_reserve_mib"],
            min_free_mib=gpu2["default_min_free_mib"],
        )
        self.assertEqual(plan2.reserve_mib, 1024)
        self.assertEqual(plan2.min_free_mib, 4096)
        self.assertEqual(plan2.cuda_context_allowance_mib, 768)
        self.assertGreaterEqual(plan2.projected_free_after_mib, 4096)

        gpu3 = subject.GPU_SPECS[3]
        plan3 = subject.build_reservation_plan(
            self.snapshot(physical_gpu_index=3, free_mib=32109),
            reserve_mib=gpu3["default_reserve_mib"],
            min_free_mib=gpu3["default_min_free_mib"],
        )
        self.assertEqual(plan3.reserve_mib, 8192)
        self.assertEqual(plan3.min_free_mib, 6144)
        self.assertGreaterEqual(plan3.projected_free_after_mib, 6144)

    def test_plan_adapts_to_safe_quantum_and_rejects_zero_capacity(self) -> None:
        plan = subject.build_reservation_plan(
            self.snapshot(physical_gpu_index=2, free_mib=5870),
            reserve_mib=1024,
            min_free_mib=4096,
        )
        self.assertEqual(plan.requested_reserve_mib, 1024)
        self.assertEqual(plan.reserve_mib, 768)
        self.assertTrue(plan.adaptive_reduction_applied)
        self.assertGreaterEqual(plan.projected_free_after_mib, 4096)
        with self.assertRaisesRegex(subject.ReservationError, "minimum 256 MiB"):
            subject.build_reservation_plan(
                self.snapshot(physical_gpu_index=2, free_mib=5000),
                reserve_mib=1024,
                min_free_mib=4096,
            )

    def test_snapshot_parser_and_uuid_validation(self) -> None:
        line = (
            "2, GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562, "
            "NVIDIA GeForce RTX 5090, 32607, 9259, 22851"
        )
        snapshot = subject.parse_gpu_snapshot(line, expected_index=2)
        self.assertEqual(snapshot.free_mib, 22851)
        self.assertEqual(snapshot.physical_gpu_index, 2)

        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=line.replace(subject.GPU_SPECS[2]["uuid"], "GPU-wrong"),
            stderr="",
        )
        with (
            mock.patch.object(
                subject.subprocess,
                "run",
                return_value=completed,
            ),
            self.assertRaisesRegex(subject.ReservationError, "UUID mismatch"),
        ):
            subject.query_gpu_snapshot(2)

    @mock.patch.object(
        subject, "inspect_existing_compute_processes", return_value=[]
    )
    def test_preflight_does_not_import_torch_or_write_state(
        self, _lane: mock.Mock
    ) -> None:
        snapshot = self.snapshot()
        with tempfile.TemporaryDirectory() as temporary:
            state_root = Path(temporary) / "state"
            with mock.patch.object(
                subject,
                "query_gpu_snapshot",
                return_value=snapshot,
            ):
                result = subject.main(
                    [
                        "preflight",
                        "--physical-gpu",
                        "2",
                        "--state-root",
                        str(state_root),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertFalse(state_root.exists())
            self.assertNotIn("torch", subject.__dict__)

    def test_gpu3_allows_existing_foreign_process(self) -> None:
        plan = subject.build_reservation_plan(
            self.snapshot(physical_gpu_index=3, free_mib=15127),
            reserve_mib=8192,
            min_free_mib=6144,
        )
        foreign = subject.ComputeProcess(
            gpu_uuid=subject.GPU_SPECS[3]["uuid"],
            pid=30477,
            process_name="/home/experiments/acc_test/bench_segment",
            used_mib=16966,
        )
        with mock.patch.object(
            subject,
            "query_compute_processes",
            return_value=[foreign],
        ):
            observed = subject.inspect_existing_compute_processes(plan)
        self.assertEqual(observed, [foreign])


    def test_gpu2_requires_fixed_v4_main_pid_but_allows_foreign(self) -> None:
        plan = subject.build_reservation_plan(
            self.snapshot(physical_gpu_index=2, free_mib=5870),
            reserve_mib=1024,
            min_free_mib=4096,
        )
        v4_process = subject.ComputeProcess(
            gpu_uuid=subject.GPU_SPECS[2]["uuid"],
            pid=18823,
            process_name=subject.GPU2_V4_PROCESS_NAME,
            used_mib=9252,
        )
        foreign = subject.ComputeProcess(
            gpu_uuid=subject.GPU_SPECS[2]["uuid"],
            pid=30456,
            process_name="/home/experiments/acc_test/bench_segment",
            used_mib=16966,
        )
        with (
            mock.patch.object(
                subject,
                "query_compute_processes",
                return_value=[v4_process, foreign],
            ),
            mock.patch.object(
                subject,
                "query_gpu2_v4_main_pid",
                return_value=18823,
            ),
        ):
            observed = subject.inspect_existing_compute_processes(plan)
        self.assertEqual(observed, [v4_process, foreign])

    def test_fixed_v4_unit_main_pid_contract(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="ActiveState=active\nSubState=running\nMainPID=18823\n",
            stderr="",
        )
        with mock.patch.object(
            subject.subprocess,
            "run",
            return_value=completed,
        ) as run:
            self.assertEqual(subject.query_gpu2_v4_main_pid(), 18823)
        command = run.call_args.args[0]
        self.assertIn(subject.GPU2_V4_UNIT, command)
        self.assertIn("--property=ActiveState,SubState,MainPID", command)

    def test_chunk_closes_early_and_metadata_uses_actual_allocation(self) -> None:
        blocked = subject.next_allocation_chunk_mib(
            free_mib=6655,
            remaining_mib=7936,
            min_free_mib=6144,
        )
        self.assertEqual(blocked, 0)
        allowed = subject.next_allocation_chunk_mib(
            free_mib=6656,
            remaining_mib=7936,
            min_free_mib=6144,
        )
        self.assertEqual(allowed, 256)
        plan = subject.build_reservation_plan(
            self.snapshot(physical_gpu_index=3, free_mib=14999),
            reserve_mib=8192,
            min_free_mib=6144,
        )
        metadata = subject.actual_allocation_metadata(plan, 7680)
        self.assertEqual(metadata["planned_reserve_mib"], 7936)
        self.assertEqual(metadata["reserve_mib"], 7680)
        self.assertTrue(metadata["adaptive_reduction_applied"])
        with self.assertRaisesRegex(subject.ReservationError, "no 256 MiB"):
            subject.actual_allocation_metadata(plan, 0)

    def test_partial_success_exception_final_state_keeps_actual_memory(self) -> None:
        plan = subject.build_reservation_plan(
            self.snapshot(physical_gpu_index=3, free_mib=14999),
            reserve_mib=8192,
            min_free_mib=6144,
        )
        base_state = {"reserve_mib": 0, "allocated_mib": 0}
        final_state = subject.build_final_reservation_state(
            base_state=base_state,
            plan=plan,
            allocated_mib=512,
            exit_code=2,
            exit_reason="allocation_or_monitor_failure",
            final_free_mib=14900,
        )
        self.assertEqual(final_state["status"], "self_released")
        self.assertEqual(final_state["reserve_mib"], 512)
        self.assertEqual(final_state["allocated_mib"], 512)
        self.assertTrue(final_state["adaptive_reduction_applied"])
        self.assertEqual(final_state["release_reason"], "allocation_or_monitor_failure")

    def test_status_marks_dead_active_record_as_inactive(self) -> None:
        with mock.patch.object(subject, "process_is_alive", return_value=False):
            payload = subject.status_payload(
                snapshot=self.snapshot(),
                recorded_state={"status": "active", "pid": 12345},
            )
        self.assertFalse(payload["active"])
        self.assertFalse(payload["holder_process_alive"])

    def test_atomic_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu2.json"
            subject.write_json_atomic(path, {"status": "active", "pid": os.getpid()})
            self.assertEqual(
                subject.read_state(path),
                {"status": "active", "pid": os.getpid()},
            )
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_launcher_has_no_gpu_zero_or_one_start_path(self) -> None:
        launcher = (
            subject.REPO_ROOT
            / "experiments"
            / "manage_gpu23_memory_reservation.sh"
        )
        text = launcher.read_text(encoding="utf-8")
        fixed_python = Path("/home/ly/BasicIRSTD/infrarenet/bin/python")
        resolved_python = fixed_python.resolve(strict=True)
        self.assertTrue(fixed_python.is_symlink())
        self.assertEqual(resolved_python, Path("/usr/bin/python3.12"))
        self.assertTrue(resolved_python.is_file())
        self.assertTrue(os.access(resolved_python, os.X_OK))
        python_probe = subprocess.run(
            [
                str(fixed_python),
                "-c",
                "import sys; print(sys.executable); print(sys.prefix)",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(python_probe.returncode, 0, python_probe.stderr)
        self.assertEqual(
            python_probe.stdout.splitlines(),
            [str(fixed_python), "/home/ly/BasicIRSTD/infrarenet"],
        )
        rejected_environment = os.environ.copy()
        rejected_environment["SCTransNet_GPU23_RESERVATION_PYTHON"] = "/usr/bin/python3"
        rejected = subprocess.run(
            [str(launcher), "status", "--physical-gpu", "2"],
            check=False,
            capture_output=True,
            text=True,
            env=rejected_environment,
        )
        self.assertEqual(rejected.returncode, 1)
        self.assertIn("unexpected_python_path", rejected.stderr)
        self.assertIn('--property=Restart=no', text)
        self.assertIn('    direct-start)', text)
        self.assertIn('GPU23_RESERVATION_DIRECT_LAUNCHED', text)
        self.assertNotIn('--property=Restart=on-failure', text)
        self.assertNotIn('watch-start', text)
        self.assertIn('readlink -f -- "$gpu23_python"', text)
        self.assertIn('"$gpu23_physical_index" != "2"', text)
        self.assertIn('"$gpu23_physical_index" != "3"', text)
        self.assertNotIn("kill -9", text)
        parsed = subprocess.run(
            ["bash", "-n", str(launcher)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(parsed.returncode, 0, parsed.stderr)


if __name__ == "__main__":
    unittest.main()
