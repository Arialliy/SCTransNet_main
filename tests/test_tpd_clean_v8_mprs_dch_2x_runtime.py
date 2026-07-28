from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LANE = (
    REPO_ROOT
    / "experiments/run_tpd_clean_v8_mprs_dch_formal800_2x5090_lane.sh"
)
LAUNCHER = (
    REPO_ROOT
    / "experiments/launch_tpd_clean_v8_mprs_dch_formal800_2x5090.sh"
)

GPU2_UUID = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
GPU3_UUID = "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
AUTHORIZATION_SCHEMA = (
    "sctransnet_tpd_clean_v8_mprs_dch_formal800_authorization_v1"
)


class TPDCleanV8MPRSDCHTwoGpuRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lane = LANE.read_text(encoding="utf-8")
        cls.launcher = LAUNCHER.read_text(encoding="utf-8")

    def test_shell_entrypoints_are_executable_and_syntax_valid(self) -> None:
        for path in (LANE, LAUNCHER):
            with self.subTest(path=path.name):
                self.assertTrue(path.stat().st_mode & 0o111)
                subprocess.run(
                    ["/usr/bin/bash", "-n", str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

    def test_counterbalanced_lanes_freeze_exactly_four_formal_jobs(self) -> None:
        self.assertIn(
            "gpu2:GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562)",
            self.lane,
        )
        self.assertIn(
            "gpu3:GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3)",
            self.lane,
        )
        gpu2_case = self.lane[
            self.lane.index(f"gpu2:{GPU2_UUID})") :
            self.lane.index(f"gpu3:{GPU3_UUID})")
        ]
        gpu3_case = self.lane[
            self.lane.index(f"gpu3:{GPU3_UUID})") :
            self.lane.index("    *)")
        ]
        self.assertLess(
            gpu2_case.index("tpd_clean_v8_mprs_dch_full"),
            gpu2_case.index("tpd_clean_v8_mprs_dch_capacity"),
        )
        self.assertLess(
            gpu3_case.index("tpd_clean_v8_mprs_dch_capacity"),
            gpu3_case.index("tpd_clean_v8_mprs_dch_full"),
        )
        self.assertEqual(self.lane.count("v8_seeds=(42 3407)"), 2)
        self.assertIn("for v8_index in 0 1; do", self.lane)
        self.assertNotRegex(self.lane, r"(?:^|[;\s])&(?:$|[;\s])")

    def test_only_registered_physical_gpu2_and_gpu3_are_visible(self) -> None:
        self.assertIn(
            'export CUDA_VISIBLE_DEVICES="$v8_gpu_uuid"',
            self.lane,
        )
        self.assertIn(
            'export TPD_V8_MPRS_DCH_PHYSICAL_GPU_INDEX="$v8_physical_index"',
            self.lane,
        )
        self.assertIn(
            'export TPD_V8_MPRS_DCH_PHYSICAL_GPU_UUID="$v8_gpu_uuid"',
            self.lane,
        )
        self.assertIn("torch.cuda.device_count() != 1", self.lane)
        self.assertIn("NVIDIA GeForce RTX 5090", self.lane)
        self.assertNotRegex(
            self.lane,
            r"CUDA_VISIBLE_DEVICES\s*=\s*[\"']?[01](?:[\"']|\s|$)",
        )
        self.assertNotIn("TPD_DCH_PHYSICAL_GPU_INDEX", self.lane)
        self.assertNotIn("TPD_DCH_PHYSICAL_GPU_UUID", self.lane)

    def test_lane_selects_fresh_resume_or_validated_complete(self) -> None:
        command = self.lane.index(
            '"$v8_python" experiments/train_tpd_clean_v8_mprs_dch_exact.py'
        )
        self.assertLess(self.lane.index('print("fresh")'), command)
        self.assertLess(self.lane.index('print("exact-resume")'), command)
        self.assertLess(self.lane.index('print("complete")'), command)
        self.assertIn('v8_init_flag="--fresh"', self.lane)
        self.assertIn('v8_init_flag="--exact-resume"', self.lane)
        self.assertIn(
            "TPDCLEANV8MPRSDCH_2X_IDEMPOTENT_COMPLETE",
            self.lane,
        )
        self.assertIn("len(events) != 800", self.lane)
        self.assertIn("does not store all 17 metrics", self.lane)
        self.assertIn("incomplete V8 directory has no committed exact journal", self.lane)

    def test_lane_explicitly_forbids_v7_cross_version_resume(self) -> None:
        self.assertIn(
            "sctransnet_tpd_clean_v7_dch_exact_entry_v1",
            self.lane,
        )
        self.assertIn(
            "cross-version exact resume from a V7 protocol/journal is forbidden",
            self.lane,
        )
        self.assertIn(
            '"tpd_clean_v7_dch_exact_source_lock" in source_locks',
            self.lane,
        )
        trainer_command = self.lane[
            self.lane.index(
                '"$v8_python" experiments/train_tpd_clean_v8_mprs_dch_exact.py'
            ) :
        ]
        self.assertNotIn("train_tpd_clean_v7_dch_exact.py", trainer_command)

    def test_training_command_freezes_all_formal_axes(self) -> None:
        required = (
            'export PYTHONHASHSEED="$v8_seed"',
            'export CUBLAS_WORKSPACE_CONFIG=":4096:8"',
            "--device cuda:0",
            "--epochs 800",
            "--batch-size 16",
            "--patch-size 256",
            "--workers 0",
            "--split-seed 20260722",
            "--val-fraction 0.20",
            "--eval-every 1",
            "--base-lr 0.001",
            "--min-lr 0.00001",
            "--warmup-epochs 10",
            "--threshold 0.5",
            "--match-radius 3",
            "--tiny-area 9",
            "--eps 0.000001",
            '--exact-source-lock "$v8_source_lock"',
            '"$v8_init_flag"',
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.lane)
        command = self.lane[
            self.lane.index(
                '"$v8_python" experiments/train_tpd_clean_v8_mprs_dch_exact.py'
            ) :
        ]
        self.assertNotIn("--amp", command)

    def test_launcher_validates_fail_closed_manifest_and_all_sha_bindings(
        self,
    ) -> None:
        self.assertIn(AUTHORIZATION_SCHEMA, self.launcher)
        self.assertIn(
            'authorization.get("formal_training_authorized") is not True',
            self.launcher,
        )
        for field in (
            "training_source_lock_sha256",
            "acceptance_source_lock_sha256",
            "protocol_sha256",
            "counterfactual_report_sha256",
            "compute_benchmark_sha256",
            "cpu_smoke_sha256",
            "gpu2_smoke_sha256",
            "gpu3_smoke_sha256",
        ):
            with self.subTest(field=field):
                self.assertIn(field, self.launcher)
        for gate in (
            "v8_exact_resume_passed",
            "counterfactual_finite_passed",
            "target_correction_lift_passed",
            "fragmentation_gate_passed",
            "shift_consistency_gate_passed",
            "compute_memory_gate_passed",
            "cpu_gpu_smoke_passed",
            "source_locks_passed",
        ):
            with self.subTest(gate=gate):
                self.assertIn(gate, self.launcher)
        self.assertIn("counterfactual_gate_pass", self.launcher)
        self.assertIn("compute_memory_gate_pass", self.launcher)
        self.assertIn(
            "sctransnet_tpd_clean_v8_mprs_counterfactual_v2",
            self.launcher,
        )
        self.assertIn(
            "sctransnet_tpd_clean_v8_mprs_block_benchmark_v2",
            self.launcher,
        )
        self.assertIn(
            "sctransnet_tpd_clean_v8_mprs_dch_acceptance_source_lock_v1",
            self.launcher,
        )
        self.assertIn(
            "acceptance_source_lock_training_binding_mismatch",
            self.launcher,
        )
        self.assertIn(
            "REQUIRED_ACCEPTANCE_SOURCES",
            self.launcher,
        )
        self.assertIn(
            "TPD_CLEAN_V8_MPRS_DCH_PREFLIGHT_AMENDMENT_V1.md",
            self.launcher,
        )
        self.assertIn(
            "tpd_clean_v8_mprs_benchmark_v3/gpu2.json",
            self.launcher,
        )
        self.assertIn(
            "tpd_clean_v8_mprs_smoke_v3/gpu3.json",
            self.launcher,
        )
        self.assertIn(
            'str(smoke.get("physical_gpu_index"))',
            self.launcher,
        )
        self.assertIn(
            '"cublas_workspace_config"',
            self.launcher,
        )
        self.assertIn(
            "compute_benchmark_variant_matrix_mismatch",
            self.launcher,
        )
        self.assertIn(
            "compute_benchmark_shape_gate_failed",
            self.launcher,
        )
        self.assertIn(
            'f"{label}_variant_evidence_failed"',
            self.launcher,
        )
        self.assertIn(
            "validate_report_sources",
            self.launcher,
        )
        self.assertIn(
            "BENCHMARK_REPORT_SOURCES",
            self.launcher,
        )
        self.assertIn(
            "SMOKE_REPORT_SOURCES",
            self.launcher,
        )
        self.assertIn(
            "counterfactual_hardening_evidence_failed",
            self.launcher,
        )
        self.assertIn(
            "counterfactual_group_matrix_mismatch",
            self.launcher,
        )
        self.assertIn(
            "counterfactual_job_output_digest_mismatch",
            self.launcher,
        )

    def test_authorization_and_source_lock_precede_lane_or_systemd(self) -> None:
        authorization = self.launcher.index(
            '"$v8_validation_python" -'
        )
        first_lane = self.launcher.index(
            '"$v8_lane_runner" --preflight gpu2'
        )
        first_systemd = self.launcher.index("systemd-run --user")
        self.assertLess(authorization, first_lane)
        self.assertLess(first_lane, first_systemd)
        self.assertEqual(self.launcher.count("systemd-run --user"), 2)
        self.assertEqual(
            self.launcher.count("--property=Restart=on-failure"),
            2,
        )
        self.assertIn("concurrent_tasks_per_gpu=1", self.launcher)
        self.assertIn('"$v8_launcher" --validate-only', self.lane)

    def test_missing_or_false_authorization_stops_without_launching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing-authorization.json"
            environment = os.environ.copy()
            environment.update(
                {
                    "TPD_V8_MPRS_DCH_REPO": str(REPO_ROOT),
                    "TPD_V8_MPRS_DCH_VALIDATION_PYTHON": "/usr/bin/python3",
                    "TPD_V8_MPRS_DCH_FORMAL800_AUTHORIZATION": str(missing),
                }
            )
            absent = subprocess.run(
                ["/usr/bin/bash", str(LAUNCHER), "--validate-only"],
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(absent.returncode, 0)
            self.assertIn(
                "reason=missing_formal_authorization",
                absent.stderr,
            )
            self.assertNotIn("systemd-run", absent.stdout + absent.stderr)

            denied = root / "denied-authorization.json"
            denied.write_text(
                json.dumps(
                    {
                        "schema": AUTHORIZATION_SCHEMA,
                        "formal_training_authorized": False,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            environment[
                "TPD_V8_MPRS_DCH_FORMAL800_AUTHORIZATION"
            ] = str(denied)
            rejected = subprocess.run(
                ["/usr/bin/bash", str(LAUNCHER), "--validate-only"],
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(
                "reason=formal_training_not_authorized",
                rejected.stderr,
            )
            self.assertNotIn(
                "missing_or_nonregular_training_source_lock",
                rejected.stderr,
            )

    def test_invalid_lane_mapping_stops_before_authorization_or_gpu(self) -> None:
        completed = subprocess.run(
            [
                "/usr/bin/bash",
                str(LANE),
                "gpu2",
                GPU3_UUID,
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("reason=invalid_lane_mapping", completed.stderr)
        self.assertNotIn("missing_formal_authorization", completed.stderr)
        self.assertNotIn("gpu_identity_mismatch", completed.stderr)


if __name__ == "__main__":
    unittest.main()
