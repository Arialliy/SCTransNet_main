from __future__ import annotations

import re
import stat
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
WORKER = EXPERIMENTS / "run_tpd_clean_v4_screen800_2x5090_worker.sh"
LAUNCHER = EXPERIMENTS / "launch_tpd_clean_v4_screen800_2x5090.sh"
STATUS = EXPERIMENTS / "status_tpd_clean_v4_screen800_2x5090.sh"

GPU2 = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
GPU3 = "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
GPU0 = "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70"
GPU1 = "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640"
RESULT_ROOT = "tpd_clean_v4_screen800_2x5090_v1"
RUN_TAG = "screen800_pd_fp32_shared2x5090_v1"
UNIT_PREFIX = "sctransnet-tpd-clean-v4-2x-"
SOURCE_LOCK = "tpd_clean_v4_screen800_2x_source_lock.json"
SOURCE_LOCK_SCHEMA = "sctransnet_tpd_clean_v4_screen800_2x_source_lock_v1"
LAUNCH_SCHEMA = "sctransnet_tpd_clean_v4_screen800_2x5090_launch_v1"
CANDIDATE_FAMILY = "spd_anchored_tpd_clean_v4_single_logit_kcs"
TRAINER = "experiments/train_tpd_clean_v4.py"
EVALUATOR = "experiments/evaluate_tpd_clean_v4_pd_fa.py"

EXPECTED_MAPPING = (
    ("tpd_clean_v4_full", 42, GPU2, "full-s42"),
    ("tpd_clean_v4_sal_capacity", 42, GPU3, "cap-s42"),
    ("tpd_clean_v4_full", 3407, GPU3, "full-s3407"),
    ("tpd_clean_v4_sal_capacity", 3407, GPU2, "cap-s3407"),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TpdCleanV4TwoGpuRuntimeTests(unittest.TestCase):
    def test_shell_scripts_are_executable_and_parse(self) -> None:
        for script in (WORKER, LAUNCHER, STATUS):
            self.assertTrue(script.is_file(), script)
            self.assertTrue(
                script.stat().st_mode & stat.S_IXUSR,
                f"{script} must be executable",
            )
            completed = subprocess.run(
                ["bash", "-n", str(script)],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"{script}: {completed.stderr}",
            )

    def test_only_physical_gpu2_and_gpu3_are_bound(self) -> None:
        for path in (WORKER, LAUNCHER, STATUS):
            text = _read(path)
            self.assertIn(GPU2, text)
            self.assertIn(GPU3, text)
            self.assertNotIn(GPU0, text)
            self.assertNotIn(GPU1, text)
        launcher = _read(LAUNCHER)
        self.assertIn('"$v4_actual_index" != "2"', launcher)
        self.assertIn('"$v4_actual_index" != "3"', launcher)

    def test_counterbalanced_four_job_mapping_is_exact(self) -> None:
        worker = _read(WORKER)
        launcher = _read(LAUNCHER)
        status = _read(STATUS)
        for variant, seed, gpu_uuid, unit_tag in EXPECTED_MAPPING:
            mapping = f"{variant}:{seed}:{gpu_uuid}"
            expected_job = f"{mapping}:{unit_tag}"
            self.assertIn(mapping, worker)
            self.assertIn(expected_job, launcher)
            self.assertIn(unit_tag, status)
        self.assertEqual(
            tuple(item[2] for item in EXPECTED_MAPPING).count(GPU2), 2
        )
        self.assertEqual(
            tuple(item[2] for item in EXPECTED_MAPPING).count(GPU3), 2
        )
        self.assertIn("invalid_gpu_multiplicity", launcher)
        self.assertIn("counterbalanced_mapping_mismatch", launcher)
        self.assertIn("concurrent_jobs_per_gpu=2", launcher)

    def test_v4_variant_schema_root_and_family_are_isolated(self) -> None:
        worker = _read(WORKER)
        combined = "\n".join(
            _read(path) for path in (WORKER, LAUNCHER, STATUS)
        )
        for expected in (
            RESULT_ROOT,
            RUN_TAG,
            UNIT_PREFIX,
            SOURCE_LOCK,
            SOURCE_LOCK_SCHEMA,
            LAUNCH_SCHEMA,
            "tpd_clean_v4_full",
            "tpd_clean_v4_sal_capacity",
        ):
            self.assertIn(expected, combined)
        self.assertIn(CANDIDATE_FAMILY, worker)
        self.assertIn('"allowed_gpu_indices": [2, 3]', worker)
        self.assertIn('"concurrent_jobs_per_gpu": 2', worker)
        self.assertIn('"counterbalanced_mapping": True', worker)
        self.assertIn(
            "tpd_clean_v3_screen800_source_lock.json",
            worker,
        )
        for forbidden_active_binding in (
            "train_tpd_clean_v3.py",
            "evaluate_tpd_clean_v3",
            "tpd_clean_v3_full",
            "tpd_clean_v3_sal_capacity",
            "sctransnet-tpd-clean-v3",
        ):
            self.assertNotIn(forbidden_active_binding, combined)
        self.assertNotIn("TPDCLEANV3", combined)

    def test_all_worker_and_launcher_markers_use_v4_2x_prefix(self) -> None:
        for path in (WORKER, LAUNCHER):
            text = _read(path)
            self.assertIsNone(
                re.search(r"TPDCLEANV4(?!_2X)", text),
                f"unscoped marker in {path}",
            )

    def test_training_is_fresh_800_epochs_and_has_no_resume_path(self) -> None:
        worker = _read(WORKER)
        launcher = _read(LAUNCHER)
        combined = "\n".join((worker, launcher, _read(STATUS)))
        self.assertIn(TRAINER, worker)
        self.assertIn("--epochs 800", worker)
        self.assertIn(
            '[[ "$(wc -l < "$v4_run_dir/metrics.jsonl")" -eq 800 ]]',
            worker,
        )
        self.assertIn('"fresh_run": True', worker)
        self.assertIn("run_path_not_fresh", worker)
        self.assertIn("run_path_not_fresh", launcher)
        self.assertNotRegex(combined, r"(?i)(?:^|[^a-z])resume(?:[^a-z]|$)")
        self.assertNotIn("--expected-resume-epoch", combined)
        self.assertNotIn("resume_training", combined)

    def test_closed_interval_evaluator_and_endpoint_audit_are_fixed(self) -> None:
        worker = _read(WORKER)
        launcher = _read(LAUNCHER)
        self.assertIn(EVALUATOR, worker)
        self.assertIn("evaluate_tpd_clean_v4_pd_fa", launcher)
        self.assertNotIn("evaluate_tpd_clean_v3", worker)
        for expected in (
            ".threshold_provenance.posthoc_endpoint_completion == false",
            ".threshold_provenance.preregistered_endpoint_completion == true",
            (
                '.threshold_provenance.endpoint_protocol_stage == '
                '"before_formal_training"'
            ),
            ".threshold_provenance.closed_probability_interval == true",
            '.threshold_provenance.score_dtype == "float32"',
            ".threshold_provenance.upper_boundary_threshold == 1",
            (
                ".threshold_provenance.upper_boundary_comparison "
                '== "prediction > threshold"'
            ),
            (
                ".threshold_provenance.upper_boundary_semantics "
                '== "empty_prediction_pd0_fa0"'
            ),
            ".points[-1].threshold == 1",
            ".points[-1].pd == 0",
            ".points[-1].fa == 0",
        ):
            self.assertIn(expected, worker)
        self.assertEqual(
            worker.count(
                "for v4_checkpoint in best.pth.tar best_miou.pth.tar"
            ),
            1,
        )

    def test_one_cpu_thread_cap_precedes_all_python_work(self) -> None:
        worker = _read(WORKER)
        cap_assignment = worker.index("v4_cpu_threads=1")
        first_python = worker.index('"$v4_python"')
        self.assertLess(cap_assignment, first_python)
        for variable in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            export = f'export {variable}="$v4_cpu_threads"'
            self.assertIn(export, worker)
            self.assertLess(worker.index(export), first_python)
            self.assertIn(f'"{variable}",', worker)
        self.assertIn('"cpu_threads_per_job": 1', worker)
        self.assertIn("cpu_threads=$v4_cpu_threads", worker)
        self.assertIn("threads_per_job=1", _read(LAUNCHER))

    def test_static_preflight_gate_precedes_launch(self) -> None:
        launcher = _read(LAUNCHER)
        gate = 'if [[ "$v4_mode" == "--preflight" ]]'
        self.assertIn(gate, launcher)
        self.assertIn("systemd-run --user", launcher)
        self.assertLess(
            launcher.index(gate),
            launcher.index("systemd-run --user"),
        )
        self.assertIn(
            'v4_unit="sctransnet-tpd-clean-v4-2x-$v4_tag"',
            launcher,
        )


if __name__ == "__main__":
    unittest.main()
