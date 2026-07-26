from __future__ import annotations

import re
import stat
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"
WORKER = EXPERIMENTS / "run_tpd_clean_v3_screen800_2x5090_worker.sh"
LAUNCHER = EXPERIMENTS / "launch_tpd_clean_v3_screen800_2x5090.sh"
STATUS = EXPERIMENTS / "status_tpd_clean_v3_screen800_2x5090.sh"
PROTOCOL = EXPERIMENTS / "TPD_CLEAN_V3_2GPU_PROTOCOL.md"
FOUR_GPU_WORKER = (
    EXPERIMENTS / "run_tpd_clean_v3_screen800_4x5090_worker.sh"
)

GPU2 = "GPU-4a0f4ab5-9d4e-20d9-4e7a-515e2d4e0562"
GPU3 = "GPU-8d68eb9e-49d3-67f6-f715-6ef2ac4975c3"
GPU0 = "GPU-9ac47fe9-13d6-06e8-d0d6-6de812bc3c70"
GPU1 = "GPU-3cc18a8a-e7fd-ee2f-c302-e778feabe640"
RESULT_ROOT = "tpd_clean_v3_screen800_2x5090_v1"
RUN_TAG = "screen800_pd_fp32_shared2x5090_v1"
UNIT_PREFIX = "sctransnet-tpd-clean-v3-2x-"
SOURCE_LOCK = "tpd_clean_v3_screen800_2x_source_lock.json"
SOURCE_LOCK_SCHEMA = "sctransnet_tpd_clean_v3_screen800_2x_source_lock_v1"
LAUNCH_SCHEMA = "sctransnet_tpd_clean_v3_screen800_2x5090_launch_v1"

EXPECTED_MAPPING = (
    ("tpd_clean_v3_full", 42, GPU2, "full-s42"),
    ("tpd_clean_v3_sal_capacity", 42, GPU3, "cap-s42"),
    ("tpd_clean_v3_full", 3407, GPU3, "full-s3407"),
    ("tpd_clean_v3_sal_capacity", 3407, GPU2, "cap-s3407"),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _training_command(text: str) -> str:
    start = text.index('"$v3_python" experiments/train_tpd_clean_v3.py')
    end = text.index('\n\n[[ "$(wc -l', start)
    return text[start:end]


class TpdCleanV3TwoGpuRuntimeTests(unittest.TestCase):
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

    def test_only_gpu2_and_gpu3_are_bound(self) -> None:
        for path in (WORKER, LAUNCHER, STATUS, PROTOCOL):
            text = _read(path)
            self.assertIn(GPU2, text)
            self.assertIn(GPU3, text)
            self.assertNotIn(GPU0, text)
            self.assertNotIn(GPU1, text)

    def test_counterbalanced_mapping_is_exact(self) -> None:
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

    def test_isolated_names_and_contracts(self) -> None:
        combined = "\n".join(
            _read(path) for path in (WORKER, LAUNCHER, STATUS, PROTOCOL)
        )
        for expected in (
            RESULT_ROOT,
            RUN_TAG,
            UNIT_PREFIX,
            SOURCE_LOCK,
            SOURCE_LOCK_SCHEMA,
            LAUNCH_SCHEMA,
        ):
            self.assertIn(expected, combined)
        self.assertIn('"allowed_gpu_indices": [2, 3]', _read(WORKER))
        self.assertIn('"concurrent_jobs_per_gpu": 2', _read(WORKER))
        self.assertIn('"counterbalanced_mapping": True', _read(WORKER))

    def test_all_runtime_markers_use_2x_prefix(self) -> None:
        for path in (WORKER, LAUNCHER):
            text = _read(path)
            self.assertIsNone(
                re.search(r"TPDCLEANV3(?!_2X)", text),
                f"unscoped marker in {path}",
            )

    def test_training_command_matches_four_gpu_runner(self) -> None:
        self.assertEqual(
            _training_command(_read(WORKER)),
            _training_command(_read(FOUR_GPU_WORKER)),
        )

    def test_launcher_does_not_start_during_static_validation(self) -> None:
        launcher = _read(LAUNCHER)
        self.assertIn('if [[ "$v3_mode" == "--preflight" ]]', launcher)
        self.assertIn("systemd-run --user", launcher)
        self.assertIn('v3_unit="sctransnet-tpd-clean-v3-2x-$v3_tag"', launcher)


if __name__ == "__main__":
    unittest.main()
