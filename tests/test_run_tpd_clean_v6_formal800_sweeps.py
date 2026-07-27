from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from experiments import run_tpd_clean_v6_formal800_sweeps as subject
from experiments import summarize_tpd_clean_v6_formal800 as summary


class V6Formal800SweepRunnerTests(unittest.TestCase):
    def test_fixed_job_matrix_has_two_roles_for_four_runs(self) -> None:
        jobs = subject.sweep_jobs("cpu")
        self.assertEqual(len(jobs), 8)
        self.assertEqual(
            {
                (job["variant"], job["seed"], job["role"])
                for job in jobs
            },
            {
                (variant, seed, role)
                for variant in summary.VARIANTS
                for seed in summary.SEEDS
                for role in summary.ROLE_SPECS
            },
        )
        for job in jobs:
            command = job["command"]
            spec = summary.ROLE_SPECS[job["role"]]
            self.assertEqual(
                command[command.index("--checkpoint") + 1],
                spec["checkpoint"],
            )
            self.assertEqual(
                command[command.index("--expected-epochs") + 1],
                "800",
            )
            self.assertNotIn("--overwrite", command)
            self.assertEqual(
                Path(job["output"]).name,
                spec["sweep"],
            )

    def test_preflight_never_starts_a_subprocess(self) -> None:
        incomplete = {
            "formal_matrix_complete": False,
            "gate_evaluated": False,
            "engineering_gate_passed": None,
            "runs": {},
        }
        with (
            mock.patch.object(
                summary,
                "inspect_training_readiness",
                return_value=incomplete,
            ),
            mock.patch.object(subject.subprocess, "run") as process,
        ):
            result = subject.preflight("cpu")
        process.assert_not_called()
        self.assertFalse(result["formal_matrix_complete"])
        self.assertEqual(result["subprocesses_started"], 0)
        self.assertEqual(result["outputs_written"], 0)

    def test_run_mode_is_forbidden_until_all_four_runs_are_complete(self) -> None:
        with (
            mock.patch.object(
                summary,
                "inspect_training_readiness",
                return_value={"formal_matrix_complete": False},
            ),
            mock.patch.object(subject.subprocess, "run") as process,
        ):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                subject.run_sweeps("cpu")
        process.assert_not_called()

    def test_valid_existing_sweep_is_strictly_checked_then_skipped(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "pd_fa_sweep_best.pth.json"
            output.write_bytes(b"existing")
            job = {
                "variant": summary.PRIMARY_VARIANT,
                "seed": 42,
                "role": "pd_primary",
                "run_directory": str(root),
                "output": str(output),
                "command": ["never"],
            }
            training = {
                "source_sha256": {
                    "experiments/evaluate_tpd_clean_v6_pd_fa.py": "1" * 64
                }
            }
            with (
                mock.patch.object(
                    summary,
                    "inspect_training_readiness",
                    return_value={"formal_matrix_complete": True},
                ),
                mock.patch.object(
                    summary,
                    "_validate_current_training_contract",
                    return_value=(
                        training,
                        summary.EXPECTED_TRAINING_LOCK_SHA256,
                    ),
                ),
                mock.patch.object(
                    summary,
                    "validate_postprocess_source_lock",
                    return_value=({}, "0" * 64),
                ),
                mock.patch.object(subject, "sweep_jobs", return_value=[job]),
                mock.patch.object(
                    summary, "validate_existing_sweep", return_value={}
                ) as validate,
                mock.patch.object(subject.subprocess, "run") as process,
            ):
                result = subject.run_sweeps("cpu")
            process.assert_not_called()
            validate.assert_called_once()
            self.assertEqual(result["created"], [])
            self.assertEqual(result["validated_existing"], [output])
            self.assertEqual(output.read_bytes(), b"existing")

    def test_invalid_existing_sweep_stops_without_overwrite(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "pd_fa_sweep_best.pth.json"
            output.write_bytes(b"invalid")
            job = {
                "variant": summary.PRIMARY_VARIANT,
                "seed": 42,
                "role": "pd_primary",
                "run_directory": str(root),
                "output": str(output),
                "command": ["never"],
            }
            training = {
                "source_sha256": {
                    "experiments/evaluate_tpd_clean_v6_pd_fa.py": "1" * 64
                }
            }
            with (
                mock.patch.object(
                    summary,
                    "inspect_training_readiness",
                    return_value={"formal_matrix_complete": True},
                ),
                mock.patch.object(
                    summary,
                    "_validate_current_training_contract",
                    return_value=(
                        training,
                        summary.EXPECTED_TRAINING_LOCK_SHA256,
                    ),
                ),
                mock.patch.object(
                    summary,
                    "validate_postprocess_source_lock",
                    return_value=({}, "0" * 64),
                ),
                mock.patch.object(subject, "sweep_jobs", return_value=[job]),
                mock.patch.object(
                    summary,
                    "validate_existing_sweep",
                    side_effect=summary.IncompleteArtifact("invalid"),
                ),
                mock.patch.object(subject.subprocess, "run") as process,
            ):
                with self.assertRaises(summary.IncompleteArtifact):
                    subject.run_sweeps("cpu")
            process.assert_not_called()
            self.assertEqual(output.read_bytes(), b"invalid")

    def test_cuda_run_requires_an_explicit_physical_gpu2_or3(self) -> None:
        with self.assertRaisesRegex(ValueError, "physical-gpu"):
            subject._gpu_environment("cuda:0", None)
        with self.assertRaisesRegex(ValueError, "physical-gpu"):
            subject._gpu_environment("cuda:0", "0")
        self.assertEqual(subject.parse_args(["--run"]).device, "cpu")
        with self.assertRaises(SystemExit):
            subject.parse_args(["--run", "--device", "cuda:0"])


if __name__ == "__main__":
    unittest.main()
