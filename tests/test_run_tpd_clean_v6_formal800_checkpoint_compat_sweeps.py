from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from experiments import run_tpd_clean_v6_formal800_checkpoint_compat_sweeps as subject
from experiments import summarize_tpd_clean_v6_formal800 as summary


class CompatibilitySweepRunnerTests(unittest.TestCase):
    def test_fixed_matrix_uses_adapter_for_original_eight_paths(self) -> None:
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
            spec = summary.ROLE_SPECS[job["role"]]
            self.assertEqual(Path(job["command"][1]), subject.EVALUATOR)
            self.assertEqual(Path(job["output"]).name, spec["sweep"])
            self.assertEqual(
                job["command"][job["command"].index("--checkpoint") + 1],
                spec["checkpoint"],
            )
            self.assertNotIn("--overwrite", job["command"])

    def test_preflight_does_not_start_subprocess(self) -> None:
        incomplete = {"formal_matrix_complete": False, "runs": {}}
        with (
            mock.patch.object(
                subject.compat,
                "validate_compatibility_source_lock",
                return_value=({}, "a" * 64),
            ),
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

    def test_run_rejects_incomplete_matrix_before_subprocess(self) -> None:
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

    def test_existing_output_is_old_and_compat_validated_then_skipped(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            output = run_dir / "pd_fa_sweep_best.pth.json"
            output.write_bytes(b"{}")
            job = {
                "variant": summary.PRIMARY_VARIANT,
                "seed": 42,
                "role": "pd_primary",
                "run_directory": str(run_dir),
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
                    return_value=({}, "2" * 64),
                ),
                mock.patch.object(
                    subject.old_acceptance,
                    "validate_supplemental_source_lock",
                    return_value="3" * 64,
                ),
                mock.patch.object(
                    subject.compat,
                    "validate_compatibility_source_lock",
                    return_value=({}, "4" * 64),
                ),
                mock.patch.object(
                    subject.frozen_runner,
                    "_gpu_environment",
                    return_value={},
                ),
                mock.patch.object(
                    subject,
                    "_exclusive_postprocess_lock",
                    return_value=nullcontext(),
                ),
                mock.patch.object(subject, "sweep_jobs", return_value=[job]),
                mock.patch.object(
                    summary, "validate_existing_sweep", return_value={}
                ) as old_validate,
                mock.patch.object(
                    subject.strict,
                    "validate_sweep_payload",
                    return_value=None,
                ) as strict_validate,
                mock.patch.object(
                    subject.validator,
                    "validate_compatibility_sweep",
                    return_value={"valid": True},
                ) as compat_validate,
                mock.patch.object(subject.subprocess, "run") as process,
            ):
                result = subject.run_sweeps("cpu")
            process.assert_not_called()
            old_validate.assert_called_once()
            strict_validate.assert_called_once()
            compat_validate.assert_called_once()
            self.assertEqual(result["created"], [])
            self.assertEqual(result["validated_existing"], [output])
            self.assertEqual(output.read_bytes(), b"{}")

    def test_invalid_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            output = run_dir / "pd_fa_sweep_best.pth.json"
            output.write_bytes(b"invalid")
            job = {
                "variant": summary.PRIMARY_VARIANT,
                "seed": 42,
                "role": "pd_primary",
                "run_directory": str(run_dir),
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
                    return_value=({}, "2" * 64),
                ),
                mock.patch.object(
                    subject.old_acceptance,
                    "validate_supplemental_source_lock",
                    return_value="3" * 64,
                ),
                mock.patch.object(
                    subject.compat,
                    "validate_compatibility_source_lock",
                    return_value=({}, "4" * 64),
                ),
                mock.patch.object(
                    subject.frozen_runner,
                    "_gpu_environment",
                    return_value={},
                ),
                mock.patch.object(
                    subject,
                    "_exclusive_postprocess_lock",
                    return_value=nullcontext(),
                ),
                mock.patch.object(subject, "sweep_jobs", return_value=[job]),
                mock.patch.object(
                    summary,
                    "validate_existing_sweep",
                    side_effect=ValueError("invalid"),
                ),
                mock.patch.object(subject.subprocess, "run") as process,
            ):
                with self.assertRaisesRegex(ValueError, "invalid"):
                    subject.run_sweeps("cpu")
            process.assert_not_called()
            self.assertEqual(output.read_bytes(), b"invalid")

    def test_cli_forbids_cpu_formal_run_but_allows_cpu_preflight(
        self,
    ) -> None:
        self.assertEqual(
            subject.parse_args(["--preflight", "--device", "cpu"]).device,
            "cpu",
        )
        with self.assertRaises(SystemExit):
            subject.parse_args(["--run", "--device", "cpu"])
        parsed = subject.parse_args(
            [
                "--run",
                "--device",
                "cuda:0",
                "--physical-gpu",
                "training",
            ]
        )
        self.assertEqual(parsed.physical_gpu, "training")

    def test_job_environment_replays_training_gpu_seed_and_threads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            gpu_uuid = subject.POSTPROCESS_GPUS["2"]
            (run_dir / "protocol.json").write_text(
                json.dumps(
                    {
                        "run_identity": {
                            "training_contract": {
                                "environment": {
                                    "device_uuid": gpu_uuid,
                                    "cuda_visible_devices": gpu_uuid,
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            job = {"run_directory": str(run_dir), "seed": 42}
            with mock.patch.object(
                subject.frozen_runner,
                "_gpu_environment",
                return_value={"CUDA_VISIBLE_DEVICES": gpu_uuid},
            ) as gpu_environment:
                environment = subject._job_environment(
                    "cuda:0",
                    "training",
                    job,
                )
            gpu_environment.assert_called_once_with("cuda:0", "2")
            self.assertEqual(environment["PYTHONHASHSEED"], "42")
            self.assertEqual(
                environment["CUBLAS_WORKSPACE_CONFIG"],
                subject.compat.FORMAL_CUBLAS_WORKSPACE_CONFIG,
            )
            for key, value in subject.THREAD_ENVIRONMENT.items():
                self.assertEqual(environment[key], value)

    def test_runner_shares_nonblocking_postprocess_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".postprocess.lock"
            with subject._exclusive_postprocess_lock(path):
                with self.assertRaisesRegex(RuntimeError, "already held"):
                    with subject._exclusive_postprocess_lock(path):
                        self.fail("second lock acquisition unexpectedly passed")


if __name__ == "__main__":
    unittest.main()
