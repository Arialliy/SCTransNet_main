from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import run_tpd_clean_v7_dch_formal800_sweeps as subject
from experiments import (
    validate_tpd_clean_v7_dch_formal800_completion as completion,
)


class V7DCHFormal800SweepRunnerTests(unittest.TestCase):
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
                for variant in completion.VARIANTS
                for seed in completion.SEEDS
                for role in completion.ROLE_SPECS
            },
        )
        for job in jobs:
            command = job["command"]
            spec = completion.ROLE_SPECS[job["role"]]
            self.assertEqual(command[:2], [sys.executable, str(subject.EVALUATOR)])
            self.assertEqual(
                command[command.index("--checkpoint") + 1],
                spec["checkpoint"],
            )
            self.assertEqual(
                command[command.index("--expected-epochs") + 1],
                "800",
            )
            self.assertEqual(
                command[command.index("--fa-budgets") + 1 :],
                ["1e-06", "5e-06", "1e-05", "5e-05", "0.0001"],
            )
            self.assertNotIn("--overwrite", command)
            self.assertEqual(Path(job["output"]).name, spec["sweep"])
            self.assertIn("tpd_clean_v7_dch", job["variant"])

    def test_preflight_never_starts_subprocess_or_writes(self) -> None:
        incomplete = {
            "formal_matrix_complete": False,
            "runs": {},
        }
        with (
            mock.patch.object(
                completion,
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
        self.assertEqual(result["expected_runs"], 4)
        self.assertEqual(result["expected_checkpoints"], 12)
        self.assertEqual(result["expected_sweeps"], 8)
        self.assertEqual(result["native_validation_field_count"], 17)
        self.assertEqual(
            result["execution_provenance_contract"]["determinism"],
            subject.DETERMINISM_SETTINGS,
        )

    def test_run_is_forbidden_until_all_four_runs_are_complete(self) -> None:
        with (
            mock.patch.object(
                completion,
                "inspect_training_readiness",
                return_value={"formal_matrix_complete": False},
            ),
            mock.patch.object(subject.subprocess, "run") as process,
        ):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                subject.run_sweeps("cpu")
        process.assert_not_called()

    def test_valid_existing_sweep_is_checked_then_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "pd_fa_sweep_best.pth.json"
            output.write_text(
                json.dumps(
                    {
                        subject.EXECUTION_PROVENANCE_KEY:
                        subject._execution_provenance("cpu", None),
                    }
                ),
                encoding="utf-8",
            )
            job = {
                "variant": completion.PRIMARY_VARIANT,
                "seed": 42,
                "role": "pd_primary",
                "run_directory": str(root),
                "output": str(output),
                "command": ["never"],
            }
            lock = {
                "source_sha256": {
                    str(subject.EVALUATOR.relative_to(subject.REPO_ROOT)):
                    completion.sha256_file(subject.EVALUATOR)
                }
            }
            with (
                mock.patch.object(
                    completion,
                    "inspect_training_readiness",
                    return_value={"formal_matrix_complete": True},
                ),
                mock.patch.object(
                    completion,
                    "validate_acceptance_source_lock",
                    return_value=(lock, "0" * 64),
                ),
                mock.patch.object(subject, "sweep_jobs", return_value=[job]),
                mock.patch.object(
                    completion,
                    "validate_existing_sweep",
                    return_value={},
                ) as validate,
                mock.patch.object(subject.subprocess, "run") as process,
            ):
                result = subject.run_sweeps("cpu")
            process.assert_not_called()
            validate.assert_called_once()
            self.assertEqual(result["created"], [])
            self.assertEqual(result["validated_existing"], [output])
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))[
                    subject.EXECUTION_PROVENANCE_KEY
                ],
                subject._execution_provenance("cpu", None),
            )

    def test_invalid_existing_sweep_stops_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "pd_fa_sweep_best.pth.json"
            output.write_bytes(b"invalid")
            job = {
                "variant": completion.PRIMARY_VARIANT,
                "seed": 42,
                "role": "pd_primary",
                "run_directory": str(root),
                "output": str(output),
                "command": ["never"],
            }
            lock = {
                "source_sha256": {
                    str(subject.EVALUATOR.relative_to(subject.REPO_ROOT)):
                    completion.sha256_file(subject.EVALUATOR)
                }
            }
            with (
                mock.patch.object(
                    completion,
                    "inspect_training_readiness",
                    return_value={"formal_matrix_complete": True},
                ),
                mock.patch.object(
                    completion,
                    "validate_acceptance_source_lock",
                    return_value=(lock, "0" * 64),
                ),
                mock.patch.object(subject, "sweep_jobs", return_value=[job]),
                mock.patch.object(
                    completion,
                    "validate_existing_sweep",
                    side_effect=completion.IncompleteArtifact("invalid"),
                ),
                mock.patch.object(subject.subprocess, "run") as process,
            ):
                with self.assertRaises(completion.IncompleteArtifact):
                    subject.run_sweeps("cpu")
            process.assert_not_called()
            self.assertEqual(output.read_bytes(), b"invalid")

    def test_new_sweep_gets_strict_execution_provenance_before_validation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "pd_fa_sweep_best.pth.json"
            job = {
                "variant": completion.PRIMARY_VARIANT,
                "seed": 42,
                "role": "pd_primary",
                "run_directory": str(root),
                "output": str(output),
                "command": ["evaluator"],
            }
            lock = {
                "source_sha256": {
                    str(subject.EVALUATOR.relative_to(subject.REPO_ROOT)):
                    completion.sha256_file(subject.EVALUATOR)
                }
            }

            def create_output(*args, **kwargs):
                output.write_text(json.dumps({"audit": {}}), encoding="utf-8")
                return mock.Mock(returncode=0)

            with (
                mock.patch.object(
                    completion,
                    "inspect_training_readiness",
                    return_value={"formal_matrix_complete": True},
                ),
                mock.patch.object(
                    completion,
                    "validate_acceptance_source_lock",
                    return_value=(lock, "0" * 64),
                ),
                mock.patch.object(subject, "sweep_jobs", return_value=[job]),
                mock.patch.object(
                    completion,
                    "validate_existing_sweep",
                    return_value={},
                ) as validate,
                mock.patch.object(
                    subject.subprocess,
                    "run",
                    side_effect=create_output,
                ),
            ):
                result = subject.run_sweeps("cpu")
            validate.assert_called_once()
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                payload[subject.EXECUTION_PROVENANCE_KEY],
                subject._execution_provenance("cpu", None),
            )
            self.assertEqual(result["created"], [output])

    def test_cuda_environment_passes_exact_cublas_and_uuid(self) -> None:
        gpu_uuid = subject.POSTPROCESS_GPUS["2"]
        query = mock.Mock(
            stdout=f"2, NVIDIA GeForce RTX 5090, {gpu_uuid}\n"
        )
        with mock.patch.object(
            subject.subprocess, "run", return_value=query
        ):
            environment = subject._gpu_environment("cuda:0", "2")
        self.assertEqual(
            environment["CUBLAS_WORKSPACE_CONFIG"],
            subject.CUBLAS_WORKSPACE_CONFIG,
        )
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], gpu_uuid)
        provenance = subject._execution_provenance("cuda:0", "2")
        self.assertEqual(provenance["physical_gpu_index"], 2)
        self.assertEqual(provenance["physical_gpu_uuid"], gpu_uuid)
        self.assertEqual(
            provenance["determinism"], subject.DETERMINISM_SETTINGS
        )

    def test_cuda_jobs_replay_each_run_training_gpu(self) -> None:
        jobs = subject.sweep_jobs("cuda:0")
        for job in jobs:
            expected_index, expected_uuid = completion.GPU_ASSIGNMENTS[
                (job["variant"], job["seed"])
            ]
            self.assertEqual(job["physical_gpu"], str(expected_index))
            self.assertEqual(job["physical_gpu_uuid"], expected_uuid)
        per_run = {
            (job["variant"], job["seed"]): job["physical_gpu"]
            for job in jobs
        }
        self.assertEqual(
            per_run[(completion.PRIMARY_VARIANT, 42)], "2"
        )
        self.assertEqual(
            per_run[(completion.CONTROL_VARIANT, 3407)], "2"
        )
        self.assertEqual(
            per_run[(completion.CONTROL_VARIANT, 42)], "3"
        )
        self.assertEqual(
            per_run[(completion.PRIMARY_VARIANT, 3407)], "3"
        )

    def test_cuda_run_uses_each_jobs_mapped_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output2 = root / "gpu2.json"
            output3 = root / "gpu3.json"
            jobs = [
                {
                    "variant": completion.PRIMARY_VARIANT,
                    "seed": 42,
                    "role": "pd_primary",
                    "run_directory": str(root),
                    "output": str(output2),
                    "physical_gpu": "2",
                    "physical_gpu_uuid": subject.POSTPROCESS_GPUS["2"],
                    "command": ["gpu2-evaluator"],
                },
                {
                    "variant": completion.CONTROL_VARIANT,
                    "seed": 42,
                    "role": "pd_primary",
                    "run_directory": str(root),
                    "output": str(output3),
                    "physical_gpu": "3",
                    "physical_gpu_uuid": subject.POSTPROCESS_GPUS["3"],
                    "command": ["gpu3-evaluator"],
                },
            ]
            lock = {
                "source_sha256": {
                    str(subject.EVALUATOR.relative_to(subject.REPO_ROOT)):
                    completion.sha256_file(subject.EVALUATOR)
                }
            }
            environments = {
                "2": {
                    "CUDA_VISIBLE_DEVICES": subject.POSTPROCESS_GPUS["2"],
                    "CUBLAS_WORKSPACE_CONFIG":
                    subject.CUBLAS_WORKSPACE_CONFIG,
                },
                "3": {
                    "CUDA_VISIBLE_DEVICES": subject.POSTPROCESS_GPUS["3"],
                    "CUBLAS_WORKSPACE_CONFIG":
                    subject.CUBLAS_WORKSPACE_CONFIG,
                },
            }

            def create_output(command, **kwargs):
                selected = "2" if command[0] == "gpu2-evaluator" else "3"
                self.assertEqual(kwargs["env"], environments[selected])
                path = output2 if selected == "2" else output3
                path.write_text(
                    json.dumps(
                        {
                            "audit": {
                                "cuda_visible_devices":
                                subject.POSTPROCESS_GPUS[selected]
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                return mock.Mock(returncode=0)

            with (
                mock.patch.object(
                    completion,
                    "inspect_training_readiness",
                    return_value={"formal_matrix_complete": True},
                ),
                mock.patch.object(
                    completion,
                    "validate_acceptance_source_lock",
                    return_value=(lock, "0" * 64),
                ),
                mock.patch.object(subject, "sweep_jobs", return_value=jobs),
                mock.patch.object(
                    subject,
                    "_gpu_environment",
                    side_effect=lambda device, gpu: environments[str(gpu)],
                ) as gpu_environment,
                mock.patch.object(
                    completion,
                    "validate_existing_sweep",
                    return_value={},
                ) as validate,
                mock.patch.object(
                    subject.subprocess,
                    "run",
                    side_effect=create_output,
                ),
            ):
                result = subject.run_sweeps("cuda:0")
            self.assertEqual(
                gpu_environment.call_args_list,
                [mock.call("cuda:0", "2"), mock.call("cuda:0", "3")],
            )
            self.assertEqual(validate.call_count, 2)
            self.assertEqual(result["created"], [output2, output3])
            for selected, path in (("2", output2), ("3", output3)):
                provenance = json.loads(
                    path.read_text(encoding="utf-8")
                )[subject.EXECUTION_PROVENANCE_KEY]
                self.assertEqual(
                    provenance["physical_gpu_index"], int(selected)
                )
                self.assertEqual(
                    provenance["physical_gpu_uuid"],
                    subject.POSTPROCESS_GPUS[selected],
                )

    def test_cuda_run_requires_physical_gpu2_or3(self) -> None:
        with self.assertRaisesRegex(ValueError, "physical-gpu"):
            subject._gpu_environment("cuda:0", None)
        with self.assertRaisesRegex(ValueError, "physical-gpu"):
            subject._gpu_environment("cuda:0", "0")
        self.assertEqual(subject.parse_args(["--run"]).device, "cpu")
        self.assertEqual(
            subject.parse_args(["--run", "--device", "cuda:0"]).device,
            "cuda:0",
        )
        with self.assertRaises(SystemExit):
            subject.parse_args(
                [
                    "--run",
                    "--device",
                    "cuda:0",
                    "--physical-gpu",
                    "2",
                ]
            )


if __name__ == "__main__":
    unittest.main()
