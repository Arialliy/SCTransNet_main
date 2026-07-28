from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from experiments import smoke_tpd_clean_v8_mprs_dch as smoke
from model.tpd_clean_v8_mprs_dch import (
    SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS,
    TPDCleanV8MPRSDCHBlock,
)


class SmokeTPDCleanV8MPRSDCHContractTests(unittest.TestCase):
    def test_contract_has_seven_scales_and_fourteen_keep_parameters(
        self,
    ) -> None:
        self.assertEqual(len(smoke.EXPECTED_SCALE_PARAMETER_NAMES), 7)
        self.assertEqual(len(smoke.EXPECTED_KEEP_PARAMETER_NAMES), 14)
        self.assertEqual(
            smoke.EXPECTED_SCALE_PARAMETER_NAMES,
            frozenset(
                {
                    *{
                        f"embeddings_1.blocks.{index}.saliency_scale"
                        for index in range(4)
                    },
                    *{
                        f"embeddings_2.blocks.{index}.saliency_scale"
                        for index in range(3)
                    },
                }
            ),
        )
        self.assertEqual(
            smoke.EXPECTED_MPRS_DIAGNOSTIC_KEYS,
            {
                "context_aligned",
                "saliency_v7",
                "phase_correction",
                "saliency_v8",
                "scale",
                "modulation",
                "headroom",
            },
        )

    def test_full_and_capacity_block_diagnostics_use_three_convolutions(
        self,
    ) -> None:
        for gate in (1.0, 0.0):
            with self.subTest(context_gate=gate):
                block = TPDCleanV8MPRSDCHBlock(
                    5,
                    activate=True,
                    context_gate=gate,
                )
                evidence = smoke._validate_mprs_block(
                    block,
                    f"block_gate_{gate}",
                    torch.device("cpu"),
                )
                self.assertEqual(
                    evidence["ordinary_forward_conv2d_calls"],
                    3,
                )
                self.assertEqual(
                    evidence["diagnostic_forward_conv2d_calls"],
                    3,
                )
                self.assertFalse(
                    evidence["explicit_phase_sources_in_production"]
                )
                self.assertTrue(evidence["diagnostic_output_exact"])
                self.assertEqual(
                    evidence["reuse_identity_max_abs_difference"],
                    0.0,
                )

    def test_pair_evidence_requires_exact_initial_and_first_adam_states(
        self,
    ) -> None:
        reports = [
            {
                "variant": variant,
                "initial_model_checksum": "a" * 64,
                "first_step_model_checksum": "b" * 64,
                "first_step_optimizer_checksum": "c" * 64,
            }
            for variant in SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS
        ]
        self.assertEqual(
            smoke._pair_evidence(
                reports,
                SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS,
            ),
            (True, "verified", "a" * 64, True),
        )
        self.assertEqual(
            smoke._pair_evidence(
                reports[:1],
                SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS[:1],
            ),
            (None, "not_checked_single_variant", None, None),
        )
        mismatched = [dict(item) for item in reports]
        mismatched[1]["first_step_optimizer_checksum"] = "d" * 64
        with self.assertRaisesRegex(RuntimeError, "first Adam step"):
            smoke._pair_evidence(
                mismatched,
                SUPPORTED_CLEAN_V8_MPRS_DCH_VARIANTS,
            )

    def test_nested_state_hash_reuses_order_stable_v7_contract(self) -> None:
        left = {
            "state": {
                1: {
                    "step": torch.tensor(1.0),
                    "avg": torch.tensor([2.0]),
                }
            },
            "groups": [{"lr": 1e-3, "params": [1]}],
        }
        right = {
            "groups": [{"params": [1], "lr": 1e-3}],
            "state": {
                1: {
                    "avg": torch.tensor([2.0]),
                    "step": torch.tensor(1.0),
                }
            },
        }
        self.assertIs(smoke._state_sha256, smoke.v7_smoke._state_sha256)
        self.assertEqual(
            smoke._state_sha256(left),
            smoke._state_sha256(right),
        )
        right["state"][1]["avg"][0] = 3.0
        self.assertNotEqual(
            smoke._state_sha256(left),
            smoke._state_sha256(right),
        )

    def test_validation_rejects_invalid_axes_before_build(self) -> None:
        with self.assertRaisesRegex(ValueError, "must equal 2"):
            smoke.run_smoke(
                variant="tpd_clean_v8_mprs_dch_full",
                device_text="cpu",
                batch_size=2,
                patch_size=32,
                steps=1,
                seed=42,
            )
        with self.assertRaisesRegex(ValueError, "unsupported variant"):
            smoke.run_smoke(
                variant="tpd_clean_v7_dch_full",
                device_text="cpu",
                batch_size=2,
                patch_size=32,
                steps=2,
                seed=42,
            )

    def test_cpu_device_contract_never_claims_physical_gpu(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "2"},
            clear=True,
        ):
            device, name, contract = (
                smoke.v6_smoke._resolve_device_contract(
                    "cpu",
                    None,
                    None,
                )
            )
        self.assertEqual(str(device), "cpu")
        self.assertEqual(name, "cpu")
        self.assertFalse(contract["applicable"])
        self.assertFalse(contract["validated"])
        self.assertIsNone(contract["device_uuid"])

    def test_cuda_contract_accepts_only_registered_gpu2_or_gpu3_uuid(
        self,
    ) -> None:
        expected_uuid = smoke.PHYSICAL_GPU_UUIDS["2"]
        properties = SimpleNamespace(uuid=expected_uuid)
        with (
            mock.patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "2"},
                clear=True,
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=1),
            mock.patch.object(
                torch.cuda,
                "get_device_name",
                return_value=smoke.EXPECTED_DEVICE_NAME,
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=properties,
            ),
        ):
            device, _, contract = smoke.v6_smoke._resolve_device_contract(
                "cuda:0",
                smoke.EXPECTED_DEVICE_NAME,
                "2",
            )
        self.assertEqual(str(device), "cuda:0")
        self.assertTrue(contract["validated"])
        self.assertEqual(contract["declared_physical_index"], "2")
        self.assertEqual(contract["device_uuid"], expected_uuid)

        with self.assertRaisesRegex(ValueError, "exactly 2 or 3"):
            smoke.v6_smoke._external_cuda_mapping("cuda:0", "0")

    def test_cli_help_works_outside_repository(self) -> None:
        script = Path(smoke.__file__).resolve()
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=temporary,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertIn("tpd_clean_v8_mprs_dch_full", completed.stdout)
        self.assertIn("tpd_clean_v8_mprs_dch_capacity", completed.stdout)
        self.assertIn("--expected-cuda-visible-devices", completed.stdout)
        self.assertIn("--output", completed.stdout)

    def test_deterministic_execution_is_scoped_and_explicit(self) -> None:
        previous_deterministic = (
            torch.are_deterministic_algorithms_enabled()
        )
        previous_warn_only = (
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
        previous_cudnn_deterministic = (
            torch.backends.cudnn.deterministic
        )
        previous_cudnn_benchmark = torch.backends.cudnn.benchmark
        previous_cuda_matmul_tf32 = (
            torch.backends.cuda.matmul.allow_tf32
        )
        previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        with mock.patch.dict(
            os.environ,
            {"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
            clear=False,
        ):
            with smoke._deterministic_smoke_execution(
                "cuda:0"
            ) as evidence:
                self.assertTrue(evidence["enabled"])
                self.assertTrue(evidence["deterministic_algorithms"])
                self.assertFalse(evidence["warn_only"])
                self.assertTrue(evidence["cudnn_deterministic"])
                self.assertFalse(evidence["cudnn_benchmark"])
                self.assertFalse(evidence["cuda_matmul_allow_tf32"])
                self.assertFalse(evidence["cudnn_allow_tf32"])
                self.assertEqual(
                    evidence["cublas_workspace_config"],
                    smoke.CUBLAS_WORKSPACE_CONFIG,
                )
                self.assertEqual(
                    os.environ["CUBLAS_WORKSPACE_CONFIG"],
                    smoke.CUBLAS_WORKSPACE_CONFIG,
                )
            self.assertEqual(
                os.environ["CUBLAS_WORKSPACE_CONFIG"],
                ":16:8",
            )
        self.assertEqual(
            torch.are_deterministic_algorithms_enabled(),
            previous_deterministic,
        )
        self.assertEqual(
            torch.is_deterministic_algorithms_warn_only_enabled(),
            previous_warn_only,
        )
        self.assertEqual(
            torch.backends.cudnn.deterministic,
            previous_cudnn_deterministic,
        )
        self.assertEqual(
            torch.backends.cudnn.benchmark,
            previous_cudnn_benchmark,
        )
        self.assertEqual(
            torch.backends.cuda.matmul.allow_tf32,
            previous_cuda_matmul_tf32,
        )
        self.assertEqual(
            torch.backends.cudnn.allow_tf32,
            previous_cudnn_tf32,
        )

    def test_cpu_run_records_non_applicable_cuda_contract(self) -> None:
        observed = {}
        previous = torch.are_deterministic_algorithms_enabled()

        def fake_impl(**_: object) -> dict[str, object]:
            observed["deterministic"] = (
                torch.are_deterministic_algorithms_enabled()
            )
            observed["warn_only"] = (
                torch.is_deterministic_algorithms_warn_only_enabled()
            )
            return {"schema": smoke.SCHEMA, "status": "complete"}

        with mock.patch.object(
            smoke,
            "_run_smoke_impl",
            side_effect=fake_impl,
        ):
            report = smoke.run_smoke(
                variant="all",
                device_text="cpu",
                batch_size=2,
                patch_size=32,
                steps=2,
                seed=42,
            )
        self.assertEqual(observed["deterministic"], previous)
        self.assertEqual(
            observed["warn_only"],
            torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        self.assertFalse(report["deterministic_execution"]["applicable"])
        self.assertFalse(report["deterministic_execution"]["enabled"])
        self.assertEqual(
            report["deterministic_execution"]["purpose"],
            "byte_exact_sequential_first_adam_pair",
        )

    def test_output_argument_atomically_writes_pure_json(self) -> None:
        expected = {
            "schema": smoke.SCHEMA,
            "status": "complete",
            "value": [1, 2, 3],
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "report.json"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                mock.patch.object(
                    smoke,
                    "run_smoke",
                    return_value=expected,
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                smoke.main(["--output", str(output)])
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                expected,
            )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                smoke._json_payload(expected) + "\n",
            )
            self.assertIn(str(output.resolve()), stderr.getvalue())
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.*.tmp")),
                [],
            )

    @unittest.skipUnless(
        os.environ.get("RUN_V8_MPRS_DCH_FULL_SMOKE_TEST") == "1",
        "manual full-model CPU smoke",
    )
    def test_manual_cpu_pair_two_step_smoke(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            report = smoke.run_smoke(
                variant="all",
                device_text="cpu",
                batch_size=2,
                patch_size=32,
                steps=2,
                seed=42,
            )
        self.assertTrue(report["paired_initialization"])
        self.assertTrue(report["paired_first_adam_step_exact"])
        self.assertEqual(
            set(report["source_sha256"]),
            set(smoke.SOURCE_RELATIVES),
        )
        self.assertEqual(
            report["source_sha256"][
                "experiments/smoke_tpd_clean_v8_mprs_dch.py"
            ],
            smoke.file_sha256(smoke.REPO_ROOT / smoke.SOURCE_RELATIVES[0]),
        )
        self.assertFalse(
            report["deterministic_execution"]["applicable"]
        )
        self.assertEqual(report["headroom_bound"], [0.75, 1.25])
        self.assertEqual(
            report["standard_forward_conv2d_calls_per_block"],
            3,
        )
        self.assertIsNone(report["physical_gpu_index"])
        self.assertIsNone(report["physical_gpu_uuid"])
        for variant_report in report["variants"]:
            self.assertTrue(variant_report["step_zero_exact_spd"])
            self.assertTrue(variant_report["mprs_diagnostics_verified"])
            self.assertEqual(variant_report["mprs_block_count"], 7)
            self.assertTrue(
                variant_report["all_observed_gradients_finite"]
            )
            self.assertTrue(
                variant_report["all_updated_parameters_finite"]
            )


if __name__ == "__main__":
    unittest.main()
