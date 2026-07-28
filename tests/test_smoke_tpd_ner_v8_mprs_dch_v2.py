from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments import smoke_tpd_ner_v8_mprs_dch_v2 as smoke
from model.tpd_ner_v8_mprs_dch_v2 import (
    PRODUCTION_PARENT_PARAMETERS,
    PRODUCTION_V2_RELAY_ON_PARAMETERS,
    PRODUCTION_V2_RELAY_PARAMETERS,
)


class SmokeTPDNERV8MPRSDCHV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_cpu_two_steps_reload_identity_and_v2_numerics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v2-smoke.pth.tar"
            report = smoke.run_smoke(
                device_text="cpu",
                batch_size=2,
                patch_size=32,
                checkpoint_output=checkpoint,
            )
            payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )
        self.assertEqual(report["schema"], smoke.SCHEMA)
        self.assertEqual(
            report["variant"],
            smoke.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
        )
        self.assertEqual(
            report["control_variant"],
            smoke.V1_RELAY_OFF_REFERENCE,
        )
        self.assertFalse(report["relay_off_retrained"])
        self.assertEqual(report["training_seed"], 42)
        self.assertEqual(report["split_seed"], 20260722)
        self.assertEqual(report["relay_width"], 8)
        self.assertEqual(report["output_count"], 6)
        self.assertTrue(report["step_zero_output_exact"])
        self.assertEqual(report["step_zero_max_abs_differences"], [0.0] * 6)
        self.assertTrue(report["parent_relay_off_output_exact"])
        self.assertTrue(report["common_state_exact"])
        self.assertEqual(report["extra_state_key_count"], 16)
        self.assertEqual(
            report["relay_off_parameters"],
            PRODUCTION_PARENT_PARAMETERS,
        )
        self.assertEqual(
            report["relay_parameters"],
            PRODUCTION_V2_RELAY_PARAMETERS,
        )
        self.assertEqual(
            report["relay_on_parameters"],
            PRODUCTION_V2_RELAY_ON_PARAMETERS,
        )
        self.assertEqual(len(report["losses"]), 2)
        self.assertGreater(report["gate_gradient_l1"][0], 0.0)
        self.assertEqual(report["fusion_gradient_l1"][0], 0.0)
        self.assertGreater(report["fusion_gradient_l1"][1], 0.0)
        self.assertEqual(set(report["batch_norm_update_counts"]), {2})
        self.assertTrue(report["strict_model_reload"])
        self.assertTrue(report["optimizer_state_exact"])
        self.assertTrue(report["paired_resume_step_exact"])
        self.assertTrue(report["paired_resume_model_state_exact"])
        self.assertTrue(report["paired_resume_optimizer_state_exact"])
        self.assertTrue(report["relay_values_finite"])
        self.assertTrue(report["relay_rms_finite"])
        self.assertTrue(report["relay_masks_finite"])
        self.assertTrue(report["relay_masks_within_open_bounds"])
        self.assertFalse(report["deterministic_runtime"]["applicable"])
        self.assertFalse(report["deterministic_runtime"]["enabled"])
        self.assertGreater(report["relay_value_observation_count"], 0)
        self.assertGreaterEqual(report["relay_rms_min"], 0.0)
        self.assertLess(report["relay_mask_abs_max"], 0.5)
        self.assertFalse(report["formal_training_started"])

        self.assertEqual(payload["schema"], smoke.SCHEMA)
        self.assertEqual(
            payload["variant"],
            smoke.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
        )
        self.assertTrue(payload["relay_enabled"])
        self.assertEqual(payload["training_seed"], 42)

    def test_cuda_deterministic_runtime_is_scoped_and_explicit(self) -> None:
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
        previous_float32_precision = (
            torch.get_float32_matmul_precision()
        )
        with (
            mock.patch.object(
                torch.cuda,
                "is_initialized",
                return_value=False,
            ),
            mock.patch.dict(
                os.environ,
                {"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
                clear=False,
            ),
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
                    evidence["float32_matmul_precision"],
                    "highest",
                )
                self.assertEqual(
                    evidence["cublas_workspace_config"],
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
        self.assertEqual(
            torch.get_float32_matmul_precision(),
            previous_float32_precision,
        )

    def test_cuda_initialized_before_workspace_contract_is_rejected(
        self,
    ) -> None:
        with (
            mock.patch.object(
                torch.cuda,
                "is_initialized",
                return_value=True,
            ),
            mock.patch.dict(
                os.environ,
                {"CUBLAS_WORKSPACE_CONFIG": ":16:8"},
                clear=False,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "CUDA was initialized before",
            ),
        ):
            with smoke._deterministic_smoke_execution("cuda:0"):
                pass

    def test_device_resolution_covers_optional_cuda_identity(self) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=1),
            mock.patch.object(
                torch.cuda,
                "get_device_name",
                return_value="NVIDIA GeForce RTX 5090",
            ),
        ):
            device, identity = smoke._resolve_device(
                "cuda:0",
                "NVIDIA GeForce RTX 5090",
            )
            self.assertEqual(device, torch.device("cuda:0"))
            self.assertEqual(identity["type"], "cuda")
            self.assertEqual(identity["name"], "NVIDIA GeForce RTX 5090")
            with self.assertRaisesRegex(RuntimeError, "unexpected CUDA"):
                smoke._resolve_device("cuda:0", "other")
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                smoke._resolve_device("cuda:0", None)

    def test_cli_exposes_no_seed_width_or_relay_off_override(self) -> None:
        args = smoke.parse_args(["--device", "cpu", "--patch-size", "64"])
        self.assertEqual(
            args.variant,
            smoke.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
        )
        self.assertFalse(hasattr(args, "seed"))
        self.assertFalse(hasattr(args, "split_seed"))
        self.assertFalse(hasattr(args, "relay_width"))
        for arguments in (
            ["--seed", "7"],
            ["--variant", smoke.V1_RELAY_OFF_REFERENCE],
            ["--variant", "tpd_ner_v8_mprs_dch_v2_full_relay_off"],
        ):
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        smoke.parse_args(arguments)

    def test_main_emits_one_json_record(self) -> None:
        fixture = {
            "schema": smoke.SCHEMA,
            "status": "complete",
            "variant": smoke.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
        }
        stdout = io.StringIO()
        with (
            mock.patch.object(smoke, "run_smoke", return_value=fixture),
            contextlib.redirect_stdout(stdout),
        ):
            smoke.main(["--device", "cpu"])
        lines = [line for line in stdout.getvalue().splitlines() if line]
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), fixture)

    def test_cli_main_forwards_explicit_cuda_two_step_smoke_path(self) -> None:
        fixture = {
            "schema": smoke.SCHEMA,
            "status": "complete",
            "variant": smoke.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
            "device": {"type": "cuda", "index": 0},
        }
        with (
            mock.patch.object(smoke, "run_smoke", return_value=fixture) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            smoke.main(
                [
                    "--device",
                    "cuda:0",
                    "--expected-device-name",
                    "NVIDIA GeForce RTX 5090",
                    "--batch-size",
                    "2",
                    "--patch-size",
                    "32",
                ]
            )
        run.assert_called_once_with(
            variant=smoke.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,
            device_text="cuda:0",
            expected_device_name="NVIDIA GeForce RTX 5090",
            batch_size=2,
            patch_size=32,
            learning_rate=1e-3,
            checkpoint_output=None,
        )


if __name__ == "__main__":
    unittest.main()
