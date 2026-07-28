from __future__ import annotations

import contextlib
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments import smoke_tpd_ner_v8_mprs_dch_v3 as smoke
from model.tpd_ner_v8_mprs_dch_v3 import (
    PRODUCTION_PARENT_PARAMETERS,
    PRODUCTION_V3_RELAY_ON_PARAMETERS,
    PRODUCTION_V3_RELAY_PARAMETERS,
    V3_RELAY_VERSION,
)


class SmokeTPDNERV8MPRSDCHV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls.previous_threads)

    def test_cpu_two_steps_reload_offsets_rng_and_v3_numerics(
        self,
    ) -> None:
        rng_before = smoke._capture_global_rng_state()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "v3-smoke.pth.tar"
            with (
                mock.patch.object(
                    torch.cuda,
                    "is_available",
                    return_value=False,
                ),
                mock.patch.object(
                    torch.cuda,
                    "device_count",
                    return_value=0,
                ),
            ):
                report = smoke.run_smoke(
                    device_text="cpu",
                    batch_size=2,
                    patch_size=32,
                    checkpoint_output=checkpoint,
                )
            rng_after = smoke._capture_global_rng_state()
            payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )

        self.assertTrue(
            smoke._global_rng_states_equal(rng_before, rng_after)
        )
        self.assertTrue(report["global_rng_preserved"])
        self.assertEqual(report["schema"], smoke.SCHEMA)
        self.assertEqual(
            report["variant"],
            smoke.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
        )
        self.assertEqual(
            report["control_variant"],
            smoke.V1_RELAY_OFF_REFERENCE,
        )
        self.assertEqual(
            report["structural_predecessor"],
            smoke.V2_RELAY_ON_REFERENCE,
        )
        self.assertFalse(report["relay_off_retrained"])
        self.assertEqual(report["training_seed"], 42)
        self.assertEqual(report["split_seed"], 20260722)
        self.assertEqual(report["relay_width"], 8)
        self.assertEqual(report["output_count"], 6)
        self.assertTrue(report["outputs_finite"])
        self.assertTrue(report["step_zero_output_exact"])
        self.assertEqual(
            report["step_zero_max_abs_differences"],
            [0.0] * 6,
        )
        self.assertTrue(report["parent_relay_off_output_exact"])
        self.assertTrue(report["common_state_exact"])
        self.assertEqual(
            report["extra_state_key_count"],
            smoke.EXPECTED_RELAY_STATE_KEY_COUNT,
        )
        self.assertEqual(report["relay_state_key_count"], 19)
        self.assertEqual(
            report["relay_off_parameters"],
            PRODUCTION_PARENT_PARAMETERS,
        )
        self.assertEqual(
            report["relay_parameters"],
            PRODUCTION_V3_RELAY_PARAMETERS,
        )
        self.assertEqual(
            report["relay_on_parameters"],
            PRODUCTION_V3_RELAY_ON_PARAMETERS,
        )
        self.assertEqual(
            (
                report["relay_state_key_count"],
                report["relay_parameters"],
                report["relay_on_parameters"],
            ),
            (19, 11_291, 10_854_446),
        )
        self.assertEqual(report["relay_version"], V3_RELAY_VERSION)
        self.assertEqual(report["dc_offset_count"], 3)
        self.assertEqual(
            set(report["dc_offset_stages"]),
            {"4", "3", "2"},
        )
        self.assertEqual(
            set(report["dc_offset_state_keys"]),
            set(smoke.OFFSET_STATE_KEYS),
        )
        self.assertTrue(
            all(
                value == 0.0
                for value in report["dc_offset_initial_values"].values()
            )
        )

        self.assertEqual(len(report["losses"]), 2)
        self.assertTrue(
            all(math.isfinite(value) for value in report["losses"])
        )
        self.assertGreater(report["gate_gradient_l1"][0], 0.0)
        self.assertEqual(report["fusion_gradient_l1"][0], 0.0)
        self.assertGreater(report["fusion_gradient_l1"][1], 0.0)
        self.assertEqual(len(report["offset_gradient_l1"]), 2)
        self.assertTrue(
            all(value > 0.0 for value in report["offset_gradient_l1"])
        )
        self.assertEqual(
            set(report["dc_offset_gradient_l1"]),
            {"4", "3", "2"},
        )
        for stage, gradients in report["dc_offset_gradient_l1"].items():
            with self.subTest(stage=stage):
                self.assertEqual(len(gradients), 2)
                self.assertTrue(
                    all(
                        math.isfinite(value) and value > 0.0
                        for value in gradients
                    )
                )
        self.assertTrue(report["dc_offset_gradients_finite"])
        self.assertTrue(report["dc_offset_gradients_nonzero"])
        self.assertEqual(
            set(report["dc_offset_update_l1"]),
            {"4", "3", "2"},
        )
        self.assertTrue(
            all(
                math.isfinite(value) and value > 0.0
                for value in report["dc_offset_update_l1"].values()
            )
        )
        self.assertTrue(report["dc_offsets_updated"])
        self.assertTrue(report["dc_offsets_nonzero_after_two_steps"])

        self.assertEqual(set(report["batch_norm_update_counts"]), {2})
        self.assertEqual(report["train_forward_count"], 2)
        self.assertTrue(report["parameters_finite"])
        self.assertTrue(report["strict_model_reload"])
        self.assertTrue(report["optimizer_state_exact"])
        self.assertTrue(report["paired_resume_step_exact"])
        self.assertTrue(report["paired_resume_model_state_exact"])
        self.assertTrue(report["paired_resume_optimizer_state_exact"])
        self.assertTrue(report["checkpoint_preserved"])
        self.assertTrue(report["checkpoint_dc_offsets_nonzero"])
        self.assertTrue(report["checkpoint_dc_offsets_exact"])
        self.assertTrue(report["strict_reload_dc_offsets_exact"])
        self.assertEqual(
            set(report["checkpoint_dc_offset_keys"]),
            set(smoke.OFFSET_STATE_KEYS),
        )
        self.assertTrue(
            all(
                value != 0.0
                for value in report[
                    "checkpoint_dc_offset_values"
                ].values()
            )
        )

        self.assertTrue(report["relay_values_finite"])
        self.assertTrue(report["relay_rms_finite"])
        self.assertTrue(report["relay_masks_finite"])
        self.assertTrue(report["relay_masks_within_open_bounds"])
        self.assertGreater(report["relay_value_observation_count"], 0)
        self.assertGreaterEqual(report["relay_rms_min"], 0.0)
        self.assertLess(report["relay_mask_abs_max"], 0.5)
        self.assertFalse(report["deterministic_runtime"]["applicable"])
        self.assertFalse(report["deterministic_runtime"]["enabled"])
        self.assertFalse(report["formal_training_started"])

        self.assertEqual(payload["schema"], smoke.SCHEMA)
        self.assertEqual(
            payload["variant"],
            smoke.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
        )
        self.assertTrue(payload["relay_enabled"])
        self.assertEqual(payload["training_seed"], 42)
        relay_state_keys = {
            name
            for name in payload["state_dict"]
            if name.startswith("tpd_ner.")
        }
        offset_state_keys = {
            name
            for name in relay_state_keys
            if name.startswith("tpd_ner.dc_offsets.")
        }
        self.assertEqual(len(relay_state_keys), 19)
        self.assertEqual(offset_state_keys, set(smoke.OFFSET_STATE_KEYS))
        for stage in smoke.OFFSET_STAGES:
            value = payload["state_dict"][
                f"tpd_ner.dc_offsets.{stage}"
            ]
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(int(torch.count_nonzero(value)), 1)
            self.assertEqual(
                float(value),
                report["checkpoint_dc_offset_values"][stage],
            )

    def test_cpu_deterministic_runtime_contract_is_non_mutating(
        self,
    ) -> None:
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
        previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        with smoke._deterministic_smoke_execution("cpu") as evidence:
            self.assertFalse(evidence["applicable"])
            self.assertFalse(evidence["enabled"])
            self.assertIsNone(evidence["cublas_workspace_config"])
            self.assertIsNone(
                evidence["cuda_initialized_before_contract"]
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
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            previous_workspace,
        )

    def test_cuda_deterministic_runtime_is_mocked_scoped_and_explicit(
        self,
    ) -> None:
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

    def test_device_resolution_covers_mocked_optional_cuda_identity(
        self,
    ) -> None:
        with (
            mock.patch.object(
                torch.cuda,
                "is_available",
                return_value=True,
            ),
            mock.patch.object(
                torch.cuda,
                "device_count",
                return_value=1,
            ),
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
            self.assertEqual(
                identity["name"],
                "NVIDIA GeForce RTX 5090",
            )
            with self.assertRaisesRegex(RuntimeError, "unexpected CUDA"):
                smoke._resolve_device("cuda:0", "other")
        with mock.patch.object(
            torch.cuda,
            "is_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                smoke._resolve_device("cuda:0", None)

    def test_only_v3_variant_is_accepted(self) -> None:
        args = smoke.parse_args(
            ["--device", "cpu", "--patch-size", "64"]
        )
        self.assertEqual(
            args.variant,
            smoke.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
        )
        self.assertFalse(hasattr(args, "seed"))
        self.assertFalse(hasattr(args, "split_seed"))
        self.assertFalse(hasattr(args, "relay_width"))
        for arguments in (
            ["--seed", "7"],
            ["--variant", smoke.V1_RELAY_OFF_REFERENCE],
            ["--variant", smoke.V2_RELAY_ON_REFERENCE],
            [
                "--variant",
                "tpd_ner_v8_mprs_dch_v3_full_relay_off",
            ],
        ):
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        smoke.parse_args(arguments)
        with self.assertRaisesRegex(ValueError, "only the formal V3"):
            smoke.run_smoke(variant=smoke.V2_RELAY_ON_REFERENCE)

    def test_main_emits_one_json_record(self) -> None:
        fixture = {
            "schema": smoke.SCHEMA,
            "status": "complete",
            "variant": smoke.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
        }
        stdout = io.StringIO()
        with (
            mock.patch.object(
                smoke,
                "run_smoke",
                return_value=fixture,
            ),
            contextlib.redirect_stdout(stdout),
        ):
            smoke.main(["--device", "cpu"])
        lines = [
            line for line in stdout.getvalue().splitlines() if line
        ]
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0]), fixture)

    def test_cli_main_forwards_mocked_cuda_two_step_smoke_path(
        self,
    ) -> None:
        fixture = {
            "schema": smoke.SCHEMA,
            "status": "complete",
            "variant": smoke.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
            "device": {"type": "cuda", "index": 0},
        }
        with (
            mock.patch.object(
                smoke,
                "run_smoke",
                return_value=fixture,
            ) as run,
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
            variant=smoke.TPD_NER_V8_MPRS_DCH_V3_FULL_RELAY_ON,
            device_text="cuda:0",
            expected_device_name="NVIDIA GeForce RTX 5090",
            batch_size=2,
            patch_size=32,
            learning_rate=1e-3,
            checkpoint_output=None,
        )


if __name__ == "__main__":
    unittest.main()
