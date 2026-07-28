from __future__ import annotations

import contextlib
import io
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import numpy as np

from experiments import smoke_tpd_ner_v8_mprs_dch as smoke
from model.tpd_ner_v8_mprs_dch import (
    PRODUCTION_PARENT_PARAMETERS,
    PRODUCTION_RELAY_ON_PARAMETERS,
    PRODUCTION_RELAY_PARAMETERS,
)


class SmokeTPDNERV8MPRSDCHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._previous_thread_count = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        torch.set_num_threads(cls._previous_thread_count)

    def test_cpu_two_step_checkpoint_and_parent_immutability(self) -> None:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        hash_seed = os.environ.get("PYTHONHASHSEED")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "smoke.pth.tar"
            report = smoke.run_smoke(
                variant="tpd_clean_v8_mprs_dch_full",
                device_text="cpu",
                batch_size=2,
                patch_size=32,
                checkpoint_output=checkpoint,
            )
            self.assertTrue(checkpoint.is_file())
            payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )

        self.assertEqual(random.getstate(), python_state)
        observed_numpy_state = np.random.get_state()
        self.assertEqual(observed_numpy_state[0], numpy_state[0])
        self.assertTrue(
            np.array_equal(observed_numpy_state[1], numpy_state[1])
        )
        self.assertEqual(observed_numpy_state[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), hash_seed)
        self.assertEqual(report["schema"], smoke.SCHEMA)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["training_seed"], 42)
        self.assertEqual(report["relay_initialization_seed"], 42)
        self.assertEqual(report["split_seed"], 20260722)
        self.assertEqual(report["relay_width"], 8)
        self.assertEqual(report["device"]["type"], "cpu")
        self.assertEqual(report["output_count"], 6)
        self.assertTrue(report["outputs_finite"])
        self.assertTrue(report["common_state_exact"])
        self.assertTrue(report["parent_relay_off_output_exact"])
        self.assertEqual(
            report["parent_relay_off_max_abs_differences"],
            [0.0] * 6,
        )
        self.assertTrue(report["step_zero_output_exact"])
        self.assertTrue(report["parent_relay_off_output_exact"])
        self.assertEqual(report["step_zero_max_abs_differences"], [0.0] * 6)
        self.assertEqual(report["extra_state_key_count"], 19)
        self.assertEqual(
            report["relay_off_parameters"],
            PRODUCTION_PARENT_PARAMETERS,
        )
        self.assertEqual(
            report["relay_on_parameters"],
            PRODUCTION_RELAY_ON_PARAMETERS,
        )
        self.assertEqual(
            report["relay_parameters"],
            PRODUCTION_RELAY_PARAMETERS,
        )
        self.assertEqual(len(report["losses"]), 2)
        self.assertTrue(all(value > 0.0 for value in report["losses"]))
        self.assertGreater(report["gate_gradient_l1"][0], 0.0)
        self.assertEqual(report["fusion_gradient_l1"][0], 0.0)
        self.assertGreater(report["fusion_gradient_l1"][1], 0.0)
        self.assertTrue(report["first_step_gate_active"])
        self.assertTrue(report["first_step_fusion_blocked"])
        self.assertTrue(report["second_step_fusion_active"])
        self.assertTrue(report["parameters_finite"])
        self.assertTrue(report["strict_model_reload"])
        self.assertTrue(report["optimizer_reload"])
        self.assertTrue(report["optimizer_state_exact"])
        self.assertEqual(report["reload_output_max_abs_difference"], 0.0)
        self.assertTrue(report["paired_resume_step_exact"])
        self.assertEqual(report["paired_resume_loss_difference"], 0.0)
        self.assertTrue(report["paired_resume_model_state_exact"])
        self.assertTrue(report["paired_resume_optimizer_state_exact"])
        self.assertTrue(report["batch_norm_update_counts"])
        self.assertEqual(set(report["batch_norm_update_counts"]), {2})
        self.assertEqual(report["train_forward_count"], 2)
        self.assertTrue(report["checkpoint_preserved"])
        self.assertTrue(report["parent_state_unchanged_after_adaptation"])
        self.assertTrue(report["parent_state_unchanged_after_smoke"])
        self.assertFalse(report["formal_training_started"])

        self.assertEqual(payload["schema"], smoke.SCHEMA)
        self.assertEqual(payload["training_seed"], 42)
        self.assertEqual(payload["split_seed"], 20260722)
        self.assertEqual(payload["relay_width"], 8)
        self.assertTrue(payload["relay_enabled"])
        checkpoint_bn_counts = [
            int(value)
            for name, value in payload["state_dict"].items()
            if name.endswith("num_batches_tracked")
        ]
        self.assertTrue(checkpoint_bn_counts)
        self.assertEqual(set(checkpoint_bn_counts), {2})
        checkpoint_optimizer_steps = [
            int(state["step"])
            for state in payload["optimizer"]["state"].values()
        ]
        self.assertTrue(checkpoint_optimizer_steps)
        self.assertEqual(set(checkpoint_optimizer_steps), {2})

    def test_capacity_cpu_two_step_and_paired_resume(self) -> None:
        report = smoke.run_smoke(
            variant="tpd_clean_v8_mprs_dch_capacity",
            device_text="cpu",
            batch_size=2,
            patch_size=32,
        )
        self.assertEqual(
            report["variant"],
            "tpd_clean_v8_mprs_dch_capacity",
        )
        self.assertTrue(report["step_zero_output_exact"])
        self.assertEqual(set(report["batch_norm_update_counts"]), {2})
        self.assertTrue(report["optimizer_state_exact"])
        self.assertTrue(report["paired_resume_step_exact"])
        self.assertTrue(report["paired_resume_model_state_exact"])
        self.assertTrue(report["paired_resume_optimizer_state_exact"])

    def test_cuda_device_resolution_is_optional_and_identity_checked(self) -> None:
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
            self.assertEqual(identity["index"], 0)
            self.assertEqual(identity["name"], "NVIDIA GeForce RTX 5090")
            with self.assertRaisesRegex(RuntimeError, "unexpected CUDA"):
                smoke._resolve_device("cuda:0", "other")

        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                smoke._resolve_device("cuda:0", None)
        with self.assertRaisesRegex(ValueError, "only for CUDA"):
            smoke._resolve_device("cpu", "NVIDIA GeForce RTX 5090")
        with self.assertRaisesRegex(ValueError, "only CPU or CUDA"):
            smoke._resolve_device("mps", None)

    def test_cli_has_no_experiment_seed_override(self) -> None:
        args = smoke.parse_args(
            [
                "--variant",
                "tpd_clean_v8_mprs_dch_capacity",
                "--device",
                "cpu",
                "--patch-size",
                "64",
            ]
        )
        self.assertEqual(args.patch_size, 64)
        self.assertFalse(hasattr(args, "seed"))
        self.assertFalse(hasattr(args, "split_seed"))
        self.assertFalse(hasattr(args, "relay_width"))

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                smoke.parse_args(["--seed", "3407"])

    def test_main_emits_one_json_record(self) -> None:
        fixture = {
            "schema": smoke.SCHEMA,
            "status": "complete",
            "training_seed": 42,
            "split_seed": 20260722,
            "relay_width": 8,
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


if __name__ == "__main__":
    unittest.main()
