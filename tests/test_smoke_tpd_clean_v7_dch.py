from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments import smoke_tpd_clean_v7_dch as dch_smoke
from model.tpd_clean_v7_dch import SUPPORTED_CLEAN_V7_DCH_VARIANTS


class SmokeTPDCleanV7DCHContractTests(unittest.TestCase):
    def test_contract_has_seven_scales_and_fourteen_keep_parameters(
        self,
    ) -> None:
        self.assertEqual(len(dch_smoke.EXPECTED_SCALE_PARAMETER_NAMES), 7)
        self.assertEqual(len(dch_smoke.EXPECTED_KEEP_PARAMETER_NAMES), 14)
        self.assertEqual(
            dch_smoke.EXPECTED_SCALE_PARAMETER_NAMES,
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

    def test_pair_evidence_requires_exact_initial_and_first_adam_states(
        self,
    ) -> None:
        optimizer_state = {
            "parameter_state_count": 10,
            "step_values": [1.0],
            "exp_avg_l1": 2.0,
            "exp_avg_sq_l1": 3.0,
        }
        reports = [
            {
                "variant": variant,
                "initial_model_checksum": "a" * 64,
                "first_step_model_checksum": "b" * 64,
                "first_step_optimizer_checksum": "c" * 64,
                "first_step_optimizer_state": optimizer_state,
            }
            for variant in SUPPORTED_CLEAN_V7_DCH_VARIANTS
        ]
        self.assertEqual(
            dch_smoke._pair_evidence(
                reports,
                SUPPORTED_CLEAN_V7_DCH_VARIANTS,
            ),
            (True, "verified", "a" * 64, True),
        )
        self.assertEqual(
            dch_smoke._pair_evidence(
                reports[:1],
                SUPPORTED_CLEAN_V7_DCH_VARIANTS[:1],
            ),
            (None, "not_checked_single_variant", None, None),
        )
        mismatched = [dict(item) for item in reports]
        mismatched[1]["first_step_optimizer_checksum"] = "d" * 64
        with self.assertRaisesRegex(RuntimeError, "first Adam step"):
            dch_smoke._pair_evidence(
                mismatched,
                SUPPORTED_CLEAN_V7_DCH_VARIANTS,
            )

    def test_nested_state_hash_is_exact_and_order_stable(self) -> None:
        left = {
            "state": {
                1: {"step": torch.tensor(1.0), "avg": torch.tensor([2.0])}
            },
            "groups": [{"lr": 1e-3, "params": [1]}],
        }
        right = {
            "groups": [{"params": [1], "lr": 1e-3}],
            "state": {
                1: {"avg": torch.tensor([2.0]), "step": torch.tensor(1.0)}
            },
        }
        self.assertEqual(
            dch_smoke._state_sha256(left),
            dch_smoke._state_sha256(right),
        )
        right["state"][1]["avg"][0] = 3.0
        self.assertNotEqual(
            dch_smoke._state_sha256(left),
            dch_smoke._state_sha256(right),
        )

    def test_validation_rejects_invalid_axes_before_build(self) -> None:
        with self.assertRaisesRegex(ValueError, "must equal 2"):
            dch_smoke.run_smoke(
                variant="tpd_clean_v7_dch_full",
                device_text="cpu",
                batch_size=2,
                patch_size=32,
                steps=1,
                seed=42,
            )
        with self.assertRaisesRegex(ValueError, "unsupported variant"):
            dch_smoke.run_smoke(
                variant="tpd_clean_v6_full",
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
                dch_smoke.v6_smoke._resolve_device_contract(
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

    def test_cli_help_works_outside_repository(self) -> None:
        script = Path(dch_smoke.__file__).resolve()
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
        self.assertIn("tpd_clean_v7_dch_full", completed.stdout)
        self.assertIn("tpd_clean_v7_dch_capacity", completed.stdout)
        self.assertIn("--expected-cuda-visible-devices", completed.stdout)

    @unittest.skipUnless(
        os.environ.get("RUN_V7_DCH_FULL_SMOKE_TEST") == "1",
        "manual full-model CPU smoke",
    )
    def test_manual_cpu_pair_two_step_smoke(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            report = dch_smoke.run_smoke(
                variant="all",
                device_text="cpu",
                batch_size=2,
                patch_size=32,
                steps=2,
                seed=42,
            )
        self.assertTrue(report["paired_initialization"])
        self.assertTrue(report["paired_first_adam_step_exact"])
        self.assertEqual(report["headroom_bound"], [0.75, 1.25])


if __name__ == "__main__":
    unittest.main()
