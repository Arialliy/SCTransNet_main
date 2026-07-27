from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from experiments import smoke_tpd_clean_v6 as v6_smoke
from experiments.smoke_tpd_clean_v6 import (
    EXPECTED_DEVICE_NAME,
    EXPECTED_KEEP_PARAMETER_NAMES,
    EXPECTED_SCALE_PARAMETER_NAMES,
    PHYSICAL_GPU_UUIDS,
    _external_cuda_mapping,
    normalized_gpu_uuid,
    _paired_initialization_fields,
    _resolve_device_contract,
    run_smoke,
    six_output_bce_loss,
)
from model.tpd_clean_v6 import SUPPORTED_CLEAN_V6_VARIANTS


class SmokeTPDCleanV6ContractTests(unittest.TestCase):
    def test_six_output_loss_is_the_explicit_baseline_sum(self) -> None:
        target = torch.zeros(2, 1, 4, 4)
        outputs = tuple(
            torch.full_like(target, 0.2 + 0.05 * index, requires_grad=True)
            for index in range(6)
        )
        criterion = nn.BCELoss(reduction="mean")
        actual, heads = six_output_bce_loss(outputs, target, criterion)
        expected = v6_smoke.v3_smoke.deep_supervision_loss(
            outputs,
            target,
            criterion,
        )
        self.assertEqual(len(heads), 6)
        torch.testing.assert_close(actual, sum(heads), rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        with self.assertRaisesRegex(RuntimeError, "exactly six"):
            six_output_bce_loss(outputs[:5], target, criterion)

    def test_frozen_scale_and_keep_parameter_name_sets_are_exact(self) -> None:
        self.assertEqual(len(EXPECTED_SCALE_PARAMETER_NAMES), 7)
        self.assertEqual(len(EXPECTED_KEEP_PARAMETER_NAMES), 14)
        self.assertEqual(
            EXPECTED_SCALE_PARAMETER_NAMES,
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
            EXPECTED_KEEP_PARAMETER_NAMES,
            frozenset(
                {
                    *{
                        "embeddings_1.blocks."
                        f"{index}.phase_compress.{parameter}"
                        for index in range(4)
                        for parameter in ("weight", "bias")
                    },
                    *{
                        "embeddings_2.blocks."
                        f"{index}.phase_compress.{parameter}"
                        for index in range(3)
                        for parameter in ("weight", "bias")
                    },
                }
            ),
        )

    def test_cpu_contract_never_claims_a_validated_cuda_mapping(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "2"},
            clear=True,
        ):
            self.assertIsNone(_external_cuda_mapping("cpu", None))
            device, name, contract = _resolve_device_contract(
                "cpu",
                None,
                None,
            )
        self.assertEqual(device, torch.device("cpu"))
        self.assertEqual(name, "cpu")
        self.assertFalse(contract["applicable"])
        self.assertFalse(contract["validated"])
        self.assertIsNone(contract["declared_physical_index"])
        self.assertIsNone(contract["logical_device"])
        self.assertIsNone(contract["device_uuid"])

    def test_cuda_contract_binds_mask_logical_device_model_and_uuid(
        self,
    ) -> None:
        properties = SimpleNamespace(
            uuid=PHYSICAL_GPU_UUIDS["2"].removeprefix("GPU-")
        )
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
                return_value=EXPECTED_DEVICE_NAME,
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=properties,
            ),
        ):
            device, name, contract = _resolve_device_contract(
                "cuda:0",
                None,
                "2",
            )
        self.assertEqual(device, torch.device("cuda:0"))
        self.assertEqual(name, EXPECTED_DEVICE_NAME)
        self.assertTrue(contract["applicable"])
        self.assertTrue(contract["validated"])
        self.assertEqual(contract["declared_physical_index"], "2")
        self.assertEqual(contract["visible_device_count"], 1)
        self.assertEqual(contract["device_uuid"], PHYSICAL_GPU_UUIDS["2"])
        self.assertEqual(
            normalized_gpu_uuid(PHYSICAL_GPU_UUIDS["2"]),
            PHYSICAL_GPU_UUIDS["2"],
        )

    def test_cuda_contract_rejects_unbound_or_wrong_physical_selection(
        self,
    ) -> None:
        with mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "2"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "required for CUDA"):
                _external_cuda_mapping("cuda:0", None)
            with self.assertRaisesRegex(ValueError, "exactly 2 or 3"):
                _external_cuda_mapping("cuda:0", "0")
            with self.assertRaisesRegex(ValueError, "logical device cuda:0"):
                _external_cuda_mapping("cuda", "2")
            with self.assertRaisesRegex(
                RuntimeError,
                "unexpected CUDA_VISIBLE_DEVICES",
            ):
                _external_cuda_mapping("cuda:0", "3")

    def test_cuda_contract_rejects_wrong_uuid_and_wrong_model(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "3"},
                clear=True,
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=1),
            mock.patch.object(
                torch.cuda,
                "get_device_name",
                return_value=EXPECTED_DEVICE_NAME,
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(uuid=PHYSICAL_GPU_UUIDS["2"]),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "UUID differs"):
                _resolve_device_contract("cuda:0", None, "3")

        with (
            mock.patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "3"},
                clear=True,
            ),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=1),
            mock.patch.object(
                torch.cuda,
                "get_device_name",
                return_value="NVIDIA GeForce RTX 4090",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected device"):
                _resolve_device_contract("cuda:0", None, "3")

    def test_pairing_is_verified_only_for_the_complete_pair(self) -> None:
        checksum = "a" * 64
        paired = [
            {
                "variant": variant,
                "initial_model_checksum": checksum,
            }
            for variant in SUPPORTED_CLEAN_V6_VARIANTS
        ]
        self.assertEqual(
            _paired_initialization_fields(
                paired,
                SUPPORTED_CLEAN_V6_VARIANTS,
            ),
            (True, "verified", checksum),
        )
        self.assertEqual(
            _paired_initialization_fields(
                paired[:1],
                SUPPORTED_CLEAN_V6_VARIANTS[:1],
            ),
            (None, "not_checked_single_variant", None),
        )
        mismatched = [dict(item) for item in paired]
        mismatched[1]["initial_model_checksum"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "not exactly paired"):
            _paired_initialization_fields(
                mismatched,
                SUPPORTED_CLEAN_V6_VARIANTS,
            )

    def test_validation_requires_exactly_two_steps_before_model_build(
        self,
    ) -> None:
        for steps in (1, 3):
            with self.subTest(steps=steps):
                with self.assertRaisesRegex(ValueError, "must equal 2"):
                    run_smoke(
                        variant="tpd_clean_v6_full",
                        device_text="cpu",
                        batch_size=2,
                        patch_size=32,
                        steps=steps,
                        seed=42,
                    )
        with self.assertRaisesRegex(ValueError, "unsupported variant"):
            run_smoke(
                variant="tpd_clean_v5_full",
                device_text="cpu",
                batch_size=2,
                patch_size=32,
                steps=2,
                seed=42,
            )

    def test_absolute_cli_help_without_pythonpath(self) -> None:
        script = Path(v6_smoke.__file__).resolve()
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
        self.assertIn("--expected-cuda-visible-devices", completed.stdout)
        self.assertIn("tpd_clean_v6_full", completed.stdout)
        self.assertIn("tpd_clean_v6_phase_capacity", completed.stdout)

    @unittest.skipUnless(
        os.environ.get("RUN_V6_FULL_SMOKE_TEST") == "1",
        "manual full-model CPU smoke",
    )
    def test_manual_full_model_cpu_two_step_smoke(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            report = run_smoke(
                variant="all",
                device_text="cpu",
                batch_size=2,
                patch_size=32,
                steps=2,
                seed=42,
            )
        self.assertTrue(report["paired_initialization"])
        self.assertEqual(report["paired_initialization_status"], "verified")
        self.assertIsNone(report["cuda_visible_devices"])
        self.assertFalse(report["cuda_device_contract"]["validated"])
        for entry in report["variants"]:
            self.assertEqual(entry["loss_head_count"], 6)
            self.assertEqual(len(entry["per_head_losses"]), 2)
            self.assertEqual(set(entry["scale_gradient_l1"]), set(
                EXPECTED_SCALE_PARAMETER_NAMES
            ))
            self.assertEqual(set(entry["phase_gradient_l1"]), set(
                EXPECTED_KEEP_PARAMETER_NAMES
            ))


if __name__ == "__main__":
    unittest.main()
