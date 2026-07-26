from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments import smoke_tpd_ner_v5 as smoke
from experiments.smoke_tpd_ner_v5 import (
    PRODUCTION_PARAMETER_CONTRACT,
    SCHEMA,
    run_smoke,
)
from experiments.train_tpd_ner_v5 import SUPPORTED_TPD_NER_V5_VARIANTS
from experiments.train_tpd_ner_v5 import build_tpd_ner_v5_model


torch.set_num_threads(1)


class SmokeTPDNERV5Tests(unittest.TestCase):
    def test_light_cpu_all_variants_two_step_contract(self) -> None:
        report = run_smoke(
            variant="all",
            device_text="cpu",
            batch_size=2,
            patch_size=32,
            steps=2,
            seed=42,
        )
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["requested_variant"], "all")
        self.assertEqual(report["device"], "cpu")
        self.assertTrue(report["lightweight_cpu"])
        self.assertEqual(report["dimension_profile"], "cpu_light_base4_relay2")
        self.assertTrue(report["off_on_step_zero_exact"])
        self.assertFalse(report["formal_training_started"])
        self.assertEqual(
            report["production_parameter_contract"],
            PRODUCTION_PARAMETER_CONTRACT,
        )
        self.assertFalse(
            report["production_parameter_contract_verified_in_this_run"]
        )

        self.assertEqual(len(report["pair_checks"]), 2)
        for pair in report["pair_checks"]:
            with self.subTest(pair=pair["pair"]):
                self.assertTrue(pair["common_state_exact"])
                self.assertTrue(pair["step_zero_output_exact"])
                self.assertEqual(pair["step_zero_max_abs_difference"], 0.0)
                self.assertEqual(pair["extra_state_prefix"], "tpd_ner.")
                self.assertEqual(pair["extra_state_key_count"], 19)
                self.assertEqual(pair["shallow_embedding_parameters"], 1_104)
                self.assertEqual(pair["relay_parameter_delta"], 545)

        self.assertEqual(
            tuple(item["variant"] for item in report["variants"]),
            SUPPORTED_TPD_NER_V5_VARIANTS,
        )
        for item in report["variants"]:
            with self.subTest(variant=item["variant"]):
                self.assertEqual(item["status"], "complete")
                self.assertEqual(item["output_count"], 6)
                self.assertEqual(len(item["losses"]), 2)
                self.assertTrue(all(value > 0.0 for value in item["losses"]))
                self.assertEqual(len(item["control_gradient_l1"]), 7)
                self.assertEqual(len(item["control_update_l1"]), 7)
                self.assertTrue(
                    all(value > 0.0 for value in item["control_gradient_l1"].values())
                )
                self.assertTrue(
                    all(value > 0.0 for value in item["control_update_l1"].values())
                )
                self.assertTrue(item["strict_rebuild_load"])
                self.assertEqual(item["strict_reload_max_abs_difference"], 0.0)
                self.assertFalse(
                    item["metadata_contract"]["fourth_parallel_branch_added"]
                )
                self.assertTrue(
                    item["metadata_contract"][
                        "shallow_parameter_match_verified"
                    ]
                )
                self.assertFalse(
                    item["metadata_contract"][
                        "production_parameter_contract_verified"
                    ]
                )
                counts = item["parameter_counts"]
                self.assertEqual(counts["total"], counts["trainable"])
                self.assertEqual(counts["shallow_embedding"], 1_104)
                if item["relay_enabled"]:
                    self.assertEqual(counts["total"], 100_804)
                    self.assertEqual(counts["common"], 100_259)
                    self.assertEqual(counts["relay"], 545)
                    self.assertEqual(counts["relay_gate"], 9)
                    self.assertEqual(len(item["gate_gradient_l1"]), 6)
                    self.assertEqual(len(item["gate_update_l1"]), 6)
                    self.assertEqual(len(item["fusion_gradient_l1"]), 13)
                    self.assertEqual(len(item["fusion_update_l1"]), 13)
                    self.assertTrue(
                        all(
                            value > 0.0
                            for value in item["gate_gradient_l1"].values()
                        )
                    )
                    self.assertTrue(
                        all(
                            value > 0.0
                            for value in item["gate_update_l1"].values()
                        )
                    )
                    self.assertTrue(
                        all(
                            value == 0.0
                            for value in item[
                                "fusion_step1_zero_gradient"
                            ].values()
                        )
                    )
                    self.assertTrue(
                        all(
                            value == 0.0
                            for value in item[
                                "fusion_step1_zero_update"
                            ].values()
                        )
                    )
                    self.assertTrue(
                        all(
                            value > 0.0
                            for value in item["fusion_gradient_l1"].values()
                        )
                    )
                    self.assertTrue(
                        all(
                            value > 0.0
                            for value in item["fusion_update_l1"].values()
                        )
                    )
                else:
                    self.assertEqual(counts["total"], 100_259)
                    self.assertEqual(counts["relay"], 0)
                    self.assertEqual(counts["relay_gate"], 0)
                    for key in (
                        "gate_gradient_l1",
                        "gate_update_l1",
                        "fusion_step1_zero_gradient",
                        "fusion_step1_zero_update",
                        "fusion_gradient_l1",
                        "fusion_update_l1",
                    ):
                        self.assertEqual(item[key], {})

    def test_production_builds_match_every_frozen_parameter_count(self) -> None:
        for variant in SUPPORTED_TPD_NER_V5_VARIANTS:
            with self.subTest(variant=variant):
                model, metadata = build_tpd_ner_v5_model(variant, seed=42)
                smoke._validate_production_parameter_contract(metadata)
                self.assertEqual(
                    sum(parameter.numel() for parameter in model.parameters()),
                    metadata["total_parameters"],
                )
                del model

    def test_control_kind_follows_the_tokenizer(self) -> None:
        report = run_smoke(
            variant="progressive_relay_off",
            device_text="cpu",
            batch_size=2,
            patch_size=32,
            steps=2,
            seed=17,
        )
        self.assertEqual(len(report["variants"]), 1)
        self.assertEqual(
            report["variants"][0]["control_kind"],
            "progressive_channel_gain",
        )
        self.assertEqual(len(report["pair_checks"]), 1)
        self.assertEqual(report["pair_checks"][0]["pair"], "progressive")

    def test_absolute_cli_stdout_ends_with_json(self) -> None:
        script = Path(smoke.__file__).resolve()
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--variant",
                    "tpd_clean_v5_full_relay_on",
                    "--device",
                    "cpu",
                    "--batch-size",
                    "2",
                    "--patch-size",
                    "32",
                    "--steps",
                    "2",
                    "--seed",
                    "42",
                ],
                cwd=temporary,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        stdout_lines = [
            line for line in completed.stdout.splitlines() if line.strip()
        ]
        self.assertEqual(len(stdout_lines), 1)
        report = json.loads(stdout_lines[-1])
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            report["variants"][0]["variant"],
            "tpd_clean_v5_full_relay_on",
        )
        self.assertTrue(report["off_on_step_zero_exact"])

    def test_cuda_resolution_requires_one_visible_device(self) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=2),
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly one visible"):
                smoke._resolve_device("cuda:0", None)
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device_count", return_value=1),
            mock.patch.object(
                torch.cuda,
                "get_device_name",
                return_value="NVIDIA GeForce RTX 5090",
            ),
        ):
            device, name = smoke._resolve_device(
                "cuda",
                "NVIDIA GeForce RTX 5090",
            )
        self.assertEqual(device, torch.device("cuda:0"))
        self.assertEqual(name, "NVIDIA GeForce RTX 5090")


if __name__ == "__main__":
    unittest.main()
