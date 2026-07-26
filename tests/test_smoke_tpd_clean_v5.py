from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments import smoke_tpd_clean_v3 as v3_smoke
from experiments import smoke_tpd_clean_v5 as v5_smoke
from experiments.smoke_tpd_clean_v5 import run_smoke
from model.tpd_clean_v5 import SUPPORTED_CLEAN_V5_VARIANTS


class SmokeTPDCleanV5Tests(unittest.TestCase):
    def test_cpu_two_step_all_variants_and_scoped_bindings(self) -> None:
        original_builder = v3_smoke.build_clean_v3_model
        original_variants = v3_smoke.SUPPORTED_CLEAN_V3_VARIANTS
        original_schema = v3_smoke.SCHEMA
        original_scales = v3_smoke._kcs_parameters
        original_stdout = sys.stdout

        report = run_smoke(
            variant="all",
            device_text="cpu",
            batch_size=2,
            patch_size=32,
            steps=2,
            seed=42,
        )

        self.assertEqual(report["schema"], "sctransnet_tpd_clean_v5_smoke_v1")
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["paired_initialization"])
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["steps"], 2)
        self.assertEqual(report["context_selector_range"], [0.5, 1.5])
        self.assertEqual(report["learned_scales_per_block"], 1)
        self.assertEqual(
            report["fusion_formula"],
            "K+S*tanh(saliency_scale*(1+0.5*context_code))",
        )
        self.assertEqual(len(report["variants"]), 2)
        self.assertEqual(
            tuple(item["variant"] for item in report["variants"]),
            SUPPORTED_CLEAN_V5_VARIANTS,
        )
        for variant in report["variants"]:
            with self.subTest(variant=variant["variant"]):
                self.assertEqual(variant["output_count"], 6)
                self.assertEqual(len(variant["losses"]), 2)
                self.assertTrue(variant["step_zero_exact_spd"])
                self.assertTrue(variant["strict_rebuild_load"])
                self.assertEqual(variant["strict_reload_max_abs_difference"], 0.0)
                self.assertEqual(len(variant["scale_gradient_l1"]), 7)
                self.assertEqual(len(variant["scale_update_l1"]), 7)
                self.assertEqual(len(variant["phase_gradient_l1"]), 14)
                self.assertEqual(len(variant["phase_update_l1"]), 14)
                self.assertEqual(variant["total_parameters"], 10_843_155)
                self.assertEqual(
                    variant["shallow_embedding_parameters"],
                    66_176,
                )
                for key in (
                    "scale_gradient_l1",
                    "scale_update_l1",
                    "phase_gradient_l1",
                    "phase_update_l1",
                ):
                    self.assertTrue(
                        all(value > 0.0 for value in variant[key].values()),
                        msg=f"{variant['variant']}.{key}",
                    )

        self.assertIs(v3_smoke.build_clean_v3_model, original_builder)
        self.assertIs(v3_smoke.SUPPORTED_CLEAN_V3_VARIANTS, original_variants)
        self.assertIs(v3_smoke.SCHEMA, original_schema)
        self.assertIs(v3_smoke._kcs_parameters, original_scales)
        self.assertIs(sys.stdout, original_stdout)

    def test_absolute_cli_without_pythonpath(self) -> None:
        script = (
            Path(v5_smoke.__file__).resolve()
        )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--variant",
                    "tpd_clean_v5_full",
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
        report = json.loads(completed.stdout.splitlines()[-1])
        self.assertEqual(report["schema"], "sctransnet_tpd_clean_v5_smoke_v1")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(
            tuple(item["variant"] for item in report["variants"]),
            ("tpd_clean_v5_full",),
        )
        self.assertTrue(report["variants"][0]["step_zero_exact_spd"])
        self.assertTrue(report["variants"][0]["strict_rebuild_load"])

    def test_exception_never_rebinds_v3_harness_symbols(self) -> None:
        original_builder = v3_smoke.build_clean_v3_model
        original_variants = v3_smoke.SUPPORTED_CLEAN_V3_VARIANTS
        original_schema = v3_smoke.SCHEMA
        original_scales = v3_smoke._kcs_parameters
        original_stdout = sys.stdout

        class InjectedFailure(BaseException):
            pass

        inputs = torch.zeros(2, 1, 32, 32)
        targets = torch.zeros_like(inputs)
        with (
            mock.patch.object(
                v3_smoke,
                "_resolve_device",
                return_value=(torch.device("cpu"), "cpu"),
            ),
            mock.patch.object(
                v3_smoke,
                "_paired_inputs",
                return_value=(inputs, targets),
            ),
            mock.patch.object(v5_smoke, "_spd_reference_v5", return_value=()),
            mock.patch.object(
                v5_smoke,
                "_run_v5_variant",
                side_effect=InjectedFailure,
            ),
        ):
            with self.assertRaises(InjectedFailure):
                run_smoke(
                    variant="tpd_clean_v5_full",
                    device_text="cpu",
                    batch_size=2,
                    patch_size=32,
                    steps=2,
                    seed=42,
                )

        self.assertIs(v3_smoke.build_clean_v3_model, original_builder)
        self.assertIs(v3_smoke.SUPPORTED_CLEAN_V3_VARIANTS, original_variants)
        self.assertIs(v3_smoke.SCHEMA, original_schema)
        self.assertIs(v3_smoke._kcs_parameters, original_scales)
        self.assertIs(sys.stdout, original_stdout)

    def test_overlapping_calls_never_rebind_v3_harness_symbols(self) -> None:
        original_builder = v3_smoke.build_clean_v3_model
        original_variants = v3_smoke.SUPPORTED_CLEAN_V3_VARIANTS
        original_schema = v3_smoke.SCHEMA
        original_scales = v3_smoke._kcs_parameters
        original_stdout = sys.stdout
        barrier = threading.Barrier(2, timeout=5)
        errors: list[BaseException] = []
        reports: list[dict[str, object]] = []
        inputs = torch.zeros(2, 1, 32, 32)
        targets = torch.zeros_like(inputs)

        def fake_variant(
            variant: str,
            **_: object,
        ) -> dict[str, object]:
            barrier.wait()
            self.assertIs(v3_smoke.build_clean_v3_model, original_builder)
            self.assertIs(
                v3_smoke.SUPPORTED_CLEAN_V3_VARIANTS,
                original_variants,
            )
            self.assertIs(v3_smoke.SCHEMA, original_schema)
            self.assertIs(v3_smoke._kcs_parameters, original_scales)
            self.assertIs(sys.stdout, original_stdout)
            return {
                "variant": variant,
                "initial_model_checksum": "paired",
            }

        def worker(variant: str) -> None:
            try:
                reports.append(
                    run_smoke(
                        variant=variant,
                        device_text="cpu",
                        batch_size=2,
                        patch_size=32,
                        steps=2,
                        seed=42,
                    )
                )
            except BaseException as error:
                errors.append(error)

        with (
            mock.patch.object(
                v3_smoke,
                "_resolve_device",
                return_value=(torch.device("cpu"), "cpu"),
            ),
            mock.patch.object(
                v3_smoke,
                "_paired_inputs",
                return_value=(inputs, targets),
            ),
            mock.patch.object(v5_smoke, "_spd_reference_v5", return_value=()),
            mock.patch.object(
                v5_smoke,
                "_run_v5_variant",
                side_effect=fake_variant,
            ),
        ):
            threads = [
                threading.Thread(
                    target=worker,
                    args=(variant,),
                    daemon=True,
                )
                for variant in (
                    "tpd_clean_v5_full",
                    "tpd_clean_v5_sal_capacity",
                )
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self.assertTrue(all(not thread.is_alive() for thread in threads))

        self.assertEqual(errors, [])
        self.assertEqual(len(reports), 2)
        self.assertIs(v3_smoke.build_clean_v3_model, original_builder)
        self.assertIs(v3_smoke.SUPPORTED_CLEAN_V3_VARIANTS, original_variants)
        self.assertIs(v3_smoke.SCHEMA, original_schema)
        self.assertIs(v3_smoke._kcs_parameters, original_scales)
        self.assertIs(sys.stdout, original_stdout)


if __name__ == "__main__":
    unittest.main()
