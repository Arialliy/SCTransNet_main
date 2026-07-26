from __future__ import annotations

import unittest

from experiments.smoke_tpd_clean_v4 import SCHEMA, run_smoke
from model.tpd_clean_v4 import SUPPORTED_CLEAN_V4_VARIANTS


class SmokeTPDCleanV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_smoke(
            variant="all",
            device_text="cpu",
            batch_size=2,
            patch_size=32,
            steps=2,
            seed=42,
        )

    def test_cpu_smoke_completes_for_both_variants(self) -> None:
        self.assertEqual(self.report["schema"], SCHEMA)
        self.assertEqual(self.report["status"], "complete")
        self.assertEqual(
            tuple(item["variant"] for item in self.report["variants"]),
            SUPPORTED_CLEAN_V4_VARIANTS,
        )
        self.assertTrue(self.report["paired_initialization"])
        self.assertEqual(self.report["device"], "cpu")

    def test_step_zero_gradients_updates_and_reload_are_verified(self) -> None:
        for item in self.report["variants"]:
            with self.subTest(variant=item["variant"]):
                self.assertTrue(item["step_zero_exact_spd"])
                self.assertTrue(item["strict_rebuild_load"])
                self.assertEqual(item["strict_reload_max_abs_difference"], 0.0)
                self.assertEqual(len(item["scale_gradient_l1"]), 14)
                self.assertEqual(len(item["phase_gradient_l1"]), 14)
                self.assertTrue(
                    all(value > 0 for value in item["scale_update_l1"].values())
                )
                self.assertTrue(
                    all(value > 0 for value in item["phase_update_l1"].values())
                )

    def test_v4_fusion_contract_is_recorded(self) -> None:
        self.assertEqual(
            self.report["residual_bound"],
            "absolute_residual_at_most_absolute_saliency",
        )
        self.assertIn("0.5*tanh(context_scale)", self.report["fusion_formula"])


if __name__ == "__main__":
    unittest.main()
