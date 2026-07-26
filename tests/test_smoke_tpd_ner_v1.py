from __future__ import annotations

import unittest

from experiments.smoke_tpd_ner_v1 import (
    SCHEMA,
    UINT32_MAX,
    _next_rebuild_seed,
    run_smoke,
)


class TPDNERSmokeTests(unittest.TestCase):
    def test_rebuild_seed_wraps_inside_uint32_without_changing_seed42(self) -> None:
        self.assertEqual(_next_rebuild_seed(42), 43)
        self.assertEqual(_next_rebuild_seed(UINT32_MAX), 0)
        for invalid_seed in (-1, UINT32_MAX + 1):
            with self.subTest(invalid_seed=invalid_seed):
                with self.assertRaisesRegex(ValueError, "seed must lie"):
                    _next_rebuild_seed(invalid_seed)

    def test_full_builder_two_step_cpu_smoke(self) -> None:
        report = run_smoke(
            device_text="cpu",
            batch_size=2,
            patch_size=32,
            steps=2,
            seed=113,
        )

        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["variant"], "tpd_clean_full_ner")
        self.assertEqual(report["device"], "cpu")
        self.assertEqual(report["output_count"], 6)
        self.assertEqual(len(report["losses"]), 2)
        self.assertTrue(all(value > 0.0 for value in report["losses"]))
        self.assertEqual(set(report["gate_gradient_l1"]), {"2", "3", "4"})
        self.assertEqual(set(report["gate_update_l1"]), {"2", "3", "4"})
        self.assertEqual(set(report["fusion_gradient_l1"]), {"2", "3", "4"})
        self.assertEqual(set(report["fusion_update_l1"]), {"2", "3", "4"})
        self.assertTrue(
            all(value > 0.0 for value in report["gate_gradient_l1"].values())
        )
        self.assertTrue(
            all(value > 0.0 for value in report["gate_update_l1"].values())
        )
        self.assertTrue(
            all(value > 0.0 for value in report["fusion_gradient_l1"].values())
        )
        self.assertTrue(
            all(value > 0.0 for value in report["fusion_update_l1"].values())
        )
        self.assertGreater(report["tpd_scale_update_l1"], 0.0)
        self.assertTrue(report["strict_rebuild_load"])
        self.assertEqual(report["relay_parameters"], 11_291)
        self.assertEqual(report["total_parameters"], 10_854_766)
        self.assertIsNone(report["cuda_memory"])


if __name__ == "__main__":
    unittest.main()
