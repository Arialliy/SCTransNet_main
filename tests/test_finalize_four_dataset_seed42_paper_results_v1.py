from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments import finalize_four_dataset_seed42_paper_results_v1 as finalizer


class ResultTemplateTests(unittest.TestCase):
    def test_initializer_contains_only_explicit_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = finalizer.initialize_templates(root, overwrite=False)
            self.assertTrue(payload["no_experimental_result_generated"])
            self.assertTrue(payload["no_fabricated_results"])
            self.assertFalse(payload["stability_claim_supported"])
            self.assertIsNone(
                payload["fixed_seed42_four_dataset_performance_supported"]
            )
            table = (root / "tables" / "table2_best_miou.md").read_text(
                encoding="utf-8"
            )
            self.assertEqual(table.count("| SIRST3 | SIRST3 |"), 2)
            self.assertIn("TBD", table)
            self.assertNotIn("nan", table.lower())

    def test_initializer_refuses_implicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            finalizer.initialize_templates(root, overwrite=False)
            with self.assertRaises(FileExistsError):
                finalizer.initialize_templates(root, overwrite=False)

    def test_both_saved_sirst3_roles_have_six_placeholder_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            finalizer.initialize_templates(root, overwrite=False)
            for suffix in (
                "table3a_best_miou_sirst3_three_sources.md",
                "table3b_best_pd_sirst3_three_sources.md",
            ):
                text = (root / "tables" / suffix).read_text(encoding="utf-8")
                data_rows = [
                    line
                    for line in text.splitlines()
                    if line.startswith("| SIRST3 |")
                ]
                self.assertEqual(len(data_rows), 6)


if __name__ == "__main__":
    unittest.main()
