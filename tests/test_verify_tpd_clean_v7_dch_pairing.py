from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments import verify_tpd_clean_v7_dch_pairing as pairing


class V7DCHPairingManifestTests(unittest.TestCase):
    def test_real_preflight_evidence_builds_complete_manifest(self) -> None:
        payload = pairing.build_pairing_manifest()
        self.assertEqual(payload["schema"], pairing.SCHEMA)
        self.assertEqual(payload["status"], "complete")
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["mainline_contract"], "Keep-Context-Saliency")
        self.assertEqual(payload["formal_runs"], 4)
        self.assertEqual(payload["seeds"], [42, 3407])
        self.assertEqual(len(payload["schedule"]), 4)
        self.assertEqual(len(payload["pair_evidence"]), 2)
        self.assertEqual(
            {
                (
                    item["physical_gpu_index"],
                    item["variant"],
                    item["seed"],
                )
                for item in payload["schedule"]
            },
            {
                (2, "tpd_clean_v7_dch_full", 42),
                (2, "tpd_clean_v7_dch_capacity", 3407),
                (3, "tpd_clean_v7_dch_capacity", 42),
                (3, "tpd_clean_v7_dch_full", 3407),
            },
        )
        self.assertFalse(payload["paper_core_established"])
        self.assertFalse(payload["stability_claim_supported"])

    def test_writer_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pairing.json"
            pairing.write_new_json(output, {"schema": "fixture"})
            self.assertTrue(output.is_file())
            with self.assertRaises(FileExistsError):
                pairing.write_new_json(output, {"schema": "replacement"})


if __name__ == "__main__":
    unittest.main()
