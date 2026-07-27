from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments import (
    capture_tpd_clean_v7_dch_smoke_report as capture,
)
from experiments import (
    verify_tpd_clean_v7_dch_smoke_reports as verifier,
)


class VerifyTPDCleanV7DCHSmokeReportsTests(unittest.TestCase):
    def test_source_manifest_covers_owned_and_shared_smoke_sources(
        self,
    ) -> None:
        manifest = capture.source_manifest()
        self.assertIn("model/tpd_clean_v7_dch.py", manifest)
        self.assertIn("experiments/train_tpd_clean_v7_dch.py", manifest)
        self.assertIn("experiments/smoke_tpd_clean_v7_dch.py", manifest)
        self.assertIn("experiments/smoke_tpd_clean_v6.py", manifest)
        self.assertIn("experiments/smoke_tpd_clean_v3.py", manifest)
        self.assertIn(
            "experiments/verify_tpd_clean_v7_dch_smoke_reports.py",
            manifest,
        )
        self.assertTrue(all(len(value) == 64 for value in manifest.values()))

    def test_envelope_binds_device_contract_and_sources(self) -> None:
        report = {
            "environment_cuda_visible_devices": "2",
            "cuda_visible_devices": "2",
            "cuda_device_contract": {"device_uuid": "GPU-example"},
        }
        envelope = capture.build_envelope(
            report,
            source_sha256={"example.py": "a" * 64},
            created_at_utc="2026-07-27T00:00:00+00:00",
        )
        self.assertEqual(envelope["schema"], capture.SCHEMA)
        self.assertEqual(envelope["source_sha256"], {"example.py": "a" * 64})
        self.assertEqual(
            envelope["cuda_device_contract"],
            report["cuda_device_contract"],
        )

    def test_exclusive_write_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            capture.exclusive_write_json(path, {"status": "complete"})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "complete"},
            )
            with self.assertRaises(FileExistsError):
                capture.exclusive_write_json(path, {"status": "changed"})

    def test_verifier_requires_the_exact_three_report_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                verifier.SmokeReportError,
                "exactly cpu_all.json",
            ):
                verifier.validate_smoke_reports(root)

    def test_frozen_report_matrix_binds_gpu2_and_gpu3(self) -> None:
        self.assertEqual(
            set(verifier.EXPECTED_REPORTS),
            {"cpu_all.json", "gpu2_full.json", "gpu3_capacity.json"},
        )
        self.assertEqual(
            verifier.EXPECTED_REPORTS["gpu2_full.json"][
                "physical_index"
            ],
            "2",
        )
        self.assertEqual(
            verifier.EXPECTED_REPORTS["gpu3_capacity.json"][
                "physical_index"
            ],
            "3",
        )


if __name__ == "__main__":
    unittest.main()
