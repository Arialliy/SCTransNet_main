from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import freeze_tpd_clean_v6_postprocess_source_lock as subject
from experiments import summarize_tpd_clean_v6_formal800 as summary


class FreezeV6PostprocessSourceLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = subject.build_source_lock_payload()

    def test_payload_binds_training_lock_data_sources_and_references(self) -> None:
        payload = self.payload
        self.assertEqual(payload["schema"], subject.SCHEMA)
        self.assertEqual(
            payload["training_source_lock_sha256"],
            summary.EXPECTED_TRAINING_LOCK_SHA256,
        )
        self.assertEqual(
            payload["training_data_sha256"],
            summary.EXPECTED_TRAINING_DATA_SHA256,
        )
        expected_sources = {
            str(path.resolve().relative_to(subject.REPO_ROOT))
            for path in subject.POSTPROCESS_SOURCE_PATHS
        }
        expected_references = {
            str(path.resolve().relative_to(subject.REPO_ROOT))
            for path in subject.FROZEN_REFERENCE_PATHS
        }
        self.assertEqual(set(payload["source_sha256"]), expected_sources)
        self.assertEqual(
            expected_sources,
            set(summary.POSTPROCESS_SOURCE_RELATIVES),
        )
        self.assertEqual(
            set(payload["frozen_reference_sha256"]),
            expected_references,
        )
        self.assertEqual(
            expected_references,
            set(summary.FROZEN_REFERENCE_RELATIVES),
        )
        self.assertNotIn(
            "experiments/train_tpd_clean_v6_exact.py",
            payload["source_sha256"],
        )
        self.assertTrue(
            payload["policy"]["formal_report_overwrite_forbidden"]
        )
        self.assertTrue(
            payload["policy"][
                "gate_evaluation_before_four_complete_runs_forbidden"
            ]
        )

    def test_temporary_lock_is_accepted_by_the_summarizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "postprocess-lock.json"
            subject.write_new_json(path, self.payload)
            loaded, digest = summary.validate_postprocess_source_lock(path)
            self.assertEqual(loaded, self.payload)
            self.assertEqual(digest, summary.sha256_file(path))

    def test_summarizer_rejects_a_lock_that_omits_one_source_or_reference(
        self,
    ) -> None:
        for field in ("source_sha256", "frozen_reference_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                payload = json.loads(json.dumps(self.payload))
                payload[field].pop(next(iter(payload[field])))
                path = Path(directory) / "incomplete-lock.json"
                subject.write_new_json(path, payload)
                with self.assertRaises(summary.IncompleteArtifact):
                    summary.validate_postprocess_source_lock(path)

    def test_freeze_refuses_to_replace_an_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lock.json"
            output.write_bytes(b"unchanged")
            with (
                mock.patch.object(
                    subject,
                    "build_source_lock_payload",
                    return_value=self.payload,
                ),
                self.assertRaises(FileExistsError),
            ):
                subject.freeze_source_lock(output)
            self.assertEqual(output.read_bytes(), b"unchanged")

    def test_write_new_json_is_canonical_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lock.json"
            subject.write_new_json(output, self.payload)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                self.payload,
            )
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                subject.write_new_json(output, self.payload)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
