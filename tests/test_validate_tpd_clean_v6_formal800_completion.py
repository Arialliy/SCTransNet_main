from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import summarize_tpd_clean_v6_formal800 as summary
from experiments import validate_tpd_clean_v6_formal800_completion as subject


class V6Formal800CompletionTests(unittest.TestCase):
    def test_manifest_binds_each_exact_input_by_digest_and_size(self) -> None:
        with tempfile.TemporaryDirectory(dir=summary.REPO_ROOT) as directory:
            root = Path(directory)
            first = root / "first.bin"
            second = root / "second.bin"
            lock = root / "lock.json"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            lock.write_bytes(b"lock")
            records = [
                ("first", "candidate_training", first),
                ("second", "candidate_sweep", second),
            ]
            with mock.patch.object(subject, "_input_paths", return_value=records):
                manifest = subject.build_manifest(
                    {"candidate_runs": {}},
                    postprocess_source_lock=lock,
                )
            self.assertEqual(manifest["schema"], subject.MANIFEST_SCHEMA)
            self.assertEqual(manifest["input_count"], 2)
            self.assertEqual(
                manifest["category_counts"],
                {"candidate_training": 1, "candidate_sweep": 1},
            )
            self.assertEqual(
                manifest["inputs"][0]["sha256"],
                summary.sha256_file(first),
            )
            self.assertEqual(manifest["inputs"][0]["size_bytes"], 5)

    def test_publish_is_exclusive_and_marker_binds_three_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / summary.JSON_OUTPUT_NAME
            markdown_path = root / summary.MARKDOWN_OUTPUT_NAME
            manifest_path = root / subject.MANIFEST_NAME
            marker_path = root / subject.MARKER_NAME
            json_path.write_text("{}\n", encoding="utf-8")
            markdown_path.write_text("# report\n", encoding="utf-8")
            report = {"decision": "ENGINEERING_GATE_FAIL"}
            manifest = {
                "schema": subject.MANIFEST_SCHEMA,
                "input_count": 0,
                "inputs": [],
            }
            with (
                mock.patch.object(
                    subject,
                    "_report_paths",
                    return_value=(
                        json_path,
                        markdown_path,
                        manifest_path,
                        marker_path,
                    ),
                ),
                mock.patch.object(
                    subject,
                    "validate_published_report",
                    return_value=report,
                ),
                mock.patch.object(
                    subject,
                    "build_manifest",
                    return_value=manifest,
                ),
            ):
                result = subject.publish_completion(root)
                with self.assertRaises(FileExistsError):
                    subject.publish_completion(root)
            self.assertEqual(result["status"], "complete")
            rows = marker_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 3)
            self.assertTrue(rows[0].endswith(summary.JSON_OUTPUT_NAME))
            self.assertTrue(rows[1].endswith(summary.MARKDOWN_OUTPUT_NAME))
            self.assertTrue(rows[2].endswith(subject.MANIFEST_NAME))

    def test_verify_rejects_a_tampered_completion_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / summary.JSON_OUTPUT_NAME
            markdown_path = root / summary.MARKDOWN_OUTPUT_NAME
            manifest_path = root / subject.MANIFEST_NAME
            marker_path = root / subject.MARKER_NAME
            json_path.write_text("{}\n", encoding="utf-8")
            markdown_path.write_text("# report\n", encoding="utf-8")
            manifest = {
                "schema": subject.MANIFEST_SCHEMA,
                "input_count": 0,
                "inputs": [],
            }
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            marker_path.write_text("0" * 64 + "  bad\n", encoding="utf-8")
            report = {
                "decision": "ENGINEERING_GATE_FAIL",
                "engineering_gate_passed": False,
                "ner_stage_authorized": False,
            }
            with (
                mock.patch.object(
                    subject,
                    "_report_paths",
                    return_value=(
                        json_path,
                        markdown_path,
                        manifest_path,
                        marker_path,
                    ),
                ),
                mock.patch.object(
                    subject,
                    "validate_published_report",
                    return_value=report,
                ),
                mock.patch.object(
                    subject,
                    "build_manifest",
                    return_value=manifest,
                ),
            ):
                with self.assertRaisesRegex(
                    summary.IncompleteArtifact, "marker"
                ):
                    subject.verify_completion(root)

    def test_cli_requires_an_explicit_mode(self) -> None:
        self.assertEqual(subject.parse_args(["preflight"]).mode, "preflight")
        self.assertEqual(subject.parse_args(["publish"]).mode, "publish")
        self.assertEqual(subject.parse_args(["verify"]).mode, "verify")
        with self.assertRaises(SystemExit):
            subject.parse_args([])


if __name__ == "__main__":
    unittest.main()
