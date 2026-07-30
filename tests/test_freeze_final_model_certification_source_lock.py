from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import freeze_final_model_certification_source_lock as lock


class FinalModelCertificationSourceLockTests(unittest.TestCase):
    def test_live_payload_has_explicit_engineering_only_scope(self) -> None:
        payload = lock.build_source_lock_payload()
        self.assertEqual(payload["schema"], lock.SCHEMA)
        self.assertEqual(
            payload["execution_scope"],
            "fixed_parent_engineering_b_d_only",
        )
        self.assertTrue(
            payload["readiness"]["engineering_b_d_execution_ready"]
        )
        self.assertTrue(
            payload["readiness"][
                "f1_executable_six_mode_audit_complete"
            ]
        )
        self.assertFalse(
            payload["readiness"]["f1_six_mode_audit_execution_complete"]
        )
        self.assertFalse(
            payload["readiness"][
                "confirmatory_full_pipeline_execution_ready"
            ]
        )
        self.assertEqual(payload["engineering_matrix"]["run_count"], 4)
        self.assertFalse(payload["official_test_accessed"])

    def test_source_map_is_exact_and_excludes_only_output_self_reference(
        self,
    ) -> None:
        payload = lock.build_source_lock_payload()
        self.assertEqual(
            tuple(payload["source_sha256"]),
            tuple(sorted(lock.SOURCE_PATHS)),
        )
        self.assertNotIn(lock.DEFAULT_OUTPUT_RELATIVE, lock.SOURCE_PATHS)
        self.assertIn(
            "experiments/freeze_final_model_certification_source_lock.py",
            lock.SOURCE_PATHS,
        )
        self.assertIn(
            "experiments/freeze_final_model_certification_parent_lock.py",
            lock.SOURCE_PATHS,
        )
        self.assertIn(
            "experiments/freeze_tpd_ner_v4_survival_exact_source_lock.py",
            lock.SOURCE_PATHS,
        )
        self.assertIn(
            (
                "experiments/"
                "freeze_tpd_ner_v4_qfg_v2_croa_exact_source_lock.py"
            ),
            lock.SOURCE_PATHS,
        )
        self.assertIn(
            (
                "experiments/"
                "freeze_tpd_ner_v4_qfg_v2_croa_operational_closure_v2.py"
            ),
            lock.SOURCE_PATHS,
        )
        self.assertIn(
            "experiments/freeze_tpd_clean_v8_mprs_dch_source_locks.py",
            lock.SOURCE_PATHS,
        )
        self.assertIn(
            "analysis/run_final_qfg_six_mode_audit.py",
            lock.SOURCE_PATHS,
        )
        self.assertIn(
            "tests/test_run_final_qfg_six_mode_audit.py",
            lock.SOURCE_PATHS,
        )

    def test_write_once_refuses_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            path = Path(directory_text) / "lock.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                lock.write_source_lock_once(path)

    def test_verify_rejects_changed_source_digest(self) -> None:
        payload = lock.build_source_lock_payload()
        relative = next(iter(payload["source_sha256"]))
        payload["source_sha256"][relative] = "0" * 64
        with tempfile.TemporaryDirectory() as directory_text:
            path = Path(directory_text) / "lock.json"
            path.write_bytes(lock.canonical_json_bytes(payload))
            with self.assertRaises(lock.CertificationSourceLockError):
                lock.verify_source_lock(path)

    def test_plan_does_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            path = Path(directory_text) / "lock.json"
            with mock.patch(
                "builtins.print",
            ):
                lock.main(["--plan", "--output", str(path)])
            self.assertFalse(path.exists())

    def test_json_round_trip_is_canonical(self) -> None:
        payload = lock.build_source_lock_payload()
        raw = lock.canonical_json_bytes(payload)
        self.assertEqual(json.loads(raw.decode("utf-8")), payload)


if __name__ == "__main__":
    unittest.main()
