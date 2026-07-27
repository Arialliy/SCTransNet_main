from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments import evaluate_tpd_clean_v6_pd_fa_checkpoint_compat as compat
from experiments import freeze_tpd_clean_v6_checkpoint_metric_compatibility_source_lock as subject


class FreezeCheckpointMetricCompatibilitySourceLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = subject.build_source_lock_payload()

    def test_payload_binds_three_old_locks_evaluators_and_new_sources(
        self,
    ) -> None:
        payload = self.payload
        self.assertEqual(payload["schema"], compat.SOURCE_LOCK_SCHEMA)
        self.assertEqual(
            set(payload["frozen_lock_sha256"]),
            set(compat.EXPECTED_LOCK_BINDINGS),
        )
        for name, (_, expected_sha) in compat.EXPECTED_LOCK_BINDINGS.items():
            self.assertEqual(
                payload["frozen_lock_sha256"][name], expected_sha
            )
        self.assertEqual(
            set(payload["source_sha256"]),
            set(compat.COMPATIBILITY_SOURCE_RELATIVES),
        )
        self.assertEqual(
            payload["source_count"],
            len(compat.COMPATIBILITY_SOURCE_RELATIVES),
        )
        self.assertEqual(
            payload["base_evaluator_sha256"][
                "experiments/evaluate_tpd_clean_v6_pd_fa.py"
            ],
            compat.sha256_file(compat.FROZEN_EVALUATOR),
        )
        self.assertEqual(
            payload["base_evaluator_sha256"][
                "experiments/evaluate_pd_fa_sweep.py"
            ],
            compat.sha256_file(compat.GENERIC_BASE_EVALUATOR),
        )
        self.assertTrue(
            payload["policy"]["audit_supplement_is_in_memory_only"]
        )
        self.assertTrue(
            payload["policy"][
                "old_acceptance_runs_before_compatibility_acceptance"
            ]
        )
        self.assertEqual(
            payload["policy"]["non_strict_numeric_delta_limits"],
            compat.NON_STRICT_NUMERIC_DELTA_LIMITS,
        )
        self.assertTrue(
            payload["policy"]["sweep_task_metric_points_preserved"]
        )
        self.assertTrue(
            payload["policy"]["sweep_val_loss_normalized_to_checkpoint"]
        )
        self.assertTrue(
            payload["policy"][
                "raw_fixed_audit_preserved_before_normalization"
            ]
        )
        self.assertTrue(
            payload["policy"][
                "formal_inference_replays_training_environment"
            ]
        )

    def test_checked_in_lock_is_canonical_and_current(self) -> None:
        path = compat.DEFAULT_COMPATIBILITY_SOURCE_LOCK
        self.assertTrue(path.is_file())
        self.assertFalse(path.is_symlink())
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8")),
            self.payload,
        )
        loaded, digest = compat.validate_compatibility_source_lock(path)
        self.assertEqual(loaded, self.payload)
        self.assertEqual(digest, compat.sha256_file(path))

    def test_tampered_source_digest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = json.loads(json.dumps(self.payload))
            relative = next(iter(payload["source_sha256"]))
            payload["source_sha256"][relative] = "0" * 64
            path = Path(directory) / "tampered.json"
            subject.write_new_json(path, payload)
            with self.assertRaisesRegex(ValueError, "source differs"):
                compat.validate_compatibility_source_lock(path)

    def test_write_is_canonical_exclusive_and_does_not_replace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compat-lock.json"
            subject.write_new_json(path, self.payload)
            original = path.read_bytes()
            self.assertEqual(
                json.loads(original.decode("utf-8")), self.payload
            )
            with self.assertRaises(FileExistsError):
                subject.write_new_json(path, self.payload)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
