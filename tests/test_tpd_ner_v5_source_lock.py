from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOCK = REPO_ROOT / "experiments/tpd_ner_v5_source_lock.json"

REQUIRED_SOURCES = {
    "model/tpd_ner_v5.py",
    "experiments/train_tpd_ner_v5.py",
    "experiments/evaluate_tpd_ner_v5_pd_fa.py",
    "experiments/smoke_tpd_ner_v5.py",
    "experiments/TPD_NER_V5_PROTOCOL.md",
    "tests/test_tpd_ner_v5.py",
    "tests/test_train_tpd_ner_v5.py",
    "tests/test_evaluate_tpd_ner_v5_pd_fa.py",
    "tests/test_smoke_tpd_ner_v5.py",
    "tests/test_tpd_ner_v5_source_lock.py",
    "experiments/tpd_ner_runtime.py",
    "model/tpd_sctransnet.py",
    "model/tpd_relay.py",
    "model/tpd_clean_v5.py",
    "experiments/evaluate_tpd_clean_v5_pd_fa.py",
    "experiments/train_tpd_pilot.py",
    "experiments/evaluate_pd_fa_sweep.py",
    "model/SCTransNet.py",
    "model/Config.py",
    "dataset.py",
    "utils.py",
    "warmup_scheduler.py",
    "experiments/tpd_clean_v5_screen800_2x_source_lock.json",
    "experiments/tpd_ner_v1_source_lock.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TPDNERV5SourceLockTests(unittest.TestCase):
    def test_source_lock_recomputes_and_preserves_mainline_contract(self) -> None:
        payload = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["schema"],
            "sctransnet_tpd_ner_v5_source_lock_v1",
        )
        self.assertEqual(payload["mainline_contract"], "Keep-Context-Saliency")
        self.assertFalse(payload["fourth_parallel_branch_added"])
        self.assertEqual(
            payload["evidence_nodes"],
            ["h11", "h12", "h13", "h21", "h22"],
        )
        self.assertEqual(payload["relay_stage_order"], [4, 3, 2])
        self.assertEqual(payload["model_seeds"], [42, 3407])
        self.assertEqual(
            payload["variants"],
            [
                "tpd_clean_v5_full_relay_off",
                "tpd_clean_v5_full_relay_on",
                "progressive_relay_off",
                "progressive_relay_on",
            ],
        )
        self.assertFalse(payload["formal_training_started"])
        self.assertEqual(payload["gate_connection"], "tpd_clean_v5_gate_a_to_e")
        self.assertEqual(
            payload["production_parameter_contract"],
            {
                "shallow_embedding_parameters": 66_176,
                "relay_off_total_parameters": 10_843_155,
                "relay_on_total_parameters": 10_854_446,
                "relay_parameters": 11_291,
                "relay_gate_parameters": 27,
            },
        )
        self.assertEqual(set(payload["source_sha256"]), REQUIRED_SOURCES)
        for relative, expected in payload["source_sha256"].items():
            path = REPO_ROOT / relative
            with self.subTest(relative=relative):
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertEqual(sha256(path), expected)

    def test_recursive_frozen_locks_match_their_declared_hashes(self) -> None:
        payload = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
        bindings = {
            "experiments/tpd_clean_v5_screen800_2x_source_lock.json": (
                "frozen_v5_source_lock_sha256"
            ),
            "experiments/tpd_ner_v1_source_lock.json": (
                "frozen_legacy_ner_source_lock_sha256"
            ),
        }
        for relative, field in bindings.items():
            with self.subTest(relative=relative):
                self.assertEqual(
                    payload[field],
                    payload["source_sha256"][relative],
                )
                self.assertEqual(
                    payload[field],
                    sha256(REPO_ROOT / relative),
                )


if __name__ == "__main__":
    unittest.main()
