from __future__ import annotations

import copy
import importlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments import (
    freeze_tpd_ner_v8_mprs_dch_v2_source_locks as subject,
)
from experiments import train_tpd_ner_v8_mprs_dch_v2 as ordinary


class V2SourceLockContractTests(unittest.TestCase):
    def test_single_seed_single_candidate_contract(self) -> None:
        formal = subject.formal_contract()
        self.assertEqual(
            tuple(formal["candidate_variants"]),
            (ordinary.TPD_NER_V8_MPRS_DCH_V2_FULL_RELAY_ON,),
        )
        self.assertEqual(formal["training_seed"], 42)
        self.assertEqual(formal["split_seed"], 20260722)
        self.assertFalse(formal["multi_seed_scheduled"])
        self.assertEqual(
            formal["required_control"],
            ordinary.V1_RELAY_OFF_REFERENCE,
        )
        self.assertFalse(formal["relay_off_retrained"])

    def test_runtime_source_set_is_complete_and_unique(self) -> None:
        relatives = subject.training_source_relatives()
        self.assertEqual(len(relatives), len(set(relatives)))
        required = {
            "experiments/TPD_NER_V8_MPRS_DCH_V2_PROTOCOL.md",
            "experiments/train_tpd_ner_v8_mprs_dch_v2.py",
            "experiments/train_tpd_ner_v8_mprs_dch_v2_exact.py",
            "model/tpd_ner_v8_mprs_dch_v2.py",
        }
        self.assertTrue(required.issubset(relatives))
        for relative in relatives:
            path = (subject.REPO_ROOT / relative).resolve()
            self.assertTrue(path.is_file(), relative)
            self.assertFalse(path.is_symlink(), relative)

    def test_acceptance_sources_cover_the_complete_closure(self) -> None:
        relatives = subject.ACCEPTANCE_SOURCE_RELATIVES
        self.assertEqual(len(relatives), len(set(relatives)))
        required = {
            "experiments/evaluate_tpd_ner_v8_mprs_dch_v2_pd_fa.py",
            "experiments/postprocess_tpd_ner_v8_mprs_dch_v2_formal800.py",
            "experiments/smoke_tpd_ner_v8_mprs_dch_v2.py",
            "experiments/smoke_tpd_ner_v8_mprs_dch.py",
            "experiments/handoff_tpd_ner_v8_v1_to_v2.py",
            "experiments/run_tpd_ner_v8_mprs_dch_v2_formal800_1x5090_lane.sh",
            "experiments/launch_tpd_ner_v8_mprs_dch_v2_formal800_1x5090.sh",
            "experiments/evaluate_tpd_ner_v8_mprs_dch_pd_fa.py",
            (
                "experiments/"
                "evaluate_sctransnet_baseline_reference_closed_interval.py"
            ),
            "experiments/evaluate_pd_fa_sweep.py",
        }
        self.assertTrue(required.issubset(relatives))
        for relative in relatives:
            self.assertTrue((subject.REPO_ROOT / relative).is_file(), relative)

    def test_performance_contract_does_not_gate_on_v1_off(self) -> None:
        contract = subject.performance_gate_contract()
        self.assertEqual(contract["anchor_target_count"], 189)
        self.assertFalse(contract["v1_off_absolute_gate_required"])
        self.assertFalse(contract["baseline_affects_decision"])
        paired = contract[
            "paired_v2_on_vs_v1_off_each_checkpoint_role"
        ]
        self.assertEqual(paired["minimum_non_inferior_budget_count"], 4)
        self.assertEqual(paired["minimum_strictly_better_budget_count"], 1)
        self.assertEqual(paired["budget_count"], 5)

    def test_acceptance_policy_enforces_the_single_seed_contract(
        self,
    ) -> None:
        payload = subject.build_acceptance_lock(
            subject.DEFAULT_TRAINING_LOCK
        )
        policy = payload["policy"]
        expected = {
            "training_seed": 42,
            "split_seed": 20260722,
            "multi_seed_scheduled": False,
            "v1_off_sweeps_read_only": True,
            "baseline_sweeps_read_only": True,
        }
        for name, value in expected.items():
            self.assertEqual(policy[name], value)

        invalid = {
            "training_seed": 7,
            "split_seed": 17,
            "multi_seed_scheduled": True,
            "v1_off_sweeps_read_only": False,
            "baseline_sweeps_read_only": False,
        }
        for name, value in invalid.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(payload)
                changed["policy"][name] = value
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "acceptance.json"
                    path.write_text(
                        json.dumps(changed, sort_keys=True),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        f"acceptance policy differs: {name}",
                    ):
                        subject.verify_acceptance_lock(
                            path,
                            subject.DEFAULT_TRAINING_LOCK,
                        )

    def test_import_does_not_publish_or_modify_source_locks(self) -> None:
        paths = (
            subject.DEFAULT_TRAINING_LOCK,
            subject.DEFAULT_ACCEPTANCE_LOCK,
        )

        def state(path: Path) -> tuple[bool, int | None, int | None]:
            if not path.exists() and not path.is_symlink():
                return False, None, None
            stat = path.stat()
            return True, stat.st_size, stat.st_mtime_ns

        before = {str(path): state(Path(path)) for path in paths}
        importlib.reload(subject)
        after = {str(path): state(Path(path)) for path in paths}
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
