from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments import (
    tpd_ner_v8_mprs_dch_v3_dc_knockout_spec as subject,
)


class V3DCKnockoutSpecTests(unittest.TestCase):
    def test_matrix_is_exact_checkpoint_major_two_by_four(self) -> None:
        self.assertEqual(
            subject.CHECKPOINTS,
            ("best.pth.tar", "best_miou.pth.tar"),
        )
        self.assertEqual(
            subject.KNOCKOUT_MODES,
            (
                "zero_all_dc",
                "zero_dc_stage4",
                "zero_dc_stage3",
                "zero_dc_stage2",
            ),
        )
        rows = subject.matrix_rows()
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [(row["checkpoint"], row["knockout_mode"]) for row in rows],
            [
                (checkpoint, mode)
                for checkpoint in subject.CHECKPOINTS
                for mode in subject.KNOCKOUT_MODES
            ],
        )
        self.assertEqual(
            len({row["row_id"] for row in rows}),
            8,
        )

    def test_knockout_key_registry_is_exact(self) -> None:
        self.assertEqual(
            subject.DC_OFFSET_KEYS,
            (
                "tpd_ner.dc_offsets.4",
                "tpd_ner.dc_offsets.3",
                "tpd_ner.dc_offsets.2",
            ),
        )
        self.assertEqual(
            subject.KNOCKOUT_ZERO_KEYS["zero_all_dc"],
            subject.DC_OFFSET_KEYS,
        )
        for stage in ("4", "3", "2"):
            self.assertEqual(
                subject.KNOCKOUT_ZERO_KEYS[f"zero_dc_stage{stage}"],
                (f"tpd_ner.dc_offsets.{stage}",),
            )

    def test_spec_is_diagnostic_only_and_contains_no_gate_authority(
        self,
    ) -> None:
        contract = subject.fixed_specification()
        self.assertEqual(
            subject.validate_specification(contract),
            contract,
        )
        self.assertTrue(contract["diagnostic_only"])
        self.assertFalse(contract["affects_formal_gate"])
        self.assertFalse(contract["formal_decision_authority"])
        self.assertEqual(contract["formal_gate_components"], [])
        self.assertFalse(
            contract["publication_contract"][
                "aggregate_may_contain_decision"
            ]
        )
        self.assertFalse(contract["multi_seed_scheduled"])
        self.assertFalse(contract["official_test_accessed"])
        self.assertEqual(contract["row_count"], 8)
        self.assertRegex(subject.specification_sha256(), r"^[0-9a-f]{64}$")
        self.assertTrue(contract["schema"].endswith("_v2"))
        execution = contract["execution_contract"]
        self.assertEqual(
            execution["cublas_workspace_config_env"],
            "CUBLAS_WORKSPACE_CONFIG",
        )
        self.assertEqual(
            execution["cublas_workspace_config"],
            ":4096:8",
        )
        self.assertEqual(execution["pythonhashseed_env"], "PYTHONHASHSEED")
        self.assertEqual(execution["pythonhashseed"], "42")
        self.assertTrue(
            execution[
                "cublas_workspace_config_required_before_evaluator_start"
            ]
        )
        self.assertTrue(
            execution["pythonhashseed_required_before_evaluator_start"]
        )

    def test_threshold_contract_matches_locked_no_zero_lower_endpoint(
        self,
    ) -> None:
        threshold = subject.threshold_contract()
        self.assertFalse(threshold["include_zero"])
        self.assertTrue(threshold["include_one"])
        self.assertTrue(threshold["include_last_float32_below_one"])
        self.assertEqual(
            tuple(threshold["fa_budgets"]),
            subject.FA_BUDGETS,
        )
        self.assertEqual(threshold["prediction_comparison"], "prediction > threshold")

    def test_output_root_and_marker_are_formal_path_isolated(self) -> None:
        self.assertTrue(
            str(subject.DEFAULT_OUTPUT_ROOT).endswith(
                "tpd_ner_v8_mprs_dch_v3_dc_knockout_v2"
            )
        )
        self.assertNotEqual(
            subject.DEFAULT_OUTPUT_ROOT.resolve(),
            subject.FORMAL_RESULT_ROOT.resolve(),
        )
        self.assertTrue(
            subject.DEFAULT_RUN_DIR.is_relative_to(
                subject.DEFAULT_OUTPUT_ROOT
            )
        )
        paths = subject.aggregate_paths()
        self.assertTrue(
            all(path.is_relative_to(subject.DEFAULT_OUTPUT_ROOT) for path in paths)
        )
        self.assertEqual(paths[-1].name, "DC_KNOCKOUT_COMPLETE.json")
        for checkpoint in subject.CHECKPOINTS:
            path = subject.sweep_path(checkpoint)
            self.assertTrue(path.is_relative_to(subject.DEFAULT_OUTPUT_ROOT))
            self.assertEqual(
                path.name,
                subject.SWEEP_FILENAMES[checkpoint],
            )

    def test_formal_output_root_is_rejected_as_diagnostic_root(self) -> None:
        forbidden = (
            subject.FORMAL_RESULT_ROOT,
            subject.FORMAL_RESULT_ROOT / "nested-diagnostic",
        )
        for root in forbidden:
            with self.subTest(root=root):
                with self.assertRaisesRegex(ValueError, "formal V3"):
                    subject.run_directory(root)
                with self.assertRaisesRegex(ValueError, "formal V3"):
                    subject.aggregate_paths(root)

    def test_symbolic_link_output_root_or_ancestor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            real = temporary / "real"
            real.mkdir()
            link = temporary / "linked"
            link.symlink_to(real, target_is_directory=True)
            for root in (link, link / "nested"):
                with self.subTest(root=root):
                    with self.assertRaisesRegex(
                        ValueError,
                        "symbolic-link component",
                    ):
                        subject.run_directory(root)
                    with self.assertRaisesRegex(
                        ValueError,
                        "symbolic-link component",
                    ):
                        subject.aggregate_paths(root)

    def test_spec_rejects_any_mutated_contract(self) -> None:
        changed = subject.fixed_specification()
        changed["multi_seed_scheduled"] = True
        with self.assertRaisesRegex(ValueError, "differs"):
            subject.validate_specification(changed)


if __name__ == "__main__":
    unittest.main()
