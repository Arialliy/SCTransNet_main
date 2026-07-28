from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import (
    summarize_tpd_ner_v8_mprs_dch_v4_tail_aware_comparative_performance
    as subject,
)


def _load_real_source() -> dict:
    return json.loads(
        subject.SOURCE_COMPARISON.read_text(encoding="utf-8")
    )


def _find_row(
    source: dict,
    variant: str,
    role: str,
) -> dict:
    matches = [
        row
        for row in source["rows"]
        if row["variant"] == variant
        and row["checkpoint_role"] == role
    ]
    if len(matches) != 1:
        raise ValueError(f"fixture row lookup differs: {variant}:{role}")
    return matches[0]


def _write_source_authority(
    root: Path,
    source: dict,
) -> tuple[Path, Path, str, str]:
    source_dir = root / "source"
    source_dir.mkdir(parents=True)
    comparison = source_dir / "comparison.json"
    markdown = source_dir / "comparison.md"
    marker = source_dir / "POSTPROCESS_COMPLETE.json"
    comparison.write_text(
        json.dumps(
            source,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    markdown.write_text("synthetic frozen comparison\n", encoding="utf-8")
    marker_value = {
        "schema": subject.SOURCE_MARKER_SCHEMA,
        "status": "complete",
        "decision": source["decision"],
        "aggregate_full_model_gate_passed": source[
            "aggregate_full_model_gate_passed"
        ],
        "outputs": {
            comparison.name: subject.sha256_file(comparison),
            markdown.name: subject.sha256_file(markdown),
        },
    }
    marker.write_text(
        json.dumps(
            marker_value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        comparison,
        marker,
        subject.sha256_file(comparison),
        subject.sha256_file(marker),
    )


def _aggregate_synthetic(root: Path, source: dict) -> dict:
    comparison, marker, comparison_sha, marker_sha = (
        _write_source_authority(root, source)
    )
    return subject.aggregate(
        comparison_path=comparison,
        marker_path=marker,
        expected_comparison_sha256=comparison_sha,
        expected_marker_sha256=marker_sha,
    )


class ComparativePerformanceDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.real_report = subject.aggregate()

    def test_real_frozen_result_confirms_relative_success_with_tradeoff(
        self,
    ) -> None:
        report = self.real_report
        self.assertEqual(
            report["decision"],
            "RELATIVE_MODEL_IMPROVEMENT_CONFIRMED_WITH_TRADEOFF",
        )
        self.assertTrue(report["comprehensive_relative_gate_passed"])
        self.assertTrue(report["model_iteration_success"])
        self.assertTrue(report["next_model_stage_authorized"])
        self.assertTrue(
            all(report["relative_success_components"].values())
        )
        self.assertTrue(report["tradeoff_present"])
        self.assertFalse(report["universal_dominance"])
        self.assertFalse(report["strictest_fa_budget_advantage"])
        self.assertEqual(
            report["policy"]["timing"],
            "post_training_user_confirmed_engineering_decision",
        )
        self.assertFalse(
            report["policy"]["four_of_five_rule_used_as_veto"]
        )
        self.assertEqual(
            set(report["relative_success_components"]),
            {
                "v4_contributes_global_fixed_pareto_point",
                (
                    "v4_strictly_improves_historical_envelope_"
                    "at_any_fa_budget"
                ),
                "v4_two_fixed_checkpoints_tiny_pd_no_regression",
            },
        )

    def test_global_fixed_frontier_is_symmetric_over_all_ten_points(
        self,
    ) -> None:
        frontier = self.real_report["global_fixed_pareto_frontier"]
        self.assertEqual(frontier["point_count"], 10)
        self.assertEqual(
            frontier["frontier_point_keys"],
            [
                f"{subject.V1_VARIANT}:miou_secondary",
                f"{subject.V4_VARIANT}:pd_primary",
                f"{subject.V4_VARIANT}:miou_secondary",
            ],
        )
        self.assertEqual(frontier["frontier_point_count"], 3)
        self.assertEqual(len(frontier["points"]), 10)
        self.assertTrue(
            all(
                point["is_global_pareto"]
                for point in frontier["frontier_points"]
            )
        )

    def test_v4_model_envelope_uses_both_checkpoints_transparently(
        self,
    ) -> None:
        envelopes = self.real_report["model_fa_budget_envelopes"]
        self.assertEqual(set(envelopes), set(subject.ALL_VARIANTS))
        for variant in subject.ALL_VARIANTS:
            with self.subTest(variant=variant):
                self.assertEqual(
                    set(envelopes[variant]["points"]),
                    set(subject.BUDGET_KEYS),
                )
                self.assertTrue(
                    all(
                        len(point["role_candidates"]) == 2
                        for point in envelopes[variant][
                            "points"
                        ].values()
                    )
                )
        envelope = envelopes[
            subject.V4_VARIANT
        ]
        self.assertEqual(
            envelope["matched_target_profile_in_budget_order"],
            [0, 188, 189, 189, 189],
        )
        self.assertEqual(
            envelope["selected_role_profile_in_budget_order"],
            [
                "pd_primary",
                "miou_secondary",
                "pd_primary",
                "pd_primary",
                "pd_primary",
            ],
        )
        self.assertEqual(
            envelope["points"]["1e-06"]["co_leader_role_names"],
            ["pd_primary", "miou_secondary"],
        )
        self.assertEqual(
            envelope["points"]["5e-06"]["selected_role_name"],
            "miou_secondary",
        )
        self.assertEqual(
            envelope["points"]["1e-05"]["selected_role_name"],
            "pd_primary",
        )

    def test_global_budget_leaders_and_strict_new_budget_are_exact(
        self,
    ) -> None:
        leaders = self.real_report["global_fa_budget_leaders"]
        self.assertEqual(
            leaders["v4_global_leader_budget_keys"],
            ["5e-06", "1e-05", "5e-05", "0.0001"],
        )
        self.assertEqual(leaders["v4_global_leader_budget_count"], 4)
        self.assertEqual(
            leaders["v4_strict_new_detection_budget_keys"],
            ["1e-05"],
        )
        self.assertEqual(
            leaders["points"]["5e-06"]["leader_variants"],
            [subject.V3_VARIANT, subject.V4_VARIANT],
        )
        self.assertFalse(
            leaders["points"]["1e-06"][
                "v4_strictly_exceeds_all_historical_matched"
            ]
        )
        self.assertTrue(
            leaders["points"]["1e-05"][
                "v4_strictly_exceeds_all_historical_matched"
            ]
        )

    def test_real_pairwise_matrix_preserves_dominance_and_tradeoffs(
        self,
    ) -> None:
        pairwise = self.real_report["pairwise_by_reference_and_role"]
        expected = {
            subject.BASELINE_VARIANT: {
                "pd_primary": ("candidate_dominates", 5, 3, 0),
                "miou_secondary": ("tradeoff", 4, 4, 1),
            },
            subject.V1_VARIANT: {
                "pd_primary": ("candidate_dominates", 4, 2, 1),
                "miou_secondary": ("tradeoff", 4, 4, 1),
            },
            subject.V2_VARIANT: {
                "pd_primary": ("tradeoff", 4, 3, 1),
                "miou_secondary": ("tradeoff", 5, 4, 0),
            },
            subject.V3_VARIANT: {
                "pd_primary": ("tradeoff", 3, 3, 2),
                "miou_secondary": ("candidate_dominates", 5, 4, 0),
            },
        }
        for reference, roles in expected.items():
            for role_name, expected_value in roles.items():
                with self.subTest(reference=reference, role=role_name):
                    value = pairwise[reference][role_name]
                    budget = value["pd_at_fa_budget"]
                    observed = (
                        value["fixed_threshold_0_5"]["relation"],
                        budget["noninferior_budget_count"],
                        budget["strictly_better_budget_count"],
                        budget["strictly_worse_budget_count"],
                    )
                    self.assertEqual(observed, expected_value)

    def test_original_absolute_failure_is_diagnostic_not_veto(self) -> None:
        original = self.real_report["source_original_decision"]
        self.assertFalse(original["aggregate_full_model_gate_passed"])
        self.assertEqual(
            original["decision"],
            "RETURN_TO_MODEL_OPTIMIZATION",
        )
        self.assertEqual(
            original["role"],
            "retained_unchanged_diagnostic_only",
        )
        self.assertFalse(original["veto_applied_to_relative_decision"])
        self.assertTrue(
            self.real_report["comprehensive_relative_gate_passed"]
        )

    def test_tradeoff_is_never_reported_as_candidate_dominance(self) -> None:
        authority = subject.validate_source_authority()
        role = subject.CHECKPOINT_ROLES["best.pth.tar"]
        value = subject.fixed_pareto_assessment(
            authority["rows"][(subject.V2_VARIANT, role)],
            authority["rows"][(subject.V4_VARIANT, role)],
        )
        self.assertEqual(value["relation"], "tradeoff")
        self.assertFalse(value["candidate_dominates"])
        self.assertFalse(value["reference_dominates"])
        self.assertIn(
            "better",
            value["objective_group_relations"].values(),
        )
        self.assertIn(
            "worse",
            value["objective_group_relations"].values(),
        )

    def test_budget_profile_with_equal_and_worse_points_is_reference_dominated(
        self,
    ) -> None:
        authority = subject.validate_source_authority()
        role = subject.CHECKPOINT_ROLES["best.pth.tar"]
        reference = authority["rows"][(subject.V2_VARIANT, role)]
        candidate = copy.deepcopy(reference)
        candidate["variant"] = subject.V4_VARIANT
        candidate["pd_at_fa_budget"]["5e-06"][
            "matched_target_count"
        ] -= 1
        candidate["pd_at_fa_budget"]["5e-06"]["pd"] = (
            candidate["pd_at_fa_budget"]["5e-06"][
                "matched_target_count"
            ]
            / subject.TARGET_COUNT
        )
        value = subject.budget_assessment(reference, candidate)
        self.assertEqual(
            value["profile_relation"],
            "reference_pd_budget_dominates",
        )
        self.assertEqual(value["strictly_better_budget_count"], 0)
        self.assertEqual(value["strictly_worse_budget_count"], 1)

    def test_tiny_pd_regression_fails_relative_gate(self) -> None:
        source = copy.deepcopy(_load_real_source())
        secondary = subject.CHECKPOINT_ROLES["best_miou.pth.tar"]
        candidate = _find_row(source, subject.V4_VARIANT, secondary)
        fixed = candidate["fixed_threshold_0_5"]
        fixed["matched_tiny_target_count"] = 38
        fixed["tiny_pd"] = 38 / subject.TINY_TARGET_COUNT
        with tempfile.TemporaryDirectory() as directory:
            report = _aggregate_synthetic(Path(directory), source)
        self.assertFalse(
            report["relative_success_components"][
                "v4_two_fixed_checkpoints_tiny_pd_no_regression"
            ]
        )
        self.assertFalse(report["comprehensive_relative_gate_passed"])

    def test_failed_markdown_does_not_claim_confirmed_iteration(
        self,
    ) -> None:
        source = copy.deepcopy(_load_real_source())
        secondary = subject.CHECKPOINT_ROLES["best_miou.pth.tar"]
        candidate = _find_row(source, subject.V4_VARIANT, secondary)
        candidate["fixed_threshold_0_5"][
            "matched_tiny_target_count"
        ] = 38
        candidate["fixed_threshold_0_5"]["tiny_pd"] = (
            38 / subject.TINY_TARGET_COUNT
        )
        with tempfile.TemporaryDirectory() as directory:
            report = _aggregate_synthetic(Path(directory), source)
        markdown = subject.render_markdown(report)
        self.assertIn(
            "does not confirm a relative model iteration",
            markdown,
        )
        self.assertNotIn(
            "confirms a relative model iteration",
            markdown,
        )

    def test_missing_budget_metric_and_duplicate_role_are_rejected(
        self,
    ) -> None:
        primary = subject.CHECKPOINT_ROLES["best.pth.tar"]
        source_missing = copy.deepcopy(_load_real_source())
        row = _find_row(source_missing, subject.V4_VARIANT, primary)
        del row["pd_at_fa_budget"]["1e-06"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "budget keys differ"):
                _aggregate_synthetic(Path(directory), source_missing)

        source_duplicate = copy.deepcopy(_load_real_source())
        baseline_rows = [
            index
            for index, row in enumerate(source_duplicate["rows"])
            if row["variant"] == subject.BASELINE_VARIANT
        ]
        source_duplicate["rows"][baseline_rows[1]] = copy.deepcopy(
            source_duplicate["rows"][baseline_rows[0]]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "duplicate source row"):
                _aggregate_synthetic(Path(directory), source_duplicate)

    def test_nonfinite_metric_and_wrong_checkpoint_role_are_rejected(
        self,
    ) -> None:
        primary = subject.CHECKPOINT_ROLES["best.pth.tar"]
        source_nonfinite = copy.deepcopy(_load_real_source())
        row = _find_row(source_nonfinite, subject.V4_VARIANT, primary)
        row["fixed_threshold_0_5"]["miou"] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "mIoU is not finite"):
                _aggregate_synthetic(Path(directory), source_nonfinite)

        source_wrong_role = copy.deepcopy(_load_real_source())
        row = _find_row(source_wrong_role, subject.V4_VARIANT, primary)
        row["checkpoint_role"] = (
            subject.CHECKPOINT_ROLES["best_miou.pth.tar"]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "checkpoint role differs",
            ):
                _aggregate_synthetic(Path(directory), source_wrong_role)

    def test_input_and_marker_hash_bindings_are_enforced(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "source comparison SHA-256 differs",
        ):
            subject.validate_source_authority(
                expected_comparison_sha256="0" * 64,
            )
        with self.assertRaisesRegex(
            ValueError,
            "source marker SHA-256 differs",
        ):
            subject.validate_source_authority(
                expected_marker_sha256="0" * 64,
            )

        source = copy.deepcopy(_load_real_source())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comparison, marker, comparison_sha, marker_sha = (
                _write_source_authority(root, source)
            )
            (marker.parent / "comparison.md").write_text(
                "changed after marker\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "marker output SHA-256 differs",
            ):
                subject.validate_source_authority(
                    comparison_path=comparison,
                    marker_path=marker,
                    expected_comparison_sha256=comparison_sha,
                    expected_marker_sha256=marker_sha,
                )

    def test_snapshot_before_must_match_just_validated_bindings(
        self,
    ) -> None:
        with mock.patch.object(
            subject,
            "_snapshot_paths",
            return_value={"comparison": "0" * 64},
        ):
            with self.assertRaisesRegex(
                ValueError,
                "changed after validation before aggregation",
            ):
                subject.aggregate()

    def test_forged_report_cannot_be_published_or_marked_complete(
        self,
    ) -> None:
        forged = copy.deepcopy(self.real_report)
        forged["decision"] = "FORGED_SUCCESS"
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "relative"
            with self.assertRaisesRegex(
                ValueError,
                "differs from default frozen authority",
            ):
                subject.publish_report(forged, output_dir=output_dir)
            self.assertFalse(output_dir.exists())

    def test_publish_is_write_once_and_verify_recomputes_everything(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "relative"
            paths = subject.publish_report(
                self.real_report,
                output_dir=output_dir,
            )
            verified = subject.verify_published(output_dir=output_dir)
            self.assertEqual(paths, verified)
            marker = json.loads(paths[2].read_text(encoding="utf-8"))
            self.assertEqual(
                marker["inputs"]["source_comparison"]["sha256"],
                subject.SOURCE_COMPARISON_SHA256,
            )
            self.assertEqual(
                marker["inputs"]["source_completion_marker"]["sha256"],
                subject.SOURCE_MARKER_SHA256,
            )
            self.assertEqual(
                marker["outputs"][paths[0].name],
                subject.sha256_file(paths[0]),
            )
            self.assertEqual(
                marker["outputs"][paths[1].name],
                subject.sha256_file(paths[1]),
            )
            with self.assertRaisesRegex(ValueError, "overwrite"):
                subject.publish_report(
                    self.real_report,
                    output_dir=output_dir,
                )
            paths[0].write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "JSON differs on recomputation",
            ):
                subject.verify_published(output_dir=output_dir)


if __name__ == "__main__":
    unittest.main()
