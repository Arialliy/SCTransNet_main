from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from experiments import summarize_tpd_clean_screen800 as summary


def point(
    matched: int,
    fa: float,
    *,
    tiny_matched: int = 39,
    miou: float = 0.8,
    threshold: float = 0.5,
) -> dict:
    return {
        "pd": matched / summary.EXPECTED_TARGET_COUNT,
        "fa": fa,
        "tiny_pd": tiny_matched / summary.EXPECTED_TINY_TARGET_COUNT,
        "miou": miou,
        "niou": miou,
        "pixel_precision": 0.8,
        "pixel_recall": 0.8,
        "pixel_f1": 0.8,
        "val_loss": 0.1,
        "false_objects_per_image": 0.0,
        "matched_target_count": matched,
        "target_count": summary.EXPECTED_TARGET_COUNT,
        "matched_tiny_target_count": tiny_matched,
        "tiny_target_count": summary.EXPECTED_TINY_TARGET_COUNT,
        "predicted_object_count": matched,
        "unmatched_predicted_object_count": 0,
        "valid_pixel_count": summary.EXPECTED_VALID_PIXELS,
        "threshold": threshold,
    }


def sweep(
    fixed: dict,
    *,
    budgets: dict[str, dict | None] | None = None,
    points: list[dict] | None = None,
) -> dict:
    return {
        "checkpoint_epoch": 10,
        "checkpoint_sha256": "a" * 64,
        "fixed_threshold_0_5": fixed,
        "best_points_under_fa_budget": (
            budgets
            if budgets is not None
            else {budget: fixed for budget in summary.EXPECTED_BUDGETS}
        ),
        "points": points if points is not None else [fixed],
    }


def run_record(
    variant: str,
    fixed: dict,
    *,
    budgets: dict[str, dict | None] | None = None,
    points: list[dict] | None = None,
) -> dict:
    primary_sweep = sweep(fixed, budgets=budgets, points=points)
    secondary_sweep = sweep(fixed, budgets=budgets, points=points)
    return {
        "variant": variant,
        "split_sha256": "1" * 64,
        "summary": {
            "model": {
                "shallow_embedding_parameters": 100,
                "total_parameters": 1_000,
                "shared_initialization_sha256": "d" * 64,
            },
            "best_pd_epoch": 10,
            "best_miou_epoch": 11,
            "best_miou_validation_metrics": {
                "pd": fixed["pd"],
                "fa": fixed["fa"],
                "tiny_pd": fixed["tiny_pd"],
                "miou": fixed["miou"],
            },
        },
        "best_sweep": primary_sweep,
        "best_miou_sweep": secondary_sweep,
        "critical_protocol": {"epochs": summary.EXPECTED_EPOCHS},
        "protocol_contract": {"selection": "frozen"},
        "run_dir": f"/fixture/{variant}",
        "checkpoints": {},
        "completion_log": {},
        "launch_manifest": (
            {"source_lock_sha256": "b" * 64}
            if variant in summary.CANDIDATE_VARIANTS
            else None
        ),
        "derived_best_miou_reference": (
            {"variant": variant}
            if variant in summary.REFERENCE_VARIANTS
            else None
        ),
        "artifact_sha256": {"split.json": "c" * 64},
    }


class BudgetOrderingTests(unittest.TestCase):
    def test_unavailable_budget_point_has_null_auditable_key(self) -> None:
        available = point(170, 9e-7)
        cases = (
            (None, None, "tie", "tie", None, None),
            (available, None, "candidate_better", "availability", "list", None),
            (None, available, "reference_better", "availability", None, "list"),
        )
        for candidate, reference, outcome, decisive, candidate_key, reference_key in cases:
            with self.subTest(candidate=candidate is not None, reference=reference is not None):
                result = summary.compare_points(candidate, reference)
                self.assertEqual(result["outcome"], outcome)
                self.assertEqual(result["decisive_metric"], decisive)
                if candidate_key == "list":
                    self.assertIsInstance(result["candidate_key"], list)
                else:
                    self.assertIs(result["candidate_key"], candidate_key)
                if reference_key == "list":
                    self.assertIsInstance(result["reference_key"], list)
                else:
                    self.assertIs(result["reference_key"], reference_key)
                summary.require_finite(result, "comparison")

    def test_cross_method_budget_order_is_frozen_lexicographic(self) -> None:
        cases = (
            (
                "availability",
                point(170, 9e-7),
                None,
                "candidate_better",
                "availability",
            ),
            (
                "pd_before_all_later_metrics",
                point(171, 9e-7, tiny_matched=37, miou=0.60),
                point(170, 0.0, tiny_matched=39, miou=0.99),
                "candidate_better",
                "pd",
            ),
            (
                "lower_fa_on_pd_tie",
                point(170, 5e-7, tiny_matched=37, miou=0.60),
                point(170, 9e-7, tiny_matched=39, miou=0.99),
                "candidate_better",
                "fa",
            ),
            (
                "tiny_pd_after_pd_and_fa",
                point(170, 5e-7, tiny_matched=39, miou=0.60),
                point(170, 5e-7, tiny_matched=38, miou=0.99),
                "candidate_better",
                "tiny_pd",
            ),
            (
                "miou_is_final_cross_method_tie_break",
                point(170, 5e-7, tiny_matched=39, miou=0.81),
                point(170, 5e-7, tiny_matched=39, miou=0.80),
                "candidate_better",
                "miou",
            ),
            (
                "calibration_threshold_is_not_a_cross_method_advantage",
                point(170, 5e-7, miou=0.80, threshold=0.4),
                point(170, 5e-7, miou=0.80, threshold=0.6),
                "tie",
                "tie",
            ),
        )
        for name, candidate, reference, outcome, decisive_metric in cases:
            with self.subTest(name=name):
                result = summary.compare_points(candidate, reference)
                self.assertEqual(result["outcome"], outcome)
                self.assertEqual(result["decisive_metric"], decisive_metric)

    def test_validate_sweep_rejects_nonoptimal_reported_budget_point(self) -> None:
        reported = point(170, 5e-7, miou=0.90, threshold=0.5)
        lexicographic_winner = point(
            171,
            9e-7,
            tiny_matched=37,
            miou=0.60,
            threshold=0.6,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "best.pth.tar"
            checkpoint.write_bytes(b"fixture checkpoint\n")
            payload = {
                "variant": "tpd_clean_full",
                "dataset": summary.EXPECTED_DATASET,
                "seed": summary.EXPECTED_SEED,
                "split_seed": summary.EXPECTED_SPLIT_SEED,
                "validation_count": summary.EXPECTED_VALIDATION_COUNT,
                "validation_split_sha256": "2" * 64,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": summary.file_sha256(checkpoint),
                "checkpoint_role": "best_validation_pd_primary",
                "checkpoint_epoch": 1,
                "official_test_accessed": False,
                "fixed_threshold_0_5": reported,
                "best_points_under_fa_budget": {
                    budget: reported for budget in summary.EXPECTED_BUDGETS
                },
                "points": [reported, lexicographic_winner],
            }
            sweep_path = root / "pd_fa_sweep_best.pth.json"
            sweep_path.write_text(
                json.dumps(payload) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "budget|lexicographic|optimal",
            ):
                summary.validate_sweep(
                    sweep_path,
                    variant="tpd_clean_full",
                    split_sha256="2" * 64,
                    expected_role="best_validation_pd_primary",
                )


class ParetoAndCoverageTests(unittest.TestCase):
    def test_pareto_merges_owners_at_identical_coordinates(self) -> None:
        zero_fa = point(170, 0.0, threshold=0.4)
        shared_a = point(171, 1e-6, miou=0.81, threshold=0.4)
        shared_b = point(171, 1e-6, miou=0.82, threshold=0.6)
        dominated = point(170, 2e-6, threshold=0.5)
        frontier = summary.pareto_frontier(
            (
                ("spd", zero_fa),
                ("tpd_clean_full", shared_a),
                ("tpd_clean_ctx", shared_b),
                ("tpd_clean_full", shared_a),
                ("tpd_clean_sal", dominated),
            )
        )

        self.assertEqual(
            [(item["pd"], item["fa"]) for item in frontier],
            [(zero_fa["pd"], 0.0), (shared_a["pd"], 1e-6)],
        )
        shared = frontier[1]
        self.assertEqual(
            set(shared["owners"]),
            {"tpd_clean_full", "tpd_clean_ctx"},
        )
        self.assertEqual(shared["owners"].count("tpd_clean_full"), 1)

    def test_frozen_reference_union_covers_crossing_budget_results(self) -> None:
        candidate_budget_points: dict[str, dict] = {}
        spd_budget_points: dict[str, dict] = {}
        tpd_budget_points: dict[str, dict] = {}
        for index, budget in enumerate(summary.EXPECTED_BUDGETS):
            candidate_budget_points[budget] = point(170, 8e-7)
            if index % 2 == 0:
                spd_budget_points[budget] = point(171, 8e-7)
                tpd_budget_points[budget] = point(169, 7e-7)
            else:
                spd_budget_points[budget] = point(169, 7e-7)
                tpd_budget_points[budget] = point(171, 8e-7)

        candidate = run_record(
            "tpd_clean_full",
            candidate_budget_points[summary.EXPECTED_BUDGETS[0]],
            budgets=candidate_budget_points,
            points=list(candidate_budget_points.values()),
        )
        references = {
            "spd": run_record(
                "spd",
                spd_budget_points[summary.EXPECTED_BUDGETS[0]],
                budgets=spd_budget_points,
                points=list(spd_budget_points.values()),
            ),
            "tpd": run_record(
                "tpd",
                tpd_budget_points[summary.EXPECTED_BUDGETS[0]],
                budgets=tpd_budget_points,
                points=list(tpd_budget_points.values()),
            ),
        }

        result = summary.summarize_candidate(candidate, references)

        self.assertEqual(result["candidate_better_than_both_budget_count"], 0)
        self.assertEqual(
            result["reference_union_covers_budget_count"],
            len(summary.EXPECTED_BUDGETS),
        )
        self.assertEqual(result["unique_candidate_pareto_points"], [])
        self.assertEqual(result["evidence_class"], "COVERED_BY_FROZEN_REFERENCES")
        for index, budget in enumerate(summary.EXPECTED_BUDGETS):
            outcomes = {
                name: comparison["outcome"]
                for name, comparison in result["comparisons"]["budgets"][
                    budget
                ].items()
            }
            better_reference = "spd" if index % 2 == 0 else "tpd"
            worse_reference = "tpd" if index % 2 == 0 else "spd"
            self.assertEqual(outcomes[better_reference], "reference_better")
            self.assertEqual(outcomes[worse_reference], "candidate_better")


class JsonAndDecisionBoundaryTests(unittest.TestCase):
    def test_load_json_rejects_duplicate_keys_at_nested_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text(
                '{"outer":{"value":1,"value":2}}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                summary.load_json(path)

    def test_main_output_never_makes_a_mainline_decision(self) -> None:
        reference_point = point(170, 8e-7)
        candidate_point = point(171, 8e-7)

        def fake_validate_run(
            _root: Path,
            _run_name: str,
            variant: str,
            *,
            require_miou_sweep: bool,
        ) -> dict:
            selected = (
                candidate_point
                if variant in summary.CANDIDATE_VARIANTS
                else reference_point
            )
            return run_record(variant, selected)

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            arguments = SimpleNamespace(
                candidate_root=Path("/fixture/candidates"),
                reference_root=Path("/fixture/references"),
                reference_miou_root=Path("/fixture/reference_miou"),
                candidate_run_name="candidate_run",
                reference_run_name="reference_run",
                output_dir=output_dir,
                overwrite=False,
            )
            with (
                mock.patch.object(summary, "parse_args", return_value=arguments),
                mock.patch.object(
                    summary,
                    "verify_frozen_comparison",
                    return_value={
                        "formal_decision": summary.FROZEN_DECISION,
                        "paper_core_established": False,
                        "stability_claim_supported": False,
                    },
                ),
                mock.patch.object(
                    summary,
                    "verify_source_lock",
                    return_value={"sha256": "b" * 64},
                ),
                mock.patch.object(
                    summary,
                    "validate_run",
                    side_effect=fake_validate_run,
                ),
                mock.patch.object(
                    summary,
                    "validate_reference_miou_sweep",
                    side_effect=lambda _root, _name, record: record,
                ),
            ):
                summary.main()

            payload = summary.load_json(
                output_dir / "tpd_clean_screen800_comparison_seed42.json"
            )
            boundary = payload["decision_boundary"]
            self.assertIs(boundary["mainline_decision_made"], False)
            self.assertIs(boundary["paper_core_established"], False)
            self.assertIs(boundary["stability_claim_supported"], False)
            self.assertIs(boundary["mainline_changed"], False)
            self.assertEqual(
                boundary["permitted_action"],
                "nominate_candidate_for_paired_seed_confirmation_only",
            )


if __name__ == "__main__":
    unittest.main()
