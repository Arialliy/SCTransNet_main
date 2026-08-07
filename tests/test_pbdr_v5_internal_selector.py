from __future__ import annotations

from fractions import Fraction
import unittest

from experiments.pbdr_v5_internal_selector import (
    FROZEN_FAMILY_ORDER,
    InternalSelectionError,
    role_key,
    select_internal_candidate,
)


def _metrics(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "intersection_pixels": 8_000,
        "union_pixels": 10_000,
        "matched_target_count": 90,
        "target_count": 100,
        "unmatched_component_pixels": 20,
        "valid_pixel_count": 20_000,
        "niou": 0.79,
        "matched_tiny_target_count": 9,
        "tiny_target_count": 10,
        "test_loss": 0.01,
    }
    values.update(updates)
    return values


def _pool(**family_updates: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        family: _metrics(**family_updates.get(family, {}))
        for family in reversed(FROZEN_FAMILY_ORDER)
    }


class PBDRV5InternalSelectorTests(unittest.TestCase):
    def test_any_strict_v5_improvement_wins_without_margin(self) -> None:
        report = select_internal_candidate(
            "best_miou",
            _pool(V5={"intersection_pixels": 8_001}),
        )
        self.assertEqual(report["winner"], "V5")
        self.assertIs(report["v5_strictly_improves_existing_envelope"], True)
        self.assertIsNone(report["performance_acceptance_margin"])
        self.assertEqual(report["candidate_families"], list(FROZEN_FAMILY_ORDER))

    def test_complete_tie_keeps_frozen_first_family_not_mapping_order(self) -> None:
        report = select_internal_candidate("best_miou", _pool())
        self.assertEqual(report["winner"], "Original")
        self.assertEqual(report["existing_envelope_winner"], "Original")
        self.assertIs(report["v5_strictly_improves_existing_envelope"], False)
        self.assertEqual(report["exact_tie_order"], list(FROZEN_FAMILY_ORDER))

    def test_v5_may_improve_baselines_but_not_existing_five_family_envelope(self) -> None:
        report = select_internal_candidate(
            "best_miou",
            _pool(
                Current={"intersection_pixels": 8_100},
                **{
                    "V4-Stage2": {"intersection_pixels": 8_300},
                    "V5": {"intersection_pixels": 8_200},
                },
            ),
        )
        self.assertEqual(report["existing_envelope_winner"], "V4-Stage2")
        self.assertEqual(report["winner"], "V4-Stage2")
        self.assertIs(report["v5_strictly_improves_existing_envelope"], False)

    def test_best_pd_uses_pd_then_exact_fa_before_miou(self) -> None:
        report = select_internal_candidate(
            "best_pd",
            _pool(
                Original={"matched_target_count": 95, "unmatched_component_pixels": 20},
                Current={"matched_target_count": 95, "unmatched_component_pixels": 19},
                V5={
                    "matched_target_count": 96,
                    "unmatched_component_pixels": 100,
                    "intersection_pixels": 1_000,
                },
            ),
        )
        self.assertEqual(report["winner"], "V5")
        self.assertIs(report["v5_strictly_improves_existing_envelope"], True)

        current = _metrics(
            matched_target_count=95,
            unmatched_component_pixels=19,
        )
        lower_fa = _metrics(
            matched_target_count=95,
            unmatched_component_pixels=18,
            intersection_pixels=1_000,
        )
        self.assertGreater(role_key("best_pd", lower_fa), role_key("best_pd", current))

    def test_exact_sufficient_statistics_and_fraction_deltas_are_reported(self) -> None:
        report = select_internal_candidate(
            "best_miou",
            _pool(
                V5={
                    "intersection_pixels": 8_001,
                    "unmatched_component_pixels": 19,
                }
            ),
        )
        comparison = report["v5_vs_existing_envelope_winner"]
        assert isinstance(comparison, dict)
        stats = comparison["exact_sufficient_statistics_delta"]
        assert isinstance(stats, dict)
        self.assertEqual(stats["intersection_pixels"], 1)
        self.assertEqual(stats["union_pixels"], 0)
        self.assertEqual(stats["unmatched_component_pixels"], -1)
        self.assertEqual(stats["target_count"], 0)

        metric_delta = comparison["exact_role_metric_delta"]
        assert isinstance(metric_delta, dict)
        self.assertEqual(
            metric_delta["miou"],
            {"numerator": Fraction(1, 10_000).numerator, "denominator": 10_000},
        )
        self.assertEqual(
            metric_delta["fa"],
            {"numerator": -1, "denominator": 20_000},
        )

    def test_rounded_summary_fields_cannot_override_exact_counts(self) -> None:
        pool = _pool(V5={"intersection_pixels": 8_001})
        pool["Original"]["miou"] = 1.0
        pool["V5"]["miou"] = 0.0
        report = select_internal_candidate("best_miou", pool)
        self.assertEqual(report["winner"], "V5")

    def test_full_role_key_reaches_late_fields(self) -> None:
        report = select_internal_candidate(
            "best_miou",
            _pool(V5={"test_loss": 0.009}),
        )
        self.assertEqual(report["winner"], "V5")
        self.assertIs(report["v5_strictly_improves_existing_envelope"], True)

    def test_family_pool_and_shared_denominators_fail_closed(self) -> None:
        missing = _pool()
        missing.pop("V5")
        with self.assertRaisesRegex(InternalSelectionError, "frozen pool"):
            select_internal_candidate("best_miou", missing)

        extra = _pool()
        extra["Other"] = _metrics()
        with self.assertRaisesRegex(InternalSelectionError, "extra"):
            select_internal_candidate("best_miou", extra)

        inconsistent = _pool(V5={"target_count": 101})
        with self.assertRaisesRegex(InternalSelectionError, "share target"):
            select_internal_candidate("best_miou", inconsistent)

    def test_invalid_exact_statistics_and_nonfinite_metrics_fail_closed(self) -> None:
        with self.assertRaisesRegex(InternalSelectionError, "intersection_pixels"):
            select_internal_candidate(
                "best_miou",
                _pool(V5={"intersection_pixels": 10_001}),
            )
        with self.assertRaisesRegex(InternalSelectionError, "finite"):
            select_internal_candidate("best_miou", _pool(V5={"niou": float("nan")}))
        with self.assertRaisesRegex(InternalSelectionError, "unsupported role"):
            select_internal_candidate("unknown", _pool())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
