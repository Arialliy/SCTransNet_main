from __future__ import annotations

from dataclasses import replace
import unittest

from experiments.pbdr_v4_zero_margin_selector import (
    EvaluationBinding,
    MetricRecord,
    ZeroMarginSelectionError,
    role_key,
    select_against_baseline_envelope,
    select_best,
    selection_report,
)


_A = "a" * 64
_B = "b" * 64
_C = "c" * 64
_D = "d" * 64


def _binding(role: str = "best_miou", **updates: object) -> EvaluationBinding:
    values: dict[str, object] = {
        "dataset": "NUAA-SIRST",
        "role": role,
        "evaluation_context_sha256": _A,
        "sample_id_order_sha256": _B,
        "target_sha256": _C,
        "metric_core_sha256": _D,
    }
    values.update(updates)
    return EvaluationBinding(**values)  # type: ignore[arg-type]


def _record(
    name: str,
    family: str = "Original",
    role: str = "best_miou",
    **updates: object,
) -> MetricRecord:
    values: dict[str, object] = {
        "name": name,
        "family": family,
        "binding": _binding(role),
        "intersection_pixels": 8000,
        "union_pixels": 10_000,
        "matched_target_count": 90,
        "target_count": 100,
        "unmatched_component_pixels": 20,
        "valid_pixels": 20_000,
        "niou": 0.79,
        "matched_tiny_target_count": 9,
        "tiny_target_count": 10,
        "loss": 0.01,
    }
    values.update(updates)
    return MetricRecord(**values)  # type: ignore[arg-type]


class PBDRV4ZeroMarginSelectorTests(unittest.TestCase):
    def test_any_strict_first_metric_improvement_wins_without_epsilon(self) -> None:
        original = _record("original")
        candidate = _record(
            "candidate",
            "V4-Stage1",
            intersection_pixels=8001,
        )
        self.assertIs(select_best("best_miou", (original, candidate)), candidate)

    def test_exact_tie_uses_frozen_order_not_caller_order(self) -> None:
        original = _record("original")
        current = _record("current", "Current")
        candidate = _record("candidate", "V4-Stage1")
        self.assertIs(
            select_best("best_miou", (candidate, current, original)),
            original,
        )

    def test_candidate_can_beat_original_but_lose_to_current(self) -> None:
        original = _record("original", intersection_pixels=8000)
        current = _record("current", "Current", intersection_pixels=8200)
        candidate = _record("candidate", "V4-Stage1", intersection_pixels=8100)
        winner = select_against_baseline_envelope(
            "best_miou",
            original=original,
            current=current,
            candidates=(candidate,),
        )
        self.assertIs(winner, current)
        report = selection_report(
            "best_miou",
            original=original,
            current=current,
            candidates=(candidate,),
        )
        self.assertEqual(report["winner"], "current")
        self.assertIsNone(report["performance_acceptance_margin"])
        self.assertNotIn("passed_gate", report)

    def test_best_pd_uses_exact_counts_then_exact_fa_fraction(self) -> None:
        original = _record(
            "original",
            role="best_pd",
            matched_target_count=95,
            unmatched_component_pixels=40,
        )
        current = _record(
            "current",
            "Current",
            "best_pd",
            matched_target_count=95,
            unmatched_component_pixels=30,
        )
        candidate = _record(
            "candidate",
            "V4-Stage1",
            "best_pd",
            matched_target_count=95,
            unmatched_component_pixels=29,
            intersection_pixels=1000,
        )
        self.assertGreater(role_key("best_pd", candidate), role_key("best_pd", current))
        self.assertIs(
            select_against_baseline_envelope(
                "best_pd",
                original=original,
                current=current,
                candidates=(candidate,),
            ),
            candidate,
        )

    def test_context_hash_not_only_counts_binds_split(self) -> None:
        original = _record("original")
        different_context = _record(
            "candidate",
            "V4-Stage1",
            binding=_binding(evaluation_context_sha256="e" * 64),
        )
        with self.assertRaisesRegex(ZeroMarginSelectionError, "evaluation context"):
            select_best("best_miou", (original, different_context))

    def test_duplicate_names_or_families_are_rejected(self) -> None:
        original = _record("same")
        with self.assertRaisesRegex(ZeroMarginSelectionError, "names"):
            select_best("best_miou", (original, _record("same", "Current")))
        with self.assertRaisesRegex(ZeroMarginSelectionError, "families"):
            select_best("best_miou", (original, _record("candidate")))

    def test_numeric_consistency_checks_are_fail_closed(self) -> None:
        with self.assertRaisesRegex(ZeroMarginSelectionError, "tiny_target_count"):
            _record("bad", tiny_target_count=101)
        with self.assertRaisesRegex(ZeroMarginSelectionError, "unmatched_component_pixels"):
            _record("bad", unmatched_component_pixels=20_001)
        with self.assertRaisesRegex(ZeroMarginSelectionError, "niou"):
            _record("bad", niou=1.1)
        with self.assertRaisesRegex(ZeroMarginSelectionError, "loss"):
            _record("bad", loss=-0.1)

    def test_mapping_requires_exact_integer_statistics(self) -> None:
        payload = _record("source").as_dict()
        payload.pop("unmatched_component_pixels")
        with self.assertRaisesRegex(ZeroMarginSelectionError, "component pixels"):
            MetricRecord.from_mapping(
                name="candidate",
                family="V4-Stage1",
                binding=_binding(),
                value=payload,
            )
        payload = _record("source").as_dict()
        payload.pop("intersection_pixels")
        with self.assertRaises(KeyError):
            MetricRecord.from_mapping(
                name="candidate",
                family="V4-Stage1",
                binding=_binding(),
                value=payload,
            )

    def test_operational_test_selection_is_marked_optimistic(self) -> None:
        report = selection_report(
            "best_miou",
            original=_record("original"),
            current=_record("current", "Current"),
            candidates=(_record("candidate", "V4-Stage1"),),
            operational_test_selected=True,
        )
        self.assertIs(report["operational_test_selected"], True)
        self.assertIs(report["selection_is_optimistic"], True)


if __name__ == "__main__":
    unittest.main()
