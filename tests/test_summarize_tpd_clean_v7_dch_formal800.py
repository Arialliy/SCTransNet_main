from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments import summarize_tpd_clean_v7_dch_formal800 as subject


def point(
    matched: int,
    fa: float,
    miou: float,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    return {
        "matched_target_count": matched,
        "target_count": subject.EXPECTED_TARGET_COUNT,
        "pd": matched / subject.EXPECTED_TARGET_COUNT,
        "fa": fa,
        "miou": miou,
        "tiny_pd": 1.0,
        "tiny_target_count": 39,
        "matched_tiny_target_count": 39 if matched else 0,
        "predicted_object_count": matched,
        "unmatched_predicted_object_count": 0,
        "threshold": threshold,
    }


def role(
    fixed: dict[str, float | int],
    *,
    capacity: bool = False,
) -> dict[str, object]:
    budgets: dict[str, dict[str, float | int]] = {}
    for budget in subject.BUDGET_KEYS:
        matched = 187 if budget == "1e-06" else 188
        fa = 0.0 if budget == "1e-06" else min(float(budget), 2e-6)
        if capacity and budget == "5e-06":
            matched -= 1
        budgets[budget] = point(matched, fa, 0.94, 0.8)
    return {"fixed_threshold_0_5": fixed, "budgets": budgets}


def passing_inputs() -> tuple[dict, dict, dict]:
    runs = {}
    for seed in subject.SEEDS:
        for variant in subject.VARIANTS:
            capacity = variant == subject.CONTROL_VARIANT
            runs[(variant, seed)] = {
                "variant": variant,
                "seed": seed,
                "roles": {
                    "pd_primary": role(
                        point(188, 1e-6, 0.94),
                        capacity=capacity,
                    ),
                    "miou_primary": role(
                        point(187, 0.0, 0.95),
                        capacity=capacity,
                    ),
                },
            }
    spd_budgets = copy.deepcopy(
        runs[(subject.PRIMARY_VARIANT, 42)]["roles"]["pd_primary"][
            "budgets"
        ]
    )
    spd_budgets["5e-06"] = point(187, 0.0, 0.94, 0.8)
    spd = {"roles": {"pd_primary": {"budgets": spd_budgets}}}
    integrity = {key: True for key in subject.INTEGRITY_KEYS}
    return runs, spd, integrity


class DCHFormal800SummaryTests(unittest.TestCase):
    def test_default_acceptance_lock_is_current_v3(self) -> None:
        self.assertEqual(
            subject.DEFAULT_ACCEPTANCE_SOURCE_LOCK,
            (
                subject.REPO_ROOT
                / "experiments/"
                "tpd_clean_v7_dch_acceptance_source_lock_v3.json"
            ),
        )
        self.assertNotEqual(
            subject.DEFAULT_ACCEPTANCE_SOURCE_LOCK,
            (
                subject.REPO_ROOT
                / "experiments/tpd_clean_v7_dch_acceptance_source_lock.json"
            ),
        )

    def test_identity_and_artifact_matrix_are_dch_only(self) -> None:
        self.assertEqual(
            subject.VARIANTS,
            (
                "tpd_clean_v7_dch_full",
                "tpd_clean_v7_dch_capacity",
            ),
        )
        self.assertEqual(subject.SEEDS, (42, 3407))
        self.assertEqual(subject.EXPECTED_RUNS, 4)
        self.assertEqual(subject.EXPECTED_CHECKPOINTS, 12)
        self.assertEqual(subject.EXPECTED_SWEEPS, 8)
        self.assertEqual(len(subject.VALIDATION_FIELDS), 17)

    def test_all_five_unchanged_gates_pass_a_qualifying_fixture(self) -> None:
        runs, spd, integrity = passing_inputs()
        result = subject.evaluate_engineering_gates(runs, spd, integrity)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["protocol"],
            "experiments/TPD_CLEAN_V7_DCH_PROTOCOL.md section 7",
        )
        self.assertTrue(result["thresholds_inherited_without_change"])
        self.assertTrue(
            all(item["passed"] for item in result["checks"].values())
        )

    def test_gate_a_threshold_is_not_relaxed(self) -> None:
        runs, spd, integrity = passing_inputs()
        runs[(subject.PRIMARY_VARIANT, 42)]["roles"]["miou_primary"][
            "fixed_threshold_0_5"
        ]["miou"] = 0.9465419
        result = subject.evaluate_engineering_gates(runs, spd, integrity)
        gate = result["checks"]["gate_a_seed42_fixed_threshold"]
        self.assertFalse(gate["passed"])
        self.assertFalse(
            gate["subchecks"]["miou_primary_miou_at_least_0_946542"]
        )

    def test_gate_d_rejects_any_capacity_strict_coverage(self) -> None:
        runs, spd, integrity = passing_inputs()
        capacity = runs[(subject.CONTROL_VARIANT, 42)]["roles"][
            "pd_primary"
        ]["fixed_threshold_0_5"]
        capacity["fa"] = 0.0
        result = subject.evaluate_engineering_gates(runs, spd, integrity)
        seed = result["checks"]["gate_d_full_vs_capacity"]["per_seed"]["42"]
        self.assertFalse(seed["passed"])
        self.assertIn(
            "pd_primary.fixed_threshold_0_5",
            seed["capacity_strict_coverage_points"],
        )

    def test_native_17_fields_are_an_independent_gate_e_requirement(
        self,
    ) -> None:
        runs, spd, integrity = passing_inputs()
        integrity["native_17_fields_complete"] = False
        result = subject.evaluate_engineering_gates(runs, spd, integrity)
        self.assertFalse(result["passed"])
        gate_e = result["checks"]["gate_e_engineering_integrity"]
        self.assertFalse(gate_e["passed"])
        self.assertFalse(
            gate_e["subchecks"]["native_17_fields_complete"]
        )

    def test_report_keeps_claim_boundaries_and_counts(self) -> None:
        runs, spd, integrity = passing_inputs()
        keyed = {
            f"{variant}/seed_{seed}": record
            for (variant, seed), record in runs.items()
        }
        report = subject.build_report_from_components(
            runs=runs,
            keyed_runs=keyed,
            spd_reference=spd,
            engineering_integrity=integrity,
            bindings={"acceptance_source_lock_sha256": "a" * 64},
        )
        self.assertEqual(report["decision"], "ENGINEERING_GATE_PASS")
        self.assertTrue(report["ner_stage_authorized"])
        self.assertFalse(report["mainline_changed"])
        self.assertFalse(report["paper_core_established"])
        self.assertFalse(report["stability_claim_supported"])
        self.assertIsNone(
            report["fragmentation_mechanism_claim_supported"]
        )
        self.assertEqual(
            report["formal_artifact_counts"],
            {"runs": 4, "checkpoints": 12, "sweeps": 8},
        )

    def test_markdown_contains_40_budget_and_24_gate_d_rows(self) -> None:
        runs, spd, integrity = passing_inputs()
        keyed = {
            f"{variant}/seed_{seed}": record
            for (variant, seed), record in runs.items()
        }
        report = subject.build_report_from_components(
            runs=runs,
            keyed_runs=keyed,
            spd_reference=spd,
            engineering_integrity=integrity,
            bindings={},
        )
        markdown = subject.render_markdown(report)
        budget = markdown.split(
            "## Registered Fa-budget operating points (40)",
            1,
        )[1].split("## Gate A", 1)[0]
        gate_d = markdown.split(
            "## Gate D — Full versus Capacity (24 comparisons)",
            1,
        )[1].split("## Claim boundary", 1)[0]
        budget_rows = [
            line for line in budget.splitlines() if line.startswith("| ")
        ]
        gate_d_rows = [
            line for line in gate_d.splitlines() if line.startswith("| ")
        ]
        self.assertEqual(len(budget_rows) - 2, 40)
        self.assertEqual(len(gate_d_rows) - 2, 24)
        serialized = json.dumps(report, sort_keys=True)
        round_tripped = json.loads(serialized)
        self.assertEqual(
            subject.render_markdown(round_tripped),
            markdown,
        )

    def test_writer_refuses_to_replace_either_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / subject.JSON_OUTPUT_NAME
            existing.write_bytes(b"unchanged")
            with self.assertRaises(FileExistsError):
                subject.write_report_once({}, root)
            self.assertEqual(existing.read_bytes(), b"unchanged")
            self.assertFalse((root / subject.MARKDOWN_OUTPUT_NAME).exists())

    def test_preflight_requires_training_and_all_eight_sweeps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = list(root.rglob("*"))
            readiness = subject.inspect_training_readiness(root)
            after = list(root.rglob("*"))
        self.assertFalse(readiness["formal_matrix_complete"])
        self.assertFalse(readiness["gate_evaluated"])
        self.assertIsNone(readiness["engineering_gate_passed"])
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
