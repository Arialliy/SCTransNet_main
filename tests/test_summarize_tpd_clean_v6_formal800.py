from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from experiments import summarize_tpd_clean_v6_formal800 as subject


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
            pd_fixed = point(188, 1e-6, 0.94)
            miou_fixed = point(187, 0.0, 0.95)
            runs[(variant, seed)] = {
                "roles": {
                    "pd_primary": role(pd_fixed, capacity=capacity),
                    "miou_primary": role(miou_fixed, capacity=capacity),
                }
            }
    spd_budgets = copy.deepcopy(
        runs[(subject.PRIMARY_VARIANT, 42)]["roles"]["pd_primary"]["budgets"]
    )
    spd_budgets["5e-06"] = point(187, 0.0, 0.94, 0.8)
    spd = {"roles": {"pd_primary": {"budgets": spd_budgets}}}
    integrity = {key: True for key in subject.INTEGRITY_KEYS}
    return runs, spd, integrity


class V6Formal800GateTests(unittest.TestCase):
    def test_closed_interval_accepts_float64_quantiles_above_nextafter(self) -> None:
        values = [
            point(188, 1e-6, 0.94, 0.5),
            point(187, 0.0, 0.93, subject.LAST_FLOAT32_BELOW_ONE),
            point(
                187,
                0.0,
                0.93,
                (subject.LAST_FLOAT32_BELOW_ONE + 1.0) / 2.0,
            ),
            point(0, 0.0, 0.0, 1.0),
        ]
        provenance = {
            "posthoc_endpoint_completion": False,
            "preregistered_endpoint_completion": True,
            "endpoint_protocol_stage": "before_formal_training",
            "closed_probability_interval": True,
            "score_dtype": "float32",
            "last_float32_below_one": subject.LAST_FLOAT32_BELOW_ONE,
            "upper_boundary_threshold": 1.0,
            "upper_boundary_comparison": "prediction > threshold",
            "upper_boundary_semantics": "empty_prediction_pd0_fa0",
        }
        subject._validate_closed_interval(values, provenance)

    def test_closed_interval_rejects_nonempty_threshold_one(self) -> None:
        values = [
            point(188, 1e-6, 0.94, 0.5),
            point(187, 0.0, 0.93, subject.LAST_FLOAT32_BELOW_ONE),
            point(1, 0.0, 0.1, 1.0),
        ]
        provenance = {
            "posthoc_endpoint_completion": False,
            "preregistered_endpoint_completion": True,
            "endpoint_protocol_stage": "before_formal_training",
            "closed_probability_interval": True,
            "score_dtype": "float32",
            "last_float32_below_one": subject.LAST_FLOAT32_BELOW_ONE,
            "upper_boundary_threshold": 1.0,
            "upper_boundary_comparison": "prediction > threshold",
            "upper_boundary_semantics": "empty_prediction_pd0_fa0",
        }
        with self.assertRaises(subject.IncompleteArtifact):
            subject._validate_closed_interval(values, provenance)

    def test_all_five_gates_pass_on_a_complete_qualifying_fixture(self) -> None:
        runs, spd, integrity = passing_inputs()
        result = subject.evaluate_engineering_gates(runs, spd, integrity)
        self.assertTrue(result["passed"])
        self.assertTrue(
            all(item["passed"] for item in result["checks"].values())
        )
        self.assertTrue(
            result["checks"]["gate_d_full_vs_capacity"]["per_seed"]["42"][
                "subchecks"
            ]["full_advantage_not_only_threshold_1_empty_endpoint"]
        )

    def test_gate_a_threshold_is_exactly_the_frozen_protocol_value(self) -> None:
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

    def test_gate_b_uses_weak_spd_coverage_and_all_five_budgets(self) -> None:
        runs, spd, integrity = passing_inputs()
        spd["roles"]["pd_primary"]["budgets"] = copy.deepcopy(
            runs[(subject.PRIMARY_VARIANT, 42)]["roles"]["pd_primary"][
                "budgets"
            ]
        )
        result = subject.evaluate_engineering_gates(runs, spd, integrity)
        gate = result["checks"]["gate_b_seed42_budget_and_spd"]
        self.assertFalse(gate["passed"])
        self.assertFalse(
            gate["subchecks"][
                "at_least_one_budget_not_covered_by_frozen_spd"
            ]
        )

    def test_gate_c_requires_four_seed3407_budgets_within_one_target(self) -> None:
        runs, spd, integrity = passing_inputs()
        budgets = runs[(subject.PRIMARY_VARIANT, 3407)]["roles"]["pd_primary"][
            "budgets"
        ]
        for key in ("1e-06", "5e-06"):
            budgets[key]["matched_target_count"] -= 2
            budgets[key]["pd"] = (
                budgets[key]["matched_target_count"]
                / subject.EXPECTED_TARGET_COUNT
            )
        result = subject.evaluate_engineering_gates(runs, spd, integrity)
        gate = result["checks"]["gate_c_seed3407_stability"]
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["budget_stability_pass_count"], 3)

    def test_gate_d_rejects_any_capacity_strict_coverage(self) -> None:
        runs, spd, integrity = passing_inputs()
        capacity = runs[(subject.CONTROL_VARIANT, 42)]["roles"]["pd_primary"][
            "fixed_threshold_0_5"
        ]
        capacity["fa"] = 0.0
        result = subject.evaluate_engineering_gates(runs, spd, integrity)
        seed = result["checks"]["gate_d_full_vs_capacity"]["per_seed"]["42"]
        self.assertFalse(seed["passed"])
        self.assertIn(
            "pd_primary.fixed_threshold_0_5",
            seed["capacity_strict_coverage_points"],
        )

    def test_gate_d_rejects_advantage_only_at_threshold_one(self) -> None:
        runs, spd, integrity = passing_inputs()
        for seed in subject.SEEDS:
            for role_name in subject.ROLE_SPECS:
                full = runs[(subject.PRIMARY_VARIANT, seed)]["roles"][role_name]
                capacity = runs[(subject.CONTROL_VARIANT, seed)]["roles"][
                    role_name
                ]
                for key in subject.BUDGET_KEYS:
                    shared = point(0, 0.0, 0.0, 1.0)
                    full["budgets"][key] = copy.deepcopy(shared)
                    capacity["budgets"][key] = copy.deepcopy(shared)
                full["budgets"]["5e-06"]["miou"] = 0.1
        result = subject.evaluate_engineering_gates(runs, spd, integrity)
        for seed in subject.SEEDS:
            record = result["checks"]["gate_d_full_vs_capacity"]["per_seed"][
                str(seed)
            ]
            self.assertFalse(record["passed"])
            self.assertTrue(record["full_strict_budget_advantages"])
            self.assertFalse(record["nonempty_full_strict_budget_advantages"])

    def test_gate_e_fails_when_any_integrity_item_is_false(self) -> None:
        runs, spd, integrity = passing_inputs()
        integrity["exact_epoch_journals_complete"] = False
        result = subject.evaluate_engineering_gates(runs, spd, integrity)
        gate = result["checks"]["gate_e_engineering_integrity"]
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["subchecks"]["exact_epoch_journals_complete"])

    def test_markdown_contains_40_budget_rows_and_24_gate_d_rows(self) -> None:
        runs, spd, integrity = passing_inputs()
        gates = subject.evaluate_engineering_gates(runs, spd, integrity)
        candidate_runs = {}
        for (variant, seed), run in runs.items():
            rendered_run = copy.deepcopy(run)
            rendered_run["artifact_sha256"] = {"protocol.json": "0" * 64}
            for role_record in rendered_run["roles"].values():
                role_record["sweep_sha256"] = "1" * 64
                role_record["checkpoint_sha256"] = "2" * 64
            candidate_runs[f"{variant}/seed_{seed}"] = rendered_run
        report = {
            "decision": "ENGINEERING_GATE_PASS",
            "engineering_gate_passed": True,
            "ner_stage_authorized": True,
            "candidate_runs": candidate_runs,
            "engineering_gate": gates,
            "training_source_lock_sha256": "3" * 64,
            "training_data_sha256": "4" * 64,
            "postprocess_source_lock_sha256": "5" * 64,
            "evaluator_sha256": "6" * 64,
            "frozen_spd_reference": {"sweep_sha256": "7" * 64},
            "smoke_validation": {
                "persisted_verification": {"sha256": "8" * 64}
            },
        }
        markdown = subject.render_markdown(report)
        budget = markdown.split(
            "## Registered Fa-budget operating points (40)", 1
        )[1].split("## Gate A", 1)[0]
        gate_d = markdown.split(
            "## Gate D — Full versus capacity (24 comparisons)", 1
        )[1].split("## Gate E", 1)[0]
        budget_rows = [
            line for line in budget.splitlines() if line.startswith("| ")
        ]
        gate_d_rows = [
            line for line in gate_d.splitlines() if line.startswith("| ")
        ]
        self.assertEqual(len(budget_rows) - 2, 40)
        self.assertEqual(len(gate_d_rows) - 2, 24)
        self.assertIn("## SHA-256 bindings", markdown)

    def test_live_preflight_is_read_only_and_never_evaluates_gates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = list(root.rglob("*"))
            result = subject.inspect_training_readiness(root)
            after = list(root.rglob("*"))
            self.assertEqual(result["mode"], "preflight")
            self.assertFalse(result["gate_evaluated"])
            self.assertIsNone(result["engineering_gate_passed"])
            self.assertFalse(result["formal_matrix_complete"])
            self.assertEqual(before, after)

    def test_formal_report_writer_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / subject.JSON_OUTPUT_NAME
            existing.write_bytes(b"unchanged")
            with self.assertRaises(FileExistsError):
                subject.write_report_once({}, root)
            self.assertEqual(existing.read_bytes(), b"unchanged")
            self.assertFalse((root / subject.MARKDOWN_OUTPUT_NAME).exists())

    def test_smoke_rederivation_binds_the_persisted_timestamp(self) -> None:
        current_a = {
            "status": "complete",
            "passed": True,
            "verified_at_utc": "later-a",
        }
        current_b = {
            "status": "complete",
            "passed": True,
            "verified_at_utc": "later-b",
        }
        persisted = {
            "status": "complete",
            "passed": True,
            "verified_at_utc": "persisted",
        }
        normalized_a = subject._bind_persisted_smoke_verification(
            current_a, persisted
        )
        normalized_b = subject._bind_persisted_smoke_verification(
            current_b, persisted
        )
        self.assertEqual(normalized_a, normalized_b)
        self.assertEqual(normalized_a["verified_at_utc"], "persisted")
        tampered = dict(current_a, passed=False)
        with self.assertRaises(subject.IncompleteArtifact):
            subject._bind_persisted_smoke_verification(tampered, persisted)


if __name__ == "__main__":
    unittest.main()
