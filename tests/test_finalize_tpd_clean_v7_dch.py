from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from experiments import finalize_tpd_clean_v7_dch as finalizer
from experiments import summarize_tpd_clean_v7_dch_formal800 as summary


def point(
    matched: int,
    fa: float,
    miou: float,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    return {
        "matched_target_count": matched,
        "target_count": 189,
        "pd": matched / 189,
        "fa": fa,
        "miou": miou,
        "tiny_pd": 1.0,
        "tiny_target_count": 39,
        "matched_tiny_target_count": 39 if matched else 0,
        "predicted_object_count": matched,
        "unmatched_predicted_object_count": 0,
        "threshold": threshold,
    }


def role(capacity: bool, *, miou_role: bool) -> dict[str, object]:
    fixed = (
        point(187, 0.0, 0.95)
        if miou_role
        else point(188, 1e-6, 0.94)
    )
    budgets = {}
    for budget in summary.BUDGET_KEYS:
        matched = 187 if budget == "1e-06" else 188
        if capacity and budget == "5e-06":
            matched -= 1
        budgets[budget] = point(
            matched,
            0.0 if budget == "1e-06" else 2e-6,
            0.94,
            0.8,
        )
    return {"fixed_threshold_0_5": fixed, "budgets": budgets}


def comparison(*, gate_pass: bool) -> dict:
    runs = {}
    for seed in summary.SEEDS:
        for variant in summary.VARIANTS:
            capacity = variant == summary.CONTROL_VARIANT
            runs[(variant, seed)] = {
                "variant": variant,
                "seed": seed,
                "roles": {
                    "pd_primary": role(capacity, miou_role=False),
                    "miou_primary": role(capacity, miou_role=True),
                },
            }
    spd_budgets = copy.deepcopy(
        runs[(summary.PRIMARY_VARIANT, 42)]["roles"]["pd_primary"][
            "budgets"
        ]
    )
    spd_budgets["5e-06"] = point(187, 0.0, 0.94, 0.8)
    spd = {"roles": {"pd_primary": {"budgets": spd_budgets}}}
    integrity = {key: True for key in summary.INTEGRITY_KEYS}
    if not gate_pass:
        integrity["native_17_fields_complete"] = False
    keyed = {
        f"{variant}/seed_{seed}": record
        for (variant, seed), record in runs.items()
    }
    return summary.build_report_from_components(
        runs=runs,
        keyed_runs=keyed,
        spd_reference=spd,
        engineering_integrity=integrity,
        bindings={},
    )


def mechanism(*, supported: bool) -> dict:
    return {
        "schema": "sctransnet_tpd_clean_v7_dch_mechanism_audit_v1",
        "status": "complete",
        "candidate_family": "tpd_clean_v7_dch",
        "dataset": summary.DATASET,
        "variants": list(summary.VARIANTS),
        "seeds": list(summary.SEEDS),
        "artifact_counts": {"runs": 4, "checkpoints": 12},
        "directions": {
            "fragment_excess_total": {"direction": "lower"},
            "in_gt_unmatched_pixels": {"direction": "lower"},
            "split_target": {"direction": "lower"},
            "largest_fragment": {"direction": "higher"},
        },
        "fragmentation_mechanism_claim_supported": supported,
        "mechanism_audit_replaces_performance_gates": False,
        "native_validation_fields": list(summary.VALIDATION_FIELDS),
        "native_validation_field_count": len(summary.VALIDATION_FIELDS),
    }


class FinalizeTPDCleanV7DCHTests(unittest.TestCase):
    def test_default_acceptance_lock_is_current_v4(self) -> None:
        self.assertEqual(
            finalizer.DEFAULT_ACCEPTANCE_SOURCE_LOCK,
            summary.DEFAULT_ACCEPTANCE_SOURCE_LOCK,
        )
        self.assertEqual(
            finalizer.DEFAULT_ACCEPTANCE_SOURCE_LOCK.name,
            "tpd_clean_v7_dch_acceptance_source_lock_v4.json",
        )

    def test_mechanism_failure_does_not_veto_passed_gates(self) -> None:
        report = finalizer.derive_final_decision(
            comparison(gate_pass=True),
            mechanism(supported=False),
        )
        self.assertEqual(report["decision"], "ENGINEERING_GATE_PASS")
        self.assertTrue(report["ner_stage_authorized"])
        self.assertFalse(
            report["fragmentation_mechanism_claim_supported"]
        )
        self.assertEqual(
            report["mechanism_decision"],
            "MECHANISM_NOT_SUPPORTED",
        )

    def test_mechanism_success_cannot_rescue_failed_gates(self) -> None:
        report = finalizer.derive_final_decision(
            comparison(gate_pass=False),
            mechanism(supported=True),
        )
        self.assertEqual(report["decision"], "ENGINEERING_GATE_FAIL")
        self.assertFalse(report["ner_stage_authorized"])
        self.assertTrue(
            report["fragmentation_mechanism_claim_supported"]
        )
        self.assertFalse(
            report["mechanism_audit_replaces_performance_gates"]
        )

    def test_final_claim_boundary_remains_conservative(self) -> None:
        report = finalizer.derive_final_decision(
            comparison(gate_pass=True),
            mechanism(supported=True),
        )
        self.assertFalse(report["mainline_changed"])
        self.assertFalse(report["paper_core_established"])
        self.assertFalse(report["stability_claim_supported"])
        self.assertTrue(report["authoritative_result_accepted"])
        self.assertEqual(
            report["formal_artifact_counts"],
            {
                "runs": 4,
                "checkpoints": 12,
                "sweeps": 8,
                "mechanism_checkpoint_audits": 12,
            },
        )

    def test_rejects_incomplete_mechanism_artifact_matrix(self) -> None:
        broken = mechanism(supported=True)
        broken["artifact_counts"]["checkpoints"] = 11
        with self.assertRaisesRegex(
            finalizer.FinalizationError,
            "4 runs and 12 checkpoints",
        ):
            finalizer.derive_final_decision(
                comparison(gate_pass=True),
                broken,
            )

    def test_preflight_is_read_only_when_results_do_not_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = list(root.rglob("*"))
            report = finalizer.inspect_readiness(
                comparison_path=root / "comparison.json",
                mechanism_path=root / "mechanism.json",
                acceptance_source_lock=root / "acceptance.json",
            )
            after = list(root.rglob("*"))
        self.assertFalse(report["ready"])
        self.assertEqual(report["writes_performed"], 0)
        self.assertEqual(before, after)

    def test_preflight_rejects_present_v1_acceptance_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comparison_path = root / "comparison.json"
            mechanism_path = root / "mechanism.json"
            v1_lock = root / "acceptance-v1.json"
            comparison_path.write_text("{}\n", encoding="utf-8")
            mechanism_path.write_text("{}\n", encoding="utf-8")
            v1_lock.write_text(
                json.dumps(
                    {
                        "schema": (
                            finalizer.source_locks
                            .ACCEPTANCE_SOURCE_LOCK_SCHEMA_V1
                        )
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = finalizer.inspect_readiness(
                comparison_path=comparison_path,
                mechanism_path=mechanism_path,
                acceptance_source_lock=v1_lock,
            )
        self.assertFalse(report["ready"])
        lock_input = report["inputs"]["acceptance_source_lock"]
        self.assertTrue(lock_input["present"])
        self.assertFalse(
            lock_input["validation"]["valid_current_lock"]
        )
        self.assertEqual(
            lock_input["validation"]["expected_schema"],
            finalizer.source_locks.ACCEPTANCE_SOURCE_LOCK_SCHEMA_V4,
        )
        self.assertIn(
            "acceptance_source_lock_v4",
            lock_input["validation"]["error"],
        )

    def test_writer_refuses_to_replace_final_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / finalizer.JSON_OUTPUT_NAME
            existing.write_bytes(b"unchanged")
            with self.assertRaises(FileExistsError):
                finalizer.write_final_report_once({}, root)
            self.assertEqual(existing.read_bytes(), b"unchanged")
            self.assertFalse(
                (root / finalizer.MARKDOWN_OUTPUT_NAME).exists()
            )


if __name__ == "__main__":
    unittest.main()
