import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import (
    repair_postprocess_tpd_ner_v8_mprs_dch_v3_dc_knockout_aggregate_field_v1
    as subject,
)


class DCKnockoutAggregateFieldRepairV1Tests(unittest.TestCase):
    def test_frozen_bug_is_reproduced_and_repaired_mapping_is_exact(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"achieved_fa must be finite",
        ):
            subject.frozen.load_formal_reference_rows()

        formal = json.loads(
            subject.FORMAL_REPAIRED_AGGREGATE.read_text(encoding="utf-8")
        )
        references = subject.load_formal_reference_rows_repaired()
        formal_rows = {
            row["checkpoint_role"]: row
            for row in formal["rows"]
            if row.get("variant") == subject.spec.VARIANT
        }
        self.assertEqual(
            set(references),
            set(subject.spec.CHECKPOINT_ROLES.values()),
        )
        for role, reference in references.items():
            for key in subject.spec.BUDGET_KEYS:
                canonical = formal_rows[role]["pd_at_fa_budget"][key]
                normalized = reference["pd_at_fa_budget"][key]
                self.assertNotIn("achieved_fa", canonical)
                self.assertEqual(normalized["achieved_fa"], canonical["fa"])
                for field in (
                    "pd",
                    "threshold",
                    "matched_target_count",
                    "target_count",
                ):
                    self.assertEqual(normalized[field], canonical[field])

    def test_attestation_binds_exact_minimal_closure(self) -> None:
        result = subject.verify_repair_attestation()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(
            set(result["bindings"]),
            set(subject.required_attestation_paths()),
        )
        self.assertEqual(
            result["bindings"]["artifact:v2_source_lock"]["sha256"],
            subject.EXPECTED_V2_SOURCE_LOCK_SHA256,
        )
        self.assertEqual(
            result["bindings"]["artifact:v2_sweep_best"]["sha256"],
            subject.EXPECTED_SWEEP_SHA256["best.pth.tar"],
        )
        self.assertEqual(
            result["bindings"]["artifact:v2_sweep_best_miou"]["sha256"],
            subject.EXPECTED_SWEEP_SHA256["best_miou.pth.tar"],
        )

    def test_attestation_tamper_and_extra_binding_are_rejected(self) -> None:
        payload = json.loads(subject.ATTESTATION.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attestation.json"
            changed = copy.deepcopy(payload)
            changed["bindings"]["artifact:v2_sweep_best"]["sha256"] = "0" * 64
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError,
                "artifact:v2_sweep_best SHA-256 differs",
            ):
                subject.verify_repair_attestation(path)

            changed = copy.deepcopy(payload)
            changed["bindings"]["extra"] = {
                "path": "experiments/extra",
                "sha256": "0" * 64,
            }
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "binding matrix differs"):
                subject.verify_repair_attestation(path)

    def test_in_memory_builder_reuses_frozen_validator_and_builder(
        self,
    ) -> None:
        attestation = subject.verify_repair_attestation()
        before = {"formal_artifact_binding": {"fixture": True}}
        source_binding = {"fixture": "source-binding"}
        reference = {
            role: {"fixture": role}
            for role in subject.spec.CHECKPOINT_ROLES.values()
        }
        rows_by_checkpoint = {
            checkpoint: [
                {
                    "checkpoint": checkpoint,
                    "knockout_mode": mode,
                }
                for mode in subject.spec.KNOCKOUT_MODES
            ]
            for checkpoint in subject.spec.CHECKPOINTS
        }
        base_report = {
            "row_count": subject.spec.EXPECTED_ROW_COUNT,
            "rows": [
                row
                for checkpoint in subject.spec.CHECKPOINTS
                for row in rows_by_checkpoint[checkpoint]
            ],
            "source_binding": {},
        }

        with (
            mock.patch.object(
                subject,
                "verify_repair_attestation",
                return_value=attestation,
            ),
            mock.patch.object(
                subject.freezer,
                "verify_source_lock",
                side_effect=[before, before],
            ) as verify_lock,
            mock.patch.object(
                subject.freezer,
                "current_source_binding",
                return_value=source_binding,
            ),
            mock.patch.object(
                subject.frozen,
                "_validate_repaired_formal_report_input",
                return_value=subject.FORMAL_REPAIRED_AGGREGATE,
            ),
            mock.patch.object(
                subject.frozen,
                "validate_checkpoint_sweep",
                side_effect=[
                    rows_by_checkpoint[checkpoint]
                    for checkpoint in subject.spec.CHECKPOINTS
                ],
            ) as validate_sweep,
            mock.patch.object(
                subject,
                "load_formal_reference_rows_repaired",
                return_value=reference,
            ),
            mock.patch.object(
                subject.frozen,
                "build_report",
                return_value=copy.deepcopy(base_report),
            ) as build_report,
        ):
            report = subject.build_repaired_report_in_memory()

        self.assertEqual(verify_lock.call_count, 2)
        self.assertEqual(
            validate_sweep.call_count,
            len(subject.spec.CHECKPOINTS),
        )
        build_report.assert_called_once()
        self.assertIn("aggregate_field_repair", report)
        bindings = report["aggregate_field_repair"]["bindings"]
        self.assertEqual(
            bindings["v2_diagnostic_source_lock"]["sha256"],
            subject.EXPECTED_V2_SOURCE_LOCK_SHA256,
        )
        self.assertEqual(
            bindings["formal_repaired_aggregate"]["sha256"],
            subject.EXPECTED_FORMAL_REPAIRED_AGGREGATE_SHA256,
        )
        self.assertEqual(
            set(bindings["v2_knockout_sweeps"]),
            set(subject.spec.CHECKPOINTS),
        )
        for key in (
            "repair_wrapper",
            "repair_protocol",
            "repair_attestation",
        ):
            self.assertEqual(set(bindings[key]), {"path", "sha256"})

    def test_markdown_uses_frozen_renderer(self) -> None:
        with mock.patch.object(
            subject.frozen,
            "render_markdown",
            return_value="FROZEN TABLE\n",
        ) as renderer:
            result = subject.render_repaired_markdown({"rows": []})
        renderer.assert_called_once_with({"rows": []})
        self.assertTrue(result.startswith("FROZEN TABLE"))
        self.assertIn('point["fa"] -> achieved_fa', result)

    def test_publication_is_versioned_and_never_original_comparison(
        self,
    ) -> None:
        paths = subject._output_paths()
        self.assertEqual(paths[0].parent, subject.REPAIR_COMPARISON_DIR)
        self.assertNotEqual(
            subject.REPAIR_COMPARISON_DIR.resolve(),
            subject.spec.DEFAULT_COMPARISON_DIR.resolve(),
        )
        plan = subject.execution_plan()
        self.assertFalse(plan["gpu_work"])
        self.assertEqual(plan["new_evaluation_count"], 0)
        self.assertFalse(plan["original_comparison_is_publication_target"])
        self.assertEqual(
            plan["reused_frozen_functions"],
            [
                "validate_checkpoint_sweep",
                "build_report",
                "render_markdown",
            ],
        )

    def test_aggregate_wrapper_delegates_to_in_memory_build_and_writer(
        self,
    ) -> None:
        report = {"row_count": subject.spec.EXPECTED_ROW_COUNT}
        paths = (
            subject.REPAIR_JSON_OUTPUT,
            subject.REPAIR_MARKDOWN_OUTPUT,
            subject.REPAIR_COMPLETE_MARKER,
        )
        with (
            mock.patch.object(
                subject,
                "build_repaired_report_in_memory",
                return_value=report,
            ) as builder,
            mock.patch.object(
                subject,
                "write_repaired_report",
                return_value=paths,
            ) as writer,
            mock.patch.object(
                subject,
                "inspect_complete",
                return_value={"repair_id": subject.REPAIR_ID},
            ) as inspector,
        ):
            observed_report, observed_paths = subject.aggregate_and_write()
        builder.assert_called_once_with()
        writer.assert_called_once_with(report)
        inspector.assert_called_once_with()
        self.assertIs(observed_report, report)
        self.assertEqual(observed_paths, paths)

    def test_inspect_complete_is_strict_and_rejects_decision_fields(
        self,
    ) -> None:
        attestation = subject.verify_repair_attestation()
        bindings = subject._repair_bindings(attestation)
        report = {
            "schema": subject.frozen.SCHEMA,
            "status": "complete",
            "artifact_kind": subject.spec.ARTIFACT_KIND,
            "diagnostic_only": True,
            "affects_formal_gate": False,
            "formal_decision_authority": False,
            "row_count": subject.spec.EXPECTED_ROW_COUNT,
            "rows": [{} for _ in range(subject.spec.EXPECTED_ROW_COUNT)],
            "source_binding": {
                "aggregate_field_repair": bindings,
            },
            "aggregate_field_repair": {
                "schema": subject.REPAIR_SCHEMA,
                "repair_id": subject.REPAIR_ID,
                "logic_override": subject.ATTESTATION_POLICY[
                    "logic_override"
                ],
                "mapping": {
                    "source_container": "formal_row.pd_at_fa_budget",
                    "source_field": "fa",
                    "normalized_field": "achieved_fa",
                },
                "reused_frozen_functions": subject.ATTESTATION_POLICY[
                    "reused_frozen_functions"
                ],
                "frozen_artifacts_modified": False,
                "new_evaluation_count": 0,
                "gpu_work": False,
                "bindings": bindings,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = (
                root / subject.REPAIR_JSON_OUTPUT.name,
                root / subject.REPAIR_MARKDOWN_OUTPUT.name,
                root / subject.REPAIR_COMPLETE_MARKER.name,
            )
            markdown_bytes = b"fixture markdown\n"
            json_bytes = subject._json_bytes(report)
            marker = subject._marker_payload(
                report,
                json_bytes,
                markdown_bytes,
            )
            paths[0].write_bytes(json_bytes)
            paths[1].write_bytes(markdown_bytes)
            paths[2].write_bytes(subject._json_bytes(marker))
            with (
                mock.patch.object(subject, "_output_paths", return_value=paths),
                mock.patch.object(
                    subject,
                    "render_repaired_markdown",
                    return_value=markdown_bytes.decode("utf-8"),
                ),
            ):
                self.assertEqual(
                    subject.inspect_complete()["repair_id"],
                    subject.REPAIR_ID,
                )

                extra_marker = copy.deepcopy(marker)
                extra_marker["extra"] = True
                paths[2].write_bytes(subject._json_bytes(extra_marker))
                with self.assertRaisesRegex(ValueError, "field set differs"):
                    subject.inspect_complete()

                changed_report = copy.deepcopy(report)
                changed_report["decision"] = "not-allowed"
                changed_json = subject._json_bytes(changed_report)
                changed_marker = subject._marker_payload(
                    changed_report,
                    changed_json,
                    markdown_bytes,
                )
                paths[0].write_bytes(changed_json)
                paths[2].write_bytes(subject._json_bytes(changed_marker))
                with self.assertRaisesRegex(
                    ValueError,
                    "contains a formal decision field",
                ):
                    subject.inspect_complete()

    def test_marker_explicitly_binds_all_required_artifacts(self) -> None:
        attestation = subject.verify_repair_attestation()
        bindings = subject._repair_bindings(attestation)
        report = {
            "aggregate_field_repair": {
                "bindings": bindings,
            }
        }
        marker = subject._marker_payload(report, b"json", b"markdown")
        self.assertEqual(marker["bindings"], bindings)
        self.assertEqual(
            set(bindings),
            {
                "v2_diagnostic_source_lock",
                "v2_knockout_sweeps",
                "formal_repaired_aggregate",
                "frozen_v2_postprocessor",
                "repair_wrapper",
                "repair_protocol",
                "repair_attestation",
            },
        )
        self.assertFalse(marker["affects_formal_gate"])
        self.assertFalse(marker["formal_decision_authority"])

    def test_cli_requires_one_mode(self) -> None:
        for option in ("--verify-only", "--plan", "--aggregate-only"):
            parsed = subject.parse_args([option])
            self.assertTrue(getattr(parsed, option[2:].replace("-", "_")))
        with self.assertRaises(SystemExit):
            subject.parse_args([])


if __name__ == "__main__":
    unittest.main()
