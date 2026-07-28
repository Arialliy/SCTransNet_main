from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import (
    postprocess_tpd_ner_v8_mprs_dch_v3_formal800 as subject,
)


def point(
    matched: int,
    *,
    fa: float = 0.0,
    miou: float = 0.95,
    matched_tiny: int = 39,
) -> dict:
    return {
        "matched_target_count": matched,
        "target_count": 189,
        "pd": matched / 189,
        "fa": fa,
        "miou": miou,
        "false_objects_per_image": 0.0,
        "threshold": 0.5,
        "tiny_pd": matched_tiny / 39,
        "matched_tiny_target_count": matched_tiny,
        "tiny_target_count": 39,
        "niou": 0.94,
        "pixel_precision": 0.93,
        "pixel_recall": 0.92,
        "pixel_f1": 0.925,
    }


def absolute_gate(
    *,
    fixed_passed: bool = True,
    budgets_passed: bool = True,
) -> dict:
    return {
        "fixed_threshold_checks": {
            "matched_targets": fixed_passed,
            "pd": fixed_passed,
            "fa": fixed_passed,
            "miou": fixed_passed,
        },
        "budget_checks": {
            key: {"passed": budgets_passed}
            for key in subject.BUDGET_KEYS
        },
        "absolute_checkpoint_gate_passed": (
            fixed_passed and budgets_passed
        ),
    }


def row(
    variant: str,
    role: str,
    counts: list[int],
    *,
    gate: dict | None = None,
) -> dict:
    return {
        "source": "fixture",
        "variant": variant,
        "checkpoint_role": role,
        "fixed_threshold_0_5": point(counts[-1]),
        "pd_at_fa_budget": {
            key: point(count)
            for key, count in zip(subject.BUDGET_KEYS, counts)
        },
        "absolute_gate": gate,
    }


def full_rows(
    *,
    v1_counts: list[int] | None = None,
    v2_counts: list[int] | None = None,
    v3_counts: list[int] | None = None,
) -> dict:
    v1_values = (
        [186, 187, 187, 187, 187]
        if v1_counts is None
        else v1_counts
    )
    v2_values = (
        [187, 188, 188, 188, 188]
        if v2_counts is None
        else v2_counts
    )
    v3_values = (
        [188, 188, 188, 188, 188]
        if v3_counts is None
        else v3_counts
    )
    result = {}
    for checkpoint, role in subject.CHECKPOINT_ROLES.items():
        result[(subject.BASELINE_VARIANT, checkpoint)] = row(
            subject.BASELINE_VARIANT,
            role,
            [185, 185, 185, 185, 185],
        )
        result[(subject.VARIANT_V1_OFF, checkpoint)] = row(
            subject.VARIANT_V1_OFF,
            role,
            list(v1_values),
            gate={"absolute_checkpoint_gate_passed": False},
        )
        result[(subject.VARIANT_V2_ON, checkpoint)] = row(
            subject.VARIANT_V2_ON,
            role,
            list(v2_values),
            gate={"absolute_checkpoint_gate_passed": True},
        )
        result[(subject.VARIANT_V3_ON, checkpoint)] = row(
            subject.VARIANT_V3_ON,
            role,
            list(v3_values),
            gate=absolute_gate(),
        )
    return result


def six_upstream_rows() -> dict:
    return {
        key: value
        for key, value in full_rows().items()
        if key[0] != subject.VARIANT_V3_ON
    }


class V3PostprocessTests(unittest.TestCase):
    def build(self, rows: dict) -> dict:
        snapshot = {"upstream": {"sha256": "a" * 64}}
        with mock.patch.object(
            subject,
            "sha256_file",
            return_value="b" * 64,
        ):
            return subject.build_report(
                rows,
                lock_bindings={"lock": "temporary-fixture"},
                comparison_contract={"passed": True},
                upstream_before=snapshot,
                upstream_after=snapshot,
            )

    def test_plan_contains_exactly_two_new_v3_evaluations(self) -> None:
        with mock.patch.object(
            subject,
            "inspect_training_readiness",
            return_value={"required_runs_complete": False},
        ):
            plan = subject.execution_plan(
                python="/python",
                device_mode="gpu23",
                physical_gpu=3,
            )
        self.assertEqual(plan["new_evaluation_count"], 2)
        self.assertEqual(plan["aggregate_row_count"], 8)
        self.assertEqual(plan["v2_on_evaluations"], 0)
        self.assertEqual(plan["v1_off_evaluations"], 0)
        self.assertEqual(plan["baseline_evaluations"], 0)
        self.assertTrue(
            all(
                entry["variant"] == subject.VARIANT_V3_ON
                for entry in plan["new_evaluations"]
            )
        )
        self.assertEqual(
            {entry["checkpoint"] for entry in plan["new_evaluations"]},
            set(subject.CHECKPOINTS),
        )
        self.assertTrue(
            all(
                entry["physical_gpu_index"] == 3
                for entry in plan["new_evaluations"]
            )
        )

    def test_evaluation_command_is_v3_only_and_single_gpu(self) -> None:
        command, environment, output = subject.evaluation_command(
            checkpoint="best.pth.tar",
            python="/python",
            physical_gpu=2,
        )
        self.assertEqual(command[0], "/python")
        self.assertEqual(Path(command[1]), subject.V3_EVALUATOR.resolve())
        self.assertIn(str(subject.V3_RUN_DIR.resolve()), command)
        self.assertNotIn(str(subject.V2_RUN_DIR.resolve()), command)
        self.assertNotIn(str(subject.V1_OFF_RUN_DIR.resolve()), command)
        self.assertEqual(
            environment["CUDA_VISIBLE_DEVICES"],
            subject.GPU_UUIDS["2"],
        )
        self.assertEqual(environment["PYTHONHASHSEED"], "42")
        self.assertEqual(
            output,
            subject.V3_RUN_DIR / "pd_fa_sweep_best.pth.json",
        )

    def test_paths_and_schemas_are_v3_owned(self) -> None:
        self.assertIn("v3", subject.SCHEMA)
        self.assertIn("v3", subject.READINESS_SCHEMA)
        self.assertIn("v3", subject.REJECTED_SCHEMA)
        self.assertTrue(
            subject.V3_RUN_DIR.is_relative_to(subject.V3_RESULT_ROOT)
        )
        self.assertTrue(
            subject.COMPARISON_DIR.is_relative_to(subject.V3_RESULT_ROOT)
        )
        self.assertNotEqual(subject.V3_RESULT_ROOT, subject.V2_RUN_DIR.parent)

    def test_gate_contract_is_preregistered_and_reuses_v2_absolute(self) -> None:
        contract = subject.formal_gate_contract()
        v2_contract = subject.v2_post.v2_eval.performance_gate_contract()
        for key in (
            "anchor_target_count",
            "pd_primary_fixed_threshold_0_5",
            "miou_secondary_fixed_threshold_0_5",
            "pd_at_fa_budget",
        ):
            self.assertEqual(contract[key], v2_contract[key])
        for name, reference in {
            "paired_v3_on_vs_v1_off_each_checkpoint_role": (
                subject.VARIANT_V1_OFF
            ),
            "paired_v3_on_vs_v2_on_each_checkpoint_role": (
                subject.VARIANT_V2_ON
            ),
        }.items():
            gate = contract[name]
            self.assertEqual(gate["reference"], reference)
            self.assertEqual(gate["candidate"], subject.VARIANT_V3_ON)
            self.assertEqual(gate["budget_count"], 5)
            self.assertEqual(gate["minimum_non_inferior_budget_count"], 4)
            self.assertEqual(
                gate["minimum_strictly_better_budget_count"],
                1,
            )
        self.assertEqual(len(contract["all_required_components"]), 6)

    def test_reused_v3_sweep_requires_current_validation_split_sha(self) -> None:
        binding = {
            "artifact_identity": {"validation_split_sha256": "a" * 64},
            "validation_split_sha256": "a" * 64,
        }
        payload = {"validation_split_sha256": "b" * 64}
        with (
            mock.patch.object(
                subject.v3_eval,
                "validate_output_identity",
            ) as validate,
            self.assertRaisesRegex(
                ValueError,
                "V3 sweep validation split SHA differs",
            ),
        ):
            subject._normalize_v3_sweep(
                payload,
                checkpoint="best.pth.tar",
                binding=binding,
            )
        validate.assert_not_called()

    def test_required_control_paired_gate_boundary(self) -> None:
        role = "best_validation_pd_primary"
        v1 = row(
            subject.VARIANT_V1_OFF,
            role,
            [187, 188, 188, 188, 188],
        )
        v3 = row(
            subject.VARIANT_V3_ON,
            role,
            [188, 188, 188, 188, 187],
        )
        gate = subject._paired_gate(v1, v3)
        self.assertEqual(gate["non_inferior_budget_count"], 4)
        self.assertEqual(gate["strictly_better_budget_count"], 1)
        self.assertTrue(gate["passed"])
        v3["pd_at_fa_budget"][subject.BUDGET_KEYS[0]] = point(187)
        self.assertFalse(subject._paired_gate(v1, v3)["passed"])

    def test_structural_predecessor_paired_gate_boundary(self) -> None:
        role = "best_validation_miou_secondary"
        v2 = row(
            subject.VARIANT_V2_ON,
            role,
            [187, 188, 188, 188, 188],
        )
        v3 = row(
            subject.VARIANT_V3_ON,
            role,
            [188, 188, 188, 188, 187],
        )
        gate = subject._predecessor_paired_gate(v2, v3)
        self.assertEqual(gate["non_inferior_budget_count"], 4)
        self.assertEqual(gate["strictly_better_budget_count"], 1)
        self.assertTrue(gate["passed"])
        v3["pd_at_fa_budget"][subject.BUDGET_KEYS[0]] = point(187)
        self.assertFalse(
            subject._predecessor_paired_gate(v2, v3)["passed"]
        )

    def test_success_has_eight_rows_and_every_decision_component(self) -> None:
        report = self.build(full_rows())
        self.assertEqual(report["row_count"], 8)
        self.assertEqual(len(report["rows"]), 8)
        for row_value in report["rows"]:
            self.assertEqual(
                set(subject.COMPLETE_FIXED_FIELDS)
                - set(row_value["fixed_threshold_0_5"]),
                set(),
            )
        self.assertEqual(
            [
                row_value["variant"]
                for row_value in report["rows"][:4]
            ],
            [
                subject.BASELINE_VARIANT,
                subject.VARIANT_V1_OFF,
                subject.VARIANT_V2_ON,
                subject.VARIANT_V3_ON,
            ],
        )
        self.assertEqual(len(report["required_decision_components"]), 8)
        self.assertEqual(
            tuple(report["preregistered_required_components"]),
            tuple(
                report["preregistered_performance_gate_contract"][
                    "all_required_components"
                ]
            ),
        )
        self.assertTrue(
            all(report["required_decision_components"].values())
        )
        self.assertTrue(report["aggregate_full_model_gate_passed"])
        self.assertEqual(report["decision"], "FULL_MODEL_GATE_PASSED")
        self.assertFalse(
            report["success_components"][
                "v1_off_absolute_gate_required"
            ]
        )
        self.assertFalse(
            report["success_components"][
                "v2_predecessor_absolute_gate_required"
            ]
        )
        self.assertFalse(
            report["success_components"]["baseline_affects_decision"]
        )
        self.assertFalse(
            report["success_components"]["tiny_pd_affects_decision"]
        )
        self.assertFalse(report["aggregate_tiny_pd_regressed"])
        self.assertFalse(
            report["v3_tiny_pd_regression_by_role"]["pd_primary"][
                "tiny_pd_regressed"
            ]
        )

    def test_tiny_pd_38_of_39_is_reported_but_does_not_change_gate(self) -> None:
        rows = full_rows()
        for checkpoint in subject.CHECKPOINTS:
            fixed = rows[(subject.VARIANT_V3_ON, checkpoint)][
                "fixed_threshold_0_5"
            ]
            fixed["matched_tiny_target_count"] = 38
            fixed["tiny_pd"] = 38 / 39
        report = self.build(rows)
        self.assertTrue(report["aggregate_tiny_pd_regressed"])
        self.assertTrue(
            all(
                audit["tiny_pd_regressed"]
                for audit in report[
                    "v3_tiny_pd_regression_by_role"
                ].values()
            )
        )
        self.assertTrue(report["aggregate_full_model_gate_passed"])
        self.assertEqual(report["decision"], "FULL_MODEL_GATE_PASSED")
        self.assertFalse(report["tiny_pd_regression_affects_decision"])
        self.assertTrue(
            report["claim_boundary"][
                "tiny_pd_regression_does_not_change_six_component_gate"
            ]
        )
        markdown = subject.render_markdown(report)
        self.assertIn("tiny-Pd regression: `true`", markdown)
        self.assertIn("38/39", markdown)

    def test_missing_complete_fixed_metric_is_rejected(self) -> None:
        for field in (
            "tiny_pd",
            "matched_tiny_target_count",
            "tiny_target_count",
            "niou",
            "pixel_precision",
            "pixel_recall",
            "pixel_f1",
        ):
            with self.subTest(field=field):
                rows = full_rows()
                del rows[(subject.VARIANT_V3_ON, "best.pth.tar")][
                    "fixed_threshold_0_5"
                ][field]
                with self.assertRaisesRegex(
                    ValueError,
                    "lacks fixed metrics",
                ):
                    self.build(rows)

    def test_markdown_displays_complete_fixed_metrics_and_tiny_audit(
        self,
    ) -> None:
        report = self.build(full_rows())
        markdown = subject.render_markdown(report)
        for heading in (
            "Tiny-Pd@0.5",
            "Tiny matched/total@0.5",
            "nIoU@0.5",
            "Pixel precision@0.5",
            "Pixel recall@0.5",
            "Pixel F1@0.5",
            "Tiny-target regression audit",
        ):
            self.assertIn(heading, markdown)
        self.assertIn("39/39", markdown)

    def test_each_absolute_fixed_and_budget_role_is_required(self) -> None:
        for checkpoint in subject.CHECKPOINTS:
            for component in ("fixed", "budgets"):
                with self.subTest(
                    checkpoint=checkpoint,
                    component=component,
                ):
                    rows = full_rows()
                    rows[(subject.VARIANT_V3_ON, checkpoint)][
                        "absolute_gate"
                    ] = absolute_gate(
                        fixed_passed=component != "fixed",
                        budgets_passed=component != "budgets",
                    )
                    report = self.build(rows)
                    self.assertFalse(
                        report["aggregate_full_model_gate_passed"]
                    )
                    self.assertEqual(
                        report["decision"],
                        "RETURN_TO_MODEL_OPTIMIZATION",
                    )

    def test_each_v1_and_v2_paired_role_is_required(self) -> None:
        for checkpoint in subject.CHECKPOINTS:
            for reference in ("v1", "v2"):
                with self.subTest(
                    checkpoint=checkpoint,
                    reference=reference,
                ):
                    rows = full_rows()
                    variant = (
                        subject.VARIANT_V1_OFF
                        if reference == "v1"
                        else subject.VARIANT_V2_ON
                    )
                    rows[(variant, checkpoint)][
                        "pd_at_fa_budget"
                    ] = {
                        key: point(188)
                        for key in subject.BUDGET_KEYS
                    }
                    report = self.build(rows)
                    self.assertFalse(
                        report["aggregate_full_model_gate_passed"]
                    )

    def test_upstream_paths_are_read_only_in_postprocess_source(self) -> None:
        source = inspect.getsource(subject)
        self.assertNotIn("prepare_baseline_reference_view(", source)
        self.assertNotIn("_run_v2_evaluation(", source)
        self.assertNotIn("run_v2_evaluations(", source)
        self.assertNotIn("quarantine_postprocess_artifacts(", source)
        self.assertEqual(source.count("subprocess.run("), 1)

    def test_upstream_snapshot_detects_byte_or_mtime_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upstream.json"
            path.write_text("one", encoding="utf-8")
            with mock.patch.object(
                subject,
                "_upstream_paths",
                return_value=(path,),
            ):
                before = subject.upstream_snapshot()
                path.write_text("two", encoding="utf-8")
                after = subject.upstream_snapshot()
        self.assertNotEqual(before, after)

    def test_upstream_failure_propagates_without_quarantine_or_repair(self) -> None:
        with (
            mock.patch.object(
                subject.v2_post,
                "load_all_rows",
                side_effect=ValueError("invalid V2 sweep"),
            ),
            mock.patch.object(
                subject,
                "quarantine_v3_postprocess_artifacts",
            ) as quarantine,
            self.assertRaisesRegex(ValueError, "invalid V2 sweep"),
        ):
            subject.load_upstream_rows()
        quarantine.assert_not_called()

    def test_upstream_rows_gain_complete_fixed_metrics_from_raw_sweeps(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dirs = {
                subject.BASELINE_VARIANT: root / "baseline",
                subject.VARIANT_V1_OFF: root / "v1",
                subject.VARIANT_V2_ON: root / "v2",
            }
            normalized = six_upstream_rows()
            for variant, run_dir in run_dirs.items():
                run_dir.mkdir()
                for checkpoint in subject.CHECKPOINTS:
                    subject.sweep_path(run_dir, checkpoint).write_text(
                        json.dumps(
                            {
                                "fixed_threshold_0_5": normalized[
                                    (variant, checkpoint)
                                ]["fixed_threshold_0_5"]
                            }
                        ),
                        encoding="utf-8",
                    )
            for value in normalized.values():
                fixed = value["fixed_threshold_0_5"]
                value["fixed_threshold_0_5"] = {
                    name: fixed[name]
                    for name in (
                        "matched_target_count",
                        "target_count",
                        "pd",
                        "fa",
                        "miou",
                        "false_objects_per_image",
                        "threshold",
                    )
                }
            with (
                mock.patch.object(
                    subject.v2_post,
                    "load_all_rows",
                    return_value=normalized,
                ),
                mock.patch.object(
                    subject,
                    "BASELINE_RUN_DIR",
                    run_dirs[subject.BASELINE_VARIANT],
                ),
                mock.patch.object(
                    subject,
                    "V1_OFF_RUN_DIR",
                    run_dirs[subject.VARIANT_V1_OFF],
                ),
                mock.patch.object(
                    subject,
                    "V2_RUN_DIR",
                    run_dirs[subject.VARIANT_V2_ON],
                ),
            ):
                rows = subject.load_upstream_rows()
            for value in rows.values():
                self.assertEqual(
                    set(subject.COMPLETE_FIXED_FIELDS),
                    set(value["fixed_threshold_0_5"]),
                )

    def test_upstream_change_during_v3_evaluation_is_a_hard_stop(self) -> None:
        stable = {"upstream": {"sha256": "a" * 64}}
        changed = {"upstream": {"sha256": "b" * 64}}
        with (
            mock.patch.object(
                subject,
                "inspect_training_readiness",
                return_value={"required_runs_complete": True},
            ),
            mock.patch.object(
                subject,
                "upstream_snapshot",
                side_effect=[stable, stable, changed],
            ),
            mock.patch.object(
                subject,
                "load_upstream_rows",
                return_value=six_upstream_rows(),
            ),
            mock.patch.object(
                subject,
                "_run_v3_evaluation",
                return_value={"status": "fixture"},
            ),
            self.assertRaisesRegex(
                ValueError,
                "upstream files changed during V3 evaluation",
            ),
        ):
            subject.run_v3_evaluations(
                python="/python",
                device_mode="cpu",
                physical_gpu=2,
            )

    def test_failed_evaluator_quarantines_only_v3_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "v3-results"
            run_dir = root / "run"
            run_dir.mkdir(parents=True)
            output = run_dir / "pd_fa_sweep_best.pth.json"

            def fail_with_partial(*args, **kwargs):
                output.write_text("{", encoding="utf-8")
                raise RuntimeError("fixture interruption")

            with (
                mock.patch.object(subject, "V3_RESULT_ROOT", root),
                mock.patch.object(
                    subject,
                    "evaluation_command",
                    return_value=(["fixture"], {}, output),
                ),
                mock.patch.object(
                    subject,
                    "current_v3_binding",
                    return_value={},
                ),
                mock.patch.object(
                    subject.subprocess,
                    "run",
                    side_effect=fail_with_partial,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "fixture interruption",
                ),
            ):
                subject._run_v3_evaluation(
                    checkpoint="best.pth.tar",
                    python="/python",
                    device_mode="cpu",
                    physical_gpu=2,
                )
            self.assertFalse(output.exists())
            rejected = list(
                run_dir.glob(
                    "rejected_postprocess/*/pd_fa_sweep_best.pth.json"
                )
            )
            self.assertEqual(len(rejected), 1)

    def test_quarantine_refuses_any_upstream_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v3_root = root / "v3"
            v3_root.mkdir()
            upstream = root / "v2-sweep.json"
            upstream.write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(subject, "V3_RESULT_ROOT", v3_root),
                self.assertRaisesRegex(
                    ValueError,
                    "non-V3 artifact",
                ),
            ):
                subject.quarantine_v3_postprocess_artifacts(
                    [upstream],
                    parent=v3_root,
                    reason="must refuse",
                )
            self.assertTrue(upstream.is_file())

    def test_quarantine_refuses_symlinked_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v3_root = root / "v3"
            outside = root / "outside"
            v3_root.mkdir()
            outside.mkdir()
            linked_run = v3_root / "run"
            linked_run.symlink_to(outside, target_is_directory=True)
            escaped = linked_run / "partial.json"
            escaped.write_text("{", encoding="utf-8")
            with (
                mock.patch.object(subject, "V3_RESULT_ROOT", v3_root),
                self.assertRaisesRegex(
                    ValueError,
                    "ancestry contains a symlink",
                ),
            ):
                subject.quarantine_v3_postprocess_artifacts(
                    [escaped],
                    parent=linked_run,
                    reason="must refuse symlink escape",
                )
            self.assertTrue((outside / "partial.json").is_file())

    def test_missing_production_v3_lock_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_training = root / "training.json"
            missing_acceptance = root / "acceptance.json"
            with (
                mock.patch.object(
                    subject,
                    "TRAINING_LOCK",
                    missing_training,
                ),
                mock.patch.object(
                    subject,
                    "ACCEPTANCE_LOCK",
                    missing_acceptance,
                ),
                mock.patch.object(
                    subject.v3_eval,
                    "verify_frozen_manifests",
                ) as verify,
                self.assertRaises(FileNotFoundError),
            ):
                subject.verify_frozen_manifests()
            verify.assert_not_called()

    def test_temporary_v3_lock_binding_is_accepted_in_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "training.json"
            acceptance = root / "acceptance.json"
            training.write_text('{"fixture":"training"}', encoding="utf-8")
            acceptance.write_text(
                '{"fixture":"acceptance"}',
                encoding="utf-8",
            )
            with (
                mock.patch.object(subject, "TRAINING_LOCK", training),
                mock.patch.object(subject, "ACCEPTANCE_LOCK", acceptance),
                mock.patch.object(
                    subject.v3_eval,
                    "verify_frozen_manifests",
                    return_value={"binding": "temporary"},
                ) as verify,
                mock.patch.object(
                    subject.v3_freeze,
                    "verify_training_lock",
                    return_value={
                        "training_data_sha256": "a" * 64,
                        "source_count": 1,
                    },
                ) as verify_training,
                mock.patch.object(
                    subject.v3_freeze,
                    "verify_acceptance_lock",
                    return_value={
                        "upstream_v2_training_data_sha256": "a" * 64,
                        "source_count": 2,
                    },
                ) as verify_acceptance,
                mock.patch.object(
                    subject.v2_post,
                    "verify_frozen_manifests",
                    return_value={"upstream": "read-only"},
                ),
            ):
                binding = subject.verify_frozen_manifests()
            verify.assert_called_once_with(
                training_lock_path=training,
                acceptance_lock_path=acceptance,
            )
            verify_training.assert_called_once_with(training)
            verify_acceptance.assert_called_once_with(acceptance, training)
            self.assertEqual(
                binding["v3_evaluation_source_binding"],
                {"binding": "temporary"},
            )
            self.assertEqual(
                binding["upstream_v2_training_data_sha256"],
                "a" * 64,
            )

    def test_atomic_report_publish_and_marker_are_v3_owned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "v3-results"
            comparison = root / "comparison"
            json_output = comparison / "report.json"
            markdown_output = comparison / "report.md"
            marker = comparison / "POSTPROCESS_COMPLETE.json"
            report = {
                "decision": "RETURN_TO_MODEL_OPTIMIZATION",
                "aggregate_full_model_gate_passed": False,
                "aggregate_tiny_pd_regressed": False,
                "rows": [],
                "v3_candidate_absolute_components_by_role": {},
                "paired_v3_on_vs_v1_off_gate_by_role": {},
                "paired_v3_on_vs_v2_on_gate_by_role": {},
            }
            with (
                mock.patch.object(subject, "V3_RESULT_ROOT", root),
                mock.patch.object(subject, "COMPARISON_DIR", comparison),
                mock.patch.object(subject, "JSON_OUTPUT", json_output),
                mock.patch.object(
                    subject,
                    "MARKDOWN_OUTPUT",
                    markdown_output,
                ),
                mock.patch.object(subject, "COMPLETE_MARKER", marker),
                mock.patch.object(
                    subject,
                    "render_markdown",
                    return_value="# fixture\n",
                ),
            ):
                paths = subject.write_report(report)
                self.assertEqual(paths, (json_output, markdown_output, marker))
                self.assertTrue(marker.is_file())
                subject.write_report(report)

    def test_single_seed_and_split_are_explicit(self) -> None:
        self.assertEqual(subject.TRAINING_SEED, 42)
        self.assertEqual(subject.SPLIT_SEED, 20260722)
        with (
            mock.patch.object(
                subject,
                "inspect_v3_progress",
                return_value={"complete": False},
            ),
            mock.patch.object(
                subject.v2_post,
                "inspect_v2_progress",
                return_value={"complete": False},
            ),
            mock.patch.object(
                subject.v1_post,
                "inspect_run_progress",
                return_value={"complete": False},
            ),
        ):
            readiness = subject.inspect_training_readiness()
        self.assertFalse(readiness["multi_seed_scheduled"])


if __name__ == "__main__":
    unittest.main()
