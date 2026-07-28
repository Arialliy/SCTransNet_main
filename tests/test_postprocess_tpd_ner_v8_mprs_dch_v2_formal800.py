from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import (
    postprocess_tpd_ner_v8_mprs_dch_v2_formal800 as subject,
)


def point(matched: int, *, fa: float = 0.0, miou: float = 0.95) -> dict:
    return {
        "matched_target_count": matched,
        "target_count": 189,
        "pd": matched / 189,
        "fa": fa,
        "miou": miou,
        "false_objects_per_image": 0.0,
        "threshold": 0.5,
    }


def row(
    variant: str,
    role: str,
    counts: list[int],
    *,
    absolute: bool | None,
) -> dict:
    gate = (
        None
        if absolute is None
        else {"absolute_checkpoint_gate_passed": absolute}
    )
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
    pd_absolute: bool = True,
    miou_absolute: bool = True,
    on_counts: list[int] | None = None,
) -> dict:
    counts = [188, 188, 188, 188, 188] if on_counts is None else on_counts
    result = {}
    for checkpoint, role in subject.CHECKPOINT_ROLES.items():
        result[(subject.BASELINE_VARIANT, checkpoint)] = row(
            subject.BASELINE_VARIANT,
            role,
            [187, 187, 187, 187, 187],
            absolute=None,
        )
        # Deliberately mark the control as failing: it must not enter success.
        result[(subject.VARIANT_V1_OFF, checkpoint)] = row(
            subject.VARIANT_V1_OFF,
            role,
            [187, 188, 188, 188, 188],
            absolute=False,
        )
        result[(subject.VARIANT_V2_ON, checkpoint)] = row(
            subject.VARIANT_V2_ON,
            role,
            counts,
            absolute=(
                pd_absolute
                if role == "best_validation_pd_primary"
                else miou_absolute
            ),
        )
    return result


class V2PostprocessTests(unittest.TestCase):
    def test_plan_contains_exactly_two_new_v2_evaluations(self) -> None:
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
        self.assertEqual(plan["v1_off_evaluations"], 0)
        self.assertEqual(plan["baseline_evaluations"], 0)
        self.assertTrue(
            all(
                entry["variant"] == subject.VARIANT_V2_ON
                for entry in plan["new_evaluations"]
            )
        )
        self.assertTrue(
            all(
                entry["physical_gpu_index"] == 3
                for entry in plan["new_evaluations"]
            )
        )

    def test_evaluation_command_is_v2_only_and_single_gpu(self) -> None:
        command, environment, _ = subject.evaluation_command(
            checkpoint="best.pth.tar",
            python="/python",
            physical_gpu=2,
        )
        self.assertEqual(command[0], "/python")
        self.assertEqual(Path(command[1]), subject.V2_EVALUATOR.resolve())
        self.assertIn(str(subject.V2_RUN_DIR.resolve()), command)
        self.assertNotIn(str(subject.V1_OFF_RUN_DIR), command)
        self.assertEqual(
            environment["CUDA_VISIBLE_DEVICES"],
            subject.GPU_UUIDS["2"],
        )
        self.assertEqual(environment["PYTHONHASHSEED"], "42")

    def test_reused_v2_sweep_requires_current_validation_split_sha(self) -> None:
        binding = {
            "artifact_identity": {"validation_split_sha256": "a" * 64},
            "validation_split_sha256": "a" * 64,
        }
        payload = {"validation_split_sha256": "b" * 64}
        with (
            mock.patch.object(
                subject.v2_eval,
                "validate_output_identity",
            ) as validate,
            self.assertRaisesRegex(
                ValueError,
                "V2 sweep validation split SHA differs",
            ),
        ):
            subject._normalize_v2_sweep(
                payload,
                checkpoint="best.pth.tar",
                binding=binding,
            )
        validate.assert_not_called()

    def test_paired_gate_boundary(self) -> None:
        role = "best_validation_pd_primary"
        off = row(
            subject.VARIANT_V1_OFF,
            role,
            [187, 188, 188, 188, 188],
            absolute=False,
        )
        on = row(
            subject.VARIANT_V2_ON,
            role,
            [188, 188, 188, 188, 187],
            absolute=True,
        )
        gate = subject._paired_gate(off, on)
        self.assertEqual(gate["non_inferior_budget_count"], 4)
        self.assertEqual(gate["strictly_better_budget_count"], 1)
        self.assertTrue(gate["passed"])
        on["pd_at_fa_budget"][subject.BUDGET_KEYS[0]] = point(187)
        self.assertFalse(subject._paired_gate(off, on)["passed"])

    def test_success_ignores_control_absolute_gate(self) -> None:
        rows = full_rows()
        snapshot = {"reference": {"sha256": "a" * 64}}
        with mock.patch.object(subject, "sha256_file", return_value="b" * 64):
            report = subject.build_report(
                rows,
                lock_bindings={"lock": "fixture"},
                comparison_contract={"passed": True},
                reference_before=snapshot,
                reference_after=snapshot,
            )
        self.assertTrue(report["aggregate_full_model_gate_passed"])
        self.assertEqual(report["decision"], "FULL_MODEL_GATE_PASSED")
        self.assertFalse(
            report["success_components"]["v1_off_absolute_gate_required"]
        )
        self.assertFalse(
            report["success_components"]["baseline_affects_decision"]
        )

    def test_each_candidate_role_is_required(self) -> None:
        snapshot = {"reference": {"sha256": "a" * 64}}
        for name, rows in {
            "pd": full_rows(pd_absolute=False),
            "miou": full_rows(miou_absolute=False),
            "paired": full_rows(on_counts=[187, 187, 187, 187, 187]),
        }.items():
            with self.subTest(name=name), mock.patch.object(
                subject,
                "sha256_file",
                return_value="b" * 64,
            ):
                report = subject.build_report(
                    rows,
                    lock_bindings={},
                    comparison_contract={},
                    reference_before=snapshot,
                    reference_after=snapshot,
                )
                self.assertFalse(
                    report["aggregate_full_model_gate_passed"]
                )
                self.assertEqual(
                    report["decision"],
                    "RETURN_TO_MODEL_OPTIMIZATION",
                )

    def test_reference_paths_are_read_only_in_postprocess_source(self) -> None:
        source = inspect.getsource(subject)
        self.assertNotIn("prepare_baseline_reference_view(", source)
        self.assertNotIn("BASELINE_EVALUATOR,", source)
        self.assertNotIn("variant=VARIANT_V1_OFF,\n            python=", source)

    def test_reference_snapshot_detects_byte_or_mtime_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text("one", encoding="utf-8")
            with mock.patch.object(
                subject,
                "_reference_paths",
                return_value=(path,),
            ):
                before = subject.reference_snapshot()
                path.write_text("two", encoding="utf-8")
                after = subject.reference_snapshot()
        self.assertNotEqual(before, after)

    def test_failed_evaluator_quarantines_only_v2_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pd_fa_sweep_best.pth.json"

            def fail_with_partial(*args, **kwargs):
                output.write_text("{", encoding="utf-8")
                raise RuntimeError("fixture interruption")

            with (
                mock.patch.object(
                    subject,
                    "evaluation_command",
                    return_value=(["fixture"], {}, output),
                ),
                mock.patch.object(
                    subject,
                    "current_v2_binding",
                    return_value={},
                ),
                mock.patch.object(
                    subject.subprocess,
                    "run",
                    side_effect=fail_with_partial,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "fixture interruption",
                ):
                    subject._run_v2_evaluation(
                        checkpoint="best.pth.tar",
                        python="/python",
                        device_mode="cpu",
                        physical_gpu=2,
                    )
            self.assertFalse(output.exists())
            rejected = list(
                (Path(directory) / "rejected_postprocess").glob(
                    "*/pd_fa_sweep_best.pth.json"
                )
            )
            self.assertEqual(len(rejected), 1)

    def test_atomic_report_publish_and_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_output = root / "report.json"
            markdown_output = root / "report.md"
            marker = root / "POSTPROCESS_COMPLETE.json"
            report = {
                "decision": "RETURN_TO_MODEL_OPTIMIZATION",
                "aggregate_full_model_gate_passed": False,
                "rows": [],
                "v2_candidate_absolute_gate_by_role": {},
                "paired_v2_on_vs_v1_off_gate_by_role": {},
            }
            with (
                mock.patch.object(subject, "COMPARISON_DIR", root),
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
                # Identical publication is idempotent.
                subject.write_report(report)

    def test_single_seed_is_explicit(self) -> None:
        self.assertEqual(subject.TRAINING_SEED, 42)
        with mock.patch.object(
            subject,
            "inspect_v2_progress",
            return_value={"complete": False},
        ), mock.patch.object(
            subject.v1_post,
            "inspect_run_progress",
            return_value={"complete": False},
        ):
            readiness = subject.inspect_training_readiness()
        self.assertFalse(readiness["multi_seed_scheduled"])


if __name__ == "__main__":
    unittest.main()
