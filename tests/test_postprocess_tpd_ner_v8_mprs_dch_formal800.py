from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import (  # noqa: E402
    postprocess_tpd_ner_v8_mprs_dch_formal800 as entry,
)
from experiments import evaluate_tpd_ner_v8_mprs_dch_pd_fa as evaluator  # noqa: E402


def point(matched: int, *, fa: float = 5e-7, miou: float = 0.95, threshold: float = 0.5):
    return {
        "matched_target_count": matched,
        "target_count": 189,
        "pd": matched / 189,
        "fa": fa,
        "miou": miou,
        "false_objects_per_image": 0.02,
        "threshold": threshold,
    }


def candidate_row(
    variant: str,
    checkpoint: str,
    counts: tuple[int, int, int, int, int],
):
    role = entry.CHECKPOINT_ROLES[checkpoint]
    fixed_count = 188 if role == "best_validation_pd_primary" else 187
    fixed_miou = 0.94 if role == "best_validation_pd_primary" else 0.95
    fixed = point(fixed_count, miou=fixed_miou)
    budgets = {
        key: point(count, threshold=0.6)
        for key, count in zip(entry.BUDGET_KEYS, counts)
    }
    return {
        "source": "new_model",
        "variant": variant,
        "seed": 42,
        "split_seed": 20260722,
        "checkpoint": checkpoint,
        "checkpoint_role": role,
        "checkpoint_epoch": 700,
        "fixed_threshold_0_5": fixed,
        "pd_at_fa_budget": budgets,
        "absolute_gate": entry._absolute_gate(role, fixed, budgets),
    }


def baseline_row(checkpoint: str):
    role = entry.CHECKPOINT_ROLES[checkpoint]
    return {
        "source": "same_protocol_external_reference",
        "variant": "baseline_sctransnet",
        "seed": 42,
        "split_seed": 20260722,
        "checkpoint": checkpoint,
        "checkpoint_role": role,
        "checkpoint_epoch": 500,
        "fixed_threshold_0_5": point(180, fa=1e-5, miou=0.90),
        "pd_at_fa_budget": {
            key: point(180, threshold=0.7)
            for key in entry.BUDGET_KEYS
        },
        "absolute_gate": None,
    }


def ner_binding(variant: str, checkpoint: str):
    run_dir = Path(f"/tmp/{variant}").resolve()
    role = entry.CHECKPOINT_ROLES[checkpoint]
    run_identity = {
        "schema": "sctransnet_tpd_run_identity_v1",
        "run_id": f"run:{variant}",
        "variant": variant,
        "seed": 42,
        "split_seed": 20260722,
    }
    checkpoint_identity = {
        "schema": "checkpoint_identity",
        "variant": variant,
        "run_id": run_identity["run_id"],
    }
    artifacts = {
        "protocol.json": "1" * 64,
        "split.json": "2" * 64,
        "summary.json": "3" * 64,
        "metrics.jsonl": "4" * 64,
        "checkpoint": "5" * 64,
        "evaluator": "6" * 64,
    }
    return {
        "variant": variant,
        "run_dir": run_dir,
        "checkpoint_path": run_dir / checkpoint,
        "checkpoint_name": checkpoint,
        "checkpoint_role": role,
        "checkpoint_sha256": artifacts["checkpoint"],
        "validation_split_sha256": "7" * 64,
        "evaluator_path": entry.NER_EVALUATOR.resolve(),
        "evaluator_sha256": artifacts["evaluator"],
        "artifact_sha256": artifacts,
        "artifact_identity": {
            "training_artifact_mode": "exact_resume_primary",
            "run_identity": run_identity,
            "checkpoint_identity": checkpoint_identity,
            "variant": variant,
            "checkpoint_filename": checkpoint,
            "checkpoint_role": role,
        },
    }


def ner_payload(
    variant: str,
    checkpoint: str,
    counts: tuple[int, int, int, int, int],
):
    binding = ner_binding(variant, checkpoint)
    row = candidate_row(variant, checkpoint, counts)
    fixed = row["fixed_threshold_0_5"]
    budgets = row["pd_at_fa_budget"]
    gate = row["absolute_gate"]
    identity = binding["artifact_identity"]
    expected_budget_coverage = {
        key: {
            "budget": budget,
            "pd": budgets[key]["pd"],
            "achieved_fa": budgets[key]["fa"],
            "threshold": budgets[key]["threshold"],
            "matched_target_count": budgets[key]["matched_target_count"],
            "target_count": budgets[key]["target_count"],
        }
        for budget, key in zip(entry.FA_BUDGETS, entry.BUDGET_KEYS)
    }
    recorded_budget_checks = {
        key: {
            "required_matched_target_count": gate["budget_checks"][key][
                "required_matched_target_count"
            ],
            "required_pd": gate["budget_checks"][key]["required_pd"],
            "observed_matched_target_count": budgets[key][
                "matched_target_count"
            ],
            "observed_target_count": budgets[key]["target_count"],
            "observed_pd": budgets[key]["pd"],
            "checks": gate["budget_checks"][key]["checks"],
            "passed": gate["budget_checks"][key]["passed"],
        }
        for key in entry.BUDGET_KEYS
    }
    upper_endpoint = {
        "threshold": evaluator.UPPER_BOUNDARY_THRESHOLD,
        "pd": 0.0,
        "fa": 0.0,
        "matched_target_count": 0,
        "predicted_object_count": 0,
        "unmatched_predicted_object_count": 0,
    }
    return {
        "schema": evaluator.EVALUATION_SCHEMA,
        "run_directory": str(binding["run_dir"]),
        "checkpoint": str(binding["checkpoint_path"]),
        "checkpoint_sha256": binding["checkpoint_sha256"],
        "checkpoint_role": entry.CHECKPOINT_ROLES[checkpoint],
        "checkpoint_epoch": 700,
        "variant": variant,
        "dataset": "NUDT-SIRST",
        "seed": 42,
        "split_seed": 20260722,
        "validation_count": 133,
        "validation_split_sha256": binding["validation_split_sha256"],
        "official_test_accessed": False,
        "match_radius": 3.0,
        "tiny_area": 9,
        "threshold_configuration": {
            "threshold_min": 0.01,
            "threshold_max": 0.99,
            "threshold_step": 0.01,
            "extra_thresholds": [0.001, 0.005, 0.995, 0.999, 0.9995, 0.9999],
            "tail_logit_step": 0.1,
            "fa_budgets": list(entry.FA_BUDGETS),
        },
        "threshold_provenance": {
            "posthoc_endpoint_completion": False,
            "preregistered_endpoint_completion": True,
            "endpoint_protocol_stage": "before_formal_training",
            "closed_probability_interval": True,
            "score_dtype": "float32",
            "added_thresholds": [
                evaluator.LAST_FLOAT32_BELOW_ONE,
                evaluator.UPPER_BOUNDARY_THRESHOLD,
            ],
            "last_float32_below_one": evaluator.LAST_FLOAT32_BELOW_ONE,
            "upper_boundary_threshold": evaluator.UPPER_BOUNDARY_THRESHOLD,
            "upper_boundary_comparison": "prediction > threshold",
            "upper_boundary_semantics": "empty_prediction_pd0_fa0",
        },
        "points": [
            {
                **upper_endpoint,
                "threshold": evaluator.LAST_FLOAT32_BELOW_ONE,
            },
            upper_endpoint,
        ],
        "fixed_threshold_0_5": fixed,
        "best_points_under_fa_budget": budgets,
        "final_metric_coverage": {
            "schema": evaluator.FINAL_METRIC_COVERAGE_SCHEMA,
            "fixed_threshold": 0.5,
            "fixed_threshold_0_5": {
                name: fixed[name]
                for name in (
                    "pd",
                    "fa",
                    "miou",
                    "false_objects_per_image",
                )
            },
            "pd_at_fa_budget": expected_budget_coverage,
            "all_required_metrics_present": True,
        },
        "performance_gate_assessment": {
            "contract": evaluator.performance_gate_contract(),
            "fixed_threshold_gate": gate["fixed_threshold_gate"],
            "fixed_threshold_observed": {
                "matched_target_count": fixed["matched_target_count"],
                "target_count": fixed["target_count"],
                "pd": fixed["pd"],
                "fa": fixed["fa"],
                "miou": fixed["miou"],
            },
            "fixed_threshold_checks": gate["fixed_threshold_checks"],
            "budget_checks": recorded_budget_checks,
            "absolute_checkpoint_gate_passed": gate["passed"],
            "paired_relay_on_gate_status": (
                "requires_relay_off_and_relay_on_aggregate"
                if variant == entry.VARIANT_ON
                else "reference_control_not_applicable"
            ),
            "formal_success_claim_authorized": False,
        },
        "artifact_identity_preflight_passed": True,
        "run_identity": identity["run_identity"],
        "source_checkpoint_identity": identity["checkpoint_identity"],
        "training_artifact_mode": identity["training_artifact_mode"],
        "evaluated_checkpoint_identity": {
            "training_artifact_mode": identity["training_artifact_mode"],
            "filename": checkpoint,
            "role": entry.CHECKPOINT_ROLES[checkpoint],
            "sha256": binding["checkpoint_sha256"],
        },
        "evaluator_contract": evaluator.evaluator_contract(),
        "audit": {
            "invocation_argv": [
                sys.executable,
                str(entry.NER_EVALUATOR.resolve()),
            ],
            "parsed_arguments": {
                "run_dir": str(binding["run_dir"]),
                "checkpoint": checkpoint,
                "expected_epochs": 800,
                "threshold_min": 0.01,
                "threshold_max": 0.99,
                "threshold_step": 0.01,
                "extra_thresholds": [
                    0.001,
                    0.005,
                    0.995,
                    0.999,
                    0.9995,
                    0.9999,
                ],
                "tail_logit_step": 0.1,
                "fa_budgets": list(entry.FA_BUDGETS),
                "match_radius": None,
                "tiny_area": None,
                "overwrite": False,
            },
            "expected_epochs": 800,
            "metrics_event_count": 800,
            "metrics_epoch_range": [1, 800],
            "summary_status": "complete",
            "integrity_checks_passed": {"all": True},
            "artifact_sha256": binding["artifact_sha256"],
        },
    }, binding


def successful_matrix():
    rows = {}
    for checkpoint in entry.CHECKPOINTS:
        rows[(entry.VARIANT_OFF, checkpoint)] = candidate_row(
            entry.VARIANT_OFF,
            checkpoint,
            (187, 188, 188, 188, 188),
        )
        rows[(entry.VARIANT_ON, checkpoint)] = candidate_row(
            entry.VARIANT_ON,
            checkpoint,
            (187, 189, 188, 188, 188),
        )
        rows[("baseline_sctransnet", checkpoint)] = baseline_row(checkpoint)
    return rows


class PostprocessReadinessTests(unittest.TestCase):
    def test_complete_requires_summary_and_800_contiguous_events(self) -> None:
        variant = entry.VARIANT_OFF
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "protocol.json",
                "split.json",
                "best.pth.tar",
                "best_miou.pth.tar",
                "last.pth.tar",
            ):
                (root / name).touch()
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "variant": variant,
                        "seed": 42,
                        "split_seed": 20260722,
                    }
                ),
                encoding="utf-8",
            )
            (root / "metrics.jsonl").write_text(
                "".join(
                    json.dumps({"epoch": epoch}) + "\n"
                    for epoch in range(1, 801)
                ),
                encoding="utf-8",
            )
            ready = entry.inspect_run_progress(variant, root)
            self.assertTrue(ready["complete"])
            self.assertEqual(ready["metrics"]["event_count"], 800)

            (root / "metrics.jsonl").write_text(
                "".join(
                    json.dumps({"epoch": epoch}) + "\n"
                    for epoch in range(1, 800)
                ),
                encoding="utf-8",
            )
            incomplete = entry.inspect_run_progress(variant, root)
            self.assertFalse(incomplete["complete"])

    def test_current_status_is_read_only_and_never_early_stops(self) -> None:
        status = entry.inspect_training_readiness()
        self.assertEqual(status["expected_epochs"], 800)
        self.assertIn(status["posttraining_action"], ("wait", "evaluate"))
        for record in status["runs"].values():
            self.assertEqual(
                set(record["metrics"]),
                {"exists", "event_count", "last_epoch", "contiguous_from_one"},
            )
            self.assertNotIn("pd", record)
            self.assertNotIn("miou", record)


class PostprocessContractTests(unittest.TestCase):
    def test_frozen_manifests_verify_and_exclude_new_orchestration(self) -> None:
        bindings = entry.verify_frozen_manifests()
        self.assertEqual(
            bindings["training_source_lock_sha256"],
            entry.EXPECTED_TRAINING_LOCK_SHA256,
        )
        for path in (entry.TRAINING_LOCK, entry.ACCEPTANCE_LOCK):
            sources = json.loads(path.read_text(encoding="utf-8"))["source_sha256"]
            self.assertNotIn(
                "experiments/postprocess_tpd_ner_v8_mprs_dch_formal800.py",
                sources,
            )
            self.assertNotIn(
                "experiments/evaluate_sctransnet_baseline_reference_closed_interval.py",
                sources,
            )

    def test_execution_order_is_role_synchronized_gpu2_gpu3(self) -> None:
        with mock.patch.object(
            entry,
            "inspect_training_readiness",
            return_value={"both_runs_complete": False},
        ):
            plan = entry.execution_plan(python="/venv/python")
        self.assertEqual(
            [phase["checkpoint"] for phase in plan["evaluation_order"]],
            ["best.pth.tar", "best_miou.pth.tar"],
        )
        for phase in plan["evaluation_order"]:
            pair = phase["parallel_candidate_pair"]
            self.assertEqual(
                [record["physical_gpu_index"] for record in pair],
                [2, 3],
            )
        command, environment, _ = entry.evaluation_command(
            variant=entry.VARIANT_ON,
            checkpoint="best.pth.tar",
            python="/venv/python",
        )
        self.assertEqual(command[0], "/venv/python")
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        self.assertEqual(
            environment["CUDA_VISIBLE_DEVICES"],
            entry.GPU_UUIDS[entry.VARIANT_ON],
        )

    def test_runtime_waits_then_runs_best_pair_before_miou_pair(self) -> None:
        calls = []
        call_lock = threading.Lock()

        def fake_run(**kwargs):
            with call_lock:
                calls.append((kwargs["variant"], kwargs["checkpoint"]))
            return {
                "variant": kwargs["variant"],
                "checkpoint": kwargs["checkpoint"],
            }

        with (
            mock.patch.object(
                entry,
                "inspect_training_readiness",
                return_value={"both_runs_complete": True},
            ),
            mock.patch.object(entry, "prepare_baseline_reference_view"),
            mock.patch.object(entry, "_run_evaluation", side_effect=fake_run),
        ):
            records = entry.run_role_synchronized_evaluations(
                python="/venv/python"
            )
        self.assertEqual(len(records), 6)
        self.assertEqual(
            [checkpoint for _, checkpoint in calls[:3]],
            ["best.pth.tar"] * 3,
        )
        self.assertEqual(
            [checkpoint for _, checkpoint in calls[3:]],
            ["best_miou.pth.tar"] * 3,
        )
        self.assertEqual(
            {variant for variant, _ in calls[:2]},
            {entry.VARIANT_OFF, entry.VARIANT_ON},
        )

    def test_baseline_reference_view_is_no_overwrite_hard_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            view = root / "view"
            source.mkdir()
            names = ("protocol.json", "best.pth.tar")
            for index, name in enumerate(names):
                (source / name).write_bytes(f"artifact-{index}".encode())
            with (
                mock.patch.object(entry, "BASELINE_SOURCE_RUN", source),
                mock.patch.object(entry, "BASELINE_VIEW_RUN", view),
                mock.patch.object(entry, "BASELINE_VIEW_FILES", names),
                mock.patch.object(
                    entry,
                    "_same_split_and_training_contract",
                    return_value={"same_split_hashes": True},
                ),
            ):
                result = entry.prepare_baseline_reference_view()
                again = entry.prepare_baseline_reference_view()
            self.assertTrue(result["reference_view_uses_hard_links"])
            self.assertEqual(result, again)
            for name in names:
                self.assertEqual(
                    os.stat(source / name).st_ino,
                    os.stat(view / name).st_ino,
                )

    def test_normalize_ner_sweep_recomputes_frozen_gate(self) -> None:
        checkpoint = "best.pth.tar"
        payload, binding = ner_payload(
            entry.VARIANT_ON,
            checkpoint,
            (187, 188, 188, 188, 189),
        )
        normalized = entry.normalize_ner_sweep(
            payload,
            variant=entry.VARIANT_ON,
            checkpoint=checkpoint,
            binding=binding,
        )
        self.assertTrue(normalized["absolute_gate"]["passed"])
        self.assertEqual(
            normalized["pd_at_fa_budget"]["0.0001"]["matched_target_count"],
            189,
        )

    def test_ner_reuse_rejects_every_current_identity_mismatch(self) -> None:
        checkpoint = "best.pth.tar"
        original, binding = ner_payload(
            entry.VARIANT_ON,
            checkpoint,
            (187, 188, 188, 188, 189),
        )
        cases = (
            (
                "checkpoint sha",
                lambda value: value.__setitem__(
                    "checkpoint_sha256",
                    "0" * 64,
                ),
                "checkpoint SHA",
            ),
            (
                "checkpoint name",
                lambda value: value.__setitem__(
                    "checkpoint",
                    str(binding["run_dir"] / "best_miou.pth.tar"),
                ),
                "checkpoint path",
            ),
            (
                "absolute run dir",
                lambda value: value.__setitem__(
                    "run_directory",
                    "/tmp/wrong-run",
                ),
                "run directory",
            ),
            (
                "training seed",
                lambda value: value.__setitem__("seed", 43),
                "seed",
            ),
            (
                "split seed",
                lambda value: value.__setitem__("split_seed", 1),
                "split seed",
            ),
            (
                "validation split",
                lambda value: value.__setitem__(
                    "validation_split_sha256",
                    "0" * 64,
                ),
                "validation split SHA",
            ),
            (
                "evaluator sha",
                lambda value: value["audit"]["artifact_sha256"].__setitem__(
                    "evaluator",
                    "0" * 64,
                ),
                "artifact SHA",
            ),
            (
                "evaluator invocation",
                lambda value: value["audit"]["invocation_argv"].__setitem__(
                    1,
                    "/tmp/wrong-evaluator.py",
                ),
                "invocation",
            ),
            (
                "parsed evaluator arguments",
                lambda value: value["audit"]["parsed_arguments"].__setitem__(
                    "threshold_step",
                    0.02,
                ),
                "parsed evaluator argument",
            ),
            (
                "preflight",
                lambda value: value.__setitem__(
                    "artifact_identity_preflight_passed",
                    False,
                ),
                "preflight",
            ),
            (
                "run identity",
                lambda value: value["run_identity"].__setitem__(
                    "run_id",
                    "wrong",
                ),
                "run identity",
            ),
            (
                "source checkpoint identity",
                lambda value: value["source_checkpoint_identity"].__setitem__(
                    "run_id",
                    "wrong",
                ),
                "source checkpoint identity",
            ),
            (
                "checkpoint role",
                lambda value: value.__setitem__(
                    "checkpoint_role",
                    "wrong",
                ),
                "checkpoint role",
            ),
            (
                "closed endpoint",
                lambda value: value["threshold_provenance"].__setitem__(
                    "upper_boundary_threshold",
                    0.99,
                ),
                "threshold provenance",
            ),
            (
                "metric coverage",
                lambda value: value["final_metric_coverage"][
                    "pd_at_fa_budget"
                ]["1e-06"].__setitem__("pd", 0.0),
                "budget metric coverage",
            ),
        )
        for label, mutate, message in cases:
            with self.subTest(label=label):
                value = copy.deepcopy(original)
                mutate(value)
                with self.assertRaisesRegex(ValueError, message):
                    entry.normalize_ner_sweep(
                        value,
                        variant=entry.VARIANT_ON,
                        checkpoint=checkpoint,
                        binding=binding,
                    )

    def test_ner_reuse_rejects_each_audit_artifact_hash(self) -> None:
        checkpoint = "best.pth.tar"
        original, binding = ner_payload(
            entry.VARIANT_ON,
            checkpoint,
            (187, 188, 188, 188, 189),
        )
        for artifact_name in (
            "protocol.json",
            "split.json",
            "summary.json",
            "metrics.jsonl",
            "checkpoint",
            "evaluator",
        ):
            with self.subTest(artifact=artifact_name):
                value = copy.deepcopy(original)
                value["audit"]["artifact_sha256"][artifact_name] = "0" * 64
                with self.assertRaisesRegex(ValueError, "artifact SHA"):
                    entry.normalize_ner_sweep(
                        value,
                        variant=entry.VARIANT_ON,
                        checkpoint=checkpoint,
                        binding=binding,
                    )

    def test_aggregate_pass_and_paired_failure_are_both_explicit(self) -> None:
        successful = successful_matrix()
        with mock.patch.object(entry, "sha256_file", return_value="a" * 64):
            report = entry.build_report(
                successful,
                lock_bindings={"training_source_lock_sha256": "b" * 64},
                baseline_contract={"same_split_hashes": True},
            )
        self.assertTrue(report["aggregate_full_model_gate_passed"])
        self.assertEqual(report["decision"], "FULL_MODEL_GATE_PASSED")
        self.assertEqual(len(report["rows"]), 6)
        self.assertEqual(len(report["comparisons_vs_baseline"]), 4)

        failing = dict(successful)
        for checkpoint in entry.CHECKPOINTS:
            failing[(entry.VARIANT_OFF, checkpoint)] = candidate_row(
                entry.VARIANT_OFF,
                checkpoint,
                (188, 189, 189, 189, 189),
            )
            failing[(entry.VARIANT_ON, checkpoint)] = candidate_row(
                entry.VARIANT_ON,
                checkpoint,
                (187, 188, 188, 188, 188),
            )
        with mock.patch.object(entry, "sha256_file", return_value="a" * 64):
            report = entry.build_report(
                failing,
                lock_bindings={"training_source_lock_sha256": "b" * 64},
                baseline_contract={"same_split_hashes": True},
            )
        self.assertTrue(report["all_four_absolute_checkpoint_gates_passed"])
        self.assertFalse(report["both_role_paired_relay_on_gates_passed"])
        self.assertEqual(report["decision"], "RETURN_TO_MODEL_OPTIMIZATION")

    def test_aggregate_refuses_incomplete_training_before_loading_sweeps(self) -> None:
        with (
            mock.patch.object(
                entry,
                "inspect_training_readiness",
                return_value={"both_runs_complete": False},
            ),
            mock.patch.object(entry, "verify_frozen_manifests") as verify,
            mock.patch.object(entry, "load_all_rows") as load_rows,
        ):
            with self.assertRaises(entry.IncompleteTraining):
                entry.aggregate_and_write()
        verify.assert_not_called()
        load_rows.assert_not_called()

    def test_load_all_rows_rebinds_all_six_sweeps(self) -> None:
        observed = []

        def fake_binding(*, variant, checkpoint):
            return {"identity": (variant, checkpoint)}

        def fake_validate(path, *, variant, checkpoint, binding):
            observed.append(
                (Path(path), variant, checkpoint, dict(binding))
            )
            return {
                "variant": variant,
                "checkpoint": checkpoint,
            }

        with (
            mock.patch.object(
                entry,
                "current_sweep_binding",
                side_effect=fake_binding,
            ) as bind,
            mock.patch.object(
                entry,
                "validate_existing_sweep",
                side_effect=fake_validate,
            ) as validate,
        ):
            rows = entry.load_all_rows()
        self.assertEqual(len(rows), 6)
        self.assertEqual(bind.call_count, 6)
        self.assertEqual(validate.call_count, 6)
        self.assertEqual(
            {
                (variant, checkpoint)
                for _, variant, checkpoint, _ in observed
            },
            {
                (variant, checkpoint)
                for variant in (
                    entry.VARIANT_OFF,
                    entry.VARIANT_ON,
                    "baseline_sctransnet",
                )
                for checkpoint in entry.CHECKPOINTS
            },
        )
        for _, variant, checkpoint, binding in observed:
            self.assertEqual(binding["identity"], (variant, checkpoint))

    def test_invalid_existing_sweep_is_moved_then_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "pd_fa_sweep_best.pth.json"
            output.write_text("{incomplete", encoding="utf-8")
            binding = ner_binding(entry.VARIANT_OFF, "best.pth.tar")

            def fake_subprocess(*args, **kwargs):
                output.write_text('{"new":"complete"}', encoding="utf-8")
                return mock.Mock(returncode=0)

            with (
                mock.patch.object(
                    entry,
                    "evaluation_command",
                    return_value=(
                        ["/venv/python", "--device", "cpu"],
                        {},
                        output,
                    ),
                ),
                mock.patch.object(
                    entry,
                    "current_sweep_binding",
                    return_value=binding,
                ),
                mock.patch.object(
                    entry,
                    "validate_existing_sweep",
                    side_effect=[ValueError("bad existing"), {"valid": True}],
                ),
                mock.patch.object(
                    entry.subprocess,
                    "run",
                    side_effect=fake_subprocess,
                ),
            ):
                result = entry._run_evaluation(
                    variant=entry.VARIANT_OFF,
                    checkpoint="best.pth.tar",
                    python="/venv/python",
                    device_mode="cpu",
                )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(output.read_text(encoding="utf-8"), '{"new":"complete"}')
            rejected = list((root / "rejected_postprocess").glob("*"))
            self.assertEqual(len(rejected), 1)
            self.assertEqual(
                (rejected[0] / output.name).read_text(encoding="utf-8"),
                "{incomplete",
            )
            self.assertTrue((rejected[0] / "reason.json").is_file())

    def test_failed_evaluator_partial_output_is_moved_aside(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "pd_fa_sweep_best.pth.json"
            binding = ner_binding(entry.VARIANT_OFF, "best.pth.tar")

            def fail_after_partial(*args, **kwargs):
                output.write_text("{partial", encoding="utf-8")
                raise subprocess.CalledProcessError(1, args[0])

            with (
                mock.patch.object(
                    entry,
                    "evaluation_command",
                    return_value=(
                        ["/venv/python", "--device", "cpu"],
                        {},
                        output,
                    ),
                ),
                mock.patch.object(
                    entry,
                    "current_sweep_binding",
                    return_value=binding,
                ),
                mock.patch.object(
                    entry.subprocess,
                    "run",
                    side_effect=fail_after_partial,
                ),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    entry._run_evaluation(
                        variant=entry.VARIANT_OFF,
                        checkpoint="best.pth.tar",
                        python="/venv/python",
                        device_mode="cpu",
                    )
            self.assertFalse(output.exists())
            rejected = list((root / "rejected_postprocess").glob("*"))
            self.assertEqual(len(rejected), 1)
            self.assertEqual(
                (rejected[0] / output.name).read_text(encoding="utf-8"),
                "{partial",
            )
            self.assertTrue((rejected[0] / "reason.json").is_file())

    def test_final_report_fills_partial_pair_and_moves_conflict(self) -> None:
        with mock.patch.object(entry, "sha256_file", return_value="a" * 64):
            report = entry.build_report(
                successful_matrix(),
                lock_bindings={"training_source_lock_sha256": "b" * 64},
                baseline_contract={"same_split_hashes": True},
            )
        with tempfile.TemporaryDirectory() as directory:
            comparison = Path(directory)
            json_path = comparison / "comparison.json"
            markdown_path = comparison / "comparison.md"
            marker_path = comparison / "COMPLETE.json"
            expected_json = entry._canonical_bytes(report)
            json_path.write_bytes(expected_json)
            with (
                mock.patch.object(entry, "COMPARISON_DIR", comparison),
                mock.patch.object(entry, "JSON_OUTPUT", json_path),
                mock.patch.object(entry, "MARKDOWN_OUTPUT", markdown_path),
                mock.patch.object(entry, "COMPLETE_MARKER", marker_path),
            ):
                paths = entry.write_report(report)
                self.assertEqual(paths, (json_path, markdown_path, marker_path))
                self.assertTrue(markdown_path.is_file())
                self.assertTrue(marker_path.is_file())

                markdown_path.write_text("partial/wrong", encoding="utf-8")
                entry.write_report(report)
            self.assertEqual(json_path.read_bytes(), expected_json)
            self.assertEqual(
                markdown_path.read_text(encoding="utf-8"),
                entry.render_markdown(report),
            )
            rejected = list((comparison / "rejected_postprocess").glob("*"))
            self.assertEqual(len(rejected), 1)
            self.assertTrue((rejected[0] / "comparison.json").is_file())
            self.assertTrue((rejected[0] / "comparison.md").is_file())
            self.assertTrue((rejected[0] / "COMPLETE.json").is_file())

    def test_wrong_completion_marker_is_replaced_without_moving_reports(self) -> None:
        with mock.patch.object(entry, "sha256_file", return_value="a" * 64):
            report = entry.build_report(
                successful_matrix(),
                lock_bindings={"training_source_lock_sha256": "b" * 64},
                baseline_contract={"same_split_hashes": True},
            )
        with tempfile.TemporaryDirectory() as directory:
            comparison = Path(directory)
            json_path = comparison / "comparison.json"
            markdown_path = comparison / "comparison.md"
            marker_path = comparison / "COMPLETE.json"
            with (
                mock.patch.object(entry, "COMPARISON_DIR", comparison),
                mock.patch.object(entry, "JSON_OUTPUT", json_path),
                mock.patch.object(entry, "MARKDOWN_OUTPUT", markdown_path),
                mock.patch.object(entry, "COMPLETE_MARKER", marker_path),
            ):
                entry.write_report(report)
                original_json = json_path.read_bytes()
                original_markdown = markdown_path.read_bytes()
                marker_path.write_text('{"status":"partial"}', encoding="utf-8")
                entry.write_report(report)
            self.assertEqual(json_path.read_bytes(), original_json)
            self.assertEqual(markdown_path.read_bytes(), original_markdown)
            rejected = list((comparison / "rejected_postprocess").glob("*"))
            self.assertEqual(len(rejected), 1)
            self.assertEqual(
                (rejected[0] / "COMPLETE.json").read_text(encoding="utf-8"),
                '{"status":"partial"}',
            )
            self.assertFalse((rejected[0] / "comparison.json").exists())
            self.assertFalse((rejected[0] / "comparison.md").exists())

    def test_full_baseline_contract_and_all_axis_mismatches(self) -> None:
        contract = entry._same_split_and_training_contract()
        for field in (
            "same_fixed_training_axes",
            "same_learning_rate_axes",
            "same_normalization",
            "same_optimizer",
            "same_loss",
            "same_selection_rules",
            "checkpoint_policies_semantically_aligned",
            "same_off_on_fixed_protocol",
            "same_split_hashes",
        ):
            self.assertIs(contract[field], True)
        self.assertFalse(
            contract[
                "endpoint_protocol_preregistered_before_historical_training"
            ]
        )

        real_load = entry.load_json
        baseline_protocol = entry.BASELINE_SOURCE_RUN / "protocol.json"
        off_protocol = entry.RUN_DIRS[entry.VARIANT_OFF] / "protocol.json"
        on_protocol = entry.RUN_DIRS[entry.VARIANT_ON] / "protocol.json"
        on_split = entry.RUN_DIRS[entry.VARIANT_ON] / "split.json"
        cases = (
            (
                "base lr",
                baseline_protocol,
                lambda value: value["arguments"].__setitem__(
                    "base_lr",
                    2e-3,
                ),
                "base_lr",
            ),
            (
                "minimum lr",
                on_protocol,
                lambda value: value["arguments"].__setitem__(
                    "min_lr",
                    2e-5,
                ),
                "min_lr",
            ),
            (
                "warmup",
                off_protocol,
                lambda value: value["arguments"].__setitem__(
                    "warmup_epochs",
                    11,
                ),
                "warmup_epochs",
            ),
            (
                "normalization",
                baseline_protocol,
                lambda value: value["normalization"].__setitem__(
                    "mean",
                    0.0,
                ),
                "normalization",
            ),
            (
                "optimizer",
                baseline_protocol,
                lambda value: value.__setitem__("optimizer", "different"),
                "optimizer",
            ),
            (
                "loss",
                baseline_protocol,
                lambda value: value.__setitem__("loss", "different"),
                "loss",
            ),
            (
                "primary selection",
                baseline_protocol,
                lambda value: value.__setitem__(
                    "primary_selection_rule",
                    ["different"],
                ),
                "primary_selection_rule",
            ),
            (
                "secondary selection",
                baseline_protocol,
                lambda value: value.__setitem__(
                    "secondary_selection_rule",
                    ["different"],
                ),
                "secondary_selection_rule",
            ),
            (
                "checkpoint policy",
                baseline_protocol,
                lambda value: value.__setitem__(
                    "checkpoint_policy",
                    "different",
                ),
                "checkpoint policy",
            ),
            (
                "relay split ids",
                on_split,
                lambda value: value["used_val_ids"].__setitem__(
                    0,
                    "different",
                ),
                "validation IDs",
            ),
            (
                "relay split hash",
                on_split,
                lambda value: value["hashes"].__setitem__(
                    "used_val_sha256",
                    "0" * 64,
                ),
                "split hashes",
            ),
        )
        for label, target, mutate, message in cases:
            with self.subTest(label=label):
                def changed_load(path):
                    value = copy.deepcopy(real_load(path))
                    if Path(path) == target:
                        mutate(value)
                    return value

                with mock.patch.object(
                    entry,
                    "load_json",
                    side_effect=changed_load,
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        entry._same_split_and_training_contract()


if __name__ == "__main__":
    unittest.main()
