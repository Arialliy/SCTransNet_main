from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import (
    postprocess_tpd_ner_v8_mprs_dch_v4_tail_aware_formal800 as subject,
)


def _fixed(
    *,
    matched: int = 189,
    fa: float = 0.0,
    miou: float = 0.96,
    tiny_matched: int = 39,
) -> dict:
    return {
        "threshold": 0.5,
        "pd": matched / subject.TARGET_COUNT,
        "fa": fa,
        "miou": miou,
        "false_objects_per_image": 0.0,
        "target_count": subject.TARGET_COUNT,
        "matched_target_count": matched,
        "tiny_pd": tiny_matched / subject.TINY_TARGET_COUNT,
        "tiny_target_count": subject.TINY_TARGET_COUNT,
        "matched_tiny_target_count": tiny_matched,
        "niou": miou,
        "pixel_precision": 0.95,
        "pixel_recall": 0.95,
        "pixel_f1": 0.95,
    }


def _budgets(counts: list[int]) -> dict:
    return {
        key: {
            "threshold": 0.9,
            "pd": matched / subject.TARGET_COUNT,
            "fa": min(budget, 1e-7),
            "miou": 0.9,
            "false_objects_per_image": 0.0,
            "tiny_pd": 1.0,
            "tiny_target_count": subject.TINY_TARGET_COUNT,
            "matched_tiny_target_count": subject.TINY_TARGET_COUNT,
            "target_count": subject.TARGET_COUNT,
            "matched_target_count": matched,
        }
        for key, budget, matched in zip(
            subject.BUDGET_KEYS,
            subject.FA_BUDGETS,
            counts,
        )
    }


def _v4_row(
    checkpoint: str,
    *,
    counts: list[int] | None = None,
    fixed: dict | None = None,
) -> dict:
    role = subject.CHECKPOINT_ROLES[checkpoint]
    return {
        "source": "synthetic_v4_sweep",
        "variant": subject.V4_ON_VARIANT,
        "checkpoint": checkpoint,
        "checkpoint_role": role,
        "checkpoint_epoch": 100,
        "checkpoint_sha256": "a" * 64,
        "run_directory": "/synthetic/v4",
        "fixed_threshold_0_5": fixed or _fixed(),
        "pd_at_fa_budget": _budgets(
            counts or [188, 189, 189, 189, 189]
        ),
        "absolute_gate": None,
        "validation_split_sha256": subject.VALIDATION_SPLIT_SHA256,
        "sweep_binding": {
            "path": f"/synthetic/{checkpoint}.json",
            "sha256": "b" * 64,
        },
        "checkpoint_binding": {
            "path": f"/synthetic/{checkpoint}",
            "sha256": "a" * 64,
        },
        "evaluation_source_binding": {
            "training_source_lock": {
                "path": "/synthetic/lock.json",
                "sha256": "c" * 64,
            }
        },
        "run_id": (
            subject.V4_RUN_ID_PREFIX
            + "NUDT-SIRST:synthetic:seed-42:split-20260722"
        ),
    }


def _coverage(fixed: dict, budgets: dict) -> dict:
    return {
        "schema": subject.V4_FINAL_METRIC_COVERAGE_SCHEMA,
        "fixed_threshold_0_5": {
            field: copy.deepcopy(fixed[field])
            for field in subject.FINAL_COVERAGE_FIELDS
        },
        "fa_budget_points": {
            key: {
                field: copy.deepcopy(point[field])
                for field in subject.FINAL_COVERAGE_FIELDS
            }
            for key, point in budgets.items()
        },
        "required_metrics": [
            "pd",
            "fa",
            "miou",
            "false_objects_per_image",
            "tiny_pd",
        ],
        "fa_budgets": list(subject.FA_BUDGETS),
        "fixed_threshold_complete": True,
        "fa_budget_curve_complete": True,
        "official_test_accessed": False,
    }


def _write_synthetic_v4_sweep(
    root: Path,
    checkpoint: str,
) -> tuple[Path, Path, str]:
    run_dir = root / "run"
    run_dir.mkdir(parents=True)
    checkpoint_path = run_dir / checkpoint
    checkpoint_path.write_bytes(f"checkpoint:{checkpoint}".encode())
    checkpoint_sha = subject.sha256_file(checkpoint_path)

    source_lock = root / "source_lock.json"
    source_lock.write_text('{"synthetic": true}\n', encoding="utf-8")
    source_lock_sha = subject.sha256_file(source_lock)
    source_files = {}
    for name in (
        "evaluator",
        "shared_metric_core",
        "closed_interval_core",
        "determinism_core",
    ):
        path = root / f"{name}.py"
        path.write_text(f"# {name}\n", encoding="utf-8")
        source_files[name] = path
    source_binding = {
        "training_source_lock": {
            "path": str(source_lock),
            "sha256": source_lock_sha,
        },
        **{
            name: {
                "path": str(path),
                "sha256": subject.sha256_file(path),
            }
            for name, path in source_files.items()
        },
    }
    fixed = _fixed(fa=4e-6)
    strict = _fixed(matched=188, fa=0.0, miou=0.95)
    strict["threshold"] = 0.9
    last = _fixed(matched=0, fa=0.0, miou=0.0, tiny_matched=0)
    last["threshold"] = subject.LAST_FLOAT32_BELOW_ONE
    upper = copy.deepcopy(last)
    upper["threshold"] = subject.UPPER_BOUNDARY_THRESHOLD

    def raw_point(point: dict) -> dict:
        ready = copy.deepcopy(point)
        ready.update(
            {
                "predicted_object_count": point["matched_target_count"],
                "unmatched_predicted_object_count": 0,
                "valid_pixel_count": subject.VALID_PIXEL_COUNT,
            }
        )
        return ready

    points = [
        raw_point(fixed),
        raw_point(strict),
        raw_point(last),
        raw_point(upper),
    ]
    budgets = {
        "1e-06": copy.deepcopy(strict),
        "5e-06": copy.deepcopy(fixed),
        "1e-05": copy.deepcopy(fixed),
        "5e-05": copy.deepcopy(fixed),
        "0.0001": copy.deepcopy(fixed),
    }
    role = subject.CHECKPOINT_ROLES[checkpoint]
    run_id = (
        subject.V4_RUN_ID_PREFIX
        + "NUDT-SIRST:synthetic:seed-42:split-20260722"
    )
    determinism = {
        "relay_version": subject.V4_RELAY_VERSION,
        "required_control": subject.V1_OFF_VARIANT,
        "paired_gate_predecessor": subject.V2_ON_VARIANT,
        "structural_predecessor": subject.V3_ON_VARIANT,
        "ner_dc_offset_support_scope": subject.V4_DC_SCOPE,
        "dc_support_mode": "complement_tail",
        "dc_support_formula_stage4": "1",
        "dc_support_formula_stage3_2": "1-P",
        "tail_z_thresholds": copy.deepcopy(
            subject.V4_TAIL_THRESHOLDS
        ),
        "tail_z_thresholds_frozen": True,
        "target_protective_complement": True,
        "fresh_training": True,
        "v3_warm_start": False,
    }
    identity = {
        "run_id": run_id,
        "variant": subject.V4_ON_VARIANT,
        "dataset": subject.DATASET,
        "seed": subject.TRAINING_SEED,
        "split_seed": subject.SPLIT_SEED,
        "source_locks": {
            subject.V4_SOURCE_LOCK_KEY: source_lock_sha,
            "training_data": subject.TRAINING_DATA_SHA256,
        },
        "training_contract": {"determinism": determinism},
    }
    checkpoint_identity = {
        "variant": subject.V4_ON_VARIANT,
        "relay_version": subject.V4_RELAY_VERSION,
        "required_control": subject.V1_OFF_VARIANT,
        "paired_gate_predecessor": subject.V2_ON_VARIANT,
        "structural_predecessor": subject.V3_ON_VARIANT,
        "ner_dc_offset_support_scope": subject.V4_DC_SCOPE,
        "dc_support_mode": "complement_tail",
        "dc_support_formula_stage3_2": "1-P",
        "tail_z_thresholds": copy.deepcopy(
            subject.V4_TAIL_THRESHOLDS
        ),
        "formula_selection_decision": "COMPLEMENT_TAIL_SELECTED",
    }
    evaluator_contract = {
        "dataset": subject.DATASET,
        "formal_variant": subject.V4_ON_VARIANT,
        "required_control": subject.V1_OFF_VARIANT,
        "paired_gate_predecessor": subject.V2_ON_VARIANT,
        "structural_predecessor": subject.V3_ON_VARIANT,
        "training_seed": subject.TRAINING_SEED,
        "split_seed": subject.SPLIT_SEED,
        "expected_epochs": subject.EXPECTED_EPOCHS,
        "fixed_threshold": 0.5,
        "fa_budgets": list(subject.FA_BUDGETS),
        "official_test_accessed": False,
        "dc_support_mode": "complement_tail",
        "dc_support_formula_stage4": "1",
        "dc_support_formula_stage3_2": "1-P",
    }
    filename = (
        "pd_fa_sweep_best.pth.json"
        if checkpoint == "best.pth.tar"
        else "pd_fa_sweep_best_miou.pth.json"
    )
    output = run_dir / filename
    payload = {
        "schema": subject.V4_SWEEP_SCHEMA,
        "run_directory": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": 100,
        "checkpoint_role": role,
        "checkpoint_validation_metrics": {},
        "variant": subject.V4_ON_VARIANT,
        "dataset": subject.DATASET,
        "seed": subject.TRAINING_SEED,
        "split_seed": subject.SPLIT_SEED,
        "validation_count": subject.VALIDATION_COUNT,
        "validation_split_sha256": (
            subject.VALIDATION_SPLIT_SHA256
        ),
        "official_test_accessed": False,
        "match_radius": 3.0,
        "tiny_area": 9,
        "threshold_configuration": {
            "fa_budgets": list(subject.FA_BUDGETS)
        },
        "fixed_threshold_0_5": fixed,
        "best_points_under_fa_budget": budgets,
        "points": points,
        "threshold_provenance": {
            "total_unique_threshold_count": len(points),
            "score_count": subject.VALID_PIXEL_COUNT,
        },
        "run_identity": identity,
        "source_checkpoint_identity": checkpoint_identity,
        "evaluated_checkpoint_identity": {
            "filename": checkpoint,
            "role": role,
            "sha256": checkpoint_sha,
        },
        "artifact_identity_preflight_passed": True,
        "evaluation_source_binding": source_binding,
        "evaluator_contract": evaluator_contract,
        "final_metric_coverage": _coverage(fixed, budgets),
        "audit": {
            "expected_epochs": subject.EXPECTED_EPOCHS,
            "metrics_event_count": subject.EXPECTED_EPOCHS,
            "metrics_epoch_range": [1, subject.EXPECTED_EPOCHS],
            "summary_status": "complete",
            "selection_source": "internal_validation_only",
            "integrity_checks_passed": {
                "summary_complete": True,
                "metrics_complete_contiguous_finite": True,
                "checkpoint_role_epoch_metrics_consistent": True,
                "state_dict_strict_load": True,
            },
        },
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return output, source_lock, source_lock_sha


class V4TailAwarePostprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.historical = subject.validate_historical_authority()

    def test_historical_authority_has_exact_eight_rows(self) -> None:
        self.assertEqual(len(self.historical["rows"]), 8)
        self.assertEqual(
            self.historical["binding"]["aggregate"]["sha256"],
            subject.HISTORICAL_AGGREGATE_SHA256,
        )
        self.assertEqual(
            len(self.historical["binding"]["sweeps"]),
            8,
        )

    def test_plan_waits_for_two_checkpoint_bound_sweeps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing_sweeps = {
                checkpoint: Path(directory) / f"{checkpoint}.json"
                for checkpoint in subject.CHECKPOINTS
            }
            with mock.patch.object(
                subject,
                "DEFAULT_V4_SWEEPS",
                missing_sweeps,
            ):
                plan = subject.execution_plan()
        self.assertEqual(
            plan["status"],
            "waiting_for_two_v4_formal_sweeps",
        )
        self.assertFalse(plan["metrics_jsonl_final_metric_source"])
        self.assertEqual(
            tuple(plan["gate_components"]),
            subject.SIX_COMPONENT_NAMES,
        )
        self.assertFalse(plan["writes_performed"])

    def test_plan_is_ready_when_two_checkpoint_bound_sweeps_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ready_sweeps = {
                checkpoint: Path(directory) / f"{checkpoint}.json"
                for checkpoint in subject.CHECKPOINTS
            }
            for path in ready_sweeps.values():
                path.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                subject,
                "DEFAULT_V4_SWEEPS",
                ready_sweeps,
            ):
                plan = subject.execution_plan()
        self.assertEqual(plan["status"], "ready")
        self.assertTrue(
            all(
                entry["exists"]
                for entry in plan["metric_inputs"].values()
            )
        )
        self.assertFalse(plan["writes_performed"])

    def test_six_component_pass_and_v3_is_delta_only(self) -> None:
        rows = {
            checkpoint: _v4_row(checkpoint)
            for checkpoint in subject.CHECKPOINTS
        }
        snapshot = {"all-inputs": "d" * 64}
        report = subject.build_report(
            self.historical,
            rows,
            input_snapshot_before=snapshot,
            input_snapshot_after=snapshot,
        )
        self.assertTrue(report["aggregate_full_model_gate_passed"])
        self.assertEqual(report["decision"], "NER_V4_GATE_PASS")
        self.assertTrue(report["v4_tail_aware_accepted"])
        self.assertTrue(report["next_model_stage_authorized"])
        self.assertTrue(all(report["six_component_gate"].values()))
        self.assertFalse(
            report["claim_boundary"][
                "v3_delta_affects_six_component_gate"
            ]
        )
        self.assertFalse(
            report["metric_provenance"][
                "training_metrics_jsonl_used_as_final_metric_source"
            ]
        )
        self.assertEqual(report["row_count"], 10)

    def test_absolute_failure_forces_return_to_optimization(self) -> None:
        rows = {
            checkpoint: _v4_row(checkpoint)
            for checkpoint in subject.CHECKPOINTS
        }
        rows["best.pth.tar"]["fixed_threshold_0_5"] = _fixed(
            fa=2e-6,
        )
        snapshot = {"all-inputs": "e" * 64}
        report = subject.build_report(
            self.historical,
            rows,
            input_snapshot_before=snapshot,
            input_snapshot_after=snapshot,
        )
        self.assertFalse(
            report["six_component_gate"]["pd_primary_absolute"]
        )
        self.assertEqual(
            report["decision"],
            "RETURN_TO_MODEL_OPTIMIZATION",
        )
        self.assertFalse(report["next_model_stage_authorized"])

    def test_paired_gate_uses_matched_counts_on_five_budgets(self) -> None:
        reference = copy.deepcopy(
            self.historical["rows"][
                (subject.V2_ON_VARIANT, "best.pth.tar")
            ]
        )
        candidate = _v4_row(
            "best.pth.tar",
            counts=[0, 0, 0, 0, 189],
        )
        gate = subject.paired_gate_assessment(
            reference,
            candidate,
            reference_variant=subject.V2_ON_VARIANT,
        )
        self.assertEqual(gate["non_inferior_budget_count"], 2)
        self.assertEqual(gate["strictly_better_budget_count"], 1)
        self.assertFalse(gate["passed"])

    def test_budget_normalizer_rejects_point_above_budget(self) -> None:
        points = _budgets([188, 188, 188, 188, 188])
        points["1e-06"]["fa"] = 2e-6
        with self.assertRaisesRegex(ValueError, "exceeds Fa"):
            subject.normalize_budgets(points, label="fixture")

    def test_training_metrics_jsonl_cannot_be_a_final_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text('{"epoch": 800}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a JSON sweep"):
                subject.load_json(path)

    def test_synthetic_v4_sweep_passes_full_identity_and_hash_checks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep, source_lock, source_lock_sha = (
                _write_synthetic_v4_sweep(root, "best.pth.tar")
            )
            row = subject.validate_v4_sweep(
                sweep,
                checkpoint="best.pth.tar",
                expected_run_dir=sweep.parent,
                source_lock_path=source_lock,
                source_lock_sha256=source_lock_sha,
            )
            self.assertEqual(row["variant"], subject.V4_ON_VARIANT)
            self.assertEqual(
                row["checkpoint_role"],
                "best_validation_pd_primary",
            )
            self.assertEqual(
                row["pd_at_fa_budget"]["1e-06"][
                    "matched_target_count"
                ],
                188,
            )

    def test_v4_sweep_recomputes_budget_choice_from_raw_points(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep, source_lock, source_lock_sha = (
                _write_synthetic_v4_sweep(root, "best.pth.tar")
            )
            payload = json.loads(sweep.read_text(encoding="utf-8"))
            payload["best_points_under_fa_budget"]["1e-06"] = copy.deepcopy(
                payload["points"][-1]
            )
            sweep.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "best raw sweep point"):
                subject.validate_v4_sweep(
                    sweep,
                    checkpoint="best.pth.tar",
                    expected_run_dir=sweep.parent,
                    source_lock_path=source_lock,
                    source_lock_sha256=source_lock_sha,
                )

    def test_legacy_final_coverage_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep, source_lock, source_lock_sha = (
                _write_synthetic_v4_sweep(root, "best.pth.tar")
            )
            payload = json.loads(sweep.read_text(encoding="utf-8"))
            payload["final_metric_coverage"] = {
                "schema": "legacy_synthetic_coverage",
                "fixed_threshold": 0.5,
                "fixed_threshold_0_5": payload["fixed_threshold_0_5"],
                "pd_at_fa_budget": payload[
                    "best_points_under_fa_budget"
                ],
                "all_required_metrics_present": True,
            }
            sweep.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage schema"):
                subject.validate_v4_sweep(
                    sweep,
                    checkpoint="best.pth.tar",
                    expected_run_dir=sweep.parent,
                    source_lock_path=source_lock,
                    source_lock_sha256=source_lock_sha,
                )

    def test_v4_sweep_rejects_wrong_role_and_changed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sweep, source_lock, source_lock_sha = (
                _write_synthetic_v4_sweep(root, "best.pth.tar")
            )
            payload = json.loads(sweep.read_text(encoding="utf-8"))
            payload["checkpoint_role"] = (
                "best_validation_miou_secondary"
            )
            sweep.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint_role"):
                subject.validate_v4_sweep(
                    sweep,
                    checkpoint="best.pth.tar",
                    expected_run_dir=sweep.parent,
                    source_lock_path=source_lock,
                    source_lock_sha256=source_lock_sha,
                )

            payload["checkpoint_role"] = (
                "best_validation_pd_primary"
            )
            sweep.write_text(json.dumps(payload), encoding="utf-8")
            (sweep.parent / "best.pth.tar").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint SHA"):
                subject.validate_v4_sweep(
                    sweep,
                    checkpoint="best.pth.tar",
                    expected_run_dir=sweep.parent,
                    source_lock_path=source_lock,
                    source_lock_sha256=source_lock_sha,
                )

    def test_publication_is_write_once_and_marker_binds_outputs(self) -> None:
        rows = {
            checkpoint: _v4_row(checkpoint)
            for checkpoint in subject.CHECKPOINTS
        }
        snapshot = {"all-inputs": "f" * 64}
        report = subject.build_report(
            self.historical,
            rows,
            input_snapshot_before=snapshot,
            input_snapshot_after=snapshot,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "comparison"
            json_path, markdown_path, marker_path = (
                subject.publish_report(report, output_dir=output_dir)
            )
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            self.assertEqual(
                marker["outputs"][json_path.name],
                subject.sha256_file(json_path),
            )
            self.assertEqual(
                marker["outputs"][markdown_path.name],
                subject.sha256_file(markdown_path),
            )
            with self.assertRaisesRegex(ValueError, "overwrite"):
                subject.publish_report(report, output_dir=output_dir)

    def test_changed_input_snapshot_is_rejected(self) -> None:
        rows = {
            checkpoint: _v4_row(checkpoint)
            for checkpoint in subject.CHECKPOINTS
        }
        with self.assertRaisesRegex(ValueError, "changed"):
            subject.build_report(
                self.historical,
                rows,
                input_snapshot_before={"input": "1" * 64},
                input_snapshot_after={"input": "2" * 64},
            )


if __name__ == "__main__":
    unittest.main()
