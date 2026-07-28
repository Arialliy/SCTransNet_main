from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import (
    freeze_tpd_ner_v8_mprs_dch_v3_dc_knockout_source_lock as freezer,
)
from experiments import (
    postprocess_tpd_ner_v8_mprs_dch_v3_dc_knockout as subject,
)
from experiments import (
    tpd_ner_v8_mprs_dch_v3_dc_knockout_spec as spec,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _raw_point(
    threshold: float,
    *,
    matched: int,
    tiny_matched: int,
    fa: float,
) -> dict[str, object]:
    return {
        "val_loss": 0.125,
        "miou": matched / spec.TARGET_COUNT,
        "niou": matched / spec.TARGET_COUNT,
        "pixel_precision": 0.9 if matched else 0.0,
        "pixel_recall": matched / spec.TARGET_COUNT,
        "pixel_f1": 0.9 if matched else 0.0,
        "pd": matched / spec.TARGET_COUNT,
        "tiny_pd": tiny_matched / spec.TINY_TARGET_COUNT,
        "fa": fa,
        "false_objects_per_image": fa * 1000.0,
        "target_count": spec.TARGET_COUNT,
        "matched_target_count": matched,
        "tiny_target_count": spec.TINY_TARGET_COUNT,
        "matched_tiny_target_count": tiny_matched,
        "predicted_object_count": matched + (1 if fa else 0),
        "unmatched_predicted_object_count": 1 if fa else 0,
        "valid_pixel_count": 1000000,
        "threshold": threshold,
    }


def _points() -> list[dict[str, object]]:
    return [
        _raw_point(0.1, matched=188, tiny_matched=39, fa=5e-7),
        _raw_point(0.5, matched=187, tiny_matched=38, fa=1e-6),
        _raw_point(
            subject.LAST_FLOAT32_BELOW_ONE,
            matched=0,
            tiny_matched=0,
            fa=0.0,
        ),
        _raw_point(1.0, matched=0, tiny_matched=0, fa=0.0),
    ]


def _fixed(point: dict[str, object]) -> dict[str, object]:
    return {
        name: copy.deepcopy(point[name])
        for name in spec.FIXED_THRESHOLD_FIELDS
    }


def _normalized_budgets(
    point: dict[str, object],
) -> dict[str, dict[str, object]]:
    return {
        key: {
            "budget": budget,
            "pd": point["pd"],
            "achieved_fa": point["fa"],
            "threshold": point["threshold"],
            "matched_target_count": point["matched_target_count"],
            "target_count": point["target_count"],
        }
        for budget, key in zip(spec.FA_BUDGETS, spec.BUDGET_KEYS)
    }


def _offset_records(
    values: dict[str, float],
) -> dict[str, dict[str, object]]:
    return {
        key: {
            "shape": [1],
            "dtype": "float32",
            "value": value,
            "tensor_sha256": _digest(f"{key}:{value}"),
        }
        for key, value in values.items()
    }


def _source_binding(lock_path: Path) -> dict[str, object]:
    return {
        "schema": freezer.SOURCE_BINDING_SCHEMA,
        "diagnostic_source_lock": {
            "path": str(lock_path.resolve()),
            "sha256": subject.sha256_file(lock_path),
        },
        "knockout_spec_sha256": spec.specification_sha256(),
        "formal_artifact_snapshot_sha256": _digest("formal snapshot"),
        "formal_training_source_lock_sha256": _digest("formal training lock"),
        "formal_acceptance_source_lock_sha256": _digest(
            "formal acceptance lock"
        ),
    }


def _evaluation(
    *,
    checkpoint: str,
    checkpoint_sha256: str,
    state_sha256: str,
    non_dc_sha256: str,
    mode: str,
    original_records: dict[str, dict[str, object]],
) -> dict[str, object]:
    points = _points()
    fixed = points[1]
    best = points[0]
    evaluated_values = {
        key: (
            0.0
            if key in spec.KNOCKOUT_ZERO_KEYS[mode]
            else float(record["value"])
        )
        for key, record in original_records.items()
    }
    budgets = _normalized_budgets(best)
    return {
        "schema": (
            "sctransnet_tpd_ner_v8_mprs_dch_v3_"
            "dc_knockout_state_transform_v1"
        ),
        "status": "complete",
        "knockout_mode": mode,
        "zeroed_state_keys": list(spec.KNOCKOUT_ZERO_KEYS[mode]),
        "effective_changed_state_keys": list(
            spec.KNOCKOUT_ZERO_KEYS[mode]
        ),
        "source_dc_offsets": copy.deepcopy(original_records),
        "evaluated_dc_offsets": _offset_records(evaluated_values),
        "source_state_dict_sha256": state_sha256,
        "evaluated_state_dict_sha256": _digest(
            f"{checkpoint}:{mode}:state"
        ),
        "source_checkpoint_sha256_before": checkpoint_sha256,
        "source_checkpoint_sha256_after": checkpoint_sha256,
        "non_dc_state_sha256_before": non_dc_sha256,
        "non_dc_state_sha256_after": non_dc_sha256,
        "validation_count": spec.VALIDATION_COUNT,
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "formal_gate_eligible": False,
        "threshold_configuration": {
            "include_zero": False,
            "include_last_float32_below_one": True,
            "include_one": True,
        },
        "threshold_provenance": {
            "total_unique_threshold_count": len(points),
        },
        "fixed_threshold_0_5": copy.deepcopy(fixed),
        "best_points_under_fa_budget": {
            key: copy.deepcopy(best) for key in spec.BUDGET_KEYS
        },
        "points": points,
        "final_metric_coverage": {
            "schema": (
                "sctransnet_tpd_ner_v8_mprs_dch_v3_"
                "dc_knockout_final_metric_coverage_v1"
            ),
            "fixed_threshold": 0.5,
            "required_fixed_threshold_fields": list(
                spec.FIXED_THRESHOLD_FIELDS
            ),
            "fixed_threshold_0_5": _fixed(fixed),
            "required_fa_budget_keys": list(spec.BUDGET_KEYS),
            "pd_at_fa_budget": budgets,
            "all_required_metrics_present": True,
        },
        "audit": {
            "source_state_strict_load": True,
            "only_requested_dc_state_keys_changed": True,
            "requested_dc_state_keys_zero": True,
            "non_zeroed_dc_offsets_preserved": True,
            "source_state_unchanged": True,
            "transformed_state_stable_during_inference": True,
            "non_dc_state_unchanged": True,
            "closed_interval_validated": True,
            "derived_checkpoint_written": False,
            "gpu_memory": {
                "device_type": "cpu",
                "max_memory_allocated_bytes": None,
                "max_memory_reserved_bytes": None,
            },
        },
    }


def _sweep_payload(
    checkpoint: str,
    source_binding: dict[str, object],
) -> dict[str, object]:
    checkpoint_sha256 = _digest(f"{checkpoint}:checkpoint")
    state_sha256 = _digest(f"{checkpoint}:source state")
    non_dc_sha256 = _digest(f"{checkpoint}:non-dc state")
    original = _offset_records(
        {
            "tpd_ner.dc_offsets.4": 0.4,
            "tpd_ner.dc_offsets.3": -0.3,
            "tpd_ner.dc_offsets.2": 0.2,
        }
    )
    checkpoint_identity = {"fixture": checkpoint}
    return {
        "schema": spec.EVALUATION_SCHEMA,
        "status": "complete",
        "artifact_kind": spec.ARTIFACT_KIND,
        "scope": "evaluation_only_same_checkpoint_counterfactual",
        "diagnostic_only": True,
        "affects_formal_gate": False,
        "formal_decision_authority": False,
        "formal_gate_eligible": False,
        "formal_gate_components": [],
        "official_test_accessed": False,
        "dataset": spec.DATASET,
        "variant": spec.VARIANT,
        "training_seed": spec.TRAINING_SEED,
        "split_seed": spec.SPLIT_SEED,
        "expected_epochs": spec.EXPECTED_EPOCHS,
        "validation_count": spec.VALIDATION_COUNT,
        "validation_split_sha256": _digest("validation split"),
        "run_directory": str(spec.FORMAL_RUN_DIR.resolve()),
        "run_identity": {
            "dataset": spec.DATASET,
            "variant": spec.VARIANT,
            "seed": spec.TRAINING_SEED,
            "split_seed": spec.SPLIT_SEED,
        },
        "checkpoint_filename": checkpoint,
        "checkpoint_role": spec.CHECKPOINT_ROLES[checkpoint],
        "checkpoint_epoch": spec.EXPECTED_EPOCHS,
        "checkpoint_validation_metrics": {"fixture": True},
        "source_checkpoint_identity": checkpoint_identity,
        "source_checkpoint": {
            "path": str((spec.FORMAL_RUN_DIR / checkpoint).resolve()),
            "filename": checkpoint,
            "role": spec.CHECKPOINT_ROLES[checkpoint],
            "epoch": spec.EXPECTED_EPOCHS,
            "sha256": checkpoint_sha256,
            "state_dict_sha256": state_sha256,
            "checkpoint_identity": checkpoint_identity,
            "validation_metrics": {"fixture": True},
        },
        "source_state_dict_sha256": state_sha256,
        "source_non_dc_state_sha256": non_dc_sha256,
        "original_dc_offsets": original,
        "knockout_modes": list(spec.KNOCKOUT_MODES),
        "knockout_specification": spec.fixed_specification(),
        "knockout_spec_sha256": spec.specification_sha256(),
        "source_binding": copy.deepcopy(source_binding),
        "diagnostic_source_lock_sha256": source_binding[
            "diagnostic_source_lock"
        ]["sha256"],
        "threshold_contract": spec.threshold_contract(),
        "evaluations": [
            _evaluation(
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                state_sha256=state_sha256,
                non_dc_sha256=non_dc_sha256,
                mode=mode,
                original_records=original,
            )
            for mode in spec.KNOCKOUT_MODES
        ],
        "artifact_sha256": {
            "diagnostic_evaluator": _digest("diagnostic evaluator"),
            "source_checkpoint": checkpoint_sha256,
        },
        "audit": {
            "source_checkpoint_sha256_before": checkpoint_sha256,
            "source_checkpoint_sha256_after": checkpoint_sha256,
            "source_state_dict_sha256_before": state_sha256,
            "source_state_dict_sha256_after": state_sha256,
            "non_dc_state_sha256_before": non_dc_sha256,
            "non_dc_state_sha256_after": non_dc_sha256,
            "diagnostic_source_binding_before": copy.deepcopy(
                source_binding
            ),
            "diagnostic_source_binding_after": copy.deepcopy(
                source_binding
            ),
            "formal_artifacts_read_only": True,
            "formal_artifacts_unchanged": True,
            "all_modes_from_pristine_source_state": True,
            "modes_evaluated_sequentially": True,
            "derived_checkpoint_written": False,
            "output_overwrite_forbidden": True,
        },
    }


def _formal_report() -> dict[str, object]:
    fixed_point = _raw_point(
        0.5,
        matched=189,
        tiny_matched=39,
        fa=5e-7,
    )
    budget_point = _raw_point(
        0.25,
        matched=189,
        tiny_matched=39,
        fa=2e-7,
    )
    rows = [
        {
            "variant": spec.VARIANT,
            "checkpoint_role": role,
            "fixed_threshold_0_5": _fixed(fixed_point),
            "pd_at_fa_budget": _normalized_budgets(budget_point),
        }
        for role in spec.CHECKPOINT_ROLES.values()
    ]
    return {
        "schema": "fixture_formal_report_v1",
        "status": "complete",
        "decision": freezer.EXPECTED_FORMAL_DECISION,
        "aggregate_full_model_gate_passed": False,
        "row_count": 8,
        "dataset": spec.DATASET,
        "training_seed": spec.TRAINING_SEED,
        "split_seed": spec.SPLIT_SEED,
        "comparison_contract": {
            "selection_contract_repair": {
                "repair_id": freezer.FORMAL_REPAIR_ID,
                "each_variant_uses_own_selected_checkpoints": True,
            },
        },
        "rows": rows,
    }


def _source_lock_payload(formal_report_path: Path) -> dict[str, object]:
    return {
        "formal_artifact_binding": {
            "snapshot_sha256": _digest("formal snapshot"),
            "formal_completion_marker": {
                "sha256": _digest("formal completion marker"),
            },
            "formal_aggregate_json": {
                "path": str(formal_report_path.resolve()),
                "sha256": subject.sha256_file(formal_report_path),
            },
            "formal_selection_contract_repair": {
                "repair_id": freezer.FORMAL_REPAIR_ID,
                "authority": (
                    "versioned_selection_contract_repair_v1_only"
                ),
                "each_variant_uses_own_selected_checkpoints": True,
                "formal_aggregate_decision": (
                    freezer.EXPECTED_FORMAL_DECISION
                ),
                "aggregate_full_model_gate_passed": False,
            },
            "formal_training_source_lock": {
                "sha256": _digest("formal training lock"),
            },
            "formal_acceptance_source_lock": {
                "sha256": _digest("formal acceptance lock"),
            },
        },
    }


class CheckpointSweepValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.lock_path = self.root / "diagnostic_source_lock.json"
        _write_json(self.lock_path, {"fixture": "diagnostic source lock"})
        self.binding = _source_binding(self.lock_path)
        self.checkpoint = spec.CHECKPOINTS[0]
        self.path = self.root / "sweep.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _validate(self, payload: dict[str, object]) -> list[dict[str, object]]:
        _write_json(self.path, payload)
        with mock.patch.object(
            subject,
            "_deep_validate_evaluator_output",
            side_effect=lambda payload, **_kwargs: payload,
        ):
            return subject.validate_checkpoint_sweep(
                self.path,
                checkpoint=self.checkpoint,
                expected_source_binding=self.binding,
            )

    def test_expands_one_checkpoint_to_four_ordered_rows(self) -> None:
        rows = self._validate(
            _sweep_payload(self.checkpoint, self.binding)
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            [row["knockout_mode"] for row in rows],
            list(spec.KNOCKOUT_MODES),
        )
        for row, mode in zip(rows, spec.KNOCKOUT_MODES):
            self.assertEqual(
                row["zeroed_state_keys"],
                list(spec.KNOCKOUT_ZERO_KEYS[mode]),
            )
            self.assertEqual(row["raw_point_count"], 4)
            self.assertEqual(
                set(row["pd_at_fa_budget"]),
                set(spec.BUDGET_KEYS),
            )

    def test_rejects_formal_decision_field(self) -> None:
        payload = _sweep_payload(self.checkpoint, self.binding)
        payload["decision"] = "pass"
        with self.assertRaisesRegex(ValueError, "formal decision"):
            self._validate(payload)

    def test_rejects_nonformal_zero_threshold(self) -> None:
        payload = _sweep_payload(self.checkpoint, self.binding)
        evaluation = payload["evaluations"][0]
        zero = _raw_point(
            0.0,
            matched=188,
            tiny_matched=39,
            fa=5e-7,
        )
        evaluation["points"].insert(0, zero)
        evaluation["threshold_provenance"][
            "total_unique_threshold_count"
        ] += 1
        with self.assertRaisesRegex(ValueError, "threshold 0"):
            self._validate(payload)

    def test_rejects_changed_nonrequested_dc_offset(self) -> None:
        payload = _sweep_payload(self.checkpoint, self.binding)
        evaluation = payload["evaluations"][1]
        preserved_key = "tpd_ner.dc_offsets.3"
        evaluation["evaluated_dc_offsets"][preserved_key]["value"] = 9.0
        with self.assertRaisesRegex(ValueError, "intervention differs"):
            self._validate(payload)

    def test_deep_validator_recomputes_artifact_digest_registry(self) -> None:
        payload = _sweep_payload(self.checkpoint, self.binding)
        expected = copy.deepcopy(payload["artifact_sha256"])
        payload["artifact_sha256"]["diagnostic_evaluator"] = _digest(
            "tampered evaluator"
        )
        checkpoint_path = spec.FORMAL_RUN_DIR / self.checkpoint
        with (
            mock.patch.object(
                subject.knockout_eval.formal_evaluator,
                "validate_run_artifacts",
                return_value={"fixture": "artifact audit"},
            ),
            mock.patch.object(
                subject,
                "_regular_file",
                return_value=checkpoint_path,
            ),
            mock.patch.object(
                subject.torch,
                "load",
                return_value={"state_dict": {}},
            ),
            mock.patch.object(
                subject.knockout_eval,
                "validate_evaluation_payload",
                return_value=payload,
            ),
            mock.patch.object(
                subject.knockout_eval,
                "_artifact_hashes",
                return_value=expected,
            ) as recompute,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "artifact SHA registry differs",
            ):
                subject._deep_validate_evaluator_output(
                    payload,
                    checkpoint=self.checkpoint,
                    expected_source_binding=self.binding,
                )
        recompute.assert_called_once_with(
            checkpoint_path=checkpoint_path,
            artifact_audit={"fixture": "artifact audit"},
            source_binding=self.binding,
        )


class AggregatePublicationTests(unittest.TestCase):
    def test_eight_row_package_is_idempotent_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "diagnostic-results"
            source_lock_path = root / "diagnostic-source-lock.json"
            formal_report_path = root / "formal-report.json"
            _write_json(source_lock_path, {"fixture": "source lock"})
            _write_json(formal_report_path, _formal_report())
            binding = _source_binding(source_lock_path)
            for checkpoint in spec.CHECKPOINTS:
                _write_json(
                    spec.sweep_path(checkpoint, output_root),
                    _sweep_payload(checkpoint, binding),
                )
            lock_payload = _source_lock_payload(formal_report_path)
            with (
                mock.patch.object(
                    subject.freezer,
                    "verify_source_lock",
                    return_value=lock_payload,
                ),
                mock.patch.object(
                    subject.freezer,
                    "current_source_binding",
                    return_value=binding,
                ),
                mock.patch.object(
                    subject,
                    "_deep_validate_evaluator_output",
                    side_effect=lambda payload, **_kwargs: payload,
                ),
                mock.patch.object(
                    subject,
                    "DEFAULT_FORMAL_REPORT",
                    formal_report_path,
                ),
            ):
                report, paths = subject.aggregate_and_write(
                    output_root=output_root,
                    source_lock_path=source_lock_path,
                    formal_report_path=formal_report_path,
                )
                self.assertEqual(report["row_count"], 8)
                self.assertEqual(
                    [
                        (row["checkpoint"], row["knockout_mode"])
                        for row in report["rows"]
                    ],
                    [
                        (row["checkpoint"], row["knockout_mode"])
                        for row in spec.matrix_rows()
                    ],
                )
                self.assertNotIn("decision", report)
                self.assertFalse(report["affects_formal_gate"])
                repair_binding = report["source_binding"][
                    "formal_selection_contract_repair"
                ]
                self.assertEqual(
                    repair_binding["authority"],
                    "versioned_selection_contract_repair_v1_only",
                )
                self.assertTrue(
                    repair_binding[
                        "each_variant_uses_own_selected_checkpoints"
                    ]
                )
                self.assertEqual(
                    report["rows"][0][
                        "signed_delta_knockout_minus_learned"
                    ]["direction"],
                    "knockout_minus_same_role_learned_v3",
                )
                marker = subject.inspect_complete(
                    output_root=output_root,
                    source_lock_path=source_lock_path,
                )
                self.assertIsNotNone(marker)
                self.assertEqual(
                    set(marker["sweep_sha256"]),
                    set(spec.CHECKPOINTS),
                )
                self.assertEqual(
                    marker[
                        "formal_selection_contract_repair_sha256"
                    ],
                    spec.canonical_sha256(repair_binding),
                )

                second_report, second_paths = subject.aggregate_and_write(
                    output_root=output_root,
                    source_lock_path=source_lock_path,
                    formal_report_path=formal_report_path,
                )
                self.assertEqual(second_report, report)
                self.assertEqual(second_paths, paths)

                source_lock_bytes = source_lock_path.read_bytes()
                source_lock_path.write_bytes(source_lock_bytes + b" ")
                with self.assertRaisesRegex(
                    ValueError,
                    "source-lock SHA differs",
                ):
                    subject.inspect_complete(
                        output_root=output_root,
                        source_lock_path=source_lock_path,
                    )
                source_lock_path.write_bytes(source_lock_bytes)

                sweep_path = spec.sweep_path(
                    spec.CHECKPOINTS[0],
                    output_root,
                )
                sweep_bytes = sweep_path.read_bytes()
                sweep_path.write_bytes(sweep_bytes + b" ")
                with self.assertRaisesRegex(
                    ValueError,
                    "sweep binding differs",
                ):
                    subject.inspect_complete(
                        output_root=output_root,
                        source_lock_path=source_lock_path,
                    )
                sweep_path.write_bytes(sweep_bytes)

                formal_marker = lock_payload["formal_artifact_binding"][
                    "formal_completion_marker"
                ]
                formal_marker_sha = formal_marker["sha256"]
                formal_marker["sha256"] = _digest("changed formal marker")
                with self.assertRaisesRegex(
                    ValueError,
                    "formal marker binding differs",
                ):
                    subject.inspect_complete(
                        output_root=output_root,
                        source_lock_path=source_lock_path,
                    )
                formal_marker["sha256"] = formal_marker_sha

                formal_repair = lock_payload["formal_artifact_binding"][
                    "formal_selection_contract_repair"
                ]
                formal_repair["each_variant_uses_own_selected_checkpoints"] = (
                    False
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "selection-contract repair binding differs",
                ):
                    subject.inspect_complete(
                        output_root=output_root,
                        source_lock_path=source_lock_path,
                    )
                formal_repair["each_variant_uses_own_selected_checkpoints"] = (
                    True
                )

                paths[1].write_text("tampered\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "hashes differ"):
                    subject.inspect_complete(
                        output_root=output_root,
                        source_lock_path=source_lock_path,
                    )

    def test_execution_plan_never_invokes_evaluator_or_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = subject.execution_plan(
                output_root=Path(temporary) / "diagnostic"
            )
        self.assertFalse(plan["invokes_evaluator"])
        self.assertFalse(plan["invokes_gpu"])
        self.assertEqual(plan["checkpoint_input_count"], 2)
        self.assertEqual(plan["knockout_row_count"], 8)


if __name__ == "__main__":
    unittest.main()
