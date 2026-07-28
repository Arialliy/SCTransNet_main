from __future__ import annotations

import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments import (
    postprocess_tpd_ner_v8_mprs_dch_v4_tail_formula_counterfactual as subject,
)


def _sha(character: str) -> str:
    return character * 64


def _point(
    matched: int,
    *,
    fa: float = 1e-7,
    miou: float = 0.8,
    threshold: float = 0.5,
    tiny_matched: int = 39,
) -> dict:
    return {
        "matched_target_count": matched,
        "target_count": 189,
        "pd": matched / 189,
        "fa": fa,
        "miou": miou,
        "matched_tiny_target_count": tiny_matched,
        "tiny_target_count": 39,
        "tiny_pd": tiny_matched / 39,
        "threshold": threshold,
    }


def _mode_row(
    budget_counts: list[int],
    *,
    fixed_matched: int = 100,
    fixed_fa: float = 1e-6,
    fixed_miou: float = 0.8,
) -> dict:
    return {
        "fixed_threshold_0_5": _point(
            fixed_matched,
            fa=fixed_fa,
            miou=fixed_miou,
        ),
        "budgets": {
            key: _point(
                matched,
                fa=min(float(limit), 1e-7),
                threshold=0.9,
            )
            for key, limit, matched in zip(
                subject.evaluator.BUDGET_KEYS,
                subject.evaluator.FA_BUDGETS,
                budget_counts,
            )
        },
    }


def _role_rows(
    *,
    direct_budgets: tuple[list[int], list[int]],
    complement_budgets: tuple[list[int], list[int]],
    direct_fixed: tuple[tuple[int, float, float], tuple[int, float, float]]
    | None = None,
    complement_fixed: (
        tuple[tuple[int, float, float], tuple[int, float, float]] | None
    ) = None,
) -> dict:
    roles = tuple(subject.evaluator.CHECKPOINT_ROLES.values())
    direct_fixed = direct_fixed or (
        (100, 1e-6, 0.81),
        (100, 1e-6, 0.81),
    )
    complement_fixed = complement_fixed or (
        (100, 1e-6, 0.81),
        (100, 1e-6, 0.81),
    )
    rows = {}
    for index, role in enumerate(roles):
        direct_values = direct_fixed[index]
        complement_values = complement_fixed[index]
        rows[role] = {
            "checkpoint_filename": subject.evaluator.CHECKPOINTS[index],
            "checkpoint_epoch": index + 1,
            "source_checkpoint_sha256": _sha("a"),
            "source_state_dict_sha256": _sha("b"),
            "modes": {
                "legacy_global": _mode_row([10, 10, 10, 10, 10]),
                "direct_tail": _mode_row(
                    direct_budgets[index],
                    fixed_matched=direct_values[0],
                    fixed_fa=direct_values[1],
                    fixed_miou=direct_values[2],
                ),
                "complement_tail": _mode_row(
                    complement_budgets[index],
                    fixed_matched=complement_values[0],
                    fixed_fa=complement_values[1],
                    fixed_miou=complement_values[2],
                ),
            },
        }
    return rows


def _dummy_implementation_hashes() -> dict:
    return {
        name: {"path": str(path.resolve()), "sha256": _sha("c")}
        for name, path in subject.evaluator._source_paths().items()
    }


def _dummy_input_hashes(checkpoint: str) -> dict:
    characters = {
        "source_checkpoint": "d",
        "canonical_v3_sweep": "e",
        "protocol.json": "f",
        "split.json": "1",
        "summary.json": "2",
        "metrics.jsonl": "3",
    }
    return {
        name: _sha(characters[name])
        for name in subject.INPUT_HASH_KEYS
    }


def _canonical(checkpoint: str) -> dict:
    return json.loads(
        subject.evaluator.canonical_sweep_path(checkpoint).read_text(
            encoding="utf-8"
        )
    )


def _synthetic_payload(
    checkpoint: str,
    *,
    input_path: Path,
) -> tuple[dict, dict]:
    canonical = _canonical(checkpoint)
    source_state_sha = _sha("4")
    input_hashes = _dummy_input_hashes(checkpoint)
    implementation_hashes = _dummy_implementation_hashes()
    evaluations = []
    for index, mode in enumerate(subject.ALL_FORMULA_MODES):
        evaluation = copy.deepcopy(canonical)
        manifest = {
            "ner_dc_offset_support_mode": mode,
            "tail_z_thresholds": {4: 1.5, 3: 2.0, 2: 2.5},
            "tail_z_thresholds_frozen": True,
        }
        evaluation.update(
            {
                "schema": subject.evaluator.MODE_EVALUATION_SCHEMA,
                "status": "complete",
                "formula_mode": mode,
                "formula_expression": (
                    subject.evaluator.FORMULA_EXPRESSIONS[mode]
                ),
                "formula_index": index,
                "source_checkpoint_sha256_before": input_hashes[
                    "source_checkpoint"
                ],
                "source_checkpoint_sha256_after": input_hashes[
                    "source_checkpoint"
                ],
                "source_state_dict_sha256": source_state_sha,
                "evaluated_state_dict_sha256": source_state_sha,
                "strict_v3_state_load": True,
                "state_changed": False,
                "derived_checkpoint_written": False,
                "validation_count": subject.evaluator.VALIDATION_COUNT,
                "fixed_threshold": subject.evaluator.FIXED_THRESHOLD,
                "fa_budgets": list(subject.evaluator.FA_BUDGETS),
                "metric_contract": copy.deepcopy(
                    subject.evaluator.METRIC_CONTRACT
                ),
                "tail_z_thresholds": {4: 1.5, 3: 2.0, 2: 2.5},
                "model": {
                    "model_class": "synthetic.V4",
                    "formula_mode": mode,
                    "formula_expression": (
                        subject.evaluator.FORMULA_EXPRESSIONS[mode]
                    ),
                    "relay_parameters": (
                        subject.evaluator.v4_model_source
                        .PRODUCTION_V4_RELAY_PARAMETERS
                    ),
                    "total_parameters": (
                        subject.evaluator.v4_model_source
                        .PRODUCTION_V4_RELAY_ON_PARAMETERS
                    ),
                    "state_key_count": 100,
                    "state_dict_sha256": source_state_sha,
                    "architecture_manifest": manifest,
                    "architecture_manifest_sha256": (
                        subject.evaluator._canonical_sha256(manifest)
                    ),
                    "parent_metadata": {},
                },
                "gpu_memory": {
                    "max_memory_allocated_bytes": 1,
                    "max_memory_reserved_bytes": 1,
                },
            }
        )
        evaluations.append(evaluation)

    equivalence = subject.evaluator.require_legacy_canonical_exact(
        evaluations[0],
        canonical,
    )
    dependency_paths = subject._fixed_dependency_paths(checkpoint)
    payload = {
        "schema": subject.evaluator.EVALUATION_SCHEMA,
        "status": "complete",
        "artifact_kind": subject.evaluator.ARTIFACT_KIND,
        "scope": "same_v3_checkpoint_three_v4_forward_formulas",
        "diagnostic_only": True,
        "zero_training": True,
        "affects_v3_formal_decision": False,
        "formal_training_authorized_by_this_artifact": False,
        "official_test_accessed": False,
        "dataset": subject.evaluator.DATASET,
        "variant": subject.evaluator.VARIANT,
        "training_seed": subject.evaluator.TRAINING_SEED,
        "split_seed": subject.evaluator.SPLIT_SEED,
        "expected_epochs": subject.evaluator.EXPECTED_EPOCHS,
        "validation_count": subject.evaluator.VALIDATION_COUNT,
        "validation_split_sha256": _sha("5"),
        "run_directory": str(subject.evaluator.FORMAL_RUN_DIR),
        "run_identity": {},
        "checkpoint_filename": checkpoint,
        "checkpoint_role": subject.evaluator.CHECKPOINT_ROLES[checkpoint],
        "checkpoint_epoch": 1,
        "checkpoint_validation_metrics": {},
        "source_checkpoint": {
            "path": str(dependency_paths["source_checkpoint"]),
            "sha256": input_hashes["source_checkpoint"],
            "identity": {},
            "state_dict_sha256": source_state_sha,
            "checkpoint_payload_state_dict_sha256": source_state_sha,
        },
        "source_state_dict_sha256": source_state_sha,
        "formula_modes": list(subject.ALL_FORMULA_MODES),
        "formula_expressions": dict(subject.evaluator.FORMULA_EXPRESSIONS),
        "mode_count": 3,
        "metric_contract": copy.deepcopy(subject.evaluator.METRIC_CONTRACT),
        "formal_default_mode": (
            subject.evaluator.v4_model_source.DEFAULT_DC_SUPPORT_MODE
        ),
        "tail_z_thresholds": {4: 1.5, 3: 2.0, 2: 2.5},
        "tail_z_thresholds_frozen": True,
        "fixed_threshold": subject.evaluator.FIXED_THRESHOLD,
        "fa_budgets": list(subject.evaluator.FA_BUDGETS),
        "data_contract": {},
        "canonical_v3_sweep": {
            "path": str(dependency_paths["canonical_v3_sweep"]),
            "sha256": input_hashes["canonical_v3_sweep"],
            "schema": canonical.get("schema"),
        },
        "legacy_canonical_equivalence": equivalence,
        "evaluations": evaluations,
        "environment": {},
        "implementation_hashes_before": implementation_hashes,
        "implementation_hashes_after": copy.deepcopy(implementation_hashes),
        "input_hashes_before": input_hashes,
        "input_hashes_after": copy.deepcopy(input_hashes),
        "derived_checkpoint_written": False,
        "output_overwrite_forbidden": True,
        "audit": {
            "formal_v3_artifacts_read_only": True,
            "formal_v3_artifacts_unchanged": True,
            "all_modes_strict_loaded_from_same_pristine_v3_state": True,
            "mode_order": list(subject.ALL_FORMULA_MODES),
            "legacy_checked_before_alternative_modes": True,
            "no_derived_checkpoint": True,
            "output_path": str(input_path.resolve()),
            "invocation_argv": [],
        },
    }
    input_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload, canonical


class FormulaSelectionRuleTests(unittest.TestCase):
    def test_only_direct_qualifies_and_is_selected(self) -> None:
        rows = _role_rows(
            direct_budgets=(
                [11, 10, 10, 10, 10],
                [10, 10, 10, 10, 10],
            ),
            complement_budgets=(
                [9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9],
            ),
        )
        result = subject.select_formula(rows)
        self.assertEqual(result["decision"], "DIRECT_TAIL_SELECTED")
        self.assertEqual(result["selected_formula_mode"], "direct_tail")
        self.assertTrue(result["formal_v4_formula_selected"])
        self.assertTrue(
            result["candidate_assessments"]["direct_tail"]["qualifies"]
        )
        self.assertFalse(
            result["candidate_assessments"]["complement_tail"]["qualifies"]
        )

    def test_only_complement_qualifies_and_is_selected(self) -> None:
        rows = _role_rows(
            direct_budgets=(
                [9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9],
            ),
            complement_budgets=(
                [11, 10, 10, 10, 10],
                [10, 10, 10, 10, 10],
            ),
        )
        result = subject.select_formula(rows)
        self.assertEqual(result["decision"], "COMPLEMENT_TAIL_SELECTED")
        self.assertEqual(
            result["selected_formula_mode"],
            "complement_tail",
        )
        self.assertTrue(result["formal_v4_formula_selected"])

    def test_both_qualify_and_direct_jointly_dominates(self) -> None:
        rows = _role_rows(
            direct_budgets=(
                [12, 10, 10, 10, 10],
                [12, 10, 10, 10, 10],
            ),
            complement_budgets=(
                [11, 10, 10, 10, 10],
                [11, 10, 10, 10, 10],
            ),
            direct_fixed=(
                (101, 8e-7, 0.83),
                (101, 8e-7, 0.83),
            ),
            complement_fixed=(
                (100, 9e-7, 0.82),
                (100, 9e-7, 0.82),
            ),
        )
        result = subject.select_formula(rows)
        self.assertEqual(result["decision"], "DIRECT_TAIL_SELECTED")
        self.assertTrue(
            result["joint_dominance"][
                "direct_tail_over_complement_tail"
            ]
        )

    def test_both_qualify_with_mixed_tradeoff_is_inconclusive(self) -> None:
        rows = _role_rows(
            direct_budgets=(
                [11, 10, 10, 10, 10],
                [10, 11, 10, 10, 10],
            ),
            complement_budgets=(
                [10, 11, 10, 10, 10],
                [11, 10, 10, 10, 10],
            ),
            direct_fixed=(
                (100, 2e-6, 0.83),
                (100, 2e-6, 0.83),
            ),
            complement_fixed=(
                (101, 1e-6, 0.79),
                (101, 1e-6, 0.79),
            ),
        )
        result = subject.select_formula(rows)
        self.assertEqual(result["decision"], "FORMULA_INCONCLUSIVE")
        self.assertIsNone(result["selected_formula_mode"])
        self.assertFalse(result["formal_v4_formula_selected"])

    def test_no_candidate_qualifies_is_local_scope_rejected(self) -> None:
        rows = _role_rows(
            direct_budgets=(
                [9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9],
            ),
            complement_budgets=(
                [9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9],
            ),
        )
        result = subject.select_formula(rows)
        self.assertEqual(result["decision"], "LOCAL_SCOPE_REJECTED")
        self.assertIsNone(result["selected_formula_mode"])
        self.assertFalse(result["formal_v4_formula_selected"])

    def test_fixed_legacy_pareto_dominance_fails_candidate(self) -> None:
        rows = _role_rows(
            direct_budgets=(
                [11, 10, 10, 10, 10],
                [11, 10, 10, 10, 10],
            ),
            complement_budgets=(
                [9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9],
            ),
            direct_fixed=(
                (100, 2e-6, 0.79),
                (100, 2e-6, 0.79),
            ),
        )
        assessment = subject.assess_local_formula(
            rows,
            mode="direct_tail",
        )
        self.assertFalse(assessment["qualifies"])
        self.assertEqual(
            assessment["legacy_fixed_pareto_dominance_count"],
            2,
        )

    def test_four_of_five_boundary_and_cross_role_strict_boundary(self) -> None:
        rows = _role_rows(
            direct_budgets=(
                [11, 10, 10, 10, 9],
                [10, 10, 10, 10, 9],
            ),
            complement_budgets=(
                [9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9],
            ),
        )
        assessment = subject.assess_local_formula(
            rows,
            mode="direct_tail",
        )
        self.assertTrue(assessment["qualifies"])
        for role in subject.evaluator.CHECKPOINT_ROLES.values():
            self.assertEqual(
                assessment["per_role"][role]["noninferior_budget_count"],
                4,
            )
        self.assertEqual(
            assessment["strict_budget_improvement_count_across_roles"],
            1,
        )


class CounterfactualPayloadValidationTests(unittest.TestCase):
    def test_synthetic_payload_validates_schema_hashes_legacy_and_same_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.json"
            payload, canonical = _synthetic_payload(
                "best.pth.tar",
                input_path=path,
            )
            record = subject.validate_counterfactual_payload(
                payload,
                checkpoint="best.pth.tar",
                input_path=path,
                canonical_payload=canonical,
                verify_live_dependencies=False,
            )
        self.assertEqual(
            tuple(record["modes"]),
            subject.ALL_FORMULA_MODES,
        )
        self.assertTrue(record["all_modes_same_state"])
        self.assertTrue(
            record["legacy_canonical_equivalence"][
                "legacy_global_canonical_exact"
            ]
        )
        self.assertEqual(
            {
                row["model"]["state_dict_sha256"]
                for row in record["modes"].values()
            },
            {record["source_state_dict_sha256"]},
        )

    def test_state_hash_or_legacy_metric_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.json"
            payload, canonical = _synthetic_payload(
                "best.pth.tar",
                input_path=path,
            )
            changed_state = copy.deepcopy(payload)
            changed_state["evaluations"][1][
                "evaluated_state_dict_sha256"
            ] = _sha("9")
            with self.assertRaisesRegex(ValueError, "evaluated state"):
                subject.validate_counterfactual_payload(
                    changed_state,
                    checkpoint="best.pth.tar",
                    input_path=path,
                    canonical_payload=canonical,
                    verify_live_dependencies=False,
                )

            changed_legacy = copy.deepcopy(payload)
            changed_fixed = changed_legacy["evaluations"][0][
                "fixed_threshold_0_5"
            ]
            changed_fixed["matched_target_count"] -= 1
            changed_fixed["pd"] = (
                changed_fixed["matched_target_count"]
                / changed_fixed["target_count"]
            )
            with self.assertRaisesRegex(ValueError, "canonically exact"):
                subject.validate_counterfactual_payload(
                    changed_legacy,
                    checkpoint="best.pth.tar",
                    input_path=path,
                    canonical_payload=canonical,
                    verify_live_dependencies=False,
                )

    def test_source_checkpoint_and_implementation_hash_tampering_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.json"
            payload, canonical = _synthetic_payload(
                "best.pth.tar",
                input_path=path,
            )
            changed_checkpoint = copy.deepcopy(payload)
            changed_checkpoint["evaluations"][2][
                "source_checkpoint_sha256_after"
            ] = _sha("8")
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256_after"):
                subject.validate_counterfactual_payload(
                    changed_checkpoint,
                    checkpoint="best.pth.tar",
                    input_path=path,
                    canonical_payload=canonical,
                    verify_live_dependencies=False,
                )

            changed_source = copy.deepcopy(payload)
            changed_source["implementation_hashes_after"][
                "v4_model"
            ]["sha256"] = _sha("7")
            with self.assertRaisesRegex(
                ValueError,
                "implementation hashes before/after",
            ):
                subject.validate_counterfactual_payload(
                    changed_source,
                    checkpoint="best.pth.tar",
                    input_path=path,
                    canonical_payload=canonical,
                    verify_live_dependencies=False,
                )

    def test_live_dependency_verification_rejects_synthetic_digests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.json"
            payload, canonical = _synthetic_payload(
                "best.pth.tar",
                input_path=path,
            )
            with self.assertRaisesRegex(ValueError, "live SHA-256"):
                subject.validate_counterfactual_payload(
                    payload,
                    checkpoint="best.pth.tar",
                    input_path=path,
                    canonical_payload=canonical,
                    verify_live_dependencies=True,
                )

    def test_two_checkpoint_three_mode_pair_builds_local_only_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {}
            for checkpoint in subject.evaluator.CHECKPOINTS:
                path = root / f"{checkpoint}.json"
                payload, canonical = _synthetic_payload(
                    checkpoint,
                    input_path=path,
                )
                records[checkpoint] = (
                    subject.validate_counterfactual_payload(
                        payload,
                        checkpoint=checkpoint,
                        input_path=path,
                        canonical_payload=canonical,
                        verify_live_dependencies=False,
                    )
                )
            report = subject.build_aggregate_report(records)
        self.assertEqual(report["checkpoint_count"], 2)
        self.assertEqual(
            set(report["role_results"]),
            set(subject.evaluator.CHECKPOINT_ROLES.values()),
        )
        for row in report["role_results"].values():
            self.assertEqual(
                tuple(row["modes"]),
                subject.ALL_FORMULA_MODES,
            )
        self.assertFalse(report["affects_v3_formal_decision"])
        self.assertFalse(report["v3_formal_decision_modified"])
        self.assertFalse(
            report["formal_training_authorized_by_this_artifact"]
        )
        self.assertEqual(report["decision"], "LOCAL_SCOPE_REJECTED")
        self.assertFalse(report["formal_v4_formula_selected"])


class AggregatePublicationTests(unittest.TestCase):
    def _publication_report(self) -> dict:
        rows = _role_rows(
            direct_budgets=(
                [11, 10, 10, 10, 10],
                [10, 10, 10, 10, 10],
            ),
            complement_budgets=(
                [9, 9, 9, 9, 9],
                [9, 9, 9, 9, 9],
            ),
        )
        selection = subject.select_formula(rows)
        return {
            **selection,
            "role_results": rows,
            "candidate_assessments": selection["candidate_assessments"],
            "input_artifacts": {
                "best.pth.tar": {"path": "/input/a", "sha256": _sha("a")},
                "best_miou.pth.tar": {
                    "path": "/input/b",
                    "sha256": _sha("b"),
                },
            },
            "source_hashes": {
                "postprocessor": {
                    "path": str(subject.POSTPROCESS_PATH),
                    "sha256": subject._sha256_file(
                        subject.POSTPROCESS_PATH
                    ),
                }
            },
        }

    def test_bundle_hashes_marker_last_and_no_overwrite(self) -> None:
        report = self._publication_report()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "aggregate.json"
            markdown_path = root / "aggregate.md"
            marker_path = root / "POSTPROCESS_COMPLETE.json"
            marker = subject.publish_bundle(
                report,
                json_path=json_path,
                markdown_path=markdown_path,
                marker_path=marker_path,
            )
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            self.assertTrue(marker_path.is_file())
            self.assertEqual(
                marker["aggregate_json"]["sha256"],
                subject._sha256_file(json_path),
            )
            self.assertEqual(
                marker["aggregate_markdown"]["sha256"],
                subject._sha256_file(markdown_path),
            )
            self.assertTrue(marker["marker_written_last"])
            self.assertFalse(marker["affects_v3_formal_decision"])
            self.assertFalse(
                marker["formal_training_authorized_by_this_artifact"]
            )
            original = (
                json_path.read_bytes(),
                markdown_path.read_bytes(),
                marker_path.read_bytes(),
            )
            with self.assertRaises(FileExistsError):
                subject.publish_bundle(
                    report,
                    json_path=json_path,
                    markdown_path=markdown_path,
                    marker_path=marker_path,
                )
            self.assertEqual(
                original,
                (
                    json_path.read_bytes(),
                    markdown_path.read_bytes(),
                    marker_path.read_bytes(),
                ),
            )

    def test_cli_actions_and_no_optimization_sensitive_asserts(self) -> None:
        verify = subject.parse_args(["--verify-inputs"])
        aggregate = subject.parse_args(["--aggregate"])
        self.assertTrue(verify.verify_inputs)
        self.assertFalse(verify.aggregate)
        self.assertTrue(aggregate.aggregate)
        for argv in ([], ["--verify-inputs", "--aggregate"]):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit):
                    subject.parse_args(argv)

        source = subject.POSTPROCESS_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertEqual(
            [
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.Assert)
            ],
            [],
        )


if __name__ == "__main__":
    unittest.main()
