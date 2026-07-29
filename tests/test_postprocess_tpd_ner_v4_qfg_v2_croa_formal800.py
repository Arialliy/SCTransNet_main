#!/usr/bin/env python3
"""Lightweight synthetic tests for the formal QFG final selector."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import (
    postprocess_tpd_ner_v4_qfg_v2_croa_formal800 as selection,
)


def point(
    threshold: float,
    *,
    matched: int,
    fa: float,
    miou: float,
    tiny_matched: int,
    unmatched: int,
) -> dict[str, object]:
    return {
        "threshold": threshold,
        "pd": matched / selection.TARGET_COUNT,
        "fa": fa,
        "miou": miou,
        "tiny_pd": tiny_matched / selection.TINY_TARGET_COUNT,
        "false_objects_per_image": (
            unmatched / selection.VALIDATION_COUNT
        ),
        "target_count": selection.TARGET_COUNT,
        "matched_target_count": matched,
        "tiny_target_count": selection.TINY_TARGET_COUNT,
        "matched_tiny_target_count": tiny_matched,
        "unmatched_predicted_object_count": unmatched,
    }


def method(
    method_id: str,
    *,
    matched: int,
    fa: float,
    miou: float,
    tiny_matched: int,
    unmatched: int,
) -> dict[str, object]:
    roles: dict[str, object] = {}
    for role_name, checkpoint, checkpoint_role in (
        (
            "pd_primary",
            "best.pth.tar",
            "best_validation_pd_primary",
        ),
        (
            "miou_secondary",
            "best_miou.pth.tar",
            "best_validation_miou_secondary",
        ),
    ):
        raw_points = [
            point(
                threshold,
                matched=matched,
                fa=fa,
                miou=miou,
                tiny_matched=tiny_matched,
                unmatched=unmatched,
            )
            for threshold in (0.4, 0.5, 0.6)
        ]
        roles[role_name] = {
            "checkpoint": checkpoint,
            "checkpoint_role": checkpoint_role,
            "role_name": role_name,
            "checkpoint_epoch": 10,
            "checkpoint_sha256": "a" * 64,
            "checkpoint_path": f"/tmp/{method_id}/{checkpoint}",
            "run_directory": f"/tmp/{method_id}",
            "fixed_threshold_0_5": copy.deepcopy(raw_points[1]),
            "fa_budget_points": {
                key: copy.deepcopy(raw_points[1])
                for key in selection.BUDGET_KEYS
            },
            "raw_points": raw_points,
            "raw_point_count": len(raw_points),
            "sweep_binding": {
                "path": f"/tmp/{method_id}/{checkpoint}.json",
                "sha256": "b" * 64,
            },
        }
    return {
        "method_id": method_id,
        "display_name": method_id,
        "variant": method_id,
        "roles": roles,
    }


def base_matrix() -> dict[str, dict[str, object]]:
    return {
        "v4": method(
            "v4",
            matched=187,
            fa=6e-6,
            miou=0.89,
            tiny_matched=38,
            unmatched=7,
        ),
        "a_control": method(
            "a_control",
            matched=187,
            fa=5e-6,
            miou=0.90,
            tiny_matched=38,
            unmatched=6,
        ),
        "b_tss": method(
            "b_tss",
            matched=187,
            fa=5e-6,
            miou=0.90,
            tiny_matched=38,
            unmatched=6,
        ),
    }


class CandidateStateTests(unittest.TestCase):
    def test_repeated_budget_selection_is_not_a_non_isolated_interval(
        self,
    ) -> None:
        methods = {
            "a_control": method(
                "a_control",
                matched=187,
                fa=5e-6,
                miou=0.90,
                tiny_matched=38,
                unmatched=6,
            ),
            "c_qfg_only": method(
                "c_qfg_only",
                matched=188,
                fa=0.0,
                miou=0.91,
                tiny_matched=39,
                unmatched=0,
            ),
        }
        for value in methods.values():
            for role in value["roles"].values():
                role["raw_points"] = [role["raw_points"][1]]
                role["raw_point_count"] = 1
        evidence = selection._budget_support(
            methods,
            "c_qfg_only",
            comparison_method_ids=("a_control", "c_qfg_only"),
            exclusive_only=False,
        )
        self.assertEqual(evidence, [])

    def test_points_shared_with_pre_qfg_method_do_not_count(self) -> None:
        shared = {
            "matched": 0,
            "fa": 0.0,
            "miou": 0.0,
            "tiny_matched": 0,
            "unmatched": 0,
        }
        methods = {
            "a_control": method("a_control", **shared),
            "c_qfg_only": method("c_qfg_only", **shared),
        }
        result = selection.analyze_candidate(
            methods,
            candidate_id="c_qfg_only",
            direct_reference_id="a_control",
        )
        self.assertGreater(result["joint_non_dominated_point_count"], 0)
        self.assertEqual(
            result["qfg_frontier_contribution_point_count"],
            0,
        )
        self.assertEqual(result["status"], selection.DOMINATED)

    def test_relative_improved(self) -> None:
        methods = {
            "a_control": method(
                "a_control",
                matched=187,
                fa=5e-6,
                miou=0.90,
                tiny_matched=38,
                unmatched=6,
            ),
            "c_qfg_only": method(
                "c_qfg_only",
                matched=188,
                fa=4e-6,
                miou=0.91,
                tiny_matched=39,
                unmatched=5,
            ),
        }
        result = selection.analyze_candidate(
            methods,
            candidate_id="c_qfg_only",
            direct_reference_id="a_control",
        )
        self.assertEqual(result["status"], selection.RELATIVE_IMPROVED)
        self.assertGreater(result["non_isolated_support_count"], 0)
        self.assertTrue(
            result["aligned_strict_improvement"][
                "non_isolated_strict_improvement"
            ]
        )
        self.assertEqual(
            result["objective_directions"],
            {
                "pd": "maximize",
                "fa": "minimize",
                "miou": "maximize",
                "tiny_pd": "maximize",
                "false_objects_per_image": "minimize",
            },
        )

    def test_pareto_mixed_tradeoff(self) -> None:
        methods = {
            "a_control": method(
                "a_control",
                matched=187,
                fa=4e-6,
                miou=0.92,
                tiny_matched=39,
                unmatched=4,
            ),
            "c_qfg_only": method(
                "c_qfg_only",
                matched=188,
                fa=6e-6,
                miou=0.90,
                tiny_matched=39,
                unmatched=6,
            ),
        }
        result = selection.analyze_candidate(
            methods,
            candidate_id="c_qfg_only",
            direct_reference_id="a_control",
        )
        self.assertEqual(
            result["status"],
            selection.PARETO_MIXED_TRADEOFF,
        )
        self.assertGreater(result["non_isolated_support_count"], 0)
        self.assertFalse(
            result["aligned_strict_improvement"][
                "non_isolated_strict_improvement"
            ]
        )

    def test_dominated(self) -> None:
        methods = {
            "a_control": method(
                "a_control",
                matched=188,
                fa=4e-6,
                miou=0.92,
                tiny_matched=39,
                unmatched=4,
            ),
            "c_qfg_only": method(
                "c_qfg_only",
                matched=187,
                fa=6e-6,
                miou=0.90,
                tiny_matched=38,
                unmatched=6,
            ),
        }
        result = selection.analyze_candidate(
            methods,
            candidate_id="c_qfg_only",
            direct_reference_id="a_control",
        )
        self.assertEqual(result["status"], selection.DOMINATED)
        self.assertEqual(result["non_isolated_support_count"], 0)


class DecisionFFTests(unittest.TestCase):
    def test_report_keeps_single_seed_claim_boundary_and_all_roles(
        self,
    ) -> None:
        methods = base_matrix()
        for method_id in ("baseline", "v1", "v2", "v3"):
            methods[method_id] = method(
                method_id,
                matched=187,
                fa=6e-6,
                miou=0.89,
                tiny_matched=38,
                unmatched=7,
            )
        methods["c_qfg_only"] = method(
            "c_qfg_only",
            matched=188,
            fa=4e-6,
            miou=0.92,
            tiny_matched=39,
            unmatched=4,
        )
        methods["d_tss_qfg"] = method(
            "d_tss_qfg",
            matched=188,
            fa=4e-6,
            miou=0.92,
            tiny_matched=39,
            unmatched=4,
        )
        report = selection.build_report(
            methods,
            authority_binding={"sha256": "a" * 64},
            input_bindings={},
        )
        self.assertFalse(report["paper_core_established"])
        self.assertFalse(report["stability_claim_supported"])
        self.assertFalse(report["official_test_accessed"])
        self.assertEqual(report["training_seed"], 42)
        self.assertEqual(len(report["methods"]), 9)
        deployment = report["deployment_selection"]
        self.assertEqual(deployment["candidate_count"], 12)
        self.assertFalse(deployment["cross_checkpoint_metric_stitching"])
        self.assertTrue(deployment["selected_point_is_checkpoint_local"])
        self.assertEqual(
            deployment["selected"]["method_id"],
            report["selection"]["selected_method_id"],
        )
        self.assertEqual(
            report["selection"]["deployment"],
            deployment,
        )
        self.assertEqual(
            report["posttraining_closure_source_lock"][
                "policy_summary_sha256"
            ],
            selection.closure_policy.policy_summary_sha256(),
        )
        for value in report["methods"].values():
            self.assertEqual(
                set(value["roles"]),
                {"pd_primary", "miou_secondary"},
            )
            for role in value["roles"].values():
                self.assertEqual(
                    set(role["fa_budget_points"]),
                    set(selection.BUDGET_KEYS),
                )
        markdown = selection.render_markdown(report)
        self.assertIn("固定阈值 0.5", markdown)
        self.assertIn("五个 Fa budget", markdown)
        self.assertIn("Decision F-F", markdown)
        self.assertIn("部署 checkpoint", markdown)
        self.assertIn("Post-training closure lock", markdown)

    def test_approximately_equivalent_c_and_d_prefers_c(self) -> None:
        methods = base_matrix()
        methods["c_qfg_only"] = method(
            "c_qfg_only",
            matched=188,
            fa=4e-6,
            miou=0.92,
            tiny_matched=39,
            unmatched=4,
        )
        methods["d_tss_qfg"] = method(
            "d_tss_qfg",
            matched=188,
            fa=4e-6,
            miou=0.92,
            tiny_matched=39,
            unmatched=4,
        )
        c_analysis = selection.analyze_candidate(
            methods,
            candidate_id="c_qfg_only",
            direct_reference_id="a_control",
        )
        d_analysis = selection.analyze_candidate(
            methods,
            candidate_id="d_tss_qfg",
            direct_reference_id="b_tss",
        )
        result = selection.decide_final_recipe(
            methods,
            c_analysis,
            d_analysis,
        )
        self.assertNotEqual(c_analysis["status"], selection.DOMINATED)
        self.assertNotEqual(d_analysis["status"], selection.DOMINATED)
        self.assertEqual(result["selected_method_id"], "c_qfg_only")
        self.assertFalse(result["final_training_uses_tss"])
        self.assertFalse(result["d_unique_over_c"])

    def test_d_unique_pareto_interval_retains_tss(self) -> None:
        methods = base_matrix()
        methods["c_qfg_only"] = method(
            "c_qfg_only",
            matched=188,
            fa=3e-6,
            miou=0.92,
            tiny_matched=39,
            unmatched=3,
        )
        methods["d_tss_qfg"] = method(
            "d_tss_qfg",
            matched=189,
            fa=5e-6,
            miou=0.91,
            tiny_matched=39,
            unmatched=5,
        )
        c_analysis = selection.analyze_candidate(
            methods,
            candidate_id="c_qfg_only",
            direct_reference_id="a_control",
        )
        d_analysis = selection.analyze_candidate(
            methods,
            candidate_id="d_tss_qfg",
            direct_reference_id="b_tss",
        )
        result = selection.decide_final_recipe(
            methods,
            c_analysis,
            d_analysis,
        )
        self.assertEqual(result["selected_method_id"], "d_tss_qfg")
        self.assertTrue(result["final_training_uses_tss"])
        self.assertTrue(result["d_unique_over_c"])
        self.assertGreater(
            result["d_vs_c_analysis"][
                "exclusive_non_isolated_support_count"
            ],
            0,
        )

    def test_both_dominated_falls_back_to_v4(self) -> None:
        methods = base_matrix()
        methods["v4"] = method(
            "v4",
            matched=189,
            fa=0.0,
            miou=0.95,
            tiny_matched=39,
            unmatched=0,
        )
        methods["c_qfg_only"] = method(
            "c_qfg_only",
            matched=187,
            fa=5e-6,
            miou=0.90,
            tiny_matched=38,
            unmatched=6,
        )
        methods["d_tss_qfg"] = method(
            "d_tss_qfg",
            matched=187,
            fa=6e-6,
            miou=0.89,
            tiny_matched=38,
            unmatched=7,
        )
        c_analysis = selection.analyze_candidate(
            methods,
            candidate_id="c_qfg_only",
            direct_reference_id="a_control",
        )
        d_analysis = selection.analyze_candidate(
            methods,
            candidate_id="d_tss_qfg",
            direct_reference_id="b_tss",
        )
        result = selection.decide_final_recipe(
            methods,
            c_analysis,
            d_analysis,
        )
        self.assertEqual(c_analysis["status"], selection.DOMINATED)
        self.assertEqual(d_analysis["status"], selection.DOMINATED)
        self.assertEqual(result["selected_method_id"], "v4")
        self.assertEqual(result["decision"], "FALLBACK_TO_FROZEN_V4")


class InputAndWriteOnceTests(unittest.TestCase):
    def test_extension_sweeps_call_both_evaluator_validators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary).resolve()
            payloads: dict[str, dict[str, object]] = {}
            audits: dict[str, dict[str, object]] = {}
            for checkpoint in selection.CHECKPOINTS:
                checkpoint_path = run_dir / checkpoint
                checkpoint_path.write_bytes(
                    f"checkpoint:{checkpoint}".encode("utf-8")
                )
                checkpoint_sha = selection.sha256_file(checkpoint_path)
                checkpoint_role = selection.CHECKPOINT_ROLES[checkpoint]
                fixed = point(
                    0.5,
                    matched=188,
                    fa=0.0,
                    miou=0.91,
                    tiny_matched=39,
                    unmatched=0,
                )
                payload = {
                    "dataset": selection.DATASET,
                    "seed": selection.TRAINING_SEED,
                    "split_seed": selection.SPLIT_SEED,
                    "validation_count": selection.VALIDATION_COUNT,
                    "validation_split_sha256": (
                        selection.VALIDATION_SPLIT_SHA256
                    ),
                    "official_test_accessed": False,
                    "variant": "synthetic_extension",
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_role": checkpoint_role,
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_epoch": 10,
                    "threshold_selection_scope": (
                        "single_checkpoint_only"
                    ),
                    "cross_checkpoint_point_pooling": False,
                    "evaluated_checkpoint_count": 1,
                    "fixed_threshold_0_5": fixed,
                    "best_points_under_fa_budget": {
                        key: copy.deepcopy(fixed)
                        for key in selection.BUDGET_KEYS
                    },
                    "points": [fixed],
                    "run_directory": str(run_dir),
                    "evaluation_source_binding": {
                        "schema": "synthetic_binding"
                    },
                    "evaluator_contract": {
                        "schema": "synthetic_contract"
                    },
                    "audit": {
                        "device_assignment": {"device": "cpu"}
                    },
                }
                sweep_path = (
                    run_dir / selection._sweep_filename(checkpoint)
                )
                sweep_path.write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                payloads[checkpoint] = payload
                audits[checkpoint] = {
                    "variant": "synthetic_extension",
                    "checkpoint_filename": checkpoint,
                    "checkpoint_role": checkpoint_role,
                    "checkpoint_epoch": 10,
                    "checkpoint_sha256": checkpoint_sha,
                    "state_dict_strict_load": True,
                    "checkpoint_path": str(checkpoint_path),
                    "source_binding": {
                        "schema": "synthetic_binding"
                    },
                }

            fake_evaluator = mock.Mock()
            fake_evaluator.validate_run_artifacts.side_effect = (
                lambda _run_dir, checkpoint: audits[checkpoint]
            )
            fake_evaluator.evaluator_contract.return_value = {
                "schema": "synthetic_contract"
            }
            spec = selection.ExtensionSpec(
                method_id="synthetic",
                display_name="synthetic",
                variant="synthetic_extension",
                evaluator_module="synthetic.evaluator",
                run_dir=run_dir,
            )
            with mock.patch.object(
                selection.importlib,
                "import_module",
                return_value=fake_evaluator,
            ):
                result = selection.validate_extension_method(spec)
            self.assertEqual(
                set(result["roles"]),
                {"pd_primary", "miou_secondary"},
            )
            self.assertEqual(
                fake_evaluator.validate_run_artifacts.call_count,
                2,
            )
            self.assertEqual(
                fake_evaluator.validate_output_identity.call_count,
                2,
            )

    def test_selector_role_mismatch_is_rejected(self) -> None:
        fixed = point(
            0.5,
            matched=188,
            fa=4e-6,
            miou=0.91,
            tiny_matched=39,
            unmatched=5,
        )
        payload = {
            "dataset": selection.DATASET,
            "seed": selection.TRAINING_SEED,
            "split_seed": selection.SPLIT_SEED,
            "validation_count": selection.VALIDATION_COUNT,
            "validation_split_sha256": (
                selection.VALIDATION_SPLIT_SHA256
            ),
            "official_test_accessed": False,
            "variant": "qfg_only",
            "checkpoint": "/tmp/qfg/best.pth.tar",
            "checkpoint_role": "best_validation_miou_secondary",
            "checkpoint_sha256": "a" * 64,
            "checkpoint_epoch": 10,
            "threshold_selection_scope": "single_checkpoint_only",
            "cross_checkpoint_point_pooling": False,
            "evaluated_checkpoint_count": 1,
            "fixed_threshold_0_5": fixed,
            "best_points_under_fa_budget": {
                key: fixed for key in selection.BUDGET_KEYS
            },
            "points": [fixed],
            "run_directory": "/tmp/qfg",
        }
        with self.assertRaisesRegex(ValueError, "checkpoint_role"):
            selection.normalize_sweep_payload(
                payload,
                method_id="c_qfg_only",
                display_name="C",
                expected_variant="qfg_only",
                checkpoint="best.pth.tar",
            )

    def test_frozen_authority_file_has_expected_sha(self) -> None:
        self.assertEqual(
            selection.sha256_file(selection.FROZEN_AUTHORITY_PATH),
            selection.FROZEN_AUTHORITY_SHA256,
        )

    def test_json_and_markdown_are_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "selection.json"
            markdown_path = root / "selection.md"
            report = {"schema": "synthetic", "status": "complete"}
            with mock.patch.object(
                selection,
                "render_markdown",
                return_value="# synthetic\n",
            ):
                selection.write_outputs_once(
                    report,
                    json_output=json_path,
                    markdown_output=markdown_path,
                )
                original_json = json_path.read_bytes()
                original_markdown = markdown_path.read_bytes()
                with self.assertRaisesRegex(
                    FileExistsError,
                    "refusing to replace",
                ):
                    selection.write_outputs_once(
                        {"schema": "changed"},
                        json_output=json_path,
                        markdown_output=markdown_path,
                    )
            self.assertEqual(json_path.read_bytes(), original_json)
            self.assertEqual(markdown_path.read_bytes(), original_markdown)
            self.assertEqual(
                json.loads(original_json),
                report,
            )

    def test_preexisting_markdown_prevents_partial_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "selection.json"
            markdown_path = root / "selection.md"
            markdown_path.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                selection.write_outputs_once(
                    {"schema": "synthetic"},
                    json_output=json_path,
                    markdown_output=markdown_path,
                )
            self.assertFalse(json_path.exists())
            self.assertEqual(
                markdown_path.read_text(encoding="utf-8"),
                "existing\n",
            )

    def test_collision_rollback_never_unlinks_external_replacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path = root / "selection.json"
            markdown_path = root / "selection.md"
            real_link = selection.os.link
            calls = 0

            def racing_link(source: Path, destination: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 1:
                    real_link(source, destination)
                    return
                json_path.unlink()
                json_path.write_bytes(b"external-json\n")
                markdown_path.write_bytes(b"external-markdown\n")
                raise FileExistsError("simulated concurrent publisher")

            with (
                mock.patch.object(
                    selection,
                    "render_markdown",
                    return_value="# synthetic\n",
                ),
                mock.patch.object(
                    selection.os,
                    "link",
                    side_effect=racing_link,
                ),
            ):
                with self.assertRaises(FileExistsError):
                    selection.write_outputs_once(
                        {"schema": "synthetic"},
                        json_output=json_path,
                        markdown_output=markdown_path,
                    )
            self.assertEqual(json_path.read_bytes(), b"external-json\n")
            self.assertEqual(
                markdown_path.read_bytes(),
                b"external-markdown\n",
            )


if __name__ == "__main__":
    unittest.main()
